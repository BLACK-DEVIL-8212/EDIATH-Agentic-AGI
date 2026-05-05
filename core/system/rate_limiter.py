"""
EDIATH Rate Limiter Module
Production-grade distributed rate limiting for autonomous AI system
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from core.utils.logger import logger

# #==================== Enums #====================


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""

    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_LOG = "sliding_log"


class RateLimitScope(Enum):
    """Rate limit scopes"""

    GLOBAL = "global"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"
    PER_COMPONENT = "per_component"


class RateLimitAction(Enum):
    """Actions when rate limit exceeded"""

    BLOCK = "block"
    DELAY = "delay"
    QUEUE = "queue"
    THROTTLE = "throttle"
    DEGRADE = "degrade"
    ALERT_ONLY = "alert_only"


# #==================== Data Classes #====================


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""

    name: str
    key_pattern: str
    limit: int  # Number of requests
    window_seconds: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    scope: RateLimitScope = RateLimitScope.GLOBAL
    action: RateLimitAction = RateLimitAction.BLOCK
    burst_limit: Optional[int] = None  # For token bucket
    refill_rate: Optional[float] = None  # Tokens per second
    queue_size: int = 100  # For queue strategy
    queue_timeout_seconds: int = 30  # For queue strategy
    whitelist: List[str] = field(default_factory=list)
    blacklist: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0  # Higher priority rules checked first


@dataclass
class RateLimitMetrics:
    """Rate limit metrics"""

    total_requests: int = 0
    blocked_requests: int = 0
    delayed_requests: int = 0
    queued_requests: int = 0
    throttled_requests: int = 0
    average_wait_time_ms: float = 0.0
    peak_queue_size: int = 0
    last_blocked_timestamp: Optional[datetime] = None
    violations_per_minute: float = 0.0


# #==================== Window Trackers #====================


class FixedWindowTracker:
    """Fixed window counter"""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.window_start = time.time()
        self.count = 0

    def allow(self) -> bool:
        now = time.time()
        if now - self.window_start >= self.window_seconds:
            self.window_start = now
            self.count = 0

        if self.count < self.limit:
            self.count += 1
            return True
        return False


class SlidingWindowTracker:
    """Sliding window with timestamp list"""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.timestamps: deque = deque()

    def allow(self) -> bool:
        now = time.time()

        # Remove old timestamps
        while self.timestamps and now - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()

        if len(self.timestamps) < self.limit:
            self.timestamps.append(now)
            return True
        return False

    def get_remaining(self) -> int:
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()
        return max(0, self.limit - len(self.timestamps))

    def get_reset_time(self) -> float:
        if not self.timestamps:
            return 0
        return self.window_seconds - (time.time() - self.timestamps[0])


class TokenBucketTracker:
    """Token bucket algorithm"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def allow(self) -> bool:
        now = time.time()

        # Refill tokens
        elapsed = now - self.last_refill
        refill = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def get_tokens(self) -> float:
        now = time.time()
        elapsed = now - self.last_refill
        refill = elapsed * self.refill_rate
        return min(self.capacity, self.tokens + refill)


class LeakyBucketTracker:
    """Leaky bucket algorithm"""

    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.water = 0
        self.last_leak = time.time()

    def allow(self) -> bool:
        now = time.time()

        # Leak water
        elapsed = now - self.last_leak
        leaked = elapsed * self.leak_rate
        self.water = max(0, self.water - leaked)
        self.last_leak = now

        if self.water < self.capacity:
            self.water += 1
            return True
        return False


class SlidingLogTracker:
    """Sliding log with sorted list"""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.log: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds

        # Filter log
        self.log = [t for t in self.log if t > cutoff]

        if len(self.log) < self.limit:
            self.log.append(now)
            return True
        return False


# #==================== Queue Manager #====================


class RateLimitQueue:
    """Queue for queued rate limit strategy"""

    def __init__(self, max_size: int, timeout_seconds: int):
        self.max_size = max_size
        self.timeout_seconds = timeout_seconds
        self._queue: deque = deque()
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, key: str, data: Any) -> Optional[Any]:
        """Enqueue a request"""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return None

            future = asyncio.Future()
            self._queue.append((key, data, future, time.time()))
            self._pending[key] = future

            try:
                return await asyncio.wait_for(future, timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                return None

    async def dequeue(self) -> Optional[tuple]:
        """Dequeue next request"""
        async with self._lock:
            if not self._queue:
                return None

            key, data, future, enqueue_time = self._queue.popleft()
            if key in self._pending:
                del self._pending[key]

            return (key, data, future, enqueue_time)

    def complete(self, key: str, result: Any):
        """Complete a queued request"""
        if key in self._pending:
            self._pending[key].set_result(result)

    def get_size(self) -> int:
        return len(self._queue)


# #==================== Rate Limiter #====================


class RateLimiter:
    """
    Production-grade rate limiter with multiple strategies
    """

    def __init__(self):
        self._rules: Dict[str, RateLimitRule] = {}
        self._trackers: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, RateLimitMetrics] = {}
        self._queues: Dict[str, RateLimitQueue] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running = False

        # Default rules
        self._init_default_rules()

        logger.info("RateLimiter initialized")

    def _init_default_rules(self):
        """Initialize default rate limit rules"""

        # Global API rate limit
        self.add_rule(
            RateLimitRule(
                name="global_api",
                key_pattern="*",
                limit=1000,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
                action=RateLimitAction.BLOCK,
                priority=0,
            )
        )

        # LLM calls rate limit
        self.add_rule(
            RateLimitRule(
                name="llm_calls",
                key_pattern="llm:*",
                limit=100,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                burst_limit=20,
                refill_rate=2.0,
                action=RateLimitAction.QUEUE,
                queue_size=50,
                queue_timeout_seconds=30,
                priority=1,
            )
        )

        # File operations rate limit
        self.add_rule(
            RateLimitRule(
                name="file_ops",
                key_pattern="file:*",
                limit=500,
                window_seconds=60,
                strategy=RateLimitStrategy.FIXED_WINDOW,
                action=RateLimitAction.BLOCK,
                priority=2,
            )
        )

        # Browser automation rate limit
        self.add_rule(
            RateLimitRule(
                name="browser_ops",
                key_pattern="browser:*",
                limit=60,
                window_seconds=60,
                strategy=RateLimitStrategy.LEAKY_BUCKET,
                burst_limit=10,
                action=RateLimitAction.DELAY,
                priority=3,
            )
        )

        # Code execution rate limit
        self.add_rule(
            RateLimitRule(
                name="code_execution",
                key_pattern="code:*",
                limit=30,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_LOG,
                action=RateLimitAction.BLOCK,
                priority=4,
            )
        )

        # User-specific rate limit
        self.add_rule(
            RateLimitRule(
                name="per_user",
                key_pattern="user:*",
                limit=200,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.PER_USER,
                action=RateLimitAction.THROTTLE,
                priority=5,
            )
        )

    def add_rule(self, rule: RateLimitRule):
        """Add or update a rate limit rule"""
        self._rules[rule.name] = rule
        self._metrics[rule.name] = RateLimitMetrics()
        logger.info(
            f"Rate limit rule added: {rule.name} (limit={rule.limit}/{rule.window_seconds}s)"
        )

    def remove_rule(self, name: str) -> bool:
        """Remove a rate limit rule"""
        if name in self._rules:
            del self._rules[name]
            if name in self._metrics:
                del self._metrics[name]
            if name in self._queues:
                del self._queues[name]
            logger.info(f"Rate limit rule removed: {name}")
            return True
        return False

    def _get_tracker(self, rule: RateLimitRule, key: str):
        """Get or create tracker for a key"""
        tracker_key = f"{rule.name}:{key}"

        if tracker_key not in self._trackers:
            if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                capacity = rule.burst_limit or rule.limit
                refill_rate = rule.refill_rate or (rule.limit / rule.window_seconds)
                self._trackers[tracker_key] = TokenBucketTracker(capacity, refill_rate)

            elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
                capacity = rule.burst_limit or rule.limit
                leak_rate = rule.limit / rule.window_seconds
                self._trackers[tracker_key] = LeakyBucketTracker(capacity, leak_rate)

            elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                self._trackers[tracker_key] = FixedWindowTracker(
                    rule.limit, rule.window_seconds
                )

            elif rule.strategy == RateLimitStrategy.SLIDING_LOG:
                self._trackers[tracker_key] = SlidingLogTracker(
                    rule.limit, rule.window_seconds
                )

            else:  # SLIDING_WINDOW default
                self._trackers[tracker_key] = SlidingWindowTracker(
                    rule.limit, rule.window_seconds
                )

            # Create queue if needed
            if rule.action == RateLimitAction.QUEUE and rule.name not in self._queues:
                self._queues[rule.name] = RateLimitQueue(
                    rule.queue_size, rule.queue_timeout_seconds
                )

        return self._trackers[tracker_key]

    def _get_key(
        self, rule: RateLimitRule, identifier: str, context: Dict[str, Any]
    ) -> str:
        """Generate key for rate limit check"""
        key_pattern = rule.key_pattern

        # Replace wildcards
        if key_pattern == "*":
            return "global"

        # Replace placeholders
        key = key_pattern
        if "user" in key_pattern and "user_id" in context:
            key = key.replace("*", context.get("user_id", "unknown"))
        if "ip" in key_pattern and "ip" in context:
            key = key.replace("*", context.get("ip", "unknown"))
        if "endpoint" in key_pattern and "endpoint" in context:
            key = key.replace("*", context.get("endpoint", "unknown"))

        return f"{identifier}:{key}"

    def _check_whitelist(self, rule: RateLimitRule, key: str) -> bool:
        """Check if key is whitelisted"""
        for pattern in rule.whitelist:
            if pattern in key or key in pattern:
                return True
        return False

    def _check_blacklist(self, rule: RateLimitRule, key: str) -> bool:
        """Check if key is blacklisted"""
        for pattern in rule.blacklist:
            if pattern in key or key in pattern:
                return True
        return False

    async def check_rate_limit(
        self, identifier: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits

        Returns:
            {
                "allowed": bool,
                "rule": str,
                "remaining": int,
                "reset_time": float,
                "action": str,
                "wait_time": float (if delayed)
            }
        """
        context = context or {}

        # Get applicable rules sorted by priority
        applicable_rules = sorted(
            [r for r in self._rules.values() if r.enabled], key=lambda r: r.priority
        )

        async with self._lock:
            for rule in applicable_rules:
                # Generate key
                key = self._get_key(rule, identifier, context)

                # Check whitelist
                if self._check_whitelist(rule, key):
                    continue

                # Check blacklist
                if self._check_blacklist(rule, key):
                    self._update_metrics(rule.name, blocked=True)
                    return {
                        "allowed": False,
                        "rule": rule.name,
                        "reason": "blacklisted",
                        "action": rule.action.value,
                    }

                # Get tracker
                tracker = self._get_tracker(rule, key)

                # Check limit
                allowed = tracker.allow()

                # Get remaining and reset time
                remaining = 0
                reset_time = 0

                if hasattr(tracker, "get_remaining"):
                    remaining = tracker.get_remaining()
                elif hasattr(tracker, "get_tokens"):
                    remaining = int(tracker.get_tokens())
                else:
                    remaining = max(0, rule.limit - getattr(tracker, "count", 0))

                if hasattr(tracker, "get_reset_time"):
                    reset_time = tracker.get_reset_time()

                # Handle action based on strategy
                if not allowed:
                    self._update_metrics(rule.name, blocked=True)

                    if rule.action == RateLimitAction.DELAY:
                        wait_time = min(1.0, reset_time if reset_time > 0 else 1.0)
                        await asyncio.sleep(wait_time)
                        self._update_metrics(rule.name, delayed=True)
                        return {
                            "allowed": True,
                            "rule": rule.name,
                            "action": "delayed",
                            "wait_time": wait_time,
                            "remaining": remaining,
                            "reset_time": reset_time,
                        }

                    elif (
                        rule.action == RateLimitAction.QUEUE
                        and rule.name in self._queues
                    ):
                        queue = self._queues[rule.name]
                        result = await queue.enqueue(key, context)
                        self._update_metrics(rule.name, queued=True)

                        if result is not None:
                            return {
                                "allowed": True,
                                "rule": rule.name,
                                "action": "queued",
                                "queue_position": queue.get_size(),
                                "remaining": remaining,
                                "reset_time": reset_time,
                            }
                        else:
                            return {
                                "allowed": False,
                                "rule": rule.name,
                                "reason": "queue_full",
                                "action": "blocked",
                                "remaining": 0,
                                "reset_time": reset_time,
                            }

                    elif rule.action == RateLimitAction.THROTTLE:
                        # Throttle by adding delay proportional to excess
                        excess = getattr(tracker, "count", 0) - rule.limit
                        throttle_time = min(5.0, excess * 0.1)
                        await asyncio.sleep(throttle_time)
                        self._update_metrics(rule.name, throttled=True)
                        return {
                            "allowed": True,
                            "rule": rule.name,
                            "action": "throttled",
                            "throttle_time": throttle_time,
                            "remaining": 0,
                            "reset_time": reset_time,
                        }

                    elif rule.action == RateLimitAction.ALERT_ONLY:
                        logger.warning(f"Rate limit exceeded for {rule.name}: {key}")
                        self._update_metrics(
                            rule.name, blocked=False
                        )  # Don't block, just alert
                        return {
                            "allowed": True,
                            "rule": rule.name,
                            "action": "alerted",
                            "warning": f"Rate limit exceeded for {rule.name}",
                            "remaining": remaining,
                            "reset_time": reset_time,
                        }

                    else:  # BLOCK
                        return {
                            "allowed": False,
                            "rule": rule.name,
                            "reason": "rate_limit_exceeded",
                            "action": "blocked",
                            "remaining": 0,
                            "reset_time": reset_time,
                        }

                # Allowed
                self._update_metrics(rule.name, allowed=True)

                return {
                    "allowed": True,
                    "rule": rule.name,
                    "remaining": remaining,
                    "reset_time": reset_time,
                    "action": "allowed",
                }

        # No rules matched, allow by default
        return {
            "allowed": True,
            "rule": "default",
            "remaining": -1,
            "reset_time": 0,
            "action": "allowed",
        }

    def _update_metrics(
        self,
        rule_name: str,
        allowed: bool = False,
        blocked: bool = False,
        delayed: bool = False,
        queued: bool = False,
        throttled: bool = False,
    ):
        """Update metrics for a rule"""
        if rule_name not in self._metrics:
            self._metrics[rule_name] = RateLimitMetrics()

        metrics = self._metrics[rule_name]
        metrics.total_requests += 1

        if blocked:
            metrics.blocked_requests += 1
            metrics.last_blocked_timestamp = datetime.now()
        if delayed:
            metrics.delayed_requests += 1
        if queued:
            metrics.queued_requests += 1
        if throttled:
            metrics.throttled_requests += 1

    async def get_metrics(self, rule_name: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limit metrics"""
        if rule_name:
            if rule_name in self._metrics:
                metrics = self._metrics[rule_name]
                return {
                    rule_name: {
                        "total_requests": metrics.total_requests,
                        "blocked_requests": metrics.blocked_requests,
                        "delayed_requests": metrics.delayed_requests,
                        "queued_requests": metrics.queued_requests,
                        "throttled_requests": metrics.throttled_requests,
                        "block_rate": metrics.blocked_requests
                        / max(1, metrics.total_requests),
                        "last_blocked": (
                            metrics.last_blocked_timestamp.isoformat()
                            if metrics.last_blocked_timestamp
                            else None
                        ),
                    }
                }
            return {}

        # Return all metrics
        result = {}
        for name, metrics in self._metrics.items():
            result[name] = {
                "total_requests": metrics.total_requests,
                "blocked_requests": metrics.blocked_requests,
                "delayed_requests": metrics.delayed_requests,
                "queued_requests": metrics.queued_requests,
                "throttled_requests": metrics.throttled_requests,
                "block_rate": metrics.blocked_requests / max(1, metrics.total_requests),
                "last_blocked": (
                    metrics.last_blocked_timestamp.isoformat()
                    if metrics.last_blocked_timestamp
                    else None
                ),
            }
        return result

    async def reset_metrics(self, rule_name: Optional[str] = None):
        """Reset metrics for a rule or all rules"""
        if rule_name:
            if rule_name in self._metrics:
                self._metrics[rule_name] = RateLimitMetrics()
                logger.info(f"Metrics reset for rule: {rule_name}")
        else:
            for name in self._metrics:
                self._metrics[name] = RateLimitMetrics()
            logger.info("All rate limit metrics reset")

    async def start(self):
        """Start rate limiter background tasks"""
        if self._is_running:
            return

        self._is_running = True
        self._loop = asyncio.get_running_loop()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("RateLimiter started")

    async def _cleanup_loop(self):
        """Periodic cleanup of old trackers"""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes

                # Clean old trackers (inactive for > 1 hour)
                now = time.time()
                to_remove = []

                for key, tracker in self._trackers.items():
                    # Check if tracker has last_access attribute
                    if hasattr(tracker, "last_access"):
                        if now - tracker.last_access > 3600:
                            to_remove.append(key)

                for key in to_remove:
                    del self._trackers[key]

                if to_remove:
                    logger.debug(f"Cleaned {len(to_remove)} inactive trackers")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _stop_impl(self):
        """Internal stop implementation using the rate limiter's loop."""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("RateLimiter stopped")

    async def stop(self):
        """Stop rate limiter"""
        current_loop = asyncio.get_running_loop()
        if self._loop and self._loop is not current_loop:
            if self._loop.is_closed():
                return await self._stop_impl()
            future = asyncio.run_coroutine_threadsafe(self._stop_impl(), self._loop)
            return await asyncio.wrap_future(future, loop=current_loop)
        return await self._stop_impl()

    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        return {
            "is_running": self._is_running,
            "total_rules": len(self._rules),
            "active_trackers": len(self._trackers),
            "active_queues": len(self._queues),
            "rules": list(self._rules.keys()),
            "metrics_summary": {
                name: {
                    "total": metrics.total_requests,
                    "blocked": metrics.blocked_requests,
                    "block_rate": metrics.blocked_requests
                    / max(1, metrics.total_requests),
                }
                for name, metrics in self._metrics.items()
            },
        }


# #==================== Rate Limiter Decorator #====================


def rate_limited(rule_name: str, identifier_key: str = "default"):
    """Decorator for rate limiting functions"""

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Get rate limiter instance
            from core.system.rate_limiter import get_rate_limiter

            limiter = await get_rate_limiter()

            # Get identifier
            identifier = kwargs.get(identifier_key, identifier_key)

            # Check rate limit
            result = await limiter.check_rate_limit(str(identifier))

            if not result["allowed"]:
                logger.warning(f"Rate limit exceeded for {func.__name__}: {result}")
                raise Exception(f"Rate limit exceeded: {result['rule']}")

            # Execute function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# #==================== Global Instance #====================

_rate_limiter: Optional[RateLimiter] = None
_limiter_lock = asyncio.Lock()


async def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    async with _limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = RateLimiter()
            await _rate_limiter.start()
    return _rate_limiter


async def shutdown_rate_limiter():
    """Shutdown rate limiter"""
    global _rate_limiter
    if _rate_limiter:
        await _rate_limiter.stop()
        _rate_limiter = None


# #==================== Example Usage #====================


async def example():
    """Example usage of rate limiter"""
    limiter = await get_rate_limiter()

    # Check rate limit
    result = await limiter.check_rate_limit(
        identifier="user_123", context={"user_id": "123", "endpoint": "/api/chat"}
    )

    print(f"Rate limit check: {result}")

    # Add custom rule
    custom_rule = RateLimitRule(
        name="custom_api",
        key_pattern="api:*",
        limit=50,
        window_seconds=60,
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        burst_limit=10,
        refill_rate=1.0,
    )
    limiter.add_rule(custom_rule)

    # Get metrics
    metrics = await limiter.get_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Get status
    status = limiter.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")


# #==================== Export #====================

__all__ = [
    "RateLimiter",
    "RateLimitRule",
    "RateLimitStrategy",
    "RateLimitScope",
    "RateLimitAction",
    "RateLimitMetrics",
    "get_rate_limiter",
    "shutdown_rate_limiter",
    "rate_limited",
]

if __name__ == "__main__":
    asyncio.run(example())
