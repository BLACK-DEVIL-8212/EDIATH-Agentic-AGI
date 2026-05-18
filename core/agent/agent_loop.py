"""
🔥 FIXED Agent Loop - EDIATH (STABLE VERSION)
✔ No random thinking
✔ No multi-LLM spam
✔ User-driven execution only
✔ Concurrency safe
✔ Production ready
✔ Multi-Agent compatible
✔ Circuit breaker pattern
✔ Request queuing
✔ Performance metrics
✔ Graceful degradation
✔ Enhanced error recovery
✔ Memory optimization
✔ Event streaming
"""

import asyncio
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
import time
import json
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logger import logger
from ..brain import ContextManager, LLMEngine, DecisionEngine
from ..memory import MemoryManager
from ..system.system_controller import SystemController
from .action_router import ActionRouter

# =========================
# ENUMS AND CONSTANTS
# =========================


class LoopState(Enum):
    """Agent loop state enumeration"""

    IDLE = "idle"
    RUNNING = "running"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class InputSource(Enum):
    """Input source types"""

    VOICE = "voice"
    TEXT = "text"
    API = "api"
    SYSTEM = "system"
    SCHEDULED = "scheduled"


@dataclass
class ProcessingEvent:
    """Event for processing pipeline"""

    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    source: InputSource


@dataclass
class LoopMetrics:
    """Performance metrics for agent loop"""

    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    total_inputs_processed: int = 0
    average_processing_time_ms: float = 0.0
    processing_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    last_iteration_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    queue_size: int = 0
    memory_usage_mb: float = 0.0


# =========================
# MAIN AGENT LOOP CLASS
# =========================


class AgentLoop:
    """Production-ready agent loop with comprehensive features"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        try:
            # ------------------------
            # CORE COMPONENTS
            # ------------------------
            self.context = ContextManager()
            self.llm = self.config.get("llm") or self.config.get("llm_engine")
            self.decision_engine = DecisionEngine()
            self.memory = MemoryManager()
            self.system = SystemController()
            self.action_router = ActionRouter()

            from .agent_core import RAgent

            self.r_agent = RAgent(self)

            # ------------------------
            # STATE MANAGEMENT
            # ------------------------
            self.state = LoopState.IDLE
            self.iteration_count = 0
            self.last_error: Optional[str] = None
            self.error_count = 0
            self.consecutive_errors = 0
            self.max_consecutive_errors = 10

            # User control flag
            self.user_active = False

            # ------------------------
            # LOCKS (THREAD SAFETY)
            # ------------------------
            self._llm_lock = asyncio.Lock()
            self._input_lock = asyncio.Lock()
            self._iteration_lock = asyncio.Lock()
            self._queue_lock = asyncio.Lock()
            self._event_lock = asyncio.Lock()

            # ------------------------
            # INPUT MANAGEMENT
            # ------------------------
            self._last_input = None
            self._last_input_time = 0.0
            self._last_user_input = None
            self._latest_input = None
            self._input_cooldown = 1.0
            self._input_queue: deque = deque(maxlen=1000)
            self._queue_processor_task: Optional[asyncio.Task] = None
            self._queue_enabled = True

            # ------------------------
            # EVENT STREAMING
            # ------------------------
            self._event_stream: deque = deque(maxlen=10000)
            self._event_subscribers: List[Callable] = []
            self._streaming_enabled = True

            # ------------------------
            # PERFORMANCE TRACKING
            # ------------------------
            self.metrics = LoopMetrics()
            self.processing_times: List[float] = []
            self._start_time = datetime.now()

            # ------------------------
            # CIRCUIT BREAKER
            # ------------------------
            self._circuit_open = False
            self._circuit_open_until: Optional[datetime] = None
            self.circuit_breaker_threshold = 5
            self.circuit_breaker_timeout = 60

            # ------------------------
            # RATE LIMITING
            # ------------------------
            self._request_timestamps: deque = deque(maxlen=100)
            self.requests_per_minute = 60
            self.min_request_interval = 0.1

            # ------------------------
            # PERSISTENCE
            # ------------------------
            self.persist_path: Optional[Path] = None
            self._last_persist = datetime.now()

            # ------------------------
            # HEALTH MONITORING
            # ------------------------
            self._health_status = "healthy"
            self._health_check_task: Optional[asyncio.Task] = None
            self.health_check_interval = 30

            # ------------------------
            # MIDDLEWARE
            # ------------------------
            self._before_process_hooks: List[Callable] = []
            self._after_process_hooks: List[Callable] = []
            self._error_hooks: List[Callable] = []

            # ------------------------
            # CONFIGURATION
            # ------------------------
            self.loop_delay = self.config.get("loop_delay", 0.8)
            self.max_iterations = self.config.get("max_iterations", 100000)
            self.input_timeout = self.config.get("input_timeout", 30)
            self.processing_timeout = self.config.get("processing_timeout", 45)

            # Initialize components
            self._setup_from_config()

            logger.info("🧠 Agent loop initialized successfully")

        except Exception as e:
            logger.warning(f"[Agent Init Error] {e}")
            self._initialize_fallback()
            self.last_error = str(e)

    def _initialize_fallback(self):
        """Initialize fallback state on error"""
        self.context = None
        self.llm = None
        self.decision_engine = None
        self.memory = None
        self.system = None
        self.action_router = None
        self.r_agent = None
        self.state = LoopState.ERROR

    def _setup_from_config(self):
        """Setup configuration from provided config"""
        self.loop_delay = self.config.get("loop_delay", 0.8)
        self.max_iterations = self.config.get("max_iterations", 100000)
        self.input_timeout = self.config.get("input_timeout", 30)
        self.processing_timeout = self.config.get("processing_timeout", 45)
        self._input_cooldown = self.config.get("input_cooldown", 1.0)
        self.requests_per_minute = self.config.get("requests_per_minute", 60)
        self.min_request_interval = self.config.get("min_request_interval", 0.1)

        # Queue settings
        self._queue_enabled = self.config.get("queue_enabled", True)
        max_queue = self.config.get("max_queue_size", 1000)
        self._input_queue = deque(maxlen=max_queue)

        # Circuit breaker
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)

        # Health check
        self.health_check_interval = self.config.get("health_check_interval", 30)

    # =========================
    # EVENT STREAMING
    # =========================

    async def _emit_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: InputSource = InputSource.SYSTEM,
    ):
        """Non-blocking, high-performance event emitter"""

        if not self._streaming_enabled:
            return

        try:
            event = ProcessingEvent(
                event_id=f"{int(time.time() * 1000)}-{len(self._event_stream)}",
                event_type=str(event_type),
                timestamp=datetime.utcnow(),
                data=data if isinstance(data, dict) else {},
                source=source,
            )

            # -------------------------
            # APPEND (LOCK MINIMIZED)
            # -------------------------
            async with self._event_lock:
                self._event_stream.append(event)
                subscribers = list(
                    self._event_subscribers
                )  # copy to avoid mutation issues

            # -------------------------
            # NOTIFY SUBSCRIBERS (NON-BLOCKING)
            # -------------------------
            tasks = []

            for subscriber in subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        tasks.append(asyncio.create_task(subscriber(event)))
                    else:
                        tasks.append(asyncio.to_thread(subscriber, event))
                except Exception:
                    continue

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.debug(f"Event emit failed: {e}")

    def subscribe_events(self, callback: Callable):
        """Subscribe to event stream"""
        self._event_subscribers.append(callback)

    def unsubscribe_events(self, callback: Callable):
        """Unsubscribe from event stream"""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    def get_events(
        self, limit: int = 100, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent events"""
        events = list(self._event_stream)[-limit:]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "data": e.data,
                "source": e.source.value,
            }
            for e in events
        ]

    # =========================
    # CIRCUIT BREAKER
    # =========================

    async def _check_circuit_breaker(self) -> bool:
        """Non-blocking, safe circuit breaker check with consistent state"""

        try:
            # -------------------------
            # FAST PATH (CLOSED)
            # -------------------------
            if not self._circuit_open:
                return True

            now = datetime.utcnow()

            # -------------------------
            # RECOVERY CHECK
            # -------------------------
            if self._circuit_open_until and now >= self._circuit_open_until:
                self._circuit_open = False
                self.consecutive_errors = 0
                self._health_status = "healthy"

                # Emit event without blocking main flow
                try:
                    asyncio.create_task(
                        self._emit_event(
                            "circuit_closed", {"timestamp": now.isoformat()}
                        )
                    )
                except Exception:
                    pass

                logger.info("✅ Circuit breaker closed")
                return True

            # -------------------------
            # STILL OPEN
            # -------------------------
            self._health_status = "circuit_open"
            return False

        except Exception as e:
            logger.debug(f"Circuit check error: {e}")
            return False

    async def _record_failure(self, error_type: str = "unknown"):
        """Thread-safe, non-blocking failure recorder with circuit breaker"""

        try:
            # -------------------------
            # UPDATE COUNTERS (SAFE)
            # -------------------------
            self.consecutive_errors += 1
            self.error_count += 1

            if not hasattr(self.metrics, "error_counts"):
                self.metrics.error_counts = {}

            self.metrics.error_counts[error_type] = (
                self.metrics.error_counts.get(error_type, 0) + 1
            )

            self.metrics.failed_iterations += 1

            # -------------------------
            # CIRCUIT BREAKER TRIGGER
            # -------------------------
            if self.consecutive_errors >= self.circuit_breaker_threshold:
                now = datetime.utcnow()

                self._circuit_open = True
                self._circuit_open_until = now + timedelta(
                    seconds=self.circuit_breaker_timeout
                )
                self._health_status = "degraded"

                # -------------------------
                # NON-BLOCKING EVENT EMIT (CRITICAL FIX)
                # -------------------------
                try:
                    asyncio.create_task(
                        self._emit_event(
                            "circuit_opened",
                            {
                                "consecutive_errors": self.consecutive_errors,
                                "threshold": self.circuit_breaker_threshold,
                                "timestamp": now.isoformat(),
                            },
                        )
                    )
                except Exception:
                    pass

                logger.error(
                    f"🚨 Circuit breaker opened after {self.consecutive_errors} failures"
                )

        except Exception as e:
            logger.debug(f"Failure recording error: {e}")

    async def _record_success(self):
        """Record a successful iteration"""
        self.consecutive_errors = 0
        self.metrics.successful_iterations += 1
        self._health_status = "healthy"

    # =========================
    # RATE LIMITING
    # =========================

    async def _check_rate_limit(self) -> bool:
        """Non-blocking, accurate rate limiter with sliding window cleanup"""

        try:
            now = time.monotonic()  # more stable than time.time()

            # -------------------------
            # MIN INTERVAL (NON-BLOCKING FIX)
            # -------------------------
            elapsed = now - getattr(self, "_last_input_time", 0)
            if elapsed < self.min_request_interval:
                wait = self.min_request_interval - elapsed
                try:
                    await asyncio.wait_for(asyncio.sleep(wait), timeout=wait + 0.1)
                except asyncio.CancelledError:
                    return False

            # -------------------------
            # CLEAN OLD TIMESTAMPS (CRITICAL FIX)
            # -------------------------
            window_start = now - 60
            self._request_timestamps = [
                t for t in self._request_timestamps if t >= window_start
            ]

            # -------------------------
            # RATE LIMIT CHECK
            # -------------------------
            if len(self._request_timestamps) >= self.requests_per_minute:
                oldest = self._request_timestamps[0]
                wait_time = 60 - (now - oldest)

                if wait_time > 0:
                    logger.debug(f"Rate limit hit, sleeping {wait_time:.2f}s")

                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(wait_time), timeout=wait_time + 0.1
                        )
                    except asyncio.CancelledError:
                        return False

            # -------------------------
            # RECORD REQUEST
            # -------------------------
            now = time.monotonic()
            self._request_timestamps.append(now)
            self._last_input_time = now

            return True

        except Exception as e:
            logger.debug(f"Rate limiter error: {e}")
            return True

    # =========================
    # INPUT QUEUE
    # =========================

    async def _start_queue_processor(self):
        """Start background queue processor"""
        if not self._queue_enabled:
            return

        self._queue_processor_task = asyncio.create_task(self._process_queue())
        logger.debug("Queue processor started")

    async def _process_queue(self):
        """Non-blocking, efficient queue processor with safe concurrency"""

        try:
            while self.state in [LoopState.RUNNING, LoopState.PROCESSING]:

                try:
                    input_item = None

                    # -------------------------
                    # POP ITEM (MINIMAL LOCK)
                    # -------------------------
                    async with self._queue_lock:
                        if self._input_queue:
                            input_item = self._input_queue.popleft()

                    if input_item is not None:
                        # -------------------------
                        # NON-BLOCKING EVENT (CRITICAL FIX)
                        # -------------------------
                        try:
                            asyncio.create_task(
                                self._emit_event(
                                    "queue_processed", {"input": str(input_item)[:100]}
                                )
                            )
                        except Exception:
                            pass

                        # -------------------------
                        # PROCESS INPUT (SAFE)
                        # -------------------------
                        await self._process_single_input(input_item)

                    else:
                        # -------------------------
                        # EFFICIENT WAIT (CRITICAL FIX)
                        # -------------------------
                        await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    logger.error(f"❌ Queue processor error: {e}")
                    await asyncio.sleep(0.05)  # prevent tight error loop

        finally:
            logger.info("✅ Queue processor stopped")

    async def queue_input(
        self, text: str, source: InputSource = InputSource.TEXT
    ) -> bool:
        """Queue input for processing"""
        if not self._queue_enabled:
            return await self.handle_user_input(text)

        text = str(text).strip()
        if not text:
            return False

        async with self._queue_lock:
            self._input_queue.append(text)
            self.metrics.queue_size = len(self._input_queue)

        await self._emit_event(
            "input_queued", {"text": text[:100], "source": source.value}, source
        )
        logger.debug(f"Input queued: {text[:50]}...")
        return True

    # =========================
    # MAIN LOOP
    # =========================

    async def run(self, max_iterations: Optional[int] = None):
        """Robust, non-blocking main agent loop"""

        self.state = LoopState.RUNNING
        self._start_time = datetime.utcnow()
        iteration = 0
        max_iterations = max_iterations or self.max_iterations

        try:
            # -------------------------
            # START BACKGROUND TASKS (SAFE)
            # -------------------------
            await self._start_queue_processor()

            self._health_check_task = asyncio.create_task(
                self._health_check_loop(), name="health_check"
            )

            # Non-blocking event
            asyncio.create_task(
                self._emit_event(
                    "loop_started", {"timestamp": datetime.utcnow().isoformat()}
                )
            )

            logger.info("🧠 Agent loop started")

            # -------------------------
            # MAIN LOOP
            # -------------------------
            while self.state == LoopState.RUNNING:

                if max_iterations and iteration >= max_iterations:
                    asyncio.create_task(
                        self._emit_event(
                            "max_iterations_reached", {"iterations": iteration}
                        )
                    )
                    break

                try:
                    await self._iteration()

                    iteration += 1
                    self.iteration_count = iteration
                    self.metrics.total_iterations = iteration
                    self.metrics.last_iteration_time = datetime.utcnow()

                    # -------------------------
                    # METRICS UPDATE (SAFE)
                    # -------------------------
                    self.metrics.uptime_seconds = (
                        datetime.utcnow() - self._start_time
                    ).total_seconds()

                    # -------------------------
                    # DRIFT-SAFE SLEEP
                    # -------------------------
                    await asyncio.sleep(max(self.loop_delay, 0.001))

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    self.last_error = str(e)

                    await self._record_failure(type(e).__name__)

                    logger.warning(f"[Loop Error] {e}")

                    # -------------------------
                    # FAILURE THRESHOLD
                    # -------------------------
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        logger.error(
                            f"Too many errors ({self.consecutive_errors}), stopping loop"
                        )
                        break

                    await asyncio.sleep(0.1)  # prevent tight crash loop

            return True

        except asyncio.CancelledError:
            logger.info("🛑 Loop cancelled")
            return False

        except Exception as e:
            logger.error(f"❌ Loop fatal error: {e}")
            return False

        finally:
            # -------------------------
            # CLEAN SHUTDOWN (CRITICAL FIX)
            # -------------------------
            try:
                if self._health_check_task:
                    self._health_check_task.cancel()
                    await asyncio.gather(
                        self._health_check_task, return_exceptions=True
                    )
            except Exception:
                pass

            await self._shutdown()

    async def _shutdown(self):
        """Safe, non-blocking shutdown with full cleanup"""

        self.state = LoopState.SHUTTING_DOWN

        try:
            # -------------------------
            # CANCEL BACKGROUND TASKS (SAFE)
            # -------------------------
            tasks_to_cancel = []

            if getattr(self, "_queue_processor_task", None):
                tasks_to_cancel.append(self._queue_processor_task)

            if getattr(self, "_health_check_task", None):
                tasks_to_cancel.append(self._health_check_task)

            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()

            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

            # -------------------------
            # PERSIST STATE (SAFE)
            # -------------------------
            try:
                await self._persist_state()
            except Exception as e:
                logger.debug(f"Persist failed during shutdown: {e}")

            # -------------------------
            # NON-BLOCKING EVENT (CRITICAL FIX)
            # -------------------------
            try:
                asyncio.create_task(
                    self._emit_event(
                        "loop_stopped", {"timestamp": datetime.utcnow().isoformat()}
                    )
                )
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Shutdown error: {e}")

        finally:
            # -------------------------
            # FINAL STATE
            # -------------------------
            self.state = LoopState.IDLE
            logger.info(f"🛑 Agent stopped at {self.iteration_count} iterations")

    # =========================
    # HEALTH CHECK LOOP
    # =========================

    async def _health_check_loop(self):
        """Non-blocking, stable health check loop with safe state transitions"""

        try:
            interval = max(getattr(self, "health_check_interval", 5), 1)

            while self.state == LoopState.RUNNING:

                try:
                    # -------------------------
                    # INTERRUPTIBLE SLEEP (CRITICAL FIX)
                    # -------------------------
                    await asyncio.wait_for(
                        asyncio.sleep(interval), timeout=interval + 1
                    )

                    if self.state != LoopState.RUNNING:
                        break

                    # -------------------------
                    # HEALTH CHECK (SAFE)
                    # -------------------------
                    is_healthy = await self.health_check()

                    now_iso = datetime.utcnow().isoformat()
                    prev_status = self._health_status

                    # -------------------------
                    # STATE TRANSITIONS (SAFE)
                    # -------------------------
                    if not is_healthy and prev_status == "healthy":
                        self._health_status = "degraded"

                        asyncio.create_task(
                            self._emit_event("health_degraded", {"timestamp": now_iso})
                        )

                    elif is_healthy and prev_status != "healthy":
                        self._health_status = "healthy"

                        asyncio.create_task(
                            self._emit_event("health_restored", {"timestamp": now_iso})
                        )

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    logger.debug(f"Health check error: {e}")
                    await asyncio.sleep(0.5)  # prevent tight error loop

        finally:
            logger.info("✅ Health check loop stopped")

    # =========================
    # SINGLE ITERATION
    # =========================

    async def _iteration(self):
        """Ultra-safe, non-blocking iteration loop with performance optimizations"""

        if self._iteration_lock.locked():
            return

        async with self._iteration_lock:
            try:
                # -------------------------
                # CIRCUIT BREAKER
                # -------------------------
                if not await self._check_circuit_breaker():
                    return

                # -------------------------
                # INPUT
                # -------------------------
                context = await self._process_input()
                if not context:
                    return

                user_input = context.get("user_input")
                if not user_input:
                    return

                self.user_active = True
                start_time = time.monotonic()

                try:
                    # -------------------------
                    # INTERRUPT SPEAKER (SAFE)
                    # -------------------------
                    try:
                        speaker = getattr(self.system, "speaker", None)
                        if speaker:
                            await asyncio.to_thread(speaker.stop)
                    except Exception:
                        pass

                    # -------------------------
                    # PROCESS WITH TIMEOUT
                    # -------------------------
                    result = await asyncio.wait_for(
                        self._process_with_context(context),
                        timeout=self.processing_timeout,
                    )

                    # -------------------------
                    # METRICS (OPTIMIZED)
                    # -------------------------
                    processing_time = (time.monotonic() - start_time) * 1000

                    self.processing_times.append(processing_time)
                    self.metrics.processing_times.append(processing_time)
                    self.metrics.total_inputs_processed += 1

                    # Fast average (no full sum every time)
                    count = len(self.metrics.processing_times)
                    prev_avg = self.metrics.average_processing_time_ms
                    self.metrics.average_processing_time_ms = (
                        (prev_avg * (count - 1) + processing_time) / count
                        if count > 1
                        else processing_time
                    )

                    # Trim efficiently
                    if count > 1000:
                        self.metrics.processing_times = self.metrics.processing_times[
                            -500:
                        ]

                    # -------------------------
                    # SUCCESS
                    # -------------------------
                    await self._record_success()

                    # Non-blocking event
                    asyncio.create_task(
                        self._emit_event(
                            "iteration_completed",
                            {
                                "processing_time_ms": processing_time,
                                "input_length": len(user_input),
                            },
                            context.get("source", InputSource.SYSTEM),
                        )
                    )

                except asyncio.TimeoutError:
                    asyncio.create_task(
                        self._emit_event(
                            "iteration_timeout", {"timeout": self.processing_timeout}
                        )
                    )
                    logger.warning("⏱ Iteration timeout")
                    raise

                finally:
                    self.user_active = False

            except Exception as e:
                self.last_error = str(e)

                await self._record_failure(type(e).__name__)

                # Non-blocking error event
                try:
                    asyncio.create_task(
                        self._emit_event("iteration_error", {"error": str(e)})
                    )
                except Exception:
                    pass

                logger.error(f"❌ Iteration error: {e}")

    async def _process_with_context(self, context: Dict[str, Any]) -> Optional[str]:
        """Optimized, non-blocking context processing with fault isolation"""

        try:
            # -------------------------
            # BEFORE HOOKS (SAFE)
            # -------------------------
            try:
                await self._run_before_hooks(context)
            except Exception as e:
                logger.debug(f"Before hooks failed: {e}")

            # -------------------------
            # THINK (LOCK MINIMIZED)
            # -------------------------
            thought = None
            try:
                async with self._llm_lock:
                    thought = await self._safe_think(context)
            except Exception as e:
                logger.warning(f"Thinking failed: {e}")

            if not thought:
                return None

            # -------------------------
            # DECISION (SAFE)
            # -------------------------
            try:
                decision = self._decide(thought)
            except Exception as e:
                logger.warning(f"Decision failed: {e}")
                return None

            if not decision:
                return None

            # -------------------------
            # EXECUTION (SAFE)
            # -------------------------
            result = None
            try:
                result = await self._execute_action(decision)
            except Exception as e:
                logger.warning(f"Execution failed: {e}")
                return None

            # -------------------------
            # LEARNING (NON-BLOCKING FIX)
            # -------------------------
            try:
                asyncio.create_task(self._learn(context, decision, result))
            except Exception:
                pass

            # -------------------------
            # AFTER HOOKS (SAFE)
            # -------------------------
            try:
                await self._run_after_hooks(context, result)
            except Exception as e:
                logger.debug(f"After hooks failed: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Process with context error: {e}")
            return None

    # =========================
    # SAFE THINK
    # =========================

    async def _safe_think(self, context: Dict[str, Any]) -> str:
        """Robust, timeout-safe, non-blocking LLM reasoning"""

        try:
            user_input = str(context.get("user_input", "")).strip()

            if not user_input:
                return ""

            # -------------------------
            # PROMPT (SANITIZED)
            # -------------------------
            prompt = (
                "User Input:\n"
                f"{user_input[:2000]}\n\n"  # prevent prompt explosion
                "Respond ONLY in JSON:\n"
                '{"action":"respond","params":{}}'
            )

            # -------------------------
            # LLM CALL (TIMEOUT SAFE)
            # -------------------------
            result = await asyncio.wait_for(
                self.llm.generate(prompt), timeout=self.input_timeout
            )

            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not result or not isinstance(result, dict):
                return ""

            response = result.get("response")

            if not isinstance(response, str):
                return ""

            return response.strip()

        except asyncio.TimeoutError:
            logger.warning("⏱ Think timeout")

            # Non-blocking event
            try:
                asyncio.create_task(
                    self._emit_event("think_timeout", {"timeout": self.input_timeout})
                )
            except Exception:
                pass

            return ""

        except Exception as e:
            logger.debug(f"Think error: {e}")
            return ""

    # =========================
    # INPUT PROCESSING
    # =========================

    async def _process_input(self) -> Optional[Dict[str, Any]]:
        """Safe, non-blocking, deduplicated input handler"""

        try:
            async with self._input_lock:
                user_input = None
                source = InputSource.TEXT
                listener = getattr(self.system, "listener", None)

                # -------------------------
                # VOICE INPUT (NON-BLOCKING FIX)
                # -------------------------
                try:
                    if listener:
                        if hasattr(listener, "get_last_input"):
                            user_input = await asyncio.to_thread(
                                listener.get_last_input
                            )
                        elif hasattr(listener, "last_input"):
                            user_input = listener.last_input

                        if user_input:
                            source = InputSource.VOICE

                            # Consume safely
                            if hasattr(listener, "clear_input"):
                                await asyncio.to_thread(listener.clear_input)
                            elif hasattr(listener, "last_input"):
                                listener.last_input = None

                except Exception as e:
                    logger.debug(f"Voice input error: {e}")
                    user_input = None

                # -------------------------
                # FALLBACK INPUT
                # -------------------------
                if not user_input and getattr(self, "_latest_input", None):
                    user_input = self._latest_input
                    self._latest_input = None

                if not user_input:
                    return None

                user_input = str(user_input).strip()
                if not user_input:
                    return None

                now = time.monotonic()

                # -------------------------
                # RATE LIMIT
                # -------------------------
                if not await self._check_rate_limit():
                    return None

                # -------------------------
                # DUPLICATE CHECK (IMPROVED)
                # -------------------------
                if user_input == getattr(self, "_last_input", None):
                    return None

                # -------------------------
                # DEBOUNCE
                # -------------------------
                if now - getattr(self, "_last_input_time", 0) < self._input_cooldown:
                    return None

                # -------------------------
                # UPDATE STATE
                # -------------------------
                self._last_input = user_input
                self._last_input_time = now

            # -------------------------
            # BUILD CONTEXT (OUTSIDE LOCK)
            # -------------------------
            context = {
                "time": datetime.utcnow().isoformat(),
                "iteration": getattr(self, "iteration_count", 0),
                "user_input": user_input,
                "source": source,
            }

            # -------------------------
            # NON-BLOCKING EVENT (CRITICAL FIX)
            # -------------------------
            try:
                asyncio.create_task(
                    self._emit_event(
                        "input_processed",
                        {"input": user_input[:100]},
                        source,
                    )
                )
            except Exception:
                pass

            return context

        except Exception as e:
            logger.warning(f"[ProcessInput Error] {e}")
            return None

    # =========================
    # DECISION
    # =========================

    def _decide(self, thought: str) -> Optional[Dict[str, Any]]:
        """Parse decision from thought"""
        try:
            # Try to parse as JSON
            data = json.loads(thought)

            return {
                "action": data.get("action", "respond"),
                "params": data.get("params", {}),
            }

        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return {"action": "respond", "params": {"text": thought}}
        except Exception as e:
            logger.debug(f"Decision parsing error: {e}")
            return None

    # =========================
    # EXECUTE ACTION
    # =========================

    async def _execute_action(
        self, decision: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Safe, non-blocking action execution with timeout + validation"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not isinstance(decision, dict):
                return None

            action = str(decision.get("action", "respond"))
            params = decision.get("params", {})

            if not isinstance(params, dict):
                params = {}

            # -------------------------
            # ROUTER EXECUTION (SAFE)
            # -------------------------
            if self.action_router and hasattr(self.action_router, "route"):
                try:
                    result = self.action_router.route(action, params)

                    # Handle async/sync properly
                    if asyncio.iscoroutine(result):
                        result = await asyncio.wait_for(result, timeout=10)
                    elif callable(result):
                        result = await asyncio.to_thread(result)

                    return result

                except asyncio.TimeoutError:
                    logger.warning(f"⏱ Action '{action}' timed out")
                    return {"status": "timeout", "action": action}

                except Exception as e:
                    logger.warning(f"Action router failed: {e}")
                    return {"status": "error", "action": action, "error": str(e)}

            # -------------------------
            # FALLBACK (SAFE)
            # -------------------------
            return {"status": "executed", "action": action, "params": params}

        except Exception as e:
            logger.debug(f"Execute action error: {e}")
            return None

    # =========================
    # LEARNING
    # =========================

    async def _learn(
        self,
        context: Dict[str, Any],
        decision: Dict[str, Any],
        result: Optional[Dict[str, Any]],
    ):
        """Non-blocking, safe learning storage with validation"""

        try:
            if not self.memory:
                return

            # -------------------------
            # SAFE SERIALIZATION (CRITICAL FIX)
            # -------------------------
            memory_payload = {
                "type": "agent_interaction",
                "context": str(context)[:500],
                "decision": str(decision)[:500],
                "result": str(result)[:500] if result is not None else None,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # -------------------------
            # NON-BLOCKING STORE (CRITICAL FIX)
            # -------------------------
            try:
                store_result = self.memory.store(memory_payload)

                if asyncio.iscoroutine(store_result):
                    await store_result
                else:
                    await asyncio.to_thread(self.memory.store, memory_payload)

            except Exception as e:
                logger.debug(f"Memory store failed: {e}")
                return

            # -------------------------
            # NON-BLOCKING EVENT
            # -------------------------
            try:
                asyncio.create_task(
                    self._emit_event(
                        "learning_stored", {"memory_type": "agent_interaction"}
                    )
                )
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Learning error: {e}")

    # =========================
    # MIDDLEWARE HOOKS
    # =========================

    def add_before_hook(self, hook: Callable):
        """Add hook to run before processing"""
        self._before_process_hooks.append(hook)

    def add_after_hook(self, hook: Callable):
        """Add hook to run after processing"""
        self._after_process_hooks.append(hook)

    def add_error_hook(self, hook: Callable):
        """Add hook to run on error"""
        self._error_hooks.append(hook)

    async def _run_before_hooks(self, context: Dict[str, Any]):
        """Run all before hooks"""
        for hook in self._before_process_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context)
                else:
                    hook(context)
            except Exception as e:
                logger.debug(f"Before hook error: {e}")

    async def _run_after_hooks(
        self, context: Dict[str, Any], result: Optional[Dict[str, Any]]
    ):
        """Run all after hooks"""
        for hook in self._after_process_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(context, result)
                else:
                    hook(context, result)
            except Exception as e:
                logger.debug(f"After hook error: {e}")

    async def _run_error_hooks(self, error: Exception):
        """Run all error hooks"""
        for hook in self._error_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(error)
                else:
                    hook(error)
            except Exception as e:
                logger.debug(f"Error hook error: {e}")

    # =========================
    # PERSISTENCE
    # =========================

    def enable_persistence(self, path: Path):
        """Enable state persistence to disk"""
        self.persist_path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    async def _persist_state(self):
        """Non-blocking, atomic state persistence"""

        if not self.persist_path:
            return

        try:
            now = datetime.utcnow()

            # -------------------------
            # BUILD STATE (SAFE)
            # -------------------------
            state = {
                "metrics": {
                    "total_iterations": self.metrics.total_iterations,
                    "successful_iterations": self.metrics.successful_iterations,
                    "failed_iterations": self.metrics.failed_iterations,
                    "total_inputs_processed": self.metrics.total_inputs_processed,
                    "average_processing_time_ms": self.metrics.average_processing_time_ms,
                },
                "error_count": self.error_count,
                "last_error": self.last_error,
                "timestamp": now.isoformat(),
            }

            # -------------------------
            # ATOMIC WRITE (CRITICAL FIX)
            # -------------------------
            def _write():
                try:
                    temp_path = str(self.persist_path) + ".tmp"

                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)

                    # atomic replace using resolved destination path
                    try:
                        dest = str(self.persist_path)
                    except Exception:
                        dest = str(Path(self.persist_path).resolve())

                    import os

                    os.replace(temp_path, dest)

                except Exception:
                    pass

            # -------------------------
            # NON-BLOCKING FILE WRITE
            # -------------------------
            await asyncio.to_thread(_write)

            self._last_persist = now

        except Exception as e:
            logger.debug(f"Persist state failed: {e}")

    # =========================
    # USER INPUT HANDLER
    # =========================

    async def handle_user_input(self, text: str) -> str:
        """Safe, non-blocking full pipeline user input handler"""

        try:
            # -------------------------
            # VALIDATION
            # -------------------------
            if not text:
                return ""

            text = str(text).strip()
            if not text:
                return ""

            # -------------------------
            # DUPLICATE CHECK
            # -------------------------
            if text == getattr(self, "_last_user_input", None):
                return ""

            self._last_user_input = text

            # -------------------------
            # UI SEND (NON-BLOCKING FIX)
            # -------------------------
            try:
                if hasattr(self.system, "send_ui_message"):
                    await asyncio.to_thread(self.system.send_ui_message, "YOU", text)
            except Exception:
                pass

            # -------------------------
            # INTERRUPT SPEAKER (SAFE)
            # -------------------------
            try:
                speaker = getattr(self.system, "speaker", None)
                if speaker:
                    await asyncio.to_thread(speaker.stop)
            except Exception:
                pass

            # -------------------------
            # PREVENT PARALLEL EXECUTION
            # -------------------------
            if getattr(self, "user_active", False):
                return "Processing previous request..."

            self.user_active = True
            start_time = time.monotonic()

            try:
                decision_engine = getattr(self.system, "decision_engine", None)
                vision_engine = getattr(self.system, "vision_engine_instance", None)
                memory_api = getattr(self.system, "memory_api", None)

                # -------------------------
                # VISION CONTEXT (SAFE)
                # -------------------------
                vision_context = ""
                try:
                    if vision_engine and decision_engine:
                        vision_data = getattr(vision_engine, "current_frame_data", None)
                        if vision_data:
                            vision_context = await asyncio.wait_for(
                                decision_engine.process_vision(vision_data), timeout=2
                            )
                except Exception:
                    vision_context = ""

                # -------------------------
                # MEMORY CONTEXT (SAFE)
                # -------------------------
                memory_context = ""
                try:
                    if memory_api and hasattr(memory_api, "get_recent"):
                        recent = await asyncio.wait_for(
                            memory_api.get_recent(limit=2), timeout=2
                        )
                        if recent:
                            memory_context = f"\nRecent: {recent}"
                except Exception:
                    pass

                # -------------------------
                # BUILD CONTEXT
                # -------------------------
                full_context = f"{vision_context}\n{memory_context}\n\n{text}"

                # -------------------------
                # DECISION + AGENT (SAFE)
                # -------------------------
                if decision_engine and hasattr(decision_engine, "intelligent_decision"):
                    try:
                        decision = await asyncio.wait_for(
                            decision_engine.intelligent_decision(full_context),
                            timeout=5,
                        )
                    except Exception:
                        decision = text

                    response = await asyncio.wait_for(
                        self.r_agent.run(decision), timeout=10
                    )
                else:
                    response = await asyncio.wait_for(
                        self.r_agent.run(text), timeout=10
                    )

            except asyncio.TimeoutError:
                response = "⚠️ Request timed out. Please try again."

            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                response = "⚠️ Error processing request."

            # -------------------------
            # CLEAN RESPONSE
            # -------------------------
            if not response:
                response = "Sorry, I couldn't process that."

            response = str(response).strip()
            response = (
                response.replace("assistant:", "").replace("ASSISTANT:", "").strip()
            )

            # -------------------------
            # UI RESPONSE (NON-BLOCKING)
            # -------------------------
            try:
                if hasattr(self.system, "send_ui_message"):
                    await asyncio.to_thread(self.system.send_ui_message, "AI", response)
            except Exception:
                pass

            # -------------------------
            # SPEAK RESPONSE (SAFE)
            # -------------------------
            try:
                speaker = getattr(self.system, "speaker", None)
                if speaker and hasattr(speaker, "speak_async"):
                    asyncio.create_task(speaker.speak_async(response))
            except Exception:
                pass

            # -------------------------
            # PERFORMANCE LOG
            # -------------------------
            processing_time = (time.monotonic() - start_time) * 1000
            logger.info(f"📊 Pipeline complete: {processing_time:.1f}ms")

            return response

        except Exception as e:
            logger.error(f"User input handler error: {e}")
            return "⚠️ Unexpected error occurred."

        finally:
            self.user_active = False

    # =========================
    # PROCESS COMMANDS (for orchestrator)
    # =========================

    async def process(self, commands: List[Any]) -> Optional[Dict[str, Any]]:
        """Safe, non-blocking command processor with batching + validation"""

        if not commands or not isinstance(commands, list):
            return None

        results = []

        try:
            # -------------------------
            # LIMIT COMMANDS (PROTECTION)
            # -------------------------
            for cmd in commands[:3]:  # prevent overload

                try:
                    response = None

                    # -------------------------
                    # DICT COMMAND
                    # -------------------------
                    if isinstance(cmd, dict):
                        action = str(cmd.get("action", "process"))
                        params = cmd.get("params", {})

                        if not isinstance(params, dict):
                            params = {}

                        if action == "user_input" and "text" in params:
                            response = await asyncio.wait_for(
                                self.handle_user_input(params["text"]), timeout=10
                            )

                    # -------------------------
                    # STRING COMMAND
                    # -------------------------
                    elif isinstance(cmd, str):
                        response = await asyncio.wait_for(
                            self.handle_user_input(cmd), timeout=10
                        )

                    # -------------------------
                    # STORE RESULT
                    # -------------------------
                    if response:
                        results.append({"response": response, "processed": True})

                except asyncio.TimeoutError:
                    results.append(
                        {"response": "⚠️ Command timeout", "processed": False}
                    )

                except Exception as e:
                    logger.debug(f"Command failed: {e}")
                    results.append(
                        {"response": "⚠️ Command failed", "processed": False}
                    )

            # -------------------------
            # FINAL RESULT
            # -------------------------
            if results:
                return {"results": results, "processed": True}

            return {"status": "no_action", "processed": False}

        except Exception as e:
            logger.error(f"Process commands error: {e}")
            return {"status": "error", "processed": False}

    # =========================
    # HEALTH CHECK
    # =========================

    async def health_check(self) -> bool:
        """Robust, fast, non-blocking health check"""

        try:
            # -------------------------
            # CIRCUIT BREAKER
            # -------------------------
            if self._circuit_open:
                return False

            # -------------------------
            # STATE CHECK
            # -------------------------
            if self.state in [LoopState.ERROR]:
                return False

            # -------------------------
            # LLM CHECK (SAFE + LIGHTWEIGHT)
            # -------------------------
            if self.llm and hasattr(self.llm, "generate"):
                try:
                    test_result = await asyncio.wait_for(
                        self.llm.generate("ping"), timeout=3  # lighter than "test"
                    )

                    if test_result is None:
                        return False

                except asyncio.TimeoutError:
                    logger.debug("LLM health check timeout")
                    return False

                except Exception as e:
                    logger.debug(f"LLM health check failed: {e}")
                    return False

            return True

        except Exception as e:
            logger.debug(f"Health check error: {e}")
            return False

    # =========================
    # STATISTICS
    # =========================

    def get_stats(self) -> Dict[str, Any]:
        """Get agent loop statistics"""
        avg_processing_time = sum(self.processing_times) / max(
            1, len(self.processing_times)
        )

        return {
            "state": self.state.value,
            "health": self._health_status,
            "is_running": self.state == LoopState.RUNNING,
            "iteration_count": self.iteration_count,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "user_active": self.user_active,
            "circuit_open": self._circuit_open,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "queue_size": len(self._input_queue),
            "last_error": self.last_error,
            "llm_available": self.llm is not None,
            "memory_available": self.memory is not None,
            "action_router_available": self.action_router is not None,
            "event_subscribers": len(self._event_subscribers),
            "uptime_seconds": (
                (datetime.now() - self._start_time).total_seconds()
                if hasattr(self, "_start_time")
                else 0
            ),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        return {
            "iterations": {
                "total": self.metrics.total_iterations,
                "successful": self.metrics.successful_iterations,
                "failed": self.metrics.failed_iterations,
                "success_rate": (
                    self.metrics.successful_iterations
                    / max(1, self.metrics.total_iterations)
                )
                * 100,
            },
            "inputs": {
                "total_processed": self.metrics.total_inputs_processed,
                "queue_size": self.metrics.queue_size,
            },
            "performance": {
                "average_processing_time_ms": round(
                    self.metrics.average_processing_time_ms, 2
                ),
                "recent_processing_times_ms": (
                    self.metrics.processing_times[-10:]
                    if self.metrics.processing_times
                    else []
                ),
                "uptime_seconds": round(self.metrics.uptime_seconds, 2),
            },
            "errors": self.metrics.error_counts,
            "events": len(self._event_stream),
        }

    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics = LoopMetrics()
        self.processing_times.clear()
        self.error_count = 0
        self.consecutive_errors = 0
        logger.info("Agent loop metrics cleared")

    # =========================
    # CONTROL METHODS
    # =========================

    def stop(self):
        """Stop the agent loop"""
        self.state = LoopState.IDLE
        logger.info("Agent loop stop requested")
        return True

    def pause(self):
        """Pause the agent loop"""
        if self.state == LoopState.RUNNING:
            self.state = LoopState.PAUSED
            logger.info("Agent loop paused")
            return True
        return False

    def resume(self):
        """Resume the agent loop"""
        if self.state == LoopState.PAUSED:
            self.state = LoopState.RUNNING
            logger.info("Agent loop resumed")
            return True
        return False

    async def start(self):
        """Safe start for agent loop (prevents duplicate runs + tracks task)"""

        try:
            # -------------------------
            # PREVENT DUPLICATE START
            # -------------------------
            if getattr(self, "_main_task", None) and not self._main_task.done():
                return True

            if self.state == LoopState.RUNNING:
                return True

            # -------------------------
            # START LOOP (TRACK TASK)
            # -------------------------
            self._main_task = asyncio.create_task(self.run(), name="agent_main_loop")

            return True

        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            return False

    async def initialize(self, *args, **kwargs):
        """Safe initialization with idempotency and state setup"""

        try:
            # -------------------------
            # PREVENT DUPLICATE INIT
            # -------------------------
            if getattr(self, "_initialized", False):
                return True

            self._initialized = True

            # -------------------------
            # INITIAL STATE SETUP
            # -------------------------
            if not hasattr(self, "state"):
                self.state = LoopState.IDLE

            self._start_time = None
            self.last_error = None

            # -------------------------
            # INIT INTERNAL STRUCTURES
            # -------------------------
            if not hasattr(self, "_main_task"):
                self._main_task = None

            if not hasattr(self, "_health_check_task"):
                self._health_check_task = None

            # -------------------------
            # LOG (NON-BLOCKING SAFE)
            # -------------------------
            logger.info("✅ Agent loop initialized")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def shutdown(self):
        """Safe shutdown wrapper with task cleanup and idempotency"""

        try:
            # -------------------------
            # PREVENT DUPLICATE SHUTDOWN
            # -------------------------
            if self.state in [LoopState.SHUTTING_DOWN, LoopState.IDLE]:
                return True

            self.state = LoopState.SHUTTING_DOWN

            # -------------------------
            # CANCEL MAIN TASK (CRITICAL FIX)
            # -------------------------
            if getattr(self, "_main_task", None):
                if not self._main_task.done():
                    self._main_task.cancel()
                    await asyncio.gather(self._main_task, return_exceptions=True)

            # -------------------------
            # INTERNAL CLEANUP
            # -------------------------
            await self._shutdown()

            return True

        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False


# =========================
# PUBLIC API
# =========================

__all__ = ["AgentLoop", "LoopState", "InputSource", "ProcessingEvent", "LoopMetrics"]
