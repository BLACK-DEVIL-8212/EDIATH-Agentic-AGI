"""
Advanced Safety Layer for EDIATH (AI Protection System)
"""

import asyncio
import time

from typing import Callable, Any, Dict, Optional, List
from datetime import datetime
from enum import Enum

from ..utils.logger import logger


class SafetyViolation(Exception):
    """
    Production-grade safety violation exception
    """

    def __init__(
        self,
        message: str,
        code: str = "SAFETY_ERROR",
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Production-grade SafetyViolation initialization
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not message or not isinstance(message, str):
                message = "Unknown safety violation"

            if not isinstance(code, str) or not code:
                code = "SAFETY_ERROR"

            if not isinstance(severity, str):
                severity = "medium"

            # normalize severity
            severity = severity.strip().lower()
            allowed = {"low", "medium", "high", "critical"}

            if severity not in allowed:
                severity = "medium"

            if context is None:
                context = {}
            elif not isinstance(context, dict):
                context = {"raw": str(context)}

            # ------------------------
            # 🔥 BASE INIT
            # ------------------------
            super().__init__(message)

            # ------------------------
            # 🔥 ASSIGN
            # ------------------------
            self.message = message
            self.code = code.strip().upper()
            self.severity = severity
            self.context = context

            # ------------------------
            # 🔥 TIMESTAMP
            # ------------------------
            try:
                self.timestamp = datetime.utcnow()
            except Exception:
                self.timestamp = None

            # ------------------------
            # 🔥 METADATA
            # ------------------------
            self.error_id = f"{self.code}-{int(time.time() * 1000)}"

        except Exception as e:
            # ------------------------
            # 🔥 HARD FAILSAFE
            # ------------------------
            super().__init__("Critical safety initialization failure")

            self.message = "Critical safety initialization failure"
            self.code = "CRITICAL_ERROR"
            self.severity = "critical"
            self.context = {"init_error": str(e)}
            self.timestamp = None
            self.error_id = "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize SafetyViolation safely (production-grade)
        """

        try:
            # ------------------------
            # 🔥 SAFE TIMESTAMP
            # ------------------------
            try:
                timestamp = (
                    self.timestamp.isoformat()
                    if getattr(self, "timestamp", None)
                    else None
                )
            except Exception:
                timestamp = None

            # ------------------------
            # 🔥 SAFE CONTEXT
            # ------------------------
            context = getattr(self, "context", {})
            if not isinstance(context, dict):
                context = {"raw": str(context)}

            # ------------------------
            # 🔥 BUILD RESULT
            # ------------------------
            data = {
                "message": getattr(self, "message", "Unknown error"),
                "code": getattr(self, "code", "SAFETY_ERROR"),
                "severity": getattr(self, "severity", "medium"),
                "context": context,
                "timestamp": timestamp,
            }

            # ------------------------
            # 🔥 OPTIONAL METADATA
            # ------------------------
            try:
                if hasattr(self, "error_id"):
                    data["error_id"] = self.error_id
            except Exception:
                pass

            return data

        except Exception as e:
            # ------------------------
            # 🔥 HARD FAILSAFE
            # ------------------------
            try:
                logger.error(f"❌ to_dict failed: {e}")
            except Exception:
                pass

            return {
                "message": "Serialization failed",
                "code": "SERIALIZATION_ERROR",
                "severity": "critical",
                "context": {"error": str(e)},
                "timestamp": None,
            }

    def __str__(self) -> str:
        """
        Human-readable error string (production-grade)
        """
        try:
            code = getattr(self, "code", "SAFETY_ERROR")
            message = getattr(self, "message", "Unknown error")
            severity = getattr(self, "severity", "medium")

            return f"[{code}] ({severity}) {message}"

        except Exception:
            return "[SAFETY_ERROR] (unknown) Serialization failure"

    def __repr__(self) -> str:
        """
        Debug representation (production-grade)
        """
        try:
            code = getattr(self, "code", "SAFETY_ERROR")
            message = getattr(self, "message", "Unknown error")
            severity = getattr(self, "severity", "medium")

            error_id = getattr(self, "error_id", "N/A")

            return (
                f"<SafetyViolation "
                f"id={error_id} "
                f"code={code} "
                f"severity={severity} "
                f"message='{message}'>"
            )

        except Exception:
            return "<SafetyViolation INVALID>"


class SafetyManager:
    """
    Central safety validation system
    """

    def __init__(self):
        """
        Production-grade SafetyManager initialization
        """

        import threading
        import time

        try:
            # ------------------------
            # 🔥 RULE STORAGE
            # ------------------------
            self.rules: Dict[str, Callable] = {}

            # ------------------------
            # 🔥 VIOLATION TRACKING
            # ------------------------
            self.violation_count = 0
            self.last_violation: Optional[SafetyViolation] = None
            self.violation_history: List[Dict[str, Any]] = []

            self.max_history = 500  # prevent memory overflow

            # ------------------------
            # 🔥 STATE FLAGS
            # ------------------------
            self.enabled = True
            self.strict_mode = False  # if True → fail fast

            # ------------------------
            # 🔥 THREAD SAFETY
            # ------------------------
            self._lock = threading.Lock()

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            self.validation_count = 0
            self.execution_count = 0
            self.blocked_count = 0

            self.created_at = time.time()
            self.last_check = None

            # ------------------------
            # 🔥 PERFORMANCE CONFIG
            # ------------------------
            self.max_rules = 1000

            # ------------------------
            # 🔥 ERROR TRACKING
            # ------------------------
            self.error_count = 0
            self.last_error: Optional[str] = None

            # ------------------------
            # 🔥 LOGGING
            # ------------------------
            try:
                logger.info("🛡️ SafetyManager initialized")
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ SafetyManager init failed: {e}")
            except Exception:
                pass

            raise RuntimeError(f"SafetyManager initialization failed: {e}")

    # ------------------------
    # 🔥 REGISTER RULE
    # ------------------------
    def register_rule(self, name: str, func: Callable, overwrite: bool = False) -> bool:
        """
        Register validation rule safely (production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not isinstance(name, str):
                raise ValueError("Invalid rule name")

            if not callable(func):
                raise ValueError("Rule must be callable")

            name = name.strip().lower()

            # ------------------------
            # 🔥 LIMIT CHECK
            # ------------------------
            max_rules = getattr(self, "max_rules", 1000)

            if len(self.rules) >= max_rules and name not in self.rules:
                try:
                    logger.warning("⚠ Max rules limit reached")
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 THREAD-SAFE REGISTRATION
            # ------------------------
            lock = getattr(self, "_lock", None)

            if lock:
                lock.acquire()

            try:
                # ------------------------
                # 🔥 DUPLICATE CHECK
                # ------------------------
                if name in self.rules and not overwrite:
                    try:
                        logger.warning(f"⚠ Rule already exists: {name}")
                    except Exception:
                        pass
                    return False

                # ------------------------
                # 🔥 REGISTER
                # ------------------------
                self.rules[name] = func

            finally:
                if lock:
                    lock.release()

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            try:
                self.rule_count = len(self.rules)
            except Exception:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                logger.debug(f"🛡️ Rule registered: {name}")
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
                logger.error(f"❌ Rule registration failed: {name} - {e}")
            except Exception:
                pass

            return False

    # ------------------------
    # 🔥 VALIDATE INPUT
    # ------------------------
    def validate(self, name: str, value: Any) -> bool:
        """
        Validate input against registered rule (production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not isinstance(name, str):
                raise ValueError("Invalid rule name")

            name = name.strip().lower()

            # ------------------------
            # 🔥 CHECK ENABLED
            # ------------------------
            if not getattr(self, "enabled", True):
                return True

            # ------------------------
            # 🔥 RULE EXISTS?
            # ------------------------
            rule = self.rules.get(name)

            if not rule:
                return True  # no rule = allow

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            try:
                self.validation_count = getattr(self, "validation_count", 0) + 1
                self.last_check = time.time()
            except Exception:
                pass

            # ------------------------
            # 🔥 EXECUTE RULE (THREAD SAFE)
            # ------------------------
            lock = getattr(self, "_lock", None)

            if lock:
                lock.acquire()

            try:
                result = rule(value)
            finally:
                if lock:
                    lock.release()

            # ------------------------
            # 🔥 RESULT CHECK
            # ------------------------
            if result is False:
                raise SafetyViolation(
                    message=f"Validation failed for {name}",
                    code="VALIDATION_FAILED",
                    severity="high",
                    context={"value": value, "rule": name},
                )

            return True

        except SafetyViolation as e:
            try:
                self.blocked_count = getattr(self, "blocked_count", 0) + 1
            except:
                pass

            self._log_violation(e)

            if getattr(self, "strict_mode", False):
                raise

            return False

        except Exception as e:
            violation = SafetyViolation(
                message=str(e),
                code="VALIDATION_ERROR",
                severity="critical",
                context={"rule": name},
            )

            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
            except:
                pass

            self._log_violation(violation)

            if getattr(self, "strict_mode", False):
                raise violation

            return False

    # ------------------------
    # 🔥 SAFE EXECUTION
    # ------------------------
    def safe_execute(
        self, func: Callable, *args, timeout: Optional[float] = None, **kwargs
    ) -> Any:
        """
        Execute function safely (production-grade)
        """

        import time
        import asyncio

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if func is None or not callable(func):
                raise ValueError("Invalid function")

            if not getattr(self, "enabled", True):
                return func(*args, **kwargs)

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            try:
                self.execution_count = getattr(self, "execution_count", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 EXECUTION
            # ------------------------
            if asyncio.iscoroutinefunction(func):

                async def _run():
                    return await func(*args, **kwargs)

                if timeout:
                    result = asyncio.run(asyncio.wait_for(_run(), timeout))
                else:
                    result = asyncio.run(_run())

            else:
                # sync execution
                if timeout:

                    async def _wrap():
                        return await asyncio.wait_for(
                            asyncio.to_thread(func, *args, **kwargs), timeout
                        )

                    result = asyncio.run(_wrap())
                else:
                    result = func(*args, **kwargs)

            # ------------------------
            # 🔥 SUCCESS METRICS
            # ------------------------
            try:
                self.success_count = getattr(self, "success_count", 0) + 1
            except Exception:
                pass

            return result

        except SafetyViolation:
            raise

        except asyncio.TimeoutError:
            violation = SafetyViolation(
                message="Execution timeout", code="TIMEOUT", severity="high"
            )

            self._log_violation(violation)

            try:
                self.blocked_count = getattr(self, "blocked_count", 0) + 1
            except Exception:
                pass

            raise violation

        except Exception as e:
            violation = SafetyViolation(
                message=str(e),
                code="EXECUTION_ERROR",
                severity="high",
                context={"function": getattr(func, "__name__", "unknown")},
            )

            self._log_violation(violation)

            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
            except Exception:
                pass

            raise violation

        finally:
            # ------------------------
            # 🔥 EXECUTION TIME
            # ------------------------
            try:
                elapsed = time.time() - start
                self.last_execution_time = round(elapsed, 4)
            except Exception:
                pass

    # ------------------------
    # 🔥 LOG VIOLATION
    # ------------------------
    def _log_violation(self, violation: SafetyViolation):
        try:
            self.violation_count += 1
            self.last_violation = violation

            logger.error(f"🚨 Safety violation: {violation}")

        except Exception:
            pass


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RateLimiter:
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def is_allowed(self) -> bool:
        now = datetime.now()
        self.calls = [
            t for t in self.calls if (now - t).total_seconds() < self.time_window
        ]

        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False


class SafetyLayer:
    def __init__(self):
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.execution_timeouts: Dict[str, float] = {}
        self.allowed_operations: set = set()

        self.violation_log = []
        self.enabled = True

    # ------------------------
    # PERMISSIONS 🔒
    # ------------------------
    def allow_operation(self, operation: str):
        self.allowed_operations.add(operation)

    def check_permission(self, operation: str):
        if self.allowed_operations and operation not in self.allowed_operations:
            self._log_violation(
                f"Unauthorized operation: {operation}", Severity.CRITICAL
            )
            raise SafetyViolation(f"Operation not allowed: {operation}")

    # ------------------------
    # RATE LIMIT
    # ------------------------
    def register_rate_limit(
        self, operation: str, max_calls: int, time_window: int = 60
    ):
        self.rate_limiters[operation] = RateLimiter(max_calls, time_window)

    def check_rate_limit(self, operation: str):
        limiter = self.rate_limiters.get(operation)
        if limiter and not limiter.is_allowed():
            self._log_violation(f"Rate limit exceeded: {operation}", Severity.MEDIUM)
            raise SafetyViolation(f"Rate limit exceeded: {operation}")

    # ------------------------
    # TIMEOUT 🔥
    # ------------------------
    def set_execution_timeout(self, operation: str, timeout: float):
        self.execution_timeouts[operation] = timeout

    async def _run_with_timeout(self, func, timeout, *args, **kwargs):
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

    # ------------------------
    # SAFE EXECUTION DECORATOR 🔥
    # ------------------------
    def safe_execution(self, operation: str):
        def decorator(func: Callable):

            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)

                self.check_permission(operation)
                self.check_rate_limit(operation)

                timeout = self.execution_timeouts.get(operation)

                try:
                    if timeout:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout)
                    else:
                        result = await func(*args, **kwargs)

                    return result

                except asyncio.TimeoutError:
                    self._log_violation(f"Timeout: {operation}", Severity.HIGH)
                    raise SafetyViolation(f"Timeout in {operation}")

                except Exception as e:
                    self._log_violation(f"Error in {operation}: {e}", Severity.MEDIUM)
                    raise

            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                self.check_permission(operation)
                self.check_rate_limit(operation)

                start = datetime.now()

                try:
                    result = func(*args, **kwargs)

                    elapsed = (datetime.now() - start).total_seconds()
                    timeout = self.execution_timeouts.get(operation)

                    if timeout and elapsed > timeout:
                        raise SafetyViolation(f"Timeout in {operation}")

                    return result

                except Exception as e:
                    self._log_violation(f"Error in {operation}: {e}", Severity.MEDIUM)
                    raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    # ------------------------
    # CONTROL
    # ------------------------
    def enable(self):
        self.enabled = True
        logger.info("Safety layer enabled")

    def disable(self):
        self.enabled = False
        logger.warning("⚠ Safety layer disabled")

    # ------------------------
    # LOGGING
    # ------------------------
    def _log_violation(self, message: str, severity: Severity):
        entry = {
            "message": message,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
        }

        self.violation_log.append(entry)
        logger.warning(f"[{severity.value.upper()}] {message}")

    def get_violations(self, limit=10):
        return self.violation_log[-limit:]


# Global instance
safety = SafetyLayer()
