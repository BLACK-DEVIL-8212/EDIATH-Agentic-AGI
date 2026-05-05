"""
Global Shutdown Manager - Graceful System Termination
Handles coordinated shutdown across multiple asyncio loops/threads
Integrates with task_registry for complete cleanup
"""

import asyncio
import signal
import logging
import threading
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import asynccontextmanager

from .task_registry import task_registry

logger = logging.getLogger(__name__)


class ShutdownManager:
    """
    Global shutdown coordinator for multi-loop asyncio systems.

    Features:
    - Distributed shutdown flag (Event + atomic state)
    - Multi-signal handling (SIGINT, SIGTERM, SIGQUIT)
    - Timeout-aware graceful shutdown
    - Cross-thread/loop notification
    - Integration with GlobalTaskRegistry
    """

    _instance: Optional["ShutdownManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        # Core shutdown state
        self._shutdown_requested = threading.Event()
        self._shutdown_complete = threading.Event()
        self._shutdown_start_time: Optional[float] = None
        self._shutdown_timeout = 30.0  # 30s max graceful shutdown

        # Signal tracking
        self._signals_received: List[int] = []
        self._original_handlers: Dict[int, callable] = {}
        self._signals_installed = False

        # Cancellation groups
        self._cancel_groups: Dict[str, List[asyncio.Task]] = {}

        # Stats
        self.shutdown_phase = "idle"
        self._shutdown_stats = {
            "requested_at": None,
            "completed_at": None,
            "duration": None,
            "tasks_cancelled": 0,
            "tasks_remaining": 0,
            "timeout": False,
        }

        self._initialized = True
        logger.info("ShutdownManager initialized")

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_requested.is_set()

    @property
    def shutdown_complete(self) -> bool:
        """Check if shutdown sequence is fully complete"""
        return self._shutdown_complete.is_set()

    def request_shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Request immediate shutdown (thread-safe, loop-safe).
        """
        try:
            if self._shutdown_requested.is_set():
                logger.debug("Shutdown already requested")
                return

            # Safe timeout update
            if timeout is not None:
                self._shutdown_timeout = float(timeout)

            # Set flag FIRST (atomic)
            self._shutdown_requested.set()
            self.shutdown_phase = "requested"

            # Safe time handling (NO event loop dependency)
            try:
                loop = asyncio.get_running_loop()
                self._shutdown_start_time = loop.time()
            except RuntimeError:
                # No running loop (thread / shutdown phase)
                import time

                self._shutdown_start_time = time.time()

            # Stats
            self._shutdown_stats["requested_at"] = datetime.utcnow().isoformat()

            logger.critical(
                f"🛑 Shutdown requested (timeout: {self._shutdown_timeout}s)"
            )

        except Exception as e:
            logger.error(f"❌ request_shutdown failed: {e}")

    async def graceful_shutdown(
        self,
        cancel_tasks: bool = True,
        await_completion: bool = True,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Safe, deterministic graceful shutdown (thread-safe + async-safe).
        """

        try:
            if self._shutdown_complete.is_set():
                logger.info("Shutdown already complete")
                return True

            # -------------------------
            # TIME HANDLING (SAFE)
            # -------------------------
            timeout = float(timeout or self._shutdown_timeout)
            start_time = asyncio.get_running_loop().time()

            self.shutdown_phase = "executing"

            logger.info(f"🔄 Starting graceful shutdown (timeout: {timeout}s)")

            # -------------------------
            # PHASE 1: FLAG ONLY (NO SIGNAL INSTALL HERE)
            # -------------------------
            self.request_shutdown(timeout)

            success = True

            # -------------------------
            # PHASE 2: CANCEL TASKS
            # -------------------------
            if cancel_tasks:
                try:
                    cancelled = await asyncio.wait_for(
                        self.cancel_all_tasks(), timeout=min(10.0, timeout)
                    )
                    cancelled_count = (
                        len(cancelled) if isinstance(cancelled, list) else cancelled
                    )
                    self._shutdown_stats["tasks_cancelled"] = cancelled_count
                    logger.info(f"Cancelled {cancelled_count} tasks")
                except asyncio.TimeoutError:
                    logger.warning("⚠ Task cancellation timeout")
                    success = False
                except Exception as e:
                    logger.error(f"Cancel error: {e}")
                    success = False

            # -------------------------
            # PHASE 3: AWAIT TASK COMPLETION
            # -------------------------
            if await_completion:
                remaining_time = max(
                    1.0, timeout - (asyncio.get_running_loop().time() - start_time)
                )

                try:
                    done = await asyncio.wait_for(
                        self.await_all_tasks(), timeout=remaining_time
                    )
                    success = success and bool(done)
                except asyncio.TimeoutError:
                    logger.warning("⚠ Await tasks timeout")
                    success = False
                except Exception as e:
                    logger.error(f"Await error: {e}")
                    success = False

            # -------------------------
            # PHASE 4: FINAL STATE
            # -------------------------
            self._shutdown_complete.set()
            self.shutdown_phase = "complete"

            # -------------------------
            # STATS (SAFE TIME)
            # -------------------------
            import time

            duration = time.time() - (self._shutdown_start_time or time.time())

            self._shutdown_stats["duration"] = duration
            self._shutdown_stats["completed_at"] = datetime.utcnow().isoformat()
            self._shutdown_stats["tasks_remaining"] = task_registry.get_active_tasks()
            self._shutdown_stats["timeout"] = not success

            logger.info(f"✅ Shutdown complete ({duration:.2f}s, success: {success})")

            return success

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

            # FORCE COMPLETE (prevent infinite loop)
            self._shutdown_complete.set()
            self.shutdown_phase = "failed"

            return False

    async def cancel_all_tasks(self, timeout: float = 5.0) -> int:
        """Cancel all tracked tasks via GlobalTaskRegistry"""
        cancelled = await task_registry.cancel_all_tasks(timeout=timeout)
        return len(cancelled)

    async def await_all_tasks(self, timeout: float = 10.0) -> bool:
        """Wait for all tasks to complete"""
        return await task_registry.await_all_tasks(timeout=timeout)

    def get_shutdown_stats(self) -> Dict[str, Any]:
        """Get shutdown statistics (thread-safe, loop-safe)"""
        try:
            import time

            # -------------------------
            # SAFE TIME CALCULATION
            # -------------------------
            if self._shutdown_start_time:
                now = time.time()
                elapsed = now - self._shutdown_start_time
                remaining = max(0.0, self._shutdown_timeout - elapsed)
            else:
                remaining = self._shutdown_timeout

            return {
                **self._shutdown_stats,
                "phase": self.shutdown_phase,
                "is_shutting_down": self.is_shutting_down,
                "shutdown_complete": self.shutdown_complete,
                "active_tasks": task_registry.get_active_tasks(),
                "timeout_remaining": remaining,
            }

        except Exception as e:
            logger.error(f"❌ get_shutdown_stats failed: {e}")
            return {"phase": "error", "timeout_remaining": 0.0, "active_tasks": 0}

    def _install_signal_handlers(self) -> None:
        """Install signal handlers (thread-safe, no async calls)"""

        if self._signals_installed:
            return

        def signal_handler(signum, frame):
            try:
                logger.warning(f"Signal {signum} received")

                # Track signal safely
                self._signals_received.append(signum)

                # ONLY trigger shutdown flag (NO asyncio here)
                self.request_shutdown()

            except Exception as e:
                logger.error(f"Signal handler error: {e}")

        # -------------------------
        # INSTALL SIGNALS (MAIN THREAD ONLY)
        # -------------------------
        if threading.current_thread() is not threading.main_thread():
            logger.warning("⚠ Signal handlers can only be installed in main thread")
            return

        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)

        for signum in signals:
            try:
                self._original_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, signal_handler)
            except Exception as e:
                logger.warning(f"⚠ Failed to install signal {signum}: {e}")

        self._signals_installed = True
        logger.info("Signal handlers installed safely")

    def restore_signal_handlers(self):
        """Restore original signal handlers"""
        for signum, handler in self._original_handlers.items():
            signal.signal(signum, handler)
        self._signals_installed = False
        logger.debug("Signal handlers restored")


# Global convenience instance
shutdown_manager = ShutdownManager()


# Async context manager for RAII-style shutdown
@asynccontextmanager
async def managed_shutdown(timeout: float = 30.0):
    """Context manager: auto graceful shutdown on exit"""
    try:
        yield shutdown_manager
    finally:
        if not shutdown_manager.shutdown_complete:
            await shutdown_manager.graceful_shutdown(timeout=timeout)


# Convenience functions
async def check_shutdown() -> bool:
    """Check if shutdown requested (used in loops)"""
    return not shutdown_manager.is_shutting_down


def immediate_shutdown():
    """Emergency shutdown (no graceful wait)"""
    shutdown_manager.request_shutdown(5.0)


__all__ = [
    "ShutdownManager",
    "shutdown_manager",
    "check_shutdown",
    "managed_shutdown",
    "immediate_shutdown",
]
