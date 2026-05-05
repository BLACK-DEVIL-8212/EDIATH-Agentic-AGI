"""
Global Task Registry - Graceful Shutdown Solution
Tracks all asyncio tasks across multiple event loops/threads
Provides coordinated shutdown for entire EDIATH system
"""

import asyncio
import weakref
from typing import Dict, List, Optional, Set
from threading import Lock, Event
import logging
import signal

logger = logging.getLogger(__name__)


class GlobalTaskRegistry:
    """
    Singleton registry tracking ALL asyncio tasks across multiple event loops.

    Handles:
    - Cross-thread task cancellation
    - Graceful shutdown coordination
    - Signal handling (SIGINT/SIGTERM)
    - Timeout protection
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Global shutdown coordination
        self.shutdown_event = Event()
        self.shutdown_complete = Event()
        self.shutdown_timeout = 30.0  # Max 30s graceful shutdown

        # Task tracking by event loop
        self._tasks_by_loop: Dict[str, Set[asyncio.Task]] = {}
        self._loop_refs: Dict[str, weakref.ref[asyncio.AbstractEventLoop]] = {}
        self._loop_lock = Lock()

        # Signal handlers
        self._original_handlers = {}
        self._signals_installed = False

        # Stats
        self.total_tasks_created = 0
        self.shutdown_start_time = None

        self._initialized = True
        logger.info("GlobalTaskRegistry initialized")

    def register_task(self, task: asyncio.Task, loop_id: Optional[str] = None) -> None:
        """Register a task with optional custom loop identifier"""
        if task.done():
            return

        loop = task.get_loop()
        loop_id = loop_id or id(loop)

        with self._loop_lock:
            if loop_id not in self._tasks_by_loop:
                self._tasks_by_loop[loop_id] = set()
                self._loop_refs[loop_id] = weakref.ref(loop)

            self._tasks_by_loop[loop_id].add(task)
            self.total_tasks_created += 1

        logger.debug(f"Task registered: {task} in loop {loop_id}")

    def unregister_task(
        self, task: asyncio.Task, loop_id: Optional[str] = None
    ) -> None:
        """Unregister completed/cancelled task"""
        loop_id = loop_id or id(task.get_loop())

        with self._loop_lock:
            if loop_id in self._tasks_by_loop:
                self._tasks_by_loop[loop_id].discard(task)

    def get_active_tasks(self) -> int:
        """Count all active registered tasks"""
        total = 0
        with self._loop_lock:
            for tasks in self._tasks_by_loop.values():
                total += len(tasks)
        return total

    async def cancel_all_tasks(self, timeout: float = 5.0) -> List[asyncio.Task]:
        """Cancel all registered tasks across all loops"""
        pending_tasks = []

        logger.info(f"Cancelling {self.get_active_tasks()} tasks...")

        with self._loop_lock:
            for loop_id, tasks in list(self._tasks_by_loop.items()):
                loop_ref = self._loop_refs.get(loop_id)
                loop = loop_ref() if loop_ref else None

                if not loop or loop.is_closed():
                    logger.debug(f"Skipping closed loop {loop_id}")
                    continue

                for task in list(tasks):
                    try:
                        if not task.done():
                            if loop.is_running():
                                # Cross-thread safe cancellation
                                def _cancel():
                                    task.cancel()

                                loop.call_soon_threadsafe(_cancel)
                                pending_tasks.append(task)
                            else:
                                task.cancel()
                                pending_tasks.append(task)
                    except Exception as e:
                        logger.debug(f"Task cancel failed {task}: {e}")

        # Wait for cancellation with timeout
        if pending_tasks:
            try:
                await asyncio.wait(
                    pending_tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
                )
            except asyncio.TimeoutError:
                logger.warning(f"Task cancellation timeout after {timeout}s")

        logger.info(f"Cancelled {len(pending_tasks)} tasks")
        return pending_tasks

    async def await_all_tasks(self, timeout: float = 10.0) -> bool:
        """Wait for all tasks to complete or timeout"""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            active = self.get_active_tasks()
            if active == 0:
                logger.info("All tasks completed")
                return True

            logger.debug(f"Waiting for {active} tasks to complete...")
            await asyncio.sleep(0.1)

        logger.warning(
            f"Task await timeout after {timeout}s ({self.get_active_tasks()} remaining)"
        )
        return False

    def initiate_shutdown(self, timeout: float = 30.0) -> None:
        """Initiate graceful shutdown sequence"""
        if self.shutdown_event.is_set():
            logger.warning("Shutdown already in progress")
            return

        self.shutdown_event.set()
        self.shutdown_start_time = asyncio.get_event_loop().time()

        logger.info(f"Initiating graceful shutdown (timeout: {timeout}s)")

        # Install signal handlers if not already
        self._install_signal_handlers()

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated"""
        return self.shutdown_event.is_set()

    async def graceful_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Execute complete graceful shutdown sequence"""
        if timeout is None:
            timeout = self.shutdown_timeout

        self.initiate_shutdown(timeout)

        # Phase 1: Cancel all tasks
        await self.cancel_all_tasks(timeout=10.0)

        # Phase 2: Wait for completion
        success = await self.await_all_tasks(timeout=10.0)

        # Phase 3: Signal completion
        self.shutdown_complete.set()

        elapsed = asyncio.get_event_loop().time() - self.shutdown_start_time
        logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")

        return success

    def _install_signal_handlers(self):
        """Install SIGINT/SIGTERM handlers"""
        if self._signals_installed:
            return

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.graceful_shutdown())

        # Store original handlers
        self._original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        self._original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)

        # Install handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._signals_installed = True
        logger.info("Signal handlers installed")

    def restore_signal_handlers(self):
        """Restore original signal handlers"""
        for signum, handler in self._original_handlers.items():
            signal.signal(signum, handler)
        self._signals_installed = False

    def get_stats(self) -> Dict:
        """Get registry statistics"""
        return {
            "active_tasks": self.get_active_tasks(),
            "total_created": self.total_tasks_created,
            "shutting_down": self.is_shutting_down(),
            "shutdown_complete": self.shutdown_complete.is_set(),
            "loop_count": len(self._tasks_by_loop),
        }


# Global convenience instance
task_registry = GlobalTaskRegistry()


# Context manager for easy task registration
class tracked_task:
    """Context manager: auto-register/unregister tasks"""

    def __init__(self, loop_id: Optional[str] = None, name: Optional[str] = None):
        self.loop_id = loop_id
        self.name = name
        self.task = None

    async def __aenter__(self):
        self.task = asyncio.current_task()
        if self.task:
            task_registry.register_task(self.task, self.loop_id)
            if self.name:
                self.task.set_name(self.name)
        return self.task

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            task_registry.unregister_task(self.task, self.loop_id)


# Auto-register decorator
def tracked(loop_id: Optional[str] = None, name: Optional[str] = None):
    """Decorator: auto-register async functions"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            coro_task = asyncio.current_task()
            task_registry.register_task(coro_task, loop_id)
            if name:
                coro_task.set_name(name)
            try:
                return await func(*args, **kwargs)
            finally:
                task_registry.unregister_task(coro_task, loop_id)

        return wrapper

    return decorator


__all__ = ["GlobalTaskRegistry", "task_registry", "tracked_task", "tracked"]


def create_task(
    coro, name: Optional[str] = None, loop_id: Optional[str] = None
) -> Optional[asyncio.Task]:
    """
    Convenience wrapper to create an asyncio.Task and register it with the global
    `task_registry`. Ensures tasks are tracked for graceful shutdown and
    automatically unregistered when done.
    """
    try:
        task = asyncio.create_task(coro)
        if name and hasattr(task, "set_name"):
            try:
                task.set_name(name)
            except Exception:
                pass

        try:
            task_registry.register_task(task, loop_id)
        except Exception:
            pass

        def _done(t: asyncio.Task):
            try:
                task_registry.unregister_task(t, loop_id)
            except Exception:
                pass

        try:
            task.add_done_callback(_done)
        except Exception:
            pass

        return task
    except Exception:
        return None
