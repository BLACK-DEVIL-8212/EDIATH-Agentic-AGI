"""
TaskTracker - Production Task Management with Fire-and-Forget for API Leaks

Prevents asyncio.create_task leaks by tracking + auto-cleanup.
Fire-and-forget tracked_task() for logging/metrics - won't block main flow.
"""

import asyncio
from typing import Any
from weakref import WeakSet
import logging

logger = logging.getLogger(__name__)


class TaskTracker:
    """Global task tracker with auto-cleanup"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.active_tasks = WeakSet()
        self._is_shutting_down = False
        self.pending_shutdown = None

        # Created lazily inside a running event loop. Creating tasks at import
        # time breaks modules that import this registry during normal startup.
        self._cleanup_loop = None

    def _ensure_cleanup_loop(self):
        """Start cleanup only when an event loop is already running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self.pending_shutdown is None:
            self.pending_shutdown = asyncio.Event()

        if self._cleanup_loop is None or self._cleanup_loop.done():
            self._cleanup_loop = loop.create_task(
                self._background_cleanup(), name="task_tracker_cleanup"
            )

    async def _background_cleanup(self):
        """Clean up completed tasks periodically"""
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(5.0)
                # WeakSet auto-cleans dead refs, but log periodically
                if len(self.active_tasks) > 1000:
                    logger.warning(f"High task count: {len(self.active_tasks)}")
            except asyncio.CancelledError:
                break

    def create_task(self, coro: Any, *, name: str = None, **kwargs) -> asyncio.Task:
        """Create tracked task"""
        if self._is_shutting_down:
            if hasattr(coro, "close"):
                coro.close()
            raise RuntimeError("Cannot create tasks during shutdown")

        self._ensure_cleanup_loop()
        task = asyncio.create_task(coro, name=name, **kwargs)
        self.active_tasks.add(task)

        def _task_done_cb(t):
            try:
                self.active_tasks.discard(t)
            except Exception:
                pass

        task.add_done_callback(_task_done_cb)
        return task

    async def cancel_all_tasks(self):
        """Cancel all tracked tasks"""
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        if self.pending_shutdown is not None:
            self.pending_shutdown.set()

        # Cancel cleanup loop
        if hasattr(self, "_cleanup_loop") and self._cleanup_loop:
            self._cleanup_loop.cancel()
            try:
                if self._cleanup_loop.get_loop() is asyncio.get_running_loop():
                    await self._cleanup_loop
            except asyncio.CancelledError:
                pass

        cancelled = 0
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
                cancelled += 1

        # Wait for completion
        pending = [t for t in self.active_tasks if not t.done()]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for {len(pending)} tasks")

        logger.info(f"Cancelled {cancelled} tracked tasks")

    def initiate_shutdown(self):
        """Initiate shutdown sequence"""
        self._is_shutting_down = True
        if self.pending_shutdown is not None:
            self.pending_shutdown.set()

    def is_shutting_down(self) -> bool:
        """Check if shutdown initiated"""
        return self._is_shutting_down


# Global instance
task_registry = TaskTracker()


async def tracked_task(coro: Any, *, name: str = None, **kwargs) -> None:
    """
    Fire-and-forget task for logging/metrics.
    Won't block calling code. Perfect for API logging.
    """
    try:
        task = task_registry.create_task(coro, name=name or "tracked", **kwargs)
        # Don't await - fire and forget
    except Exception as e:
        logger.debug(f"Failed to create tracked task: {e}")


__all__ = ["task_registry", "tracked_task"]
