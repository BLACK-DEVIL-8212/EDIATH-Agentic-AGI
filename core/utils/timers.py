"""Timer and scheduling utilities for EDIATH."""

import asyncio
import time
from typing import Callable, Optional
from datetime import datetime
from threading import Thread, Event
import uuid


class Timer:
    """Simple timer for measuring elapsed time."""

    def __init__(self, name: str = "Timer"):
        """Initialize timer."""
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.paused_time: float = 0
        self.is_paused = False

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.is_paused = False
        self.paused_time = 0

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started")

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time - self.paused_time
        return elapsed

    def pause(self):
        """Pause the timer."""
        if self.is_paused or self.start_time is None:
            return
        self.is_paused = True
        self.pause_time = time.time()

    def resume(self):
        """Resume the paused timer."""
        if not self.is_paused:
            return
        self.paused_time += time.time() - self.pause_time
        self.is_paused = False

    def elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time - self.paused_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.paused_time = 0
        self.is_paused = False

    def __str__(self) -> str:
        elapsed = self.elapsed()
        minutes, seconds = divmod(int(elapsed), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{self.name}: {hours:02d}:{minutes:02d}:{seconds:02d}"


class Timeout:
    """Context manager for timeout handling."""

    def __init__(self, seconds: float, message: str = "Operation timed out"):
        """Initialize timeout."""
        self.seconds = seconds
        self.message = message
        self.timer: Optional[Timer] = None

    def __enter__(self):
        """Enter context manager."""
        self.timer = Timer("Timeout")
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.timer:
            elapsed = self.timer.stop()
            if elapsed > self.seconds:
                raise TimeoutError(self.message)

    def is_expired(self) -> bool:
        """Check if timeout has expired."""
        if self.timer is None:
            return False
        return self.timer.elapsed() > self.seconds


class AsyncTimeout:
    """Async context manager for timeout handling."""

    def __init__(self, seconds: float, message: str = "Operation timed out"):
        """Initialize async timeout."""
        self.seconds = seconds
        self.message = message

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass

    async def wait_for(self, coro):
        """Wait for a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=self.seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(self.message)


class ScheduledTask:
    """Represents a scheduled task."""

    def __init__(
        self, task_id: str, func: Callable, interval: float, run_at_start: bool = False
    ):
        """Initialize scheduled task."""
        self.task_id = task_id
        self.func = func
        self.interval = interval
        self.run_at_start = run_at_start
        self.thread: Optional[Thread] = None
        self.stop_event = Event()
        self.last_run: Optional[datetime] = None

    def start(self):
        """Start the scheduled task."""
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the scheduled task."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _run_loop(self):
        """Internal loop for scheduled execution."""
        if self.run_at_start:
            try:
                self.func()
                self.last_run = datetime.now()
            except Exception as e:
                print(f"Error in scheduled task {self.task_id}: {e}")

        while not self.stop_event.is_set():
            self.stop_event.wait(self.interval)
            if not self.stop_event.is_set():
                try:
                    self.func()
                    self.last_run = datetime.now()
                except Exception as e:
                    print(f"Error in scheduled task {self.task_id}: {e}")


class TaskScheduler:
    """Manages scheduled tasks."""

    def __init__(self):
        """Initialize task scheduler."""
        self.tasks: dict[str, ScheduledTask] = {}

    def schedule(
        self,
        func: Callable,
        interval: float,
        task_id: Optional[str] = None,
        run_at_start: bool = False,
    ) -> str:
        """Schedule a function to run at regular intervals."""
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

        if task_id in self.tasks:
            self.tasks[task_id].stop()

        task = ScheduledTask(task_id, func, interval, run_at_start)
        task.start()
        self.tasks[task_id] = task

        return task_id

    def cancel(self, task_id: str):
        """Cancel a scheduled task."""
        if task_id in self.tasks:
            self.tasks[task_id].stop()
            del self.tasks[task_id]

    def cancel_all(self):
        """Cancel all scheduled tasks."""
        for task in self.tasks.values():
            task.stop()
        self.tasks.clear()

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID."""
        return self.tasks.get(task_id)


__all__ = ["Timer", "Timeout", "AsyncTimeout", "ScheduledTask", "TaskScheduler"]
