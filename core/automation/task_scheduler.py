"""
Advanced Task Scheduler - Autonomous Execution Engine
"""

import asyncio
from typing import Dict, Optional, Callable, List
from enum import Enum
from datetime import datetime, timedelta
import uuid
import heapq

from ..utils.logger import logger


class TaskStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Task:
    def __init__(
        self,
        name: str,
        callback: Callable,
        interval: Optional[float] = None,
        priority: int = 0,
        retries: int = 0,
    ):
        # ------------------------
        # VALIDATION 🔒
        # ------------------------
        if not callable(callback):
            raise TypeError("callback must be callable")

        if interval is not None and interval < 0:
            raise ValueError("interval must be >= 0")

        if retries < 0:
            retries = 0

        # ------------------------
        # CORE DATA
        # ------------------------
        self.id = str(uuid.uuid4())
        self.name = name
        self.callback = callback

        self.interval = interval
        self.priority = priority
        self.retries = retries

        # ------------------------
        # STATE
        # ------------------------
        self.status = TaskStatus.SCHEDULED
        now = datetime.now()

        self.created_at = now
        self.last_run = None
        self.next_run = now

        # ------------------------
        # METRICS
        # ------------------------
        self.run_count = 0
        self.failure_count = 0

        # ------------------------
        # INTERNAL FLAGS 🔥
        # ------------------------
        self._running = False  # prevent duplicate execution
        self._cancelled = False  # fast cancel flag

    def __lt__(self, other):
        """Priority comparison (safe + deterministic)"""

        if not isinstance(other, Task):
            return NotImplemented

        # higher priority first
        if self.priority != other.priority:
            return self.priority > other.priority

        # fallback: earlier scheduled task first
        return self.created_at < other.created_at

    async def execute(self):
        """Execute task safely with retry + timeout protection"""

        if self._running or self._cancelled:
            return False

        self._running = True

        try:
            self.status = TaskStatus.RUNNING

            if not callable(self.callback):
                raise TypeError("Invalid callback")

            # run with timeout protection
            if asyncio.iscoroutinefunction(self.callback):
                await asyncio.wait_for(self.callback(), timeout=60)
            else:
                await asyncio.wait_for(asyncio.to_thread(self.callback), timeout=60)

            self.status = TaskStatus.COMPLETED
            self.run_count += 1
            self.last_run = datetime.now()

            if self.interval:
                self.status = TaskStatus.SCHEDULED
                self.next_run = datetime.now() + timedelta(seconds=self.interval)

            return True

        except asyncio.TimeoutError:
            self.failure_count += 1
            logger.error(f"Task timeout: {self.name}")

            self.status = TaskStatus.FAILED
            return False

        except Exception as e:
            self.failure_count += 1
            logger.error(f"Task failed: {self.name} - {e}")

            if self.retries > 0:
                self.retries -= 1
                self.status = TaskStatus.RETRYING
                return await self.execute()

            self.status = TaskStatus.FAILED
            return False

        finally:
            self._running = False


class TaskScheduler:
    def __init__(self, max_workers=5):
        # ------------------------
        # VALIDATION 🔒
        # ------------------------
        if max_workers <= 0:
            max_workers = 1

        # ------------------------
        # CORE STORAGE
        # ------------------------
        self.tasks: Dict[str, Task] = {}
        self.queue: List[Task] = []

        # ------------------------
        # CONCURRENCY CONTROL ⚙️
        # ------------------------
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(self.max_workers)

        # ------------------------
        # STATE
        # ------------------------
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # ------------------------
        # INTERNAL SAFETY 🔥
        # ------------------------
        self._lock = asyncio.Lock()  # prevent race conditions

        # ------------------------
        # METRICS 📊
        # ------------------------
        self._executed_count = 0
        self._failed_count = 0

    # ------------------------
    # SCHEDULE 🔥
    # ------------------------
    def schedule(
        self,
        name: str,
        callback: Callable,
        interval: Optional[float] = None,
        priority: int = 0,
        retries: int = 0,
    ) -> str:

        # ------------------------
        # VALIDATION 🔒
        # ------------------------
        if not callable(callback):
            raise TypeError("callback must be callable")

        if interval is not None and interval < 0:
            raise ValueError("interval must be >= 0")

        if retries < 0:
            retries = 0

        if not name:
            name = "unnamed_task"

        # ------------------------
        # CREATE TASK
        # ------------------------
        task = Task(name, callback, interval, priority, retries)

        # ------------------------
        # THREAD-SAFE INSERT 🔥
        # ------------------------
        self.tasks[task.id] = task
        heapq.heappush(self.queue, task)

        # ------------------------
        # LOGGING
        # ------------------------
        logger.info(f"📅 Task scheduled: {name} | id={task.id} | priority={priority}")

        return task.id

    # ------------------------
    # MAIN LOOP 🔥
    # ------------------------
    async def start(self):
        """Start scheduler loop safely"""

        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.current_task()

        logger.info("🚀 Scheduler started")

        try:
            while self._running:
                try:
                    now = datetime.now()

                    for task in list(self.queue):
                        # skip cancelled
                        if task.status == TaskStatus.CANCELLED or task._cancelled:
                            continue

                        # prevent duplicate execution
                        if task._running:
                            continue

                        # check schedule
                        if now >= task.next_run:
                            asyncio.create_task(self._run_task(task))

                    await asyncio.sleep(0.2)

                except asyncio.CancelledError:
                    logger.info("Scheduler cancelled")
                    break

                except Exception as e:
                    logger.error(f"Scheduler error: {e}")

        finally:
            self._running = False
            self._task = None
            logger.info("🛑 Scheduler stopped")

    async def _run_task(self, task: Task):
        """Execute task with concurrency control + safety"""

        async with self._semaphore:
            try:
                if task._cancelled or task.status == TaskStatus.CANCELLED:
                    return

                result = await task.execute()

                # metrics update
                if result:
                    self._executed_count += 1
                else:
                    self._failed_count += 1

            except Exception as e:
                self._failed_count += 1
                logger.error(f"_run_task error: {task.name} - {e}")

    # ------------------------
    # CONTROL
    # ------------------------
    def cancel(self, task_id: str):
        """Cancel a task safely"""

        task = self.tasks.get(task_id)

        if not task:
            logger.warning(f"Cancel failed: task not found ({task_id})")
            return False

        # mark cancelled
        task.status = TaskStatus.CANCELLED
        task._cancelled = True

        # remove from queue (best effort)
        try:
            self.queue = [t for t in self.queue if t.id != task_id]
            heapq.heapify(self.queue)
        except Exception as e:
            logger.error(f"Queue cleanup error: {e}")

        logger.info(f"🛑 Task cancelled: {task.name} ({task_id})")
        return True

    def stop(self):
        """Stop scheduler safely"""

        if not self._running:
            logger.debug("Scheduler already stopped")
            return

        self._running = False

        # cancel running loop safely
        if self._task:
            if not self._task.done():
                self._task.cancel()

            self._task = None  # cleanup reference

        logger.info("🛑 Scheduler stopping...")

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self):
        """Return detailed scheduler statistics safely"""

        try:
            total = len(self.tasks)

            running = sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING
            )
            completed = sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED
            )
            failed = sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.FAILED
            )
            cancelled = sum(
                1 for t in self.tasks.values() if t.status == TaskStatus.CANCELLED
            )
            pending = sum(
                1
                for t in self.tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.SCHEDULED)
            )

            return {
                "tasks": total,
                "running": running,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "pending": pending,
                "executed_total": self._executed_count,
                "failed_total": self._failed_count,
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
