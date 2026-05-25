"""
EDIATH Task Queue Module
Production-grade distributed task queue with MongoDB persistence and monitoring
"""

import asyncio
import json
import uuid
import heapq
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
from pathlib import Path
import hashlib
from functools import wraps

import pymongo

from core.utils.task_registry import task_registry
from core.utils.logger import logger
from core.memory.mongo_client import (
    mongo_client,
    get_metrics_collection,
    get_audit_collection,
)

# ==================== Configuration ====================


class TaskQueueConfig:
    """Global configuration for task queue with MongoDB support"""

    MAX_QUEUE_SIZE = 10000
    PERSIST_INTERVAL_SECONDS = 30
    DEFAULT_TASK_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5
    HEALTH_CHECK_INTERVAL = 60
    CLEANUP_INTERVAL = 300  # 5 minutes
    MAX_COMPLETED_TASKS = 10000
    TASK_RETENTION_DAYS = 7

    # MongoDB collections
    MONGO_TASKS_COLLECTION = "queue_tasks"
    MONGO_METRICS_COLLECTION = "queue_metrics"
    MONGO_AUDIT_COLLECTION = "queue_audit"

    # Data paths (fallback when MongoDB unavailable)
    DATA_DIR = Path("data/task_queue")
    BACKUP_DIR = Path("data/task_queue/backups")

    # Performance settings
    BATCH_SIZE = 100
    MONGO_BATCH_SIZE = 50

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Enums ====================


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = 0  # System critical tasks (highest priority)
    HIGH = 1  # High priority tasks
    NORMAL = 2  # Normal priority tasks
    LOW = 3  # Low priority tasks
    BACKGROUND = 4  # Background maintenance tasks (lowest priority)


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class QueueType(Enum):
    """Queue type enumeration"""

    PRIORITY = "priority"
    FIFO = "fifo"
    LIFO = "lifo"
    ROUND_ROBIN = "round_robin"


# ==================== Data Classes ====================


@dataclass
class Task:
    """Task data structure with MongoDB compatibility"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = TaskQueueConfig.DEFAULT_TASK_TIMEOUT
    max_retries: int = TaskQueueConfig.MAX_RETRIES
    retry_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    queue_type: QueueType = QueueType.PRIORITY
    tags: List[str] = field(default_factory=list)
    priority_value: int = 2  # 0=highest, 4=lowest
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_task_id: Optional[str] = None

    def __lt__(self, other):
        """For heapq comparison"""
        return self.priority_value < other.priority_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for MongoDB"""
        data = asdict(self)
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        data["queue_type"] = self.queue_type.value
        data["created_at"] = self.created_at
        data["started_at"] = self.started_at
        data["completed_at"] = self.completed_at
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary"""
        data = data.copy()

        # Handle enum conversions
        if "priority" in data and isinstance(data["priority"], str):
            data["priority"] = TaskPriority(data["priority"])
        elif "priority" in data and isinstance(data["priority"], int):
            data["priority"] = TaskPriority(data["priority"])

        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus(data["status"])

        if "queue_type" in data and isinstance(data["queue_type"], str):
            data["queue_type"] = QueueType(data["queue_type"])

        # Handle datetime conversions
        for field in ["created_at", "started_at", "completed_at"]:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)

    def get_cache_key(self) -> str:
        """Generate cache key for task result"""
        return hashlib.sha256(f"{self.id}:{self.updated_at()}".encode()).hexdigest()

    def updated_at(self) -> str:
        """Get last update timestamp"""
        return (self.completed_at or self.started_at or self.created_at).isoformat()


@dataclass
class QueueMetrics:
    """Queue performance metrics"""

    total_tasks_processed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0
    current_queue_size: int = 0
    peak_queue_size: int = 0
    throughput_per_minute: float = 0.0
    last_processed_time: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    success_rate: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def update_success_rate(self):
        """Update success rate based on processed/failed counts"""
        total = self.total_tasks_processed + self.total_tasks_failed
        if total > 0:
            self.success_rate = (self.total_tasks_processed / total) * 100


# ==================== Task Queue ====================


class TaskQueue:
    """Priority-based task queue with MongoDB persistence"""

    def __init__(
        self, name: str = "default", queue_type: QueueType = QueueType.PRIORITY
    ):
        self.name = name
        self.queue_type = queue_type
        self._queue = []  # Heap for priority queue
        self._fifo_queue = deque()  # FIFO queue
        self._lifo_queue = []  # LIFO stack
        self._rr_queues = {}  # Round-robin queues by priority
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._not_empty: Optional[asyncio.Condition] = None  # lazily initialized in async context
        self._max_queue_size = TaskQueueConfig.MAX_QUEUE_SIZE

    @property
    def _cond(self) -> asyncio.Condition:
        """Lazily create Condition in the current event loop to avoid loop-crossing bugs."""
        if self._not_empty is None:
            self._not_empty = asyncio.Condition(self._lock)
        try:
            current = asyncio.get_event_loop()
            if self._not_empty._loop is not None and self._not_empty._loop is not current:
                self._not_empty = asyncio.Condition(self._lock)
        except Exception:
            self._not_empty = asyncio.Condition(self._lock)
        return self._not_empty

    def __init__(self, name: str = "default", queue_type: QueueType = QueueType.PRIORITY):
        self.name = name
        self.queue_type = queue_type
        self._queue = []  # Heap for priority queue
        self._fifo_queue = deque()  # FIFO queue
        self._lifo_queue = []  # LIFO stack
        self._rr_queues = {}  # Round-robin queues by priority
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._not_empty: Optional[asyncio.Condition] = None
        self._max_queue_size = TaskQueueConfig.MAX_QUEUE_SIZE
        self._shutdown = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._mongo_sync_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = QueueMetrics()
        self._wait_times: List[float] = []
        self._process_times: List[float] = []
        self._processing_history: List[Dict] = []

        # MongoDB collections
        self.tasks_collection = None
        self.metrics_collection = get_metrics_collection()
        self.audit_collection = get_audit_collection()

        # Persistence
        TaskQueueConfig.ensure_directories()
        # Use an absolute, resolved path for persistence to avoid cross-workdir
        # os.replace issues on Windows when temp file is absolute but dest is relative
        self.persist_path = (TaskQueueConfig.DATA_DIR / f"{name}_tasks.json").resolve()

        # Initialize MongoDB collections
        self._init_mongo_collections()

        logger.info(f"TaskQueue '{name}' initialized with type {queue_type.value}")

    def _init_mongo_collections(self):
        """Safe MongoDB collection initialization with idempotent indexes"""

        # -------------------------
        # CHECK MONGO STATUS
        # -------------------------
        if not getattr(mongo_client, "enabled", False):
            logger.warning("⚠ MongoDB disabled → skipping collection init")
            self.tasks_collection = None
            return

        try:
            db = getattr(mongo_client, "db", None)
            if db is None:
                logger.warning("⚠ MongoDB DB not available")
                self.tasks_collection = None
                return

            self.tasks_collection = db[TaskQueueConfig.MONGO_TASKS_COLLECTION]

            # -------------------------
            # SAFE INDEX CREATION (IDEMPOTENT)
            # -------------------------
            existing_indexes = set()
            try:
                existing_indexes = {
                    idx["name"] for idx in self.tasks_collection.list_indexes()
                }
            except Exception:
                pass  # safe fallback if listing fails

            def _ensure_index(fields, name):
                if name not in existing_indexes:
                    try:
                        self.tasks_collection.create_index(
                            fields, name=name, background=True
                        )
                    except Exception as e:
                        logger.debug(f"Index '{name}' creation skipped: {e}")

            _ensure_index([("name", 1)], "idx_name")
            _ensure_index([("status", 1)], "idx_status")
            _ensure_index([("priority", 1)], "idx_priority")
            _ensure_index([("created_at", -1)], "idx_created_at")
            _ensure_index([("owner_id", 1)], "idx_owner_id")
            _ensure_index([("parent_task_id", 1)], "idx_parent_task_id")
            _ensure_index([("tags", 1)], "idx_tags")

            logger.info(f"✅ MongoDB collections ready for queue '{self.name}'")

        except Exception as e:
            logger.warning(f"⚠ MongoDB init failed: {e}")
            self.tasks_collection = None

    async def _sync_to_mongo(self, task: Task, operation: str = "update"):
        """
        Safe async MongoDB sync (optimized + production-grade)
        """

        import asyncio

        # -------------------------
        # 🔥 VALIDATION
        # -------------------------
        if not getattr(mongo_client, "enabled", False):
            return

        if not getattr(self, "tasks_collection", None):
            return

        if not task or not hasattr(task, "to_dict"):
            logger.warning("Invalid task for Mongo sync")
            return

        try:
            task_dict = task.to_dict() or {}

            if not isinstance(task_dict, dict):
                raise ValueError("Task serialization failed")

            task_id = task_dict.get("id")
            if not task_id:
                logger.warning("Task missing ID → skipping Mongo sync")
                return

            # -------------------------
            # ⚡ TRUE NON-BLOCKING FIX
            # -------------------------
            def _run():
                try:
                    if operation == "insert":
                        self.tasks_collection.insert_one(task_dict)
                    else:
                        self.tasks_collection.update_one(
                            {"id": task_id}, {"$set": task_dict}, upsert=True
                        )
                except Exception as e:
                    logger.debug(f"Mongo thread op failed: {e}")

            await asyncio.to_thread(_run)

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.mongo_writes = getattr(self, "mongo_writes", 0) + 1
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Mongo sync failed ({operation}): {e}")

    async def _batch_sync_to_mongo(self, tasks: List[Task]) -> None:
        """
        Safe, non-blocking batch Mongo sync (optimized + production-grade)
        """

        import asyncio

        # -------------------------
        # 🔥 VALIDATION
        # -------------------------
        if not getattr(mongo_client, "enabled", False):
            return

        if not getattr(self, "tasks_collection", None) or not tasks:
            return

        try:
            operations = []

            # -------------------------
            # 📦 BUILD OPERATIONS (SAFE)
            # -------------------------
            for task in tasks:
                try:
                    if not task or not hasattr(task, "to_dict"):
                        continue

                    task_dict = task.to_dict() or {}
                    task_id = task_dict.get("id")

                    if not task_id:
                        continue

                    operations.append(
                        pymongo.operations.UpdateOne(
                            {"id": task_id}, {"$set": task_dict}, upsert=True
                        )
                    )

                except Exception:
                    continue

            if not operations:
                return

            # -------------------------
            # ⚡ CHUNKING
            # -------------------------
            batch_size = getattr(TaskQueueConfig, "MONGO_BATCH_SIZE", 100)

            # -------------------------
            # ⚡ TRUE THREAD EXECUTION FIX
            # -------------------------
            def _run_chunk(chunk):
                try:
                    self.tasks_collection.bulk_write(chunk, ordered=False)
                except Exception as e:
                    logger.debug(f"Mongo chunk failed: {e}")

            # -------------------------
            # 🚀 EXECUTE (PARALLEL SAFE)
            # -------------------------
            tasks_thread = []
            for i in range(0, len(operations), batch_size):
                chunk = operations[i : i + batch_size]
                tasks_thread.append(asyncio.to_thread(_run_chunk, chunk))

            await asyncio.gather(*tasks_thread, return_exceptions=True)

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.mongo_batch_writes = getattr(self, "mongo_batch_writes", 0) + len(
                    operations
                )
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Batch Mongo sync failed: {e}")

    async def _log_audit(
        self, task: Task, action: str, details: Optional[Dict] = None
    ) -> None:
        """
        Safe, non-blocking audit logging (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        # -------------------------
        # 🔥 VALIDATION
        # -------------------------
        if not getattr(mongo_client, "enabled", False):
            return

        if not getattr(self, "audit_collection", None):
            return

        if not task or not hasattr(task, "id"):
            logger.warning("Invalid task for audit log")
            return

        try:
            # -------------------------
            # 📦 BUILD AUDIT ENTRY (SAFE)
            # -------------------------
            try:
                audit_entry = {
                    "timestamp": datetime.utcnow(),
                    "queue_name": str(getattr(self, "name", "unknown")),
                    "task_id": str(getattr(task, "id", "")),
                    "task_name": str(getattr(task, "name", "")),
                    "action": str(action or "unknown"),
                    "status": (
                        getattr(task.status, "value", str(task.status))
                        if hasattr(task, "status")
                        else None
                    ),
                    "owner_id": getattr(task, "owner_id", None),
                    "session_id": getattr(task, "session_id", None),
                    "details": (
                        details
                        if isinstance(details, dict)
                        else {"value": str(details)}
                    ),
                    "retry_count": int(getattr(task, "retry_count", 0)),
                }
            except Exception as e:
                raise RuntimeError(f"Audit entry build failed: {e}")

            # -------------------------
            # ⚡ TRUE NON-BLOCKING FIX
            # -------------------------
            def _run():
                try:
                    self.audit_collection.insert_one(audit_entry)
                except Exception as e:
                    logger.debug(f"Audit insert failed: {e}")

            await asyncio.to_thread(_run)

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.audit_logs = getattr(self, "audit_logs", 0) + 1
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Audit log failed: {e}")

    async def put(self, task: Task) -> bool:
        """Safe enqueue with validation + overflow protection"""

        # -------------------------
        # VALIDATION (CRITICAL FIX)
        # -------------------------
        if not task or not isinstance(task, Task):
            logger.warning("Invalid task object")
            return False

        async with self._lock:
            try:
                # -------------------------
                # QUEUE LIMIT CHECK
                # -------------------------
                if len(self._tasks) >= self._max_queue_size:
                    logger.warning(f"Queue '{self.name}' full → rejecting {task.id}")
                    return False

                # -------------------------
                # DUPLICATE CHECK (CRITICAL FIX)
                # -------------------------
                if task.id in self._tasks or task.id in self._running_tasks:
                    logger.warning(f"Duplicate task ignored: {task.id}")
                    return False

                # -------------------------
                # PREPARE TASK
                # -------------------------
                task.status = TaskStatus.QUEUED
                task.priority_value = getattr(task.priority, "value", 2)

                # -------------------------
                # INSERT INTO QUEUE (SAFE)
                # -------------------------
                if self.queue_type == QueueType.PRIORITY:
                    heapq.heappush(self._queue, (task.priority_value, task.id, task))

                elif self.queue_type == QueueType.FIFO:
                    self._fifo_queue.append(task)

                elif self.queue_type == QueueType.LIFO:
                    self._lifo_queue.append(task)

                elif self.queue_type == QueueType.ROUND_ROBIN:
                    if task.priority not in self._rr_queues:
                        self._rr_queues[task.priority] = deque()
                    self._rr_queues[task.priority].append(task)

                else:
                    logger.warning(f"Unknown queue type: {self.queue_type}")
                    return False

                # -------------------------
                # STORE TASK
                # -------------------------
                self._tasks[task.id] = task

                # -------------------------
                # METRICS UPDATE
                # -------------------------
                size = len(self._tasks)
                self.metrics.current_queue_size = size
                self.metrics.peak_queue_size = max(self.metrics.peak_queue_size, size)

                # -------------------------
                # NON-BLOCKING DB SYNC
                # -------------------------
                try:
                    asyncio.create_task(self._sync_to_mongo(task, "insert"))
                    asyncio.create_task(self._log_audit(task, "enqueued"))
                except Exception:
                    pass

                # -------------------------
                # NOTIFY WAITERS (SAFE)
                # -------------------------
                try:
                    self._cond.notify_all()
                except RuntimeError:
                    pass

                logger.debug(f"Task {task.id} queued in '{self.name}'")
                return True

            except Exception as e:
                logger.error(f"❌ Put failed: {e}")
                return False

    async def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """Safe, non-blocking task retrieval with proper waiting"""

        loop = asyncio.get_event_loop()
        start_time = loop.time()

        while not self._shutdown:
            try:
                async with self._lock:
                    # -------------------------
                    # TRY GET TASK
                    # -------------------------
                    task = self._get_next_task()

                    if task:
                        now = datetime.utcnow()

                        task.status = TaskStatus.RUNNING
                        task.started_at = now
                        self._running_tasks[task.id] = task

                        # Remove from pending
                        self._tasks.pop(task.id, None)

                        self.metrics.current_queue_size = len(self._tasks)

                        # -------------------------
                        # WAIT TIME METRICS
                        # -------------------------
                        try:
                            wait_time_val = (now - task.created_at).total_seconds()
                            self._wait_times.append(wait_time_val)

                            if len(self._wait_times) > 1000:
                                self._wait_times.pop(0)

                            self.metrics.average_wait_time = sum(
                                self._wait_times
                            ) / len(self._wait_times)
                        except Exception:
                            pass

                        # -------------------------
                        # BACKGROUND TASKS
                        # -------------------------
                        asyncio.create_task(self._sync_to_mongo(task, "update"))
                        asyncio.create_task(self._log_audit(task, "dequeued"))

                        logger.debug(
                            f"Task {task.id} ('{task.name}') dequeued from '{self.name}'"
                        )
                        return task

                    # -------------------------
                    # TIMEOUT CALCULATION
                    # -------------------------
                    if timeout is not None:
                        elapsed = loop.time() - start_time
                        if elapsed >= timeout:
                            return None
                        remaining = timeout - elapsed
                    else:
                        remaining = None

                    # -------------------------
                    # WAIT (FIXED: INSIDE LOCK)
                    # -------------------------
                    try:
                        if remaining is not None:
                            await asyncio.wait_for(self._cond.wait(), remaining)
                        else:
                            await self._cond.wait()
                    except asyncio.TimeoutError:
                        return None

            except Exception as e:
                logger.error(f"❌ Queue get error: {e}")
                await asyncio.sleep(0.1)  # prevent tight loop

        return None

    def _get_next_task(self) -> Optional[Task]:
        """
        Safe task selection (optimized + production-grade)
        """

        import heapq

        try:
            # -------------------------
            # 🔥 PRIORITY QUEUE (FAIR + SAFE)
            # -------------------------
            if self.queue_type == QueueType.PRIORITY:
                temp = []
                selected = None

                while self._queue:
                    item = heapq.heappop(self._queue)
                    priority_value, task_id, task = item

                    if not task:
                        continue

                    if self._is_task_ready(task):
                        selected = task
                        break

                    temp.append(item)

                # restore queue
                for item in temp:
                    heapq.heappush(self._queue, item)

                return selected

            # -------------------------
            # 📦 FIFO (SAFE ROTATION)
            # -------------------------
            elif self.queue_type == QueueType.FIFO:
                size = len(self._fifo_queue)
                selected = None

                for _ in range(size):
                    task = self._fifo_queue.popleft()

                    if task and self._is_task_ready(task):
                        selected = task
                        break

                    if task and task.status != TaskStatus.BLOCKED:
                        self._fifo_queue.append(task)

                return selected

            # -------------------------
            # 🔄 LIFO (SAFE ROTATION)
            # -------------------------
            elif self.queue_type == QueueType.LIFO:
                size = len(self._lifo_queue)
                selected = None

                for _ in range(size):
                    task = self._lifo_queue.pop()

                    if task and self._is_task_ready(task):
                        selected = task
                        break

                    if task and task.status != TaskStatus.BLOCKED:
                        self._lifo_queue.insert(0, task)

                return selected

            # -------------------------
            # ⚖️ ROUND ROBIN (FAIRNESS + STARVATION FIX)
            # -------------------------
            elif self.queue_type == QueueType.ROUND_ROBIN:
                priorities = [
                    TaskPriority.CRITICAL,
                    TaskPriority.HIGH,
                    TaskPriority.NORMAL,
                    TaskPriority.LOW,
                    TaskPriority.BACKGROUND,
                ]

                selected = None

                for priority in priorities:
                    queue = self._rr_queues.get(priority)
                    if not queue:
                        continue

                    size = len(queue)

                    for _ in range(size):
                        task = queue.popleft()

                        if task and self._is_task_ready(task):
                            selected = task
                            break

                        if task and task.status != TaskStatus.BLOCKED:
                            queue.append(task)

                    if selected:
                        break

                return selected

            # -------------------------
            # ❌ UNKNOWN QUEUE TYPE
            # -------------------------
            return None

        except Exception as e:
            logger.error(f"❌ Error getting next task: {e}")
            return None

    def _is_task_ready(self, task: Task) -> bool:
        """
        Robust dependency check (optimized + production-grade)
        """

        try:
            # -------------------------
            # 🔥 VALIDATION
            # -------------------------
            if not task:
                return False

            dependencies = getattr(task, "dependencies", None)

            if not dependencies:
                return True

            # -------------------------
            # 🔁 CHECK DEPENDENCIES
            # -------------------------
            for dep_id in dependencies:

                if not dep_id:
                    return False

                # -------------------------
                # ⚡ RUNNING → NOT READY
                # -------------------------
                if dep_id in getattr(self, "_running_tasks", set()):
                    return False

                # -------------------------
                # 📦 CHECK ACTIVE TASKS
                # -------------------------
                dep_task = getattr(self, "_tasks", {}).get(dep_id)
                if dep_task:
                    if getattr(dep_task, "status", None) != TaskStatus.COMPLETED:
                        return False
                    continue

                # -------------------------
                # ✅ CHECK COMPLETED CACHE
                # -------------------------
                completed_task = getattr(self, "_completed_tasks", {}).get(dep_id)
                if completed_task:
                    if getattr(completed_task, "status", None) != TaskStatus.COMPLETED:
                        return False
                    continue

                # -------------------------
                # ❌ UNKNOWN DEPENDENCY
                # -------------------------
                logger.warning(
                    f"Dependency {dep_id} not found for task {getattr(task, 'id', 'unknown')}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Dependency check failed: {e}")
            return False

    async def complete(
        self, task_id: str, result: Any = None, error: Optional[str] = None
    ):
        """Safe, non-blocking task completion with optimized performance"""

        now = datetime.utcnow()

        try:
            # -------------------------
            # GET TASK (LOCK MINIMIZED - CRITICAL FIX)
            # -------------------------
            async with self._lock:
                task = self._running_tasks.pop(task_id, None)

            if not task:
                logger.warning(f"Task {task_id} not in running tasks")
                return

            task.completed_at = now

            # -------------------------
            # FAILURE HANDLING
            # -------------------------
            if error:
                task.status = TaskStatus.FAILED
                task.error = error

                self.metrics.total_tasks_failed += 1
                self.metrics.error_count += 1
                self.metrics.last_error = error
                self.metrics.update_success_rate()

                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRYING
                    task.created_at = now

                    logger.info(
                        f"Task {task_id} retrying ({task.retry_count}/{task.max_retries})"
                    )

                    asyncio.create_task(self._retry_with_delay(task))
                    return  # IMPORTANT: exit early

                logger.error(f"Task {task_id} failed permanently: {error}")

            else:
                task.status = TaskStatus.COMPLETED
                task.result = result

                self.metrics.total_tasks_processed += 1
                self.metrics.update_success_rate()

                logger.info(f"Task {task_id} completed")

            # -------------------------
            # STORE COMPLETED TASK
            # -------------------------
            self._completed_tasks[task_id] = task

            # -------------------------
            # FAST TRIM (CRITICAL FIX)
            # -------------------------
            if len(self._completed_tasks) > TaskQueueConfig.MAX_COMPLETED_TASKS:
                # remove oldest without sorting full dict
                oldest_key = min(
                    self._completed_tasks,
                    key=lambda k: self._completed_tasks[k].completed_at,
                )
                self._completed_tasks.pop(oldest_key, None)

            # -------------------------
            # HISTORY (SAFE)
            # -------------------------
            try:
                duration = (
                    (task.completed_at - task.started_at).total_seconds()
                    if task.started_at
                    else 0
                )

                self._processing_history.append(
                    {
                        "task_id": task_id,
                        "name": task.name,
                        "status": task.status.value,
                        "error": error,
                        "duration": duration,
                        "timestamp": task.completed_at.isoformat(),
                    }
                )

                if len(self._processing_history) > 1000:
                    self._processing_history = self._processing_history[-1000:]

            except Exception:
                pass

            # -------------------------
            # METRICS (SAFE)
            # -------------------------
            if task.started_at:
                process_time = (task.completed_at - task.started_at).total_seconds()
                self._process_times.append(process_time)

                if len(self._process_times) > 1000:
                    self._process_times.pop(0)

                self.metrics.average_processing_time = sum(self._process_times) / len(
                    self._process_times
                )

            self.metrics.last_processed_time = now
            self._update_throughput()

            # -------------------------
            # BACKGROUND TASKS (NON-BLOCKING)
            # -------------------------
            asyncio.create_task(self._sync_to_mongo(task, "update"))
            asyncio.create_task(
                self._log_audit(
                    task,
                    "failed" if error else "completed",
                    {"error": error} if error else None,
                )
            )

            asyncio.create_task(self._update_metrics_in_mongo())

        except Exception as e:
            logger.error(f"❌ Complete failed: {e}")

    async def _retry_with_delay(self, task: Task) -> None:
        """
        Safe retry with delay (optimized + production-grade)
        """

        import asyncio

        try:
            # -------------------------
            # 🔥 VALIDATION
            # -------------------------
            if not task or not hasattr(task, "id"):
                return

            if getattr(self, "_shutdown", False):
                logger.warning(f"Retry skipped (shutdown): {task.id}")
                return

            # -------------------------
            # ⏱️ SAFE DELAY
            # -------------------------
            delay = max(getattr(TaskQueueConfig, "RETRY_DELAY_SECONDS", 1), 0)

            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.debug(f"Retry cancelled: {task.id}")
                return

            # -------------------------
            # 🛑 DUPLICATE PREVENTION
            # -------------------------
            if task.id in getattr(self, "_running_tasks", set()) or task.id in getattr(
                self, "_tasks", {}
            ):
                logger.debug(f"Retry skipped (already active): {task.id}")
                return

            # -------------------------
            # 🔁 RESET STATE (SAFE)
            # -------------------------
            try:
                task.started_at = None
                task.completed_at = None
                if hasattr(task, "status"):
                    task.status = TaskStatus.PENDING
            except Exception:
                pass

            # -------------------------
            # 📥 REQUEUE SAFELY
            # -------------------------
            try:
                if hasattr(self, "put"):
                    await self.put(task)
                else:
                    raise RuntimeError("Queue put() not available")
            except Exception as e:
                logger.error(f"Requeue failed: {e}")
                return

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.retries = getattr(self, "retries", 0) + 1
            except Exception:
                pass

            logger.info(f"🔁 Task retried: {task.id}")

        except Exception as e:
            logger.error(f"❌ Retry failed: {e}")

    async def _update_metrics_in_mongo(self) -> None:
        """
        Safe, non-blocking metrics update (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        # -------------------------
        # 🔥 VALIDATION
        # -------------------------
        if not getattr(mongo_client, "enabled", False):
            return

        if not getattr(self, "metrics_collection", None):
            return

        try:
            now = datetime.utcnow()

            # -------------------------
            # 📦 BUILD DOCUMENT (SAFE)
            # -------------------------
            try:
                metrics_doc = {
                    "queue_name": str(getattr(self, "name", "unknown")),
                    "timestamp": now,
                    "metrics": (
                        self.metrics.to_dict() if hasattr(self, "metrics") else {}
                    ),
                    "queue_type": getattr(
                        self.queue_type, "value", str(self.queue_type)
                    ),
                }
            except Exception as e:
                raise RuntimeError(f"Metrics build failed: {e}")

            # -------------------------
            # ⚡ INSERT (THREAD SAFE)
            # -------------------------
            def _insert():
                try:
                    self.metrics_collection.insert_one(metrics_doc)
                except Exception as e:
                    logger.debug(f"Metrics insert failed: {e}")

            await asyncio.to_thread(_insert)

            # -------------------------
            # ⚡ OPTIMIZED CLEANUP
            # -------------------------
            def _cleanup():
                try:
                    max_docs = 1000

                    count = self.metrics_collection.count_documents(
                        {"queue_name": metrics_doc["queue_name"]}
                    )

                    if count <= max_docs:
                        return

                    delete_count = count - max_docs

                    old_ids = list(
                        self.metrics_collection.find(
                            {"queue_name": metrics_doc["queue_name"]}, {"_id": 1}
                        )
                        .sort("timestamp", 1)
                        .limit(delete_count)
                    )

                    if old_ids:
                        self.metrics_collection.delete_many(
                            {"_id": {"$in": [doc["_id"] for doc in old_ids]}}
                        )

                except Exception as e:
                    logger.debug(f"Metrics cleanup failed: {e}")

            # run cleanup in background without blocking main flow
            asyncio.create_task(asyncio.to_thread(_cleanup))

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.mongo_metrics_updates = (
                    getattr(self, "mongo_metrics_updates", 0) + 1
                )
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Metrics update failed: {e}")

    async def cancel(self, task_id: str) -> bool:
        """Safe task cancellation with proper cleanup"""

        try:
            # -------------------------
            # MINIMAL LOCK (CRITICAL FIX)
            # -------------------------
            async with self._lock:
                task = self._tasks.get(task_id)

                if not task:
                    return False

                if task.status not in (TaskStatus.PENDING, TaskStatus.QUEUED):
                    return False

                # mark cancelled
                task.status = TaskStatus.CANCELLED

                # REMOVE FROM QUEUE STORAGE (IMPORTANT FIX)
                self._tasks.pop(task_id, None)

                self.metrics.total_tasks_cancelled += 1
                self.metrics.update_success_rate()

            # -------------------------
            # OUTSIDE LOCK (NON-BLOCKING)
            # -------------------------
            logger.info(f"🚫 Task {task_id} cancelled")

            asyncio.create_task(self._sync_to_mongo(task, "update"))
            asyncio.create_task(self._log_audit(task, "cancelled"))

            return True

        except Exception as e:
            logger.error(f"❌ Cancel failed: {e}")
            return False

    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Safe task status retrieval (memory + Mongo fallback)"""

        if not task_id:
            return None

        try:
            # -------------------------
            # MEMORY LOOKUP (FAST PATH)
            # -------------------------
            task = (
                self._tasks.get(task_id)
                or self._running_tasks.get(task_id)
                or self._completed_tasks.get(task_id)
            )

            if task and hasattr(task, "to_dict"):
                try:
                    return task.to_dict()
                except Exception:
                    pass  # fallback to Mongo

            # -------------------------
            # MONGO FALLBACK (NON-BLOCKING)
            # -------------------------
            if (
                getattr(mongo_client, "enabled", False)
                and self.tasks_collection is not None
            ):

                def _find():
                    try:
                        return self.tasks_collection.find_one({"id": task_id})
                    except Exception:
                        return None

                task_doc = await asyncio.to_thread(_find)

                if task_doc:
                    return task_doc

            return None

        except Exception as e:
            logger.error(f"❌ Get status failed: {e}")
            return None

    def _update_throughput(self):
        """Accurate, stable throughput calculation"""

        try:
            now = datetime.utcnow()

            last_time = getattr(self.metrics, "last_processed_time", None)

            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not last_time:
                return

            total_tasks = getattr(self.metrics, "total_tasks_processed", 0)
            if total_tasks <= 0:
                self.metrics.throughput_per_minute = 0
                return

            # -------------------------
            # TIME CALCULATION (FIXED)
            # -------------------------
            elapsed_seconds = (now - last_time).total_seconds()

            if elapsed_seconds <= 0:
                return

            elapsed_minutes = elapsed_seconds / 60

            # -------------------------
            # STABILITY LIMIT (CRITICAL FIX)
            # -------------------------
            if elapsed_minutes > 60:
                # avoid meaningless long gaps
                return

            # -------------------------
            # THROUGHPUT CALCULATION
            # -------------------------
            throughput = total_tasks / elapsed_minutes

            # clamp to avoid spikes
            self.metrics.throughput_per_minute = round(min(throughput, 10000), 2)

        except Exception as e:
            logger.debug(f"Throughput update failed: {e}")

    async def persist(self) -> None:
        """
        Safe, non-blocking persistence (optimized + production-grade)
        """

        import asyncio
        import json
        from datetime import datetime

        try:
            # -------------------------
            # 🔒 SNAPSHOT (MIN LOCK TIME)
            # -------------------------
            async with self._lock:
                data = {
                    "name": str(self.name),
                    "queue_type": getattr(
                        self.queue_type, "value", str(self.queue_type)
                    ),
                    "tasks": [
                        t.to_dict()
                        for t in self._tasks.values()
                        if hasattr(t, "to_dict")
                    ],
                    "running_tasks": [
                        t.to_dict()
                        for t in self._running_tasks.values()
                        if hasattr(t, "to_dict")
                    ],
                    "completed_tasks": [
                        t.to_dict()
                        for t in list(self._completed_tasks.values())[-1000:]
                        if hasattr(t, "to_dict")
                    ],
                    "metrics": (
                        self.metrics.to_dict() if hasattr(self, "metrics") else {}
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "2.2.0",
                }

            # -------------------------
            # ⚡ ATOMIC WRITE (THREAD SAFE)
            # -------------------------
            def _write():
                try:
                    self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                    import tempfile
                    import time

                    # ensure we operate in the same directory (absolute)
                    dirpath = str(self.persist_path.parent)

                    # Write to a uniquely named temp file in the same directory
                    for attempt in range(3):
                        try:
                            fd, tmpname = tempfile.mkstemp(
                                prefix=f"{self.name}_",
                                suffix=".tmp",
                                dir=dirpath,
                                text=True,
                            )
                            try:
                                with os.fdopen(fd, "w", encoding="utf-8") as f:
                                    json.dump(
                                        data,
                                        f,
                                        indent=2,
                                        ensure_ascii=False,
                                        default=str,
                                    )
                                    f.flush()
                                    try:
                                        os.fsync(f.fileno())
                                    except Exception:
                                        pass

                                # atomic replace using resolved absolute destination
                                try:
                                    dest_path = str(self.persist_path)
                                except Exception:
                                    dest_path = str(
                                        (
                                            TaskQueueConfig.DATA_DIR
                                            / f"{self.name}_tasks.json"
                                        ).resolve()
                                    )

                                os.replace(tmpname, dest_path)
                                break
                            finally:
                                # if file still exists (on failure), attempt cleanup
                                if os.path.exists(tmpname):
                                    try:
                                        os.remove(tmpname)
                                    except Exception:
                                        pass
                        except PermissionError as e:
                            logger.error(
                                f"Persist write permission error (attempt {attempt+1}): {e}"
                            )
                            time.sleep(0.05)
                            continue
                        except OSError as e:
                            logger.error(
                                f"Persist write OS error (attempt {attempt+1}): {e}"
                            )
                            time.sleep(0.05)
                            continue
                    else:
                        logger.error("Persist write failed after retries")

                except Exception as e:
                    logger.error(f"Persist write failed: {e}")

            await asyncio.to_thread(_write)

            # -------------------------
            # 💾 SMART BACKUP (DEBOUNCED)
            # -------------------------
            now = datetime.utcnow()

            should_backup = False
            try:
                last_backup = getattr(self, "_last_backup_time", None)
                if not last_backup or (now - last_backup).total_seconds() > 1800:
                    should_backup = True
                    self._last_backup_time = now
            except Exception:
                should_backup = True

            if should_backup:

                def _backup():
                    try:
                        backup_dir = TaskQueueConfig.BACKUP_DIR.resolve()
                        backup_dir.mkdir(parents=True, exist_ok=True)

                        backup_path = (
                            backup_dir / f"{self.name}_{int(now.timestamp())}.json"
                        )

                        with open(backup_path, "w", encoding="utf-8") as f:
                            json.dump(
                                data, f, indent=2, ensure_ascii=False, default=str
                            )

                        # keep last 10 backups
                        backups = sorted(backup_dir.glob(f"{self.name}_*.json"))

                        for old in backups[:-10]:
                            try:
                                old.unlink()
                            except Exception:
                                pass

                    except Exception as e:
                        logger.warning(f"Backup failed: {e}")

                asyncio.create_task(asyncio.to_thread(_backup))

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.persist_count = getattr(self, "persist_count", 0) + 1
            except Exception:
                pass

            logger.debug(f"💾 Queue '{self.name}' persisted safely")

        except Exception as e:
            logger.error(f"❌ Persist failed: {e}")

    async def load(self):
        """Safe, non-blocking queue load with validation + fallback"""

        try:
            # -------------------------
            # FILE EXISTS CHECK
            # -------------------------
            if not self.persist_path.exists():
                await self._load_from_mongo()
                return

            # -------------------------
            # NON-BLOCKING FILE READ (CRITICAL FIX)
            # -------------------------
            def _read():
                try:
                    with open(self.persist_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None

            data = await asyncio.to_thread(_read)

            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid persist file for '{self.name}'")
                return

            # -------------------------
            # PREPARE DATA (NO LOCK)
            # -------------------------
            tasks_data = data.get("tasks", [])[:1000]
            completed_data = data.get("completed_tasks", [])[:1000]
            metrics_data = data.get("metrics", {})

            tasks_to_add = []
            completed_to_add = {}

            # -------------------------
            # PARSE TASKS SAFELY
            # -------------------------
            for task_data in tasks_data:
                try:
                    task = Task.from_dict(task_data)

                    if task.status in (
                        TaskStatus.PENDING,
                        TaskStatus.QUEUED,
                        TaskStatus.RETRYING,
                    ):
                        tasks_to_add.append(task)
                except Exception:
                    continue

            # -------------------------
            # PARSE COMPLETED
            # -------------------------
            for task_data in completed_data:
                try:
                    task = Task.from_dict(task_data)
                    completed_to_add[task.id] = task
                except Exception:
                    continue

            # -------------------------
            # FIX METRICS (SAFE)
            # -------------------------
            try:
                if metrics_data:
                    if isinstance(metrics_data.get("last_processed_time"), str):
                        metrics_data["last_processed_time"] = datetime.fromisoformat(
                            metrics_data["last_processed_time"]
                        )
                    new_metrics = QueueMetrics(**metrics_data)
                else:
                    new_metrics = None
            except Exception:
                new_metrics = None

            # -------------------------
            # APPLY STATE (LOCK MINIMIZED)
            # -------------------------
            async with self._lock:
                # load completed
                self._completed_tasks.update(completed_to_add)

                # update metrics
                if new_metrics:
                    self.metrics = new_metrics

            # -------------------------
            # ADD TASKS OUTSIDE LOCK (CRITICAL FIX)
            # -------------------------
            for task in tasks_to_add:
                try:
                    await self.put(task)
                except Exception:
                    continue

            logger.info(
                f"📂 Queue '{self.name}' loaded safely "
                f"({len(tasks_to_add)} pending, {len(completed_to_add)} completed)"
            )

        except Exception as e:
            logger.error(f"❌ Load failed: {e}")

    async def _load_from_mongo(self) -> None:
        """
        Safe, non-blocking MongoDB load (optimized + production-grade)
        """

        import asyncio

        # -------------------------
        # 🔥 VALIDATION
        # -------------------------
        if not getattr(mongo_client, "enabled", False):
            return

        if not getattr(self, "tasks_collection", None):
            return

        try:
            # -------------------------
            # 📥 FETCH DATA (THREAD SAFE)
            # -------------------------
            def _fetch():
                try:
                    cursor = self.tasks_collection.find(
                        {
                            "status": {"$in": ["pending", "queued", "retrying"]},
                            "queue_type": getattr(
                                self.queue_type, "value", str(self.queue_type)
                            ),
                        }
                    ).limit(1000)

                    return list(cursor)
                except Exception as e:
                    logger.debug(f"Mongo fetch failed: {e}")
                    return []

            docs = await asyncio.to_thread(_fetch)

            if not docs:
                return

            tasks_to_add = []

            # -------------------------
            # 🔁 PARSE TASKS (SAFE)
            # -------------------------
            for doc in docs:
                try:
                    task = Task.from_dict(doc)

                    if not task or not hasattr(task, "id"):
                        continue

                    # deduplication
                    if task.id not in getattr(
                        self, "_tasks", {}
                    ) and task.id not in getattr(self, "_running_tasks", set()):
                        tasks_to_add.append(task)

                except Exception:
                    continue

            if not tasks_to_add:
                return

            # -------------------------
            # ⚡ BATCH ADD (PERFORMANCE FIX)
            # -------------------------
            for task in tasks_to_add:
                try:
                    await self.put(task)
                except Exception:
                    continue

            # -------------------------
            # 📊 METRICS
            # -------------------------
            try:
                self.mongo_loads = getattr(self, "mongo_loads", 0) + len(tasks_to_add)
            except Exception:
                pass

            logger.info(
                f"📥 Queue '{self.name}' loaded {len(tasks_to_add)} tasks from MongoDB"
            )

        except Exception as e:
            logger.error(f"❌ Mongo load failed: {e}")

    async def clear(self):
        """Clear all tasks from queue"""
        async with self._lock:
            self._queue.clear()
            self._fifo_queue.clear()
            self._lifo_queue.clear()
            self._rr_queues.clear()
            self._tasks.clear()
            self._running_tasks.clear()
            self.metrics.current_queue_size = 0
            logger.info(f"Queue '{self.name}' cleared")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        async with self._lock:
            return {
                "queue_name": self.name,
                "queue_type": self.queue_type.value,
                "metrics": self.metrics.to_dict(),
                "recent_history": self._processing_history[-50:],  # Last 50 tasks
                "pending_tasks": len(self._tasks),
                "running_tasks": len(self._running_tasks),
                "completed_tasks": len(self._completed_tasks),
            }

    async def cleanup(self):
        """Safe, non-blocking cleanup with fast removal + Mongo sync"""

        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(days=TaskQueueConfig.TASK_RETENTION_DAYS)

            # -------------------------
            # FIND TASKS TO REMOVE (NO LOCK)
            # -------------------------
            to_remove = []

            for task_id, task in list(self._completed_tasks.items()):
                try:
                    if task.completed_at and task.completed_at < cutoff:
                        to_remove.append(task_id)
                except Exception:
                    continue

            # -------------------------
            # REMOVE TASKS (MIN LOCK)
            # -------------------------
            if to_remove:
                async with self._lock:
                    for task_id in to_remove:
                        self._completed_tasks.pop(task_id, None)

                logger.info(f"🧹 Cleaned {len(to_remove)} old tasks from '{self.name}'")

            # -------------------------
            # MONGO CLEANUP (NON-BLOCKING)
            # -------------------------
            if (
                getattr(mongo_client, "enabled", False)
                and self.tasks_collection is not None
            ):

                def _mongo_cleanup():
                    try:
                        result = self.tasks_collection.delete_many(
                            {
                                "status": {"$in": ["completed", "failed", "cancelled"]},
                                "completed_at": {"$lt": cutoff},
                            }
                        )
                        return result.deleted_count
                    except Exception:
                        return 0

                deleted = await asyncio.to_thread(_mongo_cleanup)

                if deleted > 0:
                    logger.info(f"🧹 Cleaned {deleted} old tasks from MongoDB")

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

    async def start_cleanup(self):
        """Robust periodic cleanup loop with cancellation + drift control"""

        try:
            interval = max(getattr(TaskQueueConfig, "CLEANUP_INTERVAL", 60), 1)

            while not self._shutdown:
                try:
                    # -------------------------
                    # INTERRUPTIBLE SLEEP (CRITICAL FIX)
                    # -------------------------
                    await asyncio.wait_for(
                        asyncio.sleep(interval), timeout=interval + 1
                    )

                    if self._shutdown:
                        break

                    # -------------------------
                    # RUN CLEANUP (SAFE)
                    # -------------------------
                    await self.cleanup()

                except asyncio.CancelledError:
                    logger.info("🛑 Cleanup loop cancelled")
                    break

                except Exception as e:
                    logger.error(f"❌ Cleanup error: {e}")
                    await asyncio.sleep(1)  # prevent tight error loop

        except Exception as e:
            logger.error(f"❌ Cleanup loop failed: {e}")

    async def shutdown(self):
        """Graceful, non-blocking queue shutdown"""

        try:
            logger.info(f"🛑 Shutting down queue '{self.name}'")

            # -------------------------
            # SIGNAL SHUTDOWN (CRITICAL)
            # -------------------------
            self._shutdown = True

            # -------------------------
            # CANCEL CLEANUP LOOP (SAFE)
            # -------------------------
            if getattr(self, "_cleanup_task", None):
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # -------------------------
            # CANCEL RUNNING TASKS (IMPORTANT FIX)
            # -------------------------
            for task in list(self._running_tasks.values()):
                try:
                    task.status = TaskStatus.CANCELLED
                except Exception:
                    continue

            self._running_tasks.clear()

            # -------------------------
            # FINAL PERSIST (NON-BLOCKING SAFE)
            # -------------------------
            try:
                await asyncio.wait_for(self.persist(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("⚠ Persist timeout during shutdown")

            # -------------------------
            # CLEAR SIGNALS / EVENTS
            # -------------------------
            try:
                if self._not_empty is not None:
                    self._not_empty.notify_all()
            except Exception:
                pass

            logger.info(f"✅ Queue '{self.name}' shutdown complete")

        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")


# ==================== Task Worker ====================


class TaskWorker:
    """Worker that processes tasks from queue with monitoring"""

    def __init__(self, queue: TaskQueue, worker_id: str, handler: Callable):
        self.queue = queue
        self.worker_id = worker_id
        self.handler = handler
        self.is_running = False
        self.current_task: Optional[Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self.tasks_processed = 0
        self.errors_count = 0
        self.total_processing_time = 0.0

        logger.info(f"TaskWorker {worker_id} created for queue '{queue.name}'")

    async def start(self):
        """Start worker"""
        self.is_running = True
        self._worker_task = asyncio.create_task(self._run())
        logger.info(f"TaskWorker {self.worker_id} started")

    async def _run(self):
        """Robust worker loop with proper idle handling + shutdown safety"""

        try:
            idle_sleep = 0.05  # prevents CPU spinning

            while self.is_running and not task_registry.is_shutting_down():

                try:
                    # -------------------------
                    # GET TASK (NON-BLOCKING SAFE)
                    # -------------------------
                    task = await self.queue.get(timeout=1.0)

                    if not task:
                        await asyncio.sleep(idle_sleep)
                        continue

                    self.current_task = task
                    start_time = datetime.utcnow()

                    # -------------------------
                    # PROCESS TASK (SAFE)
                    # -------------------------
                    try:
                        await self._process_task(task)
                    except Exception as task_error:
                        logger.error(f"❌ Task processing failed: {task_error}")
                        self.errors_count += 1

                    # -------------------------
                    # METRICS (SAFE)
                    # -------------------------
                    try:
                        processing_time = (
                            datetime.utcnow() - start_time
                        ).total_seconds()
                        self.total_processing_time += processing_time
                        self.tasks_processed += 1
                    except Exception:
                        pass

                    self.current_task = None

                except asyncio.TimeoutError:
                    # -------------------------
                    # IDLE WAIT (CRITICAL FIX)
                    # -------------------------
                    await asyncio.sleep(idle_sleep)

                except asyncio.CancelledError:
                    logger.info(f"🛑 Worker {self.worker_id} cancelled")
                    break

                except Exception as e:
                    logger.error(f"❌ Worker {self.worker_id} error: {e}")
                    self.errors_count += 1
                    await asyncio.sleep(0.1)  # prevent tight crash loop

        finally:
            # -------------------------
            # CLEAN EXIT (CRITICAL FIX)
            # -------------------------
            self.current_task = None
            logger.info(f"✅ Worker {self.worker_id} stopped safely")

    async def _process_task(self, task: Task):
        """Robust task execution with timeout, cancellation safety, and clean completion"""

        start_time = datetime.utcnow()

        try:
            # -------------------------
            # VALIDATION (CRITICAL)
            # -------------------------
            if not task or not hasattr(task, "id"):
                return

            timeout = max(getattr(task, "timeout_seconds", 30), 1)

            # -------------------------
            # EXECUTE TASK (SAFE TIMEOUT)
            # -------------------------
            result = await asyncio.wait_for(self.handler(task), timeout=timeout)

            # -------------------------
            # NORMALIZE RESULT
            # -------------------------
            if result is None:
                result = {"status": "completed", "note": "No result returned"}

            # -------------------------
            # COMPLETE TASK (NON-BLOCKING SAFE)
            # -------------------------
            await self.queue.complete(task.id, result=result)

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"✅ Task {task.id} completed in {duration:.2f}s")

        except asyncio.TimeoutError:
            # -------------------------
            # TIMEOUT HANDLING (CRITICAL FIX)
            # -------------------------
            error_msg = f"Timeout after {timeout}s"

            logger.warning(f"⏱️ Task timeout | {task.id}")

            try:
                await self.queue.complete(task.id, error=error_msg)
            except Exception:
                pass

        except asyncio.CancelledError:
            # -------------------------
            # CANCELLATION (CRITICAL FIX)
            # -------------------------
            logger.info(f"🛑 Task cancelled: {task.id}")

            try:
                await self.queue.complete(task.id, error="cancelled")
            except Exception:
                pass

            raise  # IMPORTANT: propagate cancellation

        except Exception as e:
            # -------------------------
            # SAFE ERROR HANDLING
            # -------------------------
            error_msg = str(e)

            logger.error(f"❌ Task {task.id} failed: {error_msg}")

            try:
                await self.queue.complete(task.id, error=error_msg)
            except Exception:
                pass

        finally:
            # -------------------------
            # FINAL SAFETY (NO LEAKS)
            # -------------------------
            try:
                if hasattr(self, "current_task") and self.current_task == task:
                    self.current_task = None
            except Exception:
                pass

    async def stop(self):
        """Graceful worker stop with safe cancellation and metrics"""

        try:
            logger.info(f"🛑 Stopping worker {self.worker_id}")

            # -------------------------
            # SIGNAL STOP (CRITICAL)
            # -------------------------
            self.is_running = False

            # -------------------------
            # CANCEL WORKER TASK (SAFE)
            # -------------------------
            if getattr(self, "_worker_task", None):
                self._worker_task.cancel()

                try:
                    await asyncio.wait_for(self._worker_task, timeout=5)
                except asyncio.TimeoutError:
                    logger.warning(f"⚠ Worker {self.worker_id} cancel timeout")
                except asyncio.CancelledError:
                    pass

            # -------------------------
            # CLEAN CURRENT TASK (IMPORTANT FIX)
            # -------------------------
            if getattr(self, "current_task", None):
                try:
                    await self.queue.complete(
                        self.current_task.id, error="worker stopped"
                    )
                except Exception:
                    pass
                finally:
                    self.current_task = None

            # -------------------------
            # SAFE METRICS CALCULATION
            # -------------------------
            processed = max(self.tasks_processed, 0)
            avg_time = self.total_processing_time / processed if processed > 0 else 0

            logger.info(
                f"✅ Worker {self.worker_id} stopped | "
                f"processed={processed}, errors={self.errors_count}, "
                f"avg_time={avg_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"❌ Worker stop failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        avg_time = (
            self.total_processing_time / self.tasks_processed
            if self.tasks_processed > 0
            else 0
        )
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "tasks_processed": self.tasks_processed,
            "errors_count": self.errors_count,
            "avg_processing_time": avg_time,
            "current_task": self.current_task.id if self.current_task else None,
        }


# ==================== Task Scheduler ====================


class TaskScheduler:
    """Schedule tasks for future execution with cron support"""

    def __init__(self, queue: TaskQueue):
        self.queue = queue
        self._scheduled_tasks: List[Tuple[datetime, Task, Optional[str]]] = []
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._cron_jobs: Dict[str, Dict] = {}

        logger.info(f"TaskScheduler created for queue '{queue.name}'")

    async def schedule(
        self,
        task: Task,
        delay_seconds: float = 0,
        cron: Optional[str] = None,
        cron_id: Optional[str] = None,
    ) -> str:
        """Safe scheduling with validation, deduplication, and stable timing"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL)
            # -------------------------
            if not task or not hasattr(task, "id"):
                raise ValueError("Invalid task for scheduling")

            delay = max(delay_seconds, 0)
            execution_time = datetime.utcnow() + timedelta(seconds=delay)

            # -------------------------
            # PREVENT DUPLICATES (FIX)
            # -------------------------
            if any(t[1].id == task.id for t in self._scheduled_tasks):
                logger.warning(f"Task {task.id} already scheduled → skipping duplicate")
                return task.id

            # -------------------------
            # SAFE INSERT (LOCK MINIMAL)
            # -------------------------
            async with self._lock:
                heapq.heappush(self._scheduled_tasks, (execution_time, task, cron))

                # -------------------------
                # CRON HANDLING (SAFE)
                # -------------------------
                if cron and cron_id:
                    self._cron_jobs[cron_id] = {
                        "task": task,
                        "cron": cron,
                        "next_execution": execution_time,
                    }

            logger.info(f"⏳ Task {task.id} scheduled for {execution_time.isoformat()}")

            return task.id

        except Exception as e:
            logger.error(f"❌ Schedule failed: {e}")
            return ""

    async def start(self):
        """Start scheduler"""
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        logger.info("TaskScheduler started")

    async def _run_scheduler(self):
        """Robust scheduler loop with drift control + non-blocking execution"""

        try:
            idle_sleep = 0.05  # prevents CPU spike

            while self.is_running:
                try:
                    now = datetime.utcnow()
                    tasks_to_execute = []

                    # -------------------------
                    # COLLECT READY TASKS (MIN LOCK)
                    # -------------------------
                    async with self._lock:
                        while (
                            self._scheduled_tasks and self._scheduled_tasks[0][0] <= now
                        ):
                            execution_time, task, cron = heapq.heappop(
                                self._scheduled_tasks
                            )

                            # avoid duplicate execution
                            if not task or not hasattr(task, "id"):
                                continue

                            tasks_to_execute.append((task, cron))

                            # -------------------------
                            # CRON RESCHEDULE (SAFE)
                            # -------------------------
                            if cron:
                                try:
                                    next_time = self._calculate_next_cron(
                                        cron, execution_time
                                    )
                                    if next_time:
                                        heapq.heappush(
                                            self._scheduled_tasks,
                                            (next_time, task, cron),
                                        )
                                except Exception:
                                    continue

                    # -------------------------
                    # EXECUTE OUTSIDE LOCK (CRITICAL FIX)
                    # -------------------------
                    for task, _ in tasks_to_execute:
                        try:
                            await self.queue.put(task)
                        except Exception:
                            continue

                    # -------------------------
                    # SMART SLEEP (CRITICAL FIX)
                    # -------------------------
                    if not tasks_to_execute:
                        await asyncio.sleep(idle_sleep)

                    else:
                        await asyncio.sleep(0)  # yield control

                except asyncio.CancelledError:
                    logger.info("🛑 Scheduler stopped")
                    break

                except Exception as e:
                    logger.error(f"❌ Scheduler error: {e}")
                    await asyncio.sleep(0.5)  # prevent crash loop

        finally:
            logger.info("✅ Scheduler exited safely")

    def _calculate_next_cron(
        self, cron: str, last_time: datetime
    ) -> Optional[datetime]:
        """Safe cron parser (basic) with validation and forward-only scheduling"""

        try:
            if not cron or not isinstance(cron, str):
                return None

            parts = cron.strip().split()
            if len(parts) != 5:
                return None

            minute, hour, day, month, weekday = parts

            # -------------------------
            # START FROM NEXT MINUTE
            # -------------------------
            next_time = (last_time + timedelta(minutes=1)).replace(
                second=0, microsecond=0
            )

            # -------------------------
            # SAFE PARSER
            # -------------------------
            def _match(value: int, pattern: str) -> bool:
                if pattern == "*":
                    return True
                try:
                    return int(pattern) == value
                except Exception:
                    return False

            # -------------------------
            # SEARCH NEXT VALID TIME (LIMITED LOOP)
            # -------------------------
            for _ in range(1440):  # max 24 hours search (safety limit)
                if (
                    _match(next_time.minute, minute)
                    and _match(next_time.hour, hour)
                    and _match(next_time.day, day)
                    and _match(next_time.month, month)
                    and _match(next_time.weekday(), weekday if weekday != "*" else "*")
                ):
                    return next_time

                next_time += timedelta(minutes=1)

            return None  # no valid time found (safe fallback)

        except Exception as e:
            logger.debug(f"Cron parse failed: {e}")
            return None

    async def stop(self):
        """Stop scheduler"""
        self.is_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskScheduler stopped")


# ==================== Task Queue Manager ====================


class TaskQueueManager:
    """Manages multiple task queues with MongoDB integration"""

    def __init__(self):
        self.queues: Dict[str, TaskQueue] = {}
        self.workers: Dict[str, List[TaskWorker]] = {}
        self.schedulers: Dict[str, TaskScheduler] = {}
        self._lock = asyncio.Lock()
        self._persist_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.is_running = False

        # Statistics tracking
        self.global_metrics = {
            "start_time": None,
            "total_tasks_processed": 0,
            "total_errors": 0,
            "active_queues": 0,
            "active_workers": 0,
        }

        TaskQueueConfig.ensure_directories()

        logger.info("TaskQueueManager initialized with MongoDB support")

    async def create_queue(
        self, name: str, queue_type: QueueType = QueueType.PRIORITY
    ) -> TaskQueue:
        """Create a new task queue"""
        async with self._lock:
            if name in self.queues:
                logger.warning(f"Queue '{name}' already exists")
                return self.queues[name]

            queue = TaskQueue(name, queue_type)
            await queue.load()
            self.queues[name] = queue
            self.workers[name] = []
            self.schedulers[name] = TaskScheduler(queue)

            # Start cleanup for this queue and track the task
            queue._cleanup_task = asyncio.create_task(queue.start_cleanup())

            self.global_metrics["active_queues"] = len(self.queues)

            logger.info(f"Queue '{name}' created")
            return queue

    async def add_worker(
        self, queue_name: str, worker_id: str, handler: Callable
    ) -> TaskWorker:
        """Add worker to queue"""
        if queue_name not in self.queues:
            await self.create_queue(queue_name)

        worker = TaskWorker(self.queues[queue_name], worker_id, handler)
        self.workers[queue_name].append(worker)

        if self.is_running:
            await worker.start()

        self.global_metrics["active_workers"] = sum(
            len(w) for w in self.workers.values()
        )

        logger.info(f"Worker '{worker_id}' added to queue '{queue_name}'")
        return worker

    async def remove_worker(self, queue_name: str, worker_id: str) -> bool:
        """Remove worker from queue"""
        if queue_name not in self.workers:
            return False

        for i, worker in enumerate(self.workers[queue_name]):
            if worker.worker_id == worker_id:
                await worker.stop()
                self.workers[queue_name].pop(i)
                self.global_metrics["active_workers"] = sum(
                    len(w) for w in self.workers.values()
                )
                logger.info(f"Worker '{worker_id}' removed from queue '{queue_name}'")
                return True

        return False

    async def start(self):
        """Robust, non-blocking startup with parallel initialization"""

        try:
            logger.info("🚀 Starting TaskQueueManager...")

            # -------------------------
            # INIT STATE
            # -------------------------
            self.is_running = True
            self._loop = asyncio.get_running_loop()
            self.global_metrics["start_time"] = datetime.utcnow()

            # -------------------------
            # START SCHEDULERS (PARALLEL - CRITICAL FIX)
            # -------------------------
            scheduler_tasks = []
            for scheduler in self.schedulers.values():
                try:
                    scheduler_tasks.append(asyncio.create_task(scheduler.start()))
                except Exception as e:
                    logger.error(f"Scheduler start failed: {e}")

            # -------------------------
            # START WORKERS (PARALLEL - CRITICAL FIX)
            # -------------------------
            worker_tasks = []
            for workers in self.workers.values():
                for worker in workers:
                    try:
                        worker_tasks.append(asyncio.create_task(worker.start()))
                    except Exception as e:
                        logger.error(f"Worker start failed: {e}")

            # wait for all startup tasks safely
            await asyncio.gather(
                *scheduler_tasks, *worker_tasks, return_exceptions=True
            )

            # -------------------------
            # BACKGROUND TASKS (SAFE)
            # -------------------------
            self._persist_task = asyncio.create_task(self._persist_loop())
            self._health_task = asyncio.create_task(self._health_check_loop())

            logger.info("✅ TaskQueueManager started successfully")

        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            self.is_running = False

    async def _persist_loop(self):
        """Robust periodic persistence loop with parallel execution + cancellation safety"""

        try:
            interval = max(getattr(TaskQueueConfig, "PERSIST_INTERVAL_SECONDS", 60), 1)

            while self.is_running:
                try:
                    # -------------------------
                    # INTERRUPTIBLE SLEEP (CRITICAL FIX)
                    # -------------------------
                    await asyncio.wait_for(
                        asyncio.sleep(interval), timeout=interval + 1
                    )

                    if not self.is_running:
                        break

                    # -------------------------
                    # PARALLEL PERSIST (CRITICAL FIX)
                    # -------------------------
                    tasks = []
                    for queue in self.queues.values():
                        try:
                            tasks.append(asyncio.create_task(queue.persist()))
                        except Exception:
                            continue

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                except asyncio.CancelledError:
                    logger.info("🛑 Persist loop cancelled")
                    break

                except Exception as e:
                    logger.error(f"❌ Persistence loop error: {e}")
                    await asyncio.sleep(1)  # prevent tight crash loop

        finally:
            logger.info("✅ Persist loop stopped safely")

    async def _health_check_loop(self):
        """Robust health monitoring loop with safe restart + parallel checks"""

        try:
            interval = max(getattr(TaskQueueConfig, "HEALTH_CHECK_INTERVAL", 30), 1)

            while self.is_running:
                try:
                    # -------------------------
                    # INTERRUPTIBLE SLEEP (CRITICAL FIX)
                    # -------------------------
                    await asyncio.wait_for(
                        asyncio.sleep(interval), timeout=interval + 1
                    )

                    if not self.is_running:
                        break

                    # -------------------------
                    # QUEUE HEALTH CHECK
                    # -------------------------
                    for name, queue in self.queues.items():
                        try:
                            if queue.metrics.error_count > 10:
                                logger.warning(
                                    f"⚠ Queue '{name}' high errors: {queue.metrics.error_count}"
                                )

                            if queue.metrics.current_queue_size > 100:
                                logger.warning(
                                    f"⚠ Queue '{name}' backlog: {queue.metrics.current_queue_size}"
                                )
                        except Exception:
                            continue

                    # -------------------------
                    # WORKER HEALTH (SAFE RESTART)
                    # -------------------------
                    restart_tasks = []

                    for workers in self.workers.values():
                        for worker in workers:
                            try:
                                if not worker.is_running:
                                    logger.warning(
                                        f"🔄 Restarting worker {worker.worker_id}"
                                    )

                                    # prevent restart storm
                                    if getattr(worker, "_restarting", False):
                                        continue

                                    worker._restarting = True

                                    async def _restart(w=worker):
                                        try:
                                            await w.start()
                                        finally:
                                            w._restarting = False

                                    restart_tasks.append(
                                        asyncio.create_task(_restart())
                                    )

                            except Exception:
                                continue

                    if restart_tasks:
                        await asyncio.gather(*restart_tasks, return_exceptions=True)

                    # -------------------------
                    # GLOBAL METRICS (SAFE)
                    # -------------------------
                    try:
                        self.global_metrics["total_tasks_processed"] = sum(
                            q.metrics.total_tasks_processed
                            for q in self.queues.values()
                        )
                        self.global_metrics["total_errors"] = sum(
                            q.metrics.total_tasks_failed for q in self.queues.values()
                        )
                    except Exception:
                        pass

                except asyncio.CancelledError:
                    logger.info("🛑 Health check loop cancelled")
                    break

                except Exception as e:
                    logger.error(f"❌ Health check error: {e}")
                    await asyncio.sleep(1)  # prevent tight crash loop

        finally:
            logger.info("✅ Health check loop stopped safely")

    async def get_queue(self, name: str) -> Optional[TaskQueue]:
        """Get queue by name"""
        return self.queues.get(name)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all queues"""
        metrics = {}
        for name, queue in self.queues.items():
            metrics[name] = await queue.get_metrics()

        # Add global metrics
        metrics["_global"] = {
            "start_time": (
                self.global_metrics["start_time"].isoformat()
                if self.global_metrics["start_time"]
                else None
            ),
            "uptime_seconds": (
                (datetime.now() - self.global_metrics["start_time"]).total_seconds()
                if self.global_metrics["start_time"]
                else 0
            ),
            "total_tasks_processed": self.global_metrics["total_tasks_processed"],
            "total_errors": self.global_metrics["total_errors"],
            "active_queues": self.global_metrics["active_queues"],
            "active_workers": self.global_metrics["active_workers"],
            "is_running": self.is_running,
        }

        return metrics

    async def get_all_workers_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers"""
        stats = {}
        for queue_name, workers in self.workers.items():
            stats[queue_name] = [worker.get_stats() for worker in workers]
        return stats

    async def shutdown(self):
        """Graceful, parallel shutdown of all queues, workers, and background tasks"""

        logger.info("🛑 Shutting down TaskQueueManager...")
        self.is_running = False

        try:
            # -------------------------
            # STOP SCHEDULERS (PARALLEL)
            # -------------------------
            scheduler_tasks = [
                asyncio.create_task(s.stop()) for s in self.schedulers.values()
            ]
            await asyncio.gather(*scheduler_tasks, return_exceptions=True)

            # -------------------------
            # STOP WORKERS (PARALLEL)
            # -------------------------
            worker_tasks = []
            for workers in self.workers.values():
                for worker in workers:
                    worker_tasks.append(asyncio.create_task(worker.stop()))

            await asyncio.gather(*worker_tasks, return_exceptions=True)

            # -------------------------
            # CANCEL BACKGROUND TASKS (SAFE)
            # -------------------------
            async def _cancel(task):
                if task:
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5)
                    except asyncio.CancelledError:
                        pass
                    except asyncio.TimeoutError:
                        logger.warning("⚠ Background task cancel timeout")

            await asyncio.gather(
                _cancel(self._persist_task),
                _cancel(self._health_task),
                return_exceptions=True,
            )

            # -------------------------
            # FINAL PERSIST + QUEUE SHUTDOWN (PARALLEL)
            # -------------------------
            queue_tasks = []
            for queue in self.queues.values():
                queue_tasks.append(asyncio.create_task(queue.persist()))
                queue_tasks.append(asyncio.create_task(queue.shutdown()))

            await asyncio.gather(*queue_tasks, return_exceptions=True)

            logger.info("✅ TaskQueueManager shutdown complete")

        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")


# ==================== Global Instance ====================

_task_queue_manager: Optional[TaskQueueManager] = None
_manager_lock = asyncio.Lock()


async def get_task_queue_manager() -> TaskQueueManager:
    """Get global task queue manager instance"""
    global _task_queue_manager
    async with _manager_lock:
        if _task_queue_manager is None:
            _task_queue_manager = TaskQueueManager()
    return _task_queue_manager


async def init_task_queue_system():
    """Initialize the entire task queue system"""
    manager = await get_task_queue_manager()
    await manager.start()
    return manager


async def shutdown_task_queue_system():
    """Shutdown the task queue system"""
    global _task_queue_manager
    if _task_queue_manager:
        await _task_queue_manager.shutdown()
        _task_queue_manager = None


# ==================== Decorators ====================


def task_handler(queue_name: str = "default", max_retries: int = 3, timeout: int = 30):
    """Decorator to register a function as a task handler"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._is_task_handler = True
        wrapper._queue_name = queue_name
        wrapper._max_retries = max_retries
        wrapper._timeout = timeout

        return wrapper

    return decorator


# ==================== Example Usage ====================


@task_handler(queue_name="main", max_retries=2, timeout=10)
async def example_task_handler(task: Task):
    """Example task handler with decorator"""
    logger.info(f"Processing task {task.id}: {task.name}")

    # Simulate work
    await asyncio.sleep(1)

    # Access task data
    data = task.data.get("value", 0)
    result = data * 2

    return {
        "result": result,
        "original": data,
        "processed_at": datetime.now().isoformat(),
    }


async def example_usage():
    """Example usage of task queue system"""
    # Initialize
    manager = await init_task_queue_system()

    # Create queue
    queue = await manager.create_queue("main", QueueType.PRIORITY)

    # Add worker with handler
    await manager.add_worker("main", "worker-1", example_task_handler)

    # Create and submit tasks
    for i in range(10):
        task = Task(
            name=f"example_task_{i}",
            priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH,
            data={"value": i, "iteration": i},
            timeout_seconds=5,
            tags=["example", f"iteration_{i}"],
            metadata={"source": "example", "created_by": "demo"},
            owner_id="user_123",
            session_id="session_456",
        )
        await queue.put(task)
        logger.info(f"Submitted task {task.id}")

    # Schedule a future task
    scheduler = manager.schedulers["main"]
    future_task = Task(
        name="future_task",
        data={"value": 999, "type": "scheduled"},
        tags=["scheduled", "future"],
    )
    await scheduler.schedule(future_task, delay_seconds=10)
    logger.info(f"Scheduled future task {future_task.id}")

    # Monitor progress
    await asyncio.sleep(15)

    # Get metrics
    metrics = await manager.get_metrics()
    logger.info(f"Queue Metrics: {json.dumps(metrics, indent=2, default=str)}")

    # Get worker stats
    worker_stats = await manager.get_all_workers_stats()
    logger.info(f"Worker Stats: {json.dumps(worker_stats, indent=2)}")

    # Get specific task status
    if future_task:
        status = await queue.get_status(future_task.id)
        logger.info(f"Task {future_task.id} status: {status}")

    # Shutdown
    await shutdown_task_queue_system()

    logger.info("Example completed")


# ==================== Export ====================

__all__ = [
    "Task",
    "TaskQueue",
    "TaskWorker",
    "TaskScheduler",
    "TaskQueueManager",
    "TaskPriority",
    "TaskStatus",
    "QueueType",
    "TaskQueueConfig",
    "task_handler",
    "get_task_queue_manager",
    "init_task_queue_system",
    "shutdown_task_queue_system",
]

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
