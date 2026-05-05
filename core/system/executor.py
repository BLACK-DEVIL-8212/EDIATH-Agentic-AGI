"""
Advanced Executor - Parallel Task Execution Engine
"""

import asyncio
import uuid

from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, value: str):
        """Safe parser for ExecutionStatus with normalization and fallback"""

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if value is None:
                return cls.PENDING

            value = str(value).strip().lower()

            if not value:
                return cls.PENDING

            # ------------------------
            # 🔥 DIRECT MATCH (FAST PATH)
            # ------------------------
            try:
                return cls(value)
            except ValueError:
                pass

            # ------------------------
            # 🔥 ALIAS MAP (CRITICAL FIX)
            # ------------------------
            alias_map = {
                "done": cls.COMPLETED,
                "success": cls.COMPLETED,
                "ok": cls.COMPLETED,
                "in_progress": cls.RUNNING,
                "processing": cls.RUNNING,
                "error": cls.FAILED,
                "fail": cls.FAILED,
                "stopped": cls.CANCELLED,
                "abort": cls.CANCELLED,
            }

            if value in alias_map:
                return alias_map[value]

            # ------------------------
            # 🔥 PARTIAL MATCH (SAFE)
            # ------------------------
            for status in cls:
                if value in status.value:
                    return status

            # ------------------------
            # 🔥 DEFAULT FALLBACK
            # ------------------------
            return cls.PENDING

        except Exception:
            # ------------------------
            # 🔥 HARD FAILSAFE
            # ------------------------
            return cls.PENDING


class Task(ABC):
    def __init__(
        self, task_id: str, name: str, retries: int = 0, timeout: Optional[int] = None
    ):
        try:

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not task_id:
                task_id = str(uuid.uuid4())

            if not name:
                raise ValueError("Task name cannot be empty")

            # ------------------------
            # 🔥 NORMALIZATION
            # ------------------------
            self.task_id = str(task_id).strip()
            self.name = str(name).strip()

            self.retries = max(0, int(retries))
            self.timeout = int(timeout) if timeout else None

            # ------------------------
            # 🔥 STATUS
            # ------------------------
            self.status = ExecutionStatus.PENDING

            # ------------------------
            # 🔥 TIMESTAMPS
            # ------------------------
            self.created_at = datetime.utcnow()
            self.started_at = None
            self.completed_at = None

            # ------------------------
            # 🔥 EXECUTION DATA
            # ------------------------
            self.result = None
            self.error = None

            # ------------------------
            # 🔥 INTERNAL STATE
            # ------------------------
            self._future: Optional[asyncio.Task] = None
            self._retry_count = 0
            self._cancelled = False

            # ------------------------
            # 🔥 METADATA
            # ------------------------
            self.metadata = {}
            self.tags = []

        except Exception as e:
            raise RuntimeError(f"Task initialization failed: {e}")

    @abstractmethod
    async def execute(self) -> Any:
        """Robust task execution with retries, timeout, and safe state handling"""

        # ------------------------
        # 🔥 PRE-CHECK (CRITICAL FIX)
        # ------------------------
        if getattr(self, "status", None) != ExecutionStatus.PENDING:
            raise RuntimeError("Task already started or finished")

        if not hasattr(self, "run") or not callable(self.run):
            raise RuntimeError("Task missing run() method")

        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self._retry_count = getattr(self, "_retry_count", 0)

        try:
            while self._retry_count <= int(getattr(self, "retries", 0)):

                try:
                    # ------------------------
                    # 🔥 EXECUTION CALL (TIMEOUT SAFE)
                    # ------------------------
                    coro = self.run()

                    if asyncio.iscoroutine(coro):
                        if getattr(self, "timeout", None):
                            self.result = await asyncio.wait_for(
                                coro, timeout=self.timeout
                            )
                        else:
                            self.result = await coro
                    else:
                        # Support sync run()
                        self.result = coro

                    # ------------------------
                    # 🔥 SUCCESS
                    # ------------------------
                    self.status = ExecutionStatus.COMPLETED
                    self.completed_at = datetime.utcnow()
                    return self.result

                except asyncio.TimeoutError:
                    self.error = "timeout"
                    self._retry_count += 1

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    self.error = str(e)
                    self._retry_count += 1

                # ------------------------
                # 🔥 RETRY EXIT
                # ------------------------
                if self._retry_count > getattr(self, "retries", 0):
                    break

                # ------------------------
                # 🔥 BACKOFF (CRITICAL FIX)
                # ------------------------
                await asyncio.sleep(min(2.0, 0.2 * (2**self._retry_count)))

            # ------------------------
            # 🔥 FAILURE
            # ------------------------
            self.status = ExecutionStatus.FAILED
            self.completed_at = datetime.utcnow()
            return None

        except asyncio.CancelledError:
            # ------------------------
            # 🔥 CANCELLED (CLEAN STATE)
            # ------------------------
            self.status = ExecutionStatus.CANCELLED
            self.completed_at = datetime.utcnow()
            self._cancelled = True
            return None

        except Exception as e:
            # ------------------------
            # 🔥 CRITICAL FAILURE
            # ------------------------
            self.status = ExecutionStatus.FAILED
            self.error = str(e)
            self.completed_at = datetime.utcnow()
            return None

    async def run(self):
        """Robust task runner with retries, timeout, and safe state handling"""

        # ------------------------
        # 🔥 PRE-CHECK (CRITICAL FIX)
        # ------------------------
        if getattr(self, "status", None) == ExecutionStatus.RUNNING:
            raise RuntimeError("Task already running")

        if not hasattr(self, "_execute_fn") or not callable(self._execute_fn):
            raise RuntimeError("No execution function defined")

        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self._retry_count = getattr(self, "_retry_count", 0)

        try:
            while self._retry_count <= int(getattr(self, "retries", 0)):

                try:
                    # ------------------------
                    # 🔥 EXECUTION (SYNC + ASYNC SAFE)
                    # ------------------------
                    result = self._execute_fn()

                    if asyncio.iscoroutine(result):
                        if getattr(self, "timeout", None):
                            result = await asyncio.wait_for(
                                result, timeout=self.timeout
                            )
                        else:
                            result = await result

                    # ------------------------
                    # 🔥 SUCCESS
                    # ------------------------
                    self.result = result
                    self.status = ExecutionStatus.COMPLETED
                    return result

                except asyncio.TimeoutError:
                    self.error = "timeout"
                    self._retry_count += 1

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    self.error = str(e)
                    self._retry_count += 1

                # ------------------------
                # 🔥 RETRY EXIT
                # ------------------------
                if self._retry_count > getattr(self, "retries", 0):
                    break

                # ------------------------
                # 🔥 SAFE LOGGING
                # ------------------------
                try:
                    self.logger.warning(
                        f"Retrying task: {getattr(self, 'name', 'unknown')} "
                        f"({self._retry_count}/{self.retries})"
                    )
                except Exception:
                    pass

                # ------------------------
                # 🔥 EXPONENTIAL BACKOFF (CRITICAL FIX)
                # ------------------------
                await asyncio.sleep(min(2.0, 0.2 * (2**self._retry_count)))

            # ------------------------
            # 🔥 FAILURE
            # ------------------------
            self.status = ExecutionStatus.FAILED

            try:
                self.logger.error(
                    f"Task failed: {getattr(self, 'name', 'unknown')} - {self.error}"
                )
            except Exception:
                pass

            return None

        except asyncio.CancelledError:
            # ------------------------
            # 🔥 CANCEL HANDLING
            # ------------------------
            self.status = ExecutionStatus.CANCELLED
            self.error = "cancelled"
            return None

        except Exception as e:
            # ------------------------
            # 🔥 CRITICAL FAILURE
            # ------------------------
            self.status = ExecutionStatus.FAILED
            self.error = str(e)
            return None

        finally:
            # ------------------------
            # 🔥 ALWAYS EXECUTES
            # ------------------------
            self.completed_at = datetime.utcnow()


class Executor:
    def __init__(self, max_workers: int = 4):

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            max_workers = int(max_workers)
            if max_workers <= 0:
                raise ValueError("max_workers must be > 0")

            # ------------------------
            # 🔥 CORE CONFIG
            # ------------------------
            self.max_workers = max_workers

            # ------------------------
            # 🔥 TASK STORAGE
            # ------------------------
            self.tasks: Dict[str, Task] = {}
            self._running_tasks: Dict[str, asyncio.Task] = {}

            # ------------------------
            # 🔥 EXECUTION METRICS
            # ------------------------
            self.execution_count = 0
            self.completed_count = 0
            self.failed_count = 0
            self.cancelled_count = 0

            # ------------------------
            # 🔥 CONCURRENCY CONTROL
            # ------------------------
            self._semaphore = asyncio.Semaphore(self.max_workers)

            # ------------------------
            # 🔥 QUEUE SYSTEM (OPTIONAL FUTURE)
            # ------------------------
            self._queue: asyncio.Queue = asyncio.Queue()

            # ------------------------
            # 🔥 CONTROL FLAGS
            # ------------------------
            self._shutdown = False
            self._paused = False

            # ------------------------
            # 🔥 LOCK (THREAD SAFETY)
            # ------------------------
            self._lock = asyncio.Lock()

            # ------------------------
            # 🔥 WORKER TASK
            # ------------------------
            self._worker_task: Optional[asyncio.Task] = None

            self._registry = {}

        except Exception as e:
            raise RuntimeError(f"TaskExecutor initialization failed: {e}")

    def register(self, name: str, func) -> Dict[str, Any]:
        """Register a function safely and return executor status"""

        try:
            from datetime import datetime

            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            name = str(name or "").strip()
            if not name:
                return {"success": False, "error": "invalid_name"}

            if not callable(func):
                return {"success": False, "error": "function_not_callable"}

            # ------------------------
            # 🔥 INIT REGISTRY (CRITICAL FIX)
            # ------------------------
            if not hasattr(self, "_registry") or not isinstance(self._registry, dict):
                self._registry = {}

            # ------------------------
            # 🔥 REGISTER FUNCTION
            # ------------------------
            self._registry[name] = func

            # ------------------------
            # 🔥 SAFE STATE EXTRACTION
            # ------------------------
            tasks = getattr(self, "tasks", {}) or {}
            running = getattr(self, "_running_tasks", {}) or {}

            total_tasks = len(tasks)
            running_tasks = len(running)

            try:
                max_workers = int(getattr(self, "max_workers", 0) or 0)
            except Exception:
                max_workers = 0

            # ------------------------
            # 🔥 LOAD CALCULATION
            # ------------------------
            load = round(running_tasks / max_workers, 3) if max_workers > 0 else 0.0

            # ------------------------
            # 🔥 HEALTH LOGIC
            # ------------------------
            if max_workers == 0:
                health = "misconfigured"
            elif load >= 1.0:
                health = "overloaded"
            elif load >= 0.7:
                health = "high_load"
            elif total_tasks > 0 and running_tasks == 0:
                health = "idle"
            else:
                health = "healthy"

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            execution_count = int(getattr(self, "execution_count", 0) or 0)
            completed = int(getattr(self, "completed_count", 0) or 0)
            failed = int(getattr(self, "failed_count", 0) or 0)
            cancelled = int(getattr(self, "cancelled_count", 0) or 0)

            success_rate = (
                round(completed / execution_count, 3) if execution_count > 0 else 0.0
            )

            # ------------------------
            # 🔥 REGISTERED FUNCTIONS
            # ------------------------
            registered_functions = list(self._registry.keys())

            # ------------------------
            # 🔥 TIMESTAMP
            # ------------------------
            timestamp = datetime.utcnow().isoformat()

            # ------------------------
            # 🔥 FINAL RESPONSE
            # ------------------------
            return {
                "success": True,
                "name": "Executor",
                "version": "1.0",
                "status": "ready",
                "timestamp": timestamp,
                "registered_function": name,
                "workers": {
                    "max": max_workers,
                    "running": running_tasks,
                    "available": max(max_workers - running_tasks, 0),
                    "load": load,
                },
                "tasks": {
                    "total": total_tasks,
                    "queued": max(total_tasks - running_tasks, 0),
                },
                "metrics": {
                    "execution_count": execution_count,
                    "completed": completed,
                    "failed": failed,
                    "cancelled": cancelled,
                    "success_rate": success_rate,
                },
                "functions": {
                    "registered": len(registered_functions),
                    "names": registered_functions,
                },
                "health": health,
            }

        except Exception as e:
            try:
                self.logger.error(f"Executor register critical error: {e}")
            except Exception:
                pass

            return {
                "success": False,
                "name": "Executor",
                "version": "unknown",
                "status": "error",
                "health": "critical",
                "message": "Registration failed",
            }

    def register_function(self, name: str, func, overwrite: bool = True) -> bool:
        """Register callable function into executor (safe + production-ready)"""

        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not name or not isinstance(name, str):
                raise ValueError("Function name must be a non-empty string")

            name = name.strip()

            if not callable(func):
                raise ValueError(f"{name} is not callable")

            # ------------------------
            # 🔥 INIT REGISTRY (SAFE)
            # ------------------------
            if not hasattr(self, "_registry") or not isinstance(self._registry, dict):
                self._registry = {}

            if not hasattr(self, "_function_meta") or not isinstance(
                self._function_meta, dict
            ):
                self._function_meta = {}

            # ------------------------
            # 🔥 DUPLICATE HANDLING
            # ------------------------
            if name in self._registry and not overwrite:
                raise ValueError(f"Function already exists: {name}")

            # ------------------------
            # 🔥 STORE FUNCTION (CRITICAL FIX)
            # ------------------------
            self._registry[name] = func

            # ------------------------
            # 🔥 METADATA (IMPROVED)
            # ------------------------
            try:
                is_async = asyncio.iscoroutinefunction(func)
            except Exception:
                is_async = False

            self._function_meta[name] = {
                "registered_at": time.time(),
                "is_async": is_async,
                "callable": True,
                "name": name,
            }

            # ------------------------
            # 🔥 OPTIONAL METRICS
            # ------------------------
            try:
                self.function_count = int(getattr(self, "function_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"✅ Function registered: {name}")
            except Exception:
                pass

            return True

        except Exception as e:
            # ------------------------
            # 🔥 ERROR TRACKING (CRITICAL FIX)
            # ------------------------
            try:
                self.error_count = int(getattr(self, "error_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Function registration failed: {name} - {e}")
            except Exception:
                pass

            return False

    async def execute_function(
        self, name: str, *args, timeout: Optional[int] = None, **kwargs
    ):
        """Execute registered function dynamically (safe + production-ready)"""

        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not name or not isinstance(name, str):
                raise ValueError("Invalid function name")

            name = name.strip()

            registry = getattr(self, "_registry", {}) or {}

            if name not in registry:
                raise ValueError(f"Function not found: {name}")

            func = registry[name]

            # ------------------------
            # 🔥 METADATA (SAFE)
            # ------------------------
            meta = getattr(self, "_function_meta", {}).get(name, {})

            try:
                is_async = meta.get("is_async", asyncio.iscoroutinefunction(func))
            except Exception:
                is_async = False

            # ------------------------
            # 🔥 EXECUTION WRAPPER (CRITICAL FIX)
            # ------------------------
            async def _run():
                try:
                    if is_async:
                        return await func(*args, **kwargs)
                    else:
                        return await asyncio.to_thread(func, *args, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"{name} execution failed: {e}")

            # ------------------------
            # 🔥 TIMEOUT SUPPORT
            # ------------------------
            if timeout and timeout > 0:
                result = await asyncio.wait_for(_run(), timeout=timeout)
            else:
                result = await _run()

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.execution_count = int(getattr(self, "execution_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(
                        f"🚀 Function executed: {name} "
                        f"({round(time.time() - start, 3)}s)"
                    )
            except Exception:
                pass

            return {
                "success": True,
                "function": name,
                "result": result,
                "execution_time": round(time.time() - start, 4),
            }

        except asyncio.TimeoutError:
            error_msg = f"Function timed out: {name}"

        except asyncio.CancelledError:
            raise

        except Exception as e:
            error_msg = str(e)

        # ------------------------
        # 🔥 ERROR HANDLING (CRITICAL FIX)
        # ------------------------
        try:
            self.failed_count = int(getattr(self, "failed_count", 0)) + 1
        except Exception:
            pass

        try:
            if hasattr(self, "logger"):
                self.logger.error(f"❌ Function execution failed: {name} - {error_msg}")
        except Exception:
            pass

        return {
            "success": False,
            "function": name,
            "error": error_msg,
            "execution_time": round(time.time() - start, 4),
        }

    # ------------------------
    # REGISTER
    # ------------------------
    def register_task(self, task: Task) -> str:
        """Register a task safely with validation and thread safety"""

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not task:
                raise ValueError("Task cannot be None")

            if not hasattr(task, "task_id"):
                raise ValueError("Invalid task object (missing task_id)")

            task_id = str(task.task_id).strip()

            if not task_id:
                raise ValueError("Task ID cannot be empty")

            # ------------------------
            # 🔥 INIT STORAGE (CRITICAL FIX)
            # ------------------------
            if not hasattr(self, "tasks") or not isinstance(self.tasks, dict):
                self.tasks = {}

            # ------------------------
            # 🔥 THREAD-SAFE REGISTRATION (BEST EFFORT)
            # ------------------------
            lock = getattr(self, "_lock", None)

            if lock and hasattr(lock, "acquire"):
                lock.acquire()
                locked = True
            else:
                locked = False

            try:
                # ------------------------
                # 🔥 DUPLICATE CHECK
                # ------------------------
                if task_id in self.tasks:
                    raise ValueError(f"Task already exists: {task_id}")

                # ------------------------
                # 🔥 REGISTER TASK
                # ------------------------
                self.tasks[task_id] = task

            finally:
                if locked:
                    try:
                        lock.release()
                    except Exception:
                        pass

            # ------------------------
            # 🔥 METADATA (SAFE)
            # ------------------------
            try:
                if hasattr(task, "metadata") and isinstance(task.metadata, dict):
                    task.metadata.setdefault(
                        "registered_at", datetime.utcnow().isoformat()
                    )
            except Exception:
                pass

            # ------------------------
            # 🔥 METRICS (OPTIONAL)
            # ------------------------
            try:
                self.task_count = int(getattr(self, "task_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Task registered: {task_id}")
            except Exception:
                pass

            return task_id

        except Exception as e:
            # ------------------------
            # 🔥 ERROR TRACKING
            # ------------------------
            try:
                self.error_count = int(getattr(self, "error_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Task registration failed: {e}")
            except Exception:
                pass

            raise

    # ------------------------
    # EXECUTE SINGLE
    # ------------------------
    async def execute_task(self, task: Task) -> Any:
        """Execute a task safely with concurrency control, metrics, and cleanup"""

        # ------------------------
        # 🔥 VALIDATION (CRITICAL FIX)
        # ------------------------
        if not task:
            raise ValueError("Task cannot be None")

        if not hasattr(task, "task_id"):
            raise ValueError("Invalid task (missing task_id)")

        # ------------------------
        # 🔥 INIT STATE (CRITICAL FIX)
        # ------------------------
        if not hasattr(self, "_running_tasks") or not isinstance(
            self._running_tasks, dict
        ):
            self._running_tasks = {}

        if not hasattr(self, "_semaphore"):
            self._semaphore = asyncio.Semaphore(10)  # safe default

        async with self._semaphore:
            try:
                task_id = str(task.task_id)

                # ------------------------
                # 🔥 DUPLICATE RUN CHECK
                # ------------------------
                if task_id in self._running_tasks:
                    raise RuntimeError(f"Task already running: {task_id}")

                # ------------------------
                # 🔥 CREATE TASK (SAFE)
                # ------------------------
                coro = task.run()
                if not asyncio.iscoroutine(coro):
                    raise RuntimeError("task.run() must return coroutine")

                future = asyncio.create_task(coro)

                task._future = future
                self._running_tasks[task_id] = future

                # ------------------------
                # 🔥 EXECUTION
                # ------------------------
                result = await future

                # ------------------------
                # 🔥 METRICS (SAFE)
                # ------------------------
                try:
                    self.execution_count = int(getattr(self, "execution_count", 0)) + 1
                except Exception:
                    pass

                try:
                    if task.status == ExecutionStatus.COMPLETED:
                        self.completed_count = (
                            int(getattr(self, "completed_count", 0)) + 1
                        )
                    elif task.status == ExecutionStatus.FAILED:
                        self.failed_count = int(getattr(self, "failed_count", 0)) + 1
                except Exception:
                    pass

                # ------------------------
                # 🔥 SAFE LOGGING
                # ------------------------
                try:
                    if hasattr(self, "logger"):
                        self.logger.info(
                            f"✅ Task completed: {getattr(task, 'name', task_id)}"
                        )
                except Exception:
                    pass

                return result

            except asyncio.CancelledError:
                # ------------------------
                # 🔥 CANCEL HANDLING
                # ------------------------
                task.status = ExecutionStatus.CANCELLED

                try:
                    self.cancelled_count = int(getattr(self, "cancelled_count", 0)) + 1
                except Exception:
                    pass

                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"⚠ Task cancelled: {getattr(task, 'name', task_id)}"
                        )
                except Exception:
                    pass

                raise

            except Exception as e:
                # ------------------------
                # 🔥 FAILURE HANDLING
                # ------------------------
                task.status = ExecutionStatus.FAILED
                task.error = str(e)

                try:
                    self.failed_count = int(getattr(self, "failed_count", 0)) + 1
                except Exception:
                    pass

                try:
                    if hasattr(self, "logger"):
                        self.logger.error(
                            f"❌ Task failed: {getattr(task, 'name', task_id)} - {e}"
                        )
                except Exception:
                    pass

                return None

            finally:
                # ------------------------
                # 🔥 CLEANUP (CRITICAL FIX)
                # ------------------------
                try:
                    self._running_tasks.pop(str(task.task_id), None)
                except Exception:
                    pass

                try:
                    if not getattr(task, "completed_at", None):
                        task.completed_at = datetime.utcnow()
                except Exception:
                    pass

    # ------------------------
    # PARALLEL EXECUTION 🔥
    # ------------------------
    async def execute_parallel(self, task_ids: List[str]) -> List[Any]:
        """Execute multiple tasks in parallel safely"""

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not task_ids or not isinstance(task_ids, list):
                return []

            # ------------------------
            # 🔥 INIT STORAGE (SAFE)
            # ------------------------
            tasks_store = getattr(self, "tasks", {}) or {}

            coroutines = []
            valid_task_ids = []

            # ------------------------
            # 🔥 BUILD TASK LIST
            # ------------------------
            for task_id in task_ids:
                try:
                    task_id = str(task_id)
                    task = tasks_store.get(task_id)

                    if not task:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Task not found: {task_id}")
                        continue

                    valid_task_ids.append(task_id)
                    coroutines.append(self.execute_task(task))

                except Exception:
                    continue

            if not coroutines:
                return []

            # ------------------------
            # 🔥 EXECUTE IN PARALLEL
            # ------------------------
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            # ------------------------
            # 🔥 POST-PROCESS RESULTS
            # ------------------------
            final_results = []

            for task_id, result in zip(valid_task_ids, results):
                try:
                    if isinstance(result, Exception):
                        task = tasks_store.get(task_id)

                        if task:
                            task.status = ExecutionStatus.FAILED
                            task.error = str(result)

                        if hasattr(self, "logger"):
                            self.logger.error(
                                f"❌ Parallel task failed: {task_id} - {result}"
                            )

                        final_results.append(None)

                    else:
                        final_results.append(result)

                except Exception:
                    final_results.append(None)

            return final_results

        except asyncio.CancelledError:
            raise

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Parallel execution error: {e}")
            except Exception:
                pass

            return []

    # ------------------------
    # SEQUENTIAL
    # ------------------------
    async def execute_sequential(self, task_ids: List[str]) -> List[Any]:
        """Execute tasks sequentially with full safety and consistency"""

        results = []

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not task_ids or not isinstance(task_ids, list):
                return results

            # ------------------------
            # 🔥 SAFE TASK STORE
            # ------------------------
            tasks_store = getattr(self, "tasks", {}) or {}

            for task_id in task_ids:
                try:
                    task_id = str(task_id)
                    task = tasks_store.get(task_id)

                    # ------------------------
                    # 🔥 TASK CHECK
                    # ------------------------
                    if not task:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Task not found: {task_id}")
                        results.append(None)
                        continue

                    try:
                        # ------------------------
                        # 🔥 EXECUTE ONE BY ONE
                        # ------------------------
                        result = await self.execute_task(task)
                        results.append(result)

                    except asyncio.CancelledError:
                        # ------------------------
                        # 🔥 CANCEL HANDLING
                        # ------------------------
                        task.status = ExecutionStatus.CANCELLED
                        results.append(None)

                        try:
                            if hasattr(self, "logger"):
                                self.logger.warning(
                                    f"⚠ Sequential task cancelled: {getattr(task, 'name', task_id)}"
                                )
                        except Exception:
                            pass

                        raise

                    except Exception as e:
                        # ------------------------
                        # 🔥 FAILURE HANDLING
                        # ------------------------
                        task.status = ExecutionStatus.FAILED
                        task.error = str(e)
                        results.append(None)

                        try:
                            if hasattr(self, "logger"):
                                self.logger.error(
                                    f"❌ Sequential task failed: {getattr(task, 'name', task_id)} - {e}"
                                )
                        except Exception:
                            pass

                except Exception:
                    results.append(None)

            return results

        except asyncio.CancelledError:
            raise

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Sequential execution error: {e}")
            except Exception:
                pass

            return results

    # ------------------------
    # CANCEL TASK 🔥
    # ------------------------
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task safely"""

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not task_id:
                return False

            task_id = str(task_id).strip()

            # ------------------------
            # 🔥 SAFE STORAGE ACCESS
            # ------------------------
            running_tasks = getattr(self, "_running_tasks", {}) or {}
            tasks_store = getattr(self, "tasks", {}) or {}

            future = running_tasks.get(task_id)
            task = tasks_store.get(task_id)

            if not future:
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"Cancel failed: task not running ({task_id})"
                        )
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 CANCEL TASK (CRITICAL FIX)
            # ------------------------
            cancelled = False
            try:
                if not future.done():
                    cancelled = future.cancel()
            except Exception:
                cancelled = False

            # ------------------------
            # 🔥 UPDATE TASK STATE
            # ------------------------
            if task:
                try:
                    task.status = ExecutionStatus.CANCELLED
                    task.error = "Cancelled by user"
                    task.completed_at = datetime.utcnow()
                except Exception:
                    pass

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            if cancelled:
                try:
                    self.cancelled_count = int(getattr(self, "cancelled_count", 0)) + 1
                except Exception:
                    pass

            # ------------------------
            # 🔥 CLEANUP (CRITICAL FIX)
            # ------------------------
            try:
                running_tasks.pop(task_id, None)
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"⚠ Task cancelled: {task_id}")
            except Exception:
                pass

            return cancelled

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Cancel task error: {task_id} - {e}")
            except Exception:
                pass

            return False

    # ------------------------
    # STATUS / RESULTS
    # ------------------------
    def get_task_status(self, task_id: str):
        """Safely retrieve task status with full metadata"""

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not task_id:
                return None

            task_id = str(task_id).strip()

            # ------------------------
            # 🔥 SAFE STORAGE ACCESS
            # ------------------------
            tasks_store = getattr(self, "tasks", {}) or {}
            running_tasks = getattr(self, "_running_tasks", {}) or {}

            task = tasks_store.get(task_id)

            if not task:
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Task not found: {task_id}")
                except Exception:
                    pass
                return None

            # ------------------------
            # 🔥 SAFE STATUS EXTRACTION
            # ------------------------
            status = getattr(task, "status", None)

            try:
                status_value = status.value if status else None
            except Exception:
                status_value = str(status)

            try:
                is_terminal = status.is_terminal() if status else False
            except Exception:
                is_terminal = False

            try:
                is_success = status.is_success() if status else False
            except Exception:
                is_success = False

            # ------------------------
            # 🔥 SAFE TIME FIELDS
            # ------------------------
            def safe_time(val):
                try:
                    return val.isoformat() if val else None
                except Exception:
                    return str(val) if val else None

            # ------------------------
            # 🔥 BUILD RESPONSE
            # ------------------------
            status_data = {
                "task_id": getattr(task, "task_id", task_id),
                "name": getattr(task, "name", None),
                "status": status_value,
                "is_terminal": is_terminal,
                "is_success": is_success,
                "created_at": safe_time(getattr(task, "created_at", None)),
                "started_at": safe_time(getattr(task, "started_at", None)),
                "completed_at": safe_time(getattr(task, "completed_at", None)),
                "error": getattr(task, "error", None),
                "running": task_id in running_tasks,
            }

            return status_data

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Get task status error: {task_id} - {e}")
            except Exception:
                pass

            return None

    def get_task_result(self, task_id: str):
        """Safely retrieve task result with full state handling"""

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not task_id:
                return None

            task_id = str(task_id).strip()

            # ------------------------
            # 🔥 SAFE STORAGE ACCESS
            # ------------------------
            tasks_store = getattr(self, "tasks", {}) or {}
            task = tasks_store.get(task_id)

            if not task:
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Task not found: {task_id}")
                except Exception:
                    pass
                return None

            status = getattr(task, "status", None)

            # ------------------------
            # 🔥 SAFE STATUS CHECKS
            # ------------------------
            if status == ExecutionStatus.PENDING:
                return {"status": "pending", "message": "Task not started yet"}

            if status == ExecutionStatus.RUNNING:
                return {"status": "running", "message": "Task is still running"}

            # ------------------------
            # 🔥 FAILED / CANCELLED
            # ------------------------
            if status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                return {
                    "status": getattr(status, "value", str(status)),
                    "error": getattr(task, "error", None),
                }

            # ------------------------
            # 🔥 SUCCESS
            # ------------------------
            def safe_time(val):
                try:
                    return val.isoformat() if val else None
                except Exception:
                    return str(val) if val else None

            return {
                "status": "completed",
                "result": getattr(task, "result", None),
                "completed_at": safe_time(getattr(task, "completed_at", None)),
            }

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Get task result error: {task_id} - {e}")
            except Exception:
                pass

            return {"status": "error", "message": "Failed to retrieve result"}

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self):
        """Return safe executor statistics with health evaluation"""

        try:
            # ------------------------
            # 🔥 SAFE STORAGE ACCESS (CRITICAL FIX)
            # ------------------------
            tasks_store = getattr(self, "tasks", {}) or {}
            running_store = getattr(self, "_running_tasks", {}) or {}

            total_tasks = len(tasks_store)
            running_tasks = len(running_store)

            # ------------------------
            # 🔥 STATUS DISTRIBUTION (SAFE)
            # ------------------------
            statuses = {}
            try:
                for status in ExecutionStatus:
                    statuses[status.value] = sum(
                        1
                        for t in tasks_store.values()
                        if getattr(t, "status", None) == status
                    )
            except Exception:
                statuses = {}

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            completed = int(getattr(self, "completed_count", 0) or 0)
            failed = int(getattr(self, "failed_count", 0) or 0)
            cancelled = int(getattr(self, "cancelled_count", 0) or 0)
            execution_count = int(getattr(self, "execution_count", 0) or 0)

            # ------------------------
            # 🔥 SUCCESS RATE (SAFE DIVISION)
            # ------------------------
            success_rate = (
                round(completed / execution_count, 3) if execution_count > 0 else 0.0
            )

            # ------------------------
            # 🔥 LOAD FACTOR (SAFE)
            # ------------------------
            try:
                max_workers = int(getattr(self, "max_workers", 1) or 1)
            except Exception:
                max_workers = 1

            load = round(running_tasks / max_workers, 2) if max_workers > 0 else 0.0

            # ------------------------
            # 🔥 HEALTH SCORE (ROBUST)
            # ------------------------
            health_score = 1.0

            if execution_count > 0:
                if failed > execution_count * 0.3:
                    health_score -= 0.3

                if cancelled > execution_count * 0.2:
                    health_score -= 0.2

            if load > 0.9:
                health_score -= 0.2

            health_score = max(0.0, round(health_score, 2))

            # ------------------------
            # 🔥 STATUS LABEL
            # ------------------------
            if health_score > 0.7:
                system_status = "healthy"
            elif health_score > 0.4:
                system_status = "degraded"
            else:
                system_status = "critical"

            # ------------------------
            # 🔥 FINAL OUTPUT
            # ------------------------
            return {
                "total_tasks": total_tasks,
                "running_tasks": running_tasks,
                "execution_count": execution_count,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "success_rate": success_rate,
                "load": load,
                "max_workers": max_workers,
                "statuses": statuses,
                "health_score": health_score,
                "status": system_status,
            }

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Executor stats error: {e}")
            except Exception:
                pass

            return {
                "status": "error",
                "total_tasks": 0,
                "running_tasks": 0,
                "health_score": 0.0,
            }

    # ------------------------
    # CLEANUP
    # ------------------------
    def clear_tasks(
        self, cancel_running: bool = True, reset_metrics: bool = False
    ) -> bool:
        """Clear all tasks safely with optional cancellation and metrics reset"""

        try:
            # ------------------------
            # 🔥 SAFE STORAGE ACCESS (CRITICAL FIX)
            # ------------------------
            tasks_store = getattr(self, "tasks", {}) or {}
            running_store = getattr(self, "_running_tasks", {}) or {}

            # ------------------------
            # 🔥 CANCEL RUNNING TASKS
            # ------------------------
            if cancel_running and running_store:
                for task_id, future in list(running_store.items()):
                    try:
                        if future and not future.done():
                            future.cancel()

                        task = tasks_store.get(task_id)
                        if task:
                            try:
                                task.status = ExecutionStatus.CANCELLED
                                task.error = "Cleared by system"
                                task.completed_at = datetime.utcnow()
                            except Exception:
                                pass

                    except Exception:
                        continue

                # safe clear
                try:
                    running_store.clear()
                except Exception:
                    pass

            # ------------------------
            # 🔥 CLEAR TASK STORAGE
            # ------------------------
            cleared_count = len(tasks_store)

            try:
                tasks_store.clear()
            except Exception:
                pass

            # ------------------------
            # 🔥 RESET METRICS
            # ------------------------
            if reset_metrics:
                try:
                    self.execution_count = 0
                    self.completed_count = 0
                    self.failed_count = 0
                    self.cancelled_count = 0
                except Exception:
                    pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"🧹 Tasks cleared ({cleared_count} removed)")
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Clear tasks error: {e}")
            except Exception:
                pass

            return False


# 🔥 MODULE LEVEL REGISTER (PRODUCTION FIX)


_executor_instance: Optional[Executor] = None


def register(name, func) -> Dict[str, Any]:
    """
    Module-level register function (safe + production-ready)
    Creates singleton Executor and registers function properly
    """

    global _executor_instance

    try:
        # ------------------------
        # 🔥 VALIDATION (CRITICAL FIX)
        # ------------------------
        if not name or not isinstance(name, str):
            raise ValueError("Invalid function name")

        name = name.strip()

        if not callable(func):
            raise ValueError(f"{name} is not callable")

        # ------------------------
        # 🔥 SINGLETON INIT (SAFE)
        # ------------------------
        if _executor_instance is None:
            _executor_instance = Executor()

        executor = _executor_instance

        # ------------------------
        # 🔥 SAFE REGISTRY INIT
        # ------------------------
        if not hasattr(executor, "_registry") or not isinstance(
            executor._registry, dict
        ):
            executor._registry = {}

        # ------------------------
        # 🔥 REGISTER FUNCTION
        # ------------------------
        executor._registry[name] = func

        # ------------------------
        # 🔥 OPTIONAL METADATA
        # ------------------------
        try:
            if not hasattr(executor, "_function_meta"):
                executor._function_meta = {}

            executor._function_meta[name] = {"registered": True}
        except Exception:
            pass

        # ------------------------
        # 🔥 SAFE LOGGING
        # ------------------------
        try:
            if hasattr(executor, "logger"):
                executor.logger.info(f"✅ Module function registered: {name}")
        except Exception:
            pass

        # ------------------------
        # 🔥 RETURN STATUS
        # ------------------------
        return {"status": "registered", "function": name, "executor_initialized": True}

    except Exception as e:
        try:
            if _executor_instance and hasattr(_executor_instance, "logger"):
                _executor_instance.logger.error(f"Executor module register failed: {e}")
        except Exception:
            pass

        return {
            "name": "Executor",
            "version": "unknown",
            "status": "error",
            "health": "critical",
            "message": "Module registration failed",
        }
