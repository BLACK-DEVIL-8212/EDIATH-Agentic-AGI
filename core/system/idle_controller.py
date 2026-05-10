"""
Advanced Idle Controller for EDIATH (Event-Driven + Safe)
with Chunking, Pipeline Processing & Fixed Auto-Healing Loop Protection
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from ..utils.logger import logger


class IdleState(Enum):
    """Idle state enumeration"""

    ACTIVE = "active"
    IDLE = "idle"
    DEEP_IDLE = "deep_idle"
    RECOVERING = "recovering"


@dataclass
class IdleCallback:
    """Idle callback with metadata"""

    callback: Callable
    name: str
    created_at: float = field(default_factory=time.time)
    execution_count: int = 0
    last_execution_time: float = 0.0
    total_execution_time: float = 0.0
    error_count: int = 0


@dataclass
class CallbackChunk:
    """Chunk of callbacks for batch processing"""

    chunk_id: int
    callbacks: List[IdleCallback]
    total_chunks: int
    created_at: float = field(default_factory=time.time)


class IdleController:
    """
    Production-grade Idle Manager with chunking and pipeline processing
    """

    def __init__(self, idle_threshold: int = 300, cooldown: int = 60):
        """
        Production-grade Idle Manager initialization with pipeline support
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            try:
                idle_threshold = int(idle_threshold)
            except Exception:
                idle_threshold = 300

            try:
                cooldown = int(cooldown)
            except Exception:
                cooldown = 60

            self.idle_threshold = max(1, idle_threshold)
            self.cooldown = max(0, cooldown)

            # Deep idle configuration
            self.deep_idle_threshold = max(
                1, self.idle_threshold * 2
            )  # 2x regular idle
            self.recovery_cooldown = max(1, self.cooldown)
            self.idle_state = IdleState.ACTIVE

            # ------------------------
            # 🔥 TIME BASE (CONSISTENT)
            # ------------------------
            now = datetime.utcnow()

            self.last_activity: Optional[datetime] = now
            self.last_triggered: Optional[datetime] = None
            self.last_deep_triggered: Optional[datetime] = None
            self.idle_start_time: Optional[datetime] = None
            self.deep_idle_start_time: Optional[datetime] = None

            # ------------------------
            # 🔥 STATE FLAGS
            # ------------------------
            self.is_idle = False
            self.is_deep_idle = False
            self._running = False
            self._paused = False
            self._shutdown = False

            # ------------------------
            # 🔥 TASK MANAGEMENT
            # ------------------------
            self._task: Optional[asyncio.Task] = None
            self._pipeline_tasks: List[asyncio.Task] = []
            self._shutdown_event = asyncio.Event()

            # ------------------------
            # 🔥 CALLBACK SYSTEM with chunking
            # ------------------------
            self._callbacks: List[IdleCallback] = []
            self._deep_idle_callbacks: List[IdleCallback] = []
            self._max_callbacks = 100
            self._callback_lock = asyncio.Lock()

            # Chunking configuration
            self.chunk_size = 10
            self.pipeline_workers = 4
            self.callback_queue: Optional[asyncio.Queue] = None
            self.deep_callback_queue: Optional[asyncio.Queue] = None

            # Auto-healing protection (FIXED)
            self._last_healing_attempt = 0
            self._healing_cooldown = 30  # seconds between healing attempts
            self._healing_consecutive = 0
            self._healing_lock = asyncio.Lock()

            # Cache cleanup protection
            self._last_cache_cleanup = 0
            self._cache_cleanup_interval = 60  # seconds
            self._memory_threshold = 0.9  # 90%

            # ------------------------
            # 🔥 LOCK (ASYNC SAFE)
            # ------------------------
            self._lock = asyncio.Lock()

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            self._trigger_count = 0
            self._deep_trigger_count = 0
            self._check_count = 0
            self._last_check_time: Optional[datetime] = None
            self._chunks_processed = 0

            self.created_at = now
            self.last_state_change: Optional[datetime] = None

            # ------------------------
            # 🔥 PERFORMANCE CONFIG
            # ------------------------
            self.check_interval = 1.0  # seconds
            self.max_idle_cycles = 100000  # safety cap
            self.callback_timeout = 30.0  # seconds per callback

            # ------------------------
            # 🔥 ERROR TRACKING
            # ------------------------
            self.error_count = 0
            self.last_error: Optional[str] = None

            self._max_callbacks = 100
            self._log_every_n = 5
            self._callback_count = 0
            self._deep_callback_count = 0

            # ------------------------
            # 🔥 LOGGING
            # ------------------------
            try:
                logger.info(
                    f"🕒 IdleManager initialized "
                    f"(threshold={self.idle_threshold}s, deep={self.deep_idle_threshold}s, "
                    f"cooldown={self.cooldown}s)"
                )
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ IdleManager init failed: {e}")
            except Exception:
                pass

            raise RuntimeError(f"IdleManager initialization failed: {e}")

    # ------------------------
    # START MONITORING
    # ------------------------
    async def start_monitoring(self, interval: int = 5) -> bool:
        """
        Start idle monitoring loop with chunking and pipeline support
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if interval <= 0:
                interval = 1

            if getattr(self, "_running", False):
                try:
                    logger.warning("⚠ Idle monitor already running")
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 STATE INIT
            # ------------------------
            self._running = True
            self._shutdown = False
            self._paused = False
            self._shutdown_event.clear()

            # Initialize queues
            self.callback_queue = asyncio.Queue(maxsize=200)
            self.deep_callback_queue = asyncio.Queue(maxsize=200)

            # Start pipeline workers
            for worker_id in range(self.pipeline_workers):
                task = asyncio.create_task(
                    self._callback_pipeline_worker(worker_id, "normal"),
                    name=f"idle_worker_{worker_id}",
                )
                self._pipeline_tasks.append(task)

                task = asyncio.create_task(
                    self._callback_pipeline_worker(worker_id, "deep"),
                    name=f"deep_idle_worker_{worker_id}",
                )
                self._pipeline_tasks.append(task)

            # Start metrics collector
            task = asyncio.create_task(
                self._collect_metrics(), name="idle_metrics_collector"
            )
            self._pipeline_tasks.append(task)

            self._task = asyncio.current_task()

            try:
                logger.info(
                    f"🕒 Idle monitor started with {self.pipeline_workers} workers"
                )
            except Exception:
                pass

            # ------------------------
            # 🔥 MAIN LOOP
            # ------------------------
            chunk_buffer = []
            last_chunk_time = time.time()
            chunk_interval = 1.0

            while self._running and not self._shutdown:

                try:
                    # ------------------------
                    # 🔥 PAUSE SUPPORT
                    # ------------------------
                    if getattr(self, "_paused", False):
                        await asyncio.sleep(interval)
                        continue

                    # ------------------------
                    # 🔥 CHECK IDLE STATE
                    # ------------------------
                    current_state = await self._check_idle_state()

                    # Buffer callbacks for chunking
                    if current_state in ["idle", "deep_idle"]:
                        callbacks_to_run = []

                        if current_state == "idle":
                            callbacks_to_run = self._callbacks.copy()
                        elif current_state == "deep_idle":
                            callbacks_to_run = self._deep_idle_callbacks.copy()

                        if callbacks_to_run:
                            # Add to buffer
                            chunk_buffer.extend(callbacks_to_run)

                            # Create chunk when buffer is full or time expired
                            current_time = time.time()
                            if len(chunk_buffer) >= self.chunk_size or (
                                chunk_buffer
                                and current_time - last_chunk_time >= chunk_interval
                            ):

                                await self._process_callback_chunk(
                                    chunk_buffer.copy(), current_state
                                )
                                chunk_buffer.clear()
                                last_chunk_time = current_time

                    # ------------------------
                    # 🔥 METRICS
                    # ------------------------
                    try:
                        self._check_count = getattr(self, "_check_count", 0) + 1
                        self._last_check_time = datetime.utcnow()
                    except Exception:
                        pass

                    await asyncio.sleep(interval)

                except asyncio.CancelledError:
                    try:
                        logger.info("🛑 Idle monitor cancelled")
                    except Exception:
                        pass
                    break

                except Exception as e:
                    try:
                        self.error_count = getattr(self, "error_count", 0) + 1
                        self.last_error = str(e)
                        logger.error(f"❌ Idle monitor error: {e}")
                    except Exception:
                        pass

                    await asyncio.sleep(interval)

            return True

        finally:
            # ------------------------
            # 🔥 CLEANUP
            # ------------------------
            self._running = False

            # Cancel pipeline workers
            for task in self._pipeline_tasks:
                if not task.done():
                    task.cancel()

            if self._pipeline_tasks:
                await asyncio.gather(*self._pipeline_tasks, return_exceptions=True)
                self._pipeline_tasks.clear()

            self._task = None

            try:
                logger.info("🛑 Idle monitor stopped")
            except Exception:
                pass

    # ------------------------
    # IDLE STATE CHECK (FIXED - NO SPAM)
    # ------------------------
    async def _check_idle_state(self) -> str:
        """
        Check idle state with rate limiting and circuit breaker
        """

        try:
            if not self.last_activity:
                return "active"

            now = datetime.utcnow()
            elapsed = (now - self.last_activity).total_seconds()

            if elapsed < 0:
                elapsed = 0

            was_idle = self.is_idle
            was_deep_idle = self.is_deep_idle

            # Determine new state
            is_now_idle = elapsed >= self.idle_threshold
            is_now_deep_idle = elapsed >= self.deep_idle_threshold

            self.is_idle = is_now_idle
            self.is_deep_idle = is_now_deep_idle

            # Update idle state enum
            if is_now_deep_idle:
                new_state = IdleState.DEEP_IDLE
            elif is_now_idle:
                new_state = IdleState.IDLE
            else:
                new_state = IdleState.ACTIVE

            # Track state changes with rate limiting
            if new_state != self.idle_state:
                self.idle_state = new_state
                self.last_state_change = now

                if is_now_deep_idle:
                    self.deep_idle_start_time = self.deep_idle_start_time or now

                    # Rate-limited logging
                    if self._can_log():
                        try:
                            logger.info(f"🟠 System became DEEP idle ({int(elapsed)}s)")
                        except Exception:
                            pass

                elif is_now_idle and not was_idle:
                    self.idle_start_time = now

                    if self._can_log():
                        try:
                            logger.info(f"🟡 System became idle ({int(elapsed)}s)")
                        except Exception:
                            pass

                elif not is_now_idle and was_idle:
                    self.idle_start_time = None
                    self.deep_idle_start_time = None

                    if self._can_log():
                        try:
                            logger.info("🟢 System active again")
                        except Exception:
                            pass

            # Return state string
            if is_now_deep_idle:
                return "deep_idle"
            elif is_now_idle:
                return "idle"
            else:
                return "active"

        except Exception as e:
            try:
                self.error_count += 1
                self.last_error = str(e)
                logger.warning(f"⚠ _check_idle_state failed: {e}")
            except Exception:
                pass
            return "active"

    def _can_log(self) -> bool:
        """Rate limit logging to prevent spam"""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = 0
            self._log_count = 0

        # Reset counter every minute
        if now - self._last_log_time > 60:
            self._last_log_time = now
            self._log_count = 0

        # Allow log if under limit
        if self._log_count < 5:
            self._log_count += 1
            return True
        return False

    # ------------------------
    # ACTIVITY TRACKING
    # ------------------------
    def register_activity(self) -> None:
        """
        Mark user/system activity safely (production-grade)
        """

        try:
            now = datetime.utcnow()

            # Reset idle timers
            self.last_activity = now
            self._last_check_time = now

            # Reset idle states if needed
            if self.is_idle or self.is_deep_idle:
                self.is_idle = False
                self.is_deep_idle = False
                self.idle_state = IdleState.ACTIVE
                self.last_state_change = now

                if self._can_log():
                    try:
                        logger.info("🟢 System active (activity detected)")
                    except Exception:
                        pass

            # Metrics
            try:
                self.activity_count = getattr(self, "activity_count", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                self.error_count += 1
                self.last_error = str(e)
                logger.warning(f"⚠ register_activity failed: {e}")
            except Exception:
                pass

    # ------------------------
    # CALLBACK MANAGEMENT
    # ------------------------
    def add_idle_listener(self, callback: Callable, name: str = None) -> bool:
        """
        Add idle listener safely with metadata
        """
        return self._add_callback(callback, self._callbacks, "idle", name)

    async def add_deep_idle_listener(
        self, callback: callable, name: str = None
    ) -> bool:
        """
        Register a deep idle listener callback with metadata
        """
        return self._add_callback(
            callback, self._deep_idle_callbacks, "deep_idle", name
        )

    def _add_callback(
        self,
        callback: Callable,
        callback_list: List,
        callback_type: str,
        name: str = None,
    ) -> bool:
        """
        Add callback to specified list with validation, chunking support, and error handling

        Args:
            callback: Callable function to register
            callback_list: List to store the callback in
            callback_type: Type of callback ('idle' or 'deep_idle')
            name: Optional custom name for the callback

        Returns:
            bool: True if successfully added, False otherwise
        """
        import asyncio
        import time

        try:
            # -------------------------
            # 🔥 VALIDATION
            # -------------------------
            if callback is None:
                raise ValueError("Callback cannot be None")

            if not callable(callback):
                raise TypeError(f"Callback must be callable, got {type(callback)}")

            if callback_type not in ["idle", "deep_idle", "normal"]:
                callback_type = "idle"  # Default to idle

            # -------------------------
            # 🔥 CHECK SHUTDOWN STATE
            # -------------------------
            if getattr(self, "_shutdown", False):
                try:
                    logger.debug(
                        f"⚠ Cannot add {callback_type} listener - system shutting down"
                    )
                except Exception:
                    pass
                return False

            # -------------------------
            # 🔒 THREAD/ASYNC SAFE LOCK (FIXED)
            # -------------------------
            # Use asyncio lock if in async context, otherwise use threading lock
            if not hasattr(self, "_callback_lock"):
                self._callback_lock = asyncio.Lock()

            # For sync context, we need to handle differently
            # Since this method might be called from sync code, we'll use a threading lock fallback
            if not hasattr(self, "_thread_lock"):
                import threading

                self._thread_lock = threading.RLock()

            # Use thread lock for sync safety (asyncio lock can't be used in sync context)
            with self._thread_lock:
                # -------------------------
                # 🔥 DEDUPLICATION CHECK
                # -------------------------
                for existing in callback_list:
                    # Check by callback reference
                    if existing.callback == callback:
                        try:
                            logger.debug(
                                f"⚠ {callback_type} listener already registered: {getattr(existing, 'name', 'unknown')}"
                            )
                        except Exception:
                            pass
                        return False

                    # Also check by name if provided
                    if name and hasattr(existing, "name") and existing.name == name:
                        try:
                            logger.debug(
                                f"⚠ {callback_type} listener with name '{name}' already exists"
                            )
                        except Exception:
                            pass
                        return False

                # -------------------------
                # 🔥 LIMIT CHECK
                # -------------------------
                max_callbacks = getattr(self, "_max_callbacks", 100)
                if len(callback_list) >= max_callbacks:
                    try:
                        logger.warning(
                            f"⚠ Max {callback_type} listeners reached ({max_callbacks})"
                        )
                    except Exception:
                        pass
                    return False

                # -------------------------
                # 🔥 CREATE CALLBACK WRAPPER WITH METADATA
                # -------------------------
                callback_name = name or getattr(
                    callback, "__name__", f"{callback_type}_{len(callback_list)}"
                )

                # Import IdleCallback if not already in scope
                try:
                    from dataclasses import dataclass, field

                    @dataclass
                    class CallbackWrapper:
                        callback: Callable
                        name: str
                        created_at: float = field(default_factory=time.time)
                        execution_count: int = 0
                        last_execution_time: float = 0.0
                        total_execution_time: float = 0.0
                        error_count: int = 0
                        callback_type: str = callback_name

                    callback_wrapper = CallbackWrapper(
                        callback=callback,
                        name=callback_name,
                        callback_type=callback_type,
                    )
                except Exception:
                    # Fallback to simple tuple if dataclass not available
                    callback_wrapper = {
                        "callback": callback,
                        "name": callback_name,
                        "created_at": time.time(),
                        "execution_count": 0,
                        "last_execution_time": 0.0,
                        "total_execution_time": 0.0,
                        "error_count": 0,
                        "callback_type": callback_type,
                    }

                # -------------------------
                # 🔥 ADD TO LIST
                # -------------------------
                callback_list.append(callback_wrapper)

                # -------------------------
                # 🔥 UPDATE METRICS
                # -------------------------
                try:
                    if callback_type == "idle":
                        self._callback_count = len(callback_list)
                    elif callback_type == "deep_idle":
                        self._deep_callback_count = len(callback_list)
                except Exception:
                    pass

                # -------------------------
                # 🔥 LOGGING (RATE LIMITED)
                # -------------------------
                try:
                    # Only log every 5th addition to prevent spam
                    log_every = getattr(self, "_log_every_n", 5)
                    current_count = len(callback_list)

                    if current_count % log_every == 0 or current_count <= 3:
                        logger.debug(
                            f"➕ {callback_type} listener added: {callback_name} "
                            f"(total={current_count}, max={max_callbacks})"
                        )
                    else:
                        logger.debug(
                            f"➕ {callback_type} listener added: {callback_name}"
                        )
                except Exception:
                    pass

                # -------------------------
                # 🔥 AUTO-HEALING CHECK (OPTIONAL)
                # -------------------------
                # If we're adding callbacks, the system is likely healthy
                try:
                    self._healing_consecutive = max(
                        0, getattr(self, "_healing_consecutive", 0) - 1
                    )
                except Exception:
                    pass

                return True

        except TypeError as e:
            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
                logger.error(f"❌ Type error adding {callback_type} listener: {e}")
            except Exception:
                pass
            return False

        except ValueError as e:
            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
                logger.error(f"❌ Value error adding {callback_type} listener: {e}")
            except Exception:
                pass
            return False

        except Exception as e:
            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
                logger.error(f"❌ Failed to add {callback_type} listener: {e}")
            except Exception:
                pass
            return False

    def remove_callback(
        self, callback: Callable = None, name: str = None, callback_type: str = None
    ) -> bool:
        """
        Remove a callback by reference or name

        Args:
            callback: The callback function to remove
            name: Name of the callback to remove
            callback_type: Type of callback ('idle', 'deep_idle', or None for both)

        Returns:
            bool: True if removed, False otherwise
        """
        import threading

        try:
            if callback is None and name is None:
                raise ValueError("Either callback or name must be provided")

            if not hasattr(self, "_thread_lock"):
                import threading

                self._thread_lock = threading.RLock()

            removed = False

            with self._thread_lock:
                # Determine which lists to search
                lists_to_search = []
                if callback_type is None or callback_type == "idle":
                    if hasattr(self, "_callbacks"):
                        lists_to_search.append(("idle", self._callbacks))
                if callback_type is None or callback_type == "deep_idle":
                    if hasattr(self, "_deep_idle_callbacks"):
                        lists_to_search.append(("deep_idle", self._deep_idle_callbacks))

                for list_type, callback_list in lists_to_search:
                    for i, cb_wrapper in enumerate(callback_list):
                        # Check by callback reference
                        if (
                            callback
                            and hasattr(cb_wrapper, "callback")
                            and cb_wrapper.callback == callback
                        ):
                            callback_list.pop(i)
                            removed = True
                            try:
                                logger.debug(
                                    f"➖ {list_type} listener removed by callback: {getattr(cb_wrapper, 'name', 'unknown')}"
                                )
                            except Exception:
                                pass
                            break

                        # Check by name
                        if (
                            name
                            and hasattr(cb_wrapper, "name")
                            and cb_wrapper.name == name
                        ):
                            callback_list.pop(i)
                            removed = True
                            try:
                                logger.debug(
                                    f"➖ {list_type} listener removed by name: {name}"
                                )
                            except Exception:
                                pass
                            break

                return removed

        except Exception as e:
            try:
                logger.error(f"❌ Failed to remove callback: {e}")
            except Exception:
                pass
            return False

    def get_callbacks(self, callback_type: str = None) -> List[Dict[str, Any]]:
        """
        Get list of registered callbacks with their metadata

        Args:
            callback_type: Type of callbacks to retrieve ('idle', 'deep_idle', or None for both)

        Returns:
            List of callback information dictionaries
        """
        result = []

        try:
            if callback_type is None or callback_type == "idle":
                for cb in getattr(self, "_callbacks", []):
                    result.append(
                        {
                            "name": getattr(cb, "name", "unknown"),
                            "type": "idle",
                            "created_at": getattr(cb, "created_at", 0),
                            "execution_count": getattr(cb, "execution_count", 0),
                            "error_count": getattr(cb, "error_count", 0),
                            "avg_execution_time": (
                                getattr(cb, "total_execution_time", 0)
                                / max(1, getattr(cb, "execution_count", 0))
                            ),
                        }
                    )

            if callback_type is None or callback_type == "deep_idle":
                for cb in getattr(self, "_deep_idle_callbacks", []):
                    result.append(
                        {
                            "name": getattr(cb, "name", "unknown"),
                            "type": "deep_idle",
                            "created_at": getattr(cb, "created_at", 0),
                            "execution_count": getattr(cb, "execution_count", 0),
                            "error_count": getattr(cb, "error_count", 0),
                            "avg_execution_time": (
                                getattr(cb, "total_execution_time", 0)
                                / max(1, getattr(cb, "execution_count", 0))
                            ),
                        }
                    )

            return result

        except Exception as e:
            try:
                logger.debug(f"Failed to get callbacks: {e}")
            except Exception:
                pass
            return []

    # ------------------------
    # CALLBACK PIPELINE WORKER
    # ------------------------
    async def _callback_pipeline_worker(self, worker_id: int, callback_type: str):
        """
        Worker for processing callbacks in parallel pipeline
        """
        logger.debug(f"🏭 {callback_type} callback worker {worker_id} started")

        while self._running and not self._shutdown:
            try:
                # Determine which queue to use
                queue = (
                    self.callback_queue
                    if callback_type == "normal"
                    else self.deep_callback_queue
                )
                if not queue:
                    await asyncio.sleep(0.1)
                    continue

                # Get chunk from queue
                chunk_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                if chunk_data is None:
                    break

                chunk = chunk_data.get("callbacks", [])
                chunk_id = chunk_data.get("chunk_id", 0)

                # Process each callback in chunk
                for callback_wrapper in chunk:
                    if not self._running:
                        break

                    await self._execute_callback(callback_wrapper, worker_id)

                queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if queue:
                    queue.task_done()
                await asyncio.sleep(0.1)

        logger.debug(f"🏭 {callback_type} callback worker {worker_id} stopped")

    async def _process_callback_chunk(
        self, callbacks: List[IdleCallback], callback_type: str
    ):
        """
        Process a chunk of callbacks through the pipeline
        """
        try:
            chunk = CallbackChunk(
                chunk_id=self._chunks_processed,
                callbacks=callbacks,
                total_chunks=self._chunks_processed + 1,
            )

            queue = (
                self.callback_queue
                if callback_type == "idle"
                else self.deep_callback_queue
            )

            if queue:
                await queue.put(
                    {
                        "chunk_id": chunk.chunk_id,
                        "callbacks": chunk.callbacks,
                        "type": callback_type,
                        "timestamp": time.time(),
                    }
                )

                self._chunks_processed += 1

        except Exception as e:
            logger.error(f"Failed to process callback chunk: {e}")

    async def _execute_callback(self, callback_wrapper: IdleCallback, worker_id: int):
        """
        Execute a single callback safely with timeout and metrics
        """
        start_time = time.time()

        try:
            callback = callback_wrapper.callback

            # Execute based on type
            if asyncio.iscoroutinefunction(callback):
                task = asyncio.create_task(callback())
                result = await asyncio.wait_for(task, timeout=self.callback_timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(callback), timeout=self.callback_timeout
                )

            # Update metrics
            execution_time = time.time() - start_time
            callback_wrapper.execution_count += 1
            callback_wrapper.last_execution_time = execution_time
            callback_wrapper.total_execution_time += execution_time

            return result

        except asyncio.TimeoutError:
            callback_wrapper.error_count += 1
            logger.warning(
                f"⏱ Callback timeout: {callback_wrapper.name} (worker {worker_id})"
            )
            return None

        except asyncio.CancelledError:
            logger.debug(f"Callback cancelled: {callback_wrapper.name}")
            raise

        except Exception as e:
            callback_wrapper.error_count += 1
            logger.error(f"❌ Callback failed {callback_wrapper.name}: {e}")
            return None

    def _can_trigger(self) -> bool:
        """
        Check if cooldown passed with rate limiting
        """
        try:
            if not self.last_triggered:
                return True

            now = datetime.utcnow()
            elapsed = (now - self.last_triggered).total_seconds()

            if elapsed < 0:
                elapsed = 0

            can_trigger = elapsed >= self.cooldown

            # Rate-limited debug logging
            if not can_trigger and self._can_log():
                try:
                    logger.debug(
                        f"⏳ Cooldown active ({round(elapsed,2)}s/{self.cooldown}s)"
                    )
                except Exception:
                    pass

            return can_trigger

        except Exception as e:
            try:
                self.error_count += 1
                self.last_error = str(e)
                logger.warning(f"⚠ _can_trigger failed: {e}")
            except Exception:
                pass
            return False

    # ------------------------
    # AUTO-HEALING (FIXED - NO SPAM)
    # ------------------------
    async def auto_heal(self):
        """
        Production-safe auto-healing with cooldown and circuit breaker
        """
        async with self._healing_lock:
            now = time.time()

            # Cooldown check - prevent spam
            if now - self._last_healing_attempt < self._healing_cooldown:
                return

            # Circuit breaker - if too many failures, stop trying
            if self._healing_consecutive >= 5:
                logger.warning(
                    "⚠ Auto-healing circuit breaker OPEN - too many attempts"
                )
                return

            self._last_healing_attempt = now

            try:
                # Check if healing is actually needed
                if not await self._needs_healing():
                    self._healing_consecutive = 0
                    return

                logger.info("🔄 Running auto-healing procedures")

                # Perform healing actions
                success = await self._perform_healing()

                if success:
                    self._healing_consecutive = 0
                    logger.info("✅ Auto-healing successful")
                else:
                    self._healing_consecutive += 1
                    logger.warning(
                        f"⚠ Auto-healing failed (attempt {self._healing_consecutive}/5)"
                    )

            except Exception as e:
                self._healing_consecutive += 1
                logger.error(f"❌ Auto-healing error: {e}")

    async def _needs_healing(self) -> bool:
        """Check if system actually needs healing"""
        # Check if error rate is too high
        total_ops = max(1, self._check_count)
        error_rate = self.error_count / total_ops

        # Check if callbacks are failing
        failed_callbacks = sum(1 for cb in self._callbacks if cb.error_count > 3)

        needs_heal = error_rate > 0.2 or failed_callbacks > 0

        if not needs_heal:
            logger.debug(f"System healthy (error rate: {error_rate:.1%})")

        return needs_heal

    async def _perform_healing(self) -> bool:
        """Perform actual healing operations"""
        try:
            # Reset error counter if partially healing
            if self.error_count > 10:
                self.error_count = max(0, self.error_count - 5)

            # Clear failed callbacks if too many errors
            expired_callbacks = []
            for cb in self._callbacks:
                if cb.error_count > 5:
                    expired_callbacks.append(cb)

            for cb in expired_callbacks:
                self._callbacks.remove(cb)
                logger.info(f"Removed failed callback: {cb.name}")

            return True

        except Exception as e:
            logger.error(f"Healing execution error: {e}")
            return False

    # ------------------------
    # RATE-LIMITED CACHE CLEANUP
    # ------------------------
    async def _rate_limited_cache_cleanup(self):
        """Clean cache with rate limiting to prevent spam"""
        now = time.time()

        # Only cleanup if enough time has passed
        if now - self._last_cache_cleanup < self._cache_cleanup_interval:
            return

        # Check if memory is actually high
        if not self._is_memory_high():
            return

        self._last_cache_cleanup = now

        # Perform cleanup
        try:
            # Clean old callback metrics
            for cb in self._callbacks:
                if time.time() - cb.created_at > 3600:  # Older than 1 hour
                    cb.total_execution_time = 0  # Reset metrics

            logger.info("🧹 Cache cleared (rate-limited)")
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")

    def _is_memory_high(self) -> bool:
        """Check if memory usage is actually high"""
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > self._memory_threshold
        except ImportError:
            return False  # Don't clean if can't check

    # ------------------------
    # METRICS COLLECTOR
    # ------------------------
    async def _collect_metrics(self):
        """Collect and log metrics periodically with rate limiting"""
        while self._running and not self._shutdown:
            await asyncio.sleep(30)  # Every 30 seconds

            if self._check_count > 0:
                logger.info(
                    f"📊 Idle Metrics: checks={self._check_count}, "
                    f"idle_triggers={self._trigger_count}, "
                    f"deep_triggers={self._deep_trigger_count}, "
                    f"callbacks={len(self._callbacks)}, "
                    f"deep_callbacks={len(self._deep_idle_callbacks)}, "
                    f"chunks={self._chunks_processed}, "
                    f"errors={self.error_count}"
                )

            # Run rate-limited operations
            await self._rate_limited_cache_cleanup()
            await self.auto_heal()

    # ------------------------
    # CONTROL METHODS
    # ------------------------
    def pause(self):
        """Pause idle monitoring"""
        self._paused = True
        logger.info("⏸️ Idle monitor paused")

    def resume(self):
        """Resume idle monitoring"""
        self._paused = False
        logger.info("▶️ Idle monitor resumed")

    def stop(self, force: bool = False) -> bool:
        """
        Stop idle monitoring safely with pipeline cleanup and timeout protection

        Args:
            force: If True, force immediate cleanup even if tasks are running

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        import asyncio

        try:
            # -------------------------
            # 🔥 VALIDATION - ALREADY STOPPED
            # -------------------------
            if not getattr(self, "_running", False):
                try:
                    logger.debug("⚠ Idle monitor already stopped")
                except Exception:
                    pass
                return True  # Already stopped, consider success

            # -------------------------
            # 🔥 UPDATE STATE FLAGS
            # -------------------------
            self._running = False
            self._shutdown = True
            self._shutdown_event.set()

            # -------------------------
            # 🔥 STOP MAIN TASK
            # -------------------------
            task = getattr(self, "_task", None)
            if task:
                try:
                    if not task.done():
                        task.cancel()
                        # Don't await here - let it cancel asynchronously
                except Exception as e:
                    try:
                        logger.debug(f"Main task cancellation error: {e}")
                    except Exception:
                        pass
                finally:
                    self._task = None

            # -------------------------
            # 🔥 STOP PIPELINE WORKERS WITH TIMEOUT
            # -------------------------
            if hasattr(self, "_pipeline_tasks") and self._pipeline_tasks:
                # Set shutdown event to signal workers
                if hasattr(self, "_shutdown_event"):
                    self._shutdown_event.set()

                # Cancel all pipeline tasks
                cancelled_count = 0
                for pipeline_task in self._pipeline_tasks:
                    if pipeline_task and not pipeline_task.done():
                        try:
                            pipeline_task.cancel()
                            cancelled_count += 1
                        except Exception as e:
                            try:
                                logger.debug(f"Pipeline task cancel error: {e}")
                            except Exception:
                                pass

                try:
                    logger.debug(f"Cancelled {cancelled_count} pipeline tasks")
                except Exception:
                    pass

                # Clear the reference (actual cleanup happens in start_monitoring finally block)
                self._pipeline_tasks.clear()

            # -------------------------
            # 🔥 CLEAR QUEUES (TO PREVENT MEMORY LEAKS)
            # -------------------------
            try:
                if hasattr(self, "callback_queue") and self.callback_queue:
                    # Clear the queue without waiting
                    while not self.callback_queue.empty():
                        try:
                            self.callback_queue.get_nowait()
                            self.callback_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                        except Exception:
                            break
            except Exception:
                pass

            try:
                if hasattr(self, "deep_callback_queue") and self.deep_callback_queue:
                    while not self.deep_callback_queue.empty():
                        try:
                            self.deep_callback_queue.get_nowait()
                            self.deep_callback_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                        except Exception:
                            break
            except Exception:
                pass

            # -------------------------
            # 🔥 FORCE CLEANUP (OPTIONAL)
            # -------------------------
            if force:
                try:
                    # Clear callback lists
                    if hasattr(self, "_callback_lock"):
                        # Try to acquire lock but don't block if not available
                        try:
                            # Use asyncio.create_task to run async operation
                            # but in stop() we can't await, so we'll do best effort
                            if hasattr(self, "_callbacks"):
                                self._callbacks.clear()
                            if hasattr(self, "_deep_idle_callbacks"):
                                self._deep_idle_callbacks.clear()
                        except Exception:
                            # Fallback to direct clear
                            if hasattr(self, "_callbacks"):
                                self._callbacks.clear()
                            if hasattr(self, "_deep_idle_callbacks"):
                                self._deep_idle_callbacks.clear()
                    else:
                        if hasattr(self, "_callbacks"):
                            self._callbacks.clear()
                        if hasattr(self, "_deep_idle_callbacks"):
                            self._deep_idle_callbacks.clear()

                    # Reset state flags
                    self.is_idle = False
                    self.is_deep_idle = False
                    if hasattr(self, "idle_state"):
                        self.idle_state = IdleState.ACTIVE

                    # Reset timers
                    self.last_activity = datetime.utcnow()
                    self.idle_start_time = None
                    self.deep_idle_start_time = None

                    try:
                        logger.info("🧹 Force cleanup completed")
                    except Exception:
                        pass

                except Exception as e:
                    try:
                        logger.debug(f"Force cleanup error: {e}")
                    except Exception:
                        pass

            # -------------------------
            # 🔥 UPDATE METRICS
            # -------------------------
            try:
                self.stopped_at = datetime.utcnow()
                self.stop_count = getattr(self, "stop_count", 0) + 1
                self.stop_duration = (self.stopped_at - self.created_at).total_seconds()
            except Exception:
                pass

            # -------------------------
            # 🔥 LOGGING
            # -------------------------
            try:
                logger.info(f"🛑 Idle monitor stopped (force={force})")
            except Exception:
                pass

            return True

        except Exception as e:
            # -------------------------
            # 🔥 ERROR HANDLING
            # -------------------------
            try:
                self.error_count = getattr(self, "error_count", 0) + 1
                self.last_error = str(e)
                logger.error(f"❌ Idle monitor stop failed: {e}")
            except Exception:
                pass

            # Even on error, try to mark as stopped
            try:
                self._running = False
                self._shutdown = True
            except Exception:
                pass

            return False

    async def stop_async(self, force: bool = False, timeout: float = 5.0) -> bool:
        """
        Async version of stop with proper awaiting of pipeline workers

        Args:
            force: If True, force immediate cleanup
            timeout: Timeout in seconds for waiting on pipeline workers

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            if not getattr(self, "_running", False):
                return True

            # Set shutdown flags
            self._running = False
            self._shutdown = True
            if hasattr(self, "_shutdown_event"):
                self._shutdown_event.set()

            # Cancel and wait for pipeline tasks
            if hasattr(self, "_pipeline_tasks") and self._pipeline_tasks:
                for task in self._pipeline_tasks:
                    if task and not task.done():
                        task.cancel()

                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._pipeline_tasks, return_exceptions=True),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Pipeline worker cleanup timeout after {timeout}s")
                except Exception as e:
                    logger.debug(f"Pipeline cleanup error: {e}")

                self._pipeline_tasks.clear()

            # Call sync stop for remaining cleanup
            return self.stop(force=force)

        except Exception as e:
            logger.error(f"Async stop failed: {e}")
            return False

    async def _cleanup(self):
        """Internal cleanup method for pipeline workers"""
        if hasattr(self, "_pipeline_tasks") and self._pipeline_tasks:
            for task in self._pipeline_tasks:
                if not task.done():
                    task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._pipeline_tasks, return_exceptions=True),
                    timeout=3.0,
                )
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            self._pipeline_tasks.clear()

    # ------------------------
    # STATUS METHODS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "is_running": self._running,
            "is_idle": self.is_idle,
            "is_deep_idle": self.is_deep_idle,
            "idle_state": self.idle_state.value,
            "idle_threshold": self.idle_threshold,
            "deep_idle_threshold": self.deep_idle_threshold,
            "cooldown": self.cooldown,
            "trigger_count": self._trigger_count,
            "deep_trigger_count": self._deep_trigger_count,
            "check_count": self._check_count,
            "chunks_processed": self._chunks_processed,
            "callbacks_count": len(self._callbacks),
            "deep_callbacks_count": len(self._deep_idle_callbacks),
            "error_count": self.error_count,
            "pipeline_workers": self.pipeline_workers,
            "last_error": self.last_error,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the idle controller"""
        return {
            "status": (
                "healthy" if self._running and self.error_count < 10 else "degraded"
            ),
            "running": self._running,
            "paused": self._paused,
            "stats": self.get_stats(),
            "healing_cooldown_remaining": max(
                0, self._healing_cooldown - (time.time() - self._last_healing_attempt)
            ),
        }


__all__ = ["IdleController", "IdleState"]
