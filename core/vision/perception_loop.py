# perception_loop.py

import asyncio
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import json
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DAG = "dag"
    PRIORITY = "priority"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ErrorHandlingStrategy(Enum):
    STOP = "stop"  # Stop pipeline on error
    SKIP = "skip"  # Skip failed step and continue
    RETRY = "retry"  # Retry failed step
    FALLBACK = "fallback"  # Use fallback processor


@dataclass
class PerceptionConfig:
    fps_target: float = 30.0
    max_queue_size: int = 10
    default_timeout: float = 2.0
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.SKIP
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_metrics: bool = True
    enable_monitoring: bool = True
    enable_checkpointing: bool = False
    adaptive_fps: bool = True
    min_fps: float = 10.0
    max_fps: float = 60.0
    backpressure_threshold: float = 0.8  # 80% queue full triggers backpressure


# ------------------------
# PERCEPTION RESULT
# ------------------------
class PerceptionResult:
    def __init__(self, success: bool, data: Any, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = time.time()
        self.processing_time = 0.0
        self.metadata: Dict = {}
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data if self.success else None,
            "error": self.error,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def success(cls, data: Any) -> 'PerceptionResult':
        return cls(True, data)
    
    @classmethod
    def failure(cls, error: str) -> 'PerceptionResult':
        return cls(False, None, error)


# ------------------------
# ADVANCED PERCEPTION STEP
# ------------------------
class PerceptionStep:
    def __init__(
        self,
        name: str,
        processor: Callable,
        timeout: float = 2.0,
        retries: int = 0,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        fallback: Optional[Callable] = None,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None
    ):
        self.name = name
        self.processor = processor
        self.timeout = timeout
        self.retries = retries
        self.dependencies = dependencies or []
        self.priority = priority
        self.fallback = fallback
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
        # Statistics
        self.execution_count = 0
        self.total_time = 0.0
        self.failures = 0
        self.retry_count = 0
        self.timeout_count = 0
        self.execution_times = deque(maxlen=100)
        self.status_history = deque(maxlen=100)
        
        # State
        self.last_result: Optional[PerceptionResult] = None
        self.last_execution_time = 0.0
        self.status = StepStatus.PENDING
        
        # Performance
        self.avg_time = 0.0
        self.p95_time = 0.0
        
        logger.info(f"➕ Step registered: {name} (priority={priority}, timeout={timeout}s)")
    
    async def execute(self, data: Any) -> PerceptionResult:
        """Execute perception step with advanced error handling"""
        start = time.time()
        self.status = StepStatus.RUNNING
        self.execution_count += 1
        
        try:
            # Apply preprocessor if available
            if self.preprocessor:
                data = await self._run_with_timeout(self.preprocessor, data, "preprocessor")
            
            # Execute with retries
            result = await self._execute_with_retries(data)
            
            # Apply postprocessor if available
            if result.success and self.postprocessor:
                result = await self._run_with_timeout(self.postprocessor, result.data, "postprocessor")
                if isinstance(result, PerceptionResult):
                    result = result
                else:
                    result = PerceptionResult.success(result)
            
            # Update statistics
            elapsed = time.time() - start
            self.execution_times.append(elapsed)
            self.total_time += elapsed
            self.last_execution_time = elapsed
            self.avg_time = np.mean(self.execution_times) if self.execution_times else 0
            
            if len(self.execution_times) >= 10:
                self.p95_time = np.percentile(self.execution_times, 95)
            
            self.status = StepStatus.COMPLETED
            self.status_history.append((time.time(), StepStatus.COMPLETED))
            self.last_result = result
            
            return result
            
        except asyncio.CancelledError:
            self.status = StepStatus.FAILED
            self.failures += 1
            raise
            
        except Exception as e:
            self.status = StepStatus.FAILED
            self.failures += 1
            elapsed = time.time() - start
            self.total_time += elapsed
            
            error_msg = f"Step '{self.name}' failed: {str(e)}"
            logger.error(error_msg)
            
            result = PerceptionResult.failure(error_msg)
            self.last_result = result
            return result
    
    async def _execute_with_retries(self, data: Any) -> PerceptionResult:
        """Execute with retry logic"""
        last_error = None
        
        for attempt in range(self.retries + 1):
            try:
                result = await self._run_with_timeout(self.processor, data, "processor")
                
                if isinstance(result, PerceptionResult):
                    if result.success:
                        if attempt > 0:
                            logger.info(f"Step '{self.name}' succeeded after {attempt} retries")
                        return result
                    else:
                        last_error = result.error
                        continue
                else:
                    if attempt > 0:
                        logger.info(f"Step '{self.name}' succeeded after {attempt} retries")
                    return PerceptionResult.success(result)
                    
            except asyncio.TimeoutError:
                self.timeout_count += 1
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(f"Step '{self.name}' timeout (attempt {attempt + 1}/{self.retries + 1})")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step '{self.name}' error (attempt {attempt + 1}/{self.retries + 1}): {e}")
            
            # Wait before retry
            if attempt < self.retries:
                self.retry_count += 1
                await asyncio.sleep(self.retry_delay)
        
        # All retries exhausted
        if self.fallback:
            logger.info(f"Using fallback for step '{self.name}'")
            try:
                fallback_result = await self._run_with_timeout(self.fallback, data, "fallback")
                if isinstance(fallback_result, PerceptionResult):
                    return fallback_result
                return PerceptionResult.success(fallback_result)
            except Exception as e:
                return PerceptionResult.failure(f"Fallback also failed: {str(e)}")
        
        return PerceptionResult.failure(f"All retries exhausted: {last_error}")
    
    async def _run_with_timeout(self, func: Callable, data: Any, context: str) -> Any:
        """Run function with timeout"""
        async def run():
            if asyncio.iscoroutinefunction(func):
                return await func(data)
            return await asyncio.to_thread(func, data)
        
        return await asyncio.wait_for(run(), timeout=self.timeout)
    
    def get_stats(self) -> Dict:
        """Get step statistics"""
        success_rate = (self.execution_count - self.failures) / max(1, self.execution_count)
        
        return {
            "name": self.name,
            "executions": self.execution_count,
            "failures": self.failures,
            "retries": self.retry_count,
            "timeouts": self.timeout_count,
            "success_rate": round(success_rate, 3),
            "avg_time_ms": round(self.avg_time * 1000, 2),
            "p95_time_ms": round(self.p95_time * 1000, 2),
            "last_execution_ms": round(self.last_execution_time * 1000, 2),
            "status": self.status.value,
            "priority": self.priority
        }


# ------------------------
# METRICS COLLECTOR
# ------------------------
class MetricsCollector:
    def __init__(self, enable_prometheus: bool = False):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_history = deque(maxlen=1000)
        
        if self.enable_prometheus:
            self.step_duration = Histogram('perception_step_duration_seconds', 'Step execution duration', ['step_name'])
            self.step_failures = Counter('perception_step_failures_total', 'Step failures', ['step_name'])
            self.frame_processing_time = Histogram('perception_frame_processing_seconds', 'Frame processing time')
            self.queue_size = Gauge('perception_queue_size', 'Frame queue size')
            self.fps_gauge = Gauge('perception_fps', 'Current FPS')
    
    def record_step(self, step_name: str, duration: float, success: bool):
        """Record step execution metrics"""
        self.metrics_history.append({
            "timestamp": time.time(),
            "type": "step",
            "step_name": step_name,
            "duration": duration,
            "success": success
        })
        
        if self.enable_prometheus:
            self.step_duration.labels(step_name=step_name).observe(duration)
            if not success:
                self.step_failures.labels(step_name=step_name).inc()
    
    def record_frame(self, duration: float, queue_size: int, fps: float):
        """Record frame processing metrics"""
        self.metrics_history.append({
            "timestamp": time.time(),
            "type": "frame",
            "duration": duration,
            "queue_size": queue_size,
            "fps": fps
        })
        
        if self.enable_prometheus:
            self.frame_processing_time.observe(duration)
            self.queue_size.set(queue_size)
            self.fps_gauge.set(fps)
    
    def get_recent_metrics(self, seconds: int = 60) -> List[Dict]:
        """Get recent metrics within time window"""
        cutoff = time.time() - seconds
        return [m for m in self.metrics_history if m["timestamp"] > cutoff]
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        if not self.metrics_history:
            return {}
        
        step_metrics = {}
        frame_durations = [m["duration"] for m in self.metrics_history if m["type"] == "frame"]
        
        for metric in self.metrics_history:
            if metric["type"] == "step":
                step_name = metric["step_name"]
                if step_name not in step_metrics:
                    step_metrics[step_name] = {"durations": [], "failures": 0}
                step_metrics[step_name]["durations"].append(metric["duration"])
                if not metric["success"]:
                    step_metrics[step_name]["failures"] += 1
        
        summary = {
            "total_metrics": len(self.metrics_history),
            "avg_frame_time_ms": np.mean(frame_durations) * 1000 if frame_durations else 0,
            "p95_frame_time_ms": np.percentile(frame_durations, 95) * 1000 if frame_durations else 0,
            "steps": {}
        }
        
        for step_name, data in step_metrics.items():
            summary["steps"][step_name] = {
                "avg_time_ms": np.mean(data["durations"]) * 1000,
                "p95_time_ms": np.percentile(data["durations"], 95) * 1000,
                "failure_rate": data["failures"] / len(data["durations"])
            }
        
        return summary


# ------------------------
# ADAPTIVE SCHEDULER
# ------------------------
class AdaptiveScheduler:
    def __init__(self, target_fps: float = 30.0, min_fps: float = 10.0, max_fps: float = 60.0):
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = target_fps
        self.processing_times = deque(maxlen=30)
        self.adjustment_factor = 1.0
        self.last_adjustment = time.time()
    
    def update(self, processing_time: float):
        """Update scheduler with latest processing time"""
        self.processing_times.append(processing_time)
        avg_time = np.mean(self.processing_times) if self.processing_times else processing_time
        
        # Calculate achievable FPS
        achievable_fps = 1.0 / avg_time if avg_time > 0 else self.target_fps
        
        # Adjust target FPS based on system load
        if achievable_fps < self.target_fps * 0.8:
            # System is overloaded, reduce target
            self.current_fps = max(self.min_fps, self.current_fps * 0.95)
        elif achievable_fps > self.target_fps * 1.2 and self.current_fps < self.target_fps:
            # System has headroom, increase target
            self.current_fps = min(self.target_fps, self.current_fps * 1.05)
        
        self.current_fps = np.clip(self.current_fps, self.min_fps, self.max_fps)
        
        return self.current_fps
    
    def get_sleep_time(self) -> float:
        """Calculate sleep time to maintain target FPS"""
        if not self.processing_times:
            return 1.0 / self.target_fps
        
        avg_processing = np.mean(self.processing_times)
        target_frame_time = 1.0 / self.current_fps
        
        return max(0, target_frame_time - avg_processing)


# ------------------------
# CHECKPOINT MANAGER
# ------------------------
class CheckpointManager:
    def __init__(self, save_dir: str = "./checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = {}
    
    async def save_checkpoint(self, step_name: str, data: Any, frame_id: int):
        """Save checkpoint for a step"""
        checkpoint_path = self.save_dir / f"{step_name}_frame_{frame_id}.json"
        
        checkpoint_data = {
            "step": step_name,
            "frame_id": frame_id,
            "data": data if isinstance(data, (dict, list)) else str(data),
            "timestamp": time.time()
        }
        
        def write():
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
        
        await asyncio.to_thread(write)
        self.checkpoints[f"{step_name}_{frame_id}"] = checkpoint_data
    
    async def load_checkpoint(self, step_name: str, frame_id: int) -> Optional[Any]:
        """Load checkpoint for a step"""
        checkpoint_path = self.save_dir / f"{step_name}_frame_{frame_id}.json"
        
        if not checkpoint_path.exists():
            return None
        
        def read():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        
        try:
            data = await asyncio.to_thread(read)
            return data.get("data")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


# ------------------------
# MAIN PERCEPTION LOOP
# ------------------------
class PerceptionLoop:
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        
        # Pipeline components
        self.steps: List[PerceptionStep] = []
        self.step_dict: Dict[str, PerceptionStep] = {}
        
        # State
        self.is_running = False
        self.frames_processed = 0
        self.skipped_frames = 0
        self.error_frames = 0
        
        # Timing
        self.frame_time = 1.0 / self.config.fps_target
        self.last_frame_time = 0
        self.start_time = 0
        self.processing_times = deque(maxlen=100)
        
        # Queue for backpressure
        self._frame_queue: Optional[asyncio.Queue] = None
        self.max_queue = self.config.max_queue_size
        
        # Advanced components
        self.metrics = MetricsCollector()
        self.scheduler = AdaptiveScheduler(
            self.config.fps_target,
            self.config.min_fps,
            self.config.max_fps
        ) if self.config.adaptive_fps else None
        self.checkpoint_manager = CheckpointManager() if self.config.enable_checkpointing else None
        
        # Performance monitoring
        self.system_monitor = SystemMonitor() if self.config.enable_monitoring else None
        
        # Callbacks
        self.on_frame_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        logger.info(f"🚀 Perception Loop Initialized - Mode: {self.config.execution_mode.value}, FPS Target: {self.config.fps_target}")
    
    # ------------------------
    # STEP MANAGEMENT
    # ------------------------
    def add_step(
        self,
        name: str,
        processor: Callable,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        fallback: Optional[Callable] = None,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None
    ) -> 'PerceptionLoop':
        """Add perception step with advanced configuration"""
        
        if name in self.step_dict:
            logger.warning(f"Step '{name}' already exists, overwriting")
        
        step = PerceptionStep(
            name=name,
            processor=processor,
            timeout=timeout or self.config.default_timeout,
            retries=retries or self.config.max_retries,
            dependencies=dependencies,
            priority=priority,
            fallback=fallback,
            preprocessor=preprocessor,
            postprocessor=postprocessor
        )
        
        self.steps.append(step)
        self.step_dict[name] = step
        
        # Sort by priority (higher priority first)
        self.steps.sort(key=lambda s: s.priority, reverse=True)
        
        logger.info(f"➕ Step added: {name} (priority={priority})")
        return self
    
    def remove_step(self, name: str) -> bool:
        """Remove a step from pipeline"""
        if name in self.step_dict:
            self.steps.remove(self.step_dict[name])
            del self.step_dict[name]
            logger.info(f"➖ Step removed: {name}")
            return True
        return False
    
    # ------------------------
    # CALLBACK MANAGEMENT
    # ------------------------
    def on_frame(self, callback: Callable) -> 'PerceptionLoop':
        """Register callback for each processed frame"""
        self.on_frame_callbacks.append(callback)
        return self
    
    def on_error(self, callback: Callable) -> 'PerceptionLoop':
        """Register callback for errors"""
        self.on_error_callbacks.append(callback)
        return self
    
    # ------------------------
    # PIPELINE EXECUTION
    # ------------------------
    async def _execute_sequential(self, frame: np.ndarray, results: Dict) -> Dict:
        """Execute steps sequentially"""
        current_data = frame
        
        for step in self.steps:
            step_start = time.time()
            result = await step.execute(current_data)
            step_duration = time.time() - step_start
            
            # Record metrics
            self.metrics.record_step(step.name, step_duration, result.success)
            
            if not result.success:
                if self.config.error_handling == ErrorHandlingStrategy.STOP:
                    raise Exception(f"Pipeline stopped at step '{step.name}': {result.error}")
                elif self.config.error_handling == ErrorHandlingStrategy.SKIP:
                    results[step.name] = {"error": result.error, "skipped": True}
                    continue
            
            results[step.name] = result.data
            current_data = result.data
            
            # Update results for next steps
            if isinstance(result.data, dict):
                results.update(result.data)
            elif isinstance(result.data, (list, tuple, np.ndarray)):
                results[f"step_{step.name}_output"] = result.data
        
        return results
    
    async def _execute_parallel(self, frame: np.ndarray, results: Dict) -> Dict:
        """Execute independent steps in parallel"""
        # Build dependency graph
        tasks = []
        for step in self.steps:
            if not step.dependencies:
                tasks.append(self._execute_with_deps(step, frame, results))
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks)
        
        return results
    
    async def _execute_with_deps(self, step: PerceptionStep, frame: np.ndarray, results: Dict):
        """Execute step with dependency resolution"""
        # Wait for dependencies
        for dep_name in step.dependencies:
            while dep_name not in results:
                await asyncio.sleep(0.01)
        
        # Prepare input data (merge from dependencies)
        input_data = {}
        for dep_name in step.dependencies:
            input_data[dep_name] = results.get(dep_name)
        
        step_start = time.time()
        result = await step.execute(input_data if len(step.dependencies) > 1 else frame)
        step_duration = time.time() - step_start
        
        self.metrics.record_step(step.name, step_duration, result.success)
        
        if result.success:
            results[step.name] = result.data
    
    async def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process single frame through pipeline"""
        if not self.is_running:
            return None
        
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        
        start_time = time.time()
        results = {
            "frame_id": self.frames_processed,
            "timestamp": start_time,
            "original_frame": frame
        }
        
        try:
            # Execute pipeline based on mode
            if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential(frame, results)
            elif self.config.execution_mode == ExecutionMode.PARALLEL:
                results = await self._execute_parallel(frame, results)
            elif self.config.execution_mode == ExecutionMode.PRIORITY:
                results = await self._execute_sequential(frame, results)  # Priority already handled by sorting
            else:
                results = await self._execute_sequential(frame, results)
            
            # Add performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frames_processed += 1
            
            results["_perception_metrics"] = {
                "processing_time": round(processing_time, 4),
                "fps": 1.0 / processing_time if processing_time > 0 else 0,
                "timestamp": time.time(),
                "frames_processed": self.frames_processed,
                "skipped_frames": self.skipped_frames
            }
            
            # Record frame metrics
            queue_size = self._frame_queue.qsize() if self._frame_queue else 0
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            self.metrics.record_frame(processing_time, queue_size, current_fps)
            
            # Adaptive scheduling
            if self.scheduler:
                new_fps = self.scheduler.update(processing_time)
                if new_fps != self.config.fps_target:
                    self.config.fps_target = new_fps
                    self.frame_time = 1.0 / new_fps
                    logger.debug(f"Adjusted FPS target: {new_fps:.1f}")
            
            # Execute callbacks
            for callback in self.on_frame_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(results)
                    else:
                        callback(results)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            return results
            
        except Exception as e:
            self.error_frames += 1
            error_msg = f"Frame processing error: {str(e)}"
            logger.error(error_msg)
            
            # Execute error callbacks
            for callback in self.on_error_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(error_msg, frame)
                    else:
                        callback(error_msg, frame)
                except Exception as cb_err:
                    logger.error(f"Error callback error: {cb_err}")
            
            return None
    
    # ------------------------
    # MAIN LOOP
    # ------------------------
    async def run(
        self,
        frame_source: Callable,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run perception loop with advanced control"""
        
        results = []
        self.start_time = time.time()
        self.is_running = True
        
        # Initialize frame queue
        self._frame_queue = asyncio.Queue(maxsize=self.max_queue)
        
        # Start producer and consumer tasks
        producer_task = asyncio.create_task(self._frame_producer(frame_source))
        consumer_task = asyncio.create_task(self._frame_consumer(results, max_frames))
        
        try:
            # Run until duration or max_frames reached
            if duration:
                await asyncio.sleep(duration)
                self.is_running = False
            else:
                await consumer_task
                
        except asyncio.CancelledError:
            logger.info("Perception loop cancelled")
        except Exception as e:
            logger.error(f"Perception loop error: {e}")
        finally:
            self.is_running = False
            producer_task.cancel()
            consumer_task.cancel()
            
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
            
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        
        return results
    
    async def _frame_producer(self, frame_source: Callable):
        """Produce frames from source"""
        try:
            if asyncio.iscoroutinefunction(frame_source):
                async for frame in frame_source():
                    if not self.is_running:
                        break
                    
                    # Backpressure control
                    if self._frame_queue.full():
                        self.skipped_frames += 1
                        logger.debug(f"Frame queue full, skipping frame (total skipped: {self.skipped_frames})")
                        continue
                    
                    await self._frame_queue.put(frame)
            else:
                # Synchronous generator
                for frame in frame_source():
                    if not self.is_running:
                        break
                    
                    if self._frame_queue.full():
                        self.skipped_frames += 1
                        continue
                    
                    await self._frame_queue.put(frame)
                    
        except Exception as e:
            logger.error(f"Frame producer error: {e}")
    
    async def _frame_consumer(self, results: List, max_frames: Optional[int]):
        """Consume and process frames"""
        try:
            while self.is_running:
                # Check max frames limit
                if max_frames and len(results) >= max_frames:
                    break
                
                # Get frame from queue with timeout
                try:
                    frame = await asyncio.wait_for(
                        self._frame_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if not self.is_running:
                        break
                    continue
                
                # Process frame
                result = await self.process_frame(frame)
                
                if result:
                    results.append(result)
                
                # FPS control
                if self.scheduler:
                    sleep_time = self.scheduler.get_sleep_time()
                else:
                    elapsed = time.time() - self.last_frame_time
                    sleep_time = max(0, self.frame_time - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                self.last_frame_time = time.time()
                
        except Exception as e:
            logger.error(f"Frame consumer error: {e}")
    
    # ------------------------
    # STATISTICS & MONITORING
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        
        total_frames = self.frames_processed + self.skipped_frames + self.error_frames
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0
        
        stats = {
            "status": "running" if self.is_running else "stopped",
            "runtime_seconds": time.time() - self.start_time if self.start_time else 0,
            "frames_processed": self.frames_processed,
            "skipped_frames": self.skipped_frames,
            "error_frames": self.error_frames,
            "total_frames": total_frames,
            "drop_rate": (self.skipped_frames + self.error_frames) / max(1, total_frames),
            "avg_processing_time_ms": avg_processing * 1000,
            "current_fps": 1.0 / avg_processing if avg_processing > 0 else 0,
            "target_fps": self.config.fps_target,
            "queue_size": self._frame_queue.qsize() if self._frame_queue else 0,
            "steps": [step.get_stats() for step in self.steps],
            "metrics_summary": self.metrics.get_summary(),
            "config": {
                "execution_mode": self.config.execution_mode.value,
                "error_handling": self.config.error_handling.value,
                "adaptive_fps": self.config.adaptive_fps
            }
        }
        
        # Add system monitor stats
        if self.system_monitor:
            stats["system"] = self.system_monitor.get_stats()
        
        return stats
    
    def get_step_stats(self, step_name: str) -> Optional[Dict]:
        """Get statistics for specific step"""
        step = self.step_dict.get(step_name)
        return step.get_stats() if step else None
    
    def reset_stats(self):
        """Reset all statistics"""
        self.frames_processed = 0
        self.skipped_frames = 0
        self.error_frames = 0
        self.processing_times.clear()
        
        for step in self.steps:
            step.execution_count = 0
            step.total_time = 0
            step.failures = 0
            step.retry_count = 0
            step.timeout_count = 0
            step.execution_times.clear()
        
        logger.info("📊 Statistics reset")
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "metrics_history": self.metrics.get_recent_metrics(seconds=300)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📁 Metrics exported to {filepath}")


# ------------------------
# SYSTEM MONITOR
# ------------------------
class SystemMonitor:
    def __init__(self):
        self.cpu_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        
    def get_stats(self) -> Dict:
        """Get system resource usage"""
        stats = {}
        
        if PSUTIL_AVAILABLE:
            stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            stats["memory_percent"] = psutil.virtual_memory().percent
            stats["disk_usage"] = psutil.disk_usage('/').percent
            
            self.cpu_history.append(stats["cpu_percent"])
            self.memory_history.append(stats["memory_percent"])
            
            stats["cpu_avg"] = np.mean(self.cpu_history) if self.cpu_history else 0
            stats["memory_avg"] = np.mean(self.memory_history) if self.memory_history else 0
        else:
            stats["error"] = "psutil not available"
        
        return stats


# ------------------------
# USAGE EXAMPLE
# ------------------------
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure perception loop
    config = PerceptionConfig(
        fps_target=30.0,
        execution_mode=ExecutionMode.SEQUENTIAL,
        error_handling=ErrorHandlingStrategy.SKIP,
        adaptive_fps=True,
        enable_metrics=True,
        enable_monitoring=True
    )
    
    # Create perception loop
    loop = PerceptionLoop(config)
    
    # Add perception steps
    def preprocess(frame):
        """Preprocessing step"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 else frame
    
    def detect_objects(frame):
        """Object detection step (mock)"""
        return {"objects": ["person", "car"], "count": 2}
    
    def classify_activity(data):
        """Activity classification step"""
        return {"activity": "walking", "confidence": 0.85}
    
    # Register steps
    loop.add_step("preprocess", preprocess, priority=10)
    loop.add_step("detect", detect_objects, dependencies=["preprocess"])
    loop.add_step("classify", classify_activity, dependencies=["detect"])
    
    # Add callbacks
    def on_frame_callback(results):
        logger.info(f"Frame processed: {results.get('_perception_metrics', {}).get('fps', 0):.1f} FPS")
    
    loop.on_frame(on_frame_callback)
    
    # Create frame source
    async def frame_source():
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    yield frame
                await asyncio.sleep(0.01)
        finally:
            cap.release()
    
    # Run perception loop
    print("🚀 Starting perception loop (press Ctrl+C to stop)")
    try:
        results = await loop.run(frame_source, duration=10.0)
        
        # Print statistics
        print("\n" + "="*50)
        print("FINAL STATISTICS:")
        print(json.dumps(loop.get_stats(), indent=2))
        
        # Export metrics
        loop.export_metrics("perception_metrics.json")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping perception loop...")
    finally:
        await loop.stop()


if __name__ == "__main__":
    import cv2
    asyncio.run(main())