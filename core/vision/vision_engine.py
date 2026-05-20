"""
Vision Engine - FULLY INTEGRATED AI VISION SYSTEM
Advanced features: distributed processing, GPU acceleration, adaptive resolution
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from core.utils.logger import logger as core_logger
except ImportError:
    core_logger = logger

from .shared_memory import SharedVisionMemory, VisionFrame, DataSource, DataPriority


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class VisionMode(Enum):
    LOW_POWER = "low_power"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    PRECISION = "precision"


class Resolution(Enum):
    LOW = (320, 240)
    MEDIUM = (640, 480)
    HIGH = (1280, 720)
    ULTRA = (1920, 1080)


class ProcessingStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINED = "pipelined"
    ADAPTIVE = "adaptive"


@dataclass
class VisionConfig:
    mode: VisionMode = VisionMode.BALANCED
    resolution: Resolution = Resolution.MEDIUM
    fps_target: int = 30
    processing_strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL
    
    # CNN Pipeline
    enable_cnn: bool = True
    enable_pipeline: bool = True
    chunk_size: int = 10
    pipeline_workers: int = 4
    cnn_batch_size: int = 32
    cnn_input_size: int = 224
    
    # Feature flags
    enable_detection: bool = True
    enable_caption: bool = True
    enable_activity: bool = True
    enable_face: bool = True
    enable_decision: bool = False
    
    # Performance
    enable_gpu: bool = True
    enable_adaptive_fps: bool = True
    enable_frame_skip: bool = True
    max_frame_skip: int = 5
    processing_timeout: float = 2.0
    queue_size: int = 100
    
    # Monitoring
    enable_metrics: bool = True
    enable_health_check: bool = True
    metrics_interval: int = 10
    
    # Storage
    enable_persistence: bool = False
    storage_path: str = "./vision_data"
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "resolution": self.resolution.value,
            "fps_target": self.fps_target,
            "processing_strategy": self.processing_strategy.value,
            "enable_cnn": self.enable_cnn,
            "enable_pipeline": self.enable_pipeline,
            "chunk_size": self.chunk_size,
            "pipeline_workers": self.pipeline_workers,
            "enable_gpu": self.enable_gpu
        }


# ------------------------
# PERFORMANCE MONITOR
# ------------------------
class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.gpu_utilization = deque(maxlen=window_size)
        self.cpu_utilization = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.frame_drops = 0
        self.start_time = time.time()
        
    def record_frame(self, processing_time: float, frame_interval: float):
        """Record frame processing metrics"""
        self.processing_times.append(processing_time)
        self.frame_times.append(frame_interval)
        
    def record_system_metrics(self, gpu_util: float = 0, cpu_util: float = 0, memory: float = 0):
        """Record system resource metrics"""
        self.gpu_utilization.append(gpu_util)
        self.cpu_utilization.append(cpu_util)
        self.memory_usage.append(memory)
        
    def record_frame_drop(self):
        """Record a frame drop"""
        self.frame_drops += 1
        
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if not self.processing_times:
            return 0
        avg_time = np.mean(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0
    
    def get_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        fps = self.get_fps()
        total_frames = len(self.processing_times)
        drop_rate = self.frame_drops / max(1, total_frames + self.frame_drops)
        
        return {
            "fps": round(fps, 2),
            "avg_processing_ms": round(np.mean(self.processing_times) * 1000, 2) if self.processing_times else 0,
            "p95_processing_ms": round(np.percentile(self.processing_times, 95) * 1000, 2) if self.processing_times else 0,
            "p99_processing_ms": round(np.percentile(self.processing_times, 99) * 1000, 2) if self.processing_times else 0,
            "frame_drops": self.frame_drops,
            "drop_rate": round(drop_rate, 3),
            "total_frames": total_frames,
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "system": {
                "avg_gpu": round(np.mean(self.gpu_utilization), 2) if self.gpu_utilization else 0,
                "avg_cpu": round(np.mean(self.cpu_utilization), 2) if self.cpu_utilization else 0,
                "avg_memory": round(np.mean(self.memory_usage), 2) if self.memory_usage else 0
            }
        }


# ------------------------
# ADAPTIVE RESOLUTION MANAGER
# ------------------------
class AdaptiveResolutionManager:
    def __init__(self, initial_resolution: Resolution, target_fps: int):
        self.current_resolution = initial_resolution
        self.target_fps = target_fps
        self.performance_history = deque(maxlen=30)
        self.resolution_levels = list(Resolution)
        self.current_level = self.resolution_levels.index(initial_resolution)
        
    def update(self, current_fps: float, processing_time: float):
        """Update resolution based on performance"""
        self.performance_history.append((current_fps, processing_time))
        
        if len(self.performance_history) < 10:
            return self.current_resolution
        
        avg_fps = np.mean([p[0] for p in self.performance_history])
        
        if avg_fps < self.target_fps * 0.8 and self.current_level > 0:
            # Lower resolution
            self.current_level -= 1
            self.current_resolution = self.resolution_levels[self.current_level]
            logger.info(f"⬇️ Lowering resolution to {self.current_resolution.value}")
            
        elif avg_fps > self.target_fps * 1.2 and self.current_level < len(self.resolution_levels) - 1:
            # Increase resolution if we have headroom
            self.current_level += 1
            self.current_resolution = self.resolution_levels[self.current_level]
            logger.info(f"⬆️ Increasing resolution to {self.current_resolution.value}")
        
        return self.current_resolution
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution dimensions"""
        return self.current_resolution.value


# ------------------------
# FRAME BUFFER MANAGER
# ------------------------
class FrameBufferManager:
    def __init__(self, max_size: int = 100, chunk_size: int = 10):
        self.buffer = deque(maxlen=max_size)
        self.chunk_size = chunk_size
        self._lock = asyncio.Lock()
        
    async def add_frame(self, frame: np.ndarray, metadata: Dict = None):
        """Add frame to buffer"""
        async with self._lock:
            self.buffer.append({
                "frame": frame,
                "timestamp": time.time(),
                "metadata": metadata or {}
            })
    
    async def get_chunk(self) -> Optional[List[np.ndarray]]:
        """Get a chunk of frames for batch processing"""
        async with self._lock:
            if len(self.buffer) >= self.chunk_size:
                chunk = []
                for _ in range(self.chunk_size):
                    if self.buffer:
                        chunk.append(self.buffer.popleft())
                return chunk
            return None
    
    async def get_latest(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        async with self._lock:
            if self.buffer:
                return self.buffer[-1]["frame"]
            return None
    
    async def clear(self):
        """Clear the buffer"""
        async with self._lock:
            self.buffer.clear()
    
    async def size(self) -> int:
        """Get buffer size"""
        async with self._lock:
            return len(self.buffer)


# ------------------------
# CNN ENGINE INTERFACE
# ------------------------
class CNNEngineInterface:
    """Wrapper for CNN Engine with async support"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.is_running = False
        self.device = "cuda" if config.get("enable_gpu", True) and TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
    async def start(self):
        """Start CNN engine"""
        try:
            # Import here to avoid circular imports
            from core.brain.cnn_engine import CNNEngine
            self.model = CNNEngine(self.config)
            await self.model.start()
            self.is_running = True
            logger.info(f"✅ CNN Engine started on {self.device}")
        except Exception as e:
            logger.error(f"CNN Engine start failed: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop CNN engine"""
        if self.model:
            await self.model.stop()
        self.is_running = False
    
    async def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame"""
        if not self.is_running or self.model is None:
            return {}
        
        try:
            result = await self.model.process_frame(frame)
            return result if result else {}
        except Exception as e:
            logger.debug(f"CNN frame processing error: {e}")
            return {}
    
    async def process_frames_chunked(self, frames: List[np.ndarray]) -> List[Dict]:
        """Process chunk of frames"""
        if not self.is_running or self.model is None:
            return []
        
        try:
            results = await self.model.process_batch(frames)
            return results if results else []
        except Exception as e:
            logger.debug(f"CNN chunk processing error: {e}")
            return []
    
    async def health_check(self) -> Dict:
        """Health check"""
        return {
            "running": self.is_running,
            "device": self.device,
            "model_loaded": self.model is not None
        }
    
    def get_metrics(self) -> Dict:
        """Get engine metrics"""
        if self.model and hasattr(self.model, 'get_metrics'):
            return self.model.get_metrics()
        return {}


# ------------------------
# VISION ENGINE (ENHANCED)
# ------------------------
class VisionEngine:
    """Central vision system with advanced features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Configuration
        self.config = config or {}
        self.vision_config = self._build_vision_config()
        
        # State
        self.is_active = False
        self.is_initialized = False
        self.start_time = None
        
        # Components (lazy loaded)
        self.webcam = None
        self.object_detector = None
        self.captioner = None
        self.perception_loop = None
        self.decision_engine = None
        self.activity_engine = None
        self.face_engine = None
        
        # CNN Engine
        self.cnn_engine = None
        self.vision_memory = None
        
        # Frame data
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Dict] = []
        self.current_caption: Optional[str] = None
        self.current_activity: Optional[Dict] = None
        self.current_faces: List[Dict] = []
        self.cnn_results: List[Dict] = []
        
        # Performance components
        self.performance_monitor = PerformanceMonitor()
        self.resolution_manager = AdaptiveResolutionManager(
            self.vision_config.resolution,
            self.vision_config.fps_target
        )
        self.frame_buffer = FrameBufferManager(
            max_size=self.vision_config.queue_size,
            chunk_size=self.vision_config.chunk_size
        )
        
        # Feature flags
        self.vision_enabled = {
            "detection": self.vision_config.enable_detection,
            "caption": self.vision_config.enable_caption,
            "activity": self.vision_config.enable_activity,
            "face": self.vision_config.enable_face,
            "decision": self.vision_config.enable_decision,
            "cnn": self.vision_config.enable_cnn,
        }
        
        # Pipeline workers
        self.pipeline_tasks: List[asyncio.Task] = []
        self.frame_queue: Optional[asyncio.Queue] = None
        
        # Control
        self._lock = asyncio.Lock()
        self._error_count = 0
        self._shutdown_event = asyncio.Event()
        self._main_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.frames_processed = 0
        self.chunks_processed = 0
        self.fps = 0
        self.last_frame_time = 0
        
        logger.info(f"🚀 VisionEngine initialized - Mode: {self.vision_config.mode.value}")
    
    def _build_vision_config(self) -> VisionConfig:
        """Build vision configuration from dict"""
        mode_str = self.config.get("mode", "balanced")
        mode_map = {
            "low_power": VisionMode.LOW_POWER,
            "balanced": VisionMode.BALANCED,
            "high_performance": VisionMode.HIGH_PERFORMANCE,
            "precision": VisionMode.PRECISION
        }
        
        resolution_str = self.config.get("resolution", "medium")
        resolution_map = {
            "low": Resolution.LOW,
            "medium": Resolution.MEDIUM,
            "high": Resolution.HIGH,
            "ultra": Resolution.ULTRA
        }
        
        strategy_str = self.config.get("processing_strategy", "parallel")
        strategy_map = {
            "sequential": ProcessingStrategy.SEQUENTIAL,
            "parallel": ProcessingStrategy.PARALLEL,
            "pipelined": ProcessingStrategy.PIPELINED,
            "adaptive": ProcessingStrategy.ADAPTIVE
        }
        
        return VisionConfig(
            mode=mode_map.get(mode_str, VisionMode.BALANCED),
            resolution=resolution_map.get(resolution_str, Resolution.MEDIUM),
            fps_target=self.config.get("fps_target", 30),
            processing_strategy=strategy_map.get(strategy_str, ProcessingStrategy.PARALLEL),
            enable_cnn=self.config.get("enable_cnn", True),
            enable_pipeline=self.config.get("enable_pipeline", True),
            chunk_size=self.config.get("chunk_size", 10),
            pipeline_workers=self.config.get("pipeline_workers", 4),
            cnn_batch_size=self.config.get("cnn_batch_size", 32),
            cnn_input_size=self.config.get("cnn_input_size", 224),
            enable_detection=self.config.get("enable_detection", True),
            enable_caption=self.config.get("enable_caption", True),
            enable_activity=self.config.get("enable_activity", True),
            enable_face=self.config.get("enable_face", True),
            enable_decision=self.config.get("enable_decision", False),
            enable_gpu=self.config.get("enable_gpu", True),
            enable_adaptive_fps=self.config.get("enable_adaptive_fps", True),
            enable_frame_skip=self.config.get("enable_frame_skip", True),
            max_frame_skip=self.config.get("max_frame_skip", 5),
            processing_timeout=self.config.get("processing_timeout", 2.0),
            queue_size=self.config.get("queue_size", 100),
            enable_metrics=self.config.get("enable_metrics", True),
            enable_health_check=self.config.get("enable_health_check", True),
            enable_persistence=self.config.get("enable_persistence", False),
            storage_path=self.config.get("storage_path", "./vision_data")
        )
    
    # ------------------------
    # INITIALIZATION
    # ------------------------
    async def initialize(self) -> bool:
        """Initialize all vision components"""
        async with self._lock:
            if self.is_initialized:
                logger.warning("Vision engine already initialized")
                return True
            
            try:
                logger.info("🔍 Initializing Vision System...")
                
                # Import components
                from core.vision.webcam import Webcam
                from core.vision.object_detector import ObjectDetector
                from core.vision.captioner import Captioner
                from core.vision.perception_loop import PerceptionLoop
                from core.brain.decision_engine import DecisionEngine
                from core.vision.activity_recognition import ActivityRecognitionEngine
                from core.vision.face_recognition import FaceRecognitionEngine
                
                # Initialize components
                width, height = self.vision_config.resolution.value
                self.webcam = Webcam(width=width, height=height, fps=self.vision_config.fps_target)
                self.object_detector = ObjectDetector(confidence_threshold=0.5)
                self.captioner = Captioner(model_name="advanced")
                self.perception_loop = PerceptionLoop(fps=self.vision_config.fps_target)
                self.decision_engine = DecisionEngine()
                self.activity_engine = ActivityRecognitionEngine()
                self.face_engine = FaceRecognitionEngine()
                
                # Initialize CNN engine
                if self.vision_config.enable_cnn:
                    cnn_config = {
                        "enable_gpu": self.vision_config.enable_gpu,
                        "batch_size": self.vision_config.cnn_batch_size,
                        "chunk_size": self.vision_config.chunk_size,
                        "enable_pipeline": self.vision_config.enable_pipeline,
                        "input_size": self.vision_config.cnn_input_size
                    }
                    self.cnn_engine = CNNEngineInterface(cnn_config)
                    await self.cnn_engine.start()
                    logger.info("✅ CNN Engine initialized")
                
                # Initialize shared memory
                self.vision_memory = SharedVisionMemory()
                
                # Setup perception pipeline
                if self.perception_loop:
                    self.perception_loop.add_step("detection", self._detect_objects)
                    self.perception_loop.add_step("caption", self._safe_caption)
                    self.perception_loop.add_step("activity", self._detect_activity)
                    self.perception_loop.add_step("face", self._detect_faces)
                    if self.cnn_engine:
                        self.perception_loop.add_step("cnn", self._process_cnn)
                
                # Load face database
                if self.face_engine:
                    try:
                        await asyncio.wait_for(self.face_engine.load_known_faces(), timeout=10)
                        logger.info("✅ Face recognition loaded")
                    except asyncio.TimeoutError:
                        logger.warning("Face preload timeout")
                    except Exception as e:
                        logger.warning(f"Face preload failed: {e}")
                
                # Start webcam
                webcam_ok = await self._start_webcam()
                if not webcam_ok:
                    logger.warning("Webcam failed to start, running in fallback mode")
                
                self.is_initialized = True
                self.is_active = True
                self.start_time = time.time()
                
                logger.info("✅ Vision System Fully Initialized")
                return True
                
            except Exception as e:
                logger.error(f"Vision initialization failed: {e}")
                self.is_initialized = False
                return False
    
    async def _start_webcam(self) -> bool:
        """Start webcam with timeout"""
        try:
            if self.webcam:
                return await asyncio.wait_for(self.webcam.start(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Webcam start timeout")
        except Exception as e:
            logger.error(f"Webcam start error: {e}")
        return False
    
    # ------------------------
    # MAIN LOOP
    # ------------------------
    async def start_vision(self) -> bool:
        """Start vision processing"""
        if not self.is_initialized:
            logger.error("Vision engine not initialized")
            return False
        
        if self._main_task and not self._main_task.done():
            logger.warning("Vision already running")
            return True
        
        try:
            # Create frame queue
            self.frame_queue = asyncio.Queue(maxsize=self.vision_config.queue_size)
            
            # Start pipeline workers
            for worker_id in range(self.vision_config.pipeline_workers):
                task = asyncio.create_task(
                    self._pipeline_worker(worker_id),
                    name=f"vision_worker_{worker_id}"
                )
                self.pipeline_tasks.append(task)
            
            # Start main loop
            self._main_task = asyncio.create_task(self._main_loop(), name="vision_main")
            
            logger.info(f"✅ Vision started with {self.vision_config.pipeline_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start vision: {e}")
            return False
    
    async def _main_loop(self):
        """Main vision processing loop"""
        frame_skip_counter = 0
        last_metrics_time = time.time()
        
        try:
            while self.is_active and not self._shutdown_event.is_set():
                loop_start = time.time()
                
                try:
                    # Capture frame
                    frame = await self._capture_frame()
                    if frame is None:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Adaptive frame skipping
                    if self.vision_config.enable_frame_skip:
                        processing_time = self.performance_monitor.get_stats().get("avg_processing_ms", 33) / 1000
                        target_time = 1.0 / self.vision_config.fps_target
                        
                        if processing_time > target_time and frame_skip_counter < self.vision_config.max_frame_skip:
                            frame_skip_counter += 1
                            self.performance_monitor.record_frame_drop()
                            continue
                        else:
                            frame_skip_counter = 0
                    
                    # Process frame
                    result = await self._process_frame_with_timeout(frame)
                    
                    if result:
                        self.frames_processed += 1
                        
                        # Update performance metrics
                        processing_time = time.time() - loop_start
                        frame_interval = loop_start - self.last_frame_time if self.last_frame_time else 0
                        self.performance_monitor.record_frame(processing_time, frame_interval)
                        self.last_frame_time = loop_start
                        
                        # Adaptive resolution
                        if self.vision_config.enable_adaptive_fps:
                            current_fps = self.performance_monitor.get_fps()
                            new_resolution = self.resolution_manager.update(current_fps, processing_time)
                            if new_resolution != self.vision_config.resolution:
                                self.vision_config.resolution = new_resolution
                                await self._update_webcam_resolution()
                    
                    # Periodic metrics logging
                    if self.vision_config.enable_metrics and (time.time() - last_metrics_time) >= self.vision_config.metrics_interval:
                        stats = self.performance_monitor.get_stats()
                        logger.info(f"📊 Vision Stats - FPS: {stats['fps']:.1f}, "
                                  f"Processing: {stats['avg_processing_ms']:.1f}ms, "
                                  f"Drop Rate: {stats['drop_rate']:.1%}")
                        last_metrics_time = time.time()
                    
                    # FPS control
                    elapsed = time.time() - loop_start
                    target_frame_time = 1.0 / self.vision_config.fps_target
                    sleep_time = max(0, target_frame_time - elapsed)
                    await asyncio.sleep(sleep_time)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Main loop error: {e}")
                    await asyncio.sleep(0.1)
                    
        finally:
            logger.info("Main loop stopped")
    
    async def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame with timeout and resolution adaptation"""
        if not self.webcam:
            return None
        
        try:
            frame = await asyncio.wait_for(self.webcam.capture_frame(), timeout=2.0)
            
            # Apply resolution if needed
            if frame is not None and self.vision_config.resolution != Resolution.MEDIUM:
                target_size = self.vision_config.resolution.value
                if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                    frame = cv2.resize(frame, target_size)
            
            return frame
            
        except asyncio.TimeoutError:
            logger.debug("Frame capture timeout")
            return None
        except Exception as e:
            logger.debug(f"Frame capture error: {e}")
            return None
    
    async def _process_frame_with_timeout(self, frame: np.ndarray) -> Optional[Dict]:
        """Process frame with timeout protection"""
        try:
            return await asyncio.wait_for(
                self.process_single_frame(frame),
                timeout=self.vision_config.processing_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Frame processing timeout")
            return None
    
    async def _pipeline_worker(self, worker_id: int):
        """Worker for processing frame chunks"""
        while self.is_active and not self._shutdown_event.is_set():
            try:
                chunk = await asyncio.wait_for(
                    self.frame_buffer.get_chunk(),
                    timeout=1.0
                )
                
                if chunk and self.cnn_engine:
                    frames = [item["frame"] for item in chunk]
                    results = await self.cnn_engine.process_frames_chunked(frames)
                    
                    async with self._lock:
                        self.cnn_results.extend(results)
                        self.chunks_processed += 1
                        
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    # ------------------------
    # FRAME PROCESSING
    # ------------------------
    async def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame with all enabled features"""
        start_time = time.time()
        
        results = {
            "frame_id": self.frames_processed,
            "timestamp": start_time,
            "detections": [],
            "caption": None,
            "activity": None,
            "faces": [],
            "cnn_features": None,
            "processing_time": 0
        }
        
        try:
            self.current_frame = frame
            
            # Add to buffer for chunk processing
            if self.cnn_engine:
                await self.frame_buffer.add_frame(frame, {"frame_id": self.frames_processed})
            
            # Parallel task execution
            tasks = {}
            
            if self.vision_enabled["detection"] and self.object_detector:
                tasks["detection"] = self._detect_objects(frame)
            
            if self.vision_enabled["caption"] and self.captioner:
                tasks["caption"] = self._safe_caption(frame)
            
            if self.vision_enabled["activity"] and self.activity_engine:
                tasks["activity"] = self._detect_activity(frame)
            
            if self.vision_enabled["face"] and self.face_engine:
                tasks["face"] = self._detect_faces(frame)
            
            if self.vision_enabled["cnn"] and self.cnn_engine:
                tasks["cnn"] = self._process_cnn(frame)
            
            # Execute all tasks
            if tasks:
                if self.vision_config.processing_strategy == ProcessingStrategy.PARALLEL:
                    task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                    
                    for (key, _), result in zip(tasks.items(), task_results):
                        if isinstance(result, Exception):
                            logger.debug(f"{key} error: {result}")
                            continue
                        
                        results[key] = result if result else results[key]
                        
                else:  # Sequential
                    for key, task in tasks.items():
                        try:
                            result = await task
                            if result:
                                results[key] = result
                        except Exception as e:
                            logger.debug(f"{key} error: {e}")
            
            # Update current state
            self.current_detections = results["detections"] if isinstance(results["detections"], list) else []
            self.current_caption = results["caption"] if isinstance(results["caption"], str) else None
            self.current_activity = results["activity"] if isinstance(results["activity"], dict) else {}
            self.current_faces = results["faces"] if isinstance(results["faces"], list) else []
            self.cnn_results = results["cnn_features"] if isinstance(results["cnn_features"], list) else []
            
            # Add metrics
            results["processing_time"] = time.time() - start_time
            results["fps"] = self.performance_monitor.get_fps()
            
            # Publish to shared memory
            if self.vision_memory:
                asyncio.create_task(self._publish_to_memory(results))
            
            return results
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Frame processing error: {e}")
            results["error"] = str(e)
            return results
    
    async def _publish_to_memory(self, results: Dict):
        """Publish results to shared memory"""
        try:
            await self.vision_memory.publish_frame(
                data={
                    "detections": results.get("detections", []),
                    "caption": results.get("caption"),
                    "activity": results.get("activity"),
                    "faces": results.get("faces", []),
                    "cnn_features": results.get("cnn_features"),
                    "fps": results.get("fps", 0),
                    "metadata": {
                        "frame_id": results.get("frame_id"),
                        "processing_time": results.get("processing_time")
                    }
                },
                source=DataSource.CAMERA,
                priority=DataPriority.NORMAL
            )
        except Exception as e:
            logger.debug(f"Memory publish error: {e}")
    
    # ------------------------
    # PIPELINE STEPS
    # ------------------------
    async def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame"""
        if not self.object_detector:
            return []
        
        try:
            if asyncio.iscoroutinefunction(self.object_detector.detect):
                detections = await self.object_detector.detect(frame)
            else:
                detections = await asyncio.to_thread(self.object_detector.detect, frame)
            
            return [d.to_dict() if hasattr(d, 'to_dict') else d for d in (detections or [])]
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return []
    
    async def _detect_activity(self, frame: np.ndarray) -> Dict:
        """Detect activity in frame"""
        if not self.activity_engine:
            return {}
        
        try:
            if asyncio.iscoroutinefunction(self.activity_engine.process_frame):
                return await self.activity_engine.process_frame(frame)
            else:
                return await asyncio.to_thread(self.activity_engine.process_frame, frame)
        except Exception as e:
            logger.debug(f"Activity detection error: {e}")
            return {"activity": "unknown", "error": str(e)}
    
    async def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame"""
        if not self.face_engine:
            return []
        
        try:
            if asyncio.iscoroutinefunction(self.face_engine.recognize_faces):
                return await self.face_engine.recognize_faces(frame)
            else:
                return await asyncio.to_thread(self.face_engine.recognize_faces, frame)
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []
    
    async def _safe_caption(self, frame: np.ndarray) -> Optional[str]:
        """Generate caption safely"""
        if not self.captioner:
            return None
        
        try:
            if hasattr(self.captioner, 'caption'):
                result = await asyncio.to_thread(self.captioner.caption, frame)
                return result.text if hasattr(result, 'text') else str(result)
            return None
        except Exception as e:
            logger.debug(f"Caption error: {e}")
            return None
    
    async def _process_cnn(self, frame: np.ndarray) -> Dict:
        """Process frame with CNN engine"""
        if not self.cnn_engine:
            return {}
        
        try:
            return await self.cnn_engine.process_frame(frame)
        except Exception as e:
            logger.debug(f"CNN processing error: {e}")
            return {}
    
    # ------------------------
    # UTILITY METHODS
    # ------------------------
    async def _update_webcam_resolution(self):
        """Update webcam resolution"""
        if self.webcam and hasattr(self.webcam, 'set_resolution'):
            width, height = self.vision_config.resolution.value
            await self.webcam.set_resolution(width, height)
            logger.info(f"Updated webcam resolution to {width}x{height}")
    
    async def stop_vision(self):
        """Stop vision processing"""
        self._shutdown_event.set()
        self.is_active = False
        
        # Cancel main task
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        # Cancel pipeline workers
        for task in self.pipeline_tasks:
            if not task.done():
                task.cancel()
        
        if self.pipeline_tasks:
            await asyncio.gather(*self.pipeline_tasks, return_exceptions=True)
            self.pipeline_tasks.clear()
        
        # Stop components
        if self.cnn_engine:
            await self.cnn_engine.stop()
        
        if self.webcam:
            await self.webcam.stop()
        
        logger.info("Vision system stopped")
    
    # ------------------------
    # STATE & STATISTICS
    # ------------------------
    def get_current_state(self) -> Dict:
        """Get current vision state"""
        return {
            "is_active": self.is_active,
            "is_initialized": self.is_initialized,
            "frames_processed": self.frames_processed,
            "chunks_processed": self.chunks_processed,
            "fps": self.performance_monitor.get_fps(),
            "error_count": self._error_count,
            "resolution": self.vision_config.resolution.value,
            "mode": self.vision_config.mode.value,
            "features": self.vision_enabled.copy(),
            "timestamp": time.time()
        }
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            "vision": self.get_current_state(),
            "performance": self.performance_monitor.get_stats(),
            "config": self.vision_config.to_dict(),
            "pipeline": {
                "workers": len(self.pipeline_tasks),
                "queue_size": self.frame_queue.qsize() if self.frame_queue else 0,
                "buffer_size": asyncio.run_coroutine_threadsafe(
                    self.frame_buffer.size(), asyncio.get_event_loop()
                ).result() if self.frame_buffer else 0
            }
        }
        
        # Add CNN metrics
        if self.cnn_engine:
            stats["cnn"] = self.cnn_engine.get_metrics()
        
        # Add component stats
        for name, component in [
            ("detector", self.object_detector),
            ("captioner", self.captioner),
            ("activity", self.activity_engine),
            ("face", self.face_engine)
        ]:
            if component and hasattr(component, 'get_stats'):
                try:
                    stats[name] = component.get_stats()
                except Exception:
                    stats[name] = {}
        
        return stats
    
    async def health_check(self) -> Dict:
        """Perform health check"""
        health = {
            "status": "healthy" if self.is_active else "inactive",
            "webcam": self.webcam is not None,
            "cnn_engine": self.cnn_engine is not None and self.cnn_engine.is_running,
            "memory": self.vision_memory is not None,
            "frames_per_second": self.performance_monitor.get_fps(),
            "error_rate": self._error_count / max(1, self.frames_processed),
            "uptime_seconds": time.time() - (self.start_time or time.time())
        }
        
        if self.cnn_engine:
            health["cnn_health"] = await self.cnn_engine.health_check()
        
        return health
    
    # ------------------------
    # CONTROL METHODS
    # ------------------------
    def enable_feature(self, feature: str, enabled: bool) -> bool:
        """Enable/disable specific features"""
        if feature in self.vision_enabled:
            self.vision_enabled[feature] = enabled
            logger.info(f"Feature '{feature}' {'enabled' if enabled else 'disabled'}")
            return True
        return False
    
    async def clear_buffer(self):
        """Clear frame buffer"""
        if self.frame_buffer:
            await self.frame_buffer.clear()
            logger.info("Frame buffer cleared")
    
    async def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame"""
        return self.current_frame
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_monitor = PerformanceMonitor()
        self.frames_processed = 0
        self.chunks_processed = 0
        self._error_count = 0
        logger.info("Statistics reset")


# ------------------------
# CONVENIENCE FUNCTIONS
# ------------------------
async def create_vision_engine(config: Dict = None) -> VisionEngine:
    """Factory function for VisionEngine"""
    engine = VisionEngine(config)
    if await engine.initialize():
        return engine
    raise RuntimeError("Failed to initialize vision engine")


# ------------------------
# USAGE EXAMPLE
# ------------------------
async def main():
    """Example usage"""
    # Configure vision engine
    config = {
        "mode": "balanced",
        "resolution": "medium",
        "fps_target": 30,
        "enable_cnn": True,
        "pipeline_workers": 4,
        "chunk_size": 10,
        "enable_adaptive_fps": True
    }
    
    # Create and initialize engine
    engine = await create_vision_engine(config)
    
    # Start vision
    await engine.start_vision()
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop
    await engine.stop_vision()
    
    # Print stats
    print(json.dumps(engine.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())