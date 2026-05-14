"""
Vision Engine - FULLY INTEGRATED AI VISION SYSTEM
✔ Real-time webcam processing
✔ Works with PerceptionLoop
✔ Feeds AI (DecisionEngine ready)
✔ Stable async loop
✔ Clean state tracking
✔ Fixed captioner compatibility
✔ CNN Engine integration with chunking & pipeline
"""

import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import time
from datetime import datetime
from core.utils.logger import logger
from .shared_memory import SharedVisionMemory


def _create_cnn_engine(config: Dict[str, Any]):
    from core.brain.cnn_engine import CNNEngine

    return CNNEngine(config)


class VisionEngine:
    """Central vision system for EDIATH with CNN pipeline support"""

    def __init__(self, config: Dict[str, Any] = None):
        # ------------------------
        # STATE
        # ------------------------
        self.is_active = False
        self.config = config or {}

        # ------------------------
        # COMPONENTS
        # ------------------------
        self.webcam = None
        self.object_detector = None
        self.captioner = None
        self.perception_loop = None

        # AI
        self.decision_engine = None
        self.activity_engine = None
        self.face_engine = None

        # CNN Engine (NEW)
        self.cnn_engine = None
        self.vision_memory = None

        # ------------------------
        # FRAME DATA
        # ------------------------
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Dict[str, Any]] = []
        self.current_caption: Optional[str] = None
        self.current_activity: Optional[Dict[str, Any]] = None
        self.current_faces: List[Dict[str, Any]] = []
        self.cnn_results: List[Dict[str, Any]] = []

        # ------------------------
        # PERFORMANCE METRICS
        # ------------------------
        self.frames_processed = 0
        self.last_frame_time = 0
        self.fps = 0
        self.chunks_processed = 0
        self.pipeline_tasks = []

        # ------------------------
        # FEATURE FLAGS
        # ------------------------
        self.vision_enabled = {
            "detection": True,
            "caption": True,
            "activity": True,
            "face": True,
            "decision": False,
            "cnn": self.config.get("enable_cnn", True),
        }

        # ------------------------
        # CHUNKING CONFIG
        # ------------------------
        self.chunk_size = self.config.get("chunk_size", 10)
        self.pipeline_workers = self.config.get("pipeline_workers", 4)
        self.frame_buffer = []
        self.chunk_lock = asyncio.Lock()
        self.frame_queue = None

        # ------------------------
        # INTERNAL CONTROL
        # ------------------------
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._error_count = 0
        self._shutdown_event = asyncio.Event()

        # Shared memory (updated)
        self.vision_memory = SharedVisionMemory()

    # ------------------------
    # INITIALIZE
    # ------------------------
    async def initialize(self, *args, **kwargs) -> bool:
        """
        Initialize full vision system safely with CNN engine
        """

        import asyncio

        try:
            # ------------------------
            # 🔒 PREVENT DOUBLE INIT (CRITICAL FIX)
            # ------------------------
            async with self._lock:
                if getattr(self, "is_active", False):
                    if hasattr(self, "logger"):
                        self.logger.warning("Vision system already initialized")
                    return True

                if hasattr(self, "logger"):
                    self.logger.info(
                        "🔍 Initializing Vision System with CNN Pipeline..."
                    )

                # ------------------------
                # 🔥 SAFE IMPORTS
                # ------------------------
                try:
                    from core.vision.webcam import Webcam
                    from core.vision.object_detector import ObjectDetector
                    from core.vision.captioner import Captioner
                    from core.vision.perception_loop import PerceptionLoop
                    from core.brain.decision_engine import DecisionEngine
                    from core.vision.activity_recognition import (
                        ActivityRecognitionEngine,
                    )
                    from core.vision.face_recognition import FaceRecognitionEngine
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Import failure: {e}")
                    return False

                # ------------------------
                # 🔥 CNN ENGINE INIT (NEW)
                # ------------------------
                try:
                    cnn_config = {
                        "enable_cuda": self.config.get("enable_cuda", True),
                        "batch_size": self.config.get("cnn_batch_size", 32),
                        "chunk_size": self.chunk_size,
                        "pipeline_workers": self.pipeline_workers,
                        "enable_pipeline": self.config.get("enable_pipeline", True),
                        "model_path": self.config.get("cnn_model_path", None),
                        "input_size": self.config.get("cnn_input_size", 224),
                    }

                    self.cnn_engine = _create_cnn_engine(cnn_config)
                    await self.cnn_engine.start()

                    if hasattr(self, "logger"):
                        self.logger.info(
                            "✓ CNN Engine initialized with pipeline support"
                        )

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"CNN Engine init failed: {e}")
                    self.cnn_engine = None

                # ------------------------
                # 🔥 SAFE COMPONENT INIT
                # ------------------------
                try:
                    self.webcam = Webcam(width=640, height=480, fps=30)
                    self.object_detector = ObjectDetector(confidence_threshold=0.5)
                    self.captioner = Captioner(model_name="advanced")
                    self.perception_loop = PerceptionLoop(fps=30)
                    self.decision_engine = DecisionEngine()
                    self.activity_engine = ActivityRecognitionEngine()
                    self.face_engine = FaceRecognitionEngine()
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Component init failed: {e}")
                    return False

                # ------------------------
                # 🔥 FACE PRELOAD (TIMEOUT SAFE)
                # ------------------------
                try:
                    if self.face_engine and hasattr(
                        self.face_engine, "load_known_faces"
                    ):
                        await asyncio.wait_for(
                            self.face_engine.load_known_faces(), timeout=10
                        )
                        if hasattr(self, "logger"):
                            self.logger.info("Face recognition loaded")
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Face preload timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Face preload failed: {e}")

                # ------------------------
                # 🔥 PIPELINE SETUP (SAFE)
                # ------------------------
                try:
                    if self.perception_loop:
                        self.perception_loop.add_step("detection", self._detect_objects)
                        self.perception_loop.add_step("caption", self._generate_caption)
                        self.perception_loop.add_step("activity", self._detect_activity)
                        self.perception_loop.add_step("face", self._detect_faces)
                        self.perception_loop.add_step("cnn", self._process_cnn_chunked)
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Pipeline setup failed: {e}")
                    return False

                # ------------------------
                # 🔥 START WEBCAM (SAFE)
                # ------------------------
                webcam_ok = False
                try:
                    if self.webcam:
                        webcam_ok = await asyncio.wait_for(
                            self.webcam.start(), timeout=5
                        )
                        if webcam_ok:
                            if hasattr(self, "logger"):
                                self.logger.info("Webcam started successfully")
                        else:
                            if hasattr(self, "logger"):
                                self.logger.warning("Webcam failed → fallback active")
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Webcam start timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Webcam start error: {e}")

                # ------------------------
                # 🔥 HEALTH CHECK
                # ------------------------
                self.health = {
                    "webcam": webcam_ok,
                    "detector": self.object_detector is not None,
                    "captioner": self.captioner is not None,
                    "perception": self.perception_loop is not None,
                    "decision": self.decision_engine is not None,
                    "activity": self.activity_engine is not None,
                    "face": self.face_engine is not None,
                    "cnn": self.cnn_engine is not None,
                }

                # ------------------------
                # 🔥 STATE SET
                # ------------------------
                self.is_active = True
                self.initialized_at = datetime.utcnow().isoformat()
                self._error_count = 0

                # ------------------------
                # 🔥 FINAL VALIDATION
                # ------------------------
                if not all(self.health.values()):
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Partial initialization: {self.health}")
                    return False

                if hasattr(self, "logger"):
                    self.logger.info(
                        "Vision System Fully Initialized with CNN Pipeline"
                    )

                return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Vision init critical failure: {e}")
            except Exception:
                pass

            self.is_active = False
            self.health = {}
            return False

    # ------------------------
    # START
    # ------------------------
    async def start_vision(self) -> bool:
        """
        Start vision system safely with pipeline support
        """

        import asyncio

        try:
            # ------------------------
            # 🔒 LOCK (CRITICAL FIX)
            # ------------------------
            async with self._lock:

                # ------------------------
                # 🔥 VALIDATION
                # ------------------------
                if not getattr(self, "is_active", False):
                    if hasattr(self, "logger"):
                        self.logger.warning("Vision system not initialized")
                    return False

                if self._task and not self._task.done():
                    if hasattr(self, "logger"):
                        self.logger.warning("Vision already running")
                    return True

                # ------------------------
                # 🔥 START WEBCAM (SAFE + TIMEOUT)
                # ------------------------
                webcam_ok = False
                try:
                    if self.webcam:
                        webcam_ok = await asyncio.wait_for(
                            self.webcam.start(), timeout=5
                        )
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Webcam start timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Webcam start error: {e}")

                if not webcam_ok:
                    if hasattr(self, "logger"):
                        self.logger.error("Webcam failed to start")
                    return False

                # ------------------------
                # 🔥 START CNN ENGINE IF ENABLED
                # ------------------------
                if self.cnn_engine and self.vision_enabled.get("cnn", False):
                    try:
                        await self.cnn_engine.start()
                        if hasattr(self, "logger"):
                            self.logger.info("CNN Engine pipeline started")
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"CNN Engine start failed: {e}")

                # ------------------------
                # 🔥 START PERCEPTION LOOP
                # ------------------------
                try:
                    if self.perception_loop:
                        await asyncio.wait_for(self.perception_loop.start(), timeout=5)
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Perception loop start timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Perception loop start failed: {e}")
                    return False

                # ------------------------
                # 🔥 START CHUNK PROCESSING TASKS
                # ------------------------
                self.frame_queue = asyncio.Queue(maxsize=100)

                for worker_id in range(self.pipeline_workers):
                    task = asyncio.create_task(
                        self._chunk_processing_worker(worker_id),
                        name=f"vision_chunk_worker_{worker_id}",
                    )
                    self.pipeline_tasks.append(task)

                # ------------------------
                # 🔥 START MAIN LOOP
                # ------------------------
                try:
                    self._task = asyncio.create_task(
                        self._vision_loop(), name="vision-main-loop"
                    )
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Failed to start vision loop: {e}")
                    return False

                # ------------------------
                # 🔥 FINAL STATE
                # ------------------------
                if hasattr(self, "logger"):
                    self.logger.info(
                        f"Vision system started with {self.pipeline_workers} workers"
                    )

                return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"[Vision Start Error]: {e}")
            except Exception:
                pass

            return False

    # ------------------------
    # STOP
    # ------------------------
    async def stop_vision(self) -> None:
        """
        Stop vision system safely
        """

        import asyncio

        try:
            # ------------------------
            # 🔒 LOCK (CRITICAL FIX)
            # ------------------------
            async with self._lock:
                self._shutdown_event.set()

                # ------------------------
                # 🔥 ALREADY STOPPED
                # ------------------------
                if not getattr(self, "is_active", False):
                    if hasattr(self, "logger"):
                        self.logger.debug("Vision already stopped")
                    return

                # ------------------------
                # 🔥 STATE UPDATE
                # ------------------------
                self.is_active = False

                # ------------------------
                # 🔥 STOP CNN ENGINE
                # ------------------------
                if self.cnn_engine:
                    try:
                        await asyncio.wait_for(self.cnn_engine.stop(), timeout=5)
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"CNN Engine stop error: {e}")

                # ------------------------
                # 🔥 STOP CHUNK WORKERS
                # ------------------------
                for task in self.pipeline_tasks:
                    if not task.done():
                        task.cancel()

                if self.pipeline_tasks:
                    await asyncio.gather(*self.pipeline_tasks, return_exceptions=True)
                    self.pipeline_tasks.clear()

                # ------------------------
                # 🔥 STOP MAIN LOOP (SAFE)
                # ------------------------
                if self._task:
                    try:
                        if not self._task.done():
                            self._task.cancel()
                            await asyncio.wait_for(
                                asyncio.gather(self._task, return_exceptions=True),
                                timeout=5,
                            )
                    except asyncio.TimeoutError:
                        if hasattr(self, "logger"):
                            self.logger.warning("Vision task cancel timeout")
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.error(f"Vision task cancel error: {e}")
                    finally:
                        self._task = None

                # ------------------------
                # 🔥 STOP PERCEPTION LOOP FIRST
                # ------------------------
                try:
                    if self.perception_loop:
                        await asyncio.wait_for(self.perception_loop.stop(), timeout=5)
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Perception loop stop timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Perception loop stop error: {e}")

                # ------------------------
                # 🔥 STOP WEBCAM LAST
                # ------------------------
                try:
                    if self.webcam:
                        await asyncio.wait_for(self.webcam.stop(), timeout=5)
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Webcam stop timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Webcam stop error: {e}")

                # ------------------------
                # 🔥 RESET STATE
                # ------------------------
                self.current_frame = None
                self.current_detections = []
                self.current_caption = None
                self.current_activity = None
                self.current_faces = []
                self.cnn_results = []
                self.frame_buffer = []

                # ------------------------
                # 🔥 LOG SUCCESS
                # ------------------------
                if hasattr(self, "logger"):
                    self.logger.info("Vision system stopped")

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ stop_vision failed: {e}")
            except Exception:
                pass

    # ------------------------
    # CHUNK PROCESSING WORKER
    # ------------------------
    async def _chunk_processing_worker(self, worker_id: int):
        """Worker for processing frame chunks in parallel"""

        while self.is_active and not self._shutdown_event.is_set():
            try:
                # Get chunk from queue
                chunk_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                if chunk_data is None:
                    break

                frames = chunk_data.get("frames", [])
                chunk_id = chunk_data.get("chunk_id", 0)

                if frames and self.cnn_engine:
                    # Process chunk through CNN engine
                    results = await self.cnn_engine.process_frames_chunked(frames)

                    async with self.chunk_lock:
                        self.cnn_results.extend(results)
                        self.chunks_processed += 1

                self.frame_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

    # ------------------------
    # MAIN LOOP
    # ------------------------
    async def _vision_loop(self):
        """
        Main vision loop with chunking support
        """

        import asyncio
        import builtins

        try:
            if hasattr(self, "logger"):
                self.logger.info("Vision loop started with chunking")

            frame_batch = []
            last_chunk_time = time.time()
            chunk_interval = 1.0  # Process chunks every second

            while getattr(self, "is_active", False):

                try:
                    start_time = time.time()

                    # ------------------------
                    # 🔥 SAFE FRAME CAPTURE (TIMEOUT FIX)
                    # ------------------------
                    frame = None
                    try:
                        frame = await asyncio.wait_for(
                            self.webcam.capture_frame(), timeout=2
                        )
                    except asyncio.TimeoutError:
                        if hasattr(self, "logger"):
                            self.logger.warning("Frame capture timeout")
                        await asyncio.sleep(0.05)
                        continue
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Capture error: {e}")
                        await asyncio.sleep(0.05)
                        continue

                    if frame is None:
                        await asyncio.sleep(0.01)
                        continue

                    # ------------------------
                    # 🔥 BUFFER FRAMES FOR CHUNKING
                    # ------------------------
                    frame_batch.append(frame)

                    # Process chunk when buffer is full or time expired
                    current_time = time.time()
                    if (
                        len(frame_batch) >= self.chunk_size
                        or (current_time - last_chunk_time) >= chunk_interval
                    ):

                        if frame_batch:
                            chunk_id = self.chunks_processed
                            await self.frame_queue.put(
                                {
                                    "chunk_id": chunk_id,
                                    "frames": frame_batch.copy(),
                                    "timestamp": current_time,
                                }
                            )

                            frame_batch.clear()
                            last_chunk_time = current_time

                    # ------------------------
                    # 🔥 SAFE UI UPDATE (NON-BLOCKING)
                    # ------------------------
                    try:
                        ui = getattr(builtins, "EDIATH_UI", None)
                        if ui and hasattr(ui, "update_camera_frame"):
                            await asyncio.to_thread(ui.update_camera_frame, frame)
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"UI update skipped: {e}")

                    # ------------------------
                    # 🔥 PROCESS SINGLE FRAME (for real-time features)
                    # ------------------------
                    result = None
                    try:
                        result = await self.process_single_frame(frame)
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Frame process failed: {e}")

                    # ------------------------
                    # 🔥 OPTIONAL AI (NON-BLOCKING SAFETY)
                    # ------------------------
                    if (
                        result
                        and self.vision_enabled.get("decision")
                        and self.decision_engine
                    ):
                        try:
                            await asyncio.wait_for(
                                self._process_with_ai(result), timeout=5
                            )
                        except asyncio.TimeoutError:
                            if hasattr(self, "logger"):
                                self.logger.warning("AI processing timeout")
                        except Exception as e:
                            if hasattr(self, "logger"):
                                self.logger.warning(f"AI process error: {e}")

                    # ------------------------
                    # 🔥 METRICS (SAFE)
                    # ------------------------
                    self.frames_processed += 1

                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        current_fps = 1 / elapsed
                        self.fps = (
                            (self.fps * 0.8 + current_fps * 0.2)
                            if self.fps
                            else current_fps
                        )

                    # ------------------------
                    # 🔥 SMART FRAME RATE CONTROL
                    # ------------------------
                    target_delay = max(0, (1 / 30) - elapsed)
                    await asyncio.sleep(target_delay)

                except asyncio.CancelledError:
                    if hasattr(self, "logger"):
                        self.logger.info("Vision loop cancelled")
                    break

                except Exception as e:
                    self._error_count += 1
                    if hasattr(self, "logger"):
                        self.logger.error(f"[Vision Loop Error]: {e}")
                    await asyncio.sleep(0.2)

        finally:
            self.is_active = False
            try:
                if hasattr(self, "logger"):
                    self.logger.info("Vision loop stopped")
            except Exception:
                pass

    # ------------------------
    # PROCESS SINGLE FRAME
    # ------------------------
    async def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Production-grade frame processor with CNN integration
        """

        import asyncio
        import numpy as np

        start = time.time()

        # ------------------------
        # 🔥 VALIDATION
        # ------------------------
        if frame is None or not isinstance(frame, np.ndarray):
            return {}

        results = {
            "frame_id": getattr(self, "frames_processed", 0),
            "detections": [],
            "caption": None,
            "activity": None,
            "faces": [],
            "cnn_features": None,
            "timestamp": start,
            "processing_time": 0.0,
        }

        try:
            self.current_frame = frame
            tasks = {}

            # ------------------------
            # 🔥 SAFE TASK CREATION
            # ------------------------
            try:
                if self.vision_enabled.get("detection") and self.object_detector:
                    tasks["detection"] = asyncio.create_task(
                        asyncio.to_thread(self.object_detector.detect, frame)
                    )
            except Exception:
                pass

            try:
                if self.vision_enabled.get("caption") and self.captioner:
                    tasks["caption"] = asyncio.create_task(self._safe_caption(frame))
            except Exception:
                pass

            try:
                if self.vision_enabled.get("activity") and self.activity_engine:
                    tasks["activity"] = asyncio.create_task(
                        self.activity_engine.process_frame(frame)
                    )
            except Exception:
                pass

            try:
                if self.vision_enabled.get("face") and self.face_engine:
                    tasks["face"] = asyncio.create_task(
                        self.face_engine.recognize_faces(frame)
                    )
            except Exception:
                pass

            try:
                if self.vision_enabled.get("cnn") and self.cnn_engine:
                    tasks["cnn"] = asyncio.create_task(
                        self.cnn_engine.process_frame(frame)
                    )
            except Exception:
                pass

            # ------------------------
            # 🔥 EXECUTE WITH TIMEOUT (CRITICAL FIX)
            # ------------------------
            task_results = []
            if tasks:
                try:
                    task_results = await asyncio.wait_for(
                        asyncio.gather(*tasks.values(), return_exceptions=True),
                        timeout=2,
                    )
                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Frame processing timeout")
                    for t in tasks.values():
                        t.cancel()

            # ------------------------
            # 🔥 SAFE RESULT MAPPING
            # ------------------------
            for key, result in zip(tasks.keys(), task_results):

                if isinstance(result, Exception):
                    if hasattr(self, "logger"):
                        self.logger.debug(f"{key} error: {result}")
                    continue

                try:
                    if key == "detection":
                        processed = [
                            d.to_dict() if hasattr(d, "to_dict") else d
                            for d in (result or [])
                        ]
                        self.current_detections = processed
                        results["detections"] = processed

                    elif key == "caption":
                        if hasattr(result, "text"):
                            text = result.text
                        elif isinstance(result, dict):
                            text = result.get("text")
                        else:
                            text = str(result) if result else None

                        self.current_caption = text
                        results["caption"] = text

                    elif key == "activity":
                        self.current_activity = result
                        results["activity"] = result

                    elif key == "face":
                        self.current_faces = result or []
                        results["faces"] = result or []

                    elif key == "cnn":
                        results["cnn_features"] = result

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"{key} parse error: {e}")

        except Exception as e:
            self._error_count += 1
            if hasattr(self, "logger"):
                self.logger.error(f"Frame processing error: {e}")

        # ------------------------
        # 🔥 METRICS
        # ------------------------
        elapsed = time.time() - start

        results["processing_time"] = round(elapsed, 4)
        results["fps"] = (
            self.fps if self.fps > 0 else (1 / elapsed if elapsed > 0 else 0)
        )
        results["camera_status"] = "active" if self.webcam else "fallback"

        # ------------------------
        # 🔥 SAFE MEMORY PUBLISH (NON-BLOCKING)
        # ------------------------
        try:
            if self.vision_memory:
                asyncio.create_task(
                    self.vision_memory.publish_frame(
                        {
                            "detections": results["detections"],
                            "caption": results["caption"],
                            "activity": results["activity"],
                            "faces": results["faces"],
                            "cnn_features": results.get("cnn_features"),
                            "fps": results["fps"],
                            "timestamp": results["timestamp"],
                        }
                    )
                )
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.debug(f"Memory publish error: {e}")

        return results

    # ------------------------
    # PROCESS CNN CHUNKED (NEW)
    # ------------------------
    async def _process_cnn_chunked(self, frame: np.ndarray):
        """Process frame through CNN engine with chunking"""
        if not self.cnn_engine or not self.vision_enabled.get("cnn", False):
            return None

        try:
            result = await self.cnn_engine.process_frame(frame)
            return result
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.debug(f"CNN chunked processing error: {e}")
            return None

    # ------------------------
    # SAFE CAPTION WRAPPER
    # ------------------------
    async def _safe_caption(self, frame: np.ndarray) -> Optional[str]:
        """
        Safe caption generator (optimized + production-grade)
        """

        import asyncio
        import numpy as np

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if (
                self.captioner is None
                or frame is None
                or not isinstance(frame, np.ndarray)
            ):
                return None

            # ------------------------
            # 🔍 METHOD FALLBACKS
            # ------------------------
            method_names = ["generate", "caption", "generate_caption", "describe"]

            for method_name in method_names:

                if not hasattr(self.captioner, method_name):
                    continue

                method = getattr(self.captioner, method_name)

                try:
                    # ------------------------
                    # ⏱️ EXECUTE WITH TIMEOUT (CRITICAL FIX)
                    # ------------------------
                    if asyncio.iscoroutinefunction(method):
                        result = await asyncio.wait_for(method(frame), timeout=3)
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(method, frame), timeout=3
                        )

                    if not result:
                        continue

                    # ------------------------
                    # 🔥 RESULT NORMALIZATION
                    # ------------------------
                    if hasattr(result, "text"):
                        text = result.text

                    elif isinstance(result, dict):
                        text = (
                            result.get("text")
                            or result.get("caption")
                            or result.get("description")
                        )

                    elif isinstance(result, str):
                        text = result

                    else:
                        text = str(result)

                    # ------------------------
                    # 🧹 CLEAN OUTPUT
                    # ------------------------
                    if text:
                        text = str(text).strip()
                        if text:
                            return text

                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"{method_name} timeout")
                    continue

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"{method_name} failed: {e}")
                    continue

            return None

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Caption generation error: {e}")
            except Exception:
                pass

            return None

    # ------------------------
    # AI PROCESSING
    # ------------------------
    async def _process_with_ai(self, vision_data: Dict[str, Any]):
        """
        AI processing pipeline (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not self.decision_engine or not vision_data:
                if hasattr(self, "logger"):
                    self.logger.debug("Decision engine not available or empty data")
                return None

            # ------------------------
            # 🔥 STEP 1: BUILD VISUAL CONTEXT (WITH TIMEOUT)
            # ------------------------
            try:
                vision_context = await asyncio.wait_for(
                    self.decision_engine.process_vision(vision_data), timeout=3
                )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("Vision context processing timeout")
                return None

            # Skip if throttled / empty
            if not vision_context or not isinstance(vision_context, str):
                return None

            # ------------------------
            # 🔥 STEP 2: BUILD FINAL INPUT (CLEAN)
            # ------------------------
            final_input = (
                f"{vision_context.strip()}\n\n"
                "Describe what you see and respond naturally."
            )

            # ------------------------
            # 🔥 STEP 3: RUN DECISION ENGINE (WITH TIMEOUT)
            # ------------------------
            try:
                decision = await asyncio.wait_for(
                    self.decision_engine.intelligent_decision(final_input), timeout=5
                )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("AI decision timeout")
                return None

            if not decision:
                return None

            # ------------------------
            # 🔥 STEP 4: STORE RESULT (SAFE)
            # ------------------------
            try:
                self.last_decision = decision
            except Exception:
                pass

            # ------------------------
            # 🔥 METRICS (OPTIONAL)
            # ------------------------
            try:
                self.ai_calls = getattr(self, "ai_calls", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"🧠 AI Decision: {str(decision)[:120]}")
            except Exception:
                pass

            return decision

        except asyncio.CancelledError:
            try:
                if hasattr(self, "logger"):
                    self.logger.info("AI processing cancelled")
            except Exception:
                pass
            raise

        except Exception as e:
            self._error_count += 1
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"[AI Vision Error]: {e}")
            except Exception:
                pass

            return None

    # ------------------------
    # PIPELINE STEPS
    # ------------------------
    async def _detect_objects(self, frame: np.ndarray):
        """Detect objects safely"""
        import asyncio
        import numpy as np

        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return []

            if not getattr(self, "object_detector", None):
                return []

            detect_fn = getattr(self.object_detector, "detect", None)
            if not callable(detect_fn):
                return []

            try:
                if asyncio.iscoroutinefunction(detect_fn):
                    detections = await asyncio.wait_for(detect_fn(frame), timeout=2)
                else:
                    detections = await asyncio.wait_for(
                        asyncio.to_thread(detect_fn, frame), timeout=2
                    )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("Object detection timeout")
                return []

            if detections is None:
                return []

            if not isinstance(detections, (list, tuple)):
                detections = [detections]

            results = []

            for d in detections:
                if d is None:
                    continue

                try:
                    if hasattr(d, "to_dict"):
                        parsed = d.to_dict()
                    elif isinstance(d, dict):
                        parsed = d
                    else:
                        parsed = {"type": "unknown", "value": str(d)}

                    if isinstance(parsed, dict):
                        parsed.setdefault("confidence", None)
                        parsed.setdefault("label", parsed.get("type", "unknown"))

                    results.append(parsed)
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"Detection parse error: {e}")

            try:
                self.current_detections = results
            except Exception:
                pass

            return results

        except Exception as e:
            self._error_count += 1
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Object detection error: {e}")
            except Exception:
                pass
            return []

    async def _detect_activity(self, frame: np.ndarray):
        """Detect activity safely"""
        import asyncio
        import numpy as np

        start = time.time()

        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return {}

            engine = getattr(self, "activity_engine", None)
            if not engine:
                return {}

            process_fn = getattr(engine, "process_frame", None)
            if not callable(process_fn):
                return {}

            try:
                if asyncio.iscoroutinefunction(process_fn):
                    result = await asyncio.wait_for(process_fn(frame), timeout=2)
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(process_fn, frame), timeout=2
                    )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("Activity detection timeout")
                return getattr(self, "current_activity", {"activity": "timeout"})
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Activity execution error: {e}")
                return getattr(self, "current_activity", {"activity": "error"})

            if result is None:
                result = {}

            if not isinstance(result, dict):
                result = {"activity": str(result)}

            result.setdefault("activity", "unknown")
            result.setdefault("confidence", None)

            try:
                self.current_activity = result
            except Exception:
                pass

            try:
                elapsed = time.time() - start
                self.activity_latency = round(elapsed, 4)
            except Exception:
                pass

            return result

        except Exception as e:
            self._error_count += 1
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Activity detection error: {e}")
            except Exception:
                pass
            return {"activity": "error"}

    async def _detect_faces(self, frame: np.ndarray):
        """Detect faces safely"""
        import asyncio
        import numpy as np

        start = time.time()

        try:
            if frame is None or not isinstance(frame, np.ndarray):
                return []

            engine = getattr(self, "face_engine", None)
            if not engine:
                return []

            recognize_fn = getattr(engine, "recognize_faces", None)
            if not callable(recognize_fn):
                return []

            try:
                if asyncio.iscoroutinefunction(recognize_fn):
                    result = await asyncio.wait_for(recognize_fn(frame), timeout=2)
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(recognize_fn, frame), timeout=2
                    )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("Face detection timeout")
                return getattr(self, "current_faces", [])
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Face execution error: {e}")
                return getattr(self, "current_faces", [])

            if result is None:
                result = []

            if not isinstance(result, (list, tuple)):
                result = [result]

            faces = []

            for face in result:
                if face is None:
                    continue

                try:
                    if hasattr(face, "to_dict"):
                        parsed = face.to_dict()
                    elif isinstance(face, dict):
                        parsed = face
                    else:
                        parsed = {"type": "unknown_face", "value": str(face)}

                    if isinstance(parsed, dict):
                        parsed.setdefault("name", "unknown")
                        parsed.setdefault("confidence", None)

                    faces.append(parsed)
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"Face parse error: {e}")

            try:
                self.current_faces = faces
            except Exception:
                pass

            try:
                elapsed = time.time() - start
                self.face_latency = round(elapsed, 4)
            except Exception:
                pass

            return faces

        except Exception as e:
            self._error_count += 1
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Face detection error: {e}")
            except Exception:
                pass
            return []

    async def _generate_caption(self, frame: np.ndarray):
        """Generate caption for frame"""
        return await self._safe_caption(frame)

    # ------------------------
    # CONTROL METHODS
    # ------------------------
    def enable_detection(self, enabled: bool = True) -> bool:
        previous = self.vision_enabled.get("detection", False)
        self.vision_enabled["detection"] = enabled
        logger.info(f"Detection {'enabled' if enabled else 'disabled'}")
        return previous != enabled

    def enable_caption(self, enabled: bool = True) -> bool:
        previous = self.vision_enabled.get("caption", False)
        self.vision_enabled["caption"] = enabled
        logger.info(f"Caption {'enabled' if enabled else 'disabled'}")
        return previous != enabled

    def enable_ai_decision(self, enabled: bool = True) -> bool:
        previous = self.vision_enabled.get("decision", False)
        self.vision_enabled["decision"] = enabled
        if enabled and not self.decision_engine:
            logger.warning("AI decision enabled but decision_engine not initialized")
        logger.info(f"AI Decision {'enabled' if enabled else 'disabled'}")
        return previous != enabled

    def enable_cnn(self, enabled: bool = True) -> bool:
        previous = self.vision_enabled.get("cnn", False)
        self.vision_enabled["cnn"] = enabled
        logger.info(f"CNN Engine {'enabled' if enabled else 'disabled'}")
        return previous != enabled

    # ------------------------
    # STATE METHODS
    # ------------------------
    def get_current_state(self) -> Dict[str, Any]:
        """Get current vision system state"""

        try:
            state = {
                "is_active": bool(getattr(self, "is_active", False)),
                "frames_processed": int(getattr(self, "frames_processed", 0)),
                "fps": float(getattr(self, "fps", 0.0)),
                "chunks_processed": int(getattr(self, "chunks_processed", 0)),
                "detections": list(getattr(self, "current_detections", []) or []),
                "caption": getattr(self, "current_caption", None),
                "activity": dict(getattr(self, "current_activity", {}) or {}),
                "faces": list(getattr(self, "current_faces", []) or []),
                "errors": int(getattr(self, "_error_count", 0)),
                "features": dict(getattr(self, "vision_enabled", {}) or {}),
                "latency": {
                    "activity": getattr(self, "activity_latency", None),
                    "face": getattr(self, "face_latency", None),
                },
                "timestamp": time.time(),
            }

            # Add CNN metrics if available
            if self.cnn_engine:
                state["cnn_metrics"] = self.cnn_engine.get_metrics()

            return state

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"State fetch error: {e}")
            except Exception:
                pass
            return {"is_active": False, "error": "state_fetch_failed"}

    def get_stats(self) -> Dict[str, Any]:
        """Get system stats"""

        try:
            stats = {
                "frames_processed": int(getattr(self, "frames_processed", 0)),
                "fps": float(getattr(self, "fps", 0.0)),
                "chunks_processed": int(getattr(self, "chunks_processed", 0)),
                "pipeline_workers": self.pipeline_workers,
                "chunk_size": self.chunk_size,
                "vision_enabled": dict(getattr(self, "vision_enabled", {}) or {}),
                "errors": int(getattr(self, "_error_count", 0)),
                "detections_count": len(getattr(self, "current_detections", []) or []),
                "faces_count": len(getattr(self, "current_faces", []) or []),
                "last_activity": dict(getattr(self, "current_activity", {}) or {}),
                "timestamp": time.time(),
            }

            # Add component stats
            def safe_stats(component):
                try:
                    if component and hasattr(component, "get_stats"):
                        return component.get_stats()
                except Exception:
                    return {}
                return {}

            stats["detector"] = safe_stats(getattr(self, "object_detector", None))
            stats["captioner"] = safe_stats(getattr(self, "captioner", None))
            stats["activity"] = safe_stats(getattr(self, "activity_engine", None))
            stats["face"] = safe_stats(getattr(self, "face_engine", None))

            # Add CNN stats
            if self.cnn_engine:
                stats["cnn"] = self.cnn_engine.get_metrics()

            # Memory stats
            try:
                if getattr(self, "vision_memory", None):
                    stats["memory"] = self.vision_memory.get_stats()
                else:
                    stats["memory"] = {}
            except Exception:
                stats["memory"] = {}

            stats["latency"] = {
                "activity": getattr(self, "activity_latency", None),
                "face": getattr(self, "face_latency", None),
            }

            return stats

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Stats error: {e}")
            except Exception:
                pass
            return {"error": "stats_failed", "timestamp": time.time()}

    async def get_latest_frame(self):
        return self.current_frame

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "status": "healthy" if self.is_active else "inactive",
            "webcam": self.webcam is not None,
            "cnn_engine": self.cnn_engine is not None,
            "decision_engine": self.decision_engine is not None,
            "pipeline_workers": len(self.pipeline_tasks),
            "frames_per_second": self.fps,
            "error_rate": self._error_count / max(1, self.frames_processed),
        }

        if self.cnn_engine:
            health_status["cnn_health"] = await self.cnn_engine.health_check()

        return health_status


__all__ = ["VisionEngine"]
