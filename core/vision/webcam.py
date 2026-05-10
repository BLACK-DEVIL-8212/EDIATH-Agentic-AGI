"""
Webcam - Production Ready Video Capture
✔ Real webcam support (OpenCV)
✔ Fallback simulation (if camera not available)
✔ Async safe
✔ Stable FPS control
✔ Frame buffer optimized
✔ No duplicate methods
"""

import asyncio
from typing import Dict, Any, Optional, List
import numpy as np
import time
import threading

try:
    import cv2
except ImportError:
    cv2 = None

from core.utils.logger import logger


# ------------------------
# FRAME BUFFER
# ------------------------
class FrameBuffer:
    def __init__(self, max_frames: int = 30):
        try:
            # Validation
            max_frames = int(max_frames)
            if max_frames <= 0:
                raise ValueError("max_frames must be > 0")

            self.max_frames = max_frames

            # Storage
            self.frames: List[np.ndarray] = []
            self.timestamps: List[float] = []

            # Lock (thread safe)
            self._lock = threading.Lock()

            # Metrics
            self.total_added = 0
            self.total_dropped = 0

        except Exception as e:
            logger.warning(f"FrameBuffer init failed: {e}")
            # Fallback without numpy
            self.frames: List[Any] = []
            self.timestamps: List[float] = []
            self._lock = threading.Lock()
            self.total_added = 0
            self.total_dropped = 0

    def add_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        try:
            with self._lock:
                self.frames.append(frame)
                self.timestamps.append(time.time())
                self.total_added += 1

                # Overflow control
                if len(self.frames) > self.max_frames:
                    self.frames.pop(0)
                    self.timestamps.pop(0)
                    self.total_dropped += 1
        except Exception:
            pass

    def get_latest(self) -> Optional[np.ndarray]:
        try:
            with self._lock:
                return self.frames[-1] if self.frames else None
        except Exception:
            return None

    def get_last_n(self, n: int) -> List[np.ndarray]:
        try:
            n = max(1, int(n))
            with self._lock:
                return self.frames[-n:] if self.frames else []
        except Exception:
            return []

    def get_timestamps(self) -> List[float]:
        try:
            with self._lock:
                return list(self.timestamps)
        except Exception:
            return []

    def clear(self) -> None:
        try:
            with self._lock:
                self.frames.clear()
                self.timestamps.clear()
        except Exception:
            pass

    def size(self) -> int:
        try:
            with self._lock:
                return len(self.frames)
        except Exception:
            return 0

    def is_empty(self) -> bool:
        return self.size() == 0

    def get_stats(self) -> Dict[str, Any]:
        try:
            with self._lock:
                return {
                    "current_size": len(self.frames),
                    "max_frames": self.max_frames,
                    "total_added": self.total_added,
                    "total_dropped": self.total_dropped,
                    "drop_rate": (
                        round(self.total_dropped / self.total_added, 3)
                        if self.total_added > 0
                        else 0
                    ),
                }
        except Exception:
            return {
                "current_size": 0,
                "max_frames": self.max_frames,
                "total_added": 0,
                "total_dropped": 0,
                "drop_rate": 0,
            }

    def export_frames(self) -> List[np.ndarray]:
        try:
            with self._lock:
                return list(self.frames)
        except Exception:
            return []


# ------------------------
# WEBCAM
# ------------------------
class Webcam:
    def __init__(
        self, width: int = 640, height: int = 480, fps: int = 30, device_id: int = 0
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.device_id = device_id

        self.is_active = False
        self.frame_count = 0
        self.frame_buffer = FrameBuffer()
        self._last_capture = 0

        self.cap = None

        # Check dependencies
        self.has_numpy = np is not None
        self.has_cv2 = cv2 is not None

        if not self.has_numpy:
            logger.warning("NumPy not available - using fallback mode")
        if not self.has_cv2:
            logger.warning("OpenCV not available - using simulation mode")

    # ------------------------
    # START
    # ------------------------
    async def start(self) -> bool:
        self.frame_count = 0
        self.frame_buffer.clear()
        self._last_capture = 0

        if cv2:
            try:
                # Force DSHOW on Windows
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)

                if not self.cap or not self.cap.isOpened():
                    logger.warning("⚠ Camera failed with DSHOW, trying default...")
                    # fallback
                    self.cap = cv2.VideoCapture(self.device_id)

                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                    self.is_active = True
                    logger.info("📷 Webcam started successfully")
                    return True
                else:
                    logger.warning("⚠ Webcam not available → switching to simulation")

            except Exception as e:
                logger.error(f"Camera init error: {e}")

        # Fallback mode (safe)
        self.cap = None
        self.is_active = True
        logger.warning("⚠ Using simulated frames (no real camera)")
        return True

    # ------------------------
    # STOP
    # ------------------------
    async def stop(self) -> None:
        self.is_active = False

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    # ------------------------
    # CAPTURE SINGLE FRAME
    # ------------------------
    async def capture_frame(self) -> Optional[np.ndarray]:
        if not self.is_active:
            return None

        start_time = time.time()
        frame = None

        # Real camera
        if self.cap and self.has_cv2:
            try:
                ret, img = self.cap.read()
                if ret and img is not None:
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"Frame grab failed: {e}")

        # Fallback simulation
        if frame is None:
            # Return last frame if available
            last = self.frame_buffer.get_latest()
            if last is not None:
                frame = last
            else:
                # Create a blank frame
                if self.has_numpy:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frame = [[0 for _ in range(self.width)] for _ in range(self.height)]

        # Store frame
        try:
            self.frame_count += 1
            self.frame_buffer.add_frame(frame)
        except Exception:
            pass

        # FPS control
        elapsed = time.time() - start_time
        sleep_time = max(0, (1 / self.fps) - elapsed)
        await asyncio.sleep(sleep_time)

        return frame

    # ------------------------
    # CAPTURE WITH THROTTLE (FOR VISION AI)
    # ------------------------
    async def capture_frame_throttled(
        self, min_interval: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Capture frame with minimum time interval between captures.
        Use this for AI vision processing to avoid overloading.
        """
        if not self.is_active:
            return None

        now = time.time()
        if now - self._last_capture < min_interval:
            await asyncio.sleep(0.1)
            return None

        self._last_capture = now
        return await self.capture_frame()

    # ------------------------
    # STREAM MULTIPLE FRAMES
    # ------------------------
    async def stream(self, duration: float = 10.0) -> List[np.ndarray]:
        """Capture multiple frames over a duration"""
        frames = []
        start_time = time.time()

        while time.time() - start_time < duration:
            frame = await self.capture_frame()
            if frame is not None:
                frames.append(frame)

        return frames

    # ------------------------
    # GET LATEST FRAME
    # ------------------------
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from buffer (non-blocking)"""
        return self.frame_buffer.get_latest()

    # ------------------------
    # GET STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "device_id": self.device_id,
            "buffer_stats": self.frame_buffer.get_stats(),
            "has_cv2": self.has_cv2,
            "has_numpy": self.has_numpy,
            "real_camera": self.cap is not None,
        }

    # ------------------------
    # CLEAR BUFFER
    # ------------------------
    def clear_buffer(self) -> None:
        self.frame_buffer.clear()
