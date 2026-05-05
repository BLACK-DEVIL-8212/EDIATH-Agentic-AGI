# activity_recognition.py

import cv2
import numpy as np
import asyncio
from typing import Dict, Optional
from datetime import datetime


class ActivityRecognitionEngine:
    def __init__(self):
        # ------------------------
        # STATE
        # ------------------------
        self.last_activity: Optional[str] = None
        self.last_detected_time: Optional[datetime] = None

        self.prev_frame = None
        self._running = False

        # 🔥 NEW: smoothing buffer
        self.motion_history = []

        print("✅ Activity Engine Initialized (Improved)")

    # --------------------------------------------------
    # MOTION DETECTION (IMPROVED)
    # --------------------------------------------------
    def _detect_motion(self, gray):
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0

        # 🔥 Frame difference
        diff = cv2.absdiff(self.prev_frame, gray)

        # 🔥 Noise reduction
        blur = cv2.GaussianBlur(diff, (5, 5), 0)

        # 🔥 Thresholding
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # 🔥 Remove noise blobs
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 🔥 Motion score
        motion = np.sum(thresh) / 255

        self.prev_frame = gray
        return motion

    # --------------------------------------------------
    # SMOOTHING (NEW)
    # --------------------------------------------------
    def _smooth_motion(self, motion):
        self.motion_history.append(motion)

        if len(self.motion_history) > 5:
            self.motion_history.pop(0)

        return np.mean(self.motion_history)

    # --------------------------------------------------
    # ACTIVITY CLASSIFICATION (IMPROVED)
    # --------------------------------------------------
    def _classify_activity(self, motion: float) -> str:
        if motion < 800:
            return "idle"
        elif motion < 4000:
            return "slow_movement"
        elif motion < 15000:
            return "walking"
        else:
            return "running"

    # --------------------------------------------------
    # PROCESS FRAME
    # --------------------------------------------------
    async def process_frame(self, frame) -> Dict:
        try:
            if frame is None:
                return {}

            # 🔥 FIX: OpenCV uses BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            motion = self._detect_motion(gray)
            motion = self._smooth_motion(motion)

            activity = self._classify_activity(motion)

            self.last_activity = activity
            self.last_detected_time = datetime.now()

            return {
                "activity": activity,
                "motion_level": int(motion),
                "timestamp": self.last_detected_time.isoformat(),
            }

        except Exception as e:
            print(f"Activity error: {e}")
            return {}

    # --------------------------------------------------
    # MONITOR LOOP
    # --------------------------------------------------
    async def monitor(self, frame_source_callable):
        self._running = True
        print("🎥 Activity monitoring started")

        try:
            while self._running:
                frame = await frame_source_callable()

                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                result = await self.process_frame(frame)

                if result:
                    print(f"Activity: {result}")

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            print("Activity monitor cancelled")

        except Exception as e:
            print(f"Monitor error: {e}")

        finally:
            self._running = False
            print("🛑 Activity monitoring stopped")

    # --------------------------------------------------
    # STOP
    # --------------------------------------------------
    def stop(self):
        self._running = False

    # --------------------------------------------------
    # STATE
    # --------------------------------------------------
    def get_current_activity(self) -> Dict:
        return {
            "activity": self.last_activity,
            "last_detected": (
                self.last_detected_time.isoformat() if self.last_detected_time else None
            ),
        }

    # --------------------------------------------------
    # ALERT SYSTEM
    # --------------------------------------------------
    def is_suspicious(self) -> bool:
        return self.last_activity in ["running"]

    # --------------------------------------------------
    # DRAW OVERLAY
    # --------------------------------------------------
    def draw_activity(self, frame, activity_data: Dict):
        try:
            activity = activity_data.get("activity", "unknown")

            cv2.putText(
                frame,
                f"Activity: {activity}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        except Exception:
            pass

        return frame

    # --------------------------------------------------
    # STATS
    # --------------------------------------------------
    def get_stats(self):
        return {
            "mode": "opencv_motion_improved",
            "last_activity": self.last_activity,
            "history_size": len(self.motion_history),
        }
