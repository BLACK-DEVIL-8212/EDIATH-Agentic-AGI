"""
Advanced Wakeword Engine - Real-Time Detection
"""

from typing import Callable, Dict, Any, List, Optional
from enum import Enum

from ..utils.logger import logger


class WakewordStatus(Enum):
    WAITING = "waiting"
    DETECTED = "detected"
    ACTIVE = "active"


class Wakeword:
    def __init__(self, wakeword: str = "hello"):
        self.wakeword = wakeword.lower()
        self.status = WakewordStatus.WAITING

        self.detections = 0
        self.history: List[str] = []

    # ------------------------
    # TEXT MATCH (STT OUTPUT)
    # ------------------------
    def detect_text(self, text: str) -> bool:
        text = text.lower()

        detected = self.wakeword in text

        if detected:
            self.detections += 1
            self.status = WakewordStatus.DETECTED
            logger.info(f"🟢 Wakeword detected: {self.wakeword}")

        return detected

    # ------------------------
    # STREAM LOOP 🔥
    # ------------------------
    async def listen_loop(self, stt_callback, on_wake: Optional[Callable] = None):
        """
        Continuous listening loop
        stt_callback = function that returns transcribed text
        """

        logger.info("🎤 Wakeword listening started")

        while True:
            try:
                text = await stt_callback()

                if not text:
                    continue

                self.history.append(text)

                if self.detect_text(text):
                    self.status = WakewordStatus.ACTIVE

                    if on_wake:
                        await on_wake()

            except Exception as e:
                logger.error(f"Wakeword error: {e}")

    # ------------------------
    # RESET
    # ------------------------
    def reset(self):
        self.status = WakewordStatus.WAITING

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "wakeword": self.wakeword,
            "status": self.status.value,
            "detections": self.detections,
            "history_size": len(self.history),
        }
