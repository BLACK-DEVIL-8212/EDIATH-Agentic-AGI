# shared_memory.py (IMPROVED - SAFE VERSION)

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from threading import Lock
import logging

logger = logging.getLogger(__name__)


# ------------------------
# DATA MODEL
# ------------------------
@dataclass
class VisionFrame:
    frame_id: int
    timestamp: float
    detections: List[Dict[str, Any]] = field(default_factory=list)
    caption: Optional[str] = None
    activity: Optional[Dict[str, Any]] = None
    faces: List[Dict[str, Any]] = field(default_factory=list)
    fps: float = 0.0

    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections": self.detections,
            "caption": self.caption,
            "activity": self.activity,
            "faces": self.faces,
            "fps": self.fps,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


# ------------------------
# SUBSCRIPTION
# ------------------------
class Subscription:
    def __init__(self, queue: asyncio.Queue, sid: str, filter_fn=None):
        self.queue = queue
        self.sid = sid
        self.filter_fn = filter_fn
        self.last_activity = time.time()

    def is_active(self):
        return time.time() - self.last_activity < 30


# ------------------------
# MAIN MEMORY
# ------------------------
class SharedVisionMemory:
    _instance = None
    _lock = Lock()

    MAX_FRAMES = 30
    TTL = 60
    QUEUE_SIZE = 10

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_init"):
            return

        self._init = True

        self.frames: List[VisionFrame] = []
        self.subscribers: Dict[str, Subscription] = {}

        self.frame_lock = asyncio.Lock()
        self.sub_lock = asyncio.Lock()

        self.frame_id = 0
        self.running = True

        print("✅ SharedVisionMemory Ready")

    # ------------------------
    # PUBLISH
    # ------------------------
    async def publish_frame(self, data: Dict[str, Any]) -> bool:
        try:
            self.frame_id += 1

            frame = VisionFrame(
                frame_id=self.frame_id,
                timestamp=time.time(),
                detections=data.get("detections", []),
                caption=data.get("caption"),
                activity=data.get("activity"),
                faces=data.get("faces", []),
                fps=data.get("fps", 0.0),
            )

            async with self.frame_lock:
                self.frames.append(frame)
                self._prune()

            await self._notify(frame)

            return True

        except Exception as e:
            print("Publish error:", e)
            return False

    # ------------------------
    # PRUNE (SAFE)
    # ------------------------
    def _prune(self):
        now = time.time()

        self.frames = [f for f in self.frames if now - f.timestamp < self.TTL]

        if len(self.frames) > self.MAX_FRAMES:
            self.frames = self.frames[-self.MAX_FRAMES :]

    # ------------------------
    # SUBSCRIBE (FIXED)
    # ------------------------
    async def subscribe(self, sid: str, filter_fn=None) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=self.QUEUE_SIZE)

        sub = Subscription(queue, sid, filter_fn)

        async with self.sub_lock:
            self.subscribers[sid] = sub

        print(f"Subscriber {sid} connected")
        return queue

    # ------------------------
    # UNSUBSCRIBE
    # ------------------------
    async def unsubscribe(self, sid: str):
        async with self.sub_lock:
            if sid in self.subscribers:
                del self.subscribers[sid]

    # ------------------------
    # NOTIFY (IMPROVED)
    # ------------------------
    async def _notify(self, frame: VisionFrame):
        async with self.sub_lock:
            subs = list(self.subscribers.values())

        for sub in subs:
            try:
                # filter
                if sub.filter_fn:
                    if not sub.filter_fn(frame):
                        continue

                # 🔥 DROP IF FULL (REAL-TIME FIX)
                if sub.queue.full():
                    sub.queue.get_nowait()

                sub.queue.put_nowait(frame.to_dict())
                sub.last_activity = time.time()

            except Exception as e:
                print("Notify error:", e)

    # ------------------------
    # GET LATEST (FIXED LOCK)
    # ------------------------
    async def get_latest_frame(self):
        async with self.frame_lock:
            if self.frames:
                return self.frames[-1].to_dict()
        return None

    # ------------------------
    # GET RECENT
    # ------------------------
    async def get_recent_frames(self, n=5):
        async with self.frame_lock:
            return [f.to_dict() for f in self.frames[-n:]]

    # ------------------------
    # CLEANUP (IMPROVED)
    # ------------------------
    async def cleanup(self):
        while self.running:
            await asyncio.sleep(10)

            async with self.sub_lock:
                dead = [
                    sid for sid, sub in self.subscribers.items() if not sub.is_active()
                ]

                for sid in dead:
                    del self.subscribers[sid]

    # ------------------------
    # STATS
    # ------------------------
    async def get_stats(self):
        async with self.frame_lock:
            frame_count = len(self.frames)

        async with self.sub_lock:
            sub_count = len(self.subscribers)

        return {"frames": frame_count, "subscribers": sub_count}
