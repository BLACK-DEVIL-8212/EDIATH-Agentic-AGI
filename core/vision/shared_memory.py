# shared_memory.py

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import json
import hashlib
import logging
from threading import Lock, RLock
import pickle
import base64
from pathlib import Path

# ⚠️ CRITICAL FIX: Add numpy import
import numpy as np

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class DataPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class DataSource(Enum):
    CAMERA = "camera"
    DETECTOR = "detector"
    CAPTIONER = "captioner"
    ACTIVITY = "activity"
    FACE_RECOGNITION = "face_recognition"
    USER = "user"
    SYSTEM = "system"


class StorageBackend(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"
    HYBRID = "hybrid"


@dataclass
class SharedMemoryConfig:
    max_frames: int = 100
    ttl_seconds: int = 60
    queue_size: int = 20
    enable_persistence: bool = False
    storage_backend: StorageBackend = StorageBackend.MEMORY
    redis_url: str = "redis://localhost:6379"
    sqlite_path: str = "./vision_memory.db"
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    max_subscribers: int = 100
    subscriber_timeout: int = 30
    cleanup_interval: int = 10
    enable_metrics: bool = True
    enable_replication: bool = False
    replication_factor: int = 2


# ------------------------
# DATA MODELS
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
    source: DataSource = DataSource.CAMERA
    priority: DataPriority = DataPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # Now np is defined!
    hash_id: Optional[str] = None
    
    def __post_init__(self):
        if self.hash_id is None:
            self.hash_id = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute unique hash for frame"""
        content = f"{self.frame_id}_{self.timestamp}_{self.caption}_{len(self.detections)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert non-serializable objects
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist() if hasattr(self.embedding, 'tolist') else str(self.embedding)
        data['source'] = self.source.value if isinstance(self.source, DataSource) else self.source
        data['priority'] = self.priority.value if isinstance(self.priority, DataPriority) else self.priority
        data['datetime'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VisionFrame':
        """Create VisionFrame from dict"""
        # Convert back enum values
        if 'source' in data and isinstance(data['source'], str):
            data['source'] = DataSource(data['source'])
        if 'priority' in data and isinstance(data['priority'], int):
            data['priority'] = DataPriority(data['priority'])
        
        # Restore embedding if present
        embedding = data.pop('embedding', None)
        frame = cls(**data)
        if embedding is not None:
            try:
                # Try to convert back to numpy array
                if isinstance(embedding, list):
                    frame.embedding = np.array(embedding)
                elif isinstance(embedding, np.ndarray):
                    frame.embedding = embedding
            except Exception as e:
                logger.warning(f"Failed to restore embedding: {e}")
                frame.embedding = None
        
        return frame


@dataclass
class FrameEvent:
    event_id: str
    event_type: str
    frame: VisionFrame
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------
# PERSISTENCE LAYER
# ------------------------
class PersistenceLayer:
    def __init__(self, config: SharedMemoryConfig):
        self.config = config
        self.redis_client = None
        self.sqlite_conn = None
        
        if config.storage_backend in [StorageBackend.REDIS, StorageBackend.HYBRID]:
            self._init_redis()
        
        if config.storage_backend in [StorageBackend.SQLITE, StorageBackend.HYBRID]:
            self._init_sqlite()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, falling back to memory")
            return
        
        try:
            self.redis_client = redis.from_url(self.config.redis_url, decode_responses=False)
            self.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        if not SQLITE_AVAILABLE:
            logger.warning("SQLite not available")
            return
        
        try:
            self.sqlite_conn = sqlite3.connect(self.config.sqlite_path, check_same_thread=False)
            self._create_tables()
            logger.info(f"✅ SQLite database initialized at {self.config.sqlite_path}")
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}")
            self.sqlite_conn = None
    
    def _create_tables(self):
        """Create necessary tables"""
        cursor = self.sqlite_conn.cursor()
        
        # Frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY,
                hash_id TEXT UNIQUE,
                timestamp REAL,
                data BLOB,
                created_at REAL
            )
        ''')
        
        # Events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                frame_id INTEGER,
                timestamp REAL,
                metadata TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames (frame_id)
            )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON frames(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)')
        
        self.sqlite_conn.commit()
    
    async def save_frame(self, frame: VisionFrame):
        """Save frame to persistent storage"""
        if not self.config.enable_persistence:
            return
        
        tasks = []
        
        if self.redis_client:
            tasks.append(self._save_to_redis(frame))
        
        if self.sqlite_conn:
            tasks.append(self._save_to_sqlite(frame))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _save_to_redis(self, frame: VisionFrame):
        """Save to Redis"""
        try:
            key = f"frame:{frame.hash_id}"
            data = pickle.dumps(frame.to_dict())
            
            if self.config.enable_compression and len(data) > self.config.compression_threshold:
                import zlib
                data = zlib.compress(data)
            
            await asyncio.to_thread(
                self.redis_client.setex,
                key,
                self.config.ttl_seconds,
                data
            )
            
            # Add to sorted set for ordering
            await asyncio.to_thread(
                self.redis_client.zadd,
                "frames:timeline",
                {frame.hash_id: frame.timestamp}
            )
            
        except Exception as e:
            logger.error(f"Redis save error: {e}")
    
    async def _save_to_sqlite(self, frame: VisionFrame):
        """Save to SQLite"""
        try:
            data = pickle.dumps(frame.to_dict())
            
            def save():
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO frames (frame_id, hash_id, timestamp, data, created_at) VALUES (?, ?, ?, ?, ?)",
                    (frame.frame_id, frame.hash_id, frame.timestamp, data, time.time())
                )
                self.sqlite_conn.commit()
            
            await asyncio.to_thread(save)
            
        except Exception as e:
            logger.error(f"SQLite save error: {e}")
    
    async def load_frame(self, hash_id: str) -> Optional[VisionFrame]:
        """Load frame from persistent storage"""
        if not self.config.enable_persistence:
            return None
        
        # Try Redis first
        if self.redis_client:
            data = await asyncio.to_thread(self.redis_client.get, f"frame:{hash_id}")
            if data:
                try:
                    if self.config.enable_compression:
                        import zlib
                        try:
                            data = zlib.decompress(data)
                        except:
                            pass
                    
                    frame_dict = pickle.loads(data)
                    return VisionFrame.from_dict(frame_dict)
                except Exception as e:
                    logger.error(f"Redis load error: {e}")
        
        # Try SQLite
        if self.sqlite_conn:
            def load():
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT data FROM frames WHERE hash_id = ?", (hash_id,))
                row = cursor.fetchone()
                return row[0] if row else None
            
            data = await asyncio.to_thread(load)
            if data:
                try:
                    frame_dict = pickle.loads(data)
                    return VisionFrame.from_dict(frame_dict)
                except Exception as e:
                    logger.error(f"SQLite load error: {e}")
        
        return None
    
    async def get_recent_frames(self, limit: int = 100) -> List[VisionFrame]:
        """Get recent frames from persistence"""
        frames = []
        
        if self.redis_client:
            # Get from Redis sorted set
            hash_ids = await asyncio.to_thread(
                self.redis_client.zrevrange,
                "frames:timeline",
                0,
                limit - 1
            )
            
            for hash_id in hash_ids:
                frame = await self.load_frame(hash_id.decode() if isinstance(hash_id, bytes) else hash_id)
                if frame:
                    frames.append(frame)
        
        elif self.sqlite_conn:
            def load():
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "SELECT data FROM frames ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                return [pickle.loads(row[0]) for row in cursor.fetchall()]
            
            frame_dicts = await asyncio.to_thread(load)
            frames = [VisionFrame.from_dict(fd) for fd in frame_dicts if fd]
        
        return frames
    
    def close(self):
        """Close persistence connections"""
        if self.redis_client:
            self.redis_client.close()
        
        if self.sqlite_conn:
            self.sqlite_conn.close()


# ------------------------
# SUBSCRIPTION MANAGER
# ------------------------
class Subscription:
    def __init__(
        self,
        queue: asyncio.Queue,
        subscriber_id: str,
        filter_fn: Optional[Callable] = None,
        priority: DataPriority = DataPriority.NORMAL
    ):
        self.queue = queue
        self.sid = subscriber_id
        self.filter_fn = filter_fn
        self.priority = priority
        self.last_activity = time.time()
        self.created_at = time.time()
        self.delivered_count = 0
        self.dropped_count = 0
    
    def is_active(self) -> bool:
        return time.time() - self.last_activity < 30
    
    def can_deliver(self) -> bool:
        return not self.queue.full()
    
    def get_stats(self) -> Dict:
        return {
            "subscriber_id": self.sid,
            "priority": self.priority.value,
            "delivered": self.delivered_count,
            "dropped": self.dropped_count,
            "queue_size": self.queue.qsize(),
            "active_seconds": time.time() - self.last_activity
        }


class SubscriptionManager:
    def __init__(self, max_subscribers: int = 100):
        self.subscribers: Dict[str, Subscription] = {}
        self.max_subscribers = max_subscribers
        self._lock = asyncio.Lock()
    
    async def add(
        self,
        subscriber_id: str,
        filter_fn: Optional[Callable] = None,
        priority: DataPriority = DataPriority.NORMAL
    ) -> asyncio.Queue:
        """Add new subscriber"""
        async with self._lock:
            if len(self.subscribers) >= self.max_subscribers:
                # Remove oldest inactive subscriber
                oldest = min(
                    [(sid, sub) for sid, sub in self.subscribers.items() if not sub.is_active()],
                    key=lambda x: x[1].last_activity,
                    default=None
                )
                if oldest:
                    del self.subscribers[oldest[0]]
                else:
                    raise RuntimeError("Maximum subscribers reached")
            
            queue = asyncio.Queue(maxsize=20)
            subscription = Subscription(queue, subscriber_id, filter_fn, priority)
            self.subscribers[subscriber_id] = subscription
            
            logger.info(f"📡 Subscriber added: {subscriber_id} (priority={priority.value})")
            return queue
    
    async def remove(self, subscriber_id: str):
        """Remove subscriber"""
        async with self._lock:
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
                logger.info(f"📡 Subscriber removed: {subscriber_id}")
    
    async def get_active(self) -> List[Subscription]:
        """Get active subscribers"""
        async with self._lock:
            return [sub for sub in self.subscribers.values() if sub.is_active()]
    
    async def cleanup(self):
        """Remove inactive subscribers"""
        async with self._lock:
            inactive = [sid for sid, sub in self.subscribers.items() if not sub.is_active()]
            for sid in inactive:
                del self.subscribers[sid]
            
            if inactive:
                logger.debug(f"Cleaned up {len(inactive)} inactive subscribers")
    
    async def get_stats(self) -> Dict:
        """Get subscription statistics"""
        async with self._lock:
            return {
                "total_subscribers": len(self.subscribers),
                "active_subscribers": len([s for s in self.subscribers.values() if s.is_active()]),
                "subscribers": [s.get_stats() for s in self.subscribers.values()]
            }


# ------------------------
# FRAME BUFFER (RING BUFFER)
# ------------------------
class FrameBuffer:
    def __init__(self, max_size: int = 100, ttl: int = 60):
        self.max_size = max_size
        self.ttl = ttl
        self.frames: deque = deque(maxlen=max_size)
        self.frame_map: Dict[int, VisionFrame] = {}
        self._lock = asyncio.Lock()
    
    async def add(self, frame: VisionFrame):
        """Add frame to buffer"""
        async with self._lock:
            self.frames.append(frame)
            self.frame_map[frame.frame_id] = frame
            self._prune()
    
    async def get_latest(self) -> Optional[VisionFrame]:
        """Get latest frame"""
        async with self._lock:
            if self.frames:
                return self.frames[-1]
            return None
    
    async def get_by_id(self, frame_id: int) -> Optional[VisionFrame]:
        """Get frame by ID"""
        async with self._lock:
            return self.frame_map.get(frame_id)
    
    async def get_range(self, start_id: int, end_id: int) -> List[VisionFrame]:
        """Get frame range"""
        async with self._lock:
            return [f for f in self.frames if start_id <= f.frame_id <= end_id]
    
    async def get_recent(self, count: int = 10) -> List[VisionFrame]:
        """Get recent frames"""
        async with self._lock:
            return list(self.frames)[-count:]
    
    def _prune(self):
        """Prune expired frames"""
        now = time.time()
        
        # Remove expired
        while self.frames and now - self.frames[0].timestamp > self.ttl:
            expired = self.frames.popleft()
            if expired.frame_id in self.frame_map:
                del self.frame_map[expired.frame_id]
        
        # Maintain size limit
        while len(self.frames) > self.max_size:
            removed = self.frames.popleft()
            if removed.frame_id in self.frame_map:
                del self.frame_map[removed.frame_id]
    
    async def size(self) -> int:
        """Get buffer size"""
        async with self._lock:
            return len(self.frames)


# ------------------------
# EVENT BUS
# ------------------------
class EventBus:
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()
    
    async def emit(self, event_type: str, data: Any, source: str = "system"):
        """Emit event to all listeners"""
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time()
        }
        
        self.event_history.append(event)
        
        listeners = self.listeners.get(event_type, []) + self.listeners.get("*", [])
        
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def on(self, event_type: str, callback: Callable):
        """Register event listener"""
        self.listeners[event_type].append(callback)
        logger.debug(f"Listener registered for event: {event_type}")
    
    def off(self, event_type: str, callback: Callable):
        """Remove event listener"""
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)
    
    async def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get event history"""
        if event_type:
            return [e for e in self.event_history if e["type"] == event_type][-limit:]
        return list(self.event_history)[-limit:]


# ------------------------
# METRICS COLLECTOR
# ------------------------
class MetricsCollector:
    def __init__(self):
        self.frame_rates: deque = deque(maxlen=100)
        self.processing_times: deque = deque(maxlen=100)
        self.queue_sizes: deque = deque(maxlen=100)
        self.errors: List[Dict] = []
        self._lock = asyncio.Lock()
    
    async def record_frame(self, frame_id: int, processing_time: float):
        """Record frame metrics"""
        async with self._lock:
            self.processing_times.append(processing_time)
            
            # Calculate FPS
            if len(self.processing_times) > 1:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.frame_rates.append(fps)
    
    async def record_queue_size(self, size: int):
        """Record queue size"""
        async with self._lock:
            self.queue_sizes.append(size)
    
    async def record_error(self, error: str, source: str):
        """Record error"""
        async with self._lock:
            self.errors.append({
                "error": error,
                "source": source,
                "timestamp": time.time()
            })
            # Keep last 100 errors
            if len(self.errors) > 100:
                self.errors.pop(0)
    
    async def get_stats(self) -> Dict:
        """Get metrics statistics"""
        async with self._lock:
            stats = {
                "avg_fps": sum(self.frame_rates) / len(self.frame_rates) if self.frame_rates else 0,
                "max_fps": max(self.frame_rates) if self.frame_rates else 0,
                "min_fps": min(self.frame_rates) if self.frame_rates else 0,
                "avg_processing_time_ms": (sum(self.processing_times) / len(self.processing_times) * 1000) if self.processing_times else 0,
                "avg_queue_size": sum(self.queue_sizes) / len(self.queue_sizes) if self.queue_sizes else 0,
                "error_count": len(self.errors),
                "last_error": self.errors[-1] if self.errors else None
            }
            
            # Add percentile if we have enough data
            if len(self.processing_times) >= 10:
                stats["p95_processing_time_ms"] = np.percentile(self.processing_times, 95) * 1000
            
            return stats


# ------------------------
# MAIN SHARED VISION MEMORY
# ------------------------
class SharedVisionMemory:
    _instance = None
    _lock = RLock()
    
    def __new__(cls, config: Optional[SharedMemoryConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[SharedMemoryConfig] = None):
        if hasattr(self, "_initialized"):
            return
        
        self.config = config or SharedMemoryConfig()
        
        # Core components
        self.buffer = FrameBuffer(self.config.max_frames, self.config.ttl_seconds)
        self.subscription_manager = SubscriptionManager(self.config.max_subscribers)
        self.event_bus = EventBus()
        self.metrics = MetricsCollector() if self.config.enable_metrics else None
        
        # Persistence
        self.persistence = PersistenceLayer(self.config) if self.config.enable_persistence else None
        
        # State
        self.frame_counter = 0
        self.running = True
        self.start_time = time.time()
        
        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        
        self._initialized = True
        
        logger.info(f"🚀 SharedVisionMemory initialized - Backend: {self.config.storage_backend.value}")
    
    # ------------------------
    # PUBLISHING
    # ------------------------
    async def publish_frame(
        self,
        data: Dict[str, Any],
        source: DataSource = DataSource.SYSTEM,
        priority: DataPriority = DataPriority.NORMAL
    ) -> bool:
        """Publish frame to shared memory"""
        try:
            self.frame_counter += 1
            
            frame = VisionFrame(
                frame_id=self.frame_counter,
                timestamp=time.time(),
                detections=data.get("detections", []),
                caption=data.get("caption"),
                activity=data.get("activity"),
                faces=data.get("faces", []),
                fps=data.get("fps", 0.0),
                source=source,
                priority=priority,
                metadata=data.get("metadata", {})
            )
            
            # Add to buffer
            await self.buffer.add(frame)
            
            # Persist if enabled
            if self.persistence:
                await self.persistence.save_frame(frame)
            
            # Notify subscribers
            await self._notify_subscribers(frame)
            
            # Emit event
            await self.event_bus.emit("frame_published", frame.to_dict(), source.value)
            
            # Record metrics
            if self.metrics:
                await self.metrics.record_queue_size(await self.buffer.size())
            
            logger.debug(f"📤 Frame {frame.frame_id} published (source={source.value})")
            return True
            
        except Exception as e:
            logger.error(f"Publish error: {e}")
            if self.metrics:
                await self.metrics.record_error(str(e), "publish")
            return False
    
    async def publish_batch(self, frames_data: List[Dict[str, Any]]) -> List[bool]:
        """Publish multiple frames"""
        tasks = [self.publish_frame(data) for data in frames_data]
        return await asyncio.gather(*tasks)
    
    # ------------------------
    # SUBSCRIPTION
    # ------------------------
    async def subscribe(
        self,
        subscriber_id: str,
        filter_fn: Optional[Callable] = None,
        priority: DataPriority = DataPriority.NORMAL
    ) -> asyncio.Queue:
        """Subscribe to frame stream"""
        return await self.subscription_manager.add(subscriber_id, filter_fn, priority)
    
    async def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from frame stream"""
        await self.subscription_manager.remove(subscriber_id)
    
    async def _notify_subscribers(self, frame: VisionFrame):
        """Notify all subscribers"""
        subscribers = await self.subscription_manager.get_active()
        
        for sub in subscribers:
            try:
                # Apply filter
                if sub.filter_fn and not await self._apply_filter(sub.filter_fn, frame):
                    continue
                
                # Check queue capacity
                if not sub.can_deliver():
                    sub.dropped_count += 1
                    continue
                
                # Deliver frame
                await sub.queue.put(frame.to_dict())
                sub.delivered_count += 1
                sub.last_activity = time.time()
                
            except Exception as e:
                logger.error(f"Notification error for {sub.sid}: {e}")
    
    async def _apply_filter(self, filter_fn: Callable, frame: VisionFrame) -> bool:
        """Apply filter function"""
        try:
            if asyncio.iscoroutinefunction(filter_fn):
                return await filter_fn(frame)
            return filter_fn(frame)
        except Exception as e:
            logger.error(f"Filter error: {e}")
            return False
    
    # ------------------------
    # DATA RETRIEVAL
    # ------------------------
    async def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """Get latest frame"""
        frame = await self.buffer.get_latest()
        return frame.to_dict() if frame else None
    
    async def get_frame_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """Get frame by ID"""
        frame = await self.buffer.get_by_id(frame_id)
        return frame.to_dict() if frame else None
    
    async def get_recent_frames(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent frames"""
        frames = await self.buffer.get_recent(count)
        return [f.to_dict() for f in frames]
    
    async def get_frame_range(self, start_id: int, end_id: int) -> List[Dict[str, Any]]:
        """Get frame range"""
        frames = await self.buffer.get_range(start_id, end_id)
        return [f.to_dict() for f in frames]
    
    async def search_frames(
        self,
        query: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search frames by criteria"""
        frames = await self.buffer.get_recent(limit * 2)
        
        results = []
        for frame in frames:
            match = True
            for key, value in query.items():
                if hasattr(frame, key):
                    if getattr(frame, key) != value:
                        match = False
                        break
                elif key in frame.metadata:
                    if frame.metadata[key] != value:
                        match = False
                        break
            
            if match:
                results.append(frame.to_dict())
                if len(results) >= limit:
                    break
        
        return results
    
    # ------------------------
    # STATISTICS & MONITORING
    # ------------------------
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "frame_counter": self.frame_counter,
            "buffer_size": await self.buffer.size(),
            "uptime_seconds": time.time() - self.start_time,
            "config": {
                "max_frames": self.config.max_frames,
                "ttl_seconds": self.config.ttl_seconds,
                "storage_backend": self.config.storage_backend.value,
                "persistence_enabled": self.config.enable_persistence
            }
        }
        
        # Add subscription stats
        stats["subscriptions"] = await self.subscription_manager.get_stats()
        
        # Add metrics
        if self.metrics:
            stats["metrics"] = await self.metrics.get_stats()
        
        return stats
    
    async def get_subscriber_stats(self, subscriber_id: str) -> Optional[Dict]:
        """Get specific subscriber stats"""
        subscribers = await self.subscription_manager.get_stats()
        for sub in subscribers.get("subscribers", []):
            if sub.get("subscriber_id") == subscriber_id:
                return sub
        return None
    
    # ------------------------
    # MAINTENANCE
    # ------------------------
    async def cleanup(self):
        """Periodic cleanup task"""
        while self.running:
            await asyncio.sleep(self.config.cleanup_interval)
            
            # Cleanup inactive subscribers
            await self.subscription_manager.cleanup()
            
            # Log metrics periodically
            if self.metrics and self.frame_counter % 100 == 0:
                stats = await self.metrics.get_stats()
                logger.info(f"📊 Metrics: {stats['avg_fps']:.1f} FPS, "
                          f"Queue: {stats['avg_queue_size']:.1f}, "
                          f"Errors: {stats['error_count']}")
    
    async def start(self):
        """Start shared memory service"""
        self.running = True
        self.cleanup_task = asyncio.create_task(self.cleanup())
        await self.event_bus.emit("service_started", {"service": "SharedVisionMemory"}, "system")
        logger.info("✅ SharedVisionMemory service started")
    
    async def stop(self):
        """Stop shared memory service"""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.persistence:
            await asyncio.to_thread(self.persistence.close)
        
        await self.event_bus.emit("service_stopped", {"service": "SharedVisionMemory"}, "system")
        logger.info("🛑 SharedVisionMemory service stopped")
    
    async def clear(self):
        """Clear all data"""
        self.frame_counter = 0
        self.buffer = FrameBuffer(self.config.max_frames, self.config.ttl_seconds)
        await self.event_bus.emit("data_cleared", {}, "system")
        logger.info("🧹 Shared memory cleared")
    
    # ------------------------
    # BACKUP & RESTORE
    # ------------------------
    async def backup(self, filepath: str):
        """Backup shared memory to file"""
        frames = await self.buffer.get_recent(self.config.max_frames)
        
        backup_data = {
            "timestamp": time.time(),
            "frame_counter": self.frame_counter,
            "frames": [f.to_dict() for f in frames],
            "config": asdict(self.config)
        }
        
        def write_backup():
            with open(filepath, 'w') as f:
                json.dump(backup_data, f, indent=2)
        
        await asyncio.to_thread(write_backup)
        logger.info(f"💾 Backup saved to {filepath}")
    
    async def restore(self, filepath: str):
        """Restore shared memory from backup"""
        def read_backup():
            with open(filepath, 'r') as f:
                return json.load(f)
        
        backup_data = await asyncio.to_thread(read_backup)
        
        self.frame_counter = backup_data["frame_counter"]
        
        for frame_dict in backup_data["frames"]:
            frame = VisionFrame.from_dict(frame_dict)
            await self.buffer.add(frame)
        
        logger.info(f"🔄 Restored {len(backup_data['frames'])} frames from {filepath}")


# ------------------------
# CONVENIENCE FUNCTIONS
# ------------------------
async def create_filter_by_source(*sources: DataSource) -> Callable:
    """Create filter for specific sources"""
    source_set = set(sources)
    
    async def filter_fn(frame: VisionFrame) -> bool:
        return frame.source in source_set
    
    return filter_fn


async def create_filter_by_priority(min_priority: DataPriority) -> Callable:
    """Create filter for minimum priority"""
    async def filter_fn(frame: VisionFrame) -> bool:
        return frame.priority.value >= min_priority.value
    
    return filter_fn


# ------------------------
# USAGE EXAMPLE
# ------------------------
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure shared memory
    config = SharedMemoryConfig(
        max_frames=100,
        ttl_seconds=60,
        enable_persistence=True,
        storage_backend=StorageBackend.HYBRID,
        enable_metrics=True
    )
    
    # Initialize shared memory
    memory = SharedVisionMemory(config)
    await memory.start()
    
    # Create subscriber
    subscriber_id = "test_subscriber"
    queue = await memory.subscribe(subscriber_id)
    
    # Define filter for high-priority frames only
    filter_fn = await create_filter_by_priority(DataPriority.HIGH)
    
    # Publish some frames
    for i in range(10):
        frame_data = {
            "detections": [{"class": "person", "confidence": 0.9}],
            "caption": f"Frame {i}",
            "fps": 30.0,
            "metadata": {"index": i}
        }
        
        priority = DataPriority.HIGH if i % 3 == 0 else DataPriority.NORMAL
        await memory.publish_frame(frame_data, DataSource.CAMERA, priority)
        await asyncio.sleep(0.1)
    
    # Consume frames
    print("\n📺 Consuming frames:")
    try:
        for _ in range(5):
            frame = await asyncio.wait_for(queue.get(), timeout=1.0)
            print(f"  Received: Frame {frame['frame_id']} - {frame.get('caption', 'No caption')}")
    except asyncio.TimeoutError:
        print("  Timeout waiting for frames")
    
    # Get statistics
    stats = await memory.get_stats()
    print(f"\n📊 Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Get recent frames
    recent = await memory.get_recent_frames(5)
    print(f"\n📸 Recent frames: {len(recent)}")
    
    # Search frames
    results = await memory.search_frames({"priority": DataPriority.HIGH.value}, limit=3)
    print(f"\n🔍 Search results: {len(results)} high-priority frames")
    
    # Backup
    await memory.backup("memory_backup.json")
    
    # Cleanup
    await memory.unsubscribe(subscriber_id)
    await memory.stop()
    
    print("\n✅ Shared memory demo completed")


if __name__ == "__main__":
    asyncio.run(main())