"""
Memory Manager - coordinates all memory systems with unified interface,
caching, monitoring, and advanced memory operations.
(FULLY FIXED PRODUCTION VERSION - NO AWAIT ON BOOL)
"""

from typing import Any, Dict, List, Optional, Union, Callable
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory, KnowledgeCategory, RelationType
from .vector_memory import VectorMemory
from ..utils.logger import logger
from ..utils.helpers import get_project_root, ensure_dir


class MemoryOperation(Enum):
    """Memory operation types."""

    STORE = "store"
    RETRIEVE = "retrieve"
    SEARCH = "search"
    DELETE = "delete"
    UPDATE = "update"
    LINK = "link"


class MemoryPriority(Enum):
    """Priority levels for memory operations."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MemoryOperationResult:
    """Result of a memory operation."""

    success: bool
    operation: MemoryOperation
    memory_type: str
    key: Optional[str] = None
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryStats:
    """Comprehensive memory statistics."""

    episodic_count: int = 0
    semantic_count: int = 0
    vector_count: int = 0
    total_count: int = 0
    episodic_avg_importance: float = 0.0
    semantic_avg_confidence: float = 0.0
    cache_hit_rate: float = 0.0
    operation_count: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cache_size: int = 0


class MemoryManager:
    """Unified interface for all memory systems with advanced coordination."""

    _instance = None
    _initialized = False
    _lock = threading.RLock()

    def __new__(cls):
        """Singleton pattern to prevent multiple initializations."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize memory manager (safe, non-blocking, production-ready)"""

        if MemoryManager._initialized:
            return

        self._use_mongodb = False
        self._mongo_client = None

        # -------------------------
        # SAFE MONGO INIT (NON-BLOCKING FIX)
        # -------------------------
        def _init_mongo():
            try:
                from .mongo_client import mongo_client as mc

                if mc and getattr(mc, "enabled", False):
                    if hasattr(mc, "is_connected") and mc.is_connected():
                        self._use_mongodb = True
                        self._mongo_client = mc
                        logger.info("✅ MongoDB connected (reuse)")
                        return

                    # Try async connect safely - FIXED: no await on bool
                    try:
                        loop = asyncio.get_event_loop()
                        if not loop.is_running():
                            # Run the async connect
                            future = asyncio.ensure_future(mc.connect())
                            loop.run_until_complete(future)
                        else:
                            asyncio.create_task(mc.connect())
                    except RuntimeError:
                        # Create new loop for connection
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(mc.connect())
                        new_loop.close()

                    if hasattr(mc, "is_connected") and mc.is_connected():
                        self._use_mongodb = True
                        self._mongo_client = mc
                        logger.info("✅ MongoDB connected")
                    else:
                        logger.warning("⚠ MongoDB unavailable → offline mode")

            except Exception as e:
                logger.warning(f"⚠ Mongo init failed: {e}")

        # RUN ASYNC INIT IN BACKGROUND (CRITICAL FIX)
        threading.Thread(target=_init_mongo, daemon=True).start()

        # -------------------------
        # INIT MEMORY SYSTEMS (SAFE)
        # -------------------------
        try:
            self.episodic = EpisodicMemory()
            self.semantic = SemanticMemory()
            self.vector = VectorMemory()

            # Attach mongo later if ready
            if self._use_mongodb and self._mongo_client:
                for m in (self.episodic, self.semantic, self.vector):
                    if hasattr(m, "set_mongo_client"):
                        try:
                            # FIXED: No await on non-async method
                            m.set_mongo_client(self._mongo_client)
                        except Exception:
                            pass

            logger.info("✅ Memory systems initialized")

        except Exception as e:
            logger.error(f"❌ Memory init failed: {e}")
            self.episodic = None
            self.semantic = None
            self.vector = None

        # -------------------------
        # DIRECTORIES
        # -------------------------
        self.project_root = get_project_root()
        self.memory_dir = ensure_dir(Path(self.project_root) / "data" / "memory")
        self.backup_dir = ensure_dir(Path(self.project_root) / "data" / "backups")

        # -------------------------
        # CACHE (OPTIMIZED)
        # -------------------------
        self._recent_cache = deque(maxlen=200)
        self._frequent_access_cache = {}
        self._cache_limit = 200
        self._cache_ttl = 300
        self._cache_timestamps = {}

        # -------------------------
        # STATS
        # -------------------------
        self._operation_history = deque(maxlen=1000)
        self._operation_stats = defaultdict(
            lambda: {
                "count": 0,
                "success": 0,
                "failed": 0,
                "total_time_ms": 0,
                "avg_time_ms": 0,
            }
        )

        self._error_count = 0
        self._last_error = None
        self._last_error_time = None

        # -------------------------
        # CALLBACKS
        # -------------------------
        self._operation_callbacks = []
        self._error_callbacks = []

        # -------------------------
        # ASYNC SYSTEM (SAFE)
        # -------------------------
        try:
            self._operation_queue = asyncio.Queue(maxsize=1000)
        except RuntimeError:
            # fallback if no loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._operation_queue = asyncio.Queue(maxsize=1000)

        self._worker_task = None
        self._running = False

        # -------------------------
        # QUERY CACHE
        # -------------------------
        self._query_cache = {}
        self._query_cache_ttl = 60

        # -------------------------
        # CROSS REFERENCES
        # -------------------------
        self._cross_references = defaultdict(lambda: defaultdict(list))

        # -------------------------
        # GLOBAL STATS
        # -------------------------
        self._stats = MemoryStats()
        self._start_time = datetime.now()

        # -------------------------
        # ASYNC LOCK (SAFE)
        # -------------------------
        try:
            self._async_lock = asyncio.Lock()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._async_lock = asyncio.Lock()

        self._initialized_runtime = False
        MemoryManager._initialized = True

        mode_msg = "MongoDB (async)" if self._use_mongodb else "Offline"
        logger.info(f"✅ MemoryManager initialized → {mode_msg}")

        try:
            self._log_system_status()
        except Exception:
            pass

    def _init_offline_memory(self):
        """Safe offline memory initialization with fallback protection"""

        try:
            # -------------------------
            # NORMAL INIT
            # -------------------------
            self.episodic = EpisodicMemory()
            self.semantic = SemanticMemory()
            self.vector = VectorMemory()

            logger.info("📁 Offline memory initialized")

        except Exception as e:
            logger.error(f"❌ Offline memory init failed: {e}")

            # -------------------------
            # SAFE FALLBACK (CRITICAL FIX)
            # -------------------------
            class _DummyMemory:
                def store(self, *args, **kwargs):
                    return False

                def retrieve(self, *args, **kwargs):
                    return None

                def search(self, *args, **kwargs):
                    return []

                def delete(self, *args, **kwargs):
                    return False
                
                def get_stats(self, *args, **kwargs):
                    return {}
                
                def reset(self, *args, **kwargs):
                    pass

            self.episodic = _DummyMemory()
            self.semantic = _DummyMemory()
            self.vector = _DummyMemory()

            logger.warning("⚠ Using dummy memory (minimal functionality)")

        # -------------------------
        # FINAL VALIDATION
        # -------------------------
        for name in ("episodic", "semantic", "vector"):
            if getattr(self, name, None) is None:
                logger.warning(f"⚠ {name} memory missing → assigning dummy")
                setattr(self, name, _DummyMemory())

    # #================#================#============#=============
    # INITIALIZATION & STATUS
    # #================#================#============#=============

    def _log_system_status(self):
        """Log initial system status."""
        try:
            if self.episodic and hasattr(self.episodic, "get_stats"):
                episodic_stats = self.episodic.get_stats() if self.episodic else {}
                semantic_stats = self.semantic.get_stats() if self.semantic else {}
                vector_stats = self.vector.get_vector_stats() if self.vector else {}

                logger.info(
                    f"📊 Memory Stats: Episodic={episodic_stats.get('total_episodes', 0)}, "
                    f"Semantic={semantic_stats.get('total_knowledge', 0)}, "
                    f"Vector={vector_stats.get('count', 0)}"
                )
        except Exception as e:
            logger.debug(f"Status log error: {e}")

    # #================#================#============#=============
    # CORE MEMORY OPERATIONS
    # #================#================#============#=============

    async def store(
        self,
        data: Dict[str, Any],
        memory_type: str = "episodic",
        priority: MemoryPriority = MemoryPriority.NORMAL,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryOperationResult:
        """Safe, non-blocking memory store with validation + fallback"""

        import time

        start_time = time.time()

        # -------------------------
        # VALIDATION (CRITICAL FIX)
        # -------------------------
        if not isinstance(data, dict):
            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.STORE,
                memory_type=memory_type,
                error="Invalid data type",
            )

        # SAFE KEY
        key = data.get("key") or f"{memory_type}_{int(time.time() * 1000)}"

        # COPY DATA (PREVENT MUTATION BUG)
        data = data.copy()

        try:
            # -------------------------
            # LOCK WITH TIMEOUT (CRITICAL FIX)
            # -------------------------
            try:
                await asyncio.wait_for(self._async_lock.acquire(), timeout=3)
            except asyncio.TimeoutError:
                raise Exception("Store lock timeout")

            try:
                # -------------------------
                # ROUTE MEMORY TYPE
                # -------------------------
                if memory_type == "episodic":
                    if not self.episodic:
                        raise Exception("Episodic memory unavailable")

                    # FIXED: No await on sync method
                    self.episodic.store(
                        key=key,
                        content=data.get("content", data),
                        timestamp=data.get("timestamp"),
                        context=data.get("context"),
                        importance=float(data.get("importance", 0.5)),
                        tags=tags or data.get("tags", []),
                    )

                elif memory_type == "semantic":
                    if not self.semantic:
                        raise Exception("Semantic memory unavailable")

                    # FIXED: No await on sync method
                    self.semantic.store(
                        key=key,
                        content=data.get("content", data),
                        category=data.get("category", "general"),
                        confidence=float(data.get("confidence", 0.8)),
                        source=data.get("source"),
                        tags=tags or data.get("tags", []),
                    )

                elif memory_type == "vector":
                    if not self.vector:
                        raise Exception("Vector memory unavailable")

                    if "embedding" not in data:
                        raise ValueError("Vector memory requires 'embedding'")

                    # FIXED: No await on sync method
                    self.vector.store(
                        key=key,
                        embedding=data["embedding"],
                        associated_content=data.get("content"),
                        tags=tags or data.get("tags", []),
                    )

                else:
                    raise ValueError(f"Unknown memory type: {memory_type}")

                # -------------------------
                # CACHE + REFERENCES (SAFE)
                # -------------------------
                try:
                    self._update_cache(key, data, memory_type)
                    self._update_cross_reference(key, memory_type, tags or [])
                except Exception:
                    pass

                # -------------------------
                # TRACKING
                # -------------------------
                execution_time = (time.time() - start_time) * 1000

                self._track_operation(
                    MemoryOperation.STORE, memory_type, True, execution_time
                )

                result = MemoryOperationResult(
                    success=True,
                    operation=MemoryOperation.STORE,
                    memory_type=memory_type,
                    key=key,
                    data=data,
                    execution_time_ms=execution_time,
                )

                try:
                    self._add_to_history(result)
                    self._trigger_callbacks(result)
                except Exception:
                    pass

                logger.debug(f"📝 Stored → {memory_type}:{key}")
                return result

            finally:
                try:
                    self._async_lock.release()
                except Exception:
                    pass

        # -------------------------
        # ERROR HANDLING
        # -------------------------
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self._track_operation(
                MemoryOperation.STORE, memory_type, False, execution_time
            )

            self._handle_error(e)

            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.STORE,
                memory_type=memory_type,
                key=key,
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def retrieve(
        self, key: str, memory_type: str = "episodic", use_cache: bool = True
    ) -> MemoryOperationResult:
        """Safe, fast retrieve with cache + timeout protection"""

        import time

        start_time = time.time()

        # -------------------------
        # VALIDATION
        # -------------------------
        if not key:
            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.RETRIEVE,
                memory_type=memory_type,
                error="Invalid key",
            )

        now = datetime.utcnow()
        cache_key = f"{memory_type}:{key}"

        # -------------------------
        # CACHE CHECK (FIXED TTL BUG)
        # -------------------------
        if use_cache:
            try:
                cached = self._frequent_access_cache.get(cache_key)
                cached_time = self._cache_timestamps.get(cache_key)

                if cached and cached_time:
                    if (now - cached_time).total_seconds() < self._cache_ttl:
                        self._stats.cache_hit_rate = (
                            self._stats.cache_hit_rate * 0.9 + 0.1
                        )

                        return MemoryOperationResult(
                            success=True,
                            operation=MemoryOperation.RETRIEVE,
                            memory_type=memory_type,
                            key=key,
                            data=cached.copy() if isinstance(cached, dict) else cached,
                            execution_time_ms=(time.time() - start_time) * 1000,
                        )
            except Exception:
                pass

        try:
            # -------------------------
            # LOCK WITH TIMEOUT (CRITICAL FIX)
            # -------------------------
            try:
                await asyncio.wait_for(self._async_lock.acquire(), timeout=3)
            except asyncio.TimeoutError:
                raise Exception("Retrieve lock timeout")

            try:
                # -------------------------
                # MEMORY ROUTING (FIXED: No await on sync methods)
                # -------------------------
                if memory_type == "episodic":
                    data = self.episodic.retrieve(key) if self.episodic else None
                elif memory_type == "semantic":
                    data = self.semantic.retrieve(key) if self.semantic else None
                elif memory_type == "vector":
                    data = self.vector.retrieve(key) if self.vector else None
                else:
                    raise ValueError(f"Unknown memory type: {memory_type}")

                success = data is not None
                execution_time = (time.time() - start_time) * 1000

                # -------------------------
                # CACHE STORE (SAFE)
                # -------------------------
                if success and use_cache:
                    try:
                        self._update_cache(key, data, memory_type)
                    except Exception:
                        pass

                # -------------------------
                # TRACKING
                # -------------------------
                self._track_operation(
                    MemoryOperation.RETRIEVE, memory_type, success, execution_time
                )

                result = MemoryOperationResult(
                    success=success,
                    operation=MemoryOperation.RETRIEVE,
                    memory_type=memory_type,
                    key=key,
                    data=data.copy() if isinstance(data, dict) else data,
                    execution_time_ms=execution_time,
                    error=None if success else "Not found",
                )

                try:
                    self._add_to_history(result)
                    self._trigger_callbacks(result)
                except Exception:
                    pass

                return result

            finally:
                try:
                    self._async_lock.release()
                except Exception:
                    pass

        # -------------------------
        # ERROR HANDLING
        # -------------------------
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self._track_operation(
                MemoryOperation.RETRIEVE, memory_type, False, execution_time
            )

            self._handle_error(e)

            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.RETRIEVE,
                memory_type=memory_type,
                key=key,
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def search(
        self,
        query: str,
        memory_type: str = "all",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> MemoryOperationResult:
        """Fast, non-blocking parallel search across memory types"""

        import time

        start_time = time.time()

        # -------------------------
        # VALIDATION
        # -------------------------
        if not query or not isinstance(query, str):
            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.SEARCH,
                memory_type=memory_type,
                error="Invalid query",
            )

        # LIMIT FIX (CRITICAL)
        limit = min(max(limit, 1), 50)

        results = {}

        try:
            # -------------------------
            # PARALLEL SEARCH (CRITICAL FIX)
            # -------------------------
            async def safe_search(mem, fn, name):
                try:
                    if mem:
                        # FIXED: No await on sync search methods
                        res = await asyncio.to_thread(fn, query, limit)
                        return name, res
                except Exception:
                    return name, []
                return name, []

            tasks = []

            if memory_type in ["episodic", "all"] and self.episodic:
                if hasattr(self.episodic, 'search'):
                    tasks.append(
                        safe_search(self.episodic, self.episodic.search, "episodic")
                    )

            if memory_type in ["semantic", "all"] and self.semantic:
                if hasattr(self.semantic, 'search'):
                    tasks.append(
                        safe_search(self.semantic, self.semantic.search, "semantic")
                    )

            if memory_type in ["vector", "all"]:
                # vector skipped (no embedding)
                results["vector"] = []

            # RUN PARALLEL
            if tasks:
                done = await asyncio.gather(*tasks, return_exceptions=True)

                for item in done:
                    if isinstance(item, Exception):
                        continue
                    name, res = item
                    results[name] = res if isinstance(res, list) else []

            execution_time = (time.time() - start_time) * 1000

            # -------------------------
            # TRACKING
            # -------------------------
            self._track_operation(
                MemoryOperation.SEARCH, memory_type, True, execution_time
            )

            result = MemoryOperationResult(
                success=True,
                operation=MemoryOperation.SEARCH,
                memory_type=memory_type,
                data=results,
                execution_time_ms=execution_time,
            )

            try:
                self._add_to_history(result)
            except Exception:
                pass

            return result

        # -------------------------
        # ERROR HANDLING
        # -------------------------
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self._track_operation(
                MemoryOperation.SEARCH, memory_type, False, execution_time
            )

            self._handle_error(e)

            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.SEARCH,
                memory_type=memory_type,
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def delete(
        self, key: str, memory_type: str = "episodic", cascade: bool = False
    ) -> MemoryOperationResult:
        """Safe, non-blocking delete with cache + timeout protection"""

        import time

        start_time = time.time()

        # -------------------------
        # VALIDATION
        # -------------------------
        if not key:
            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.DELETE,
                memory_type=memory_type,
                error="Invalid key",
            )

        try:
            # -------------------------
            # LOCK WITH TIMEOUT (CRITICAL FIX)
            # -------------------------
            try:
                await asyncio.wait_for(self._async_lock.acquire(), timeout=3)
            except asyncio.TimeoutError:
                raise Exception("Delete lock timeout")

            try:
                success = False

                # -------------------------
                # MEMORY ROUTING (FIXED: No await on sync methods)
                # -------------------------
                if memory_type == "episodic":
                    success = self.episodic.delete(key) if self.episodic else False

                elif memory_type == "semantic":
                    success = (
                        self.semantic.delete(key, cascade=cascade)
                        if self.semantic
                        else False
                    )

                elif memory_type == "vector":
                    success = self.vector.delete(key) if self.vector else False

                else:
                    raise ValueError(f"Unknown memory type: {memory_type}")

                # -------------------------
                # CACHE CLEANUP (SAFE)
                # -------------------------
                try:
                    cache_key = f"{memory_type}:{key}"
                    self._frequent_access_cache.pop(cache_key, None)
                    self._cache_timestamps.pop(cache_key, None)
                except Exception:
                    pass

                # -------------------------
                # TRACKING
                # -------------------------
                execution_time = (time.time() - start_time) * 1000

                self._track_operation(
                    MemoryOperation.DELETE, memory_type, success, execution_time
                )

                result = MemoryOperationResult(
                    success=success,
                    operation=MemoryOperation.DELETE,
                    memory_type=memory_type,
                    key=key,
                    execution_time_ms=execution_time,
                    error=None if success else "Not found",
                )

                try:
                    self._add_to_history(result)
                except Exception:
                    pass

                return result

            finally:
                try:
                    self._async_lock.release()
                except Exception:
                    pass

        # -------------------------
        # ERROR HANDLING
        # -------------------------
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            self._track_operation(
                MemoryOperation.DELETE, memory_type, False, execution_time
            )

            self._handle_error(e)

            return MemoryOperationResult(
                success=False,
                operation=MemoryOperation.DELETE,
                memory_type=memory_type,
                key=key,
                error=str(e),
                execution_time_ms=execution_time,
            )

    # #================#================#============#=============
    # CROSS-MEMORY OPERATIONS
    # #================#================#============#=============

    def _update_cross_reference(self, key: str, memory_type: str, tags: List[str]):
        """Update cross-memory references."""
        for tag in tags:
            self._cross_references[tag][memory_type].append(key)

    async def find_related_across_memories(
        self, key: str, source_memory: str, limit: int = 10
    ) -> Dict[str, List[Any]]:
        """Fast, parallel, deduplicated cross-memory search"""

        # -------------------------
        # INIT
        # -------------------------
        results = {"episodic": [], "semantic": [], "vector": []}

        if not key:
            return results

        limit = min(max(limit, 1), 20)

        try:
            # -------------------------
            # GET SOURCE
            # -------------------------
            source_result = await self.retrieve(key, source_memory)
            if not source_result.success:
                return results

            source_data = source_result.data or {}

            # -------------------------
            # EXTRACT TERMS (IMPROVED)
            # -------------------------
            search_terms = set()

            if isinstance(source_data, dict):
                tags = source_data.get("tags", [])
                if isinstance(tags, list):
                    search_terms.update(tags[:5])

                content = source_data.get("content")
                if isinstance(content, str):
                    search_terms.update(content.split()[:10])

            # limit terms
            search_terms = list(search_terms)[:5]

            if not search_terms:
                return results

            # -------------------------
            # PARALLEL SEARCH (CRITICAL FIX)
            # -------------------------
            async def search_task(mem_type, term):
                try:
                    res = await self.search(term, mem_type, limit)
                    if res.success:
                        return mem_type, res.data.get(mem_type, [])
                except Exception:
                    pass
                return mem_type, []

            tasks = []

            for mem_type in ["episodic", "semantic", "vector"]:
                if mem_type == source_memory:
                    continue
                for term in search_terms:
                    tasks.append(search_task(mem_type, term))

            if tasks:
                done = await asyncio.gather(*tasks, return_exceptions=True)

                # -------------------------
                # MERGE + DEDUPLICATE
                # -------------------------
                seen_keys = set()

                for item in done:
                    if isinstance(item, Exception):
                        continue
                    mem_type, items = item
                    for item_data in items:
                        try:
                            item_key = (
                                item_data.get("key") if isinstance(item_data, dict) else str(item_data)
                            )

                            if item_key and item_key not in seen_keys:
                                seen_keys.add(item_key)
                                results[mem_type].append(item_data)

                                if len(results[mem_type]) >= limit:
                                    break
                        except Exception:
                            continue

            return results

        except Exception as e:
            self._handle_error(e)
            return results

    # #================#================#============#=============
    # CACHE MANAGEMENT
    # #================#================#============#=============

    def _update_cache(self, key: str, data: Any, memory_type: str):
        """Efficient, safe cache update with fast eviction"""

        if not key:
            return

        cache_key = f"{memory_type}:{key}"
        now = datetime.utcnow()

        try:
            # -------------------------
            # SAFE STORE (COPY FIX)
            # -------------------------
            safe_data = data.copy() if isinstance(data, dict) else data

            self._frequent_access_cache[cache_key] = safe_data
            self._cache_timestamps[cache_key] = now

            # -------------------------
            # FAST EVICTION (CRITICAL FIX)
            # -------------------------
            if len(self._frequent_access_cache) > self._cache_limit:
                try:
                    # remove oldest in O(1) style using sorted keys once
                    oldest_key = min(
                        self._cache_timestamps, key=self._cache_timestamps.get
                    )
                    self._frequent_access_cache.pop(oldest_key, None)
                    self._cache_timestamps.pop(oldest_key, None)
                except Exception:
                    pass

            # -------------------------
            # RECENT CACHE (SAFE + LIGHT)
            # -------------------------
            try:
                preview = None
                if isinstance(data, dict):
                    preview = str(data.get("content", ""))[:100]
                elif data:
                    preview = str(data)[:100]

                self._recent_cache.append(
                    {
                        "key": key,
                        "memory_type": memory_type,
                        "timestamp": now.isoformat(),
                        "data_preview": preview,
                    }
                )
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Cache update error: {e}")

    def clear_cache(self):
        """Clear all caches."""
        self._recent_cache.clear()
        self._frequent_access_cache.clear()
        self._cache_timestamps.clear()
        self._query_cache.clear()
        logger.info("🗑️ Memory manager caches cleared")

    # #================#================#============#=============
    # EPISODIC MEMORY WRAPPER
    # #================#================#============#=============

    def store_event(
        self,
        key: str,
        event: Any,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store episodic event."""
        if self.episodic:
            self.episodic.store(
                key, event, context=context, importance=importance, tags=tags
            )
            self._update_cache(key, event, "episodic")

    def retrieve_event(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve episodic event."""
        return self.episodic.retrieve(key) if self.episodic else None

    def get_recent_events(
        self, hours: int = 24, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent events."""
        return self.episodic.get_recent_events(hours, limit) if self.episodic else []

    def search_events(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search episodic events."""
        return self.episodic.search(query, limit) if self.episodic else []

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get event timeline."""
        return self.episodic.get_timeline() if self.episodic else []

    def get_important_events(
        self, threshold: float = 0.7, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get important events."""
        return (
            self.episodic.get_important_events(threshold, limit)
            if self.episodic
            else []
        )

    # #================#================#============#=============
    # SEMANTIC MEMORY WRAPPER
    # #================#================#============#=============

    def store_knowledge(
        self,
        key: str,
        knowledge: Any,
        category: Union[str, KnowledgeCategory] = KnowledgeCategory.GENERAL,
        confidence: float = 0.8,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent: Optional[str] = None,
    ) -> bool:
        """Store semantic knowledge."""
        if self.semantic:
            result = self.semantic.store(
                key, knowledge, category, confidence, source, tags, parent
            )
            if result:
                self._update_cache(key, knowledge, "semantic")
            return result
        return False

    def retrieve_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve semantic knowledge."""
        return self.semantic.retrieve(key) if self.semantic else None

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search semantic knowledge."""
        return self.semantic.search(query, limit) if self.semantic else []

    def link_knowledge(
        self,
        key1: str,
        key2: str,
        relation_type: Union[RelationType, str] = RelationType.RELATED_TO,
        weight: float = 1.0,
    ) -> bool:
        """Link two knowledge items."""
        return (
            self.semantic.link_knowledge(key1, key2, relation_type, weight)
            if self.semantic
            else False
        )

    def get_related_knowledge(self, key: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Get related knowledge."""
        return self.semantic.get_related_knowledge(key, depth) if self.semantic else []

    def get_knowledge_by_category(
        self, category: Union[str, KnowledgeCategory]
    ) -> List[Dict[str, Any]]:
        """Get knowledge by category."""
        return self.semantic.search_by_category(category) if self.semantic else []

    def infer_new_knowledge(self, key: str) -> List[Any]:
        """Infer new knowledge from existing."""
        return self.semantic.infer_new_knowledge(key) if self.semantic else []

    # #================#================#============#=============
    # VECTOR MEMORY WRAPPER
    # #================#================#============#=============

    def store_embedding(
        self,
        key: str,
        embedding: List[float],
        associated_content: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store vector embedding."""
        if self.vector:
            result = self.vector.store(
                key, embedding, associated_content, tags, metadata
            )
            if result:
                self._update_cache(key, associated_content, "vector")
            return result
        return False

    def retrieve_embedding(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve embedding."""
        return self.vector.retrieve(key) if self.vector else None

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search similar vectors."""
        return (
            self.vector.search(query_vector, limit, threshold, filter_tags)
            if self.vector
            else []
        )

    def find_similar_to(self, key: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar to existing vector."""
        return self.vector.find_similar(key, limit) if self.vector else []

    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector memory statistics."""
        return self.vector.get_vector_stats() if self.vector else {}

    # #================#================#============#=============
    # MONITORING & STATISTICS
    # #================#================#============#=============

    def _track_operation(
        self,
        operation: MemoryOperation,
        memory_type: str,
        success: bool,
        execution_time_ms: float,
    ):
        """Track operation statistics."""
        key = f"{operation.value}_{memory_type}"
        stats = self._operation_stats[key]

        stats["count"] += 1
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

        stats["total_time_ms"] += execution_time_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]

        # Update global stats
        self._stats.operation_count += 1
        self._stats.avg_response_time_ms = (
            self._stats.avg_response_time_ms * (self._stats.operation_count - 1)
            + execution_time_ms
        ) / self._stats.operation_count

    def _handle_error(self, error: Exception):
        """Handle and log errors."""
        self._error_count += 1
        self._last_error = str(error)
        self._last_error_time = datetime.now()
        logger.error(f"Memory operation error: {error}")

        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _add_to_history(self, result: MemoryOperationResult):
        """Add operation to history."""
        self._operation_history.append(result)

    def _trigger_callbacks(self, result: MemoryOperationResult):
        """Trigger operation callbacks."""
        for callback in self._operation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def add_operation_callback(self, callback: Callable[[MemoryOperationResult], None]):
        """Add callback for operations."""
        self._operation_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback."""
        self._error_callbacks.append(callback)

    # #================#================#============#=============
    # STATISTICS
    # #================#================#============#=============

    def get_stats(self) -> MemoryStats:
        """Safe, fast, and accurate memory statistics collection"""

        try:
            episodic_stats = {}
            semantic_stats = {}
            vector_stats = {}

            # -------------------------
            # SAFE FETCH (CRITICAL FIX)
            # -------------------------
            try:
                if self.episodic and hasattr(self.episodic, "get_stats"):
                    episodic_stats = self.episodic.get_stats() or {}
            except Exception:
                episodic_stats = {}

            try:
                if self.semantic and hasattr(self.semantic, "get_stats"):
                    semantic_stats = self.semantic.get_stats() or {}
            except Exception:
                semantic_stats = {}

            try:
                if self.vector and hasattr(self.vector, "get_vector_stats"):
                    vector_stats = self.vector.get_vector_stats() or {}
            except Exception:
                vector_stats = {}

            # -------------------------
            # SAFE VALUE EXTRACTION
            # -------------------------
            self._stats.episodic_count = int(episodic_stats.get("total_episodes", 0))
            self._stats.semantic_count = int(semantic_stats.get("total_knowledge", 0))
            self._stats.vector_count = int(vector_stats.get("count", 0))

            self._stats.total_count = (
                self._stats.episodic_count
                + self._stats.semantic_count
                + self._stats.vector_count
            )

            self._stats.episodic_avg_importance = float(
                episodic_stats.get("avg_importance", 0)
            )

            self._stats.semantic_avg_confidence = float(
                semantic_stats.get("avg_confidence", 0)
            )

            # -------------------------
            # MEMORY USAGE (IMPROVED)
            # -------------------------
            try:
                cache_size = len(self._frequent_access_cache)
                history_size = len(self._operation_history)

                self._stats.memory_usage_mb = round(
                    (cache_size * 0.002) + (history_size * 0.0002), 4
                )
                self._stats.cache_size = cache_size
            except Exception:
                self._stats.memory_usage_mb = 0
                self._stats.cache_size = 0

        except Exception as e:
            logger.debug(f"Stats collection error: {e}")

        return self._stats

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Safe, complete, and crash-proof detailed statistics"""

        try:
            now = datetime.utcnow()
            uptime = (now - self._start_time).total_seconds()

            # -------------------------
            # SAFE STATS FETCH (CRITICAL FIX)
            # -------------------------
            try:
                episodic_stats = self.episodic.get_stats() if self.episodic else {}
            except Exception:
                episodic_stats = {}

            try:
                semantic_stats = self.semantic.get_stats() if self.semantic else {}
            except Exception:
                semantic_stats = {}

            try:
                vector_stats = self.vector.get_vector_stats() if self.vector else {}
            except Exception:
                vector_stats = {}

            # -------------------------
            # SAFE VALUES
            # -------------------------
            overview = {
                "total_memory_items": int(self._stats.total_count),
                "episodic": int(self._stats.episodic_count),
                "semantic": int(self._stats.semantic_count),
                "vector": int(self._stats.vector_count),
                "uptime_seconds": uptime,
                "uptime_hours": round(uptime / 3600, 3),
                "mongodb_enabled": bool(self._use_mongodb),
            }

            performance = {
                "total_operations": int(getattr(self._stats, "operation_count", 0)),
                "avg_response_time_ms": float(
                    getattr(self._stats, "avg_response_time_ms", 0)
                ),
                "cache_hit_rate": float(getattr(self._stats, "cache_hit_rate", 0)),
                "cache_size": len(self._frequent_access_cache),
                "recent_cache_size": len(self._recent_cache),
            }

            # -------------------------
            # SAFE OPERATIONS COPY
            # -------------------------
            try:
                operations = dict(self._operation_stats)
            except Exception:
                operations = {}

            # -------------------------
            # ERROR INFO (SAFE)
            # -------------------------
            errors = {
                "total_errors": int(self._error_count),
                "last_error": str(self._last_error) if self._last_error else None,
                "last_error_time": (
                    self._last_error_time.isoformat() if self._last_error_time else None
                ),
            }

            # -------------------------
            # MEMORY HEALTH (SAFE)
            # -------------------------
            memory_health = {
                "episodic_health": float(episodic_stats.get("avg_importance", 0)),
                "semantic_health": float(semantic_stats.get("avg_confidence", 0)),
                "vector_health": float(vector_stats.get("avg_magnitude", 0)),
            }

            return {
                "overview": overview,
                "performance": performance,
                "operations": operations,
                "errors": errors,
                "memory_health": memory_health,
            }

        except Exception as e:
            logger.warning(f"Detailed stats error: {e}")

            # -------------------------
            # FAILSAFE RESPONSE
            # -------------------------
            return {
                "overview": {},
                "performance": {},
                "operations": {},
                "errors": {"message": str(e)},
                "memory_health": {},
            }

    def get_operation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent operation history."""
        return [
            {
                "operation": r.operation.value,
                "memory_type": r.memory_type,
                "success": r.success,
                "key": r.key,
                "execution_time_ms": r.execution_time_ms,
                "timestamp": r.timestamp.isoformat(),
                "error": r.error,
            }
            for r in list(self._operation_history)[-limit:]
        ]

    # #================#================#============#=============
    # MAINTENANCE OPERATIONS
    # #================#================#============#=============

    def consolidate(self):
        """Consolidate memory statistics and optimize."""
        try:
            # Get stats from all memory systems
            episodic_stats = self.episodic.get_stats() if self.episodic else {}
            semantic_stats = self.semantic.get_stats() if self.semantic else {}
            vector_stats = self.vector.get_vector_stats() if self.vector else {}

            logger.info("📊 Memory Consolidation:")
            logger.info(
                f"   Episodic: {episodic_stats.get('total_episodes', 0)} events"
            )
            logger.info(
                f"   Semantic: {semantic_stats.get('total_knowledge', 0)} facts"
            )
            logger.info(f"   Vector: {vector_stats.get('count', 0)} embeddings")

            # Log cache status
            logger.info(f"   Cache: {len(self._frequent_access_cache)} items")
            logger.info(f"   Operations: {self._stats.operation_count} total")
            logger.info(
                f"   MongoDB Mode: {'Enabled' if self._use_mongodb else 'Disabled (Offline)'}"
            )

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    def save_to_disk(self, filepath: Optional[str] = None) -> bool:
        """Safe, efficient memory backup with size limits + serialization protection"""

        from pathlib import Path

        # -------------------------
        # FILEPATH SAFE
        # -------------------------
        try:
            if filepath is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filepath = Path(self.backup_dir) / f"memory_backup_{timestamp}.json"
            else:
                filepath = Path(filepath)

            filepath.parent.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error(f"Invalid filepath: {e}")
            return False

        try:
            # -------------------------
            # BASE STRUCTURE
            # -------------------------
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "mongodb_mode": self._use_mongodb,
                "episodic": {},
                "semantic": {},
                "vector": {},
                "stats": {},
                "metadata": {
                    "version": "2.1",
                    "backup_time": datetime.utcnow().isoformat(),
                    "total_items": int(self._stats.total_count),
                },
            }

            # -------------------------
            # SAFE STATS
            # -------------------------
            try:
                backup_data["stats"] = self.get_detailed_stats()
            except Exception:
                backup_data["stats"] = {}

            # -------------------------
            # LIMIT (CRITICAL FIX)
            # -------------------------
            max_items = 1000

            # -------------------------
            # EPISODIC (SAFE + LIMITED)
            # -------------------------
            if self.episodic and hasattr(self.episodic, "entries"):
                try:
                    for i, (key, entry) in enumerate(self.episodic.entries.items()):
                        if i >= max_items:
                            break

                        backup_data["episodic"][key] = {
                            "content": getattr(entry, "content", str(entry))[:500],
                            "importance": float(getattr(entry, "importance", 0.5)),
                            "tags": list(getattr(entry, "tags", []))[:10],
                            "timestamp": (
                                getattr(entry, "timestamp", None).isoformat()
                                if getattr(entry, "timestamp", None)
                                else None
                            ),
                        }
                except Exception as e:
                    logger.warning(f"Episodic backup partial: {e}")

            # -------------------------
            # SEMANTIC (SAFE)
            # -------------------------
            if self.semantic and hasattr(self.semantic, "knowledge_base"):
                try:
                    for i, (key, node) in enumerate(
                        self.semantic.knowledge_base.items()
                    ):
                        if i >= max_items:
                            break

                        if hasattr(node, "to_dict"):
                            backup_data["semantic"][key] = node.to_dict()
                        else:
                            backup_data["semantic"][key] = {"content": str(node)[:500]}
                except Exception as e:
                    logger.warning(f"Semantic backup partial: {e}")

            # -------------------------
            # VECTOR (SAFE)
            # -------------------------
            if self.vector and hasattr(self.vector, "vectors"):
                try:
                    for i, (key, vector) in enumerate(self.vector.vectors.items()):
                        if i >= max_items:
                            break

                        backup_data["vector"][key] = {
                            "embedding": (
                                vector.tolist()[:100]
                                if hasattr(vector, "tolist")
                                else vector
                            ),
                            "content": None,
                        }
                except Exception as e:
                    logger.warning(f"Vector backup partial: {e}")

            # -------------------------
            # WRITE FILE (SAFE)
            # -------------------------
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"💾 Backup saved → {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False

    def load_from_disk(self, filepath: Optional[str] = None) -> bool:
        """Safe, controlled memory restore from disk"""

        from pathlib import Path

        try:
            # -------------------------
            # FILE SELECTION (SAFE)
            # -------------------------
            if filepath is None:
                backups = sorted(
                    Path(self.backup_dir).glob("memory_backup_*.json"), reverse=True
                )
                if not backups:
                    logger.warning("No backup files found")
                    return False
                filepath = backups[0]

            filepath = Path(filepath)

            if not filepath.exists():
                logger.warning(f"Backup file not found: {filepath}")
                return False

            # -------------------------
            # LOAD FILE (SAFE)
            # -------------------------
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    backup_data = json.load(f)
            except Exception as e:
                logger.error(f"Invalid backup file: {e}")
                return False

            # -------------------------
            # LIMIT (CRITICAL FIX)
            # -------------------------
            max_items = 1000

            # -------------------------
            # RESTORE EPISODIC
            # -------------------------
            if self.episodic:
                try:
                    for i, (key, data) in enumerate(
                        backup_data.get("episodic", {}).items()
                    ):
                        if i >= max_items:
                            break

                        self.episodic.store(
                            key=key,
                            content=data.get("content"),
                            importance=float(data.get("importance", 0.5)),
                            tags=list(data.get("tags", [])),
                        )
                except Exception as e:
                    logger.warning(f"Episodic restore partial: {e}")

            # -------------------------
            # RESTORE SEMANTIC
            # -------------------------
            if self.semantic:
                try:
                    for i, (key, data) in enumerate(
                        backup_data.get("semantic", {}).items()
                    ):
                        if i >= max_items:
                            break

                        self.semantic.store(
                            key=key,
                            content=data.get("content"),
                            category=data.get("category", "general"),
                            confidence=float(data.get("confidence", 0.8)),
                            source=data.get("source"),
                            tags=list(data.get("tags", [])),
                        )
                except Exception as e:
                    logger.warning(f"Semantic restore partial: {e}")

            # -------------------------
            # RESTORE VECTOR
            # -------------------------
            if self.vector:
                try:
                    for i, (key, data) in enumerate(
                        backup_data.get("vector", {}).items()
                    ):
                        if i >= max_items:
                            break

                        embedding = data.get("embedding")

                        if embedding and isinstance(embedding, (list, tuple)):
                            self.vector.store(
                                key=key,
                                embedding=embedding,
                                associated_content=data.get("content"),
                            )
                except Exception as e:
                    logger.warning(f"Vector restore partial: {e}")

            logger.info(f"📂 Memory loaded safely from {filepath}")

            try:
                self._log_system_status()
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
            return False

    def clear_all(self, confirm: bool = False) -> bool:
        """Safe, complete memory wipe with validation + fallback"""

        # -------------------------
        # CONFIRMATION CHECK (CRITICAL)
        # -------------------------
        if not confirm:
            logger.warning("⚠ Clear all requires confirm=True")
            return False

        try:
            # -------------------------
            # CLEAR MEMORY SYSTEMS (SAFE)
            # -------------------------
            try:
                if self.episodic and hasattr(self.episodic, "reset"):
                    self.episodic.reset(confirm=True)
            except Exception as e:
                logger.warning(f"Episodic clear partial: {e}")

            try:
                if self.semantic and hasattr(self.semantic, "reset"):
                    self.semantic.reset(confirm=True)
            except Exception as e:
                logger.warning(f"Semantic clear partial: {e}")

            try:
                if self.vector and hasattr(self.vector, "clear"):
                    self.vector.clear(confirm=True)
            except Exception as e:
                logger.warning(f"Vector clear partial: {e}")

            # -------------------------
            # CLEAR CACHE (CRITICAL FIX)
            # -------------------------
            try:
                self._frequent_access_cache.clear()
                self._cache_timestamps.clear()
                self._recent_cache.clear()
                self._query_cache.clear()
            except Exception:
                pass

            # -------------------------
            # CLEAR INTERNAL STATE
            # -------------------------
            try:
                self._operation_history.clear()
                self._operation_stats.clear()
                self._cross_references.clear()
            except Exception:
                pass

            # -------------------------
            # RESET STATS
            # -------------------------
            self._error_count = 0
            self._last_error = None
            self._last_error_time = None
            self._stats = MemoryStats()

            logger.warning("🗑️ All memory cleared safely")
            return True

        except Exception as e:
            logger.error(f"❌ Clear all failed: {e}")
            return False

    def prune_old_memory(
        self, days: int = 30, importance_threshold: float = 0.3
    ) -> int:
        """Safe, controlled pruning with limits + protection"""

        # -------------------------
        # VALIDATION (CRITICAL FIX)
        # -------------------------
        days = max(days, 1)
        importance_threshold = min(max(importance_threshold, 0.0), 1.0)

        pruned_count = 0

        try:
            # -------------------------
            # EPISODIC PRUNE (SAFE)
            # -------------------------
            if self.episodic and hasattr(self.episodic, "clear_old_events"):
                try:
                    pruned_count += int(
                        self.episodic.clear_old_events(days, importance_threshold) or 0
                    )
                except Exception as e:
                    logger.warning(f"Episodic prune partial: {e}")

            # -------------------------
            # SEMANTIC DECAY (SAFE)
            # -------------------------
            if self.semantic:
                try:
                    if hasattr(self.semantic, "decay_knowledge"):
                        self.semantic.decay_knowledge(
                            decay_rate=0.01, days_threshold=days
                        )
                except Exception as e:
                    logger.warning(f"Semantic decay error: {e}")

                try:
                    if hasattr(self.semantic, "prune_low_confidence"):
                        pruned_count += int(
                            self.semantic.prune_low_confidence(
                                threshold=importance_threshold
                            )
                            or 0
                        )
                except Exception as e:
                    logger.warning(f"Semantic prune partial: {e}")

            # -------------------------
            # CACHE CLEANUP (CRITICAL FIX)
            # -------------------------
            try:
                # remove stale cache entries
                now = datetime.utcnow()
                to_delete = [
                    k
                    for k, t in self._cache_timestamps.items()
                    if (now - t).total_seconds() > (days * 86400)
                ]

                for k in to_delete:
                    self._frequent_access_cache.pop(k, None)
                    self._cache_timestamps.pop(k, None)
            except Exception:
                pass

            logger.info(f"✂️ Pruned {pruned_count} memories safely")
            return pruned_count

        except Exception as e:
            logger.error(f"❌ Prune failed: {e}")
            return pruned_count

    # #================#================#============#=============
    # ASYNC OPERATIONS
    # #================#================#============#=============

    async def start_worker(self):
        """Safe worker startup with duplicate protection + loop handling"""

        # -------------------------
        # PREVENT MULTIPLE WORKERS (CRITICAL FIX)
        # -------------------------
        if self._running and self._worker_task and not self._worker_task.done():
            logger.warning("⚠ Worker already running")
            return

        self._running = True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("❌ No running event loop → worker not started")
            self._running = False
            return

        try:
            # -------------------------
            # CREATE TASK (SAFE)
            # -------------------------
            self._worker_task = loop.create_task(self._worker_loop())

            # -------------------------
            # ADD ERROR HANDLER (CRITICAL FIX)
            # -------------------------
            def _handle_task_result(task):
                try:
                    task.result()
                except Exception as e:
                    logger.error(f"❌ Worker crashed: {e}")
                    self._running = False

            self._worker_task.add_done_callback(_handle_task_result)

            logger.info("🚀 Memory worker started")

        except Exception as e:
            self._running = False
            logger.error(f"❌ Failed to start worker: {e}")

    async def stop_worker(self):
        """Safe worker shutdown with timeout + cleanup"""

        # -------------------------
        # CHECK RUNNING
        # -------------------------
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            try:
                # -------------------------
                # CANCEL TASK (SAFE)
                # -------------------------
                self._worker_task.cancel()

                try:
                    # -------------------------
                    # TIMEOUT WAIT (CRITICAL FIX)
                    # -------------------------
                    await asyncio.wait_for(self._worker_task, timeout=3)
                except asyncio.TimeoutError:
                    logger.warning("⚠ Worker did not stop in time")
                except asyncio.CancelledError:
                    pass

            except Exception as e:
                logger.warning(f"Worker stop error: {e}")

            finally:
                self._worker_task = None

        logger.info("🛑 Memory worker stopped safely")

    async def _worker_loop(self):
        """Robust async worker loop with safe queue handling"""

        while self._running:
            try:
                operation = None

                # -------------------------
                # GET TASK (WITH TIMEOUT)
                # -------------------------
                try:
                    operation = await asyncio.wait_for(
                        self._operation_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # -------------------------
                # PROCESS TASK (SAFE)
                # -------------------------
                try:
                    await self._process_operation(operation)
                except Exception as exc:
                    logger.error(f"Worker task error: {exc}")

                # -------------------------
                # MARK TASK DONE (CRITICAL FIX)
                # -------------------------
                try:
                    self._operation_queue.task_done()
                except Exception:
                    pass

            # -------------------------
            # CLEAN EXIT
            # -------------------------
            except asyncio.CancelledError:
                break

            # -------------------------
            # LOOP ERROR PROTECTION
            # -------------------------
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

                # prevent tight crash loop
                await asyncio.sleep(0.2)

        logger.info("🛑 Worker loop stopped")

    async def _process_operation(self, operation: Dict[str, Any]):
        """Safe operation processor with validation + error protection"""

        if not isinstance(operation, dict):
            logger.warning("Invalid operation format")
            return

        op_type = operation.get("type")
        kwargs = operation.get("kwargs", {}) or {}

        try:
            # -------------------------
            # VALIDATE OP TYPE (CRITICAL FIX)
            # -------------------------
            if not op_type:
                logger.warning("Operation missing type")
                return

            # -------------------------
            # ROUTING (SAFE)
            # -------------------------
            if op_type == "store":
                await self.store(**kwargs)

            elif op_type == "retrieve":
                await self.retrieve(**kwargs)

            elif op_type == "search":
                await self.search(**kwargs)

            elif op_type == "delete":
                await self.delete(**kwargs)

            else:
                logger.warning(f"Unknown operation: {op_type}")

        # -------------------------
        # ERROR HANDLING (CRITICAL)
        # -------------------------
        except Exception as e:
            logger.error(f"❌ Operation failed ({op_type}): {e}")
            self._handle_error(e)

    async def queue_operation(self, operation: str, **kwargs):
        """Safe queue with validation + overflow protection"""

        # -------------------------
        # VALIDATION (CRITICAL FIX)
        # -------------------------
        if not operation or not isinstance(operation, str):
            logger.warning("Invalid operation type")
            return False

        if not hasattr(self, "_operation_queue") or self._operation_queue is None:
            logger.error("Operation queue not initialized")
            return False

        op_data = {
            "type": operation,
            "kwargs": kwargs or {},
            "queued_at": datetime.utcnow().isoformat(),
        }

        try:
            # -------------------------
            # NON-BLOCKING PUT (CRITICAL FIX)
            # -------------------------
            try:
                self._operation_queue.put_nowait(op_data)
            except asyncio.QueueFull:
                logger.warning("⚠ Queue full → dropping oldest task")

                try:
                    # remove one old item to make space
                    _ = self._operation_queue.get_nowait()
                    self._operation_queue.put_nowait(op_data)
                except Exception:
                    logger.error("Queue recovery failed")
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Queue operation failed: {e}")
            return False

    # #================#================#============#=============
    # UTILITY METHODS
    # #================#================#============#=============

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics in one dictionary."""
        return self.get_detailed_stats()

    def get_memory_usage(self) -> Dict[str, float]:
        """Accurate and safe memory usage estimation"""

        try:
            # -------------------------
            # EPISODIC (SAFE)
            # -------------------------
            episodic_size = 0.0
            if self.episodic and hasattr(self.episodic, "entries"):
                try:
                    episodic_size = len(self.episodic.entries) * 0.001
                except Exception:
                    episodic_size = 0.0

            # -------------------------
            # SEMANTIC (SAFE)
            # -------------------------
            semantic_size = 0.0
            if self.semantic and hasattr(self.semantic, "knowledge_base"):
                try:
                    semantic_size = len(self.semantic.knowledge_base) * 0.002
                except Exception:
                    semantic_size = 0.0

            # -------------------------
            # VECTOR (FIXED CALCULATION)
            # -------------------------
            vector_size = 0.0
            if self.vector and hasattr(self.vector, "vectors"):
                try:
                    dim = getattr(self.vector, "embedding_dim", 128)
                    count = len(self.vector.vectors)

                    # bytes → MB
                    vector_size = (count * dim * 4) / (1024 * 1024)
                except Exception:
                    vector_size = 0.0

            # -------------------------
            # CACHE (SAFE)
            # -------------------------
            cache_size = 0.0
            try:
                cache_size = len(self._frequent_access_cache) * 0.0005
            except Exception:
                cache_size = 0.0

            # -------------------------
            # TOTAL (FIXED)
            # -------------------------
            total = round(episodic_size + semantic_size + vector_size + cache_size, 4)

            return {
                "episodic_mb": round(episodic_size, 4),
                "semantic_mb": round(semantic_size, 4),
                "vector_mb": round(vector_size, 4),
                "cache_mb": round(cache_size, 4),
                "total_mb": total,
            }

        except Exception as e:
            logger.warning(f"Memory usage calc error: {e}")
            return {
                "episodic_mb": 0.0,
                "semantic_mb": 0.0,
                "vector_mb": 0.0,
                "cache_mb": 0.0,
                "total_mb": 0.0,
            }

    def get_cross_references(self, tag: str) -> Dict[str, List[str]]:
        """Get cross-references for a tag."""
        return dict(self._cross_references.get(tag, {}))

    def get_system_health(self) -> Dict[str, Any]:
        """Safe, accurate system health evaluation"""

        try:
            stats = self.get_stats()

            health_score = 100
            warnings = []

            # -------------------------
            # SAFE VALUE EXTRACTION
            # -------------------------
            total = int(getattr(stats, "total_count", 0))
            response_time = float(getattr(stats, "avg_response_time_ms", 0))
            cache_hit = float(getattr(stats, "cache_hit_rate", 0))
            errors = int(self._error_count)

            # -------------------------
            # MEMORY CHECK
            # -------------------------
            if total == 0:
                health_score -= 20
                warnings.append("No memories stored")

            # -------------------------
            # PERFORMANCE CHECK
            # -------------------------
            if response_time > 100:
                health_score -= 10
                warnings.append(f"High response time: {round(response_time,1)}ms")

            # -------------------------
            # ERROR CHECK (IMPROVED)
            # -------------------------
            if errors > 100:
                health_score -= 15
                warnings.append(f"High error count: {errors}")
            elif errors > 20:
                health_score -= 5
                warnings.append(f"Moderate errors: {errors}")

            # -------------------------
            # CACHE HEALTH
            # -------------------------
            if cache_hit < 0.1:
                health_score -= 5
                warnings.append("Low cache hit rate")

            # -------------------------
            # CLAMP SCORE (CRITICAL FIX)
            # -------------------------
            health_score = max(0, min(100, health_score))

            # -------------------------
            # STATUS
            # -------------------------
            if health_score >= 70:
                status = "healthy"
            elif health_score >= 40:
                status = "degraded"
            else:
                status = "critical"

            # -------------------------
            # COMPONENT STATUS (SAFE)
            # -------------------------
            components = {
                "episodic": (
                    "ok" if getattr(stats, "episodic_count", 0) > 0 else "empty"
                ),
                "semantic": (
                    "ok" if getattr(stats, "semantic_count", 0) > 0 else "empty"
                ),
                "vector": "ok" if getattr(stats, "vector_count", 0) > 0 else "empty",
                "cache": "ok" if cache_hit > 0.1 else "low",
                "mongodb": (
                    "enabled" if getattr(self, "_use_mongodb", False) else "disabled"
                ),
            }

            # -------------------------
            # RECOMMENDATIONS (SAFE)
            # -------------------------
            try:
                recommendations = self._generate_recommendations()
            except Exception:
                recommendations = []

            return {
                "health_score": health_score,
                "status": status,
                "warnings": warnings,
                "components": components,
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.warning(f"Health check error: {e}")

            return {
                "health_score": 0,
                "status": "critical",
                "warnings": [str(e)],
                "components": {},
                "recommendations": [],
            }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on system state."""
        recommendations = []
        stats = self.get_stats()

        if stats.total_count == 0:
            recommendations.append("Start storing memories to enable learning")

        if stats.cache_hit_rate < 0.1 and stats.operation_count > 100:
            recommendations.append(
                "Consider increasing cache size for better performance"
            )

        if stats.avg_response_time_ms > 100:
            recommendations.append("Optimize memory queries or upgrade hardware")

        if self._error_count > 50:
            recommendations.append("Check memory system error logs")

        if stats.episodic_count > 10000:
            recommendations.append("Run prune_old_memory() to clean old events")

        return recommendations

    def is_mongodb_enabled(self) -> bool:
        """Check if MongoDB is enabled."""
        return self._use_mongodb

    def initialize(self, *args, **kwargs) -> bool:
        """
        Initialize MemoryManager safely.
        Production-grade + async-safe + idempotent.
        """

        import asyncio
        import threading
        import gc
        import time

        try:
            # ------------------------------------------------
            # PREVENT DOUBLE INITIALIZATION
            # ------------------------------------------------
            if getattr(self, "_initialized_runtime", False):
                logger.debug("MemoryManager already initialized")
                return True

            # ------------------------------------------------
            # THREAD-SAFE INIT LOCK
            # ------------------------------------------------
            if not hasattr(self, "_runtime_init_lock"):
                self._runtime_init_lock = threading.RLock()

            with self._runtime_init_lock:

                # Double-check after lock
                if getattr(self, "_initialized_runtime", False):
                    return True

                start_time = time.monotonic()

                # ------------------------------------------------
                # SAFE DEFAULTS
                # ------------------------------------------------
                self._running = False
                self._worker_task = None
                self._shutdown_requested = False

                # ------------------------------------------------
                # VALIDATE MEMORY SYSTEMS
                # ------------------------------------------------
                if getattr(self, "episodic", None) is None:
                    logger.warning("⚠ Episodic memory missing")

                if getattr(self, "semantic", None) is None:
                    logger.warning("⚠ Semantic memory missing")

                if getattr(self, "vector", None) is None:
                    logger.warning("⚠ Vector memory missing")

                # ------------------------------------------------
                # SAFE ASYNC OBJECTS
                # ------------------------------------------------
                try:
                    asyncio.get_running_loop()

                    # Queue
                    if (
                        not hasattr(self, "_operation_queue")
                        or self._operation_queue is None
                    ):
                        self._operation_queue = asyncio.Queue(maxsize=1000)

                    # Lock
                    if (
                        not hasattr(self, "_async_lock")
                        or self._async_lock is None
                    ):
                        self._async_lock = asyncio.Lock()

                except RuntimeError:
                    # No running loop yet → defer async creation safely
                    self._operation_queue = None
                    self._async_lock = None

                # ------------------------------------------------
                # CACHE VALIDATION
                # ------------------------------------------------
                if not hasattr(self, "_frequent_access_cache"):
                    self._frequent_access_cache = {}

                if not hasattr(self, "_cache_timestamps"):
                    self._cache_timestamps = {}

                if not hasattr(self, "_recent_cache"):
                    self._recent_cache = deque(maxlen=200)

                if not hasattr(self, "_query_cache"):
                    self._query_cache = {}

                # ------------------------------------------------
                # CALLBACK SAFETY
                # ------------------------------------------------
                if not hasattr(self, "_operation_callbacks"):
                    self._operation_callbacks = []

                if not hasattr(self, "_error_callbacks"):
                    self._error_callbacks = []

                # ------------------------------------------------
                # STATS SAFETY
                # ------------------------------------------------
                if not hasattr(self, "_stats") or self._stats is None:
                    self._stats = MemoryStats()

                if not hasattr(self, "_operation_history"):
                    self._operation_history = deque(maxlen=1000)

                # ------------------------------------------------
                # MEMORY OPTIMIZATION
                # ------------------------------------------------
                try:
                    gc.collect()
                except Exception:
                    pass

                # ------------------------------------------------
                # SAFE WORKER START
                # ------------------------------------------------
                try:
                    loop = asyncio.get_running_loop()

                    if (
                        hasattr(self, "start_worker")
                        and callable(self.start_worker)
                    ):
                        try:
                            loop.create_task(self.start_worker())
                        except Exception as e:
                            logger.debug(f"Worker auto-start skipped: {e}")

                except RuntimeError:
                    # No event loop running yet
                    pass

                # ------------------------------------------------
                # FINALIZE
                # ------------------------------------------------
                self._initialized_runtime = True

                init_time = round(
                    (time.monotonic() - start_time) * 1000,
                    2,
                )

                logger.info(
                    f"🧠 MemoryManager initialized successfully "
                    f"({init_time} ms)"
                )

                return True

        # ------------------------------------------------
        # CLEAN CANCELLATION
        # ------------------------------------------------
        except asyncio.CancelledError:

            logger.warning("⚠ MemoryManager initialization cancelled")

            try:
                self._initialized_runtime = False
            except Exception:
                pass

            return False

        # ------------------------------------------------
        # FULL ERROR ISOLATION
        # ------------------------------------------------
        except Exception as e:

            try:
                logger.error(
                    f"❌ MemoryManager initialize failed: {e}",
                    exc_info=True,
                )
            except Exception:
                pass

            # ------------------------------------------------
            # SAFE RESET
            # ------------------------------------------------
            try:
                self._initialized_runtime = False
                self._running = False
            except Exception:
                pass

            return False

    def start(self) -> bool:
        """
        Start MemoryManager background systems.
        """
        try:
            if not getattr(self, "_initialized_runtime", False):
                self.initialize()

            if self._running:
                logger.warning("⚠ MemoryManager already running")
                return True

            self._running = True

            # Start async worker safely
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.start_worker())
                else:
                    loop.run_until_complete(self.start_worker())
            except Exception as e:
                logger.warning(f"⚠ Worker start fallback: {e}")

            logger.info("🚀 MemoryManager started")
            return True

        except Exception as e:
            logger.error(f"❌ MemoryManager start failed: {e}")
            self._running = False
            return False

    def stop(self) -> bool:
        """
        Stop MemoryManager safely.
        """
        try:
            if not self._running:
                return True

            self._running = False

            # Stop async worker safely
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.stop_worker())
                else:
                    loop.run_until_complete(self.stop_worker())
            except Exception as e:
                logger.warning(f"⚠ Worker stop fallback: {e}")

            logger.info("🛑 MemoryManager stopped")
            return True

        except Exception as e:
            logger.error(f"❌ MemoryManager stop failed: {e}")
            return False


# #================#================#============#=============
# GLOBAL INSTANCE
# #================#================#============#=============

_memory_manager_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Return the singleton memory manager without constructing it at import."""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance


class _LazyMemoryManager:
    """Proxy that keeps the old memory_manager API lazy and import-safe."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_memory_manager(), name)


memory_manager = _LazyMemoryManager()

# #================#================#============#=============
# CONVENIENCE FUNCTIONS
# #================#================#============#=============


async def quick_store(
    content: Any, memory_type: str = "episodic", **kwargs
) -> MemoryOperationResult:
    """Quick store helper."""
    return await memory_manager.store(
        {"content": content, **kwargs}, memory_type=memory_type
    )


async def quick_retrieve(key: str, memory_type: str = "episodic") -> Optional[Any]:
    """Quick retrieve helper."""
    result = await memory_manager.retrieve(key, memory_type)
    return result.data if result.success else None


async def quick_search(
    query: str, memory_type: str = "all", limit: int = 5
) -> List[Any]:
    """Quick search helper."""
    result = await memory_manager.search(query, memory_type, limit)
    return result.data if result.success else {}


__all__ = [
    "MemoryManager",
    "MemoryOperation",
    "MemoryPriority",
    "MemoryOperationResult",
    "MemoryStats",
    "memory_manager",
    "get_memory_manager",
    "quick_store",
    "quick_retrieve",
    "quick_search",
]