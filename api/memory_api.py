"""Memory API - RESTful API for memory operations with MongoDB integration."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import asyncio
import uuid
import hashlib
import json
from enum import Enum
from pathlib import Path

from ..core.memory import memory_manager
from ..core.memory.mongo_client import mongo_client, get_metrics_collection
from ..core.utils.logger import logger

# =========================
# ENUMS AND CONSTANTS
# =========================


class MemoryType(str, Enum):
    """Types of memory storage."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    VECTOR = "vector"
    WORKING = "working"
    PROCEDURAL = "procedural"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class Importance(str, Enum):
    """Importance levels for memories."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetrievalStrategy(str, Enum):
    """Strategies for memory retrieval."""

    RECENCY = "recency"
    RELEVANCE = "relevance"
    IMPORTANCE = "importance"
    HYBRID = "hybrid"


# =========================
# REQUEST/RESPONSE MODELS
# =========================


class MemoryStoreRequest(BaseModel):
    """Request to store in memory."""

    key: str = Field(..., description="Unique memory key", min_length=1, max_length=255)
    content: Any = Field(..., description="Memory content")
    memory_type: MemoryType = Field(MemoryType.SEMANTIC, description="Type of memory")
    importance: float = Field(0.5, description="Importance score", ge=0.0, le=1.0)
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    ttl_seconds: Optional[int] = Field(
        None, description="Time-to-live in seconds", ge=60, le=31536000
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

    @validator("key")
    def validate_key(cls, v):
        if not v or not v.strip():
            raise ValueError("Key cannot be empty")
        return v.strip()

    @validator("importance")
    def validate_importance(cls, v):
        return max(0.0, min(1.0, v))


class MemoryStoreResponse(BaseModel):
    """Response from memory storage."""

    status: str
    key: str
    memory_id: str
    memory_type: MemoryType
    timestamp: str
    expires_at: Optional[str] = None


class MemoryRetrieveRequest(BaseModel):
    """Request to retrieve from memory."""

    key: str
    memory_type: MemoryType = MemoryType.SEMANTIC


class MemoryRetrieveResponse(BaseModel):
    """Response from memory retrieval."""

    key: str
    content: Any
    memory_type: str
    importance: float
    tags: Optional[List[str]]
    metadata: Dict[str, Any]
    created_at: str
    last_accessed: str
    access_count: int


class MemorySearchRequest(BaseModel):
    """Request to search memory."""

    query: str = Field(..., description="Search query", min_length=1)
    memory_type: MemoryType = Field(
        MemoryType.SEMANTIC, description="Type of memory to search"
    )
    limit: int = Field(10, description="Maximum results", ge=1, le=100)
    offset: int = Field(0, description="Pagination offset", ge=0)
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    min_importance: float = Field(
        0.0, description="Minimum importance score", ge=0.0, le=1.0
    )
    time_range_hours: Optional[int] = Field(
        None, description="Time range in hours", ge=1, le=8760
    )
    strategy: RetrievalStrategy = Field(
        RetrievalStrategy.HYBRID, description="Retrieval strategy"
    )


class MemorySearchResponse(BaseModel):
    """Response from memory search."""

    results: List[Dict[str, Any]]
    total: int
    query: str
    memory_type: str
    search_time_ms: float
    strategy: str


class MemoryBatchStoreRequest(BaseModel):
    """Request for batch memory storage."""

    memories: List[MemoryStoreRequest] = Field(
        ..., description="List of memories to store", max_items=1000
    )
    parallel: bool = Field(True, description="Process in parallel")


class MemoryBatchStoreResponse(BaseModel):
    """Response for batch memory storage."""

    stored: List[MemoryStoreResponse]
    failed: List[Dict[str, Any]]
    total: int
    successful: int
    failed_count: int
    batch_id: str


class MemoryDeleteRequest(BaseModel):
    """Request to delete memory."""

    key: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    force: bool = Field(False, description="Force delete even if important")


class MemoryUpdateRequest(BaseModel):
    """Request to update memory."""

    key: str
    content: Optional[Any] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    memory_type: MemoryType = MemoryType.SEMANTIC


class MemoryStats(BaseModel):
    """Memory statistics."""

    total_memories: int
    by_type: Dict[str, int]
    total_size_bytes: int
    average_importance: float
    most_accessed: List[Dict[str, Any]]
    recent_memories: int
    expired_memories: int
    cache_hit_rate: float


class MemoryConsolidationRequest(BaseModel):
    """Request for memory consolidation."""

    memory_type: MemoryType = MemoryType.SEMANTIC
    min_importance: float = Field(0.7, ge=0.0, le=1.0)
    max_age_days: int = Field(30, ge=1, le=365)


class MemoryExportRequest(BaseModel):
    """Request to export memories."""

    memory_type: Optional[MemoryType] = None
    format: str = Field("json", description="Export format: json, csv")
    include_metadata: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


# =========================
# MEMORY CACHE
# =========================


class MemoryCache:
    """LRU cache for frequently accessed memories."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            cached = self._cache[key]
            cache_time = cached.get("timestamp")

            if cache_time and (datetime.now() - cache_time).seconds < self.ttl_seconds:
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                return cached.get("content")
            else:
                # Remove expired
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

        self._misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set item in cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = {"content": value, "timestamp": datetime.now()}

        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def delete(self, key: str):
        """Delete from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


# =========================
# MAIN API CLASS
# =========================


class MemoryApi:
    """Production-ready API for memory operations with caching and MongoDB integration."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize memory API.

        Args:
            data_dir: Directory for storing memory data
        """
        self.data_dir = data_dir or Path("memory_data")
        self.data_dir.mkdir(exist_ok=True)

        self.manager = memory_manager
        self.request_count = 0
        self._cache = MemoryCache()

        # Request tracking
        self._request_history: List[Dict[str, Any]] = []
        self._max_history_size = 10000

        # Metrics collection
        self.metrics_collection = get_metrics_collection()

        # Memory index file
        self._index_file = self.data_dir / "memory_index.json"

        # Load configuration
        self._load_configuration()

        # Load memory index
        self._load_index()

        logger.info("✅ Memory API initialized with MongoDB integration")

    def _load_configuration(self):
        """Load configuration from environment or database."""
        self.config = {
            "default_memory_type": "semantic",
            "max_memory_size_mb": 100,
            "enable_cache": True,
            "cache_size": 1000,
            "cache_ttl_seconds": 300,
            "enable_metrics": True,
            "enable_audit_log": True,
            "auto_consolidate": True,
            "consolidation_interval_hours": 24,
            "max_tags_per_memory": 20,
            "enable_vector_search": True,
            "vector_dimension": 1536,
            "similarity_threshold": 0.7,
        }

        # Try to load from MongoDB
        if mongo_client.enabled:
            try:
                config_collection = mongo_client.get_collection("config")
                if config_collection is not None:
                    db_config = config_collection.find_one({"type": "memory_api"})
                    if db_config:
                        self.config.update(db_config.get("config", {}))
                        logger.info("Loaded Memory API configuration from database")
            except Exception as e:
                logger.warning(f"Failed to load config from MongoDB: {e}")

        # Update cache settings
        if self.config["enable_cache"]:
            self._cache = MemoryCache(
                max_size=self.config["cache_size"],
                ttl_seconds=self.config["cache_ttl_seconds"],
            )

    def _load_index(self):
        """Load memory index from disk."""
        try:
            if self._index_file.exists():
                with open(self._index_file, "r") as f:
                    index_data = json.load(f)
                    logger.info(f"Loaded memory index with {len(index_data)} entries")
        except Exception as e:
            logger.warning(f"Failed to load memory index: {e}")

    def _save_index(self):
        """Save memory index to disk."""
        try:
            # Get index from manager if available
            if hasattr(self.manager, "get_index"):
                index = self.manager.get_index()
                with open(self._index_file, "w") as f:
                    json.dump(index, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory index: {e}")

    def _generate_memory_id(self, key: str, memory_type: MemoryType) -> str:
        """Generate unique memory ID."""
        unique_string = f"{key}_{memory_type.value}_{datetime.now().isoformat()}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]

    async def _log_metrics(
        self,
        operation: str,
        memory_type: str,
        duration_ms: float,
        success: bool,
        size_bytes: int = 0,
        error: Optional[str] = None,
    ):
        """Log metrics to MongoDB."""
        if not self.config["enable_metrics"]:
            return

        if not mongo_client.enabled or self.metrics_collection is None:
            return

        try:
            metric = {
                "timestamp": datetime.now(),
                "operation": f"memory_{operation}",
                "memory_type": memory_type,
                "duration_ms": duration_ms,
                "success": success,
                "size_bytes": size_bytes,
                "error": error,
            }

            from core.utils.task_registry import create_task

            create_task(
                self.metrics_collection.insert_one(metric), name="memoryapi.log_metric"
            )

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    async def _log_audit(
        self,
        action: str,
        key: str,
        memory_type: str,
        user_id: Optional[str],
        success: bool,
    ):
        """Log audit trail to MongoDB."""
        if not self.config["enable_audit_log"]:
            return

        audit_collection = mongo_client.get_collection("audit_log")
        if not mongo_client.enabled or audit_collection is None:
            return

        try:
            audit_entry = {
                "timestamp": datetime.now(),
                "action": f"memory_{action}",
                "key": key,
                "memory_type": memory_type,
                "user_id": user_id,
                "success": success,
            }

            from core.utils.task_registry import create_task

            create_task(
                audit_collection.insert_one(audit_entry), name="memoryapi.log_audit"
            )

        except Exception as e:
            logger.error(f"Failed to log audit: {e}")

    def _trim_history(self):
        """Trim request history to max size."""
        if len(self._request_history) > self._max_history_size:
            self._request_history = self._request_history[-self._max_history_size :]

    # =========================
    # STORE MEMORY
    # =========================

    async def store(self, request: MemoryStoreRequest) -> MemoryStoreResponse:
        """Store data in memory with caching."""
        start_time = datetime.now()
        memory_id = self._generate_memory_id(request.key, request.memory_type)

        try:
            self.request_count += 1

            # Prepare memory data
            memory_data = {
                "id": memory_id,
                "key": request.key,
                "content": request.content,
                "type": request.memory_type.value,
                "importance": request.importance,
                "tags": request.tags or [],
                "metadata": request.metadata,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
            }

            # Store based on memory type
            if request.memory_type == MemoryType.EPISODIC:
                if hasattr(self.manager, "store_event"):
                    self.manager.store_event(
                        request.key,
                        request.content,
                        request.importance,
                        request.tags,
                        metadata=request.metadata,
                    )
                else:
                    # Fallback to generic storage
                    self.manager.store(
                        request.key, memory_data, request.memory_type.value
                    )

            elif request.memory_type == MemoryType.SEMANTIC:
                if hasattr(self.manager, "store_knowledge"):
                    self.manager.store_knowledge(
                        request.key,
                        request.content,
                        tags=request.tags,
                        metadata=request.metadata,
                    )
                else:
                    self.manager.store(
                        request.key, memory_data, request.memory_type.value
                    )

            elif request.memory_type == MemoryType.VECTOR:
                if hasattr(self.manager, "store_vector"):
                    self.manager.store_vector(
                        request.key,
                        request.content,
                        request.importance,
                        tags=request.tags,
                    )
                else:
                    self.manager.store(
                        request.key, memory_data, request.memory_type.value
                    )

            else:
                # Generic storage
                self.manager.store(request.key, memory_data, request.memory_type.value)

            # Update cache
            if self.config["enable_cache"]:
                self._cache.set(request.key, memory_data)

            # Save index
            self._save_index()

            # Calculate expiration
            expires_at = None
            if request.ttl_seconds:
                expires_at = (
                    datetime.now() + timedelta(seconds=request.ttl_seconds)
                ).isoformat()

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Log metrics
            await self._log_metrics(
                "store", request.memory_type.value, duration_ms, True
            )
            await self._log_audit(
                "store", request.key, request.memory_type.value, request.user_id, True
            )

            # Update history
            self._request_history.append(
                {
                    "operation": "store",
                    "key": request.key,
                    "type": request.memory_type.value,
                    "timestamp": datetime.now(),
                    "success": True,
                }
            )
            self._trim_history()

            logger.info(f"Memory stored: {request.key} ({request.memory_type.value})")

            return MemoryStoreResponse(
                status="success",
                key=request.key,
                memory_id=memory_id,
                memory_type=request.memory_type,
                timestamp=datetime.now().isoformat(),
                expires_at=expires_at,
            )

        except Exception as e:
            logger.error(f"Memory storage error: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics(
                "store", request.memory_type.value, duration_ms, False, error=str(e)
            )

            self._request_history.append(
                {
                    "operation": "store",
                    "key": request.key,
                    "type": request.memory_type.value,
                    "timestamp": datetime.now(),
                    "success": False,
                    "error": str(e),
                }
            )
            self._trim_history()

            raise

    # =========================
    # RETRIEVE MEMORY
    # =========================

    async def retrieve(
        self, request: MemoryRetrieveRequest
    ) -> Optional[MemoryRetrieveResponse]:
        """Retrieve from memory with caching."""
        start_time = datetime.now()

        try:
            self.request_count += 1

            # Check cache first
            if self.config["enable_cache"]:
                cached = self._cache.get(request.key)
                if cached:
                    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                    await self._log_metrics(
                        "retrieve", request.memory_type.value, duration_ms, True
                    )

                    logger.info(f"Memory retrieved from cache: {request.key}")
                    return MemoryRetrieveResponse(**cached)

            # Retrieve from manager
            content = None

            if request.memory_type == MemoryType.EPISODIC:
                if hasattr(self.manager, "retrieve_event"):
                    content = self.manager.retrieve_event(request.key)
            elif request.memory_type == MemoryType.SEMANTIC:
                if hasattr(self.manager, "retrieve_knowledge"):
                    content = self.manager.retrieve_knowledge(request.key)
            else:
                content = self.manager.retrieve(request.key, request.memory_type.value)

            if content:
                # Prepare response
                response_data = {
                    "key": request.key,
                    "content": (
                        content.get("content", content)
                        if isinstance(content, dict)
                        else content
                    ),
                    "memory_type": request.memory_type.value,
                    "importance": (
                        content.get("importance", 0.5)
                        if isinstance(content, dict)
                        else 0.5
                    ),
                    "tags": (
                        content.get("tags", []) if isinstance(content, dict) else []
                    ),
                    "metadata": (
                        content.get("metadata", {}) if isinstance(content, dict) else {}
                    ),
                    "created_at": (
                        content.get("created_at", datetime.now().isoformat())
                        if isinstance(content, dict)
                        else datetime.now().isoformat()
                    ),
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": (
                        content.get("access_count", 0) + 1
                        if isinstance(content, dict)
                        else 1
                    ),
                }

                response = MemoryRetrieveResponse(**response_data)

                # Update cache
                if self.config["enable_cache"]:
                    self._cache.set(request.key, response.dict())

                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                await self._log_metrics(
                    "retrieve", request.memory_type.value, duration_ms, True
                )
                await self._log_audit(
                    "retrieve", request.key, request.memory_type.value, None, True
                )

                logger.info(f"Memory retrieved: {request.key}")
                return response

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics(
                "retrieve", request.memory_type.value, duration_ms, True
            )

            return None

        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics(
                "retrieve", request.memory_type.value, duration_ms, False, error=str(e)
            )
            raise

    # =========================
    # SEARCH MEMORY
    # =========================

    async def search(self, request: MemorySearchRequest) -> MemorySearchResponse:
        """Search memory with advanced filtering."""
        start_time = datetime.now()

        try:
            self.request_count += 1

            # Prepare search filters
            filters = {}
            if request.tags:
                filters["tags"] = {"$in": request.tags}
            if request.min_importance > 0:
                filters["importance"] = {"$gte": request.min_importance}
            if request.time_range_hours:
                cutoff_time = datetime.now() - timedelta(hours=request.time_range_hours)
                filters["created_at"] = {"$gte": cutoff_time.isoformat()}

            # Search based on memory type
            results = []
            total = 0

            if request.memory_type == MemoryType.EPISODIC:
                if hasattr(self.manager, "search_events"):
                    results = self.manager.search_events(request.query, request.limit)
                    total = len(results)
            elif request.memory_type == MemoryType.SEMANTIC:
                if hasattr(self.manager, "search_knowledge"):
                    results = self.manager.search_knowledge(
                        request.query, request.limit
                    )
                    total = len(results)
            elif request.memory_type == MemoryType.VECTOR:
                if hasattr(self.manager, "vector_search"):
                    results = self.manager.vector_search(
                        request.query,
                        request.limit,
                        similarity_threshold=self.config["similarity_threshold"],
                    )
                    total = len(results)
            else:
                # Generic search
                results = self.manager.search(
                    request.query, request.memory_type.value, request.limit
                )
                total = len(results)

            # Apply pagination
            if request.offset > 0:
                results = results[request.offset : request.offset + request.limit]
            else:
                results = results[: request.limit]

            # Apply sorting based on strategy
            if request.strategy == RetrievalStrategy.RECENCY:
                results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            elif request.strategy == RetrievalStrategy.IMPORTANCE:
                results.sort(key=lambda x: x.get("importance", 0), reverse=True)

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            await self._log_metrics(
                "search", request.memory_type.value, duration_ms, True
            )

            logger.info(f"Memory search completed: {len(results)} results")

            return MemorySearchResponse(
                results=results,
                total=total,
                query=request.query,
                memory_type=request.memory_type.value,
                search_time_ms=duration_ms,
                strategy=request.strategy.value,
            )

        except Exception as e:
            logger.error(f"Memory search error: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics(
                "search", request.memory_type.value, duration_ms, False, error=str(e)
            )
            raise

    # =========================
    # BATCH OPERATIONS
    # =========================

    async def batch_store(
        self, request: MemoryBatchStoreRequest
    ) -> MemoryBatchStoreResponse:
        """Store multiple memories in batch."""
        start_time = datetime.now()
        batch_id = str(uuid.uuid4())

        logger.info(
            f"Processing batch store {batch_id} with {len(request.memories)} memories"
        )

        stored = []
        failed = []

        if request.parallel:
            # Process in parallel
            semaphore = asyncio.Semaphore(10)

            async def store_with_limit(memory_req):
                async with semaphore:
                    try:
                        return await self.store(memory_req), None
                    except Exception as e:
                        return None, {"key": memory_req.key, "error": str(e)}

            tasks = [store_with_limit(req) for req in request.memories]
            results = await asyncio.gather(*tasks)

            for resp, err in results:
                if resp:
                    stored.append(resp)
                else:
                    failed.append(err)
        else:
            # Process sequentially
            for req in request.memories:
                try:
                    resp = await self.store(req)
                    stored.append(resp)
                except Exception as e:
                    failed.append({"key": req.key, "error": str(e)})

        total_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Batch {batch_id} completed: {len(stored)} stored, {len(failed)} failed in {total_time:.2f}s"
        )

        return MemoryBatchStoreResponse(
            stored=stored,
            failed=failed,
            total=len(request.memories),
            successful=len(stored),
            failed_count=len(failed),
            batch_id=batch_id,
        )

    # =========================
    # UPDATE MEMORY
    # =========================

    async def update(self, request: MemoryUpdateRequest) -> MemoryStoreResponse:
        """Update existing memory."""
        try:
            # First retrieve existing
            retrieve_req = MemoryRetrieveRequest(
                key=request.key, memory_type=request.memory_type
            )
            existing = await self.retrieve(retrieve_req)

            if not existing:
                raise ValueError(f"Memory not found: {request.key}")

            # Prepare update data
            update_data = {}
            if request.content is not None:
                update_data["content"] = request.content
            if request.importance is not None:
                update_data["importance"] = request.importance
            if request.tags is not None:
                update_data["tags"] = request.tags
            if request.metadata is not None:
                update_data["metadata"] = request.metadata

            update_data["updated_at"] = datetime.now().isoformat()

            # Update in manager
            if hasattr(self.manager, "update"):
                self.manager.update(request.key, update_data, request.memory_type.value)
            else:
                # Store as new with updated flag
                store_req = MemoryStoreRequest(
                    key=request.key,
                    content=update_data.get("content", existing.content),
                    memory_type=request.memory_type,
                    importance=update_data.get("importance", existing.importance),
                    tags=update_data.get("tags", existing.tags),
                    metadata={**existing.metadata, **update_data.get("metadata", {})},
                )
                return await self.store(store_req)

            # Invalidate cache
            if self.config["enable_cache"]:
                self._cache.delete(request.key)

            logger.info(f"Memory updated: {request.key}")

            return MemoryStoreResponse(
                status="success",
                key=request.key,
                memory_id=existing.get("id", ""),
                memory_type=request.memory_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Memory update error: {e}")
            raise

    # =========================
    # DELETE MEMORY
    # =========================

    async def delete(self, request: MemoryDeleteRequest) -> Dict[str, Any]:
        """Delete memory."""
        try:
            # Check if exists
            retrieve_req = MemoryRetrieveRequest(
                key=request.key, memory_type=request.memory_type
            )
            existing = await self.retrieve(retrieve_req)

            if not existing:
                raise ValueError(f"Memory not found: {request.key}")

            # Check importance
            if existing.importance > 0.8 and not request.force:
                raise ValueError(
                    f"Memory has high importance ({existing.importance}). Use force=True to delete."
                )

            # Delete from manager
            if hasattr(self.manager, "delete"):
                self.manager.delete(request.key, request.memory_type.value)
            else:
                # Mark as deleted
                self.manager.store(
                    request.key,
                    {"deleted": True, "deleted_at": datetime.now().isoformat()},
                    request.memory_type.value,
                )

            # Remove from cache
            if self.config["enable_cache"]:
                self._cache.delete(request.key)

            await self._log_audit(
                "delete", request.key, request.memory_type.value, None, True
            )

            logger.info(f"Memory deleted: {request.key}")

            return {
                "status": "success",
                "key": request.key,
                "memory_type": request.memory_type.value,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Memory delete error: {e}")
            raise

    # =========================
    # MEMORY CONSOLIDATION
    # =========================

    async def consolidate(self, request: MemoryConsolidationRequest) -> Dict[str, Any]:
        """Consolidate memories (merge similar, remove duplicates)."""
        try:
            if not hasattr(self.manager, "consolidate"):
                logger.warning("Memory consolidation not supported by manager")
                return {
                    "status": "not_supported",
                    "message": "Consolidation not available",
                    "timestamp": datetime.now().isoformat(),
                }

            # Perform consolidation
            result = self.manager.consolidate(
                memory_type=request.memory_type.value,
                min_importance=request.min_importance,
                max_age_days=request.max_age_days,
            )

            logger.info(f"Memory consolidation completed: {result}")

            return {
                "status": "success",
                "consolidated": result.get("consolidated", 0),
                "removed": result.get("removed", 0),
                "merged": result.get("merged", 0),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Memory consolidation error: {e}")
            raise

    # =========================
    # EXPORT MEMORIES
    # =========================

    async def export(self, request: MemoryExportRequest) -> Dict[str, Any]:
        """Export memories to file."""
        try:
            export_file = (
                self.data_dir
                / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{request.format}"
            )

            # Gather memories
            memories = []
            if request.memory_type:
                search_req = MemorySearchRequest(
                    query="*", memory_type=request.memory_type, limit=10000
                )
                result = await self.search(search_req)
                memories = result.results
            else:
                # Get all types
                for mem_type in MemoryType:
                    search_req = MemorySearchRequest(
                        query="*", memory_type=mem_type, limit=10000
                    )
                    result = await self.search(search_req)
                    memories.extend(result.results)

            # Apply date filters
            if request.start_date:
                memories = [
                    m
                    for m in memories
                    if m.get("created_at", "") >= request.start_date.isoformat()
                ]
            if request.end_date:
                memories = [
                    m
                    for m in memories
                    if m.get("created_at", "") <= request.end_date.isoformat()
                ]

            # Export based on format
            if request.format == "json":
                export_data = {
                    "export_date": datetime.now().isoformat(),
                    "count": len(memories),
                    "memories": (
                        memories
                        if request.include_metadata
                        else [m.get("content") for m in memories]
                    ),
                }
                with open(export_file, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif request.format == "csv":
                import csv

                with open(export_file, "w", newline="", encoding="utf-8") as f:
                    if memories and request.include_metadata:
                        writer = csv.DictWriter(f, fieldnames=memories[0].keys())
                        writer.writeheader()
                        writer.writerows(memories)
                    else:
                        writer = csv.writer(f)
                        writer.writerow(["content"])
                        for m in memories:
                            writer.writerow([m.get("content", "")])

            logger.info(f"Exported {len(memories)} memories to {export_file}")

            return {
                "status": "success",
                "export_file": str(export_file),
                "memory_count": len(memories),
                "format": request.format,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Memory export error: {e}")
            raise

    # =========================
    # STATISTICS
    # =========================

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            # Get manager stats
            manager_stats = {}
            if hasattr(self.manager, "get_all_stats"):
                manager_stats = self.manager.get_all_stats()
            elif hasattr(self.manager, "get_stats"):
                manager_stats = self.manager.get_stats()

            # Calculate additional stats
            total_requests = len(self._request_history)
            successful_requests = sum(
                1 for r in self._request_history if r.get("success", False)
            )
            success_rate = (
                (successful_requests / total_requests) if total_requests > 0 else 0
            )

            # Get cache stats
            cache_stats = (
                self._cache.get_stats()
                if self.config["enable_cache"]
                else {"enabled": False}
            )

            # Get metrics from MongoDB
            metrics_data = {}
            if mongo_client.enabled and self.metrics_collection is not None:
                # Get total operations
                total_operations = await self.metrics_collection.count_documents({})

                # Get average duration
                pipeline = [
                    {
                        "$group": {
                            "_id": "$operation",
                            "avg_duration": {"$avg": "$duration_ms"},
                            "count": {"$sum": 1},
                            "success_rate": {"$avg": {"$cond": ["$success", 1, 0]}},
                        }
                    }
                ]

                agg_result = list(
                    await self.metrics_collection.aggregate(pipeline).to_list(length=10)
                )
                metrics_data["operation_stats"] = agg_result
                metrics_data["total_operations"] = total_operations

            return {
                "request_count": self.request_count,
                "success_rate": success_rate,
                "total_requests_history": total_requests,
                "cache": cache_stats,
                "manager_stats": manager_stats,
                "metrics": metrics_data,
                "config": {
                    "enable_cache": self.config["enable_cache"],
                    "enable_vector_search": self.config["enable_vector_search"],
                    "max_memory_size_mb": self.config["max_memory_size_mb"],
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {
                "request_count": self.request_count,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # =========================
    # CACHE MANAGEMENT
    # =========================

    def clear_cache(self) -> Dict[str, Any]:
        """Clear memory cache."""
        cache_size = len(self._cache._cache) if hasattr(self._cache, "_cache") else 0
        self._cache.clear()
        logger.info(f"Cleared {cache_size} cached memories")
        return {"cleared": cache_size, "timestamp": datetime.now().isoformat()}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.config["enable_cache"]:
            return self._cache.get_stats()
        return {"enabled": False, "message": "Cache is disabled"}

    # =========================
    # HEALTH CHECK
    # =========================

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {},
        }

        # Check memory manager
        try:
            if hasattr(self.manager, "health_check"):
                manager_status = self.manager.health_check()
                status["components"]["memory_manager"] = manager_status
            else:
                # Simple test
                test_key = f"health_check_{uuid.uuid4()}"
                test_req = MemoryStoreRequest(
                    key=test_key,
                    content="test",
                    memory_type=MemoryType.SEMANTIC,
                    importance=0.1,
                )
                await self.store(test_req)
                status["components"]["memory_manager"] = {"status": "healthy"}
        except Exception as e:
            status["components"]["memory_manager"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status["status"] = "degraded"

        # Check MongoDB
        if mongo_client.enabled:
            if mongo_client.status.value == "connected":
                status["components"]["mongodb"] = {
                    "status": "healthy",
                    "database": mongo_client.database_name,
                }
            else:
                status["components"]["mongodb"] = {
                    "status": "unhealthy",
                    "error": f"Status: {mongo_client.status.value}",
                }
                status["status"] = "degraded"

        # Check cache
        status["components"]["cache"] = {
            "status": "healthy",
            "enabled": self.config["enable_cache"],
            "size": len(self._cache._cache) if hasattr(self._cache, "_cache") else 0,
        }

        return status

    # =========================
    # SHUTDOWN
    # =========================

    async def shutdown(self):
        """Gracefully shutdown the API."""
        logger.info("Shutting down Memory API...")

        # Save index
        self._save_index()

        # Save any pending metrics
        if mongo_client.enabled:
            logger.info("Saving final metrics to MongoDB...")

        # Clear cache
        self.clear_cache()

        logger.info("Memory API shutdown complete")


# =========================
# GLOBAL INSTANCE
# =========================

memory_api = MemoryApi()


__all__ = [
    "MemoryApi",
    "MemoryType",
    "Importance",
    "RetrievalStrategy",
    "MemoryStoreRequest",
    "MemoryStoreResponse",
    "MemoryRetrieveRequest",
    "MemoryRetrieveResponse",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemoryBatchStoreRequest",
    "MemoryBatchStoreResponse",
    "MemoryDeleteRequest",
    "MemoryUpdateRequest",
    "MemoryConsolidationRequest",
    "MemoryExportRequest",
    "memory_api",
]
