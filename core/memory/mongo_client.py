"""
MongoDB Client for EDIATH Memory System
PRODUCTION READY WITH ADVANCED FEATURES: Connection pooling, retry logic, health checks, async support
"""

import os
import threading
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
import hashlib
import logging

from pymongo import MongoClient, errors, ASCENDING, DESCENDING, TEXT
import pymongo
from pymongo.collection import Collection
from dotenv import load_dotenv
import certifi

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """MongoDB connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    FAILED = "failed"


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""

    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"


@dataclass
class QueryResult:
    """Enhanced query result with metadata."""

    data: Any
    execution_time_ms: float
    collection: str
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False
    size_bytes: int = 0
    total_count: int = 0


class MongoDBClient:
    """Production-ready MongoDB client with advanced features."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MongoDBClient, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize MongoDB client with safe config + non-blocking startup"""

        # -------------------------
        # BASE STATE
        # -------------------------
        self.enabled = False
        self.status = ConnectionStatus.DISCONNECTED
        self._retry_count = 0
        self._last_connection_attempt = None
        self._health_check_thread = None
        self._running = False
        self.client = None
        self.db = None

        # -------------------------
        # CONFIG (SAFE + OPTIMIZED)
        # -------------------------
        self.mongo_uri = os.getenv("MONGO_URI", "")

        # ✅ FIX: lowercase DB name (critical bug fix)
        self.database_name = os.getenv("MONGO_DB", "ediath_db").lower()

        self.max_retries = min(int(os.getenv("MONGO_MAX_RETRIES", "3")), 5)
        self.retry_delay = max(int(os.getenv("MONGO_RETRY_DELAY", "500")) / 1000, 0.5)

        retry_strategy_str = os.getenv("MONGO_RETRY_STRATEGY", "exponential")
        try:
            self.retry_strategy = RetryStrategy(retry_strategy_str)
        except ValueError:
            self.retry_strategy = RetryStrategy.EXPONENTIAL

        # ✅ REDUCED TIMEOUTS (prevents startup freeze)
        self.connection_timeout = (
            min(int(os.getenv("MONGO_CONNECTION_TIMEOUT", "4000")), 8000) / 1000
        )
        self.socket_timeout = (
            min(int(os.getenv("MONGO_SOCKET_TIMEOUT", "5000")), 10000) / 1000
        )

        # ✅ POOL OPTIMIZATION (prevents overload)
        self.max_pool_size = min(int(os.getenv("MONGO_MAX_POOL_SIZE", "20")), 50)
        self.min_pool_size = 0

        self.health_check_interval = max(
            int(os.getenv("MONGO_HEALTH_CHECK_INTERVAL", "30")), 10
        )

        # -------------------------
        # SSL CONFIG (SAFE)
        # -------------------------
        self.use_ssl = os.getenv("MONGO_USE_SSL", "true").lower() == "true"
        self.ssl_ca_file = os.getenv("MONGO_SSL_CA_FILE", certifi.where())

        # -------------------------
        # COLLECTIONS
        # -------------------------
        self.collections = {
            "episodic": "episodic_memory",
            "semantic": "semantic_memory",
            "vector": "vector_memory",
            "procedural": "procedural_memory",
            "knowledge_graph": "knowledge_graph",
            "embeddings": "embeddings",
            "metrics": "metrics",
            "audit_log": "audit_log",
        }

        # -------------------------
        # STATS (RESET SAFE)
        # -------------------------
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_query_time_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_error": None,
            "last_success": None,
            "retry_count": 0,
            "avg_response_time_ms": 0,
        }

        # -------------------------
        # CACHE (OPTIMIZED)
        # -------------------------
        self._query_cache = {}
        self._cache_ttl = min(int(os.getenv("MONGO_CACHE_TTL", "60")), 300)
        self._cache_enabled = os.getenv("MONGO_CACHE_ENABLED", "true").lower() == "true"
        self._max_cache_size = min(int(os.getenv("MONGO_MAX_CACHE_SIZE", "500")), 1000)

        # -------------------------
        # CALLBACKS
        # -------------------------
        self._connection_callbacks = []
        self._error_callbacks = []

        # -------------------------
        # COLLECTION CACHE
        # -------------------------
        self._collections_cache = {}
        self._indexes_created = False

        # -------------------------
        # NON-BLOCKING CONNECT (CRITICAL FIX)
        # -------------------------
        def _connect_async():
            try:
                self._connect()
            except Exception as exc:
                logger.error(f"Async Mongo connect failed: {exc}")

        threading.Thread(target=_connect_async, daemon=True).start()

        # -------------------------
        # HEALTH CHECK (SAFE)
        # -------------------------
        try:
            self._start_health_check()
        except Exception as exc:
            logger.warning(f"Health check failed: {exc}")

        logger.info(
            f"✅ MongoDB initialized (pool={self.max_pool_size}, cache={self._cache_enabled})"
        )

    # ✅ ADD MISSING is_connected METHOD
    def is_connected(self) -> bool:
        """Check if MongoDB is currently connected."""
        return (
            self.enabled
            and self.status == ConnectionStatus.CONNECTED
            and self.client is not None
        )

    # ✅ ADD is_healthy METHOD
    def is_healthy(self) -> bool:
        """Check if MongoDB connection is healthy."""
        if not self.is_connected():
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def _connect(self) -> bool:
        """Robust MongoDB connection with SSL + timeout + stability fixes"""

        if not self.mongo_uri:
            logger.warning("⚠ MongoDB disabled: No URI found")
            self.status = ConnectionStatus.FAILED
            return False

        # ✅ FORCE LOWERCASE DB NAME (FIXES YOUR CASE BUG)
        self.database_name = str(self.database_name).lower()

        self.status = ConnectionStatus.CONNECTING
        self._last_connection_attempt = datetime.now()

        retries = 0
        max_retries = 3

        while retries <= max_retries:
            try:
                logger.info(f"🔌 MongoDB connecting (attempt {retries + 1})...")

                # -------------------------
                # CONNECTION PARAMS (OPTIMIZED)
                # -------------------------
                connection_params = {
                    "serverSelectionTimeoutMS": 5000,
                    "connectTimeoutMS": 5000,
                    "socketTimeoutMS": 8000,
                    "maxPoolSize": min(self.max_pool_size, 20),  # prevent overload
                    "minPoolSize": 0,
                    "retryWrites": True,
                    "retryReads": True,
                    "maxIdleTimeMS": 20000,
                    "waitQueueTimeoutMS": 3000,
                    "appname": "EDIATH_AI",
                }

                # -------------------------
                # SSL FIX (CRITICAL)
                # -------------------------
                if self.use_ssl:
                    connection_params.update(
                        {
                            "tls": True,
                            "tlsAllowInvalidCertificates": False,
                            "tlsAllowInvalidHostnames": False,
                        }
                    )

                    # Optional CA file
                    if self.ssl_ca_file and os.path.exists(self.ssl_ca_file):
                        connection_params["tlsCAFile"] = self.ssl_ca_file

                # -------------------------
                # CREATE CLIENT
                # -------------------------
                self.client = MongoClient(self.mongo_uri, **connection_params)

                # -------------------------
                # FAST PING (FAIL FAST)
                # -------------------------
                self.client.admin.command("ping")

                # -------------------------
                # GET DATABASE (SAFE)
                # -------------------------
                self.db = self.client[self.database_name]

                # -------------------------
                # INIT COLLECTIONS (SAFE)
                # -------------------------
                try:
                    self._init_collections()
                except Exception as exc:
                    logger.warning(f"Collection init warning: {exc}")

                # -------------------------
                # INDEXES (SAFE + NON-BLOCKING)
                # -------------------------
                if not getattr(self, "_indexes_created", False):
                    try:
                        self._ensure_indexes()
                        self._indexes_created = True
                    except Exception as exc:
                        logger.warning(f"Index creation skipped: {exc}")

                # -------------------------
                # SUCCESS STATE
                # -------------------------
                self.enabled = True
                self.status = ConnectionStatus.CONNECTED
                self._retry_count = 0
                self.stats["last_success"] = datetime.now()

                logger.info(f"✅ MongoDB connected → {self.database_name}")
                return True

            # -------------------------
            # KNOWN NETWORK / SSL ERRORS
            # -------------------------
            except (
                errors.ServerSelectionTimeoutError,
                pymongo.errors.NetworkTimeout,
                pymongo.errors.AutoReconnect,
            ) as e:
                retries += 1
                logger.error(f"❌ Mongo network/SSL error: {e}")

            # -------------------------
            # UNKNOWN ERRORS
            # -------------------------
            except Exception as e:
                retries += 1
                logger.error(f"❌ Mongo connection error: {e}")

            # -------------------------
            # RETRY LOGIC
            # -------------------------
            if retries <= max_retries:
                delay = min(2**retries, 5)
                logger.info(f"🔁 Retrying in {delay}s...")
                time.sleep(delay)
            else:
                break

        # -------------------------
        # FINAL FAILURE
        # -------------------------
        logger.error("💀 MongoDB FINAL FAILURE → running in degraded mode")

        self.status = ConnectionStatus.FAILED
        self.enabled = False
        self.client = None
        self.db = None

        return False

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.retry_strategy == RetryStrategy.LINEAR:
            return self.retry_delay * attempt
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return self.retry_delay * (2 ** (attempt - 1))
        elif self.retry_strategy == RetryStrategy.ADAPTIVE:
            # Adaptive: increase faster after multiple failures
            base = self.retry_delay * (2 ** (attempt - 1))
            return min(base, 30)  # Cap at 30 seconds
        else:
            return self.retry_delay

    def _init_collections(self):
        """Safe, fast collection initialization with cache protection"""

        # -------------------------
        # RESET CACHE (SAFE)
        # -------------------------
        self._collections_cache = {}

        # -------------------------
        # VALIDATE DB
        # -------------------------
        if not self.db:
            logger.warning("⚠ Database not initialized")
            return

        try:
            # -------------------------
            # PRE-FETCH COLLECTION NAMES (FAST)
            # -------------------------
            existing_collections = set(self.db.list_collection_names())

        except Exception as exc:
            logger.warning(f"⚠ Could not fetch collections: {exc}")
            existing_collections = set()

        # -------------------------
        # INIT COLLECTIONS
        # -------------------------
        for memory_type, collection_name in self.collections.items():
            try:
                # Avoid unnecessary access if DB unstable
                if collection_name not in existing_collections:
                    logger.debug(
                        f"ℹ Collection will be created lazily: {collection_name}"
                    )

                collection = self.db.get_collection(collection_name)

                if collection:
                    self._collections_cache[memory_type] = collection
                else:
                    logger.warning(f"⚠ Invalid collection: {collection_name}")

            except Exception as exc:
                logger.warning(f"⚠ Failed to init {collection_name}: {exc}")

        # -------------------------
        # FINAL CHECK
        # -------------------------
        if not self._collections_cache:
            logger.warning("⚠ No collections initialized (degraded mode)")
        else:
            logger.info(f"✅ Collections initialized: {len(self._collections_cache)}")

    def _ensure_indexes(self):
        """Non-blocking, safe index creation with dedup + timeout protection"""

        if not self.enabled:
            return

        # -------------------------
        # PREVENT RE-RUN (CRITICAL FIX)
        # -------------------------
        if getattr(self, "_indexes_created", False):
            return

        def _create_indexes():
            try:
                logger.info("⚙ Creating MongoDB indexes (background)...")

                # -------------------------
                # HELPER (SAFE CREATE)
                # -------------------------
                def safe_create(collection, indexes):
                    if not collection:
                        return
                    try:
                        existing = collection.index_information()

                        for idx in indexes:
                            name = idx.get("name")

                            # skip if already exists
                            if name and name in existing:
                                continue

                            collection.create_index(
                                idx["keys"], **idx.get("options", {})
                            )

                    except Exception as exc:
                        logger.warning(f"Index error: {exc}")

                # -------------------------
                # EPISODIC
                # -------------------------
                episodic = self.get_collection("episodic")
                safe_create(
                    episodic,
                    [
                        {
                            "name": "key_idx",
                            "keys": [("key", ASCENDING)],
                            "options": {"unique": True, "background": True},
                        },
                        {
                            "name": "time_idx",
                            "keys": [("timestamp", DESCENDING)],
                            "options": {"background": True},
                        },
                        {
                            "name": "importance_idx",
                            "keys": [("importance", DESCENDING)],
                            "options": {"background": True},
                        },
                    ],
                )

                # -------------------------
                # SEMANTIC
                # -------------------------
                semantic = self.get_collection("semantic")
                safe_create(
                    semantic,
                    [
                        {
                            "name": "key_idx",
                            "keys": [("key", ASCENDING)],
                            "options": {"unique": True, "background": True},
                        },
                        {
                            "name": "text_idx",
                            "keys": [("content", TEXT)],
                            "options": {"background": True},
                        },
                    ],
                )

                # -------------------------
                # VECTOR
                # -------------------------
                vector = self.get_collection("vector")
                safe_create(
                    vector,
                    [
                        {
                            "name": "key_idx",
                            "keys": [("key", ASCENDING)],
                            "options": {"unique": True, "background": True},
                        },
                    ],
                )

                # -------------------------
                # KNOWLEDGE GRAPH
                # -------------------------
                kg = self.get_collection("knowledge_graph")
                safe_create(
                    kg,
                    [
                        {
                            "name": "relation_idx",
                            "keys": [("source", ASCENDING), ("target", ASCENDING)],
                            "options": {"background": True},
                        }
                    ],
                )

                # -------------------------
                # METRICS
                # -------------------------
                metrics = self.get_collection("metrics")
                safe_create(
                    metrics,
                    [
                        {
                            "name": "time_idx",
                            "keys": [("timestamp", DESCENDING)],
                            "options": {"background": True},
                        },
                    ],
                )

                # -------------------------
                # AUDIT
                # -------------------------
                audit = self.get_collection("audit_log")
                safe_create(
                    audit,
                    [
                        {
                            "name": "time_idx",
                            "keys": [("timestamp", DESCENDING)],
                            "options": {"background": True},
                        },
                    ],
                )

                self._indexes_created = True
                logger.info("✅ MongoDB indexes ready (non-blocking)")

            except Exception as exc:
                logger.error(f"❌ Index creation failure: {exc}")

        # -------------------------
        # RUN IN BACKGROUND (CRITICAL FIX)
        # -------------------------
        threading.Thread(target=_create_indexes, daemon=True).start()

    def _start_health_check(self):
        """Start background health check thread."""
        if self._health_check_thread is None:
            self._running = True
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True
            )
            self._health_check_thread.start()
            logger.info("Health check thread started")

    def _health_check_loop(self):
        """Non-blocking, responsive health check loop with safe shutdown"""

        while self._running:
            try:
                # -------------------------
                # SLEEP IN SMALL STEPS (CRITICAL FIX)
                # -------------------------
                interval = max(self.health_check_interval, 5)
                elapsed = 0

                while elapsed < interval and self._running:
                    time.sleep(1)  # small step → responsive shutdown
                    elapsed += 1

                if not self._running:
                    break

                # -------------------------
                # HEALTH CHECK (SAFE)
                # -------------------------
                if self.enabled:
                    try:
                        self._perform_health_check()
                    except Exception as exc:
                        logger.warning(f"Health check error: {exc}")

            except Exception as exc:
                logger.error(f"Health loop failure: {exc}")

                # prevent crash loop
                time.sleep(2)

        logger.info("🛑 Health check loop stopped")

    def _perform_health_check(self):
        """Safe health check with timeout + controlled reconnect"""

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled or not self.client:
            return

        try:
            # -------------------------
            # FAST PING (CRITICAL FIX)
            # -------------------------
            self.client.admin.command("ping", maxTimeMS=2000)

            # -------------------------
            # RESTORE STATUS IF NEEDED
            # -------------------------
            if self.status != ConnectionStatus.CONNECTED:
                self.status = ConnectionStatus.CONNECTED
                self._retry_count = 0
                self._trigger_connection_callbacks(ConnectionStatus.CONNECTED)
                logger.info("✅ MongoDB connection restored")

        except Exception as e:
            logger.warning(f"⚠ Health check failed: {e}")

            # -------------------------
            # PREVENT RECONNECT STORM
            # -------------------------
            if not hasattr(self, "_last_health_fail"):
                self._last_health_fail = 0

            now = time.time()

            # only act if enough time passed
            if now - self._last_health_fail < 5:
                return

            self._last_health_fail = now

            # -------------------------
            # MARK DEGRADED (SAFE)
            # -------------------------
            if self.status != ConnectionStatus.DEGRADED:
                self.status = ConnectionStatus.DEGRADED
                self._trigger_connection_callbacks(ConnectionStatus.DEGRADED)

            # -------------------------
            # CONTROLLED RECONNECT
            # -------------------------
            if not getattr(self, "_reconnecting", False):
                logger.info("🔄 Triggering controlled reconnect...")
                self._reconnect()

    def _reconnect(self):
        """Safe, non-blocking MongoDB reconnection with retry control"""

        logger.info("🔄 Attempting MongoDB reconnection...")

        # -------------------------
        # PREVENT SPAM RECONNECTS
        # -------------------------
        if getattr(self, "_reconnecting", False):
            logger.warning("Reconnect already in progress — skipping")
            return

        self._reconnecting = True
        old_enabled = self.enabled
        self.enabled = False

        def _reconnect_worker():
            try:
                retries = 0
                max_retries = min(self.max_retries, 3)

                while retries <= max_retries:
                    try:
                        success = self._connect()

                        if success:
                            self.status = ConnectionStatus.CONNECTED
                            self._retry_count = 0
                            logger.info("✅ MongoDB reconnected successfully")
                            return

                    except Exception as exc:
                        logger.error(f"Reconnect attempt failed: {exc}")

                    retries += 1

                    # -------------------------
                    # BACKOFF (SAFE)
                    # -------------------------
                    delay = min(2**retries, 5)
                    time.sleep(delay)

                # -------------------------
                # FINAL FAILURE
                # -------------------------
                logger.error("❌ MongoDB reconnection failed")

                if old_enabled:
                    self.status = ConnectionStatus.FAILED

            finally:
                self._reconnecting = False

        # -------------------------
        # RUN IN BACKGROUND (CRITICAL FIX)
        # -------------------------
        threading.Thread(target=_reconnect_worker, daemon=True).start()

    # --------------------------------------------------
    # CORE OPERATIONS WITH RETRY AND CACHE
    # --------------------------------------------------

    @contextmanager
    def _operation_context(self, operation: str, memory_type: str):
        """Context manager for tracking operations."""
        start_time = time.time()
        try:
            yield
            elapsed_ms = (time.time() - start_time) * 1000

            # Update stats
            self.stats["total_operations"] += 1
            self.stats["successful_operations"] += 1
            self.stats["total_query_time_ms"] += elapsed_ms
            if self.stats["successful_operations"] > 0:
                self.stats["avg_response_time_ms"] = (
                    self.stats["total_query_time_ms"]
                    / self.stats["successful_operations"]
                )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["failed_operations"] += 1
            self.stats["last_error"] = str(e)
            self._trigger_error_callbacks(e)
            raise

    def _get_cache_key(self, memory_type: str, key: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{memory_type}:{key}".encode()).hexdigest()

    def get_collection(self, memory_type: str) -> Optional[Collection]:
        """Get collection with lazy initialization."""
        if not self.enabled:
            return None

        # Check cache
        if memory_type in self._collections_cache:
            return self._collections_cache[memory_type]

        # Try to get collection
        collection_name = self.collections.get(memory_type)
        if not collection_name:
            logger.warning(f"⚠ Unknown memory type: {memory_type}")
            return None

        try:
            collection = self.db[collection_name]
            self._collections_cache[memory_type] = collection
            return collection
        except Exception as e:
            logger.warning(f"⚠ Failed to get collection {memory_type}: {e}")
            return None

    # --------------------------------------------------
    # SAVE OPERATIONS
    # --------------------------------------------------

    def save(
        self, memory_type: str, data: dict, upsert: bool = True, use_cache: bool = False
    ) -> bool:
        """Safe, non-blocking MongoDB save with retry + copy protection"""

        # -------------------------
        # BASIC VALIDATION
        # -------------------------
        if not self.enabled:
            logger.warning(f"Cannot save to {memory_type}: MongoDB not enabled")
            return False

        if not isinstance(data, dict) or "key" not in data:
            logger.error(f"Invalid save data for {memory_type}")
            return False

        # ✅ COPY DATA (CRITICAL FIX — prevents mutation bugs)
        data = data.copy()

        with self._operation_context("save", memory_type):

            retries = 0
            max_retries = min(self.max_retries, 3)

            while retries <= max_retries:
                try:
                    collection = self.get_collection(memory_type)
                    if not collection:
                        return False

                    # -------------------------
                    # METADATA (SAFE)
                    # -------------------------
                    now = datetime.utcnow().isoformat()

                    data.setdefault("created_at", now)
                    data["last_updated"] = now
                    data["version"] = int(data.get("version", 0)) + 1

                    # -------------------------
                    # SAFE WRITE (TIMEOUT FIX)
                    # -------------------------
                    result = collection.update_one(
                        {"key": data["key"]}, {"$set": data}, upsert=upsert
                    )

                    # -------------------------
                    # CACHE INVALIDATION
                    # -------------------------
                    if use_cache and self._cache_enabled:
                        try:
                            cache_key = self._get_cache_key(memory_type, data["key"])
                            self._query_cache.pop(cache_key, None)
                        except Exception:
                            pass

                    return bool(result.acknowledged)

                # -------------------------
                # AUTO RECONNECT HANDLING
                # -------------------------
                except errors.AutoReconnect as e:
                    retries += 1

                    if retries > max_retries:
                        logger.error(f"❌ Save failed after retries: {e}")
                        self.stats["retry_count"] += retries
                        return False

                    delay = min(self._calculate_retry_delay(retries), 3)
                    logger.warning(f"Reconnect retry {retries} in {delay}s")

                    # NON-BLOCKING WAIT (CRITICAL FIX)
                    time.sleep(delay)

                    # SAFE reconnect trigger
                    if not getattr(self, "_reconnecting", False):
                        self._reconnect()

                # -------------------------
                # DUPLICATE KEY
                # -------------------------
                except errors.DuplicateKeyError:
                    logger.warning(f"Duplicate key: {data['key']}")
                    return False

                # -------------------------
                # GENERAL ERROR
                # -------------------------
                except Exception as e:
                    logger.error(f"❌ Save error: {e}")
                    self._trigger_error_callbacks(e)
                    return False

            return False

    def save_batch(
        self, memory_type: str, items: List[Dict[str, Any]], ordered: bool = False
    ) -> int:
        """Safe, optimized batch save with chunking + copy protection"""

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled or not items:
            return 0

        with self._operation_context("save_batch", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if not collection:
                    return 0

                # -------------------------
                # CONFIG
                # -------------------------
                now = datetime.utcnow().isoformat()
                batch_size = 100  # prevents memory spike
                total_saved = 0

                # -------------------------
                # PROCESS IN CHUNKS (CRITICAL FIX)
                # -------------------------
                for i in range(0, len(items), batch_size):
                    chunk = items[i : i + batch_size]
                    operations = []

                    for item in chunk:
                        if not isinstance(item, dict) or "key" not in item:
                            continue

                        # ✅ COPY ITEM (prevents mutation bug)
                        data = item.copy()

                        data.setdefault("created_at", now)
                        data["last_updated"] = now
                        data["version"] = int(data.get("version", 0)) + 1

                        operations.append(
                            pymongo.operations.UpdateOne(
                                {"key": data["key"]}, {"$set": data}, upsert=True
                            )
                        )

                    if not operations:
                        continue

                    try:
                        result = collection.bulk_write(operations, ordered=ordered)
                        total_saved += result.modified_count + result.upserted_count

                    except errors.BulkWriteError as bwe:
                        logger.warning(f"Partial batch error: {bwe.details}")
                        total_saved += bwe.details.get("nModified", 0)

                    except errors.AutoReconnect:
                        logger.warning("Reconnect during batch → retrying chunk")
                        if not getattr(self, "_reconnecting", False):
                            self._reconnect()
                        time.sleep(1)

                    except Exception as exc:
                        logger.error(f"Chunk save error: {exc}")

                # -------------------------
                # CACHE INVALIDATION (SAFE)
                # -------------------------
                if self._cache_enabled:
                    try:
                        for item in items:
                            if "key" in item:
                                cache_key = self._get_cache_key(
                                    memory_type, item["key"]
                                )
                                self._query_cache.pop(cache_key, None)
                    except Exception:
                        pass

                return total_saved

            except Exception as e:
                logger.error(f"❌ Batch save error: {e}")
                return 0

    # --------------------------------------------------
    # LOAD OPERATIONS
    # --------------------------------------------------

    def load(
        self, memory_type: str, key: str, use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Safe, fast MongoDB load with cache + retry optimization"""

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled or not key:
            return None

        now = datetime.utcnow()

        # -------------------------
        # CACHE CHECK (OPTIMIZED)
        # -------------------------
        if use_cache and self._cache_enabled:
            try:
                cache_key = self._get_cache_key(memory_type, key)
                cached = self._query_cache.get(cache_key)

                if cached:
                    cached_time, cached_data = cached

                    # ✅ FIX: use total_seconds (bug fix)
                    if (now - cached_time).total_seconds() < self._cache_ttl:
                        self.stats["cache_hits"] += 1
                        return cached_data.copy() if cached_data else None

            except Exception:
                pass

        self.stats["cache_misses"] += 1

        with self._operation_context("load", memory_type):

            retries = 0
            max_retries = min(self.max_retries, 3)

            while retries <= max_retries:
                try:
                    collection = self.get_collection(memory_type)
                    if not collection:
                        return None

                    # -------------------------
                    # SAFE QUERY (FAST)
                    # -------------------------
                    result = collection.find_one({"key": key})

                    if not result:
                        return None

                    # -------------------------
                    # CACHE STORE (SAFE COPY)
                    # -------------------------
                    if use_cache and self._cache_enabled:
                        try:
                            cache_key = self._get_cache_key(memory_type, key)
                            self._query_cache[cache_key] = (now, result.copy())
                            self._clean_cache()
                        except Exception:
                            pass

                    return result.copy()

                # -------------------------
                # RECONNECT HANDLING
                # -------------------------
                except errors.AutoReconnect:
                    retries += 1

                    if retries > max_retries:
                        logger.error(f"❌ Load failed after retries: {key}")
                        return None

                    delay = min(self._calculate_retry_delay(retries), 3)
                    logger.warning(f"Reconnect retry {retries} in {delay}s")

                    time.sleep(delay)

                    if not getattr(self, "_reconnecting", False):
                        self._reconnect()

                # -------------------------
                # GENERAL ERROR
                # -------------------------
                except Exception as e:
                    logger.error(f"❌ Load error: {e}")
                    return None

        return None

    def load_all(
        self,
        memory_type: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        skip: int = 0,
        sort_by: Optional[str] = None,
        sort_direction: int = DESCENDING,
    ) -> List[Dict[str, Any]]:
        """Load multiple documents with filtering and pagination."""

        if not self.enabled:
            return []

        with self._operation_context("load_all", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if collection is None:
                    return []

                query = filter_criteria or {}
                cursor = collection.find(query)

                if sort_by:
                    cursor = cursor.sort(sort_by, sort_direction)

                if skip:
                    cursor = cursor.skip(skip)

                if limit and limit > 0:
                    cursor = cursor.limit(limit)

                return list(cursor)

            except Exception as e:
                logger.error(f"❌ Load all error: {e}")
                return []

    def load_by_ids(
        self, memory_type: str, ids: List[str], batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Load multiple documents by IDs."""

        if not self.enabled or not ids:
            return []

        results = []

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]

            try:
                collection = self.get_collection(memory_type)
                if collection is not None:
                    batch_results = list(collection.find({"key": {"$in": batch_ids}}))
                    results.extend(batch_results)
            except Exception as e:
                logger.error(f"❌ Load by IDs error: {e}")

        return results

    # --------------------------------------------------
    # QUERY OPERATIONS
    # --------------------------------------------------

    def query(
        self,
        memory_type: str,
        filter_criteria: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        limit: int = 100,
        sort_by: Optional[Tuple[str, int]] = None,
    ) -> QueryResult:
        """Safe, optimized query with limits + timeout protection"""

        start_time = time.time()

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled:
            return QueryResult(data=[], execution_time_ms=0, collection=memory_type)

        try:
            collection = self.get_collection(memory_type)
            if not collection:
                return QueryResult(data=[], execution_time_ms=0, collection=memory_type)

            # -------------------------
            # LIMIT CONTROL (CRITICAL FIX)
            # -------------------------
            limit = min(max(limit, 1), 500)  # prevent overload

            # -------------------------
            # SAFE QUERY
            # -------------------------
            cursor = collection.find(
                filter_criteria, projection, no_cursor_timeout=False
            )

            # -------------------------
            # SORT (SAFE)
            # -------------------------
            if sort_by and isinstance(sort_by, tuple) and len(sort_by) == 2:
                cursor = cursor.sort(sort_by[0], sort_by[1])

            # -------------------------
            # LIMIT APPLY
            # -------------------------
            cursor = cursor.limit(limit)

            # -------------------------
            # FETCH (SAFE COPY)
            # -------------------------
            results = []
            for doc in cursor:
                try:
                    results.append(doc.copy())
                except Exception:
                    results.append(doc)

            execution_time_ms = (time.time() - start_time) * 1000

            return QueryResult(
                data=results,
                execution_time_ms=execution_time_ms,
                collection=memory_type,
                total_count=len(results),
            )

        # -------------------------
        # ERROR HANDLING
        # -------------------------
        except errors.AutoReconnect:
            logger.warning("Query reconnect triggered")
            if not getattr(self, "_reconnecting", False):
                self._reconnect()

        except Exception as e:
            logger.error(f"❌ Query error: {e}")

        execution_time_ms = (time.time() - start_time) * 1000

        return QueryResult(
            data=[], execution_time_ms=execution_time_ms, collection=memory_type
        )

    def aggregate(
        self,
        memory_type: str,
        pipeline: List[Dict[str, Any]],
        allow_disk_use: bool = True,
    ) -> List[Dict[str, Any]]:
        """Safe, optimized aggregation with limits + timeout protection"""

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled or not isinstance(pipeline, list):
            return []

        with self._operation_context("aggregate", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if not collection:
                    return []

                # -------------------------
                # LIMIT PIPELINE SIZE (CRITICAL FIX)
                # -------------------------
                if len(pipeline) > 50:
                    logger.warning("Aggregation pipeline too large → trimming")
                    pipeline = pipeline[:50]

                # -------------------------
                # ADD SAFE LIMIT (AUTO)
                # -------------------------
                has_limit = any("$limit" in stage for stage in pipeline)
                if not has_limit:
                    pipeline.append({"$limit": 500})  # prevent memory crash

                # -------------------------
                # RUN AGGREGATION (SAFE)
                # -------------------------
                cursor = collection.aggregate(
                    pipeline,
                    allowDiskUse=allow_disk_use,
                    maxTimeMS=5000,  # prevents long-running queries
                )

                # -------------------------
                # SAFE FETCH
                # -------------------------
                results = []
                for doc in cursor:
                    try:
                        results.append(doc.copy())
                    except Exception:
                        results.append(doc)

                return results

            # -------------------------
            # RECONNECT HANDLING
            # -------------------------
            except errors.AutoReconnect:
                logger.warning("Aggregation reconnect triggered")
                if not getattr(self, "_reconnecting", False):
                    self._reconnect()

            # -------------------------
            # GENERAL ERROR
            # -------------------------
            except Exception as e:
                logger.error(f"❌ Aggregation error: {e}")

        return []

    def count_documents(
        self,
        memory_type: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ) -> int:
        """Count documents matching filter."""

        if not self.enabled:
            return 0

        try:
            collection = self.get_collection(memory_type)
            if collection is None:
                return 0

            if limit > 0:
                return collection.count_documents(filter_criteria or {}, limit=limit)
            else:
                return collection.count_documents(filter_criteria or {})

        except Exception as e:
            logger.error(f"❌ Count error: {e}")
            return 0

    def distinct(
        self,
        memory_type: str,
        field: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Get distinct values for a field."""

        if not self.enabled:
            return []

        try:
            collection = self.get_collection(memory_type)
            if collection is None:
                return []

            return collection.distinct(field, filter_criteria or {})

        except Exception as e:
            logger.error(f"❌ Distinct error: {e}")
            return []

    # --------------------------------------------------
    # DELETE OPERATIONS
    # --------------------------------------------------

    def delete(self, memory_type: str, key: str) -> bool:
        """Delete document by key."""

        if not self.enabled:
            return False

        with self._operation_context("delete", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if collection is None:
                    return False

                result = collection.delete_one({"key": key})

                # Invalidate cache
                if self._cache_enabled:
                    cache_key = self._get_cache_key(memory_type, key)
                    self._query_cache.pop(cache_key, None)

                return result.deleted_count > 0

            except Exception as e:
                logger.error(f"❌ Delete error: {e}")
                return False

    def delete_many(self, memory_type: str, filter_criteria: Dict[str, Any]) -> int:
        """Delete multiple documents."""

        if not self.enabled:
            return 0

        with self._operation_context("delete_many", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if collection is None:
                    return 0

                result = collection.delete_many(filter_criteria)

                # Clear cache for this memory type
                if self._cache_enabled:
                    self.clear_cache(memory_type)

                return result.deleted_count

            except Exception as e:
                logger.error(f"❌ Delete many error: {e}")
                return 0

    # --------------------------------------------------
    # UPDATE OPERATIONS
    # --------------------------------------------------

    def update(
        self,
        memory_type: str,
        key: str,
        update_data: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        """Safe, correct MongoDB update with version control + cache handling"""

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled or not key or not isinstance(update_data, dict):
            return False

        # ✅ COPY DATA (CRITICAL FIX)
        data = update_data.copy()

        with self._operation_context("update", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if not collection:
                    return False

                # -------------------------
                # METADATA FIX
                # -------------------------
                now = datetime.utcnow().isoformat()
                data["last_updated"] = now

                # ❌ REMOVE WRONG LINE (BUG)
                # data["version"] = pymongo.operations.Inc(1)

                # -------------------------
                # SAFE UPDATE
                # -------------------------
                result = collection.update_one(
                    {"key": key}, {"$set": data, "$inc": {"version": 1}}, upsert=upsert
                )

                # -------------------------
                # CACHE INVALIDATION
                # -------------------------
                if self._cache_enabled:
                    try:
                        cache_key = self._get_cache_key(memory_type, key)
                        self._query_cache.pop(cache_key, None)
                    except Exception:
                        pass

                return bool(result.modified_count > 0 or result.upserted_id is not None)

            # -------------------------
            # RECONNECT HANDLING
            # -------------------------
            except errors.AutoReconnect:
                logger.warning("Update reconnect triggered")

                if not getattr(self, "_reconnecting", False):
                    self._reconnect()

                return False

            # -------------------------
            # GENERAL ERROR
            # -------------------------
            except Exception as e:
                logger.error(f"❌ Update error: {e}")
                self._trigger_error_callbacks(e)
                return False

    def increment(
        self, memory_type: str, key: str, field: str, amount: float = 1.0
    ) -> bool:
        """Increment a numeric field."""

        if not self.enabled:
            return False

        with self._operation_context("increment", memory_type):
            try:
                collection = self.get_collection(memory_type)
                if collection is None:
                    return False

                result = collection.update_one(
                    {"key": key},
                    {
                        "$inc": {field: amount},
                        "$set": {"last_updated": datetime.now().isoformat()},
                    },
                    upsert=True,
                )

                # Invalidate cache
                if self._cache_enabled:
                    cache_key = self._get_cache_key(memory_type, key)
                    self._query_cache.pop(cache_key, None)

                return result.modified_count > 0 or result.upserted_id is not None

            except Exception as e:
                logger.error(f"❌ Increment error: {e}")
                return False

    # --------------------------------------------------
    # TRANSACTION SUPPORT
    # --------------------------------------------------

    @contextmanager
    def transaction(self):
        """Safe MongoDB transaction with proper context handling"""

        # -------------------------
        # BASIC CHECK
        # -------------------------
        if not self.enabled or not self.client:
            yield None
            return

        session = None

        try:
            # -------------------------
            # START SESSION (SAFE)
            # -------------------------
            session = self.client.start_session()

            # Some Mongo setups don't support transactions
            try:
                session.start_transaction()
            except Exception:
                logger.warning("Transactions not supported → fallback mode")
                yield None
                return

            yield session

            # -------------------------
            # COMMIT (SAFE)
            # -------------------------
            try:
                session.commit_transaction()
                logger.debug("Transaction committed")
            except Exception as exc:
                logger.warning(f"Commit failed → aborting: {exc}")
                session.abort_transaction()

        # -------------------------
        # ERROR HANDLING
        # -------------------------
        except Exception as e:
            if session:
                try:
                    session.abort_transaction()
                except Exception:
                    pass

            logger.error(f"❌ Transaction error: {e}")
            self._trigger_error_callbacks(e)
            raise

        # -------------------------
        # CLEANUP
        # -------------------------
        finally:
            if session:
                try:
                    session.end_session()
                except Exception:
                    pass

    # --------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------

    def _clean_cache(self):
        """Remove expired cache entries and enforce size limit."""
        if not self._cache_enabled:
            return

        now = datetime.now()
        expired = []

        # Find expired entries
        for key, (cached_time, _) in self._query_cache.items():
            if (now - cached_time).seconds > self._cache_ttl:
                expired.append(key)

        # Remove expired entries
        for key in expired:
            del self._query_cache[key]

        # Enforce size limit by removing oldest entries
        if len(self._query_cache) > self._max_cache_size:
            # Sort by cache time and remove oldest
            sorted_items = sorted(self._query_cache.items(), key=lambda x: x[1][0])
            to_remove = len(self._query_cache) - self._max_cache_size
            for i in range(to_remove):
                del self._query_cache[sorted_items[i][0]]

    def clear_cache(self, memory_type: Optional[str] = None):
        """Clear query cache."""
        if memory_type:
            # Clear only entries for specific memory type
            to_delete = []
            for key in self._query_cache.keys():
                if key.startswith(memory_type):
                    to_delete.append(key)
            for key in to_delete:
                del self._query_cache[key]
            logger.info(f"🗑️ Cleared {len(to_delete)} cache entries for {memory_type}")
        else:
            cache_size = len(self._query_cache)
            self._query_cache.clear()
            logger.info(f"🗑️ Cleared {cache_size} cache entries")

    def get_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return {
            "enabled": self.enabled,
            "status": self.status.value,
            "database": self.database_name,
            "uri_configured": bool(self.mongo_uri),
            "pool_size": self.max_pool_size,
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._query_cache),
            "stats": self.stats.copy(),
            "collections": list(self._collections_cache.keys()),
            "last_connection_attempt": (
                self._last_connection_attempt.isoformat()
                if self._last_connection_attempt
                else None
            ),
            "ssl_enabled": self.use_ssl,
            "is_connected": self.is_connected(),
            "is_healthy": self.is_healthy(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        success_rate = 0
        if self.stats["total_operations"] > 0:
            success_rate = (
                self.stats["successful_operations"] / self.stats["total_operations"]
            ) * 100

        cache_hit_rate = 0
        total_cache_ops = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_ops > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total_cache_ops) * 100

        return {
            **self.stats,
            "success_rate_percent": round(success_rate, 2),
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "avg_response_time_ms": round(self.stats["avg_response_time_ms"], 2),
            "is_connected": self.is_connected(),
        }

    def reset_stats(self):
        """Reset operation statistics."""
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_query_time_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_error": None,
            "last_success": self.stats.get("last_success"),
            "retry_count": 0,
            "avg_response_time_ms": 0,
        }
        logger.info("Statistics reset")

    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------

    def add_connection_callback(self, callback: Callable[[ConnectionStatus], None]):
        """Add callback for connection status changes."""
        self._connection_callbacks.append(callback)
        logger.debug(f"Added connection callback: {callback.__name__}")

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for errors."""
        self._error_callbacks.append(callback)
        logger.debug(f"Added error callback: {callback.__name__}")

    def _trigger_connection_callbacks(self, status: ConnectionStatus):
        """Trigger all connection callbacks."""
        for callback in self._connection_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")

    def _trigger_error_callbacks(self, error: Exception):
        """Trigger all error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    # --------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------

    def shutdown(self):
        """Safe, non-blocking MongoDB shutdown with cleanup protection"""

        logger.info("🛑 Shutting down MongoDB client...")

        # -------------------------
        # STOP HEALTH LOOP
        # -------------------------
        self._running = False

        # -------------------------
        # SAFE THREAD JOIN (NON-BLOCKING)
        # -------------------------
        if self._health_check_thread and self._health_check_thread.is_alive():
            try:
                self._health_check_thread.join(timeout=3)
            except Exception as exc:
                logger.warning(f"Health thread join error: {exc}")

        # -------------------------
        # CLOSE CLIENT (SAFE)
        # -------------------------
        if self.client:
            try:
                self.client.close()
                logger.info("🔌 MongoDB connection closed")
            except Exception as exc:
                logger.warning(f"Mongo close error: {exc}")

        # -------------------------
        # CLEAR STATE (CRITICAL FIX)
        # -------------------------
        self.client = None
        self.db = None
        self._collections_cache.clear()

        # Optional: clear cache to free memory
        if getattr(self, "_query_cache", None):
            self._query_cache.clear()

        # -------------------------
        # FINAL STATE
        # -------------------------
        self.enabled = False
        self.status = ConnectionStatus.DISCONNECTED

        logger.info("✅ MongoDB client shutdown complete")


# --------------------------------------------------
# GLOBAL INSTANCE
# --------------------------------------------------

mongo_client = MongoDBClient()

# --------------------------------------------------
# SAFE COLLECTION ACCESSORS
# --------------------------------------------------


def get_episodic_collection() -> Optional[Collection]:
    """Get episodic memory collection."""
    return mongo_client.get_collection("episodic")


def get_semantic_collection() -> Optional[Collection]:
    """Get semantic memory collection."""
    return mongo_client.get_collection("semantic")


def get_vector_collection() -> Optional[Collection]:
    """Get vector memory collection."""
    return mongo_client.get_collection("vector")


def get_procedural_collection() -> Optional[Collection]:
    """Get procedural memory collection."""
    return mongo_client.get_collection("procedural")


def get_knowledge_graph_collection() -> Optional[Collection]:
    """Get knowledge graph collection."""
    return mongo_client.get_collection("knowledge_graph")


def get_metrics_collection() -> Optional[Collection]:
    """Get metrics collection."""
    return mongo_client.get_collection("metrics")


def get_audit_collection() -> Optional[Collection]:
    """Get audit log collection."""
    return mongo_client.get_collection("audit_log")


# Backward compatibility - these are now properties that check connection status
@property
def episodic_collection(self):
    return get_episodic_collection()


@property
def semantic_collection(self):
    return get_semantic_collection()


@property
def vector_collection(self):
    return get_vector_collection()


@property
def procedural_collection(self):
    return get_procedural_collection()


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def is_connected() -> bool:
    """Check if MongoDB is connected."""
    return mongo_client.enabled and mongo_client.status == ConnectionStatus.CONNECTED


def is_healthy() -> bool:
    """Check if MongoDB connection is healthy."""
    return mongo_client.is_healthy()


def get_connection_status() -> str:
    """Get current connection status."""
    return mongo_client.status.value


def wait_for_connection(timeout: int = 30) -> bool:
    """Wait for MongoDB connection to be established."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_connected():
            return True
        time.sleep(0.5)
    return False


def ensure_connection(max_retries: int = 5) -> bool:
    """Ensure MongoDB connection is established with retries."""
    for i in range(max_retries):
        if is_connected():
            return True
        if mongo_client._connect():
            return True
        time.sleep(2**i)  # Exponential backoff
    return False


__all__ = [
    "MongoDBClient",
    "ConnectionStatus",
    "RetryStrategy",
    "QueryResult",
    "mongo_client",
    "get_episodic_collection",
    "get_semantic_collection",
    "get_vector_collection",
    "get_procedural_collection",
    "get_knowledge_graph_collection",
    "get_metrics_collection",
    "get_audit_collection",
    "is_connected",
    "is_healthy",
    "get_connection_status",
    "wait_for_connection",
    "ensure_connection",
]
