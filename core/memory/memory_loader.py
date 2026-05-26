"""
Memory loading utilities with intelligent loading, hybrid storage,
checkpoint management, and recovery mechanisms.
(PRODUCTION READY WITH DB + CHECKPOINT + CLOUD SUPPORT)
"""

from typing import Optional, Dict, Any, List, Callable
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import gzip
import pickle

from .memory_manager import MemoryManager
from .mongo_client import mongo_client, ConnectionStatus
from ..utils.logger import logger
from ..utils.helpers import get_project_root, ensure_dir


class LoadPriority(Enum):
    """Priority for memory loading."""

    DB_FIRST = "db_first"
    CHECKPOINT_FIRST = "checkpoint_first"
    HYBRID = "hybrid"
    FRESH = "fresh"


class BackupFormat(Enum):
    """Backup file formats."""

    JSON = "json"
    JSON_GZIP = "json.gz"
    PICKLE = "pickle"


@dataclass
class LoadResult:
    """Result of memory loading operation."""

    success: bool
    source: str
    episodic_count: int = 0
    semantic_count: int = 0
    vector_count: int = 0
    load_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "source": self.source,
            "episodic_count": self.episodic_count,
            "semantic_count": self.semantic_count,
            "vector_count": self.vector_count,
            "load_time_ms": self.load_time_ms,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BackupInfo:
    """Information about a backup."""

    path: str
    size_bytes: int
    created_at: datetime
    format: BackupFormat
    episodic_count: int
    semantic_count: int
    vector_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "size_mb": self.size_bytes / (1024 * 1024),
            "created_at": self.created_at.isoformat(),
            "format": self.format.value,
            "episodic_count": self.episodic_count,
            "semantic_count": self.semantic_count,
            "vector_count": self.vector_count,
        }


class MemoryLoader:
    """Advanced memory loader with hybrid storage, versioning, and recovery."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MemoryLoader, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize memory loader."""
        self.project_root = get_project_root()
        self.checkpoint_dir = ensure_dir(
            Path(self.project_root) / "data" / "checkpoints"
        )
        self.backup_dir = ensure_dir(Path(self.project_root) / "data" / "backups")
        self.export_dir = ensure_dir(Path(self.project_root) / "data" / "exports")

        # Load history
        self.load_history: List[LoadResult] = []
        self.max_history = 100

        # Auto-backup settings
        self.auto_backup_enabled = True
        self.auto_backup_interval_hours = 24
        self.last_auto_backup: Optional[datetime] = None

        # Callbacks
        self._load_callbacks: List[Callable[[LoadResult], None]] = []

        # Statistics
        self.stats = {
            "total_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "avg_load_time_ms": 0.0,
            "last_load": None,
        }

        logger.info("✅ MemoryLoader initialized")

    # --------------------------------------------------
    # 🔥 MAIN INITIALIZER (SMART LOADING)
    # --------------------------------------------------

    @staticmethod
    def initialize_memory(
        use_db: bool = True,
        checkpoint_path: Optional[str] = None,
        priority: LoadPriority = LoadPriority.DB_FIRST,
        auto_backup: bool = True,
        validate_integrity: bool = True,
    ) -> MemoryManager:
        """
        Initialize memory system with intelligent loading.

        Args:
            use_db: Whether to use MongoDB
            checkpoint_path: Path to checkpoint file
            priority: Loading priority strategy
            auto_backup: Enable auto-backup on load
            validate_integrity: Validate loaded data integrity

        Returns:
            Initialized MemoryManager
        """

        logger.info(f"Initializing memory systems (priority={priority.value})")
        start_time = datetime.now()

        memory = MemoryManager()
        loader = MemoryLoader()

        load_result = None

        try:
            # Execute based on priority
            if priority == LoadPriority.DB_FIRST:
                load_result = loader._load_db_first(
                    memory, use_db, checkpoint_path, validate_integrity
                )
            elif priority == LoadPriority.CHECKPOINT_FIRST:
                load_result = loader._load_checkpoint_first(
                    memory, checkpoint_path, use_db, validate_integrity
                )
            elif priority == LoadPriority.HYBRID:
                load_result = loader._load_hybrid(
                    memory, use_db, checkpoint_path, validate_integrity
                )
            else:  # FRESH
                load_result = LoadResult(
                    success=True,
                    source="fresh",
                    episodic_count=0,
                    semantic_count=0,
                    vector_count=0,
                )

            # Record load result
            load_result.load_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            loader._record_load(load_result)

            # Auto-backup after successful load
            if auto_backup and load_result.success and load_result.source != "fresh":
                loader._auto_backup_if_needed(memory)

            # Log final status
            logger.info(
                f"✅ Memory initialized from {load_result.source} | "
                f"Episodic: {load_result.episodic_count} | "
                f"Semantic: {load_result.semantic_count} | "
                f"Vector: {load_result.vector_count} | "
                f"Time: {load_result.load_time_ms:.1f}ms"
            )

            return memory

        except Exception as e:
            logger.error(f"Memory initialization failed: {e}")

            # Return fresh memory as fallback
            load_result = LoadResult(success=False, source="failed", error=str(e))
            loader._record_load(load_result)

            return memory

    def _load_db_first(
        self,
        memory: MemoryManager,
        use_db: bool,
        checkpoint_path: Optional[str],
        validate: bool,
    ) -> LoadResult:
        """Load from DB first, fallback to checkpoint."""

        # Try MongoDB
        if use_db and self._is_db_available():
            try:
                logger.info("Loading memory from MongoDB...")

                episodic_data = mongo_client.load_all("episodic")
                semantic_data = mongo_client.load_all("semantic")
                vector_data = mongo_client.load_all("vector")

                # Validate data integrity
                if validate:
                    self._validate_data(episodic_data, "episodic")
                    self._validate_data(semantic_data, "semantic")
                    self._validate_data(vector_data, "vector")

                # Load into memory
                self._load_to_memory(memory, episodic_data, semantic_data, vector_data)

                return LoadResult(
                    success=True,
                    source="mongodb",
                    episodic_count=len(episodic_data),
                    semantic_count=len(semantic_data),
                    vector_count=len(vector_data),
                )

            except Exception as e:
                logger.warning(f"MongoDB load failed: {e}")

        # Fallback to checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Falling back to checkpoint: {checkpoint_path}")
            return self._load_from_checkpoint_file(memory, checkpoint_path, validate)

        # Fresh memory
        logger.info("No data source available, using fresh memory")
        return LoadResult(
            success=True,
            source="fresh",
            episodic_count=0,
            semantic_count=0,
            vector_count=0,
        )

    def _load_checkpoint_first(
        self,
        memory: MemoryManager,
        checkpoint_path: Optional[str],
        use_db: bool,
        validate: bool,
    ) -> LoadResult:
        """Load from checkpoint first, fallback to DB."""

        # Try checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            result = self._load_from_checkpoint_file(memory, checkpoint_path, validate)

            if result.success:
                return result

        # Try to find latest checkpoint
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Loading from latest checkpoint: {latest_checkpoint}")
            result = self._load_from_checkpoint_file(
                memory, latest_checkpoint, validate
            )

            if result.success:
                return result

        # Fallback to DB
        if use_db and self._is_db_available():
            logger.info("Falling back to MongoDB...")
            episodic_data = mongo_client.load_all("episodic")
            semantic_data = mongo_client.load_all("semantic")
            vector_data = mongo_client.load_all("vector")

            self._load_to_memory(memory, episodic_data, semantic_data, vector_data)

            return LoadResult(
                success=True,
                source="mongodb",
                episodic_count=len(episodic_data),
                semantic_count=len(semantic_data),
                vector_count=len(vector_data),
            )

        # Fresh memory
        logger.info("No data source available, using fresh memory")
        return LoadResult(
            success=True,
            source="fresh",
            episodic_count=0,
            semantic_count=0,
            vector_count=0,
        )

    def _load_hybrid(
        self,
        memory: MemoryManager,
        use_db: bool,
        checkpoint_path: Optional[str],
        validate: bool,
    ) -> LoadResult:
        """Hybrid loading - merge DB and checkpoint data."""

        episodic_data = []
        semantic_data = []
        vector_data = []

        # Load from DB
        if use_db and self._is_db_available():
            logger.info("Loading DB data...")
            episodic_data = mongo_client.load_all("episodic")
            semantic_data = mongo_client.load_all("semantic")
            vector_data = mongo_client.load_all("vector")

        # Load from checkpoint and merge
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Merging checkpoint data: {checkpoint_path}")
            checkpoint_data = self._read_checkpoint_file(checkpoint_path)

            if checkpoint_data:
                # Merge with DB data (checkpoint takes precedence)
                episodic_data = self._merge_data(
                    episodic_data, checkpoint_data.get("episodic", [])
                )
                semantic_data = self._merge_data(
                    semantic_data, checkpoint_data.get("semantic", [])
                )
                vector_data = self._merge_data(
                    vector_data, checkpoint_data.get("vector", [])
                )

        # Validate
        if validate:
            self._validate_data(episodic_data, "episodic")
            self._validate_data(semantic_data, "semantic")
            self._validate_data(vector_data, "vector")

        # Load to memory
        self._load_to_memory(memory, episodic_data, semantic_data, vector_data)

        return LoadResult(
            success=True,
            source="hybrid",
            episodic_count=len(episodic_data),
            semantic_count=len(semantic_data),
            vector_count=len(vector_data),
        )

    # --------------------------------------------------
    # 🔥 CHECKPOINT MANAGEMENT
    # --------------------------------------------------

    @staticmethod
    def load_memory_from_checkpoint(
        checkpoint_path: str, validate: bool = True
    ) -> MemoryManager:
        """Load memory directly from checkpoint file."""

        memory = MemoryManager()
        loader = MemoryLoader()

        result = loader._load_from_checkpoint_file(memory, checkpoint_path, validate)

        if result.success:
            logger.info(f"✅ Memory loaded from checkpoint: {checkpoint_path}")
        else:
            logger.error(f"Failed to load checkpoint: {result.error}")

        return memory

    def _load_from_checkpoint_file(
        self, memory: MemoryManager, checkpoint_path: str, validate: bool
    ) -> LoadResult:
        """Load memory from checkpoint file."""

        try:
            data = self._read_checkpoint_file(checkpoint_path)

            if not data:
                return LoadResult(
                    success=False,
                    source="checkpoint",
                    error="Empty or invalid checkpoint file",
                )

            # Validate data
            if validate:
                self._validate_data(data.get("episodic", []), "episodic")
                self._validate_data(data.get("semantic", []), "semantic")
                self._validate_data(data.get("vector", []), "vector")

            # Load to memory
            self._load_to_memory(
                memory,
                data.get("episodic", []),
                data.get("semantic", []),
                data.get("vector", []),
            )

            return LoadResult(
                success=True,
                source="checkpoint",
                episodic_count=len(data.get("episodic", [])),
                semantic_count=len(data.get("semantic", [])),
                vector_count=len(data.get("vector", [])),
            )

        except Exception as e:
            logger.error(f"Checkpoint load error: {e}")
            return LoadResult(success=False, source="checkpoint", error=str(e))

    def _read_checkpoint_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Read checkpoint file in various formats."""

        path = Path(filepath)

        if not path.exists():
            return None

        try:
            # JSON format
            if path.suffix == ".json":
                with open(path, "r") as f:
                    return json.load(f)

            # Gzipped JSON
            elif path.suffix == ".gz":
                with gzip.open(path, "rt") as f:
                    return json.load(f)

            # Pickle format
            elif path.suffix == ".pkl":
                with open(path, "rb") as f:
                    return pickle.load(f)

            else:
                # Try JSON by default
                with open(path, "r") as f:
                    return json.load(f)

        except Exception as e:
            logger.error(f"Failed to read checkpoint {filepath}: {e}")
            return None

    @staticmethod
    def save_memory_checkpoint(
        memory: MemoryManager,
        checkpoint_path: str,
        format: BackupFormat = BackupFormat.JSON,
        compress: bool = False,
    ) -> bool:
        """Save memory checkpoint to disk."""

        try:
            # Export memory data
            export_data = MemoryLoader._export_memory_data(memory)

            # Save in specified format
            path = Path(checkpoint_path)

            if format == BackupFormat.JSON_GZIP or compress:
                if not path.suffix == ".gz":
                    path = path.with_suffix(".json.gz")
                with gzip.open(path, "wt") as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format == BackupFormat.PICKLE:
                if not path.suffix == ".pkl":
                    path = path.with_suffix(".pkl")
                with open(path, "wb") as f:
                    pickle.dump(export_data, f)

            else:  # JSON
                if not path.suffix == ".json":
                    path = path.with_suffix(".json")
                with open(path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

            logger.info(f"💾 Memory checkpoint saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            return False

    @staticmethod
    def _export_memory_data(memory: MemoryManager) -> Dict[str, Any]:
        """Export memory data for backup."""

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            "episodic": [],
            "semantic": [],
            "vector": [],
        }

        # Export episodic memory
        for key, entry in memory.episodic.entries.items():
            export_data["episodic"].append(
                {
                    "key": key,
                    "content": entry.content,
                    "importance": entry.importance,
                    "tags": entry.tags,
                    "timestamp": (
                        entry.timestamp.isoformat() if entry.timestamp else None
                    ),
                }
            )

        # Export semantic memory
        for key, node in memory.semantic.knowledge_base.items():
            export_data["semantic"].append(node.to_dict())

        # Export vector memory
        for key, vector in memory.vector.vectors.items():
            export_data["vector"].append(
                {
                    "key": key,
                    "embedding": vector.tolist(),
                    "content": (
                        memory.episodic.entries.get(key, {}).content
                        if key in memory.episodic.entries
                        else None
                    ),
                }
            )

        return export_data

    def _load_to_memory(
        self,
        memory: MemoryManager,
        episodic_data: List[Dict],
        semantic_data: List[Dict],
        vector_data: List[Dict],
    ):
        """Load data into memory manager."""

        # Load episodic
        for item in episodic_data:
            key = item.get("key")
            if key:
                memory.episodic.store(
                    key=key,
                    content=item.get("content"),
                    importance=item.get("importance", 0.5),
                    tags=item.get("tags", []),
                    timestamp=(
                        datetime.fromisoformat(item["timestamp"])
                        if item.get("timestamp")
                        else None
                    ),
                )

        # Load semantic
        for item in semantic_data:
            key = item.get("key")
            if key:
                memory.semantic.store(
                    key=key,
                    content=item.get("content"),
                    category=item.get("category", "general"),
                    confidence=item.get("confidence", 0.8),
                    source=item.get("source"),
                    tags=item.get("tags", []),
                )

        # Load vector
        for item in vector_data:
            key = item.get("key")
            embedding = item.get("embedding")
            if key and embedding:
                memory.vector.store(
                    key=key, embedding=embedding, associated_content=item.get("content")
                )

    def _merge_data(
        self, db_data: List[Dict], checkpoint_data: List[Dict]
    ) -> List[Dict]:
        """Merge DB and checkpoint data (checkpoint takes precedence)."""

        # Create key-indexed dict
        data_dict = {item.get("key"): item for item in db_data if item.get("key")}

        # Update with checkpoint data
        for item in checkpoint_data:
            key = item.get("key")
            if key:
                data_dict[key] = item

        return list(data_dict.values())

    # --------------------------------------------------
    # 🔥 BACKUP SYSTEM
    # --------------------------------------------------

    @staticmethod
    def backup_all(
        memory: MemoryManager,
        checkpoint_path: Optional[str] = None,
        format: BackupFormat = BackupFormat.JSON,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create comprehensive backup (disk + DB sync).

        Args:
            memory: MemoryManager instance
            checkpoint_path: Custom checkpoint path
            format: Backup format
            description: Backup description

        Returns:
            Path to backup file if successful
        """

        loader = MemoryLoader()

        # Generate backup path if not provided
        if not checkpoint_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = loader.backup_dir / f"memory_backup_{timestamp}"

        # Save checkpoint
        if MemoryLoader.save_memory_checkpoint(memory, str(checkpoint_path), format):
            # Create backup info file
            backup_info = {
                "path": str(checkpoint_path),
                "timestamp": datetime.now().isoformat(),
                "format": format.value,
                "description": description,
                "stats": (
                    memory.get_stats().__dict__
                    if hasattr(memory.get_stats(), "__dict__")
                    else memory.get_stats()
                ),
            }

            info_path = Path(str(checkpoint_path) + ".info.json")
            with open(info_path, "w") as f:
                json.dump(backup_info, f, indent=2)

            logger.info(f"✅ Hybrid backup complete: {checkpoint_path}")
            return str(checkpoint_path)

        return None

    @staticmethod
    def restore_from_backup(
        backup_path: str, restore_to_db: bool = True, validate: bool = True
    ) -> bool:
        """
        Restore memory from backup file.

        Args:
            backup_path: Path to backup file
            restore_to_db: Whether to restore to MongoDB
            validate: Validate data integrity

        Returns:
            True if successful
        """

        loader = MemoryLoader()

        try:
            # Read backup
            data = loader._read_checkpoint_file(backup_path)
            if not data:
                logger.error("Failed to read backup file")
                return False

            # Validate
            if validate:
                loader._validate_data(data.get("episodic", []), "episodic")
                loader._validate_data(data.get("semantic", []), "semantic")
                loader._validate_data(data.get("vector", []), "vector")

            # Restore to MongoDB if requested
            if restore_to_db and loader._is_db_available():
                loader._restore_to_mongodb(data)

            logger.info(f"✅ Restored from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def _restore_to_mongodb(self, data: Dict[str, Any]):
        """Restore data to MongoDB."""

        # Clear existing data
        for memory_type in ["episodic", "semantic", "vector"]:
            mongo_client.delete_many(memory_type, {})

        # Restore episodic
        for item in data.get("episodic", []):
            if item.get("key"):
                mongo_client.save("episodic", item)

        # Restore semantic
        for item in data.get("semantic", []):
            if item.get("key"):
                mongo_client.save("semantic", item)

        # Restore vector
        for item in data.get("vector", []):
            if item.get("key"):
                mongo_client.save("vector", item)

        logger.info("MongoDB restored from backup")

    def list_backups(self, limit: int = 20) -> List[BackupInfo]:
        """List available backups."""

        backups = []

        # Find backup files
        backup_files = list(self.backup_dir.glob("memory_backup_*"))
        backup_files.extend(self.checkpoint_dir.glob("*.json"))
        backup_files.extend(self.checkpoint_dir.glob("*.json.gz"))
        backup_files.extend(self.checkpoint_dir.glob("*.pkl"))

        for backup_path in sorted(
            backup_files, key=lambda x: x.stat().st_mtime, reverse=True
        )[:limit]:
            try:
                # Determine format
                if backup_path.suffix == ".gz":
                    format = BackupFormat.JSON_GZIP
                elif backup_path.suffix == ".pkl":
                    format = BackupFormat.PICKLE
                else:
                    format = BackupFormat.JSON

                # Get metadata from info file
                info_path = backup_path.with_suffix(".info.json")
                if info_path.exists():
                    with open(info_path, "r") as f:
                        info = json.load(f)

                    backups.append(
                        BackupInfo(
                            path=str(backup_path),
                            size_bytes=backup_path.stat().st_size,
                            created_at=datetime.fromisoformat(
                                info.get("timestamp", datetime.now().isoformat())
                            ),
                            format=format,
                            episodic_count=info.get("stats", {}).get(
                                "episodic_count", 0
                            ),
                            semantic_count=info.get("stats", {}).get(
                                "semantic_count", 0
                            ),
                            vector_count=info.get("stats", {}).get("vector_count", 0),
                        )
                    )
                else:
                    # Read backup to get counts
                    data = self._read_checkpoint_file(str(backup_path))
                    if data:
                        backups.append(
                            BackupInfo(
                                path=str(backup_path),
                                size_bytes=backup_path.stat().st_size,
                                created_at=datetime.fromtimestamp(
                                    backup_path.stat().st_mtime
                                ),
                                format=format,
                                episodic_count=len(data.get("episodic", [])),
                                semantic_count=len(data.get("semantic", [])),
                                vector_count=len(data.get("vector", [])),
                            )
                        )

            except Exception as e:
                logger.debug(f"Failed to read backup info: {e}")

        return backups

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""

        checkpoints = list(self.checkpoint_dir.glob("*.json"))
        checkpoints.extend(self.checkpoint_dir.glob("*.json.gz"))
        checkpoints.extend(self.checkpoint_dir.glob("*.pkl"))

        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest)

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Delete old backups, keeping only the most recent."""

        backups = self.list_backups(limit=100)
        removed = 0

        for backup in backups[keep_count:]:
            try:
                path = Path(backup.path)
                if path.exists():
                    path.unlink()

                # Remove info file
                info_path = path.with_suffix(".info.json")
                if info_path.exists():
                    info_path.unlink()

                removed += 1
            except Exception as e:
                logger.debug(f"Failed to remove backup {backup.path}: {e}")

        if removed > 0:
            logger.info(f"🧹 Cleaned up {removed} old backups")

        return removed

    # --------------------------------------------------
    # 🔥 HYBRID BACKUP (DB + FILE)
    # --------------------------------------------------

    @staticmethod
    def incremental_backup(
        memory: MemoryManager, since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Create incremental backup of recent changes.

        Args:
            memory: MemoryManager instance
            since: Only backup changes since this time

        Returns:
            Backup statistics
        """

        loader = MemoryLoader()
        since = since or datetime.now() - timedelta(hours=24)

        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "since": since.isoformat(),
            "type": "incremental",
            "episodic": [],
            "semantic": [],
            "vector": [],
        }

        # Get recent episodic entries
        for key, entry in memory.episodic.entries.items():
            if entry.timestamp and entry.timestamp > since:
                backup_data["episodic"].append(
                    {
                        "key": key,
                        "content": entry.content,
                        "importance": entry.importance,
                        "tags": entry.tags,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                )

        # Get recent semantic entries
        for key, node in memory.semantic.knowledge_base.items():
            if node.updated_at > since:
                backup_data["semantic"].append(node.to_dict())

        # Save incremental backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = loader.backup_dir / f"incremental_backup_{timestamp}.json"

        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2, default=str)

        logger.info(
            f"💾 Incremental backup saved: {backup_path} ({len(backup_data['episodic'])} new items)"
        )

        return {
            "path": str(backup_path),
            "episodic_count": len(backup_data["episodic"]),
            "semantic_count": len(backup_data["semantic"]),
            "vector_count": len(backup_data["vector"]),
        }

    # --------------------------------------------------
    # 🔥 UTILITY METHODS
    # --------------------------------------------------

    def _is_db_available(self) -> bool:
        """Check if MongoDB is available."""
        return (
            mongo_client.enabled and mongo_client.status == ConnectionStatus.CONNECTED
        )

    def _validate_data(self, data: List[Dict], memory_type: str):
        """Validate data integrity."""

        for item in data:
            if "key" not in item:
                logger.warning(f"Invalid {memory_type} data: missing 'key' field")

            if memory_type == "episodic" and "content" not in item:
                logger.warning(
                    f"Invalid episodic data: missing 'content' for key {item.get('key')}"
                )

            elif memory_type == "semantic" and "content" not in item:
                logger.warning(
                    f"Invalid semantic data: missing 'content' for key {item.get('key')}"
                )

            elif memory_type == "vector" and "embedding" not in item:
                logger.warning(
                    f"Invalid vector data: missing 'embedding' for key {item.get('key')}"
                )

    def _record_load(self, result: LoadResult):
        """Record load operation in history."""

        self.load_history.append(result)
        if len(self.load_history) > self.max_history:
            self.load_history = self.load_history[-self.max_history :]

        # Update stats
        self.stats["total_loads"] += 1
        if result.success:
            self.stats["successful_loads"] += 1
        else:
            self.stats["failed_loads"] += 1

        total_time = (
            self.stats["avg_load_time_ms"] * (self.stats["successful_loads"] - 1)
            + result.load_time_ms
        )
        self.stats["avg_load_time_ms"] = total_time / max(
            1, self.stats["successful_loads"]
        )
        self.stats["last_load"] = result.timestamp.isoformat()

        # Trigger callbacks
        for callback in self._load_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Load callback error: {e}")

    def _auto_backup_if_needed(self, memory: MemoryManager):
        """Perform auto-backup if interval has passed."""

        if not self.auto_backup_enabled:
            return

        now = datetime.now()

        if self.last_auto_backup is None:
            self.last_auto_backup = now
            return

        hours_since = (now - self.last_auto_backup).total_seconds() / 3600

        if hours_since >= self.auto_backup_interval_hours:
            logger.info("Performing auto-backup...")
            self.backup_all(memory)
            self.last_auto_backup = now

    # --------------------------------------------------
    # 🔥 RESET & CLEAR
    # --------------------------------------------------

    @staticmethod
    def reset_memory(memory: MemoryManager, clear_db: bool = False) -> bool:
        """
        Reset memory system safely.

        Args:
            memory: MemoryManager instance
            clear_db: Whether to clear MongoDB as well

        Returns:
            True if successful
        """

        try:
            # Clear RAM
            memory.clear_all()

            # Clear MongoDB if requested
            if clear_db and MemoryLoader()._is_db_available():
                for memory_type in ["episodic", "semantic", "vector"]:
                    mongo_client.delete_many(memory_type, {})
                logger.info("MongoDB cleared")

            logger.warning("Memory system reset (RAM cleared)")
            return True

        except Exception as e:
            logger.error(f"Memory reset failed: {e}")
            return False

    @staticmethod
    def verify_integrity(memory: MemoryManager) -> Dict[str, Any]:
        """
        Verify memory integrity.

        Returns:
            Integrity report
        """

        issues = []

        # Check episodic memory
        try:
            episodic_count = len(memory.episodic.entries)
            for key, entry in memory.episodic.entries.items():
                if not key:
                    issues.append("Episodic entry has no key")
                if entry.content is None:
                    issues.append(f"Episodic entry {key} has no content")
        except Exception as e:
            issues.append(f"Episodic memory error: {e}")

        # Check semantic memory
        try:
            semantic_count = len(memory.semantic.knowledge_base)
            for key, node in memory.semantic.knowledge_base.items():
                if not key:
                    issues.append("Semantic entry has no key")
                if node.content is None:
                    issues.append(f"Semantic entry {key} has no content")
                if not 0 <= node.confidence <= 1:
                    issues.append(
                        f"Semantic entry {key} has invalid confidence: {node.confidence}"
                    )
        except Exception as e:
            issues.append(f"Semantic memory error: {e}")

        # Check vector memory
        try:
            vector_count = len(memory.vector.vectors)
            for key, vector in memory.vector.vectors.items():
                if not key:
                    issues.append("Vector entry has no key")
                if len(vector) != memory.vector.embedding_dim:
                    issues.append(f"Vector entry {key} has wrong dimension")
        except Exception as e:
            issues.append(f"Vector memory error: {e}")

        return {
            "integrity_ok": len(issues) == 0,
            "total_entries": episodic_count + semantic_count + vector_count,
            "episodic_count": episodic_count,
            "semantic_count": semantic_count,
            "vector_count": vector_count,
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
        }

    # --------------------------------------------------
    # 🔥 CALLBACKS
    # --------------------------------------------------

    def add_load_callback(self, callback: Callable[[LoadResult], None]):
        """Add callback for load operations."""
        self._load_callbacks.append(callback)

    def remove_load_callback(self, callback: Callable[[LoadResult], None]):
        """Remove load callback."""
        if callback in self._load_callbacks:
            self._load_callbacks.remove(callback)

    # --------------------------------------------------
    # 🔥 STATISTICS
    # --------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""

        return {
            **self.stats,
            "load_history": [r.to_dict() for r in self.load_history[-10:]],
            "backup_count": len(self.list_backups()),
            "auto_backup_enabled": self.auto_backup_enabled,
            "auto_backup_interval_hours": self.auto_backup_interval_hours,
            "last_auto_backup": (
                self.last_auto_backup.isoformat() if self.last_auto_backup else None
            ),
            "checkpoint_dir": str(self.checkpoint_dir),
            "backup_dir": str(self.backup_dir),
        }

    def get_load_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get load operation history."""

        return [r.to_dict() for r in self.load_history[-limit:]]


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def quick_load_memory(
    use_db: bool = True, checkpoint_path: Optional[str] = None
) -> MemoryManager:
    """Quick load memory with defaults."""

    return MemoryLoader.initialize_memory(
        use_db=use_db,
        checkpoint_path=checkpoint_path,
        priority=LoadPriority.DB_FIRST,
        auto_backup=True,
    )


def quick_backup(
    memory: MemoryManager, description: Optional[str] = None
) -> Optional[str]:
    """Quick backup helper."""

    return MemoryLoader.backup_all(memory, description=description)


def quick_restore(backup_path: str, restore_to_db: bool = True) -> bool:
    """Quick restore helper."""

    return MemoryLoader.restore_from_backup(backup_path, restore_to_db)


def get_memory_health() -> Dict[str, Any]:
    """Get overall memory health status."""

    loader = MemoryLoader()

    return {
        "db_available": loader._is_db_available(),
        "latest_checkpoint": loader.get_latest_checkpoint(),
        "backup_count": len(loader.list_backups()),
        "loader_stats": loader.get_stats(),
        "auto_backup_enabled": loader.auto_backup_enabled,
    }


__all__ = [
    "MemoryLoader",
    "LoadPriority",
    "BackupFormat",
    "LoadResult",
    "BackupInfo",
    "quick_load_memory",
    "quick_backup",
    "quick_restore",
    "get_memory_health",
]
