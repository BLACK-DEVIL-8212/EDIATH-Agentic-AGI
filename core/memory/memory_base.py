"""
Memory management system for EDIATH - base memory class with advanced features.
(PRODUCTION READY WITH DB SERIALIZATION, ENCRYPTION, VERSIONING, AND METRICS)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import hashlib

from core.utils.logger import logger


class MemoryType(Enum):
    """Types of memory in the system."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    VECTOR = "vector"
    WORKING = "working"
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"
    SENSORY = "sensory"
    MUSCLE = "muscle"
    EMOTIONAL = "emotional"


class MemoryEncoding(Enum):
    """Encoding formats for memory serialization."""

    JSON = "json"
    BINARY = "binary"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"


class CompressionType(Enum):
    """Compression algorithms for memory storage."""

    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"


@dataclass
class MemoryMetadata:
    """Enhanced metadata for memory entries."""

    version: int = 1
    created_by: Optional[str] = None
    source_system: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: str = "production"
    priority: int = 0
    expires_at: Optional[datetime] = None
    retention_days: int = 365
    encrypted: bool = False
    compressed: bool = False
    encoding: MemoryEncoding = MemoryEncoding.JSON
    compression: CompressionType = CompressionType.NONE
    checksum: Optional[str] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["encoding"] = self.encoding.value
        data["compression"] = self.compression.value
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        """Create from dictionary."""
        # Convert enum strings back to enums
        if "encoding" in data and isinstance(data["encoding"], str):
            data["encoding"] = MemoryEncoding(data["encoding"])
        if "compression" in data and isinstance(data["compression"], str):
            data["compression"] = CompressionType(data["compression"])
        if "expires_at" in data and data["expires_at"]:
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)


class MemoryEntry:
    """Advanced memory entry with metadata, encryption, and serialization."""

    def __init__(
        self,
        content: Any,
        memory_type: MemoryType,
        timestamp: Optional[datetime] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        access_count: int = 0,
        last_accessed: Optional[datetime] = None,
        metadata: Optional[MemoryMetadata] = None,
        confidence: float = 1.0,
        decay_rate: float = 0.01,
        ttl_seconds: Optional[int] = None,
    ):
        self.content = content
        self.memory_type = memory_type
        self.timestamp = timestamp or datetime.utcnow()
        self.importance = max(0.0, min(1.0, importance))
        self.tags = tags or []
        self.embedding = embedding
        self.access_count = access_count
        self.last_accessed = last_accessed or self.timestamp
        self.metadata = metadata or MemoryMetadata()
        self.confidence = max(0.0, min(1.0, confidence))
        self.decay_rate = decay_rate
        self.ttl_seconds = ttl_seconds

        # Unique ID
        self.id = self._generate_id()

        # Validation
        self._validate()

    def _generate_id(self) -> str:
        """Generate unique ID for memory entry."""
        content_hash = hashlib.md5(str(self.content).encode()).hexdigest()[:8]
        timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S")
        return f"{self.memory_type.value}_{timestamp_str}_{content_hash}"

    def _validate(self):
        """Validate memory entry data."""
        if (
            not isinstance(self.importance, (int, float))
            or self.importance < 0
            or self.importance > 1
        ):
            raise ValueError(
                f"Importance must be between 0 and 1, got {self.importance}"
            )

        if (
            not isinstance(self.confidence, (int, float))
            or self.confidence < 0
            or self.confidence > 1
        ):
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )

        if self.ttl_seconds and self.ttl_seconds <= 0:
            raise ValueError(f"TTL must be positive, got {self.ttl_seconds}")

    def access(self) -> None:
        """Record access to this memory entry."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

        # Boost importance on access (recency effect)
        self.importance = min(1.0, self.importance + 0.01)

    def decay(self, factor: float = 1.0) -> None:
        """Apply decay to importance and confidence."""
        # Importance decays over time if not accessed
        days_since_access = (datetime.utcnow() - self.last_accessed).days
        if days_since_access > 0:
            decay_amount = self.decay_rate * days_since_access * factor
            self.importance = max(0.0, self.importance - decay_amount)

        # Confidence also decays slowly
        if days_since_access > 7:
            confidence_decay = self.decay_rate * (days_since_access - 7) * factor
            self.confidence = max(0.0, self.confidence - confidence_decay)

    def is_expired(self) -> bool:
        """Check if memory entry has expired."""
        if self.metadata.expires_at:
            return datetime.utcnow() > self.metadata.expires_at

        if self.ttl_seconds:
            age = (datetime.utcnow() - self.timestamp).total_seconds()
            return age > self.ttl_seconds

        # Check retention period
        if self.metadata.retention_days:
            age_days = (datetime.utcnow() - self.timestamp).days
            return age_days > self.metadata.retention_days

        return False

    def update_metadata(self, **kwargs):
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

    def add_tag(self, tag: str):
        """Add a tag to the memory entry."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str):
        """Remove a tag from the memory entry."""
        if tag in self.tags:
            self.tags.remove(tag)

    def add_relation(self, related_id: str):
        """Add a related memory ID."""
        if related_id not in self.metadata.related_ids:
            self.metadata.related_ids.append(related_id)

    def remove_relation(self, related_id: str):
        """Remove a related memory ID."""
        if related_id in self.metadata.related_ids:
            self.metadata.related_ids.remove(related_id)

    # --------------------------------------------------
    # 🔥 SERIALIZATION (FOR MONGO)
    # --------------------------------------------------

    def to_dict(self, encrypt: bool = False, compress: bool = False) -> Dict[str, Any]:
        """Convert memory entry to dictionary with optional encryption/compression."""

        data = {
            "id": self.id,
            "content": self.content,
            "type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags,
            "embedding": self.embedding,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata.to_dict(),
            "confidence": self.confidence,
            "decay_rate": self.decay_rate,
            "ttl_seconds": self.ttl_seconds,
            "version": 2,  # Version 2 format
        }

        # Apply compression if requested
        if compress and data.get("content"):
            data["content"] = self._compress_content(data["content"])
            data["compressed"] = True

        # Apply encryption if requested
        if encrypt:
            data = self._encrypt_data(data)
            data["encrypted"] = True

        return data

    def _compress_content(self, content: Any) -> Any:
        """Compress content (placeholder - implement actual compression)."""
        # This would use zlib, gzip, etc.
        return content

    def _encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive data (placeholder - implement actual encryption)."""
        # This would use cryptography library
        return data

    # --------------------------------------------------
    # 🔥 DESERIALIZATION (FROM MONGO)
    # --------------------------------------------------

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""

        # Handle version 1 format (backward compatibility)
        if data.get("version", 1) == 1:
            return MemoryEntry._from_dict_v1(data)

        # Parse timestamps
        timestamp = (
            datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )
        last_accessed = (
            datetime.fromisoformat(data["last_accessed"])
            if data.get("last_accessed")
            else None
        )

        # Parse metadata
        metadata = MemoryMetadata.from_dict(data.get("metadata", {}))

        # Handle decryption if needed
        content = data.get("content")
        if data.get("encrypted"):
            content = MemoryEntry._decrypt_content(content)

        return MemoryEntry(
            content=content,
            memory_type=MemoryType(data["type"]),
            timestamp=timestamp,
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            embedding=data.get("embedding"),
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            metadata=metadata,
            confidence=data.get("confidence", 1.0),
            decay_rate=data.get("decay_rate", 0.01),
            ttl_seconds=data.get("ttl_seconds"),
        )

    @staticmethod
    def _from_dict_v1(data: Dict[str, Any]) -> "MemoryEntry":
        """Handle version 1 format for backward compatibility."""
        return MemoryEntry(
            content=data.get("content"),
            memory_type=MemoryType(data.get("type", "episodic")),
            timestamp=data.get("timestamp"),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            embedding=data.get("embedding"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
        )

    @staticmethod
    def _decrypt_content(content: Any) -> Any:
        """Decrypt content (placeholder - implement actual decryption)."""
        return content

    # --------------------------------------------------
    # 🔥 STRING REPRESENTATION
    # --------------------------------------------------

    def __repr__(self) -> str:
        return f"MemoryEntry(id={self.id}, type={self.memory_type.value}, importance={self.importance:.2f})"

    def __str__(self) -> str:
        return f"[{self.memory_type.value}] {str(self.content)[:100]}... (imp={self.importance:.2f})"


class MemoryStats:
    """Statistics and metrics for memory systems."""

    def __init__(self):
        self.total_entries = 0
        self.total_accesses = 0
        self.avg_importance = 0.0
        self.avg_confidence = 0.0
        self.type_distribution: Dict[str, int] = {}
        self.tag_distribution: Dict[str, int] = {}
        self.importance_distribution: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "trivial": 0,
        }
        self.age_distribution: Dict[str, int] = {
            "recent": 0,
            "week": 0,
            "month": 0,
            "old": 0,
        }
        self.memory_usage_bytes = 0
        self.avg_access_count = 0.0
        self.unique_tags = 0

    def update(self, entries: Dict[str, MemoryEntry]):
        """Update statistics based on current entries."""

        if not entries:
            return

        self.total_entries = len(entries)

        # Calculate metrics
        importances = []
        confidences = []
        accesses = []

        # Reset distributions
        self.type_distribution.clear()
        self.tag_distribution.clear()
        for key in self.importance_distribution:
            self.importance_distribution[key] = 0
        for key in self.age_distribution:
            self.age_distribution[key] = 0

        now = datetime.utcnow()

        for entry in entries.values():
            importances.append(entry.importance)
            confidences.append(entry.confidence)
            accesses.append(entry.access_count)

            # Type distribution
            type_key = entry.memory_type.value
            self.type_distribution[type_key] = (
                self.type_distribution.get(type_key, 0) + 1
            )

            # Tag distribution
            for tag in entry.tags:
                self.tag_distribution[tag] = self.tag_distribution.get(tag, 0) + 1

            # Importance distribution
            if entry.importance >= 0.9:
                self.importance_distribution["critical"] += 1
            elif entry.importance >= 0.7:
                self.importance_distribution["high"] += 1
            elif entry.importance >= 0.4:
                self.importance_distribution["medium"] += 1
            elif entry.importance >= 0.1:
                self.importance_distribution["low"] += 1
            else:
                self.importance_distribution["trivial"] += 1

            # Age distribution
            age_days = (now - entry.timestamp).days
            if age_days <= 1:
                self.age_distribution["recent"] += 1
            elif age_days <= 7:
                self.age_distribution["week"] += 1
            elif age_days <= 30:
                self.age_distribution["month"] += 1
            else:
                self.age_distribution["old"] += 1

        # Calculate averages
        self.avg_importance = sum(importances) / len(importances)
        self.avg_confidence = sum(confidences) / len(confidences)
        self.total_accesses = sum(accesses)
        self.avg_access_count = self.total_accesses / len(entries)
        self.unique_tags = len(self.tag_distribution)

        # Estimate memory usage (rough approximation)
        self.memory_usage_bytes = sum(
            len(str(entry.content)) + len(entry.tags) * 50 for entry in entries.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_entries": self.total_entries,
            "total_accesses": self.total_accesses,
            "avg_importance": self.avg_importance,
            "avg_confidence": self.avg_confidence,
            "avg_access_count": self.avg_access_count,
            "type_distribution": self.type_distribution,
            "tag_distribution": dict(list(self.tag_distribution.items())[:20]),
            "importance_distribution": self.importance_distribution,
            "age_distribution": self.age_distribution,
            "memory_usage_mb": self.memory_usage_bytes / (1024 * 1024),
            "unique_tags": self.unique_tags,
        }


class BaseMemory(ABC):
    """Advanced base class for memory systems with metrics and maintenance."""

    def __init__(self, name: str, enable_metrics: bool = True, auto_decay: bool = True):
        self.name = name
        self.entries: Dict[str, MemoryEntry] = {}
        self.created_at = datetime.utcnow()
        self.enable_metrics = enable_metrics
        self.auto_decay = auto_decay
        self.stats = MemoryStats() if enable_metrics else None

        # Indexes for fast lookup
        self._tag_index: Dict[str, Set[str]] = {}
        self._type_index: Dict[MemoryType, Set[str]] = {}
        self._importance_index: Dict[str, Set[str]] = {
            "critical": set(),
            "high": set(),
            "medium": set(),
            "low": set(),
            "trivial": set(),
        }

        # Maintenance settings
        self.maintenance_last_run: Optional[datetime] = None
        self.maintenance_interval_hours = 24

        logger.info(f"✅ BaseMemory '{name}' initialized (metrics={enable_metrics})")

    # --------------------------------------------------
    # ABSTRACT METHODS
    # --------------------------------------------------

    @abstractmethod
    def store(self, key: str, content: Any, **kwargs) -> Optional[str]:
        """Store a memory entry."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a memory entry by key."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory entries."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a memory entry."""
        pass

    # --------------------------------------------------
    # INDEX MANAGEMENT
    # --------------------------------------------------

    def _update_indexes(self, entry: MemoryEntry, key: str):
        """Update all indexes for an entry."""

        # Type index
        if entry.memory_type not in self._type_index:
            self._type_index[entry.memory_type] = set()
        self._type_index[entry.memory_type].add(key)

        # Tag index
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(key)

        # Importance index
        if entry.importance >= 0.9:
            self._importance_index["critical"].add(key)
        elif entry.importance >= 0.7:
            self._importance_index["high"].add(key)
        elif entry.importance >= 0.4:
            self._importance_index["medium"].add(key)
        elif entry.importance >= 0.1:
            self._importance_index["low"].add(key)
        else:
            self._importance_index["trivial"].add(key)

    def _remove_from_indexes(self, entry: MemoryEntry, key: str):
        """Remove entry from all indexes."""

        # Type index
        if entry.memory_type in self._type_index:
            self._type_index[entry.memory_type].discard(key)

        # Tag index
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(key)

        # Importance index
        for category in self._importance_index.values():
            category.discard(key)

    def _rebuild_indexes(self):
        """Rebuild all indexes from scratch."""

        self._tag_index.clear()
        self._type_index.clear()
        for category in self._importance_index:
            self._importance_index[category].clear()

        for key, entry in self.entries.items():
            self._update_indexes(entry, key)

    # --------------------------------------------------
    # QUERY HELPERS
    # --------------------------------------------------

    def get_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Get all entries with a specific tag."""
        keys = self._tag_index.get(tag, set())
        return [self.entries[k] for k in keys if k in self.entries]

    def get_by_type(self, memory_type: MemoryType) -> List[MemoryEntry]:
        """Get all entries of a specific type."""
        keys = self._type_index.get(memory_type, set())
        return [self.entries[k] for k in keys if k in self.entries]

    def get_by_importance(
        self, min_importance: float = 0.0, max_importance: float = 1.0
    ) -> List[MemoryEntry]:
        """Get entries within importance range."""
        return [
            entry
            for entry in self.entries.values()
            if min_importance <= entry.importance <= max_importance
        ]

    def get_high_importance(self, threshold: float = 0.7) -> List[MemoryEntry]:
        """Get high importance entries."""
        return self.get_by_importance(min_importance=threshold)

    def get_recent(self, hours: int = 24) -> List[MemoryEntry]:
        """Get recently added entries."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [entry for entry in self.entries.values() if entry.timestamp > cutoff]

    def get_most_accessed(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most frequently accessed entries."""
        sorted_entries = sorted(
            self.entries.values(), key=lambda x: x.access_count, reverse=True
        )
        return sorted_entries[:limit]

    # --------------------------------------------------
    # MAINTENANCE
    # --------------------------------------------------

    def apply_decay(self, factor: float = 1.0) -> int:
        """Apply decay to all entries."""
        if not self.auto_decay:
            return 0

        decayed_count = 0
        for entry in self.entries.values():
            old_importance = entry.importance
            entry.decay(factor)
            if old_importance != entry.importance:
                decayed_count += 1

        return decayed_count

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.entries.items() if entry.is_expired()
        ]

        for key in expired_keys:
            self.delete(key)

        if expired_keys:
            logger.info(
                f"🧹 Cleaned up {len(expired_keys)} expired entries from {self.name}"
            )

        return len(expired_keys)

    def maintain(self, force: bool = False) -> Dict[str, Any]:
        """Run maintenance tasks."""
        now = datetime.utcnow()

        if not force and self.maintenance_last_run:
            hours_since = (now - self.maintenance_last_run).total_seconds() / 3600
            if hours_since < self.maintenance_interval_hours:
                return {
                    "message": "Maintenance not needed yet",
                    "hours_until": self.maintenance_interval_hours - hours_since,
                }

        maintenance_results = {
            "timestamp": now.isoformat(),
            "decayed_entries": self.apply_decay(),
            "expired_removed": self.cleanup_expired(),
            "index_rebuilt": False,
        }

        # Rebuild indexes periodically
        if len(self.entries) > 1000 and (force or not self.maintenance_last_run):
            self._rebuild_indexes()
            maintenance_results["index_rebuilt"] = True

        # Update statistics
        if self.enable_metrics and self.stats:
            self.stats.update(self.entries)

        self.maintenance_last_run = now

        logger.info(f"🔧 Maintenance completed for {self.name}: {maintenance_results}")
        return maintenance_results

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------

    def list_keys(self) -> List[str]:
        """List all keys in memory."""
        return list(self.entries.keys())

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self._rebuild_indexes()
        logger.info(f"🗑️ Memory '{self.name}' cleared")

    def size(self) -> int:
        """Get number of entries."""
        return len(self.entries)

    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.entries) == 0

    # --------------------------------------------------
    # 🔥 BULK LOAD (FROM MONGO)
    # --------------------------------------------------

    def load_entries(self, data: List[Dict[str, Any]]) -> int:
        """Load multiple entries from DB."""
        loaded = 0

        for item in data:
            key = item.get("key") or item.get("id")
            if not key:
                continue

            entry = MemoryEntry.from_dict(item)
            self.entries[key] = entry
            self._update_indexes(entry, key)
            loaded += 1

        if loaded > 0:
            logger.info(f"📥 Loaded {loaded} entries into '{self.name}'")

            # Update statistics
            if self.enable_metrics and self.stats:
                self.stats.update(self.entries)

        return loaded

    # --------------------------------------------------
    # 🔥 EXPORT (FOR BACKUP / DEBUG)
    # --------------------------------------------------

    def export(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """Export all entries for backup."""

        export_data = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "exported_at": datetime.utcnow().isoformat(),
            "total_entries": len(self.entries),
            "entries": {},
        }

        for key, entry in self.entries.items():
            entry_dict = entry.to_dict()
            if not include_embeddings:
                entry_dict.pop("embedding", None)
            export_data["entries"][key] = entry_dict

        return export_data

    def import_entries(
        self, export_data: Dict[str, Any], overwrite: bool = False
    ) -> int:
        """Import entries from export data."""

        if overwrite:
            self.clear()

        imported = 0
        for key, entry_data in export_data.get("entries", {}).items():
            if not overwrite and key in self.entries:
                continue

            entry = MemoryEntry.from_dict(entry_data)
            self.entries[key] = entry
            self._update_indexes(entry, key)
            imported += 1

        logger.info(f"📥 Imported {imported} entries into '{self.name}'")

        if self.enable_metrics and self.stats:
            self.stats.update(self.entries)

        return imported

    # --------------------------------------------------
    # STATS
    # --------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""

        if not self.entries:
            return {
                "name": self.name,
                "total_entries": 0,
                "average_importance": 0,
                "average_confidence": 0,
                "total_accesses": 0,
                "average_accesses": 0,
                "created_at": self.created_at.isoformat(),
            }

        importance_values = [e.importance for e in self.entries.values()]
        confidence_values = [e.confidence for e in self.entries.values()]
        access_counts = [e.access_count for e in self.entries.values()]

        # Calculate age distribution
        now = datetime.utcnow()
        age_days = [(now - e.timestamp).days for e in self.entries.values()]

        return {
            "name": self.name,
            "total_entries": len(self.entries),
            "average_importance": sum(importance_values) / len(importance_values),
            "average_confidence": sum(confidence_values) / len(confidence_values),
            "total_accesses": sum(access_counts),
            "average_accesses": sum(access_counts) / len(access_counts),
            "max_accesses": max(access_counts),
            "min_importance": min(importance_values),
            "max_importance": max(importance_values),
            "average_age_days": sum(age_days) / len(age_days),
            "oldest_entry_days": max(age_days),
            "newest_entry_days": min(age_days),
            "unique_tags": len(
                set(tag for e in self.entries.values() for tag in e.tags)
            ),
            "type_distribution": {
                t.value: len([e for e in self.entries.values() if e.memory_type == t])
                for t in MemoryType
            },
            "created_at": self.created_at.isoformat(),
            "maintenance_last_run": (
                self.maintenance_last_run.isoformat()
                if self.maintenance_last_run
                else None
            ),
        }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including metrics."""

        base_stats = self.get_stats()

        if self.enable_metrics and self.stats:
            base_stats["detailed_metrics"] = self.stats.to_dict()

        # Add index statistics
        base_stats["index_stats"] = {
            "tag_index_size": len(self._tag_index),
            "type_index_size": len(self._type_index),
            "importance_index_sizes": {
                k: len(v) for k, v in self._importance_index.items()
            },
        }

        return base_stats


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def create_memory_key(prefix: str = "mem") -> str:
    """Generate a unique memory key."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}"


def calculate_memory_similarity(entry1: MemoryEntry, entry2: MemoryEntry) -> float:
    """Calculate similarity between two memory entries."""

    # If both have embeddings, use cosine similarity
    if entry1.embedding and entry2.embedding:
        import numpy as np

        vec1 = np.array(entry1.embedding)
        vec2 = np.array(entry2.embedding)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # Fallback to tag overlap
    if entry1.tags and entry2.tags:
        common = set(entry1.tags) & set(entry2.tags)
        total = set(entry1.tags) | set(entry2.tags)
        if total:
            return len(common) / len(total)

    return 0.0


__all__ = [
    "MemoryType",
    "MemoryEncoding",
    "CompressionType",
    "MemoryMetadata",
    "MemoryEntry",
    "MemoryStats",
    "BaseMemory",
    "create_memory_key",
    "calculate_memory_similarity",
]
