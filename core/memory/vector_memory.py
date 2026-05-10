"""
Vector Memory - stores embeddings for similarity search with advanced indexing,
batch operations, approximate nearest neighbors, and real-time updates.
(PRODUCTION READY WITH MONGODB INTEGRATION)
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import threading
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, deque

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from annoy import AnnoyIndex

    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False

from .memory_base import BaseMemory, MemoryEntry, MemoryType
from .mongo_client import mongo_client
from ..utils.logger import logger


@dataclass
class VectorMetadata:
    """Enhanced metadata for vector entries."""

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    source: Optional[str] = None
    model_version: Optional[str] = None
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorIndexType:
    """Types of vector indexes."""

    FLAT = "flat"  # Brute force
    ANNOY = "annoy"  # Approximate nearest neighbors
    SKLEARN = "sklearn"  # Scikit-learn KDTree
    FAISS = "faiss"  # Facebook AI Similarity Search (if available)


class VectorMemory(BaseMemory):
    """Advanced vector memory with multiple index types and batch operations."""

    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = VectorIndexType.FLAT,
        use_annoy: bool = False,
        annoy_trees: int = 10,
        cache_size: int = 1000,
        enable_compression: bool = False,
        similarity_metric: str = "cosine",  # cosine, euclidean, dot
    ):
        """Initialize advanced vector memory."""
        super().__init__("vector_memory")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_annoy = use_annoy and ANNOY_AVAILABLE
        self.annoy_trees = annoy_trees
        self.cache_size = cache_size
        self.enable_compression = enable_compression
        self.similarity_metric = similarity_metric

        # Core storage
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, VectorMetadata] = {}

        # Index structures
        self._flat_index: Optional[np.ndarray] = None
        self._annoy_index = None
        self._sklearn_index = None
        self._index_keys: List[str] = []
        self._index_needs_rebuild = True

        # Caching
        self._cache: Dict[str, np.ndarray] = {}
        self._access_history: deque = deque(maxlen=1000)

        # Locks
        self._vector_lock = threading.RLock()
        self._index_lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_vectors": 0,
            "total_searches": 0,
            "avg_search_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "index_rebuilds": 0,
            "last_rebuild": None,
        }

        # Batch processing
        self._batch_buffer: List[Tuple[str, np.ndarray, Dict]] = []
        self._batch_size = 100
        self._batch_thread: Optional[threading.Thread] = None

        # Callbacks
        self._callbacks: List[Callable[[str, np.ndarray], None]] = []

        # Query cache
        self._query_cache: Dict[str, Tuple[datetime, List[Dict]]] = {}
        self._query_cache_ttl = 60

        # 🔥 LOAD FROM DB
        self._load_from_db()

        # Build index if needed
        if self.index_type != VectorIndexType.FLAT:
            self._rebuild_index()

        logger.info(
            f"✅ VectorMemory initialized (dim={embedding_dim}, index={index_type}, metric={similarity_metric})"
        )

    # --------------------------------------------------
    # LOAD FROM MONGO
    # --------------------------------------------------
    def _load_from_db(self):
        """Load all vectors from MongoDB with metadata."""
        try:
            docs = mongo_client.load_all("vector")

            for doc in docs:
                key = doc.get("key")
                embedding = doc.get("embedding")
                metadata = doc.get("metadata", {})

                if not embedding or not key:
                    continue

                vector = np.array(embedding, dtype=np.float32)

                # Normalize if using cosine similarity
                if self.similarity_metric == "cosine":
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm

                # Create entry
                entry = MemoryEntry.from_dict(doc)
                self.entries[key] = entry
                self.vectors[key] = vector

                # Load metadata
                self.metadata[key] = VectorMetadata(
                    created_at=(
                        datetime.fromisoformat(
                            metadata.get("created_at", datetime.now().isoformat())
                        )
                        if metadata.get("created_at")
                        else datetime.now()
                    ),
                    updated_at=(
                        datetime.fromisoformat(
                            metadata.get("updated_at", datetime.now().isoformat())
                        )
                        if metadata.get("updated_at")
                        else datetime.now()
                    ),
                    access_count=metadata.get("access_count", 0),
                    source=metadata.get("source"),
                    model_version=metadata.get("model_version"),
                    confidence=metadata.get("confidence", 1.0),
                    tags=metadata.get("tags", []),
                    metadata=metadata.get("metadata", {}),
                )

            self.stats["total_vectors"] = len(self.vectors)

            logger.info(
                f"📚 Vector memory loaded from DB ({len(self.vectors)} vectors)"
            )

        except Exception as e:
            logger.error(f"Vector DB load failed: {e}")

    # --------------------------------------------------
    # STORE
    # --------------------------------------------------
    def store(
        self,
        key: str,
        embedding: Union[List[float], np.ndarray],
        associated_content: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        model_version: Optional[str] = None,
        confidence: float = 1.0,
        normalize: bool = True,
    ) -> bool:
        """Store a vector embedding with enhanced metadata."""

        try:
            # Validate and convert embedding
            if isinstance(embedding, list):
                embedding_array = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                embedding_array = embedding.copy().astype(np.float32)
            else:
                raise TypeError("Embedding must be list or numpy array")

            # Check dimension
            if len(embedding_array) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {len(embedding_array)} != {self.embedding_dim}"
                )

            # Normalize if requested
            if normalize and self.similarity_metric == "cosine":
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding_array = embedding_array / norm

            # Create metadata
            vector_metadata = VectorMetadata(
                source=source,
                model_version=model_version,
                confidence=confidence,
                tags=tags or [],
                metadata=metadata or {},
            )

            # Create entry
            entry = MemoryEntry(
                content=associated_content or embedding_array.tolist(),
                memory_type=MemoryType.VECTOR,
                embedding=embedding_array.tolist(),
                tags=tags or [],
            )

            with self._vector_lock:
                # Store in memory
                self.entries[key] = entry
                self.vectors[key] = embedding_array
                self.metadata[key] = vector_metadata

                # Update cache
                self._update_cache(key, embedding_array)

                # Mark index for rebuild
                self._index_needs_rebuild = True

            # Prepare data for MongoDB
            db_doc = {
                "key": key,
                "embedding": embedding_array.tolist(),
                "content": associated_content,
                "tags": tags or [],
                "type": "vector",
                "metadata": {
                    "created_at": vector_metadata.created_at.isoformat(),
                    "updated_at": vector_metadata.updated_at.isoformat(),
                    "access_count": vector_metadata.access_count,
                    "source": source,
                    "model_version": model_version,
                    "confidence": confidence,
                    "tags": tags or [],
                    "metadata": metadata or {},
                },
            }

            # 🔥 SAVE TO MONGO (async)
            mongo_client.save("vector", db_doc)

            # Update stats
            self.stats["total_vectors"] = len(self.vectors)

            # Trigger callbacks
            self._trigger_callbacks(key, embedding_array)

            logger.debug(
                f"📝 Vector stored: {key} (dim={self.embedding_dim}, confidence={confidence})"
            )
            return True

        except Exception as e:
            logger.error(f"Vector store error: {e}")
            return False

    def store_batch(
        self,
        items: List[
            Tuple[
                str, Union[List[float], np.ndarray], Optional[Any], Optional[List[str]]
            ]
        ],
    ) -> int:
        """Store multiple vectors in batch."""
        success_count = 0

        for item in items:
            key = item[0]
            embedding = item[1]
            content = item[2] if len(item) > 2 else None
            tags = item[3] if len(item) > 3 else None

            if self.store(key, embedding, content, tags):
                success_count += 1

        # Rebuild index after batch insert
        if success_count > 0:
            self._rebuild_index()

        return success_count

    # --------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------
    def retrieve(self, key: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve vector and metadata."""

        # Check cache
        if use_cache and key in self._cache:
            self.stats["cache_hits"] += 1
            self._update_access_metadata(key)
            return {
                "embedding": self._cache[key].tolist(),
                "content": self.entries[key].content if key in self.entries else None,
                "metadata": self._get_metadata_dict(key),
            }

        self.stats["cache_misses"] += 1

        with self._vector_lock:
            if key in self.entries:
                self._update_access_metadata(key)
                entry = self.entries[key]

                return {
                    "embedding": entry.embedding,
                    "content": entry.content,
                    "metadata": self._get_metadata_dict(key),
                }

        # 🔥 FALLBACK DB
        doc = mongo_client.load("vector", key)
        if doc:
            return {
                "embedding": doc.get("embedding"),
                "content": doc.get("content"),
                "metadata": doc.get("metadata", {}),
            }

        return None

    def retrieve_batch(self, keys: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve multiple vectors."""
        results = {}
        for key in keys:
            results[key] = self.retrieve(key)
        return results

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 10,
        threshold: float = 0.0,
        filter_tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors with filtering."""

        import time

        start_time = time.time()

        # Check query cache
        cache_key = self._get_query_cache_key(
            query_vector, limit, threshold, filter_tags, min_confidence
        )
        if use_cache and cache_key in self._query_cache:
            cached_time, cached_result = self._query_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._query_cache_ttl:
                self.stats["avg_search_time_ms"] = (
                    self.stats["avg_search_time_ms"] * self.stats["total_searches"] + 0
                ) / (self.stats["total_searches"] + 1)
                return cached_result

        try:
            # Convert query vector
            if isinstance(query_vector, list):
                query = np.array(query_vector, dtype=np.float32)
            else:
                query = query_vector.copy().astype(np.float32)

            # Validate dimension
            if len(query) != self.embedding_dim:
                raise ValueError(
                    f"Query vector dimension mismatch: {len(query)} != {self.embedding_dim}"
                )

            # Normalize
            if self.similarity_metric == "cosine":
                norm = np.linalg.norm(query)
                if norm > 0:
                    query = query / norm

            # Perform search based on index type
            if (
                self.index_type == VectorIndexType.ANNOY
                and self.use_annoy
                and self._annoy_index
            ):
                results = self._search_annoy(query, limit)
            elif (
                self.index_type == VectorIndexType.SKLEARN
                and SKLEARN_AVAILABLE
                and self._sklearn_index
            ):
                results = self._search_sklearn(query, limit)
            else:
                results = self._search_flat(query, limit)

            # Apply filters
            filtered_results = []
            for result in results:
                key = result["key"]

                # Check confidence threshold
                if (
                    key in self.metadata
                    and self.metadata[key].confidence < min_confidence
                ):
                    continue

                # Check tags filter
                if filter_tags and key in self.metadata:
                    if not any(tag in self.metadata[key].tags for tag in filter_tags):
                        continue

                # Check similarity threshold
                if result["similarity"] >= threshold:
                    # Add metadata to result
                    result["metadata"] = self._get_metadata_dict(key)
                    result["content"] = (
                        self.entries[key].content if key in self.entries else None
                    )
                    filtered_results.append(result)

            # Update stats
            self.stats["total_searches"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["avg_search_time_ms"] = (
                self.stats["avg_search_time_ms"] * (self.stats["total_searches"] - 1)
                + elapsed_ms
            ) / self.stats["total_searches"]

            # Cache results
            if use_cache:
                self._query_cache[cache_key] = (datetime.now(), filtered_results)
                self._clean_query_cache()

            return filtered_results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _search_flat(self, query: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Brute force similarity search."""
        similarities = []

        with self._vector_lock:
            for key, vector in self.vectors.items():
                if self.similarity_metric == "cosine":
                    similarity = float(np.dot(query, vector))
                elif self.similarity_metric == "euclidean":
                    # Convert euclidean distance to similarity
                    distance = np.linalg.norm(query - vector)
                    similarity = 1.0 / (1.0 + distance)
                else:  # dot product
                    similarity = float(np.dot(query, vector))

                similarities.append({"key": key, "similarity": similarity})

        # Get top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:limit]

    def _search_annoy(self, query: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Approximate nearest neighbors search using Annoy."""
        if not self._annoy_index:
            return self._search_flat(query, limit)

        # Search Annoy index
        indices = self._annoy_index.get_nns_by_vector(
            query.tolist(), min(limit * 2, len(self.vectors)), include_distances=True
        )

        results = []
        for idx, distance in zip(indices[0], indices[1]):
            if idx < len(self._index_keys):
                key = self._index_keys[idx]

                # Convert distance to similarity
                if self.similarity_metric == "cosine":
                    similarity = 1.0 - (distance / 2)  # Annoy returns angular distance
                else:
                    similarity = 1.0 / (1.0 + distance)

                results.append({"key": key, "similarity": similarity})

        return results[:limit]

    def _search_sklearn(self, query: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Search using scikit-learn KDTree."""
        if not self._sklearn_index:
            return self._search_flat(query, limit)

        # Reshape query
        query_reshaped = query.reshape(1, -1)

        # Find nearest neighbors
        distances, indices = self._sklearn_index.kneighbors(
            query_reshaped, n_neighbors=limit
        )

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self._index_keys):
                key = self._index_keys[idx]

                # Convert distance to similarity
                similarity = 1.0 / (1.0 + distance)

                results.append({"key": key, "similarity": similarity})

        return results

    # --------------------------------------------------
    # SIMILARITY SEARCH
    # --------------------------------------------------
    def find_similar(
        self,
        key: str,
        limit: int = 10,
        threshold: float = 0.0,
        exclude_self: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find vectors similar to a stored vector."""

        with self._vector_lock:
            if key not in self.vectors:
                return []

            query_vector = self.vectors[key]

        results = self.search(query_vector.tolist(), limit + 1, threshold)

        if exclude_self:
            return [r for r in results if r["key"] != key][:limit]

        return results[:limit]

    def find_similar_batch(
        self, keys: List[str], limit: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find similar vectors for multiple keys."""
        results = {}
        for key in keys:
            results[key] = self.find_similar(key, limit)
        return results

    # --------------------------------------------------
    # ADVANCED SEARCH
    # --------------------------------------------------
    def search_by_range(
        self,
        query_vector: Union[List[float], np.ndarray],
        radius: float,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for vectors within a similarity radius."""
        results = self.search(query_vector, limit=limit)

        # Filter by radius
        return [r for r in results if r["similarity"] >= (1.0 - radius)]

    def search_by_tags(
        self, tags: List[str], limit: int = 50, sort_by: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """Search vectors by tags."""
        results = []

        with self._vector_lock:
            for key, metadata in self.metadata.items():
                # Check if any tag matches
                if any(tag in metadata.tags for tag in tags):
                    relevance = len([t for t in tags if t in metadata.tags]) / len(tags)

                    results.append(
                        {
                            "key": key,
                            "relevance": relevance,
                            "tags": metadata.tags,
                            "confidence": metadata.confidence,
                            "content": (
                                self.entries[key].content
                                if key in self.entries
                                else None
                            ),
                        }
                    )

        # Sort by specified field
        if sort_by == "relevance":
            results.sort(key=lambda x: x["relevance"], reverse=True)
        elif sort_by == "confidence":
            results.sort(key=lambda x: x["confidence"], reverse=True)
        elif sort_by == "recency":
            results.sort(
                key=lambda x: (
                    self.metadata[x["key"]].created_at
                    if x["key"] in self.metadata
                    else datetime.min
                ),
                reverse=True,
            )

        return results[:limit]

    def search_by_metadata(
        self, query: Dict[str, Any], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search vectors by metadata fields."""
        results = []

        with self._vector_lock:
            for key, metadata in self.metadata.items():
                match = True
                for field, value in query.items():
                    if field == "tags":
                        if not all(tag in metadata.tags for tag in value):
                            match = False
                            break
                    elif hasattr(metadata, field):
                        if getattr(metadata, field) != value:
                            match = False
                            break
                    elif field in metadata.metadata:
                        if metadata.metadata[field] != value:
                            match = False
                            break
                    else:
                        match = False
                        break

                if match:
                    results.append(
                        {
                            "key": key,
                            "metadata": self._get_metadata_dict(key),
                            "content": (
                                self.entries[key].content
                                if key in self.entries
                                else None
                            ),
                        }
                    )

        return results[:limit]

    # --------------------------------------------------
    # INDEX MANAGEMENT
    # --------------------------------------------------
    def _rebuild_index(self):
        """Rebuild vector index for faster search."""
        with self._index_lock:
            if not self._index_needs_rebuild:
                return

            if len(self.vectors) < 10:
                return

            try:
                if self.index_type == VectorIndexType.ANNOY and self.use_annoy:
                    self._rebuild_annoy_index()
                elif self.index_type == VectorIndexType.SKLEARN and SKLEARN_AVAILABLE:
                    self._rebuild_sklearn_index()

                self._index_needs_rebuild = False
                self.stats["index_rebuilds"] += 1
                self.stats["last_rebuild"] = datetime.now()

                logger.debug(f"🔄 Vector index rebuilt: {len(self.vectors)} vectors")

            except Exception as e:
                logger.error(f"Index rebuild failed: {e}")

    def _rebuild_annoy_index(self):
        """Rebuild Annoy index."""
        if not ANNOY_AVAILABLE:
            return

        # Determine metric for Annoy
        annoy_metric = "angular" if self.similarity_metric == "cosine" else "euclidean"

        # Create index
        self._annoy_index = AnnoyIndex(self.embedding_dim, annoy_metric)

        # Add vectors
        self._index_keys = []
        for idx, (key, vector) in enumerate(self.vectors.items()):
            self._annoy_index.add_item(idx, vector.tolist())
            self._index_keys.append(key)

        # Build index
        self._annoy_index.build(self.annoy_trees)

    def _rebuild_sklearn_index(self):
        """Rebuild scikit-learn KDTree index."""
        if not SKLEARN_AVAILABLE:
            return

        # Prepare vectors matrix
        vectors_list = []
        self._index_keys = []

        for key, vector in self.vectors.items():
            vectors_list.append(vector)
            self._index_keys.append(key)

        vectors_matrix = np.array(vectors_list)

        # Build KDTree
        from sklearn.neighbors import NearestNeighbors

        self._sklearn_index = NearestNeighbors(
            n_neighbors=min(10, len(vectors_list)),
            metric=self.similarity_metric,
            algorithm="kd_tree",
        )
        self._sklearn_index.fit(vectors_matrix)

    # --------------------------------------------------
    # VECTOR OPERATIONS
    # --------------------------------------------------
    def average_vectors(self, keys: List[str]) -> Optional[np.ndarray]:
        """Compute average of multiple vectors."""
        vectors = []

        with self._vector_lock:
            for key in keys:
                if key in self.vectors:
                    vectors.append(self.vectors[key])

        if not vectors:
            return None

        avg_vector = np.mean(vectors, axis=0)

        # Normalize
        if self.similarity_metric == "cosine":
            norm = np.linalg.norm(avg_vector)
            if norm > 0:
                avg_vector = avg_vector / norm

        return avg_vector

    def merge_vectors(
        self, keys: List[str], new_key: str, weights: Optional[List[float]] = None
    ) -> bool:
        """Merge multiple vectors into one using weighted average."""
        vectors = []
        weights = weights or [1.0] * len(keys)

        with self._vector_lock:
            for key, weight in zip(keys, weights):
                if key in self.vectors:
                    vectors.append(self.vectors[key] * weight)

            if not vectors:
                return False

            merged = np.sum(vectors, axis=0)

            # Normalize
            if self.similarity_metric == "cosine":
                norm = np.linalg.norm(merged)
                if norm > 0:
                    merged = merged / norm

            # Store merged vector
            return self.store(
                new_key, merged, associated_content=f"Merged from {len(keys)} vectors"
            )

    def get_vector_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector statistics."""
        if not self.vectors:
            return {"count": 0, "avg_magnitude": 0}

        magnitudes = [np.linalg.norm(v) for v in self.vectors.values()]

        # Calculate pairwise similarity distribution (sample)
        similarity_samples = []
        if len(self.vectors) > 1:
            sample_keys = list(self.vectors.keys())[: min(100, len(self.vectors))]
            for i in range(min(10, len(sample_keys))):
                for j in range(i + 1, min(10, len(sample_keys))):
                    sim = float(
                        np.dot(
                            self.vectors[sample_keys[i]], self.vectors[sample_keys[j]]
                        )
                    )
                    similarity_samples.append(sim)

        return {
            "count": len(self.vectors),
            "avg_magnitude": float(np.mean(magnitudes)),
            "min_magnitude": float(np.min(magnitudes)),
            "max_magnitude": float(np.max(magnitudes)),
            "std_magnitude": float(np.std(magnitudes)),
            "avg_similarity": (
                float(np.mean(similarity_samples)) if similarity_samples else 0
            ),
            "dimension": self.embedding_dim,
            "index_type": self.index_type,
            "index_built": not self._index_needs_rebuild,
            "cache_hit_rate": self.stats["cache_hits"]
            / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]),
            "avg_search_time_ms": self.stats["avg_search_time_ms"],
            "total_searches": self.stats["total_searches"],
        }

    def get_clusters(self, num_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster vectors using k-means (if scikit-learn available)."""
        if not SKLEARN_AVAILABLE or len(self.vectors) < num_clusters:
            return {}

        from sklearn.cluster import KMeans

        # Prepare vectors
        vectors_list = []
        keys_list = []

        for key, vector in self.vectors.items():
            vectors_list.append(vector)
            keys_list.append(key)

        vectors_matrix = np.array(vectors_list)

        # Perform clustering
        kmeans = KMeans(
            n_clusters=min(num_clusters, len(vectors_list)), random_state=42
        )
        labels = kmeans.fit_predict(vectors_matrix)

        # Group by cluster
        clusters = defaultdict(list)
        for key, label in zip(keys_list, labels):
            clusters[int(label)].append(key)

        return dict(clusters)

    # --------------------------------------------------
    # MAINTENANCE
    # --------------------------------------------------
    def delete(self, key: str, delete_metadata: bool = True) -> bool:
        """Delete a vector and its metadata."""

        with self._vector_lock:
            if key in self.entries:
                del self.entries[key]
                del self.vectors[key]

                if delete_metadata and key in self.metadata:
                    del self.metadata[key]

                # Remove from cache
                if key in self._cache:
                    del self._cache[key]

                # Mark index for rebuild
                self._index_needs_rebuild = True

                # 🔥 DELETE FROM DB
                mongo_client.delete("vector", key)

                self.stats["total_vectors"] = len(self.vectors)

                logger.debug(f"🗑️ Vector deleted: {key}")
                return True

        return False

    def delete_batch(self, keys: List[str]) -> int:
        """Delete multiple vectors."""
        success_count = 0
        for key in keys:
            if self.delete(key):
                success_count += 1

        # Rebuild index after batch delete
        if success_count > 0:
            self._rebuild_index()

        return success_count

    def clear(self, confirm: bool = False) -> bool:
        """Clear all vectors (requires confirmation)."""
        if not confirm:
            logger.warning("Clear requires confirmation")
            return False

        with self._vector_lock:
            self.entries.clear()
            self.vectors.clear()
            self.metadata.clear()
            self._cache.clear()
            self._query_cache.clear()

            self._index_keys.clear()
            self._index_needs_rebuild = True

            self.stats = {
                "total_vectors": 0,
                "total_searches": 0,
                "avg_search_time_ms": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "index_rebuilds": self.stats["index_rebuilds"],
                "last_rebuild": self.stats["last_rebuild"],
            }

            # Clear database
            try:
                mongo_client.collection("vector").delete_many({})
                logger.info("🗑️ MongoDB vector collection cleared")
            except Exception as e:
                logger.error(f"Failed to clear MongoDB: {e}")

            logger.info("🧹 Vector memory cleared")
            return True

    def optimize(self) -> Dict[str, Any]:
        """Optimize vector storage and indexes."""
        optimization_results = {
            "before_count": len(self.vectors),
            "after_count": 0,
            "removed_duplicates": 0,
            "rebuilt_index": False,
        }

        with self._vector_lock:
            # Remove duplicate vectors
            unique_vectors = {}
            vector_hashes = {}

            for key, vector in self.vectors.items():
                # Create hash of vector (rounded to 6 decimal places)
                rounded = np.round(vector, decimals=6)
                vector_hash = hashlib.md5(rounded.tobytes()).hexdigest()

                if vector_hash not in vector_hashes:
                    vector_hashes[vector_hash] = key
                    unique_vectors[key] = vector
                else:
                    # Duplicate found
                    optimization_results["removed_duplicates"] += 1
                    self.delete(key)

            optimization_results["after_count"] = len(self.vectors)

            # Rebuild index
            self._rebuild_index()
            optimization_results["rebuilt_index"] = True

        logger.info(
            f"🔧 Optimization complete: removed {optimization_results['removed_duplicates']} duplicates"
        )
        return optimization_results

    # --------------------------------------------------
    # CACHE MANAGEMENT
    # --------------------------------------------------
    def _update_cache(self, key: str, vector: np.ndarray):
        """Update LRU cache."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = vector
        self._access_history.append(key)

    def _update_access_metadata(self, key: str):
        """Update access metadata for a vector."""
        if key in self.metadata:
            self.metadata[key].access_count += 1
            self.metadata[key].last_accessed = datetime.now()

            # Update in database (async)
            try:
                mongo_client.update(
                    "vector",
                    key,
                    {
                        "metadata.access_count": self.metadata[key].access_count,
                        "metadata.last_accessed": self.metadata[
                            key
                        ].last_accessed.isoformat(),
                    },
                )
            except Exception:
                pass

    def _get_metadata_dict(self, key: str) -> Dict[str, Any]:
        """Get metadata as dictionary."""
        if key not in self.metadata:
            return {}

        meta = self.metadata[key]
        return {
            "created_at": meta.created_at.isoformat(),
            "updated_at": meta.updated_at.isoformat(),
            "access_count": meta.access_count,
            "last_accessed": (
                meta.last_accessed.isoformat() if meta.last_accessed else None
            ),
            "source": meta.source,
            "model_version": meta.model_version,
            "confidence": meta.confidence,
            "tags": meta.tags,
            "metadata": meta.metadata,
        }

    def _get_query_cache_key(self, *args) -> str:
        """Generate cache key for query."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _clean_query_cache(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired = []
        for key, (cached_time, _) in self._query_cache.items():
            if (now - cached_time).seconds > self._query_cache_ttl:
                expired.append(key)

        for key in expired:
            del self._query_cache[key]

    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------
    def add_callback(self, callback: Callable[[str, np.ndarray], None]):
        """Add callback for new vector storage."""
        self._callbacks.append(callback)

    def _trigger_callbacks(self, key: str, vector: np.ndarray):
        """Trigger all callbacks."""
        for callback in self._callbacks:
            try:
                callback(key, vector)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    # --------------------------------------------------
    # EXPORT/IMPORT
    # --------------------------------------------------
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all vectors to JSON file."""
        if not filepath:
            filepath = f"vector_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            "export_time": datetime.now().isoformat(),
            "embedding_dim": self.embedding_dim,
            "similarity_metric": self.similarity_metric,
            "vectors": [],
        }

        with self._vector_lock:
            for key, vector in self.vectors.items():
                export_data["vectors"].append(
                    {
                        "key": key,
                        "embedding": vector.tolist(),
                        "content": (
                            self.entries[key].content if key in self.entries else None
                        ),
                        "metadata": self._get_metadata_dict(key),
                    }
                )

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"📤 Exported {len(export_data['vectors'])} vectors to {filepath}")
        return filepath

    def import_from_json(self, filepath: str, overwrite: bool = False) -> int:
        """Import vectors from JSON file."""
        if overwrite:
            self.clear(confirm=True)

        with open(filepath, "r") as f:
            data = json.load(f)

        imported_count = 0
        for item in data.get("vectors", []):
            key = item["key"]
            embedding = item["embedding"]
            content = item.get("content")
            metadata = item.get("metadata", {})

            if self.store(
                key,
                embedding,
                associated_content=content,
                source=metadata.get("source"),
                confidence=metadata.get("confidence", 1.0),
                tags=metadata.get("tags", []),
            ):
                imported_count += 1

        logger.info(f"📥 Imported {imported_count} vectors from {filepath}")
        return imported_count

    # --------------------------------------------------
    # UTILITY
    # --------------------------------------------------
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_keys(self) -> List[str]:
        """Get all vector keys."""
        with self._vector_lock:
            return list(self.vectors.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats."""
        return {
            "total_vectors": len(self.vectors),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "similarity_metric": self.similarity_metric,
            "vector_stats": self.get_vector_stats(),
            "cache_stats": {
                "size": len(self._cache),
                "hits": self.stats["cache_hits"],
                "misses": self.stats["cache_misses"],
                "hit_rate": self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]),
            },
            "index_stats": {
                "needs_rebuild": self._index_needs_rebuild,
                "rebuilds": self.stats["index_rebuilds"],
                "last_rebuild": (
                    self.stats["last_rebuild"].isoformat()
                    if self.stats["last_rebuild"]
                    else None
                ),
            },
            "performance": {
                "avg_search_time_ms": self.stats["avg_search_time_ms"],
                "total_searches": self.stats["total_searches"],
            },
        }

    def info(self) -> Dict[str, Any]:
        """Get memory information."""
        return self.get_stats()


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convert euclidean distance to similarity score."""
    distance = np.linalg.norm(a - b)
    return 1.0 / (1.0 + distance)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


__all__ = [
    "VectorMemory",
    "VectorIndexType",
    "VectorMetadata",
    "cosine_similarity",
    "euclidean_similarity",
    "normalize_vector",
]
