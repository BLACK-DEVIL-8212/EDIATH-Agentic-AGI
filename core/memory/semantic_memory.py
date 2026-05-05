"""
Semantic Memory - stores general knowledge, facts, and learned information with
graph relationships, inference capabilities, and knowledge graph support.
(PRODUCTION READY WITH MONGODB INTEGRATION)
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib
import statistics
import threading

from .memory_base import BaseMemory
from .mongo_client import mongo_client
from ..utils.logger import logger


class RelationType(Enum):
    """Types of semantic relationships."""

    IS_A = "is_a"  # Subclass relationship
    PART_OF = "part_of"  # Part-whole relationship
    RELATED_TO = "related_to"  # Generic relationship
    CAUSES = "causes"  # Causal relationship
    PRECEDES = "precedes"  # Temporal relationship
    CONTRADICTS = "contradicts"  # Contradictory relationship
    SUPPORTS = "supports"  # Supporting evidence
    EXAMPLE_OF = "example_of"  # Example relationship
    DEFINED_AS = "defined_as"  # Definition
    DERIVED_FROM = "derived_from"  # Inference source


class KnowledgeCategory(Enum):
    """Categories of knowledge."""

    GENERAL = "general"
    FACT = "fact"
    RULE = "rule"
    DEFINITION = "definition"
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    HEURISTIC = "heuristic"
    BELIEF = "belief"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    STRATEGY = "strategy"


@dataclass
class KnowledgeNode:
    """Enhanced knowledge node with metadata."""

    key: str
    content: Any
    category: KnowledgeCategory
    confidence: float
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    version: int = 1
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "content": self.content,
            "category": self.category.value,
            "confidence": self.confidence,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "version": self.version,
            "parent": self.parent,
            "children": self.children,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        """Create from dictionary."""
        # Convert category string to enum
        category_str = data.get("category", "general")
        try:
            category = KnowledgeCategory(category_str)
        except ValueError:
            category = KnowledgeCategory.GENERAL

        # Parse timestamps
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now()
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now()
        )
        last_accessed = (
            datetime.fromisoformat(data["last_accessed"])
            if data.get("last_accessed")
            else None
        )

        return cls(
            key=data["key"],
            content=data["content"],
            category=category,
            confidence=data.get("confidence", 0.8),
            source=data.get("source"),
            tags=data.get("tags", []),
            created_at=created_at,
            updated_at=updated_at,
            access_count=data.get("access_count", 0),
            last_accessed=last_accessed,
            version=data.get("version", 1),
            parent=data.get("parent"),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relationship:
    """Enhanced relationship between knowledge nodes."""

    source: str
    target: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=RelationType(data.get("relation_type", "related_to")),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            metadata=data.get("metadata", {}),
        )


class InferenceEngine:
    """Knowledge inference engine for deriving new knowledge."""

    def __init__(self, memory: "SemanticMemory"):
        self.memory = memory
        self.inference_rules: List[Callable] = []

    def infer(self, knowledge: KnowledgeNode) -> List[Tuple[str, float]]:
        """Infer new knowledge from existing knowledge."""
        inferences = []

        # Rule 1: Transitive relationships
        if knowledge.category == KnowledgeCategory.FACT:
            related = self.memory.get_related_knowledge(
                knowledge.key, relation_type=RelationType.RELATED_TO
            )
            for rel in related:
                if rel.confidence > 0.7:
                    inferences.append(
                        (
                            f"{knowledge.content} is related to {rel.content}",
                            rel.confidence * 0.8,
                        )
                    )

        # Rule 2: IS-A hierarchy
        if knowledge.category == KnowledgeCategory.CONCEPT:
            parents = self.memory.get_parents(knowledge.key)
            for parent in parents:
                inferences.append(
                    (
                        f"{knowledge.content} is a type of {parent.content}",
                        parent.confidence * 0.9,
                    )
                )

        # Rule 3: Cause-effect chains
        causes = self.memory.get_relationships(
            knowledge.key, relation_type=RelationType.CAUSES
        )
        for cause in causes:
            inferences.append(
                (
                    f"{cause.content} can lead to {knowledge.content}",
                    cause.confidence * 0.7,
                )
            )

        return inferences


class SemanticMemory(BaseMemory):
    """Advanced semantic memory with knowledge graph, inference, and analytics."""

    def __init__(
        self,
        enable_inference: bool = True,
        enable_versioning: bool = True,
        cache_size: int = 1000,
        max_history: int = 100,
        auto_link: bool = True,
    ):
        """Initialize advanced semantic memory."""
        super().__init__("semantic_memory")

        # Core storage
        self.knowledge_base: Dict[str, KnowledgeNode] = {}
        self.relationships: Dict[str, List[Relationship]] = defaultdict(list)
        self.reverse_relationships: Dict[str, List[Relationship]] = defaultdict(list)

        # Indexes
        self.category_index: Dict[KnowledgeCategory, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.confidence_index: Dict[str, List[str]] = defaultdict(list)

        # Features
        self.enable_inference = enable_inference
        self.enable_versioning = enable_versioning
        self.cache_size = cache_size
        self.max_history = max_history
        self.auto_link = auto_link

        # Version history
        self.knowledge_history: Dict[str, List[KnowledgeNode]] = defaultdict(list)

        # Caching
        self._cache: Dict[str, KnowledgeNode] = {}
        self._access_history: deque = deque(maxlen=cache_size)

        # Inference engine
        self.inference_engine = InferenceEngine(self) if enable_inference else None

        # Statistics
        self.stats = {
            "total_knowledge": 0,
            "total_relationships": 0,
            "avg_confidence": 0.0,
            "most_accessed": [],
            "category_distribution": defaultdict(int),
            "inferences_made": 0,
        }

        # Query cache
        self._query_cache: Dict[str, Tuple[datetime, List[Dict]]] = {}
        self._query_cache_ttl = 60

        # Locks
        self._knowledge_lock = threading.RLock()

        # Callbacks
        self._knowledge_callbacks: List[Callable[[KnowledgeNode], None]] = []
        self._relationship_callbacks: List[Callable[[Relationship], None]] = []

        # 🔥 LOAD FROM DB
        self._load_from_db()

        logger.info(
            f"✅ SemanticMemory initialized (inference={enable_inference}, versioning={enable_versioning}, auto_link={auto_link})"
        )

    # --------------------------------------------------
    # LOAD FROM MONGO
    # --------------------------------------------------
    def _load_from_db(self):
        """Load all knowledge from MongoDB with relationships."""
        try:
            # Load knowledge nodes
            docs = mongo_client.load_all("semantic")

            for doc in docs:
                key = doc.get("key")

                # Create knowledge node
                knowledge = KnowledgeNode.from_dict(doc)

                # Store in memory
                self.knowledge_base[key] = knowledge

                # Update indexes
                self.category_index[knowledge.category].append(key)
                for tag in knowledge.tags:
                    self.tag_index[tag].append(key)

                # Load relationships from separate collection
                rel_docs = mongo_client.load_all(f"semantic_rels_{key}")
                for rel_doc in rel_docs:
                    relationship = Relationship.from_dict(rel_doc)
                    self.relationships[key].append(relationship)
                    self.reverse_relationships[relationship.target].append(relationship)

            # Update statistics
            self.stats["total_knowledge"] = len(self.knowledge_base)
            self.stats["total_relationships"] = sum(
                len(rels) for rels in self.relationships.values()
            )

            if self.knowledge_base:
                confidences = [k.confidence for k in self.knowledge_base.values()]
                self.stats["avg_confidence"] = statistics.mean(confidences)

            # Build category distribution
            for category, keys in self.category_index.items():
                self.stats["category_distribution"][category.value] = len(keys)

            logger.info(
                f"📚 Semantic memory loaded from DB ({len(self.knowledge_base)} nodes, {self.stats['total_relationships']} relationships)"
            )

        except Exception as e:
            logger.error(f"Semantic DB load failed: {e}")

    # --------------------------------------------------
    # STORE KNOWLEDGE
    # --------------------------------------------------
    def store(
        self,
        key: str,
        content: Any,
        category: Union[KnowledgeCategory, str] = KnowledgeCategory.GENERAL,
        confidence: float = 0.8,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version_note: Optional[str] = None,
    ) -> bool:
        """Store knowledge with enhanced metadata."""
        try:
            # Convert category if string
            if isinstance(category, str):
                try:
                    category = KnowledgeCategory(category)
                except ValueError:
                    category = KnowledgeCategory.GENERAL

            # Check if updating existing knowledge
            existing = self.knowledge_base.get(key)

            with self._knowledge_lock:
                # Create knowledge node
                knowledge = KnowledgeNode(
                    key=key,
                    content=content,
                    category=category,
                    confidence=confidence,
                    source=source,
                    tags=tags or [],
                    parent=parent,
                    metadata=metadata or {},
                    version=(existing.version + 1) if existing else 1,
                )

                # Store version history
                if self.enable_versioning and existing:
                    self.knowledge_history[key].append(existing)
                    # Maintain history size
                    if len(self.knowledge_history[key]) > self.max_history:
                        self.knowledge_history[key] = self.knowledge_history[key][
                            -self.max_history :
                        ]

                # Store in memory
                self.knowledge_base[key] = knowledge

                # Update indexes
                if existing and existing.category != category:
                    # Remove from old category index
                    if existing.category in self.category_index:
                        self.category_index[existing.category] = [
                            k
                            for k in self.category_index[existing.category]
                            if k != key
                        ]

                self.category_index[category].append(key)

                # Update tag index
                if existing:
                    for old_tag in existing.tags:
                        if old_tag in self.tag_index and key in self.tag_index[old_tag]:
                            self.tag_index[old_tag].remove(key)

                for tag in tags or []:
                    self.tag_index[tag].append(key)

                # Update parent-child relationships
                if parent:
                    knowledge.parent = parent
                    if parent in self.knowledge_base:
                        if key not in self.knowledge_base[parent].children:
                            self.knowledge_base[parent].children.append(key)

                # Update cache
                self._update_cache(key, knowledge)

                # Update statistics
                self.stats["total_knowledge"] = len(self.knowledge_base)
                all_confidences = [k.confidence for k in self.knowledge_base.values()]
                self.stats["avg_confidence"] = (
                    statistics.mean(all_confidences) if all_confidences else 0
                )
                self.stats["category_distribution"][category.value] = len(
                    self.category_index[category]
                )

                # Auto-link related knowledge
                if self.auto_link:
                    self._auto_link_knowledge(knowledge)

                # 🔥 SAVE TO MONGO
                mongo_client.save("semantic", knowledge.to_dict())

                # Trigger callbacks
                self._trigger_knowledge_callbacks(knowledge)

                # Perform inference
                if self.enable_inference and self.inference_engine:
                    inferences = self.inference_engine.infer(knowledge)
                    for inferred_content, inferred_conf in inferences:
                        inferred_key = hashlib.md5(
                            inferred_content.encode()
                        ).hexdigest()[:16]
                        self.store(
                            inferred_key,
                            inferred_content,
                            category=KnowledgeCategory.FACT,
                            confidence=inferred_conf,
                            source=f"inferred_from_{key}",
                        )
                        self.stats["inferences_made"] += 1

                logger.debug(
                    f"📝 Knowledge stored: {key} (category={category.value}, confidence={confidence:.2f})"
                )
                return True

        except Exception as e:
            logger.error(f"Knowledge store error: {e}")
            return False

    def _auto_link_knowledge(self, knowledge: KnowledgeNode):
        """Automatically link related knowledge based on content similarity."""
        # Find potential related knowledge
        for other_key, other_knowledge in self.knowledge_base.items():
            if other_key == knowledge.key:
                continue

            # Check tag overlap
            common_tags = set(knowledge.tags) & set(other_knowledge.tags)
            if len(common_tags) >= 2:
                self.link_knowledge(
                    knowledge.key,
                    other_key,
                    relation_type=RelationType.RELATED_TO,
                    weight=len(common_tags)
                    / max(len(knowledge.tags), len(other_knowledge.tags)),
                )

            # Check category similarity
            if knowledge.category == other_knowledge.category:
                self.link_knowledge(
                    knowledge.key,
                    other_key,
                    relation_type=RelationType.RELATED_TO,
                    weight=0.5,
                )

    # --------------------------------------------------
    # RETRIEVE KNOWLEDGE
    # --------------------------------------------------
    def retrieve(self, key: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge by key."""

        # Check cache
        if use_cache and key in self._cache:
            self.stats.get("cache_hits", 0)
            knowledge = self._cache[key]
            self._update_access_metadata(knowledge)
            return self._knowledge_to_dict(knowledge)

        with self._knowledge_lock:
            if key in self.knowledge_base:
                knowledge = self.knowledge_base[key]
                self._update_access_metadata(knowledge)
                self._update_cache(key, knowledge)
                return self._knowledge_to_dict(knowledge)

        # 🔥 FALLBACK DB
        doc = mongo_client.load("semantic", key)
        if doc:
            return doc

        return None

    def retrieve_batch(self, keys: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve multiple knowledge items."""
        results = {}
        for key in keys:
            results[key] = self.retrieve(key)
        return results

    def _knowledge_to_dict(self, knowledge: KnowledgeNode) -> Dict[str, Any]:
        """Convert knowledge node to dictionary for return."""
        return {
            "key": knowledge.key,
            "content": knowledge.content,
            "category": knowledge.category.value,
            "confidence": knowledge.confidence,
            "source": knowledge.source,
            "tags": knowledge.tags,
            "created_at": knowledge.created_at.isoformat(),
            "updated_at": knowledge.updated_at.isoformat(),
            "access_count": knowledge.access_count,
            "version": knowledge.version,
            "parent": knowledge.parent,
            "children": knowledge.children,
            "metadata": knowledge.metadata,
        }

    # --------------------------------------------------
    # RELATIONSHIP MANAGEMENT
    # --------------------------------------------------
    def link_knowledge(
        self,
        source_key: str,
        target_key: str,
        relation_type: Union[RelationType, str] = RelationType.RELATED_TO,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create relationship between knowledge nodes."""

        # Convert relation type if string
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType(relation_type)
            except ValueError:
                relation_type = RelationType.RELATED_TO

        # Check if both nodes exist
        if (
            source_key not in self.knowledge_base
            or target_key not in self.knowledge_base
        ):
            logger.warning(f"Cannot link: {source_key} or {target_key} not found")
            return False

        with self._knowledge_lock:
            # Create relationship
            relationship = Relationship(
                source=source_key,
                target=target_key,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence,
                metadata=metadata or {},
            )

            # Add relationship
            self.relationships[source_key].append(relationship)
            self.reverse_relationships[target_key].append(relationship)

            # Update statistics
            self.stats["total_relationships"] += 1

            # 🔥 SAVE TO MONGO
            mongo_client.save(f"semantic_rels_{source_key}", relationship.to_dict())

            # Trigger callbacks
            self._trigger_relationship_callbacks(relationship)

            logger.debug(
                f"🔗 Linked {source_key} -> {target_key} ({relation_type.value}, weight={weight})"
            )
            return True

    def get_relationships(
        self,
        key: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing",
    ) -> List[Relationship]:
        """Get relationships for a knowledge node."""

        if direction == "outgoing":
            relationships = self.relationships.get(key, [])
        elif direction == "incoming":
            relationships = self.reverse_relationships.get(key, [])
        else:  # both
            relationships = self.relationships.get(
                key, []
            ) + self.reverse_relationships.get(key, [])

        if relation_type:
            relationships = [
                r for r in relationships if r.relation_type == relation_type
            ]

        return relationships

    def get_related_knowledge(
        self,
        key: str,
        relation_type: Optional[RelationType] = None,
        depth: int = 1,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get related knowledge with graph traversal."""

        related = []
        visited = set()
        queue = deque([(key, 0)])

        while queue:
            current_key, current_depth = queue.popleft()

            if current_key in visited or current_depth > depth:
                continue

            visited.add(current_key)

            if current_key in self.knowledge_base:
                knowledge = self.knowledge_base[current_key]

                # Skip if confidence too low
                if knowledge.confidence < min_confidence:
                    continue

                if current_key != key:  # Don't include the original node
                    related.append(
                        {
                            "key": current_key,
                            "content": knowledge.content,
                            "category": knowledge.category.value,
                            "confidence": knowledge.confidence,
                            "depth": current_depth,
                            "relationships": [
                                {"type": r.relation_type.value, "weight": r.weight}
                                for r in self.get_relationships(
                                    current_key, relation_type, "incoming"
                                )
                                if r.source == key or r.target == key
                            ],
                        }
                    )

                # Explore outgoing relationships
                for rel in self.relationships.get(current_key, []):
                    if relation_type is None or rel.relation_type == relation_type:
                        if rel.target not in visited:
                            queue.append((rel.target, current_depth + 1))

        # Sort by depth and confidence
        related.sort(key=lambda x: (x["depth"], x["confidence"]), reverse=True)

        return related

    def get_parents(self, key: str) -> List[KnowledgeNode]:
        """Get parent nodes in hierarchy."""
        parents = []
        current = self.knowledge_base.get(key)

        while current and current.parent:
            parent = self.knowledge_base.get(current.parent)
            if parent:
                parents.append(parent)
                current = parent
            else:
                break

        return parents

    def get_children(self, key: str, recursive: bool = False) -> List[KnowledgeNode]:
        """Get child nodes in hierarchy."""
        children = []

        if key not in self.knowledge_base:
            return children

        for child_key in self.knowledge_base[key].children:
            child = self.knowledge_base.get(child_key)
            if child:
                children.append(child)
                if recursive:
                    children.extend(self.get_children(child_key, recursive=True))

        return children

    def get_knowledge_graph(self, root_key: str, max_depth: int = 3) -> Dict[str, Any]:
        """Build knowledge graph starting from root."""

        graph = {"nodes": {}, "edges": []}

        def traverse(node_key: str, depth: int):
            if depth > max_depth or node_key not in self.knowledge_base:
                return

            node = self.knowledge_base[node_key]

            # Add node
            graph["nodes"][node_key] = {
                "label": str(node.content)[:50],
                "category": node.category.value,
                "confidence": node.confidence,
            }

            # Add edges
            for rel in self.relationships.get(node_key, []):
                if rel.target in self.knowledge_base:
                    graph["edges"].append(
                        {
                            "source": node_key,
                            "target": rel.target,
                            "type": rel.relation_type.value,
                            "weight": rel.weight,
                        }
                    )
                    traverse(rel.target, depth + 1)

        traverse(root_key, 0)
        return graph

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    def search(
        self,
        query: str,
        category: Optional[Union[KnowledgeCategory, str]] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
        use_semantic: bool = False,
    ) -> List[Dict[str, Any]]:
        """Advanced search with multiple filters."""

        # Check query cache
        cache_key = self._get_query_cache_key(query, category, tags, min_confidence)
        if cache_key in self._query_cache:
            cached_time, cached_result = self._query_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._query_cache_ttl:
                return cached_result

        results = []
        query_lower = query.lower()

        # Convert category if provided
        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                category = None

        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():

                # Apply confidence filter
                if knowledge.confidence < min_confidence:
                    continue

                # Apply category filter
                if category and knowledge.category != category:
                    continue

                # Apply tags filter
                if tags and not any(tag in knowledge.tags for tag in tags):
                    continue

                # Calculate relevance score
                relevance = 0.0

                # Exact key match
                if key.lower() == query_lower:
                    relevance = 1.0

                # Content match
                elif isinstance(knowledge.content, str):
                    content_lower = knowledge.content.lower()
                    if query_lower in content_lower:
                        # Position-based relevance
                        position = content_lower.find(query_lower)
                        relevance = 0.7 * (1 - position / max(len(content_lower), 1))

                # Tag match
                for tag in knowledge.tags:
                    if query_lower in tag.lower():
                        relevance = max(relevance, 0.6)

                # Category match
                if query_lower in knowledge.category.value:
                    relevance = max(relevance, 0.5)

                if relevance > 0:
                    # Boost by confidence
                    relevance *= 0.5 + knowledge.confidence * 0.5

                    results.append(
                        {
                            "key": key,
                            "content": knowledge.content,
                            "category": knowledge.category.value,
                            "confidence": knowledge.confidence,
                            "source": knowledge.source,
                            "tags": knowledge.tags,
                            "relevance": relevance,
                            "access_count": knowledge.access_count,
                        }
                    )

        # Sort by relevance
        results.sort(key=lambda x: (x["relevance"], x["confidence"]), reverse=True)

        # Cache results
        self._query_cache[cache_key] = (datetime.now(), results[:limit])
        self._clean_query_cache()

        return results[:limit]

    def search_by_category(
        self, category: Union[KnowledgeCategory, str]
    ) -> List[Dict[str, Any]]:
        """Get all knowledge in a category."""

        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                return []

        keys = self.category_index.get(category, [])

        return [
            self._knowledge_to_dict(self.knowledge_base[key])
            for key in keys
            if key in self.knowledge_base
        ]

    def search_by_tags(
        self, tags: List[str], match_all: bool = False
    ) -> List[Dict[str, Any]]:
        """Search knowledge by tags."""

        results = []

        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if match_all:
                    if all(tag in knowledge.tags for tag in tags):
                        results.append(self._knowledge_to_dict(knowledge))
                else:
                    if any(tag in knowledge.tags for tag in tags):
                        results.append(self._knowledge_to_dict(knowledge))

        return results

    def search_similar(
        self, content: str, threshold: float = 0.5, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find semantically similar knowledge (basic text similarity)."""

        results = []
        content_lower = content.lower()

        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if isinstance(knowledge.content, str):
                    # Calculate simple text similarity
                    target_lower = knowledge.content.lower()

                    # Word overlap
                    content_words = set(content_lower.split())
                    target_words = set(target_lower.split())

                    if content_words and target_words:
                        overlap = len(content_words & target_words)
                        similarity = overlap / max(
                            len(content_words), len(target_words)
                        )

                        if similarity >= threshold:
                            results.append(
                                {
                                    "key": key,
                                    "content": knowledge.content,
                                    "similarity": similarity,
                                    "confidence": knowledge.confidence,
                                }
                            )

        results.sort(key=lambda x: (x["similarity"], x["confidence"]), reverse=True)
        return results[:limit]

    # --------------------------------------------------
    # KNOWLEDGE INFERENCE
    # --------------------------------------------------
    def infer_new_knowledge(self, key: str) -> List[KnowledgeNode]:
        """Infer new knowledge from existing knowledge node."""

        if not self.enable_inference or not self.inference_engine:
            return []

        knowledge = self.knowledge_base.get(key)
        if not knowledge:
            return []

        inferences = self.inference_engine.infer(knowledge)
        inferred_nodes = []

        for content, confidence in inferences:
            inferred_key = hashlib.md5(content.encode()).hexdigest()[:16]

            # Check if already exists
            if inferred_key not in self.knowledge_base:
                self.store(
                    inferred_key,
                    content,
                    category=KnowledgeCategory.FACT,
                    confidence=confidence,
                    source=f"inferred_from_{key}",
                )
                inferred_nodes.append(self.knowledge_base[inferred_key])

        self.stats["inferences_made"] += len(inferred_nodes)
        return inferred_nodes

    def resolve_contradictions(self) -> List[Dict[str, Any]]:
        """Find and resolve contradictory knowledge."""

        contradictions = []

        # Find contradictory relationships
        for key, knowledge in self.knowledge_base.items():
            for rel in self.relationships.get(key, []):
                if rel.relation_type == RelationType.CONTRADICTS:
                    target = self.knowledge_base.get(rel.target)
                    if target:
                        contradictions.append(
                            {
                                "source": knowledge,
                                "target": target,
                                "relationship": rel,
                                "resolution": "conflict_detected",
                            }
                        )

        # Resolve by confidence
        for contra in contradictions:
            if contra["source"].confidence > contra["target"].confidence:
                contra["resolution"] = "source_wins"
                # Reduce confidence of target
                contra["target"].confidence *= 0.8
            else:
                contra["resolution"] = "target_wins"
                contra["source"].confidence *= 0.8

        return contradictions

    # --------------------------------------------------
    # KNOWLEDGE MAINTENANCE
    # --------------------------------------------------
    def update_confidence(self, key: str, delta: float) -> bool:
        """Update confidence of knowledge node."""

        with self._knowledge_lock:
            if key not in self.knowledge_base:
                return False

            knowledge = self.knowledge_base[key]
            knowledge.confidence = max(0.0, min(1.0, knowledge.confidence + delta))
            knowledge.updated_at = datetime.now()
            knowledge.version += 1

            # Save update
            mongo_client.save("semantic", knowledge.to_dict())

            logger.debug(f"📊 Updated confidence for {key}: {knowledge.confidence:.2f}")
            return True

    def decay_knowledge(self, decay_rate: float = 0.01, days_threshold: int = 30):
        """Apply confidence decay to old knowledge."""

        cutoff = datetime.now() - timedelta(days=days_threshold)
        decayed_count = 0

        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if knowledge.last_accessed and knowledge.last_accessed < cutoff:
                    old_confidence = knowledge.confidence
                    knowledge.confidence *= 1 - decay_rate
                    knowledge.confidence = max(0.1, knowledge.confidence)

                    if old_confidence != knowledge.confidence:
                        decayed_count += 1
                        knowledge.updated_at = datetime.now()
                        mongo_client.save("semantic", knowledge.to_dict())

        logger.info(f"📉 Decayed confidence for {decayed_count} knowledge items")
        return decayed_count

    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        """Remove knowledge with confidence below threshold."""

        to_delete = []

        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if knowledge.confidence < threshold:
                    to_delete.append(key)

            for key in to_delete:
                self.delete(key)

        logger.info(f"✂️ Pruned {len(to_delete)} low-confidence knowledge items")
        return len(to_delete)

    def get_version_history(self, key: str) -> List[Dict[str, Any]]:
        """Get version history for knowledge."""

        history = self.knowledge_history.get(key, [])
        return [self._knowledge_to_dict(v) for v in history]

    def rollback_version(self, key: str, version: int) -> bool:
        """Rollback knowledge to previous version."""

        if not self.enable_versioning:
            logger.warning("Versioning not enabled")
            return False

        history = self.knowledge_history.get(key, [])

        # Find target version
        target = None
        for node in history:
            if node.version == version:
                target = node
                break

        if not target:
            logger.warning(f"Version {version} not found for {key}")
            return False

        # Store current version in history
        current = self.knowledge_base.get(key)
        if current:
            self.knowledge_history[key].append(current)

        # Restore target version
        self.knowledge_base[key] = target
        target.version += 1
        target.updated_at = datetime.now()

        # Save to database
        mongo_client.save("semantic", target.to_dict())

        logger.info(f"⏪ Rolled back {key} to version {version}")
        return True

    # --------------------------------------------------
    # STATISTICS AND ANALYTICS
    # --------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive semantic memory statistics."""

        # Calculate most accessed
        most_accessed = sorted(
            [(k, v.access_count) for k, v in self.knowledge_base.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Calculate relationship statistics
        relation_counts = defaultdict(int)
        for relationships in self.relationships.values():
            for rel in relationships:
                relation_counts[rel.relation_type.value] += 1

        # Calculate tag statistics
        tag_counts = {tag: len(keys) for tag, keys in self.tag_index.items()}
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_knowledge": len(self.knowledge_base),
            "total_relationships": self.stats["total_relationships"],
            "avg_confidence": self.stats["avg_confidence"],
            "avg_connections_per_node": self.stats["total_relationships"]
            / max(1, len(self.knowledge_base)),
            "category_distribution": dict(self.stats["category_distribution"]),
            "relationship_distribution": dict(relation_counts),
            "top_tags": dict(top_tags),
            "most_accessed": dict(most_accessed),
            "inferences_made": self.stats["inferences_made"],
            "cache_hit_rate": self._get_cache_hit_rate(),
            "knowledge_with_children": sum(
                1 for k in self.knowledge_base.values() if k.children
            ),
            "max_depth": max(
                (len(self.get_parents(k.key)) for k in self.knowledge_base.values()),
                default=0,
            ),
            "versioning_enabled": self.enable_versioning,
            "auto_link_enabled": self.auto_link,
        }

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""

        # Calculate graph metrics
        nodes = len(self.knowledge_base)
        edges = self.stats["total_relationships"]

        # Calculate density
        if nodes > 1:
            density = (2 * edges) / (nodes * (nodes - 1))
        else:
            density = 0

        # Find isolated nodes
        isolated = sum(
            1
            for key in self.knowledge_base.keys()
            if not self.relationships.get(key)
            and not self.reverse_relationships.get(key)
        )

        # Calculate clustering coefficient (simplified)
        triangles = 0
        for key in self.knowledge_base.keys():
            neighbors = set()
            for rel in self.relationships.get(key, []):
                neighbors.add(rel.target)
            for rel in self.reverse_relationships.get(key, []):
                neighbors.add(rel.source)

            # Count triangles (simplified)
            for neighbor in neighbors:
                for rel2 in self.relationships.get(neighbor, []):
                    if rel2.target in neighbors:
                        triangles += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "density": density,
            "isolated_nodes": isolated,
            "triangles": triangles // 3,  # Each triangle counted 3 times
            "avg_degree": (2 * edges) / max(1, nodes),
            "is_connected": isolated == 0,
        }

    # --------------------------------------------------
    # DELETE AND CLEANUP
    # --------------------------------------------------
    def delete(self, key: str, cascade: bool = False) -> bool:
        """Delete knowledge node with optional cascade."""

        with self._knowledge_lock:
            if key not in self.knowledge_base:
                return False

            knowledge = self.knowledge_base[key]

            # Delete children if cascade
            if cascade:
                for child_key in knowledge.children:
                    self.delete(child_key, cascade=True)

            # Remove from indexes
            if knowledge.category in self.category_index:
                self.category_index[knowledge.category] = [
                    k for k in self.category_index[knowledge.category] if k != key
                ]

            for tag in knowledge.tags:
                if tag in self.tag_index and key in self.tag_index[tag]:
                    self.tag_index[tag].remove(key)

            # Remove relationships
            for rel in self.relationships.get(key, []):
                if rel.target in self.reverse_relationships:
                    self.reverse_relationships[rel.target] = [
                        r
                        for r in self.reverse_relationships[rel.target]
                        if r.source != key
                    ]

            for rel in self.reverse_relationships.get(key, []):
                if rel.source in self.relationships:
                    self.relationships[rel.source] = [
                        r for r in self.relationships[rel.source] if r.target != key
                    ]

            # Remove from storage
            del self.knowledge_base[key]
            if key in self.relationships:
                del self.relationships[key]
            if key in self.reverse_relationships:
                del self.reverse_relationships[key]
            if key in self._cache:
                del self._cache[key]

            # Remove from parent's children
            if knowledge.parent and knowledge.parent in self.knowledge_base:
                parent = self.knowledge_base[knowledge.parent]
                if key in parent.children:
                    parent.children.remove(key)

            # 🔥 DELETE FROM DB
            mongo_client.delete("semantic", key)

            # Delete relationships collection
            try:
                mongo_client.collection(f"semantic_rels_{key}").drop()
            except Exception:
                pass

            # Update statistics
            self.stats["total_knowledge"] = len(self.knowledge_base)
            self.stats["total_relationships"] = sum(
                len(rels) for rels in self.relationships.values()
            )

            logger.debug(f"🗑️ Knowledge deleted: {key}")
            return True

    def delete_batch(self, keys: List[str], cascade: bool = False) -> int:
        """Delete multiple knowledge nodes."""
        success_count = 0
        for key in keys:
            if self.delete(key, cascade):
                success_count += 1
        return success_count

    def clear_category(self, category: Union[KnowledgeCategory, str]) -> int:
        """Clear all knowledge in a category."""

        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                return 0

        keys = self.category_index.get(category, []).copy()
        return self.delete_batch(keys, cascade=False)

    def reset(self, confirm: bool = False) -> bool:
        """Reset all semantic memory (requires confirmation)."""

        if not confirm:
            logger.warning("Reset requires confirmation")
            return False

        with self._knowledge_lock:
            self.knowledge_base.clear()
            self.relationships.clear()
            self.reverse_relationships.clear()
            self.category_index.clear()
            self.tag_index.clear()
            self.knowledge_history.clear()
            self._cache.clear()
            self._query_cache.clear()

            self.stats = {
                "total_knowledge": 0,
                "total_relationships": 0,
                "avg_confidence": 0.0,
                "most_accessed": [],
                "category_distribution": defaultdict(int),
                "inferences_made": 0,
            }

            # Clear database
            try:
                mongo_client.collection("semantic").delete_many({})
                logger.info("🗑️ MongoDB semantic collection cleared")
            except Exception as e:
                logger.error(f"Failed to clear MongoDB: {e}")

            logger.info("🔄 Semantic memory reset")
            return True

    # --------------------------------------------------
    # EXPORT/IMPORT
    # --------------------------------------------------
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all knowledge to JSON file."""

        if not filepath:
            filepath = (
                f"semantic_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_knowledge": len(self.knowledge_base),
            "total_relationships": self.stats["total_relationships"],
            "knowledge": [k.to_dict() for k in self.knowledge_base.values()],
            "relationships": [
                r.to_dict()
                for relationships in self.relationships.values()
                for r in relationships
            ],
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(
            f"📤 Exported {len(export_data['knowledge'])} knowledge nodes to {filepath}"
        )
        return filepath

    def import_from_json(self, filepath: str, overwrite: bool = False) -> int:
        """Import knowledge from JSON file."""

        if overwrite:
            self.reset(confirm=True)

        with open(filepath, "r") as f:
            data = json.load(f)

        imported_count = 0

        # Import knowledge nodes
        for knowledge_data in data.get("knowledge", []):
            knowledge = KnowledgeNode.from_dict(knowledge_data)
            self.knowledge_base[knowledge.key] = knowledge
            imported_count += 1

        # Import relationships
        for rel_data in data.get("relationships", []):
            relationship = Relationship.from_dict(rel_data)
            self.relationships[relationship.source].append(relationship)
            self.reverse_relationships[relationship.target].append(relationship)

        # Rebuild indexes
        self._rebuild_indexes()

        logger.info(f"📥 Imported {imported_count} knowledge nodes from {filepath}")
        return imported_count

    def _rebuild_indexes(self):
        """Rebuild all indexes."""

        self.category_index.clear()
        self.tag_index.clear()

        for key, knowledge in self.knowledge_base.items():
            self.category_index[knowledge.category].append(key)
            for tag in knowledge.tags:
                self.tag_index[tag].append(key)

        self.stats["total_knowledge"] = len(self.knowledge_base)
        self.stats["total_relationships"] = sum(
            len(rels) for rels in self.relationships.values()
        )

        if self.knowledge_base:
            confidences = [k.confidence for k in self.knowledge_base.values()]
            self.stats["avg_confidence"] = statistics.mean(confidences)

    # --------------------------------------------------
    # CACHE MANAGEMENT
    # --------------------------------------------------
    def _update_cache(self, key: str, knowledge: KnowledgeNode):
        """Update LRU cache."""

        if len(self._cache) >= self.cache_size:
            # Remove least recently accessed
            oldest_key = (
                next(iter(self._access_history)) if self._access_history else None
            )
            if oldest_key and oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[key] = knowledge
        self._access_history.append(key)

    def _update_access_metadata(self, knowledge: KnowledgeNode):
        """Update access metadata."""

        knowledge.access_count += 1
        knowledge.last_accessed = datetime.now()

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Simplified - would need actual tracking
        return 0.0

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
    def add_knowledge_callback(self, callback: Callable[[KnowledgeNode], None]):
        """Add callback for new knowledge."""
        self._knowledge_callbacks.append(callback)

    def add_relationship_callback(self, callback: Callable[[Relationship], None]):
        """Add callback for new relationships."""
        self._relationship_callbacks.append(callback)

    def _trigger_knowledge_callbacks(self, knowledge: KnowledgeNode):
        """Trigger all knowledge callbacks."""
        for callback in self._knowledge_callbacks:
            try:
                callback(knowledge)
            except Exception as e:
                logger.error(f"Knowledge callback error: {e}")

    def _trigger_relationship_callbacks(self, relationship: Relationship):
        """Trigger all relationship callbacks."""
        for callback in self._relationship_callbacks:
            try:
                callback(relationship)
            except Exception as e:
                logger.error(f"Relationship callback error: {e}")

    # --------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------
    def get_category(self, key: str) -> Optional[str]:
        """Get category of knowledge."""
        knowledge = self.knowledge_base.get(key)
        return knowledge.category.value if knowledge else None

    def get_confidence(self, key: str) -> float:
        """Get confidence of knowledge."""
        knowledge = self.knowledge_base.get(key)
        return knowledge.confidence if knowledge else 0.0

    def get_all_keys(self) -> List[str]:
        """Get all knowledge keys."""
        return list(self.knowledge_base.keys())

    def get_all_categories(self) -> List[str]:
        """Get all categories."""
        return [c.value for c in KnowledgeCategory]

    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_stats()."""
        return self.get_stats()


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def create_knowledge_key(prefix: str = "know") -> str:
    """Generate unique knowledge key."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}"


__all__ = [
    "SemanticMemory",
    "KnowledgeCategory",
    "RelationType",
    "KnowledgeNode",
    "Relationship",
    "InferenceEngine",
    "create_knowledge_key",
]
