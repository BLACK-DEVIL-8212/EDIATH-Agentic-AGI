"""
Semantic Memory - stores general knowledge, facts, and learned information with
graph relationships, inference capabilities, and knowledge graph support.
(PRODUCTION READY WITH MONGODB INTEGRATION)
<<<<<<< HEAD

Placement in Architecture Flowchart:
- Located in: MEMORY MANAGER / MEMORY STORAGE layer
- Interacts with: ORCHESTRATOR CORE, REASONING CORE, INFERENCE CORE
- Provides: Knowledge storage, relationship management, inference capabilities
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
import asyncio
from functools import wraps

# Import based on actual project structure
try:
    from .memory_base import BaseMemory
    from .mongo_client import mongo_client
    from ..utils.logger import logger
except ImportError:
    # Fallback for standalone testing
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    class BaseMemory:
        def __init__(self, name):
            self.name = name
    
    class mongo_client:
        @staticmethod
        def save(collection, data):
            logger.info(f"Would save to {collection}: {data.get('key', 'unknown')}")
            return True
        
        @staticmethod
        def load(collection, key):
            return None
        
        @staticmethod
        def load_all(collection):
            return []
        
        @staticmethod
        def delete(collection, key):
            return True
=======

from .memory_base import BaseMemory
from .mongo_client import mongo_client
from ..utils.logger import logger
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7


class RelationType(Enum):
    """Types of semantic relationships."""
<<<<<<< HEAD
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PRECEDES = "precedes"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    EXAMPLE_OF = "example_of"
    DEFINED_AS = "defined_as"
    DERIVED_FROM = "derived_from"
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7


class KnowledgeCategory(Enum):
    """Categories of knowledge."""
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
    embedding: Optional[List[float]] = None  # For semantic search
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

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
<<<<<<< HEAD
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
=======
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            "version": self.version,
            "parent": self.parent,
            "children": self.children,
            "metadata": self.metadata,
<<<<<<< HEAD
            "embedding": self.embedding,
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        """Create from dictionary."""
<<<<<<< HEAD
=======
        # Convert category string to enum
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        category_str = data.get("category", "general")
        try:
            category = KnowledgeCategory(category_str)
        except ValueError:
            category = KnowledgeCategory.GENERAL

<<<<<<< HEAD
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        last_accessed = datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

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
<<<<<<< HEAD
            embedding=data.get("embedding"),
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        )


@dataclass
class Relationship:
    """Enhanced relationship between knowledge nodes."""
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    source: str
    target: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
<<<<<<< HEAD
=======
        """Convert to dictionary."""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
=======
        """Create from dictionary."""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=RelationType(data.get("relation_type", "related_to")),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
<<<<<<< HEAD
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
=======
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            metadata=data.get("metadata", {}),
        )


class InferenceEngine:
<<<<<<< HEAD
    """Knowledge inference engine - connects to REASONING CORE and INFERENCE CORE"""
    
    def __init__(self, memory: "SemanticMemory"):
        self.memory = memory
        self.inference_rules: List[Callable] = []
        self._llm_inference_callback: Optional[Callable] = None

    def register_llm_callback(self, callback: Callable[[str], Any]):
        """Register callback to LLM engine for advanced inference"""
        self._llm_inference_callback = callback

    def infer(self, knowledge: KnowledgeNode) -> List[Tuple[str, float, Dict]]:
        """Infer new knowledge from existing with enhanced reasoning"""
=======
    """Knowledge inference engine for deriving new knowledge."""

    def __init__(self, memory: "SemanticMemory"):
        self.memory = memory
        self.inference_rules: List[Callable] = []

    def infer(self, knowledge: KnowledgeNode) -> List[Tuple[str, float]]:
        """Infer new knowledge from existing knowledge."""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        inferences = []

        # Rule 1: Transitive relationships
        if knowledge.category == KnowledgeCategory.FACT:
            related = self.memory.get_related_knowledge(
                knowledge.key, relation_type=RelationType.RELATED_TO
            )
            for rel in related:
<<<<<<< HEAD
                if rel.get("confidence", 0) > 0.7:
                    inferences.append((
                        f"{knowledge.content} is related to {rel.get('content', 'unknown')}",
                        rel.get("confidence", 0.5) * 0.8,
                        {"inference_type": "transitive", "source": "relationship"}
                    ))
=======
                if rel.confidence > 0.7:
                    inferences.append(
                        (
                            f"{knowledge.content} is related to {rel.content}",
                            rel.confidence * 0.8,
                        )
                    )
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

        # Rule 2: IS-A hierarchy
        if knowledge.category == KnowledgeCategory.CONCEPT:
            parents = self.memory.get_parents(knowledge.key)
            for parent in parents:
<<<<<<< HEAD
                inferences.append((
                    f"{knowledge.content} is a type of {parent.content}",
                    parent.confidence * 0.9,
                    {"inference_type": "is_a", "source": "hierarchy"}
                ))

        # Rule 3: Cause-effect chains
        causes = self.memory.get_relationships(knowledge.key, relation_type=RelationType.CAUSES)
        for cause in causes:
            inferences.append((
                f"{cause.target if cause.source == knowledge.key else cause.source} can lead to {knowledge.content}",
                cause.confidence * 0.7,
                {"inference_type": "causal", "source": "relationship"}
            ))

        # Rule 4: LLM-enhanced inference if available
        if self._llm_inference_callback and isinstance(knowledge.content, str):
            try:
                llm_result = self._llm_inference_callback(
                    f"Based on: '{knowledge.content}', infer related knowledge or implications."
                )
                if llm_result:
                    inferences.append((
                        str(llm_result),
                        0.6,
                        {"inference_type": "llm_enhanced", "source": "llm_core"}
                    ))
            except Exception as e:
                logger.debug(f"LLM inference failed: {e}")
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

        return inferences


class SemanticMemory(BaseMemory):
<<<<<<< HEAD
    """
    Advanced semantic memory with knowledge graph, inference, and analytics.
    
    Architecture Placement:
    - Receives requests from ORCHESTRATOR CORE and MEMORY MANAGER
    - Provides knowledge to REASONING CORE and INFERENCE CORE
    - Stores data in MEMORY STORAGE layer (MongoDB)
    - Integrates with VECTOR DATABASE for semantic search
    """
=======
    """Advanced semantic memory with knowledge graph, inference, and analytics."""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

    def __init__(
        self,
        enable_inference: bool = True,
        enable_versioning: bool = True,
        cache_size: int = 1000,
        max_history: int = 100,
        auto_link: bool = True,
<<<<<<< HEAD
        vector_db_client: Optional[Any] = None,  # For Qdrant/ChromaDB integration
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    ):
        """Initialize advanced semantic memory."""
        super().__init__("semantic_memory")

        # Core storage
        self.knowledge_base: Dict[str, KnowledgeNode] = {}
        self.relationships: Dict[str, List[Relationship]] = defaultdict(list)
        self.reverse_relationships: Dict[str, List[Relationship]] = defaultdict(list)

<<<<<<< HEAD
        # Indexes for fast retrieval
=======
        # Indexes
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.category_index: Dict[KnowledgeCategory, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.confidence_index: Dict[str, List[str]] = defaultdict(list)

        # Features
        self.enable_inference = enable_inference
        self.enable_versioning = enable_versioning
        self.cache_size = cache_size
        self.max_history = max_history
        self.auto_link = auto_link
<<<<<<< HEAD
        self.vector_db = vector_db_client
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

        # Version history
        self.knowledge_history: Dict[str, List[KnowledgeNode]] = defaultdict(list)

        # Caching
        self._cache: Dict[str, KnowledgeNode] = {}
        self._access_history: deque = deque(maxlen=cache_size)

<<<<<<< HEAD
        # Inference engine - connects to REASONING/INFERENCE cores
        self.inference_engine = InferenceEngine(self) if enable_inference else None

        # Statistics for OBSERVABILITY STACK
=======
        # Inference engine
        self.inference_engine = InferenceEngine(self) if enable_inference else None

        # Statistics
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.stats = {
            "total_knowledge": 0,
            "total_relationships": 0,
            "avg_confidence": 0.0,
            "most_accessed": [],
            "category_distribution": defaultdict(int),
            "inferences_made": 0,
<<<<<<< HEAD
            "cache_hits": 0,
            "cache_misses": 0,
            "vector_searches": 0,
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        }

        # Query cache
        self._query_cache: Dict[str, Tuple[datetime, List[Dict]]] = {}
        self._query_cache_ttl = 60

<<<<<<< HEAD
        # Thread safety
        self._knowledge_lock = threading.RLock()
        
        # Async task queue for background operations
        self._async_tasks: List[asyncio.Task] = []

        # Callbacks for integrating with other cores
        self._knowledge_callbacks: List[Callable[[KnowledgeNode], None]] = []
        self._relationship_callbacks: List[Callable[[Relationship], None]] = []
        self._inference_callbacks: List[Callable[[str, Any], None]] = []

        # Load from persistent storage
        self._load_from_db()

        logger.info(
            f"✅ SemanticMemory initialized (inference={enable_inference}, "
            f"versioning={enable_versioning}, auto_link={auto_link}, "
            f"vector_db={vector_db_client is not None})"
        )

    # ================================================================
    # PERSISTENCE LAYER (MEMORY STORAGE integration)
    # ================================================================
    
    def _load_from_db(self):
        """Load all knowledge from MongoDB with relationships."""
        try:
            docs = mongo_client.load_all("semantic")
            
            for doc in docs:
                key = doc.get("key")
                knowledge = KnowledgeNode.from_dict(doc)
                
                self.knowledge_base[key] = knowledge
                self.category_index[knowledge.category].append(key)
                for tag in knowledge.tags:
                    self.tag_index[tag].append(key)
                
                # Load relationships
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                rel_docs = mongo_client.load_all(f"semantic_rels_{key}")
                for rel_doc in rel_docs:
                    relationship = Relationship.from_dict(rel_doc)
                    self.relationships[key].append(relationship)
                    self.reverse_relationships[relationship.target].append(relationship)
<<<<<<< HEAD
            
            # Update statistics
            self.stats["total_knowledge"] = len(self.knowledge_base)
            self.stats["total_relationships"] = sum(len(rels) for rels in self.relationships.values())
            
            if self.knowledge_base:
                confidences = [k.confidence for k in self.knowledge_base.values()]
                self.stats["avg_confidence"] = statistics.mean(confidences)
            
            for category, keys in self.category_index.items():
                self.stats["category_distribution"][category.value] = len(keys)
            
            logger.info(f"📚 Loaded {len(self.knowledge_base)} nodes, {self.stats['total_relationships']} relationships")
            
        except Exception as e:
            logger.error(f"Load failed: {e}")

    def _save_to_db(self, knowledge: KnowledgeNode):
        """Save knowledge to persistent storage"""
        try:
            mongo_client.save("semantic", knowledge.to_dict())
            
            # Also save to vector DB if available
            if self.vector_db and knowledge.embedding:
                self._save_to_vector_db(knowledge)
                
        except Exception as e:
            logger.error(f"Save to DB failed: {e}")

    def _save_to_vector_db(self, knowledge: KnowledgeNode):
        """Save embedding to vector database for semantic search"""
        try:
            # This integrates with VECTOR DATABASE layer (Qdrant/ChromaDB)
            self.vector_db.upsert(
                collection_name="semantic_memory",
                points=[{
                    "id": knowledge.key,
                    "vector": knowledge.embedding,
                    "payload": {
                        "content": knowledge.content,
                        "category": knowledge.category.value,
                        "tags": knowledge.tags,
                    }
                }]
            )
        except Exception as e:
            logger.debug(f"Vector DB save failed: {e}")

    # ================================================================
    # STORE KNOWLEDGE (from ORCHESTRATOR/MEMORY MANAGER)
    # ================================================================
    
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Store knowledge with enhanced metadata."""
        try:
=======
    ) -> bool:
        """Store knowledge with enhanced metadata."""
        try:
            # Convert category if string
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            if isinstance(category, str):
                try:
                    category = KnowledgeCategory(category)
                except ValueError:
                    category = KnowledgeCategory.GENERAL

<<<<<<< HEAD
            existing = self.knowledge_base.get(key)

            with self._knowledge_lock:
=======
            # Check if updating existing knowledge
            existing = self.knowledge_base.get(key)

            with self._knowledge_lock:
                # Create knowledge node
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
                    embedding=embedding,
                )

                # Version history
                if self.enable_versioning and existing:
                    self.knowledge_history[key].append(existing)
                    if len(self.knowledge_history[key]) > self.max_history:
                        self.knowledge_history[key] = self.knowledge_history[key][-self.max_history:]

=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                self.knowledge_base[key] = knowledge

                # Update indexes
                if existing and existing.category != category:
<<<<<<< HEAD
                    if existing.category in self.category_index:
                        self.category_index[existing.category] = [k for k in self.category_index[existing.category] if k != key]
                
                self.category_index[category].append(key)

=======
                    # Remove from old category index
                    if existing.category in self.category_index:
                        self.category_index[existing.category] = [
                            k
                            for k in self.category_index[existing.category]
                            if k != key
                        ]

                self.category_index[category].append(key)

                # Update tag index
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                if existing:
                    for old_tag in existing.tags:
                        if old_tag in self.tag_index and key in self.tag_index[old_tag]:
                            self.tag_index[old_tag].remove(key)

                for tag in tags or []:
                    self.tag_index[tag].append(key)

<<<<<<< HEAD
                if parent:
                    knowledge.parent = parent
                    if parent in self.knowledge_base and key not in self.knowledge_base[parent].children:
                        self.knowledge_base[parent].children.append(key)

                self._update_cache(key, knowledge)
                self._update_statistics()
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

                # Auto-link related knowledge
                if self.auto_link:
                    self._auto_link_knowledge(knowledge)

<<<<<<< HEAD
                # Save to persistent storage
                self._save_to_db(knowledge)
=======
                # 🔥 SAVE TO MONGO
                mongo_client.save("semantic", knowledge.to_dict())
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

                # Trigger callbacks
                self._trigger_knowledge_callbacks(knowledge)

<<<<<<< HEAD
                # Perform inference (connects to INFERENCE CORE)
                if self.enable_inference and self.inference_engine:
                    self._perform_inference_async(knowledge)

                logger.debug(f"📝 Stored: {key} (category={category.value}, confidence={confidence:.2f})")
                return True

        except Exception as e:
            logger.error(f"Store error: {e}")
            return False

    def _perform_inference_async(self, knowledge: KnowledgeNode):
        """Perform inference asynchronously to not block storage"""
        try:
            inferences = self.inference_engine.infer(knowledge)
            for inferred_content, inferred_conf, inference_meta in inferences:
                inferred_key = hashlib.md5(inferred_content.encode()).hexdigest()[:16]
                
                # Store inferred knowledge with source tracking
                self.store(
                    inferred_key,
                    inferred_content,
                    category=KnowledgeCategory.FACT,
                    confidence=inferred_conf,
                    source=f"inferred_from_{knowledge.key}",
                    metadata={"inference_metadata": inference_meta}
                )
                self.stats["inferences_made"] += 1
                
                # Notify inference callbacks
                for callback in self._inference_callbacks:
                    try:
                        callback(knowledge.key, inferred_content)
                    except Exception as e:
                        logger.error(f"Inference callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Inference failed: {e}")

    def _auto_link_knowledge(self, knowledge: KnowledgeNode):
        """Automatically link related knowledge"""
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for other_key, other_knowledge in self.knowledge_base.items():
            if other_key == knowledge.key:
                continue

<<<<<<< HEAD
            common_tags = set(knowledge.tags) & set(other_knowledge.tags)
            if len(common_tags) >= 2:
                self.link_knowledge(
                    knowledge.key, other_key,
                    relation_type=RelationType.RELATED_TO,
                    weight=len(common_tags) / max(len(knowledge.tags), len(other_knowledge.tags))
                )

            if knowledge.category == other_knowledge.category:
                self.link_knowledge(
                    knowledge.key, other_key,
                    relation_type=RelationType.RELATED_TO,
                    weight=0.5
                )

    # ================================================================
    # RETRIEVE KNOWLEDGE (for REASONING CORE)
    # ================================================================
    
    def retrieve(self, key: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge by key."""
        if use_cache and key in self._cache:
            self.stats["cache_hits"] += 1
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            knowledge = self._cache[key]
            self._update_access_metadata(knowledge)
            return self._knowledge_to_dict(knowledge)

<<<<<<< HEAD
        self.stats["cache_misses"] += 1
        
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        with self._knowledge_lock:
            if key in self.knowledge_base:
                knowledge = self.knowledge_base[key]
                self._update_access_metadata(knowledge)
                self._update_cache(key, knowledge)
                return self._knowledge_to_dict(knowledge)

<<<<<<< HEAD
        doc = mongo_client.load("semantic", key)
        return doc

    def retrieve_batch(self, keys: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve multiple knowledge items."""
        return {key: self.retrieve(key) for key in keys}

    def retrieve_by_category(self, category: Union[KnowledgeCategory, str]) -> List[Dict[str, Any]]:
        """Retrieve all knowledge in a category"""
        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                return []
        
        keys = self.category_index.get(category, [])
        return [self._knowledge_to_dict(self.knowledge_base[k]) for k in keys if k in self.knowledge_base]

    def retrieve_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve most recently updated knowledge"""
        sorted_knowledge = sorted(
            self.knowledge_base.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )
        return [self._knowledge_to_dict(k) for k in sorted_knowledge[:limit]]

    def _knowledge_to_dict(self, knowledge: KnowledgeNode) -> Dict[str, Any]:
        """Convert knowledge node to dictionary."""
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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

<<<<<<< HEAD
    # ================================================================
    # RELATIONSHIP MANAGEMENT (Knowledge Graph)
    # ================================================================
    
=======
    # --------------------------------------------------
    # RELATIONSHIP MANAGEMENT
    # --------------------------------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
        
=======

        # Convert relation type if string
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType(relation_type)
            except ValueError:
                relation_type = RelationType.RELATED_TO

<<<<<<< HEAD
        if source_key not in self.knowledge_base or target_key not in self.knowledge_base:
=======
        # Check if both nodes exist
        if (
            source_key not in self.knowledge_base
            or target_key not in self.knowledge_base
        ):
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            logger.warning(f"Cannot link: {source_key} or {target_key} not found")
            return False

        with self._knowledge_lock:
<<<<<<< HEAD
=======
            # Create relationship
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            relationship = Relationship(
                source=source_key,
                target=target_key,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence,
                metadata=metadata or {},
            )

<<<<<<< HEAD
            self.relationships[source_key].append(relationship)
            self.reverse_relationships[target_key].append(relationship)
            self.stats["total_relationships"] += 1

            mongo_client.save(f"semantic_rels_{source_key}", relationship.to_dict())
            self._trigger_relationship_callbacks(relationship)

            logger.debug(f"🔗 Linked {source_key} -> {target_key} ({relation_type.value})")
            return True

    def get_relationships(
        self, key: str, relation_type: Optional[RelationType] = None, direction: str = "outgoing"
    ) -> List[Relationship]:
        """Get relationships for a knowledge node."""
        
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if direction == "outgoing":
            relationships = self.relationships.get(key, [])
        elif direction == "incoming":
            relationships = self.reverse_relationships.get(key, [])
<<<<<<< HEAD
        else:
            relationships = self.relationships.get(key, []) + self.reverse_relationships.get(key, [])

        if relation_type:
            relationships = [r for r in relationships if r.relation_type == relation_type]
        
        return relationships

    def get_related_knowledge(
        self, key: str, relation_type: Optional[RelationType] = None, depth: int = 1, min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get related knowledge with graph traversal."""
        
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        related = []
        visited = set()
        queue = deque([(key, 0)])

        while queue:
            current_key, current_depth = queue.popleft()
<<<<<<< HEAD
            
            if current_key in visited or current_depth > depth:
                continue
            
            visited.add(current_key)
            
            if current_key in self.knowledge_base:
                knowledge = self.knowledge_base[current_key]
                
                if knowledge.confidence >= min_confidence and current_key != key:
                    related.append({
                        "key": current_key,
                        "content": knowledge.content,
                        "category": knowledge.category.value,
                        "confidence": knowledge.confidence,
                        "depth": current_depth,
                        "relationships": [
                            {"type": r.relation_type.value, "weight": r.weight}
                            for r in self.get_relationships(current_key, relation_type, "incoming")
                            if r.source == key or r.target == key
                        ],
                    })
                
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                for rel in self.relationships.get(current_key, []):
                    if relation_type is None or rel.relation_type == relation_type:
                        if rel.target not in visited:
                            queue.append((rel.target, current_depth + 1))

<<<<<<< HEAD
        related.sort(key=lambda x: (x["depth"], x["confidence"]), reverse=True)
=======
        # Sort by depth and confidence
        related.sort(key=lambda x: (x["depth"], x["confidence"]), reverse=True)

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return related

    def get_parents(self, key: str) -> List[KnowledgeNode]:
        """Get parent nodes in hierarchy."""
        parents = []
        current = self.knowledge_base.get(key)
<<<<<<< HEAD
        
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        while current and current.parent:
            parent = self.knowledge_base.get(current.parent)
            if parent:
                parents.append(parent)
                current = parent
            else:
                break
<<<<<<< HEAD
        
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return parents

    def get_children(self, key: str, recursive: bool = False) -> List[KnowledgeNode]:
        """Get child nodes in hierarchy."""
        children = []
<<<<<<< HEAD
        
        if key not in self.knowledge_base:
            return children
        
=======

        if key not in self.knowledge_base:
            return children

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for child_key in self.knowledge_base[key].children:
            child = self.knowledge_base.get(child_key)
            if child:
                children.append(child)
                if recursive:
                    children.extend(self.get_children(child_key, recursive=True))
<<<<<<< HEAD
        
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return children

    def get_knowledge_graph(self, root_key: str, max_depth: int = 3) -> Dict[str, Any]:
        """Build knowledge graph starting from root."""
<<<<<<< HEAD
        
        graph = {"nodes": {}, "edges": []}
        
        def traverse(node_key: str, depth: int):
            if depth > max_depth or node_key not in self.knowledge_base:
                return
            
            node = self.knowledge_base[node_key]
            
=======

        graph = {"nodes": {}, "edges": []}

        def traverse(node_key: str, depth: int):
            if depth > max_depth or node_key not in self.knowledge_base:
                return

            node = self.knowledge_base[node_key]

            # Add node
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            graph["nodes"][node_key] = {
                "label": str(node.content)[:50],
                "category": node.category.value,
                "confidence": node.confidence,
            }
<<<<<<< HEAD
            
            for rel in self.relationships.get(node_key, []):
                if rel.target in self.knowledge_base:
                    graph["edges"].append({
                        "source": node_key,
                        "target": rel.target,
                        "type": rel.relation_type.value,
                        "weight": rel.weight,
                    })
                    traverse(rel.target, depth + 1)
        
        traverse(root_key, 0)
        return graph

    # ================================================================
    # SEARCH (for RETRIEVAL/RAG system)
    # ================================================================
    
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
        
=======

        # Check query cache
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        cache_key = self._get_query_cache_key(query, category, tags, min_confidence)
        if cache_key in self._query_cache:
            cached_time, cached_result = self._query_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._query_cache_ttl:
                return cached_result

        results = []
        query_lower = query.lower()

<<<<<<< HEAD
=======
        # Convert category if provided
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                category = None

<<<<<<< HEAD
        # Semantic search via vector DB if available
        if use_semantic and self.vector_db:
            self.stats["vector_searches"] += 1
            results = self._semantic_search(query, limit)
        
        # Fallback to keyword search
        if not results:
            with self._knowledge_lock:
                for key, knowledge in self.knowledge_base.items():
                    if knowledge.confidence < min_confidence:
                        continue
                    if category and knowledge.category != category:
                        continue
                    if tags and not any(tag in knowledge.tags for tag in tags):
                        continue
                    
                    relevance = self._calculate_relevance(query_lower, knowledge)
                    
                    if relevance > 0:
                        results.append({
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                            "key": key,
                            "content": knowledge.content,
                            "category": knowledge.category.value,
                            "confidence": knowledge.confidence,
                            "source": knowledge.source,
                            "tags": knowledge.tags,
                            "relevance": relevance,
                            "access_count": knowledge.access_count,
<<<<<<< HEAD
                        })
        
        results.sort(key=lambda x: (x.get("relevance", 0), x["confidence"]), reverse=True)
        self._query_cache[cache_key] = (datetime.now(), results[:limit])
        self._clean_query_cache()
        
        return results[:limit]

    def _calculate_relevance(self, query_lower: str, knowledge: KnowledgeNode) -> float:
        """Calculate relevance score for keyword search"""
        relevance = 0.0
        
        if knowledge.key.lower() == query_lower:
            relevance = 1.0
        elif isinstance(knowledge.content, str):
            content_lower = knowledge.content.lower()
            if query_lower in content_lower:
                position = content_lower.find(query_lower)
                relevance = 0.7 * (1 - position / max(len(content_lower), 1))
        
        for tag in knowledge.tags:
            if query_lower in tag.lower():
                relevance = max(relevance, 0.6)
        
        if query_lower in knowledge.category.value:
            relevance = max(relevance, 0.5)
        
        if relevance > 0:
            relevance *= 0.5 + knowledge.confidence * 0.5
        
        return relevance

    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform semantic search using vector database"""
        try:
            results = self.vector_db.search(
                collection_name="semantic_memory",
                query_text=query,
                limit=limit
            )
            return [
                {
                    "key": r.id,
                    "content": r.payload.get("content"),
                    "category": r.payload.get("category"),
                    "relevance": r.score,
                    "source": "semantic_search",
                }
                for r in results
            ]
        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return []

    def search_by_category(self, category: Union[KnowledgeCategory, str]) -> List[Dict[str, Any]]:
        """Get all knowledge in a category."""
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                return []
<<<<<<< HEAD
        
        keys = self.category_index.get(category, [])
        return [self._knowledge_to_dict(self.knowledge_base[key]) for key in keys if key in self.knowledge_base]

    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Dict[str, Any]]:
        """Search knowledge by tags."""
        results = []
        
=======

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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if match_all:
                    if all(tag in knowledge.tags for tag in tags):
                        results.append(self._knowledge_to_dict(knowledge))
                else:
                    if any(tag in knowledge.tags for tag in tags):
                        results.append(self._knowledge_to_dict(knowledge))
<<<<<<< HEAD
        
        return results

    # ================================================================
    # INFERENCE (connects to INFERENCE CORE)
    # ================================================================
    
    def infer_new_knowledge(self, key: str) -> List[KnowledgeNode]:
        """Infer new knowledge from existing knowledge node."""
        if not self.enable_inference or not self.inference_engine:
            return []
        
        knowledge = self.knowledge_base.get(key)
        if not knowledge:
            return []
        
        inferences = self.inference_engine.infer(knowledge)
        inferred_nodes = []
        
        for content, confidence, meta in inferences:
            inferred_key = hashlib.md5(content.encode()).hexdigest()[:16]
            
            if inferred_key not in self.knowledge_base:
                self.store(
                    inferred_key, content,
                    category=KnowledgeCategory.FACT,
                    confidence=confidence,
                    source=f"inferred_from_{key}",
                    metadata=meta
                )
                inferred_nodes.append(self.knowledge_base[inferred_key])
        
=======

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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.stats["inferences_made"] += len(inferred_nodes)
        return inferred_nodes

    def resolve_contradictions(self) -> List[Dict[str, Any]]:
        """Find and resolve contradictory knowledge."""
<<<<<<< HEAD
        contradictions = []
        
=======

        contradictions = []

        # Find contradictory relationships
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for key, knowledge in self.knowledge_base.items():
            for rel in self.relationships.get(key, []):
                if rel.relation_type == RelationType.CONTRADICTS:
                    target = self.knowledge_base.get(rel.target)
                    if target:
<<<<<<< HEAD
                        contradictions.append({
                            "source": knowledge,
                            "target": target,
                            "relationship": rel,
                            "resolution": "conflict_detected",
                        })
        
        for contra in contradictions:
            if contra["source"].confidence > contra["target"].confidence:
                contra["resolution"] = "source_wins"
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                contra["target"].confidence *= 0.8
            else:
                contra["resolution"] = "target_wins"
                contra["source"].confidence *= 0.8
<<<<<<< HEAD
        
        return contradictions

    # ================================================================
    # MAINTENANCE (for MEMORY MANAGER)
    # ================================================================
    
    def update_confidence(self, key: str, delta: float) -> bool:
        """Update confidence of knowledge node."""
        with self._knowledge_lock:
            if key not in self.knowledge_base:
                return False
            
=======

        return contradictions

    # --------------------------------------------------
    # KNOWLEDGE MAINTENANCE
    # --------------------------------------------------
    def update_confidence(self, key: str, delta: float) -> bool:
        """Update confidence of knowledge node."""

        with self._knowledge_lock:
            if key not in self.knowledge_base:
                return False

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            knowledge = self.knowledge_base[key]
            knowledge.confidence = max(0.0, min(1.0, knowledge.confidence + delta))
            knowledge.updated_at = datetime.now()
            knowledge.version += 1
<<<<<<< HEAD
            
            self._save_to_db(knowledge)
            return True

    def decay_knowledge(self, decay_rate: float = 0.01, days_threshold: int = 30) -> int:
        """Apply confidence decay to old knowledge."""
        cutoff = datetime.now() - timedelta(days=days_threshold)
        decayed_count = 0
        
=======

            # Save update
            mongo_client.save("semantic", knowledge.to_dict())

            logger.debug(f"📊 Updated confidence for {key}: {knowledge.confidence:.2f}")
            return True

    def decay_knowledge(self, decay_rate: float = 0.01, days_threshold: int = 30):
        """Apply confidence decay to old knowledge."""

        cutoff = datetime.now() - timedelta(days=days_threshold)
        decayed_count = 0

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if knowledge.last_accessed and knowledge.last_accessed < cutoff:
                    old_confidence = knowledge.confidence
                    knowledge.confidence *= 1 - decay_rate
                    knowledge.confidence = max(0.1, knowledge.confidence)
<<<<<<< HEAD
                    
                    if old_confidence != knowledge.confidence:
                        decayed_count += 1
                        knowledge.updated_at = datetime.now()
                        self._save_to_db(knowledge)
        
=======

                    if old_confidence != knowledge.confidence:
                        decayed_count += 1
                        knowledge.updated_at = datetime.now()
                        mongo_client.save("semantic", knowledge.to_dict())

        logger.info(f"📉 Decayed confidence for {decayed_count} knowledge items")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return decayed_count

    def prune_low_confidence(self, threshold: float = 0.3) -> int:
        """Remove knowledge with confidence below threshold."""
<<<<<<< HEAD
        to_delete = []
        
=======

        to_delete = []

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        with self._knowledge_lock:
            for key, knowledge in self.knowledge_base.items():
                if knowledge.confidence < threshold:
                    to_delete.append(key)
<<<<<<< HEAD
            
            for key in to_delete:
                self.delete(key)
        
=======

            for key in to_delete:
                self.delete(key)

        logger.info(f"✂️ Pruned {len(to_delete)} low-confidence knowledge items")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return len(to_delete)

    def get_version_history(self, key: str) -> List[Dict[str, Any]]:
        """Get version history for knowledge."""
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        history = self.knowledge_history.get(key, [])
        return [self._knowledge_to_dict(v) for v in history]

    def rollback_version(self, key: str, version: int) -> bool:
        """Rollback knowledge to previous version."""
<<<<<<< HEAD
        if not self.enable_versioning:
            logger.warning("Versioning not enabled")
            return False
        
        history = self.knowledge_history.get(key, [])
        target = None
        
=======

        if not self.enable_versioning:
            logger.warning("Versioning not enabled")
            return False

        history = self.knowledge_history.get(key, [])

        # Find target version
        target = None
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for node in history:
            if node.version == version:
                target = node
                break
<<<<<<< HEAD
        
        if not target:
            logger.warning(f"Version {version} not found for {key}")
            return False
        
        current = self.knowledge_base.get(key)
        if current:
            self.knowledge_history[key].append(current)
        
        self.knowledge_base[key] = target
        target.version += 1
        target.updated_at = datetime.now()
        
        self._save_to_db(target)
        return True

    # ================================================================
    # DELETE AND CLEANUP
    # ================================================================
    
    def delete(self, key: str, cascade: bool = False) -> bool:
        """Delete knowledge node with optional cascade."""
        with self._knowledge_lock:
            if key not in self.knowledge_base:
                return False
            
            knowledge = self.knowledge_base[key]
            
            if cascade:
                for child_key in knowledge.children:
                    self.delete(child_key, cascade=True)
            
            if knowledge.category in self.category_index:
                self.category_index[knowledge.category] = [k for k in self.category_index[knowledge.category] if k != key]
            
            for tag in knowledge.tags:
                if tag in self.tag_index and key in self.tag_index[tag]:
                    self.tag_index[tag].remove(key)
            
            for rel in self.relationships.get(key, []):
                if rel.target in self.reverse_relationships:
                    self.reverse_relationships[rel.target] = [r for r in self.reverse_relationships[rel.target] if r.source != key]
            
            for rel in self.reverse_relationships.get(key, []):
                if rel.source in self.relationships:
                    self.relationships[rel.source] = [r for r in self.relationships[rel.source] if r.target != key]
            
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            del self.knowledge_base[key]
            if key in self.relationships:
                del self.relationships[key]
            if key in self.reverse_relationships:
                del self.reverse_relationships[key]
            if key in self._cache:
                del self._cache[key]
<<<<<<< HEAD
            
=======

            # Remove from parent's children
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            if knowledge.parent and knowledge.parent in self.knowledge_base:
                parent = self.knowledge_base[knowledge.parent]
                if key in parent.children:
                    parent.children.remove(key)
<<<<<<< HEAD
            
            mongo_client.delete("semantic", key)
            
=======

            # 🔥 DELETE FROM DB
            mongo_client.delete("semantic", key)

            # Delete relationships collection
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            try:
                mongo_client.collection(f"semantic_rels_{key}").drop()
            except Exception:
                pass
<<<<<<< HEAD
            
            self._update_statistics()
=======

            # Update statistics
            self.stats["total_knowledge"] = len(self.knowledge_base)
            self.stats["total_relationships"] = sum(
                len(rels) for rels in self.relationships.values()
            )

            logger.debug(f"🗑️ Knowledge deleted: {key}")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if isinstance(category, str):
            try:
                category = KnowledgeCategory(category)
            except ValueError:
                return 0
<<<<<<< HEAD
        
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        keys = self.category_index.get(category, []).copy()
        return self.delete_batch(keys, cascade=False)

    def reset(self, confirm: bool = False) -> bool:
<<<<<<< HEAD
        """Reset all semantic memory."""
        if not confirm:
            logger.warning("Reset requires confirmation")
            return False
        
=======
        """Reset all semantic memory (requires confirmation)."""

        if not confirm:
            logger.warning("Reset requires confirmation")
            return False

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        with self._knowledge_lock:
            self.knowledge_base.clear()
            self.relationships.clear()
            self.reverse_relationships.clear()
            self.category_index.clear()
            self.tag_index.clear()
            self.knowledge_history.clear()
            self._cache.clear()
            self._query_cache.clear()
<<<<<<< HEAD
            
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            self.stats = {
                "total_knowledge": 0,
                "total_relationships": 0,
                "avg_confidence": 0.0,
                "most_accessed": [],
                "category_distribution": defaultdict(int),
                "inferences_made": 0,
<<<<<<< HEAD
                "cache_hits": 0,
                "cache_misses": 0,
                "vector_searches": 0,
            }
            
            try:
                mongo_client.collection("semantic").delete_many({})
            except Exception as e:
                logger.error(f"Failed to clear MongoDB: {e}")
            
            return True

    # ================================================================
    # STATISTICS (for OBSERVABILITY STACK)
    # ================================================================
    
    def _update_statistics(self):
        """Update internal statistics."""
        self.stats["total_knowledge"] = len(self.knowledge_base)
        self.stats["total_relationships"] = sum(len(rels) for rels in self.relationships.values())
        
        if self.knowledge_base:
            confidences = [k.confidence for k in self.knowledge_base.values()]
            self.stats["avg_confidence"] = statistics.mean(confidences)
        
        self.stats["category_distribution"] = defaultdict(int)
        for category, keys in self.category_index.items():
            self.stats["category_distribution"][category.value] = len(keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive semantic memory statistics."""
        most_accessed = sorted(
            [(k, v.access_count) for k, v in self.knowledge_base.items()],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        relation_counts = defaultdict(int)
        for relationships in self.relationships.values():
            for rel in relationships:
                relation_counts[rel.relation_type.value] += 1
        
        tag_counts = {tag: len(keys) for tag, keys in self.tag_index.items()}
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        cache_hit_rate = self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
        
        return {
            "total_knowledge": len(self.knowledge_base),
            "total_relationships": self.stats["total_relationships"],
            "avg_confidence": self.stats["avg_confidence"],
            "avg_connections_per_node": self.stats["total_relationships"] / max(1, len(self.knowledge_base)),
            "category_distribution": dict(self.stats["category_distribution"]),
            "relationship_distribution": dict(relation_counts),
            "top_tags": dict(top_tags),
            "most_accessed": dict(most_accessed),
            "inferences_made": self.stats["inferences_made"],
            "cache_hit_rate": cache_hit_rate,
            "vector_searches": self.stats["vector_searches"],
            "knowledge_with_children": sum(1 for k in self.knowledge_base.values() if k.children),
            "max_depth": max((len(self.get_parents(k.key)) for k in self.knowledge_base.values()), default=0),
            "versioning_enabled": self.enable_versioning,
            "auto_link_enabled": self.auto_link,
            "vector_db_enabled": self.vector_db is not None,
        }

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        nodes = len(self.knowledge_base)
        edges = self.stats["total_relationships"]
        
        density = (2 * edges) / (nodes * (nodes - 1)) if nodes > 1 else 0
        
        isolated = sum(1 for key in self.knowledge_base.keys() 
                      if not self.relationships.get(key) and not self.reverse_relationships.get(key))
        
        return {
            "nodes": nodes,
            "edges": edges,
            "density": density,
            "isolated_nodes": isolated,
            "avg_degree": (2 * edges) / max(1, nodes),
            "is_connected": isolated == 0,
        }

    # ================================================================
    # EXPORT/IMPORT
    # ================================================================
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all knowledge to JSON file."""
        if not filepath:
            filepath = f"semantic_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_knowledge": len(self.knowledge_base),
            "total_relationships": self.stats["total_relationships"],
            "knowledge": [k.to_dict() for k in self.knowledge_base.values()],
<<<<<<< HEAD
            "relationships": [r.to_dict() for relationships in self.relationships.values() for r in relationships],
        }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return filepath

    def import_from_json(self, filepath: str, overwrite: bool = False) -> int:
        """Import knowledge from JSON file."""
<<<<<<< HEAD
        if overwrite:
            self.reset(confirm=True)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        imported_count = 0
        
=======

        if overwrite:
            self.reset(confirm=True)

        with open(filepath, "r") as f:
            data = json.load(f)

        imported_count = 0

        # Import knowledge nodes
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for knowledge_data in data.get("knowledge", []):
            knowledge = KnowledgeNode.from_dict(knowledge_data)
            self.knowledge_base[knowledge.key] = knowledge
            imported_count += 1
<<<<<<< HEAD
        
=======

        # Import relationships
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for rel_data in data.get("relationships", []):
            relationship = Relationship.from_dict(rel_data)
            self.relationships[relationship.source].append(relationship)
            self.reverse_relationships[relationship.target].append(relationship)
<<<<<<< HEAD
        
        self._rebuild_indexes()
=======

        # Rebuild indexes
        self._rebuild_indexes()

        logger.info(f"📥 Imported {imported_count} knowledge nodes from {filepath}")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return imported_count

    def _rebuild_indexes(self):
        """Rebuild all indexes."""
<<<<<<< HEAD
        self.category_index.clear()
        self.tag_index.clear()
        
=======

        self.category_index.clear()
        self.tag_index.clear()

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        for key, knowledge in self.knowledge_base.items():
            self.category_index[knowledge.category].append(key)
            for tag in knowledge.tags:
                self.tag_index[tag].append(key)
<<<<<<< HEAD
        
        self._update_statistics()

    # ================================================================
    # CALLBACKS AND INTEGRATION
    # ================================================================
    
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def add_knowledge_callback(self, callback: Callable[[KnowledgeNode], None]):
        """Add callback for new knowledge."""
        self._knowledge_callbacks.append(callback)

    def add_relationship_callback(self, callback: Callable[[Relationship], None]):
        """Add callback for new relationships."""
        self._relationship_callbacks.append(callback)

<<<<<<< HEAD
    def add_inference_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for inference results."""
        self._inference_callbacks.append(callback)

    def register_llm_inference(self, llm_callback: Callable[[str], Any]):
        """Register LLM engine for enhanced inference."""
        if self.inference_engine:
            self.inference_engine.register_llm_callback(llm_callback)

=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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

<<<<<<< HEAD
    # ================================================================
    # CACHE MANAGEMENT
    # ================================================================
    
    def _update_cache(self, key: str, knowledge: KnowledgeNode):
        """Update LRU cache."""
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._access_history)) if self._access_history else None
            if oldest_key and oldest_key in self._cache:
                del self._cache[oldest_key]
        
        self._cache[key] = knowledge
        self._access_history.append(key)

    def _update_access_metadata(self, knowledge: KnowledgeNode):
        """Update access metadata."""
        knowledge.access_count += 1
        knowledge.last_accessed = datetime.now()

    def _get_query_cache_key(self, *args) -> str:
        """Generate cache key for query."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _clean_query_cache(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired = [key for key, (cached_time, _) in self._query_cache.items() 
                  if (now - cached_time).seconds > self._query_cache_ttl]
        
        for key in expired:
            del self._query_cache[key]

    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
=======
    # --------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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


<<<<<<< HEAD
# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================
=======
# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

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
<<<<<<< HEAD
]
=======
]
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
