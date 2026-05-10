"""
Episodic Memory - stores specific events and experiences with temporal context,
advanced querying, pattern recognition, and multi-level storage.
(PRODUCTION READY WITH MONGODB INTEGRATION)
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import json
import hashlib
import statistics
import asyncio

from .memory_base import BaseMemory
from .mongo_client import mongo_client
from ..utils.logger import logger


class EventCategory(Enum):
    """Categories for episodic events."""

    USER_INTERACTION = "user_interaction"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    SUCCESS = "success"
    LEARNING = "learning"
    DECISION = "decision"
    PERCEPTION = "perception"
    ACTION = "action"
    OBSERVATION = "observation"
    TASK = "task"
    GOAL = "goal"
    FEEDBACK = "feedback"
    KNOWLEDGE_UPDATE = "knowledge_update"
    AGENT_COMMUNICATION = "agent_communication"


class EventImportance(Enum):
    """Predefined importance levels."""

    TRIVIAL = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9
    EPIC = 1.0


@dataclass
class EpisodeContext:
    """Enhanced context for episodic events."""

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    location: Optional[str] = None
    environment: str = "default"
    preceding_event: Optional[str] = None
    following_event: Optional[str] = None
    related_events: List[str] = field(default_factory=list)
    emotional_state: Optional[str] = None
    cognitive_load: float = 0.5
    attention_level: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeContext":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Episode:
    """Enhanced episode representation."""

    key: str
    content: Any
    timestamp: datetime
    importance: float
    tags: List[str]
    category: EventCategory
    context: EpisodeContext
    outcome: Optional[Any] = None
    reward: float = 0.0
    confidence: float = 1.0
    version: int = 1
    parent_event: Optional[str] = None
    child_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "key": self.key,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "outcome": self.outcome,
            "reward": self.reward,
            "confidence": self.confidence,
            "version": self.version,
            "parent_event": self.parent_event,
            "child_events": self.child_events,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["category"] = EventCategory(data.get("category", "observation"))
        data["context"] = EpisodeContext.from_dict(data.get("context", {}))
        return cls(**data)


class EpisodicMemory(BaseMemory):
    """Advanced episodic memory with temporal context, pattern recognition, and analytics."""

    def __init__(self, cache_size: int = 1000, enable_pattern_detection: bool = True):
        """Initialize episodic memory with advanced features."""
        super().__init__("episodic_memory")

        # Core storage
        self.events: Dict[str, Episode] = {}
        self.timeline: List[Tuple[datetime, str]] = []
        self.cache_size = cache_size
        self.enable_pattern_detection = enable_pattern_detection

        # Indexes for fast querying
        self.category_index: Dict[EventCategory, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.user_index: Dict[str, List[str]] = defaultdict(list)
        self.session_index: Dict[str, List[str]] = defaultdict(list)
        self.date_index: Dict[str, List[str]] = defaultdict(list)

        # Temporal patterns
        self.patterns: Dict[str, Any] = {}
        self.sequences: Dict[str, List[str]] = defaultdict(list)
        self.transitions: Dict[Tuple[str, str], int] = defaultdict(int)

        # Statistics
        self.stats: Dict[str, Any] = {
            "total_episodes": 0,
            "by_category": defaultdict(int),
            "avg_importance": 0.0,
            "peak_hours": defaultdict(int),
            "event_frequencies": defaultdict(int),
        }

        # Cache for recent/frequent events
        self._cache: Dict[str, Episode] = {}
        self._access_count: Dict[str, int] = defaultdict(int)
        self._last_access: Dict[str, datetime] = {}

        # Query cache
        self._query_cache: Dict[str, Tuple[datetime, List[Dict[str, Any]]]] = {}
        self._query_cache_ttl = 60  # seconds

        # Callbacks
        self._event_callbacks: List[Callable[[Episode], None]] = []
        self._pattern_callbacks: List[Callable[[str, Any], None]] = []

        # Async
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None

        # 🔥 LOAD FROM MONGO ON STARTUP
        self._load_from_db()

        logger.info(
            f"✅ EpisodicMemory initialized (cache={cache_size}, patterns={enable_pattern_detection})"
        )

    # --------------------------------------------------
    # LOAD EXISTING DATA
    # --------------------------------------------------
    def _load_from_db(self):
        """Load existing episodes from MongoDB with indexing."""
        try:
            docs = mongo_client.load_all("episodic")

            for doc in docs:
                key = doc.get("key")

                # Create episode object
                episode = Episode.from_dict(doc)

                # Store in memory
                self.events[key] = episode
                self.timeline.append((episode.timestamp, key))

                # Update indexes
                self._update_indexes(episode)

                # Update statistics
                self.stats["total_episodes"] += 1
                self.stats["by_category"][episode.category.value] += 1
                self.stats["event_frequencies"][episode.category.value] += 1

                # Update hour distribution
                hour = episode.timestamp.hour
                self.stats["peak_hours"][hour] += 1

            # Sort timeline
            self.timeline.sort(key=lambda x: x[0])

            # Calculate average importance
            if self.events:
                avg_imp = statistics.mean([e.importance for e in self.events.values()])
                self.stats["avg_importance"] = avg_imp

            # Build patterns if enabled
            if self.enable_pattern_detection:
                self._build_patterns()

            logger.info(
                f"📚 Episodic memory loaded from DB ({len(self.events)} episodes)"
            )

        except Exception as e:
            logger.error(f"Failed loading episodic memory: {e}")

    def _update_indexes(self, episode: Episode):
        """Update all indexes for an episode."""
        # Category index
        self.category_index[episode.category].append(episode.key)

        # Tag index
        for tag in episode.tags:
            self.tag_index[tag].append(episode.key)

        # User index
        if episode.context.user_id:
            self.user_index[episode.context.user_id].append(episode.key)

        # Session index
        if episode.context.session_id:
            self.session_index[episode.context.session_id].append(episode.key)

        # Date index
        date_key = episode.timestamp.date().isoformat()
        self.date_index[date_key].append(episode.key)

        # Update sequences
        if episode.context.preceding_event:
            self.transitions[(episode.context.preceding_event, episode.key)] += 1

        if len(self.timeline) >= 2:
            prev_key = self.timeline[-2][1]
            self.sequences[prev_key].append(episode.key)

    # --------------------------------------------------
    # STORE
    # --------------------------------------------------
    def store(
        self,
        key: str,
        content: Any,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        category: Union[EventCategory, str] = EventCategory.OBSERVATION,
        outcome: Optional[Any] = None,
        reward: float = 0.0,
        confidence: float = 1.0,
        parent_event: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_importance: bool = True,
    ) -> str:
        """Store an episode with enhanced metadata."""

        timestamp = timestamp or datetime.now()

        # Convert category if string
        if isinstance(category, str):
            try:
                category = EventCategory(category)
            except ValueError:
                category = EventCategory.OBSERVATION

        # Auto-calculate importance if enabled
        if auto_importance:
            importance = self._calculate_importance(
                content, category, reward, confidence
            )

        # Create episode context
        episode_context = EpisodeContext()
        if context:
            episode_context = EpisodeContext.from_dict(context)

        # Create episode
        episode = Episode(
            key=key,
            content=content,
            timestamp=timestamp,
            importance=importance,
            tags=tags or [],
            category=category,
            context=episode_context,
            outcome=outcome,
            reward=reward,
            confidence=confidence,
            parent_event=parent_event,
            metadata=metadata or {},
        )

        # Update parent-child relationships
        if parent_event and parent_event in self.events:
            if key not in self.events[parent_event].child_events:
                self.events[parent_event].child_events.append(key)

        # Store in memory
        self.events[key] = episode
        self.timeline.append((timestamp, key))
        self.timeline.sort(key=lambda x: x[0])

        # Update indexes
        self._update_indexes(episode)

        # Update cache
        self._update_cache(key, episode)

        # Update statistics
        self.stats["total_episodes"] += 1
        self.stats["by_category"][category.value] += 1
        self.stats["event_frequencies"][category.value] += 1

        # Update average importance
        all_importances = [e.importance for e in self.events.values()]
        self.stats["avg_importance"] = statistics.mean(all_importances)

        # Update hour distribution
        hour = timestamp.hour
        self.stats["peak_hours"][hour] += 1

        # 🔥 SAVE TO MONGO
        mongo_client.save("episodic", episode.to_dict())

        # Trigger callbacks
        self._trigger_event_callbacks(episode)

        # Update patterns
        if self.enable_pattern_detection:
            self._update_patterns(episode)

        # Maintain size limits
        self._maintain_size()

        logger.debug(
            f"📝 Episode stored: {key} (category={category.value}, importance={importance:.2f})"
        )

        return key

    def _calculate_importance(
        self, content: Any, category: EventCategory, reward: float, confidence: float
    ) -> float:
        """Auto-calculate importance based on multiple factors."""
        importance = 0.5  # Base importance

        # Category-based importance
        category_weights = {
            EventCategory.CRITICAL: 0.3,
            EventCategory.ERROR: 0.25,
            EventCategory.SUCCESS: 0.2,
            EventCategory.DECISION: 0.15,
            EventCategory.LEARNING: 0.1,
            EventCategory.USER_INTERACTION: 0.1,
        }
        importance += category_weights.get(category, 0)

        # Reward-based boost
        importance += min(0.3, abs(reward) / 10)

        # Confidence-based adjustment
        importance *= confidence

        # Content length factor (longer content might be more important)
        if isinstance(content, str):
            importance += min(0.1, len(content) / 1000)

        return min(1.0, importance)

    def _update_cache(self, key: str, episode: Episode):
        """Update LRU cache."""
        self._cache[key] = episode
        self._access_count[key] += 1
        self._last_access[key] = datetime.now()

        # Maintain cache size
        if len(self._cache) > self.cache_size:
            # Remove least recently accessed
            oldest = min(
                self._cache.keys(), key=lambda k: self._last_access.get(k, datetime.min)
            )
            del self._cache[oldest]
            del self._access_count[oldest]
            del self._last_access[oldest]

    def _maintain_size(self):
        """Maintain memory limits by pruning old/low-importance events."""
        if len(self.events) > self.cache_size * 2:
            # Remove oldest and least important events
            to_remove = []
            for timestamp, key in self.timeline[:100]:  # Check oldest 100
                episode = self.events.get(key)
                if episode and episode.importance < 0.3:
                    to_remove.append(key)

            for key in to_remove:
                self.delete(key)

    # --------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------
    def retrieve(self, key: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Retrieve an episode by key with caching."""

        # Check cache first
        if use_cache and key in self._cache:
            episode = self._cache[key]
            self._update_cache(key, episode)
            return episode.to_dict()

        # Check memory
        if key in self.events:
            episode = self.events[key]
            episode.confidence *= 0.999  # Slight confidence decay over time
            self._update_cache(key, episode)
            return episode.to_dict()

        # 🔥 FALLBACK TO MONGO
        doc = mongo_client.load("episodic", key)
        if doc:
            episode = Episode.from_dict(doc)
            self._update_cache(key, episode)
            return episode.to_dict()

        return None

    def retrieve_episode(self, key: str) -> Optional[Episode]:
        """Retrieve full Episode object."""
        if key in self.events:
            return self.events[key]

        doc = mongo_client.load("episodic", key)
        if doc:
            return Episode.from_dict(doc)

        return None

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    def search(
        self,
        query: Optional[str] = None,
        category: Optional[Union[EventCategory, str]] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        limit: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Advanced search with multiple filters."""

        # Check query cache
        cache_key = self._get_cache_key(
            query,
            category,
            tags,
            user_id,
            session_id,
            start_time,
            end_time,
            min_importance,
            max_importance,
        )
        if use_cache and cache_key in self._query_cache:
            cached_time, cached_result = self._query_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._query_cache_ttl:
                return cached_result

        results = []

        # Get candidate keys from indexes
        candidate_keys = set()

        if category:
            if isinstance(category, str):
                try:
                    category = EventCategory(category)
                except ValueError:
                    pass
            keys = self.category_index.get(category, [])
            candidate_keys.update(keys)

        if tags:
            for tag in tags:
                keys = self.tag_index.get(tag, [])
                candidate_keys.update(keys)

        if user_id:
            keys = self.user_index.get(user_id, [])
            candidate_keys.update(keys)

        if session_id:
            keys = self.session_index.get(session_id, [])
            candidate_keys.update(keys)

        # If no specific indexes, search all
        if not candidate_keys:
            candidate_keys = set(self.events.keys())

        # Filter and score
        for key in candidate_keys:
            episode = self.events.get(key)
            if not episode:
                continue

            # Apply filters
            if (
                episode.importance < min_importance
                or episode.importance > max_importance
            ):
                continue

            if start_time and episode.timestamp < start_time:
                continue

            if end_time and episode.timestamp > end_time:
                continue

            # Calculate relevance score
            relevance = 0.0

            if query:
                # Text search
                content_str = str(episode.content).lower()
                if query.lower() in content_str:
                    relevance = 0.7
                    # Boost if query appears early
                    position = content_str.find(query.lower())
                    relevance += max(0, 0.3 * (1 - position / len(content_str)))

                # Tag match
                for tag in episode.tags:
                    if query.lower() in tag.lower():
                        relevance = max(relevance, 0.8)

            # Boost by importance
            relevance += episode.importance * 0.2

            # Boost by recency
            age_hours = (datetime.now() - episode.timestamp).total_seconds() / 3600
            recency_boost = max(0, 1 - age_hours / 168) * 0.1  # Last week
            relevance += recency_boost

            results.append(
                {
                    "key": key,
                    "content": episode.content,
                    "timestamp": episode.timestamp.isoformat(),
                    "importance": episode.importance,
                    "category": episode.category.value,
                    "tags": episode.tags,
                    "relevance": min(1.0, relevance),
                    "confidence": episode.confidence,
                }
            )

        # Sort by relevance and importance
        results.sort(key=lambda x: (x["relevance"], x["importance"]), reverse=True)

        # Cache results
        if use_cache:
            self._query_cache[cache_key] = (datetime.now(), results[:limit])
            # Clean old cache entries
            self._clean_query_cache()

        return results[:limit]

    def _get_cache_key(self, *args) -> str:
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
    # TEMPORAL QUERIES
    # --------------------------------------------------
    def get_recent_events(
        self,
        hours: int = 24,
        limit: int = 10,
        category: Optional[EventCategory] = None,
        min_importance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get recent events with filters."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []

        for timestamp, key in reversed(self.timeline):
            if timestamp < cutoff:
                break

            episode = self.events.get(key)
            if not episode:
                continue

            if category and episode.category != category:
                continue

            if episode.importance < min_importance:
                continue

            recent.append(
                {
                    "key": key,
                    "content": episode.content,
                    "timestamp": timestamp.isoformat(),
                    "importance": episode.importance,
                    "category": episode.category.value,
                }
            )

            if len(recent) >= limit:
                break

        return recent

    def get_events_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get events within date range."""
        results = []

        for timestamp, key in self.timeline:
            if start_date <= timestamp <= end_date:
                episode = self.events.get(key)
                if episode:
                    results.append(
                        {
                            "key": key,
                            "content": episode.content,
                            "timestamp": timestamp.isoformat(),
                            "importance": episode.importance,
                        }
                    )

            if len(results) >= limit:
                break

        return results

    def get_important_events(
        self,
        threshold: float = 0.7,
        limit: int = 10,
        category: Optional[EventCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Get important events with optional category filter."""
        important = []

        for key, episode in sorted(
            self.events.items(), key=lambda x: x[1].importance, reverse=True
        ):
            if episode.importance >= threshold:
                if not category or episode.category == category:
                    important.append(
                        {
                            "key": key,
                            "content": episode.content,
                            "timestamp": episode.timestamp.isoformat(),
                            "importance": episode.importance,
                            "category": episode.category.value,
                            "reward": episode.reward,
                        }
                    )

            if len(important) >= limit:
                break

        return important

    def get_events_by_category(
        self,
        category: EventCategory,
        limit: int = 20,
        sort_by: str = "timestamp",  # timestamp, importance, reward
    ) -> List[Dict[str, Any]]:
        """Get events by category with sorting."""
        keys = self.category_index.get(category, [])
        events = [self.events[k] for k in keys if k in self.events]

        if sort_by == "importance":
            events.sort(key=lambda x: x.importance, reverse=True)
        elif sort_by == "reward":
            events.sort(key=lambda x: x.reward, reverse=True)
        else:  # timestamp
            events.sort(key=lambda x: x.timestamp, reverse=True)

        return [
            {
                "key": e.key,
                "content": e.content,
                "timestamp": e.timestamp.isoformat(),
                "importance": e.importance,
                "reward": e.reward,
            }
            for e in events[:limit]
        ]

    # --------------------------------------------------
    # SEQUENCE AND PATTERN ANALYSIS
    # --------------------------------------------------
    def get_sequence(self, start_key: str, max_length: int = 10) -> List[Episode]:
        """Get sequence of events following a starting event."""
        sequence = []
        current_key = start_key

        for _ in range(max_length):
            if current_key not in self.events:
                break

            episode = self.events[current_key]
            sequence.append(episode)

            if not episode.child_events:
                break

            current_key = episode.child_events[0]  # Follow first child

        return sequence

    def find_patterns(
        self, min_occurrences: int = 3, window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Find recurring patterns in event sequences."""
        patterns = []

        # Look for repeating sequences
        sequences = defaultdict(int)

        for i in range(len(self.timeline) - window_size):
            pattern_key = []
            for j in range(window_size):
                idx = i + j
                if idx < len(self.timeline):
                    key = self.timeline[idx][1]
                    episode = self.events.get(key)
                    if episode:
                        pattern_key.append(episode.category.value)

            pattern_tuple = tuple(pattern_key)
            sequences[pattern_tuple] += 1

        # Extract patterns that occur frequently
        for pattern, count in sequences.items():
            if count >= min_occurrences:
                patterns.append(
                    {
                        "pattern": list(pattern),
                        "occurrences": count,
                        "length": len(pattern),
                        "confidence": count / len(self.timeline),
                    }
                )

        # Sort by occurrence count
        patterns.sort(key=lambda x: x["occurrences"], reverse=True)

        return patterns[:10]

    def get_transition_probability(
        self, from_category: EventCategory, to_category: EventCategory
    ) -> float:
        """Calculate probability of transitioning between event categories."""
        total = 0
        transitions = 0

        for i in range(len(self.timeline) - 1):
            current_key = self.timeline[i][1]
            next_key = self.timeline[i + 1][1]

            current_ep = self.events.get(current_key)
            next_ep = self.events.get(next_key)

            if current_ep and next_ep:
                if current_ep.category == from_category:
                    total += 1
                    if next_ep.category == to_category:
                        transitions += 1

        return transitions / total if total > 0 else 0.0

    def _build_patterns(self):
        """Build pattern database from existing events."""
        if not self.enable_pattern_detection:
            return

        # Find common sequences
        self.patterns["common_sequences"] = self.find_patterns(
            min_occurrences=3, window_size=3
        )

        # Calculate category transition matrix
        categories = list(EventCategory)
        transition_matrix = {}

        for from_cat in categories:
            transition_matrix[from_cat.value] = {}
            for to_cat in categories:
                prob = self.get_transition_probability(from_cat, to_cat)
                if prob > 0:
                    transition_matrix[from_cat.value][to_cat.value] = prob

        self.patterns["transition_matrix"] = transition_matrix

        logger.info(
            f"🔍 Built pattern database with {len(self.patterns.get('common_sequences', []))} patterns"
        )

    def _update_patterns(self, episode: Episode):
        """Update patterns with new episode."""
        if not self.enable_pattern_detection:
            return

        # Check for new patterns in recent events
        if len(self.timeline) >= 5:
            recent_keys = [self.timeline[i][1] for i in range(-5, 0)]
            recent_categories = []

            for key in recent_keys:
                ep = self.events.get(key)
                if ep:
                    recent_categories.append(ep.category.value)

            # Look for new patterns
            pattern_key = tuple(recent_categories)
            if pattern_key not in self.patterns:
                # Count occurrences
                count = 0
                for i in range(len(self.timeline) - 5):
                    pattern = []
                    for j in range(5):
                        idx = i + j
                        if idx < len(self.timeline):
                            key = self.timeline[idx][1]
                            ep = self.events.get(key)
                            if ep:
                                pattern.append(ep.category.value)

                    if tuple(pattern) == pattern_key:
                        count += 1

                if count >= 3:
                    self.patterns[pattern_key] = count
                    self._trigger_pattern_callbacks(pattern_key, count)

    # --------------------------------------------------
    # EPISODE RELATIONSHIPS
    # --------------------------------------------------
    def link_events(self, parent_key: str, child_key: str) -> bool:
        """Link two events as parent-child."""
        if parent_key not in self.events or child_key not in self.events:
            return False

        if child_key not in self.events[parent_key].child_events:
            self.events[parent_key].child_events.append(child_key)

        self.events[child_key].parent_event = parent_key

        # Update in database
        mongo_client.save("episodic", self.events[parent_key].to_dict())
        mongo_client.save("episodic", self.events[child_key].to_dict())

        logger.debug(f"🔗 Linked {parent_key} -> {child_key}")
        return True

    def get_event_chain(self, key: str, direction: str = "both") -> List[Episode]:
        """Get chain of related events (ancestors and descendants)."""
        chain = []

        if direction in ["backward", "both"]:
            # Get ancestors
            current = self.events.get(key)
            while current and current.parent_event:
                parent = self.events.get(current.parent_event)
                if parent:
                    chain.insert(0, parent)
                    current = parent
                else:
                    break

        # Add the event itself
        if key in self.events:
            chain.append(self.events[key])

        if direction in ["forward", "both"]:
            # Get descendants
            current = self.events.get(key)
            while current and current.child_events:
                child = self.events.get(current.child_events[0])
                if child:
                    chain.append(child)
                    current = child
                else:
                    break

        return chain

    # --------------------------------------------------
    # STATISTICS AND ANALYTICS
    # --------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        # Calculate memory usage
        import sys

        memory_usage = sys.getsizeof(self.events) + sys.getsizeof(self.timeline)

        # Calculate event frequency by hour
        peak_hours = sorted(
            self.stats["peak_hours"].items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Calculate retention rate
        week_ago = datetime.now() - timedelta(days=7)
        recent_count = sum(1 for t, _ in self.timeline if t > week_ago)
        retention_rate = recent_count / len(self.timeline) if self.timeline else 0

        return {
            "total_episodes": len(self.events),
            "unique_categories": len(self.category_index),
            "unique_tags": len(self.tag_index),
            "unique_users": len(self.user_index),
            "unique_sessions": len(self.session_index),
            "avg_importance": self.stats["avg_importance"],
            "by_category": dict(self.stats["by_category"]),
            "peak_hours": dict(peak_hours),
            "memory_usage_mb": memory_usage / (1024 * 1024),
            "cache_size": len(self._cache),
            "query_cache_size": len(self._query_cache),
            "total_transitions": len(self.transitions),
            "recent_events_7d": recent_count,
            "retention_rate": retention_rate,
            "patterns_detected": len(self.patterns),
            "avg_confidence": (
                statistics.mean([e.confidence for e in self.events.values()])
                if self.events
                else 0
            ),
        }

    def get_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in events."""
        if len(self.timeline) < 10:
            return {"message": "Insufficient data for pattern analysis"}

        # Calculate inter-event intervals
        intervals = []
        for i in range(1, len(self.timeline)):
            interval = (self.timeline[i][0] - self.timeline[i - 1][0]).total_seconds()
            intervals.append(interval)

        # Calculate event density by hour
        hourly_density = defaultdict(int)
        for timestamp, _ in self.timeline:
            hourly_density[timestamp.hour] += 1

        # Find recurring patterns by day of week
        daily_patterns = defaultdict(int)
        for timestamp, _ in self.timeline:
            daily_patterns[timestamp.strftime("%A")] += 1

        return {
            "avg_interval_seconds": statistics.mean(intervals) if intervals else 0,
            "median_interval_seconds": statistics.median(intervals) if intervals else 0,
            "min_interval_seconds": min(intervals) if intervals else 0,
            "max_interval_seconds": max(intervals) if intervals else 0,
            "hourly_density": dict(hourly_density),
            "peak_hour": (
                max(hourly_density, key=hourly_density.get) if hourly_density else None
            ),
            "daily_patterns": dict(daily_patterns),
            "most_active_day": (
                max(daily_patterns, key=daily_patterns.get) if daily_patterns else None
            ),
        }

    def get_insights(self) -> Dict[str, Any]:
        """Generate insights from episodic memory."""
        insights = {
            "most_common_categories": [],
            "emerging_patterns": [],
            "recommendations": [],
        }

        # Most common categories
        category_counts = self.stats["by_category"]
        insights["most_common_categories"] = sorted(
            [{"category": k, "count": v} for k, v in category_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        # Emerging patterns
        if self.enable_pattern_detection and self.patterns.get("common_sequences"):
            insights["emerging_patterns"] = self.patterns["common_sequences"][:3]

        # Recommendations based on patterns
        if self.patterns.get("transition_matrix"):
            # Find strong transitions that might indicate workflows
            for from_cat, transitions in self.patterns["transition_matrix"].items():
                for to_cat, prob in transitions.items():
                    if prob > 0.8:
                        insights["recommendations"].append(
                            f"Strong workflow detected: {from_cat} → {to_cat} ({prob:.0%})"
                        )

        # Memory health recommendations
        if len(self.events) > 10000:
            insights["recommendations"].append(
                "Memory size large, consider archiving old events"
            )

        if self.stats["avg_importance"] < 0.3:
            insights["recommendations"].append(
                "Low average importance, review event quality"
            )

        return insights

    # --------------------------------------------------
    # MAINTENANCE
    # --------------------------------------------------
    def delete(self, key: str, delete_children: bool = False) -> bool:
        """Delete an episode with optional cascade deletion."""
        if key not in self.events:
            return False

        episode = self.events[key]

        # Delete children if requested
        if delete_children and episode.child_events:
            for child_key in episode.child_events:
                self.delete(child_key, delete_children=True)

        # Remove from indexes
        self._remove_from_indexes(episode)

        # Remove from timeline
        self.timeline = [(t, k) for t, k in self.timeline if k != key]

        # Remove from storage
        del self.events[key]

        # Remove from cache
        if key in self._cache:
            del self._cache[key]

        # 🔥 DELETE FROM MONGO
        mongo_client.delete("episodic", key)

        # Update statistics
        self.stats["total_episodes"] = len(self.events)
        self.stats["by_category"][episode.category.value] -= 1
        if self.stats["by_category"][episode.category.value] == 0:
            del self.stats["by_category"][episode.category.value]

        logger.debug(f"🗑️ Episode deleted: {key}")
        return True

    def _remove_from_indexes(self, episode: Episode):
        """Remove episode from all indexes."""
        # Category index
        if episode.key in self.category_index.get(episode.category, []):
            self.category_index[episode.category].remove(episode.key)

        # Tag index
        for tag in episode.tags:
            if episode.key in self.tag_index.get(tag, []):
                self.tag_index[tag].remove(episode.key)

        # User index
        if episode.context.user_id and episode.key in self.user_index.get(
            episode.context.user_id, []
        ):
            self.user_index[episode.context.user_id].remove(episode.key)

        # Session index
        if episode.context.session_id and episode.key in self.session_index.get(
            episode.context.session_id, []
        ):
            self.session_index[episode.context.session_id].remove(episode.key)

        # Date index
        date_key = episode.timestamp.date().isoformat()
        if episode.key in self.date_index.get(date_key, []):
            self.date_index[date_key].remove(episode.key)

    def clear_old_events(self, days: int = 30, min_importance: float = 0.3) -> int:
        """Clear events older than N days with importance below threshold."""
        cutoff = datetime.now() - timedelta(days=days)
        to_delete = []

        for key, episode in self.events.items():
            if episode.timestamp < cutoff and episode.importance < min_importance:
                to_delete.append(key)

        for key in to_delete:
            self.delete(key)

        logger.info(
            f"🧹 Cleared {len(to_delete)} old events (>{days} days, importance<{min_importance})"
        )
        return len(to_delete)

    def archive_events(self, days: int = 90) -> int:
        """Archive old events (mark as archived but don't delete)."""
        cutoff = datetime.now() - timedelta(days=days)
        archived_count = 0

        for key, episode in self.events.items():
            if episode.timestamp < cutoff and not episode.metadata.get(
                "archived", False
            ):
                episode.metadata["archived"] = True
                episode.metadata["archived_date"] = datetime.now().isoformat()
                episode.importance *= 0.5  # Reduce importance of archived events

                # Update in database
                mongo_client.save("episodic", episode.to_dict())
                archived_count += 1

        logger.info(f"📦 Archived {archived_count} events (>{days} days)")
        return archived_count

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export all episodes to JSON file."""
        if not filepath:
            filepath = (
                f"episodic_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_episodes": len(self.events),
            "episodes": [episode.to_dict() for episode in self.events.values()],
            "stats": self.get_stats(),
            "patterns": self.patterns,
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"📤 Exported {len(self.events)} episodes to {filepath}")
        return filepath

    # --------------------------------------------------
    # TIMELINE
    # --------------------------------------------------
    def get_timeline(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[EventCategory] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get timeline with filters."""
        timeline = []

        for timestamp, key in self.timeline:
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue

            episode = self.events.get(key)
            if not episode:
                continue

            if category and episode.category != category:
                continue

            timeline.append(
                {
                    "key": key,
                    "timestamp": timestamp.isoformat(),
                    "importance": episode.importance,
                    "category": episode.category.value,
                    "tags": episode.tags,
                    "content_preview": (
                        str(episode.content)[:100]
                        if isinstance(episode.content, str)
                        else str(episode.content)
                    ),
                }
            )

            if len(timeline) >= limit:
                break

        return timeline

    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------
    def add_event_callback(self, callback: Callable[[Episode], None]):
        """Add callback for new events."""
        self._event_callbacks.append(callback)

    def add_pattern_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for pattern detection."""
        self._pattern_callbacks.append(callback)

    def _trigger_event_callbacks(self, episode: Episode):
        """Trigger all event callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(episode)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def _trigger_pattern_callbacks(self, pattern: str, count: int):
        """Trigger all pattern callbacks."""
        for callback in self._pattern_callbacks:
            try:
                callback(pattern, count)
            except Exception as e:
                logger.error(f"Pattern callback error: {e}")

    # --------------------------------------------------
    # ASYNC OPERATIONS
    # --------------------------------------------------
    async def start_background_analysis(self, interval_seconds: int = 300):
        """Start background pattern analysis."""
        self._running = True

        async def analyze_loop():
            while self._running:
                await asyncio.sleep(interval_seconds)
                if self.enable_pattern_detection:
                    self._build_patterns()
                    logger.debug("Background pattern analysis completed")

        self._analysis_task = asyncio.create_task(analyze_loop())
        logger.info("🚀 Background pattern analysis started")

    async def stop_background_analysis(self):
        """Stop background analysis."""
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Background pattern analysis stopped")

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------
    def reset(self, confirm: bool = False) -> bool:
        """Reset all episodic memory (requires confirmation)."""
        if not confirm:
            logger.warning("Reset requires confirmation")
            return False

        self.events.clear()
        self.timeline.clear()
        self.category_index.clear()
        self.tag_index.clear()
        self.user_index.clear()
        self.session_index.clear()
        self.date_index.clear()
        self.patterns.clear()
        self.sequences.clear()
        self.transitions.clear()
        self._cache.clear()
        self._access_count.clear()
        self._last_access.clear()
        self._query_cache.clear()

        self.stats = {
            "total_episodes": 0,
            "by_category": defaultdict(int),
            "avg_importance": 0.0,
            "peak_hours": defaultdict(int),
            "event_frequencies": defaultdict(int),
        }

        # Clear database
        try:
            mongo_client.collection("episodic").delete_many({})
            logger.info("🗑️ MongoDB episodic collection cleared")
        except Exception as e:
            logger.error(f"Failed to clear MongoDB: {e}")

        logger.info("🔄 Episodic memory reset")
        return True


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def create_episode_key(prefix: str = "ep") -> str:
    """Generate unique episode key."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}"


__all__ = [
    "EpisodicMemory",
    "EventCategory",
    "EventImportance",
    "EpisodeContext",
    "Episode",
    "create_episode_key",
]
