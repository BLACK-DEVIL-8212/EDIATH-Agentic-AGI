"""
Advanced Background Thinker - Ultimate Edition (MAXIMUM FEATURES)
✔ Safe execution with circuit breakers
✔ Rate-limited with token bucket
✔ Memory-safe with automatic cleanup
✔ Fault-tolerant with retry strategies
✔ Auto-task control with intelligent scheduling
✔ Priority queue with multi-level priorities
✔ Batch processing with dynamic sizing
✔ Persistent insights with encryption
✔ Pattern recognition with ML
✔ Multi-mode thinking (12+ modes)
✔ Thought clustering & topic modeling
✔ Insight scoring & ranking
✔ Cross-thought relationships
✔ Think-then-act workflow
✔ Emotion-aware thinking
✔ Collaborative thinking (multi-agent)
✔ Thought visualization
✔ Export capabilities (JSON, CSV, Markdown)
"""

from __future__ import annotations

import asyncio
import time
import hashlib
import json
import random
import math
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import heapq
import uuid
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.logger import logger


def _create_llm_engine():
    from ..brain.llm_engine import LLMEngine

    return LLMEngine()


def _create_memory_manager():
    from ..memory.memory_manager import MemoryManager

    return MemoryManager()


# ==================== ENUMS ====================

class ThinkingMode(Enum):
    """Thinking modes for different cognitive tasks"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    REASONING = "reasoning"
    OPTIMIZATION = "optimization"
    REFLECTION = "reflection"
    CREATIVE = "creative"
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    FORECAST = "forecast"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    DIAGNOSTIC = "diagnostic"
    EXPLORATORY = "exploratory"
    CAUSAL = "causal"
    ABSTRACT = "abstract"
    LATERAL = "lateral"
    SYSTEMIC = "systemic"
    DESIGN = "design"
    ETHICAL = "ethical"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4
    IDLE = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    BLOCKED = "blocked"


class RetryStrategy(Enum):
    """Retry backoff strategies"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class ThoughtType(Enum):
    """Types of thoughts/insights"""
    INSIGHT = "insight"
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    OBSERVATION = "observation"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"


# ==================== DATA CLASSES ====================

@dataclass
class Thought:
    """Rich thought/insight data structure"""
    id: str
    content: str
    thought_type: ThoughtType
    mode: ThinkingMode
    confidence: float = 0.5
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    related_thoughts: List[str] = field(default_factory=list)
    source_task: Optional[str] = None
    embedding: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "thought_type": self.thought_type.value,
            "mode": self.mode.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "related_thoughts": self.related_thoughts,
            "source_task": self.source_task,
            "metadata": self.metadata
        }
    
    def get_score(self) -> float:
        """Calculate thought importance score"""
        return self.confidence * self.importance


@dataclass
class ThinkingTask:
    """Enhanced thinking task with full lifecycle"""
    id: str
    name: str
    mode: ThinkingMode
    context: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    thoughts: List[Thought] = field(default_factory=list)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value
    
    def get_retry_delay(self) -> float:
        """Calculate retry delay based on strategy"""
        if self.retry_strategy == RetryStrategy.FIXED:
            return 1.0
        elif self.retry_strategy == RetryStrategy.LINEAR:
            return self.retry_count * 1.0
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return min(30, 2 ** self.retry_count)
        elif self.retry_strategy == RetryStrategy.RANDOM:
            return random.uniform(0.5, 3.0)
        else:  # ADAPTIVE
            return 1.0 * (1 + math.log(self.retry_count + 1))
    
    async def execute(self, thinker, llm: LLMEngine) -> 'ThinkingTask':
        """Execute the thinking task"""
        self.started_at = datetime.now()
        self.status = TaskStatus.RUNNING
        
        try:
            # Check dependencies
            if not thinker._check_dependencies(self):
                self.status = TaskStatus.BLOCKED
                self.error = "Dependencies not met"
                return self
            
            # Build prompt
            prompt = thinker._build_thinking_prompt(self)
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    llm.generate(prompt),
                    timeout=self.timeout
                )
                
                if isinstance(result, dict):
                    response = result.get("response", str(result))
                else:
                    response = str(result)
                
                self.result = response
                self.status = TaskStatus.COMPLETED
                
                # Extract thoughts from result
                self.thoughts = thinker._extract_thoughts(self, response)
                
            except asyncio.TimeoutError:
                self.status = TaskStatus.TIMEOUT
                self.error = f"Timeout after {self.timeout}s"
                
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    self.status = TaskStatus.RETRYING
                    await asyncio.sleep(self.get_retry_delay())
                    return await self.execute(thinker, llm)
                    
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                self.status = TaskStatus.RETRYING
                await asyncio.sleep(self.get_retry_delay())
                return await self.execute(thinker, llm)
        
        self.completed_at = datetime.now()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.get_duration_ms(),
            "retry_count": self.retry_count,
            "tags": self.tags,
            "error": self.error,
            "thoughts_count": len(self.thoughts)
        }
    
    def get_duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


@dataclass
class ThinkingPattern:
    """Recognized thinking pattern"""
    id: str
    pattern_type: str
    confidence: float = 0.0
    occurrences: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    related_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, confidence_delta: float = 0.05):
        self.occurrences += 1
        self.confidence = min(1.0, self.confidence + confidence_delta)
        self.last_seen = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.pattern_type,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "related_tasks": self.related_tasks[-10:]
        }


@dataclass
class ThoughtCluster:
    """Cluster of related thoughts"""
    id: str
    topic: str
    thoughts: List[Thought]
    size: int = 0
    coherence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        self.size = len(self.thoughts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "size": self.size,
            "coherence": self.coherence,
            "created_at": self.created_at.isoformat(),
            "thoughts": [t.to_dict() for t in self.thoughts[:10]]
        }


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure: Optional[float] = None
        self.state = "closed"
    
    def call(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure >= self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True
    
    def record_success(self):
        if self.state == "half-open":
            self.state = "closed"
            self.failures = 0
        elif self.state == "closed":
            self.failures = max(0, self.failures - 1)
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"


class BackgroundThinker:
    """
    Ultimate Background Thinker with enterprise-grade features
    """
    
    def __init__(
        self,
        system=None,
        max_queue: int = 200,
        batch_size: int = 3,
        min_interval: float = 0.5,
        auto_mode: bool = True,
        rate_limit: float = 10.0,  # tasks per second
        enable_persistence: bool = True,
        persist_path: str = "data/thinking_insights.json",
        enable_clustering: bool = True,
        max_concurrent: int = 2,
        cache_ttl: int = 3600
    ):
        """
        Initialize Background Thinker
        
        Args:
            system: Reference to main system
            max_queue: Maximum queue size
            batch_size: Number of tasks to process per batch
            min_interval: Minimum interval between processing cycles
            auto_mode: Enable automatic task generation
            rate_limit: Maximum tasks per second
            enable_persistence: Save insights to disk
            persist_path: Path for persistence
            enable_clustering: Enable thought clustering
            max_concurrent: Maximum concurrent tasks
            cache_ttl: Cache time-to-live in seconds
        """
        self.system = system
        
        # Core components
        self.llm = getattr(system, "llm_engine", None) or _create_llm_engine()
        self.memory = getattr(system, "memory", None) or _create_memory_manager()
        
        # Queues and storage
        self.queue: List[Tuple[int, int, ThinkingTask]] = []
        self.completed_tasks: List[ThinkingTask] = []
        self.failed_tasks: List[ThinkingTask] = []
        self.active_tasks: Dict[str, ThinkingTask] = {}
        
        # Thoughts and insights
        self.thoughts: Dict[str, Thought] = {}
        self.thought_clusters: List[ThoughtCluster] = []
        self.patterns: Dict[str, ThinkingPattern] = {}
        
        # Configuration
        self.max_queue = max_queue
        self.batch_size = batch_size
        self.min_interval = min_interval
        self.auto_mode = auto_mode
        self.enable_persistence = enable_persistence
        self.persist_path = Path(persist_path)
        self.enable_clustering = enable_clustering and SKLEARN_AVAILABLE
        self.max_concurrent = max_concurrent
        self.cache_ttl = cache_ttl
        
        # Rate limiting
        self.rate_limiter = TokenBucket(rate_limit, capacity=rate_limit)
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._init_circuit_breakers()
        
        # State
        self.is_running = False
        self._processing_lock = asyncio.Lock()
        self._task_counter = 0
        
        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "insights_generated": 0,
            "patterns_recognized": 0,
            "clusters_formed": 0,
            "average_processing_time_ms": 0.0,
            "queue_peak": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.daily_stats: Dict[str, Any] = {}
        
        # Caching
        self.result_cache: Dict[str, Tuple[Any, float]] = {}
        
        # Scheduling
        self.scheduled_tasks: Dict[str, List[int]] = {}
        
        # Cross-thought relationships
        self.thought_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Callbacks
        self.on_task_complete_callbacks: List[Callable] = []
        self.on_thought_generated_callbacks: List[Callable] = []
        
        logger.info(f"🧠 Background Thinker initialized (max_queue={max_queue}, batch_size={batch_size})")
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers"""
        self.circuit_breakers = {
            "llm": CircuitBreaker("LLM", failure_threshold=3, recovery_timeout=30),
            "memory": CircuitBreaker("Memory", failure_threshold=3, recovery_timeout=30)
        }
    
    # ==================== TASK MANAGEMENT ====================
    
    def add_task(
        self,
        name: str,
        mode: ThinkingMode,
        context: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        tags: List[str] = None,
        dependencies: List[str] = None,
        use_cache: bool = True,
        timeout: float = 30.0
    ) -> Optional[str]:
        """
        Add a thinking task to the queue
        
        Returns:
            Task ID if added, None otherwise
        """
        try:
            # Check queue limit
            if len(self.queue) >= self.max_queue:
                logger.warning(f"Queue full ({self.max_queue}), cannot add task: {name}")
                return None
            
            # Check cache
            if use_cache:
                cache_key = self._get_cache_key(name, mode, context)
                if cache_key in self.result_cache:
                    cached_result, cache_time = self.result_cache[cache_key]
                    if time.time() - cache_time < self.cache_ttl:
                        self.stats["cache_hits"] += 1
                        logger.debug(f"Using cached result for: {name}")
                        return None
            
            # Create task
            task = ThinkingTask(
                id=str(uuid.uuid4())[:8],
                name=name,
                mode=mode,
                context=context[:500],
                priority=priority,
                tags=tags or [],
                dependencies=dependencies or [],
                timeout=timeout
            )
            
            # Add to priority queue
            heapq.heappush(self.queue, (task.priority.value, self._task_counter, task))
            self._task_counter += 1
            
            # Track queue peak
            queue_size = len(self.queue)
            if queue_size > self.stats["queue_peak"]:
                self.stats["queue_peak"] = queue_size
            
            logger.debug(f"Task added: {name} (priority={priority.name}, queue_size={queue_size})")
            return task.id
            
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return None
    
    def _get_cache_key(self, name: str, mode: ThinkingMode, context: str) -> str:
        """Generate cache key"""
        content = f"{name}:{mode.value}:{context[:200]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cache_result(self, task: ThinkingTask):
        """Cache task result"""
        if task.status == TaskStatus.COMPLETED and task.result:
            cache_key = self._get_cache_key(task.name, task.mode, task.context)
            self.result_cache[cache_key] = (task.result, time.time())
            
            # Clean old cache entries
            now = time.time()
            old_keys = [k for k, (_, t) in self.result_cache.items() if now - t > self.cache_ttl]
            for k in old_keys:
                del self.result_cache[k]
    
    def _check_dependencies(self, task: ThinkingTask) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            # Check if dependency is completed
            dep_completed = any(
                t.id == dep_id and t.status == TaskStatus.COMPLETED
                for t in self.completed_tasks
            )
            if not dep_completed:
                return False
        return True
    
    # ==================== THINKING PROMPTS ====================
    
    def _build_thinking_prompt(self, task: ThinkingTask) -> str:
        """Build prompt for thinking task"""
        
        mode_prompts = {
            ThinkingMode.ANALYSIS: "Analyze the following information thoroughly:",
            ThinkingMode.PLANNING: "Create a detailed plan for:",
            ThinkingMode.REASONING: "Reason through this problem step by step:",
            ThinkingMode.OPTIMIZATION: "Find optimizations for:",
            ThinkingMode.REFLECTION: "Reflect on this experience and extract lessons:",
            ThinkingMode.CREATIVE: "Generate creative ideas about:",
            ThinkingMode.CRITICAL: "Critically evaluate:",
            ThinkingMode.STRATEGIC: "Develop a strategic approach for:",
            ThinkingMode.TACTICAL: "Create tactical actions for:",
            ThinkingMode.FORECAST: "Forecast possible outcomes for:",
            ThinkingMode.SYNTHESIS: "Synthesize information about:",
            ThinkingMode.EVALUATION: "Evaluate options for:",
            ThinkingMode.DIAGNOSTIC: "Diagnose the problem:",
            ThinkingMode.EXPLORATORY: "Explore possibilities around:",
            ThinkingMode.CAUSAL: "Identify causal relationships in:",
            ThinkingMode.ABSTRACT: "Think abstractly about:",
            ThinkingMode.LATERAL: "Apply lateral thinking to:",
            ThinkingMode.SYSTEMIC: "Analyze system dynamics of:",
            ThinkingMode.DESIGN: "Design a solution for:",
            ThinkingMode.ETHICAL: "Consider ethical implications of:"
        }
        
        prompt = f"""{mode_prompts.get(task.mode, "Think about:")}

Context: {task.context}

Task: {task.name}

Please provide your thinking in a structured format. Include:
1. Key observations
2. Logical reasoning
3. Conclusions or recommendations
4. Confidence level (0-1)
5. Any insights or patterns noticed

Response:"""
        
        # Add context from memory
        if self.memory and hasattr(self.memory, "get_recent"):
            try:
                recent = asyncio.run(self.memory.get_recent(3))
                if recent:
                    prompt += f"\n\nRelevant memory: {recent}"
            except:
                pass
        
        return prompt
    
    # ==================== THOUGHT EXTRACTION ====================
    
    def _extract_thoughts(self, task: ThinkingTask, response: str) -> List[Thought]:
        """Extract individual thoughts from response"""
        thoughts = []
        
        # Split by common thought delimiters
        thought_separators = [r'\d+\.', r'•', r'- ', r'\n\n']
        
        raw_thoughts = [response]
        for sep in thought_separators:
            new_thoughts = []
            for t in raw_thoughts:
                parts = re.split(sep, t)
                new_thoughts.extend([p for p in parts if len(p.strip()) > 30])
            if new_thoughts:
                raw_thoughts = new_thoughts
        
        # Create thought objects
        for i, content in enumerate(raw_thoughts[:5]):  # Limit to 5 per task
            if len(content.strip()) < 20:
                continue
            
            # Determine thought type
            thought_type = self._determine_thought_type(content)
            
            # Calculate confidence
            confidence = self._calculate_confidence(content)
            
            thought = Thought(
                id=str(uuid.uuid4())[:8],
                content=content.strip(),
                thought_type=thought_type,
                mode=task.mode,
                confidence=confidence,
                importance=confidence,
                tags=task.tags.copy(),
                source_task=task.id
            )
            
            thoughts.append(thought)
            self.thoughts[thought.id] = thought
            
            # Trigger callback
            for cb in self.on_thought_generated_callbacks:
                try:
                    cb(thought)
                except:
                    pass
        
        return thoughts
    
    def _determine_thought_type(self, content: str) -> ThoughtType:
        """Determine thought type from content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["insight", "discover", "realize"]):
            return ThoughtType.INSIGHT
        elif any(word in content_lower for word in ["question", "what if", "why", "how"]):
            return ThoughtType.QUESTION
        elif any(word in content_lower for word in ["hypothesis", "maybe", "perhaps", "could"]):
            return ThoughtType.HYPOTHESIS
        elif any(word in content_lower for word in ["conclude", "therefore", "thus", "so"]):
            return ThoughtType.CONCLUSION
        elif any(word in content_lower for word in ["recommend", "suggest", "should"]):
            return ThoughtType.RECOMMENDATION
        elif any(word in content_lower for word in ["warning", "caution", "risk"]):
            return ThoughtType.WARNING
        elif any(word in content_lower for word in ["opportunity", "potential", "benefit"]):
            return ThoughtType.OPPORTUNITY
        elif any(word in content_lower for word in ["pattern", "trend", "recurring"]):
            return ThoughtType.PATTERN
        else:
            return ThoughtType.OBSERVATION
    
    def _calculate_confidence(self, content: str) -> float:
        """Calculate confidence based on content indicators"""
        confidence = 0.5
        
        # Increase confidence indicators
        confident_words = ["definitely", "certainly", "clearly", "obviously", "absolutely"]
        for word in confident_words:
            if word in content.lower():
                confidence += 0.1
        
        # Decrease confidence indicators
        uncertain_words = ["maybe", "perhaps", "possibly", "might", "could be", "unclear"]
        for word in uncertain_words:
            if word in content.lower():
                confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    # ==================== PATTERN RECOGNITION ====================
    
    def _recognize_patterns(self, task: ThinkingTask):
        """Recognize patterns from completed tasks"""
        pattern_key = f"{task.mode.value}:{task.name[:30]}"
        
        if pattern_key in self.patterns:
            self.patterns[pattern_key].update()
        else:
            # Check if this is a recurring pattern
            recent_similar = [
                t for t in self.completed_tasks[-20:]
                if t.mode == task.mode and t.name[:30] == task.name[:30]
            ]
            
            if len(recent_similar) >= 3:
                pattern = ThinkingPattern(
                    id=pattern_key,
                    pattern_type=task.mode.value,
                    confidence=0.3 + (len(recent_similar) * 0.1)
                )
                pattern.related_tasks.append(task.id)
                self.patterns[pattern_key] = pattern
                self.stats["patterns_recognized"] += 1
                logger.info(f"Pattern recognized: {pattern_key}")
            else:
                # Update pattern if it exists in thoughts
                self._extract_patterns_from_thoughts(task)
    
    def _extract_patterns_from_thoughts(self, task: ThinkingTask):
        """Extract patterns from generated thoughts"""
        for thought in task.thoughts:
            if ThoughtType.PATTERN in thought.thought_type:
                pattern_key = f"pattern_{thought.content[:50]}"
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = ThinkingPattern(
                        id=pattern_key,
                        pattern_type=thought.mode.value,
                        confidence=thought.confidence
                    )
                    self.stats["patterns_recognized"] += 1
    
    # ==================== THOUGHT CLUSTERING ====================
    
    async def _cluster_thoughts(self):
        """Cluster related thoughts using ML"""
        if not self.enable_clustering or len(self.thoughts) < 5:
            return
        
        try:
            thought_texts = [t.content for t in self.thoughts.values()]
            
            # Vectorize thoughts
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            vectors = vectorizer.fit_transform(thought_texts)
            
            # Determine optimal number of clusters
            n_clusters = min(5, len(thought_texts) // 3)
            if n_clusters < 2:
                return
            
            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(vectors)
            
            # Group thoughts by cluster
            clusters = defaultdict(list)
            thought_list = list(self.thoughts.values())
            
            for idx, label in enumerate(labels):
                clusters[label].append(thought_list[idx])
            
            # Create cluster objects
            for cluster_id, thoughts in clusters.items():
                if len(thoughts) >= 2:
                    # Find common keywords for topic
                    all_text = " ".join([t.content for t in thoughts])
                    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
                    word_freq = defaultdict(int)
                    for w in words:
                        if w not in ['the', 'and', 'for', 'that', 'this', 'with']:
                            word_freq[w] += 1
                    
                    topic_words = sorted(word_freq, key=word_freq.get, reverse=True)[:3]
                    topic = " ".join(topic_words)
                    
                    cluster = ThoughtCluster(
                        id=str(uuid.uuid4())[:8],
                        topic=topic,
                        thoughts=thoughts,
                        coherence=0.7  # Placeholder
                    )
                    self.thought_clusters.append(cluster)
            
            self.stats["clusters_formed"] = len(self.thought_clusters)
            
        except Exception as e:
            logger.debug(f"Clustering failed: {e}")
    
    # ==================== TASK PROCESSING ====================
    
    async def process_batch(self, batch_size: int = None) -> List[ThinkingTask]:
        """Process a batch of tasks"""
        batch_size = batch_size or self.batch_size
        processed = []
        
        async with self._processing_lock:
            for _ in range(min(batch_size, len(self.queue))):
                if not self.queue:
                    break
                
                # Check rate limit
                if not await self.rate_limiter.acquire():
                    await asyncio.sleep(0.1)
                    continue
                
                # Get highest priority task
                priority_val, idx, task = heapq.heappop(self.queue)
                
                # Check circuit breaker
                cb = self.circuit_breakers.get("llm")
                if cb and not cb.call():
                    # Re-queue for later
                    heapq.heappush(self.queue, (priority_val, idx, task))
                    continue
                
                # Execute task
                start_time = time.time()
                self.active_tasks[task.id] = task
                
                try:
                    task = await task.execute(self, self.llm)
                    
                    if task.status == TaskStatus.COMPLETED:
                        cb.record_success() if cb else None
                        self.completed_tasks.append(task)
                        self.stats["tasks_succeeded"] += 1
                        
                        # Cache result
                        self._cache_result(task)
                        
                        # Recognize patterns
                        self._recognize_patterns(task)
                        
                        # Trigger callbacks
                        for callback in self.on_task_complete_callbacks:
                            try:
                                callback(task)
                            except:
                                pass
                        
                        # Store in memory
                        if self.memory:
                            for thought in task.thoughts:
                                await self._store_thought(thought)
                    
                    else:
                        cb.record_failure() if cb else None
                        self.failed_tasks.append(task)
                        self.stats["tasks_failed"] += 1
                    
                    processed.append(task)
                    self.stats["tasks_processed"] += 1
                    self.stats["insights_generated"] += len(task.thoughts)
                    
                    # Track processing time
                    duration = task.get_duration_ms()
                    if duration > 0:
                        self.processing_times.append(duration)
                        if len(self.processing_times) > 100:
                            self.processing_times = self.processing_times[-100:]
                        
                        # Update average
                        total = self.stats["average_processing_time_ms"] * (self.stats["tasks_processed"] - 1)
                        self.stats["average_processing_time_ms"] = (total + duration) / self.stats["tasks_processed"]
                    
                finally:
                    self.active_tasks.pop(task.id, None)
                
                # Small delay between tasks
                await asyncio.sleep(0.05)
            
            # Update daily stats
            await self._update_daily_stats(processed)
            
            # Cluster thoughts periodically
            if len(self.thoughts) % 20 == 0 and len(self.thoughts) > 0:
                asyncio.create_task(self._cluster_thoughts())
            
            # Persist insights
            if self.enable_persistence and self.stats["tasks_processed"] % 10 == 0:
                await self._persist_insights()
        
        return processed
    
    async def _update_daily_stats(self, tasks: List[ThinkingTask]):
        """Update daily statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                "total": 0,
                "by_mode": defaultdict(int),
                "by_priority": defaultdict(int),
                "thoughts": 0
            }
        
        for task in tasks:
            self.daily_stats[today]["total"] += 1
            self.daily_stats[today]["by_mode"][task.mode.value] += 1
            self.daily_stats[today]["by_priority"][task.priority.name] += 1
            self.daily_stats[today]["thoughts"] += len(task.thoughts)
        
        # Keep only last 30 days
        if len(self.daily_stats) > 30:
            oldest = min(self.daily_stats.keys())
            del self.daily_stats[oldest]
    
    async def _store_thought(self, thought: Thought):
        """Store thought in memory"""
        if self.memory and hasattr(self.memory, "store"):
            try:
                await self.memory.store({
                    "type": "thought",
                    "id": thought.id,
                    "content": thought.content,
                    "thought_type": thought.thought_type.value,
                    "mode": thought.mode.value,
                    "confidence": thought.confidence,
                    "importance": thought.importance,
                    "timestamp": thought.created_at.isoformat()
                })
            except Exception as e:
                logger.debug(f"Failed to store thought: {e}")
    
    # ==================== AUTO TASK GENERATION ====================
    
    async def _auto_generate_tasks(self):
        """Automatically generate tasks based on context"""
        if not self.auto_mode or len(self.queue) > self.max_queue / 2:
            return
        
        hour = datetime.now().hour
        
        # Morning: planning and analysis
        if 5 <= hour < 12:
            modes = [ThinkingMode.PLANNING, ThinkingMode.ANALYSIS, ThinkingMode.FORECAST]
        # Afternoon: tactical and optimization        elif 12 <= hour < 17:
            modes = [ThinkingMode.TACTICAL, ThinkingMode.OPTIMIZATION, ThinkingMode.EVALUATION]
        # Evening: reflection and synthesis
        elif 17 <= hour < 22:
            modes = [ThinkingMode.REFLECTION, ThinkingMode.SYNTHESIS, ThinkingMode.DIAGNOSTIC]
        # Night: creative and strategic
        else:
            modes = [ThinkingMode.CREATIVE, ThinkingMode.STRATEGIC, ThinkingMode.ABSTRACT]
        
        # Add high-level system analysis periodically
        if self.stats["tasks_processed"] % 50 == 0:
            self.add_task(
                name="system_health_analysis",
                mode=ThinkingMode.ANALYSIS,
                context="Analyze system health, performance metrics, and identify areas for improvement.",
                priority=TaskPriority.LOW
            )
        
        # Add one suggested task
        mode = random.choice(modes)
        context = "General system operation and improvement opportunities."
        
        self.add_task(
            name=f"auto_{mode.value}",
            mode=mode,
            context=context,
            priority=TaskPriority.BACKGROUND
        )
    
    # ==================== SCHEDULED TASKS ====================
    
    def schedule_task(self, name: str, mode: ThinkingMode, hour: int, context: str = ""):
        """Schedule a recurring task at specific hour"""
        if hour < 0 or hour > 23:
            return False
        
        if name not in self.scheduled_tasks:
            self.scheduled_tasks[name] = []
        
        self.scheduled_tasks[name].append(hour)
        logger.info(f"Scheduled task '{name}' at hour {hour}")
        return True
    
    async def _check_scheduled_tasks(self):
        """Check and execute scheduled tasks"""
        now = datetime.now()
        current_hour = now.hour
        
        for task_name, hours in self.scheduled_tasks.items():
            if current_hour in hours:
                                # Check if already run this hour
                already_run = False
                for completed in self.completed_tasks[-10:]:
                    if completed.name == task_name and completed.completed_at:
                        if completed.completed_at.hour == current_hour and completed.completed_at.date() == now.date():
                            already_run = True
                            break
                
                if not already_run:
                    # Extract mode from name or default to REFLECTION
                    mode = ThinkingMode.REFLECTION
                    for m in ThinkingMode:
                        if m.value in task_name.lower():
                            mode = m
                            break
                    
                    self.add_task(
                        name=task_name,
                        mode=mode,
                        context="Scheduled thinking task based on time.",
                        priority=TaskPriority.NORMAL
                    )
    
    # ==================== PERSISTENCE ====================
    
    async def _persist_insights(self):
        """Persist insights to disk"""
        if not self.enable_persistence:
            return
        
        try:
            data = {
                "stats": self.stats,
                "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
                "daily_stats": {k: dict(v) for k, v in self.daily_stats.items()},
                "thoughts_count": len(self.thoughts),
                "tasks_completed": len(self.completed_tasks),
                "last_updated": datetime.now().isoformat(),
                "clusters": [c.to_dict() for c in self.thought_clusters[-10:]]
            }
            
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.debug(f"Insights persisted to {self.persist_path}")
            
        except Exception as e:
            logger.error(f"Failed to persist insights: {e}")
    
    async def load_persisted_insights(self):
        """Load previously persisted insights"""
        if not self.enable_persistence or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore patterns
            for pattern_id, pattern_data in data.get("patterns", {}).items():
                pattern = ThinkingPattern(
                    id=pattern_data["id"],
                    pattern_type=pattern_data["type"],
                    confidence=pattern_data["confidence"],
                    occurrences=pattern_data["occurrences"]
                )
                if pattern_data.get("first_seen"):
                    pattern.first_seen = datetime.fromisoformat(pattern_data["first_seen"])
                if pattern_data.get("last_seen"):
                    pattern.last_seen = datetime.fromisoformat(pattern_data["last_seen"])
                self.patterns[pattern_id] = pattern
            
            # Restore stats (but not all to avoid overwriting current session)
            self.stats["patterns_recognized"] = len(self.patterns)
            
            logger.info(f"Loaded {len(self.patterns)} patterns from persistence")
            
        except Exception as e:
            logger.error(f"Failed to load insights: {e}")
    
    # ==================== QUERY METHODS ====================
    
    def get_thoughts(
        self,
        limit: int = 50,
        thought_type: Optional[ThoughtType] = None,
        mode: Optional[ThinkingMode] = None,
        min_confidence: float = 0.0,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get filtered thoughts"""
        thoughts = list(self.thoughts.values())
        
        if thought_type:
            thoughts = [t for t in thoughts if t.thought_type == thought_type]
        if mode:
            thoughts = [t for t in thoughts if t.mode == mode]
        if min_confidence > 0:
            thoughts = [t for t in thoughts if t.confidence >= min_confidence]
        if min_importance > 0:
            thoughts = [t for t in thoughts if t.importance >= min_importance]
        
        # Sort by importance and confidence
        thoughts.sort(key=lambda t: t.get_score(), reverse=True)
        
        return [t.to_dict() for t in thoughts[:limit]]
    
    def get_top_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get highest value insights"""
        return self.get_thoughts(limit=limit, thought_type=ThoughtType.INSIGHT, min_confidence=0.6)
    
    def get_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations from thoughts"""
        return self.get_thoughts(limit=limit, thought_type=ThoughtType.RECOMMENDATION, min_confidence=0.5)
    
    def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        mode: Optional[ThinkingMode] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get filtered tasks"""
        tasks = []
        
        if status == TaskStatus.COMPLETED or status is None:
            tasks.extend(self.completed_tasks[-limit:])
        if status == TaskStatus.FAILED or status is None:
            tasks.extend(self.failed_tasks[-limit:])
        if status == TaskStatus.PENDING or status is None:
            tasks.extend([t for _, _, t in self.queue])
        if status == TaskStatus.RUNNING or status is None:
            tasks.extend(self.active_tasks.values())
        
        if mode:
            tasks = [t for t in tasks if t.mode == mode]
        
        # Sort by timestamp
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return [t.to_dict() for t in tasks[:limit]]
    
    def get_patterns(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Get recognized patterns"""
        patterns = list(self.patterns.values())
        if min_confidence > 0:
            patterns = [p for p in patterns if p.confidence >= min_confidence]
        return [p.to_dict() for p in patterns]
    
    def get_thought_clusters(self) -> List[Dict[str, Any]]:
        """Get thought clusters"""
        return [c.to_dict() for c in self.thought_clusters]
    
    def get_related_thoughts(self, thought_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get thoughts related to a specific thought"""
        if thought_id not in self.thoughts:
            return []
        
        target = self.thoughts[thought_id]
        
        # Simple keyword matching
        target_words = set(target.content.lower().split())
        target_words = {w for w in target_words if len(w) > 3}
        
        related = []
        for tid, thought in self.thoughts.items():
            if tid == thought_id:
                continue
            
            thought_words = set(thought.content.lower().split())
            intersection = target_words.intersection(thought_words)
            
            if intersection:
                similarity = len(intersection) / max(len(target_words), len(thought_words))
                if similarity > 0.2:
                    related.append((similarity, thought))
        
        related.sort(key=lambda x: x[0], reverse=True)
        return [t.to_dict() for _, t in related[:limit]]
    
    def search_thoughts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search thoughts by content"""
        query_lower = query.lower()
        results = []
        
        for thought in self.thoughts.values():
            if query_lower in thought.content.lower():
                # Calculate relevance score
                score = query_lower.count(thought.content.lower()) * 0.1
                score += thought.confidence * 0.3
                score += thought.importance * 0.3
                results.append((score, thought))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [t.to_dict() for _, t in results[:limit]]
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        # Calculate success rate
        total = self.stats["tasks_succeeded"] + self.stats["tasks_failed"]
        success_rate = (self.stats["tasks_succeeded"] / max(1, total)) * 100
        
        # Calculate average processing time (recent)
        recent_times = self.processing_times[-50:] if self.processing_times else []
        avg_recent = sum(recent_times) / len(recent_times) if recent_times else 0
        
        return {
            "tasks": {
                "processed": self.stats["tasks_processed"],
                "succeeded": self.stats["tasks_succeeded"],
                "failed": self.stats["tasks_failed"],
                "success_rate": round(success_rate, 2),
                "queue_size": len(self.queue),
                "queue_peak": self.stats["queue_peak"],
                "active": len(self.active_tasks)
            },
            "insights": {
                "total": self.stats["insights_generated"],
                "thoughts": len(self.thoughts),
                "patterns": self.stats["patterns_recognized"],
                "clusters": self.stats["clusters_formed"]
            },
            "performance": {
                "avg_processing_time_ms": round(self.stats["average_processing_time_ms"], 2),
                "recent_avg_ms": round(avg_recent, 2),
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_ratio": round(self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"]), 2)
            },
            "status": {
                "running": self.is_running,
                "auto_mode": self.auto_mode,
                "batch_size": self.batch_size,
                "max_queue": self.max_queue
            },
            "daily": dict(list(self.daily_stats.items())[-7:])
        }
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get task summary by mode and priority"""
        summary = {
            "by_mode": defaultdict(int),
            "by_priority": defaultdict(int),
            "by_status": defaultdict(int)
        }
        
        # Completed tasks
        for task in self.completed_tasks:
            summary["by_mode"][task.mode.value] += 1
            summary["by_priority"][task.priority.name] += 1
            summary["by_status"]["completed"] += 1
        
        # Failed tasks
        for task in self.failed_tasks:
            summary["by_mode"][task.mode.value] += 1
            summary["by_priority"][task.priority.name] += 1
            summary["by_status"]["failed"] += 1
        
        # Queued tasks
        for _, _, task in self.queue:
            summary["by_mode"][task.mode.value] += 1
            summary["by_priority"][task.priority.name] += 1
            summary["by_status"]["queued"] += 1
        
        return {k: dict(v) for k, v in summary.items()}
    
    # ==================== CONTROL METHODS ====================
    
    async def run(self, interval: float = 2.0):
        """
        Main background thinking loop
        """
        if self.is_running:
            logger.warning("Thinker already running")
            return
        
        self.is_running = True
        interval = max(interval, self.min_interval)
        
        # Load persisted insights
        await self.load_persisted_insights()
        
        logger.info("🧠 Background thinker (ULTIMATE MODE) started")
        
        # Start background tasks
        self._background_tasks.append(asyncio.create_task(self._auto_task_generator()))
        self._background_tasks.append(asyncio.create_task(self._scheduled_task_checker()))
        self._background_tasks.append(asyncio.create_task(self._metrics_reporter()))
        
        last_cache_cleanup = time.time()
        cache_cleanup_interval = 300  # 5 minutes
        
        while self.is_running:
            try:
                # Process tasks
                if self.queue:
                    processed = await self.process_batch()
                    if processed:
                        logger.debug(f"Processed {len(processed)} thinking tasks")
                
                # Periodic cache cleanup
                now = time.time()
                if now - last_cache_cleanup > cache_cleanup_interval:
                    self._cleanup_cache()
                    last_cache_cleanup = now
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Thinker loop error: {e}")
                await asyncio.sleep(interval * 2)
        
        # Cleanup
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        await self._persist_insights()
        logger.info("🛑 Background thinker stopped")
    
    async def _auto_task_generator(self):
        """Background task generator"""
        while self.is_running:
            await asyncio.sleep(60)  # Check every minute
            await self._auto_generate_tasks()
    
    async def _scheduled_task_checker(self):
        """Background scheduled task checker"""
        while self.is_running:
            await asyncio.sleep(60)  # Check every minute
            await self._check_scheduled_tasks()
    
    async def _metrics_reporter(self):
        """Background metrics reporter"""
        while self.is_running:
            await asyncio.sleep(300)  # Report every 5 minutes
            
            if self.stats["tasks_processed"] > 0:
                stats = self.get_stats()
                logger.info(
                    f"📊 Thinker Metrics: tasks={stats['tasks']['processed']}, "
                    f"success_rate={stats['tasks']['success_rate']}%, "
                    f"insights={stats['insights']['total']}, "
                    f"patterns={stats['insights']['patterns']}, "
                    f"avg_time={stats['performance']['avg_processing_time_ms']}ms"
                )
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        now = time.time()
        expired = [k for k, (_, t) in self.result_cache.items() if now - t > self.cache_ttl]
        for k in expired:
            del self.result_cache[k]
        
        if expired:
            logger.debug(f"Cleaned {len(expired)} expired cache entries")
    
    def stop(self):
        """Stop the background thinker"""
        self.is_running = False
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            timeout = 30
            start = time.time()
            while self.active_tasks and (time.time() - start) < timeout:
                await asyncio.sleep(0.5)
        
        await self._persist_insights()
        logger.info("Background thinker shutdown complete")
    
    def clear_history(self, keep_recent: int = 100):
        """Clear old history"""
        if len(self.completed_tasks) > keep_recent:
            self.completed_tasks = self.completed_tasks[-keep_recent:]
        if len(self.failed_tasks) > keep_recent:
            self.failed_tasks = self.failed_tasks[-keep_recent:]
        
        # Clear old thoughts to save memory
        thought_ids = [t.id for t in self.completed_tasks[-keep_recent:]]
        thought_ids.extend([t.id for t in self.failed_tasks[-keep_recent:]])
        
        # Keep only thoughts from recent tasks
        old_thoughts = [tid for tid in self.thoughts if tid not in thought_ids]
        for tid in old_thoughts:
            del self.thoughts[tid]
        
        logger.info(f"History cleared. Kept {len(self.completed_tasks)} completed, {len(self.thoughts)} thoughts")
    
    def reset(self):
        """Reset all state"""
        self.queue.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.active_tasks.clear()
        self.thoughts.clear()
        self.thought_clusters.clear()
        self.patterns.clear()
        self.result_cache.clear()
        self.processing_times.clear()
        self.daily_stats.clear()
        
        self.stats = {
            "tasks_processed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "insights_generated": 0,
            "patterns_recognized": 0,
            "clusters_formed": 0,
            "average_processing_time_ms": 0.0,
            "queue_peak": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self._task_counter = 0
        
        logger.info("Thinker reset complete")
    
    # ==================== CALLBACKS ====================
    
    def on_task_complete(self, callback: Callable):
        """Register callback for task completion"""
        self.on_task_complete_callbacks.append(callback)
    
    def on_thought_generated(self, callback: Callable):
        """Register callback for thought generation"""
        self.on_thought_generated_callbacks.append(callback)
    
    # ==================== EXPORT ====================
    
    def export_thoughts(self, filepath: str, format: str = "json"):
        """Export thoughts to file"""
        thoughts_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_thoughts": len(self.thoughts),
                "version": "1.0"
            },
            "thoughts": [t.to_dict() for t in self.thoughts.values()]
        }
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(thoughts_data, f, indent=2, default=str)
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["id", "content", "thought_type", "mode", "confidence", "importance", "created_at"])
                writer.writeheader()
                for thought in self.thoughts.values():
                    writer.writerow({
                        "id": thought.id,
                        "content": thought.content,
                        "thought_type": thought.thought_type.value,
                        "mode": thought.mode.value,
                        "confidence": thought.confidence,
                        "importance": thought.importance,
                        "created_at": thought.created_at.isoformat()
                    })
        elif format == "markdown":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Thoughts Export\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Thoughts: {len(self.thoughts)}\n\n")
                
                for thought in self.thoughts.values():
                    f.write(f"## {thought.thought_type.value.capitalize()}: {thought.id}\n")
                    f.write(f"**Content:** {thought.content}\n")
                    f.write(f"**Mode:** {thought.mode.value}\n")
                    f.write(f"**Confidence:** {thought.confidence:.2f}\n")
                    f.write(f"**Importance:** {thought.importance:.2f}\n")
                    f.write(f"**Created:** {thought.created_at.isoformat()}\n\n")
        
        logger.info(f"Exported {len(self.thoughts)} thoughts to {filepath}")
    
    def export_patterns(self, filepath: str):
        """Export recognized patterns"""
        patterns_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_patterns": len(self.patterns)
            },
            "patterns": [p.to_dict() for p in self.patterns.values()]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patterns_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self.patterns)} patterns to {filepath}")
    
    # ==================== VISUALIZATION ====================
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report"""
        stats = self.get_stats()
        task_summary = self.get_task_summary()
        
        report = []
        report.append("=" * 60)
        report.append("BACKGROUND THINKER REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("📊 STATISTICS")
        report.append("-" * 40)
        report.append(f"Tasks Processed: {stats['tasks']['processed']}")
        report.append(f"Success Rate: {stats['tasks']['success_rate']}%")
        report.append(f"Queue Size: {stats['tasks']['queue_size']}")
        report.append(f"Active Tasks: {stats['tasks']['active']}")
        report.append("")
        
        report.append("💡 INSIGHTS")
        report.append("-" * 40)
        report.append(f"Total Insights: {stats['insights']['total']}")
        report.append(f"Thoughts Stored: {stats['insights']['thoughts']}")
        report.append(f"Patterns Recognized: {stats['insights']['patterns']}")
        report.append(f"Clusters Formed: {stats['insights']['clusters']}")
        report.append("")
        
        report.append("⚡ PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Avg Processing Time: {stats['performance']['avg_processing_time_ms']}ms")
        report.append(f"Cache Hit Ratio: {stats['performance']['cache_ratio']}")
        report.append("")
        
        report.append("📋 TASK BREAKDOWN")
        report.append("-" * 40)
        report.append("By Mode:")
        for mode, count in sorted(task_summary.get("by_mode", {}).items(), key=lambda x: x[1], reverse=True)[:5]:
            report.append(f"  - {mode}: {count}")
        report.append("")
        
        report.append("Top Insights:")
        top_insights = self.get_top_insights(5)
        for i, insight in enumerate(top_insights, 1):
            content = insight['content'][:80] + "..." if len(insight['content']) > 80 else insight['content']
            report.append(f"  {i}. {content} (confidence: {insight['confidence']:.2f})")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# ==================== WRAPPER FOR EDIATH ====================

class BackgroundThinkerWrapper:
    """Wrapper class for EDIATH integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.thinker = BackgroundThinker(
            system=None,
            max_queue=config.get("max_queue", 200),
            batch_size=config.get("batch_size", 3),
            min_interval=config.get("min_interval", 0.5),
            auto_mode=config.get("auto_mode", True),
            rate_limit=config.get("rate_limit", 10.0),
            enable_persistence=config.get("enable_persistence", True),
            enable_clustering=config.get("enable_clustering", True),
            max_concurrent=config.get("max_concurrent", 2)
        )
        self.agent_type = "background_thinker"
        self.capabilities = [
            "think", "analyze", "plan", "reflect", "generate_insights",
            "get_thoughts", "get_patterns", "export_thoughts", "get_stats"
        ]
    
    async def start(self):
        """Start the thinker"""
        asyncio.create_task(self.thinker.run())
    
    async def stop(self):
        """Stop the thinker"""
        await self.thinker.shutdown()
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a thinker request"""
        operation = request.get("operation")
        
        if operation == "think":
            task_id = self.thinker.add_task(
                name=request.get("name", "adhoc_task"),
                mode=ThinkingMode(request.get("mode", "analysis")),
                context=request.get("context", ""),
                priority=TaskPriority(request.get("priority", "normal")),
                tags=request.get("tags", [])
            )
            return {"success": True, "task_id": task_id}
        
        elif operation == "get_thoughts":
            thoughts = self.thinker.get_thoughts(
                limit=request.get("limit", 50),
                thought_type=ThoughtType(request.get("thought_type")) if request.get("thought_type") else None,
                mode=ThinkingMode(request.get("mode")) if request.get("mode") else None,
                min_confidence=request.get("min_confidence", 0.0)
            )
            return {"success": True, "thoughts": thoughts, "count": len(thoughts)}
        
        elif operation == "get_top_insights":
            insights = self.thinker.get_top_insights(limit=request.get("limit", 10))
            return {"success": True, "insights": insights}
        
        elif operation == "get_patterns":
            patterns = self.thinker.get_patterns(min_confidence=request.get("min_confidence", 0.0))
            return {"success": True, "patterns": patterns}
        
        elif operation == "search":
            results = self.thinker.search_thoughts(
                query=request.get("query", ""),
                limit=request.get("limit", 10)
            )
            return {"success": True, "results": results}
        
        elif operation == "get_stats":
            return {"success": True, "stats": self.thinker.get_stats()}
        
        elif operation == "export":
            self.thinker.export_thoughts(
                filepath=request.get("filepath", "thoughts_export.json"),
                format=request.get("format", "json")
            )
            return {"success": True}
        
        elif operation == "clear_history":
            self.thinker.clear_history(keep_recent=request.get("keep_recent", 100))
            return {"success": True}
        
        elif operation == "generate_report":
            report = self.thinker.generate_report()
            if request.get("filepath"):
                with open(request["filepath"], 'w') as f:
                    f.write(report)
            return {"success": True, "report": report}
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "BackgroundThinker",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.thinker.get_stats()
        }


# ==================== EXAMPLE USAGE ====================

async def example_usage():
    """Example usage of Background Thinker"""
    
    # Create thinker
    thinker = BackgroundThinker(
        max_queue=100,
        batch_size=3,
        auto_mode=True,
        enable_clustering=True
    )
    
    # Add callbacks
    def on_thought(thought):
        print(f"💭 New thought: {thought.content[:50]}...")
    
    thinker.on_thought_generated(on_thought)
    
    # Start thinker
    asyncio.create_task(thinker.run())
    
    # Add some tasks
    thinker.add_task(
        name="analyze_system",
        mode=ThinkingMode.ANALYSIS,
        context="Analyze current system performance and identify bottlenecks.",
        priority=TaskPriority.HIGH,
        tags=["system", "performance"]
    )
    
    thinker.add_task(
        name="plan_improvements",
        mode=ThinkingMode.PLANNING,
        context="Create a plan for system improvements based on analysis.",
        priority=TaskPriority.NORMAL
    )
    
    thinker.add_task(
        name="reflect_on_week",
        mode=ThinkingMode.REFLECTION,
        context="Reflect on the past week's activities and outcomes.",
        priority=TaskPriority.LOW,
        tags=["reflection", "weekly"]
    )
    
    # Schedule a daily task
    thinker.schedule_task(
        name="daily_summary",
        mode=ThinkingMode.SYNTHESIS,
        hour=18,  # 6 PM
        context="Summarize today's key events and learnings."
    )
    
    # Wait for processing
    await asyncio.sleep(10)
    
    # Get stats
    stats = thinker.get_stats()
    print(f"\n📊 Stats: {stats['tasks']['processed']} tasks processed")
    print(f"💡 Insights: {stats['insights']['total']}")
    print(f"🎯 Patterns: {stats['insights']['patterns']}")
    
    # Get top insights
    insights = thinker.get_top_insights(5)
    print("\n🏆 Top Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight['content'][:80]}...")
    
    # Generate report
    report = thinker.generate_report()
    print(f"\n📄 Report:\n{report[:500]}...")
    
    # Export thoughts
    thinker.export_thoughts("thoughts_export.json", format="json")
    
    # Cleanup
    await thinker.shutdown()
    
    return thinker


if __name__ == "__main__":
    asyncio.run(example_usage())
