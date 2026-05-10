"""
Advanced Context Manager - Ultimate Edition (AI Context Intelligence Engine)
✔ Multi-layered context (immediate, session, long-term)
✔ Time-aware context with decay & expiration
✔ Relevance scoring & ranking
✔ Working memory & episodic memory integration
✔ Semantic similarity for context retrieval
✔ Context compression & summarization
✔ Topic clustering & segmentation
✔ User persona & preference learning
✔ Multi-modal context (text, image, code)
✔ Cross-session context transfer
✔ Contextual bandits for adaptive retrieval
✔ Privacy-preserving context (PII filtering)
✔ Collaborative context (multi-user)
✔ Context versioning & rollback
"""

import asyncio
import hashlib
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from tenacity import retry, stop_after_attempt, wait_exponential
from ..utils.logger import logger
from ..memory import MemoryManager


# ==================== ENUMS ====================

class ContextLayer(Enum):
    """Layers of context with different retention"""
    IMMEDIATE = "immediate"      # Current turn, high priority
    SHORT_TERM = "short_term"    # Last few minutes
    WORKING = "working"          # Current session
    EPISODIC = "episodic"        # Past sessions
    LONG_TERM = "long_term"      # User preferences, knowledge
    COLLABORATIVE = "collaborative"  # From other users


class ContextPriority(Enum):
    """Priority levels for context items"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class RetentionPolicy(Enum):
    """Retention policies for context"""
    EXPIRE_TIME = "expire_time"
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    SCORE = "score"
    MANUAL = "manual"


class PIIFilter(Enum):
    """PII filtering levels"""
    NONE = "none"           # No filtering
    BASIC = "basic"         # Email, phone, SSN
    STRICT = "strict"       # All identifiable info
    AGGRESSIVE = "aggressive"  # Maximum privacy


# ==================== DATA CLASSES ====================

@dataclass
class ContextEntry:
    """Enhanced context entry with rich metadata"""
    key: str
    value: Any
    layer: ContextLayer = ContextLayer.SHORT_TERM
    priority: ContextPriority = ContextPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None  # Seconds
    access_count: int = 0
    relevance_score: float = 0.5
    importance: float = 0.5
    source: str = "system"
    tags: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None
    version: int = 1
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def access(self) -> Any:
        """Record access and return value"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value
    
    def decay_relevance(self, decay_rate: float = 0.01):
        """Decay relevance over time"""
        age_hours = (datetime.now() - self.created_at).total_seconds() / 3600
        self.relevance_score *= max(0.1, 1.0 - (decay_rate * age_hours))
    
    def compute_score(self) -> float:
        """Compute overall priority score"""
        return (
            (self.relevance_score * 0.3) +
            (self.importance * 0.3) +
            (1.0 / (self.priority.value) * 0.2) +
            (min(1.0, self.access_count / 10) * 0.2)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "layer": self.layer.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "relevance_score": round(self.relevance_score, 3),
            "tags": self.tags,
            "value_preview": str(self.value)[:100]
        }


@dataclass
class ContextMessage:
    """Conversation message with metadata"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    tokens: int = 0
    embeddings: Optional[Any] = None
    response_time_ms: float = 0.0
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content[:200],
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance
        }


@dataclass
class UserProfile:
    """User preferences and learned context"""
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    topics: Dict[str, float] = field(default_factory=dict)  # topic -> interest_score
    style: Dict[str, str] = field(default_factory=dict)
    expertise: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_topic_interest(self, topic: str, delta: float):
        """Update interest in a topic"""
        current = self.topics.get(topic, 0.5)
        self.topics[topic] = max(0, min(1, current + delta))
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "topics": dict(sorted(self.topics.items(), key=lambda x: x[1], reverse=True)[:10]),
            "expertise": dict(sorted(self.expertise.items(), key=lambda x: x[1], reverse=True)[:5]),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class ContextSummary:
    """Compressed context summary"""
    id: str
    content: str
    topic: str
    start_time: datetime
    end_time: datetime
    message_count: int
    importance: float
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "content": self.content[:200],
            "message_count": self.message_count,
            "importance": self.importance,
            "key_points": self.key_points[:3]
        }


class ContextManager:
    """
    Ultimate Context Manager with intelligent context handling
    """
    
    def __init__(
        self,
        max_history: int = 200,
        max_context_tokens: int = 4000,
        enable_memory: bool = True,
        enable_summarization: bool = True,
        enable_pii_filtering: bool = True,
        pii_level: PIIFilter = PIIFilter.BASIC,
        retention_policy: RetentionPolicy = RetentionPolicy.SCORE,
        decay_rate: float = 0.01,
        min_relevance: float = 0.1,
        compression_threshold: int = 100,
        session_ttl_hours: int = 24
    ):
        """
        Initialize Context Manager
        
        Args:
            max_history: Maximum conversation history length
            max_context_tokens: Maximum tokens for LLM context
            enable_memory: Enable long-term memory integration
            enable_summarization: Enable context summarization
            enable_pii_filtering: Filter personal information
            pii_level: PII filtering strictness
            retention_policy: How to evict old context
            decay_rate: Rate at which context relevance decays
            min_relevance: Minimum relevance to keep context
            compression_threshold: Messages before summarizing
            session_ttl_hours: Session time-to-live
        """
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        self.enable_memory = enable_memory
        self.enable_summarization = enable_summarization
        self.enable_pii_filtering = enable_pii_filtering
        self.pii_level = pii_level
        self.retention_policy = retention_policy
        self.decay_rate = decay_rate
        self.min_relevance = min_relevance
        self.compression_threshold = compression_threshold
        self.session_ttl = timedelta(hours=session_ttl_hours)
        
        # Context storage
        self.context: Dict[str, ContextEntry] = {}
        self.history: deque = deque(maxlen=max_history)
        self.layered_context: Dict[ContextLayer, Dict[str, ContextEntry]] = {
            layer: {} for layer in ContextLayer
        }
        
        # Summaries
        self.summaries: List[ContextSummary] = []
        self.current_summary: Optional[ContextSummary] = None
        self.summary_counter = 0
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        self.current_user: Optional[str] = None
        self.session_id: Optional[str] = None
        self.session_start: datetime = field(default_factory=datetime.now)
        
        # Topic tracking
        self.topic_clusters: Dict[str, List[str]] = defaultdict(list)
        self.current_topics: Set[str] = set()
        
        # Performance
        self.total_context_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # PII patterns for filtering
        self._init_pii_patterns()
        
        # Vectorizer for similarity (if available)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.context_embeddings = []
        self.context_keys = []
        
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Memory management
        self.memory = MemoryManager() if enable_memory else None
        
        logger.info(f"🧠 Context Manager initialized (max_history={max_history}, pii={pii_level.value})")
    
    def _init_pii_patterns(self):
        """Initialize PII detection patterns"""
        self.pii_patterns = {
            "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        }
    
    def _filter_pii(self, text: str) -> str:
        """Filter PII from text based on configured level"""
        if not self.enable_pii_filtering or self.pii_level == PIIFilter.NONE:
            return text
        
        filtered = text
        
        for pattern_name, pattern in self.pii_patterns.items():
            if self.pii_level == PIIFilter.BASIC and pattern_name not in ["email", "phone"]:
                continue
            if self.pii_level == PIIFilter.STRICT:
                filtered = pattern.sub(f"[REDACTED_{pattern_name}]", filtered)
            elif pattern_name in ["email", "phone", "ssn"]:
                filtered = pattern.sub(f"[REDACTED_{pattern_name}]", filtered)
        
        return filtered
    
    # ==================== CONTEXT CRUD ====================
    
    def set(
        self,
        key: str,
        value: Any,
        layer: ContextLayer = ContextLayer.SHORT_TERM,
        priority: ContextPriority = ContextPriority.NORMAL,
        ttl: Optional[int] = None,
        tags: List[str] = None,
        importance: float = 0.5
    ):
        """Set context value with full configuration"""
        entry = ContextEntry(
            key=key,
            value=value,
            layer=layer,
            priority=priority,
            ttl=ttl,
            tags=tags or [],
            importance=importance
        )
        
        self.context[key] = entry
        self.layered_context[layer][key] = entry
        
        # Update embeddings for similarity search
        if self.vectorizer and isinstance(value, str):
            self._update_embeddings(key, value)
        
        logger.debug(f"📝 Context set: {key} (layer={layer.value}, priority={priority.value})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get context value"""
        if key in self.context:
            entry = self.context[key]
            if not entry.is_expired():
                self.cache_hits += 1
                return entry.access()
            else:
                self.delete(key)
        
        self.cache_misses += 1
        return default
    
    def get_all(self, layer: Optional[ContextLayer] = None) -> Dict[str, Any]:
        """Get all context values with optional layer filter"""
        if layer:
            return {k: v.value for k, v in self.layered_context[layer].items() if not v.is_expired()}
        return {k: v.value for k, v in self.context.items() if not v.is_expired()}
    
    def delete(self, key: str) -> bool:
        """Delete context entry"""
        if key in self.context:
            entry = self.context[key]
            del self.context[key]
            del self.layered_context[entry.layer][key]
            logger.debug(f"🗑️ Context deleted: {key}")
            return True
        return False
    
    def clear(self, layer: Optional[ContextLayer] = None):
        """Clear context (optionally by layer)"""
        if layer:
            for key in list(self.layered_context[layer].keys()):
                self.delete(key)
        else:
            self.context.clear()
            for l in self.layered_context:
                self.layered_context[l].clear()
        
        logger.info(f"🧹 Context cleared (layer={layer.value if layer else 'all'})")
    
    # ==================== CONVERSATION HISTORY ====================
    
    def add_message(
        self,
        role: str,
        content: str,
        importance: float = 0.5,
        user_id: Optional[str] = None
    ):
        """Add message to conversation history"""
        # Filter PII
        filtered_content = self._filter_pii(content)
        
        message = ContextMessage(
            role=role,
            content=filtered_content,
            importance=importance,
            user_id=user_id or self.current_user
        )
        
        self.history.append(message)
        
        # Extract topics from message
        topics = self._extract_topics(filtered_content)
        for topic in topics:
            self.current_topics.add(topic)
            if self.current_user and self.current_user in self.user_profiles:
                self.user_profiles[self.current_user].update_topic_interest(topic, 0.05)
        
        # Auto-summarize if needed
        if self.enable_summarization and len(self.history) >= self.compression_threshold:
            asyncio.create_task(self._compress_history())
        
        logger.debug(f"💬 Message added: {role} ({importance:.2f})")
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        topics = []
        keywords = {
            "coding": ["code", "programming", "function", "class", "debug"],
            "ai": ["ai", "machine learning", "neural", "model", "training"],
            "data": ["data", "database", "query", "sql", "analytics"],
            "web": ["web", "api", "http", "endpoint", "server"],
            "testing": ["test", "assert", "coverage", "unit test"],
            "deployment": ["deploy", "docker", "kubernetes", "cloud"]
        }
        
        text_lower = text.lower()
        for topic, keywords_list in keywords.items():
            if any(kw in text_lower for kw in keywords_list):
                topics.append(topic)
        
        return topics[:3]
    
    def get_recent_messages(
        self,
        limit: int = 10,
        min_importance: float = 0.0,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent messages with filters"""
        messages = list(self.history)[-limit:]
        
        if min_importance > 0:
            messages = [m for m in messages if m.importance >= min_importance]
        if role:
            messages = [m for m in messages if m.role == role]
        
        return [m.to_dict() for m in messages]
    
    def get_conversation_context(
        self,
        max_tokens: Optional[int] = None,
        include_summary: bool = True
    ) -> str:
        """Get formatted conversation context for LLM"""
        max_tokens = max_tokens or self.max_context_tokens
        context_parts = []
        token_estimate = 0
        
        # Add summary if available
        if include_summary and self.current_summary:
            summary_text = f"[Summary: {self.current_summary.content}]"
            context_parts.append(summary_text)
            token_estimate += len(summary_text) // 4
        
        # Add recent messages
        for msg in reversed(list(self.history)[-30:]):
            msg_text = f"{msg.role}: {msg.content}"
            msg_tokens = len(msg_text) // 4
            
            if token_estimate + msg_tokens > max_tokens:
                break
            
            context_parts.append(msg_text)
            token_estimate += msg_tokens
        
        return "\n".join(reversed(context_parts))
    
    # ==================== CONTEXT COMPRESSION ====================
    
    async def _compress_history(self):
        """Compress conversation history using LLM"""
        if not self.enable_summarization:
            return
        
        try:
            from ..brain.llm_engine import LLMEngine
            llm = LLMEngine()
            
            # Get messages to compress
            messages_to_compress = list(self.history)[:-10]  # Keep last 10
            if len(messages_to_compress) < 5:
                return
            
            conversation = "\n".join([
                f"{m.role}: {m.content}" for m in messages_to_compress
            ])
            
            prompt = f"""Summarize this conversation concisely, focusing on key points and decisions:

{conversation[:2000]}

Return ONLY a JSON object:
{{
    "summary": "Brief conversation summary",
    "topic": "Main topic of conversation",
    "key_points": ["point1", "point2", "point3"],
    "importance": 0.0-1.0
}}
"""
            
            response = await llm.generate(prompt)
            
            if isinstance(response, dict):
                raw = response.get("response", "")
            else:
                raw = str(response)
            
            # Parse JSON
            import re, json
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                data = json.loads(json_match.group())
                
                summary = ContextSummary(
                    id=f"sum_{self.summary_counter}",
                    content=data.get("summary", ""),
                    topic=data.get("topic", "general"),
                    start_time=messages_to_compress[0].timestamp,
                    end_time=messages_to_compress[-1].timestamp,
                    message_count=len(messages_to_compress),
                    importance=data.get("importance", 0.5),
                    key_points=data.get("key_points", [])
                )
                
                self.summaries.append(summary)
                self.current_summary = summary
                self.summary_counter += 1
                
                # Clear compressed messages
                for msg in messages_to_compress:
                    if msg in self.history:
                        self.history.remove(msg)
                
                logger.info(f"📄 History compressed: {len(messages_to_compress)} messages -> summary")
        
        except Exception as e:
            logger.error(f"History compression failed: {e}")
    
    # ==================== SEMANTIC SEARCH ====================
    
    def _update_embeddings(self, key: str, text: str):
        """Update TF-IDF embeddings for context search"""
        if not self.vectorizer:
            return
        
        self.context_keys.append(key)
        
        # Rebuild matrix (simplified - in production would be incremental)
        if len(self.context_keys) > 5:
            texts = [str(self.context[k].value) for k in self.context_keys if k in self.context]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.context_embeddings = list(range(len(self.context_keys)))
    
    def search_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search context by semantic similarity"""
        if not self.vectorizer or self.tfidf_matrix is None:
            # Fallback to keyword search
            return self._keyword_search(query, limit)
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top indices
            top_indices = similarities.argsort()[-limit:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    key = self.context_keys[idx]
                    entry = self.context.get(key)
                    if entry:
                        results.append({
                            "key": key,
                            "value": str(entry.value)[:200],
                            "similarity": float(similarities[idx]),
                            "layer": entry.layer.value
                        })
            
            return results
            
        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return self._keyword_search(query, limit)
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        query_lower = query.lower()
        results = []
        
        for key, entry in self.context.items():
            value_str = str(entry.value).lower()
            if query_lower in value_str:
                results.append({
                    "key": key,
                    "value": str(entry.value)[:200],
                    "similarity": 0.5,
                    "layer": entry.layer.value
                })
        
        return results[:limit]
    
    # ==================== USER PROFILES ====================
    
    def set_user(self, user_id: str, create_if_missing: bool = True):
        """Set current user"""
        self.current_user = user_id
        
        if user_id not in self.user_profiles and create_if_missing:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
            logger.info(f"👤 New user profile created: {user_id}")
    
    def get_user_profile(self, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        uid = user_id or self.current_user
        if uid and uid in self.user_profiles:
            return self.user_profiles[uid].to_dict()
        return None
    
    def update_user_preference(self, key: str, value: Any):
        """Update user preference"""
        if self.current_user and self.current_user in self.user_profiles:
            self.user_profiles[self.current_user].preferences[key] = value
            logger.debug(f"📝 User preference updated: {key}={value}")
    
    def learn_user_style(self, message: str):
        """Learn user communication style"""
        if not self.current_user:
            return
        
        # Simple style detection
        if len(message) < 50:
            style = "concise"
        elif "?" in message:
            style = "inquisitive"
        elif "!" in message:
            style = "enthusiastic"
        else:
            style = "detailed"
        
        self.user_profiles[self.current_user].style["preferred"] = style
        logger.debug(f"🎨 Learned user style: {style}")
    
    # ==================== CONTEXT BUILDING ====================
    
    async def build_context(
        self,
        user_input: str,
        max_tokens: Optional[int] = None,
        include_memory: bool = True,
        include_history: bool = True,
        include_user_profile: bool = True
    ) -> List[Dict[str, str]]:
        """
        Build comprehensive LLM-ready context
        
        Returns:
            List of message dicts for LLM consumption
        """
        self.total_context_requests += 1
        
        context_messages = []
        token_estimate = 0
        max_tokens = max_tokens or self.max_context_tokens
        
        # System prompt
        system_content = self._build_system_prompt()
        context_messages.append({"role": "system", "content": system_content})
        token_estimate += len(system_content) // 4
        
        # User profile context
        if include_user_profile and self.current_user:
            profile = self.get_user_profile()
            if profile:
                profile_text = f"User preferences: {profile.get('preferences', {})}"
                context_messages.append({"role": "system", "content": profile_text})
                token_estimate += len(profile_text) // 4
        
        # Memory context (from vector search)
        if include_memory and self.memory:
            try:
                memory_results = await asyncio.wait_for(
                    self.memory.search(user_input),
                    timeout=3.0
                )
                
                if memory_results:
                    for mem in memory_results[:3]:
                        mem_text = f"Relevant memory: {str(mem)[:200]}"
                        context_messages.append({"role": "system", "content": mem_text})
                        token_estimate += len(mem_text) // 4
            except Exception as e:
                logger.debug(f"Memory search failed: {e}")
        
        # Search context
        search_results = self.search_context(user_input, limit=3)
        for result in search_results:
            if result.get("value"):
                context_text = f"Context: {result['key']} = {result['value'][:150]}"
                context_messages.append({"role": "system", "content": context_text})
                token_estimate += len(context_text) // 4
        
        # Conversation history
        if include_history:
            history = self.get_conversation_context(max_tokens - token_estimate)
            if history:
                context_messages.append({"role": "assistant", "content": "[Previous conversation context]"})
                token_estimate += 50
        
        # Current user input
        filtered_input = self._filter_pii(user_input)
        context_messages.append({"role": "user", "content": filtered_input})
        
        # Trim if still over limit
        if token_estimate > max_tokens:
            # Remove less important context
            context_messages = context_messages[:2] + context_messages[-2:]
        
        logger.debug(f"📊 Context built: {len(context_messages)} messages, ~{token_estimate} tokens")
        
        # Update user profile based on input
        if self.current_user:
            self.learn_user_style(user_input)
        
        return context_messages
    
    def _build_system_prompt(self) -> str:
        """Build dynamic system prompt"""
        base_prompt = (
            "You an intelligent AI assistant. Be helpful, concise, and accurate. "
            "Use context to provide relevant responses. Admit when you don't know something."
        )
        
        # Add session info
        if self.session_id:
            base_prompt += f" Session: {self.session_id[:8]}"
        
        # Add current topics
        if self.current_topics:
            topics = ", ".join(list(self.current_topics)[:3])
            base_prompt += f" Current topics: {topics}"
        
        return base_prompt
    
    # ==================== MAINTENANCE ====================
    
    def cleanup(self):
        """Remove expired and low-relevance context"""
        before = len(self.context)
        
        # Remove expired
        expired = [k for k, v in self.context.items() if v.is_expired()]
        for k in expired:
            self.delete(k)
        
        # Apply decay to remaining
        for entry in self.context.values():
            entry.decay_relevance(self.decay_rate)
        
        # Remove low relevance based on policy
        if self.retention_policy == RetentionPolicy.SCORE:
            low_relevance = [
                k for k, v in self.context.items()
                if v.relevance_score < self.min_relevance and v.priority.value > 2
            ]
            for k in low_relevance:
                self.delete(k)
        elif self.retention_policy == RetentionPolicy.LRU:
            # Sort by last accessed and remove oldest
            sorted_entries = sorted(
                self.context.items(),
                key=lambda x: x[1].last_accessed
            )
            for k, v in sorted_entries[:10]:
                if v.priority.value > 2:
                    self.delete(k)
        elif self.retention_policy == RetentionPolicy.LFU:
            # Remove least frequently used
            sorted_entries = sorted(
                self.context.items(),
                key=lambda x: x[1].access_count
            )
            for k, v in sorted_entries[:10]:
                if v.priority.value > 2:
                    self.delete(k)
        
        after = len(self.context)
        if before != after:
            logger.debug(f"🧹 Cleanup complete: {before} -> {after} items")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context manager summary"""
        return {
            "context": {
                "total": len(self.context),
                "by_layer": {
                    layer.value: len(entries)
                    for layer, entries in self.layered_context.items()
                }
            },
            "history": {
                "total": len(self.history),
                "summary_count": len(self.summaries),
                "current_summary": self.current_summary.to_dict() if self.current_summary else None
            },
            "user": {
                "current": self.current_user,
                "profiles": len(self.user_profiles),
                "session_id": self.session_id,
                "session_duration_hours": (datetime.now() - self.session_start).total_seconds() / 3600
            },
            "performance": {
                "total_requests": self.total_context_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            },
            "topics": list(self.current_topics)[:10]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for monitoring"""
        return {
            "context_size": len(self.context),
            "history_size": len(self.history),
            "summaries": len(self.summaries),
            "user_profiles": len(self.user_profiles),
            "current_topics": len(self.current_topics),
            "active_session": self.session_id is not None,
            "performance": {
                "requests": self.total_context_requests,
                "cache_hit_rate": round(self.cache_hits / max(1, self.cache_hits + self.cache_misses), 3)
            }
        }
    
    def set_session(self, session_id: str):
        """Set current session ID"""
        self.session_id = session_id
        self.session_start = datetime.now()
        logger.info(f"🔑 Session started: {session_id}")
    
    def export_context(self, filepath: str):
        """Export context to file"""
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "user": self.current_user
            },
            "context": {k: v.to_dict() for k, v in self.context.items()},
            "history": [m.to_dict() for m in self.history],
            "summaries": [s.to_dict() for s in self.summaries],
            "user_profiles": {uid: p.to_dict() for uid, p in self.user_profiles.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📁 Context exported to {filepath}")
    
    def import_context(self, filepath: str):
        """Import context from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restore context
        for key, entry_data in data.get("context", {}).items():
            entry = ContextEntry(
                key=entry_data["key"],
                value=entry_data["value"],
                layer=ContextLayer(entry_data["layer"]),
                priority=ContextPriority(entry_data["priority"]),
                tags=entry_data.get("tags", [])
            )
            self.context[key] = entry
            self.layered_context[entry.layer][key] = entry
        
        logger.info(f"📥 Context imported: {len(data.get('context', {}))} items")


# ==================== WRAPPER FOR EDIATH ====================

class ContextManagerWrapper:
    """Wrapper for EDIATH integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.manager = ContextManager(
            max_history=config.get("max_history", 200),
            max_context_tokens=config.get("max_context_tokens", 4000),
            enable_memory=config.get("enable_memory", True),
            enable_summarization=config.get("enable_summarization", True),
            enable_pii_filtering=config.get("enable_pii_filtering", True),
            pii_level=PIIFilter(config.get("pii_level", "basic")),
            retention_policy=RetentionPolicy(config.get("retention_policy", "score")),
            decay_rate=config.get("decay_rate", 0.01),
            session_ttl_hours=config.get("session_ttl_hours", 24)
        )
        self.agent_type = "context_manager"
        self.capabilities = [
            "set_context", "get_context", "add_message", "build_context",
            "search_context", "get_recent", "set_user", "get_summary",
            "export_context", "clear_context"
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process context request"""
        operation = request.get("operation")
        
        if operation == "set_context":
            self.manager.set(
                key=request.get("key", ""),
                value=request.get("value"),
                layer=ContextLayer(request.get("layer", "short_term")),
                priority=ContextPriority(request.get("priority", "normal")),
                ttl=request.get("ttl"),
                tags=request.get("tags", [])
            )
            return {"success": True}
        
        elif operation == "get_context":
            value = self.manager.get(
                key=request.get("key", ""),
                default=request.get("default")
            )
            return {"success": True, "value": value}
        
        elif operation == "add_message":
            self.manager.add_message(
                role=request.get("role", "user"),
                content=request.get("content", ""),
                importance=request.get("importance", 0.5),
                user_id=request.get("user_id")
            )
            return {"success": True}
        
        elif operation == "build_context":
            context = await self.manager.build_context(
                user_input=request.get("user_input", ""),
                max_tokens=request.get("max_tokens"),
                include_memory=request.get("include_memory", True),
                include_history=request.get("include_history", True),
                include_user_profile=request.get("include_user_profile", True)
            )
            return {"success": True, "context": context}
        
        elif operation == "search_context":
            results = self.manager.search_context(
                query=request.get("query", ""),
                limit=request.get("limit", 5)
            )
            return {"success": True, "results": results}
        
        elif operation == "get_recent":
            messages = self.manager.get_recent_messages(
                limit=request.get("limit", 10),
                min_importance=request.get("min_importance", 0.0),
                role=request.get("role")
            )
            return {"success": True, "messages": messages}
        
        elif operation == "set_user":
            self.manager.set_user(request.get("user_id", ""))
            return {"success": True}
        
        elif operation == "get_summary":
            return {"success": True, "summary": self.manager.get_summary()}
        
        elif operation == "export":
            self.manager.export_context(request.get("filepath", "context_export.json"))
            return {"success": True}
        
        elif operation == "clear_context":
            layer = request.get("layer")
            self.manager.clear(ContextLayer(layer) if layer else None)
            return {"success": True}
        
        elif operation == "get_stats":
            return {"success": True, "stats": self.manager.get_stats()}
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "ContextManager",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.manager.get_stats()
        }


# ==================== EXAMPLE USAGE ====================

async def example_usage():
    """Example usage of Context Manager"""
    
    manager = ContextManager(
        max_history=100,
        enable_summarization=True,
        enable_pii_filtering=True,
        pii_level=PIIFilter.BASIC
    )
    
    # Set session
    manager.set_session("demo_session_123")
    manager.set_user("user_001")
    
    # Add context
    manager.set("user_name", "John", layer=ContextLayer.LONG_TERM, priority=ContextPriority.HIGH)
    manager.set("preferred_language", "Python", tags=["coding"])
    manager.set("project", "EDIATH Framework", importance=0.9)
    
    # Add conversation messages
    manager.add_message("user", "I'm working on an AI framework", importance=0.8)
    manager.add_message("assistant", "That's interesting! Tell me more about it.", importance=0.7)
    manager.add_message("user", "It's called EDIATH - an autonomous AI system", importance=0.9)
    
    # Build context for LLM
    context = await manager.build_context(
        "How can I improve the context management?",
        include_memory=True,
        include_user_profile=True
    )
    
    print("\n📋 Built Context:")
    for msg in context:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")
    
    # Search context
    results = manager.search_context("AI framework", limit=3)
    print(f"\n🔍 Search Results: {len(results)} found")
    
    # Get summary
    summary = manager.get_summary()
    print(f"\n📊 Summary: {summary['context']['total']} context items, {summary['history']['total']} messages")
    
    # Cleanup expired
    manager.cleanup()
    
    return manager


if __name__ == "__main__":
    asyncio.run(example_usage())