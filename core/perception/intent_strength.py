"""Intent Strength - measures the strength of user intent with advanced linguistic analysis, urgency detection, and multi-factor scoring."""

from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import re
import statistics
from collections import deque, defaultdict

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger


class IntentLevel(Enum):
    """Intent strength levels."""

    VERY_WEAK = "very_weak"  # 0.0 - 0.15
    WEAK = "weak"  # 0.15 - 0.35
    MODERATE = "moderate"  # 0.35 - 0.55
    STRONG = "strong"  # 0.55 - 0.75
    VERY_STRONG = "very_strong"  # 0.75 - 0.9
    URGENT = "urgent"  # 0.9 - 1.0


class ActionPriority(Enum):
    """Priority levels for action execution."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    DEFERRED = "deferred"


class IntentType(Enum):
    """Types of user intent."""

    COMMAND = "command"
    REQUEST = "request"
    QUESTION = "question"
    SUGGESTION = "suggestion"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    CASUAL = "casual"


@dataclass
class IntentAnalysis:
    """Comprehensive intent analysis result."""

    strength: float
    level: IntentLevel
    priority: ActionPriority
    urgency_score: float  # 0-1
    certainty_score: float  # 0-1
    intent_type: IntentType
    key_phrases: List[str]
    modifiers: Dict[str, float]
    contextual_factors: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strength": self.strength,
            "level": self.level.value,
            "priority": self.priority.value,
            "urgency_score": self.urgency_score,
            "certainty_score": self.certainty_score,
            "intent_type": self.intent_type.value,
            "key_phrases": self.key_phrases,
            "modifiers": self.modifiers,
            "contextual_factors": self.contextual_factors,
            "timestamp": self.timestamp.isoformat(),
        }

    def requires_immediate_action(self) -> bool:
        """Check if intent requires immediate action."""
        return self.priority in [ActionPriority.IMMEDIATE, ActionPriority.HIGH]

    def get_execution_timeout(self) -> float:
        """Get recommended execution timeout in seconds."""
        if self.priority == ActionPriority.IMMEDIATE:
            return 5.0
        elif self.priority == ActionPriority.HIGH:
            return 30.0
        elif self.priority == ActionPriority.NORMAL:
            return 300.0
        else:
            return 3600.0


class IntentStrength:
    """Advanced intent strength analyzer with multi-factor scoring."""

    # Strong intent indicators
    STRONG_KEYWORDS = {
        "must": 0.25,
        "definitely": 0.25,
        "absolutely": 0.3,
        "immediately": 0.35,
        "urgent": 0.4,
        "critical": 0.45,
        "need": 0.2,
        "require": 0.25,
        "demand": 0.35,
        "essential": 0.3,
        "vital": 0.3,
        "imperative": 0.4,
        "crucial": 0.35,
        "right_now": 0.4,
        "asap": 0.45,
        "now": 0.3,
        "quickly": 0.25,
        "important": 0.25,
        "serious": 0.3,
        "emergency": 0.5,
    }

    # Weak intent indicators
    WEAK_KEYWORDS = {
        "maybe": -0.15,
        "perhaps": -0.15,
        "might": -0.2,
        "consider": -0.1,
        "could": -0.1,
        "possibly": -0.15,
        "sometime": -0.1,
        "later": -0.1,
        "eventually": -0.1,
        "if": -0.05,
        "whenever": -0.05,
        "i guess": -0.2,
        "i think": -0.1,
        "i suppose": -0.15,
        "sort of": -0.1,
        "kind of": -0.1,
    }

    # Intent type indicators
    TYPE_INDICATORS = {
        IntentType.COMMAND: [
            "do",
            "make",
            "go",
            "start",
            "stop",
            "run",
            "execute",
            "perform",
        ],
        IntentType.REQUEST: [
            "please",
            "could you",
            "would you",
            "can you",
            "i would like",
            "i need",
        ],
        IntentType.QUESTION: [
            "?",
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "which",
        ],
        IntentType.SUGGESTION: [
            "suggest",
            "recommend",
            "propose",
            "maybe we should",
            "how about",
        ],
        IntentType.FEEDBACK: ["feedback", "comment", "review", "opinion", "thought"],
        IntentType.COMPLAINT: [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "frustrated",
            "annoyed",
        ],
        IntentType.PRAISE: ["good", "great", "excellent", "awesome", "perfect", "love"],
        IntentType.CASUAL: ["hi", "hello", "hey", "ok", "hmm", "well"],
    }

    # Urgency indicators
    URGENCY_INDICATORS = {
        "urgent": 0.8,
        "asap": 0.9,
        "emergency": 1.0,
        "critical": 0.9,
        "immediately": 0.85,
        "right now": 0.8,
        "now": 0.7,
        "quick": 0.6,
        "fast": 0.6,
        "soon": 0.5,
        "time-sensitive": 0.7,
        "deadline": 0.7,
    }

    # Certainty indicators
    CERTAINTY_INDICATORS = {
        "definitely": 0.9,
        "absolutely": 0.95,
        "certainly": 0.9,
        "surely": 0.85,
        "without doubt": 0.95,
        "for sure": 0.85,
        "guaranteed": 0.9,
        "promise": 0.8,
        "maybe": 0.3,
        "perhaps": 0.3,
        "might": 0.4,
        "could": 0.4,
        "possibly": 0.35,
        "i think": 0.5,
        "i believe": 0.55,
        "i guess": 0.4,
    }

    def __init__(
        self,
        enable_historical_tracking: bool = True,
        history_size: int = 1000,
        context_window: int = 5,
        adaptive_scoring: bool = True,
        min_confidence: float = 0.3,
    ):
        """Initialize advanced intent strength analyzer."""
        self.enable_historical_tracking = enable_historical_tracking
        self.history_size = history_size
        self.context_window = context_window
        self.adaptive_scoring = adaptive_scoring
        self.min_confidence = min_confidence

        # Storage
        self.measurements: deque = deque(maxlen=history_size)
        self.user_patterns: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "avg_strength": 0.5,
                "common_intents": defaultdict(int),
                "response_times": [],
                "escalation_tendency": 0.5,
            }
        )
        self.context_buffer: deque = deque(maxlen=context_window)

        # Statistics
        self.analysis_count = 0
        self.intent_distribution: Dict[IntentType, int] = defaultdict(int)
        self.level_distribution: Dict[IntentLevel, int] = defaultdict(int)

        # Adaptive weights
        self.weights = {
            "keyword_score": 0.35,
            "punctuation_score": 0.15,
            "syntax_score": 0.15,
            "context_score": 0.15,
            "historical_score": 0.10,
            "urgency_score": 0.10,
        }

        # Performance metrics
        self.metrics = {
            "avg_analysis_time_ms": 0.0,
            "total_analyses": 0,
            "accuracy_feedback": [],
        }

        logger.info(
            f"✅ IntentStrength initialized (historical_tracking={enable_historical_tracking}, adaptive={adaptive_scoring})"
        )

    def measure(
        self,
        intent_text: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        session_history: Optional[List[str]] = None,
    ) -> IntentAnalysis:
        """Measure intent strength with comprehensive analysis.

        Args:
            intent_text: Text expressing intent
            user_id: Optional user identifier for personalization
            context: Additional contextual information
            session_history: Previous messages in the session

        Returns:
            IntentAnalysis with detailed results
        """
        import time

        start_time = time.time()

        # Initialize scores
        scores = {
            "keyword_score": 0.0,
            "punctuation_score": 0.0,
            "syntax_score": 0.0,
            "context_score": 0.0,
            "historical_score": 0.0,
            "urgency_score": 0.0,
        }

        # Calculate individual scores
        scores["keyword_score"] = self._calculate_keyword_score(intent_text)
        scores["punctuation_score"] = self._calculate_punctuation_score(intent_text)
        scores["syntax_score"] = self._calculate_syntax_score(intent_text)
        scores["context_score"] = self._calculate_context_score(
            intent_text, context, session_history
        )
        scores["historical_score"] = self._calculate_historical_score(user_id)
        scores["urgency_score"] = self._calculate_urgency_score(intent_text)

        # Calculate certainty score
        certainty_score = self._calculate_certainty_score(intent_text)

        # Determine intent type
        intent_type = self._determine_intent_type(intent_text)

        # Calculate final strength
        total_weight = sum(self.weights.values())
        raw_strength = sum(scores[k] * self.weights[k] for k in scores) / total_weight

        # Apply adaptive adjustments
        if self.adaptive_scoring and user_id:
            raw_strength = self._apply_adaptive_adjustment(raw_strength, user_id)

        # Clamp and get level
        strength = max(0.0, min(1.0, raw_strength))
        level = self._get_intent_level(strength)
        priority = self._get_action_priority(strength, scores["urgency_score"])

        # Extract key phrases
        key_phrases = self._extract_key_phrases(intent_text)

        # Compile modifiers
        modifiers = self._compile_modifiers(intent_text)

        # Create analysis result
        analysis = IntentAnalysis(
            strength=strength,
            level=level,
            priority=priority,
            urgency_score=scores["urgency_score"],
            certainty_score=certainty_score,
            intent_type=intent_type,
            key_phrases=key_phrases,
            modifiers=modifiers,
            contextual_factors=context or {},
        )

        # Store measurement
        self._store_measurement(intent_text, analysis, user_id)

        # Update statistics
        self._update_statistics(analysis, intent_type)

        # Update user patterns
        if user_id:
            self._update_user_patterns(user_id, analysis)

        # Update metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics["total_analyses"] += 1
        self.metrics["avg_analysis_time_ms"] = (
            self.metrics["avg_analysis_time_ms"] * (self.metrics["total_analyses"] - 1)
            + elapsed_ms
        ) / self.metrics["total_analyses"]

        logger.debug(
            f"Intent analysis: strength={strength:.2f}, level={level.value}, type={intent_type.value}"
        )

        return analysis

    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate score based on keywords."""
        text_lower = text.lower()
        score = 0.5  # Base score

        # Strong keywords
        for keyword, weight in self.STRONG_KEYWORDS.items():
            if keyword in text_lower:
                score += weight
                # Multiple occurrences add more
                count = text_lower.count(keyword)
                if count > 1:
                    score += weight * 0.3 * (count - 1)

        # Weak keywords
        for keyword, weight in self.WEAK_KEYWORDS.items():
            if keyword in text_lower:
                score += weight
                # Multiple occurrences weaken more
                count = text_lower.count(keyword)
                if count > 1:
                    score += weight * 0.2 * (count - 1)

        return max(0.0, min(1.0, score))

    def _calculate_punctuation_score(self, text: str) -> float:
        """Calculate score based on punctuation."""
        score = 0.5

        # Exclamation marks (strong)
        exclamation_count = text.count("!")
        score += min(0.3, exclamation_count * 0.15)

        # Question marks (weakens)
        question_count = text.count("?")
        score -= min(0.2, question_count * 0.08)

        # Ellipsis (indicates hesitation)
        if "..." in text:
            score -= 0.1

        # Multiple punctuation (e.g., "!!!" or "?!")
        if re.search(r"!{2,}", text):
            score += 0.1
        if re.search(r"\?{2,}", text):
            score -= 0.05

        return max(0.0, min(1.0, score))

    def _calculate_syntax_score(self, text: str) -> float:
        """Calculate score based on syntactic features."""
        score = 0.5

        # All caps (emphasis)
        words = text.split()
        if words and len(text) > 3:
            upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            if upper_words > 0:
                score += min(0.25, upper_words * 0.1)

        # Sentence length
        if len(words) > 10:
            score += 0.05
        elif len(words) < 3:
            score -= 0.05

        # First-person pronouns (stronger intent)
        if re.search(r"\b(i|we|me|us)\b", text.lower()):
            score += 0.05

        # Imperative mood detection
        if re.search(r"^\s*(do|make|go|run|stop|start|execute)\b", text.lower()):
            score += 0.1

        # Conditional mood (weaker)
        if re.search(r"\b(if|whether|maybe|perhaps)\b", text.lower()):
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_context_score(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
        session_history: Optional[List[str]],
    ) -> float:
        """Calculate score based on context."""
        score = 0.5

        if context:
            # Conversation topic importance
            if context.get("topic_importance"):
                score += context["topic_importance"] * 0.2

            # Previous intent strength
            if context.get("previous_intent_strength"):
                prev_strength = context["previous_intent_strength"]
                score += (prev_strength - 0.5) * 0.15

            # Time pressure
            if context.get("time_pressure"):
                score += context["time_pressure"] * 0.2

            # User's emotional state
            if context.get("emotional_intensity"):
                score += context["emotional_intensity"] * 0.15

        # Session history analysis
        if session_history and len(session_history) >= 2:
            # Repetition indicates stronger intent
            last_message = session_history[-1] if session_history else ""
            if last_message and text.lower() == last_message.lower():
                score += 0.15

            # Escalation detection
            if len(session_history) >= 3:
                recent_messages = session_history[-3:]
                if all(len(m) > 20 for m in recent_messages):
                    score += 0.05

        return max(0.0, min(1.0, score))

    def _calculate_historical_score(self, user_id: Optional[str]) -> float:
        """Calculate score based on user's historical patterns."""
        if not user_id or not self.enable_historical_tracking:
            return 0.5

        user_data = self.user_patterns.get(user_id, {})
        return user_data.get("avg_strength", 0.5)

    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on text."""
        text_lower = text.lower()
        urgency_score = 0.0

        for indicator, weight in self.URGENCY_INDICATORS.items():
            if indicator in text_lower:
                urgency_score = max(urgency_score, weight)
                # Multiple indicators increase score
                if urgency_score > 0:
                    urgency_score = min(1.0, urgency_score + 0.1)

        # Time references increase urgency
        time_refs = ["now", "immediately", "right away", "as soon as possible", "asap"]
        for ref in time_refs:
            if ref in text_lower:
                urgency_score = max(urgency_score, 0.6)

        # Exclamation marks add urgency
        exclamation_count = text.count("!")
        if exclamation_count > 0:
            urgency_score = min(1.0, urgency_score + exclamation_count * 0.1)

        return max(0.0, min(1.0, urgency_score))

    def _calculate_certainty_score(self, text: str) -> float:
        """Calculate certainty score based on text."""
        text_lower = text.lower()
        certainty = 0.6  # Base certainty

        for indicator, weight in self.CERTAINTY_INDICATORS.items():
            if indicator in text_lower:
                certainty = weight
                break

        # Question marks reduce certainty
        if "?" in text:
            certainty *= 0.7

        # Hedging phrases reduce certainty
        hedging = ["i think", "i believe", "i guess", "maybe", "perhaps"]
        for phrase in hedging:
            if phrase in text_lower:
                certainty *= 0.8

        return max(0.0, min(1.0, certainty))

    def _determine_intent_type(self, text: str) -> IntentType:
        """Determine the type of intent."""
        text_lower = text.lower()

        # Check for question first
        if "?" in text:
            return IntentType.QUESTION

        # Check other types
        for intent_type, indicators in self.TYPE_INDICATORS.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return intent_type

        # Default to command
        return IntentType.COMMAND

    def _get_intent_level(self, strength: float) -> IntentLevel:
        """Get intent level from strength score."""
        if strength >= 0.9:
            return IntentLevel.URGENT
        elif strength >= 0.75:
            return IntentLevel.VERY_STRONG
        elif strength >= 0.55:
            return IntentLevel.STRONG
        elif strength >= 0.35:
            return IntentLevel.MODERATE
        elif strength >= 0.15:
            return IntentLevel.WEAK
        else:
            return IntentLevel.VERY_WEAK

    def _get_action_priority(self, strength: float, urgency: float) -> ActionPriority:
        """Determine action priority based on strength and urgency."""
        combined = (strength * 0.6) + (urgency * 0.4)

        if combined >= 0.85:
            return ActionPriority.IMMEDIATE
        elif combined >= 0.65:
            return ActionPriority.HIGH
        elif combined >= 0.45:
            return ActionPriority.NORMAL
        elif combined >= 0.25:
            return ActionPriority.LOW
        else:
            return ActionPriority.DEFERRED

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that indicate intent."""
        phrases = []
        text_lower = text.lower()

        # Extract verb-noun pairs
        verb_noun_pattern = (
            r"\b(do|make|go|run|start|stop|create|delete|update|get|set)\s+(\w+)"
        )
        matches = re.findall(verb_noun_pattern, text_lower)
        for match in matches:
            phrases.append(f"{match[0]} {match[1]}")

        # Extract urgent phrases
        for phrase in self.URGENCY_INDICATORS.keys():
            if phrase in text_lower:
                phrases.append(phrase)

        # Extract strong intent phrases
        for phrase in self.STRONG_KEYWORDS.keys():
            if phrase in text_lower and len(phrase) > 3:
                phrases.append(phrase)

        return list(set(phrases))[:5]  # Return unique, limit to 5

    def _compile_modifiers(self, text: str) -> Dict[str, float]:
        """Compile intent modifiers and their strengths."""
        modifiers = {}
        text_lower = text.lower()

        # Urgency modifiers
        for word, weight in self.URGENCY_INDICATORS.items():
            if word in text_lower:
                modifiers["urgency"] = max(modifiers.get("urgency", 0), weight)

        # Certainty modifiers
        for word, weight in self.CERTAINTY_INDICATORS.items():
            if word in text_lower:
                modifiers["certainty"] = max(modifiers.get("certainty", 0), weight)

        # Politeness
        if "please" in text_lower:
            modifiers["politeness"] = 0.7
        if "thank" in text_lower:
            modifiers["politeness"] = modifiers.get("politeness", 0) + 0.3

        # Emotional intensity
        if text.count("!") >= 2:
            modifiers["emotional_intensity"] = 0.8
        elif text.count("!") == 1:
            modifiers["emotional_intensity"] = 0.5

        return modifiers

    def _apply_adaptive_adjustment(self, strength: float, user_id: str) -> float:
        """Apply adaptive adjustment based on user patterns."""
        user_data = self.user_patterns.get(user_id, {})
        avg_strength = user_data.get("avg_strength", 0.5)
        escalation = user_data.get("escalation_tendency", 0.5)

        # Adjust based on user's typical behavior
        if avg_strength > 0.6:
            # User tends to express intent strongly
            adjusted = strength * 0.9 + 0.05  # Slight reduction
        elif avg_strength < 0.4:
            # User tends to express intent weakly
            adjusted = strength * 1.1 - 0.05  # Slight boost
        else:
            adjusted = strength

        # Adjust for escalation tendency
        if escalation > 0.7:
            # User escalates quickly, be cautious
            adjusted = min(1.0, adjusted * 0.95)

        return max(0.0, min(1.0, adjusted))

    def _update_user_patterns(self, user_id: str, analysis: IntentAnalysis):
        """Update user-specific patterns."""
        user_data = self.user_patterns[user_id]

        # Update average strength (moving average)
        current_avg = user_data["avg_strength"]
        user_data["avg_strength"] = current_avg * 0.9 + analysis.strength * 0.1

        # Update intent distribution
        user_data["common_intents"][analysis.intent_type] += 1

        # Update escalation tendency
        if analysis.level in [
            IntentLevel.STRONG,
            IntentLevel.VERY_STRONG,
            IntentLevel.URGENT,
        ]:
            user_data["escalation_tendency"] = min(
                1.0, user_data["escalation_tendency"] + 0.05
            )
        else:
            user_data["escalation_tendency"] = max(
                0.0, user_data["escalation_tendency"] - 0.02
            )

    def _store_measurement(
        self, text: str, analysis: IntentAnalysis, user_id: Optional[str]
    ):
        """Store measurement for historical analysis."""
        self.measurements.append(
            {
                "text": text,
                "analysis": analysis,
                "user_id": user_id,
                "timestamp": datetime.now(),
            }
        )

        # Maintain size limit
        if len(self.measurements) > self.history_size:
            self.measurements.popleft()

    def _update_statistics(self, analysis: IntentAnalysis, intent_type: IntentType):
        """Update overall statistics."""
        self.analysis_count += 1
        self.intent_distribution[intent_type] += 1
        self.level_distribution[analysis.level] += 1

    # #================#================#============#=============
    # BATCH OPERATIONS
    # #================#================#============#=============

    def batch_measure(
        self,
        texts: List[str],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[IntentAnalysis]:
        """Measure intent strength for multiple texts."""
        results = []
        for text in texts:
            analysis = self.measure(text, user_id, context)
            results.append(analysis)
        return results

    def analyze_commitment_level(self, strength: float) -> Dict[str, Any]:
        """Analyze commitment level from intent strength (legacy compatibility)."""
        level = self._get_intent_level(strength)
        priority = self._get_action_priority(strength, 0.5)

        return {
            "level": level.value,
            "recommended_action": priority.value,
            "confidence": strength,
            "should_execute": strength >= 0.55,
        }

    def analyze_conversation_intent(
        self, messages: List[str], user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze intent progression in a conversation."""
        analyses = self.batch_measure(messages, user_id)

        if not analyses:
            return {"error": "No messages to analyze"}

        strengths = [a.strength for a in analyses]
        urgency_scores = [a.urgency_score for a in analyses]

        # Detect intent escalation
        escalation = False
        if len(strengths) >= 3:
            first_third = statistics.mean(strengths[: len(strengths) // 3])
            last_third = statistics.mean(strengths[-len(strengths) // 3 :])
            escalation = last_third > first_third * 1.2

        # Find critical points
        critical_points = []
        for i, analysis in enumerate(analyses):
            if analysis.priority in [ActionPriority.IMMEDIATE, ActionPriority.HIGH]:
                critical_points.append(
                    {
                        "index": i,
                        "message": messages[i][:100],
                        "strength": analysis.strength,
                        "priority": analysis.priority.value,
                    }
                )

        return {
            "total_messages": len(messages),
            "avg_strength": statistics.mean(strengths),
            "max_strength": max(strengths),
            "min_strength": min(strengths),
            "avg_urgency": statistics.mean(urgency_scores),
            "intent_escalation": escalation,
            "critical_points": critical_points,
            "dominant_intent_type": (
                max(self.intent_distribution.items(), key=lambda x: x[1])[0].value
                if self.intent_distribution
                else "unknown"
            ),
            "strength_trend": self._calculate_trend(strengths),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"

        if NUMPY_AVAILABLE:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
        else:
            # Simple linear regression
            n = len(values)
            x = list(range(n))
            slope = (
                n * sum(x[i] * values[i] for i in range(n)) - sum(x) * sum(values)
            ) / (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    # #================#================#============#=============
    # UTILITIES
    # #================#================#============#=============

    def get_user_intent_profile(self, user_id: str) -> Dict[str, Any]:
        """Get intent profile for a specific user."""
        user_data = self.user_patterns.get(user_id, {})

        if not user_data:
            return {"message": "No data for this user"}

        return {
            "user_id": user_id,
            "avg_strength": user_data.get("avg_strength", 0.5),
            "escalation_tendency": user_data.get("escalation_tendency", 0.5),
            "common_intents": dict(user_data.get("common_intents", {})),
            "total_interactions": sum(user_data.get("common_intents", {}).values()),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive analyzer statistics."""
        if not self.measurements:
            return {
                "count": 0,
                "avg_strength": 0,
                "min_strength": 0,
                "max_strength": 0,
                "avg_urgency": 0,
                "total_analyses": self.analysis_count,
            }

        strengths = [m["analysis"].strength for m in self.measurements]
        urgencies = [m["analysis"].urgency_score for m in self.measurements]
        certainties = [m["analysis"].certainty_score for m in self.measurements]

        return {
            "count": len(self.measurements),
            "total_analyses": self.analysis_count,
            "avg_strength": statistics.mean(strengths),
            "min_strength": min(strengths),
            "max_strength": max(strengths),
            "std_strength": statistics.stdev(strengths) if len(strengths) > 1 else 0,
            "avg_urgency": statistics.mean(urgencies),
            "avg_certainty": statistics.mean(certainties),
            "intent_distribution": {
                k.value: v for k, v in self.intent_distribution.items()
            },
            "level_distribution": {
                k.value: v for k, v in self.level_distribution.items()
            },
            "performance": {
                "avg_analysis_time_ms": self.metrics["avg_analysis_time_ms"],
                "total_analyses_performed": self.metrics["total_analyses"],
            },
            "unique_users": len(self.user_patterns),
            "adaptive_scoring_enabled": self.adaptive_scoring,
        }

    def get_recent_measurements(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent intent measurements."""
        recent = list(self.measurements)[-limit:]
        return [
            {
                "text": m["text"][:100],
                "strength": m["analysis"].strength,
                "level": m["analysis"].level.value,
                "priority": m["analysis"].priority.value,
                "timestamp": m["timestamp"].isoformat(),
            }
            for m in recent
        ]

    def update_weight(self, weight_name: str, new_weight: float):
        """Update a scoring weight."""
        if weight_name in self.weights:
            self.weights[weight_name] = max(0.05, min(0.5, new_weight))
            logger.info(f"Updated weight {weight_name} to {new_weight}")

    def record_accuracy_feedback(self, was_correct: bool, strength: float):
        """Record accuracy feedback for model improvement."""
        self.metrics["accuracy_feedback"].append(
            {"correct": was_correct, "strength": strength, "timestamp": datetime.now()}
        )

        # Keep limited feedback
        if len(self.metrics["accuracy_feedback"]) > 1000:
            self.metrics["accuracy_feedback"] = self.metrics["accuracy_feedback"][
                -1000:
            ]

        # Calculate running accuracy
        recent_feedback = self.metrics["accuracy_feedback"][-100:]
        accuracy = (
            sum(1 for f in recent_feedback if f["correct"]) / len(recent_feedback)
            if recent_feedback
            else 0.5
        )

        logger.debug(
            f"Intent accuracy feedback: {'correct' if was_correct else 'incorrect'}, accuracy={accuracy:.2f}"
        )

    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get accuracy metrics for intent detection."""
        if not self.metrics["accuracy_feedback"]:
            return {"message": "No accuracy data available"}

        recent = self.metrics["accuracy_feedback"][-100:]
        total_correct = sum(1 for f in recent if f["correct"])

        # Calculate calibration
        strengths = [f["strength"] for f in recent]
        avg_strength = statistics.mean(strengths) if strengths else 0.5
        accuracy = total_correct / len(recent) if recent else 0.5

        return {
            "accuracy": accuracy,
            "total_feedback": len(self.metrics["accuracy_feedback"]),
            "recent_feedback": len(recent),
            "avg_predicted_strength": avg_strength,
            "calibration_error": abs(avg_strength - accuracy),
            "well_calibrated": abs(avg_strength - accuracy) < 0.1,
        }

    def reset(self):
        """Reset all data and statistics."""
        self.measurements.clear()
        self.user_patterns.clear()
        self.context_buffer.clear()
        self.intent_distribution.clear()
        self.level_distribution.clear()
        self.analysis_count = 0
        self.metrics = {
            "avg_analysis_time_ms": 0.0,
            "total_analyses": 0,
            "accuracy_feedback": [],
        }

        logger.info("🔄 IntentStrength reset")


# #================#================#============#=============
# CONVENIENCE FUNCTIONS
# #================#================#============#=============


def get_intent_color(intent_type: IntentType) -> str:
    """Get color associated with intent type."""
    colors = {
        IntentType.COMMAND: "#FF6B6B",  # Coral Red
        IntentType.REQUEST: "#4ECDC4",  # Turquoise
        IntentType.QUESTION: "#45B7D1",  # Sky Blue
        IntentType.SUGGESTION: "#96CEB4",  # Sage Green
        IntentType.FEEDBACK: "#FFEAA7",  # Light Gold
        IntentType.COMPLAINT: "#FF6B6B",  # Coral Red
        IntentType.PRAISE: "#A8E6CF",  # Mint Green
        IntentType.CASUAL: "#DDA0DD",  # Plum
    }
    return colors.get(intent_type, "#95A5A6")  # Gray default


def get_intent_icon(intent_type: IntentType) -> str:
    """Get icon for intent type."""
    icons = {
        IntentType.COMMAND: "⚡",
        IntentType.REQUEST: "🙏",
        IntentType.QUESTION: "❓",
        IntentType.SUGGESTION: "💡",
        IntentType.FEEDBACK: "📝",
        IntentType.COMPLAINT: "⚠️",
        IntentType.PRAISE: "⭐",
        IntentType.CASUAL: "💬",
    }
    return icons.get(intent_type, "📌")


__all__ = [
    "IntentStrength",
    "IntentLevel",
    "ActionPriority",
    "IntentType",
    "IntentAnalysis",
    "get_intent_color",
    "get_intent_icon",
]
