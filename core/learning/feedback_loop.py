"""Feedback Loop - collects and processes feedback for learning with advanced analytics, ML integration, and real-time optimization."""

import asyncio
import json
import hashlib
import statistics
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..utils.logger import logger


class FeedbackType(Enum):
    """Types of feedback with priority levels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CORRECTIVE = "corrective"
    CRITICAL = "critical"
    SUGGESTION = "suggestion"
    PRAISE = "praise"
    COMPLAINT = "complaint"


class FeedbackPriority(Enum):
    """Priority levels for feedback processing."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


class LearningSignal(Enum):
    """Signals that trigger learning from feedback."""

    STRENGTHEN = "strengthen"  # Reinforce successful behavior
    WEAKEN = "weaken"  # Reduce unsuccessful behavior
    MODIFY = "modify"  # Adjust behavior
    EXPLORE = "explore"  # Try new approaches
    AVOID = "avoid"  # Avoid similar situations
    PRIORITIZE = "prioritize"  # Give higher priority


@dataclass
class FeedbackMetadata:
    """Enhanced metadata for feedback items."""

    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    environment: str = "production"
    model_version: Optional[str] = None
    processing_time_ms: float = 0.0
    confidence: float = 1.0
    importance_weight: float = 1.0
    expires_at: Optional[datetime] = None


@dataclass
class FeedbackItem:
    """Enhanced single feedback item with advanced features."""

    target: str
    score: float
    feedback_type: FeedbackType = FeedbackType.NEUTRAL
    comment: str = ""
    source: Optional[str] = None
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    metadata: FeedbackMetadata = field(default_factory=FeedbackMetadata)
    timestamp: datetime = field(default_factory=datetime.now)

    # Advanced fields
    id: str = field(
        default_factory=lambda: hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:16]
    )
    parent_id: Optional[str] = None  # For threaded feedback
    learning_signals: List[LearningSignal] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 to 1
    impact_score: float = 0.5  # 0 to 1
    recurrence_count: int = 1

    def __post_init__(self):
        """Validate and normalize fields."""
        self.score = max(0.0, min(1.0, float(self.score)))
        self.sentiment_score = max(-1.0, min(1.0, float(self.sentiment_score)))
        self.impact_score = max(0.0, min(1.0, float(self.impact_score)))

        # Auto-determine priority if not set
        if self.priority == FeedbackPriority.MEDIUM:
            if self.score < 0.3:
                self.priority = FeedbackPriority.HIGH
            elif self.score > 0.8:
                self.priority = FeedbackPriority.LOW
            elif self.feedback_type == FeedbackType.CRITICAL:
                self.priority = FeedbackPriority.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        data["priority"] = self.priority.value
        data["learning_signals"] = [s.value for s in self.learning_signals]
        data["timestamp"] = self.timestamp.isoformat()
        if self.metadata.expires_at:
            data["metadata"]["expires_at"] = self.metadata.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackItem":
        """Create from dictionary."""
        # Convert enum strings back to enums
        data["feedback_type"] = FeedbackType(data.get("feedback_type", "neutral"))
        data["priority"] = FeedbackPriority(data.get("priority", 1))
        data["learning_signals"] = [
            LearningSignal(s) for s in data.get("learning_signals", [])
        ]
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        if "metadata" in data and data["metadata"].get("expires_at"):
            data["metadata"]["expires_at"] = datetime.fromisoformat(
                data["metadata"]["expires_at"]
            )

        return cls(**data)


class FeedbackAnalyzer:
    """Advanced analytics for feedback data."""

    def __init__(self):
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.regression_model = None

    def analyze_trends(self, feedback_items: List[FeedbackItem]) -> Dict[str, Any]:
        """Analyze trends in feedback over time."""
        if not feedback_items:
            return {"trend": "insufficient_data", "slope": 0}

        # Group by time intervals
        by_day = defaultdict(list)
        for item in feedback_items:
            day_key = item.timestamp.date()
            by_day[day_key].append(item.score)

        # Calculate daily averages
        days = sorted(by_day.keys())
        if len(days) < 2:
            return {
                "trend": "stable",
                "slope": 0,
                "daily_averages": {
                    str(k): statistics.mean(v) for k, v in by_day.items()
                },
            }

        daily_avgs = [statistics.mean(by_day[day]) for day in days]

        # Calculate trend using linear regression if available
        trend = "stable"
        slope = 0

        if SKLEARN_AVAILABLE and len(daily_avgs) >= 2:
            X = np.arange(len(daily_avgs)).reshape(-1, 1)
            y = np.array(daily_avgs)
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]

            if slope > 0.05:
                trend = "improving"
            elif slope < -0.05:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "trend": trend,
            "slope": slope,
            "daily_averages": {str(k): v for k, v in zip(days, daily_avgs)},
            "improvement_rate": slope * 100,  # Percentage change per day
        }

    def analyze_patterns(self, feedback_items: List[FeedbackItem]) -> Dict[str, Any]:
        """Identify patterns in feedback."""
        patterns = {
            "common_issues": [],
            "success_patterns": [],
            "correlations": {},
            "recurring_themes": [],
        }

        if not feedback_items:
            return patterns

        # Find common issues (negative feedback with low scores)
        negative_items = [
            f
            for f in feedback_items
            if f.feedback_type
            in [FeedbackType.NEGATIVE, FeedbackType.COMPLAINT, FeedbackType.CRITICAL]
        ]
        if negative_items:
            issue_counts = defaultdict(int)
            for item in negative_items:
                for tag in item.metadata.tags:
                    issue_counts[tag] += 1

            patterns["common_issues"] = [
                {
                    "issue": tag,
                    "count": count,
                    "severity": "high" if count > 5 else "medium",
                }
                for tag, count in sorted(
                    issue_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

        # Find success patterns (positive feedback with high scores)
        positive_items = [
            f
            for f in feedback_items
            if f.feedback_type in [FeedbackType.POSITIVE, FeedbackType.PRAISE]
        ]
        if positive_items:
            success_counts = defaultdict(int)
            for item in positive_items:
                for tag in item.metadata.tags:
                    success_counts[tag] += 1

            patterns["success_patterns"] = [
                {
                    "pattern": tag,
                    "count": count,
                    "success_rate": count / len(positive_items),
                }
                for tag, count in sorted(
                    success_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

        # Analyze recurring themes
        all_comments = [f.comment for f in feedback_items if f.comment]
        if all_comments and PANDAS_AVAILABLE:
            # Simple theme extraction based on keywords
            themes = defaultdict(int)
            keywords = [
                "speed",
                "accuracy",
                "usability",
                "performance",
                "reliability",
                "interface",
                "response",
                "quality",
                "efficiency",
                "stability",
            ]

            for comment in all_comments:
                comment_lower = comment.lower()
                for keyword in keywords:
                    if keyword in comment_lower:
                        themes[keyword] += 1

            patterns["recurring_themes"] = [
                {"theme": theme, "frequency": count}
                for theme, count in sorted(
                    themes.items(), key=lambda x: x[1], reverse=True
                )
            ]

        return patterns

    def calculate_confidence_intervals(
        self, scores: List[float], confidence: float = 0.95
    ) -> Dict[str, float]:
        """Calculate confidence intervals for scores."""
        if not scores or len(scores) < 2:
            return {
                "mean": statistics.mean(scores) if scores else 0,
                "lower": 0,
                "upper": 0,
            }

        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        n = len(scores)

        # Using z-score approximation for confidence interval
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        margin = z * (stdev / (n**0.5))

        return {
            "mean": mean,
            "lower": max(0, mean - margin),
            "upper": min(1, mean + margin),
            "confidence_level": confidence,
            "sample_size": n,
        }


class FeedbackProcessor:
    """Process and enrich feedback items."""

    def __init__(self):
        self.analyzer = FeedbackAnalyzer()

    def process_feedback(self, item: FeedbackItem) -> FeedbackItem:
        """Process and enrich a feedback item."""
        # Calculate sentiment from comment if available
        if item.comment and item.sentiment_score == 0:
            item.sentiment_score = self._analyze_sentiment(item.comment)

        # Determine learning signals based on feedback
        if not item.learning_signals:
            item.learning_signals = self._determine_learning_signals(item)

        # Calculate impact score based on priority and score
        item.impact_score = self._calculate_impact_score(item)

        # Add processing metadata
        item.metadata.processing_time_ms = datetime.now().timestamp() * 1000

        return item

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to 1)."""
        # Simple keyword-based sentiment analysis
        positive_words = [
            "good",
            "great",
            "excellent",
            "awesome",
            "perfect",
            "amazing",
            "love",
            "like",
            "happy",
        ]
        negative_words = [
            "bad",
            "poor",
            "terrible",
            "awful",
            "horrible",
            "hate",
            "dislike",
            "angry",
            "frustrated",
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _determine_learning_signals(self, item: FeedbackItem) -> List[LearningSignal]:
        """Determine appropriate learning signals from feedback."""
        signals = []

        if item.score >= 0.8:
            signals.append(LearningSignal.STRENGTHEN)
            if item.feedback_type == FeedbackType.POSITIVE:
                signals.append(LearningSignal.PRIORITIZE)
        elif item.score <= 0.3:
            signals.append(LearningSignal.WEAKEN)
            if item.feedback_type == FeedbackType.CORRECTIVE:
                signals.append(LearningSignal.MODIFY)
            if item.score <= 0.1:
                signals.append(LearningSignal.AVOID)

        if item.feedback_type == FeedbackType.SUGGESTION:
            signals.append(LearningSignal.EXPLORE)

        return signals

    def _calculate_impact_score(self, item: FeedbackItem) -> float:
        """Calculate impact score based on priority and other factors."""
        base_score = item.score

        # Adjust based on priority
        priority_multipliers = {
            FeedbackPriority.LOW: 0.5,
            FeedbackPriority.MEDIUM: 1.0,
            FeedbackPriority.HIGH: 1.5,
            FeedbackPriority.CRITICAL: 2.0,
            FeedbackPriority.EMERGENCY: 3.0,
        }

        multiplier = priority_multipliers.get(item.priority, 1.0)

        # Adjust based on recurrence
        recurrence_boost = min(1.0, item.recurrence_count / 10)

        impact = base_score * multiplier * (1 + recurrence_boost)
        return min(1.0, impact)


class FeedbackLoop:
    """Advanced feedback loop with analytics, persistence, and real-time processing."""

    def __init__(
        self,
        window_size: int = 1000,
        persistence_enabled: bool = True,
        persistence_path: str = "data/feedback",
        auto_save_interval: int = 60,  # seconds
        enable_analytics: bool = True,
        enable_callbacks: bool = True,
        max_feedback_age_days: int = 90,
        enable_ml: bool = True,
    ):
        """Initialize advanced feedback loop."""
        self.feedback_items: List[FeedbackItem] = []
        self.window_size = window_size
        self.persistence_enabled = persistence_enabled
        self.persistence_path = Path(persistence_path)
        self.auto_save_interval = auto_save_interval
        self.enable_analytics = enable_analytics
        self.enable_callbacks = enable_callbacks
        self.max_feedback_age_days = max_feedback_age_days
        self.enable_ml = enable_ml and SKLEARN_AVAILABLE

        # Advanced data structures
        self.feedback_by_target: Dict[str, List[FeedbackItem]] = defaultdict(list)
        self.feedback_by_type: Dict[FeedbackType, List[FeedbackItem]] = defaultdict(
            list
        )
        self.feedback_by_user: Dict[str, List[FeedbackItem]] = defaultdict(list)
        self.feedback_by_session: Dict[str, List[FeedbackItem]] = defaultdict(list)

        # Real-time metrics
        self.realtime_scores: deque = deque(maxlen=100)
        self.score_history: List[float] = []
        self.timestamp_history: List[datetime] = []

        # Learning aggregates
        self.target_aggregates: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "total_score": 0,
                "count": 0,
                "avg_score": 0.5,
                "positive_count": 0,
                "negative_count": 0,
                "last_updated": datetime.now(),
            }
        )

        # Callbacks
        self._callbacks: List[Callable[[FeedbackItem], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Async tasks
        self._save_task: Optional[asyncio.Task] = None
        self._running = False

        # Processor and analyzer
        self.processor = FeedbackProcessor()
        self.analyzer = FeedbackAnalyzer()

        # Statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 30

        # Ensure persistence directory exists
        if self.persistence_enabled:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"✅ Advanced FeedbackLoop initialized (window={window_size}, ml={self.enable_ml})"
        )

    # #==================== Core Feedback Recording #====================

    def record_feedback(
        self,
        target: str,
        score: float,
        comment: str = "",
        feedback_type: Union[FeedbackType, str] = FeedbackType.NEUTRAL,
        source: Optional[str] = None,
        priority: Union[FeedbackPriority, str, int] = FeedbackPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        recurrence_count: int = 1,
    ) -> FeedbackItem:
        """Record feedback with advanced options."""

        # Convert string enums if needed
        if isinstance(feedback_type, str):
            feedback_type = FeedbackType(feedback_type.lower())
        if isinstance(priority, str):
            priority = FeedbackPriority(priority.upper())
        elif isinstance(priority, int):
            priority = FeedbackPriority(priority)

        # Create metadata
        feedback_metadata = FeedbackMetadata(
            context=metadata or {},
            tags=tags or [],
            user_id=user_id,
            session_id=session_id,
            importance_weight=self._calculate_importance_weight(score, feedback_type),
        )

        # Create feedback item
        item = FeedbackItem(
            target=target,
            score=score,
            feedback_type=feedback_type,
            comment=comment,
            source=source,
            priority=priority,
            metadata=feedback_metadata,
            parent_id=parent_id,
            recurrence_count=recurrence_count,
        )

        # Process and enrich
        item = self.processor.process_feedback(item)

        # Store feedback
        self._add_feedback_item(item)

        # Trigger callbacks
        if self.enable_callbacks:
            self._trigger_callbacks(item)

        # Invalidate cache
        self._invalidate_cache()

        logger.debug(
            f"📝 Feedback recorded: {target} -> score={score:.2f}, type={feedback_type.value}"
        )

        return item

    def _add_feedback_item(self, item: FeedbackItem):
        """Add feedback item to all indexes."""
        self.feedback_items.append(item)
        self.feedback_by_target[item.target].append(item)
        self.feedback_by_type[item.feedback_type].append(item)

        if item.metadata.user_id:
            self.feedback_by_user[item.metadata.user_id].append(item)
        if item.metadata.session_id:
            self.feedback_by_session[item.metadata.session_id].append(item)

        # Update realtime metrics
        self.realtime_scores.append(item.score)
        self.score_history.append(item.score)
        self.timestamp_history.append(item.timestamp)

        # Update aggregates
        agg = self.target_aggregates[item.target]
        agg["total_score"] += item.score
        agg["count"] += 1
        agg["avg_score"] = agg["total_score"] / agg["count"]
        if item.feedback_type == FeedbackType.POSITIVE:
            agg["positive_count"] += 1
        elif item.feedback_type == FeedbackType.NEGATIVE:
            agg["negative_count"] += 1
        agg["last_updated"] = datetime.now()

        # Maintain window size
        if len(self.feedback_items) > self.window_size:
            old_item = self.feedback_items.pop(0)
            self._cleanup_old_item(old_item)

        # Auto-save if enabled
        if self.persistence_enabled and len(self.feedback_items) % 10 == 0:
            asyncio.create_task(self.save_feedback())

    def _cleanup_old_item(self, item: FeedbackItem):
        """Remove item from indexes during cleanup."""
        # Remove from by_target (keep most recent)
        target_items = self.feedback_by_target[item.target]
        if item in target_items:
            target_items.remove(item)

        # Remove from by_type
        type_items = self.feedback_by_type[item.feedback_type]
        if item in type_items:
            type_items.remove(item)

        # Remove from user/session if needed
        if item.metadata.user_id and item in self.feedback_by_user.get(
            item.metadata.user_id, []
        ):
            self.feedback_by_user[item.metadata.user_id].remove(item)
        if item.metadata.session_id and item in self.feedback_by_session.get(
            item.metadata.session_id, []
        ):
            self.feedback_by_session[item.metadata.session_id].remove(item)

    # #==================== Query Methods #====================

    def get_average_score(
        self,
        target: str,
        time_range_hours: Optional[int] = None,
        feedback_types: Optional[List[FeedbackType]] = None,
    ) -> float:
        """Get average feedback score for target with filters."""
        items = self.feedback_by_target.get(target, [])

        if time_range_hours:
            cutoff = datetime.now() - timedelta(hours=time_range_hours)
            items = [f for f in items if f.timestamp > cutoff]

        if feedback_types:
            items = [f for f in items if f.feedback_type in feedback_types]

        if not items:
            return 0.5

        scores = [f.score for f in items]
        return statistics.mean(scores)

    def get_weighted_average_score(self, target: str, decay_days: int = 30) -> float:
        """Get time-weighted average score (more recent feedback has higher weight)."""
        items = self.feedback_by_target.get(target, [])
        if not items:
            return 0.5

        now = datetime.now()
        total_weight = 0
        weighted_sum = 0

        for item in items:
            age_days = (now - item.timestamp).days
            weight = 1.0 / (1.0 + age_days / decay_days) if age_days > 0 else 1.0
            weighted_sum += item.score * weight * item.metadata.importance_weight
            total_weight += weight * item.metadata.importance_weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def get_target_feedback(
        self,
        target: str,
        limit: int = 10,
        min_score: Optional[float] = None,
        feedback_type: Optional[FeedbackType] = None,
    ) -> List[FeedbackItem]:
        """Get feedback for target with filters."""
        items = self.feedback_by_target.get(target, [])

        if min_score is not None:
            items = [f for f in items if f.score >= min_score]
        if feedback_type:
            items = [f for f in items if f.feedback_type == feedback_type]

        return items[-limit:]

    def get_recent_feedback(self, minutes: int = 60) -> List[FeedbackItem]:
        """Get feedback from last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [f for f in self.feedback_items if f.timestamp > cutoff]

    def get_high_priority_feedback(self) -> List[FeedbackItem]:
        """Get high and critical priority feedback."""
        return [
            f
            for f in self.feedback_items
            if f.priority
            in [
                FeedbackPriority.HIGH,
                FeedbackPriority.CRITICAL,
                FeedbackPriority.EMERGENCY,
            ]
        ]

    def get_positive_ratio(self, target: str) -> float:
        """Get ratio of positive to total feedback for target."""
        items = self.feedback_by_target.get(target, [])
        if not items:
            return 0.5

        positive_count = sum(
            1 for f in items if f.feedback_type == FeedbackType.POSITIVE
        )
        return positive_count / len(items)

    # #==================== Analytics #====================

    def analyze_feedback(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Analyze collected feedback with advanced metrics."""

        # Use cache if available
        if self._stats_cache and self._cache_timestamp:
            age_seconds = (datetime.now() - self._cache_timestamp).total_seconds()
            if age_seconds < self._cache_ttl_seconds:
                return self._stats_cache

        items = self.feedback_items
        if target:
            items = self.feedback_by_target.get(target, [])

        if not items:
            stats = self._get_empty_stats()
            self._cache_stats(stats)
            return stats

        scores = [f.score for f in items]
        positive_count = sum(
            1
            for f in items
            if f.feedback_type in [FeedbackType.POSITIVE, FeedbackType.PRAISE]
        )
        negative_count = sum(
            1
            for f in items
            if f.feedback_type
            in [FeedbackType.NEGATIVE, FeedbackType.COMPLAINT, FeedbackType.CRITICAL]
        )

        # Calculate confidence intervals if enough data
        confidence_intervals = {}
        if len(scores) >= 5 and self.enable_ml:
            confidence_intervals = self.analyzer.calculate_confidence_intervals(scores)

        # Analyze trends
        trends = {}
        if self.enable_analytics and len(items) >= 10:
            trends = self.analyzer.analyze_trends(items)

        # Analyze patterns
        patterns = {}
        if self.enable_analytics and len(items) >= 20:
            patterns = self.analyzer.analyze_patterns(items)

        stats = {
            "total_feedback": len(items),
            "avg_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "min_score": min(scores),
            "max_score": max(scores),
            "targets": len(set(f.target for f in items)),
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "neutral_feedback": len(items) - positive_count - negative_count,
            "positive_ratio": positive_count / len(items) if items else 0,
            "negative_ratio": negative_count / len(items) if items else 0,
            "weighted_avg_score": self._calculate_weighted_avg(items),
            "confidence_intervals": confidence_intervals,
            "trends": trends,
            "patterns": patterns,
            "feedback_by_type": {
                t.value: len(self.feedback_by_type[t]) for t in FeedbackType
            },
            "top_targets": self._get_top_targets(limit=10),
            "recent_rate": self._calculate_recent_rate(minutes=60),
            "improvement_score": self._calculate_improvement_score(),
        }

        self._cache_stats(stats)
        return stats

    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics."""
        return {
            "total_feedback": 0,
            "avg_score": 0.5,
            "median_score": 0.5,
            "std_dev": 0,
            "min_score": 0,
            "max_score": 0,
            "targets": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "neutral_feedback": 0,
            "positive_ratio": 0,
            "negative_ratio": 0,
            "weighted_avg_score": 0.5,
            "confidence_intervals": {},
            "trends": {},
            "patterns": {},
            "feedback_by_type": {},
            "top_targets": [],
            "recent_rate": 0,
            "improvement_score": 0,
        }

    def _calculate_weighted_avg(self, items: List[FeedbackItem]) -> float:
        """Calculate weighted average based on priority and impact."""
        if not items:
            return 0.5

        weighted_sum = sum(
            f.score * f.impact_score * f.metadata.importance_weight for f in items
        )
        total_weight = sum(f.impact_score * f.metadata.importance_weight for f in items)

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _get_top_targets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top targets by feedback volume."""
        targets = []
        for target, items in self.feedback_by_target.items():
            targets.append(
                {
                    "target": target,
                    "count": len(items),
                    "avg_score": statistics.mean([f.score for f in items]),
                    "positive_ratio": self.get_positive_ratio(target),
                }
            )

        targets.sort(key=lambda x: x["count"], reverse=True)
        return targets[:limit]

    def _calculate_recent_rate(self, minutes: int = 60) -> float:
        """Calculate feedback rate per hour."""
        recent = self.get_recent_feedback(minutes)
        return len(recent) / (minutes / 60) if minutes > 0 else 0

    def _calculate_improvement_score(self) -> float:
        """Calculate overall improvement score (-1 to 1)."""
        if len(self.score_history) < 20:
            return 0

        # Compare recent average with older average
        recent = self.score_history[-10:]
        older = self.score_history[-20:-10]

        if not older:
            return 0

        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)

        improvement = recent_avg - older_avg
        return max(-1.0, min(1.0, improvement))

    def _calculate_importance_weight(
        self, score: float, feedback_type: FeedbackType
    ) -> float:
        """Calculate importance weight for feedback."""
        base_weight = 1.0

        # Boost critical feedback
        if feedback_type in [FeedbackType.CRITICAL, FeedbackType.CORRECTIVE]:
            base_weight = 2.0
        elif feedback_type == FeedbackType.NEGATIVE:
            base_weight = 1.5
        elif feedback_type == FeedbackType.POSITIVE:
            base_weight = 0.8

        # Boost extreme scores
        if score <= 0.2 or score >= 0.9:
            base_weight *= 1.5

        return base_weight

    # #==================== Learning Signals #====================

    def get_learning_signals(
        self, target: str, min_confidence: float = 0.5
    ) -> List[LearningSignal]:
        """Get learning signals for a target based on feedback."""
        items = self.feedback_by_target.get(target, [])
        if not items:
            return []

        # Collect signals with confidence
        signal_confidence = defaultdict(float)

        for item in items:
            for signal in item.learning_signals:
                confidence = (
                    item.score * item.impact_score * item.metadata.importance_weight
                )
                signal_confidence[signal] = max(signal_confidence[signal], confidence)

        # Return signals above threshold
        return [
            signal
            for signal, confidence in signal_confidence.items()
            if confidence >= min_confidence
        ]

    def get_improvement_suggestions(
        self, threshold: float = 0.6, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get detailed improvement suggestions for areas needing improvement."""
        suggestions = []

        for target, items in self.feedback_by_target.items():
            avg_score = self.get_weighted_average_score(target)

            if avg_score < threshold:
                # Get specific issues from feedback comments
                issues = []
                for item in items:
                    if item.score < threshold and item.comment:
                        issues.append(
                            {
                                "comment": item.comment,
                                "score": item.score,
                                "feedback_type": item.feedback_type.value,
                            }
                        )

                suggestions.append(
                    {
                        "target": target,
                        "score": avg_score,
                        "gap": threshold - avg_score,
                        "feedback_count": len(items),
                        "positive_ratio": self.get_positive_ratio(target),
                        "priority": "high" if avg_score < 0.4 else "medium",
                        "issues": issues[:5],  # Top 5 issues
                    }
                )

        suggestions.sort(key=lambda x: x["gap"], reverse=True)
        return suggestions[:limit]

    def get_success_patterns(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """Get patterns that consistently lead to positive feedback."""
        patterns = []

        for target, items in self.feedback_by_target.items():
            positive_items = [
                f for f in items if f.feedback_type == FeedbackType.POSITIVE
            ]

            if len(positive_items) >= min_occurrences:
                avg_score = statistics.mean([f.score for f in positive_items])

                # Extract common tags from positive feedback
                common_tags = []
                tag_counts = defaultdict(int)
                for item in positive_items:
                    for tag in item.metadata.tags:
                        tag_counts[tag] += 1

                common_tags = [
                    tag for tag, count in tag_counts.items() if count >= min_occurrences
                ]

                patterns.append(
                    {
                        "target": target,
                        "success_rate": len(positive_items) / len(items),
                        "avg_positive_score": avg_score,
                        "occurrences": len(positive_items),
                        "common_tags": common_tags,
                    }
                )

        patterns.sort(key=lambda x: x["success_rate"], reverse=True)
        return patterns[:10]

    # #==================== Persistence #====================

    async def save_feedback(self, filename: Optional[str] = None) -> bool:
        """Save feedback to disk."""
        if not self.persistence_enabled:
            return False

        try:
            if not filename:
                filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            filepath = self.persistence_path / filename

            data = {
                "timestamp": datetime.now().isoformat(),
                "total_items": len(self.feedback_items),
                "window_size": self.window_size,
                "items": [item.to_dict() for item in self.feedback_items],
                "target_aggregates": dict(self.target_aggregates),
            }

            # Convert aggregates for JSON serialization
            for target, agg in data["target_aggregates"].items():
                agg["last_updated"] = (
                    agg["last_updated"].isoformat()
                    if isinstance(agg["last_updated"], datetime)
                    else agg["last_updated"]
                )

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(
                f"💾 Saved {len(self.feedback_items)} feedback items to {filepath}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            if self._error_callbacks:
                for callback in self._error_callbacks:
                    try:
                        callback(e)
                    except Exception:
                        pass
            return False

    async def load_feedback(self, filepath: Optional[str] = None) -> int:
        """Load feedback from disk."""
        if not self.persistence_enabled:
            return 0

        try:
            if not filepath:
                # Load most recent file
                files = sorted(
                    self.persistence_path.glob("feedback_*.json"), reverse=True
                )
                if not files:
                    return 0
                filepath = files[0]
            else:
                filepath = Path(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

            loaded_count = 0
            for item_data in data.get("items", []):
                item = FeedbackItem.from_dict(item_data)
                self._add_feedback_item(item)
                loaded_count += 1

            logger.info(f"📂 Loaded {loaded_count} feedback items from {filepath}")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
            return 0

    async def auto_save_loop(self):
        """Background task for auto-saving feedback."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.auto_save_interval)
            if self.persistence_enabled and self.feedback_items:
                await self.save_feedback()

    # #==================== Cleanup #====================

    def clear_old_feedback(self, days: int = 7) -> int:
        """Clear feedback older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.feedback_items)

        # Keep only recent items
        new_items = [f for f in self.feedback_items if f.timestamp > cutoff]

        # Rebuild all indexes
        self.feedback_items = new_items
        self._rebuild_indexes()

        removed = original_count - len(self.feedback_items)
        logger.info(f"🗑️ Cleared {removed} old feedback items (older than {days} days)")

        return removed

    def clear_expired_feedback(self) -> int:
        """Clear feedback that has expired based on metadata."""
        now = datetime.now()
        original_count = len(self.feedback_items)

        new_items = [
            f
            for f in self.feedback_items
            if not f.metadata.expires_at or f.metadata.expires_at > now
        ]

        self.feedback_items = new_items
        self._rebuild_indexes()

        removed = original_count - len(self.feedback_items)
        if removed > 0:
            logger.info(f"🗑️ Cleared {removed} expired feedback items")

        return removed

    def _rebuild_indexes(self):
        """Rebuild all feedback indexes."""
        self.feedback_by_target.clear()
        self.feedback_by_type.clear()
        self.feedback_by_user.clear()
        self.feedback_by_session.clear()

        for item in self.feedback_items:
            self.feedback_by_target[item.target].append(item)
            self.feedback_by_type[item.feedback_type].append(item)
            if item.metadata.user_id:
                self.feedback_by_user[item.metadata.user_id].append(item)
            if item.metadata.session_id:
                self.feedback_by_session[item.metadata.session_id].append(item)

    def reset(self):
        """Reset all feedback data."""
        self.feedback_items.clear()
        self.feedback_by_target.clear()
        self.feedback_by_type.clear()
        self.feedback_by_user.clear()
        self.feedback_by_session.clear()
        self.realtime_scores.clear()
        self.score_history.clear()
        self.timestamp_history.clear()
        self.target_aggregates.clear()
        self._invalidate_cache()

        logger.info("🔄 Feedback loop reset")

    # #==================== Callbacks #====================

    def add_callback(self, callback: Callable[[FeedbackItem], None]):
        """Add callback for new feedback."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[FeedbackItem], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback."""
        self._error_callbacks.append(callback)

    def _trigger_callbacks(self, item: FeedbackItem):
        """Trigger all callbacks."""
        for callback in self._callbacks:
            try:
                callback(item)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                for err_cb in self._error_callbacks:
                    try:
                        err_cb(e)
                    except Exception:
                        pass

    # #==================== Utility Methods #====================

    def _cache_stats(self, stats: Dict[str, Any]):
        """Cache statistics."""
        self._stats_cache = stats
        self._cache_timestamp = datetime.now()

    def _invalidate_cache(self):
        """Invalidate statistics cache."""
        self._stats_cache = {}
        self._cache_timestamp = None

    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary of feedback loop."""
        return {
            "total_feedback": len(self.feedback_items),
            "unique_targets": len(self.feedback_by_target),
            "recent_rate": self._calculate_recent_rate(),
            "avg_score_realtime": (
                statistics.mean(self.realtime_scores) if self.realtime_scores else 0.5
            ),
            "positive_ratio": self.analyze_feedback().get("positive_ratio", 0),
            "high_priority_count": len(self.get_high_priority_feedback()),
            "persistence_enabled": self.persistence_enabled,
            "ml_enabled": self.enable_ml,
        }

    def export_to_dataframe(self) -> Optional["pd.DataFrame"]:
        """Export feedback to pandas DataFrame for advanced analysis."""
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available for export")
            return None

        data = []
        for item in self.feedback_items:
            data.append(
                {
                    "id": item.id,
                    "target": item.target,
                    "score": item.score,
                    "type": item.feedback_type.value,
                    "priority": item.priority.value,
                    "comment": item.comment,
                    "source": item.source,
                    "sentiment": item.sentiment_score,
                    "impact": item.impact_score,
                    "timestamp": item.timestamp,
                    "user_id": item.metadata.user_id,
                    "session_id": item.metadata.session_id,
                    "tags": ",".join(item.metadata.tags),
                }
            )

        return pd.DataFrame(data)

    async def start(self):
        """Start background tasks."""
        self._running = True
        if self.persistence_enabled and self.auto_save_interval > 0:
            self._save_task = asyncio.create_task(self.auto_save_loop())
        logger.info("🚀 Feedback loop started")

    async def stop(self):
        """Stop background tasks."""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        # Final save
        if self.persistence_enabled and self.feedback_items:
            await self.save_feedback()

        logger.info("🛑 Feedback loop stopped")


# #==================== Convenience Functions #====================

__all__ = [
    "FeedbackLoop",
    "FeedbackItem",
    "FeedbackType",
    "FeedbackPriority",
    "LearningSignal",
    "FeedbackMetadata",
    "FeedbackAnalyzer",
    "FeedbackProcessor",
]
