"""Emotion Detector - detects and analyzes emotional content with multi-modal support, sentiment analysis, and emotional memory."""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from collections import deque, defaultdict

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from ..utils.logger import logger


class EmotionType(Enum):
    """Comprehensive emotion types based on psychological models."""

    # Primary emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

    # Secondary emotions
    LOVE = "love"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    FRUSTRATION = "frustration"
    ANXIETY = "anxiety"
    GUILT = "guilt"
    SHAME = "shame"
    PRIDE = "pride"
    JEALOUSY = "jealousy"
    HOPE = "hope"
    BOREDOM = "boredom"
    CURIOSITY = "curiosity"
    CALM = "calm"
    EXCITEMENT = "excitement"
    GRATITUDE = "gratitude"
    EMPATHY = "empathy"


class EmotionalValence(Enum):
    """Emotional valence categories."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class ArousalLevel(Enum):
    """Emotional arousal levels."""

    LOW = "low"  # Calm, relaxed
    MEDIUM = "medium"  # Alert, engaged
    HIGH = "high"  # Excited, stressed
    VERY_HIGH = "very_high"  # Ecstatic, panicked


@dataclass
class EmotionResult:
    """Comprehensive emotion analysis result."""

    emotions: Dict[EmotionType, float]
    dominant: EmotionType
    valence: EmotionalValence
    arousal: ArousalLevel
    intensity: float  # 0-1 overall emotional intensity
    confidence: float  # 0-1 confidence in detection
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emotions": {k.value: v for k, v in self.emotions.items()},
            "dominant": self.dominant.value,
            "valence": self.valence.value,
            "arousal": self.arousal.value,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }

    def get_top_emotions(self, n: int = 3) -> List[Tuple[EmotionType, float]]:
        """Get top N emotions by score."""
        sorted_emotions = sorted(
            self.emotions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_emotions[:n]

    def is_positive(self) -> bool:
        """Check if dominant emotion is positive."""
        return self.valence == EmotionalValence.POSITIVE

    def is_negative(self) -> bool:
        """Check if dominant emotion is negative."""
        return self.valence == EmotionalValence.NEGATIVE


class EmotionDetector:
    """Advanced emotion detector with multi-modal support and emotional intelligence."""

    # Emotion word lists for keyword-based detection
    EMOTION_KEYWORDS = {
        EmotionType.JOY: [
            "happy",
            "joy",
            "delighted",
            "pleased",
            "glad",
            "cheerful",
            "ecstatic",
            "elated",
            "joyful",
            "merry",
            "content",
            "thrilled",
            "exuberant",
            "radiant",
        ],
        EmotionType.SADNESS: [
            "sad",
            "depressed",
            "unhappy",
            "miserable",
            "gloomy",
            "heartbroken",
            "sorrowful",
            "dejected",
            "despair",
            "grief",
            "melancholy",
            "downcast",
        ],
        EmotionType.ANGER: [
            "angry",
            "furious",
            "rage",
            "mad",
            "irritated",
            "annoyed",
            "outraged",
            "frustrated",
            "hostile",
            "resentful",
            "enraged",
            "livid",
        ],
        EmotionType.FEAR: [
            "afraid",
            "scared",
            "fearful",
            "terrified",
            "anxious",
            "worried",
            "panicked",
            "horrified",
            "nervous",
            "apprehensive",
            "dread",
        ],
        EmotionType.SURPRISE: [
            "surprised",
            "astonished",
            "amazed",
            "shocked",
            "stunned",
            "startled",
            "dumbfounded",
            "speechless",
            "awestruck",
        ],
        EmotionType.DISGUST: [
            "disgusted",
            "revolted",
            "repulsed",
            "sickened",
            "nauseated",
            "offended",
            "appalled",
            "horrified",
        ],
        EmotionType.LOVE: [
            "love",
            "adore",
            "cherish",
            "affection",
            "fond",
            "devotion",
            "romance",
            "passion",
            "tenderness",
            "warmth",
        ],
        EmotionType.TRUST: [
            "trust",
            "believe",
            "rely",
            "confident",
            "faith",
            "depend",
            "reliable",
            "honest",
            "loyal",
            "secure",
        ],
        EmotionType.ANTICIPATION: [
            "anticipate",
            "expect",
            "await",
            "look forward",
            "hopeful",
            "eager",
            "excited",
            "optimistic",
            "enthusiastic",
        ],
        EmotionType.FRUSTRATION: [
            "frustrated",
            "exasperated",
            "fed up",
            "drained",
            "stuck",
            "blocked",
            "impeded",
            "thwarted",
            "defeated",
        ],
        EmotionType.ANXIETY: [
            "anxious",
            "nervous",
            "worried",
            "uneasy",
            "tense",
            "stressed",
            "distressed",
            "apprehensive",
            "restless",
        ],
        EmotionType.GUILT: [
            "guilty",
            "ashamed",
            "remorseful",
            "contrite",
            "sorry",
            "regretful",
            "penitent",
            "culpable",
        ],
        EmotionType.PRIDE: [
            "proud",
            "accomplished",
            "achieved",
            "successful",
            "confident",
            "dignified",
            "honored",
            "self-respect",
        ],
        EmotionType.HOPE: [
            "hopeful",
            "optimistic",
            "positive",
            "encouraged",
            "inspired",
            "aspiring",
            "promising",
            "bright",
        ],
        EmotionType.CALM: [
            "calm",
            "relaxed",
            "peaceful",
            "serene",
            "tranquil",
            "composed",
            "centered",
            "balanced",
            "mindful",
        ],
        EmotionType.EXCITEMENT: [
            "excited",
            "thrilled",
            "enthusiastic",
            "eager",
            "energetic",
            "animated",
            "vivacious",
            "passionate",
        ],
        EmotionType.GRATITUDE: [
            "grateful",
            "thankful",
            "appreciative",
            "blessed",
            "touched",
            "indebted",
            "obliged",
        ],
    }

    # Valence mapping
    EMOTION_VALENCE = {
        EmotionType.JOY: EmotionalValence.POSITIVE,
        EmotionType.LOVE: EmotionalValence.POSITIVE,
        EmotionType.TRUST: EmotionalValence.POSITIVE,
        EmotionType.ANTICIPATION: EmotionalValence.POSITIVE,
        EmotionType.PRIDE: EmotionalValence.POSITIVE,
        EmotionType.HOPE: EmotionalValence.POSITIVE,
        EmotionType.CALM: EmotionalValence.POSITIVE,
        EmotionType.EXCITEMENT: EmotionalValence.POSITIVE,
        EmotionType.GRATITUDE: EmotionalValence.POSITIVE,
        EmotionType.SADNESS: EmotionalValence.NEGATIVE,
        EmotionType.ANGER: EmotionalValence.NEGATIVE,
        EmotionType.FEAR: EmotionalValence.NEGATIVE,
        EmotionType.DISGUST: EmotionalValence.NEGATIVE,
        EmotionType.FRUSTRATION: EmotionalValence.NEGATIVE,
        EmotionType.ANXIETY: EmotionalValence.NEGATIVE,
        EmotionType.GUILT: EmotionalValence.NEGATIVE,
        EmotionType.SHAME: EmotionalValence.NEGATIVE,
        EmotionType.JEALOUSY: EmotionalValence.NEGATIVE,
        EmotionType.BOREDOM: EmotionalValence.NEGATIVE,
        EmotionType.SURPRISE: EmotionalValence.MIXED,
        EmotionType.NEUTRAL: EmotionalValence.NEUTRAL,
        EmotionType.CURIOSITY: EmotionalValence.POSITIVE,
        EmotionType.EMPATHY: EmotionalValence.POSITIVE,
    }

    # Arousal mapping
    EMOTION_AROUSAL = {
        EmotionType.JOY: ArousalLevel.MEDIUM,
        EmotionType.SADNESS: ArousalLevel.LOW,
        EmotionType.ANGER: ArousalLevel.HIGH,
        EmotionType.FEAR: ArousalLevel.VERY_HIGH,
        EmotionType.SURPRISE: ArousalLevel.HIGH,
        EmotionType.DISGUST: ArousalLevel.MEDIUM,
        EmotionType.LOVE: ArousalLevel.MEDIUM,
        EmotionType.TRUST: ArousalLevel.LOW,
        EmotionType.ANTICIPATION: ArousalLevel.MEDIUM,
        EmotionType.FRUSTRATION: ArousalLevel.HIGH,
        EmotionType.ANXIETY: ArousalLevel.HIGH,
        EmotionType.GUILT: ArousalLevel.MEDIUM,
        EmotionType.PRIDE: ArousalLevel.MEDIUM,
        EmotionType.HOPE: ArousalLevel.MEDIUM,
        EmotionType.CALM: ArousalLevel.LOW,
        EmotionType.EXCITEMENT: ArousalLevel.HIGH,
        EmotionType.GRATITUDE: ArousalLevel.LOW,
        EmotionType.NEUTRAL: ArousalLevel.LOW,
        EmotionType.CURIOSITY: ArousalLevel.MEDIUM,
        EmotionType.EMPATHY: ArousalLevel.MEDIUM,
    }

    def __init__(
        self,
        use_advanced_nlp: bool = True,
        enable_emotional_memory: bool = True,
        history_size: int = 1000,
        intensity_threshold: float = 0.3,
        context_window: int = 5,
        enable_multi_modal: bool = True,
    ):
        """Initialize advanced emotion detector."""
        self.use_advanced_nlp = use_advanced_nlp and TEXTBLOB_AVAILABLE
        self.enable_emotional_memory = enable_emotional_memory
        self.history_size = history_size
        self.intensity_threshold = intensity_threshold
        self.context_window = context_window
        self.enable_multi_modal = enable_multi_modal

        # Storage
        self.detection_history: deque = deque(maxlen=history_size)
        self.emotion_memory: Dict[str, List[EmotionResult]] = defaultdict(list)
        self.emotion_trends: Dict[EmotionType, List[float]] = defaultdict(list)
        self.context_buffer: deque = deque(maxlen=context_window)

        # Statistics
        self.detection_count = 0
        self.emotion_frequencies: Dict[EmotionType, int] = defaultdict(int)
        self.avg_intensity: Dict[EmotionType, float] = defaultdict(float)

        # Emotional state tracking
        self.current_emotional_state: Optional[EmotionResult] = None
        self.emotional_history: List[EmotionResult] = []

        # Callbacks
        self._emotion_callbacks: List = []

        # NLP resources
        if self.use_advanced_nlp:
            logger.info("✅ Advanced NLP emotion detection enabled (TextBlob)")

        logger.info(
            f"✅ EmotionDetector initialized (advanced_nlp={self.use_advanced_nlp}, emotional_memory={enable_emotional_memory})"
        )

    def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.0,
    ) -> EmotionResult:
        """Detect emotions in text with advanced analysis.

        Args:
            text: Input text to analyze
            context: Additional context (speaker, situation, etc.)
            confidence_threshold: Minimum confidence to include emotions

        Returns:
            EmotionResult with comprehensive analysis
        """

        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in EmotionType}

        # 1. Keyword-based detection
        keyword_scores = self._detect_by_keywords(text)
        for emotion, score in keyword_scores.items():
            emotion_scores[emotion] += score * 0.6  # 60% weight

        # 2. Sentiment-based detection (if available)
        sentiment_scores = self._detect_by_sentiment(text)
        for emotion, score in sentiment_scores.items():
            emotion_scores[emotion] += score * 0.3  # 30% weight

        # 3. Advanced NLP (if available)
        if self.use_advanced_nlp:
            nlp_scores = self._detect_by_nlp(text)
            for emotion, score in nlp_scores.items():
                emotion_scores[emotion] += score * 0.2  # 20% weight

        # 4. Context-based adjustment
        if context:
            self._apply_context_adjustment(emotion_scores, context)

        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_score

        # Apply threshold
        if confidence_threshold > 0:
            for emotion in list(emotion_scores.keys()):
                if emotion_scores[emotion] < confidence_threshold:
                    emotion_scores[emotion] = 0.0

        # Get dominant emotion
        dominant = max(emotion_scores.items(), key=lambda x: x[1])
        dominant_emotion = dominant[0] if dominant[1] > 0 else EmotionType.NEUTRAL

        # Determine valence and arousal
        valence = self._calculate_valence(emotion_scores)
        arousal = self._calculate_arousal(emotion_scores)

        # Calculate overall intensity
        intensity = self._calculate_intensity(emotion_scores)

        # Calculate confidence
        confidence = self._calculate_confidence(emotion_scores, text)

        # Create result
        result = EmotionResult(
            emotions=emotion_scores,
            dominant=dominant_emotion,
            valence=valence,
            arousal=arousal,
            intensity=intensity,
            confidence=confidence,
            context=context or {},
        )

        # Update history and statistics
        self._update_history(result, text)
        self._update_statistics(result)

        # Update emotional memory
        if self.enable_emotional_memory:
            self._update_emotional_memory(result, context)

        # Update current state
        self.current_emotional_state = result
        self.emotional_history.append(result)

        # Trigger callbacks
        self._trigger_callbacks(result)

        self.detection_count += 1

        logger.debug(
            f"Emotion detected: {dominant_emotion.value} (intensity={intensity:.2f}, confidence={confidence:.2f})"
        )

        return result

    def _detect_by_keywords(self, text: str) -> Dict[EmotionType, float]:
        """Detect emotions using keyword matching."""
        scores = defaultdict(float)
        text_lower = text.lower()

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = 0
            for keyword in keywords:
                if keyword in text_lower:
                    count += text_lower.count(keyword)

            if count > 0:
                # Score based on keyword frequency, capped at 1.0
                scores[emotion] = min(1.0, count / 5)

        return dict(scores)

    def _detect_by_sentiment(self, text: str) -> Dict[EmotionType, float]:
        """Detect emotions based on sentiment analysis."""
        scores = {}

        # Simple sentiment detection
        positive_words = [
            "good",
            "great",
            "excellent",
            "awesome",
            "wonderful",
            "fantastic",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "poor",
            "disappointing",
        ]

        text_lower = text.lower()
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        if positive_count > negative_count:
            scores[EmotionType.JOY] = min(1.0, positive_count / 3)
            scores[EmotionType.LOVE] = min(1.0, positive_count / 5)
        elif negative_count > positive_count:
            scores[EmotionType.SADNESS] = min(1.0, negative_count / 3)
            scores[EmotionType.ANGER] = min(1.0, negative_count / 4)

        return scores

    def _detect_by_nlp(self, text: str) -> Dict[EmotionType, float]:
        """Detect emotions using NLP (TextBlob)."""
        scores = {}

        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            sentiment = blob.sentiment

            # Map sentiment to emotions
            if sentiment.polarity > 0.3:
                scores[EmotionType.JOY] = sentiment.polarity
                scores[EmotionType.LOVE] = sentiment.polarity * 0.7
            elif sentiment.polarity < -0.3:
                scores[EmotionType.SADNESS] = abs(sentiment.polarity)
                scores[EmotionType.ANGER] = abs(sentiment.polarity) * 0.8

            # Subjectivity affects confidence
            if sentiment.subjectivity > 0.6:
                scores[EmotionType.SURPRISE] = sentiment.subjectivity * 0.5

        return scores

    def _apply_context_adjustment(
        self, scores: Dict[EmotionType, float], context: Dict[str, Any]
    ):
        """Adjust emotion scores based on context."""

        # Speaker influence
        if "speaker" in context:
            speaker = context["speaker"]
            if speaker in self.emotion_memory:
                recent_emotions = self.emotion_memory[speaker][-5:]
                for result in recent_emotions:
                    for emotion, score in result.emotions.items():
                        if score > 0.3:
                            scores[emotion] = scores.get(emotion, 0) * 1.1

        # Situation influence
        if "situation" in context:
            situation = context["situation"]
            if situation == "celebration":
                scores[EmotionType.JOY] = scores.get(EmotionType.JOY, 0) * 1.3
            elif situation == "crisis":
                scores[EmotionType.FEAR] = scores.get(EmotionType.FEAR, 0) * 1.3
                scores[EmotionType.ANXIETY] = scores.get(EmotionType.ANXIETY, 0) * 1.2

        # Time of day influence
        if "timestamp" in context:
            hour = (
                context["timestamp"].hour
                if hasattr(context["timestamp"], "hour")
                else datetime.now().hour
            )
            if 22 <= hour or hour <= 5:
                scores[EmotionType.CALM] = scores.get(EmotionType.CALM, 0) * 1.2
                scores[EmotionType.FATIGUE] = scores.get(EmotionType.FATIGUE, 0) * 1.1

    def _calculate_valence(self, scores: Dict[EmotionType, float]) -> EmotionalValence:
        """Calculate emotional valence from scores."""
        positive_score = 0
        negative_score = 0

        for emotion, score in scores.items():
            valence = self.EMOTION_VALENCE.get(emotion, EmotionalValence.NEUTRAL)
            if valence == EmotionalValence.POSITIVE:
                positive_score += score
            elif valence == EmotionalValence.NEGATIVE:
                negative_score += score

        if positive_score > negative_score * 1.5:
            return EmotionalValence.POSITIVE
        elif negative_score > positive_score * 1.5:
            return EmotionalValence.NEGATIVE
        elif positive_score > 0 and negative_score > 0:
            return EmotionalValence.MIXED
        else:
            return EmotionalValence.NEUTRAL

    def _calculate_arousal(self, scores: Dict[EmotionType, float]) -> ArousalLevel:
        """Calculate emotional arousal level."""
        arousal_score = 0

        for emotion, score in scores.items():
            arousal = self.EMOTION_AROUSAL.get(emotion, ArousalLevel.MEDIUM)
            arousal_values = {
                ArousalLevel.LOW: 0.2,
                ArousalLevel.MEDIUM: 0.5,
                ArousalLevel.HIGH: 0.8,
                ArousalLevel.VERY_HIGH: 1.0,
            }
            arousal_score += score * arousal_values.get(arousal, 0.5)

        if arousal_score > 0.7:
            return ArousalLevel.VERY_HIGH
        elif arousal_score > 0.4:
            return ArousalLevel.HIGH
        elif arousal_score > 0.2:
            return ArousalLevel.MEDIUM
        else:
            return ArousalLevel.LOW

    def _calculate_intensity(self, scores: Dict[EmotionType, float]) -> float:
        """Calculate overall emotional intensity."""
        # Intensity is the maximum emotion score
        max_score = max(scores.values()) if scores else 0

        # Also consider number of active emotions
        active_emotions = sum(1 for s in scores.values() if s > 0.1)
        intensity_boost = min(0.3, active_emotions * 0.05)

        return min(1.0, max_score + intensity_boost)

    def _calculate_confidence(
        self, scores: Dict[EmotionType, float], text: str
    ) -> float:
        """Calculate confidence in emotion detection."""
        confidence = 0.5  # Base confidence

        # Higher confidence if there's a clear dominant emotion
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > sorted_scores[1] * 1.5:
            confidence += 0.2

        # Longer text gives more confidence
        text_length = len(text)
        if text_length > 100:
            confidence += 0.15
        elif text_length > 50:
            confidence += 0.1
        elif text_length > 20:
            confidence += 0.05

        # Multiple detection methods agree
        if self.use_advanced_nlp and text_length > 20:
            confidence += 0.1

        return min(1.0, confidence)

    def _update_history(self, result: EmotionResult, text: str):
        """Update detection history."""
        self.detection_history.append(
            {
                "text": text[:200],  # Truncate for storage
                "result": result,
                "timestamp": datetime.now(),
            }
        )

    def _update_statistics(self, result: EmotionResult):
        """Update detection statistics."""
        self.emotion_frequencies[result.dominant] += 1

        for emotion, score in result.emotions.items():
            if score > 0:
                # Update moving average
                old_avg = self.avg_intensity.get(emotion, 0)
                count = self.emotion_frequencies.get(emotion, 1)
                self.avg_intensity[emotion] = (old_avg * (count - 1) + score) / count
                self.emotion_trends[emotion].append(score)

                # Keep limited history
                if len(self.emotion_trends[emotion]) > 100:
                    self.emotion_trends[emotion] = self.emotion_trends[emotion][-100:]

    def _update_emotional_memory(
        self, result: EmotionResult, context: Optional[Dict[str, Any]]
    ):
        """Update emotional memory for users/sessions."""
        if context and "speaker" in context:
            speaker = context["speaker"]
            self.emotion_memory[speaker].append(result)

            # Keep limited history per speaker
            if len(self.emotion_memory[speaker]) > self.history_size:
                self.emotion_memory[speaker] = self.emotion_memory[speaker][
                    -self.history_size :
                ]

    def _trigger_callbacks(self, result: EmotionResult):
        """Trigger all emotion callbacks."""
        for callback in self._emotion_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Emotion callback error: {e}")

    # #================#================#============#=============
    # MULTI-MODAL DETECTION
    # #================#================#============#=============

    def detect_from_sentiment(self, sentiment_score: float) -> EmotionResult:
        """Detect emotions from sentiment score (-1 to 1)."""
        emotion_scores = {emotion: 0.0 for emotion in EmotionType}

        if sentiment_score > 0.5:
            emotion_scores[EmotionType.JOY] = sentiment_score
            emotion_scores[EmotionType.LOVE] = sentiment_score * 0.7
        elif sentiment_score > 0.2:
            emotion_scores[EmotionType.CALM] = sentiment_score
            emotion_scores[EmotionType.TRUST] = sentiment_score * 0.8
        elif sentiment_score < -0.5:
            emotion_scores[EmotionType.SADNESS] = abs(sentiment_score)
            emotion_scores[EmotionType.ANGER] = abs(sentiment_score) * 0.6
        elif sentiment_score < -0.2:
            emotion_scores[EmotionType.FRUSTRATION] = abs(sentiment_score)
            emotion_scores[EmotionType.ANXIETY] = abs(sentiment_score) * 0.7
        else:
            emotion_scores[EmotionType.NEUTRAL] = 1.0

        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total

        dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]

        return EmotionResult(
            emotions=emotion_scores,
            dominant=dominant,
            valence=self._calculate_valence(emotion_scores),
            arousal=self._calculate_arousal(emotion_scores),
            intensity=abs(sentiment_score),
            confidence=0.8,
        )

    def detect_from_audio_features(
        self, pitch: float, energy: float, speech_rate: float
    ) -> EmotionResult:
        """Detect emotions from audio features."""
        emotion_scores = {emotion: 0.0 for emotion in EmotionType}

        # Pitch analysis
        if pitch > 200:  # High pitch
            emotion_scores[EmotionType.EXCITEMENT] += 0.4
            emotion_scores[EmotionType.FEAR] += 0.3
        elif pitch < 100:  # Low pitch
            emotion_scores[EmotionType.SADNESS] += 0.4
            emotion_scores[EmotionType.CALM] += 0.3

        # Energy analysis
        if energy > 0.7:
            emotion_scores[EmotionType.ANGER] += 0.4
            emotion_scores[EmotionType.EXCITEMENT] += 0.3
        elif energy < 0.3:
            emotion_scores[EmotionType.SADNESS] += 0.4
            emotion_scores[EmotionType.TIRED] += 0.3

        # Speech rate analysis
        if speech_rate > 5:  # Fast speech
            emotion_scores[EmotionType.ANXIETY] += 0.3
            emotion_scores[EmotionType.EXCITEMENT] += 0.3
        elif speech_rate < 2:  # Slow speech
            emotion_scores[EmotionType.SADNESS] += 0.3
            emotion_scores[EmotionType.CALM] += 0.2

        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total

        dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]

        return EmotionResult(
            emotions=emotion_scores,
            dominant=dominant,
            valence=self._calculate_valence(emotion_scores),
            arousal=self._calculate_arousal(emotion_scores),
            intensity=max(emotion_scores.values()),
            confidence=0.7,
            context={"source": "audio_features"},
        )

    def fuse_modalities(
        self,
        text_result: Optional[EmotionResult],
        audio_result: Optional[EmotionResult],
        weights: Dict[str, float] = None,
    ) -> EmotionResult:
        """Fuse emotion results from multiple modalities."""
        if not weights:
            weights = {"text": 0.5, "audio": 0.5}

        fused_scores = {emotion: 0.0 for emotion in EmotionType}
        total_weight = 0

        if text_result:
            for emotion, score in text_result.emotions.items():
                fused_scores[emotion] += score * weights.get("text", 0.5)
            total_weight += weights.get("text", 0.5)

        if audio_result:
            for emotion, score in audio_result.emotions.items():
                fused_scores[emotion] += score * weights.get("audio", 0.5)
            total_weight += weights.get("audio", 0.5)

        # Normalize
        if total_weight > 0:
            for emotion in fused_scores:
                fused_scores[emotion] /= total_weight

        dominant = max(fused_scores.items(), key=lambda x: x[1])[0]

        return EmotionResult(
            emotions=fused_scores,
            dominant=dominant,
            valence=self._calculate_valence(fused_scores),
            arousal=self._calculate_arousal(fused_scores),
            intensity=max(fused_scores.values()),
            confidence=(text_result.confidence if text_result else 0) * 0.5
            + (audio_result.confidence if audio_result else 0) * 0.5,
            context={"fusion": True},
        )

    # #================#================#============#=============
    # EMOTIONAL INTELLIGENCE
    # #================#================#============#=============

    def get_emotional_trend(
        self, emotion: EmotionType, window: int = 10
    ) -> Dict[str, Any]:
        """Get trend for a specific emotion over time."""
        scores = self.emotion_trends.get(emotion, [])[-window:]

        if not scores:
            return {"trend": "stable", "direction": 0, "volatility": 0}

        if len(scores) >= 3:
            # Calculate trend using simple linear regression
            x = list(range(len(scores)))
            n = len(x)
            slope = (
                n * sum(x[i] * scores[i] for i in range(n)) - sum(x) * sum(scores)
            ) / (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)

            if slope > 0.05:
                trend = "increasing"
            elif slope < -0.05:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
            slope = 0

        # Calculate volatility (standard deviation)
        volatility = statistics.stdev(scores) if len(scores) > 1 else 0

        return {
            "trend": trend,
            "direction": slope,
            "volatility": volatility,
            "current": scores[-1] if scores else 0,
            "average": statistics.mean(scores) if scores else 0,
        }

    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state summary."""
        if not self.emotional_history:
            return {"state": "unknown", "intensity": 0}

        recent = self.emotional_history[-10:]

        # Calculate dominant emotion in recent history
        emotion_counts = defaultdict(int)
        for result in recent:
            emotion_counts[result.dominant] += 1

        dominant = (
            max(emotion_counts.items(), key=lambda x: x[1])[0]
            if emotion_counts
            else EmotionType.NEUTRAL
        )

        # Calculate average intensity
        avg_intensity = statistics.mean([r.intensity for r in recent])

        # Detect emotional shift
        if len(self.emotional_history) > 20:
            old_emotions = self.emotional_history[-20:-10]
            new_emotions = self.emotional_history[-10:]

            old_dominant = max(
                [r.dominant for r in old_emotions], key=lambda x: emotion_counts[x]
            )
            new_dominant = max(
                [r.dominant for r in new_emotions], key=lambda x: emotion_counts[x]
            )

            shift_detected = old_dominant != new_dominant
        else:
            shift_detected = False

        return {
            "current_dominant": dominant.value,
            "intensity": avg_intensity,
            "valence": (
                self.current_emotional_state.valence.value
                if self.current_emotional_state
                else "neutral"
            ),
            "arousal": (
                self.current_emotional_state.arousal.value
                if self.current_emotional_state
                else "medium"
            ),
            "emotional_shift": shift_detected,
            "recent_emotions": [r.dominant.value for r in recent[-5:]],
        }

    def analyze_conversation(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze emotional flow in a conversation."""
        results = []
        for text in texts:
            result = self.detect(text)
            results.append(result)

        # Calculate metrics
        emotion_sequence = [r.dominant for r in results]
        intensity_sequence = [r.intensity for r in results]

        # Detect emotional peaks
        peaks = []
        for i in range(1, len(intensity_sequence) - 1):
            if (
                intensity_sequence[i] > intensity_sequence[i - 1]
                and intensity_sequence[i] > intensity_sequence[i + 1]
            ):
                peaks.append((i, intensity_sequence[i]))

        # Calculate emotional diversity
        unique_emotions = len(set(emotion_sequence))

        # Detect sentiment shifts
        shifts = []
        for i in range(1, len(results)):
            if results[i].valence != results[i - 1].valence:
                shifts.append(i)

        return {
            "total_utterances": len(texts),
            "emotion_sequence": [e.value for e in emotion_sequence],
            "average_intensity": (
                statistics.mean(intensity_sequence) if intensity_sequence else 0
            ),
            "emotional_peaks": len(peaks),
            "peak_intensity": max(intensity_sequence) if intensity_sequence else 0,
            "emotional_diversity": unique_emotions,
            "valence_shifts": len(shifts),
            "dominant_emotion": (
                max(set(emotion_sequence), key=emotion_sequence.count)
                if emotion_sequence
                else EmotionType.NEUTRAL
            ),
            "emotional_stability": 1.0
            - (
                statistics.stdev(intensity_sequence)
                if len(intensity_sequence) > 1
                else 0
            ),
        }

    # #================#================#============#=============
    # CALLBACKS
    # #================#================#============#=============

    def add_callback(self, callback):
        """Add callback for emotion detection events."""
        self._emotion_callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove callback."""
        if callback in self._emotion_callbacks:
            self._emotion_callbacks.remove(callback)

    # #================#================#============#=============
    # UTILITIES
    # #================#================#============#=============

    def get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """Get dominant emotion from emotion dictionary."""
        return max(emotions, key=emotions.get) if emotions else "neutral"

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive detector statistics."""
        return {
            "detections": self.detection_count,
            "history_size": len(self.detection_history),
            "emotion_frequencies": {
                k.value: v for k, v in self.emotion_frequencies.items()
            },
            "avg_intensities": {k.value: v for k, v in self.avg_intensity.items()},
            "current_state": self.get_emotional_state(),
            "unique_speakers": len(self.emotion_memory),
            "trends": {
                e.value: self.get_emotional_trend(e) for e in list(EmotionType)[:5]
            },  # Top 5 emotions
            "advanced_nlp_enabled": self.use_advanced_nlp,
            "emotional_memory_enabled": self.enable_emotional_memory,
            "multi_modal_enabled": self.enable_multi_modal,
        }

    def get_emotional_summary(self, speaker: Optional[str] = None) -> Dict[str, Any]:
        """Get emotional summary for a speaker or overall."""
        if speaker and speaker in self.emotion_memory:
            history = self.emotion_memory[speaker]
        else:
            history = self.emotional_history

        if not history:
            return {"message": "No emotional history available"}

        recent = history[-50:]  # Last 50 interactions

        # Calculate primary emotions
        emotion_counts = defaultdict(int)
        for result in recent:
            emotion_counts[result.dominant] += 1

        primary_emotions = sorted(
            [{"emotion": k.value, "count": v} for k, v in emotion_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        # Calculate average metrics
        avg_intensity = statistics.mean([r.intensity for r in recent])
        avg_confidence = statistics.mean([r.confidence for r in recent])

        # Valence distribution
        valence_counts = defaultdict(int)
        for result in recent:
            valence_counts[result.valence] += 1

        return {
            "total_interactions": len(history),
            "recent_interactions": len(recent),
            "primary_emotions": primary_emotions,
            "average_intensity": avg_intensity,
            "average_confidence": avg_confidence,
            "valence_distribution": {k.value: v for k, v in valence_counts.items()},
            "most_common_valence": (
                max(valence_counts.items(), key=lambda x: x[1])[0].value
                if valence_counts
                else "neutral"
            ),
        }

    def reset(self):
        """Reset all detection history and statistics."""
        self.detection_history.clear()
        self.emotion_memory.clear()
        self.emotion_trends.clear()
        self.emotional_history.clear()
        self.emotion_frequencies.clear()
        self.avg_intensity.clear()
        self.detection_count = 0
        self.current_emotional_state = None

        logger.info("🔄 EmotionDetector reset")

    def initialize(self, *args, **kwargs) -> bool:
        """
        Initialize resources required by the emotion detector.
        Safe to call multiple times.
        """
        try:
            self._initialized = True

            # Validate dependencies
            if self.use_advanced_nlp and not TEXTBLOB_AVAILABLE:
                logger.warning("⚠ TextBlob not available, disabling advanced NLP")
                self.use_advanced_nlp = False

            logger.info("🧠 EmotionDetector initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ EmotionDetector initialize failed: {e}")
            self._initialized = False
            return False

    def start(self) -> bool:
        """
        Start the emotion detector (activate runtime behavior).
        """
        try:
            if not getattr(self, "_initialized", False):
                self.initialize()

            self._running = True

            logger.info("🚀 EmotionDetector started")
            return True

        except Exception as e:
            logger.error(f"❌ EmotionDetector start failed: {e}")
            self._running = False
            return False

    def stop(self) -> bool:
        """
        Stop the emotion detector and release runtime resources.
        """
        try:
            self._running = False

            logger.info("🛑 EmotionDetector stopped")
            return True

        except Exception as e:
            logger.error(f"❌ EmotionDetector stop failed: {e}")
            return False


# #================#================#============#=============
# CONVENIENCE FUNCTIONS
# #================#================#============#=============


def get_emotion_color(emotion: EmotionType) -> str:
    """Get color associated with an emotion."""
    colors = {
        EmotionType.JOY: "#FFD700",  # Gold
        EmotionType.SADNESS: "#4682B4",  # Steel Blue
        EmotionType.ANGER: "#DC143C",  # Crimson
        EmotionType.FEAR: "#8B008B",  # Dark Magenta
        EmotionType.SURPRISE: "#FF8C00",  # Dark Orange
        EmotionType.DISGUST: "#556B2F",  # Dark Olive Green
        EmotionType.LOVE: "#FF69B4",  # Hot Pink
        EmotionType.TRUST: "#2E8B57",  # Sea Green
        EmotionType.CALM: "#87CEEB",  # Sky Blue
        EmotionType.EXCITEMENT: "#FF4500",  # Orange Red
    }
    return colors.get(emotion, "#808080")  # Gray default


def get_emotion_emoji(emotion: EmotionType) -> str:
    """Get emoji associated with an emotion."""
    emojis = {
        EmotionType.JOY: "😊",
        EmotionType.SADNESS: "😢",
        EmotionType.ANGER: "😠",
        EmotionType.FEAR: "😨",
        EmotionType.SURPRISE: "😲",
        EmotionType.DISGUST: "🤢",
        EmotionType.LOVE: "❤️",
        EmotionType.TRUST: "🤝",
        EmotionType.CALM: "😌",
        EmotionType.EXCITEMENT: "🤩",
        EmotionType.FRUSTRATION: "😤",
        EmotionType.ANXIETY: "😰",
        EmotionType.PRIDE: "🦚",
        EmotionType.HOPE: "🌈",
        EmotionType.GRATITUDE: "🙏",
    }
    return emojis.get(emotion, "😐")


__all__ = [
    "EmotionDetector",
    "EmotionType",
    "EmotionalValence",
    "ArousalLevel",
    "EmotionResult",
    "get_emotion_color",
    "get_emotion_emoji",
]
