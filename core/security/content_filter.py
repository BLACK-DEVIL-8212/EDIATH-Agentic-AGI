"""
Content Filter - Production Version
✔ Safe filtering
✔ Emotion-aware handling
✔ Frustration detection (controlled)
✔ Analytics + logging
✔ Extensible
"""

from typing import Dict, List, Any, Tuple
from enum import Enum

import re
import time
import hashlib


# ------------------------
# CONTENT TYPES
# ------------------------
class ContentType(Enum):
    PROFANITY = "profanity"
    VIOLENCE = "violence"
    ABUSE = "abuse"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual"
    MANIPULATION = "manipulation"
    FRUSTRATION = "frustration"

    # ------------------------
    # 🔥 FROM STRING (SMART PARSE)
    # ------------------------
    @classmethod
    def from_string(cls, value: str):
        try:
            if not value:
                return None

            value = str(value).lower().strip()

            mapping = {
                "profanity": cls.PROFANITY,
                "violence": cls.VIOLENCE,
                "abuse": cls.ABUSE,
                "harassment": cls.HARASSMENT,
                "hate": cls.HATE_SPEECH,
                "hate_speech": cls.HATE_SPEECH,
                "sexual": cls.SEXUAL,
                "manipulation": cls.MANIPULATION,
                "frustration": cls.FRUSTRATION,
            }

            if value in mapping:
                return mapping[value]

            for item in cls:
                if value in item.value:
                    return item

            return None

        except Exception:
            return None

    # ------------------------
    # 🔥 PRIORITY (IMPACT LEVEL)
    # ------------------------
    def priority(self) -> int:
        if not hasattr(ContentType, "_priority_map"):
            ContentType._priority_map = {
                ContentType.PROFANITY: 1,
                ContentType.FRUSTRATION: 1,
                ContentType.ABUSE: 3,
                ContentType.HARASSMENT: 3,
                ContentType.MANIPULATION: 4,
                ContentType.HATE_SPEECH: 5,
                ContentType.VIOLENCE: 6,
                ContentType.SEXUAL: 4,
            }
        return ContentType._priority_map.get(self, 1)

    # ------------------------
    # 🔥 SEVERITY SCORE
    # ------------------------
    def severity(self) -> float:
        if not hasattr(ContentType, "_severity_map"):
            ContentType._severity_map = {
                ContentType.PROFANITY: 0.2,
                ContentType.FRUSTRATION: 0.3,
                ContentType.ABUSE: 0.6,
                ContentType.HARASSMENT: 0.7,
                ContentType.MANIPULATION: 0.75,
                ContentType.HATE_SPEECH: 0.85,
                ContentType.VIOLENCE: 0.9,
                ContentType.SEXUAL: 0.8,
            }
        return ContentType._severity_map.get(self, 0.1)

    # ------------------------
    # 🔥 IS HARMFUL
    # ------------------------
    def is_harmful(self) -> bool:
        return self not in {ContentType.PROFANITY, ContentType.FRUSTRATION}

    # ------------------------
    # 🔥 IS EMOTIONAL
    # ------------------------
    def is_emotional(self) -> bool:
        return self == ContentType.FRUSTRATION

    # ------------------------
    # 🔥 CATEGORY GROUP
    # ------------------------
    def category(self) -> str:
        if self in {ContentType.FRUSTRATION, ContentType.PROFANITY}:
            return "emotional"

        if self in {ContentType.ABUSE, ContentType.HARASSMENT}:
            return "behavior"

        if self in {ContentType.HATE_SPEECH, ContentType.VIOLENCE}:
            return "danger"

        return "other"

    # ------------------------
    # 🔥 COLOR TAG (UI / LOGGING)
    # ------------------------
    def color(self) -> str:
        if not hasattr(ContentType, "_color_map"):
            ContentType._color_map = {
                ContentType.PROFANITY: "yellow",
                ContentType.FRUSTRATION: "blue",
                ContentType.ABUSE: "orange",
                ContentType.HARASSMENT: "orange",
                ContentType.MANIPULATION: "purple",
                ContentType.HATE_SPEECH: "red",
                ContentType.VIOLENCE: "dark_red",
                ContentType.SEXUAL: "pink",
            }
        return ContentType._color_map.get(self, "white")

    # ------------------------
    # 🔥 TO DICT (ANALYTICS)
    # ------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.value,
            "name": self.name,
            "priority": self.priority(),
            "severity": self.severity(),
            "category": self.category(),
            "is_harmful": self.is_harmful(),
            "is_emotional": self.is_emotional(),
            "color": self.color(),
        }

    # ------------------------
    # 🔥 BATCH ANALYSIS
    # ------------------------
    @classmethod
    def analyze_types(cls, types: List["ContentType"]) -> Dict[str, Any]:
        try:
            if not types:
                return {}

            max_severity = max(t.severity() for t in types)
            max_priority = max(t.priority() for t in types)

            harmful = any(t.is_harmful() for t in types)
            emotional = any(t.is_emotional() for t in types)

            return {
                "count": len(types),
                "max_severity": max_severity,
                "max_priority": max_priority,
                "is_harmful": harmful,
                "is_emotional": emotional,
                "types": [t.value for t in types],
            }

        except Exception:
            return {}

    # ------------------------
    # 🔥 STRING REPRESENTATION
    # ------------------------
    def __str__(self):
        return self.value.upper()

    def __repr__(self):
        return f"<ContentType {self.name} severity={self.severity()}>"


class FilterAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDIRECT = "redirect"
    SOFTEN = "soften"  # emotional handling

    # ------------------------
    # 🔥 STRING PARSER (SAFE)
    # ------------------------
    @classmethod
    def from_string(cls, value: str):
        try:
            if not value:
                return cls.ALLOW

            value = str(value).lower().strip()

            for action in cls:
                if action.value == value:
                    return action

            return cls.ALLOW

        except Exception:
            return cls.ALLOW

    # ------------------------
    # 🔥 SAFETY CHECK
    # ------------------------
    def is_safe(self) -> bool:
        return self in {FilterAction.ALLOW, FilterAction.SOFTEN, FilterAction.WARN}

    # ------------------------
    # 🔥 BLOCKING CHECK
    # ------------------------
    def is_blocking(self) -> bool:
        return self in {FilterAction.BLOCK, FilterAction.REDIRECT}

    # ------------------------
    # 🔥 PRIORITY (DECISION WEIGHT)
    # ------------------------
    def priority(self) -> int:
        priority_map = {
            FilterAction.ALLOW: 0,
            FilterAction.SOFTEN: 1,
            FilterAction.WARN: 2,
            FilterAction.REDIRECT: 3,
            FilterAction.BLOCK: 4,
        }
        return priority_map[self]

    # ------------------------
    # 🔥 USER MESSAGE
    # ------------------------
    def message(self) -> str:
        messages = {
            FilterAction.ALLOW: "Content is safe",
            FilterAction.SOFTEN: "Emotional tone detected, softening response",
            FilterAction.WARN: "Content has minor issues",
            FilterAction.REDIRECT: "Unsafe intent detected, redirecting",
            FilterAction.BLOCK: "Content blocked due to safety concerns",
        }
        return messages.get(self, "Unknown action")

    # ------------------------
    # 🔥 LOG FORMAT
    # ------------------------
    def to_dict(self):
        return {
            "action": self.value,
            "safe": self.is_safe(),
            "blocking": self.is_blocking(),
            "priority": self.priority(),
            "message": self.message(),
        }

    # ------------------------
    # 🔥 STRING REPRESENTATION
    # ------------------------
    def __str__(self):
        return self.value

    def __repr__(self):
        return f"<FilterAction {self.name}>"


# ------------------------
# ACTIONS
# ------------------------
class ContentAnalysis:
    def __init__(self, content: str, content_types: List["ContentType"]):
        try:
            self.content = str(content)
            self.content_types = content_types or []

            # ------------------------
            # 🔥 TIMESTAMP
            # ------------------------
            self.timestamp = time.time()

            # ------------------------
            # 🔥 DERIVED METRICS
            # ------------------------
            self.count = len(self.content_types)

            self.max_severity = max(
                (t.severity() for t in self.content_types), default=0.0
            )

            self.max_priority = max(
                (t.priority() for t in self.content_types), default=0
            )

            self.is_harmful = any(t.is_harmful() for t in self.content_types)

            self.is_emotional = any(t.is_emotional() for t in self.content_types)

            # ------------------------
            # 🔥 SAFETY SCORE (0–1)
            # ------------------------
            self.safety_score = round(1.0 - self.max_severity, 2)

            # ------------------------
            # 🔥 RISK LEVEL
            # ------------------------
            if self.max_severity >= 0.85:
                self.risk_level = "critical"
            elif self.max_severity >= 0.6:
                self.risk_level = "high"
            elif self.max_severity >= 0.3:
                self.risk_level = "medium"
            else:
                self.risk_level = "low"

            # ------------------------
            # 🔥 CONTENT HASH (TRACKING)
            # ------------------------
            self.hash = hashlib.md5(self.content.encode()).hexdigest()

        except Exception:
            self.content = ""
            self.content_types = []
            self.timestamp = time.time()
            self.count = 0
            self.max_severity = 0.0
            self.max_priority = 0
            self.is_harmful = False
            self.is_emotional = False
            self.safety_score = 1.0
            self.risk_level = "low"
            self.hash = "unknown"

    # ------------------------
    # 🔥 DECISION ENGINE
    # ------------------------
    def decide_action(self) -> str:
        try:
            # ------------------------
            # HARMFUL CONTENT
            # ------------------------
            if self.is_harmful:
                if self.max_severity >= 0.85:
                    return "block"
                elif self.max_severity >= 0.6:
                    return "redirect"
                else:
                    return "warn"

            # ------------------------
            # EMOTIONAL CONTENT
            # ------------------------
            if self.is_emotional:
                return "soften"

            # ------------------------
            # DEFAULT SAFE
            # ------------------------
            return "allow"

        except Exception:
            return "allow"

    # ------------------------
    # 🔥 SUMMARY TEXT
    # ------------------------
    def summary(self) -> str:
        try:
            types = ", ".join(t.value for t in self.content_types) or "none"

            return (
                f"Types: {types} | "
                f"Severity: {self.max_severity} | "
                f"Risk: {self.risk_level}"
            )

        except Exception:
            return "Analysis unavailable"

    # ------------------------
    # 🔥 TO DICT (FULL EXPORT)
    # ------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_hash": self.hash,
            "types": [t.to_dict() for t in self.content_types],
            "count": self.count,
            "max_severity": self.max_severity,
            "max_priority": self.max_priority,
            "is_harmful": self.is_harmful,
            "is_emotional": self.is_emotional,
            "safety_score": self.safety_score,
            "risk_level": self.risk_level,
            "recommended_action": self.decide_action(),
            "timestamp": self.timestamp,
        }

    # ------------------------
    # 🔥 QUICK CHECK
    # ------------------------
    def is_safe(self) -> bool:
        return not self.is_harmful

    # ------------------------
    # 🔥 STRING REPRESENTATION
    # ------------------------
    def __str__(self):
        return f"[{self.risk_level.upper()}] {self.summary()}"

    def __repr__(self):
        return f"<ContentAnalysis risk={self.risk_level} severity={self.max_severity}>"


# ------------------------
# MAIN FILTER
# ------------------------
class ContentFilter:
    def __init__(self):
        # ------------------------
        # 🔥 PATTERNS
        # ------------------------
        self.profanity_patterns = [
            r"\b(?:damn|hell)\b",
        ]

        self.abuse_patterns = [
            r"(?:kill|hurt|harm).*(?:you|yourself)",
            r"(?:stupid|idiot|dumb)",
            r"(?:worthless|useless)",
        ]

        self.harassment_patterns = [
            r"(?:follow|find|locate).*(?:you|your)",
            r"(?:threat|threaten)",
            r"(?:bully|bullying)",
        ]

        self.hate_patterns = [
            r"(?:hate|despise).*(?:gender|race|religion)",
        ]

        self.violence_patterns = [
            r"(?:murder|kill|attack)",
            r"(?:bomb|explosive)",
            r"(?:shoot|gun)",
        ]

        # 🔥 NEW: frustration detection
        self.frustration_patterns = [
            r"(?:frustrated|annoyed|angry|irritated)",
            r"(?:this is stupid|this is annoying)",
        ]

        # ------------------------
        # 🔥 STATS
        # ------------------------
        self.allowed_content_count = 0
        self.blocked_content_count = 0
        self.filter_log: List[Dict[str, Any]] = []

    # ------------------------
    # PATTERN CHECK
    # ------------------------
    def _check_patterns(self, text: str, patterns: List[str]) -> bool:
        text = text.lower()

        for pattern in patterns:
            try:
                if re.search(pattern, text):
                    return True
            except:
                continue

        return False

    # ------------------------
    # ANALYZE
    # ------------------------
    def analyze_content(self, text: str) -> ContentAnalysis:
        detected = []
        severity = 0.0

        if self._check_patterns(text, self.profanity_patterns):
            detected.append(ContentType.PROFANITY)
            severity = max(severity, 0.2)

        if self._check_patterns(text, self.abuse_patterns):
            detected.append(ContentType.ABUSE)
            severity = max(severity, 0.6)

        if self._check_patterns(text, self.harassment_patterns):
            detected.append(ContentType.HARASSMENT)
            severity = max(severity, 0.7)

        if self._check_patterns(text, self.hate_patterns):
            detected.append(ContentType.HATE_SPEECH)
            severity = max(severity, 0.8)

        if self._check_patterns(text, self.violence_patterns):
            detected.append(ContentType.VIOLENCE)
            severity = max(severity, 0.9)

        # 🔥 NEW: frustration (safe emotion)
        if self._check_patterns(text, self.frustration_patterns):
            detected.append(ContentType.FRUSTRATION)
            severity = max(severity, 0.3)

        is_safe = (
            all(t == ContentType.FRUSTRATION for t in detected) or len(detected) == 0
        )

        analysis = ContentAnalysis(text, detected)

        self.filter_log.append(analysis.to_dict())

        return analysis

    # ------------------------
    # FILTER
    # ------------------------
    def filter_content(
        self, content: str, learn: bool = False
    ) -> Tuple[bool, FilterAction, str]:

        try:
            # ------------------------
            # INPUT NORMALIZATION
            # ------------------------
            content = str(content).strip()

            if not content:
                return True, FilterAction.ALLOW, "Empty content"

            # ------------------------
            # ANALYSIS
            # ------------------------
            analysis = self.analyze_content(content)

            # ------------------------
            # FIX: SAFE CHECK (NO CALL ERROR)
            # ------------------------
            is_safe = analysis.is_safe()

            # ------------------------
            # LEARNING CONTROL
            # ------------------------
            if learn and not is_safe:
                self.blocked_content_count += 1
                return False, FilterAction.BLOCK, "Learning blocked: unsafe content"

            # ------------------------
            # DECISION ENGINE
            # ------------------------
            action = analysis.decide_action()

            # ------------------------
            # SAFE CONTENT
            # ------------------------
            if action == "allow":
                self.allowed_content_count += 1
                return True, FilterAction.ALLOW, "Content is safe"

            # ------------------------
            # EMOTIONAL HANDLING
            # ------------------------
            if action == "soften":
                self.allowed_content_count += 1
                return (
                    True,
                    FilterAction.SOFTEN,
                    "User frustration detected → soften response",
                )

            # ------------------------
            # WARNING LEVEL
            # ------------------------
            if action == "warn":
                self.allowed_content_count += 1
                return True, FilterAction.WARN, "Minor issue detected"

            # ------------------------
            # REDIRECT
            # ------------------------
            if action == "redirect":
                self.blocked_content_count += 1
                return False, FilterAction.REDIRECT, "Unsafe intent detected"

            # ------------------------
            # BLOCK
            # ------------------------
            self.blocked_content_count += 1
            return False, FilterAction.BLOCK, "Content blocked due to high risk"

        except Exception as e:
            try:
                print(f"[ContentFilter] error: {e}")
            except:
                pass

            return False, FilterAction.BLOCK, "System safety fallback"

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        total = self.allowed_content_count + self.blocked_content_count

        return {
            "allowed": self.allowed_content_count,
            "blocked": self.blocked_content_count,
            "total": total,
            "block_rate": round(self.blocked_content_count / total, 3) if total else 0,
        }


# ------------------------
# EXPORTS
# ------------------------
__all__ = ["ContentFilter", "ContentAnalysis", "ContentType", "FilterAction"]
