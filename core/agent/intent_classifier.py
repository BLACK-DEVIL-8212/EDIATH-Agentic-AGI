"""
Advanced Intent Classifier for EDIATH (FINAL STABLE VERSION)
✔ Queue-safe
✔ JSON-safe
✔ Crash-resistant
✔ Production ready
"""

from typing import Dict
from enum import Enum
import asyncio

from ..utils.logger import logger
# NOTE: Do not import LLMEngine at module import time to avoid triggering model loads.
# The classifier should receive llm_engine via dependency injection.
LLMEngine = None

from ..brain.llm_engine import LLMEngine
from ..ui.web_ui_server import WebUIServer


# ------------------------
# ENUM
# ------------------------
class IntentType(Enum):
    # ------------------------
    # 🔥 CORE USER INTENTS
    # ------------------------
    QUERY = "query"
    COMMAND = "command"
    CONVERSATION = "conversation"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"

    # ------------------------
    # 🔥 ADVANCED / SYSTEM
    # ------------------------
    AUTOMATION = "automation"
    SYSTEM = "system"

    # ------------------------
    # 🔥 FALLBACK
    # ------------------------
    UNKNOWN = "unknown"

    # ------------------------
    # 🔥 SAFE PARSER (CRITICAL)
    # ------------------------
    @classmethod
    def from_string(cls, value: str):
        try:
            if not value:
                return cls.UNKNOWN

            value = str(value).strip().lower()

            for item in cls:
                if item.value == value:
                    return item

            return cls.UNKNOWN

        except Exception:
            return cls.UNKNOWN

    # ------------------------
    # 🔥 HELPER
    # ------------------------
    def is_action(self) -> bool:
        try:
            # 🔥 ensure valid enum instance
            if not isinstance(self, IntentType):
                return False

            # 🔥 action-type intents
            action_intents = {IntentType.COMMAND, IntentType.AUTOMATION}

            return self in action_intents

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Intent is_action Error] {e}")
            return False

    def is_conversation(self) -> bool:
        try:
            # 🔥 ensure valid enum instance
            if not isinstance(self, IntentType):
                return False

            # 🔥 conversational intents
            conversation_intents = {
                IntentType.QUERY,
                IntentType.CONVERSATION,
                IntentType.CLARIFICATION,
                IntentType.FEEDBACK,
            }

            return self in conversation_intents

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Intent is_conversation Error] {e}")
            return False


# ------------------------
# INTENT OBJECT
# ------------------------
class Intent:
    def __init__(self, intent_type: IntentType, confidence: float = 0.5):
        try:
            # 🔥 VALIDATE INTENT TYPE
            if not isinstance(intent_type, IntentType):
                intent_type = IntentType.from_string(intent_type)

            self.type = intent_type

            # 🔥 SAFE CONFIDENCE (0.0 → 1.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.5

            self.confidence = max(0.0, min(1.0, confidence))

            # 🔥 PARAMETERS (SAFE INIT)
            self.parameters: Dict[str, str] = {}

            # 🔥 META (future use)
            self.timestamp = __import__("datetime").datetime.now().isoformat()
            self.source = "classifier"

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Intent Init Error] {e}")

            # 🔥 HARD FAILSAFE
            self.type = IntentType.UNKNOWN
            self.confidence = 0.0
            self.parameters = {}
            self.timestamp = __import__("datetime").datetime.now().isoformat()
            self.source = "fallback"

    def to_dict(self):
        try:
            return {
                # 🔥 core fields
                "type": getattr(self.type, "value", "unknown"),
                "confidence": float(getattr(self, "confidence", 0.0)),
                # 🔥 safe parameters
                "parameters": dict(getattr(self, "parameters", {})),
                # 🔥 metadata (added for production)
                "timestamp": getattr(self, "timestamp", None),
                "source": getattr(self, "source", "unknown"),
            }

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Intent ToDict Error] {e}")

            # 🔥 safe fallback
            return {
                "type": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "timestamp": None,
                "source": "error",
            }


# ------------------------
# CLASSIFIER
# ------------------------
class IntentClassifier:
    def __init__(self, llm_engine=None):
        try:
            # 🔥 stats
            self.classification_count = 0

            # Reuse the shared engine when injected. Do not create a new
            # engine/model here; startup must not duplicate the GGUF load.
            self.llm = llm_engine

            # 🔥 rate limiting (prevents spam / overload)
            self._last_call = 0.0
            self.min_delay = 0.5  # seconds between LLM calls

            # 🔥 config flags
            self.use_llm_fallback = True
            self.rule_confidence_threshold = 0.6

            # 🔥 debug
            self.debug = False

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[IntentClassifier Init Error] {e}")

            # 🔥 safe fallback
            self.classification_count = 0
            self.llm = llm_engine if llm_engine is not None else None
            self.llm = None
            self._last_call = 0.0
            self.min_delay = 0.5
            self.use_llm_fallback = False
            self.rule_confidence_threshold = 0.6
            self.debug = False

    # ------------------------
    # MAIN
    # ------------------------
    async def classify(self, text: str) -> Intent:
        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            if not text or not isinstance(text, str):
                return Intent(IntentType.UNKNOWN, 0.0)

            text = text.strip()
            if not text:
                return Intent(IntentType.UNKNOWN, 0.0)

            # 🔥 stats
            self.classification_count += 1

            # ------------------------
            # 🔥 RULE-BASED FIRST (FAST)
            # ------------------------
            intent = self._rule_based(text)

            if intent and intent.confidence >= getattr(
                self, "rule_confidence_threshold", 0.6
            ):
                return intent

            # ------------------------
            # 🔥 LLM FALLBACK CONTROL
            # ------------------------
            if not getattr(self, "use_llm_fallback", True) or not self.llm:
                return intent or Intent(IntentType.UNKNOWN, 0.3)

            # ------------------------
            # 🔥 RATE LIMIT (ANTI-SPAM)
            # ------------------------
            now = asyncio.get_event_loop().time()
            min_delay = getattr(self, "min_delay", 0.5)

            if now - self._last_call < min_delay:
                return intent or Intent(IntentType.UNKNOWN, 0.3)

            self._last_call = now

            # ------------------------
            # 🔥 LLM CLASSIFICATION
            # ------------------------
            llm_intent = await self._llm_classify(text)

            if llm_intent and isinstance(llm_intent, Intent):
                return llm_intent

            # ------------------------
            # 🔥 FINAL FALLBACK
            # ------------------------
            return intent or Intent(IntentType.UNKNOWN, 0.3)

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[IntentClassifier Error] {e}")

            return Intent(IntentType.UNKNOWN, 0.0)

    # ------------------------
    # RULE BASED
    # ------------------------
    def _rule_based(self, text: str) -> Intent:
        try:
            if not text or not isinstance(text, str):
                return Intent(IntentType.UNKNOWN, 0.0)

            text_lower = text.strip().lower()
            words = text_lower.split()

            # ------------------------
            # 🔥 QUERY DETECTION
            # ------------------------
            question_words = {"what", "how", "why", "when", "where", "who"}
            if any(w in words for w in question_words) or "?" in text_lower:
                return Intent(IntentType.QUERY, 0.85)

            # ------------------------
            # 🔥 COMMAND DETECTION
            # ------------------------
            command_keywords = {"open", "run", "execute", "start", "launch"}
            if any(cmd in words for cmd in command_keywords):
                intent = Intent(IntentType.COMMAND, 0.85)

                # 🔥 extract target (next word after command)
                for cmd in command_keywords:
                    if cmd in words:
                        idx = words.index(cmd)
                        if idx + 1 < len(words):
                            intent.parameters["target"] = words[idx + 1]
                        break

                return intent

            # ------------------------
            # 🔥 FEEDBACK DETECTION
            # ------------------------
            feedback_keywords = {"thanks", "thank", "good", "bad", "great", "awesome"}
            if any(f in words for f in feedback_keywords):
                return Intent(IntentType.FEEDBACK, 0.7)

            # ------------------------
            # 🔥 CLARIFICATION DETECTION
            # ------------------------
            clarification_keywords = {"what do you mean", "explain", "clarify"}
            if any(phrase in text_lower for phrase in clarification_keywords):
                return Intent(IntentType.CLARIFICATION, 0.7)

            # ------------------------
            # 🔥 CONVERSATION DETECTION
            # ------------------------
            conversation_keywords = {"hello", "hi", "hey"}
            if any(c in words for c in conversation_keywords):
                return Intent(IntentType.CONVERSATION, 0.6)

            # ------------------------
            # 🔥 DEFAULT
            # ------------------------
            return Intent(IntentType.UNKNOWN, 0.3)

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[RuleBased Error] {e}")

            return Intent(IntentType.UNKNOWN, 0.0)

    # ------------------------
    # 🔥 LLM CLASSIFIER (FIXED)
    # ------------------------
    async def _llm_classify(self, text: str) -> Intent:
        try:
            if not text or not self.llm:
                return Intent(IntentType.UNKNOWN, 0.0)

            # ------------------------
            # 🔥 SAFE PROMPT
            # ------------------------
            prompt = f"""
    Return ONLY valid JSON.

    Format:
    {{"type":"query","confidence":0.8,"parameters":{{}}}}

    Allowed types:
    ["query","command","conversation","automation","system","unknown"]

    Text:
    {text[:150]}
    """

            # ------------------------
            # 🔥 CALL LLM (QUEUE SAFE)
            # ------------------------
            result = await self.llm.generate(prompt=prompt)

            raw = str(result).strip()

            if not raw:
                return Intent(IntentType.UNKNOWN, 0.3)

            # ------------------------
            # 🔥 CLEAN RESPONSE
            # ------------------------
            import re
            import json

            raw = raw.replace("\n", "").replace("\r", "")
            raw = re.sub(r"\s+", " ", raw)[:300]

            data = None

            # ------------------------
            # 🔥 DIRECT JSON PARSE
            # ------------------------
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                pass

            # ------------------------
            # 🔥 JSON EXTRACTION (FALLBACK)
            # ------------------------
            if not data:
                matches = re.findall(r"\{.*?\}", raw)

                for m in matches:
                    try:
                        parsed = json.loads(m)
                        if isinstance(parsed, dict):
                            data = parsed
                            break
                    except Exception:
                        continue

            # ------------------------
            # 🔥 FINAL FALLBACK
            # ------------------------
            if not data:
                return Intent(IntentType.UNKNOWN, 0.4)

            # ------------------------
            # 🔥 PARSE INTENT TYPE
            # ------------------------
            intent_type_raw = data.get("type", "unknown")
            intent_enum = IntentType.from_string(intent_type_raw)

            # ------------------------
            # 🔥 PARSE CONFIDENCE
            # ------------------------
            try:
                confidence = float(data.get("confidence", 0.5))
            except Exception:
                confidence = 0.5

            confidence = max(0.0, min(1.0, confidence))

            # ------------------------
            # 🔥 BUILD INTENT
            # ------------------------
            intent = Intent(intent_enum, confidence)

            # ------------------------
            # 🔥 PARAMETERS
            # ------------------------
            params = data.get("parameters", {})
            if isinstance(params, dict):
                # 🔥 sanitize keys/values
                clean_params = {}
                for k, v in params.items():
                    try:
                        clean_params[str(k)] = str(v)[:100]
                    except Exception:
                        continue
                intent.parameters = clean_params

            return intent

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Intent LLM Error] {e}")

            return Intent(IntentType.UNKNOWN, 0.2)

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self):
        try:
            return {
                # 🔥 core stats
                "classifications": getattr(self, "classification_count", 0),
                # 🔥 LLM status
                "llm_available": self.llm is not None,
                # 🔥 rate limiting
                "last_call_time": getattr(self, "_last_call", 0.0),
                "min_delay": getattr(self, "min_delay", 0.5),
                # 🔥 config
                "use_llm_fallback": getattr(self, "use_llm_fallback", True),
                "rule_confidence_threshold": getattr(
                    self, "rule_confidence_threshold", 0.6
                ),
                # 🔥 debug
                "debug": getattr(self, "debug", False),
            }

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[IntentClassifier GetStats Error] {e}")

            # 🔥 safe fallback
            return {
                "classifications": 0,
                "llm_available": False,
                "last_call_time": 0.0,
                "min_delay": 0.5,
                "use_llm_fallback": False,
                "rule_confidence_threshold": 0.6,
                "debug": False,
            }
