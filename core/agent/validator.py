"""
Advanced Validator for EDIATH R_AGENT (ENHANCED)
✔ Smart hallucination detection (less false positives)
✔ Response quality improvement
✔ Length + structure validation
✔ Context-aware rewriting
✔ Anti-garbage output
✔ Logging support
"""

from typing import Optional
import re

try:
    from ..utils.logger import logger
except:
    logger = None


class Validator:
    def __init__(self):
        # 🔥 HALLUCINATION / RISKY PHRASES (context-aware)
        self.banned_phrases = [
            "i see",
            "i can see",
            "i can hear you",
            "you are wearing",
            "behind you",
            "in front of you",
            "i am watching",
            "i accessed your",
        ]

        # 🔥 DANGEROUS / RESTRICTED TOPICS (hard block)
        self.restricted_keywords = [
            "ddos",
            "attack script",
            "hack",
            "exploit",
            "malware",
            "ransomware",
        ]

        # 🔥 LOW QUALITY RESPONSES (regex)
        self.low_quality_patterns = [
            r"^\.*$",  # only dots
            r"^ok$",
            r"^yes$",
            r"^no$",
            r"^hmm",
            r"^idk",
            r"^maybe$",
        ]

        # 🔥 DEFAULT SAFE RESPONSE
        self.default_response = "Sorry, I couldn't generate a reliable answer."

        # 🔥 SAFETY LIMITS
        self.max_length = 500
        self.min_length = 3

        # 🔥 RESPONSE CLEANING RULES
        self.strip_phrases = [
            "assistant:",
            "ASSISTANT:",
            "AI:",
        ]

        # 🔥 CONFIDENCE / QUALITY CONTROL FLAGS
        self.enable_hallucination_filter = True
        self.enable_quality_filter = True
        self.enable_restricted_filter = True

        # 🔥 DEBUG
        self.debug = False

    # ------------------------
    # MAIN VALIDATE
    # ------------------------
    def validate(self, response: Optional[str]) -> str:
        try:
            # ------------------------
            # 🔥 TYPE SAFETY (CRITICAL FIX)
            # ------------------------
            if isinstance(response, dict):
                response = response.get("response", "")

            if not response:
                return self._fallback("Empty response")

            if not isinstance(response, str):
                response = str(response)

            # ------------------------
            # 🔥 CLEAN
            # ------------------------
            response = self._clean(response)

            if not response:
                return self._fallback("Cleaned empty response")

            # ------------------------
            # 🔥 LIMIT SIZE
            # ------------------------
            response = response[: self.max_length]

            lower = response.lower()

            # ------------------------
            # 🔥 RESTRICTED CONTENT (HARD BLOCK)
            # ------------------------
            if getattr(self, "enable_restricted_filter", True):
                for keyword in getattr(self, "restricted_keywords", []):
                    if keyword in lower:
                        if logger:
                            logger.warning(
                                f"[Validator] Blocked restricted content: {keyword}"
                            )
                        return "I can't assist with that request."

            # ------------------------
            # 🔥 LOW QUALITY FILTER
            # ------------------------
            if getattr(self, "enable_quality_filter", True):
                if self._is_low_quality(response):
                    return self._improve_response(response)

            # ------------------------
            # 🔥 HALLUCINATION CHECK
            # ------------------------
            if getattr(self, "enable_hallucination_filter", True):
                if self._is_hallucination(response):
                    return "I don't have enough verified information to answer that accurately."

            # ------------------------
            # 🔥 FINAL CLEANUP
            # ------------------------
            response = response.strip()

            if len(response) < self.min_length:
                return self._fallback("Too short response")

            return response

        except Exception as e:
            return self._fallback(f"Validator error: {str(e)}")

    # ------------------------
    # CLEAN TEXT
    # ------------------------
    def _clean(self, text: str) -> str:
        try:
            if not text:
                return ""

            # 🔥 ensure string
            if not isinstance(text, str):
                text = str(text)

            # 🔥 basic strip
            text = text.strip()

            # 🔥 remove common AI tags
            for tag in getattr(
                self, "strip_phrases", ["assistant:", "ASSISTANT:", "AI:"]
            ):
                text = text.replace(tag, "")

            # 🔥 remove JSON blocks (important fix)
            import re

            text = re.sub(r"\{[\s\S]*?\}", "", text)

            # 🔥 remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            # 🔥 remove weird repeating characters
            text = re.sub(r"(.)\1{4,}", r"\1\1", text)

            return text.strip()

        except Exception as e:
            if logger:
                logger.warning(f"[Validator Clean Error] {e}")
            return ""

    # ------------------------
    # LOW QUALITY DETECTION
    # ------------------------
    def _is_low_quality(self, text: str) -> bool:
        try:
            if not text:
                return True

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()
            lower = text.lower()

            # ------------------------
            # 🔥 MIN LENGTH CHECK
            # ------------------------
            if len(text) < getattr(self, "min_length", 3):
                return True

            # ------------------------
            # 🔥 TOO SHORT WORD COUNT
            # ------------------------
            if len(text.split()) < 1:
                return True

            # ------------------------
            # 🔥 REGEX PATTERNS
            # ------------------------
            for pattern in getattr(self, "low_quality_patterns", []):
                if re.match(pattern, lower):
                    if logger:
                        logger.warning(f"[Validator] Low quality pattern: {text}")
                    return True

            # ------------------------
            # 🔥 REPETITIVE TEXT (e.g. ok ok ok)
            # ------------------------
            words = lower.split()
            if len(set(words)) == 1 and len(words) > 2:
                if logger:
                    logger.warning(f"[Validator] Repetitive response: {text}")
                return True

            # ------------------------
            # 🔥 TOO MANY SYMBOLS
            # ------------------------
            symbol_ratio = sum(not c.isalnum() for c in text) / len(text)
            if symbol_ratio > 0.5:
                if logger:
                    logger.warning(f"[Validator] Too many symbols: {text}")
                return True

            # ------------------------
            # 🔥 MEANINGLESS CHAR SEQUENCES
            # ------------------------
            if re.match(r"^[a-z]{1,3}$", lower):  # like "ok", "hmm"
                return True

            return False

        except Exception as e:
            if logger:
                logger.warning(f"[Validator LowQuality Error] {e}")
            return True

    # ------------------------
    # IMPROVE RESPONSE
    # ------------------------
    def _improve_response(self, text: str) -> str:
        try:
            if not text:
                return "Let me provide a clearer answer."

            if not isinstance(text, str):
                text = str(text)

            lower = text.strip().lower()

            # ------------------------
            # 🔥 VERY SHORT / WEAK RESPONSES
            # ------------------------
            if lower in ["ok", "yes", "no", "hmm", "idk", "maybe"]:
                return "Could you provide a bit more detail so I can give you a better answer?"

            # ------------------------
            # 🔥 REPETITIVE OR MEANINGLESS
            # ------------------------
            if len(set(lower.split())) == 1:
                return "Let me explain that more clearly."

            # ------------------------
            # 🔥 SYMBOL / GARBAGE TEXT
            # ------------------------
            import re

            if re.match(r"^[^a-zA-Z0-9]+$", lower):
                return (
                    "That response wasn't clear. Let me give you a better explanation."
                )

            # ------------------------
            # 🔥 DEFAULT IMPROVEMENT
            # ------------------------
            return "Let me give you a clearer and more helpful answer."

        except Exception as e:
            if logger:
                logger.warning(f"[Validator Improve Error] {e}")
            return "Let me provide a better response."

    # ------------------------
    # HALLUCINATION DETECTION (SMART)
    # ------------------------
    def _is_hallucination(self, text: str) -> bool:
        try:
            if not text:
                return False

            if not isinstance(text, str):
                text = str(text)

            lower = text.strip().lower()

            # ------------------------
            # 🔥 DIRECT PHRASE MATCH (STRONG SIGNAL)
            # ------------------------
            for phrase in getattr(self, "banned_phrases", []):
                if lower.startswith(phrase) or f". {phrase}" in lower:
                    if logger:
                        logger.warning(
                            f"[Validator] Hallucination detected (direct): {phrase}"
                        )
                    return True

            # ------------------------
            # 🔥 CONTEXTUAL VISUAL CLAIMS
            # ------------------------
            visual_patterns = [
                "i can see you",
                "i am looking at you",
                "i can hear you",
                "i see your",
            ]

            for vp in visual_patterns:
                if vp in lower:
                    if logger:
                        logger.warning(
                            f"[Validator] Hallucination detected (vision claim): {vp}"
                        )
                    return True

            # ------------------------
            # 🔥 OVERCONFIDENT FALSE CLAIMS
            # ------------------------
            risky_phrases = [
                "definitely",
                "100% sure",
                "without a doubt",
            ]

            if any(rp in lower for rp in risky_phrases):
                if logger:
                    logger.warning(f"[Validator] Overconfidence detected: {text}")
                return True

            # ------------------------
            # 🔥 CONTRADICTION / NONSENSE PATTERN
            # ------------------------
            import re

            if re.search(r"(cannot|can't).*but.*can", lower):
                if logger:
                    logger.warning(f"[Validator] Contradiction detected: {text}")
                return True

            return False

        except Exception as e:
            if logger:
                logger.warning(f"[Validator Hallucination Error] {e}")
            return False

    # ------------------------
    # FALLBACK
    # ------------------------
    def _fallback(self, reason: str) -> str:
        try:
            if logger:
                logger.warning(f"[Validator Fallback] {reason}")

            reason = str(reason).lower()

            # ------------------------
            # 🔥 CONTEXT-AWARE FALLBACKS
            # ------------------------
            if "empty" in reason:
                return "I didn't receive enough information to answer that."

            if "short" in reason:
                return "The response wasn't detailed enough. Let me try again."

            if "cleaned" in reason:
                return "I couldn't extract a meaningful answer from that."

            if "restricted" in reason:
                return "I can't assist with that request."

            if "hallucination" in reason:
                return "I don't have enough verified information to answer that accurately."

            if "error" in reason:
                return "Something went wrong while processing your request."

            # ------------------------
            # 🔥 DEFAULT SAFE RESPONSE
            # ------------------------
            return getattr(
                self,
                "default_response",
                "Sorry, I couldn't generate a reliable answer.",
            )

        except Exception as e:
            if logger:
                logger.warning(f"[Validator Fallback Error] {e}")

            return "Sorry, something went wrong."

    # ------------------------
    # EXTEND RULES
    # ------------------------
    def add_banned_phrase(self, phrase: str):
        try:
            if not phrase:
                return False

            # 🔥 ensure string + normalize
            if not isinstance(phrase, str):
                phrase = str(phrase)

            phrase = phrase.strip().lower()

            if not phrase:
                return False

            # 🔥 initialize if missing (safety)
            if not hasattr(self, "banned_phrases") or not isinstance(
                self.banned_phrases, list
            ):
                self.banned_phrases = []

            # 🔥 prevent duplicates
            if phrase in self.banned_phrases:
                return False

            # 🔥 add safely
            self.banned_phrases.append(phrase)

            if logger:
                logger.info(f"[Validator] Added banned phrase: {phrase}")

            return True

        except Exception as e:
            if logger:
                logger.warning(f"[Validator AddPhrase Error] {e}")
            return False

    def get_rules(self):
        try:
            return {
                # 🔥 safety rules
                "banned_phrases": list(getattr(self, "banned_phrases", [])),
                "restricted_keywords": list(getattr(self, "restricted_keywords", [])),
                # 🔥 quality rules
                "low_quality_patterns": list(getattr(self, "low_quality_patterns", [])),
                # 🔥 limits
                "max_length": getattr(self, "max_length", 500),
                "min_length": getattr(self, "min_length", 3),
                # 🔥 flags
                "enable_hallucination_filter": getattr(
                    self, "enable_hallucination_filter", True
                ),
                "enable_quality_filter": getattr(self, "enable_quality_filter", True),
                "enable_restricted_filter": getattr(
                    self, "enable_restricted_filter", True
                ),
                # 🔥 fallback
                "default_response": getattr(self, "default_response", "N/A"),
            }

        except Exception as e:
            if logger:
                logger.warning(f"[Validator GetRules Error] {e}")

            return {}
