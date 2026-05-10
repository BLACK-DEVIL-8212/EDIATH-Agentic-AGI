import asyncio
import io
import re
import shutil
import logging
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd

# ── logger with fallback ──────────────────────────────────────────────────────
try:
    from core.logging_config import get_logger  # type: ignore

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# FIX 11: move scipy import to module level
try:
    from scipy import signal as scipy_signal

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Suppress comtypes spam on Windows
logging.getLogger("comtypes").setLevel(logging.CRITICAL)

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    import pyttsx3  # type: ignore
except ImportError:
    pyttsx3 = None  # type: ignore

try:
    from gtts import gTTS  # type: ignore
    from pydub import AudioSegment  # type: ignore
    from pydub.effects import normalize as pydub_normalize  # type: ignore

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS / pydub not available — multi-language support limited")

try:
    import edge_tts  # type: ignore

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


# =============================================================================
# Enums
# =============================================================================


class Language(Enum):
    AUTO = "auto"
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    TURKISH = "tr"
    DUTCH = "nl"
    POLISH = "pl"
    SWEDISH = "sv"
    DANISH = "da"
    FINNISH = "fi"
    NORWEGIAN = "no"
    GREEK = "el"
    CZECH = "cs"
    # FIX 13: these were in _language_patterns but missing from the enum
    HEBREW = "he"
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"


class VoiceGender(Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceQuality(Enum):
    DEFAULT = "default"
    NATURAL = "natural"
    WARM = "warm"
    CLEAR = "clear"
    SOFT = "soft"
    EXPRESSIVE = "expressive"


# =============================================================================
# Speaker
# =============================================================================


class Speaker:

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        volume: float = 0.9,
        language: Language = Language.ENGLISH,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_input = False
        self._is_playing = False
        self._volume = max(0.0, min(1.0, volume))
        self._engine = None
        # FIX 16: Lock created lazily (see property below)
        self._lock_obj: Optional[asyncio.Lock] = None
        self._stop_signal = False
        self._current_task = None
        self.language = language
        self.auto_detect_language = True
        self._language_patterns = self._init_language_patterns()
        self.voice_gender = VoiceGender.FEMALE
        self.voice_quality = VoiceQuality.NATURAL
        # FIX 18: documented valid range
        self.speech_rate = 0  # percent, Edge TTS: -100 to +200
        self.speech_pitch = 1.05  # multiplier; converted to Hz for Edge TTS
        self.voice_emotion = "cheerful"
        self._enable_smoothing = True
        self._smoothing_factor = 0.98

        self.natural_voice_map: Dict[Language, Dict[str, str]] = {
            Language.ENGLISH: {
                "female": "en-US-JennyNeural",
                "male": "en-US-GuyNeural",
                "natural_female": "en-US-AriaNeural",
                "expressive_female": "en-US-JennyMultilingualNeural",
            },
            Language.SPANISH: {
                "female": "es-ES-ElviraNeural",
                "male": "es-ES-AlvaroNeural",
            },
            Language.FRENCH: {
                "female": "fr-FR-DeniseNeural",
                "male": "fr-FR-HenriNeural",
            },
            Language.GERMAN: {
                "female": "de-DE-KatjaNeural",
                "male": "de-DE-ConradNeural",
            },
            Language.ITALIAN: {
                "female": "it-IT-ElsaNeural",
                "male": "it-IT-DiegoNeural",
            },
            Language.PORTUGUESE: {
                "female": "pt-BR-FranciscaNeural",
                "male": "pt-BR-AntonioNeural",
            },
            Language.RUSSIAN: {
                "female": "ru-RU-SvetlanaNeural",
                "male": "ru-RU-DmitryNeural",
            },
            Language.CHINESE: {
                "female": "zh-CN-XiaoxiaoNeural",
                "male": "zh-CN-YunxiNeural",
            },
            Language.JAPANESE: {
                "female": "ja-JP-NanamiNeural",
                "male": "ja-JP-KeitaNeural",
            },
            Language.KOREAN: {
                "female": "ko-KR-SunHiNeural",
                "male": "ko-KR-InJoonNeural",
            },
            Language.HINDI: {
                "female": "hi-IN-SwaraNeural",
                "male": "hi-IN-MadhurNeural",
            },
            Language.ARABIC: {
                "female": "ar-EG-SalmaNeural",
                "male": "ar-EG-ShakirNeural",
            },
            Language.TURKISH: {
                "female": "tr-TR-EmelNeural",
                "male": "tr-TR-AhmetNeural",
            },
        }

        # pyttsx3
        if pyttsx3:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
                self._engine.setProperty("volume", self._volume)
                self._set_natural_female_voice()
                self._apply_voice_quality()
                # Callbacks: signatures are (name, loc, length) and (name, completed)
                self._engine.connect("started-word", lambda *a: None)
                self._engine.connect("finished-utterance", lambda *a: None)
                logger.info("pyttsx3 engine initialised")
            except Exception as exc:
                logger.error(f"pyttsx3 init failed: {exc}")
                self._engine = None

        self.edge_tts_available = EDGE_TTS_AVAILABLE
        self.gtts_available = GTTS_AVAILABLE
        self._ffmpeg_available = (
            shutil.which("ffmpeg") is not None or shutil.which("ffprobe") is not None
        )

        logger.info(
            f"Speaker initialised — language={self.language.value}, "
            f"ffmpeg={'yes' if self._ffmpeg_available else 'NO'}"
        )

    # ── lazy asyncio.Lock ─────────────────────────────────────────────────────

    @property
    def _lock(self) -> asyncio.Lock:
        # FIX 16: only create the Lock when we are inside a running event loop
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    # ── language detection ────────────────────────────────────────────────────

    def _init_language_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "zh": re.compile(r"[\u4e00-\u9fff]"),
            "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]"),
            "ko": re.compile(r"[\uac00-\ud7af]"),
            "ru": re.compile(r"[\u0400-\u04ff]"),
            "ar": re.compile(r"[\u0600-\u06ff]"),
            "he": re.compile(r"[\u0590-\u05ff]"),
            "hi": re.compile(r"[\u0900-\u097f]"),
            "th": re.compile(r"[\u0e00-\u0e7f]"),
        }

    def detect_language(self, text: str) -> Language:
        if not self.auto_detect_language:
            return self.language

        for lang_code, pattern in self._language_patterns.items():
            if pattern.search(text):
                try:
                    return Language(lang_code)  # FIX 13: enum now has all codes
                except ValueError:
                    pass

        text_lower = text.lower()
        language_indicators: Dict[Language, list] = {
            Language.SPANISH: [
                "hola",
                "como",
                "esta",
                "gracias",
                "adios",
                "bueno",
                "por favor",
            ],
            Language.FRENCH: [
                "bonjour",
                "merci",
                "comment",
                "oui",
                "non",
                "tres",
                "bien",
            ],
            Language.GERMAN: ["hallo", "danke", "bitte", "gut", "nein", "ja", "schön"],
            Language.ITALIAN: ["ciao", "grazie", "prego", "buono", "si", "no", "bene"],
            Language.PORTUGUESE: [
                "olá",
                "obrigado",
                "bom",
                "sim",
                "não",
                "tudo",
                "obrigada",
            ],
            Language.RUSSIAN: [
                "привет",
                "спасибо",
                "хорошо",
                "да",
                "нет",
                "пожалуйста",
            ],
            Language.DUTCH: [
                "hallo",
                "dank",
                "goed",
                "ja",
                "nee",
                "alsjeblieft",
                "bedankt",
            ],
            Language.HINDI: ["नमस्ते", "धन्यवाद", "अच्छा", "हाँ", "नहीं", "कृपया"],
        }
        for lang, indicators in language_indicators.items():
            if any(ind in text_lower for ind in indicators):
                return lang

        return self.language

    # ── voice helpers ─────────────────────────────────────────────────────────

    def _set_natural_female_voice(self) -> None:
        if not self._engine:
            return
        try:
            voices = self._engine.getProperty("voices")
            female_kw = [
                "female",
                "zira",
                "hazel",
                "susan",
                "kate",
                "linda",
                "anna",
                "emma",
            ]
            for voice in voices:
                vn, vi = voice.name.lower(), voice.id.lower()
                if any(kw in vn or kw in vi for kw in female_kw):
                    self._engine.setProperty("voice", voice.id)
                    logger.debug(f"Female voice: {voice.name}")
                    return
            if voices:
                self._engine.setProperty("voice", voices[0].id)
        except Exception as exc:
            logger.debug(f"Voice selection error: {exc}")

    def _apply_voice_quality(self) -> None:
        if not self._engine:
            return
        try:
            rate_map = {
                VoiceQuality.NATURAL: 150,
                VoiceQuality.WARM: 140,
                VoiceQuality.CLEAR: 165,
                VoiceQuality.SOFT: 145,
                VoiceQuality.EXPRESSIVE: 155,
            }
            rate = rate_map.get(self.voice_quality, 160)
            self._engine.setProperty("rate", rate)
            if self.voice_quality == VoiceQuality.WARM:
                self._engine.setProperty("volume", min(1.0, self._volume * 1.05))
            elif self.voice_quality == VoiceQuality.SOFT:
                self._engine.setProperty("volume", min(1.0, self._volume * 0.9))
        except Exception as exc:
            logger.debug(f"Voice quality error: {exc}")

    def _set_pyttsx3_voice(self) -> None:
        if not self._engine:
            return
        try:
            voices = self._engine.getProperty("voices")
            target = self.language.value
            female_kw = ["female", "zira", "hazel"]
            for voice in voices:
                vl, vn = voice.id.lower(), voice.name.lower()
                if (target in vl or vl.startswith(target)) and any(
                    kw in vn for kw in female_kw
                ):
                    self._engine.setProperty("voice", voice.id)
                    return
            for voice in voices:
                vl = voice.id.lower()
                if target in vl or vl.startswith(target):
                    self._engine.setProperty("voice", voice.id)
                    return
            self._set_natural_female_voice()
        except Exception as exc:
            logger.debug(f"Voice selection error: {exc}")

    # ── audio enhancement ─────────────────────────────────────────────────────

    def _enhance_audio_quality(self, samples: np.ndarray) -> np.ndarray:
        """
        Audio enhancement (optimized + production-grade)
        """

        import numpy as np

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if (
                samples is None
                or not isinstance(samples, np.ndarray)
                or samples.size < 100
                or not getattr(self, "_enable_smoothing", False)
                or not globals().get("_SCIPY_AVAILABLE", False)
            ):
                return samples

            # ------------------------
            # 🔧 ENSURE FLOAT32
            # ------------------------
            samples = samples.astype(np.float32, copy=False)

            # ------------------------
            # 🔊 APPLY SMOOTHING FILTER
            # ------------------------
            try:
                b = scipy_signal.firwin(5, 0.8, window="hamming")
                smoothed = scipy_signal.filtfilt(b, [1.0], samples)
            except Exception:
                return samples

            # ------------------------
            # 🔥 CLIPPING PROTECTION (CRITICAL FIX)
            # ------------------------
            max_val = float(np.max(np.abs(smoothed)))
            if max_val > 1e-6:
                smoothed = smoothed / max_val * 0.95

            # ------------------------
            # 🔄 MIX ORIGINAL + SMOOTHED
            # ------------------------
            output = smoothed * 0.7 + samples * 0.3

            # ------------------------
            # 🔥 FINAL SAFETY CLAMP
            # ------------------------
            output = np.clip(output, -1.0, 1.0)

            return output.astype(np.float32)

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Audio enhancement failed: {e}")
            except Exception:
                pass

            return samples

    # ── text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        # FIX 17: strip special tokens BEFORE generic HTML tags
        text = re.sub(r"<\|.*?\|>", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[<>|]", "", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list:
        sentences = re.split(r"(?<=[.!?]) +", text)
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) <= 200:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                # FIX 10: sentence itself may exceed 200 chars
                while len(sentence) > 200:
                    chunks.append(sentence[:200])
                    sentence = sentence[200:]
                current = sentence + " "
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text[:200]]

    # ── edge TTS pitch helper ─────────────────────────────────────────────────

    def _edge_pitch_str(self) -> str:
        # FIX 7: clamp to Edge TTS limit of ±20 Hz
        hz = int((self.speech_pitch - 1.0) * 20)
        hz = max(-20, min(20, hz))
        return f"{hz:+d}Hz"

    def _edge_rate_str(self) -> str:
        r = max(-50, min(100, self.speech_rate))  # FIX 18
        return f"{r:+d}%"

    # ── public speak() ────────────────────────────────────────────────────────

    async def speak(
        self,
        text: str,
        language: Optional[Language] = None,
    ) -> bool:
        """
        Main speech entry point (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not text or not isinstance(text, str):
                return False

            # ------------------------
            # 🔥 SINGLE EXECUTION LOCK (CRITICAL FIX)
            # ------------------------
            async with self._lock:

                # ------------------------
                # 🔥 RESET STOP SIGNAL
                # ------------------------
                self._stop_signal = False

                # ------------------------
                # 🔥 CLEAN TEXT
                # ------------------------
                clean_text = self._clean_text(text)
                if not clean_text:
                    return False

                # ------------------------
                # 🔥 LENGTH LIMIT
                # ------------------------
                if len(clean_text) > 500:
                    clean_text = clean_text[:500] + "…"
                    if hasattr(self, "logger"):
                        self.logger.debug("Text truncated to 500 chars")

                # ------------------------
                # 🔥 LANGUAGE DETECTION
                # ------------------------
                try:
                    target_language = language or self.detect_language(clean_text)
                except Exception:
                    target_language = self.language

                # ------------------------
                # 🛑 EARLY STOP CHECK
                # ------------------------
                if self._stop_signal:
                    return False

                # ========================
                # 🔊 PYTTSX3
                # ========================
                if getattr(self, "_engine", None):
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(self._safe_tts_stream, clean_text),
                            timeout=15,
                        )

                        if hasattr(self, "logger"):
                            self.logger.info(f"🔊 pyttsx3: {clean_text[:80]}")

                        return True

                    except asyncio.TimeoutError:
                        if hasattr(self, "logger"):
                            self.logger.warning("pyttsx3 timeout")

                    except Exception as exc:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"pyttsx3 failed: {exc}")

                # ========================
                # 🌐 EDGE TTS
                # ========================
                if getattr(self, "edge_tts_available", False):
                    try:
                        ok = await asyncio.wait_for(
                            self._speak_edge_tts_safe(clean_text, target_language),
                            timeout=20,
                        )
                        if ok:
                            return True
                    except asyncio.TimeoutError:
                        if hasattr(self, "logger"):
                            self.logger.warning("Edge TTS timeout")
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"Edge TTS failed: {e}")

                # ========================
                # 🌍 GTTS
                # ========================
                if getattr(self, "gtts_available", False):
                    try:
                        ok = await asyncio.wait_for(
                            self._speak_gtts_safe(clean_text, target_language),
                            timeout=20,
                        )
                        if ok:
                            return True
                    except asyncio.TimeoutError:
                        if hasattr(self, "logger"):
                            self.logger.warning("gTTS timeout")
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"gTTS failed: {e}")

                # ========================
                # 🔔 FALLBACK
                # ========================
                try:
                    await self._fallback_beep()
                except Exception:
                    pass

                return False

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ speak() failed: {e}")
            except Exception:
                pass

            return False

    # ── backend: Edge TTS ─────────────────────────────────────────────────────

    async def _speak_edge_tts_safe(self, text: str, language: Language) -> bool:
        """
        Edge TTS speech (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not text:
                return False

            # ------------------------
            # 🔥 VOICE SELECTION (SAFE)
            # ------------------------
            voice_info = self.natural_voice_map.get(
                language, self.natural_voice_map.get(Language.ENGLISH, {})
            )

            if (
                self.voice_quality == VoiceQuality.NATURAL
                and "natural_female" in voice_info
            ):
                voice = voice_info["natural_female"]
            elif (
                self.voice_quality == VoiceQuality.EXPRESSIVE
                and "expressive_female" in voice_info
            ):
                voice = voice_info["expressive_female"]
            elif self.voice_gender == VoiceGender.FEMALE:
                voice = voice_info.get("female", "en-US-JennyNeural")
            else:
                voice = voice_info.get("male", "en-US-GuyNeural")

            # ------------------------
            # 🔥 INIT EDGE TTS
            # ------------------------
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=self._edge_rate_str(),
                pitch=self._edge_pitch_str(),
            )

            # ------------------------
            # 🔁 STREAM AUDIO (SAFE + LIMIT)
            # ------------------------
            audio_chunks = []
            max_bytes = 10 * 1024 * 1024  # 10MB safety cap
            total_size = 0

            try:
                async for chunk in communicate.stream():

                    # 🛑 STOP SIGNAL
                    if self._stop_signal:
                        return False

                    if chunk.get("type") == "audio":
                        data = chunk.get("data", b"")

                        if not data:
                            continue

                        total_size += len(data)

                        # 🔥 MEMORY SAFETY
                        if total_size > max_bytes:
                            if hasattr(self, "logger"):
                                self.logger.warning(
                                    "Edge TTS audio too large, aborting"
                                )
                            return False

                        audio_chunks.append(data)

            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Edge stream error: {e}")
                return False

            if not audio_chunks:
                return False

            # ------------------------
            # 🔥 MERGE AUDIO
            # ------------------------
            audio_data = b"".join(audio_chunks)

            # ------------------------
            # 🔄 STOP CHECK BEFORE DECODE
            # ------------------------
            if self._stop_signal:
                return False

            # ------------------------
            # 🔥 DECODE MP3
            # ------------------------
            try:
                samples = self._decode_mp3_bytes(audio_data)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"MP3 decode failed: {e}")
                return False

            if samples is None or len(samples) == 0:
                return False

            # ------------------------
            # 🔊 PROCESS AUDIO
            # ------------------------
            try:
                samples = self._enhance_audio_quality(samples) * self._volume
            except Exception:
                pass

            # ------------------------
            # 🛑 FINAL STOP CHECK
            # ------------------------
            if self._stop_signal:
                return False

            # ------------------------
            # 🔥 PLAY AUDIO (TIMEOUT SAFE)
            # ------------------------
            try:
                await asyncio.wait_for(
                    self.play_audio(samples, blocking=True), timeout=30
                )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("Audio playback timeout")
                return False

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"🔊 Edge TTS: {text[:80]}")
            except Exception:
                pass

            return True

        except Exception as exc:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Edge TTS failed: {exc}")
            except Exception:
                pass

            return False

    async def _speak_edge_tts(self, text: str, language: Language) -> None:
        await self._speak_edge_tts_safe(text, language)

    # ── backend: gTTS ────────────────────────────────────────────────────────

    async def _speak_gtts_safe(self, text: str, language: Language) -> bool:
        """
        gTTS speech (optimized + production-grade)
        """

        import asyncio
        import numpy as np

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not text:
                return False

            if not getattr(self, "_ffmpeg_available", False):
                if hasattr(self, "logger"):
                    self.logger.warning("gTTS requires ffmpeg — not found. Skipping.")
                return False

            # ------------------------
            # 🔥 GENERATE AUDIO (SAFE)
            # ------------------------
            try:
                tts = gTTS(text=text, lang=language.value, slow=False)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"gTTS init failed: {e}")
                return False

            fp = io.BytesIO()

            try:
                await asyncio.to_thread(tts.write_to_fp, fp)
                fp.seek(0)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"gTTS generation failed: {e}")
                return False

            # ------------------------
            # 🔥 DECODE MP3 (SAFE)
            # ------------------------
            try:
                audio = AudioSegment.from_file(fp, format="mp3")
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"MP3 decode failed (ffmpeg missing or corrupt): {e}"
                    )
                return False

            # ------------------------
            # 🔧 PITCH ADJUSTMENT (SAFE)
            # ------------------------
            if self.voice_gender == VoiceGender.FEMALE and self.speech_pitch != 1.0:
                try:
                    octaves = (self.speech_pitch - 1.0) * 0.5
                    new_rate = int(audio.frame_rate * (2.0**octaves))
                    audio = audio._spawn(
                        audio.raw_data, overrides={"frame_rate": new_rate}
                    )
                    audio = audio.set_frame_rate(self.sample_rate)
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"Pitch adjust failed: {e}")

            # ------------------------
            # 🔥 NORMALIZE + CONVERT
            # ------------------------
            try:
                audio = pydub_normalize(audio)
                raw = np.array(audio.get_array_of_samples())

                if raw.size == 0:
                    return False

                samples = raw.astype(np.float32) / 32768.0
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"Audio conversion failed: {e}")
                return False

            # ------------------------
            # 🔊 ENHANCE AUDIO
            # ------------------------
            try:
                samples = self._enhance_audio_quality(samples) * self._volume
            except Exception:
                pass

            # ------------------------
            # 🛑 STOP CHECK
            # ------------------------
            if self._stop_signal:
                return False

            # ------------------------
            # 🔥 PLAY AUDIO (TIMEOUT SAFE)
            # ------------------------
            try:
                await asyncio.wait_for(
                    self.play_audio(samples, blocking=True), timeout=30
                )
            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("gTTS playback timeout")
                return False

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"🔊 gTTS: {text[:80]}")
            except Exception:
                pass

            return True

        except Exception as exc:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ gTTS failed: {exc}")
            except Exception:
                pass

            return False

    async def _speak_gtts(self, text: str, language: Language) -> None:
        await self._speak_gtts_safe(text, language)

    # ── pyttsx3 sync helper ───────────────────────────────────────────────────

    def _safe_tts_stream(self, text: str) -> None:
        """
        Safe pyttsx3 streaming (optimized + production-grade)
        Runs inside thread pool
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not text or not getattr(self, "_engine", None):
                return

            # ------------------------
            # 🔧 APPLY VOICE SETTINGS (SAFE)
            # ------------------------
            try:
                self._apply_voice_quality()
            except Exception:
                pass

            # ------------------------
            # 🔁 PROCESS TEXT IN CHUNKS
            # ------------------------
            chunks = list(self._chunk_text(text)) or []

            for chunk in chunks:

                # 🛑 STOP SIGNAL CHECK
                if self._stop_signal:
                    break

                if not chunk:
                    continue

                try:
                    # ------------------------
                    # 🔥 HARD STOP PREVIOUS AUDIO (CRITICAL FIX)
                    # ------------------------
                    try:
                        self._engine.stop()
                    except Exception:
                        pass

                    # ------------------------
                    # 🔊 SPEAK CHUNK
                    # ------------------------
                    self._engine.say(chunk)
                    self._engine.runAndWait()

                except RuntimeError:
                    # engine busy / internal state issue
                    try:
                        self._engine.stop()
                    except Exception:
                        pass

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"TTS chunk failed: {e}")
                    continue

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ TTS stream failed: {e}")
            except Exception:
                pass

    # ── MP3 decode helper ─────────────────────────────────────────────────────

    def _decode_mp3_bytes(self, data: bytes) -> Optional[np.ndarray]:
        """
        Decode raw MP3 bytes to float32 samples (optimized + production-grade)
        """

        import numpy as np

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not data or not isinstance(data, (bytes, bytearray)):
                return None

            # ------------------------
            # 🎧 PYDUB (PRIMARY)
            # ------------------------
            if getattr(self, "_ffmpeg_available", False):
                try:
                    seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
                    seg = seg.set_frame_rate(self.sample_rate).set_channels(1)

                    raw = np.array(seg.get_array_of_samples())

                    if raw.size == 0:
                        return None

                    return raw.astype(np.float32) / 32768.0

                except Exception as exc:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"pydub MP3 decode failed: {exc}")

            # ------------------------
            # 🔁 SOUNDFILE FALLBACK
            # ------------------------
            try:
                import soundfile as sf  # type: ignore

                samples, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)

                if samples is None or len(samples) == 0:
                    return None

                # 🔧 Ensure mono
                if isinstance(samples, np.ndarray) and samples.ndim > 1:
                    samples = samples.mean(axis=1)

                return samples.astype(np.float32)

            except Exception as exc:
                if hasattr(self, "logger"):
                    self.logger.debug(f"soundfile MP3 decode failed: {exc}")

            # ------------------------
            # 🔥 FINAL WARNING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.warning(
                        "Cannot decode MP3 — install ffmpeg or soundfile"
                    )
            except Exception:
                pass

            return None

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ MP3 decode failed: {e}")
            except Exception:
                pass

            return None

    # ── audio playback ────────────────────────────────────────────────────────

    async def play_audio(self, audio: np.ndarray, blocking: bool = True) -> None:
        """
        Play audio safely (optimized + production-grade)
        """

        import asyncio
        import numpy as np

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if audio is None or not isinstance(audio, np.ndarray) or audio.size == 0:
                return

            # ------------------------
            # 🔒 PREVENT OVERLAP (CRITICAL FIX)
            # ------------------------
            async with self._lock:
                self._is_playing = True
                self.block_input = True

                # ------------------------
                # 🔧 SAFE PLAY FUNCTION
                # ------------------------
                def _play():
                    try:
                        sd.stop()  # 🔥 stop any previous playback
                    except Exception:
                        pass

                    sd.play(audio, self.sample_rate)

                    if blocking:
                        sd.wait()

                # ------------------------
                # 🔥 EXECUTION
                # ------------------------
                if blocking:
                    await asyncio.to_thread(_play)
                else:
                    # non-blocking still protected
                    await asyncio.to_thread(_play)

        except asyncio.CancelledError:
            # ------------------------
            # 🛑 HANDLE CANCELLATION
            # ------------------------
            try:
                sd.stop()
            except Exception:
                pass
            raise

        except Exception as exc:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Audio playback failed: {exc}")
            except Exception:
                pass

        finally:
            # ------------------------
            # 🔄 RESET STATE
            # ------------------------
            self._is_playing = False
            self.block_input = False

    async def _fallback_beep(self) -> None:
        duration = 0.4
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = (np.sin(2 * np.pi * 440 * t) * self._volume).astype(np.float32)
        await self.play_audio(audio)

    # ── control ───────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """
        Stop all speech + audio playback safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🛑 SET STOP SIGNAL (CRITICAL FIRST)
            # ------------------------
            self._stop_signal = True

            # ------------------------
            # 🔊 STOP TTS ENGINE
            # ------------------------
            try:
                if getattr(self, "_engine", None):
                    self._engine.stop()
            except Exception as exc:
                if hasattr(self, "logger"):
                    self.logger.debug(f"engine.stop() error: {exc}")

            # ------------------------
            # 🔉 STOP AUDIO OUTPUT
            # ------------------------
            try:
                sd.stop()
            except Exception as exc:
                if hasattr(self, "logger"):
                    self.logger.debug(f"sd.stop() error: {exc}")

            # ------------------------
            # 🔄 RESET STATE FLAGS
            # ------------------------
            self._is_playing = False
            self.block_input = False

            # ------------------------
            # 🔥 OPTIONAL QUEUE CLEANUP
            # ------------------------
            try:
                if hasattr(self, "_audio_queue"):
                    while not self._audio_queue.empty():
                        self._audio_queue.get_nowait()
            except Exception:
                pass

            try:
                if hasattr(self, "_text_queue"):
                    while not self._text_queue.empty():
                        self._text_queue.get_nowait()
            except Exception:
                pass

            # ------------------------
            # 📢 LOG SUCCESS
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info("🛑 Speech system stopped")
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ stop() failed: {e}")
            except Exception:
                pass

    def set_language(self, language: Language) -> None:
        self.language = language
        self._set_pyttsx3_voice()
        logger.info(f"Language → {language.value}")

    def set_voice_gender(self, gender: VoiceGender) -> None:
        self.voice_gender = gender
        self._set_pyttsx3_voice()
        logger.info(f"Voice gender → {gender.value}")

    def set_voice_quality(self, quality: VoiceQuality) -> None:
        self.voice_quality = quality
        self._apply_voice_quality()
        logger.info(f"Voice quality → {quality.value}")

    def set_speech_rate(self, rate: int) -> None:
        # FIX 18: valid Edge TTS range is roughly -100% to +200%; -50/+100 is safe
        self.speech_rate = max(-50, min(100, rate))
        logger.info(f"Speech rate → {self.speech_rate}%")

    def set_speech_pitch(self, pitch: float) -> None:
        self.speech_pitch = max(0.5, min(1.5, pitch))
        logger.info(f"Speech pitch → {self.speech_pitch}")

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        if self._engine:
            try:
                self._engine.setProperty("volume", self._volume)
            except Exception as exc:
                logger.debug(f"Volume set error: {exc}")
        logger.info(f"Volume → {self._volume}")

    # ── properties / status ───────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def initialize(self, *args, **kwargs) -> bool:
        logger.info("Speaker ready")
        return True

    def get_available_languages(self) -> list:
        return [lang.value for lang in Language if lang != Language.AUTO]

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_playing": self._is_playing,
            "block_input": self.block_input,
            "language": self.language.value,
            "voice_gender": self.voice_gender.value,
            "voice_quality": self.voice_quality.value,
            "speech_rate": self.speech_rate,
            "speech_pitch": self.speech_pitch,
            "volume": self._volume,
            "engine_available": self._engine is not None,
            "edge_tts_available": self.edge_tts_available,
            "gtts_available": self.gtts_available,
            "ffmpeg_available": self._ffmpeg_available,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = ["Speaker", "Language", "VoiceGender", "VoiceQuality"]
