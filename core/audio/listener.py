import asyncio
import queue
import inspect
import time
import re
from typing import Optional, List, Callable

import numpy as np
import speech_recognition as sr

# ── optional Whisper ──────────────────────────────────────────────────────────
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# ── logger ────────────────────────────────────────────────────────────────────
try:
    from core.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Whisper commonly hallucinates these when there is only silence / noise
_WHISPER_NOISE_PHRASES = {
    "you",
    "thanks for watching",
    "thank you",
    "thank you for watching",
    ".",
    "..",
    "...",
    "uh",
    "um",
    "hmm",
    "hm",
}


class AudioListener:
    """
    Continuously listens to the microphone, transcribes speech (Whisper
    primary, Google fallback), and makes recognised text available via
    get_text() / has_text() or registered callbacks.
    """

    def __init__(
        self,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        phrase_time_limit: float = 5.0,
        language: str = "en-US",
        whisper_model: str = "base",
    ):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True

        self._phrase_time_limit = phrase_time_limit
        self.language = language

        self._audio_queue: queue.Queue = queue.Queue(maxsize=20)
        self._text_queue: queue.Queue = queue.Queue(maxsize=50)
        self._running = False
        self.is_listening = False

        # FIX 3: protect dedup state with a lock
        self._dedup_lock = asyncio.Lock()
        self._last_voice: str = ""
        self._last_time: float = 0.0

        self._callbacks: List[Callable] = []

        # FIX 1: try CUDA, fall back to CPU
        self._whisper_model = None
        if WHISPER_AVAILABLE:
            for device in ("cuda", "cpu"):
                try:
                    self._whisper_model = whisper.load_model(
                        whisper_model, device=device
                    )
                    logger.info(f"[Whisper] Model '{whisper_model}' loaded on {device}")
                    break
                except Exception as exc:
                    logger.warning(f"[Whisper] Could not load on {device}: {exc}")

            if self._whisper_model is None:
                logger.error(
                    "[Whisper] Model could not be loaded on any device; will use Google only."
                )

        logger.info("[Listener] Initialised")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the listener (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 PREVENT DOUBLE START (CRITICAL FIX)
            # ------------------------
            if getattr(self, "_running", False):
                if hasattr(self, "logger"):
                    self.logger.warning("Listener already running")
                return

            # ------------------------
            # 🔥 INIT STATE
            # ------------------------
            self._running = True
            self.is_listening = True

            # ------------------------
            # 🔥 CALIBRATION (SAFE)
            # ------------------------
            try:
                await asyncio.to_thread(self._calibrate)
            except Exception as e:
                raise RuntimeError(f"Calibration failed: {e}")

            # ------------------------
            # 🔥 CREATE TASKS (SAFE)
            # ------------------------
            try:
                self._loop_task = asyncio.create_task(
                    self._loop(), name="audio-capture"
                )

                self._process_task = asyncio.create_task(
                    self._process_audio_continuous(), name="audio-process"
                )

            except Exception as e:
                raise RuntimeError(f"Task creation failed: {e}")

            # ------------------------
            # 🔥 SUCCESS LOG
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info("🎤 Listener started successfully")
            except Exception:
                pass

        except Exception as exc:
            # ------------------------
            # 🔥 CLEANUP ON FAILURE (CRITICAL FIX)
            # ------------------------
            self._running = False
            self.is_listening = False

            # cancel partially created tasks
            for attr in ("_loop_task", "_process_task"):
                task = getattr(self, attr, None)
                if task and not task.done():
                    task.cancel()

            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Listener start failed: {exc}")
            except Exception:
                pass

            raise

    def start_listening(self) -> None:
        """
        Synchronous entry-point.  Works whether an event loop is already
        running (schedules a task) or not (creates a new loop).
        """
        try:
            loop = asyncio.get_running_loop()
            # We are inside a running loop – schedule as a task.
            loop.create_task(self.start())
        except RuntimeError:
            # No running loop.
            asyncio.run(self.start())

    def get_text(self) -> Optional[str]:
        """Non-blocking.  Returns the next recognised utterance or None."""
        try:
            return self._text_queue.get_nowait()
        except queue.Empty:
            return None

    def has_text(self) -> bool:
        return not self._text_queue.empty()

    def add_callback(self, cb: Callable) -> None:
        """Register a callback (plain function OR async coroutine function)."""
        self._callbacks.append(cb)

    async def stop(self) -> None:
        """
        Cancel all background tasks and release resources (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 PREVENT DOUBLE STOP
            # ------------------------
            if not getattr(self, "_running", False):
                if hasattr(self, "logger"):
                    self.logger.warning("Listener already stopped")
                return

            # ------------------------
            # 🔥 UPDATE STATE
            # ------------------------
            self._running = False
            self.is_listening = False

            # ------------------------
            # 🔥 COLLECT TASKS (SAFE)
            # ------------------------
            tasks = []
            for attr in ("_loop_task", "_process_task"):
                task = getattr(self, attr, None)
                if task and not task.done():
                    task.cancel()
                    tasks.append(task)

            # ------------------------
            # 🔥 AWAIT CANCELLATION (CRITICAL FIX)
            # ------------------------
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception:
                    pass

            # ------------------------
            # 🔥 CLEAR TASK REFERENCES
            # ------------------------
            try:
                self._loop_task = None
                self._process_task = None
            except Exception:
                pass

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
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info("🛑 Listener stopped")
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Listener stop failed: {e}")
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Internal – calibration
    # ──────────────────────────────────────────────────────────────────────────

    def _calibrate(self) -> None:
        """Run once synchronously before the async tasks start."""
        with self.microphone as source:
            logger.info("🎤 Calibrating mic …")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info("🎤 Calibration done")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal – capture loop
    # ──────────────────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """
        Audio capture loop (optimized + production-grade)
        Keeps microphone open and handles async safely
        """

        import asyncio

        try:
            if hasattr(self, "logger"):
                self.logger.info("🎙️ Audio capture loop started")

            # ------------------------
            # 🔥 OPEN MIC ONCE (CRITICAL FIX)
            # ------------------------
            with self.microphone as source:

                def _listen_blocking() -> Optional[sr.AudioData]:
                    try:
                        return self.recognizer.listen(
                            source,
                            timeout=1.0,
                            phrase_time_limit=self._phrase_time_limit,
                        )
                    except sr.WaitTimeoutError:
                        return None
                    except Exception as exc:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"Listen error: {exc}")
                        return None

                # ------------------------
                # 🔄 MAIN LOOP
                # ------------------------
                while getattr(self, "_running", False):

                    try:
                        audio = await asyncio.to_thread(_listen_blocking)

                        # ------------------------
                        # 🔥 SAFE QUEUE INSERT
                        # ------------------------
                        if audio is not None:
                            try:
                                if not self._audio_queue.full():
                                    self._audio_queue.put_nowait(audio)
                            except Exception:
                                pass

                        await asyncio.sleep(0.01)

                    except asyncio.CancelledError:
                        break

                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Loop iteration error: {e}")
                        await asyncio.sleep(0.1)

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Audio loop failed: {e}")
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Internal – process loop
    # ──────────────────────────────────────────────────────────────────────────

    async def _process_audio_continuous(self) -> None:
        """
        Audio processing loop (optimized + production-grade)
        Non-blocking, safe, and resilient
        """

        import asyncio
        import queue

        try:
            if hasattr(self, "logger"):
                self.logger.info("🔄 Audio processing loop started")

            while getattr(self, "_running", False):

                try:
                    # ------------------------
                    # 🔥 NON-BLOCKING QUEUE READ
                    # ------------------------
                    try:
                        audio = self._audio_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue

                    # ------------------------
                    # 🔥 SAFE TRANSCRIPTION
                    # ------------------------
                    try:
                        text = await self._transcribe(audio)
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Transcription failed: {e}")
                        continue

                    if not text or not self._valid(text):
                        continue

                    # ------------------------
                    # 🔥 CLEAN TEXT
                    # ------------------------
                    try:
                        text = self._clean(text)
                    except Exception:
                        continue

                    # ------------------------
                    # 🔁 DEDUP (CRITICAL FIX)
                    # ------------------------
                    async with self._dedup_lock:
                        now = time.monotonic()
                        if text == self._last_voice and now - self._last_time < 1.0:
                            continue
                        self._last_voice = text
                        self._last_time = now

                    # ------------------------
                    # 🔥 LOG OUTPUT
                    # ------------------------
                    try:
                        if hasattr(self, "logger"):
                            self.logger.info(f"🎤 Recognised: '{text}'")
                    except Exception:
                        pass

                    # ------------------------
                    # 🔥 SAFE TEXT QUEUE INSERT
                    # ------------------------
                    try:
                        if self._text_queue.full():
                            try:
                                self._text_queue.get_nowait()
                            except queue.Empty:
                                pass

                        self._text_queue.put_nowait(text)
                    except Exception:
                        pass

                    # ------------------------
                    # 🔥 CALLBACK DISPATCH (SAFE)
                    # ------------------------
                    try:
                        await self._dispatch_callbacks(text)
                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.error(f"Callback dispatch failed: {e}")

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Processing loop error: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Audio processing loop crashed: {e}")
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Internal – transcription
    # ──────────────────────────────────────────────────────────────────────────

    async def _transcribe(self, audio: sr.AudioData) -> Optional[str]:
        """
        Transcribe audio using Whisper (primary) and Google (fallback)
        Optimized + production-grade
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if audio is None:
                return None

            # ------------------------
            # 🎤 WHISPER FIRST (SAFE)
            # ------------------------
            if getattr(self, "_whisper_model", None):
                try:
                    text = await asyncio.wait_for(self._whisper_rec(audio), timeout=8.0)

                    if text and self._valid(text):
                        return text

                except asyncio.TimeoutError:
                    if hasattr(self, "logger"):
                        self.logger.warning("Whisper timeout")
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.debug(f"Whisper failed: {e}")

            # ------------------------
            # 🌐 GOOGLE FALLBACK (SAFE)
            # ------------------------
            try:
                text = await asyncio.wait_for(self._google_rec(audio), timeout=6.0)

                if text and self._valid(text):
                    return text

            except asyncio.TimeoutError:
                if hasattr(self, "logger"):
                    self.logger.warning("Google recognition timeout")
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.debug(f"Google recognition failed: {e}")

            # ------------------------
            # 🔥 FINAL FALLBACK
            # ------------------------
            return None

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Transcription failed: {e}")
            except Exception:
                pass

            return None

    async def _whisper_rec(self, audio: sr.AudioData) -> Optional[str]:
        """
        Whisper transcription (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if audio is None or not getattr(self, "_whisper_model", None):
                return None

            def run() -> Optional[str]:
                try:
                    # ------------------------
                    # 🔥 SAFE AUDIO CONVERSION
                    # ------------------------
                    raw_bytes = audio.get_raw_data()
                    if not raw_bytes:
                        return None

                    raw = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

                    if raw.size == 0:
                        return None

                    # normalize
                    raw /= 32768.0

                    # ------------------------
                    # 🔥 WHISPER TRANSCRIBE
                    # ------------------------
                    result = self._whisper_model.transcribe(
                        raw,
                        fp16=False,
                        language="en",
                        temperature=0.0,
                        no_speech_threshold=0.6,
                    )

                    text = result.get("text", "")
                    text = text.strip() if text else ""

                    return text or None

                except Exception as exc:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"Whisper error: {exc}")
                    except Exception:
                        pass
                    return None

            # ------------------------
            # 🔥 THREAD EXECUTION (SAFE)
            # ------------------------
            return await asyncio.to_thread(run)

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Whisper recognition failed: {e}")
            except Exception:
                pass

            return None

    async def _google_rec(self, audio: sr.AudioData) -> Optional[str]:
        """
        Google Speech Recognition (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if audio is None:
                return None

            def run() -> Optional[str]:
                try:
                    text = self.recognizer.recognize_google(
                        audio, language=getattr(self, "language", "en-US")
                    )

                    if not text:
                        return None

                    text = str(text).strip()
                    return text or None

                except sr.UnknownValueError:
                    # no speech detected
                    return None

                except sr.RequestError as e:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Google API error: {e}")
                    except Exception:
                        pass
                    return None

                except Exception as e:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.debug(f"Google recognition error: {e}")
                    except Exception:
                        pass
                    return None

            # ------------------------
            # 🔥 TIMEOUT PROTECTION (CRITICAL FIX)
            # ------------------------
            return await asyncio.wait_for(asyncio.to_thread(run), timeout=6.0)

        except asyncio.TimeoutError:
            try:
                if hasattr(self, "logger"):
                    self.logger.warning("Google recognition timeout")
            except Exception:
                pass
            return None

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Google recognition failed: {e}")
            except Exception:
                pass
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal – validation & cleaning
    # ──────────────────────────────────────────────────────────────────────────

    def _valid(self, text: str) -> bool:
        if not text:
            return False
        stripped = text.strip().lower()
        if len(stripped) < 2:
            return False
        # FIX 8: reject hallucinated Whisper noise phrases
        if stripped in _WHISPER_NOISE_PHRASES:
            return False
        # FIX 8: only reject if the ENTIRE text is digits (not "2 pm")
        if re.fullmatch(r"\d+", stripped):
            return False
        return True

    def _clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    # ──────────────────────────────────────────────────────────────────────────
    # Internal – callback dispatch
    # ──────────────────────────────────────────────────────────────────────────

    async def _dispatch_callbacks(self, text: str) -> None:
        """
        Dispatch callbacks safely (optimized + production-grade)
        Supports sync + async callbacks with isolation
        """

        import asyncio

        try:
            callbacks = list(getattr(self, "_callbacks", []) or [])

            if not callbacks:
                return

            tasks = []

            for cb in callbacks:
                try:
                    # ------------------------
                    # 🔥 EXECUTE CALLBACK
                    # ------------------------
                    result = cb(text)

                    # ------------------------
                    # 🔁 ASYNC CALLBACK
                    # ------------------------
                    if inspect.isawaitable(result):
                        tasks.append(result)

                except Exception as exc:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Callback error (sync): {exc}")

            # ------------------------
            # 🔥 AWAIT ASYNC CALLBACKS (PARALLEL)
            # ------------------------
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        if hasattr(self, "logger"):
                            self.logger.error(f"Callback error (async): {res}")

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Callback dispatch failed: {e}")
            except Exception:
                pass
