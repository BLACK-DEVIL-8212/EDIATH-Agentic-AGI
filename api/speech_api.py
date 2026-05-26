"""Speech API - RESTful API for speech operations with TTS/STT integration."""

from typing import Optional, Dict, Any, List, AsyncIterator
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid
import base64
import json
from enum import Enum
from pathlib import Path
import hashlib

from ..core.audio import (
    Listener,
    Speaker,
    SpeechToText,
    WakeWordDetector,
    DuplexController,
    NoiseFilter,
)
from ..core.utils.logger import logger
from ..core.memory.mongo_client import mongo_client, get_metrics_collection

# =========================
# ENUMS AND CONSTANTS
# =========================


class AudioFormat(str, Enum):
    """Audio format types."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    RAW = "raw"


class VoiceGender(str, Enum):
    """Voice gender options."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceLanguage(str, Enum):
    """Supported languages."""

    EN_US = "en-US"
    EN_GB = "en-GB"
    ES_ES = "es-ES"
    FR_FR = "fr-FR"
    DE_DE = "de-DE"
    IT_IT = "it-IT"
    PT_BR = "pt-BR"
    JA_JP = "ja-JP"
    KO_KR = "ko-KR"
    ZH_CN = "zh-CN"


class SpeechMode(str, Enum):
    """Speech processing modes."""

    COMMAND = "command"
    CONVERSATION = "conversation"
    DICTATION = "dictation"
    TRANSCRIPTION = "transcription"


# =========================
# REQUEST/RESPONSE MODELS
# =========================


class ListenRequest(BaseModel):
    """Request to listen to audio."""

    duration: float = Field(
        5.0, description="Listen duration in seconds", ge=0.5, le=300.0
    )
    format: AudioFormat = Field(AudioFormat.WAV, description="Audio format")
    sample_rate: int = Field(16000, description="Sample rate in Hz", ge=8000, le=48000)
    channels: int = Field(1, description="Number of audio channels", ge=1, le=2)
    enable_vad: bool = Field(True, description="Enable voice activity detection")
    silence_timeout: float = Field(
        1.5, description="Silence timeout in seconds", ge=0.5, le=5.0
    )
    save_to_file: bool = Field(False, description="Save audio to file")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ListenResponse(BaseModel):
    """Response from listening."""

    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_format: str
    duration: float
    sample_rate: int
    channels: int
    has_speech: bool
    speech_duration: Optional[float] = None
    recording_id: str
    timestamp: str
    file_path: Optional[str] = None


class TranscribeRequest(BaseModel):
    """Request to transcribe audio."""

    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_file: Optional[str] = Field(None, description="Path to audio file")
    language: VoiceLanguage = Field(VoiceLanguage.EN_US, description="Audio language")
    mode: SpeechMode = Field(SpeechMode.COMMAND, description="Speech mode")
    enable_diarization: bool = Field(False, description="Enable speaker diarization")
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class TranscribeResponse(BaseModel):
    """Response from transcription."""

    text: str
    confidence: float
    language: str
    mode: str
    duration: float
    processing_time_ms: float
    transcription_id: str
    timestamp: str
    alternatives: Optional[List[Dict[str, Any]]] = None
    speaker_segments: Optional[List[Dict[str, Any]]] = None


class SpeakRequest(BaseModel):
    """Request to speak text."""

    text: str = Field(..., description="Text to speak", min_length=1, max_length=5000)
    voice: Optional[str] = Field(None, description="Voice name or ID")
    gender: VoiceGender = Field(VoiceGender.FEMALE, description="Voice gender")
    language: VoiceLanguage = Field(VoiceLanguage.EN_US, description="Voice language")
    rate: float = Field(
        150.0, description="Speech rate in words per minute", ge=50, le=300
    )
    pitch: float = Field(1.0, description="Voice pitch multiplier", ge=0.5, le=2.0)
    volume: float = Field(1.0, description="Volume level", ge=0.0, le=2.0)
    format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")
    save_to_file: bool = Field(False, description="Save audio to file")
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class SpeakResponse(BaseModel):
    """Response from speaking."""

    status: str
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_format: str
    duration: float
    characters: int
    speech_id: str
    timestamp: str
    file_path: Optional[str] = None


class WakeWordRequest(BaseModel):
    """Request for wake word detection."""

    word: str = Field("hey computer", description="Wake word or phrase")
    sensitivity: float = Field(0.5, description="Detection sensitivity", ge=0.1, le=1.0)
    timeout: float = Field(
        10.0, description="Detection timeout in seconds", ge=1, le=60
    )


class WakeWordResponse(BaseModel):
    """Response from wake word detection."""

    detected: bool
    word: str
    confidence: float
    detection_time: float
    timestamp: str


class SpeechCommand(BaseModel):
    """Speech command recognition."""

    command: str
    intent: str
    confidence: float
    parameters: Dict[str, Any] = Field(default_factory=dict)


class StreamAudioChunk(BaseModel):
    """Chunk of streaming audio."""

    sequence: int
    audio_data: str  # Base64 encoded
    timestamp: float
    is_final: bool = False


# =========================
# AUDIO CACHE
# =========================


class AudioCache:
    """Cache for generated speech audio."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_cache_key(self, text: str, voice: str, rate: float, pitch: float) -> str:
        """Generate cache key for speech request."""
        cache_data = {"text": text, "voice": voice, "rate": rate, "pitch": pitch}
        return hashlib.sha256(
            json.dumps(cache_data, sort_keys=True).encode()
        ).hexdigest()

    def get(self, key: str) -> Optional[bytes]:
        """Get cached audio."""
        if key in self._cache:
            cached = self._cache[key]
            cache_time = cached.get("timestamp")

            if cache_time and (datetime.now() - cache_time).seconds < self.ttl_seconds:
                return cached.get("audio_data")
            else:
                del self._cache[key]
        return None

    def set(self, key: str, audio_data: bytes):
        """Cache audio data."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].get("timestamp", datetime.min),
            )
            del self._cache[oldest_key]

        self._cache[key] = {"audio_data": audio_data, "timestamp": datetime.now()}

    def clear(self):
        """Clear cache."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


# =========================
# MAIN API CLASS
# =========================


class SpeechApi:
    """Production-ready API for speech/audio operations with TTS/STT integration."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize speech API.

        Args:
            data_dir: Directory for storing audio data
        """
        self.data_dir = data_dir or Path("audio_data")
        self.data_dir.mkdir(exist_ok=True)
        self.audio_outputs_dir = self.data_dir / "outputs"
        self.audio_outputs_dir.mkdir(exist_ok=True)

        # Initialize components
        self.listener = Listener()
        self.speaker = Speaker()
        self.stt = SpeechToText()
        self.wakeword = WakeWordDetector()
        self.noise_filter = NoiseFilter()
        self.duplex_controller = DuplexController()

        self.request_count = 0
        self._audio_cache = AudioCache()

        # Request tracking
        self._request_history: List[Dict[str, Any]] = []
        self._max_history_size = 10000
        self._active_recordings: Dict[str, datetime] = {}

        # Metrics collection
        self.metrics_collection = get_metrics_collection()

        # Available voices
        self._available_voices: Dict[str, Dict[str, Any]] = {}
        self._load_voices()

        # Load configuration
        self._load_configuration()

        logger.info("✅ Speech API initialized with TTS/STT integration")

    def _load_configuration(self):
        """Load configuration from environment or database."""
        self.config = {
            "default_voice": "en-US-Female-1",
            "default_language": "en-US",
            "default_rate": 150,
            "enable_cache": True,
            "cache_max_size": 100,
            "cache_ttl_seconds": 3600,
            "enable_metrics": True,
            "enable_audit_log": True,
            "enable_vad": True,
            "vad_silence_timeout": 1.5,
            "max_recording_duration": 300,
            "sample_rate": 16000,
            "audio_format": "wav",
            "stt_model": "base",
            "tts_engine": "default",
            "wake_word_sensitivity": 0.5,
            "noise_filtering": True,
            "auto_gain_control": True,
        }

        # Try to load from MongoDB
        if mongo_client.enabled:
            try:
                config_collection = mongo_client.get_collection("config")
                if config_collection is not None:
                    db_config = config_collection.find_one({"type": "speech_api"})
                    if db_config:
                        self.config.update(db_config.get("config", {}))
                        logger.info("Loaded Speech API configuration from database")
            except Exception as e:
                logger.warning(f"Failed to load config from MongoDB: {e}")

        # Update cache settings
        if self.config["enable_cache"]:
            self._audio_cache = AudioCache(
                max_size=self.config["cache_max_size"],
                ttl_seconds=self.config["cache_ttl_seconds"],
            )

    def _load_voices(self):
        """Load available voices."""
        # Default voices
        self._available_voices = {
            "en-US-Female-1": {
                "name": "Emma",
                "gender": "female",
                "language": "en-US",
                "description": "Natural female voice",
            },
            "en-US-Male-1": {
                "name": "James",
                "gender": "male",
                "language": "en-US",
                "description": "Natural male voice",
            },
            "en-GB-Female-1": {
                "name": "Sophie",
                "gender": "female",
                "language": "en-GB",
                "description": "British female voice",
            },
            "en-GB-Male-1": {
                "name": "Oliver",
                "gender": "male",
                "language": "en-GB",
                "description": "British male voice",
            },
            "es-ES-Female-1": {
                "name": "Lucia",
                "gender": "female",
                "language": "es-ES",
                "description": "Spanish female voice",
            },
            "fr-FR-Female-1": {
                "name": "Claire",
                "gender": "female",
                "language": "fr-FR",
                "description": "French female voice",
            },
            "de-DE-Female-1": {
                "name": "Anna",
                "gender": "female",
                "language": "de-DE",
                "description": "German female voice",
            },
            "ja-JP-Female-1": {
                "name": "Sakura",
                "gender": "female",
                "language": "ja-JP",
                "description": "Japanese female voice",
            },
        }

    def _save_audio_file(
        self, audio_data: bytes, format: str, prefix: str = "recording"
    ) -> Path:
        """Save audio data to file."""
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{format}"
        filepath = self.audio_outputs_dir / filename
        filepath.write_bytes(audio_data)
        return filepath

    async def _log_metrics(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        size_bytes: int = 0,
        error: Optional[str] = None,
    ):
        """Log metrics to MongoDB."""
        if not self.config["enable_metrics"]:
            return

        if not mongo_client.enabled or self.metrics_collection is None:
            return

        try:
            metric = {
                "timestamp": datetime.now(),
                "operation": f"speech_{operation}",
                "duration_ms": duration_ms,
                "success": success,
                "size_bytes": size_bytes,
                "error": error,
            }

            from core.utils.task_registry import create_task

            create_task(
                self.metrics_collection.insert_one(metric), name="speechapi.log_metric"
            )

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    async def _log_audit(
        self,
        action: str,
        user_id: Optional[str],
        success: bool,
        details: Dict[str, Any] = None,
    ):
        """Log audit trail to MongoDB."""
        if not self.config["enable_audit_log"]:
            return

        audit_collection = mongo_client.get_collection("audit_log")
        if not mongo_client.enabled or audit_collection is None:
            return

        try:
            audit_entry = {
                "timestamp": datetime.now(),
                "action": f"speech_{action}",
                "user_id": user_id,
                "success": success,
                "details": details or {},
            }

            from core.utils.task_registry import create_task

            create_task(
                audit_collection.insert_one(audit_entry), name="speechapi.log_audit"
            )

        except Exception as e:
            logger.error(f"Failed to log audit: {e}")

    def _trim_history(self):
        """Trim request history to max size."""
        if len(self._request_history) > self._max_history_size:
            self._request_history = self._request_history[-self._max_history_size :]

    # =========================
    # LISTEN / RECORD
    # =========================

    async def listen(self, request: ListenRequest) -> ListenResponse:
        """Listen to audio input with VAD support."""
        start_time = datetime.now()
        recording_id = str(uuid.uuid4())

        try:
            self.request_count += 1

            # Apply noise filtering if enabled
            if self.config["noise_filtering"]:
                self.noise_filter.enable()

            # Start recording
            self._active_recordings[recording_id] = start_time

            # Listen with VAD if enabled
            audio_data = None
            has_speech = False
            speech_duration = None

            if request.enable_vad and hasattr(self.listener, "listen_vad"):
                audio_data, speech_info = await self.listener.listen_vad(
                    timeout=request.duration, silence_timeout=request.silence_timeout
                )
                has_speech = speech_info.get("has_speech", False)
                speech_duration = speech_info.get("speech_duration")
            else:
                audio_data = await self.listener.listen(request.duration)
                has_speech = audio_data is not None and len(audio_data) > 0

            # Save to file if requested
            file_path = None
            if request.save_to_file and audio_data:
                file_path = str(
                    self._save_audio_file(audio_data, request.format.value, "recording")
                )

            # Prepare response
            response = ListenResponse(
                audio_data=(
                    base64.b64encode(audio_data).decode() if audio_data else None
                ),
                audio_format=request.format.value,
                duration=request.duration,
                sample_rate=request.sample_rate,
                channels=request.channels,
                has_speech=has_speech,
                speech_duration=speech_duration,
                recording_id=recording_id,
                timestamp=datetime.now().isoformat(),
                file_path=file_path,
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics(
                "listen", duration_ms, True, len(audio_data) if audio_data else 0
            )
            await self._log_audit(
                "listen", request.user_id, True, {"duration": request.duration}
            )

            # Update history
            self._request_history.append(
                {
                    "operation": "listen",
                    "recording_id": recording_id,
                    "duration": request.duration,
                    "timestamp": datetime.now(),
                    "success": True,
                }
            )
            self._trim_history()

            logger.info(f"Listen completed: {request.duration}s, speech: {has_speech}")
            return response

        except Exception as e:
            logger.error(f"Listen error: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics("listen", duration_ms, False, error=str(e))

            self._request_history.append(
                {
                    "operation": "listen",
                    "recording_id": recording_id,
                    "timestamp": datetime.now(),
                    "success": False,
                    "error": str(e),
                }
            )
            self._trim_history()

            raise
        finally:
            self._active_recordings.pop(recording_id, None)

    # =========================
    # TRANSCRIBE (STT)
    # =========================

    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """Transcribe audio to text using STT."""
        start_time = datetime.now()
        transcription_id = str(uuid.uuid4())

        try:
            self.request_count += 1

            # Load audio data
            audio_bytes = None
            if request.audio_data:
                audio_bytes = base64.b64decode(request.audio_data)
            elif request.audio_file:
                file_path = Path(request.audio_file)
                if file_path.exists():
                    audio_bytes = file_path.read_bytes()
                else:
                    raise ValueError(f"Audio file not found: {request.audio_file}")
            else:
                raise ValueError("Either audio_data or audio_file must be provided")

            # Transcribe
            result = await self.stt.transcribe(
                audio_bytes,
                language=request.language.value,
                mode=request.mode.value,
                enable_diarization=request.enable_diarization,
            )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            response = TranscribeResponse(
                text=result.get("text", ""),
                confidence=result.get("confidence", 0.0),
                language=request.language.value,
                mode=request.mode.value,
                duration=result.get("duration", 0),
                processing_time_ms=duration_ms,
                transcription_id=transcription_id,
                timestamp=datetime.now().isoformat(),
                alternatives=result.get("alternatives"),
                speaker_segments=result.get("speaker_segments"),
            )

            await self._log_metrics("transcribe", duration_ms, True, len(audio_bytes))
            await self._log_audit(
                "transcribe",
                request.user_id,
                True,
                {"text_length": len(response.text), "language": request.language.value},
            )

            logger.info(
                f"Transcription completed: {len(response.text)} chars, confidence: {response.confidence:.2f}"
            )
            return response

        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics("transcribe", duration_ms, False, error=str(e))
            raise

    # =========================
    # SPEAK (TTS)
    # =========================

    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        """Convert text to speech with caching."""
        start_time = datetime.now()
        speech_id = str(uuid.uuid4())

        try:
            self.request_count += 1

            # Select voice
            voice_name = request.voice
            if not voice_name:
                # Find voice matching gender and language
                for voice_id, voice_info in self._available_voices.items():
                    if (
                        voice_info["gender"] == request.gender.value
                        and voice_info["language"] == request.language.value
                    ):
                        voice_name = voice_id
                        break

                if not voice_name:
                    voice_name = self.config["default_voice"]

            # Check cache
            audio_data = None
            if self.config["enable_cache"]:
                cache_key = self._audio_cache._get_cache_key(
                    request.text, voice_name, request.rate, request.pitch
                )
                audio_data = self._audio_cache.get(cache_key)

            # Generate speech if not cached
            if not audio_data:
                audio_data = await self.speaker.speak(
                    text=request.text,
                    voice=voice_name,
                    rate=request.rate,
                    pitch=request.pitch,
                    volume=request.volume,
                    format=request.format.value,
                )

                # Save to cache
                if self.config["enable_cache"]:
                    cache_key = self._audio_cache._get_cache_key(
                        request.text, voice_name, request.rate, request.pitch
                    )
                    self._audio_cache.set(cache_key, audio_data)

            # Save to file if requested
            file_path = None
            if request.save_to_file and audio_data:
                file_path = str(
                    self._save_audio_file(audio_data, request.format.value, "speech")
                )

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            duration_seconds = (
                len(request.text) / request.rate * 60
            )  # Approximate duration

            response = SpeakResponse(
                status="success",
                audio_data=(
                    base64.b64encode(audio_data).decode() if audio_data else None
                ),
                audio_format=request.format.value,
                duration=duration_seconds,
                characters=len(request.text),
                speech_id=speech_id,
                timestamp=datetime.now().isoformat(),
                file_path=file_path,
            )

            await self._log_metrics(
                "speak", duration_ms, True, len(audio_data) if audio_data else 0
            )
            await self._log_audit(
                "speak",
                request.user_id,
                True,
                {"text_length": len(request.text), "voice": voice_name},
            )

            logger.info(
                f"Speech generated: {len(request.text)} chars, duration: {duration_seconds:.2f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Speak error: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_metrics("speak", duration_ms, False, error=str(e))
            raise

    # =========================
    # WAKE WORD DETECTION
    # =========================

    async def detect_wake_word(self, request: WakeWordRequest) -> WakeWordResponse:
        """Detect wake word in audio stream."""
        start_time = datetime.now()

        try:
            self.request_count += 1

            # Configure wake word detector
            self.wakeword.set_sensitivity(request.sensitivity)

            # Detect wake word
            result = await self.wakeword.detect(
                wake_word=request.word, timeout=request.timeout
            )

            detection_time = (datetime.now() - start_time).total_seconds()

            response = WakeWordResponse(
                detected=result.get("detected", False),
                word=request.word,
                confidence=result.get("confidence", 0.0),
                detection_time=detection_time,
                timestamp=datetime.now().isoformat(),
            )

            logger.info(
                f"Wake word detection: {response.detected}, confidence: {response.confidence:.2f}"
            )
            return response

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            raise

    # =========================
    # DUPLEX COMMUNICATION
    # =========================

    async def start_duplex(self, session_id: str) -> Dict[str, Any]:
        """Start duplex audio session."""
        try:
            await self.duplex_controller.start(session_id)

            return {
                "status": "success",
                "session_id": session_id,
                "message": "Duplex session started",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Start duplex error: {e}")
            raise

    async def stop_duplex(self, session_id: str) -> Dict[str, Any]:
        """Stop duplex audio session."""
        try:
            await self.duplex_controller.stop(session_id)

            return {
                "status": "success",
                "session_id": session_id,
                "message": "Duplex session stopped",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Stop duplex error: {e}")
            raise

    async def process_duplex_audio(
        self, session_id: str, audio_chunk: bytes
    ) -> AsyncIterator[bytes]:
        """Process audio chunk in duplex mode."""
        async for response_chunk in self.duplex_controller.process(
            session_id, audio_chunk
        ):
            yield response_chunk

    # =========================
    # VOICE COMMANDS
    # =========================

    async def recognize_command(self, audio_data: str) -> SpeechCommand:
        """Recognize voice command from audio."""
        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_data)

            # Transcribe first
            transcription = await self.stt.transcribe(audio_bytes)

            # Extract intent and parameters
            result = await self._extract_intent(transcription.get("text", ""))

            return SpeechCommand(
                command=result.get("command", ""),
                intent=result.get("intent", "unknown"),
                confidence=result.get("confidence", 0.0),
                parameters=result.get("parameters", {}),
            )

        except Exception as e:
            logger.error(f"Command recognition error: {e}")
            raise

    async def _extract_intent(self, text: str) -> Dict[str, Any]:
        """Extract intent and parameters from text."""
        # Simple keyword-based intent extraction
        # In production, this would use an NLP model
        text_lower = text.lower()

        intents = {
            "search": ["search", "find", "look up", "google"],
            "open": ["open", "launch", "start"],
            "close": ["close", "exit", "quit"],
            "play": ["play", "listen", "music"],
            "stop": ["stop", "pause", "cease"],
            "volume": ["volume", "louder", "quieter", "mute"],
            "weather": ["weather", "temperature", "forecast"],
            "time": ["time", "clock", "what time"],
            "date": ["date", "day", "today"],
            "reminder": ["remind", "reminder", "alert", "notify"],
        }

        for intent, keywords in intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        "intent": intent,
                        "command": text,
                        "confidence": 0.8,
                        "parameters": {},
                    }

        return {
            "intent": "unknown",
            "command": text,
            "confidence": 0.3,
            "parameters": {},
        }

    # =========================
    # VOICE MANAGEMENT
    # =========================

    async def get_voices(
        self, language: Optional[VoiceLanguage] = None
    ) -> List[Dict[str, Any]]:
        """Get available voices."""
        voices = []

        for voice_id, voice_info in self._available_voices.items():
            if language and voice_info["language"] != language.value:
                continue

            voices.append(
                {
                    "id": voice_id,
                    "name": voice_info["name"],
                    "gender": voice_info["gender"],
                    "language": voice_info["language"],
                    "description": voice_info.get("description", ""),
                }
            )

        return voices

    async def preview_voice(
        self, voice_id: str, text: str = "Hello, this is a voice preview."
    ) -> bytes:
        """Preview a voice with sample text."""
        request = SpeakRequest(text=text, voice=voice_id, save_to_file=False)

        response = await self.speak(request)
        if response.audio_data:
            return base64.b64decode(response.audio_data)
        return b""

    # =========================
    # AUDIO PROCESSING
    # =========================

    async def filter_noise(self, audio_data: str) -> str:
        """Apply noise filtering to audio."""
        try:
            audio_bytes = base64.b64decode(audio_data)
            filtered_bytes = self.noise_filter.process(audio_bytes)
            return base64.b64encode(filtered_bytes).decode()
        except Exception as e:
            logger.error(f"Noise filtering error: {e}")
            raise

    async def convert_format(self, audio_data: str, target_format: AudioFormat) -> str:
        """Convert audio format."""
        try:
            audio_bytes = base64.b64decode(audio_data)

            # Format conversion logic would go here
            # This is a placeholder for actual conversion

            return base64.b64encode(audio_bytes).decode()
        except Exception as e:
            logger.error(f"Format conversion error: {e}")
            raise

    # =========================
    # STATISTICS
    # =========================

    async def get_stats(self) -> Dict[str, Any]:
        """Get API statistics."""
        try:
            # Calculate success rate
            total_requests = len(self._request_history)
            successful_requests = sum(
                1 for r in self._request_history if r.get("success", False)
            )
            success_rate = (
                (successful_requests / total_requests) if total_requests > 0 else 0
            )

            # Get cache stats
            cache_stats = (
                self._audio_cache.get_stats()
                if self.config["enable_cache"]
                else {"enabled": False}
            )

            # Get metrics from MongoDB
            metrics_data = {}
            if mongo_client.enabled and self.metrics_collection is not None:
                total_operations = await self.metrics_collection.count_documents({})

                pipeline = [
                    {
                        "$group": {
                            "_id": "$operation",
                            "avg_duration": {"$avg": "$duration_ms"},
                            "count": {"$sum": 1},
                            "success_rate": {"$avg": {"$cond": ["$success", 1, 0]}},
                        }
                    }
                ]

                agg_result = list(
                    await self.metrics_collection.aggregate(pipeline).to_list(length=10)
                )
                metrics_data["operation_stats"] = agg_result
                metrics_data["total_operations"] = total_operations

            return {
                "request_count": self.request_count,
                "success_rate": success_rate,
                "total_requests_history": total_requests,
                "active_recordings": len(self._active_recordings),
                "cache": cache_stats,
                "available_voices": len(self._available_voices),
                "metrics": metrics_data,
                "config": {
                    "enable_cache": self.config["enable_cache"],
                    "enable_vad": self.config["enable_vad"],
                    "default_voice": self.config["default_voice"],
                    "stt_model": self.config["stt_model"],
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {
                "request_count": self.request_count,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # =========================
    # CACHE MANAGEMENT
    # =========================

    def clear_cache(self) -> Dict[str, Any]:
        """Clear audio cache."""
        cache_stats = self._audio_cache.get_stats()
        self._audio_cache.clear()
        logger.info(f"Cleared {cache_stats['size']} cached audio files")
        return {"cleared": cache_stats["size"], "timestamp": datetime.now().isoformat()}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.config["enable_cache"]:
            return self._audio_cache.get_stats()
        return {"enabled": False, "message": "Cache is disabled"}

    # =========================
    # HEALTH CHECK
    # =========================

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {},
        }

        # Check listener
        try:
            if hasattr(self.listener, "health_check"):
                listener_status = self.listener.health_check()
            else:
                listener_status = {"status": "healthy"}
            status["components"]["listener"] = listener_status
        except Exception as e:
            status["components"]["listener"] = {"status": "unhealthy", "error": str(e)}
            status["status"] = "degraded"

        # Check speaker
        try:
            if hasattr(self.speaker, "health_check"):
                speaker_status = self.speaker.health_check()
            else:
                speaker_status = {"status": "healthy"}
            status["components"]["speaker"] = speaker_status
        except Exception as e:
            status["components"]["speaker"] = {"status": "unhealthy", "error": str(e)}
            status["status"] = "degraded"

        # Check STT
        try:
            if hasattr(self.stt, "health_check"):
                stt_status = self.stt.health_check()
            else:
                stt_status = {"status": "healthy"}
            status["components"]["stt"] = stt_status
        except Exception as e:
            status["components"]["stt"] = {"status": "unhealthy", "error": str(e)}
            status["status"] = "degraded"

        # Check MongoDB
        if mongo_client.enabled:
            if mongo_client.status.value == "connected":
                status["components"]["mongodb"] = {
                    "status": "healthy",
                    "database": mongo_client.database_name,
                }
            else:
                status["components"]["mongodb"] = {
                    "status": "unhealthy",
                    "error": f"Status: {mongo_client.status.value}",
                }
                status["status"] = "degraded"

        # Check cache
        status["components"]["cache"] = {
            "status": "healthy",
            "enabled": self.config["enable_cache"],
            "size": (
                self._audio_cache.get_stats()["size"]
                if self.config["enable_cache"]
                else 0
            ),
        }

        return status

    # =========================    # SHUTDOWN
    # =========================

    async def shutdown(self):
        """Gracefully shutdown the API."""
        logger.info("Shutting down Speech API...")

        # Stop all duplex sessions
        if hasattr(self.duplex_controller, "shutdown"):
            await self.duplex_controller.shutdown()

        # Clear cache
        self.clear_cache()

        # Save any pending metrics
        if mongo_client.enabled:
            logger.info("Saving final metrics to MongoDB...")

        # Close audio devices
        if hasattr(self.listener, "close"):
            await self.listener.close()
        if hasattr(self.speaker, "close"):
            await self.speaker.close()

        logger.info("Speech API shutdown complete")


# =========================
# GLOBAL INSTANCE
# =========================

speech_api = SpeechApi()


__all__ = [
    "SpeechApi",
    "AudioFormat",
    "VoiceGender",
    "VoiceLanguage",
    "SpeechMode",
    "ListenRequest",
    "ListenResponse",
    "TranscribeRequest",
    "TranscribeResponse",
    "SpeakRequest",
    "SpeakResponse",
    "WakeWordRequest",
    "WakeWordResponse",
    "SpeechCommand",
    "StreamAudioChunk",
    "speech_api",
]
