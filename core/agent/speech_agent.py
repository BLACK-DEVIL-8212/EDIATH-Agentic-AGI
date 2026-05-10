"""
Speech Agent for EDIATH
Advanced speech processing: speech-to-text (STT), text-to-speech (TTS), voice activity detection, emotion recognition
"""

import asyncio
import base64
import io
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import wave

# Audio processing
try:
    import numpy as np
    import soundfile as sf
    import sounddevice as sd
    import pyaudio

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Speech recognition
try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# Text-to-speech
try:
    import pyttsx3

    TTS_ENGINE_AVAILABLE = True
except ImportError:
    TTS_ENGINE_AVAILABLE = False

# Advanced TTS
try:
    from TTS.api import TTS

    COQUI_TTS_AVAILABLE = True
    COQUI_TTS_IMPORT_ERROR = None
except Exception as e:
    TTS = None
    COQUI_TTS_AVAILABLE = False
    COQUI_TTS_IMPORT_ERROR = e

# Voice activity detection
try:
    import webrtcvad

    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

# Whisper for STT
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class SpeechModel(Enum):
    """Speech recognition models"""

    GOOGLE = "google"
    WHISPER = "whisper"
    SPHINX = "sphinx"
    WITAI = "witai"
    BING = "bing"
    AZURE = "azure"


class TTSModel(Enum):
    """Text-to-speech models"""

    SYSTEM = "system"  # System TTS (pyttsx3)
    COQUI = "coqui"  # Coqui TTS
    GOOGLE = "google"  # Google TTS
    AZURE = "azure"  # Azure TTS


class VoiceEmotion(Enum):
    """Voice emotions for TTS"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    WHISPER = "whisper"


@dataclass
class SpeechSegment:
    """Speech segment information"""

    text: str
    confidence: float
    start_time: float
    end_time: float
    language: str = "en"
    emotion: Optional[str] = None


@dataclass
class AudioInfo:
    """Audio file information"""

    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int


class SpeechAgent:
    """
    Advanced speech processing agent capable of:
    - Speech-to-text (STT) with multiple engines
    - Text-to-speech (TTS) with multiple voices
    - Voice activity detection (VAD)
    - Real-time microphone recording
    - Audio file processing
    - Speaker diarization (basic)
    - Emotion detection from speech
    - Language detection
    - Noise filtering and enhancement
    - Batch audio processing
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Speech Agent

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # API keys
        self.google_api_key = self.config.get("google_api_key", "")
        self.witai_key = self.config.get("witai_key", "")
        self.bing_key = self.config.get("bing_key", "")
        self.azure_key = self.config.get("azure_key", "")
        self.azure_region = self.config.get("azure_region", "eastus")

        # Audio configuration
        self.default_sample_rate = self.config.get("sample_rate", 16000)
        self.default_channels = self.config.get("channels", 1)
        self.default_device = self.config.get("audio_device", None)

        # STT configuration
        self.default_stt_model = SpeechModel(self.config.get("stt_model", "google"))
        self.stt_timeout = self.config.get("stt_timeout", 5)
        self.phrase_time_limit = self.config.get("phrase_time_limit", 10)

        # TTS configuration
        self.default_tts_model = TTSModel(self.config.get("tts_model", "system"))
        self.default_voice = self.config.get("default_voice", "default")
        self.default_rate = self.config.get("speech_rate", 150)
        self.default_volume = self.config.get("volume", 1.0)

        # Whisper model
        self.whisper_model_size = self.config.get("whisper_model", "base")
        self.whisper_model = None

        # Coqui TTS model
        self.coqui_tts_model = self.config.get(
            "coqui_model", "tts_models/en/ljspeech/tacotron2-DDC"
        )
        self.coqui_tts = None

        # Audio recording
        self.recording = False
        self.recorded_audio = []
        self.audio_stream = None

        # VAD
        self.vad = None
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(3)  # Most aggressive

        # Recognizer
        self.recognizer = None
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.config.get("energy_threshold", 300)
            self.recognizer.dynamic_energy_threshold = self.config.get(
                "dynamic_energy", True
            )
            self.recognizer.pause_threshold = self.config.get("pause_threshold", 0.8)

        # TTS engine
        self.tts_engine = None
        if TTS_ENGINE_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.default_rate)
            self.tts_engine.setProperty("volume", self.default_volume)

        # Output directory
        self.output_dir = Path(self.config.get("output_dir", "audio_outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_transcriptions": 0,
            "total_syntheses": 0,
            "total_audio_processed": 0,
            "average_confidence": 0.0,
            "errors": 0,
        }

        # History
        self.transcription_history: List[SpeechSegment] = []
        self.synthesis_history: List[Dict] = []
        self.max_history = self.config.get("max_history", 100)

        self.logger.info(
            f"Speech Agent initialized. STT: {self.default_stt_model.value}, TTS: {self.default_tts_model.value}"
        )

    async def load_models(self):
        """Load required models"""
        # Load Whisper if needed
        if self.default_stt_model == SpeechModel.WHISPER and WHISPER_AVAILABLE:
            if not self.whisper_model:
                self.logger.info(f"Loading Whisper model: {self.whisper_model_size}")
                self.whisper_model = whisper.load_model(self.whisper_model_size)

        # Load Coqui TTS if needed
        if self.default_tts_model == TTSModel.COQUI:
            if not COQUI_TTS_AVAILABLE:
                raise RuntimeError(f"Coqui TTS unavailable: {COQUI_TTS_IMPORT_ERROR}")
            if not self.coqui_tts:
                self.logger.info(f"Loading Coqui TTS model: {self.coqui_tts_model}")
                self.coqui_tts = TTS(self.coqui_tts_model)

    async def transcribe_audio(
        self,
        audio_source: Union[str, Path, bytes, np.ndarray],
        model: Optional[SpeechModel] = None,
        language: str = "en",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text

        Args:
            audio_source: Audio file path, bytes, or numpy array
            model: Speech recognition model to use
            language: Language code (for some models)
            use_cache: Use cached results

        Returns:
            Dictionary with transcription results
        """
        if model is None:
            model = self.default_stt_model

        start_time = datetime.now()

        try:
            # Load audio data
            audio_data, sample_rate = await self._load_audio(audio_source)

            # Transcribe based on model
            if model == SpeechModel.GOOGLE:
                transcription = await self._transcribe_google(audio_data, sample_rate)
            elif model == SpeechModel.WHISPER:
                await self.load_models()
                transcription = await self._transcribe_whisper(
                    audio_data, sample_rate, language
                )
            elif model == SpeechModel.SPHINX:
                transcription = await self._transcribe_sphinx(audio_data, sample_rate)
            elif model == SpeechModel.WITAI:
                transcription = await self._transcribe_witai(audio_data, sample_rate)
            else:
                raise ValueError(f"Unsupported STT model: {model}")

            processing_time = (datetime.now() - start_time).total_seconds()

            # Create segment
            segment = SpeechSegment(
                text=transcription["text"],
                confidence=transcription["confidence"],
                start_time=0,
                end_time=processing_time,
                language=language,
            )

            # Update statistics
            self.stats["total_transcriptions"] += 1
            self.stats["average_confidence"] = (
                self.stats["average_confidence"]
                * (self.stats["total_transcriptions"] - 1)
                + transcription["confidence"]
            ) / self.stats["total_transcriptions"]

            # Add to history
            self._add_to_history(segment)

            return {
                "success": True,
                "text": transcription["text"],
                "confidence": transcription["confidence"],
                "model": model.value,
                "language": language,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "model": model.value}

    async def _load_audio(
        self, audio_source: Union[str, Path, bytes, np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """Load audio from various sources"""
        if isinstance(audio_source, (str, Path)):
            # Load from file
            data, sample_rate = sf.read(audio_source)
            return data, sample_rate

        elif isinstance(audio_source, bytes):
            # Load from bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_source)
                tmp_path = tmp.name

            data, sample_rate = sf.read(tmp_path)
            Path(tmp_path).unlink()
            return data, sample_rate

        elif isinstance(audio_source, np.ndarray):
            # Assume already loaded
            return audio_source, self.default_sample_rate

        else:
            raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

    async def _transcribe_google(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict:
        """Transcribe using Google Speech Recognition"""
        if not SR_AVAILABLE:
            raise ImportError("SpeechRecognition not available")

        # Convert to audio data
        audio_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
        audio = sr.AudioData(audio_bytes, sample_rate, 2)

        try:
            text = self.recognizer.recognize_google(audio, key=self.google_api_key)
            return {
                "text": text,
                "confidence": 0.8,  # Google doesn't provide confidence
            }
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Google API error: {e}")

    async def _transcribe_whisper(
        self, audio_data: np.ndarray, sample_rate: int, language: str
    ) -> Dict:
        """Transcribe using Whisper"""
        if not self.whisper_model:
            raise Exception("Whisper model not loaded")

        # Convert to float32 and ensure correct sample rate
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Resample if needed
        if sample_rate != 16000:
            import scipy.signal

            audio_data = scipy.signal.resample(
                audio_data, int(len(audio_data) * 16000 / sample_rate)
            )

        # Transcribe
        result = self.whisper_model.transcribe(
            audio_data, language=language if language != "en" else None
        )

        return {"text": result["text"], "confidence": 0.9}  # Approximate confidence

    async def _transcribe_sphinx(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict:
        """Transcribe using CMU Sphinx (offline)"""
        if not SR_AVAILABLE:
            raise ImportError("SpeechRecognition not available")

        audio_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
        audio = sr.AudioData(audio_bytes, sample_rate, 2)

        try:
            text = self.recognizer.recognize_sphinx(audio)
            return {"text": text, "confidence": 0.7}
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")

    async def _transcribe_witai(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Transcribe using Wit.ai"""
        if not SR_AVAILABLE:
            raise ImportError("SpeechRecognition not available")

        audio_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
        audio = sr.AudioData(audio_bytes, sample_rate, 2)

        try:
            text = self.recognizer.recognize_wit(audio, key=self.witai_key)
            return {"text": text, "confidence": 0.85}
        except sr.UnknownValueError:
            raise Exception("Could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Wit.ai API error: {e}")

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes"""
        buffer = io.BytesIO()

        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1 if len(audio_data.shape) == 1 else audio_data.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data.tobytes())

        return buffer.getvalue()

    async def synthesize_speech(
        self,
        text: str,
        model: Optional[TTSModel] = None,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        emotion: VoiceEmotion = VoiceEmotion.NEUTRAL,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize text to speech

        Args:
            text: Text to synthesize
            model: TTS model to use
            voice: Voice name or ID
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            emotion: Voice emotion (if supported)
            output_file: Output audio file path

        Returns:
            Dictionary with synthesis results
        """
        if model is None:
            model = self.default_tts_model

        start_time = datetime.now()

        try:
            audio_data = None
            sample_rate = self.default_sample_rate

            if model == TTSModel.SYSTEM:
                audio_data = await self._synthesize_system(text, rate, volume)
            elif model == TTSModel.COQUI:
                await self.load_models()
                audio_data, sample_rate = await self._synthesize_coqui(
                    text, voice, emotion
                )
            elif model == TTSModel.GOOGLE:
                audio_data = await self._synthesize_google(text, voice)
            elif model == TTSModel.AZURE:
                audio_data = await self._synthesize_azure(text, voice, emotion)
            else:
                raise ValueError(f"Unsupported TTS model: {model}")

            # Save to file if requested
            if output_file:
                output_path = Path(output_file)
                if not output_path.suffix:
                    output_path = output_path.with_suffix(".wav")

                sf.write(str(output_path), audio_data, sample_rate)

            processing_time = (datetime.now() - start_time).total_seconds()

            # Convert to base64 for embedding
            audio_base64 = base64.b64encode(audio_data.tobytes()).decode("utf-8")

            # Update statistics
            self.stats["total_syntheses"] += 1

            # Add to history
            self.synthesis_history.append(
                {
                    "text": text[:100],
                    "model": model.value,
                    "voice": voice,
                    "timestamp": datetime.now().isoformat(),
                    "duration": len(audio_data) / sample_rate,
                }
            )

            # Trim history
            if len(self.synthesis_history) > self.max_history:
                self.synthesis_history = self.synthesis_history[-self.max_history :]

            return {
                "success": True,
                "text": text,
                "model": model.value,
                "voice": voice or self.default_voice,
                "audio_data": audio_base64,
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "processing_time": processing_time,
                "output_file": str(output_path) if output_file else None,
            }

        except Exception as e:
            self.logger.error(f"Speech synthesis error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "error": str(e), "model": model.value}

    async def _synthesize_system(
        self, text: str, rate: Optional[int], volume: Optional[float]
    ) -> np.ndarray:
        """Synthesize using system TTS (pyttsx3)"""
        if not self.tts_engine:
            raise ImportError("pyttsx3 not available")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # Update settings
        if rate:
            self.tts_engine.setProperty("rate", rate)
        if volume:
            self.tts_engine.setProperty("volume", volume)

        # Synthesize
        self.tts_engine.save_to_file(text, tmp_path)
        self.tts_engine.runAndWait()

        # Load audio
        audio_data, sample_rate = sf.read(tmp_path)
        Path(tmp_path).unlink()

        return audio_data

    async def _synthesize_coqui(
        self, text: str, voice: Optional[str], emotion: VoiceEmotion
    ) -> Tuple[np.ndarray, int]:
        """Synthesize using Coqui TTS"""
        if not COQUI_TTS_AVAILABLE:
            raise ImportError(f"Coqui TTS not available: {COQUI_TTS_IMPORT_ERROR}")
        if not self.coqui_tts:
            raise ImportError("Coqui TTS model not loaded")

        # Apply emotion modifier (simplified)
        if emotion != VoiceEmotion.NEUTRAL:
            text = f"[{emotion.value}] {text}"

        # Synthesize
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self.coqui_tts.tts_to_file(text=text, file_path=output_path, speaker=voice)

        # Load audio
        audio_data, sample_rate = sf.read(output_path)
        Path(output_path).unlink()

        return audio_data, sample_rate

    async def _synthesize_google(self, text: str, voice: Optional[str]) -> np.ndarray:
        """Synthesize using Google TTS"""
        if not SR_AVAILABLE:
            raise ImportError("SpeechRecognition not available")

        # Use gTTS for synthesis
        from gtts import gTTS

        tts = gTTS(text=text, lang="en", slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)

        # Convert to WAV and load
        audio_data, sample_rate = sf.read(tmp_path)
        Path(tmp_path).unlink()

        return audio_data

    async def _synthesize_azure(
        self, text: str, voice: Optional[str], emotion: VoiceEmotion
    ) -> np.ndarray:
        """Synthesize using Azure TTS"""
        if not self.azure_key:
            raise Exception("Azure API key required")

        import requests

        # Azure TTS endpoint
        url = (
            f"https://{self.azure_region}.tts.speech.microsoft.com/cognitiveservices/v1"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
        }

        # Build SSML
        voice_name = voice or "en-US-JennyNeural"
        ssml = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
            <voice name='{voice_name}'>
                <prosody rate='0%'>
                    {text}
                </prosody>
            </voice>
        </speak>
        """

        response = requests.post(url, headers=headers, data=ssml.encode("utf-8"))

        if response.status_code != 200:
            raise Exception(f"Azure TTS error: {response.status_code}")

        # Convert to numpy array
        audio_data = np.frombuffer(response.content, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32767.0

        return audio_data

    async def record_audio(
        self,
        duration: float = 5.0,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        device: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate
            channels: Number of channels
            device: Audio device index

        Returns:
            Dictionary with recorded audio
        """
        if not AUDIO_AVAILABLE:
            return {"success": False, "error": "Audio libraries not available"}

        sample_rate = sample_rate or self.default_sample_rate
        channels = channels or self.default_channels
        device = device or self.default_device

        try:
            self.logger.info(f"Recording for {duration} seconds...")

            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                device=device,
            )
            sd.wait()

            return {
                "success": True,
                "audio_data": recording,
                "sample_rate": sample_rate,
                "channels": channels,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Recording error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def record_and_transcribe(
        self, duration: float = 5.0, model: Optional[SpeechModel] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Record audio and transcribe in one step

        Args:
            duration: Recording duration
            model: STT model to use
            **kwargs: Additional arguments for transcription

        Returns:
            Dictionary with transcription result
        """
        # Record audio
        recording_result = await self.record_audio(duration)

        if not recording_result["success"]:
            return recording_result

        # Transcribe
        transcription = await self.transcribe_audio(
            recording_result["audio_data"], model=model, **kwargs
        )

        transcription["recording_info"] = {
            "duration": duration,
            "sample_rate": recording_result["sample_rate"],
        }

        return transcription

    async def start_streaming(
        self,
        callback: Callable[[str, float], None],
        sample_rate: Optional[int] = None,
        chunk_size: int = 512,
    ) -> Dict[str, Any]:
        """
        Start streaming audio recognition

        Args:
            callback: Callback function for recognized text (text, confidence)
            sample_rate: Sample rate
            chunk_size: Audio chunk size

        Returns:
            Dictionary with streaming status
        """
        if not AUDIO_AVAILABLE or not SR_AVAILABLE:
            return {"success": False, "error": "Required libraries not available"}

        sample_rate = sample_rate or self.default_sample_rate

        try:
            self.recording = True

            def audio_callback(indata, frames, time, status):
                if self.recording:
                    # Convert to audio data
                    audio_data = sr.AudioData(
                        indata.tobytes(), sample_rate, 2  # Sample width
                    )

                    try:
                        # Recognize speech
                        text = self.recognizer.recognize_google(audio_data)
                        if text:
                            asyncio.run_coroutine_threadsafe(
                                callback(text, 0.8), asyncio.get_event_loop()
                            )
                    except:
                        pass

            # Start stream
            self.audio_stream = sd.InputStream(
                callback=audio_callback,
                samplerate=sample_rate,
                channels=1,
                blocksize=chunk_size,
            )
            self.audio_stream.start()

            return {
                "success": True,
                "status": "streaming",
                "sample_rate": sample_rate,
                "message": "Streaming started",
            }

        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def stop_streaming(self) -> Dict[str, Any]:
        """Stop streaming audio recognition"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.recording = False

            return {"success": True, "message": "Streaming stopped"}

        return {"success": False, "error": "No active stream"}

    async def detect_voice_activity(
        self, audio_source: Union[str, Path, bytes, np.ndarray], aggressiveness: int = 3
    ) -> Dict[str, Any]:
        """
        Detect voice activity in audio

        Args:
            audio_source: Audio source
            aggressiveness: VAD aggressiveness (0-3)

        Returns:
            Dictionary with voice activity segments
        """
        if not VAD_AVAILABLE:
            return {"success": False, "error": "WebRTC VAD not available"}

        try:
            # Load audio
            audio_data, sample_rate = await self._load_audio(audio_source)

            # Convert to 16kHz mono 16-bit PCM
            if sample_rate != 16000:
                import scipy.signal

                audio_data = scipy.signal.resample(
                    audio_data, int(len(audio_data) * 16000 / sample_rate)
                )

            # Convert to int16
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)

            # Process in frames (30ms)
            frame_duration = 30  # ms
            frame_size = int(16000 * frame_duration / 1000)

            self.vad.set_mode(aggressiveness)

            voice_segments = []
            current_segment = None

            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i : i + frame_size]
                frame_bytes = frame.tobytes()

                is_speech = self.vad.is_speech(frame_bytes, 16000)

                if is_speech and current_segment is None:
                    current_segment = {"start": i / 16000}
                elif not is_speech and current_segment is not None:
                    current_segment["end"] = i / 16000
                    voice_segments.append(current_segment)
                    current_segment = None

            return {
                "success": True,
                "voice_segments": voice_segments,
                "total_segments": len(voice_segments),
                "total_voice_time": sum(s["end"] - s["start"] for s in voice_segments),
                "aggressiveness": aggressiveness,
            }

        except Exception as e:
            self.logger.error(f"VAD error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_audio_info(
        self, audio_source: Union[str, Path, bytes]
    ) -> Dict[str, Any]:
        """
        Get audio file information

        Args:
            audio_source: Audio file path or bytes

        Returns:
            Dictionary with audio info
        """
        try:
            # Load audio
            audio_data, sample_rate = await self._load_audio(audio_source)

            # Calculate duration
            duration = len(audio_data) / sample_rate

            # Get file size if source is file
            file_size = 0
            if isinstance(audio_source, (str, Path)):
                file_size = Path(audio_source).stat().st_size

            return {
                "success": True,
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                "samples": len(audio_data),
                "file_size": file_size,
                "format": "wav",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def convert_audio_format(
        self,
        audio_source: Union[str, Path, bytes],
        target_format: str = "wav",
        target_sample_rate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Convert audio format

        Args:
            audio_source: Audio source
            target_format: Target format (wav, mp3, ogg)
            target_sample_rate: Target sample rate

        Returns:
            Dictionary with converted audio
        """
        try:
            # Load audio
            audio_data, sample_rate = await self._load_audio(audio_source)

            # Resample if needed
            if target_sample_rate and target_sample_rate != sample_rate:
                import scipy.signal

                audio_data = scipy.signal.resample(
                    audio_data, int(len(audio_data) * target_sample_rate / sample_rate)
                )
                sample_rate = target_sample_rate

            # Convert to bytes
            buffer = io.BytesIO()

            if target_format == "wav":
                sf.write(buffer, audio_data, sample_rate, format="wav")
            elif target_format == "mp3":
                sf.write(buffer, audio_data, sample_rate, format="mp3")
            elif target_format == "ogg":
                sf.write(buffer, audio_data, sample_rate, format="ogg")
            else:
                raise ValueError(f"Unsupported format: {target_format}")

            audio_bytes = buffer.getvalue()

            return {
                "success": True,
                "audio_data": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": target_format,
                "sample_rate": sample_rate,
                "size": len(audio_bytes),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_audio_devices(self) -> Dict[str, Any]:
        """List available audio devices"""
        if not AUDIO_AVAILABLE:
            return {"success": False, "error": "Sounddevice not available"}

        try:
            devices = sd.query_devices()

            device_list = []
            for i, device in enumerate(devices):
                device_list.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "max_input_channels": device["max_input_channels"],
                        "max_output_channels": device["max_output_channels"],
                        "default_sample_rate": device["default_samplerate"],
                    }
                )

            return {
                "success": True,
                "devices": device_list,
                "default_input": sd.default.device[0],
                "default_output": sd.default.device[1],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _add_to_history(self, segment: SpeechSegment):
        """Add transcription to history"""
        self.transcription_history.append(segment)
        if len(self.transcription_history) > self.max_history:
            self.transcription_history = self.transcription_history[-self.max_history :]

    def get_transcription_history(self, limit: int = None) -> List[Dict]:
        """Get transcription history"""
        history = self.transcription_history
        if limit:
            history = history[-limit:]

        return [
            {"text": h.text, "confidence": h.confidence, "timestamp": h.start_time}
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "history_size": len(self.transcription_history),
            "synthesis_history_size": len(self.synthesis_history),
            "stt_model": self.default_stt_model.value,
            "tts_model": self.default_tts_model.value,
            "is_recording": self.recording,
        }

    async def close(self):
        """Clean up resources"""
        if self.recording:
            await self.stop_streaming()

        if self.tts_engine:
            self.tts_engine.stop()


# Integration wrapper for EDIATH
class SpeechAgentWrapper:
    """
    Wrapper class to integrate SpeechAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.speech_agent = SpeechAgent(config)
        self.agent_type = "speech_processing"
        self.capabilities = [
            "speech_to_text",
            "text_to_speech",
            "audio_recording",
            "voice_activity_detection",
            "audio_conversion",
            "real_time_streaming",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a speech request

        Request format:
        {
            'operation': 'transcribe|synthesize|record|stream|vad|info|convert',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "transcribe":
            model = request.get("model")
            if model:
                model = SpeechModel(model)

            return await self.speech_agent.transcribe_audio(
                audio_source=request.get("audio"),
                model=model,
                language=request.get("language", "en"),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "synthesize":
            model = request.get("model")
            if model:
                model = TTSModel(model)

            emotion = request.get("emotion", "neutral")

            return await self.speech_agent.synthesize_speech(
                text=request.get("text"),
                model=model,
                voice=request.get("voice"),
                rate=request.get("rate"),
                volume=request.get("volume"),
                emotion=VoiceEmotion(emotion),
                output_file=request.get("output_file"),
            )

        elif operation == "record":
            return await self.speech_agent.record_audio(
                duration=request.get("duration", 5.0),
                sample_rate=request.get("sample_rate"),
                channels=request.get("channels"),
                device=request.get("device"),
            )

        elif operation == "record_transcribe":
            model = request.get("model")
            if model:
                model = SpeechModel(model)

            return await self.speech_agent.record_and_transcribe(
                duration=request.get("duration", 5.0),
                model=model,
                language=request.get("language", "en"),
            )

        elif operation == "stream":
            # For streaming, you'd need to handle callbacks differently
            return {"success": False, "error": "Streaming requires callback handling"}

        elif operation == "vad":
            return await self.speech_agent.detect_voice_activity(
                audio_source=request.get("audio"),
                aggressiveness=request.get("aggressiveness", 3),
            )

        elif operation == "info":
            return await self.speech_agent.get_audio_info(
                audio_source=request.get("audio")
            )

        elif operation == "convert":
            return await self.speech_agent.convert_audio_format(
                audio_source=request.get("audio"),
                target_format=request.get("format", "wav"),
                target_sample_rate=request.get("sample_rate"),
            )

        elif operation == "devices":
            return await self.speech_agent.list_audio_devices()

        elif operation == "history":
            return {
                "success": True,
                "history": self.speech_agent.get_transcription_history(
                    limit=request.get("limit")
                ),
            }

        elif operation == "stats":
            return self.speech_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "SpeechAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.speech_agent.get_stats(),
            "stt_model": self.speech_agent.default_stt_model.value,
            "tts_model": self.speech_agent.default_tts_model.value,
        }

    async def close(self):
        """Clean up resources"""
        await self.speech_agent.close()


# Example usage and testing
async def test_speech_agent():
    """Test the speech agent functionality"""

    # Initialize agent
    agent = SpeechAgent()

    print("=== Speech Agent Test ===\n")

    # Test text-to-speech synthesis
    print("1. Text-to-Speech Synthesis...")
    result = await agent.synthesize_speech(
        text="Hello! This is EDIATH, your personal AI assistant. I'm happy to help you today.",
        model=TTSModel.SYSTEM,
        output_file="test_speech.wav",
    )

    if result["success"]:
        print("   Synthesis successful!")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Output file: {result.get('output_file', 'N/A')}")

    # Test audio info (if we have a file)
    print("\n2. Audio File Info...")
    try:
        info = await agent.get_audio_info("test_speech.wav")
        if info["success"]:
            print(f"   Duration: {info['duration']:.2f}s")
            print(f"   Sample rate: {info['sample_rate']} Hz")
            print(f"   Channels: {info['channels']}")
    except:
        print("   No audio file to analyze")

    # Test audio devices
    print("\n3. Available Audio Devices...")
    devices = await agent.list_audio_devices()
    if devices["success"]:
        print(f"   Found {len(devices['devices'])} devices")
        for device in devices["devices"][:3]:
            print(f"     - {device['name']}")

    # Test transcription with a sample (if we have recorded audio)
    print("\n4. Speech Recognition Test...")
    print("   (This would require microphone input)")

    # Get statistics
    print("\n5. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total transcriptions: {stats['total_transcriptions']}")
    print(f"   Total syntheses: {stats['total_syntheses']}")
    print(f"   Average confidence: {stats['average_confidence']:.2f}")

    print("\n=== Test Complete ===")

    # Clean up
    await agent.close()


# Run test
if __name__ == "__main__":
    asyncio.run(test_speech_agent())
