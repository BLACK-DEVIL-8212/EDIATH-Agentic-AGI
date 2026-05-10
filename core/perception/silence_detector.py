"""Silence Detector - detects silence in audio streams with advanced VAD, adaptive thresholding, and real-time processing."""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import time
import threading
from collections import deque
import math

try:
    import webrtcvad

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from ..utils.logger import logger


class SilenceState(Enum):
    """Current silence detection state."""

    SPEAKING = "speaking"
    SILENT = "silent"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


class VADMode(Enum):
    """Voice Activity Detection modes (WebRTC)."""

    VERY_AGGRESSIVE = 3
    AGGRESSIVE = 2
    NORMAL = 1
    RELAXED = 0


@dataclass
class SilenceRegion:
    """Represents a detected silence region."""

    start_time: float
    end_time: float
    duration: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AudioStats:
    """Real-time audio statistics."""

    energy_level: float = 0.0
    peak_energy: float = 0.0
    rms_energy: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    background_noise_level: float = 0.0
    snr_db: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class SilenceDetector:
    """Advanced silence detector with VAD, adaptive thresholding, and real-time processing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 480,  # 30ms frames
        vad_mode: VADMode = VADMode.NORMAL,
        use_webrtc: bool = True,
        energy_threshold: float = 0.01,
        min_silence_duration: float = 0.3,
        max_silence_duration: float = 10.0,
        adaptive_threshold: bool = True,
        adaptation_rate: float = 0.01,
        history_size: int = 100,
        voice_hold_time: float = 0.5,
    ):
        """Initialize advanced silence detector.

        Args:
            sample_rate: Audio sample rate (Hz)
            frame_size: Frame size for analysis (samples)
            vad_mode: WebRTC VAD mode (if available)
            use_webrtc: Use WebRTC VAD if available
            energy_threshold: Energy threshold for silence detection
            min_silence_duration: Minimum silence duration (seconds)
            max_silence_duration: Maximum silence duration (seconds)
            adaptive_threshold: Adapt threshold based on background noise
            adaptation_rate: Rate of threshold adaptation
            history_size: Size of audio history buffer
            voice_hold_time: Time to hold voice state after detection (seconds)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.vad_mode = vad_mode
        self.use_webrtc = use_webrtc and WEBRTC_AVAILABLE
        self.energy_threshold = energy_threshold
        self.min_silence_duration = min_silence_duration
        self.max_silence_duration = max_silence_duration
        self.adaptive_threshold = adaptive_threshold
        self.adaptation_rate = adaptation_rate
        self.history_size = history_size
        self.voice_hold_time = voice_hold_time

        # WebRTC VAD instance
        self._vad = None
        if self.use_webrtc:
            self._vad = webrtcvad.Vad()
            self._vad.set_mode(vad_mode.value)
            logger.info(f"✅ WebRTC VAD enabled (mode={vad_mode.name})")

        # Audio buffers
        self.audio_buffer: deque = deque(maxlen=history_size)
        self.energy_history: deque = deque(maxlen=history_size)
        self.noise_floor_history: deque = deque(maxlen=history_size * 2)

        # State tracking
        self.current_state: SilenceState = SilenceState.UNCERTAIN
        self.silence_start_time: Optional[float] = None
        self.speech_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None

        # Statistics
        self.detection_count = 0
        self.silence_regions: List[SilenceRegion] = []
        self.speech_segments: List[Tuple[float, float]] = []
        self.audio_stats: List[AudioStats] = []

        # Adaptive threshold
        self.background_noise_level = 0.0
        self.dynamic_threshold = energy_threshold

        # Real-time processing
        self._processing = False
        self._processing_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._state_callbacks: List[Callable] = []

        # Performance metrics
        self.metrics = {
            "total_audio_processed_seconds": 0.0,
            "total_silence_detected_seconds": 0.0,
            "avg_silence_duration": 0.0,
            "vad_decisions": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        logger.info(
            f"✅ SilenceDetector initialized (rate={sample_rate}Hz, frame={frame_size}, vad={self.use_webrtc})"
        )

    # #================#================#============#=============
    # CORE DETECTION
    # #================#================#============#=============

    def is_silent(
        self,
        audio_data: Optional[bytes] = None,
        threshold: Optional[float] = None,
        use_vad: bool = True,
        return_confidence: bool = False,
    ) -> Union[bool, Tuple[bool, float]]:
        """Check if audio contains silence with advanced detection.

        Args:
            audio_data: Audio data bytes
            threshold: Custom energy threshold
            use_vad: Use WebRTC VAD if available
            return_confidence: Return confidence score

        Returns:
            True if silent, False otherwise (or tuple with confidence)
        """
        if audio_data is None:
            return (True, 1.0) if return_confidence else True

        try:
            # Convert to numpy array
            audio_array = self._bytes_to_array(audio_data)

            # Calculate energy
            energy = self._calculate_energy(audio_array)
            rms = self._calculate_rms(audio_array)
            zcr = self._calculate_zcr(audio_array)

            # Update audio stats
            self._update_audio_stats(audio_array, energy, rms, zcr)

            # Update adaptive threshold
            if self.adaptive_threshold:
                self._update_adaptive_threshold(energy)

            # Use VAD if available and requested
            if use_vad and self.use_webrtc and self._vad:
                is_speech = self._vad.is_speech(audio_data, self.sample_rate)
                is_silent = not is_speech
                confidence = 0.9 if is_silent else 0.1

                self.metrics["vad_decisions"] += 1

                # Update state
                self._update_state(is_silent)

                if return_confidence:
                    return is_silent, confidence
                return is_silent

            # Use energy-based detection
            current_threshold = threshold or self.dynamic_threshold
            is_silent_energy = energy < current_threshold

            # Calculate confidence based on energy ratio
            if current_threshold > 0:
                confidence = min(1.0, current_threshold / max(energy, 0.001))
            else:
                confidence = 1.0 if is_silent_energy else 0.0

            # Also consider zero-crossing rate for better detection
            if zcr < 50:  # Low ZCR indicates silence
                confidence = max(confidence, 0.7)

            # Update state
            self._update_state(is_silent_energy)

            self.detection_count += 1

            if return_confidence:
                return is_silent_energy, confidence

            return is_silent_energy

        except Exception as e:
            logger.debug(f"Silence detection error: {e}")
            return (True, 0.5) if return_confidence else True

    def _bytes_to_array(self, audio_data: bytes) -> np.ndarray:
        """Convert bytes to numpy array."""
        dtype = np.int16
        array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
        array = array / 32768.0  # Normalize to [-1, 1]
        return array

    def _calculate_energy(self, audio_array: np.ndarray) -> float:
        """Calculate RMS energy of audio."""
        return float(np.sqrt(np.mean(audio_array**2)))

    def _calculate_rms(self, audio_array: np.ndarray) -> float:
        """Calculate Root Mean Square energy."""
        return float(np.sqrt(np.mean(audio_array**2)))

    def _calculate_zcr(self, audio_array: np.ndarray) -> float:
        """Calculate Zero Crossing Rate."""
        if len(audio_array) < 2:
            return 0.0

        crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / 2
        return float(crossings / len(audio_array))

    def _calculate_spectral_centroid(self, audio_array: np.ndarray) -> float:
        """Calculate spectral centroid (brightness of sound)."""
        fft = np.fft.fft(audio_array)
        magnitude = np.abs(fft[: len(fft) // 2])
        frequencies = np.fft.fftfreq(len(audio_array), 1 / self.sample_rate)[
            : len(fft) // 2
        ]

        if np.sum(magnitude) == 0:
            return 0.0

        centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        return float(centroid)

    def _update_adaptive_threshold(self, energy: float):
        """Update adaptive threshold based on background noise."""
        # Update noise floor using minimum tracking
        if energy < self.dynamic_threshold:
            # Silence: update noise floor
            self.noise_floor_history.append(energy)
            if len(self.noise_floor_history) > 10:
                self.background_noise_level = np.mean(self.noise_floor_history)

        # Dynamic threshold = noise floor + margin
        margin = max(0.01, self.background_noise_level * 2)
        new_threshold = self.background_noise_level + margin

        # Smooth update
        self.dynamic_threshold = (
            self.dynamic_threshold * (1 - self.adaptation_rate)
            + new_threshold * self.adaptation_rate
        )

        # Clamp threshold
        self.dynamic_threshold = max(0.001, min(0.5, self.dynamic_threshold))

    def _update_audio_stats(
        self, audio_array: np.ndarray, energy: float, rms: float, zcr: float
    ):
        """Update real-time audio statistics."""
        self.audio_buffer.append(audio_array)
        self.energy_history.append(energy)

        # Calculate additional stats
        peak_energy = float(np.max(np.abs(audio_array)))
        spectral_centroid = self._calculate_spectral_centroid(audio_array)

        # Estimate SNR
        snr = 0.0
        if self.background_noise_level > 0:
            snr = 20 * math.log10(
                max(energy, 0.001) / max(self.background_noise_level, 0.001)
            )

        stats = AudioStats(
            energy_level=energy,
            peak_energy=peak_energy,
            rms_energy=rms,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            background_noise_level=self.background_noise_level,
            snr_db=snr,
        )

        self.audio_stats.append(stats)

        # Maintain history size
        if len(self.audio_stats) > self.history_size:
            self.audio_stats = self.audio_stats[-self.history_size :]

    def _update_state(self, is_silent: bool):
        """Update silence detection state machine."""
        current_time = time.time()

        if is_silent:
            if self.current_state != SilenceState.SILENT:
                # Transition to silence
                self.silence_start_time = current_time
                self.current_state = SilenceState.TRANSITION
            elif (
                self.silence_start_time
                and current_time - self.silence_start_time >= self.min_silence_duration
            ):
                # Confirmed silence
                if self.current_state != SilenceState.SILENT:
                    self.current_state = SilenceState.SILENT
                    self._on_silence_start()
        else:
            # Voice detected
            self.last_voice_time = current_time

            if self.current_state == SilenceState.SILENT:
                # End of silence
                if self.silence_start_time:
                    duration = current_time - self.silence_start_time
                    if duration >= self.min_silence_duration:
                        self._record_silence_region(
                            self.silence_start_time, current_time
                        )
                self.current_state = SilenceState.SPEAKING
                self._on_speech_start()
            elif self.current_state == SilenceState.TRANSITION:
                # Short interruption, not enough for silence
                self.current_state = SilenceState.SPEAKING
            elif self.current_state != SilenceState.SPEAKING:
                self.current_state = SilenceState.SPEAKING
                self._on_speech_start()

    def _on_silence_start(self):
        """Handle silence start event."""
        logger.debug("🔇 Silence started")
        self._trigger_state_callbacks("silence_start")

    def _on_speech_start(self):
        """Handle speech start event."""
        logger.debug("🎤 Speech started")
        self._trigger_state_callbacks("speech_start")

    def _record_silence_region(self, start_time: float, end_time: float):
        """Record a detected silence region."""
        duration = end_time - start_time

        # Only record if within limits
        if self.min_silence_duration <= duration <= self.max_silence_duration:
            region = SilenceRegion(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                confidence=min(1.0, duration / self.min_silence_duration),
            )
            self.silence_regions.append(region)

            self.metrics["total_silence_detected_seconds"] += duration

            # Update average
            total_duration = sum(r.duration for r in self.silence_regions)
            self.metrics["avg_silence_duration"] = total_duration / len(
                self.silence_regions
            )

            self._trigger_callbacks(region)

    # #================#================#============#=============
    # SILENCE REGION DETECTION
    # #================#================#============#=============

    def detect_silence_regions(
        self,
        audio_data: bytes,
        threshold: Optional[float] = None,
        min_silence_duration: Optional[float] = None,
        return_regions: bool = True,
    ) -> List[Union[SilenceRegion, Tuple[float, float]]]:
        """Detect regions of silence in audio.

        Args:
            audio_data: Audio data
            threshold: Energy threshold
            min_silence_duration: Minimum silence duration (seconds)
            return_regions: Return SilenceRegion objects

        Returns:
            List of silence regions
        """
        min_duration = min_silence_duration or self.min_silence_duration

        try:
            audio_array = self._bytes_to_array(audio_data)
            n_frames = len(audio_array) // self.frame_size

            if n_frames == 0:
                return []

            # Calculate energy for each frame
            energies = []
            for i in range(n_frames):
                frame = audio_array[i * self.frame_size : (i + 1) * self.frame_size]
                if len(frame) == self.frame_size:
                    energy = self._calculate_energy(frame)
                    energies.append(energy)
                else:
                    energies.append(0.0)

            # Use VAD if available
            if self.use_webrtc and self._vad:
                return self._detect_regions_vad(
                    audio_data, min_duration, return_regions
                )

            # Use energy-based detection
            current_threshold = threshold or self.dynamic_threshold
            is_silent = [e < current_threshold for e in energies]

            # Find silence regions
            min_silence_frames = int(min_duration * self.sample_rate / self.frame_size)
            regions = []
            in_silence = False
            silence_start = 0

            for i, silent in enumerate(is_silent):
                if silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not silent and in_silence:
                    duration_frames = i - silence_start
                    if duration_frames >= min_silence_frames:
                        start_time = silence_start * self.frame_size / self.sample_rate
                        end_time = i * self.frame_size / self.sample_rate

                        if return_regions:
                            region = SilenceRegion(
                                start_time=start_time,
                                end_time=end_time,
                                duration=end_time - start_time,
                                confidence=min(
                                    1.0, duration_frames / min_silence_frames
                                ),
                            )
                            regions.append(region)
                        else:
                            regions.append((start_time, end_time))
                    in_silence = False

            self.detection_count += 1
            return regions

        except Exception as e:
            logger.error(f"Silence region detection error: {e}")
            return []

    def _detect_regions_vad(
        self, audio_data: bytes, min_duration: float, return_regions: bool
    ) -> List[Union[SilenceRegion, Tuple[float, float]]]:
        """Detect silence regions using WebRTC VAD."""
        regions = []
        frame_duration_ms = int(self.frame_size / self.sample_rate * 1000)

        # Process in frames
        n_frames = len(audio_data) // (
            self.frame_size * 2
        )  # 16-bit = 2 bytes per sample
        in_silence = False
        silence_start = 0

        for i in range(n_frames):
            start_byte = i * self.frame_size * 2
            end_byte = start_byte + self.frame_size * 2
            frame = audio_data[start_byte:end_byte]

            if len(frame) == self.frame_size * 2:
                is_speech = self._vad.is_speech(frame, self.sample_rate)
                is_silent = not is_speech

                if is_silent and not in_silence:
                    silence_start = i * frame_duration_ms / 1000.0
                    in_silence = True
                elif not is_silent and in_silence:
                    end_time = i * frame_duration_ms / 1000.0
                    duration = end_time - silence_start

                    if duration >= min_duration:
                        if return_regions:
                            region = SilenceRegion(
                                start_time=silence_start,
                                end_time=end_time,
                                duration=duration,
                                confidence=min(1.0, duration / min_duration),
                            )
                            regions.append(region)
                        else:
                            regions.append((silence_start, end_time))
                    in_silence = False

        return regions

    def detect_speech_segments(
        self, audio_data: bytes, min_speech_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Detect speech segments in audio."""
        # Detect silence regions first
        silence_regions = self.detect_silence_regions(audio_data, return_regions=False)

        if not silence_regions:
            # No silence detected, entire audio might be speech
            total_duration = len(audio_data) / (self.sample_rate * 2)
            return (
                [(0.0, total_duration)] if total_duration >= min_speech_duration else []
            )

        # Speech segments are gaps between silence regions
        speech_segments = []
        last_end = 0.0

        for start, end in silence_regions:
            if start - last_end >= min_speech_duration:
                speech_segments.append((last_end, start))
            last_end = end

        # Last segment after final silence
        total_duration = len(audio_data) / (self.sample_rate * 2)
        if total_duration - last_end >= min_speech_duration:
            speech_segments.append((last_end, total_duration))

        self.speech_segments = speech_segments
        return speech_segments

    # #================#================#============#=============
    # REAL-TIME PROCESSING
    # #================#================#============#=============

    def start_real_time_processing(self, callback: Callable[[bool, float], None]):
        """Start real-time silence detection loop."""
        self._processing = True
        self._callbacks.append(callback)

        self._processing_thread = threading.Thread(
            target=self._real_time_loop, daemon=True
        )
        self._processing_thread.start()

        logger.info("🚀 Real-time silence detection started")

    def stop_real_time_processing(self):
        """Stop real-time processing."""
        self._processing = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        logger.info("🛑 Real-time silence detection stopped")

    def _real_time_loop(self):
        """Real-time processing loop (placeholder - requires audio input stream)."""
        # This would integrate with a microphone input stream
        # For now, it's a placeholder for future implementation
        while self._processing:
            # In real implementation, this would read from audio stream
            time.sleep(0.1)

    def process_frame(self, audio_frame: bytes) -> Tuple[bool, float]:
        """Process a single audio frame in real-time."""
        is_silent, confidence = self.is_silent(audio_frame, return_confidence=True)

        # Update metrics
        self.metrics["total_audio_processed_seconds"] += (
            self.frame_size / self.sample_rate
        )

        return is_silent, confidence

    # #================#================#============#=============
    # NOISE GATE AND THRESHOLD MANAGEMENT
    # #================#================#============#=============

    def get_noise_gate_threshold(
        self, audio_data: bytes, method: str = "energy"
    ) -> float:
        """Automatically determine noise gate threshold.

        Args:
            audio_data: Audio data
            method: Method to use ("energy", "percentile", "webrtc")

        Returns:
            Recommended threshold value
        """
        try:
            audio_array = self._bytes_to_array(audio_data)

            if method == "energy":
                # Calculate energy statistics
                energies = []
                n_frames = len(audio_array) // self.frame_size

                for i in range(min(n_frames, 100)):  # Limit frames
                    frame = audio_array[i * self.frame_size : (i + 1) * self.frame_size]
                    if len(frame) == self.frame_size:
                        energy = self._calculate_energy(frame)
                        energies.append(energy)

                if energies:
                    # Use 95th percentile of quiet frames
                    sorted_energies = sorted(energies)
                    percentile_idx = int(len(sorted_energies) * 0.95)
                    threshold = (
                        sorted_energies[percentile_idx]
                        if percentile_idx < len(sorted_energies)
                        else sorted_energies[-1]
                    )
                    return max(0.001, threshold * 1.5)

            elif method == "percentile":
                # Use percentile of all samples
                abs_samples = np.abs(audio_array)
                threshold = np.percentile(abs_samples, 95)
                return float(threshold)

            elif method == "webrtc" and self.use_webrtc:
                # WebRTC VAD already has built-in threshold
                return 0.5  # Placeholder

            # Default fallback
            return 0.01

        except Exception as e:
            logger.debug(f"Noise gate calculation error: {e}")
            return 0.1

    def calibrate_background_noise(self, audio_data: bytes, duration: float = 2.0):
        """Calibrate background noise level from audio sample."""
        try:
            audio_array = self._bytes_to_array(audio_data)

            # Calculate noise floor from quietest segments
            n_frames = len(audio_array) // self.frame_size
            frame_energies = []

            for i in range(min(n_frames, 100)):
                frame = audio_array[i * self.frame_size : (i + 1) * self.frame_size]
                if len(frame) == self.frame_size:
                    energy = self._calculate_energy(frame)
                    frame_energies.append(energy)

            if frame_energies:
                # Use 10th percentile as noise floor
                self.background_noise_level = np.percentile(frame_energies, 10)
                self.dynamic_threshold = self.background_noise_level * 3

                logger.info(
                    f"🔧 Calibrated: noise_floor={self.background_noise_level:.4f}, threshold={self.dynamic_threshold:.4f}"
                )

        except Exception as e:
            logger.error(f"Calibration error: {e}")

    # #================#================#============#=============
    # STATISTICS AND ANALYTICS
    # #================#================#============#=============

    def get_current_stats(self) -> AudioStats:
        """Get current audio statistics."""
        if self.audio_stats:
            return self.audio_stats[-1]
        return AudioStats()

    def get_silence_ratio(self, audio_duration: float) -> float:
        """Get ratio of silence to total audio duration."""
        if audio_duration <= 0:
            return 0.0

        total_silence = sum(r.duration for r in self.silence_regions)
        return min(1.0, total_silence / audio_duration)

    def get_speech_to_silence_ratio(self, audio_duration: float) -> float:
        """Get speech-to-silence ratio."""
        silence_ratio = self.get_silence_ratio(audio_duration)
        if silence_ratio >= 1.0:
            return 0.0
        return (1.0 - silence_ratio) / max(silence_ratio, 0.001)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive detector statistics."""
        avg_energy = np.mean(self.energy_history) if self.energy_history else 0
        avg_noise = np.mean(self.noise_floor_history) if self.noise_floor_history else 0

        return {
            "detections": self.detection_count,
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "current_state": self.current_state.value,
            "dynamic_threshold": self.dynamic_threshold,
            "background_noise_level": self.background_noise_level,
            "avg_energy_level": float(avg_energy),
            "avg_noise_floor": float(avg_noise),
            "silence_regions_detected": len(self.silence_regions),
            "speech_segments_detected": len(self.speech_segments),
            "metrics": self.metrics,
            "vad_enabled": self.use_webrtc,
            "vad_mode": self.vad_mode.name if self.use_webrtc else None,
            "adaptive_threshold": self.adaptive_threshold,
            "min_silence_duration": self.min_silence_duration,
            "max_silence_duration": self.max_silence_duration,
            "recent_stats": self.get_current_stats().__dict__,
        }

    def get_silence_regions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get detected silence regions."""
        return [r.to_dict() for r in self.silence_regions[-limit:]]

    def get_speech_segments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get detected speech segments."""
        return [
            {"start": start, "end": end, "duration": end - start}
            for start, end in self.speech_segments[-limit:]
        ]

    # #================#================#============#=============
    # CALLBACKS
    # #================#================#============#=============

    def add_callback(self, callback: Callable[[SilenceRegion], None]):
        """Add callback for silence region detection."""
        self._callbacks.append(callback)

    def add_state_callback(self, callback: Callable[[str], None]):
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    def _trigger_callbacks(self, region: SilenceRegion):
        """Trigger all callbacks."""
        for callback in self._callbacks:
            try:
                callback(region)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _trigger_state_callbacks(self, state: str):
        """Trigger state callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    # #================#================#============#=============
    # UTILITIES
    # #================#================#============#=============

    def reset(self):
        """Reset all detector state and statistics."""
        self.silence_regions.clear()
        self.speech_segments.clear()
        self.audio_stats.clear()
        self.audio_buffer.clear()
        self.energy_history.clear()
        self.noise_floor_history.clear()

        self.silence_start_time = None
        self.speech_start_time = None
        self.current_state = SilenceState.UNCERTAIN
        self.detection_count = 0

        self.metrics = {
            "total_audio_processed_seconds": 0.0,
            "total_silence_detected_seconds": 0.0,
            "avg_silence_duration": 0.0,
            "vad_decisions": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        self.dynamic_threshold = self.energy_threshold
        self.background_noise_level = 0.0

        logger.info("🔄 SilenceDetector reset")

    def set_vad_mode(self, mode: VADMode):
        """Change WebRTC VAD mode."""
        if self.use_webrtc and self._vad:
            self.vad_mode = mode
            self._vad.set_mode(mode.value)
            logger.info(f"VAD mode changed to {mode.name}")

    def set_energy_threshold(self, threshold: float):
        """Set energy threshold manually."""
        self.energy_threshold = max(0.001, min(0.5, threshold))
        self.dynamic_threshold = self.energy_threshold
        logger.info(f"Energy threshold set to {threshold:.4f}")


# #================#================#============#=============
# CONVENIENCE FUNCTIONS
# #================#================#============#=============


def detect_optimal_frame_size(sample_rate: int) -> int:
    """Get optimal frame size for given sample rate."""
    # Common frame durations: 10ms, 20ms, 30ms
    durations_ms = [10, 20, 30]

    for duration in durations_ms:
        frame_size = int(sample_rate * duration / 1000)
        if frame_size % 2 == 0:  # Even number for byte alignment
            return frame_size

    return 480  # Default 30ms at 16kHz


def calculate_energy_percentiles(
    audio_data: bytes, percentiles: List[int] = [10, 50, 90]
) -> Dict[int, float]:
    """Calculate energy percentiles for audio data."""
    try:
        audio_array = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        frame_size = 480
        n_frames = len(audio_array) // frame_size

        energies = []
        for i in range(min(n_frames, 100)):
            frame = audio_array[i * frame_size : (i + 1) * frame_size]
            if len(frame) == frame_size:
                energy = np.sqrt(np.mean(frame**2))
                energies.append(energy)

        if energies:
            return {p: float(np.percentile(energies, p)) for p in percentiles}

        return {p: 0.0 for p in percentiles}

    except Exception:
        return {p: 0.0 for p in percentiles}


__all__ = [
    "SilenceDetector",
    "SilenceState",
    "VADMode",
    "SilenceRegion",
    "AudioStats",
    "detect_optimal_frame_size",
    "calculate_energy_percentiles",
]
