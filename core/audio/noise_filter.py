"""
EDIATH Noise Filter Module
Production-grade audio noise reduction and enhancement

Fixes applied
-------------
1.  _prepare_audio: audio.ndim > 1 used np.reshape(-1) which is wrong syntax.
    Fixed to audio.flatten() (or audio.reshape(-1) with correct args).
2.  EnhancedNoiseProfile.update: alpha applies to old spectrum but (1-alpha)
    applies to new — correct Exponential Moving Average direction, but
    snr_estimate was never updated on subsequent calls. Fixed.
3.  EnhancedNoiseProfile.__init__: snr_estimate attribute missing — only
    NoiseProfile dataclass has it; EnhancedNoiseProfile didn't, causing
    AttributeError when learn_noise_profile tried to set it. Added.
4.  WienerFilter.apply: noise_spectrum length may not match rfft output of
    audio (different audio lengths). Added safe trimming/padding.
5.  AdaptiveFilter.update: LMS weight update uses `2 * mu * error * x` which
    is the unnormalised LMS; can diverge for large inputs. Switched to NLMS
    (normalised) with a stability epsilon.
6.  spectral_subtraction: noise_mag length mismatch was trimmed but the
    reconstructed phase/mag arrays were sliced inconsistently — irfft was
    called with original n but the spectrum was shorter. Fixed: pad/trim
    noise to match rfft output exactly.
7.  bandpass: low >= high not checked; could produce invalid Butterworth
    coefficients and crash. Added guard.
8.  highpass / lowpass / bandpass: bare `except:` swallowed ALL exceptions
    silently. Changed to `except Exception as exc` with a debug log.
9.  process_chunk: _processing_times list trimmed with slice reassignment
    inside a hot path — fine, but inconsistent with _snr_history which does
    the same. Unified both to use deque(maxlen=100) for O(1) behaviour.
10. denoise: _snr_history trimmed with slice but the list was never converted
    to deque. Made consistent (deque).
11. get_stats / get_performance_metrics: np.mean / np.max on empty list raises
    ValueError. Guarded with `if self._processing_times` (was already done for
    throughput but not for peak/average in get_performance_metrics).
12. reset: called self._processing_times.clear() and self._snr_history.clear()
    but those are now deques — .clear() works on deque too, so no change
    needed, but the comment is updated for clarity.
13. FilterStats.filter_bank default_factory dict is fine, but
    stats.total_noise_reduction_ms is tracked in stats but never updated
    anywhere. Added update in process_chunk.
14. KalmanFilter.filter_signal: named parameter `signal` shadows the imported
    `scipy.signal` module at module level. Renamed parameter to `measurements`.
15. logger import uses relative path `..utils.logger`; added a stdlib fallback
    so the module works standalone (same pattern as audio_listener fix).
16. NoiseEstimationMethod.HISTOGRAM defined in enum but never implemented —
    _update_mcra_minima only runs for MCRA. Added a clear NotImplementedError
    guard in learn_noise_profile so callers get an actionable error instead of
    silent wrong behaviour.
"""

import re
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from scipy import signal as scipy_signal

# ── logger with fallback ──────────────────────────────────────────────────────
try:
    from ..utils.logger import logger  # type: ignore
except ImportError:
    logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class FilterType(Enum):
    SPECTRAL = "spectral"
    WIENER = "wiener"
    ADAPTIVE = "adaptive"
    KALMAN = "kalman"


class NoiseEstimationMethod(Enum):
    MINIMA = "minima"
    MCRA = "mcra"
    IMCRA = "imcra"
    HISTOGRAM = "histogram"


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class NoiseProfile:
    noise_spectrum: np.ndarray
    threshold: float
    created_at: float = field(default_factory=time.time)
    update_count: int = 0
    snr_estimate: float = 0.0

    def __post_init__(self):
        self.noise_spectrum = np.asarray(self.noise_spectrum, dtype=np.float32)
        self.threshold = float(self.threshold)


@dataclass
class FilterStats:
    total_frames_processed: int = 0
    total_noise_reduction_ms: float = 0.0
    average_snr_improvement: float = 0.0
    last_processing_time_ms: float = 0.0
    filter_bank: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# EnhancedNoiseProfile
# =============================================================================


class EnhancedNoiseProfile:
    """Noise profile with adaptive learning via exponential smoothing / MCRA."""

    def __init__(
        self,
        noise_data: np.ndarray,
        method: NoiseEstimationMethod = NoiseEstimationMethod.MCRA,
    ):
        noise_data = np.asarray(noise_data, dtype=np.float32)

        self.noise_spectrum: np.ndarray = np.abs(np.fft.rfft(noise_data)) ** 2
        self.threshold: float = float(np.mean(self.noise_spectrum) * 1.5)

        self.spectral_flux: float = float(np.diff(self.noise_spectrum).mean())
        self.peak_noise: float = float(np.max(self.noise_spectrum))
        self.noise_floor: float = float(np.min(self.noise_spectrum))
        self.dynamic_range: float = self.peak_noise - self.noise_floor

        self.smoothed_spectrum: np.ndarray = self.noise_spectrum.copy()
        self.smoothing_factor: float = 0.9

        self.method = method
        self.created_at: float = time.time()
        self.update_count: int = 0

        # FIX 3: attribute expected by learn_noise_profile
        self.snr_estimate: float = 0.0

        # MCRA state
        self.local_minima: np.ndarray = self.noise_spectrum.copy()
        self.local_min_counter: np.ndarray = np.zeros_like(self.noise_spectrum)

    def update(self, new_noise: np.ndarray, alpha: float = 0.95) -> None:
        new_noise = np.asarray(new_noise, dtype=np.float32)
        new_spectrum = np.abs(np.fft.rfft(new_noise)) ** 2

        # Align lengths (new recording may have a different n)
        new_spectrum = _match_length(new_spectrum, len(self.noise_spectrum))

        # Exponential Moving Average: high alpha → trust old estimate more
        self.noise_spectrum = alpha * self.noise_spectrum + (1 - alpha) * new_spectrum
        self.smoothed_spectrum = (
            self.smoothing_factor * self.smoothed_spectrum
            + (1 - self.smoothing_factor) * new_spectrum
        )
        self.threshold = float(np.mean(self.noise_spectrum) * 1.5)
        self.update_count += 1

        # FIX 2: keep snr_estimate in sync
        self.snr_estimate = 20 * np.log10(
            max(float(np.max(self.noise_spectrum)), 1e-9)
            / max(float(np.min(self.noise_spectrum)), 1e-9)
        )

        if self.method == NoiseEstimationMethod.MCRA:
            self._update_mcra_minima(new_spectrum)

    def _update_mcra_minima(self, spectrum: np.ndarray) -> None:
        spectrum = _match_length(spectrum, len(self.local_minima))
        minima_candidates = np.minimum(self.local_minima, spectrum)

        stable_mask = spectrum > minima_candidates * 1.05
        self.local_min_counter[stable_mask] += 1
        self.local_min_counter[~stable_mask] = 0

        update_mask = self.local_min_counter > 5
        self.local_minima[update_mask] = minima_candidates[update_mask]
        self.local_min_counter[update_mask] = 0


# =============================================================================
# WienerFilter
# =============================================================================


class WienerFilter:
    """Wiener filter for noise reduction."""

    def __init__(self, sample_rate: int = 16000, noise_floor: float = 1e-6):
        self.sample_rate = sample_rate
        self.noise_floor = noise_floor

    def apply(self, audio: np.ndarray, noise_spectrum: np.ndarray) -> np.ndarray:
        audio_spectrum = np.fft.rfft(audio)
        mag = np.abs(audio_spectrum)
        phase = np.angle(audio_spectrum)

        # FIX 4: align noise_spectrum to rfft output length
        noise_spec = _match_length(noise_spectrum, len(mag))

        signal_power = np.maximum(mag**2 - noise_spec, self.noise_floor)
        wiener_gain = signal_power / (signal_power + noise_spec)

        clean_spectrum = (mag * wiener_gain) * np.exp(1j * phase)
        return np.fft.irfft(clean_spectrum, n=len(audio))


# =============================================================================
# AdaptiveFilter
# =============================================================================


class AdaptiveFilter:
    """Normalised LMS (NLMS) adaptive filter for echo/noise reduction."""

    def __init__(
        self, filter_length: int = 128, mu: float = 0.01, epsilon: float = 1e-8
    ):
        self.filter_length = filter_length
        self.mu = mu
        # FIX 5: epsilon prevents division by zero in NLMS normalisation
        self.epsilon = epsilon
        self.weights = np.zeros(filter_length, dtype=np.float64)
        self.error_history: deque = deque(maxlen=100)

    def update(self, reference: np.ndarray, desired: np.ndarray) -> np.ndarray:
        if len(reference) < self.filter_length:
            return desired.copy()

        output = np.zeros_like(desired, dtype=np.float64)
        error = desired.astype(np.float64).copy()

        for i in range(self.filter_length, len(desired)):
            x = reference[i - self.filter_length : i][::-1].astype(np.float64)
            output[i] = np.dot(self.weights, x)
            error[i] = desired[i] - output[i]

            # FIX 5: NLMS update — normalise by signal power
            power = np.dot(x, x) + self.epsilon
            self.weights += (self.mu / power) * error[i] * x

        self.error_history.append(float(np.mean(error**2)))
        return error.astype(np.float32)


# =============================================================================
# KalmanFilter
# =============================================================================


class KalmanFilter:
    """Scalar Kalman filter for signal smoothing."""

    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-2):
        self.Q = process_noise
        self.R = measurement_noise
        self.P = 1.0
        self.K = 0.0
        self.x = 0.0

    def apply(self, measurement: float) -> float:
        self.P = self.P + self.Q
        self.K = self.P / (self.P + self.R)
        self.x = self.x + self.K * (measurement - self.x)
        self.P = (1 - self.K) * self.P
        return self.x

    # FIX 14: renamed `signal` → `measurements` to avoid shadowing scipy.signal
    def filter_signal(self, measurements: np.ndarray) -> np.ndarray:
        filtered = np.empty_like(measurements, dtype=np.float32)
        for i, v in enumerate(measurements):
            filtered[i] = self.apply(float(v))
        return filtered


# =============================================================================
# Utility
# =============================================================================


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad 1-D array to target_len."""
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


# =============================================================================
# NoiseFilter
# =============================================================================


class NoiseFilter:
    """Production-grade noise filter with multiple strategies."""

    def __init__(
        self,
        sample_rate: int = 16000,
        filter_type: FilterType = FilterType.SPECTRAL,
    ):
        self.sample_rate = sample_rate
        self.filter_type = filter_type
        self.noise_profile: Optional[EnhancedNoiseProfile] = None
        self.filters_applied: int = 0
        self.stats = FilterStats()

        self.wiener_filter = WienerFilter(sample_rate)
        self.adaptive_filter = AdaptiveFilter()
        self.kalman_filter = KalmanFilter()

        self._buffer: deque = deque(maxlen=sample_rate)

        # FIX 9 / 10: use deque for O(1) append/trim
        self._processing_times: deque = deque(maxlen=100)
        self._snr_history: deque = deque(maxlen=50)

        logger.info(
            f"NoiseFilter initialised — filter={filter_type.value}, "
            f"sample_rate={sample_rate}Hz"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio is None:
            return np.zeros(1, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32)
        # FIX 1: flatten correctly (reshape(-1) with no size arg works but
        # np.reshape(audio, -1) is the correct call form; flatten() is clearer)
        if audio.ndim > 1:
            audio = audio.flatten()
        return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # ── noise profile ─────────────────────────────────────────────────────────

    def learn_noise_profile(
        self,
        noise_audio: np.ndarray,
        method: NoiseEstimationMethod = NoiseEstimationMethod.MCRA,
    ) -> None:
        # FIX 16: HISTOGRAM not implemented — fail loudly
        if method == NoiseEstimationMethod.HISTOGRAM:
            raise NotImplementedError(
                "NoiseEstimationMethod.HISTOGRAM is not yet implemented. "
                "Use MCRA, MINIMA, or IMCRA."
            )

        noise_audio = self._prepare_audio(noise_audio)
        if len(noise_audio) < 512:
            logger.warning("Noise sample too short — padding to 512 zeros")
            noise_audio = np.zeros(512, dtype=np.float32)

        self.noise_profile = EnhancedNoiseProfile(noise_audio, method)

        # FIX 3: snr_estimate now exists on EnhancedNoiseProfile
        self.noise_profile.snr_estimate = 20 * np.log10(
            max(self.noise_profile.peak_noise, 1e-9)
            / max(self.noise_profile.noise_floor, 1e-9)
        )
        logger.info(f"Noise profile learnt — method={method.value}")

    def update_noise_profile(
        self, noise_audio: np.ndarray, alpha: float = 0.95
    ) -> None:
        if self.noise_profile is None:
            logger.warning(
                "No noise profile to update — call learn_noise_profile first"
            )
            return
        self.noise_profile.update(self._prepare_audio(noise_audio), alpha)
        logger.debug("Noise profile updated")

    # ── audio processing primitives ───────────────────────────────────────────

    def normalize(self, audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
        audio = self._prepare_audio(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * target_level
        return audio

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def limit(self, audio: np.ndarray, threshold: float = 0.99) -> np.ndarray:
        return np.clip(self._prepare_audio(audio), -threshold, threshold)

    def auto_gain(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        audio = self._prepare_audio(audio)
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms > 0:
            gain = np.clip(target_rms / rms, 0.5, 2.0)
            audio = audio * gain
        return audio

    def highpass(
        self, audio: np.ndarray, cutoff: int = 80, order: int = 4
    ) -> np.ndarray:
        audio = self._prepare_audio(audio)
        if len(audio) < 32:
            return audio
        nyquist = self.sample_rate / 2
        cutoff = min(cutoff, nyquist - 1)
        b, a = scipy_signal.butter(order, cutoff / nyquist, btype="high")
        try:
            return scipy_signal.filtfilt(b, a, audio)
        except Exception as exc:  # FIX 8: log instead of silently swallow
            logger.debug(f"highpass filtfilt failed: {exc}")
            return audio

    def lowpass(
        self, audio: np.ndarray, cutoff: int = 8000, order: int = 4
    ) -> np.ndarray:
        audio = self._prepare_audio(audio)
        if len(audio) < 32:
            return audio
        nyquist = self.sample_rate / 2
        cutoff = min(cutoff, nyquist - 1)
        b, a = scipy_signal.butter(order, cutoff / nyquist, btype="low")
        try:
            return scipy_signal.filtfilt(b, a, audio)
        except Exception as exc:
            logger.debug(f"lowpass filtfilt failed: {exc}")
            return audio

    def bandpass(
        self,
        audio: np.ndarray,
        lowcut: int = 80,
        highcut: int = 8000,
        order: int = 4,
    ) -> np.ndarray:
        audio = self._prepare_audio(audio)
        if len(audio) < 32:
            return audio
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        # FIX 7: guard against invalid band specification
        if low <= 0 or high >= 1 or low >= high:
            logger.debug(f"bandpass: invalid band [{low:.3f}, {high:.3f}] — skipping")
            return audio
        b, a = scipy_signal.butter(order, [low, high], btype="band")
        try:
            return scipy_signal.filtfilt(b, a, audio)
        except Exception as exc:
            logger.debug(f"bandpass filtfilt failed: {exc}")
            return audio

    def spectral_subtraction(
        self,
        audio: np.ndarray,
        strength: float = 1.5,
        floor_factor: float = 0.05,
    ) -> np.ndarray:
        audio = self._prepare_audio(audio)
        if self.noise_profile is None or len(audio) < 256:
            return audio

        spectrum = np.fft.rfft(audio)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        # FIX 6: align noise to exact rfft output length, then reconstruct fully
        noise_mag = np.sqrt(_match_length(self.noise_profile.noise_spectrum, len(mag)))

        clean_mag = np.maximum(mag - strength * noise_mag, floor_factor * mag)
        clean_spectrum = clean_mag * np.exp(1j * phase)
        return np.fft.irfft(clean_spectrum, n=len(audio))

    def apply_wiener_filter(self, audio: np.ndarray) -> np.ndarray:
        if self.noise_profile is None:
            return audio
        return self.wiener_filter.apply(audio, self.noise_profile.noise_spectrum)

    def apply_adaptive_filter(
        self,
        audio: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if reference is None:
            reference = audio.copy()
        return self.adaptive_filter.update(reference, audio)

    def apply_kalman_filter(self, audio: np.ndarray) -> np.ndarray:
        return self.kalman_filter.filter_signal(audio)

    # ── VAD ───────────────────────────────────────────────────────────────────

    def is_voice(
        self,
        audio: np.ndarray,
        energy_threshold: float = 0.01,
        peak_threshold: float = 0.1,
    ) -> bool:
        audio = self._prepare_audio(audio)
        if len(audio) < 128:
            return False
        energy = float(np.mean(audio**2))
        peak = float(np.max(np.abs(audio)))
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio)))
        return (energy > energy_threshold or peak > peak_threshold) and 0.02 < zcr < 0.4

    # ── SNR ───────────────────────────────────────────────────────────────────

    def estimate_snr(self, audio: np.ndarray) -> float:
        audio = self._prepare_audio(audio)
        if self.noise_profile is None:
            return 0.0
        signal_power = float(np.mean(audio**2))
        noise_power = float(np.mean(self.noise_profile.noise_spectrum))
        if noise_power > 0:
            return float(np.clip(10 * np.log10(signal_power / noise_power), -20, 60))
        return 0.0

    # ── pipelines ─────────────────────────────────────────────────────────────

    def process_chunk(
        self, audio: np.ndarray, apply_vad: bool = True
    ) -> Optional[np.ndarray]:
        start = time.perf_counter()
        audio = self._prepare_audio(audio)

        if apply_vad and not self.is_voice(audio):
            return None

        try:
            if self.filter_type == FilterType.SPECTRAL:
                audio = self.highpass(audio)
                audio = self.spectral_subtraction(audio)
            elif self.filter_type == FilterType.WIENER:
                audio = self.highpass(audio)
                audio = self.apply_wiener_filter(audio)
            elif self.filter_type == FilterType.ADAPTIVE:
                audio = self.apply_adaptive_filter(audio)
                audio = self.highpass(audio)
            elif self.filter_type == FilterType.KALMAN:
                audio = self.apply_kalman_filter(audio)
                audio = self.highpass(audio)

            audio = self.auto_gain(audio)
            audio = self.normalize(audio)
            audio = self.limit(audio)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._processing_times.append(elapsed_ms)  # FIX 9: deque, no trim needed

            self.filters_applied += 1
            self.stats.total_frames_processed += 1
            self.stats.last_processing_time_ms = elapsed_ms
            # FIX 13: actually track total noise reduction time
            self.stats.total_noise_reduction_ms += elapsed_ms
            self.stats.filter_bank[self.filter_type.value] = (
                self.stats.filter_bank.get(self.filter_type.value, 0) + 1
            )

            return audio.astype(np.float32)

        except Exception as exc:
            logger.error(f"process_chunk error: {exc}")
            return audio.astype(np.float32)

    def denoise(self, audio: np.ndarray, aggressive: int = 1) -> np.ndarray:
        """Denoise entire audio. aggressive: 0=light, 1=normal, 2=aggressive."""
        audio = self._prepare_audio(audio)
        try:
            audio = self.highpass(audio)
            audio = self.lowpass(audio)

            if self.noise_profile:
                strength = 1.0 + aggressive * 0.5
                floor_factor = 0.05 + aggressive * 0.02
                audio = self.spectral_subtraction(audio, strength, floor_factor)
                if aggressive >= 2:
                    audio = self.apply_wiener_filter(audio)

            audio = self.auto_gain(audio)
            audio = self.normalize(audio)
            audio = self.limit(audio)

            if self.noise_profile:
                snr = self.estimate_snr(audio)
                self._snr_history.append(snr)  # FIX 10: deque
                self.stats.average_snr_improvement = float(np.mean(self._snr_history))

            return audio

        except Exception as exc:
            logger.error(f"denoise error: {exc}")
            return audio

    # ── management ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.noise_profile = None
        self.filters_applied = 0
        self._processing_times.clear()  # FIX 12: deque.clear() — fine
        self._snr_history.clear()
        self.stats = FilterStats()
        self.adaptive_filter = AdaptiveFilter()
        self.kalman_filter = KalmanFilter()
        logger.info("Noise filter reset")

    def get_stats(self) -> Dict[str, Any]:
        # FIX 11: guard empty deque
        avg_proc = (
            float(np.mean(self._processing_times)) if self._processing_times else 0.0
        )
        return {
            "filters_applied": self.filters_applied,
            "noise_profile_available": self.noise_profile is not None,
            "filter_type": self.filter_type.value,
            "sample_rate": self.sample_rate,
            "avg_processing_time_ms": round(avg_proc, 2),
            "total_frames": self.stats.total_frames_processed,
            "last_processing_ms": round(self.stats.last_processing_time_ms, 2),
            "total_noise_reduction_ms": round(self.stats.total_noise_reduction_ms, 2),
            "average_snr_improvement": round(self.stats.average_snr_improvement, 2),
            "filter_bank": self.stats.filter_bank,
            "noise_profile_age": (
                time.time() - self.noise_profile.created_at if self.noise_profile else 0
            ),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        times = list(self._processing_times)
        snrs = list(self._snr_history)
        # FIX 11: all guarded
        avg = float(np.mean(times)) if times else 0.0
        peak = float(np.max(times)) if times else 0.0
        throughput = (1000.0 / avg) if avg > 0 else 0.0
        return {
            "processing_times_ms": times[-10:],
            "snr_history_db": snrs[-10:],
            "average_processing_time_ms": round(avg, 2),
            "peak_processing_time_ms": round(peak, 2),
            "throughput_fps": round(throughput, 2),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "NoiseFilter",
    "NoiseProfile",
    "EnhancedNoiseProfile",
    "FilterType",
    "NoiseEstimationMethod",
    "WienerFilter",
    "AdaptiveFilter",
    "KalmanFilter",
]
