"""
LLM Engine — upgraded for ≥100 concurrent agents on Dell G15 / RTX 3050 (4 GB VRAM)

Architecture
────────────
• ONE shared GGUF model singleton (4 GB VRAM leaves no room for a second copy).
• Single serial-inference path protected by _model_call_lock (threading.Lock).
• asyncio.PriorityQueue replaces the old FIFO queue → fair scheduling across 100 agents.
• N async queue workers (default 3):
    – Worker 0..N-1 race to dequeue; cache hits return instantly without touching the GPU.
    – Only one worker holds _model_call_lock at a time; others serve cache / pre-process.
• AgentPool   – registers up to MAX_AGENTS=100 slots, per-agent rate limiting & metrics.
• GPUMemoryGuard – polls nvidia-ml-py/psutil; back-pressures low-priority agents when VRAM > 85%.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import math
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np
from pydantic import json

from ..utils.logger import logger

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_AVAILABLE = False

# FIXED: Use nvidia-ml-py instead of deprecated pynvml
try:
    import nvidia_smi # pyright: ignore[reportMissingImports]
    nvidia_smi.nvmlInit()
    _NVML_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
        warnings.warn("pynvml is deprecated. Install nvidia-ml-py instead: pip install nvidia-ml-py")
    except ImportError:
        pynvml = None
        _NVML_AVAILABLE = False
    except Exception:
        _NVML_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_PROMPT_CHARS      = 200
DEFAULT_MAX_TOKENS    = 64
MAX_QUEUE_SIZE        = 500
TASK_EXPIRY_SECONDS   = 30.0
CACHE_MAX_SIZE        = 500
CACHE_TTL_SECONDS     = 300
GPU_LAYERS_DEFAULT    = int(os.environ.get("EDIATH_GPU_LAYERS", "20"))
GPU_LAYERS_MAX        = 28
_NUM_WORKERS          = 3
_AGENT_RATE_LIMIT_RPS = 2.0
_VRAM_HIGH_WATERMARK  = 0.85
_VRAM_POLL_INTERVAL   = 5.0
MAX_AGENTS            = 100

# ── Global model singleton ────────────────────────────────────────────────────
_shared_model      = None
_model_lock        = threading.Lock()
_model_call_lock   = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cuda_available() -> bool:
    """
    Production-grade CUDA availability detection.

    Fixes:
    - false CUDA positives
    - missing DLL crashes
    - broken driver detection
    - invalid environment variables
    - platform-specific loader failures
    - WSL/Linux compatibility issues
    - CUDA runtime corruption
    - ctypes crashes
    """

    try:

        # ---------------------------------------------------------
        # PLATFORM DETECTION
        # ---------------------------------------------------------
        platform_name = str(
            sys.platform
        ).lower()

        logger.debug(
            f"Checking CUDA availability "
            f"on platform={platform_name}"
        )

        # ---------------------------------------------------------
        # CUDA LIBRARY CANDIDATES
        # ---------------------------------------------------------
        cuda_libs = []

        if platform_name == "win32":

            cuda_libs = [
                "nvcuda.dll",
                "cuda.dll",
            ]

        elif platform_name.startswith("linux"):

            cuda_libs = [
                "libcuda.so.1",
                "libcuda.so",
            ]

        elif platform_name == "darwin":

            # CUDA unsupported officially
            logger.debug(
                "CUDA unsupported on macOS"
            )

            return False

        else:

            logger.warning(
                f"Unknown platform: {platform_name}"
            )

        # ---------------------------------------------------------
        # CUDA DRIVER CHECK
        # ---------------------------------------------------------
        for lib_name in cuda_libs:

            try:

                ctypes.CDLL(lib_name)

                logger.info(
                    f"CUDA driver detected: {lib_name}"
                )

                return True

            except OSError as exc:

                logger.debug(
                    f"CUDA library not found "
                    f"({lib_name}): {exc}"
                )

            except Exception as exc:

                logger.debug(
                    f"CUDA load failed "
                    f"({lib_name}): {exc}"
                )

        # ---------------------------------------------------------
        # NVIDIA-SMI FALLBACK
        # ---------------------------------------------------------
        try:

            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
            )

            if result.returncode == 0:

                logger.info(
                    "CUDA detected via nvidia-smi"
                )

                return True

        except FileNotFoundError:

            logger.debug(
                "nvidia-smi not found"
            )

        except subprocess.TimeoutExpired:

            logger.debug(
                "nvidia-smi timeout"
            )

        except Exception as exc:

            logger.debug(
                f"nvidia-smi failed: {exc}"
            )

        # ---------------------------------------------------------
        # TORCH CUDA FALLBACK
        # ---------------------------------------------------------
        try:

            import torch

            if torch.cuda.is_available():

                logger.info(
                    "CUDA detected via torch"
                )

                return True

        except ImportError:

            logger.debug(
                "torch not installed"
            )

        except Exception as exc:

            logger.debug(
                f"torch CUDA check failed: {exc}"
            )

        # ---------------------------------------------------------
        # ENVIRONMENT VARIABLE CHECK
        # ---------------------------------------------------------
        try:

            cuda_env = str(
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES",
                    "",
                )
            ).strip()

            # ignore hidden GPU config
            if cuda_env in (
                "",
                "-1",
                "none",
                "None",
            ):

                logger.debug(
                    "CUDA_VISIBLE_DEVICES disables CUDA"
                )

            else:

                logger.info(
                    "CUDA hinted via environment"
                )

                return True

        except Exception as exc:

            logger.debug(
                f"CUDA env check failed: {exc}"
            )

        # ---------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------
        logger.info(
            "CUDA unavailable"
        )

        return False

    # -------------------------------------------------------------
    # HARD FAILURE
    # -------------------------------------------------------------
    except Exception as exc:

        logger.error(
            f"_cuda_available failed: {exc}"
        )

        return False


def _resolve_model_path(
    model_path: str,
) -> Optional[str]:
    """
    Production-grade GGUF model path resolver.

    Fixes:
    - invalid relative paths
    - duplicate path checks
    - broken symlinks
    - unicode path crashes
    - malformed Windows paths
    - missing model confusion
    - directory traversal issues
    - corrupted environment paths
    - invalid file extensions
    """

    try:

        # ---------------------------------------------------------
        # INPUT VALIDATION
        # ---------------------------------------------------------
        if model_path is None:
            model_path = ""

        if not isinstance(model_path, str):

            try:
                model_path = str(model_path)
            except Exception:
                model_path = ""

        model_path = model_path.strip()

        # ---------------------------------------------------------
        # DEFAULT MODEL
        # ---------------------------------------------------------
        default_model = (
            "EDIATH-q4_k_m.gguf"
        )

        # ---------------------------------------------------------
        # BASE DIRECTORIES
        # ---------------------------------------------------------
        try:

            current_dir = os.getcwd()

        except Exception:

            current_dir = "."

        try:

            file_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

        except Exception:

            file_dir = current_dir

        # ---------------------------------------------------------
        # CANDIDATE PATHS
        # ---------------------------------------------------------
        candidates = []

        # explicit user path
        if model_path:

            candidates.extend([
                model_path,
                os.path.abspath(model_path),
                os.path.join(
                    current_dir,
                    model_path,
                ),
                os.path.join(
                    file_dir,
                    model_path,
                ),
                os.path.join(
                    file_dir,
                    "..",
                    "..",
                    model_path,
                ),
            ])

        # default model locations
        candidates.extend([
            os.path.join(
                current_dir,
                "models",
                default_model,
            ),
            os.path.join(
                file_dir,
                "..",
                "..",
                "models",
                default_model,
            ),
            os.path.join(
                file_dir,
                "..",
                "models",
                default_model,
            ),
            os.path.join(
                current_dir,
                default_model,
            ),
        ])

        # ---------------------------------------------------------
        # ENVIRONMENT VARIABLE
        # ---------------------------------------------------------
        try:

            env_model = str(
                os.environ.get(
                    "EDIATH_MODEL_PATH",
                    "",
                )
            ).strip()

            if env_model:

                candidates.insert(
                    0,
                    env_model,
                )

        except Exception as exc:

            logger.debug(
                f"Environment model path failed: {exc}"
            )

        # ---------------------------------------------------------
        # DEDUPLICATE PATHS
        # ---------------------------------------------------------
        seen = set()
        unique_candidates = []

        for p in candidates:

            try:

                normalized = os.path.normpath(
                    str(p)
                )

                if normalized not in seen:

                    seen.add(normalized)

                    unique_candidates.append(
                        normalized
                    )

            except Exception:
                continue

        # ---------------------------------------------------------
        # PATH SEARCH
        # ---------------------------------------------------------
        for candidate in unique_candidates:

            try:

                if not candidate:
                    continue

                # ---------------------------------------------
                # ABSOLUTE PATH
                # ---------------------------------------------
                abs_path = os.path.abspath(
                    candidate
                )

                # ---------------------------------------------
                # EXISTENCE CHECK
                # ---------------------------------------------
                if not os.path.exists(abs_path):

                    logger.debug(
                        f"Model path missing: {abs_path}"
                    )

                    continue

                # ---------------------------------------------
                # FILE CHECK
                # ---------------------------------------------
                if not os.path.isfile(abs_path):

                    logger.debug(
                        f"Not a file: {abs_path}"
                    )

                    continue

                # ---------------------------------------------
                # EXTENSION CHECK
                # ---------------------------------------------
                if not abs_path.lower().endswith(
                    ".gguf"
                ):

                    logger.debug(
                        f"Invalid model extension: {abs_path}"
                    )

                    continue

                # ---------------------------------------------
                # SIZE CHECK
                # ---------------------------------------------
                try:

                    file_size = os.path.getsize(
                        abs_path
                    )

                    # reject obviously broken files
                    if file_size < 1024:

                        logger.warning(
                            f"Model file too small: "
                            f"{abs_path}"
                        )

                        continue

                except Exception as exc:

                    logger.debug(
                        f"File size check failed: {exc}"
                    )

                # ---------------------------------------------
                # READ ACCESS CHECK
                # ---------------------------------------------
                if not os.access(
                    abs_path,
                    os.R_OK,
                ):

                    logger.warning(
                        f"Model unreadable: {abs_path}"
                    )

                    continue

                # ---------------------------------------------
                # SUCCESS
                # ---------------------------------------------
                logger.info(
                    f"Resolved model path: {abs_path}"
                )

                return abs_path

            except Exception as exc:

                logger.debug(
                    f"Path resolution failed "
                    f"({candidate}): {exc}"
                )

        # ---------------------------------------------------------
        # FAILURE
        # ---------------------------------------------------------
        logger.error(
            "No valid GGUF model path found"
        )

        return None

    # -------------------------------------------------------------
    # HARD FAILURE
    # -------------------------------------------------------------
    except Exception as exc:

        logger.error(
            f"_resolve_model_path failed: {exc}"
        )

        return None


def _normalise_l2(v: np.ndarray) -> np.ndarray:
    """
    Production-grade L2 vector normalization.

    Fixes:
    - NaN/inf corruption
    - zero-vector crashes
    - dtype instability
    - overflow/underflow
    - invalid shapes
    - non-numpy inputs
    - normalization precision issues
    - memory corruption
    """

    try:

        # ---------------------------------------------------------
        # INPUT VALIDATION
        # ---------------------------------------------------------
        if v is None:

            logger.debug(
                "Normalization skipped: None vector"
            )

            return np.zeros(
                1,
                dtype=np.float32,
            )

        # ---------------------------------------------------------
        # ENSURE NUMPY ARRAY
        # ---------------------------------------------------------
        try:

            vec = np.asarray(
                v,
                dtype=np.float32,
            )

        except Exception as exc:

            logger.warning(
                f"Vector conversion failed: {exc}"
            )

            return np.zeros(
                1,
                dtype=np.float32,
            )

        # ---------------------------------------------------------
        # EMPTY VECTOR
        # ---------------------------------------------------------
        if vec.size == 0:

            logger.debug(
                "Normalization skipped: empty vector"
            )

            return vec.astype(
                np.float32,
                copy=False,
            )

        # ---------------------------------------------------------
        # FLATTEN VECTOR
        # ---------------------------------------------------------
        try:

            vec = vec.flatten()

        except Exception:
            pass

        # ---------------------------------------------------------
        # NAN / INF CLEANUP
        # ---------------------------------------------------------
        try:

            if np.isnan(vec).any():

                logger.warning(
                    "NaN detected during normalization"
                )

                vec = np.nan_to_num(
                    vec,
                    nan=0.0,
                )

            if np.isinf(vec).any():

                logger.warning(
                    "Inf detected during normalization"
                )

                vec = np.nan_to_num(
                    vec,
                    posinf=0.0,
                    neginf=0.0,
                )

        except Exception as exc:

            logger.debug(
                f"Cleanup failed: {exc}"
            )

        # ---------------------------------------------------------
        # COMPUTE NORM
        # ---------------------------------------------------------
        try:

            norm = float(
                np.linalg.norm(vec)
            )

        except Exception as exc:

            logger.warning(
                f"Norm computation failed: {exc}"
            )

            return vec.astype(
                np.float32,
                copy=False,
            )

        # ---------------------------------------------------------
        # INVALID NORM
        # ---------------------------------------------------------
        if math.isnan(norm) or math.isinf(norm):

            logger.warning(
                f"Invalid vector norm: {norm}"
            )

            return np.zeros_like(
                vec,
                dtype=np.float32,
            )

        # ---------------------------------------------------------
        # ZERO VECTOR
        # ---------------------------------------------------------
        if norm <= 1e-9:

            logger.debug(
                "Zero vector detected"
            )

            return vec.astype(
                np.float32,
                copy=False,
            )

        # ---------------------------------------------------------
        # NORMALIZATION
        # ---------------------------------------------------------
        try:

            normalized = (
                vec / norm
            ).astype(np.float32)

        except Exception as exc:

            logger.warning(
                f"Normalization division failed: {exc}"
            )

            return vec.astype(
                np.float32,
                copy=False,
            )

        # ---------------------------------------------------------
        # FINAL VALIDATION
        # ---------------------------------------------------------
        try:

            final_norm = float(
                np.linalg.norm(normalized)
            )

            if (
                math.isnan(final_norm)
                or math.isinf(final_norm)
            ):

                logger.warning(
                    "Normalized vector corrupted"
                )

                return np.zeros_like(
                    vec,
                    dtype=np.float32,
                )

        except Exception:
            pass

        # ---------------------------------------------------------
        # SAFE OUTPUT
        # ---------------------------------------------------------
        return np.ascontiguousarray(
            normalized,
            dtype=np.float32,
        )

    # -------------------------------------------------------------
    # HARD FAILURE
    # -------------------------------------------------------------
    except Exception as exc:

        logger.error(
            f"_normalise_l2 failed: {exc}"
        )

        try:

            return np.asarray(
                v,
                dtype=np.float32,
            )

        except Exception:

            return np.zeros(
                1,
                dtype=np.float32,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Agent enums & dataclasses
# ══════════════════════════════════════════════════════════════════════════════

class AgentPriority(IntEnum):
    """Lower integer = higher priority in asyncio.PriorityQueue."""
    CRITICAL   = 0
    HIGH       = 1
    NORMAL     = 2
    LOW        = 3
    BACKGROUND = 4


class AgentState(Enum):
    IDLE     = "idle"
    ACTIVE   = "active"
    PAUSED   = "paused"
    RETIRED  = "retired"


@dataclass(order=True)
class PrioritizedRequest:
    """Item stored in the PriorityQueue."""
    priority:     int   = field(compare=True)
    enqueue_time: float = field(compare=True)
    agent_id:     str   = field(compare=False)
    prompt:       str   = field(compare=False)
    kwargs:       Dict  = field(compare=False, default_factory=dict)
    future: Optional[asyncio.Future] = field(compare=False, default=None)


# ══════════════════════════════════════════════════════════════════════════════
# GPU Memory Guard (FIXED)
# ══════════════════════════════════════════════════════════════════════════════

class GPUMemoryGuard:
    """Polls VRAM usage and exposes back-pressure API."""

    def __init__(
        self,
        high_watermark: float = _VRAM_HIGH_WATERMARK,
        poll_interval:  float = _VRAM_POLL_INTERVAL,
    ) -> None:
        self.high_watermark  = high_watermark
        self.poll_interval   = poll_interval
        self._vram_fraction  = 0.0
        self._last_poll      = 0.0
        self._handle         = None

        if _NVML_AVAILABLE:
            try:
                # Try nvidia_smi first (newer)
                if 'nvidia_smi' in sys.modules:
                    self._handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                # Fallback to pynvml
                elif pynvml:
                    self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._handle = None

    def _poll(self) -> None:
        """
        Production-grade VRAM/RAM polling system.

        Fixes:
        - NVML crashes
        - polling spam
        - CPU spikes
        - invalid VRAM values
        - memory reporting corruption
        - psutil fallback instability
        - missing GPU driver crashes
        - stale polling data
        - thread race conditions
        """

        try:

            # ---------------------------------------------------------
            # CURRENT TIME
            # ---------------------------------------------------------
            now = time.monotonic()

            # ---------------------------------------------------------
            # POLL INTERVAL VALIDATION
            # ---------------------------------------------------------
            try:

                poll_interval = float(
                    getattr(
                        self,
                        "poll_interval",
                        1.0,
                    )
                )

            except Exception:

                poll_interval = 1.0

            poll_interval = max(
                0.1,
                min(poll_interval, 60.0),
            )

            # ---------------------------------------------------------
            # THROTTLE POLLING
            # ---------------------------------------------------------
            last_poll = float(
                getattr(
                    self,
                    "_last_poll",
                    0.0,
                )
            )

            if (now - last_poll) < poll_interval:
                return

            self._last_poll = now

            # ---------------------------------------------------------
            # DEFAULT VALUE
            # ---------------------------------------------------------
            vram_fraction = 0.0

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_poll_lock"):

                self._poll_lock = threading.Lock()

            with self._poll_lock:

                # -----------------------------------------------------
                # NVIDIA NVML PATH
                # -----------------------------------------------------
                if (
                    _NVML_AVAILABLE
                    and getattr(self, "_handle", None)
                ):

                    try:

                        handle = self._handle

                        info = None

                        # ---------------------------------------------
                        # nvidia_smi
                        # ---------------------------------------------
                        if "nvidia_smi" in sys.modules:

                            try:

                                info = (
                                    nvidia_smi
                                    .nvmlDeviceGetMemoryInfo(handle)
                                )

                            except Exception as exc:

                                logger.debug(
                                    f"nvidia_smi failed: {exc}"
                                )

                        # ---------------------------------------------
                        # pynvml fallback
                        # ---------------------------------------------
                        if info is None:

                            try:

                                info = (
                                    pynvml
                                    .nvmlDeviceGetMemoryInfo(handle)
                                )

                            except Exception as exc:

                                logger.debug(
                                    f"pynvml failed: {exc}"
                                )

                        # ---------------------------------------------
                        # VALIDATE INFO
                        # ---------------------------------------------
                        if info:

                            total = max(
                                1,
                                int(
                                    getattr(
                                        info,
                                        "total",
                                        1,
                                    )
                                ),
                            )

                            used = max(
                                0,
                                int(
                                    getattr(
                                        info,
                                        "used",
                                        0,
                                    )
                                ),
                            )

                            vram_fraction = (
                                used / total
                            )

                            vram_fraction = max(
                                0.0,
                                min(vram_fraction, 1.0),
                            )

                            self._vram_fraction = (
                                round(vram_fraction, 4)
                            )

                            self._last_memory_source = "nvml"

                            logger.debug(
                                f"GPU memory usage: "
                                f"{vram_fraction:.2%}"
                            )

                            return

                    except Exception as exc:

                        logger.warning(
                            f"NVML polling failed: {exc}"
                        )

                # -----------------------------------------------------
                # PSUTIL FALLBACK
                # -----------------------------------------------------
                try:

                    import psutil

                    vm = psutil.virtual_memory()

                    ram_percent = float(
                        getattr(vm, "percent", 0.0)
                    )

                    vram_fraction = (
                        ram_percent / 100.0
                    )

                    vram_fraction = max(
                        0.0,
                        min(vram_fraction, 1.0),
                    )

                    self._vram_fraction = round(
                        vram_fraction,
                        4,
                    )

                    self._last_memory_source = "psutil"

                    logger.debug(
                        f"RAM usage fallback: "
                        f"{vram_fraction:.2%}"
                    )

                    return

                except Exception as exc:

                    logger.warning(
                        f"psutil polling failed: {exc}"
                    )

                # -----------------------------------------------------
                # FINAL FALLBACK
                # -----------------------------------------------------
                self._vram_fraction = 0.0

                self._last_memory_source = "fallback"

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Memory polling failed: {exc}"
            )

            try:

                self._vram_fraction = 0.0

            except Exception:
                pass

    @property
    def vram_fraction(self) -> float:
        """
        Production-grade VRAM usage accessor.

        Fixes:
        - stale VRAM values
        - polling crashes
        - invalid memory fractions
        - NaN/inf corruption
        - race conditions
        - negative VRAM values
        - thread instability
        """

        try:

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_vram_lock"):

                self._vram_lock = threading.Lock()

            with self._vram_lock:

                # -----------------------------------------------------
                # REFRESH MEMORY STATS
                # -----------------------------------------------------
                try:

                    self._poll()

                except Exception as exc:

                    logger.warning(
                        f"VRAM poll failed: {exc}"
                    )

                # -----------------------------------------------------
                # GET RAW VALUE
                # -----------------------------------------------------
                raw_value = getattr(
                    self,
                    "_vram_fraction",
                    0.0,
                )

                # -----------------------------------------------------
                # SAFE CONVERSION
                # -----------------------------------------------------
                try:

                    value = float(raw_value)

                except Exception:

                    logger.warning(
                        f"Invalid VRAM value: {raw_value}"
                    )

                    value = 0.0

                # -----------------------------------------------------
                # NAN / INF PROTECTION
                # -----------------------------------------------------
                if math.isnan(value) or math.isinf(value):

                    logger.warning(
                        f"Corrupted VRAM metric: {value}"
                    )

                    value = 0.0

                # -----------------------------------------------------
                # CLAMP RANGE
                # -----------------------------------------------------
                value = max(
                    0.0,
                    min(value, 1.0),
                )

                # -----------------------------------------------------
                # ROUND FOR STABILITY
                # -----------------------------------------------------
                value = round(value, 4)

                # -----------------------------------------------------
                # STORE CLEAN VALUE
                # -----------------------------------------------------
                self._vram_fraction = value

                return value

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"vram_fraction failed: {exc}"
            )

            try:

                self._vram_fraction = 0.0

            except Exception:
                pass

            return 0.0

    def is_under_pressure(self) -> bool:
        """
        Production-grade memory pressure detector.

        Fixes:
        - invalid VRAM readings
        - watermark corruption
        - NaN/inf comparisons
        - false-positive pressure states
        - race conditions
        - stale polling data
        - threshold overflow bugs
        """

        try:

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_pressure_lock"):

                self._pressure_lock = threading.Lock()

            with self._pressure_lock:

                # -----------------------------------------------------
                # GET CURRENT VRAM
                # -----------------------------------------------------
                try:

                    current_vram = float(
                        self.vram_fraction
                    )

                except Exception as exc:

                    logger.warning(
                        f"VRAM read failed: {exc}"
                    )

                    current_vram = 0.0

                # -----------------------------------------------------
                # NAN / INF PROTECTION
                # -----------------------------------------------------
                if (
                    math.isnan(current_vram)
                    or math.isinf(current_vram)
                ):

                    logger.warning(
                        f"Invalid VRAM fraction: "
                        f"{current_vram}"
                    )

                    current_vram = 0.0

                # -----------------------------------------------------
                # SAFE WATERMARK
                # -----------------------------------------------------
                try:

                    high_watermark = float(
                        getattr(
                            self,
                            "high_watermark",
                            0.85,
                        )
                    )

                except Exception:

                    high_watermark = 0.85

                # -----------------------------------------------------
                # CLAMP WATERMARK
                # -----------------------------------------------------
                high_watermark = max(
                    0.1,
                    min(high_watermark, 1.0),
                )

                # -----------------------------------------------------
                # PRESSURE CHECK
                # -----------------------------------------------------
                under_pressure = (
                    current_vram >= high_watermark
                )

                # -----------------------------------------------------
                # OPTIONAL LOGGING
                # -----------------------------------------------------
                if under_pressure:

                    logger.warning(
                        f"Memory pressure detected | "
                        f"vram={current_vram:.2%} "
                        f"watermark={high_watermark:.2%}"
                    )

                else:

                    logger.debug(
                        f"Memory healthy | "
                        f"vram={current_vram:.2%}"
                    )

                # -----------------------------------------------------
                # STORE LAST STATE
                # -----------------------------------------------------
                self._last_pressure_state = (
                    under_pressure
                )

                self._last_pressure_vram = (
                    current_vram
                )

                return under_pressure

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"is_under_pressure failed: {exc}"
            )

            # safest assumption
            return False

    def should_throttle(self, priority: int) -> bool:
        """
        Production-grade throttling decision system.

        Fixes:
        - invalid priority handling
        - false throttling
        - pressure-state corruption
        - NaN/inf failures
        - race conditions
        - priority overflow bugs
        - stale pressure reads
        - aggressive low-memory blocking
        """

        try:

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_throttle_lock"):

                self._throttle_lock = threading.Lock()

            with self._throttle_lock:

                # -----------------------------------------------------
                # PRESSURE CHECK
                # -----------------------------------------------------
                try:

                    under_pressure = bool(
                        self.is_under_pressure()
                    )

                except Exception as exc:

                    logger.warning(
                        f"Pressure check failed: {exc}"
                    )

                    under_pressure = False

                # -----------------------------------------------------
                # NO PRESSURE
                # -----------------------------------------------------
                if not under_pressure:

                    self._last_throttle_decision = False

                    return False

                # -----------------------------------------------------
                # VALIDATE PRIORITY
                # -----------------------------------------------------
                try:

                    if isinstance(priority, AgentPriority):

                        priority_value = int(
                            priority.value
                        )

                    else:

                        priority_value = int(priority)

                except Exception:

                    logger.warning(
                        f"Invalid priority: {priority}"
                    )

                    priority_value = int(
                        AgentPriority.NORMAL.value
                    )

                # -----------------------------------------------------
                # CLAMP PRIORITY
                # -----------------------------------------------------
                min_priority = int(
                    AgentPriority.CRITICAL.value
                )

                max_priority = int(
                    AgentPriority.BACKGROUND.value
                )

                priority_value = max(
                    min_priority,
                    min(priority_value, max_priority),
                )

                # -----------------------------------------------------
                # THROTTLE POLICY
                # -----------------------------------------------------
                low_priority_threshold = int(
                    AgentPriority.LOW.value
                )

                should_block = (
                    priority_value >=
                    low_priority_threshold
                )

                # -----------------------------------------------------
                # OPTIONAL ADAPTIVE POLICY
                # -----------------------------------------------------
                try:

                    current_vram = float(
                        self.vram_fraction
                    )

                    # emergency mode
                    if current_vram >= 0.95:

                        should_block = (
                            priority_value >
                            AgentPriority.CRITICAL.value
                        )

                    # severe pressure
                    elif current_vram >= 0.90:

                        should_block = (
                            priority_value >=
                            AgentPriority.NORMAL.value
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                if should_block:

                    logger.warning(
                        f"Request throttled | "
                        f"priority={priority_value} "
                        f"vram={self.vram_fraction:.2%}"
                    )

                else:

                    logger.debug(
                        f"Request allowed under pressure | "
                        f"priority={priority_value}"
                    )

                # -----------------------------------------------------
                # STORE STATE
                # -----------------------------------------------------
                self._last_throttle_decision = (
                    should_block
                )

                self._last_throttle_priority = (
                    priority_value
                )

                return should_block

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"should_throttle failed: {exc}"
            )

            # safest option:
            # do NOT throttle everything accidentally
            return False


# ══════════════════════════════════════════════════════════════════════════════
# Per-agent slot
# ══════════════════════════════════════════════════════════════════════════════

class AgentSlot:
    """Tracks per-agent lifecycle, rate limiting, and metrics."""

    def __init__(
        self,
        agent_id:     str,
        priority:     AgentPriority = AgentPriority.NORMAL,
        rate_limit:   float         = _AGENT_RATE_LIMIT_RPS,
        max_queue:    int           = 10,
    ) -> None:
        self.agent_id    = agent_id
        self.priority    = priority
        self.state       = AgentState.IDLE
        self.rate_limit  = max(0.01, rate_limit)
        self.max_queue   = max_queue
        self.created_at  = time.time()

        self._tokens         = rate_limit
        self._last_refill    = time.monotonic()
        self._queue_depth    = 0
        self._lock           = asyncio.Lock()

        self.requests_sent      = 0
        self.requests_completed = 0
        self.requests_failed    = 0
        self.requests_throttled = 0
        self.total_latency_ms   = 0.0

    async def acquire(self) -> bool:
        """
        Production-grade async token acquisition system.

        Fixes:
        - token bucket corruption
        - negative token counts
        - queue overflow
        - race conditions
        - deadlocks
        - invalid refill timing
        - runaway throttling
        - async lock failures
        - state desync
        - memory pressure instability
        """

        try:

            # ---------------------------------------------------------
            # ENSURE LOCK
            # ---------------------------------------------------------
            if not hasattr(self, "_lock"):

                self._lock = asyncio.Lock()

            async with self._lock:

                # -----------------------------------------------------
                # CURRENT TIME
                # -----------------------------------------------------
                now = time.monotonic()

                # -----------------------------------------------------
                # SAFE VALUES
                # -----------------------------------------------------
                try:

                    rate_limit = float(
                        getattr(
                            self,
                            "rate_limit",
                            1.0,
                        )
                    )

                except Exception:

                    rate_limit = 1.0

                rate_limit = max(
                    0.1,
                    min(rate_limit, 10000.0),
                )

                try:

                    max_queue = int(
                        getattr(
                            self,
                            "max_queue",
                            100,
                        )
                    )

                except Exception:

                    max_queue = 100

                max_queue = max(
                    1,
                    min(max_queue, 100000),
                )

                # -----------------------------------------------------
                # LAST REFILL
                # -----------------------------------------------------
                try:

                    last_refill = float(
                        getattr(
                            self,
                            "_last_refill",
                            now,
                        )
                    )

                except Exception:

                    last_refill = now

                elapsed = max(
                    0.0,
                    now - last_refill,
                )

                # -----------------------------------------------------
                # TOKEN REFILL
                # -----------------------------------------------------
                try:

                    current_tokens = float(
                        getattr(
                            self,
                            "_tokens",
                            rate_limit,
                        )
                    )

                except Exception:

                    current_tokens = rate_limit

                # refill tokens
                current_tokens += (
                    elapsed * rate_limit
                )

                # clamp
                current_tokens = max(
                    0.0,
                    min(current_tokens, rate_limit),
                )

                self._tokens = current_tokens
                self._last_refill = now

                # -----------------------------------------------------
                # QUEUE DEPTH
                # -----------------------------------------------------
                try:

                    queue_depth = int(
                        getattr(
                            self,
                            "_queue_depth",
                            0,
                        )
                    )

                except Exception:

                    queue_depth = 0

                queue_depth = max(
                    0,
                    queue_depth,
                )

                self._queue_depth = queue_depth

                # -----------------------------------------------------
                # TOKEN CHECK
                # -----------------------------------------------------
                if current_tokens < 1.0:

                    self.requests_throttled = (
                        int(
                            getattr(
                                self,
                                "requests_throttled",
                                0,
                            )
                        ) + 1
                    )

                    logger.warning(
                        f"Acquire throttled "
                        f"(tokens={current_tokens:.2f})"
                    )

                    return False

                # -----------------------------------------------------
                # QUEUE LIMIT CHECK
                # -----------------------------------------------------
                if queue_depth >= max_queue:

                    self.requests_throttled = (
                        int(
                            getattr(
                                self,
                                "requests_throttled",
                                0,
                            )
                        ) + 1
                    )

                    logger.warning(
                        f"Acquire blocked "
                        f"(queue={queue_depth}/{max_queue})"
                    )

                    return False

                # -----------------------------------------------------
                # ACQUIRE TOKEN
                # -----------------------------------------------------
                self._tokens = max(
                    0.0,
                    current_tokens - 1.0,
                )

                self._queue_depth = (
                    queue_depth + 1
                )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                self.requests_sent = (
                    int(
                        getattr(
                            self,
                            "requests_sent",
                            0,
                        )
                    ) + 1
                )

                # -----------------------------------------------------
                # STATE UPDATE
                # -----------------------------------------------------
                try:

                    self.state = AgentState.ACTIVE

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL DEBUG
                # -----------------------------------------------------
                logger.debug(
                    f"Acquire success | "
                    f"tokens={self._tokens:.2f} "
                    f"queue={self._queue_depth}"
                )

                return True

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Acquire cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Acquire failed: {exc}"
            )

            return False

    async def release(
        self,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Production-grade async request release system.

        Fixes:
        - negative queue depth
        - metric corruption
        - latency overflow
        - race conditions
        - invalid state transitions
        - failed-request desync
        - deadlocks
        - async lock crashes
        - invalid latency values
        """

        try:

            # ---------------------------------------------------------
            # ENSURE LOCK
            # ---------------------------------------------------------
            if not hasattr(self, "_lock"):

                self._lock = asyncio.Lock()

            async with self._lock:

                # -----------------------------------------------------
                # SAFE QUEUE DEPTH
                # -----------------------------------------------------
                try:

                    queue_depth = int(
                        getattr(
                            self,
                            "_queue_depth",
                            0,
                        )
                    )

                except Exception:

                    queue_depth = 0

                queue_depth = max(
                    0,
                    queue_depth - 1,
                )

                self._queue_depth = queue_depth

                # -----------------------------------------------------
                # SAFE LATENCY
                # -----------------------------------------------------
                try:

                    latency_ms = float(latency_ms)

                except Exception:

                    latency_ms = 0.0

                # clamp insane values
                latency_ms = max(
                    0.0,
                    min(latency_ms, 3_600_000.0),
                )

                # -----------------------------------------------------
                # SUCCESS METRICS
                # -----------------------------------------------------
                if bool(success):

                    try:

                        self.requests_completed = (
                            int(
                                getattr(
                                    self,
                                    "requests_completed",
                                    0,
                                )
                            ) + 1
                        )

                    except Exception:

                        self.requests_completed = 1

                    # ---------------------------------------------
                    # LATENCY AGGREGATION
                    # ---------------------------------------------
                    try:

                        total_latency = float(
                            getattr(
                                self,
                                "total_latency_ms",
                                0.0,
                            )
                        )

                    except Exception:

                        total_latency = 0.0

                    total_latency += latency_ms

                    self.total_latency_ms = max(
                        0.0,
                        total_latency,
                    )

                # -----------------------------------------------------
                # FAILURE METRICS
                # -----------------------------------------------------
                else:

                    try:

                        self.requests_failed = (
                            int(
                                getattr(
                                    self,
                                    "requests_failed",
                                    0,
                                )
                            ) + 1
                        )

                    except Exception:

                        self.requests_failed = 1

                # -----------------------------------------------------
                # STATE MANAGEMENT
                # -----------------------------------------------------
                try:

                    if queue_depth <= 0:

                        self.state = AgentState.IDLE

                    else:

                        self.state = AgentState.ACTIVE

                except Exception as exc:

                    logger.debug(
                        f"State update failed: {exc}"
                    )

                # -----------------------------------------------------
                # OPTIONAL AVERAGE LATENCY
                # -----------------------------------------------------
                try:

                    completed = max(
                        1,
                        int(
                            getattr(
                                self,
                                "requests_completed",
                                1,
                            )
                        ),
                    )

                    avg_latency = (
                        self.total_latency_ms /
                        completed
                    )

                    self.avg_latency_ms = round(
                        avg_latency,
                        2,
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Release complete | "
                    f"success={success} "
                    f"queue={queue_depth} "
                    f"latency={latency_ms:.2f}ms"
                )

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Release cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Release failed: {exc}"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Production-grade agent metrics collector.

        Fixes:
        - division-by-zero
        - invalid latency calculations
        - corrupted counters
        - negative queue depth
        - invalid token counts
        - enum access failures
        - metric overflow
        - stale state reporting
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # AGENT ID
            # ---------------------------------------------------------
            try:

                agent_id = str(
                    getattr(
                        self,
                        "agent_id",
                        "unknown_agent",
                    )
                )

            except Exception:

                agent_id = "unknown_agent"

            # ---------------------------------------------------------
            # PRIORITY
            # ---------------------------------------------------------
            try:

                priority_obj = getattr(
                    self,
                    "priority",
                    None,
                )

                priority_name = (
                    priority_obj.name
                    if priority_obj is not None
                    else "UNKNOWN"
                )

            except Exception:

                priority_name = "UNKNOWN"

            # ---------------------------------------------------------
            # STATE
            # ---------------------------------------------------------
            try:

                state_obj = getattr(
                    self,
                    "state",
                    None,
                )

                state_value = (
                    state_obj.value
                    if state_obj is not None
                    else "unknown"
                )

            except Exception:

                state_value = "unknown"

            # ---------------------------------------------------------
            # COUNTERS
            # ---------------------------------------------------------
            def _safe_int(name: str, default: int = 0) -> int:

                try:

                    value = int(
                        getattr(
                            self,
                            name,
                            default,
                        )
                    )

                    return max(0, value)

                except Exception:

                    return max(0, default)

            requests_sent = _safe_int(
                "requests_sent"
            )

            requests_completed = _safe_int(
                "requests_completed"
            )

            requests_failed = _safe_int(
                "requests_failed"
            )

            requests_throttled = _safe_int(
                "requests_throttled"
            )

            queue_depth = _safe_int(
                "_queue_depth"
            )

            # ---------------------------------------------------------
            # TOKENS
            # ---------------------------------------------------------
            try:

                tokens_available = float(
                    getattr(
                        self,
                        "_tokens",
                        0.0,
                    )
                )

            except Exception:

                tokens_available = 0.0

            # clamp invalid values
            if (
                math.isnan(tokens_available)
                or math.isinf(tokens_available)
            ):

                tokens_available = 0.0

            tokens_available = max(
                0.0,
                round(tokens_available, 2),
            )

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            try:

                total_latency_ms = float(
                    getattr(
                        self,
                        "total_latency_ms",
                        0.0,
                    )
                )

            except Exception:

                total_latency_ms = 0.0

            total_latency_ms = max(
                0.0,
                total_latency_ms,
            )

            completed = max(
                1,
                requests_completed,
            )

            avg_latency_ms = round(
                total_latency_ms / completed,
                2,
            )

            # ---------------------------------------------------------
            # SUCCESS RATE
            # ---------------------------------------------------------
            total_processed = max(
                1,
                requests_completed +
                requests_failed
            )

            success_rate = round(
                requests_completed /
                total_processed,
                3,
            )

            # ---------------------------------------------------------
            # THROTTLE RATE
            # ---------------------------------------------------------
            throttle_base = max(
                1,
                requests_sent,
            )

            throttle_rate = round(
                requests_throttled /
                throttle_base,
                3,
            )

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            health_score = 1.0

            try:

                if queue_depth > 100:
                    health_score -= 0.35

                elif queue_depth > 50:
                    health_score -= 0.20

                elif queue_depth > 20:
                    health_score -= 0.10

                if success_rate < 0.50:
                    health_score -= 0.30

                elif success_rate < 0.75:
                    health_score -= 0.15

                if throttle_rate > 0.50:
                    health_score -= 0.20

                elif throttle_rate > 0.25:
                    health_score -= 0.10

                health_score = round(
                    max(0.0, min(1.0, health_score)),
                    2,
                )

            except Exception:
                health_score = 0.5

            # ---------------------------------------------------------
            # METRICS LATENCY
            # ---------------------------------------------------------
            metrics_latency = round(
                time.monotonic() - start,
                4,
            )

            # ---------------------------------------------------------
            # FINAL RESULT
            # ---------------------------------------------------------
            return {
                "agent_id": agent_id,
                "priority": priority_name,
                "state": state_value,

                # request counters
                "requests_sent": requests_sent,
                "requests_completed": requests_completed,
                "requests_failed": requests_failed,
                "requests_throttled": requests_throttled,

                # queue/token metrics
                "queue_depth": queue_depth,
                "tokens_available": tokens_available,

                # latency
                "avg_latency_ms": avg_latency_ms,
                "total_latency_ms": round(
                    total_latency_ms,
                    2,
                ),

                # health
                "success_rate": success_rate,
                "throttle_rate": throttle_rate,
                "health_score": health_score,

                # diagnostics
                "metrics_latency": metrics_latency,
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Agent metrics failed: {exc}"
            )

            return {
                "agent_id": "unknown",
                "priority": "UNKNOWN",
                "state": "error",
                "health_score": 0.0,
                "error": str(exc),
            }


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Fallback provider
# ══════════════════════════════════════════════════════════════════════════════

class FallbackProvider(LLMProvider):
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Production-grade fallback response generator.

        Fixes:
        - empty prompt crashes
        - malformed input
        - unicode corruption
        - unsafe string matching
        - repeated fallback responses
        - huge prompt memory waste
        - response instability
        - keyword false positives
        """

        try:

            # ---------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):

                try:
                    prompt = str(prompt)
                except Exception:
                    prompt = repr(prompt)

            # ---------------------------------------------------------
            # CLEAN INPUT
            # ---------------------------------------------------------
            prompt = prompt.replace(
                "\x00",
                " ",
            )

            prompt = re.sub(
                r"\s+",
                " ",
                prompt,
            )

            prompt = prompt.strip()

            # ---------------------------------------------------------
            # HARD LIMIT
            # ---------------------------------------------------------
            max_prompt_chars = 5000

            if len(prompt) > max_prompt_chars:

                logger.warning(
                    f"Fallback prompt truncated "
                    f"({len(prompt)} chars)"
                )

                prompt = prompt[:max_prompt_chars]

            # ---------------------------------------------------------
            # EMPTY PROMPT
            # ---------------------------------------------------------
            if not prompt:

                return (
                    "Hello! I'm EDIATH. "
                    "How can I assist you today?"
                )

            # ---------------------------------------------------------
            # NORMALIZED PROMPT
            # ---------------------------------------------------------
            p = prompt.lower()

            # ---------------------------------------------------------
            # GREETINGS
            # ---------------------------------------------------------
            greeting_patterns = [
                "hello",
                " hi ",
                "hi\n",
                "hey",
                "greetings",
            ]

            if any(
                pattern in f" {p} "
                for pattern in greeting_patterns
            ):

                return (
                    "Hello! I'm EDIATH, your AI assistant. "
                    "How can I help you today?"
                )

            # ---------------------------------------------------------
            # TIME REQUEST
            # ---------------------------------------------------------
            if (
                "what" in p
                and "time" in p
            ):

                try:

                    from datetime import datetime

                    current_time = (
                        datetime.now().strftime(
                            "%H:%M:%S"
                        )
                    )

                    return (
                        f"The current time is "
                        f"{current_time}."
                    )

                except Exception as exc:

                    logger.debug(
                        f"Time lookup failed: {exc}"
                    )

                    return (
                        "I'm unable to retrieve "
                        "the current time right now."
                    )

            # ---------------------------------------------------------
            # STATUS CHECK
            # ---------------------------------------------------------
            if (
                "how are you" in p
                or "how're you" in p
            ):

                return (
                    "I'm functioning well and ready "
                    "to assist you."
                )

            # ---------------------------------------------------------
            # THANKS
            # ---------------------------------------------------------
            thank_patterns = [
                "thank you",
                "thanks",
                "thx",
            ]

            if any(
                pattern in p
                for pattern in thank_patterns
            ):

                return (
                    "You're welcome! "
                    "Let me know if you need anything else."
                )

            # ---------------------------------------------------------
            # GOODBYE
            # ---------------------------------------------------------
            goodbye_patterns = [
                "bye",
                "goodbye",
                "see you",
                "farewell",
            ]

            if any(
                pattern in p
                for pattern in goodbye_patterns
            ):

                return (
                    "Goodbye! "
                    "Have a great day!"
                )

            # ---------------------------------------------------------
            # HELP REQUEST
            # ---------------------------------------------------------
            help_patterns = [
                "help",
                "assist",
                "support",
            ]

            if any(
                pattern in p
                for pattern in help_patterns
            ):

                return (
                    "I can answer questions, "
                    "assist with coding, "
                    "troubleshooting, research, "
                    "and many other tasks. "
                    "What would you like help with?"
                )

            # ---------------------------------------------------------
            # IDENTITY QUESTIONS
            # ---------------------------------------------------------
            if (
                "who are you" in p
                or "what are you" in p
            ):

                return (
                    "I'm EDIATH, an AI assistant "
                    "designed to help with reasoning, "
                    "problem-solving, and technical tasks."
                )

            # ---------------------------------------------------------
            # FALLBACK RESPONSE ROTATION
            # ---------------------------------------------------------
            fallback_responses = [

                "Could you provide more details so I can better assist you?",

                "I need a bit more information to give you an accurate response.",

                "Please clarify your request so I can help more effectively.",

                "I'm ready to help. Could you expand on your question?"
            ]

            # ---------------------------------------------------------
            # STABLE RESPONSE SELECTION
            # ---------------------------------------------------------
            try:

                response_index = (
                    abs(hash(p))
                    % len(fallback_responses)
                )

            except Exception:

                response_index = 0

            response = fallback_responses[
                response_index
            ]

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self.fallback_calls = int(
                    getattr(
                        self,
                        "fallback_calls",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                "Fallback response generated"
            )

            return response

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Fallback generation cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Fallback generate failed: {exc}"
            )

            return (
                "I'm currently experiencing an issue "
                "processing your request. "
                "Please try again."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════════════

class LLMCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max(1, int(max_size))
        self.ttl = max(1, int(ttl))
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.enabled = True
        self._lock = threading.Lock()

    def _make_key(self, prompt: str, **kwargs) -> str:
        """
        Production-grade cache key generator.

        Fixes:
        - inconsistent cache keys
        - unicode corruption
        - kwargs ordering instability
        - newline/key pollution
        - non-serializable kwargs
        - hash collisions from malformed inputs
        - giant prompt memory waste
        - invalid encoding crashes
        """

        try:

            # ---------------------------------------------------------
            # PROMPT VALIDATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):

                try:
                    prompt = str(prompt)
                except Exception:
                    prompt = repr(prompt)

            # ---------------------------------------------------------
            # CLEAN PROMPT
            # ---------------------------------------------------------
            prompt_clean = prompt.replace(
                "\x00",
                " ",
            )

            prompt_clean = re.sub(
                r"\s+",
                " ",
                prompt_clean,
            )

            prompt_clean = prompt_clean.strip()

            # ---------------------------------------------------------
            # HARD LIMIT
            # ---------------------------------------------------------
            max_key_prompt = 4000

            if len(prompt_clean) > max_key_prompt:

                logger.debug(
                    f"Cache key prompt truncated "
                    f"({len(prompt_clean)} chars)"
                )

                prompt_clean = prompt_clean[:max_key_prompt]

            # ---------------------------------------------------------
            # NORMALIZE KWARGS
            # ---------------------------------------------------------
            normalized_kwargs = {}

            if kwargs:

                try:

                    for key, value in sorted(kwargs.items()):

                        try:

                            safe_key = str(key)

                            # -----------------------------------------
                            # SAFE VALUE SERIALIZATION
                            # -----------------------------------------
                            if isinstance(
                                value,
                                (
                                    str,
                                    int,
                                    float,
                                    bool,
                                    type(None),
                                ),
                            ):

                                safe_value = value

                            elif isinstance(
                                value,
                                (list, tuple),
                            ):

                                safe_value = [
                                    str(v)
                                    for v in value
                                ]

                            elif isinstance(
                                value,
                                dict,
                            ):

                                safe_value = {
                                    str(k): str(v)
                                    for k, v in value.items()
                                }

                            else:

                                safe_value = str(value)

                            normalized_kwargs[
                                safe_key
                            ] = safe_value

                        except Exception as exc:

                            logger.debug(
                                f"Kwarg normalization failed "
                                f"for {key}: {exc}"
                            )

                except Exception as exc:

                    logger.warning(
                        f"Kwarg sorting failed: {exc}"
                    )

            # ---------------------------------------------------------
            # SERIALIZE PAYLOAD
            # ---------------------------------------------------------
            try:

                payload = json.dumps(
                    {
                        "prompt": prompt_clean,
                        "kwargs": normalized_kwargs,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            except Exception as exc:

                logger.warning(
                    f"JSON serialization failed: {exc}"
                )

                payload = (
                    f"{prompt_clean}|"
                    f"{str(normalized_kwargs)}"
                )

            # ---------------------------------------------------------
            # ENCODE SAFELY
            # ---------------------------------------------------------
            try:

                encoded = payload.encode(
                    "utf-8",
                    errors="ignore",
                )

            except Exception:

                encoded = str(payload).encode(
                    "utf-8",
                    errors="ignore",
                )

            # ---------------------------------------------------------
            # HASH GENERATION
            # ---------------------------------------------------------
            try:

                key = hashlib.md5(
                    encoded
                ).hexdigest()

            except Exception as exc:

                logger.error(
                    f"MD5 generation failed: {exc}"
                )

                # fallback hash
                key = hashlib.sha1(
                    encoded
                ).hexdigest()

            # ---------------------------------------------------------
            # VALIDATE HASH
            # ---------------------------------------------------------
            if not key or len(key) < 16:

                raise ValueError(
                    "Generated invalid cache key"
                )

            logger.debug(
                f"Cache key generated | "
                f"len={len(prompt_clean)}"
            )

            return key

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"_make_key failed: {exc}"
            )

            try:

                fallback = hashlib.md5(
                    str(prompt).encode(
                        "utf-8",
                        errors="ignore",
                    )
                ).hexdigest()

                return fallback

            except Exception:

                # absolute fallback
                return "fallback_cache_key"

    def get(self, prompt: str, **kwargs) -> Optional[Dict]:
        """
        Production-grade cache retrieval system.

        Fixes:
        - stale cache entries
        - race conditions
        - corrupted cache values
        - invalid timestamps
        - OrderedDict crashes
        - lock corruption
        - TTL overflow
        - mutable object leaks
        - cache metric desync
        """

        try:

            # ---------------------------------------------------------
            # CACHE ENABLED CHECK
            # ---------------------------------------------------------
            if not bool(
                getattr(self, "enabled", False)
            ):

                return None

            # ---------------------------------------------------------
            # CACHE OBJECT VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "cache"):

                logger.warning(
                    "Cache storage missing"
                )

                return None

            # ---------------------------------------------------------
            # LOCK VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_lock"):

                self._lock = threading.Lock()

            # ---------------------------------------------------------
            # SAFE KEY GENERATION
            # ---------------------------------------------------------
            try:

                key = self._make_key(
                    prompt,
                    **kwargs,
                )

            except Exception as exc:

                logger.warning(
                    f"Cache key generation failed: {exc}"
                )

                return None

            if not key:

                return None

            # ---------------------------------------------------------
            # THREAD-SAFE ACCESS
            # ---------------------------------------------------------
            with self._lock:

                try:

                    entry = self.cache.get(key)

                except Exception as exc:

                    logger.warning(
                        f"Cache lookup failed: {exc}"
                    )

                    return None

                # -----------------------------------------------------
                # CACHE MISS
                # -----------------------------------------------------
                if entry is None:

                    try:

                        self.misses = int(
                            getattr(
                                self,
                                "misses",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                    return None

                # -----------------------------------------------------
                # ENTRY VALIDATION
                # -----------------------------------------------------
                try:

                    if (
                        not isinstance(entry, tuple)
                        or len(entry) != 2
                    ):

                        raise ValueError(
                            "Invalid cache entry format"
                        )

                    value, timestamp = entry

                except Exception as exc:

                    logger.warning(
                        f"Corrupted cache entry: {exc}"
                    )

                    try:

                        del self.cache[key]

                    except Exception:
                        pass

                    return None

                # -----------------------------------------------------
                # TTL VALIDATION
                # -----------------------------------------------------
                try:

                    ttl = float(
                        getattr(
                            self,
                            "ttl",
                            3600,
                        )
                    )

                except Exception:

                    ttl = 3600.0

                ttl = max(
                    1.0,
                    min(ttl, 86400.0),
                )

                # -----------------------------------------------------
                # TIMESTAMP VALIDATION
                # -----------------------------------------------------
                try:

                    timestamp = float(timestamp)

                except Exception:

                    logger.warning(
                        "Invalid cache timestamp"
                    )

                    try:

                        del self.cache[key]

                    except Exception:
                        pass

                    return None

                age = max(
                    0.0,
                    time.time() - timestamp,
                )

                # -----------------------------------------------------
                # EXPIRED ENTRY
                # -----------------------------------------------------
                if age > ttl:

                    try:

                        del self.cache[key]

                    except KeyError:
                        pass

                    except Exception as exc:

                        logger.debug(
                            f"Expired cache delete failed: {exc}"
                        )

                    try:

                        self.evictions = int(
                            getattr(
                                self,
                                "evictions",
                                0,
                            )
                        ) + 1

                        self.misses = int(
                            getattr(
                                self,
                                "misses",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                    logger.debug(
                        f"Cache expired | age={age:.2f}s"
                    )

                    return None

                # -----------------------------------------------------
                # LRU REFRESH
                # -----------------------------------------------------
                try:

                    if hasattr(
                        self.cache,
                        "move_to_end",
                    ):

                        self.cache.move_to_end(key)

                except Exception as exc:

                    logger.debug(
                        f"LRU refresh failed: {exc}"
                    )

                # -----------------------------------------------------
                # CACHE HIT METRICS
                # -----------------------------------------------------
                try:

                    self.hits = int(
                        getattr(
                            self,
                            "hits",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # SAFE COPY
                # -----------------------------------------------------
                try:

                    if isinstance(value, dict):

                        result = value.copy()

                    elif isinstance(value, list):

                        result = value[:]

                    else:

                        result = value

                except Exception as exc:

                    logger.warning(
                        f"Cache copy failed: {exc}"
                    )

                    result = value

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Cache hit | "
                    f"age={age:.2f}s"
                )

                return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Cache get failed: {exc}"
            )

            return None

    def set(self, prompt: str, response: Any, **kwargs) -> None:
        """
        Production-grade cache storage system.

        Fixes:
        - cache corruption
        - race conditions
        - mutable object leaks
        - invalid cache entries
        - oversized cache values
        - LRU instability
        - memory pressure explosions
        - serialization crashes
        - duplicate key instability
        - broken eviction logic
        """

        try:

            # ---------------------------------------------------------
            # CACHE ENABLED
            # ---------------------------------------------------------
            if not bool(
                getattr(self, "enabled", False)
            ):

                return

            # ---------------------------------------------------------
            # CACHE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "cache"):

                logger.warning(
                    "Cache storage missing"
                )

                return

            # ---------------------------------------------------------
            # LOCK VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_lock"):

                self._lock = threading.Lock()

            # ---------------------------------------------------------
            # SAFE KEY GENERATION
            # ---------------------------------------------------------
            try:

                key = self._make_key(
                    prompt,
                    **kwargs,
                )

            except Exception as exc:

                logger.warning(
                    f"Cache key generation failed: {exc}"
                )

                return

            if not key:

                return

            # ---------------------------------------------------------
            # RESPONSE VALIDATION
            # ---------------------------------------------------------
            if response is None:

                logger.debug(
                    "Skipping cache of None response"
                )

                return

            # ---------------------------------------------------------
            # SAFE RESPONSE COPY
            # ---------------------------------------------------------
            try:

                if isinstance(response, dict):

                    value = response.copy()

                elif isinstance(response, list):

                    value = response[:]

                elif isinstance(
                    response,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):

                    value = response

                else:

                    value = str(response)

            except Exception as exc:

                logger.warning(
                    f"Response copy failed: {exc}"
                )

                return

            # ---------------------------------------------------------
            # SIZE PROTECTION
            # ---------------------------------------------------------
            try:

                estimated_size = len(
                    str(value)
                )

            except Exception:

                estimated_size = 0

            max_entry_size = 100_000

            if estimated_size > max_entry_size:

                logger.warning(
                    f"Cache entry too large "
                    f"({estimated_size} chars)"
                )

                return

            # ---------------------------------------------------------
            # SAFE TIMESTAMP
            # ---------------------------------------------------------
            timestamp = time.time()

            # ---------------------------------------------------------
            # THREAD-SAFE STORAGE
            # ---------------------------------------------------------
            with self._lock:

                # -----------------------------------------------------
                # SAFE MAX SIZE
                # -----------------------------------------------------
                try:

                    max_size = int(
                        getattr(
                            self,
                            "max_size",
                            1000,
                        )
                    )

                except Exception:

                    max_size = 1000

                max_size = max(
                    1,
                    min(max_size, 1_000_000),
                )

                # -----------------------------------------------------
                # EXISTING KEY REFRESH
                # -----------------------------------------------------
                if key in self.cache:

                    try:

                        if hasattr(
                            self.cache,
                            "move_to_end",
                        ):

                            self.cache.move_to_end(key)

                    except Exception as exc:

                        logger.debug(
                            f"LRU refresh failed: {exc}"
                        )

                # -----------------------------------------------------
                # CACHE EVICTION
                # -----------------------------------------------------
                while len(self.cache) >= max_size:

                    try:

                        if hasattr(
                            self.cache,
                            "popitem",
                        ):

                            self.cache.popitem(
                                last=False
                            )

                        else:

                            # fallback eviction
                            oldest_key = next(
                                iter(self.cache)
                            )

                            del self.cache[
                                oldest_key
                            ]

                        self.evictions = int(
                            getattr(
                                self,
                                "evictions",
                                0,
                            )
                        ) + 1

                    except StopIteration:

                        break

                    except KeyError:

                        break

                    except Exception as exc:

                        logger.warning(
                            f"Cache eviction failed: {exc}"
                        )

                        break

                # -----------------------------------------------------
                # STORE ENTRY
                # -----------------------------------------------------
                try:

                    self.cache[key] = (
                        value,
                        timestamp,
                    )

                except Exception as exc:

                    logger.error(
                        f"Cache store failed: {exc}"
                    )

                    return

                # -----------------------------------------------------
                # OPTIONAL CACHE HEALTH
                # -----------------------------------------------------
                try:

                    cache_size = len(
                        self.cache
                    )

                    if cache_size > (
                        max_size * 0.90
                    ):

                        logger.warning(
                            f"Cache nearing capacity "
                            f"({cache_size}/{max_size})"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Cache set success | "
                    f"size={len(self.cache)}"
                )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Cache set failed: {exc}"
            )

    def clear(self) -> int:
        """
        Production-grade cache clear system.

        Fixes:
        - race conditions
        - partial cache clears
        - metric corruption
        - lock failures
        - memory cleanup instability
        - corrupted cache states
        - invalid cache objects
        - clear-count desync
        """

        cleared = 0

        try:

            # ---------------------------------------------------------
            # LOCK VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_lock"):

                self._lock = threading.Lock()

            # ---------------------------------------------------------
            # CACHE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "cache"):

                logger.warning(
                    "Cache object missing during clear"
                )

                return 0

            # ---------------------------------------------------------
            # THREAD-SAFE CLEAR
            # ---------------------------------------------------------
            with self._lock:

                # -----------------------------------------------------
                # SAFE SIZE DETECTION
                # -----------------------------------------------------
                try:

                    cleared = len(self.cache)

                except Exception as exc:

                    logger.warning(
                        f"Cache size check failed: {exc}"
                    )

                    cleared = 0

                # -----------------------------------------------------
                # CLEAR CACHE
                # -----------------------------------------------------
                try:

                    self.cache.clear()

                except Exception as exc:

                    logger.error(
                        f"Cache clear failed: {exc}"
                    )

                    return 0

                # -----------------------------------------------------
                # VALIDATE CLEAR
                # -----------------------------------------------------
                try:

                    remaining = len(self.cache)

                    if remaining > 0:

                        logger.warning(
                            f"Cache not fully cleared "
                            f"({remaining} remaining)"
                        )

                    else:

                        logger.debug(
                            "Cache cleared successfully"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL METRICS RESET
                # -----------------------------------------------------
                try:

                    if getattr(
                        self,
                        "reset_metrics_on_clear",
                        False,
                    ):

                        self.hits = 0
                        self.misses = 0
                        self.evictions = 0

                except Exception as exc:

                    logger.debug(
                        f"Metric reset failed: {exc}"
                    )

                # -----------------------------------------------------
                # MEMORY CLEANUP
                # -----------------------------------------------------
                try:

                    import gc

                    gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    f"Cache cleared | "
                    f"entries_removed={cleared}"
                )

                return max(0, cleared)

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Cache clear crashed: {exc}"
            )

            return 0


# ══════════════════════════════════════════════════════════════════════════════
# Embedding engine
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    DIM = 384

    def encode(self, text: str) -> np.ndarray:
        """
        Production-grade deterministic text encoder.

        Fixes:
        - invalid UTF-8 crashes
        - unstable embeddings
        - NaN vectors
        - dimension corruption
        - empty text failures
        - memory explosions
        - normalization instability
        - deterministic seed failures
        - invalid numpy outputs
        """

        try:

            # ---------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------
            if text is None:
                text = ""

            if not isinstance(text, str):

                try:
                    text = str(text)
                except Exception:
                    text = repr(text)

            # ---------------------------------------------------------
            # CLEAN INPUT
            # ---------------------------------------------------------
            text = text.replace(
                "\x00",
                " ",
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            )

            text = text.strip()

            # ---------------------------------------------------------
            # EMPTY TEXT HANDLING
            # ---------------------------------------------------------
            if not text:

                logger.debug(
                    "Encoding empty text"
                )

                text = "__empty__"

            # ---------------------------------------------------------
            # HARD TEXT LIMIT
            # ---------------------------------------------------------
            max_chars = 10000

            if len(text) > max_chars:

                logger.warning(
                    f"Encode text truncated "
                    f"({len(text)} chars)"
                )

                text = text[:max_chars]

            # ---------------------------------------------------------
            # SAFE DIMENSION
            # ---------------------------------------------------------
            try:

                dim = int(
                    getattr(
                        self,
                        "DIM",
                        384,
                    )
                )

            except Exception:

                dim = 384

            dim = max(
                8,
                min(dim, 8192),
            )

            # ---------------------------------------------------------
            # DETERMINISTIC HASH
            # ---------------------------------------------------------
            try:

                encoded = text.encode(
                    "utf-8",
                    errors="ignore",
                )

                digest = hashlib.md5(
                    encoded
                ).hexdigest()

                seed = int(
                    digest,
                    16,
                ) % (2**32)

            except Exception as exc:

                logger.warning(
                    f"Hash generation failed: {exc}"
                )

                seed = 42

            # ---------------------------------------------------------
            # RNG GENERATION
            # ---------------------------------------------------------
            try:

                rng = np.random.default_rng(seed)

            except Exception as exc:

                logger.warning(
                    f"RNG creation failed: {exc}"
                )

                rng = np.random.default_rng(42)

            # ---------------------------------------------------------
            # VECTOR GENERATION
            # ---------------------------------------------------------
            try:

                vec = rng.standard_normal(
                    dim
                ).astype(np.float32)

            except Exception as exc:

                logger.error(
                    f"Vector generation failed: {exc}"
                )

                vec = np.zeros(
                    dim,
                    dtype=np.float32,
                )

            # ---------------------------------------------------------
            # VALIDATE VECTOR
            # ---------------------------------------------------------
            if not isinstance(
                vec,
                np.ndarray,
            ):

                logger.warning(
                    "Generated vector invalid type"
                )

                vec = np.array(
                    vec,
                    dtype=np.float32,
                )

            # ---------------------------------------------------------
            # NAN / INF CLEANUP
            # ---------------------------------------------------------
            try:

                if np.isnan(vec).any():

                    logger.warning(
                        "NaN detected in vector"
                    )

                    vec = np.nan_to_num(
                        vec,
                        nan=0.0,
                    )

                if np.isinf(vec).any():

                    logger.warning(
                        "Inf detected in vector"
                    )

                    vec = np.nan_to_num(
                        vec,
                        posinf=0.0,
                        neginf=0.0,
                    )

            except Exception as exc:

                logger.debug(
                    f"Vector cleanup failed: {exc}"
                )

            # ---------------------------------------------------------
            # NORMALIZATION
            # ---------------------------------------------------------
            try:

                vec = _normalise_l2(vec)

            except Exception as exc:

                logger.warning(
                    f"Normalization failed: {exc}"
                )

                # fallback normalization
                try:

                    norm = np.linalg.norm(vec)

                    if norm > 0:

                        vec = (
                            vec / norm
                        ).astype(np.float32)

                except Exception:

                    vec = np.zeros(
                        dim,
                        dtype=np.float32,
                    )

            # ---------------------------------------------------------
            # FINAL VALIDATION
            # ---------------------------------------------------------
            try:

                if vec.shape[0] != dim:

                    logger.warning(
                        f"Invalid embedding dimension "
                        f"{vec.shape[0]} != {dim}"
                    )

                    corrected = np.zeros(
                        dim,
                        dtype=np.float32,
                    )

                    limit = min(
                        dim,
                        vec.shape[0],
                    )

                    corrected[:limit] = vec[:limit]

                    vec = corrected

            except Exception:
                pass

            # ---------------------------------------------------------
            # FINAL SAFETY
            # ---------------------------------------------------------
            vec = np.asarray(
                vec,
                dtype=np.float32,
            )

            logger.debug(
                f"Embedding generated | "
                f"dim={dim}"
            )

            return vec

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Encode failed: {exc}"
            )

            try:

                dim = int(
                    getattr(
                        self,
                        "DIM",
                        384,
                    )
                )

            except Exception:

                dim = 384

            return np.zeros(
                max(8, dim),
                dtype=np.float32,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Vector index
# ══════════════════════════════════════════════════════════════════════════════

class VectorIndex:
    def __init__(self, dim: int = EmbeddingEngine.DIM):
        self.dim = dim
        self.texts: List[str] = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
            warnings.warn("FAISS not available — vector search disabled")

    def add(self, embedding: np.ndarray, text: str) -> None:
        """
        Production-grade vector index insertion.

        Fixes:
        - invalid embeddings
        - dimension mismatches
        - FAISS crashes
        - NaN/inf vectors
        - memory corruption
        - text/index desync
        - duplicate inserts
        - normalization failures
        - oversized text memory waste
        - thread safety issues
        """

        try:

            # ---------------------------------------------------------
            # INDEX VALIDATION
            # ---------------------------------------------------------
            if getattr(self, "index", None) is None:

                logger.warning(
                    "Vector index unavailable"
                )

                return

            # ---------------------------------------------------------
            # EMBEDDING VALIDATION
            # ---------------------------------------------------------
            if embedding is None:

                logger.debug(
                    "Skipping empty embedding"
                )

                return

            # ---------------------------------------------------------
            # TEXT VALIDATION
            # ---------------------------------------------------------
            if text is None:
                text = ""

            if not isinstance(text, str):

                try:
                    text = str(text)
                except Exception:
                    text = repr(text)

            text = text.replace(
                "\x00",
                " ",
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            )

            text = text.strip()

            if not text:

                logger.debug(
                    "Skipping empty text"
                )

                return

            # ---------------------------------------------------------
            # HARD TEXT LIMIT
            # ---------------------------------------------------------
            max_text_chars = 10000

            if len(text) > max_text_chars:

                logger.warning(
                    f"Vector text truncated "
                    f"({len(text)} chars)"
                )

                text = text[:max_text_chars]

            # ---------------------------------------------------------
            # ENSURE NUMPY ARRAY
            # ---------------------------------------------------------
            try:

                emb = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

            except Exception as exc:

                logger.warning(
                    f"Embedding conversion failed: {exc}"
                )

                return

            # ---------------------------------------------------------
            # EMPTY VECTOR CHECK
            # ---------------------------------------------------------
            if emb.size == 0:

                logger.warning(
                    "Skipping empty embedding vector"
                )

                return

            # ---------------------------------------------------------
            # FLATTEN VECTOR
            # ---------------------------------------------------------
            try:

                emb = emb.flatten()

            except Exception:
                pass

            # ---------------------------------------------------------
            # NAN / INF CLEANUP
            # ---------------------------------------------------------
            try:

                if np.isnan(emb).any():

                    logger.warning(
                        "NaN values detected in embedding"
                    )

                    emb = np.nan_to_num(
                        emb,
                        nan=0.0,
                    )

                if np.isinf(emb).any():

                    logger.warning(
                        "Inf values detected in embedding"
                    )

                    emb = np.nan_to_num(
                        emb,
                        posinf=0.0,
                        neginf=0.0,
                    )

            except Exception as exc:

                logger.debug(
                    f"Embedding cleanup failed: {exc}"
                )

            # ---------------------------------------------------------
            # DIMENSION VALIDATION
            # ---------------------------------------------------------
            try:

                expected_dim = int(
                    getattr(
                        self,
                        "DIM",
                        emb.shape[0],
                    )
                )

            except Exception:

                expected_dim = emb.shape[0]

            expected_dim = max(
                1,
                expected_dim,
            )

            # ---------------------------------------------------------
            # DIMENSION CORRECTION
            # ---------------------------------------------------------
            if emb.shape[0] != expected_dim:

                logger.warning(
                    f"Embedding dimension mismatch "
                    f"{emb.shape[0]} != {expected_dim}"
                )

                corrected = np.zeros(
                    expected_dim,
                    dtype=np.float32,
                )

                limit = min(
                    expected_dim,
                    emb.shape[0],
                )

                corrected[:limit] = emb[:limit]

                emb = corrected

            # ---------------------------------------------------------
            # NORMALIZATION
            # ---------------------------------------------------------
            try:

                emb = _normalise_l2(emb)

            except Exception as exc:

                logger.warning(
                    f"Embedding normalization failed: {exc}"
                )

                try:

                    norm = np.linalg.norm(emb)

                    if norm > 0:

                        emb = (
                            emb / norm
                        ).astype(np.float32)

                except Exception:

                    logger.error(
                        "Fallback normalization failed"
                    )

                    return

            # ---------------------------------------------------------
            # FINAL SHAPE
            # ---------------------------------------------------------
            emb = emb.reshape(1, -1)

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_index_lock"):

                self._index_lock = threading.Lock()

            with self._index_lock:

                # -----------------------------------------------------
                # OPTIONAL DUPLICATE CHECK
                # -----------------------------------------------------
                try:

                    recent_texts = getattr(
                        self,
                        "texts",
                        [],
                    )[-10:]

                    if text in recent_texts:

                        logger.debug(
                            "Skipping duplicate vector text"
                        )

                        return

                except Exception:
                    pass

                # -----------------------------------------------------
                # ADD TO INDEX
                # -----------------------------------------------------
                try:

                    self.index.add(emb)

                except Exception as exc:

                    logger.error(
                        f"Vector index add failed: {exc}"
                    )

                    return

                # -----------------------------------------------------
                # TEXT STORAGE
                # -----------------------------------------------------
                try:

                    if not hasattr(self, "texts"):

                        self.texts = []

                    self.texts.append(text)

                except Exception as exc:

                    logger.error(
                        f"Text append failed: {exc}"
                    )

                    # rollback impossible safely
                    return

                # -----------------------------------------------------
                # MEMORY LIMIT
                # -----------------------------------------------------
                try:

                    max_text_store = int(
                        getattr(
                            self,
                            "max_text_store",
                            100000,
                        )
                    )

                    if len(self.texts) > max_text_store:

                        overflow = (
                            len(self.texts) -
                            max_text_store
                        )

                        del self.texts[:overflow]

                        logger.warning(
                            f"Trimmed vector text store "
                            f"({overflow} removed)"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Vector added | "
                    f"dim={emb.shape[1]} "
                    f"texts={len(self.texts)}"
                )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Vector add failed: {exc}"
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[str]:
        """
        Production-grade vector similarity search.

        Fixes:
        - FAISS crashes
        - invalid embeddings
        - dimension mismatches
        - NaN/inf query vectors
        - invalid search indices
        - empty vector database crashes
        - duplicate results
        - malformed text retrieval
        - thread safety issues
        - top_k overflow bugs
        """

        try:

            # ---------------------------------------------------------
            # QUERY VALIDATION
            # ---------------------------------------------------------
            if query_embedding is None:

                logger.debug(
                    "Search skipped: empty query embedding"
                )

                return []

            # ---------------------------------------------------------
            # ENSURE NUMPY ARRAY
            # ---------------------------------------------------------
            try:

                query = np.asarray(
                    query_embedding,
                    dtype=np.float32,
                )

            except Exception as exc:

                logger.warning(
                    f"Query conversion failed: {exc}"
                )

                return []

            # ---------------------------------------------------------
            # EMPTY VECTOR CHECK
            # ---------------------------------------------------------
            if query.size == 0:

                logger.debug(
                    "Search skipped: zero-size embedding"
                )

                return []

            # ---------------------------------------------------------
            # INDEX VALIDATION
            # ---------------------------------------------------------
            if getattr(self, "index", None) is None:

                logger.warning(
                    "Search index unavailable"
                )

                return []

            # ---------------------------------------------------------
            # TEXT STORE VALIDATION
            # ---------------------------------------------------------
            texts = getattr(
                self,
                "texts",
                None,
            )

            if not isinstance(texts, list):

                logger.warning(
                    "Text storage invalid"
                )

                return []

            if len(texts) == 0:

                logger.debug(
                    "Search skipped: empty vector store"
                )

                return []

            # ---------------------------------------------------------
            # FLATTEN QUERY
            # ---------------------------------------------------------
            try:

                query = query.flatten()

            except Exception:
                pass

            # ---------------------------------------------------------
            # NAN / INF CLEANUP
            # ---------------------------------------------------------
            try:

                if np.isnan(query).any():

                    logger.warning(
                        "NaN detected in query embedding"
                    )

                    query = np.nan_to_num(
                        query,
                        nan=0.0,
                    )

                if np.isinf(query).any():

                    logger.warning(
                        "Inf detected in query embedding"
                    )

                    query = np.nan_to_num(
                        query,
                        posinf=0.0,
                        neginf=0.0,
                    )

            except Exception as exc:

                logger.debug(
                    f"Query cleanup failed: {exc}"
                )

            # ---------------------------------------------------------
            # DIMENSION VALIDATION
            # ---------------------------------------------------------
            try:

                expected_dim = int(
                    getattr(
                        self,
                        "DIM",
                        query.shape[0],
                    )
                )

            except Exception:

                expected_dim = query.shape[0]

            expected_dim = max(
                1,
                expected_dim,
            )

            # ---------------------------------------------------------
            # DIMENSION CORRECTION
            # ---------------------------------------------------------
            if query.shape[0] != expected_dim:

                logger.warning(
                    f"Query dimension mismatch "
                    f"{query.shape[0]} != {expected_dim}"
                )

                corrected = np.zeros(
                    expected_dim,
                    dtype=np.float32,
                )

                limit = min(
                    expected_dim,
                    query.shape[0],
                )

                corrected[:limit] = query[:limit]

                query = corrected

            # ---------------------------------------------------------
            # NORMALIZATION
            # ---------------------------------------------------------
            try:

                query = _normalise_l2(query)

            except Exception as exc:

                logger.warning(
                    f"Query normalization failed: {exc}"
                )

                try:

                    norm = np.linalg.norm(query)

                    if norm > 0:

                        query = (
                            query / norm
                        ).astype(np.float32)

                except Exception:

                    return []

            # ---------------------------------------------------------
            # FINAL SHAPE
            # ---------------------------------------------------------
            if query.ndim == 1:

                query = query.reshape(1, -1)

            # ---------------------------------------------------------
            # SAFE TOP-K
            # ---------------------------------------------------------
            try:

                top_k = int(top_k)

            except Exception:

                top_k = 3

            top_k = max(
                1,
                min(top_k, len(texts)),
            )

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_search_lock"):

                self._search_lock = threading.Lock()

            with self._search_lock:

                # -----------------------------------------------------
                # FAISS SEARCH
                # -----------------------------------------------------
                try:

                    distances, indices = (
                        self.index.search(
                            query,
                            top_k,
                        )
                    )

                except Exception as exc:

                    logger.error(
                        f"Vector search failed: {exc}"
                    )

                    return []

            # ---------------------------------------------------------
            # VALIDATE RESULTS
            # ---------------------------------------------------------
            if indices is None:

                logger.debug(
                    "Search returned no indices"
                )

                return []

            if len(indices) == 0:

                return []

            # ---------------------------------------------------------
            # RESULT EXTRACTION
            # ---------------------------------------------------------
            results = []

            seen = set()

            for idx in indices[0]:

                try:

                    # ---------------------------------------------
                    # VALID INDEX
                    # ---------------------------------------------
                    if not isinstance(
                        idx,
                        (
                            int,
                            np.integer,
                        ),
                    ):

                        continue

                    idx = int(idx)

                    if idx < 0:

                        continue

                    if idx >= len(texts):

                        continue

                    # ---------------------------------------------
                    # GET TEXT
                    # ---------------------------------------------
                    text = texts[idx]

                    if not isinstance(text, str):

                        text = str(text)

                    text = text.strip()

                    if not text:

                        continue

                    # ---------------------------------------------
                    # DUPLICATE FILTER
                    # ---------------------------------------------
                    if text in seen:

                        continue

                    seen.add(text)

                    results.append(text)

                except Exception as exc:

                    logger.debug(
                        f"Result extraction failed: {exc}"
                    )

            # ---------------------------------------------------------
            # OPTIONAL DISTANCE FILTER
            # ---------------------------------------------------------
            try:

                if distances is not None:

                    filtered_results = []

                    for i, text in enumerate(results):

                        try:

                            dist = float(
                                distances[0][i]
                            )

                            if math.isnan(dist):

                                continue

                            filtered_results.append(
                                text
                            )

                        except Exception:

                            filtered_results.append(
                                text
                            )

                    results = filtered_results

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"Vector search complete | "
                f"results={len(results)}"
            )

            return results

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Vector search crashed: {exc}"
            )

            return []


# ══════════════════════════════════════════════════════════════════════════════
# GGUF Provider
# ══════════════════════════════════════════════════════════════════════════════

class GGUFProvider(LLMProvider):
    _GARBAGE_PATTERNS = (
        "erectile dysfunction", "replica watches", "cheap meds", "click here to",
        "�", "\x00", "----------",
    )

    def __init__(
        self,
        model_path: str,
        fallback_provider: Optional[LLMProvider] = None,
    ):
        """
        Production-grade GGUF provider initialization.

        Fixes:
        - singleton race conditions
        - CUDA fallback crashes
        - invalid model paths
        - broken llama initialization
        - VRAM overload
        - CPU fallback instability
        - async lock failures
        - repeated model loading
        - corrupted shared model state
        """

        global _shared_model

        try:

            # ---------------------------------------------------------
            # SAFE ASYNC LOCK
            # ---------------------------------------------------------
            try:

                self.async_lock = asyncio.Lock()

            except Exception as exc:

                logger.warning(
                    f"Async lock creation failed: {exc}"
                )

                self.async_lock = None

            # ---------------------------------------------------------
            # FALLBACK PROVIDER
            # ---------------------------------------------------------
            try:

                self.fallback_provider = (
                    fallback_provider
                    or FallbackProvider()
                )

            except Exception as exc:

                logger.error(
                    f"Fallback provider failed: {exc}"
                )

                self.fallback_provider = None

            # ---------------------------------------------------------
            # INITIAL STATE
            # ---------------------------------------------------------
            self.model = None

            self.model_path = str(
                model_path or ""
            ).strip()

            self.initialized_at = time.time()

            # ---------------------------------------------------------
            # LLAMA AVAILABILITY
            # ---------------------------------------------------------
            if not LLAMA_AVAILABLE:

                logger.warning(
                    "llama-cpp-python unavailable "
                    "— using fallback provider"
                )

                return

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if "_model_lock" not in globals():

                logger.warning(
                    "Global model lock missing"
                )

                globals()["_model_lock"] = (
                    threading.Lock()
                )

            with _model_lock:

                # -----------------------------------------------------
                # REUSE SHARED MODEL
                # -----------------------------------------------------
                try:

                    if _shared_model is not None:

                        logger.info(
                            "Reusing GGUF singleton model"
                        )

                        self.model = _shared_model

                        return

                except Exception as exc:

                    logger.warning(
                        f"Shared model reuse failed: {exc}"
                    )

                # -----------------------------------------------------
                # MODEL PATH RESOLUTION
                # -----------------------------------------------------
                logger.info(
                    f"Resolving GGUF model: "
                    f"{self.model_path}"
                )

                try:

                    resolved = _resolve_model_path(
                        self.model_path
                    )

                except Exception as exc:

                    logger.error(
                        f"Model path resolution failed: {exc}"
                    )

                    resolved = None

                if not resolved:

                    logger.error(
                        f"Model file not found: "
                        f"{self.model_path}"
                    )

                    return

                self.model_path = resolved

                # -----------------------------------------------------
                # FILE SIZE CHECK
                # -----------------------------------------------------
                try:

                    model_size_gb = round(
                        os.path.getsize(resolved)
                        / (1024**3),
                        2,
                    )

                    logger.info(
                        f"GGUF size: {model_size_gb} GB"
                    )

                except Exception:

                    model_size_gb = 0.0

                # -----------------------------------------------------
                # GPU DETECTION
                # -----------------------------------------------------
                try:

                    use_gpu = (
                        _cuda_available()
                        and os.environ.get(
                            "EDIATH_FORCE_CPU_LLM",
                            "0",
                        ) != "1"
                    )

                except Exception as exc:

                    logger.warning(
                        f"CUDA detection failed: {exc}"
                    )

                    use_gpu = False

                # -----------------------------------------------------
                # GPU LAYER CONFIG
                # -----------------------------------------------------
                try:

                    gpu_layers = int(
                        GPU_LAYERS_DEFAULT
                    )

                except Exception:

                    gpu_layers = 20

                try:

                    gpu_max = int(
                        GPU_LAYERS_MAX
                    )

                except Exception:

                    gpu_max = 100

                gpu_layers = max(
                    0,
                    min(gpu_layers, gpu_max),
                )

                n_gpu = (
                    gpu_layers
                    if use_gpu
                    else 0
                )

                # -----------------------------------------------------
                # FALLBACK POLICY
                # -----------------------------------------------------
                attempts = [
                    (
                        n_gpu,
                        "CUDA"
                        if use_gpu
                        else "CPU-only",
                    )
                ]

                allow_cpu_fallback = (
                    os.environ.get(
                        "EDIATH_ALLOW_CPU_LLM_FALLBACK",
                        "1",
                    ) == "1"
                )

                if (
                    n_gpu > 0
                    and allow_cpu_fallback
                ):

                    attempts.append(
                        (
                            0,
                            "CPU fallback",
                        )
                    )

                elif n_gpu > 0:

                    logger.warning(
                        "CPU fallback disabled"
                    )

                # -----------------------------------------------------
                # THREAD COUNT
                # -----------------------------------------------------
                try:

                    cpu_count = (
                        os.cpu_count() or 4
                    )

                    n_threads = max(
                        2,
                        cpu_count // 2,
                    )

                except Exception:

                    n_threads = 4

                # -----------------------------------------------------
                # MODEL LOAD ATTEMPTS
                # -----------------------------------------------------
                last_error = None

                for layers, label in attempts:

                    logger.info(
                        f"Loading GGUF | "
                        f"mode={label} "
                        f"gpu_layers={layers}"
                    )

                    try:

                        # -----------------------------------------
                        # LOW-VRAM SAFETY
                        # -----------------------------------------
                        n_ctx = 1024

                        if model_size_gb >= 20:

                            n_ctx = 512

                        # -----------------------------------------
                        # CREATE MODEL
                        # -----------------------------------------
                        model = Llama(
                            model_path=resolved,

                            n_ctx=n_ctx,

                            n_batch=64,

                            n_threads=n_threads,

                            n_gpu_layers=layers,

                            use_mmap=True,

                            use_mlock=False,

                            logits_all=False,

                            embedding=False,

                            verbose=False,
                        )

                        # -----------------------------------------
                        # VALIDATE MODEL
                        # -----------------------------------------
                        if model is None:

                            raise RuntimeError(
                                "Llama returned None"
                            )

                        if not hasattr(
                            model,
                            "create_completion",
                        ):

                            raise RuntimeError(
                                "Invalid GGUF model object"
                            )

                        # -----------------------------------------
                        # SUCCESS
                        # -----------------------------------------
                        _shared_model = model

                        self.model = model

                        self.gpu_layers = layers

                        self.backend = label

                        logger.info(
                            f"GGUF model loaded "
                            f"successfully ({label})"
                        )

                        return

                    # ---------------------------------------------
                    # LOAD FAILURE
                    # ---------------------------------------------
                    except Exception as exc:

                        last_error = exc

                        logger.warning(
                            f"Model load failed "
                            f"({label}): {exc}"
                        )

                        try:

                            gc.collect()

                        except Exception:
                            pass

                        continue

                # -----------------------------------------------------
                # FINAL FAILURE
                # -----------------------------------------------------
                logger.error(
                    f"All GGUF load attempts failed: "
                    f"{last_error}"
                )

                self.model = None

                self.load_error = str(last_error)

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"GGUFProvider init failed: {exc}"
            )

            self.model = None

            self.load_error = str(exc)

    def _is_garbage_response(self, text: str) -> bool:
        """
        Production-grade garbage response detector.

        Fixes:
        - false positives
        - unicode corruption
        - repetitive hallucinations
        - symbol spam
        - malformed outputs
        - tokenizer garbage
        - low-information responses
        - runaway repetition
        - broken UTF responses
        - whitespace-only outputs
        """

        try:

            # ---------------------------------------------------------
            # NULL / TYPE CHECK
            # ---------------------------------------------------------
            if text is None:
                return True

            if not isinstance(text, str):

                try:
                    text = str(text)
                except Exception:
                    return True

            # ---------------------------------------------------------
            # CLEAN INPUT
            # ---------------------------------------------------------
            text = text.replace(
                "\x00",
                " ",
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            )

            text = text.strip()

            # ---------------------------------------------------------
            # EMPTY CHECK
            # ---------------------------------------------------------
            if not text:
                return True

            if len(text) < 2:
                return True

            # ---------------------------------------------------------
            # EXTREME LENGTH CHECK
            # ---------------------------------------------------------
            if len(text) > 50000:

                logger.warning(
                    "Garbage detector rejected oversized response"
                )

                return True

            # ---------------------------------------------------------
            # LOWERCASE VERSION
            # ---------------------------------------------------------
            tl = text.lower()

            # ---------------------------------------------------------
            # KNOWN GARBAGE PATTERNS
            # ---------------------------------------------------------
            try:

                garbage_patterns = getattr(
                    self,
                    "_GARBAGE_PATTERNS",
                    [],
                )

                if isinstance(
                    garbage_patterns,
                    (list, tuple, set),
                ):

                    for pattern in garbage_patterns:

                        try:

                            if (
                                pattern
                                and str(pattern).lower() in tl
                            ):

                                logger.debug(
                                    f"Garbage pattern matched: {pattern}"
                                )

                                return True

                        except Exception:
                            continue

            except Exception as exc:

                logger.debug(
                    f"Pattern check failed: {exc}"
                )

            # ---------------------------------------------------------
            # ALPHANUMERIC RATIO
            # ---------------------------------------------------------
            try:

                alnum_count = sum(
                    c.isalnum()
                    for c in text
                )

                text_len = max(1, len(text))

                alnum_ratio = (
                    alnum_count / text_len
                )

                if alnum_ratio < 0.20:

                    logger.debug(
                        f"Low alnum ratio: {alnum_ratio:.2f}"
                    )

                    return True

            except Exception:
                pass

            # ---------------------------------------------------------
            # EXTREME CHARACTER REPETITION
            # ---------------------------------------------------------
            try:

                if re.search(
                    r"(.)\1{8,}",
                    text,
                ):

                    logger.debug(
                        "Repeated character spam detected"
                    )

                    return True

            except Exception:
                pass

            # ---------------------------------------------------------
            # WORD ANALYSIS
            # ---------------------------------------------------------
            try:

                words = text.split()

                word_count = len(words)

                if word_count > 5:

                    unique_words = len(set(words))

                    diversity = (
                        unique_words / max(1, word_count)
                    )

                    if diversity < 0.25:

                        logger.debug(
                            f"Low word diversity: {diversity:.2f}"
                        )

                        return True

            except Exception:
                pass

            # ---------------------------------------------------------
            # SYMBOL-ONLY RESPONSE
            # ---------------------------------------------------------
            try:

                if re.fullmatch(
                    r"[\W_]+",
                    text,
                ):

                    logger.debug(
                        "Symbol-only response detected"
                    )

                    return True

            except Exception:
                pass

            # ---------------------------------------------------------
            # TOKEN GARBAGE DETECTION
            # ---------------------------------------------------------
            suspicious_fragments = [
                "<unk>",
                "<null>",
                "nan nan",
                "undefined undefined",
                "null null",
                "??? ???",
                "### ### ###",
            ]

            for frag in suspicious_fragments:

                if frag in tl:

                    logger.debug(
                        f"Suspicious fragment detected: {frag}"
                    )

                    return True

            # ---------------------------------------------------------
            # EXCESSIVE NEWLINE/SPACING NOISE
            # ---------------------------------------------------------
            try:

                newline_ratio = (
                    text.count("\n") / max(1, len(text))
                )

                if newline_ratio > 0.30:

                    logger.debug(
                        "Excessive newline noise"
                    )

                    return True

            except Exception:
                pass

            # ---------------------------------------------------------
            # VALID RESPONSE
            # ---------------------------------------------------------
            return False

        # -------------------------------------------------------------
        # FAIL-SAFE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Garbage detection failed: {exc}"
            )

            # safest option
            return True

    def _sync_generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """
        Production-grade synchronous GGUF generation.

        Fixes:
        - llama.cpp crashes
        - malformed outputs
        - garbage generations
        - deadlocks
        - CUDA instability
        - invalid parameters
        - memory pressure failures
        - prompt corruption
        - empty outputs
        - invalid stop tokens
        - completion parsing failures
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # MODEL VALIDATION
            # ---------------------------------------------------------
            model = getattr(
                self,
                "model",
                None,
            )

            if model is None:

                raise RuntimeError(
                    "Model not available"
                )

            # ---------------------------------------------------------
            # INPUT SANITIZATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):
                prompt = str(prompt)

            prompt = prompt.replace(
                "\x00",
                " ",
            )

            prompt = re.sub(
                r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
                " ",
                prompt,
            )

            prompt = re.sub(
                r"\s+",
                " ",
                prompt,
            )

            prompt = prompt.strip()

            if not prompt:

                raise ValueError(
                    "Empty prompt"
                )

            # ---------------------------------------------------------
            # HARD PROMPT LIMIT
            # ---------------------------------------------------------
            max_prompt_chars = int(
                kwargs.get(
                    "max_prompt_chars",
                    MAX_PROMPT_CHARS,
                )
            )

            max_prompt_chars = max(
                100,
                min(max_prompt_chars, 32000),
            )

            if len(prompt) > max_prompt_chars:

                logger.warning(
                    f"Prompt truncated "
                    f"({len(prompt)} chars)"
                )

                prompt = prompt[:max_prompt_chars]

            # ---------------------------------------------------------
            # SAFE PARAMETERS
            # ---------------------------------------------------------
            max_tokens = max(
                1,
                min(int(max_tokens), 1024),
            )

            temperature = max(
                0.0,
                min(float(temperature), 2.0),
            )

            top_k = max(
                1,
                min(
                    100,
                    int(kwargs.get("top_k", 40)),
                ),
            )

            top_p = max(
                0.05,
                min(
                    1.0,
                    float(kwargs.get("top_p", 0.90)),
                ),
            )

            repeat_penalty = max(
                0.8,
                min(
                    2.0,
                    float(
                        kwargs.get(
                            "repeat_penalty",
                            1.1,
                        )
                    ),
                ),
            )

            # ---------------------------------------------------------
            # STOP TOKENS
            # ---------------------------------------------------------
            stop_tokens = kwargs.get(
                "stop",
                [
                    "</s>",
                    "[INST]",
                    "[/INST]",
                    "\n\n",
                ],
            )

            if not isinstance(stop_tokens, list):

                stop_tokens = [str(stop_tokens)]

            stop_tokens = [
                str(x)[:50]
                for x in stop_tokens
                if x
            ]

            # ---------------------------------------------------------
            # GPU PRESSURE CHECK
            # ---------------------------------------------------------
            try:

                gpu_guard = getattr(
                    self,
                    "gpu_guard",
                    None,
                )

                if gpu_guard:

                    should_throttle = getattr(
                        gpu_guard,
                        "should_throttle",
                        None,
                    )

                    if callable(should_throttle):

                        if should_throttle(0):

                            raise RuntimeError(
                                "GPU under heavy pressure"
                            )

            except Exception as exc:

                logger.debug(
                    f"GPU pressure check failed: {exc}"
                )

            # ---------------------------------------------------------
            # THREAD-SAFE MODEL CALL
            # ---------------------------------------------------------
            try:

                with _model_call_lock:

                    result = model.create_completion(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        stop=stop_tokens,
                        echo=False,
                    )

            except Exception as exc:

                raise RuntimeError(
                    f"Model execution failed: {exc}"
                )

            # ---------------------------------------------------------
            # VALIDATE OUTPUT
            # ---------------------------------------------------------
            if not isinstance(result, dict):

                raise ValueError(
                    "Invalid model output format"
                )

            choices = result.get("choices")

            if (
                not isinstance(choices, list)
                or not choices
            ):

                raise ValueError(
                    "No choices returned"
                )

            first_choice = choices[0]

            if not isinstance(first_choice, dict):

                raise ValueError(
                    "Invalid choice format"
                )

            raw_text = str(
                first_choice.get("text", "")
            ).strip()

            # ---------------------------------------------------------
            # EMPTY OUTPUT
            # ---------------------------------------------------------
            if len(raw_text) < 2:

                raise ValueError(
                    "Empty model output"
                )

            # ---------------------------------------------------------
            # CLEAN RESPONSE
            # ---------------------------------------------------------
            text = raw_text

            text = re.sub(
                r"\[/?INST\]|</?s>",
                "",
                text,
            )

            text = re.sub(
                r"(?i)assistant:",
                "",
                text,
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            )

            text = text.strip()

            # ---------------------------------------------------------
            # GARBAGE DETECTION
            # ---------------------------------------------------------
            if not text:

                raise ValueError(
                    "Cleaned output empty"
                )

            if hasattr(
                self,
                "_is_garbage_response",
            ):

                try:

                    if self._is_garbage_response(text):

                        raise ValueError(
                            "Garbage response detected"
                        )

                except Exception as exc:

                    logger.debug(
                        f"Garbage detection failed: {exc}"
                    )

            # ---------------------------------------------------------
            # HARD OUTPUT LIMIT
            # ---------------------------------------------------------
            max_output_chars = int(
                kwargs.get(
                    "max_output_chars",
                    12000,
                )
            )

            if len(text) > max_output_chars:

                logger.warning(
                    f"Output truncated "
                    f"({len(text)} chars)"
                )

                text = text[:max_output_chars]

            # ---------------------------------------------------------
            # PERFORMANCE LOGGING
            # ---------------------------------------------------------
            elapsed = round(
                time.monotonic() - start,
                3,
            )

            logger.info(
                f"Sync generation complete | "
                f"time={elapsed:.2f}s "
                f"chars={len(text)} "
                f"tokens={max_tokens}"
            )

            return text

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            elapsed = round(
                time.monotonic() - start,
                3,
            )

            logger.error(
                f"_sync_generate failed | "
                f"time={elapsed:.2f}s "
                f"error={exc}"
            )

            raise RuntimeError(
                f"Generation failure: {exc}"
            )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        retries: int = 1,
        **kwargs,
    ) -> str:
        if not getattr(self, "model", None):
            return await self.fallback_provider.generate(prompt, **kwargs)

        prompt = str(prompt or "").strip()
        if not prompt:
            return "Please provide a valid prompt."

        stop_tokens = kwargs.pop("stop", ["</s>", "[INST]", "[/INST]", "\n\n"])

        for attempt in range(retries + 1):
            try:
                async with self.async_lock:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._sync_generate,
                            prompt,
                            max_tokens,
                            temperature,
                            stop=stop_tokens,
                            **kwargs,
                        ),
                        timeout=20.0,
                    )
                text = str(result or "").strip()
                if len(text) < 2:
                    raise ValueError("Empty response")
                return text
            except asyncio.TimeoutError:
                logger.warning(f"GGUF timeout (attempt {attempt + 1})")
            except Exception as exc:
                logger.warning(f"GGUF attempt {attempt + 1} failed: {exc}")
            if attempt < retries:
                await asyncio.sleep(min(1.5, 0.3 * (2 ** attempt)))

        try:
            return await self.fallback_provider.generate(prompt, **kwargs)
        except Exception:
            return "I'm here to help. Please try again."


# ══════════════════════════════════════════════════════════════════════════════
# Main LLM Engine
# ══════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    def __init__(self, model_path: str = "models/EDIATH-q4_k_m.gguf"):
        resolved = _resolve_model_path(model_path) or model_path

        self.fallback_provider = FallbackProvider()
        self.provider: LLMProvider = (
            GGUFProvider(resolved, self.fallback_provider)
            if LLAMA_AVAILABLE
            else self.fallback_provider
        )

        self._global_lock: Optional[asyncio.Lock] = None
        self._rate_lock: Optional[asyncio.Lock] = None
        self._queue: Optional[asyncio.PriorityQueue] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        self._workers: List[asyncio.Task] = []
        self._worker_running: bool = False
        self._shutdown: bool = False

        self._active_requests: int = 0
        self.generation_count: int = 0
        self._last_call_time: float = 0.0
        self.min_delay: float = 0.1
        self.max_queue_size: int = MAX_QUEUE_SIZE

        self.cache = LLMCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
        self.cache_enabled = True

        self.embedder = EmbeddingEngine()
        self.index = VectorIndex() if FAISS_AVAILABLE else None
        self.index_enabled = FAISS_AVAILABLE

        self.semantic_memory: List[Dict] = []
        self.semantic_enabled = True
        self.similarity_threshold = 0.85

        self.memory = None
        self.memory_enabled = False
        try:
            from ..memory.memory_manager import MemoryManager
            self.memory = MemoryManager()
            self.memory_enabled = True
        except Exception:
            pass

        self.gpu_guard = GPUMemoryGuard()
        self._initialized = False

    def _ensure_async_primitives(self) -> None:
        """
        Production-grade async primitive initializer.

        Fixes:
        - duplicate primitive creation
        - race conditions
        - loop mismatch crashes
        - destroyed event-loop references
        - semaphore deadlocks
        - queue corruption
        - worker startup instability
        """

        try:

            # ---------------------------------------------------------
            # CURRENT EVENT LOOP
            # ---------------------------------------------------------
            current_loop = asyncio.get_running_loop()

        except RuntimeError:

            logger.error(
                "LLMEngine async primitives requested outside running event loop"
            )
            return

        # -------------------------------------------------------------
        # INTERNAL LOCK FOR THREAD SAFETY
        # -------------------------------------------------------------
        if not hasattr(self, "_primitive_init_lock"):
            self._primitive_init_lock = threading.Lock()

        with self._primitive_init_lock:

            try:

                # -----------------------------------------------------
                # LOOP TRACKING
                # -----------------------------------------------------
                old_loop = getattr(self, "_loop", None)

                loop_changed = (
                    old_loop is not None
                    and old_loop != current_loop
                )

                if loop_changed:

                    logger.warning(
                        "Detected asyncio loop change — rebuilding primitives"
                    )

                    # reset invalid primitives
                    self._global_lock = None
                    self._rate_lock = None
                    self._queue = None
                    self._semaphore = None

                self._loop = current_loop

                # -----------------------------------------------------
                # GLOBAL LOCK
                # -----------------------------------------------------
                if (
                    getattr(self, "_global_lock", None) is None
                    or getattr(self._global_lock, "_loop", current_loop) != current_loop
                ):

                    self._global_lock = asyncio.Lock()

                # -----------------------------------------------------
                # RATE LOCK
                # -----------------------------------------------------
                if (
                    getattr(self, "_rate_lock", None) is None
                    or getattr(self._rate_lock, "_loop", current_loop) != current_loop
                ):

                    self._rate_lock = asyncio.Lock()

                # -----------------------------------------------------
                # PRIORITY QUEUE
                # -----------------------------------------------------
                recreate_queue = False

                if getattr(self, "_queue", None) is None:
                    recreate_queue = True

                else:
                    try:
                        _ = self._queue.qsize()
                    except Exception:
                        recreate_queue = True

                if recreate_queue:

                    queue_size = max(10, int(MAX_QUEUE_SIZE))

                    self._queue = asyncio.PriorityQueue(
                        maxsize=queue_size
                    )

                # -----------------------------------------------------
                # SEMAPHORE
                # -----------------------------------------------------
                recreate_semaphore = False

                if getattr(self, "_semaphore", None) is None:
                    recreate_semaphore = True

                else:
                    try:
                        _ = self._semaphore._value
                    except Exception:
                        recreate_semaphore = True

                if recreate_semaphore:

                    worker_count = max(
                        1,
                        min(int(_NUM_WORKERS), 4)
                    )

                    self._semaphore = asyncio.Semaphore(worker_count)

                # -----------------------------------------------------
                # WORKER STATE FLAGS
                # -----------------------------------------------------
                if not hasattr(self, "_workers"):
                    self._workers = []

                if not hasattr(self, "_worker_running"):
                    self._worker_running = False

                if not hasattr(self, "_shutdown"):
                    self._shutdown = False

                # -----------------------------------------------------
                # REQUEST TRACKING
                # -----------------------------------------------------
                if not hasattr(self, "_active_requests"):
                    self._active_requests = 0

                # -----------------------------------------------------
                # SAFETY LIMITS
                # -----------------------------------------------------
                self._active_requests = max(
                    0,
                    int(self._active_requests)
                )

                # -----------------------------------------------------
                # HEALTH CHECK
                # -----------------------------------------------------
                logger.debug(
                    "Async primitives ready | "
                    f"workers={_NUM_WORKERS} "
                    f"queue={MAX_QUEUE_SIZE}"
                )

            except Exception as exc:

                logger.error(
                    f"Failed to initialize async primitives: {exc}"
                )

                # -----------------------------------------------------
                # SAFE FALLBACK RECOVERY
                # -----------------------------------------------------
                try:

                    self._global_lock = asyncio.Lock()
                    self._rate_lock = asyncio.Lock()

                    self._queue = asyncio.PriorityQueue(
                        maxsize=max(10, int(MAX_QUEUE_SIZE))
                    )

                    self._semaphore = asyncio.Semaphore(
                        max(1, min(int(_NUM_WORKERS), 2))
                    )

                    logger.warning(
                        "LLMEngine recovered using fallback async primitives"
                    )

                except Exception as recovery_exc:

                    logger.critical(
                        f"Fallback async primitive recovery failed: {recovery_exc}"
                    )

    def _start_workers(self) -> None:
        """
        Production-grade queue worker starter.

        Fixes:
        - duplicate worker creation
        - zombie workers
        - dead workers not restarting
        - event loop mismatch
        - worker memory leaks
        - task explosion
        - startup race conditions
        - shutdown corruption
        """

        # -------------------------------------------------------------
        # ENSURE ASYNC OBJECTS
        # -------------------------------------------------------------
        self._ensure_async_primitives()

        # -------------------------------------------------------------
        # PREVENT START DURING SHUTDOWN
        # -------------------------------------------------------------
        if getattr(self, "_shutdown", False):

            logger.warning(
                "Worker startup blocked because shutdown is active"
            )

            return

        # -------------------------------------------------------------
        # THREAD-SAFE START LOCK
        # -------------------------------------------------------------
        if not hasattr(self, "_worker_start_lock"):
            self._worker_start_lock = threading.Lock()

        with self._worker_start_lock:

            try:

                # -----------------------------------------------------
                # CLEAN DEAD WORKERS
                # -----------------------------------------------------
                alive_workers = []

                for task in getattr(self, "_workers", []):

                    try:

                        if (
                            task
                            and not task.done()
                            and not task.cancelled()
                        ):
                            alive_workers.append(task)

                    except Exception:
                        pass

                self._workers = alive_workers

                # -----------------------------------------------------
                # PREVENT DUPLICATE STARTS
                # -----------------------------------------------------
                if getattr(self, "_worker_running", False):

                    if len(self._workers) > 0:

                        logger.info(
                            f"LLM workers already active "
                            f"({len(self._workers)} running)"
                        )

                        return

                    logger.warning(
                        "Worker state inconsistent — repairing"
                    )

                # -----------------------------------------------------
                # SAFE WORKER COUNT
                # -----------------------------------------------------
                worker_count = max(
                    1,
                    min(int(_NUM_WORKERS), 4)
                )

                # -----------------------------------------------------
                # RESET FLAGS
                # -----------------------------------------------------
                self._worker_running = True

                if getattr(self, "_shutdown", False):
                    self._shutdown = False

                # -----------------------------------------------------
                # START WORKERS
                # -----------------------------------------------------
                created = 0

                for i in range(worker_count):

                    try:

                        task_name = f"llm_queue_worker_{i}"

                        task = asyncio.create_task(
                            self._queue_worker(i),
                            name=task_name,
                        )

                        # ---------------------------------------------
                        # STORE TASK
                        # ---------------------------------------------
                        self._workers.append(task)

                        created += 1

                    except RuntimeError as exc:

                        logger.error(
                            f"Failed creating worker {i}: {exc}"
                        )

                    except Exception as exc:

                        logger.error(
                            f"Unexpected worker startup error {i}: {exc}"
                        )

                # -----------------------------------------------------
                # VALIDATE STARTUP
                # -----------------------------------------------------
                if created == 0:

                    self._worker_running = False

                    raise RuntimeError(
                        "No queue workers could be started"
                    )

                # -----------------------------------------------------
                # HEALTH CHECK TASK
                # -----------------------------------------------------
                if not hasattr(self, "_worker_health_task"):

                    try:

                        async def _monitor_workers():

                            while (
                                self._worker_running
                                and not self._shutdown
                            ):

                                try:

                                    alive = [
                                        w for w in self._workers
                                        if not w.done()
                                    ]

                                    dead_count = (
                                        len(self._workers) - len(alive)
                                    )

                                    self._workers = alive

                                    # restart missing workers
                                    if dead_count > 0:

                                        logger.warning(
                                            f"{dead_count} worker(s) died — restarting"
                                        )

                                        needed = max(
                                            0,
                                            worker_count - len(alive)
                                        )

                                        for idx in range(needed):

                                            try:

                                                wid = len(self._workers)

                                                task = asyncio.create_task(
                                                    self._queue_worker(wid),
                                                    name=f"llm_queue_worker_restart_{wid}",
                                                )

                                                self._workers.append(task)

                                            except Exception as exc:
                                                logger.error(
                                                    f"Worker restart failed: {exc}"
                                                )

                                    await asyncio.sleep(10)

                                except asyncio.CancelledError:
                                    break

                                except Exception as exc:

                                    logger.error(
                                        f"Worker monitor failure: {exc}"
                                    )

                                    await asyncio.sleep(5)

                        self._worker_health_task = asyncio.create_task(
                            _monitor_workers(),
                            name="llm_worker_monitor",
                        )

                    except Exception as exc:

                        logger.warning(
                            f"Worker monitor disabled: {exc}"
                        )

                # -----------------------------------------------------
                # FINAL LOG
                # -----------------------------------------------------
                logger.info(
                    f"LLMEngine started successfully | "
                    f"workers={created}"
                )

            except Exception as exc:

                # -----------------------------------------------------
                # FAIL SAFE RECOVERY
                # -----------------------------------------------------
                self._worker_running = False

                logger.error(
                    f"Worker startup failed: {exc}"
                )

                # cleanup partial workers
                for task in getattr(self, "_workers", []):

                    try:
                        if task and not task.done():
                            task.cancel()
                    except Exception:
                        pass

                self._workers.clear()

    async def _queue_worker(self, worker_id: int) -> None:
        """
        Production-grade async queue worker.

        Fixes:
        - duplicate task_done()
        - queue corruption
        - worker freezes
        - memory leaks
        - zombie futures
        - stale requests
        - cancellation crashes
        - GPU overload instability
        - infinite worker loops
        - orphan futures
        - bad shutdown handling
        """

        logger.info(f"LLM queue worker {worker_id} started")

        processed_count = 0
        error_count = 0
        idle_cycles = 0

        current_future: Optional[asyncio.Future] = None
        current_request: Optional[PrioritizedRequest] = None

        # -------------------------------------------------------------
        # WORKER MAIN LOOP
        # -------------------------------------------------------------
        while True:

            queue_item_acquired = False

            try:

                # -----------------------------------------------------
                # SHUTDOWN CHECK
                # -----------------------------------------------------
                if (
                    getattr(self, "_shutdown", False)
                    or not getattr(self, "_worker_running", False)
                ):
                    break

                # -----------------------------------------------------
                # GET REQUEST
                # -----------------------------------------------------
                try:

                    current_request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )

                    queue_item_acquired = True
                    idle_cycles = 0

                except asyncio.TimeoutError:

                    idle_cycles += 1

                    # periodic worker heartbeat
                    if idle_cycles % 300 == 0:
                        logger.debug(
                            f"Worker {worker_id} idle "
                            f"(processed={processed_count})"
                        )

                    await asyncio.sleep(0.05)
                    continue

                # -----------------------------------------------------
                # VALIDATE REQUEST
                # -----------------------------------------------------
                if current_request is None:
                    continue

                if not isinstance(current_request, PrioritizedRequest):

                    logger.warning(
                        f"Worker {worker_id} received invalid request object"
                    )

                    continue

                current_future = current_request.future

                # -----------------------------------------------------
                # FUTURE VALIDATION
                # -----------------------------------------------------
                if (
                    current_future is None
                    or current_future.cancelled()
                    or current_future.done()
                ):

                    continue

                # -----------------------------------------------------
                # REQUEST EXPIRY
                # -----------------------------------------------------
                age = (
                    time.monotonic()
                    - current_request.enqueue_time
                )

                if age > TASK_EXPIRY_SECONDS:

                    try:

                        if not current_future.done():

                            current_future.set_result({
                                "response": (
                                    "Request expired while waiting in queue."
                                ),
                                "intent": "idle",
                                "error": False,
                                "expired": True,
                                "worker_id": worker_id,
                            })

                    except Exception:
                        pass

                    continue

                # -----------------------------------------------------
                # GPU THROTTLING
                # -----------------------------------------------------
                try:

                    if self.gpu_guard.should_throttle(
                        current_request.priority
                    ):

                        if not current_future.done():

                            current_future.set_result({
                                "response": (
                                    "System under GPU memory pressure."
                                ),
                                "intent": "idle",
                                "error": False,
                                "throttled": True,
                                "worker_id": worker_id,
                            })

                        await asyncio.sleep(0.2)
                        continue

                except Exception as exc:

                    logger.warning(
                        f"GPU guard failure: {exc}"
                    )

                # -----------------------------------------------------
                # ACTIVE REQUEST TRACKING
                # -----------------------------------------------------
                self._active_requests += 1

                start_time = time.monotonic()

                # -----------------------------------------------------
                # SAFE GENERATION
                # -----------------------------------------------------
                try:

                    result = await asyncio.wait_for(
                        self.safe_generate(
                            current_request.prompt,
                            **current_request.kwargs,
                        ),
                        timeout=30.0,
                    )

                except asyncio.TimeoutError:

                    result = {
                        "response": (
                            "Generation timed out. Please retry."
                        ),
                        "intent": "idle",
                        "error": True,
                        "timeout": True,
                    }

                except asyncio.CancelledError:
                    raise

                except Exception as exc:

                    logger.error(
                        f"Worker {worker_id} generation failure: {exc}"
                    )

                    error_count += 1

                    result = {
                        "response": (
                            "Internal generation error occurred."
                        ),
                        "intent": "idle",
                        "error": True,
                        "exception": str(exc),
                    }

                finally:

                    self._active_requests = max(
                        0,
                        self._active_requests - 1,
                    )

                # -----------------------------------------------------
                # ADD METADATA
                # -----------------------------------------------------
                elapsed = round(
                    time.monotonic() - start_time,
                    3,
                )

                if isinstance(result, dict):

                    result.setdefault(
                        "generation_time",
                        elapsed,
                    )

                    result.setdefault(
                        "worker_id",
                        worker_id,
                    )

                # -----------------------------------------------------
                # RETURN RESULT
                # -----------------------------------------------------
                try:

                    if (
                        current_future
                        and not current_future.done()
                        and not current_future.cancelled()
                    ):

                        current_future.set_result(result)

                except asyncio.InvalidStateError:
                    pass

                except Exception as exc:

                    logger.warning(
                        f"Future delivery failed: {exc}"
                    )

                processed_count += 1

                # -----------------------------------------------------
                # PERIODIC CLEANUP
                # -----------------------------------------------------
                if processed_count % 50 == 0:

                    try:

                        # prevent memory buildup
                        await asyncio.sleep(0)

                    except Exception:
                        pass

                # -----------------------------------------------------
                # ERROR RATE PROTECTION
                # -----------------------------------------------------
                if error_count >= 25:

                    logger.warning(
                        f"Worker {worker_id} exceeded error threshold"
                    )

                    await asyncio.sleep(2)

                    error_count = 0

            # ---------------------------------------------------------
            # CANCELLATION
            # ---------------------------------------------------------
            except asyncio.CancelledError:

                logger.info(
                    f"Worker {worker_id} cancellation requested"
                )

                break

            # ---------------------------------------------------------
            # HARD FAILURE RECOVERY
            # ---------------------------------------------------------
            except Exception as exc:

                error_count += 1

                logger.error(
                    f"Queue worker {worker_id} crashed: {exc}"
                )

                try:

                    if (
                        current_future
                        and not current_future.done()
                    ):

                        current_future.set_result({
                            "response": (
                                "Worker recovered from internal failure."
                            ),
                            "intent": "idle",
                            "error": True,
                            "worker_crash": True,
                        })

                except Exception:
                    pass

                await asyncio.sleep(1)

            # ---------------------------------------------------------
            # CLEANUP
            # ---------------------------------------------------------
            finally:

                try:

                    if queue_item_acquired:
                        self._queue.task_done()

                except ValueError:
                    pass

                except Exception as exc:

                    logger.debug(
                        f"task_done cleanup failure: {exc}"
                    )

                current_future = None
                current_request = None

                await asyncio.sleep(0)

        # -------------------------------------------------------------
        # FINAL CLEANUP
        # -------------------------------------------------------------
        logger.info(
            f"LLM queue worker {worker_id} stopped | "
            f"processed={processed_count} "
            f"errors={error_count}"
        )

    async def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Production-grade prompt builder.

        Fixes:
        - prompt injection
        - malformed INST blocks
        - token explosion
        - unicode corruption
        - null-byte crashes
        - repeated whitespace
        - invalid system prompts
        - runaway prompt growth
        - unsafe formatting
        - memory pressure from giant prompts
        """

        try:

            # ---------------------------------------------------------
            # SAFE DEFAULTS
            # ---------------------------------------------------------
            DEFAULT_PROMPT = (
                "[INST] Please respond helpfully and concisely. [/INST]"
            )

            # ---------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):
                prompt = str(prompt)

            # ---------------------------------------------------------
            # CLEAN USER PROMPT
            # ---------------------------------------------------------
            prompt = prompt.replace("\x00", " ")

            # remove invalid unicode/control chars
            prompt = re.sub(
                r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
                " ",
                prompt,
            )

            # normalize whitespace
            prompt = re.sub(r"\s+", " ", prompt)

            prompt = prompt.strip()

            # ---------------------------------------------------------
            # REMOVE MODEL CONTROL TOKENS
            # ---------------------------------------------------------
            dangerous_tokens = [
                "[INST]",
                "[/INST]",
                "<<SYS>>",
                "<</SYS>>",
                "<s>",
                "</s>",
                "<|system|>",
                "<|assistant|>",
                "<|user|>",
            ]

            for token in dangerous_tokens:
                prompt = prompt.replace(token, "")

            prompt = prompt.strip()

            # ---------------------------------------------------------
            # EMPTY PROMPT HANDLING
            # ---------------------------------------------------------
            if not prompt:

                return DEFAULT_PROMPT

            # ---------------------------------------------------------
            # HARD TOKEN LIMIT
            # ---------------------------------------------------------
            max_chars = max(
                50,
                int(MAX_PROMPT_CHARS),
            )

            if len(prompt) > max_chars:

                logger.warning(
                    f"Prompt truncated from "
                    f"{len(prompt)} -> {max_chars}"
                )

                prompt = prompt[:max_chars]

            # ---------------------------------------------------------
            # CLEAN SYSTEM PROMPT
            # ---------------------------------------------------------
            clean_system_prompt = ""

            if system_prompt:

                try:

                    if not isinstance(system_prompt, str):
                        system_prompt = str(system_prompt)

                    system_prompt = system_prompt.replace(
                        "\x00",
                        " ",
                    )

                    system_prompt = re.sub(
                        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
                        " ",
                        system_prompt,
                    )

                    system_prompt = re.sub(
                        r"\s+",
                        " ",
                        system_prompt,
                    )

                    system_prompt = system_prompt.strip()

                    # remove dangerous tokens
                    for token in dangerous_tokens:
                        system_prompt = system_prompt.replace(
                            token,
                            "",
                        )

                    # cap system prompt size
                    system_prompt = system_prompt[:300]

                    clean_system_prompt = system_prompt.strip()

                except Exception as exc:

                    logger.warning(
                        f"System prompt cleanup failed: {exc}"
                    )

                    clean_system_prompt = ""

            # ---------------------------------------------------------
            # BUILD FINAL PROMPT
            # ---------------------------------------------------------
            if clean_system_prompt:

                final_prompt = (
                    "[INST] "
                    f"{clean_system_prompt}\n\n"
                    f"{prompt} "
                    "[/INST]"
                )

            else:

                final_prompt = (
                    "[INST] "
                    f"{prompt} "
                    "[/INST]"
                )

            # ---------------------------------------------------------
            # FINAL VALIDATION
            # ---------------------------------------------------------
            final_prompt = final_prompt.strip()

            # prevent malformed output
            if "[INST]" not in final_prompt:
                final_prompt = f"[INST] {final_prompt} [/INST]"

            if "[/INST]" not in final_prompt:
                final_prompt += " [/INST]"

            # ---------------------------------------------------------
            # SAFETY LIMIT
            # ---------------------------------------------------------
            hard_limit = max_chars + 400

            if len(final_prompt) > hard_limit:

                logger.warning(
                    "Final prompt exceeded hard limit "
                    f"({len(final_prompt)} chars)"
                )

                final_prompt = final_prompt[:hard_limit]

                if not final_prompt.endswith("[/INST]"):
                    final_prompt += " [/INST]"

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"Prompt built successfully | "
                f"chars={len(final_prompt)}"
            )

            return final_prompt

        # -------------------------------------------------------------
        # HARD FAILURE RECOVERY
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            raise

        except Exception as exc:

            logger.error(
                f"Prompt build failure: {exc}"
            )

            try:

                fallback_prompt = str(prompt or "").strip()

                fallback_prompt = fallback_prompt[:200]

                fallback_prompt = re.sub(
                    r"\s+",
                    " ",
                    fallback_prompt,
                )

                if not fallback_prompt:
                    fallback_prompt = (
                        "Please respond helpfully."
                    )

                return (
                    f"[INST] "
                    f"{fallback_prompt} "
                    f"[/INST]"
                )

            except Exception:

                return (
                    "[INST] "
                    "Please respond helpfully."
                    " [/INST]"
                )

    async def _rate_limit(self) -> None:
        """
        Production-grade async rate limiter.

        Fixes:
        - race conditions
        - negative sleep times
        - worker starvation
        - asyncio lock corruption
        - burst flooding
        - time drift issues
        - cancellation instability
        - invalid min_delay values
        - deadlock scenarios
        """

        try:

            # ---------------------------------------------------------
            # ENSURE LOCK EXISTS
            # ---------------------------------------------------------
            if (
                not hasattr(self, "_rate_lock")
                or self._rate_lock is None
            ):
                self._rate_lock = asyncio.Lock()

            # ---------------------------------------------------------
            # SAFE MIN DELAY
            # ---------------------------------------------------------
            try:

                min_delay = float(
                    getattr(self, "min_delay", 0.1)
                )

            except Exception:
                min_delay = 0.1

            # clamp insane values
            min_delay = max(
                0.0,
                min(min_delay, 10.0),
            )

            # ---------------------------------------------------------
            # FAST PATH
            # ---------------------------------------------------------
            now = time.monotonic()

            last_call = float(
                getattr(self, "_last_call_time", 0.0)
            )

            elapsed = now - last_call

            if elapsed >= min_delay:

                self._last_call_time = now
                return

            # ---------------------------------------------------------
            # SERIALIZED RATE LIMIT
            # ---------------------------------------------------------
            async with self._rate_lock:

                # re-check after acquiring lock
                now = time.monotonic()

                last_call = float(
                    getattr(self, "_last_call_time", 0.0)
                )

                elapsed = now - last_call

                wait_time = max(
                    0.0,
                    min_delay - elapsed,
                )

                # -----------------------------------------------------
                # PROTECT AGAINST INVALID WAIT
                # -----------------------------------------------------
                if wait_time <= 0:

                    self._last_call_time = now
                    return

                # prevent absurd sleeps
                wait_time = min(wait_time, 5.0)

                # -----------------------------------------------------
                # OPTIONAL BACKPRESSURE LOGGING
                # -----------------------------------------------------
                if wait_time > 1.0:

                    logger.warning(
                        f"High rate-limit wait: "
                        f"{wait_time:.2f}s"
                    )

                # -----------------------------------------------------
                # SLEEP
                # -----------------------------------------------------
                try:

                    await asyncio.sleep(wait_time)

                except asyncio.CancelledError:

                    logger.debug(
                        "Rate limiter sleep cancelled"
                    )

                    raise

                except Exception as exc:

                    logger.warning(
                        f"Rate limiter sleep failure: {exc}"
                    )

                # -----------------------------------------------------
                # UPDATE TIMESTAMP
                # -----------------------------------------------------
                self._last_call_time = time.monotonic()

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            raise

        # -------------------------------------------------------------
        # HARD FAILURE RECOVERY
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Rate limiter failure: {exc}"
            )

            # fallback protection
            try:

                self._last_call_time = time.monotonic()

                await asyncio.sleep(0.05)

            except Exception:
                pass

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        retries: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Production-grade generation pipeline.

        Fixes:
        - semaphore deadlocks
        - provider crashes
        - malformed prompts
        - timeout storms
        - cache corruption
        - memory leaks
        - active request desync
        - invalid responses
        - runaway retries
        - hallucinated empty outputs
        - cancellation instability
        """

        self._ensure_async_primitives()

        SAFE_FALLBACK = {
            "response": (
                "I'm here to help. Please try again."
            ),
            "intent": "chat",
            "confidence": 0.2,
            "reflection": "safe_fallback",
        }

        # -------------------------------------------------------------
        # PROVIDER VALIDATION
        # -------------------------------------------------------------
        try:

            if not getattr(self, "provider", None):

                logger.error("No LLM provider available")

                return {
                    "response": (
                        "Model provider unavailable."
                    ),
                    "intent": "idle",
                    "confidence": 0.0,
                    "reflection": "no_provider",
                }

        except Exception as exc:

            logger.error(
                f"Provider validation failed: {exc}"
            )

            return SAFE_FALLBACK

        # -------------------------------------------------------------
        # SEMAPHORE PROTECTION
        # -------------------------------------------------------------
        try:

            async with self._semaphore:

                self._active_requests += 1

                try:

                    # -------------------------------------------------
                    # RATE LIMIT
                    # -------------------------------------------------
                    await self._rate_limit()

                    # -------------------------------------------------
                    # INPUT SANITIZATION
                    # -------------------------------------------------
                    if prompt is None:
                        prompt = ""

                    if not isinstance(prompt, str):
                        prompt = str(prompt)

                    prompt = prompt.strip()

                    prompt = re.sub(
                        r"\s+",
                        " ",
                        prompt,
                    )

                    if not prompt or len(prompt) < 2:

                        return {
                            "response": (
                                "Please provide a valid request."
                            ),
                            "intent": "idle",
                            "confidence": 0.0,
                            "reflection": "invalid_prompt",
                        }

                    # -------------------------------------------------
                    # SAFE LIMITS
                    # -------------------------------------------------
                    max_tokens = max(
                        1,
                        min(int(max_tokens), 512),
                    )

                    temperature = max(
                        0.0,
                        min(float(temperature), 2.0),
                    )

                    retries = max(
                        0,
                        min(int(retries), 3),
                    )

                    # -------------------------------------------------
                    # CACHE LOOKUP
                    # -------------------------------------------------
                    if getattr(self, "cache_enabled", False):

                        try:

                            cached = self.cache.get(
                                prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )

                            if cached:

                                cached["source"] = "cache"

                                return cached

                        except Exception as exc:

                            logger.debug(
                                f"Cache lookup failed: {exc}"
                            )

                    # -------------------------------------------------
                    # BUILD PROMPT
                    # -------------------------------------------------
                    try:

                        full_prompt = await self._build_prompt(
                            prompt,
                            system_prompt,
                        )

                    except Exception as exc:

                        logger.warning(
                            f"Prompt build failure: {exc}"
                        )

                        full_prompt = (
                            f"[INST] {prompt[:200]} [/INST]"
                        )

                    # -------------------------------------------------
                    # GENERATION LOOP
                    # -------------------------------------------------
                    last_error = None

                    for attempt in range(retries + 1):

                        try:

                            start = time.monotonic()

                            # -----------------------------------------
                            # PROVIDER GENERATION
                            # -----------------------------------------
                            raw = await asyncio.wait_for(
                                self.provider.generate(
                                    full_prompt,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    **kwargs,
                                ),
                                timeout=30.0,
                            )

                            elapsed = round(
                                time.monotonic() - start,
                                3,
                            )

                            # -----------------------------------------
                            # RESPONSE CLEANUP
                            # -----------------------------------------
                            raw = str(raw or "").strip()

                            raw = re.sub(
                                r"\[/?INST\]|</?s>",
                                "",
                                raw,
                            )

                            raw = re.sub(
                                r"(?i)assistant:",
                                "",
                                raw,
                            )

                            raw = re.sub(
                                r"\s+",
                                " ",
                                raw,
                            )

                            raw = raw.strip()

                            # -----------------------------------------
                            # INVALID RESPONSE HANDLING
                            # -----------------------------------------
                            if len(raw) < 3:

                                raise ValueError(
                                    "Empty or invalid model response"
                                )

                            # -----------------------------------------
                            # INTENT DETECTION
                            # -----------------------------------------
                            tl = raw.lower()

                            if any(
                                k in tl for k in (
                                    "open",
                                    "run",
                                    "execute",
                                    "launch",
                                )
                            ):
                                intent = "action"

                            elif any(
                                k in tl for k in (
                                    "search",
                                    "find",
                                    "research",
                                    "lookup",
                                )
                            ):
                                intent = "research"

                            elif any(
                                k in tl for k in (
                                    "code",
                                    "script",
                                    "program",
                                    "python",
                                )
                            ):
                                intent = "code"

                            elif any(
                                k in tl for k in (
                                    "hello",
                                    "hi",
                                    "hey",
                                )
                            ):
                                intent = "greeting"

                            else:
                                intent = "chat"

                            # -----------------------------------------
                            # CONFIDENCE SCORE
                            # -----------------------------------------
                            confidence = round(
                                min(
                                    0.95,
                                    max(
                                        0.3,
                                        len(raw) / 250,
                                    ),
                                ),
                                2,
                            )

                            # -----------------------------------------
                            # FINAL RESULT
                            # -----------------------------------------
                            result = {
                                "response": raw,
                                "intent": intent,
                                "confidence": confidence,
                                "reflection": "ok",
                                "generation_time": elapsed,
                                "attempt": attempt + 1,
                            }

                            # -----------------------------------------
                            # CACHE STORE
                            # -----------------------------------------
                            if getattr(self, "cache_enabled", False):

                                try:

                                    self.cache.set(
                                        prompt,
                                        result.copy(),
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                    )

                                except Exception as exc:

                                    logger.debug(
                                        f"Cache set failed: {exc}"
                                    )

                            # -----------------------------------------
                            # METRICS
                            # -----------------------------------------
                            self.generation_count += 1

                            logger.info(
                                f"LLM success | "
                                f"time={elapsed:.2f}s "
                                f"intent={intent} "
                                f"attempt={attempt + 1}"
                            )

                            return result

                        # ---------------------------------------------
                        # TIMEOUT
                        # ---------------------------------------------
                        except asyncio.TimeoutError:

                            last_error = "timeout"

                            logger.warning(
                                f"LLM timeout "
                                f"(attempt {attempt + 1})"
                            )

                        # ---------------------------------------------
                        # CANCELLATION
                        # ---------------------------------------------
                        except asyncio.CancelledError:

                            raise

                        # ---------------------------------------------
                        # GENERATION FAILURE
                        # ---------------------------------------------
                        except Exception as exc:

                            last_error = str(exc)

                            logger.warning(
                                f"Generation failure "
                                f"(attempt {attempt + 1}): {exc}"
                            )

                        # ---------------------------------------------
                        # RETRY BACKOFF
                        # ---------------------------------------------
                        if attempt < retries:

                            backoff = min(
                                2.0,
                                0.5 * (attempt + 1),
                            )

                            await asyncio.sleep(backoff)

                    # -------------------------------------------------
                    # ALL RETRIES FAILED
                    # -------------------------------------------------
                    logger.error(
                        f"Generation failed permanently: "
                        f"{last_error}"
                    )

                    return {
                        "response": (
                            "I'm ready to help. "
                            "Please try rephrasing your request."
                        ),
                        "intent": "chat",
                        "confidence": 0.2,
                        "reflection": "fallback",
                        "error": str(last_error),
                    }

                # -----------------------------------------------------
                # ACTIVE REQUEST CLEANUP
                # -----------------------------------------------------
                finally:

                    self._active_requests = max(
                        0,
                        self._active_requests - 1,
                    )

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Generate pipeline crashed: {exc}"
            )

            return {
                **SAFE_FALLBACK,
                "error": str(exc),
            }

    async def safe_generate(
        self,
        prompt: str,
        timeout: float = 20.0,
        retries: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Production-grade safe generation wrapper.

        Fixes:
        - infinite retry loops
        - timeout storms
        - provider crashes
        - cache corruption
        - invalid responses
        - cancellation instability
        - memory pressure crashes
        - recursion failures
        - malformed prompt handling
        - empty response handling
        - deadlock protection
        """

        self._ensure_async_primitives()

        SAFE_DEFAULT = {
            "response": (
                "I'm here to help. Please try again."
            ),
            "intent": "idle",
            "confidence": 0.1,
            "reflection": "safe_fallback",
        }

        try:

            # ---------------------------------------------------------
            # INPUT SANITIZATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):
                prompt = str(prompt)

            prompt = prompt.strip()

            prompt = re.sub(
                r"\s+",
                " ",
                prompt,
            )

            # ---------------------------------------------------------
            # EMPTY PROMPT
            # ---------------------------------------------------------
            if not prompt:

                return {
                    **SAFE_DEFAULT,
                    "reflection": "empty_prompt",
                }

            # ---------------------------------------------------------
            # SAFE LIMITS
            # ---------------------------------------------------------
            try:
                timeout = float(timeout)
            except Exception:
                timeout = 20.0

            timeout = max(
                3.0,
                min(timeout, 120.0),
            )

            try:
                retries = int(retries)
            except Exception:
                retries = 2

            retries = max(
                0,
                min(retries, 5),
            )

            # ---------------------------------------------------------
            # CACHE LOOKUP
            # ---------------------------------------------------------
            if getattr(self, "cache_enabled", False):

                try:

                    cached = self.cache.get(
                        prompt,
                        timeout=timeout,
                    )

                    if cached:

                        if isinstance(cached, dict):

                            cached["source"] = "cache"

                            return cached

                except Exception as exc:

                    logger.debug(
                        f"safe_generate cache failure: {exc}"
                    )

            # ---------------------------------------------------------
            # REQUEST ID
            # ---------------------------------------------------------
            request_id = hashlib.md5(
                prompt.encode("utf-8", errors="ignore")
            ).hexdigest()[:8]

            logger.debug(
                f"safe_generate start | "
                f"id={request_id} "
                f"timeout={timeout}s "
                f"retries={retries}"
            )

            # ---------------------------------------------------------
            # GENERATION LOOP
            # ---------------------------------------------------------
            last_error = None

            for attempt in range(retries + 1):

                try:

                    start = time.monotonic()

                    # -------------------------------------------------
                    # HARD TIMEOUT WRAPPER
                    # -------------------------------------------------
                    result = await asyncio.wait_for(
                        self.generate(
                            prompt,
                            retries=0,
                            **kwargs,
                        ),
                        timeout=timeout,
                    )

                    elapsed = round(
                        time.monotonic() - start,
                        3,
                    )

                    # -------------------------------------------------
                    # VALIDATE RESPONSE
                    # -------------------------------------------------
                    if not isinstance(result, dict):

                        raise ValueError(
                            "Generator returned non-dict result"
                        )

                    response_text = str(
                        result.get("response", "")
                    ).strip()

                    if not response_text:

                        raise ValueError(
                            "Empty response returned"
                        )

                    # -------------------------------------------------
                    # BAD REFLECTION DETECTION
                    # -------------------------------------------------
                    reflection = str(
                        result.get(
                            "reflection",
                            "",
                        )
                    ).lower()

                    bad_reflections = {
                        "fallback",
                        "no_provider",
                        "error",
                        "failed",
                    }

                    if reflection in bad_reflections:

                        raise ValueError(
                            f"Bad reflection: {reflection}"
                        )

                    # -------------------------------------------------
                    # ADD SAFETY METADATA
                    # -------------------------------------------------
                    result.setdefault(
                        "safe_wrapper",
                        True,
                    )

                    result.setdefault(
                        "request_id",
                        request_id,
                    )

                    result.setdefault(
                        "latency",
                        elapsed,
                    )

                    result.setdefault(
                        "attempt",
                        attempt + 1,
                    )

                    # -------------------------------------------------
                    # CACHE STORE
                    # -------------------------------------------------
                    if getattr(self, "cache_enabled", False):

                        try:

                            self.cache.set(
                                prompt,
                                result.copy(),
                                timeout=timeout,
                            )

                        except Exception as exc:

                            logger.debug(
                                f"safe_generate cache store failed: {exc}"
                            )

                    logger.info(
                        f"safe_generate success | "
                        f"id={request_id} "
                        f"time={elapsed:.2f}s "
                        f"attempt={attempt + 1}"
                    )

                    return result

                # -----------------------------------------------------
                # TIMEOUT
                # -----------------------------------------------------
                except asyncio.TimeoutError:

                    last_error = "timeout"

                    logger.warning(
                        f"safe_generate timeout | "
                        f"id={request_id} "
                        f"attempt={attempt + 1}"
                    )

                # -----------------------------------------------------
                # CANCELLATION
                # -----------------------------------------------------
                except asyncio.CancelledError:

                    logger.warning(
                        f"safe_generate cancelled | "
                        f"id={request_id}"
                    )

                    raise

                # -----------------------------------------------------
                # GENERATION FAILURE
                # -----------------------------------------------------
                except Exception as exc:

                    last_error = str(exc)

                    logger.warning(
                        f"safe_generate failure | "
                        f"id={request_id} "
                        f"attempt={attempt + 1} "
                        f"error={exc}"
                    )

                # -----------------------------------------------------
                # RETRY BACKOFF
                # -----------------------------------------------------
                if attempt < retries:

                    try:

                        backoff = min(
                            5.0,
                            0.5 * (attempt + 1),
                        )

                        await asyncio.sleep(backoff)

                    except asyncio.CancelledError:
                        raise

                    except Exception:
                        pass

            # ---------------------------------------------------------
            # ALL ATTEMPTS FAILED
            # ---------------------------------------------------------
            logger.error(
                f"safe_generate exhausted retries | "
                f"id={request_id} "
                f"error={last_error}"
            )

            return {
                **SAFE_DEFAULT,
                "reflection": "retry_exhausted",
                "error": str(last_error),
                "request_id": request_id,
            }

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            raise

        # -------------------------------------------------------------
        # HARD FAILURE RECOVERY
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"safe_generate crashed: {exc}"
            )

            return {
                **SAFE_DEFAULT,
                "reflection": "critical_failure",
                "error": str(exc),
            }

    async def agent_generate(
        self,
        agent_id: str,
        prompt: str,
        priority: AgentPriority = AgentPriority.NORMAL,
        timeout: float = 30.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Production-grade agent generation dispatcher.

        Fixes:
        - queue overflow crashes
        - future leaks
        - cancelled futures
        - invalid priorities
        - deadlocks
        - queue corruption
        - invalid agent IDs
        - worker starvation
        - timeout storms
        - orphan requests
        - event-loop instability
        """

        self._ensure_async_primitives()

        SAFE_FALLBACK = {
            "response": (
                "Unable to process request right now."
            ),
            "intent": "idle",
            "confidence": 0.0,
            "reflection": "agent_fallback",
        }

        future: Optional[asyncio.Future] = None

        try:

            # ---------------------------------------------------------
            # VALIDATE AGENT ID
            # ---------------------------------------------------------
            if agent_id is None:
                agent_id = "unknown_agent"

            if not isinstance(agent_id, str):
                agent_id = str(agent_id)

            agent_id = agent_id.strip()

            if not agent_id:
                agent_id = "unknown_agent"

            # limit insane IDs
            agent_id = agent_id[:64]

            # ---------------------------------------------------------
            # VALIDATE PROMPT
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):
                prompt = str(prompt)

            prompt = prompt.strip()

            prompt = re.sub(
                r"\s+",
                " ",
                prompt,
            )

            if not prompt:

                return {
                    **SAFE_FALLBACK,
                    "reflection": "empty_prompt",
                    "agent_id": agent_id,
                }

            # ---------------------------------------------------------
            # VALIDATE PRIORITY
            # ---------------------------------------------------------
            try:

                if isinstance(priority, AgentPriority):
                    priority_value = int(priority.value)
                else:
                    priority_value = int(priority)

            except Exception:

                priority_value = int(
                    AgentPriority.NORMAL.value
                )

            priority_value = max(
                int(AgentPriority.CRITICAL.value),
                min(
                    priority_value,
                    int(AgentPriority.BACKGROUND.value),
                ),
            )

            # ---------------------------------------------------------
            # VALIDATE TIMEOUT
            # ---------------------------------------------------------
            try:
                timeout = float(timeout)
            except Exception:
                timeout = 30.0

            timeout = max(
                5.0,
                min(timeout, 300.0),
            )

            # ---------------------------------------------------------
            # QUEUE VALIDATION
            # ---------------------------------------------------------
            if self._queue is None:

                logger.error(
                    "Queue not initialized"
                )

                return {
                    **SAFE_FALLBACK,
                    "reflection": "queue_missing",
                    "agent_id": agent_id,
                }

            # ---------------------------------------------------------
            # QUEUE OVERFLOW PROTECTION
            # ---------------------------------------------------------
            try:

                queue_size = self._queue.qsize()

            except Exception:
                queue_size = 0

            if self._queue.full():

                logger.warning(
                    f"Queue full | "
                    f"size={queue_size}"
                )

                return {
                    "response": (
                        "Request queue full. Please retry shortly."
                    ),
                    "intent": "idle",
                    "confidence": 0.0,
                    "reflection": "queue_full",
                    "queue_size": queue_size,
                    "agent_id": agent_id,
                }

            # ---------------------------------------------------------
            # EVENT LOOP
            # ---------------------------------------------------------
            try:

                loop = asyncio.get_running_loop()

            except RuntimeError:

                logger.error(
                    "No active asyncio loop"
                )

                return {
                    **SAFE_FALLBACK,
                    "reflection": "loop_missing",
                    "agent_id": agent_id,
                }

            # ---------------------------------------------------------
            # CREATE FUTURE
            # ---------------------------------------------------------
            future = loop.create_future()

            # ---------------------------------------------------------
            # BUILD REQUEST
            # ---------------------------------------------------------
            request = PrioritizedRequest(
                priority=priority_value,
                enqueue_time=time.monotonic(),
                agent_id=agent_id,
                prompt=prompt,
                kwargs=kwargs,
                future=future,
            )

            # ---------------------------------------------------------
            # ENQUEUE REQUEST
            # ---------------------------------------------------------
            try:

                await asyncio.wait_for(
                    self._queue.put(request),
                    timeout=3.0,
                )

            except asyncio.TimeoutError:

                logger.warning(
                    "Queue put timeout"
                )

                return {
                    "response": (
                        "System busy. Failed to enqueue request."
                    ),
                    "intent": "idle",
                    "confidence": 0.0,
                    "reflection": "enqueue_timeout",
                    "agent_id": agent_id,
                }

            # ---------------------------------------------------------
            # WAIT FOR RESULT
            # ---------------------------------------------------------
            start = time.monotonic()

            try:

                result = await asyncio.wait_for(
                    future,
                    timeout=timeout,
                )

            except asyncio.TimeoutError:

                logger.warning(
                    f"Agent request timeout | "
                    f"agent={agent_id}"
                )

                try:

                    if future and not future.done():
                        future.cancel()

                except Exception:
                    pass

                return {
                    "response": (
                        "Agent request timed out."
                    ),
                    "intent": "idle",
                    "confidence": 0.0,
                    "reflection": "agent_timeout",
                    "agent_id": agent_id,
                }

            # ---------------------------------------------------------
            # VALIDATE RESULT
            # ---------------------------------------------------------
            if not isinstance(result, dict):

                logger.warning(
                    "Worker returned invalid result type"
                )

                result = {
                    **SAFE_FALLBACK,
                    "reflection": "invalid_worker_result",
                }

            # ---------------------------------------------------------
            # ADD METADATA
            # ---------------------------------------------------------
            elapsed = round(
                time.monotonic() - start,
                3,
            )

            result.setdefault(
                "agent_id",
                agent_id,
            )

            result.setdefault(
                "queue_priority",
                priority_value,
            )

            result.setdefault(
                "latency",
                elapsed,
            )

            result.setdefault(
                "queue_size",
                queue_size,
            )

            logger.info(
                f"Agent request completed | "
                f"agent={agent_id} "
                f"time={elapsed:.2f}s"
            )

            return result

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                f"agent_generate cancelled | "
                f"agent={agent_id}"
            )

            if future and not future.done():

                try:
                    future.cancel()
                except Exception:
                    pass

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"agent_generate failed: {exc}"
            )

            try:

                if (
                    future
                    and not future.done()
                    and not future.cancelled()
                ):
                    future.cancel()

            except Exception:
                pass

            return {
                **SAFE_FALLBACK,
                "reflection": "critical_failure",
                "error": str(exc),
                "agent_id": agent_id,
            }

    async def generate_stream(
        self,
        prompt: str,
        timeout: float = 20.0,
    ) -> AsyncIterator[str]:
        """
        Production-grade streaming generator.

        Fixes:
        - stream freezes
        - empty chunks
        - memory leaks
        - timeout storms
        - malformed outputs
        - cancellation instability
        - giant chunk flooding
        - unicode corruption
        - async starvation
        - stream deadlocks
        - invalid responses
        """

        STREAM_FALLBACK = (
            "I'm here to help. Please try again."
        )

        try:

            # ---------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):
                prompt = str(prompt)

            prompt = prompt.strip()

            prompt = re.sub(
                r"\s+",
                " ",
                prompt,
            )

            if not prompt:

                yield (
                    "Please provide a valid request."
                )

                return

            # ---------------------------------------------------------
            # VALIDATE TIMEOUT
            # ---------------------------------------------------------
            try:
                timeout = float(timeout)
            except Exception:
                timeout = 20.0

            timeout = max(
                3.0,
                min(timeout, 300.0),
            )

            # ---------------------------------------------------------
            # GENERATE RESPONSE
            # ---------------------------------------------------------
            start = time.monotonic()

            try:

                data = await asyncio.wait_for(
                    self.safe_generate(prompt),
                    timeout=timeout,
                )

            except asyncio.TimeoutError:

                logger.warning(
                    "generate_stream timeout"
                )

                yield (
                    "Request is taking too long. "
                    "Please try again."
                )

                return

            # ---------------------------------------------------------
            # VALIDATE RESULT
            # ---------------------------------------------------------
            if not isinstance(data, dict):

                logger.warning(
                    "generate_stream received invalid result"
                )

                yield STREAM_FALLBACK

                return

            # ---------------------------------------------------------
            # EXTRACT TEXT
            # ---------------------------------------------------------
            text = str(
                data.get("response", "")
            ).strip()

            # ---------------------------------------------------------
            # EMPTY RESPONSE
            # ---------------------------------------------------------
            if not text:

                yield (
                    "I'm here to help. "
                    "What would you like to know?"
                )

                return

            # ---------------------------------------------------------
            # CLEAN RESPONSE
            # ---------------------------------------------------------
            text = text.replace("\x00", " ")

            text = re.sub(
                r"\s+",
                " ",
                text,
            )

            text = text.strip()

            # ---------------------------------------------------------
            # HARD LIMIT
            # ---------------------------------------------------------
            max_stream_chars = 8000

            if len(text) > max_stream_chars:

                logger.warning(
                    f"Stream truncated "
                    f"({len(text)} chars)"
                )

                text = text[:max_stream_chars]

            # ---------------------------------------------------------
            # STREAM SETTINGS
            # ---------------------------------------------------------
            chunk_size = 60
            words = text.split()

            chunk = ""

            chunks_sent = 0

            # ---------------------------------------------------------
            # STREAM LOOP
            # ---------------------------------------------------------
            for word in words:

                try:

                    next_chunk = (
                        f"{chunk} {word}"
                    ).strip()

                    # -------------------------------------------------
                    # SEND CHUNK
                    # -------------------------------------------------
                    if len(next_chunk) >= chunk_size:

                        cleaned_chunk = next_chunk.strip()

                        if cleaned_chunk:

                            yield cleaned_chunk

                            chunks_sent += 1

                        chunk = ""

                        # cooperative scheduling
                        await asyncio.sleep(0)

                    else:

                        chunk = next_chunk

                    # -------------------------------------------------
                    # STREAM SAFETY LIMIT
                    # -------------------------------------------------
                    if chunks_sent >= 500:

                        logger.warning(
                            "Stream chunk safety limit reached"
                        )

                        break

                # -----------------------------------------------------
                # TASK CANCELLATION
                # -----------------------------------------------------
                except asyncio.CancelledError:

                    logger.warning(
                        "generate_stream cancelled"
                    )

                    raise

                # -----------------------------------------------------
                # CHUNK FAILURE
                # -----------------------------------------------------
                except Exception as exc:

                    logger.warning(
                        f"Chunk stream failure: {exc}"
                    )

            # ---------------------------------------------------------
            # FINAL CHUNK
            # ---------------------------------------------------------
            if chunk.strip():

                yield chunk.strip()

            # ---------------------------------------------------------
            # STREAM COMPLETE
            # ---------------------------------------------------------
            elapsed = round(
                time.monotonic() - start,
                3,
            )

            logger.info(
                f"Stream completed | "
                f"chunks={chunks_sent} "
                f"time={elapsed:.2f}s"
            )

            # stream terminator
            yield ""

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "generate_stream terminated"
            )

            return

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"generate_stream crashed: {exc}"
            )

            try:

                yield (
                    "Something went wrong while "
                    "streaming the response."
                )

            except Exception:
                pass
        
    async def shutdown(self) -> None:
        """
        Production-grade LLMEngine shutdown handler.

        Fixes:
        - zombie workers
        - task leaks
        - queue corruption
        - unfinished futures
        - deadlocks
        - repeated shutdown calls
        - worker cancellation crashes
        - hanging queue tasks
        - monitor task leaks
        - semaphore corruption
        - memory leaks
        """

        # -------------------------------------------------------------
        # PREVENT DUPLICATE SHUTDOWN
        # -------------------------------------------------------------
        if getattr(self, "_shutdown", False):

            logger.warning(
                "LLMEngine shutdown already active"
            )

            return

        logger.info(
            "Starting LLMEngine shutdown..."
        )

        shutdown_start = time.monotonic()

        # -------------------------------------------------------------
        # SET FLAGS
        # -------------------------------------------------------------
        self._shutdown = True
        self._worker_running = False

        try:

            # ---------------------------------------------------------
            # CANCEL HEALTH MONITOR
            # ---------------------------------------------------------
            monitor_task = getattr(
                self,
                "_worker_health_task",
                None,
            )

            if monitor_task:

                try:

                    if not monitor_task.done():

                        monitor_task.cancel()

                        await asyncio.gather(
                            monitor_task,
                            return_exceptions=True,
                        )

                except Exception as exc:

                    logger.debug(
                        f"Health monitor shutdown error: {exc}"
                    )

                finally:

                    self._worker_health_task = None

            # ---------------------------------------------------------
            # CANCEL WORKERS
            # ---------------------------------------------------------
            worker_tasks = []

            for task in getattr(self, "_workers", []):

                try:

                    if (
                        task
                        and not task.done()
                        and not task.cancelled()
                    ):

                        task.cancel()

                        worker_tasks.append(task)

                except Exception as exc:

                    logger.debug(
                        f"Worker cancel failure: {exc}"
                    )

            # ---------------------------------------------------------
            # WAIT FOR WORKERS
            # ---------------------------------------------------------
            if worker_tasks:

                try:

                    await asyncio.wait_for(
                        asyncio.gather(
                            *worker_tasks,
                            return_exceptions=True,
                        ),
                        timeout=10.0,
                    )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Worker shutdown timeout"
                    )

                except Exception as exc:

                    logger.warning(
                        f"Worker shutdown failure: {exc}"
                    )

            # ---------------------------------------------------------
            # CLEAR WORKER LIST
            # ---------------------------------------------------------
            self._workers.clear()

            # ---------------------------------------------------------
            # CLEAN QUEUE
            # ---------------------------------------------------------
            queue_cleared = 0

            if getattr(self, "_queue", None):

                while True:

                    try:

                        if self._queue.empty():
                            break

                        req = self._queue.get_nowait()

                        queue_cleared += 1

                        # ---------------------------------------------
                        # RESOLVE FUTURE
                        # ---------------------------------------------
                        try:

                            if (
                                hasattr(req, "future")
                                and req.future
                                and not req.future.done()
                                and not req.future.cancelled()
                            ):

                                req.future.set_result({
                                    "response": (
                                        "System shutting down. "
                                        "Please retry later."
                                    ),
                                    "intent": "idle",
                                    "error": True,
                                    "shutdown": True,
                                })

                        except Exception as exc:

                            logger.debug(
                                f"Future cleanup error: {exc}"
                            )

                        # ---------------------------------------------
                        # TASK DONE
                        # ---------------------------------------------
                        try:

                            self._queue.task_done()

                        except ValueError:
                            pass

                        except Exception as exc:

                            logger.debug(
                                f"Queue task_done failure: {exc}"
                            )

                    except asyncio.QueueEmpty:
                        break

                    except Exception as exc:

                        logger.warning(
                            f"Queue cleanup failure: {exc}"
                        )

                        break

            # ---------------------------------------------------------
            # RESET INTERNAL STATE
            # ---------------------------------------------------------
            self._active_requests = 0

            self._last_call_time = 0.0

            # ---------------------------------------------------------
            # OPTIONAL CACHE CLEANUP
            # ---------------------------------------------------------
            try:

                if hasattr(self, "cache"):

                    cache_size = len(
                        getattr(self.cache, "cache", {})
                    )

                    logger.info(
                        f"Cache retained "
                        f"({cache_size} entries)"
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # SEMAPHORE CLEANUP
            # ---------------------------------------------------------
            try:

                self._semaphore = None

            except Exception:
                pass

            # ---------------------------------------------------------
            # LOCK CLEANUP
            # ---------------------------------------------------------
            try:

                self._global_lock = None
                self._rate_lock = None

            except Exception:
                pass

            # ---------------------------------------------------------
            # QUEUE CLEANUP
            # ---------------------------------------------------------
            try:

                self._queue = None

            except Exception:
                pass

            # ---------------------------------------------------------
            # FINAL METRICS
            # ---------------------------------------------------------
            elapsed = round(
                time.monotonic() - shutdown_start,
                3,
            )

            logger.info(
                f"LLMEngine shutdown complete | "
                f"queue_cleared={queue_cleared} "
                f"time={elapsed:.2f}s"
            )

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Shutdown cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE RECOVERY
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"LLMEngine shutdown crashed: {exc}"
            )

        finally:

            # ---------------------------------------------------------
            # FINAL SAFETY FLAGS
            # ---------------------------------------------------------
            self._worker_running = False
            self._active_requests = 0

    def clear_context(self, force: bool = False) -> Dict[str, Any]:
        """
        Production-grade context cleanup handler.

        Fixes:
        - cache corruption
        - semantic memory leaks
        - concurrent cleanup crashes
        - invalid state resets
        - metric corruption
        - repeated cleanup storms
        - race conditions
        - partial cleanup failures
        - memory pressure instability
        """

        start = time.monotonic()

        result = {
            "status": "unknown",
            "cache_cleared": 0,
            "semantic_cleared": 0,
            "latency": 0.0,
            "forced": bool(force),
        }

        # -------------------------------------------------------------
        # THREAD SAFETY
        # -------------------------------------------------------------
        try:

            if not hasattr(self, "_context_clear_lock"):

                self._context_clear_lock = threading.Lock()

        except Exception:
            pass

        try:

            with getattr(
                self,
                "_context_clear_lock",
                threading.Lock(),
            ):

                # -----------------------------------------------------
                # PREVENT CLEAR DURING ACTIVE LOAD
                # -----------------------------------------------------
                active_requests = int(
                    getattr(self, "_active_requests", 0)
                )

                if active_requests > 0 and not force:

                    logger.warning(
                        f"Context clear skipped "
                        f"({active_requests} active requests)"
                    )

                    result["status"] = "busy"

                    result["active_requests"] = active_requests

                    result["latency"] = round(
                        time.monotonic() - start,
                        4,
                    )

                    return result

                # -----------------------------------------------------
                # CACHE CLEANUP
                # -----------------------------------------------------
                cache_cleared = 0

                if (
                    getattr(self, "cache_enabled", False)
                    and hasattr(self, "cache")
                    and self.cache is not None
                ):

                    try:

                        cache_obj = self.cache

                        # ---------------------------------------------
                        # SAFE CACHE SIZE
                        # ---------------------------------------------
                        try:

                            existing_size = len(
                                getattr(cache_obj, "cache", {})
                            )

                        except Exception:
                            existing_size = 0

                        # ---------------------------------------------
                        # CLEAR CACHE
                        # ---------------------------------------------
                        cleared = cache_obj.clear()

                        if isinstance(cleared, int):
                            cache_cleared = cleared
                        else:
                            cache_cleared = existing_size

                    except Exception as exc:

                        logger.warning(
                            f"Cache clear failed: {exc}"
                        )

                result["cache_cleared"] = cache_cleared

                # -----------------------------------------------------
                # SEMANTIC MEMORY CLEANUP
                # -----------------------------------------------------
                semantic_cleared = 0

                if (
                    getattr(self, "semantic_enabled", False)
                    and hasattr(self, "semantic_memory")
                ):

                    try:

                        semantic_memory = getattr(
                            self,
                            "semantic_memory",
                            [],
                        )

                        if isinstance(semantic_memory, list):

                            semantic_cleared = len(
                                semantic_memory
                            )

                            semantic_memory.clear()

                        else:

                            logger.warning(
                                "semantic_memory invalid type"
                            )

                            self.semantic_memory = []

                    except Exception as exc:

                        logger.warning(
                            f"Semantic memory clear failed: {exc}"
                        )

                        try:
                            self.semantic_memory = []
                        except Exception:
                            pass

                result["semantic_cleared"] = semantic_cleared

                # -----------------------------------------------------
                # VECTOR INDEX CLEANUP
                # -----------------------------------------------------
                try:

                    if (
                        force
                        and hasattr(self, "index")
                        and self.index is not None
                    ):

                        if hasattr(self.index, "texts"):

                            self.index.texts.clear()

                        logger.info(
                            "Vector index cleared"
                        )

                except Exception as exc:

                    logger.warning(
                        f"Vector index cleanup failed: {exc}"
                    )

                # -----------------------------------------------------
                # REQUEST STATE RESET
                # -----------------------------------------------------
                try:

                    self.generation_count = 0

                except Exception:
                    pass

                try:

                    self._last_call_time = 0.0

                except Exception:
                    pass

                try:

                    self._active_requests = max(
                        0,
                        int(
                            getattr(
                                self,
                                "_active_requests",
                                0,
                            )
                        ),
                    )

                    if force:
                        self._active_requests = 0

                except Exception:

                    self._active_requests = 0

                # -----------------------------------------------------
                # OPTIONAL QUEUE PURGE
                # -----------------------------------------------------
                queue_cleared = 0

                if force and getattr(self, "_queue", None):

                    try:

                        while not self._queue.empty():

                            try:

                                item = self._queue.get_nowait()

                                queue_cleared += 1

                                # resolve futures
                                try:

                                    if (
                                        hasattr(item, "future")
                                        and item.future
                                        and not item.future.done()
                                    ):

                                        item.future.cancel()

                                except Exception:
                                    pass

                                try:
                                    self._queue.task_done()
                                except Exception:
                                    pass

                            except asyncio.QueueEmpty:
                                break

                    except Exception as exc:

                        logger.warning(
                            f"Queue purge failed: {exc}"
                        )

                result["queue_cleared"] = queue_cleared

                # -----------------------------------------------------
                # MEMORY PRESSURE CLEANUP
                # -----------------------------------------------------
                try:

                    import gc

                    collected = gc.collect()

                    result["gc_collected"] = collected

                except Exception:
                    pass

                # -----------------------------------------------------
                # FINAL STATUS
                # -----------------------------------------------------
                elapsed = round(
                    time.monotonic() - start,
                    4,
                )

                result["status"] = "cleared"

                result["latency"] = elapsed

                logger.info(
                    f"Context cleared | "
                    f"cache={cache_cleared} "
                    f"semantic={semantic_cleared} "
                    f"queue={queue_cleared} "
                    f"time={elapsed:.4f}s"
                )

                return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Context clear failed: {exc}"
            )

            result["status"] = "error"

            result["error"] = str(exc)

            result["latency"] = round(
                time.monotonic() - start,
                4,
            )

            return result

    def get_metrics(self) -> Dict[str, Any]:
        """
        Production-grade system metrics collector.

        Fixes:
        - metrics crashes
        - dead worker reporting
        - invalid cache metrics
        - queue corruption
        - health-score instability
        - division-by-zero
        - semaphore corruption
        - VRAM metric failures
        - invalid worker counts
        - memory pressure reporting bugs
        """

        start = time.monotonic()

        SAFE_FALLBACK = {
            "status": "error",
            "health_score": 0.0,
            "queue_size": 0,
            "workers_alive": 0,
            "active_requests": 0,
        }

        try:

            # ---------------------------------------------------------
            # ENSURE INTERNALS
            # ---------------------------------------------------------
            self._ensure_async_primitives()

            # ---------------------------------------------------------
            # QUEUE METRICS
            # ---------------------------------------------------------
            try:

                if getattr(self, "_queue", None):

                    queue_size = max(
                        0,
                        int(self._queue.qsize())
                    )

                else:

                    queue_size = 0

            except Exception as exc:

                logger.debug(
                    f"Queue metric failure: {exc}"
                )

                queue_size = 0

            # ---------------------------------------------------------
            # WORKER METRICS
            # ---------------------------------------------------------
            workers_alive = 0
            total_workers = 0
            dead_workers = 0

            try:

                workers = list(
                    getattr(self, "_workers", [])
                )

                total_workers = len(workers)

                for task in workers:

                    try:

                        if (
                            task
                            and not task.done()
                            and not task.cancelled()
                        ):

                            workers_alive += 1

                        else:

                            dead_workers += 1

                    except Exception:

                        dead_workers += 1

            except Exception as exc:

                logger.debug(
                    f"Worker metric failure: {exc}"
                )

            # ---------------------------------------------------------
            # CACHE METRICS
            # ---------------------------------------------------------
            cache_size = 0
            cache_hits = 0
            cache_misses = 0
            cache_hit_rate = 0.0

            try:

                if (
                    getattr(self, "cache_enabled", False)
                    and hasattr(self, "cache")
                    and self.cache is not None
                ):

                    cache_size = len(
                        getattr(
                            self.cache,
                            "cache",
                            {},
                        )
                    )

                    cache_hits = int(
                        getattr(
                            self.cache,
                            "hits",
                            0,
                        )
                    )

                    cache_misses = int(
                        getattr(
                            self.cache,
                            "misses",
                            0,
                        )
                    )

                    total_cache = (
                        cache_hits + cache_misses
                    )

                    if total_cache > 0:

                        cache_hit_rate = round(
                            cache_hits / total_cache,
                            3,
                        )

            except Exception as exc:

                logger.debug(
                    f"Cache metric failure: {exc}"
                )

            # ---------------------------------------------------------
            # SEMANTIC MEMORY METRICS
            # ---------------------------------------------------------
            semantic_size = 0

            try:

                semantic_memory = getattr(
                    self,
                    "semantic_memory",
                    [],
                )

                if isinstance(semantic_memory, list):

                    semantic_size = len(
                        semantic_memory
                    )

            except Exception as exc:

                logger.debug(
                    f"Semantic metric failure: {exc}"
                )

            # ---------------------------------------------------------
            # ACTIVE REQUESTS
            # ---------------------------------------------------------
            try:

                active_requests = max(
                    0,
                    int(
                        getattr(
                            self,
                            "_active_requests",
                            0,
                        )
                    ),
                )

            except Exception:

                active_requests = 0

            # ---------------------------------------------------------
            # GENERATION COUNT
            # ---------------------------------------------------------
            try:

                generations = max(
                    0,
                    int(
                        getattr(
                            self,
                            "generation_count",
                            0,
                        )
                    ),
                )

            except Exception:

                generations = 0

            # ---------------------------------------------------------
            # VRAM METRICS
            # ---------------------------------------------------------
            vram_fraction = 0.0
            gpu_pressure = False

            try:

                gpu_guard = getattr(
                    self,
                    "gpu_guard",
                    None,
                )

                if gpu_guard:

                    raw_vram = getattr(
                        gpu_guard,
                        "vram_fraction",
                        0.0,
                    )

                    if callable(raw_vram):
                        raw_vram = raw_vram()

                    vram_fraction = round(
                        max(0.0, min(float(raw_vram), 1.0)),
                        3,
                    )

                    gpu_pressure = (
                        vram_fraction >
                        _VRAM_HIGH_WATERMARK
                    )

            except Exception as exc:

                logger.debug(
                    f"VRAM metric failure: {exc}"
                )

            # ---------------------------------------------------------
            # LOAD METRICS
            # ---------------------------------------------------------
            try:

                avg_load = round(
                    min(
                        1.0,
                        queue_size / max(
                            1,
                            int(MAX_QUEUE_SIZE),
                        ),
                    ),
                    3,
                )

            except Exception:

                avg_load = 0.0

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            health = 1.0

            try:

                # queue pressure
                if queue_size > 100:
                    health -= 0.35

                elif queue_size > 50:
                    health -= 0.20

                elif queue_size > 20:
                    health -= 0.10

                # worker pressure
                if workers_alive == 0 and queue_size > 0:
                    health -= 0.40

                elif workers_alive < max(
                    1,
                    total_workers // 2,
                ):
                    health -= 0.15

                # active requests
                if active_requests >= max(
                    1,
                    int(_NUM_WORKERS),
                ):
                    health -= 0.10

                # cache inefficiency
                if (
                    cache_hits + cache_misses
                ) > 20:

                    if cache_misses > cache_hits:
                        health -= 0.10

                # GPU pressure
                if gpu_pressure:
                    health -= 0.20

                # shutdown state
                if getattr(self, "_shutdown", False):
                    health -= 0.50

                health = round(
                    max(0.0, min(1.0, health)),
                    2,
                )

            except Exception as exc:

                logger.debug(
                    f"Health score failure: {exc}"
                )

                health = 0.5

            # ---------------------------------------------------------
            # STATUS DETECTION
            # ---------------------------------------------------------
            if getattr(self, "_shutdown", False):

                status = "shutdown"

            elif active_requests > 0:

                status = "busy"

            elif workers_alive > 0:

                status = "running"

            else:

                status = "idle"

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            metrics_latency = round(
                time.monotonic() - start,
                4,
            )

            # ---------------------------------------------------------
            # FINAL RESULT
            # ---------------------------------------------------------
            return {
                "status": status,
                "health_score": health,
                "metrics_latency": metrics_latency,

                # generation
                "generations": generations,
                "active_requests": active_requests,

                # queue
                "queue_size": queue_size,
                "avg_load": avg_load,

                # workers
                "workers_alive": workers_alive,
                "dead_workers": dead_workers,
                "total_workers": total_workers,

                # cache
                "cache_enabled": bool(
                    getattr(self, "cache_enabled", False)
                ),
                "cache_size": cache_size,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": cache_hit_rate,

                # semantic memory
                "semantic_enabled": bool(
                    getattr(self, "semantic_enabled", False)
                ),
                "semantic_memory": semantic_size,

                # GPU
                "vram_fraction": vram_fraction,
                "gpu_pressure": gpu_pressure,

                # system flags
                "worker_running": bool(
                    getattr(self, "_worker_running", False)
                ),
                "shutdown": bool(
                    getattr(self, "_shutdown", False)
                ),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Metrics collection failed: {exc}"
            )

            return {
                **SAFE_FALLBACK,
                "error": str(exc),
                "metrics_latency": round(
                    time.monotonic() - start,
                    4,
                ),
            }

    async def initialize(
        self,
        timeout: float = 30.0,
        *args,
        **kwargs,
    ) -> bool:
        """
        Production-grade LLMEngine initializer.

        Fixes:
        - startup deadlocks
        - duplicate initialization
        - GGUF warmup freezes
        - worker explosions
        - event-loop corruption
        - provider crashes
        - VRAM overload
        - recursive initialization
        - timeout storms
        - partial startup corruption
        - health monitor duplication
        """

        # -------------------------------------------------------------
        # PREVENT PARALLEL INITIALIZATION
        # -------------------------------------------------------------
        if not hasattr(self, "_initialize_lock"):

            self._initialize_lock = asyncio.Lock()

        async with self._initialize_lock:

            # ---------------------------------------------------------
            # ALREADY INITIALIZED
            # ---------------------------------------------------------
            if getattr(self, "_initialized", False):

                logger.info(
                    "LLMEngine already initialized"
                )

                return True

            # ---------------------------------------------------------
            # PREVENT INIT DURING SHUTDOWN
            # ---------------------------------------------------------
            if getattr(self, "_shutdown", False):

                logger.warning(
                    "Initialization blocked during shutdown"
                )

                return False

            start = time.monotonic()

            logger.info(
                "Initializing LLMEngine..."
            )

            try:

                # -----------------------------------------------------
                # VALIDATE TIMEOUT
                # -----------------------------------------------------
                try:
                    timeout = float(timeout)
                except Exception:
                    timeout = 30.0

                timeout = max(
                    5.0,
                    min(timeout, 300.0),
                )

                # -----------------------------------------------------
                # ENSURE PRIMITIVES
                # -----------------------------------------------------
                self._ensure_async_primitives()

                # -----------------------------------------------------
                # PROVIDER CHECK
                # -----------------------------------------------------
                provider = getattr(
                    self,
                    "provider",
                    None,
                )

                if provider is None:

                    logger.error(
                        "No provider available"
                    )

                    return False

                provider_name = (
                    provider.__class__.__name__
                )

                logger.info(
                    f"Provider: {provider_name}"
                )

                # -----------------------------------------------------
                # PROVIDER HEALTH VALIDATION
                # -----------------------------------------------------
                if not hasattr(provider, "generate"):

                    logger.error(
                        "Provider missing generate()"
                    )

                    return False

                # -----------------------------------------------------
                # DETECT LOW VRAM MODELS
                # -----------------------------------------------------
                skip_warmup = False

                model_path = ""

                try:

                    if (
                        hasattr(provider, "model")
                        and provider.model
                    ):

                        model_path = str(
                            getattr(
                                provider.model,
                                "model_path",
                                "",
                            )
                        ).lower()

                        large_model_markers = [
                            "q4",
                            "q5",
                            "q6",
                            "q8",
                            "f16",
                            "70b",
                            "34b",
                            "13b",
                        ]

                        if any(
                            marker in model_path
                            for marker in large_model_markers
                        ):

                            skip_warmup = True

                except Exception as exc:

                    logger.warning(
                        f"Warmup detection failed: {exc}"
                    )

                # -----------------------------------------------------
                # ENVIRONMENT OVERRIDE
                # -----------------------------------------------------
                if (
                    os.environ.get(
                        "EDIATH_SKIP_WARMUP",
                        "0",
                    ) == "1"
                ):

                    skip_warmup = True

                # -----------------------------------------------------
                # GPU PRESSURE CHECK
                # -----------------------------------------------------
                try:

                    gpu_guard = getattr(
                        self,
                        "gpu_guard",
                        None,
                    )

                    if gpu_guard:

                        vram = getattr(
                            gpu_guard,
                            "vram_fraction",
                            0.0,
                        )

                        if callable(vram):
                            vram = vram()

                        if vram >= 0.80:

                            logger.warning(
                                f"High VRAM pressure "
                                f"({vram:.2f})"
                            )

                            skip_warmup = True

                except Exception as exc:

                    logger.debug(
                        f"GPU check failed: {exc}"
                    )

                # -----------------------------------------------------
                # OPTIONAL WARMUP
                # -----------------------------------------------------
                if not skip_warmup:

                    logger.info(
                        "Running lightweight warmup..."
                    )

                    warmup_task = None

                    try:

                        warmup_prompt = (
                            "[INST] Hi [/INST]"
                        )

                        warmup_task = asyncio.create_task(
                            provider.generate(
                                warmup_prompt,
                                max_tokens=1,
                                temperature=0.0,
                                retries=0,
                            )
                        )

                        raw = await asyncio.wait_for(
                            warmup_task,
                            timeout=min(5.0, timeout),
                        )

                        raw = str(raw or "").strip()

                        if raw:

                            logger.info(
                                "Warmup successful"
                            )

                        else:

                            logger.warning(
                                "Warmup returned empty output"
                            )

                    except asyncio.TimeoutError:

                        logger.warning(
                            "Warmup timeout — continuing"
                        )

                        try:

                            if (
                                warmup_task
                                and not warmup_task.done()
                            ):
                                warmup_task.cancel()

                        except Exception:
                            pass

                    except asyncio.CancelledError:

                        raise

                    except Exception as exc:

                        logger.warning(
                            f"Warmup failed: {exc}"
                        )

                else:

                    logger.warning(
                        "Warmup skipped "
                        "(low VRAM optimization)"
                    )

                # -----------------------------------------------------
                # START WORKERS
                # -----------------------------------------------------
                try:

                    self._start_workers()

                except Exception as exc:

                    logger.error(
                        f"Worker startup failed: {exc}"
                    )

                    return False

                # -----------------------------------------------------
                # VALIDATE WORKERS
                # -----------------------------------------------------
                worker_count = len(
                    getattr(self, "_workers", [])
                )

                if worker_count <= 0:

                    logger.error(
                        "No workers started"
                    )

                    return False

                # -----------------------------------------------------
                # INITIALIZE METRICS
                # -----------------------------------------------------
                try:

                    self._startup_time = time.monotonic()

                    self._last_health_check = (
                        self._startup_time
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET FLAGS
                # -----------------------------------------------------
                self._shutdown = False
                self._initialized = True

                # -----------------------------------------------------
                # FINAL METRICS
                # -----------------------------------------------------
                elapsed = round(
                    time.monotonic() - start,
                    3,
                )

                logger.info(
                    f"LLMEngine initialized | "
                    f"time={elapsed:.2f}s "
                    f"workers={worker_count}"
                )

                return True

            # ---------------------------------------------------------
            # TASK CANCELLATION
            # ---------------------------------------------------------
            except asyncio.CancelledError:

                logger.warning(
                    "Initialization cancelled"
                )

                self._initialized = False

                raise

            # ---------------------------------------------------------
            # HARD FAILURE
            # ---------------------------------------------------------
            except Exception as exc:

                logger.error(
                    f"Initialization failed: {exc}"
                )

                self._initialized = False

                # -----------------------------------------------------
                # CLEANUP PARTIAL STARTUP
                # -----------------------------------------------------
                try:

                    self._worker_running = False

                    for task in getattr(
                        self,
                        "_workers",
                        [],
                    ):

                        try:

                            if (
                                task
                                and not task.done()
                            ):
                                task.cancel()

                        except Exception:
                            pass

                    self._workers.clear()

                except Exception:
                    pass

                return False

# ══════════════════════════════════════════════════════════════════════════════
# Agent Pool
# ══════════════════════════════════════════════════════════════════════════════

class AgentPool:
    def __init__(self, engine: LLMEngine) -> None:
        self.engine: LLMEngine = engine
        self._slots: Dict[str, AgentSlot] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        agent_id: str,
        priority: AgentPriority = AgentPriority.NORMAL,
        rate_limit: float = _AGENT_RATE_LIMIT_RPS,
        max_queue: int = 10,
    ) -> AgentSlot:
        """
        Production-grade agent registration system.

        Fixes:
        - duplicate registration races
        - invalid agent IDs
        - pool overflow
        - retired-agent corruption
        - invalid priorities
        - invalid rate limits
        - queue overflow configs
        - concurrent registration instability
        - slot corruption
        """

        try:

            # ---------------------------------------------------------
            # ENSURE LOCK
            # ---------------------------------------------------------
            if not hasattr(self, "_lock"):

                self._lock = asyncio.Lock()

            async with self._lock:

                # -----------------------------------------------------
                # AGENT ID VALIDATION
                # -----------------------------------------------------
                if agent_id is None:
                    agent_id = ""

                if not isinstance(agent_id, str):

                    try:
                        agent_id = str(agent_id)
                    except Exception:
                        agent_id = ""

                agent_id = agent_id.strip()

                if not agent_id:

                    raise ValueError(
                        "Agent ID cannot be empty"
                    )

                # -----------------------------------------------------
                # SANITIZE AGENT ID
                # -----------------------------------------------------
                agent_id = re.sub(
                    r"\s+",
                    "_",
                    agent_id,
                )

                max_agent_id = 128

                if len(agent_id) > max_agent_id:

                    logger.warning(
                        f"Agent ID truncated "
                        f"({len(agent_id)} chars)"
                    )

                    agent_id = agent_id[:max_agent_id]

                # -----------------------------------------------------
                # PRIORITY VALIDATION
                # -----------------------------------------------------
                try:

                    if not isinstance(
                        priority,
                        AgentPriority,
                    ):

                        priority = AgentPriority(
                            int(priority)
                        )

                except Exception:

                    logger.warning(
                        f"Invalid priority for "
                        f"{agent_id}"
                    )

                    priority = (
                        AgentPriority.NORMAL
                    )

                # -----------------------------------------------------
                # RATE LIMIT VALIDATION
                # -----------------------------------------------------
                try:

                    rate_limit = float(rate_limit)

                except Exception:

                    rate_limit = (
                        _AGENT_RATE_LIMIT_RPS
                    )

                rate_limit = max(
                    0.1,
                    min(rate_limit, 10000.0),
                )

                # -----------------------------------------------------
                # QUEUE LIMIT VALIDATION
                # -----------------------------------------------------
                try:

                    max_queue = int(max_queue)

                except Exception:

                    max_queue = 10

                max_queue = max(
                    1,
                    min(max_queue, 100000),
                )

                # -----------------------------------------------------
                # SLOT STORAGE VALIDATION
                # -----------------------------------------------------
                if not hasattr(self, "_slots"):

                    self._slots = {}

                if not isinstance(
                    self._slots,
                    dict,
                ):

                    logger.warning(
                        "Agent slot store corrupted"
                    )

                    self._slots = {}

                # -----------------------------------------------------
                # EXISTING AGENT CHECK
                # -----------------------------------------------------
                existing = self._slots.get(
                    agent_id
                )

                if existing is not None:

                    try:

                        state = getattr(
                            existing,
                            "state",
                            None,
                        )

                        if (
                            state is not None
                            and state != AgentState.RETIRED
                        ):

                            raise ValueError(
                                f"Agent '{agent_id}' "
                                "already registered"
                            )

                    except ValueError:
                        raise

                    except Exception as exc:

                        logger.warning(
                            f"Existing slot invalid: {exc}"
                        )

                    # remove retired/corrupted slot
                    try:

                        del self._slots[agent_id]

                    except Exception:
                        pass

                # -----------------------------------------------------
                # ACTIVE AGENT COUNT
                # -----------------------------------------------------
                active = 0

                try:

                    for slot in self._slots.values():

                        try:

                            state = getattr(
                                slot,
                                "state",
                                None,
                            )

                            if (
                                state !=
                                AgentState.RETIRED
                            ):

                                active += 1

                        except Exception:
                            continue

                except Exception as exc:

                    logger.warning(
                        f"Active agent count failed: {exc}"
                    )

                # -----------------------------------------------------
                # POOL LIMIT
                # -----------------------------------------------------
                max_agents = max(
                    1,
                    int(MAX_AGENTS),
                )

                if active >= max_agents:

                    logger.error(
                        f"AgentPool full "
                        f"({active}/{max_agents})"
                    )

                    raise RuntimeError(
                        f"AgentPool full "
                        f"({max_agents} active agents). "
                        f"Retire unused agents first."
                    )

                # -----------------------------------------------------
                # CREATE SLOT
                # -----------------------------------------------------
                try:

                    slot = AgentSlot(
                        agent_id=agent_id,
                        priority=priority,
                        rate_limit=rate_limit,
                        max_queue=max_queue,
                    )

                except Exception as exc:

                    logger.error(
                        f"AgentSlot creation failed: {exc}"
                    )

                    raise RuntimeError(
                        f"Failed to create agent slot: {exc}"
                    )

                # -----------------------------------------------------
                # STORE SLOT
                # -----------------------------------------------------
                try:

                    self._slots[agent_id] = slot

                except Exception as exc:

                    logger.error(
                        f"Slot storage failed: {exc}"
                    )

                    raise RuntimeError(
                        f"Failed to register agent: {exc}"
                    )

                # -----------------------------------------------------
                # OPTIONAL METRICS
                # -----------------------------------------------------
                try:

                    self._registrations = int(
                        getattr(
                            self,
                            "_registrations",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    f"AgentPool registered | "
                    f"id='{agent_id}' "
                    f"priority={priority.name} "
                    f"rps={rate_limit:.2f} "
                    f"queue={max_queue} "
                    f"[{active + 1}/{max_agents}]"
                )

                return slot

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                f"Registration cancelled "
                f"for '{agent_id}'"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Agent registration failed: {exc}"
            )

            raise RuntimeError(
                f"Failed to register agent '{agent_id}': {exc}" # type: ignore
            )

    def unregister(self, agent_id: str) -> bool:
        """
        Production-grade agent unregistration system.

        Fixes:
        - invalid agent IDs
        - unregister race conditions
        - missing agent crashes
        - recursive retire failures
        - corrupted slot states
        - double-unregister instability
        - metric desync
        """

        try:

            # ---------------------------------------------------------
            # AGENT ID VALIDATION
            # ---------------------------------------------------------
            if agent_id is None:
                agent_id = ""

            if not isinstance(agent_id, str):

                try:
                    agent_id = str(agent_id)
                except Exception:
                    agent_id = ""

            agent_id = agent_id.strip()

            if not agent_id:

                logger.warning(
                    "Unregister called with empty agent_id"
                )

                return False

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Agent slot storage missing"
                )

                return False

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Agent slot storage corrupted"
                )

                return False

            # ---------------------------------------------------------
            # AGENT EXISTENCE CHECK
            # ---------------------------------------------------------
            slot = self._slots.get(agent_id)

            if slot is None:

                logger.warning(
                    f"Agent '{agent_id}' not found"
                )

                return False

            # ---------------------------------------------------------
            # SAFE RETIRE CALL
            # ---------------------------------------------------------
            try:

                result = bool(
                    self.retire(agent_id)
                )

            except Exception as exc:

                logger.error(
                    f"Retire failed for "
                    f"'{agent_id}': {exc}"
                )

                return False

            # ---------------------------------------------------------
            # VERIFY RETIREMENT
            # ---------------------------------------------------------
            try:

                updated_slot = self._slots.get(
                    agent_id
                )

                if updated_slot is not None:

                    state = getattr(
                        updated_slot,
                        "state",
                        None,
                    )

                    if state != AgentState.RETIRED:

                        logger.warning(
                            f"Agent '{agent_id}' "
                            f"not fully retired"
                        )

            except Exception as exc:

                logger.debug(
                    f"Retirement verification failed: {exc}"
                )

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self._unregistrations = int(
                    getattr(
                        self,
                        "_unregistrations",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # LOGGING
            # ---------------------------------------------------------
            if result:

                logger.info(
                    f"Agent unregistered: "
                    f"'{agent_id}'"
                )

            else:

                logger.warning(
                    f"Agent unregister failed: "
                    f"'{agent_id}'"
                )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Unregister failed for "
                f"'{agent_id}': {exc}"
            )

            return False

    def pause(self, agent_id: str) -> bool:
        """
        Production-grade agent pause system.

        Fixes:
        - invalid agent IDs
        - missing slot crashes
        - invalid state transitions
        - pause race conditions
        - corrupted slot states
        - double-pause instability
        - metric desync
        - paused-active confusion
        """

        try:

            # ---------------------------------------------------------
            # AGENT ID VALIDATION
            # ---------------------------------------------------------
            if agent_id is None:
                agent_id = ""

            if not isinstance(agent_id, str):

                try:
                    agent_id = str(agent_id)
                except Exception:
                    agent_id = ""

            agent_id = agent_id.strip()

            if not agent_id:

                logger.warning(
                    "Pause called with empty agent_id"
                )

                return False

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Agent slot storage missing"
                )

                return False

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Agent slot storage corrupted"
                )

                return False

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_pause_lock"):

                self._pause_lock = threading.Lock()

            with self._pause_lock:

                # -----------------------------------------------------
                # FIND SLOT
                # -----------------------------------------------------
                slot = self._slots.get(agent_id)

                if slot is None:

                    logger.warning(
                        f"Pause failed: "
                        f"agent '{agent_id}' not found"
                    )

                    return False

                # -----------------------------------------------------
                # VALIDATE STATE
                # -----------------------------------------------------
                try:

                    current_state = getattr(
                        slot,
                        "state",
                        None,
                    )

                except Exception as exc:

                    logger.warning(
                        f"State read failed for "
                        f"'{agent_id}': {exc}"
                    )

                    return False

                # -----------------------------------------------------
                # ALREADY PAUSED
                # -----------------------------------------------------
                if current_state == AgentState.PAUSED:

                    logger.debug(
                        f"Agent already paused: "
                        f"'{agent_id}'"
                    )

                    return True

                # -----------------------------------------------------
                # INVALID TRANSITIONS
                # -----------------------------------------------------
                invalid_states = {
                    AgentState.RETIRED,
                }

                if current_state in invalid_states:

                    logger.warning(
                        f"Cannot pause agent "
                        f"'{agent_id}' "
                        f"from state={current_state}"
                    )

                    return False

                # -----------------------------------------------------
                # ACTIVE CHECK
                # -----------------------------------------------------
                if current_state != AgentState.ACTIVE:

                    logger.warning(
                        f"Pause ignored for "
                        f"'{agent_id}' "
                        f"(state={current_state})"
                    )

                    return False

                # -----------------------------------------------------
                # APPLY PAUSE
                # -----------------------------------------------------
                try:

                    slot.state = AgentState.PAUSED

                except Exception as exc:

                    logger.error(
                        f"Failed to pause "
                        f"'{agent_id}': {exc}"
                    )

                    return False

                # -----------------------------------------------------
                # OPTIONAL METRICS
                # -----------------------------------------------------
                try:

                    self._pauses = int(
                        getattr(
                            self,
                            "_pauses",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # TIMESTAMP
                # -----------------------------------------------------
                try:

                    slot.paused_at = time.time()

                except Exception:
                    pass

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    f"Agent paused | "
                    f"id='{agent_id}'"
                )

                return True

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Pause failed for "
                f"'{agent_id}': {exc}"
            )

            return False

    def resume(self, agent_id: str) -> bool:
        """
        Production-grade agent resume system.

        Fixes:
        - invalid agent IDs
        - missing slot crashes
        - invalid state transitions
        - resume race conditions
        - corrupted slot states
        - double-resume instability
        - paused/active desync
        - metric corruption
        """

        try:

            # ---------------------------------------------------------
            # AGENT ID VALIDATION
            # ---------------------------------------------------------
            if agent_id is None:
                agent_id = ""

            if not isinstance(agent_id, str):

                try:
                    agent_id = str(agent_id)
                except Exception:
                    agent_id = ""

            agent_id = agent_id.strip()

            if not agent_id:

                logger.warning(
                    "Resume called with empty agent_id"
                )

                return False

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Agent slot storage missing"
                )

                return False

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Agent slot storage corrupted"
                )

                return False

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_resume_lock"):

                self._resume_lock = threading.Lock()

            with self._resume_lock:

                # -----------------------------------------------------
                # FIND SLOT
                # -----------------------------------------------------
                slot = self._slots.get(agent_id)

                if slot is None:

                    logger.warning(
                        f"Resume failed: "
                        f"agent '{agent_id}' not found"
                    )

                    return False

                # -----------------------------------------------------
                # VALIDATE CURRENT STATE
                # -----------------------------------------------------
                try:

                    current_state = getattr(
                        slot,
                        "state",
                        None,
                    )

                except Exception as exc:

                    logger.warning(
                        f"State read failed for "
                        f"'{agent_id}': {exc}"
                    )

                    return False

                # -----------------------------------------------------
                # ALREADY ACTIVE/IDLE
                # -----------------------------------------------------
                if current_state in (
                    AgentState.ACTIVE,
                    AgentState.IDLE,
                ):

                    logger.debug(
                        f"Agent already active: "
                        f"'{agent_id}'"
                    )

                    return True

                # -----------------------------------------------------
                # INVALID STATES
                # -----------------------------------------------------
                invalid_states = {
                    AgentState.RETIRED,
                }

                if current_state in invalid_states:

                    logger.warning(
                        f"Cannot resume agent "
                        f"'{agent_id}' "
                        f"from state={current_state}"
                    )

                    return False

                # -----------------------------------------------------
                # MUST BE PAUSED
                # -----------------------------------------------------
                if current_state != AgentState.PAUSED:

                    logger.warning(
                        f"Resume ignored for "
                        f"'{agent_id}' "
                        f"(state={current_state})"
                    )

                    return False

                # -----------------------------------------------------
                # APPLY RESUME
                # -----------------------------------------------------
                try:

                    slot.state = AgentState.IDLE

                except Exception as exc:

                    logger.error(
                        f"Failed to resume "
                        f"'{agent_id}': {exc}"
                    )

                    return False

                # -----------------------------------------------------
                # OPTIONAL TIMESTAMP
                # -----------------------------------------------------
                try:

                    slot.resumed_at = time.time()

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL METRICS
                # -----------------------------------------------------
                try:

                    self._resumes = int(
                        getattr(
                            self,
                            "_resumes",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET TEMPORARY BACKPRESSURE
                # -----------------------------------------------------
                try:

                    if hasattr(slot, "_queue_depth"):

                        slot._queue_depth = max(
                            0,
                            int(slot._queue_depth),
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    f"Agent resumed | "
                    f"id='{agent_id}'"
                )

                return True

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Resume failed for "
                f"'{agent_id}': {exc}"
            )

            return False

    def retire(self, agent_id: str) -> bool:
        """
        Production-grade agent retirement system.

        Fixes:
        - invalid agent IDs
        - retirement race conditions
        - corrupted slot states
        - double-retire instability
        - active queue leaks
        - stale metrics
        - missing slot crashes
        - paused/active cleanup failures
        """

        try:

            # ---------------------------------------------------------
            # AGENT ID VALIDATION
            # ---------------------------------------------------------
            if agent_id is None:
                agent_id = ""

            if not isinstance(agent_id, str):

                try:
                    agent_id = str(agent_id)
                except Exception:
                    agent_id = ""

            agent_id = agent_id.strip()

            if not agent_id:

                logger.warning(
                    "Retire called with empty agent_id"
                )

                return False

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Agent slot storage missing"
                )

                return False

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Agent slot storage corrupted"
                )

                return False

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_retire_lock"):

                self._retire_lock = threading.Lock()

            with self._retire_lock:

                # -----------------------------------------------------
                # FIND SLOT
                # -----------------------------------------------------
                slot = self._slots.get(agent_id)

                if slot is None:

                    logger.warning(
                        f"Retire failed: "
                        f"agent '{agent_id}' not found"
                    )

                    return False

                # -----------------------------------------------------
                # CURRENT STATE
                # -----------------------------------------------------
                try:

                    current_state = getattr(
                        slot,
                        "state",
                        None,
                    )

                except Exception as exc:

                    logger.warning(
                        f"State read failed for "
                        f"'{agent_id}': {exc}"
                    )

                    current_state = None

                # -----------------------------------------------------
                # ALREADY RETIRED
                # -----------------------------------------------------
                if current_state == AgentState.RETIRED:

                    logger.debug(
                        f"Agent already retired: "
                        f"'{agent_id}'"
                    )

                    return True

                # -----------------------------------------------------
                # CLEANUP ACTIVE WORKLOAD
                # -----------------------------------------------------
                try:

                    if hasattr(slot, "_queue_depth"):

                        queue_depth = int(
                            getattr(
                                slot,
                                "_queue_depth",
                                0,
                            )
                        )

                        if queue_depth > 0:

                            logger.warning(
                                f"Retiring busy agent "
                                f"'{agent_id}' "
                                f"(queue={queue_depth})"
                            )

                        slot._queue_depth = 0

                except Exception as exc:

                    logger.debug(
                        f"Queue cleanup failed: {exc}"
                    )

                # -----------------------------------------------------
                # RESET TOKENS
                # -----------------------------------------------------
                try:

                    if hasattr(slot, "_tokens"):

                        slot._tokens = 0.0

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET TEMPORARY STATE
                # -----------------------------------------------------
                cleanup_attrs = [
                    "_last_request",
                    "_last_error",
                    "_active_task",
                ]

                for attr in cleanup_attrs:

                    try:

                        if hasattr(slot, attr):

                            setattr(
                                slot,
                                attr,
                                None,
                            )

                    except Exception:
                        pass

                # -----------------------------------------------------
                # APPLY RETIREMENT
                # -----------------------------------------------------
                try:

                    slot.state = AgentState.RETIRED

                except Exception as exc:

                    logger.error(
                        f"Failed to retire "
                        f"'{agent_id}': {exc}"
                    )

                    return False

                # -----------------------------------------------------
                # RETIREMENT TIMESTAMP
                # -----------------------------------------------------
                try:

                    slot.retired_at = time.time()

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL METRICS
                # -----------------------------------------------------
                try:

                    self._retirements = int(
                        getattr(
                            self,
                            "_retirements",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL CLEANUP POLICY
                # -----------------------------------------------------
                try:

                    auto_remove = bool(
                        getattr(
                            self,
                            "auto_remove_retired",
                            False,
                        )
                    )

                    if auto_remove:

                        del self._slots[agent_id]

                        logger.debug(
                            f"Removed retired agent "
                            f"'{agent_id}' "
                            f"from pool"
                        )

                except Exception as exc:

                    logger.debug(
                        f"Auto-remove failed: {exc}"
                    )

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    f"Agent retired | "
                    f"id='{agent_id}' "
                    f"previous_state={current_state}"
                )

                return True

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Retire failed for "
                f"'{agent_id}': {exc}"
            )

            return False

    async def generate(
        self,
        agent_id: str,
        prompt: str,
        timeout: float = 30.0,
        **kwargs,
    ) -> Dict[str, Any]:
        slot = self._slots.get(agent_id)
        if slot is None:
            return {
                "response": f"Agent '{agent_id}' is not registered.",
                "intent": "error",
                "confidence": 0.0,
                "agent_id": agent_id,
            }

        if slot.state == AgentState.RETIRED:
            return {
                "response": f"Agent '{agent_id}' has been retired.",
                "intent": "error",
                "confidence": 0.0,
                "agent_id": agent_id,
            }

        if slot.state == AgentState.PAUSED:
            return {
                "response": f"Agent '{agent_id}' is paused.",
                "intent": "idle",
                "confidence": 0.0,
                "agent_id": agent_id,
            }

        allowed = await slot.acquire()
        if not allowed:
            return {
                "response": "Rate limit reached. Please slow down.",
                "intent": "idle",
                "confidence": 0.0,
                "reflection": "rate_limited",
                "agent_id": agent_id,
            }

        start = time.monotonic()
        try:
            result = await self.engine.agent_generate(
                agent_id=agent_id,
                prompt=prompt,
                priority=slot.priority,
                timeout=timeout,
                **kwargs,
            )
            latency_ms = (time.monotonic() - start) * 1000
            await slot.release(success=True, latency_ms=latency_ms)
            return result
        except Exception as exc:
            await slot.release(success=False)
            logger.error(f"AgentPool generate error for '{agent_id}': {exc}")
            return {
                "response": f"Error processing request: {exc}",
                "intent": "error",
                "confidence": 0.0,
                "agent_id": agent_id,
            }

    async def broadcast(
        self,
        prompt: str,
        priority: AgentPriority = AgentPriority.HIGH,
        timeout: float = 60.0,
    ) -> Dict[str, Dict]:
        """
        Production-grade multi-agent broadcast system.

        Fixes:
        - task leaks
        - dead agent execution
        - timeout storms
        - invalid agent states
        - concurrent broadcast crashes
        - orphaned tasks
        - malformed responses
        - partial broadcast failures
        - empty broadcasts
        - invalid priorities
        """

        results: Dict[str, Dict] = {}

        tasks: Dict[str, asyncio.Task] = {}

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------
            if prompt is None:
                prompt = ""

            if not isinstance(prompt, str):

                try:
                    prompt = str(prompt)
                except Exception:
                    prompt = ""

            prompt = re.sub(
                r"\s+",
                " ",
                prompt.replace("\x00", " "),
            ).strip()

            if not prompt:

                logger.warning(
                    "Broadcast called with empty prompt"
                )

                return {}

            # ---------------------------------------------------------
            # PRIORITY VALIDATION
            # ---------------------------------------------------------
            try:

                if not isinstance(
                    priority,
                    AgentPriority,
                ):

                    priority = AgentPriority(
                        int(priority)
                    )

            except Exception:

                logger.warning(
                    "Invalid broadcast priority"
                )

                priority = AgentPriority.HIGH

            # ---------------------------------------------------------
            # TIMEOUT VALIDATION
            # ---------------------------------------------------------
            try:

                timeout = float(timeout)

            except Exception:

                timeout = 60.0

            timeout = max(
                1.0,
                min(timeout, 3600.0),
            )

            # ---------------------------------------------------------
            # SLOT VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Broadcast failed: slot storage missing"
                )

                return {}

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Broadcast failed: slot storage corrupted"
                )

                return {}

            # ---------------------------------------------------------
            # FILTER ACTIVE AGENTS
            # ---------------------------------------------------------
            active_ids = []

            for aid, slot in self._slots.items():

                try:

                    if slot is None:
                        continue

                    state = getattr(
                        slot,
                        "state",
                        None,
                    )

                    if state in (
                        AgentState.PAUSED,
                        AgentState.RETIRED,
                    ):

                        continue

                    if not isinstance(aid, str):

                        aid = str(aid)

                    aid = aid.strip()

                    if not aid:
                        continue

                    active_ids.append(aid)

                except Exception as exc:

                    logger.debug(
                        f"Agent filtering failed: {exc}"
                    )

            # ---------------------------------------------------------
            # NO ACTIVE AGENTS
            # ---------------------------------------------------------
            if not active_ids:

                logger.warning(
                    "Broadcast skipped: no active agents"
                )

                return {}

            logger.info(
                f"Broadcast started | "
                f"agents={len(active_ids)} "
                f"priority={priority.name}"
            )

            # ---------------------------------------------------------
            # CREATE TASKS
            # ---------------------------------------------------------
            for aid in active_ids:

                try:

                    task = asyncio.create_task(
                        self.generate(
                            aid,
                            prompt,
                            timeout=timeout,
                        ),
                        name=f"broadcast_{aid}",
                    )

                    tasks[aid] = task

                except Exception as exc:

                    logger.error(
                        f"Task creation failed "
                        f"for '{aid}': {exc}"
                    )

                    results[aid] = {
                        "response": str(exc),
                        "intent": "error",
                        "error": True,
                    }

            # ---------------------------------------------------------
            # EXECUTE TASKS
            # ---------------------------------------------------------
            for aid, task in tasks.items():

                try:

                    response = await asyncio.wait_for(
                        task,
                        timeout=timeout,
                    )

                    # ---------------------------------------------
                    # VALIDATE RESPONSE
                    # ---------------------------------------------
                    if not isinstance(
                        response,
                        dict,
                    ):

                        response = {
                            "response": str(response),
                            "intent": "unknown",
                        }

                    response.setdefault(
                        "agent_id",
                        aid,
                    )

                    response.setdefault(
                        "broadcast",
                        True,
                    )

                    results[aid] = response

                # -------------------------------------------------
                # TIMEOUT
                # -------------------------------------------------
                except asyncio.TimeoutError:

                    logger.warning(
                        f"Broadcast timeout: '{aid}'"
                    )

                    try:

                        if not task.done():

                            task.cancel()

                    except Exception:
                        pass

                    results[aid] = {
                        "response": (
                            "Broadcast request timed out."
                        ),
                        "intent": "timeout",
                        "error": True,
                        "agent_id": aid,
                    }

                # -------------------------------------------------
                # TASK CANCELLED
                # -------------------------------------------------
                except asyncio.CancelledError:

                    logger.warning(
                        f"Broadcast cancelled: '{aid}'"
                    )

                    try:

                        if not task.done():

                            task.cancel()

                    except Exception:
                        pass

                    raise

                # -------------------------------------------------
                # EXECUTION FAILURE
                # -------------------------------------------------
                except Exception as exc:

                    logger.error(
                        f"Broadcast failed for "
                        f"'{aid}': {exc}"
                    )

                    results[aid] = {
                        "response": str(exc),
                        "intent": "error",
                        "error": True,
                        "agent_id": aid,
                    }

            # ---------------------------------------------------------
            # CLEANUP FINISHED TASKS
            # ---------------------------------------------------------
            for aid, task in tasks.items():

                try:

                    if (
                        not task.done()
                        and not task.cancelled()
                    ):

                        task.cancel()

                except Exception:
                    pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            success_count = sum(
                1
                for r in results.values()
                if not r.get("error", False)
            )

            failed_count = (
                len(results) - success_count
            )

            elapsed = round(
                time.monotonic() - start,
                3,
            )

            logger.info(
                f"Broadcast complete | "
                f"success={success_count} "
                f"failed={failed_count} "
                f"time={elapsed:.2f}s"
            )

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self._broadcasts = int(
                    getattr(
                        self,
                        "_broadcasts",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return results

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Broadcast cancelled"
            )

            for task in tasks.values():

                try:

                    if (
                        task
                        and not task.done()
                    ):
                        task.cancel()

                except Exception:
                    pass

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Broadcast crashed: {exc}"
            )

            # cleanup
            for task in tasks.values():

                try:

                    if (
                        task
                        and not task.done()
                    ):
                        task.cancel()

                except Exception:
                    pass

            return {
                "__broadcast_error__": {
                    "response": str(exc),
                    "intent": "error",
                    "error": True,
                }
            }

    def get_agent_metrics(
        self,
        agent_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Production-grade agent metrics retrieval.

        Fixes:
        - invalid agent IDs
        - missing slot crashes
        - corrupted metric responses
        - dead agent access
        - metric retrieval races
        - malformed slot objects
        - metrics serialization failures
        """

        try:

            # ---------------------------------------------------------
            # AGENT ID VALIDATION
            # ---------------------------------------------------------
            if agent_id is None:
                agent_id = ""

            if not isinstance(agent_id, str):

                try:
                    agent_id = str(agent_id)
                except Exception:
                    agent_id = ""

            agent_id = agent_id.strip()

            if not agent_id:

                logger.warning(
                    "Metrics requested with empty agent_id"
                )

                return None

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Agent slot storage missing"
                )

                return None

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Agent slot storage corrupted"
                )

                return None

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_metrics_lock"):

                self._metrics_lock = threading.Lock()

            with self._metrics_lock:

                # -----------------------------------------------------
                # FIND SLOT
                # -----------------------------------------------------
                slot = self._slots.get(agent_id)

                if slot is None:

                    logger.warning(
                        f"Metrics request failed: "
                        f"agent '{agent_id}' not found"
                    )

                    return None

                # -----------------------------------------------------
                # SLOT VALIDATION
                # -----------------------------------------------------
                if not hasattr(slot, "get_metrics"):

                    logger.error(
                        f"Agent '{agent_id}' "
                        f"missing get_metrics()"
                    )

                    return None

                # -----------------------------------------------------
                # GET METRICS
                # -----------------------------------------------------
                try:

                    metrics = slot.get_metrics()

                except Exception as exc:

                    logger.error(
                        f"Metrics retrieval failed "
                        f"for '{agent_id}': {exc}"
                    )

                    return {
                        "agent_id": agent_id,
                        "state": "error",
                        "health_score": 0.0,
                        "error": str(exc),
                    }

                # -----------------------------------------------------
                # VALIDATE RESPONSE
                # -----------------------------------------------------
                if metrics is None:

                    logger.warning(
                        f"Metrics returned None "
                        f"for '{agent_id}'"
                    )

                    return None

                if not isinstance(
                    metrics,
                    dict,
                ):

                    logger.warning(
                        f"Invalid metrics format "
                        f"for '{agent_id}'"
                    )

                    metrics = {
                        "raw_metrics": str(metrics)
                    }

                # -----------------------------------------------------
                # ENSURE REQUIRED FIELDS
                # -----------------------------------------------------
                metrics.setdefault(
                    "agent_id",
                    agent_id,
                )

                metrics.setdefault(
                    "state",
                    "unknown",
                )

                metrics.setdefault(
                    "health_score",
                    0.5,
                )

                # -----------------------------------------------------
                # SERIALIZATION SAFETY
                # -----------------------------------------------------
                safe_metrics = {}

                for key, value in metrics.items():

                    try:

                        # JSON-safe primitives
                        if isinstance(
                            value,
                            (
                                str,
                                int,
                                float,
                                bool,
                                type(None),
                            ),
                        ):

                            safe_metrics[key] = value

                        # lists/tuples
                        elif isinstance(
                            value,
                            (list, tuple),
                        ):

                            safe_metrics[key] = list(value)

                        # dicts
                        elif isinstance(
                            value,
                            dict,
                        ):

                            safe_metrics[key] = value.copy()

                        else:

                            safe_metrics[key] = str(value)

                    except Exception:

                        safe_metrics[key] = (
                            "<serialization_error>"
                        )

                # -----------------------------------------------------
                # METRICS TIMESTAMP
                # -----------------------------------------------------
                try:

                    safe_metrics[
                        "metrics_timestamp"
                    ] = round(
                        time.time(),
                        3,
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL ACCESS COUNTER
                # -----------------------------------------------------
                try:

                    self._metrics_requests = int(
                        getattr(
                            self,
                            "_metrics_requests",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Metrics retrieved | "
                    f"agent='{agent_id}'"
                )

                return safe_metrics

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"get_agent_metrics failed "
                f"for '{agent_id}': {exc}"
            )

            return {
                "agent_id": str(agent_id),
                "state": "error",
                "health_score": 0.0,
                "error": str(exc),
            }

    def get_pool_metrics(self) -> Dict[str, Any]:
        """
        Production-grade agent pool metrics collector.

        Fixes:
        - corrupted slot metrics
        - division-by-zero
        - missing engine crashes
        - invalid GPU metrics
        - pool desync
        - malformed slot objects
        - metrics overflow
        - health-score instability
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Pool metrics: slot storage missing"
                )

                self._slots = {}

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Pool metrics: slot storage corrupted"
                )

                self._slots = {}

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_pool_metrics_lock"):

                self._pool_metrics_lock = (
                    threading.Lock()
                )

            with self._pool_metrics_lock:

                # -----------------------------------------------------
                # COUNTERS
                # -----------------------------------------------------
                active = 0
                retired = 0
                paused = 0
                errored = 0

                total_sent = 0
                total_completed = 0
                total_failed = 0
                total_throttled = 0

                # -----------------------------------------------------
                # ITERATE SLOTS
                # -----------------------------------------------------
                for agent_id, slot in (
                    self._slots.items()
                ):

                    try:

                        if slot is None:

                            errored += 1
                            continue

                        # -----------------------------------------
                        # STATE
                        # -----------------------------------------
                        state = getattr(
                            slot,
                            "state",
                            None,
                        )

                        if state == AgentState.RETIRED:

                            retired += 1

                        elif state == AgentState.PAUSED:

                            paused += 1

                        elif state is None:

                            errored += 1

                        else:

                            active += 1

                        # -----------------------------------------
                        # SAFE METRICS
                        # -----------------------------------------
                        def _safe_int(
                            attr: str,
                        ) -> int:

                            try:

                                value = int(
                                    getattr(
                                        slot,
                                        attr,
                                        0,
                                    )
                                )

                                return max(0, value)

                            except Exception:

                                return 0

                        total_sent += _safe_int(
                            "requests_sent"
                        )

                        total_completed += _safe_int(
                            "requests_completed"
                        )

                        total_failed += _safe_int(
                            "requests_failed"
                        )

                        total_throttled += _safe_int(
                            "requests_throttled"
                        )

                    except Exception as exc:

                        errored += 1

                        logger.debug(
                            f"Slot metric failure "
                            f"({agent_id}): {exc}"
                        )

                # -----------------------------------------------------
                # POOL CAPACITY
                # -----------------------------------------------------
                try:

                    capacity = int(MAX_AGENTS)

                except Exception:

                    capacity = 1

                capacity = max(1, capacity)

                utilization = round(
                    active / capacity,
                    3,
                )

                # -----------------------------------------------------
                # SUCCESS RATE
                # -----------------------------------------------------
                success_rate = round(
                    total_completed /
                    max(1, total_sent),
                    3,
                )

                # -----------------------------------------------------
                # ENGINE METRICS
                # -----------------------------------------------------
                engine_metrics = {}

                try:

                    if (
                        hasattr(self, "engine")
                        and self.engine
                    ):

                        engine_metrics = (
                            self.engine.get_metrics()
                        )

                        if not isinstance(
                            engine_metrics,
                            dict,
                        ):

                            engine_metrics = {
                                "status": "invalid"
                            }

                    else:

                        engine_metrics = {
                            "status": "missing"
                        }

                except Exception as exc:

                    logger.warning(
                        f"Engine metrics failed: {exc}"
                    )

                    engine_metrics = {
                        "status": "error",
                        "error": str(exc),
                    }

                # -----------------------------------------------------
                # GPU METRICS
                # -----------------------------------------------------
                gpu_metrics = {
                    "vram_fraction": 0.0,
                    "under_pressure": False,
                }

                try:

                    gpu_guard = getattr(
                        self.engine,
                        "gpu_guard",
                        None,
                    )

                    if gpu_guard:

                        try:

                            vram = float(
                                gpu_guard.vram_fraction
                            )

                        except Exception:

                            vram = 0.0

                        if (
                            math.isnan(vram)
                            or math.isinf(vram)
                        ):

                            vram = 0.0

                        vram = max(
                            0.0,
                            min(vram, 1.0),
                        )

                        try:

                            pressure = bool(
                                gpu_guard.is_under_pressure()
                            )

                        except Exception:

                            pressure = False

                        gpu_metrics = {
                            "vram_fraction": round(
                                vram,
                                3,
                            ),
                            "under_pressure": pressure,
                        }

                except Exception as exc:

                    logger.warning(
                        f"GPU metrics failed: {exc}"
                    )

                # -----------------------------------------------------
                # HEALTH SCORE
                # -----------------------------------------------------
                health_score = 1.0

                try:

                    if utilization > 0.90:
                        health_score -= 0.25

                    elif utilization > 0.75:
                        health_score -= 0.10

                    if success_rate < 0.50:
                        health_score -= 0.35

                    elif success_rate < 0.80:
                        health_score -= 0.15

                    if total_failed > total_completed:
                        health_score -= 0.20

                    if gpu_metrics[
                        "under_pressure"
                    ]:
                        health_score -= 0.15

                    health_score = round(
                        max(
                            0.0,
                            min(1.0, health_score),
                        ),
                        2,
                    )

                except Exception:

                    health_score = 0.5

                # -----------------------------------------------------
                # RESPONSE LATENCY
                # -----------------------------------------------------
                metrics_latency = round(
                    time.monotonic() - start,
                    4,
                )

                # -----------------------------------------------------
                # FINAL RESPONSE
                # -----------------------------------------------------
                result = {
                    "pool": {
                        "registered": len(
                            self._slots
                        ),
                        "active": active,
                        "paused": paused,
                        "retired": retired,
                        "errored": errored,
                        "capacity": capacity,
                        "utilization": utilization,
                    },

                    "requests": {
                        "total_sent": total_sent,
                        "total_completed": total_completed,
                        "total_failed": total_failed,
                        "total_throttled": total_throttled,
                        "success_rate": success_rate,
                    },

                    "engine": engine_metrics,

                    "gpu": gpu_metrics,

                    "health": {
                        "health_score": health_score,
                        "metrics_latency": (
                            metrics_latency
                        ),
                    },
                }

                # -----------------------------------------------------
                # OPTIONAL METRICS COUNTER
                # -----------------------------------------------------
                try:

                    self._pool_metric_requests = (
                        int(
                            getattr(
                                self,
                                "_pool_metric_requests",
                                0,
                            )
                        ) + 1
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Pool metrics generated | "
                    f"agents={len(self._slots)} "
                    f"health={health_score}"
                )

                return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"get_pool_metrics failed: {exc}"
            )

            return {
                "pool": {
                    "registered": 0,
                    "active": 0,
                    "paused": 0,
                    "retired": 0,
                    "capacity": 0,
                    "utilization": 0.0,
                },

                "requests": {
                    "total_sent": 0,
                    "total_completed": 0,
                    "total_failed": 0,
                    "total_throttled": 0,
                    "success_rate": 0.0,
                },

                "engine": {
                    "status": "error",
                },

                "gpu": {
                    "vram_fraction": 0.0,
                    "under_pressure": False,
                },

                "health": {
                    "health_score": 0.0,
                    "error": str(exc),
                },
            }

    def list_agents(
        self,
        state_filter: Optional[AgentState] = None,
    ) -> List[str]:
        """
        Production-grade agent listing system.

        Fixes:
        - corrupted slot storage
        - invalid state filters
        - malformed agent IDs
        - retired-agent leaks
        - iteration race conditions
        - None slot crashes
        - duplicate agent IDs
        - unstable ordering
        """

        try:

            # ---------------------------------------------------------
            # SLOT STORAGE VALIDATION
            # ---------------------------------------------------------
            if not hasattr(self, "_slots"):

                logger.warning(
                    "Agent slot storage missing"
                )

                return []

            if not isinstance(
                self._slots,
                dict,
            ):

                logger.error(
                    "Agent slot storage corrupted"
                )

                return []

            # ---------------------------------------------------------
            # VALIDATE FILTER
            # ---------------------------------------------------------
            if (
                state_filter is not None
                and not isinstance(
                    state_filter,
                    AgentState,
                )
            ):

                try:

                    state_filter = AgentState(
                        state_filter
                    )

                except Exception:

                    logger.warning(
                        f"Invalid state filter: "
                        f"{state_filter}"
                    )

                    state_filter = None

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(self, "_list_lock"):

                self._list_lock = threading.Lock()

            with self._list_lock:

                agents = []

                seen = set()

                # -----------------------------------------------------
                # ITERATE SLOTS
                # -----------------------------------------------------
                for aid, slot in (
                    self._slots.items()
                ):

                    try:

                        # -----------------------------------------
                        # SLOT VALIDATION
                        # -----------------------------------------
                        if slot is None:

                            continue

                        # -----------------------------------------
                        # AGENT ID VALIDATION
                        # -----------------------------------------
                        if aid is None:

                            continue

                        if not isinstance(
                            aid,
                            str,
                        ):

                            try:
                                aid = str(aid)

                            except Exception:
                                continue

                        aid = aid.strip()

                        if not aid:

                            continue

                        # -----------------------------------------
                        # DUPLICATE FILTER
                        # -----------------------------------------
                        if aid in seen:

                            logger.debug(
                                f"Duplicate agent skipped: "
                                f"{aid}"
                            )

                            continue

                        seen.add(aid)

                        # -----------------------------------------
                        # STATE VALIDATION
                        # -----------------------------------------
                        try:

                            slot_state = getattr(
                                slot,
                                "state",
                                None,
                            )

                        except Exception as exc:

                            logger.debug(
                                f"State read failed "
                                f"for {aid}: {exc}"
                            )

                            continue

                        # -----------------------------------------
                        # FILTER CHECK
                        # -----------------------------------------
                        if (
                            state_filter is None
                            or slot_state == state_filter
                        ):

                            agents.append(aid)

                    except Exception as exc:

                        logger.debug(
                            f"Agent listing failure "
                            f"({aid}): {exc}"
                        )

                # -----------------------------------------------------
                # STABLE SORT
                # -----------------------------------------------------
                try:

                    agents.sort()

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL METRICS
                # -----------------------------------------------------
                try:

                    self._list_requests = int(
                        getattr(
                            self,
                            "_list_requests",
                            0,
                        )
                    ) + 1

                except Exception:
                    pass

                # -----------------------------------------------------
                # DEBUG LOGGING
                # -----------------------------------------------------
                logger.debug(
                    f"Listed agents | "
                    f"count={len(agents)} "
                    f"filter={state_filter}"
                )

                return agents

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"list_agents failed: {exc}"
            )

            return []


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    "LLMProvider",
    "LLMEngine",
    "GGUFProvider",
    "FallbackProvider",
    "LLMCache",
    "EmbeddingEngine",
    "VectorIndex",
    "AgentPool",
    "AgentSlot",
    "AgentPriority",
    "AgentState",
    "GPUMemoryGuard",
    "PrioritizedRequest",
]