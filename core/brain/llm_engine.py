<<<<<<< HEAD
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
• GPUMemoryGuard – polls pynvml/psutil; back-pressures low-priority agents when VRAM > 85%.

Upgrade notes vs original
─────────────────────────
  [NEW]  AgentPriority enum            – CRITICAL / HIGH / NORMAL / LOW / BACKGROUND
  [NEW]  AgentState enum               – IDLE / ACTIVE / PAUSED / RETIRED
  [NEW]  PrioritizedRequest dataclass  – priority-queue item with per-agent metadata
  [NEW]  AgentSlot class               – per-agent state, rate-limiter, metrics
  [NEW]  GPUMemoryGuard class          – VRAM-aware back-pressure for RTX 3050
  [NEW]  AgentPool class               – register / pause / retire / broadcast 100 agents
  [MOD]  LLMEngine._ensure_async_primitives – creates PriorityQueue + Semaphore
  [MOD]  LLMEngine._start_workers      – launches N workers instead of 1
  [MOD]  LLMEngine._queue_worker       – priority-aware, agent-tagged, VRAM-aware
  [MOD]  LLMEngine.generate            – removes single-busy flag; uses semaphore
  [MOD]  LLMEngine.agent_generate      – high-level entry-point for AgentPool
  [MOD]  LLMEngine.shutdown            – cancels all N workers
  [FIX]  GGUFProvider.__init__         – async_lock created lazily (loop-safe)
  [FIX]  LLMEngine constructor         – _busy replaced by _active_requests counter
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
=======
import asyncio
import hashlib
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
import os
import re
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
<<<<<<< HEAD
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
=======
from typing import Any, AsyncIterator, Dict, List, Optional
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

import numpy as np

from ..utils.logger import logger

<<<<<<< HEAD
# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import faiss
=======
# ── Optional dependencies ──────────────────────────────────────────────────
try:
    import faiss

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from llama_cpp import Llama
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    LLAMA_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_AVAILABLE = False

<<<<<<< HEAD
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None
    _NVML_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_PROMPT_CHARS      = 200
DEFAULT_MAX_TOKENS    = 64
MAX_QUEUE_SIZE        = 500          # raised: 100 agents × 5 queued requests each
TASK_EXPIRY_SECONDS   = 30.0
CACHE_MAX_SIZE        = 500
CACHE_TTL_SECONDS     = 300
GPU_LAYERS_DEFAULT    = int(os.environ.get("EDIATH_GPU_LAYERS", "20"))
GPU_LAYERS_MAX        = 28           # RTX 3050 safe ceiling

# Agent-pool limits
MAX_AGENTS            = 100
_NUM_WORKERS          = 3            # async queue workers (see Architecture note above)
_AGENT_RATE_LIMIT_RPS = 2.0         # max requests-per-second per agent (default)
_VRAM_HIGH_WATERMARK  = 0.85        # fraction; above this → back-pressure LOW/BACKGROUND
_VRAM_POLL_INTERVAL   = 5.0         # seconds between VRAM polls

# ── Global model singleton ────────────────────────────────────────────────────
_shared_model      = None
_model_lock        = threading.Lock()
_model_call_lock   = threading.Lock()   # serialises all calls into llama.cpp C layer
=======
# ── Constants ──────────────────────────────────────────────────────────────
MAX_PROMPT_CHARS = 200  # single source-of-truth for prompt truncation
DEFAULT_MAX_TOKENS = 64  # sensible default; callers may raise
MAX_QUEUE_SIZE = 50
TASK_EXPIRY_SECONDS = 25.0
CACHE_MAX_SIZE = 500
CACHE_TTL_SECONDS = 300
GPU_LAYERS_DEFAULT = int(os.environ.get("EDIATH_GPU_LAYERS", "20"))
GPU_LAYERS_MAX = 35

# ── Global model singleton ─────────────────────────────────────────────────
_shared_model = None
_model_lock = threading.Lock()
_model_call_lock = threading.Lock()
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

def _cuda_available() -> bool:
    try:
        import ctypes
        ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        return True
    except OSError:
        pass
=======
def _cuda_available() -> bool:
    """
    FIX #1 – detect CUDA properly instead of `"cuda" in sys.platform`.
    Tries nvidia-smi via ctypes; falls back to checking for nvml.dll / libcuda.
    """
    try:
        import ctypes

        if sys.platform == "win32":
            ctypes.CDLL("nvcuda.dll")
        else:
            ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        pass

    # Secondary check: environment variable set by CUDA toolkit
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    return bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())


def _resolve_model_path(model_path: str) -> Optional[str]:
<<<<<<< HEAD
=======
    """
    FIX #23 – resolve model path once; returns absolute path or None.
    """
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    candidates = [
        model_path,
        os.path.abspath(model_path),
        os.path.join(os.getcwd(), model_path),
        os.path.join(os.path.dirname(__file__), "..", "..", model_path),
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        except Exception:
            pass
    return None


def _normalise_l2(v: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
=======
    """L2-normalise a 1-D float32 vector; return as-is if zero."""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD
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
    """
    Item stored in the PriorityQueue.
    Ordering: (priority, enqueue_time) so same-priority requests are FIFO.
    future and all non-comparable fields are excluded from ordering.
    """
    priority:     int   = field(compare=True)
    enqueue_time: float = field(compare=True)
    agent_id:     str   = field(compare=False)
    prompt:       str   = field(compare=False)
    kwargs:       Dict  = field(compare=False, default_factory=dict)
    future: Optional[asyncio.Future] = field(compare=False, default=None)


# ══════════════════════════════════════════════════════════════════════════════
# GPU Memory Guard  (RTX 3050 – 4 GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════

class GPUMemoryGuard:
    """
    Polls VRAM usage and exposes a simple back-pressure API.
    Falls back to CPU-memory monitoring when pynvml is unavailable.
    """

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
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._handle = None

    def _poll(self) -> None:
        now = time.monotonic()
        if now - self._last_poll < self.poll_interval:
            return
        self._last_poll = now

        if _NVML_AVAILABLE and self._handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._vram_fraction = info.used / max(1, info.total)
                return
            except Exception:
                pass

        # Fallback: RAM pressure as a proxy
        try:
            import psutil
            self._vram_fraction = psutil.virtual_memory().percent / 100.0
        except Exception:
            self._vram_fraction = 0.0

    @property
    def vram_fraction(self) -> float:
        self._poll()
        return self._vram_fraction

    def is_under_pressure(self) -> bool:
        return self.vram_fraction >= self.high_watermark

    def should_throttle(self, priority: int) -> bool:
        """Return True if a request at *priority* should be deferred."""
        if not self.is_under_pressure():
            return False
        return priority >= AgentPriority.LOW


# ══════════════════════════════════════════════════════════════════════════════
# Per-agent slot
# ══════════════════════════════════════════════════════════════════════════════

class AgentSlot:
    """
    Tracks per-agent lifecycle, rate limiting, and metrics.
    Thread/coroutine safe for the rate-limiter fields.
    """

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

        # Rate-limiter state (token-bucket)
        self._tokens         = rate_limit          # start full
        self._last_refill    = time.monotonic()
        self._queue_depth    = 0
        self._lock           = asyncio.Lock()

        # Metrics
        self.requests_sent      = 0
        self.requests_completed = 0
        self.requests_failed    = 0
        self.requests_throttled = 0
        self.total_latency_ms   = 0.0

    async def acquire(self) -> bool:
        """
        Token-bucket rate limiter.
        Returns True if the request is allowed, False if throttled.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self.rate_limit,
                self._tokens + elapsed * self.rate_limit,
            )
            self._last_refill = now

            if self._tokens < 1.0:
                self.requests_throttled += 1
                return False

            if self._queue_depth >= self.max_queue:
                self.requests_throttled += 1
                return False

            self._tokens      -= 1.0
            self._queue_depth += 1
            self.requests_sent += 1
            self.state = AgentState.ACTIVE
            return True

    async def release(self, success: bool, latency_ms: float = 0.0) -> None:
        async with self._lock:
            self._queue_depth = max(0, self._queue_depth - 1)
            if success:
                self.requests_completed += 1
                self.total_latency_ms   += latency_ms
            else:
                self.requests_failed += 1
            if self._queue_depth == 0:
                self.state = AgentState.IDLE

    def get_metrics(self) -> Dict[str, Any]:
        completed = max(1, self.requests_completed)
        return {
            "agent_id":          self.agent_id,
            "priority":          self.priority.name,
            "state":             self.state.value,
            "requests_sent":     self.requests_sent,
            "requests_completed":self.requests_completed,
            "requests_failed":   self.requests_failed,
            "requests_throttled":self.requests_throttled,
            "avg_latency_ms":    round(self.total_latency_ms / completed, 2),
            "queue_depth":       self._queue_depth,
            "tokens_available":  round(self._tokens, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════

=======
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Fallback provider
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
class FallbackProvider(LLMProvider):
    """Rule-based responses when no LLM model is available."""

    async def generate(self, prompt: str, **kwargs) -> str:
        p = prompt.lower().strip()
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if any(w in p for w in ("hello", " hi ", "hi\n")):
            return "Hello! I'm EDIATH, your AI assistant. How can I help you today?"
        if "what" in p and "time" in p:
            from datetime import datetime
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        if "how are you" in p:
            return "I'm doing well, thank you! How can I assist you?"
        if "thank" in p:
            return "You're welcome! Is there anything else I can help with?"
        if any(w in p for w in ("bye", "goodbye")):
            return "Goodbye! Have a great day!"
        if "help" in p:
            return "I can answer questions and help with various tasks. What would you like to know?"
        return "Could you please provide more details so I can better assist you?"


# ══════════════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

class LLMCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self.cache:    OrderedDict = OrderedDict()
        self.max_size  = max(1, int(max_size))
        self.ttl       = max(1, int(ttl))
        self.hits      = 0
        self.misses    = 0
        self.evictions = 0
        self.enabled   = True
        self._lock     = threading.Lock()

    def _make_key(self, prompt: str, **kwargs) -> str:
=======
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
        # FIX #8 – do NOT lowercase; preserve case for correct cache keys
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        try:
            prompt_clean = str(prompt).strip().replace("\n", " ")
            extra = str(sorted(kwargs.items())) if kwargs else ""
            return hashlib.md5(f"{prompt_clean}|{extra}".encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.md5(str(prompt).encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[Dict]:
<<<<<<< HEAD
        if not getattr(self, "enabled", False):
            return None
        try:
            key = self._make_key(prompt, **kwargs)
            with self._lock:
                entry = self.cache.get(key)
                if entry is None:
                    self.misses += 1
                    return None
                value, timestamp = entry
                if (time.time() - timestamp) > self.ttl:
=======
        """Thread-safe cache lookup with TTL + safe return"""

        if not getattr(self, "enabled", False):
            return None

        try:
            key = self._make_key(prompt, **kwargs)

            with self._lock:
                entry = self.cache.get(key)

                # -------------------------
                # MISS
                # -------------------------
                if entry is None:
                    self.misses += 1
                    return None

                value, timestamp = entry

                # -------------------------
                # TTL CHECK (CRITICAL FIX)
                # -------------------------
                now = time.time()
                if (now - timestamp) > self.ttl:
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                    try:
                        del self.cache[key]
                    except KeyError:
                        pass
<<<<<<< HEAD
                    self.evictions += 1
                    self.misses    += 1
                    return None
=======

                    self.evictions += 1
                    self.misses += 1
                    return None

                # -------------------------
                # LRU UPDATE
                # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                try:
                    self.cache.move_to_end(key)
                except Exception:
                    pass
<<<<<<< HEAD
                self.hits += 1
                return value.copy() if isinstance(value, dict) else value
        except Exception as exc:
            logger.debug("Cache get error: %s", exc)
            return None

    def set(self, prompt: str, response: Any, **kwargs) -> None:
        if not getattr(self, "enabled", False):
            return
        try:
            key   = self._make_key(prompt, **kwargs)
            value = response.copy() if isinstance(response, dict) else response
            with self._lock:
=======

                self.hits += 1

                # -------------------------
                # RETURN COPY (CRITICAL FIX)
                # -------------------------
                if isinstance(value, dict):
                    return value.copy()

                return value

        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None

    def set(self, prompt: str, response: Any, **kwargs) -> None:
        """Thread-safe cache insert with LRU + safe storage"""

        if not getattr(self, "enabled", False):
            return

        try:
            key = self._make_key(prompt, **kwargs)

            # -------------------------
            # SAFE COPY (CRITICAL FIX)
            # -------------------------
            if isinstance(response, dict):
                value = response.copy()
            else:
                value = response

            with self._lock:
                # -------------------------
                # UPDATE EXISTING
                # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                if key in self.cache:
                    try:
                        self.cache.move_to_end(key)
                    except Exception:
                        pass
<<<<<<< HEAD
=======

                # -------------------------
                # EVICTION (SAFE)
                # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                elif len(self.cache) >= self.max_size:
                    try:
                        self.cache.popitem(last=False)
                        self.evictions += 1
                    except KeyError:
                        pass
<<<<<<< HEAD
                self.cache[key] = (value, time.time())
        except Exception as exc:
            logger.debug("Cache set error: %s", exc)
=======

                # -------------------------
                # INSERT WITH TIMESTAMP
                # -------------------------
                self.cache[key] = (value, time.time())

        except Exception as e:
            logger.debug(f"Cache set error: {e}")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

    def clear(self) -> int:
        with self._lock:
            n = len(self.cache)
            self.cache.clear()
            return n


# ══════════════════════════════════════════════════════════════════════════════
# Embedding engine
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

class EmbeddingEngine:
    """Deterministic hash-seeded embeddings (replace with sentence-transformers for production)."""
=======
class EmbeddingEngine:
    """
    FIX #10 – deterministic hash-seeded embeddings so the same text always
    produces the same vector. Replace encode() with a real sentence-transformer
    when available.
    """
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7

    DIM = 384

    def encode(self, text: str) -> np.ndarray:
<<<<<<< HEAD
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
        rng  = np.random.default_rng(seed)
        vec  = rng.standard_normal(self.DIM).astype(np.float32)
=======
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return _normalise_l2(vec)


# ══════════════════════════════════════════════════════════════════════════════
# Vector index
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

class VectorIndex:
    def __init__(self, dim: int = EmbeddingEngine.DIM):
        self.dim   = dim
=======
class VectorIndex:
    """
    FIX #11 – L2-normalise all vectors before add/search so that inner-product
    scores equal cosine similarity.
    """

    def __init__(self, dim: int = EmbeddingEngine.DIM):
        self.dim = dim
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.texts: List[str] = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
            warnings.warn("FAISS not available — vector search disabled")

    def add(self, embedding: np.ndarray, text: str) -> None:
        if embedding is None or self.index is None:
            return
        emb = _normalise_l2(embedding.astype(np.float32)).reshape(1, -1)
        self.index.add(emb)
        self.texts.append(text)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
<<<<<<< HEAD
        try:
=======
        """Safe vector search with validation + robust indexing"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            if (
                query_embedding is None
                or not isinstance(query_embedding, np.ndarray)
                or query_embedding.size == 0
            ):
                return []
<<<<<<< HEAD
            if not self.texts or self.index is None:
                return []
            query = query_embedding.astype(np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)
            query = _normalise_l2(query)
            k = max(1, min(int(top_k), len(self.texts)))
            distances, indices = self.index.search(query, k)
            if indices is None or len(indices) == 0:
                return []
            return [
                self.texts[i]
                for i in indices[0]
                if isinstance(i, (int, np.integer)) and 0 <= i < len(self.texts)
            ]
        except Exception as exc:
            logger.debug("Vector search error: %s", exc)
=======

            if not self.texts or self.index is None:
                return []

            # -------------------------
            # NORMALIZE + SHAPE
            # -------------------------
            query = query_embedding.astype(np.float32)

            if query.ndim == 1:
                query = query.reshape(1, -1)

            query = _normalise_l2(query)

            # -------------------------
            # SAFE TOP-K
            # -------------------------
            k = max(1, min(int(top_k), len(self.texts)))

            # -------------------------
            # INDEX SEARCH (GUARDED)
            # -------------------------
            distances, indices = self.index.search(query, k)

            if indices is None or len(indices) == 0:
                return []

            # -------------------------
            # SAFE RESULT EXTRACTION
            # -------------------------
            results = []
            for i in indices[0]:
                if isinstance(i, (int, np.integer)) and 0 <= i < len(self.texts):
                    results.append(self.texts[i])

            return results

        except Exception as e:
            logger.debug(f"Vector search error: {e}")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            return []


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval engine
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

class RetrievalEngine:
    def __init__(self, embedder: EmbeddingEngine):
        self.embedder  = embedder
=======
class RetrievalEngine:
    def __init__(self, embedder: EmbeddingEngine):
        self.embedder = embedder
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.documents: List[tuple] = []

    def add_document(self, text: str) -> None:
        emb = self.embedder.encode(text)
        self.documents.append((emb, text))

    def search(self, query: str, top_k: int = 3) -> List[str]:
<<<<<<< HEAD
        try:
            if not query or not isinstance(query, str):
                return []
            if not getattr(self, "documents", None):
                return []
            query = query.strip()
            if not query:
                return []
=======
        """Safe semantic search with validation + optimized ranking"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not query or not isinstance(query, str):
                return []

            if not getattr(self, "documents", None):
                return []

            query = query.strip()
            if not query:
                return []

            # -------------------------
            # EMBEDDING (SAFE)
            # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            try:
                q = self.embedder.encode(query)
                if not isinstance(q, np.ndarray) or q.size == 0:
                    return []
<<<<<<< HEAD
            except Exception as exc:
                logger.debug("Embedding failed: %s", exc)
                return []
=======
            except Exception as e:
                logger.debug(f"Embedding failed: {e}")
                return []

            # -------------------------
            # NORMALIZATION (IMPROVED)
            # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return []
            q = q / q_norm
<<<<<<< HEAD
=======

            # -------------------------
            # SCORING (SAFE + FAST)
            # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            scores = []
            for emb, txt in self.documents:
                try:
                    if not isinstance(emb, np.ndarray) or emb.size == 0:
                        continue
<<<<<<< HEAD
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm == 0:
                        continue
                    scores.append((float(np.dot(q, emb / emb_norm)), txt))
                except Exception:
                    continue
            if not scores:
                return []
            scores.sort(key=lambda x: x[0], reverse=True)
            return [txt for _, txt in scores[: max(1, min(int(top_k), len(scores)))]]
        except Exception as exc:
            logger.debug("Retrieval search error: %s", exc)
=======

                    emb_norm = np.linalg.norm(emb)
                    if emb_norm == 0:
                        continue

                    sim = float(np.dot(q, emb / emb_norm))
                    scores.append((sim, txt))

                except Exception:
                    continue

            if not scores:
                return []

            # -------------------------
            # TOP-K SELECTION (OPTIMIZED)
            # -------------------------
            scores.sort(key=lambda x: x[0], reverse=True)

            k = max(1, min(int(top_k), len(scores)))

            return [txt for _, txt in scores[:k]]

        except Exception as e:
            logger.debug(f"Retrieval search error: {e}")
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            return []


# ══════════════════════════════════════════════════════════════════════════════
# Compression / Summarisation engines
# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD

=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
class CompressionEngine:
    def __init__(self, llm_engine: "LLMEngine"):
        self.llm = llm_engine

    async def compress(self, text: str, max_tokens: int = 100) -> str:
        if not text:
            return ""
<<<<<<< HEAD
        snippet = text[:MAX_PROMPT_CHARS]
        result  = await self.llm.safe_generate(f"Compress into key info only:\n{snippet}\nCOMPRESSED:")
=======
        # FIX #21 – cap input so assembled prompt stays within engine limit
        snippet = text[:MAX_PROMPT_CHARS]
        prompt = f"Compress into key info only:\n{snippet}\nCOMPRESSED:"
        result = await self.llm.safe_generate(prompt)
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return result.get("response", "").strip()

    async def compress_aggressive(self, text: str) -> str:
        if not text:
            return ""
        snippet = text[:MAX_PROMPT_CHARS]
<<<<<<< HEAD
        result  = await self.llm.safe_generate(f"Bullet-point summary, minimal words:\n{snippet}\nOUTPUT:")
=======
        prompt = f"Bullet-point summary, minimal words:\n{snippet}\nOUTPUT:"
        result = await self.llm.safe_generate(prompt)
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return result.get("response", "").strip()


class SummarizationEngine:
    def __init__(self, llm_engine: "LLMEngine"):
        self.llm = llm_engine

    async def summarize(self, text: str, max_length: int = 150) -> str:
        if not text:
            return ""
        snippet = text[:MAX_PROMPT_CHARS]
<<<<<<< HEAD
        result  = await self.llm.safe_generate(f"Summarize concisely:\n{snippet}\nSummary:")
        return result.get("response", "").strip()

    async def summarize_long(self, text: str, chunk_size: int = 800) -> str:
        chunks    = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
=======
        prompt = f"Summarize concisely:\n{snippet}\nSummary:"
        result = await self.llm.safe_generate(prompt)
        return result.get("response", "").strip()

    async def summarize_long(self, text: str, chunk_size: int = 800) -> str:
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        summaries = [await self.summarize(chunk) for chunk in chunks]
        return await self.summarize(" ".join(summaries))


# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD
# GGUF Provider  (RTX 3050 — 4 GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════

class GGUFProvider(LLMProvider):
=======
# GGUF Provider
# ══════════════════════════════════════════════════════════════════════════════
class GGUFProvider(LLMProvider):
    """
    llama-cpp-python GGUF provider, optimised for RTX 3050 (4 GB VRAM).
    Falls back to FallbackProvider when the model is unavailable.
    """

    # Patterns that are unambiguously adversarial/spam
    # FIX #12 – removed innocent words like 'baby', 'jesus', 'thelma'
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    _GARBAGE_PATTERNS = (
        "erectile dysfunction",
        "replica watches",
        "cheap meds",
        "click here to",
<<<<<<< HEAD
        "？",
        "\x00",
        "----------",
=======
        "�",
        "\x00",
        "----------",  # 10+ dashes
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    )

    def __init__(
        self,
        model_path: str,
        fallback_provider: Optional[LLMProvider] = None,
    ):
        global _shared_model

<<<<<<< HEAD
        # FIX: async_lock created lazily so it binds to the correct event loop.
        self._async_lock_instance: Optional[asyncio.Lock] = None
=======
        self.async_lock = asyncio.Lock()
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.fallback_provider = fallback_provider or FallbackProvider()
        self.model = None

        if not LLAMA_AVAILABLE:
<<<<<<< HEAD
            logger.warning("llama-cpp-python not installed — using fallback provider")
=======
            logger.warning(
                "❌ llama-cpp-python not installed — using fallback provider"
            )
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            return

        with _model_lock:
            if _shared_model is not None:
<<<<<<< HEAD
                logger.info("Reusing cached GGUF model singleton")
                self.model = _shared_model
                return

            logger.info("Loading GGUF model: %s", model_path)
            resolved = _resolve_model_path(model_path)
            if not resolved:
                logger.error("Model file not found: %s", model_path)
                return

            use_gpu = _cuda_available()
            n_gpu   = min(max(0, GPU_LAYERS_DEFAULT) if use_gpu else 0, GPU_LAYERS_MAX)
            logger.info("GPU layers: %d (%s)", n_gpu, "CUDA" if use_gpu else "CPU-only")

            try:
                model = Llama(
                    model_path  = resolved,
                    n_ctx       = 1024,
                    n_batch     = 64,
                    n_threads   = max(2, (os.cpu_count() or 4) // 2),
                    n_gpu_layers= n_gpu,
                    use_mmap    = True,
                    use_mlock   = False,
                    logits_all  = False,
                    embedding   = False,
                    verbose     = False,
                )
                _shared_model = model
                self.model    = model
                logger.info("GGUF model loaded (warmup deferred)")
            except Exception as exc:
                logger.error("Model load failed: %s", exc)
                self.model = None

    @property
    def async_lock(self) -> asyncio.Lock:
        """Lazy lock creation — safe when called from inside a running event loop."""
        if self._async_lock_instance is None:
            self._async_lock_instance = asyncio.Lock()
        return self._async_lock_instance

    def _is_garbage_response(self, text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return True
        tl = text.lower()
        for pattern in self._GARBAGE_PATTERNS:
            if pattern in tl:
                return True
        alnum = sum(c.isalnum() for c in text)
        if len(text) > 0 and alnum / len(text) < 0.25:
            return True
        if re.search(r"(.)\1{6,}", text):
            return True
        words = text.split()
        if len(words) > 5 and len(set(words)) / len(words) < 0.3:
            return True
        if re.fullmatch(r"[\W_]+", text):
            return True
        return False

    def _sync_generate(
        self,
        prompt:      str,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        if not getattr(self, "model", None):
            raise RuntimeError("Model not available")
        prompt = str(prompt or "").replace("\x00", "").strip()
        if not prompt:
            raise ValueError("Empty prompt")
        stop_tokens    = kwargs.get("stop",          ["</s>", "[INST]", "[/INST]", "\n\n"])
        top_k          = max(1,   min(100, int(kwargs.get("top_k",          40))))
        top_p          = max(0.1, min(1.0, float(kwargs.get("top_p",        0.90))))
        repeat_penalty = max(0.8, min(2.0, float(kwargs.get("repeat_penalty", 1.1))))
        try:
            with _model_call_lock:
                result = self.model.create_completion(
                    prompt         = prompt,
                    max_tokens     = int(max_tokens),
                    temperature    = float(temperature),
                    top_k          = top_k,
                    top_p          = top_p,
                    repeat_penalty = repeat_penalty,
                    stop           = stop_tokens,
                    echo           = False,
                )
        except Exception as exc:
            raise RuntimeError(f"Model execution failed: {exc}")
        if not isinstance(result, dict):
            raise ValueError("Invalid model output format")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("No choices in output")
        raw_text = str(choices[0].get("text", "") or "").strip()
        if len(raw_text) < 2:
            raise ValueError("Empty text in output")
        if self._is_garbage_response(raw_text):
            raise ValueError("Garbage response detected")
        text = re.sub(r"\[/?INST\]|</?s>", "", raw_text)
        text = re.sub(r"(?i)assistant:", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("Cleaned output is empty")
        return text

    async def generate(
        self,
        prompt:      str,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        retries:     int   = 1,
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
=======
                logger.info("✓ Reusing cached GGUF model singleton")
                self.model = _shared_model
                return

            logger.info(f"🔥 Loading GGUF model: {model_path}")
            resolved = _resolve_model_path(model_path)
            if not resolved:
                logger.error(f"❌ Model file not found: {model_path}")
                return

            # FIX #1 – proper CUDA detection
            use_gpu = _cuda_available()
            n_gpu = min(
                max(0, GPU_LAYERS_DEFAULT) if use_gpu else 0,
                GPU_LAYERS_MAX,
            )
            logger.info(
                f"GPU layers: {n_gpu} ({'CUDA detected' if use_gpu else 'CPU-only'})"
            )

            try:
                model = Llama(
                    model_path=resolved,
                    n_ctx=1024,
                    n_batch=64,
                    n_threads=max(2, (os.cpu_count() or 4) // 2),
                    n_gpu_layers=n_gpu,
                    use_mmap=True,
                    use_mlock=False,
                    logits_all=False,
                    embedding=False,
                    verbose=False,
                )

                # NOTE: skip immediate synchronous warmup/test inference here.
                # Some GGML backends crash when called concurrently during init.
                # Warmup will be performed lazily on first request (and is
                # protected by _model_call_lock in generate()).

                _shared_model = model
                self.model = model
                logger.info("✅ GGUF model loaded (warmup deferred)")

            except Exception as e:
                logger.error(f"❌ Model load failed: {e}")
                self.model = None

    # ── Garbage detection ──────────────────────────────────────────────────
    def _is_garbage_response(self, text: str) -> bool:
        """Robust garbage detection with better signal filtering"""

        if not text:
            return True

        text = str(text).strip()

        # -------------------------
        # LENGTH CHECK
        # -------------------------
        if len(text) < 2:
            return True

        tl = text.lower()

        # -------------------------
        # PATTERN MATCH (SAFE)
        # -------------------------
        for pattern in getattr(self, "_GARBAGE_PATTERNS", []):
            try:
                if pattern in tl:
                    return True
            except Exception:
                continue

        # -------------------------
        # ALPHANUMERIC RATIO (IMPROVED)
        # -------------------------
        total_chars = len(text)
        alnum_chars = sum(c.isalnum() for c in text)

        if total_chars > 0:
            ratio = alnum_chars / total_chars
            if ratio < 0.25:  # slightly relaxed (was 0.30)
                return True

        # -------------------------
        # REPEATED CHARACTER DETECTION (IMPROVED)
        # -------------------------
        if re.search(r"(.)\1{6,}", text):  # slightly stricter pattern
            return True

        # -------------------------
        # LOW WORD DIVERSITY (NEW FIX)
        # -------------------------
        words = text.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return True

        # -------------------------
        # NON-LANGUAGE NOISE (NEW FIX)
        # -------------------------
        if re.fullmatch(r"[\W_]+", text):
            return True

        return False

    # ── Synchronous generation (called from thread) ────────────────────────
    def _sync_generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """Safe synchronous GGUF generation with strict validation"""

        if not getattr(self, "model", None):
            raise RuntimeError("Model not available")

        # -------------------------
        # INPUT SANITIZATION (CRITICAL FIX)
        # -------------------------
        prompt = str(prompt or "").replace("\x00", "").strip()
        if not prompt:
            raise ValueError("Empty prompt")

        # -------------------------
        # SAFE PARAM EXTRACTION
        # -------------------------
        stop_tokens = kwargs.get("stop", ["</s>", "[INST]", "[/INST]", "\n\n"])
        top_k = max(1, min(100, int(kwargs.get("top_k", 40))))
        top_p = max(0.1, min(1.0, float(kwargs.get("top_p", 0.90))))
        repeat_penalty = max(0.8, min(2.0, float(kwargs.get("repeat_penalty", 1.1))))

        try:
            # -------------------------
            # MODEL CALL (GUARDED)
            # -------------------------
            # serialize access to the underlying C model to avoid concurrent
            # calls into llama.cpp which can trigger internal assertions
            with _model_call_lock:
                result = self.model.create_completion(
                    prompt=prompt,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_tokens,
                    echo=False,
                )

        except Exception as e:
            raise RuntimeError(f"Model execution failed: {e}")

        # -------------------------
        # OUTPUT VALIDATION (CRITICAL FIX)
        # -------------------------
        if not isinstance(result, dict):
            raise ValueError("Invalid model output format")

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("No choices in output")

        raw_text = str(choices[0].get("text", "") or "").strip()

        if len(raw_text) < 2:
            raise ValueError("Empty text in output")

        # -------------------------
        # GARBAGE FILTER (IMPROVED)
        # -------------------------
        if self._is_garbage_response(raw_text):
            raise ValueError("Garbage response detected")

        # -------------------------
        # CLEAN OUTPUT
        # -------------------------
        text = re.sub(r"\[/?INST\]|</?s>", "", raw_text)
        text = re.sub(r"(?i)assistant:", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            raise ValueError("Cleaned output is empty")

        return text

    # ── Async generation ───────────────────────────────────────────────────
    async def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        retries: int = 1,
        **kwargs,
    ) -> str:
        """Thread-safe, timeout-safe GGUF generation with optimized retries"""

        # -------------------------
        # FALLBACK EARLY
        # -------------------------
        if not getattr(self, "model", None):
            return await self.fallback_provider.generate(prompt, **kwargs)

        prompt = str(prompt or "").strip()
        if not prompt:
            return "Please provide a valid prompt."

        # -------------------------
        # SAFE STOP TOKENS
        # -------------------------
        stop_tokens = kwargs.pop("stop", ["</s>", "[INST]", "[/INST]", "\n\n"])

        for attempt in range(retries + 1):
            try:
                # -------------------------
                # LOCK ONLY DURING EXECUTION (CRITICAL FIX)
                # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
                text = str(result or "").strip()
                if len(text) < 2:
                    raise ValueError("Empty response")
                return text
            except asyncio.TimeoutError:
                logger.warning("GGUF timeout (attempt %d)", attempt + 1)
            except Exception as exc:
                logger.warning("GGUF attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries:
                await asyncio.sleep(min(1.5, 0.3 * (2 ** attempt)))
=======

                # -------------------------
                # VALIDATION (CRITICAL FIX)
                # -------------------------
                text = str(result or "").strip()
                if len(text) < 2:
                    raise ValueError("Empty response")

                return text

            except asyncio.TimeoutError:
                logger.warning(f"GGUF timeout (attempt {attempt + 1})")

            except Exception as e:
                logger.warning(f"GGUF attempt {attempt + 1} failed: {e}")

            # -------------------------
            # SMART BACKOFF
            # -------------------------
            if attempt < retries:
                await asyncio.sleep(min(1.5, 0.3 * (2**attempt)))

        # -------------------------
        # SAFE FALLBACK
        # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        try:
            return await self.fallback_provider.generate(prompt, **kwargs)
        except Exception:
            return "I'm here to help. Please try again."


# ══════════════════════════════════════════════════════════════════════════════
<<<<<<< HEAD
# Main LLM Engine  — upgraded for 100-agent concurrency
# ══════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    High-level async LLM engine.

    Concurrency model
    -----------------
    • asyncio.PriorityQueue (size MAX_QUEUE_SIZE) receives PrioritizedRequest items.
    • _NUM_WORKERS async workers drain the queue concurrently.
    • Because the underlying GGUF model is single-threaded (C layer, protected by
      _model_call_lock), only ONE worker does real inference at a time.
      The others serve cache hits instantly or wait on the model lock.
    • A Semaphore(_NUM_WORKERS) ensures we never spin more workers than slots.

    Agent interface
    ---------------
    • Use agent_generate(agent_id, prompt, **kwargs) to submit prioritised requests.
    • The AgentPool class (below) provides full lifecycle management for 100 agents.
    """

    def __init__(self, model_path: str = "models/EDIATH-q4_k_m.gguf"):
=======
# Main LLM Engine
# ══════════════════════════════════════════════════════════════════════════════
class LLMEngine:
    """
    High-level async LLM engine with caching, queueing, rate-limiting,
    embedding, retrieval, summarisation, and compression.
    """

    def __init__(self, model_path: str = "models/EDIATH-q4_k_m.gguf"):
        # FIX #23 – resolve path once here; pass resolved path to GGUFProvider
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        resolved = _resolve_model_path(model_path) or model_path

        self.fallback_provider = FallbackProvider()
        self.provider: LLMProvider = (
            GGUFProvider(resolved, self.fallback_provider)
            if LLAMA_AVAILABLE
            else self.fallback_provider
        )

<<<<<<< HEAD
        # Async primitives (created lazily)
        self._global_lock:   Optional[asyncio.Lock]          = None
        self._rate_lock:     Optional[asyncio.Lock]          = None
        self._queue:         Optional[asyncio.PriorityQueue] = None
        self._semaphore:     Optional[asyncio.Semaphore]     = None

        # Worker management
        self._workers:         List[asyncio.Task] = []
        self._worker_running:  bool = False
        self._shutdown:        bool = False

        # Request tracking (replaces the single _busy flag)
        self._active_requests: int  = 0
        self.generation_count: int  = 0
        self._last_call_time:  float = 0.0
        self.min_delay:        float = 0.1  # reduced: workers handle fair scheduling
        self.max_queue_size:   int   = MAX_QUEUE_SIZE

        # Cache
        self.cache         = LLMCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
        self.cache_enabled = True

        # Embedding / retrieval
        self.embedder      = EmbeddingEngine()
        self.index         = VectorIndex() if FAISS_AVAILABLE else None
        self.index_enabled = FAISS_AVAILABLE

        # Summarization / compression
        self.summarizer           = SummarizationEngine(self)
        self.summarization_enabled= True
        self.compressor           = CompressionEngine(self)
        self.compression_enabled  = True

        # Semantic memory
        self.semantic_memory:    List[Dict] = []
        self.semantic_enabled:   bool       = True
        self.similarity_threshold: float    = 0.85

        # Persistent memory
        self.memory         = None
        self.memory_enabled = False
        try:
            from ..memory.memory_manager import MemoryManager
            self.memory         = MemoryManager()
=======
        # FIX #14 – do NOT create asyncio primitives at import / init time.
        # They are created lazily in _ensure_async_primitives().
        self._global_lock: Optional[asyncio.Lock] = None
        self._rate_lock: Optional[asyncio.Lock] = None
        self._queue: Optional[asyncio.Queue] = None

        self.generation_count = 0
        self._last_call_time = 0.0
        self.min_delay = 0.3
        self._busy = False
        self._worker_task: Optional[asyncio.Task] = None
        self._worker_running = False
        self._shutdown = False
        self.max_queue_size = MAX_QUEUE_SIZE

        self.cache = LLMCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
        self.cache_enabled = True

        self.embedder = EmbeddingEngine()
        self.index = VectorIndex() if FAISS_AVAILABLE else None
        self.index_enabled = FAISS_AVAILABLE

        self.summarizer = SummarizationEngine(self)
        self.summarization_enabled = True
        self.compressor = CompressionEngine(self)
        self.compression_enabled = True

        self.semantic_memory: List[Dict] = []
        self.semantic_enabled = True
        self.similarity_threshold = 0.85

        self.memory = None
        self.memory_enabled = False
        try:
            from ..memory.memory_manager import MemoryManager

            self.memory = MemoryManager()
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            self.memory_enabled = True
        except Exception:
            pass

<<<<<<< HEAD
        # GPU guard (RTX 3050 back-pressure)
        self.gpu_guard = GPUMemoryGuard()

        self._initialized = False

    # ── Lazy async-primitive creation ─────────────────────────────────────────

    def _ensure_async_primitives(self) -> None:
        """Create asyncio primitives on first use inside a running event loop."""
=======
        self._initialized = False

    # ── Lazy async-primitive creation ──────────────────────────────────────
    def _ensure_async_primitives(self) -> None:
        """Create asyncio primitives on first use within a running event loop."""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        if self._global_lock is None:
            self._global_lock = asyncio.Lock()
        if self._rate_lock is None:
            self._rate_lock = asyncio.Lock()
        if self._queue is None:
<<<<<<< HEAD
            self._queue = asyncio.PriorityQueue(maxsize=MAX_QUEUE_SIZE)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(_NUM_WORKERS)

    # ── Worker management ─────────────────────────────────────────────────────

    def _start_workers(self) -> None:
        """Launch _NUM_WORKERS async queue workers (idempotent)."""
        self._ensure_async_primitives()
        if self._worker_running:
            return
        self._worker_running = True
        self._workers.clear()
        for i in range(_NUM_WORKERS):
            task = asyncio.create_task(
                self._queue_worker(i),
                name=f"llm_queue_worker_{i}",
            )
            self._workers.append(task)
        logger.info("LLMEngine: started %d queue workers", _NUM_WORKERS)

    async def _queue_worker(self, worker_id: int) -> None:
        """
        Priority-aware queue consumer.
        Cache hits are served instantly; model calls wait on _model_call_lock.
        Back-pressure is applied when GPU memory is high.
        """
        logger.debug("LLM queue worker %d started", worker_id)
        current_future: Optional[asyncio.Future] = None

        while self._worker_running and not self._shutdown:
            try:
                req: PrioritizedRequest = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )

                current_future = req.future
                created_at     = req.enqueue_time

                # Expire stale requests
                if time.monotonic() - created_at > TASK_EXPIRY_SECONDS:
                    if current_future and not current_future.done():
                        current_future.set_result({
                            "response": "Request expired in queue. Please try again.",
                            "intent": "idle", "error": False, "stale": True,
                        })
                    self._queue.task_done()
                    continue

                # Back-pressure: defer low-priority requests under VRAM pressure
                if self.gpu_guard.should_throttle(req.priority):
                    if current_future and not current_future.done():
                        current_future.set_result({
                            "response": "System under memory pressure. Request deferred.",
                            "intent": "idle", "error": False, "throttled": True,
                        })
                    self._queue.task_done()
                    continue

                result = await self.safe_generate(req.prompt, **req.kwargs)

                if current_future and not current_future.done():
                    current_future.set_result(result)

            except asyncio.TimeoutError:
                current_future = None
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Queue worker %d error: %s", worker_id, exc)
                if current_future and not current_future.done():
                    current_future.set_result({
                        "response": "Worker recovered from error. Please try again.",
                        "intent": "idle", "error": True,
                    })
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass
                await asyncio.sleep(0)

        logger.debug("LLM queue worker %d stopped", worker_id)

    # ── Prompt builder ────────────────────────────────────────────────────────

    async def _build_prompt(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        try:
            prompt = str(prompt or "").replace("\x00", "").strip()
            prompt = re.sub(r"\s+", " ", prompt)[:MAX_PROMPT_CHARS]
            if not prompt:
                return "[INST] Please respond helpfully. [/INST]"
            if system_prompt:
                sp = str(system_prompt).replace("\x00", "").strip()
                sp = re.sub(r"\s+", " ", sp)[:120]
                sp = sp.replace("[INST]", "").replace("[/INST]", "")
                return f"[INST] {sp}\n\n{prompt} [/INST]"
            return f"[INST] {prompt} [/INST]"
        except Exception as exc:
            logger.debug("Prompt build error: %s", exc)
            return f"[INST] {prompt[:200]} [/INST]"

    # ── Rate limiter ──────────────────────────────────────────────────────────

    async def _rate_limit(self) -> None:
        now     = time.monotonic()
        elapsed = now - getattr(self, "_last_call_time", 0)
        wait    = self.min_delay - elapsed
        if wait <= 0:
            self._last_call_time = now
            return
        async with self._rate_lock:
            now     = time.monotonic()
            elapsed = now - self._last_call_time
            wait    = self.min_delay - elapsed
=======
            self._queue = asyncio.Queue()

    # ── Prompt builder ─────────────────────────────────────────────────────
    async def _build_prompt(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Safe, sanitized prompt builder with injection protection"""

        try:
            # -------------------------
            # SANITIZE INPUT (CRITICAL FIX)
            # -------------------------
            prompt = str(prompt or "").replace("\x00", "").strip()
            prompt = re.sub(r"\s+", " ", prompt)[:MAX_PROMPT_CHARS]

            if not prompt:
                return "[INST] Please respond helpfully. [/INST]"

            # -------------------------
            # SANITIZE SYSTEM PROMPT
            # -------------------------
            if system_prompt:
                sp = str(system_prompt).replace("\x00", "").strip()
                sp = re.sub(r"\s+", " ", sp)[:120]

                # prevent breaking instruction format
                sp = sp.replace("[INST]", "").replace("[/INST]", "")

                return f"[INST] {sp}\n\n{prompt} [/INST]"

            # -------------------------
            # DEFAULT PROMPT
            # -------------------------
            return f"[INST] {prompt} [/INST]"

        except Exception as e:
            logger.debug(f"Prompt build error: {e}")
            return f"[INST] {prompt[:200]} [/INST]"

    # ── Rate limiter ────────────────────────────────────────────────────────
    async def _rate_limit(self) -> None:
        """Adaptive, low-contention rate limiter"""

        now = time.monotonic()

        # -------------------------
        # FAST PATH (NO LOCK IF NO WAIT)
        # -------------------------
        elapsed = now - getattr(self, "_last_call_time", 0)
        queue_size = self._queue.qsize() if getattr(self, "_queue", None) else 0

        adaptive_delay = self.min_delay + min(0.2, queue_size * 0.01)
        wait = adaptive_delay - elapsed

        if wait <= 0:
            self._last_call_time = now
            return

        # -------------------------
        # LOCK ONLY WHEN NEEDED (CRITICAL FIX)
        # -------------------------
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            wait = adaptive_delay - elapsed

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            if wait > 0:
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    return
<<<<<<< HEAD
            self._last_call_time = time.monotonic()

    # ── Core generate ─────────────────────────────────────────────────────────

    async def generate(
        self,
        prompt:        str,
        system_prompt: Optional[str] = None,
        max_tokens:    int           = DEFAULT_MAX_TOKENS,
        temperature:   float         = 0.3,
        retries:       int           = 1,
        **kwargs,
    ) -> Dict[str, Any]:
=======

            self._last_call_time = time.monotonic()

    # ── Core generate ───────────────────────────────────────────────────────
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.3,
        retries: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self._ensure_async_primitives()

        if not self.provider:
            return {
<<<<<<< HEAD
                "response":   "Model not available. Please check installation.",
                "intent":     "idle",
=======
                "response": "Model not available. Please check installation.",
                "intent": "idle",
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                "confidence": 0.0,
                "reflection": "no_provider",
            }

<<<<<<< HEAD
        # Use semaphore to limit active concurrent calls
        async with self._semaphore:
            self._active_requests += 1
            try:
                await self._rate_limit()

                prompt = str(prompt).strip()
                if not prompt or len(prompt) < 2:
                    return {
                        "response":   "Please provide a valid question or request.",
                        "intent":     "idle",
                        "confidence": 0.0,
                        "reflection": "invalid_prompt",
                    }

                if self.cache_enabled:
                    try:
                        cached = self.cache.get(prompt)
                        if cached:
                            return {**cached, "source": "cache"}
                    except Exception:
                        pass

                prompt    = prompt[:MAX_PROMPT_CHARS]
                try:
                    full_prompt = await self._build_prompt(prompt, system_prompt)
                except Exception as exc:
                    logger.warning("Prompt build failed: %s", exc)
                    full_prompt = prompt

                for attempt in range(retries + 1):
                    try:
                        start = time.monotonic()
                        raw   = await asyncio.wait_for(
                            self.provider.generate(
                                full_prompt,
                                max_tokens  = max_tokens,
                                temperature = temperature,
                                **kwargs,
                            ),
                            timeout=20.0,
                        )
                        elapsed = time.monotonic() - start

                        raw = str(raw or "").strip()
                        raw = re.sub(r"\[/?INST\]|</?s>", "", raw)
                        raw = re.sub(r"(?i)assistant:", "", raw)
                        raw = re.sub(r"\s+", " ", raw).strip()

                        if len(raw) < 3:
                            raw = "I understand. Could you please rephrase your question?"

                        tl = raw.lower()
                        if any(k in tl for k in ("open", "run", "execute", "launch")):
                            intent = "action"
                        elif any(k in tl for k in ("search", "find", "look up", "research")):
                            intent = "research"
                        elif any(k in tl for k in ("code", "write", "script", "program")):
                            intent = "code"
                        elif any(k in tl for k in ("hello", "hi", "hey", "greetings")):
                            intent = "greeting"
                        else:
                            intent = "chat"

                        confidence = round(min(0.9, max(0.3, len(raw) / 200)), 2)

                        result = {
                            "response":         raw,
                            "intent":           intent,
                            "confidence":       confidence,
                            "reflection":       "ok",
                            "generation_time":  round(elapsed, 3),
                        }

                        if self.cache_enabled:
                            try:
                                self.cache.set(prompt, result.copy())
                            except Exception:
                                pass

                        self.generation_count += 1
                        logger.info("LLM done in %.2fs | intent=%s", elapsed, intent)
                        return result

                    except asyncio.TimeoutError:
                        logger.warning("Timeout (attempt %d)", attempt + 1)
                        if attempt < retries:
                            await asyncio.sleep(0.5)
                    except Exception as exc:
                        logger.warning("Generate attempt %d failed: %s", attempt + 1, exc)
                        if attempt < retries:
                            await asyncio.sleep(0.5 * (attempt + 1))

                return {
                    "response":   "I'm ready to help. What can I assist you with?",
                    "intent":     "chat",
                    "confidence": 0.3,
                    "reflection": "fallback",
                }
            finally:
                self._active_requests = max(0, self._active_requests - 1)

    # ── safe_generate ─────────────────────────────────────────────────────────

    async def safe_generate(
        self,
        prompt:  str,
        timeout: float = 20.0,
        retries: int   = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        self._ensure_async_primitives()
        _safe_default = {
            "response":   "I'm here to help. Please try again.",
            "intent":     "idle",
            "confidence": 0.1,
            "reflection": "safe_fallback",
        }
        try:
            prompt = str(prompt or "").strip()
            if not prompt:
                return _safe_default

=======
        # -------------------------
        # FAST BUSY CHECK (NO LOCK BLOCKING)
        # -------------------------
        if self._busy:
            return {
                "response": "I'm processing another request. Please wait a moment.",
                "intent": "idle",
                "confidence": 0.0,
                "reflection": "busy",
            }

        # -------------------------
        # SET BUSY (SAFE)
        # -------------------------
        self._busy = True

        try:
            await self._rate_limit()

            prompt = str(prompt).strip()
            if not prompt or len(prompt) < 2:
                return {
                    "response": "Please provide a valid question or request.",
                    "intent": "idle",
                    "confidence": 0.0,
                    "reflection": "invalid_prompt",
                }

            # -------------------------
            # CACHE CHECK
            # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            if self.cache_enabled:
                try:
                    cached = self.cache.get(prompt)
                    if cached:
                        return {**cached, "source": "cache"}
                except Exception:
                    pass

<<<<<<< HEAD
            for attempt in range(retries + 1):
                try:
                    result = await asyncio.wait_for(
                        self.generate(prompt, retries=0, **kwargs),
                        timeout=timeout,
                    )
                    if result.get("reflection") not in ("fallback", "no_provider"):
                        return result
                except asyncio.TimeoutError:
                    logger.warning("safe_generate timeout (attempt %d)", attempt + 1)
                except asyncio.CancelledError:
                    return _safe_default
                except Exception as exc:
                    logger.warning("safe_generate attempt %d: %s", attempt + 1, exc)
                if attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

            return _safe_default
        except Exception as exc:
            logger.error("safe_generate failed: %s", exc)
            return _safe_default

    # ── Agent-aware generate ──────────────────────────────────────────────────

    async def agent_generate(
        self,
        agent_id:  str,
        prompt:    str,
        priority:  AgentPriority = AgentPriority.NORMAL,
        timeout:   float         = 30.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit a generation request tagged with agent_id and priority.
        Returns a future result; the actual inference may be queued behind
        other agents' requests.
        """
        self._ensure_async_primitives()

        if self._queue.full():
            return {
                "response":   "Request queue full. Please retry later.",
                "intent":     "idle",
                "confidence": 0.0,
                "reflection": "queue_full",
                "agent_id":   agent_id,
            }

        loop   = asyncio.get_event_loop()
        future = loop.create_future()

        req = PrioritizedRequest(
            priority     = int(priority),
            enqueue_time = time.monotonic(),
            agent_id     = agent_id,
            prompt       = prompt,
            kwargs       = kwargs,
            future       = future,
        )

        await self._queue.put(req)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            result["agent_id"] = agent_id
            return result
        except asyncio.TimeoutError:
            if not future.done():
                future.cancel()
            return {
                "response":   "Agent request timed out.",
                "intent":     "idle",
                "confidence": 0.0,
                "reflection": "agent_timeout",
                "agent_id":   agent_id,
            }

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def generate_stream(
        self, prompt: str, timeout: float = 20.0
    ) -> AsyncIterator[str]:
        try:
            data  = await asyncio.wait_for(self.safe_generate(prompt), timeout=timeout)
            text  = str(data.get("response", "")).strip()
            if not text:
                yield "I'm here to help. What would you like to know?"
                return
            chunk = ""
            for word in text.split():
                next_chunk = f"{chunk} {word}".strip()
                if len(next_chunk) >= 40:
                    yield next_chunk
                    chunk = ""
                    await asyncio.sleep(0)
                else:
                    chunk = next_chunk
            if chunk:
                yield chunk
            yield ""
        except asyncio.TimeoutError:
            yield "Request is taking longer than expected. Please try again."
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Stream error: %s", exc)
            yield "Something went wrong. Please try again."

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        if getattr(self, "_shutdown", False):
            return
        self._shutdown       = True
        self._worker_running = False

        for task in self._workers:
            if not task.done():
                task.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        if self._queue:
            while not self._queue.empty():
                try:
                    req = self._queue.get_nowait()
                    if req.future and not req.future.done():
                        req.future.set_result({
                            "response":   "System shutting down. Please retry.",
                            "intent":     "idle",
                            "error":      True,
                            "shutdown":   True,
                        })
                    self._queue.task_done()
                except Exception:
                    break

        logger.info("LLMEngine shut down cleanly")

    # ── Context management ────────────────────────────────────────────────────

    def clear_context(self, force: bool = False) -> Dict[str, Any]:
        start       = time.monotonic()
        cache_cleared = 0
        sem_cleared   = 0
        try:
            if self.cache_enabled and hasattr(self, "cache"):
                try:
                    cache_cleared = self.cache.clear()
                except Exception as exc:
                    logger.debug("Cache clear failed: %s", exc)
=======
            # -------------------------
            # TRUNCATE
            # -------------------------
            prompt = prompt[:MAX_PROMPT_CHARS]

            # -------------------------
            # BUILD PROMPT
            # -------------------------
            try:
                full_prompt = await self._build_prompt(prompt, system_prompt)
            except Exception as e:
                logger.warning(f"Prompt build failed: {e}")
                full_prompt = prompt

            # -------------------------
            # GENERATION LOOP
            # -------------------------
            for attempt in range(retries + 1):
                try:
                    start = time.monotonic()

                    raw = await asyncio.wait_for(
                        self.provider.generate(
                            full_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        ),
                        timeout=20.0,
                    )

                    elapsed = time.monotonic() - start

                    # -------------------------
                    # CLEAN OUTPUT
                    # -------------------------
                    raw = str(raw or "").strip()
                    raw = re.sub(r"\[/?INST\]|</?s>", "", raw)
                    raw = re.sub(r"(?i)assistant:", "", raw)
                    raw = re.sub(r"\s+", " ", raw).strip()

                    if len(raw) < 3:
                        raw = "I understand. Could you please rephrase your question?"

                    # -------------------------
                    # INTENT DETECTION
                    # -------------------------
                    tl = raw.lower()
                    if any(k in tl for k in ("open", "run", "execute", "launch")):
                        intent = "action"
                    elif any(
                        k in tl for k in ("search", "find", "look up", "research")
                    ):
                        intent = "research"
                    elif any(k in tl for k in ("code", "write", "script", "program")):
                        intent = "code"
                    elif any(k in tl for k in ("hello", "hi", "hey", "greetings")):
                        intent = "greeting"
                    else:
                        intent = "chat"

                    confidence = round(min(0.9, max(0.3, len(raw) / 200)), 2)

                    result = {
                        "response": raw,
                        "intent": intent,
                        "confidence": confidence,
                        "reflection": "ok",
                        "generation_time": round(elapsed, 3),
                    }

                    # -------------------------
                    # CACHE STORE (SAFE)
                    # -------------------------
                    if self.cache_enabled:
                        try:
                            self.cache.set(prompt, result.copy())
                        except Exception:
                            pass

                    self.generation_count += 1
                    logger.info(f"LLM done in {elapsed:.2f}s | intent={intent}")

                    return result

                except asyncio.TimeoutError:
                    logger.warning(f"⚠ Timeout (attempt {attempt + 1})")
                    if attempt < retries:
                        await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"⚠ Generate attempt {attempt + 1} failed: {e}")
                    if attempt < retries:
                        await asyncio.sleep(0.5 * (attempt + 1))

            # -------------------------
            # FINAL FALLBACK
            # -------------------------
            return {
                "response": "I'm ready to help. What can I assist you with?",
                "intent": "chat",
                "confidence": 0.3,
                "reflection": "fallback",
            }

        finally:
            # -------------------------
            # GUARANTEED RELEASE (NO DEADLOCK)
            # -------------------------
            self._busy = False

    # ── safe_generate ───────────────────────────────────────────────────────

    async def safe_generate(
        self,
        prompt: str,
        timeout: float = 20.0,
        retries: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Robust, timeout-safe, retry-optimized generation wrapper"""

        self._ensure_async_primitives()

        prompt = str(prompt).strip()[: MAX_PROMPT_CHARS * 2]

        if not prompt:
            return {
                "response": "Please provide a valid prompt.",
                "intent": "idle",
                "confidence": 0.0,
                "status": "invalid",
            }

        last_error: Optional[str] = None
        start = time.monotonic()

        for attempt in range(retries + 1):
            try:
                # -------------------------
                # TIMEOUT PROTECTED CALL
                # -------------------------
                result = await asyncio.wait_for(
                    self.generate(prompt, **kwargs),
                    timeout=timeout,
                )

                if not isinstance(result, dict):
                    raise ValueError("Invalid result format")

                # -------------------------
                # VALIDATION (CRITICAL FIX)
                # -------------------------
                response = str(result.get("response", "")).strip()

                if len(response) < 3:
                    raise ValueError("Empty or too-short response")

                confidence = float(result.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return {
                    **result,
                    "response": response,
                    "confidence": confidence,
                    "latency": round(time.monotonic() - start, 3),
                    "attempt": attempt,
                    "status": "success",
                }

            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"⚠ safe_generate timeout (attempt {attempt + 1})")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"⚠ safe_generate attempt {attempt + 1} failed: {e}")

            # -------------------------
            # BACKOFF (SMART)
            # -------------------------
            if attempt < retries:
                await asyncio.sleep(min(1.5, 0.3 * (2**attempt)))

        # -------------------------
        # FINAL FALLBACK
        # -------------------------
        return {
            "response": "I'm here to help. What would you like to know?",
            "intent": "chat",
            "confidence": 0.3,
            "error": True,
            "reason": last_error,
            "latency": round(time.monotonic() - start, 3),
            "attempt": retries + 1,
            "status": "fallback",
        }

    # ── queued_generate ─────────────────────────────────────────────────────

    async def queued_generate(
        self, prompt: str, timeout: float = 20.0, **kwargs
    ) -> Dict[str, Any]:
        """Safe queued generation with worker control + timeout protection"""

        self._ensure_async_primitives()

        prompt = str(prompt).strip()[: MAX_PROMPT_CHARS * 2]

        if not prompt:
            return {
                "response": "Invalid input.",
                "intent": "idle",
                "error": True,
            }

        # -------------------------
        # SAFE CACHE (NO MUTATION BUG)
        # -------------------------
        if self.cache_enabled:
            try:
                cached = self.cache.get(prompt)
                if cached:
                    return {**cached, "source": "cache-fast"}
            except Exception:
                pass

        # -------------------------
        # QUEUE LIMIT PROTECTION
        # -------------------------
        if self._queue.qsize() >= self.max_queue_size:
            return {
                "response": "System is busy. Please try again in a moment.",
                "intent": "idle",
                "error": False,
                "queue_full": True,
            }

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        # -------------------------
        # ENQUEUE TASK
        # -------------------------
        try:
            await self._queue.put(
                {
                    "prompt": prompt,
                    "kwargs": kwargs,
                    "future": future,
                    "created_at": time.monotonic(),
                }
            )
        except Exception as e:
            logger.error(f"Queue put failed: {e}")
            return {
                "response": "Queue error. Please try again.",
                "intent": "idle",
                "error": True,
            }

        # -------------------------
        # START WORKER (CRITICAL FIX)
        # -------------------------
        if not getattr(self, "_worker_running", False):
            self._worker_running = True
            self._worker_task = asyncio.create_task(
                self._queue_worker(), name="llm_queue_worker"
            )

        # -------------------------
        # WAIT FOR RESULT
        # -------------------------
        try:
            result = await asyncio.wait_for(future, timeout=timeout)

            if not isinstance(result, dict):
                raise ValueError("Invalid worker result")

            return result

        except asyncio.TimeoutError:
            if not future.done():
                future.cancel()

            return {
                "response": "Request is taking longer than expected. Please try again.",
                "intent": "idle",
                "error": False,
                "timeout": True,
            }

        except Exception as e:
            logger.error(f"queued_generate failed: {e}")
            return {
                "response": "System encountered an error. Please try again.",
                "intent": "idle",
                "error": True,
            }

    # ── Queue worker ────────────────────────────────────────────────────────
    async def _queue_worker(self) -> None:
        """Robust, single-instance queue worker with full safety"""

        # -------------------------
        # PREVENT DUPLICATE WORKERS (CRITICAL FIX)
        # -------------------------
        if getattr(self, "_worker_running", False):
            return

        self._worker_running = True

        try:
            while not getattr(self, "_shutdown", False):

                current_future: Optional[asyncio.Future] = None

                try:
                    # -------------------------
                    # GET TASK (SAFE)
                    # -------------------------
                    task: Dict = await self._queue.get()

                    prompt = task.get("prompt", "")
                    kwargs = task.get("kwargs", {})
                    current_future = task.get("future")
                    created_at = task.get("created_at", time.monotonic())

                    # -------------------------
                    # EXPIRE OLD TASKS
                    # -------------------------
                    if time.monotonic() - created_at > TASK_EXPIRY_SECONDS:
                        if current_future and not current_future.done():
                            current_future.set_result(
                                {
                                    "response": "Request expired in queue. Please try again.",
                                    "intent": "idle",
                                    "error": False,
                                    "stale": True,
                                }
                            )
                        continue

                    # -------------------------
                    # PROCESS TASK (SAFE)
                    # -------------------------
                    result = await self.safe_generate(prompt, **kwargs)

                    if current_future and not current_future.done():
                        current_future.set_result(result)

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    logger.error(f"Queue worker error: {e}")

                    if current_future and not current_future.done():
                        current_future.set_result(
                            {
                                "response": "Worker recovered from error. Please try again.",
                                "intent": "idle",
                                "error": True,
                            }
                        )

                finally:
                    # -------------------------
                    # ALWAYS MARK DONE (CRITICAL FIX)
                    # -------------------------
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass

                    # -------------------------
                    # PREVENT CPU SPIN
                    # -------------------------
                    await asyncio.sleep(0)

        finally:
            # -------------------------
            # CLEAN EXIT
            # -------------------------
            self._worker_running = False
            self._worker_task = None

    # ── Streaming ───────────────────────────────────────────────────────────
    async def generate_stream(
        self, prompt: str, timeout: float = 20.0
    ) -> AsyncIterator[str]:
        """Non-blocking, low-latency streaming with chunk optimization"""

        try:
            # -------------------------
            # GENERATE (TIMEOUT SAFE)
            # -------------------------
            data = await asyncio.wait_for(self.safe_generate(prompt), timeout=timeout)

            text = str(data.get("response", "")).strip()

            if not text:
                yield "I'm here to help. What would you like to know?"
                return

            # -------------------------
            # SMART CHUNKING (CRITICAL FIX)
            # -------------------------
            chunk = ""
            for word in text.split():
                next_chunk = f"{chunk} {word}".strip()

                if len(next_chunk) >= 40:
                    yield next_chunk
                    chunk = ""
                    await asyncio.sleep(0)  # yield control (non-blocking)
                else:
                    chunk = next_chunk

            if chunk:
                yield chunk

            # -------------------------
            # END SIGNAL
            # -------------------------
            yield ""

        except asyncio.TimeoutError:
            yield "⚠️ Request is taking longer than expected. Please try again."

        except asyncio.CancelledError:
            # graceful cancellation (no noise)
            return

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield "⚠️ Something went wrong. Please try again."

    # ── Shutdown ────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Safe, idempotent shutdown with worker stop + queue cleanup"""

        try:
            # -------------------------
            # PREVENT DUPLICATE SHUTDOWN
            # -------------------------
            if getattr(self, "_shutdown", False):
                return

            self._shutdown = True

            # -------------------------
            # CANCEL WORKER (CRITICAL FIX)
            # -------------------------
            if getattr(self, "_worker_task", None):
                if not self._worker_task.done():
                    self._worker_task.cancel()
                    await asyncio.gather(self._worker_task, return_exceptions=True)

            # -------------------------
            # FAIL PENDING QUEUE TASKS (IMPORTANT)
            # -------------------------
            while not self._queue.empty():
                try:
                    task = self._queue.get_nowait()
                    future = task.get("future")

                    if future and not future.done():
                        future.set_result(
                            {
                                "response": "System shutting down. Please retry.",
                                "intent": "idle",
                                "error": True,
                                "shutdown": True,
                            }
                        )

                    self._queue.task_done()

                except Exception:
                    break

            # -------------------------
            # CLEAN STATE
            # -------------------------
            self._worker_running = False
            self._worker_task = None

            logger.info("✅ LLMEngine shut down cleanly")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    # ── Context management ──────────────────────────────────────────────────
    def clear_context(self, force: bool = False) -> Dict[str, Any]:
        """Safe, consistent context reset with full state cleanup"""

        start = time.monotonic()

        cache_cleared = 0
        sem_cleared = 0

        try:
            # -------------------------
            # CACHE CLEAR (SAFE)
            # -------------------------
            if self.cache_enabled and hasattr(self, "cache"):
                try:
                    if force or getattr(self.cache, "cache", None):
                        cache_cleared = self.cache.clear()
                except Exception as e:
                    logger.debug(f"Cache clear failed: {e}")

            # -------------------------
            # SEMANTIC MEMORY CLEAR (SAFE)
            # -------------------------
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            if self.semantic_enabled and hasattr(self, "semantic_memory"):
                try:
                    sem_cleared = len(self.semantic_memory)
                    self.semantic_memory.clear()
<<<<<<< HEAD
                except Exception as exc:
                    logger.debug("Semantic clear failed: %s", exc)
            self.generation_count = 0
            self._last_call_time  = 0.0
            self._active_requests = 0
            elapsed = time.monotonic() - start
            logger.info(
                "Context cleared | cache=%d semantic=%d time=%.4fs",
                cache_cleared, sem_cleared, elapsed,
            )
            return {
                "status":          "cleared",
                "cache_cleared":   cache_cleared,
                "semantic_cleared":sem_cleared,
                "latency":         round(elapsed, 4),
            }
        except Exception as exc:
            logger.error("Context clear failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        self._ensure_async_primitives()
        try:
            queue_size    = self._queue.qsize() if self._queue else 0
            workers_alive = sum(1 for t in self._workers if not t.done())
            cache_size    = len(getattr(self.cache, "cache", {}))
            cache_hits    = getattr(self.cache, "hits", 0)
            cache_misses  = getattr(self.cache, "misses", 0)
            sem_size      = len(getattr(self, "semantic_memory", []))

            health = 1.0
            if self._active_requests >= _NUM_WORKERS:
                health -= 0.1
            if queue_size > 100:
                health -= 0.3
            elif queue_size > 50:
                health -= 0.15
            if workers_alive == 0 and queue_size > 0:
                health -= 0.25
            if (cache_hits + cache_misses) > 10 and cache_misses > cache_hits:
                health -= 0.1
            vram = self.gpu_guard.vram_fraction
            if vram > _VRAM_HIGH_WATERMARK:
                health -= 0.2
            health = max(0.0, round(health, 2))

            if self._active_requests > 0:
                status = "busy"
            elif workers_alive > 0:
=======
                except Exception as e:
                    logger.debug(f"Semantic clear failed: {e}")

            # -------------------------
            # RESET INTERNAL STATE (CRITICAL FIX)
            # -------------------------
            self.generation_count = 0
            self._last_call_time = 0.0

            # Optional resets for stability
            if hasattr(self, "_busy"):
                self._busy = False

            if hasattr(self, "_request_timestamps"):
                self._request_timestamps.clear()

            # -------------------------
            # METRICS RESET (SAFE)
            # -------------------------
            if hasattr(self, "metrics"):
                try:
                    self.metrics.processing_times = []
                    self.metrics.average_processing_time_ms = 0
                except Exception:
                    pass

            elapsed = time.monotonic() - start

            logger.info(
                f"🧹 Context cleared | cache={cache_cleared}, "
                f"semantic={sem_cleared}, time={elapsed:.4f}s"
            )

            return {
                "status": "cleared",
                "cache_cleared": cache_cleared,
                "semantic_cleared": sem_cleared,
                "latency": round(elapsed, 4),
            }

        except Exception as e:
            logger.error(f"Context clear failed: {e}")
            return {
                "status": "error",
                "cache_cleared": cache_cleared,
                "semantic_cleared": sem_cleared,
                "error": str(e),
            }

    # ── Metrics ─────────────────────────────────────────────────────────────
    def get_metrics(self) -> Dict[str, Any]:
        """Robust, safe metrics snapshot with improved health scoring"""

        self._ensure_async_primitives()

        try:
            # -------------------------
            # SAFE DATA COLLECTION
            # -------------------------
            queue_size = self._queue.qsize() if getattr(self, "_queue", None) else 0

            worker_active = bool(
                getattr(self, "_worker_task", None) and not self._worker_task.done()
            )

            cache_size = 0
            cache_hits = 0
            cache_misses = 0

            if getattr(self, "cache", None):
                try:
                    cache_size = len(getattr(self.cache, "cache", {}))
                    cache_hits = getattr(self.cache, "hits", 0)
                    cache_misses = getattr(self.cache, "misses", 0)
                except Exception:
                    pass

            semantic_size = len(getattr(self, "semantic_memory", []))

            index_size = 0
            if getattr(self, "index", None):
                try:
                    index_size = len(getattr(self.index, "texts", []))
                except Exception:
                    pass

            mem_size = 0
            if getattr(self, "memory_enabled", False) and getattr(self, "memory", None):
                try:
                    mem_size = self.memory.get_stats().get("total_memories", 0)
                except Exception:
                    pass

            # -------------------------
            # HEALTH SCORE (IMPROVED)
            # -------------------------
            health = 1.0

            if getattr(self, "_busy", False):
                health -= 0.1

            if queue_size > 20:
                health -= 0.3
            elif queue_size > 10:
                health -= 0.15

            if not worker_active and queue_size > 0:
                health -= 0.25

            if (cache_hits + cache_misses) > 10:
                if cache_misses > cache_hits:
                    health -= 0.1

            health = max(0.0, round(health, 2))

            # -------------------------
            # STATUS
            # -------------------------
            if getattr(self, "_busy", False):
                status = "busy"
            elif worker_active:
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                status = "running"
            else:
                status = "idle"

            return {
<<<<<<< HEAD
                "status":          status,
                "generations":     self.generation_count,
                "active_requests": self._active_requests,
                "queue_size":      queue_size,
                "workers_alive":   workers_alive,
                "total_workers":   _NUM_WORKERS,
                "avg_load":        round(min(1.0, queue_size / max(1, MAX_QUEUE_SIZE)), 2),
                "cache_size":      cache_size,
                "cache_hits":      cache_hits,
                "cache_misses":    cache_misses,
                "semantic_memory": sem_size,
                "vram_fraction":   round(vram, 3),
                "health_score":    health,
            }
        except Exception as exc:
            logger.error("Metrics error: %s", exc)
            return {"status": "error", "health_score": 0.0}

    # ── Initialization ────────────────────────────────────────────────────────

    async def initialize(self, timeout: float = 30.0, *args, **kwargs) -> bool:
        self._ensure_async_primitives()

        if getattr(self, "_initialized", False):
            logger.info("LLM already initialized")
=======
                "status": status,
                "generations": getattr(self, "generation_count", 0),
                "queue_size": queue_size,
                "worker_active": worker_active,
                "avg_load": round(min(1.0, queue_size / 10), 2),
                "cache_size": cache_size,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "semantic_memory": semantic_size,
                "vector_index": index_size,
                "persistent_memory": mem_size,
                "health_score": health,
            }

        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {
                "status": "error",
                "health_score": 0.0,
            }

    # ── Initialization ──────────────────────────────────────────────────────
    async def initialize(self, timeout: float = 30.0, *args, **kwargs) -> bool:
        """Safe, idempotent initialization with robust warmup + fallback"""

        self._ensure_async_primitives()

        # -------------------------
        # PREVENT DUPLICATE INIT (CRITICAL FIX)
        # -------------------------
        if getattr(self, "_initialized", False):
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            return True

        start = time.monotonic()

<<<<<<< HEAD
        if not getattr(self, "provider", None):
            logger.error("No provider available")
            return False

        provider_name = self.provider.__class__.__name__
        logger.info("Provider: %s", provider_name)

        skip_warmup = False
        try:
            if hasattr(self.provider, "model") and self.provider.model:
                model_path = getattr(self.provider.model, "model_path", "")
                if "f16" in str(model_path).lower():
                    skip_warmup = True
        except Exception:
            pass

        if os.environ.get("EDIATH_SKIP_WARMUP", "0") == "1":
            skip_warmup = True

        if not skip_warmup:
            logger.info("Running LLM warmup…")
            try:
                raw = await asyncio.wait_for(
                    self.provider.generate("Hello", max_tokens=4, temperature=0.0, retries=0),
                    timeout=min(5.0, timeout),
                )
                if not str(raw).strip():
                    raise RuntimeError("Warmup produced empty response")
                logger.info("Warmup successful")
            except asyncio.TimeoutError:
                logger.warning("Warmup timeout")
            except Exception as exc:
                logger.warning("Warmup failed: %s", exc)
        else:
            logger.warning("Warmup skipped (large model / low VRAM)")

        # Start queue workers
        self._start_workers()

        self._initialized = True
        elapsed = time.monotonic() - start
        logger.info("LLMEngine initialized in %.2fs (%d workers)", elapsed, _NUM_WORKERS)
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Agent Pool  — manages up to MAX_AGENTS=100 concurrent agent slots
# ══════════════════════════════════════════════════════════════════════════════

class AgentPool:
    """
    Registers and manages up to 100 agent slots on top of a shared LLMEngine.

    Usage
    -----
    pool  = AgentPool(engine)
    slot  = pool.register("agent_42", priority=AgentPriority.HIGH)
    result= await pool.generate("agent_42", "What is the capital of France?")
    stats = pool.get_agent_metrics("agent_42")
    await pool.broadcast("System maintenance in 5 minutes")
    pool.pause("agent_42")
    pool.retire("agent_42")
    """

    def __init__(self, engine: LLMEngine) -> None:
        self.engine: LLMEngine = engine
        self._slots:  Dict[str, AgentSlot] = {}
        self._lock    = asyncio.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    async def register(
        self,
        agent_id:   str,
        priority:   AgentPriority = AgentPriority.NORMAL,
        rate_limit: float         = _AGENT_RATE_LIMIT_RPS,
        max_queue:  int           = 10,
    ) -> AgentSlot:
        """Register a new agent. Raises ValueError if pool is full or ID is duplicate."""
        async with self._lock:
            if agent_id in self._slots:
                existing = self._slots[agent_id]
                if existing.state != AgentState.RETIRED:
                    raise ValueError(f"Agent '{agent_id}' already registered")
                del self._slots[agent_id]   # allow re-registration after retire

            active = sum(
                1 for s in self._slots.values() if s.state != AgentState.RETIRED
            )
            if active >= MAX_AGENTS:
                raise RuntimeError(
                    f"AgentPool full ({MAX_AGENTS} active agents). "
                    "Retire unused agents before registering new ones."
                )

            slot = AgentSlot(
                agent_id   = agent_id,
                priority   = priority,
                rate_limit = rate_limit,
                max_queue  = max_queue,
            )
            self._slots[agent_id] = slot
            logger.info(
                "AgentPool: registered '%s' (priority=%s, rps=%.1f) [%d/%d active]",
                agent_id, priority.name, rate_limit, active + 1, MAX_AGENTS,
            )
            return slot

    def unregister(self, agent_id: str) -> bool:
        """Alias for retire()."""
        return self.retire(agent_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def pause(self, agent_id: str) -> bool:
        slot = self._slots.get(agent_id)
        if slot and slot.state == AgentState.ACTIVE:
            slot.state = AgentState.PAUSED
            logger.info("AgentPool: paused '%s'", agent_id)
            return True
        return False

    def resume(self, agent_id: str) -> bool:
        slot = self._slots.get(agent_id)
        if slot and slot.state == AgentState.PAUSED:
            slot.state = AgentState.IDLE
            logger.info("AgentPool: resumed '%s'", agent_id)
            return True
        return False

    def retire(self, agent_id: str) -> bool:
        slot = self._slots.get(agent_id)
        if slot:
            slot.state = AgentState.RETIRED
            logger.info("AgentPool: retired '%s'", agent_id)
            return True
        return False

    # ── Generation ────────────────────────────────────────────────────────────

    async def generate(
        self,
        agent_id: str,
        prompt:   str,
        timeout:  float = 30.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Submit a generation request for *agent_id*.
        Enforces per-agent rate limiting, priority routing, and lifecycle checks.
        """
        slot = self._slots.get(agent_id)
        if slot is None:
            return {
                "response":   f"Agent '{agent_id}' is not registered.",
                "intent":     "error",
                "confidence": 0.0,
                "agent_id":   agent_id,
            }

        if slot.state == AgentState.RETIRED:
            return {
                "response":   f"Agent '{agent_id}' has been retired.",
                "intent":     "error",
                "confidence": 0.0,
                "agent_id":   agent_id,
            }

        if slot.state == AgentState.PAUSED:
            return {
                "response":   f"Agent '{agent_id}' is paused.",
                "intent":     "idle",
                "confidence": 0.0,
                "agent_id":   agent_id,
            }

        # Per-agent rate limiting
        allowed = await slot.acquire()
        if not allowed:
            return {
                "response":   "Rate limit reached. Please slow down.",
                "intent":     "idle",
                "confidence": 0.0,
                "reflection": "rate_limited",
                "agent_id":   agent_id,
            }

        start = time.monotonic()
        try:
            result = await self.engine.agent_generate(
                agent_id = agent_id,
                prompt   = prompt,
                priority = slot.priority,
                timeout  = timeout,
                **kwargs,
            )
            latency_ms = (time.monotonic() - start) * 1000
            await slot.release(success=True, latency_ms=latency_ms)
            return result
        except Exception as exc:
            await slot.release(success=False)
            logger.error("AgentPool generate error for '%s': %s", agent_id, exc)
            return {
                "response":   f"Error processing request: {exc}",
                "intent":     "error",
                "confidence": 0.0,
                "agent_id":   agent_id,
            }

    # ── Broadcast ─────────────────────────────────────────────────────────────

    async def broadcast(
        self,
        prompt:   str,
        priority: AgentPriority = AgentPriority.HIGH,
        timeout:  float         = 60.0,
    ) -> Dict[str, Dict]:
        """
        Send the same prompt to ALL active agents concurrently.
        Returns a dict of {agent_id: result}.
        """
        active_ids = [
            aid for aid, slot in self._slots.items()
            if slot.state not in (AgentState.PAUSED, AgentState.RETIRED)
        ]
        tasks = {
            aid: asyncio.create_task(self.generate(aid, prompt, timeout=timeout))
            for aid in active_ids
        }
        results: Dict[str, Dict] = {}
        for aid, task in tasks.items():
            try:
                results[aid] = await task
            except Exception as exc:
                results[aid] = {"response": str(exc), "intent": "error"}
        return results

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        slot = self._slots.get(agent_id)
        return slot.get_metrics() if slot else None

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all slots."""
        active = retired = paused = 0
        total_sent = total_completed = total_failed = total_throttled = 0

        for slot in self._slots.values():
            if slot.state == AgentState.RETIRED:
                retired += 1
            elif slot.state == AgentState.PAUSED:
                paused += 1
            else:
                active += 1
            total_sent      += slot.requests_sent
            total_completed += slot.requests_completed
            total_failed    += slot.requests_failed
            total_throttled += slot.requests_throttled

        engine_metrics = self.engine.get_metrics()

        return {
            "pool": {
                "registered":        len(self._slots),
                "active":            active,
                "paused":            paused,
                "retired":           retired,
                "capacity":          MAX_AGENTS,
                "utilization":       round(active / MAX_AGENTS, 3),
            },
            "requests": {
                "total_sent":        total_sent,
                "total_completed":   total_completed,
                "total_failed":      total_failed,
                "total_throttled":   total_throttled,
                "success_rate":      round(
                    total_completed / max(1, total_sent), 3
                ),
            },
            "engine":              engine_metrics,
            "gpu": {
                "vram_fraction":     round(self.engine.gpu_guard.vram_fraction, 3),
                "under_pressure":    self.engine.gpu_guard.is_under_pressure(),
            },
        }

    def list_agents(self, state_filter: Optional[AgentState] = None) -> List[str]:
        """Return agent IDs, optionally filtered by state."""
        return [
            aid for aid, slot in self._slots.items()
            if state_filter is None or slot.state == state_filter
        ]


# ── Public API ────────────────────────────────────────────────────────────────
=======
        if not self.provider:
            logger.error("❌ LLM not available")
            return False

        logger.info("🧪 Running LLM warmup…")

        try:
            # -------------------------
            # SAFE WARMUP CALL
            # -------------------------
            raw = await asyncio.wait_for(
                self.provider.generate(
                    "[INST] Hello [/INST]",
                    max_tokens=10,
                    temperature=0.1,
                ),
                timeout=min(10.0, timeout),
            )

            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not raw or len(str(raw).strip()) < 2:
                raise ValueError("Warmup returned empty output")

            logger.info("✅ LLM warmup successful")

        except asyncio.TimeoutError:
            logger.warning("⚠ Warmup timed out — continuing anyway")

        except Exception as e:
            logger.error(f"❌ Warmup failed: {e}")

            # -------------------------
            # SAFE FALLBACK SWITCH
            # -------------------------
            if getattr(self, "fallback_provider", None):
                self.provider = self.fallback_provider
                logger.warning("⚠ Switched to FallbackProvider")
            else:
                logger.error("❌ No fallback provider available")
                return False

        # -------------------------
        # FINALIZE INIT
        # -------------------------
        self._initialized = True

        elapsed = time.monotonic() - start
        logger.info(f"✅ LLMEngine ready ({elapsed:.2f}s)")

        return True


# ── Public API ─────────────────────────────────────────────────────────────
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
__all__ = [
    "LLMProvider",
    "LLMEngine",
    "GGUFProvider",
    "FallbackProvider",
    "LLMCache",
    "EmbeddingEngine",
    "VectorIndex",
    "RetrievalEngine",
    "SummarizationEngine",
    "CompressionEngine",
<<<<<<< HEAD
    "AgentPool",
    "AgentSlot",
    "AgentPriority",
    "AgentState",
    "GPUMemoryGuard",
    "PrioritizedRequest",
]
=======
]
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
