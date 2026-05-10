from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.utils.task_registry import task_registry

from ..utils.logger import logger

# ── Safe imports ───────────────────────────────────────────────────────────
try:
    from ..brain.llm_engine import LLMEngine
except ImportError:
    LLMEngine = None  # type: ignore

try:
    from ..memory.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None  # type: ignore

try:
    from ..vision.vision_engine import VisionEngine
except ImportError:
    VisionEngine = None  # type: ignore

try:
    from ..autonomy.curiosity_engine import CuriosityEngine
except ImportError:
    CuriosityEngine = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════


class DecisionType(Enum):
    ACTION = "action"
    ROUTE = "route"
    PARAMETER = "parameter"
    PRIORITY = "priority"


class SuggestionType(Enum):
    TASK_SUGGESTION = "task_suggestion"
    IMPROVEMENT = "improvement"
    EXPLORATION = "exploration"
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    ALERT = "alert"


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CuriositySuggestion:
    """A suggestion produced by the curiosity engine."""

    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    suggestion_type: SuggestionType = field(default=SuggestionType.TASK_SUGGESTION)
    title: str = ""
    description: str = ""
    confidence: float = 0.5
    priority: int = 1  # 1=highest … 5=lowest
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_taken: bool = False
    feedback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "suggestion_type": self.suggestion_type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "priority": self.priority,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "is_taken": self.is_taken,
        }


@dataclass
class Option:
    """A single option inside a Decision."""

    name: str
    score: float = 0.0
    # FIX #4 / #21 – use field(default=None) for mutable defaults
    pros: Optional[List[str]] = field(default=None)
    cons: Optional[List[str]] = field(default=None)
    confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self.pros is None:
            self.pros = []
        if self.cons is None:
            self.cons = []
        if self.metadata is None:
            self.metadata = {}


class Decision:
    """Represents a single decision with options and evaluation state."""

    def __init__(self, decision_type: DecisionType, context: str) -> None:
        self.id = str(uuid.uuid4())
        self.decision_type = decision_type
        self.context = context
        self.options: List[Option] = []
        self.selected_option: Optional[Option] = None
        self.rationale: str = ""
        self.confidence: float = 0.0
        self.params: Dict[str, Any] = {}
        self.execution_status: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.source: str = "ai"
        self.metadata: Dict[str, Any] = {}

    def add_option(self, option: Option) -> None:
        self.options.append(option)

    def evaluate_options(self) -> Option:
        if not self.options:
            raise ValueError("No options available to evaluate")
        best = max(self.options, key=lambda o: o.score)
        self.selected_option = best
        self.confidence = best.confidence
        self.rationale = f"Selected '{best.name}' (score={best.score:.3f})"
        return best


# ══════════════════════════════════════════════════════════════════════════════
# DecisionEngine
# ══════════════════════════════════════════════════════════════════════════════


class DecisionEngine:
    """
    Core decision-making engine with LLM integration, curiosity suggestions,
    and vision-context processing.

    Typical lifecycle::

        engine = DecisionEngine()
        await engine.start()          # starts background workers
        decision = engine.make_decision(DecisionType.ACTION, "context", ...)
        name, conf = engine.evaluate_decision(decision)
        result = await engine.execute({"action": name, "confidence": conf}, "context")
        await engine.shutdown()
    """

    # Enhancement C – single source of truth for allowed actions
    ALLOWED_ACTIONS = frozenset(
        {
            "think",
            "explore",
            "respond",
            "code",
            "system_status",
            "read_memory",
            "learn",
            "get_suggestions",
            "get_next_suggestion",
        }
    )

    def __init__(self) -> None:
        # FIX #2 – self.logger must be set before any method can use it
        self.logger = logger

        self.decisions: Dict[str, Decision] = {}
        self.decision_count: int = 0  # FIX #20

        # Rate limiting
        self._last_decision_time: float = 0.0
        self._last_vision_time: float = 0.0
        self._last_vision_context: str = ""

        # Curiosity integration
        self.curiosity_engine: Optional[Any] = None
        self._active_suggestions: List[CuriositySuggestion] = []
        self._suggestion_history: List[CuriositySuggestion] = []
        self._last_suggestion_time: float = 0.0
        self._suggestion_cooldown: float = 10.0

        # Task / activity tracking
        self.current_task: Optional[str] = None
        self.task_history: List[Dict[str, Any]] = []
        self.user_activity_log: List[Dict[str, Any]] = []

        # ── Component initialisation ───────────────────────────────────────
        self.llm: Optional[Any] = None
        try:
            if LLMEngine:
                self.llm = LLMEngine()
        except Exception as e:
            self.logger.warning(f"LLM init failed: {e}")

        self.memory: Optional[Any] = None
        try:
            if MemoryManager:
                self.memory = MemoryManager()
        except Exception as e:
            self.logger.warning(f"Memory init failed: {e}")

        self.vision: Optional[Any] = None
        try:
            if VisionEngine:
                self.vision = VisionEngine()
        except Exception as e:
            self.logger.warning(f"Vision init failed: {e}")

        try:
            if CuriosityEngine:
                self.curiosity_engine = CuriosityEngine()
                self.logger.info("CuriosityEngine integrated")
        except Exception as e:
            self.logger.warning(f"CuriosityEngine init failed: {e}")

        # Alias for compatibility with callers that use llm_engine
        self.llm_engine = self.llm

<<<<<<< HEAD
    def set_llm_engine(self, llm_engine: Any) -> None:
        """Inject the shared LLM engine instance (single source of truth)."""
        if llm_engine is None:
            return
        self.llm = llm_engine
        self.llm_engine = llm_engine


=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        # Queue / worker (created in start())
        self._decision_queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._curiosity_task: Optional[asyncio.Task] = None  # FIX #22
        self._running: bool = False

    # ── Repr (Enhancement A) ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<DecisionEngine decisions={self.decision_count} "
            f"llm={'yes' if self.llm else 'no'} "
            f"curiosity={'yes' if self.curiosity_engine else 'no'}>"
        )

    # ── Startup / Shutdown ────────────────────────────────────────────────

    async def start(self) -> None:
        """Safe, idempotent startup for DecisionEngine workers"""

        # -------------------------
        # PREVENT DUPLICATE START (CRITICAL FIX)
        # -------------------------
        if getattr(self, "_running", False):
            return

        self._running = True

        try:
            # -------------------------
            # SAFE QUEUE INIT
            # -------------------------
            if not getattr(self, "_decision_queue", None):
                self._decision_queue = asyncio.Queue()

            # -------------------------
            # START MAIN WORKER (SAFE)
            # -------------------------
            if not getattr(self, "_worker_task", None) or self._worker_task.done():
                self._worker_task = asyncio.create_task(self._process_queue())

            # -------------------------
            # START CURIOSITY WORKER (SAFE)
            # -------------------------
            if self.curiosity_engine:
                if (
                    not getattr(self, "_curiosity_task", None)
                    or self._curiosity_task.done()
                ):
                    self._curiosity_task = asyncio.create_task(
                        self._curiosity_background_loop()
                    )

            self.logger.info("✅ DecisionEngine workers started")

        except Exception as e:
            # -------------------------
            # FAIL-SAFE RESET (CRITICAL)
            # -------------------------
            self._running = False
            self.logger.error(f"Failed to start DecisionEngine: {e}")

    async def shutdown(self) -> None:
        """Graceful, idempotent shutdown of DecisionEngine workers"""

        # -------------------------
        # PREVENT DUPLICATE SHUTDOWN
        # -------------------------
        if not getattr(self, "_running", False):
            return

        self._running = False

        try:
            tasks = []

            # -------------------------
            # COLLECT ACTIVE TASKS
            # -------------------------
            for task in (
                getattr(self, "_worker_task", None),
                getattr(self, "_curiosity_task", None),
            ):
                if task and not task.done():
                    task.cancel()
                    tasks.append(task)

            # -------------------------
            # AWAIT ALL TASKS SAFELY
            # -------------------------
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # -------------------------
            # DRAIN QUEUE (CRITICAL FIX)
            # -------------------------
            if getattr(self, "_decision_queue", None):
                while not self._decision_queue.empty():
                    try:
                        self._decision_queue.get_nowait()
                        self._decision_queue.task_done()
                    except Exception:
                        break

            # -------------------------
            # CLEAN STATE
            # -------------------------
            self._worker_task = None
            self._curiosity_task = None

            self.logger.info("✅ DecisionEngine shut down cleanly")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

    # ── Queue worker ──────────────────────────────────────────────────────

    async def _process_queue(self) -> None:
        """Robust background worker for processing decisions safely"""

        while getattr(self, "_running", False):
            decision = None

            try:
                # -------------------------
                # GET TASK (CANCELLATION SAFE)
                # -------------------------
                decision = await self._decision_queue.get()

                try:
                    # -------------------------
                    # EXECUTE DECISION
                    # -------------------------
                    result = await self._execute_decision(decision)

                    if isinstance(result, dict) and result.get("status") == "success":
                        decision.execution_status = "completed"
                    else:
                        decision.execution_status = "failed"

                    decision.error = (
                        result.get("error") if isinstance(result, dict) else None
                    )

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    # -------------------------
                    # ERROR HANDLING (CRITICAL FIX)
                    # -------------------------
                    self.logger.error(f"Queue worker error: {e}")
                    decision.execution_status = "error"
                    decision.error = str(e)

                finally:
                    # -------------------------
                    # ALWAYS UPDATE STATE (CRITICAL FIX)
                    # -------------------------
                    if decision:
                        decision.updated_at = datetime.now()
                        try:
                            self.decisions[decision.id] = decision
                        except Exception:
                            pass

                    # -------------------------
                    # ALWAYS MARK DONE (CRITICAL FIX)
                    # -------------------------
                    try:
                        self._decision_queue.task_done()
                    except Exception:
                        pass

            except asyncio.CancelledError:
                break

            except Exception as e:
                # -------------------------
                # LOOP SAFETY
                # -------------------------
                self.logger.error(f"Unexpected queue loop error: {e}")
                await asyncio.sleep(0)  # prevent tight error loop

    async def _execute_decision(self, decision: Decision) -> Dict[str, Any]:
        """Safe decision execution with validation, timeout, and extensibility"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not decision:
                return {
                    "status": "failed",
                    "error": "Invalid decision object",
                }

            # -------------------------
            # EXECUTION (EXTENSIBLE)
            # -------------------------
            if hasattr(self, "decision_handler") and callable(self.decision_handler):
                try:
                    result = await asyncio.wait_for(
                        self.decision_handler(decision), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    return {
                        "status": "failed",
                        "error": "Decision execution timeout",
                    }
            else:
                # Default fallback behavior
                result = {"status": "success", "result": "Executed"}

            # -------------------------
            # RESULT VALIDATION
            # -------------------------
            if not isinstance(result, dict):
                return {
                    "status": "failed",
                    "error": "Invalid execution result format",
                }

            return result

        except asyncio.CancelledError:
            raise

        except Exception as e:
            # -------------------------
            # ERROR ISOLATION (CRITICAL FIX)
            # -------------------------
            return {
                "status": "error",
                "error": str(e),
            }

    # ── Curiosity background loop ─────────────────────────────────────────

    async def _curiosity_background_loop(self) -> None:
        """Safe curiosity loop with throttling, limits, and shutdown awareness"""

        try:
            while (
                getattr(self, "_running", False)
                and not task_registry.is_shutting_down()
            ):

                try:
                    # -------------------------
                    # VALIDATION (CRITICAL FIX)
                    # -------------------------
                    if not self.current_task or not self.curiosity_engine:
                        continue

                    # -------------------------
                    # GENERATE SUGGESTIONS (SAFE)
                    # -------------------------
                    suggestions = await self.get_task_suggestions(self.current_task)

                    if not suggestions or not self._can_send_suggestion():
                        continue

                    self._last_suggestion_time = time.time()

                    # -------------------------
                    # LIMIT GROWTH (CRITICAL FIX)
                    # -------------------------
                    for s in suggestions[:2]:
                        try:
                            if len(self._active_suggestions) >= 50:
                                self._active_suggestions.pop(0)

                            self._active_suggestions.append(s)

                            self.logger.info(
                                f"New suggestion: {getattr(s, 'title', 'unknown')!r} "
                                f"(conf={getattr(s, 'confidence', 0.0):.2f})"
                            )

                        except Exception:
                            continue

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    # -------------------------
                    # ERROR ISOLATION
                    # -------------------------
                    self.logger.debug(f"Curiosity background error: {e}")

                finally:
                    # -------------------------
                    # CONTROL LOOP SPEED (CRITICAL FIX)
                    # -------------------------
                    try:
                        await asyncio.sleep(15)
                    except asyncio.CancelledError:
                        break

        finally:
            # -------------------------
            # CLEAN EXIT
            # -------------------------
            self.logger.debug("Curiosity loop stopped")

    def _can_send_suggestion(self) -> bool:
        """Return True if the cooldown has elapsed and the queue isn't full."""
        if time.time() - self._last_suggestion_time < self._suggestion_cooldown:
            return False
        return len(self._active_suggestions) <= 3

    # ── Suggestion helpers ────────────────────────────────────────────────

    async def get_task_suggestions(
        self, task: str, limit: int = 3
    ) -> List[CuriositySuggestion]:
        """Safe suggestion generator with validation, fallback, and bounded history"""

        if not task or not isinstance(task, str):
            return []

        if not self.curiosity_engine:
            return []

        suggestions: List[CuriositySuggestion] = []

        try:
            # -------------------------
            # GET ANALYSIS (SAFE)
            # -------------------------
            analysis = None

            if hasattr(self.curiosity_engine, "analyze_task"):
                analysis = await self.curiosity_engine.analyze_task(task)

            elif hasattr(self.curiosity_engine, "suggest_next_action"):
                analysis = await self.curiosity_engine.suggest_next_action(task)

            else:
                analysis = await self._generate_llm_suggestions(task)

            # -------------------------
            # VALIDATE ANALYSIS (CRITICAL FIX)
            # -------------------------
            if not isinstance(analysis, (list, tuple)):
                analysis = []

            # -------------------------
            # BUILD SUGGESTIONS (SAFE)
            # -------------------------
            for i, sd in enumerate(analysis[: max(1, limit)]):
                try:
                    if not isinstance(sd, dict):
                        continue

                    suggestions.append(
                        CuriositySuggestion(
                            suggestion_type=self._determine_suggestion_type(sd),
                            title=str(sd.get("title", f"Suggestion {i + 1}"))[:120],
                            description=str(sd.get("description", ""))[:300],
                            confidence=max(
                                0.0, min(1.0, float(sd.get("confidence", 0.6)))
                            ),
                            priority=max(1, min(5, int(sd.get("priority", 3)))),
                            context={
                                "task": task,
                                "source": "curiosity_engine",
                            },
                        )
                    )

                except Exception:
                    continue

            # -------------------------
            # SAFE HISTORY TRACKING (CRITICAL FIX)
            # -------------------------
            try:
                if not hasattr(self, "task_history"):
                    self.task_history = []

                self.task_history.append(
                    {
                        "task": task[:200],
                        "suggestions": len(suggestions),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # bounded history
                if len(self.task_history) > 50:
                    self.task_history = self.task_history[-50:]

            except Exception:
                pass

        except asyncio.CancelledError:
            raise

        except Exception as e:
            self.logger.warning(f"Failed to get task suggestions: {e}")

        return suggestions

    async def _generate_llm_suggestions(self, task: str) -> List[Dict[str, Any]]:
        """Robust LLM-based suggestion generator with safe parsing"""

        if not getattr(self, "llm", None):
            return []

        task = str(task or "").strip()
        if not task:
            return []

        # -------------------------
        # SAFE PROMPT (INJECTION RESISTANT)
        # -------------------------
        safe_task = task.replace("[", "").replace("]", "").strip()

        prompt = (
            f"User task: {safe_task}\n\n"
            "Generate 2-3 helpful suggestions.\n"
            "Return ONLY valid JSON (no text, no explanation):\n"
            '[{"title":"...","description":"...","confidence":0.8,"priority":2}]'
        )

        try:
            # -------------------------
            # TIMEOUT SAFE CALL
            # -------------------------
            result = await asyncio.wait_for(self.llm.generate(prompt), timeout=10)

            if not isinstance(result, dict):
                return []

            response = str(result.get("response", "")).strip()

            if not response:
                return []

            # -------------------------
            # SAFE JSON EXTRACTION (CRITICAL FIX)
            # -------------------------
            matches = re.findall(r"\[[\s\S]*?\]", response)

            for m in matches:
                try:
                    parsed = json.loads(m)

                    if isinstance(parsed, list):
                        # validate entries
                        clean = []
                        for item in parsed:
                            if not isinstance(item, dict):
                                continue

                            clean.append(
                                {
                                    "title": str(item.get("title", ""))[:120],
                                    "description": str(item.get("description", ""))[
                                        :300
                                    ],
                                    "confidence": max(
                                        0.0,
                                        min(1.0, float(item.get("confidence", 0.6))),
                                    ),
                                    "priority": max(
                                        1, min(5, int(item.get("priority", 3)))
                                    ),
                                }
                            )

                        if clean:
                            return clean[:3]

                except Exception:
                    continue

        except asyncio.TimeoutError:
            self.logger.warning("LLM suggestion generation timed out")

        except Exception as e:
            self.logger.debug(f"LLM suggestion generation failed: {e}")

        return []

    def _determine_suggestion_type(self, data: Dict[str, Any]) -> SuggestionType:
        """Map a raw suggestion dict to a SuggestionType enum value."""
        mapping = {
            "improvement": SuggestionType.IMPROVEMENT,
            "exploration": SuggestionType.EXPLORATION,
            "optimization": SuggestionType.OPTIMIZATION,
            "learning": SuggestionType.LEARNING,
            "alert": SuggestionType.ALERT,
        }
        return mapping.get(data.get("type", "").lower(), SuggestionType.TASK_SUGGESTION)

    async def get_next_suggestion(self) -> Optional[CuriositySuggestion]:
        """Return the highest-priority pending suggestion (lowest priority number)."""
        if not self._active_suggestions:
            return None
        self._active_suggestions.sort(key=lambda s: s.priority)
        return self._active_suggestions[0]

    async def take_suggestion(
        self, suggestion_id: str, feedback: Optional[str] = None
    ) -> bool:
        """Mark a suggestion as taken and move it to history."""
        # FIX #14 – avoid mutating list during iteration
        for suggestion in list(self._active_suggestions):
            if suggestion.suggestion_id == suggestion_id:
                suggestion.is_taken = True
                suggestion.feedback = feedback
                self._active_suggestions.remove(suggestion)
                self._suggestion_history.append(suggestion)

                if len(self._suggestion_history) > 100:
                    self._suggestion_history = self._suggestion_history[-100:]

                if self.curiosity_engine and hasattr(
                    self.curiosity_engine, "record_feedback"
                ):
                    await self.curiosity_engine.record_feedback(suggestion_id, feedback)

                self.logger.info(f"Suggestion taken: {suggestion.title!r}")
                return True
        return False

    async def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion without acting on it."""
        # FIX #15 – avoid mutating list during iteration
        for suggestion in list(self._active_suggestions):
            if suggestion.suggestion_id == suggestion_id:
                self._active_suggestions.remove(suggestion)
                self.logger.debug(f"Suggestion dismissed: {suggestion.title!r}")
                return True
        return False

    async def update_current_task(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the current task and optionally trigger fresh suggestions."""
        old_task = self.current_task
        self.current_task = task

        self.user_activity_log.append(
            {
                "old_task": old_task,
                "new_task": task,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self.user_activity_log) > 100:
            self.user_activity_log = self.user_activity_log[-100:]

        if old_task != task and self.curiosity_engine:
            suggestions = await self.get_task_suggestions(task, limit=1)
            if suggestions and self._can_send_suggestion():
                self._active_suggestions.extend(suggestions)
                self._last_suggestion_time = time.time()

        self.logger.info(f"Current task updated: {task!r}")

    async def record_user_action(self, action: str, result: Any = None) -> None:
        """Record a user action for curiosity-engine learning."""
        if self.curiosity_engine and hasattr(self.curiosity_engine, "record_action"):
            await self.curiosity_engine.record_action(action, result)

        self.user_activity_log.append(
            {
                "action": action,
                "result": str(result)[:200] if result is not None else None,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def get_active_suggestions(self) -> List[Dict[str, Any]]:
        """Return all active suggestions as serialisable dicts."""
        return [s.to_dict() for s in self._active_suggestions]

    async def get_suggestion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the last *limit* taken/dismissed suggestions."""
        return [s.to_dict() for s in self._suggestion_history[-limit:]]

    # ── Vision processing ─────────────────────────────────────────────────

    async def process_vision(self, vision_data: dict) -> Optional[str]:
        """
        Safe, structured vision processing with throttling, deduplication, and clean output.
        Always returns a VISUAL string.
        """

        now = time.time()

        # -------------------------
        # LOG INPUT
        # -------------------------
        try:
            self.logger.info(f"👁 Vision input: {vision_data}")
        except Exception:
            pass

        # -------------------------
        # THROTTLING (CRITICAL FIX)
        # -------------------------
        if now - getattr(self, "_last_vision_time", 0) < 0.6:
            context = (
                f"VISUAL: {self._last_vision_context}"
                if getattr(self, "_last_vision_context", None)
                else "VISUAL: Scene unchanged."
            )
            return context

        self._last_vision_time = now

        try:
            # -------------------------
            # VALIDATION
            # -------------------------
            if not isinstance(vision_data, dict):
                return "VISUAL: Camera input unavailable."

            detections = vision_data.get("detections") or []
            caption = vision_data.get("caption")
            faces = vision_data.get("faces") or []

            # -------------------------
            # OBJECT EXTRACTION (SAFE)
            # -------------------------
            objects = []
            for d in detections[:5]:
                try:
                    cls = (
                        d.get("class")
                        if isinstance(d, dict)
                        else getattr(d, "class_name", None)
                    )
                    if cls:
                        name = str(cls).strip().lower()
                        if len(name) > 1 and name not in objects:
                            objects.append(name)
                except Exception:
                    continue

            obj_text = ", ".join(objects) if objects else None

            # -------------------------
            # FACE EXTRACTION (SAFE)
            # -------------------------
            face_names = []
            for f in faces[:3]:
                try:
                    n = f.get("name") if isinstance(f, dict) else None
                    if n and str(n).lower() != "unknown":
                        face_names.append(str(n))
                except Exception:
                    continue

            face_text = ", ".join(face_names) if face_names else None

            # -------------------------
            # CAPTION CLEANING (CRITICAL FIX)
            # -------------------------
            caption_text = ""

            try:
                if hasattr(caption, "text"):
                    caption_text = str(caption.text)
                elif isinstance(caption, dict):
                    caption_text = str(caption.get("text", ""))
                elif isinstance(caption, str):
                    caption_text = caption
            except Exception:
                pass

            # remove garbage patterns
            if "<" in caption_text and "object at" in caption_text:
                caption_text = ""

            caption_text = re.sub(r"\s+", " ", caption_text).strip()[:120]

            # -------------------------
            # BUILD CONTEXT (CLEAN)
            # -------------------------
            parts = []

            if face_text:
                parts.append(f"I can see {face_text}".rstrip("."))

            if obj_text:
                prefix = "and " if face_text else "I can see "
                parts.append(f"{prefix}{obj_text}".rstrip("."))

            if caption_text:
                parts.append(caption_text.rstrip("."))

            context = (
                ". ".join(parts)
                if parts
                else "I am observing the environment but cannot detect clear details."
            )

            context = context.strip()[:200]

            # -------------------------
            # DUPLICATE SUPPRESSION (IMPROVED)
            # -------------------------
            if context == getattr(self, "_last_vision_context", None):
                if obj_text:
                    return f"VISUAL: Still seeing {obj_text}."
                return "VISUAL: Scene unchanged."

            # -------------------------
            # SAVE STATE
            # -------------------------
            self._last_vision_context = context

            output = f"VISUAL: {context}"

            try:
                self.logger.info(f"🔍 Vision output: {output}")
            except Exception:
                pass

            return output

        except Exception as e:
            # -------------------------
            # ERROR SAFETY
            # -------------------------
            try:
                self.logger.debug(f"Vision processing error: {e}")
            except Exception:
                pass

            return "VISUAL: Vision processing failed."

    # ── Retry helper ──────────────────────────────────────────────────────

    async def _retry(
        self, func, *args, retries: int = 2, timeout: float = 10.0, **kwargs
    ) -> Any:
        """Robust retry wrapper with timeout, backoff, and safe fallback"""

        last_error = None

        for attempt in range(retries + 1):
            try:
                # -------------------------
                # SUPPORT SYNC + ASYNC (CRITICAL FIX)
                # -------------------------
                result = func(*args, **kwargs)

                if asyncio.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=timeout)

                return result

            except asyncio.TimeoutError:
                last_error = "timeout"
                self.logger.warning(f"Retry timeout (attempt {attempt + 1})")

            except asyncio.CancelledError:
                raise

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")

            # -------------------------
            # EXPONENTIAL BACKOFF (CRITICAL FIX)
            # -------------------------
            if attempt < retries:
                await asyncio.sleep(min(2.0, 0.3 * (2**attempt)))

        # -------------------------
        # SAFE FALLBACK (CRITICAL FIX)
        # -------------------------
        return {"status": "failed", "error": last_error or "unknown_error"}

    # ── Intelligent decision (LLM) ────────────────────────────────────────

    async def intelligent_decision(self, context: str) -> Dict[str, Any]:
        """Robust decision engine with safe LLM usage, parsing, and fallbacks"""

        if not getattr(self, "llm", None):
            return {
                "action": "respond",
                "confidence": 0.5,
                "reasoning": "LLM unavailable",
                "params": {},
            }

        # -------------------------
        # RATE LIMIT (NON-BLOCKING SAFE)
        # -------------------------
        now = time.monotonic()
        last = getattr(self, "_last_decision_time", 0)
        wait = 0.5 - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_decision_time = time.monotonic()

        # -------------------------
        # VALIDATION
        # -------------------------
        context = str(context or "").strip()
        if not context:
            return {
                "action": "respond",
                "confidence": 0.3,
                "reasoning": "Empty input",
                "params": {},
            }

        context = context[:600]

        # -------------------------
        # SMART INTENT DETECTION (IMPROVED)
        # -------------------------
        tl = context.lower()
        is_chat = any(
            k in tl
            for k in (
                "hi",
                "hello",
                "what",
                "who",
                "why",
                "how",
                "tell",
                "explain",
                "help",
                "can you",
            )
        )

        # =========================
        # CHAT MODE
        # =========================
        if is_chat:
            try:
                result = await asyncio.wait_for(
                    self._retry(
                        self.llm.generate,
                        context,
                        system_prompt="You are a helpful AI assistant. Keep answers short and clear.",
                    ),
                    timeout=10,
                )

                response = ""
                if isinstance(result, dict):
                    response = str(result.get("response", "")).strip()
                else:
                    response = str(result).strip()

                if not response:
                    response = "I couldn't process that."

                return {
                    "action": "respond",
                    "confidence": 0.9,
                    "reasoning": response[:300],
                    "params": {},
                }

            except asyncio.TimeoutError:
                return {
                    "action": "respond",
                    "confidence": 0.4,
                    "reasoning": "Response timeout",
                    "params": {},
                }

            except Exception as e:
                self.logger.warning(f"Chat decision error: {e}")
                return {
                    "action": "respond",
                    "confidence": 0.5,
                    "reasoning": "Error processing request.",
                    "params": {},
                }

        # =========================
        # AGENT MODE
        # =========================
        try:
            prompt = (
                "Return ONLY JSON:\n"
                '{"action":"explore","confidence":0.8}\n\n'
                f"Context:\n{context}"
            )

            result = await asyncio.wait_for(
                self._retry(
                    self.llm.generate, prompt, system_prompt="STRICT JSON ONLY"
                ),
                timeout=12,
            )

            raw = ""
            if isinstance(result, dict):
                raw = str(result.get("response", "")).strip()
            else:
                raw = str(result).strip()

            raw = re.sub(r"[\n\r\t]", " ", raw)[:300]

            # -------------------------
            # SAFE JSON EXTRACTION (CRITICAL FIX)
            # -------------------------
            match = re.search(r"\{.*?\}", raw)
            if match:
                raw = match.group(0)

            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError

                action = str(data.get("action", "respond")).lower()
                confidence = float(data.get("confidence", 0.5))

            except Exception:
                action = "respond"
                confidence = 0.6

            return {
                "action": action,
                "confidence": max(0.0, min(1.0, confidence)),
                "reasoning": "",
                "params": {},
            }

        except asyncio.TimeoutError:
            return {
                "action": "respond",
                "confidence": 0.4,
                "reasoning": "timeout",
                "params": {},
            }

        except Exception as e:
            self.logger.warning(f"Decision failed: {e}")
            return {
                "action": "respond",
                "confidence": 0.5,
                "reasoning": "exception",
                "params": {},
            }

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(self, decision: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Safe execution pipeline with strict sanitization and consistent output"""

        try:
            # -------------------------
            # SANITIZE ACTION (CRITICAL FIX)
            # -------------------------
            raw_action = str(decision.get("action", "respond"))
            action = raw_action.strip().lower()
            action = re.sub(r"[\n\r\t]", "", action)
            action = re.sub(r"[^a-z_]", "", action)

            # normalize aliases
            if action in {"systemstatus", "check_system", "status"}:
                action = "system_status"

            # -------------------------
            # SAFE CONFIDENCE
            # -------------------------
            try:
                confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.5))))
            except Exception:
                confidence = 0.5

            params = decision.get("params", {}) or {}

            self.logger.info(f"⚡ execute: {action!r} (conf={confidence:.2f})")

            # -------------------------
            # MEMORY ACTIONS
            # -------------------------
            if action in {"read_memory", "memory_check"}:
                if not getattr(self, "memory", None):
                    return {
                        "status": "error",
                        "action": action,
                        "result": "Memory unavailable",
                    }

                try:
                    if hasattr(self.memory, "get_recent"):
                        data = await asyncio.wait_for(
                            self.memory.get_recent(5 if action == "read_memory" else 3),
                            timeout=5,
                        )
                    else:
                        data = []
                    return {"status": "success", "action": action, "result": data}

                except Exception as e:
                    self.logger.warning(f"Memory action failed: {e}")
                    return {
                        "status": "error",
                        "action": action,
                        "result": "Memory error",
                    }

            if action.startswith("memory"):
                return {
                    "status": "success",
                    "action": action,
                    "result": f"Memory-related action handled: {action}",
                }

            # -------------------------
            # SYSTEM STATUS
            # -------------------------
            if action == "system_status":
                return {
                    "status": "running",
                    "action": action,
                    "components": {
                        "llm": self.llm is not None,
                        "memory": self.memory is not None,
                        "curiosity": self.curiosity_engine is not None,
                    },
                    "active_suggestions": len(getattr(self, "_active_suggestions", [])),
                    "time": datetime.now().isoformat(),
                }

            # -------------------------
            # CORE ACTIONS
            # -------------------------
            simple_actions = {
                "think": lambda: f"Thinking: {context}",
                "explore": lambda: f"Exploring: {context}",
                "learn": lambda: f"Learning: {context}",
                "respond": lambda: context,
                "code": lambda: f"Generating code for: {context}",
            }

            if action in simple_actions:
                try:
                    result = simple_actions[action]()
                    return {
                        "status": "success",
                        "action": action,
                        "result": str(result)[:500],
                    }
                except Exception:
                    return {
                        "status": "error",
                        "action": action,
                        "result": "Execution error",
                    }

            # -------------------------
            # CURIOSITY ACTIONS
            # -------------------------
            if action == "get_suggestions":
                try:
                    result = await asyncio.wait_for(
                        self.get_active_suggestions(), timeout=5
                    )
                    return {"status": "success", "action": action, "result": result}
                except Exception:
                    return {"status": "error", "action": action, "result": []}

            if action == "get_next_suggestion":
                try:
                    s = await asyncio.wait_for(self.get_next_suggestion(), timeout=5)
                    return {
                        "status": "success",
                        "action": action,
                        "result": s.to_dict() if s else None,
                    }
                except Exception:
                    return {"status": "error", "action": action, "result": None}

            # -------------------------
            # DYNAMIC FALLBACK (SAFE)
            # -------------------------
            self.logger.warning(f"Unknown action: {action!r} — fallback used")

            return {
                "status": "success",
                "action": action,
                "result": f"Dynamic action executed: {action}",
            }

        except asyncio.CancelledError:
            raise

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return {
                "status": "error",
                "action": "unknown",
                "result": "Execution error",
            }

    # ── Main run ──────────────────────────────────────────────────────────

    async def run(self, context: str) -> Dict[str, Any]:
        """Safe LLM execution entry-point with validation + timeout + robust parsing"""

        try:
            # -------------------------
            # VALIDATION
            # -------------------------
            context = str(context or "").strip()
            if not context:
                return {"success": False, "error": "empty_context"}

            if not getattr(self, "llm_engine", None):
                return {"success": False, "error": "llm_missing"}

            # -------------------------
            # SELECT METHOD (CRITICAL FIX)
            # -------------------------
            if hasattr(self.llm_engine, "safe_generate"):
                call = self.llm_engine.safe_generate
            elif hasattr(self.llm_engine, "generate"):
                call = self.llm_engine.generate
            else:
                return {"success": False, "error": "no_generate_method"}

            # -------------------------
            # TIMEOUT PROTECTION (CRITICAL FIX)
            # -------------------------
            raw = await asyncio.wait_for(call(context), timeout=15)

            # -------------------------
            # SAFE EXTRACTION
            # -------------------------
            response = ""

            if isinstance(raw, dict):
                response = (
                    raw.get("response") or raw.get("output") or raw.get("result") or ""
                )
            else:
                response = str(raw) if raw else ""

            response = str(response).strip()

            # -------------------------
            # FALLBACK (CRITICAL FIX)
            # -------------------------
            if not response:
                return {
                    "success": False,
                    "error": "empty_response",
                }

            return {
                "success": True,
                "output": response[:1000],  # prevent oversized output
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "timeout",
            }

        except asyncio.CancelledError:
            raise

        except Exception as e:
            self.logger.error(f"Run failed: {e}")
            return {
                "success": False,
                "error": "execution_error",
            }

    # ── Legacy / sync API ─────────────────────────────────────────────────

    def make_decision(
        self,
        decision_type: DecisionType,
        context: str,
        options: Optional[List[Tuple[str, float]]] = None,
        normalize: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """Robust decision creation with safe normalization and validation"""

        # -------------------------
        # VALIDATION (CRITICAL FIX)
        # -------------------------
        context = str(context or "").strip()
        if not context:
            context = "no_context"

        decision = Decision(decision_type, context)

        if metadata and isinstance(metadata, dict):
            decision.metadata = metadata.copy()

        try:
            # -------------------------
            # PROCESS OPTIONS (SAFE)
            # -------------------------
            valid_options = []

            if options and isinstance(options, list):
                for item in options:
                    try:
                        name, score = item
                        name = str(name).strip()
                        score = float(score)

                        if not name:
                            continue

                        valid_options.append((name, score))

                    except Exception:
                        continue

            # -------------------------
            # NORMALIZATION
            # -------------------------
            if valid_options:
                scores = [s for _, s in valid_options]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                for name, score in valid_options:
                    if normalize and score_range > 0:
                        norm = (score - min_score) / score_range
                    else:
                        norm = score

                    # CLAMP (CRITICAL FIX)
                    norm = max(0.0, min(1.0, float(norm)))

                    decision.add_option(
                        Option(
                            name=name[:100],
                            score=norm,
                            confidence=norm,
                        )
                    )

            else:
                # -------------------------
                # DEFAULT OPTION
                # -------------------------
                decision.add_option(
                    Option(name="default_action", score=0.5, confidence=0.5)
                )

        except Exception as e:
            # -------------------------
            # FAILSAFE OPTION
            # -------------------------
            self.logger.warning(f"Decision creation error: {e}")
            decision.add_option(Option(name="fallback", score=0.5, confidence=0.5))

        # -------------------------
        # STORE DECISION (SAFE)
        # -------------------------
        try:
            self.decisions[decision.id] = decision
        except Exception:
            pass

        # -------------------------
        # METRICS (CRITICAL FIX)
        # -------------------------
        try:
            self.decision_count = int(getattr(self, "decision_count", 0)) + 1
        except Exception:
            self.decision_count = 1

        return decision

    def evaluate_decision(self, decision: Decision) -> Tuple[str, float]:
        """Safely evaluate a decision and return (name, confidence)"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not decision or not hasattr(decision, "evaluate_options"):
                return "default_action", 0.0

            # -------------------------
            # EVALUATE OPTIONS
            # -------------------------
            selected = decision.evaluate_options()

            if not selected or not hasattr(selected, "name"):
                raise ValueError("Invalid selected option")

            # -------------------------
            # SAFE CONFIDENCE (CRITICAL FIX)
            # -------------------------
            try:
                confidence = float(getattr(decision, "confidence", 0.0))
            except Exception:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))

            # -------------------------
            # UPDATE STATE
            # -------------------------
            decision.execution_status = "evaluated"

            return str(selected.name), confidence

        except ValueError as e:
            self.logger.warning(f"evaluate_decision: {e}")

        except Exception as e:
            self.logger.error(f"evaluate_decision unexpected error: {e}")

        # -------------------------
        # SAFE FALLBACK (CRITICAL FIX)
        # -------------------------
        try:
            decision.execution_status = "failed"
        except Exception:
            pass

        return "default_action", 0.0

    # ── Statistics ────────────────────────────────────────────────────────

    def get_curiosity_stats(self) -> Dict[str, Any]:
        """Return a summary of curiosity engine state."""
        stats: Dict[str, Any] = {
            "active_suggestions": len(self._active_suggestions),
            "suggestion_history": len(self._suggestion_history),
            "current_task": self.current_task,
            "suggestion_cooldown": self._suggestion_cooldown,
            "curiosity_available": self.curiosity_engine is not None,
        }

        if self.curiosity_engine and hasattr(self.curiosity_engine, "get_stats"):
            # FIX #7 – named exception with debug log
            try:
                stats["curiosity_engine"] = self.curiosity_engine.get_stats()
            except Exception as e:
                self.logger.debug(f"Curiosity get_stats failed: {e}")

        return stats

    # Enhancement D – lightweight suggestion summary
    def get_suggestion_stats(self) -> Dict[str, Any]:
        """Return a lightweight suggestion-only summary."""
        return {
            "active": len(self._active_suggestions),
            "history": len(self._suggestion_history),
            "task": self.current_task,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return safe and consistent engine statistics"""

        try:
            # -------------------------
            # SAFE ACCESS (CRITICAL FIX)
            # -------------------------
            decisions = getattr(self, "decisions", {}) or {}
            total = len(decisions)

            # -------------------------
            # SAFE AVG CONFIDENCE
            # -------------------------
            avg_conf = 0.0
            if total > 0:
                try:
                    avg_conf = (
                        sum(
                            float(getattr(d, "confidence", 0.0))
                            for d in decisions.values()
                        )
                        / total
                    )
                except Exception:
                    avg_conf = 0.0

            avg_conf = max(0.0, min(1.0, avg_conf))

            # -------------------------
            # SAFE EVALUATED COUNT
            # -------------------------
            evaluated = 0
            try:
                evaluated = sum(
                    1
                    for d in decisions.values()
                    if getattr(d, "execution_status", "") == "evaluated"
                )
            except Exception:
                evaluated = 0

            # -------------------------
            # SAFE CURIOSITY STATS
            # -------------------------
            try:
                curiosity_stats = (
                    self.get_curiosity_stats()
                    if hasattr(self, "get_curiosity_stats")
                    else {}
                )
            except Exception:
                curiosity_stats = {}

            # -------------------------
            # FINAL RESPONSE
            # -------------------------
            return {
                "total_decisions": int(getattr(self, "decision_count", total)),
                "stored_decisions": total,
                "evaluated": evaluated,
                "avg_confidence": round(avg_conf, 3),
                "status": "active" if getattr(self, "_running", False) else "stopped",
                "curiosity": curiosity_stats,
            }

        except Exception as e:
            self.logger.error(f"Stats error: {e}")

            # -------------------------
            # FAILSAFE RETURN
            # -------------------------
            return {
                "total_decisions": 0,
                "stored_decisions": 0,
                "evaluated": 0,
                "avg_confidence": 0.0,
                "status": "error",
                "curiosity": {},
            }

    # Enhancement B – maintenance helper
    def clear_old_decisions(self, max_age_seconds: float = 3600) -> int:
        """Safely remove stale decisions and return count"""

        try:
            # -------------------------
            # VALIDATION (CRITICAL FIX)
            # -------------------------
            if not isinstance(max_age_seconds, (int, float)) or max_age_seconds <= 0:
                max_age_seconds = 3600

            now_ts = datetime.now().timestamp()
            cutoff = now_ts - max_age_seconds

            decisions = getattr(self, "decisions", {}) or {}

            # -------------------------
            # SAFE COLLECTION (NO MUTATION DURING ITERATION)
            # -------------------------
            stale_keys = []

            for k, d in list(decisions.items()):
                try:
                    ts = getattr(d, "updated_at", None)
                    if ts and hasattr(ts, "timestamp"):
                        if ts.timestamp() < cutoff:
                            stale_keys.append(k)
                    else:
                        # remove corrupted entries
                        stale_keys.append(k)
                except Exception:
                    stale_keys.append(k)

            # -------------------------
            # DELETE SAFELY
            # -------------------------
            for k in stale_keys:
                try:
                    decisions.pop(k, None)
                except Exception:
                    continue

            # -------------------------
            # LOGGING
            # -------------------------
            if stale_keys:
                self.logger.debug(f"Purged {len(stale_keys)} old decisions")

            return len(stale_keys)

        except Exception as e:
            self.logger.error(f"clear_old_decisions error: {e}")
            return 0


__all__ = [
    "DecisionEngine",
    "Decision",
    "DecisionType",
    "Option",
    "CuriositySuggestion",
    "SuggestionType",
]
