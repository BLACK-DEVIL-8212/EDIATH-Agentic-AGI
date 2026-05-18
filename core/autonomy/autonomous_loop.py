
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger


# ── Lazy factory helpers ──────────────────────────────────────────────────────

def _create_memory_manager():
    from ..memory.memory_manager import MemoryManager
    return MemoryManager()


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class AgentMode(Enum):
    AUTONOMOUS = "autonomous"
    ASSISTANT  = "assistant"
    CHAT       = "chat"
    TASK       = "task"
    RESEARCH   = "research"
    CODE       = "code"
    CREATIVE   = "creative"


class EmotionState(Enum):
    HAPPY      = "happy"
    SAD        = "sad"
    ANGRY      = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL    = "neutral"
    EXCITED    = "excited"
    CONFUSED   = "confused"


class ResponseStyle(Enum):
    CONCISE      = "concise"
    DETAILED     = "detailed"
    TECHNICAL    = "technical"
    FRIENDLY     = "friendly"
    PROFESSIONAL = "professional"
    CREATIVE     = "creative"


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationTurn:
    id:                 str
    user_input:         str
    assistant_response: str
    timestamp:          datetime = field(default_factory=datetime.now)
    duration_ms:        float    = 0.0
    emotion:            Optional[EmotionState] = None
    confidence:         float    = 0.0
    tokens_used:        int      = 0
    tools_called:       List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                 self.id,
            "user_input":         self.user_input[:200],
            "assistant_response": self.assistant_response[:200],
            "timestamp":          self.timestamp.isoformat(),
            "duration_ms":        self.duration_ms,
            "emotion":            self.emotion.value if self.emotion else None,
            "confidence":         self.confidence,
        }


@dataclass
class SessionMetrics:
    session_id:           str
    start_time:           datetime = field(default_factory=datetime.now)
    end_time:             Optional[datetime] = None
    total_turns:          int   = 0
    total_tokens:         int   = 0
    avg_response_time_ms: float = 0.0
    user_satisfaction:    float = 0.0
    tools_used:           List[str] = field(default_factory=list)
    errors:               int   = 0
    rate_limits_hit:      int   = 0


# ══════════════════════════════════════════════════════════════════════════════
# Rate limiter
# ══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Token-bucket rate limiter, concurrency-safe."""

    def __init__(self, max_calls: int = 60, time_window: float = 60.0) -> None:
        self.max_calls   = max_calls
        self.time_window = time_window
        self.calls:      deque = deque()
        # FIX: asyncio.Lock must be created lazily (inside a running loop)
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> bool:
        async with self._get_lock():
            now = time.time()
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                wait = self.time_window - (now - self.calls[0])
                if wait > 0:
                    await asyncio.sleep(wait)
            self.calls.append(time.time())
            return True


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment analyser
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    def __init__(self) -> None:
        self.emotion_keywords: Dict[EmotionState, List[str]] = {
            EmotionState.HAPPY:      ["happy", "great", "excellent", "awesome", "love", "wonderful", "amazing"],
            EmotionState.SAD:        ["sad", "unhappy", "depressed", "terrible", "awful", "bad", "sorry"],
            EmotionState.ANGRY:      ["angry", "mad", "furious", "hate", "annoying", "horrible"],
            EmotionState.FRUSTRATED: ["frustrated", "confused", "stuck", "difficult", "hard", "struggling"],
            EmotionState.EXCITED:    ["excited", "thrilled", "cool", "fantastic", "incredible", "wow"],
            EmotionState.CONFUSED:   ["confused", "what", "huh", "understand", "unclear", "explain"],
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        counts: Dict[EmotionState, int] = {}
        for emotion, keywords in self.emotion_keywords.items():
            c = sum(1 for kw in keywords if kw in text_lower)
            if c:
                counts[emotion] = c

        primary = max(counts, key=counts.get) if counts else EmotionState.NEUTRAL
        total   = max(1, sum(counts.values()))
        conf    = min(1.0, sum(counts.values()) / 5)

        positive = {EmotionState.HAPPY, EmotionState.EXCITED}
        negative = {EmotionState.SAD, EmotionState.ANGRY, EmotionState.FRUSTRATED}
        score    = sum(c if e in positive else -c if e in negative else 0
                       for e, c in counts.items())

        return {
            "emotion":         primary,
            "confidence":      conf,
            "sentiment":       "positive" if score > 0 else "negative" if score < 0 else "neutral",
            # FIX: normalise consistently
            "sentiment_score": max(-1.0, min(1.0, score / total)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Context manager
# ══════════════════════════════════════════════════════════════════════════════

class ContextManager:
    def __init__(self, max_turns: int = 20, max_tokens: int = 4000) -> None:
        self.max_turns  = max_turns
        self.max_tokens = max_tokens
        self.history:   List[ConversationTurn] = []
        self.summary:   Optional[str] = None

    def add_turn(self, turn: ConversationTurn) -> None:
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    # FIX: renamed parameter n_turns to avoid shadowing self.max_turns
    def get_context(self, n_turns: Optional[int] = None) -> str:
        limit = n_turns if n_turns is not None else self.max_turns
        turns = self.history[-limit:] if limit else self.history
        lines = []
        if self.summary:
            lines.append(f"Summary of previous conversation: {self.summary}")
        for t in turns:
            lines.append(f"User: {t.user_input}")
            lines.append(f"Assistant: {t.assistant_response}")
        return "\n".join(lines)

    async def summarize(self, llm: Optional[Any] = None) -> str:
        if not self.history:
            return ""
        if llm:
            prompt = f"Summarize this conversation concisely:\n\n{self.get_context(10)}"
            try:
                response = await llm.generate(prompt)
                raw = response.get("response", "") if isinstance(response, dict) else str(response)
                self.summary = raw[:500]
            except Exception as exc:
                logger.debug("Summary generation failed: %s", exc)
                self.summary = f"Conversation with {len(self.history)} turns"
        return self.summary or ""


# ══════════════════════════════════════════════════════════════════════════════
# Tool registry
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    def __init__(self) -> None:
        self.tools:             Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str]      = {}

    def register(self, name: str, description: str, func: Callable) -> Callable:
        """Register a tool and return func (so it can be used as a decorator)."""
        self.tools[name]             = func
        self.tool_descriptions[name] = description
        logger.info("Tool registered: %s", name)
        # FIX: return func so decorator usage works
        return func

    async def execute(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        func = self.tools[name]
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        return func(**kwargs)

    def get_tools_prompt(self) -> str:
        if not self.tools:
            return ""
        lines = ["\nAvailable tools:"]
        for name, desc in self.tool_descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Autonomous loop
# ══════════════════════════════════════════════════════════════════════════════

class AutonomousLoop:
    """
    Unified Brain loop — voice, text, tools, emotion, context, memory.
    """

    def __init__(
        self,
        system,
        mode:               AgentMode     = AgentMode.ASSISTANT,
        response_style:     ResponseStyle = ResponseStyle.FRIENDLY,
        enable_emotion:     bool          = True,
        enable_context:     bool          = True,
        enable_tools:       bool          = True,
        max_history_turns:  int           = 20,
        rate_limit_calls:   int           = 60,
        rate_limit_window:  float         = 60.0,
        streaming:          bool          = False,
        personality:        Optional[str] = None,
    ) -> None:
        self.system         = system
        self.mode           = mode
        self.response_style = response_style
        self.enable_emotion = enable_emotion
        self.enable_context = enable_context
        self.enable_tools   = enable_tools
        self.streaming      = streaming
        self.personality    = personality or "Helpful and friendly AI assistant"

        self.memory            = _create_memory_manager()
        self.llm               = (
            getattr(system, "llm", None)
            or getattr(system, "llm_engine", None)
            or getattr(system, "shared_llm", None)
        )
        self.tool_registry     = ToolRegistry()
        self.sentiment_analyzer= SentimentAnalyzer() if enable_emotion else None
        self.context_manager   = ContextManager(max_turns=max_history_turns) if enable_context else None
        self.rate_limiter      = RateLimiter(max_calls=rate_limit_calls, time_window=rate_limit_window)

        # State
        self.is_running   = False
        self.user_input:  Optional[str] = None
        self.is_speaking  = False
        self.processing   = False
        self.thinking     = False
        # FIX: dedicated pause flag instead of hijacking self.processing
        self._paused      = False

        # Input guards
        self._last_input_time: float = 0.0
        self.input_cooldown:   float = 1.0
        self._last_response:   str   = ""
        self._last_input_hash: str   = ""

        # Session
        self.current_session    = SessionMetrics(session_id=str(uuid.uuid4())[:8])
        self.conversation_turns: List[ConversationTurn] = []

        self.ignore_words = {"you", "uh", "hmm", "noise", "um", "ah", "like", "well"}

        # Counters
        self.inputs_received  = 0
        self.inputs_processed = 0
        self.messages_spoken  = 0
        self.errors           = 0
        self.total_tokens     = 0

        self._background_tasks: List[asyncio.Task] = []

        self.response_cache: Dict[str, str] = {}
        self.cache_max_size = 100

        self.pre_process_hooks:  List[Callable] = []
        self.post_process_hooks: List[Callable] = []

        # FIX: register tools via proper methods — no decorator abuse
        self._register_default_tools()

        logger.info(
            "Autonomous Loop initialized (mode=%s, style=%s)",
            mode.value, response_style.value,
        )

    # ── Default tool implementations ──────────────────────────────────────────

    async def _tool_get_time(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _tool_calculate(self, expression: str) -> str:
        try:
            allowed = {"abs": abs, "round": round, "min": min, "max": max}
            result  = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
            return f"Result: {result}"
        except Exception as exc:
            return f"Error: {exc}"

    async def _tool_search_memory(self, query: str) -> str:
        try:
            results = await self.memory.search(query, limit=3)
            return json.dumps(results, default=str) if results else "No results found"
        except Exception as exc:
            return f"Search error: {exc}"

    async def _tool_remember(self, key: str, value: str) -> str:
        try:
            await self.memory.store({"key": key, "value": value})
            return f"Remembered: {key}"
        except Exception as exc:
            return f"Memory error: {exc}"

    def _register_default_tools(self) -> None:
        """
        Register built-in tools via direct calls.

        FIX: the original used @self.tool_registry.register(...) as a decorator
        on locally-scoped functions while also passing self._tool_get_time
        (non-existent at that point) as the third positional argument.
        register() returned None, so the decorator tried to call None(func)
        → AttributeError crash at startup.
        """
        self.tool_registry.register("get_time",       "Get current time",                    self._tool_get_time)
        self.tool_registry.register("calculate",      "Perform mathematical calculation",    self._tool_calculate)
        self.tool_registry.register("search_memory",  "Search memory for information",       self._tool_search_memory)
        self.tool_registry.register("remember",       "Store information in memory",         self._tool_remember)

    # ── Input handling ────────────────────────────────────────────────────────

    def set_input(self, text: str) -> bool:
        try:
            if not text or not isinstance(text, str):
                return False
            text = text.strip()
            if not text or len(text) < 2:
                return False
            if text.lower() in self.ignore_words:
                return False
            now = time.time()
            if now - self._last_input_time < self.input_cooldown:
                return False
            input_hash = hashlib.md5(text.encode()).hexdigest()
            if input_hash == self._last_input_hash:
                return False
            if self.is_speaking or self.processing or self.thinking:
                return False
            self._last_input_time  = now
            self._last_input_hash  = input_hash
            self.user_input        = text
            self.inputs_received  += 1
            logger.info("Input accepted: %s…", text[:50])
            return True
        except Exception as exc:
            logger.error("Input handling failed: %s", exc)
            return False

    def set_mode(self, mode: AgentMode) -> None:
        self.mode = mode
        logger.info("Mode changed to: %s", mode.value)

    def set_response_style(self, style: ResponseStyle) -> None:
        self.response_style = style
        logger.info("Response style changed to: %s", style.value)

    def add_pre_hook(self, hook: Callable) -> None:
        self.pre_process_hooks.append(hook)

    def add_post_hook(self, hook: Callable) -> None:
        self.post_process_hooks.append(hook)

    # ── Speech output ─────────────────────────────────────────────────────────

    async def speak(self, text: str, priority: bool = False) -> bool:
        try:
            if not text or not isinstance(text, str):
                return False
            text = text.strip()
            if not text:
                return False
            if self.is_speaking and not priority:
                return False
            self.is_speaking = True
            prefix = "🔊🔊" if priority else "🔊"
            print(f"{prefix} {text}")
            duration = min(3.0, max(0.3, len(text) * 0.03))
            await asyncio.sleep(duration)
            self.messages_spoken += 1
            return True
        except Exception as exc:
            logger.error("Speak failed: %s", exc)
            return False
        finally:
            self.is_speaking = False

    # ── Response generation ───────────────────────────────────────────────────

    async def generate_response(self, user_input: str) -> str:
        prompt    = self._build_prompt(user_input)
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.debug("Cache hit for response")
            return self.response_cache[cache_key]

        await self.rate_limiter.acquire()

        try:
            if not self.llm:
                return self._fallback_response(user_input)

            if self.streaming:
                response = await self._stream_response(prompt)
            else:
                response = await self.llm.generate(prompt)

            response_text = (
                response.get("response", str(response))
                if isinstance(response, dict)
                else str(response)
            )
            response_text = self._clean_response(response_text)

            if len(self.response_cache) >= self.cache_max_size:
                try:
                    self.response_cache.pop(next(iter(self.response_cache)))
                except StopIteration:
                    pass
            self.response_cache[cache_key] = response_text
            return response_text

        except Exception as exc:
            logger.error("Response generation failed: %s", exc)
            return self._fallback_response(user_input)

    def _build_prompt(self, user_input: str) -> str:
        system_prompt = (
            f"You are {self.personality}\n\n"
            f"Mode: {self.mode.value}\n"
            f"Response Style: {self.response_style.value}\n\n"
            "Guidelines:\n"
            "- Be helpful, accurate, and concise\n"
            f"- Use {self.response_style.value} tone\n"
            "- Stay within your capabilities\n"
            "- Admit when you don't know something\n"
            "- Be safe and ethical\n"
        )

        if self.enable_tools:
            system_prompt += self.tool_registry.get_tools_prompt()

        if self.enable_context and self.context_manager:
            ctx = self.context_manager.get_context()
            if ctx:
                system_prompt += f"\n\nConversation History:\n{ctx}\n"

        if self.enable_emotion and self.sentiment_analyzer:
            analysis = self.sentiment_analyzer.analyze(user_input)
            system_prompt += (
                f"\nUser emotion: {analysis['emotion'].value} "
                f"(confidence: {analysis['confidence']:.2f})"
            )

        return f"{system_prompt}\n\nCurrent user input: {user_input}\n\nRespond naturally and helpfully:"

    async def _stream_response(self, prompt: str) -> str:
        try:
            if not self.llm:
                raise RuntimeError("LLM is not ready")
            full_response = await self.llm.generate(prompt)
            text = full_response if isinstance(full_response, str) else str(full_response)
            for word in text.split():
                print(word, end=" ", flush=True)
                await asyncio.sleep(0.05)
            print()
            return text
        except Exception as exc:
            logger.error("Streaming failed: %s", exc)
            raise

    def _clean_response(self, response: str) -> str:
        response = re.sub(r"\s+", " ", response)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        response = response.strip()
        if response and response[-1] not in ".!?":
            response += "."
        if response:
            response = response[0].upper() + response[1:]
        return response.strip()

    def _fallback_response(self, user_input: str) -> str:
        return random.choice([
            "I'm having trouble processing that request. Could you please rephrase?",
            "I encountered an issue. Let me try again differently.",
            "Sorry, I couldn't process that. Could you simplify your request?",
            "I need a moment. Can you try again?",
            "I'm experiencing some difficulty. Please try a different approach.",
        ])

    # ── Tool execution ────────────────────────────────────────────────────────

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        try:
            result = await self.tool_registry.execute(tool_name, **kwargs)
            self.current_session.tools_used.append(tool_name)
            return result
        except Exception as exc:
            logger.error("Tool execution failed: %s", exc)
            return f"Tool error: {exc}"

    # ── Main processing ───────────────────────────────────────────────────────

    async def process_input(self, user_input: str) -> Optional[str]:
        # Pre-hooks
        for hook in self.pre_process_hooks:
            try:
                result = hook(user_input)
                if result is not None:
                    user_input = result
            except Exception as exc:
                logger.warning("Pre-hook failed: %s", exc)

        emotion: Optional[EmotionState] = None
        if self.enable_emotion and self.sentiment_analyzer:
            analysis = self.sentiment_analyzer.analyze(user_input)
            emotion  = analysis["emotion"]
            if emotion == EmotionState.ANGRY:
                await self.speak("I understand you're frustrated. Let me help.", priority=True)
            elif emotion == EmotionState.SAD:
                await self.speak("I hear that you're feeling down. I'm here to help.", priority=True)

        start_ms = time.time()
        response = await self.generate_response(user_input)
        duration_ms = (time.time() - start_ms) * 1000

        turn = ConversationTurn(
            id                 = str(uuid.uuid4())[:8],
            user_input         = user_input,
            assistant_response = response,
            duration_ms        = duration_ms,
            emotion            = emotion,
            tools_called       = list(self.current_session.tools_used),
        )

        if self.enable_context and self.context_manager:
            self.context_manager.add_turn(turn)

        # FIX: safe fire-and-forget memory store
        try:
            asyncio.ensure_future(
                self.memory.store({
                    "type":      "conversation",
                    "input":     user_input,
                    "response":  response[:500],
                    "emotion":   emotion.value if emotion else None,
                    "timestamp": datetime.now().isoformat(),
                })
            )
        except Exception as exc:
            logger.debug("Memory store skipped: %s", exc)

        # Update session metrics
        self.current_session.total_turns += 1
        self.current_session.total_tokens += turn.tokens_used
        n = self.current_session.total_turns
        prev = self.current_session.avg_response_time_ms
        # FIX: incremental running average (no floating-point accumulation drift)
        self.current_session.avg_response_time_ms = prev + (duration_ms - prev) / n

        self.inputs_processed += 1

        # Post-hooks
        for hook in self.post_process_hooks:
            try:
                result = hook(response)
                if result is not None:
                    response = result
            except Exception as exc:
                logger.warning("Post-hook failed: %s", exc)

        return response

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self, duration: Optional[float] = None) -> None:
        if self.is_running:
            logger.warning("Loop already running")
            return

        self.is_running = True
        self._background_tasks.append(asyncio.create_task(self._periodic_summary()))
        logger.info("EDIATH started — mode: %s", self.mode.value)

        start_time = time.monotonic()

        while self.is_running:
            try:
                if duration and (time.monotonic() - start_time) > duration:
                    break

                if not self.user_input:
                    await asyncio.sleep(0.05)
                    continue

                # FIX: check _paused flag (not self.processing)
                if self._paused or self.thinking or self.processing:
                    await asyncio.sleep(0.1)
                    continue

                self.processing = True
                self.thinking   = True

                user_text       = self.user_input
                self.user_input = None

                logger.info("User: %s", user_text)

                response = await self.process_input(user_text)

                if response:
                    logger.info("AI: %s…", response[:100])
                    await self.speak(response)
                    self._last_response = response
                    if hasattr(self.system, "emit"):
                        try:
                            await self.system.emit("response", {"response": response})
                        except Exception:
                            pass

                self.processing = False
                self.thinking   = False

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Loop error: %s", exc)
                self.errors += 1
                self.current_session.errors += 1
                self.processing = False
                self.thinking   = False
                if self.errors > 5:
                    logger.critical("Too many errors — entering recovery mode")
                    await asyncio.sleep(5)
                    self.errors = 0
                else:
                    await asyncio.sleep(0.5)

        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        self.current_session.end_time = datetime.now()
        logger.info("EDIATH stopped")

    async def _periodic_summary(self) -> None:
        while self.is_running:
            await asyncio.sleep(300)
            if (
                self.enable_context
                and self.context_manager
                and len(self.context_manager.history) > 10
            ):
                try:
                    await self.context_manager.summarize(self.llm)
                    logger.debug("Conversation summary updated")
                except Exception as exc:
                    logger.debug("Summary generation failed: %s", exc)

    # ── Control ───────────────────────────────────────────────────────────────

    def stop(self) -> bool:
        if not self.is_running:
            logger.warning("Loop already stopped")
            return False
        self.is_running = False
        self.processing = False
        self.thinking   = False
        self.is_speaking= False
        self._paused    = False
        logger.info("Stop signal sent")
        return True

    def pause(self) -> None:
        # FIX: set dedicated _paused flag; do NOT touch self.processing
        self._paused = True
        logger.info("Paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("Resumed")

    def clear_context(self) -> None:
        if self.context_manager:
            self.context_manager.history.clear()
            self.context_manager.summary = None
        logger.info("Context cleared")

    def reset(self) -> None:
        self.stop()
        # FIX: removed time.sleep(0.5) — callers must use asyncio.sleep if needed
        self.inputs_received  = 0
        self.inputs_processed = 0
        self.messages_spoken  = 0
        self.errors           = 0
        self.total_tokens     = 0
        self.clear_context()
        self.response_cache.clear()
        self.current_session   = SessionMetrics(session_id=str(uuid.uuid4())[:8])
        self.user_input        = None
        self._last_input_time  = 0.0
        self._last_response    = ""
        self._last_input_hash  = ""
        logger.info("System reset complete")

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        end = self.current_session.end_time or datetime.now()
        duration = (end - self.current_session.start_time).total_seconds()
        return {
            "session": {
                "id":                    self.current_session.session_id,
                "duration_seconds":      duration,
                "total_turns":           self.current_session.total_turns,
                "avg_response_time_ms":  round(self.current_session.avg_response_time_ms, 2),
                "errors":                self.current_session.errors,
                "rate_limits_hit":       self.current_session.rate_limits_hit,
                "tools_used":            self.current_session.tools_used,
            },
            "performance": {
                "inputs_received":   self.inputs_received,
                "inputs_processed":  self.inputs_processed,
                "messages_spoken":   self.messages_spoken,
                "total_tokens":      self.total_tokens,
                "cache_size":        len(self.response_cache),
                "conversation_turns":len(self.conversation_turns),
            },
            "status": {
                "is_running":     self.is_running,
                "is_speaking":    self.is_speaking,
                "processing":     self.processing,
                "thinking":       self.thinking,
                "paused":         self._paused,
                "mode":           self.mode.value,
                "response_style": self.response_style.value,
            },
            "memory": {
                "context_turns": len(self.context_manager.history) if self.context_manager else 0,
                "has_summary":   self.context_manager.summary is not None if self.context_manager else False,
            },
        }

    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        if self.context_manager:
            return [t.to_dict() for t in self.context_manager.history[-limit:]]
        return []

    def export_session(self, filepath: str) -> None:
        data = {
            "session": {
                "id":         self.current_session.session_id,
                "start_time": self.current_session.start_time.isoformat(),
                "end_time":   self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                "metrics": {
                    "total_turns":           self.current_session.total_turns,
                    "avg_response_time_ms":  self.current_session.avg_response_time_ms,
                    "errors":                self.current_session.errors,
                },
            },
            "conversation": self.get_conversation_history(limit=100),
            "stats":        self.get_stats(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Session exported to %s", filepath)


# ══════════════════════════════════════════════════════════════════════════════
# CLI interface
# ══════════════════════════════════════════════════════════════════════════════

class EDIATHCLI:
    def __init__(self, loop: AutonomousLoop) -> None:
        self.loop    = loop
        self.running = False

    async def start(self) -> None:
        self.running = True
        print("\n" + "=" * 50)
        print("🔥 EDIATH AI Assistant — Unified Brain")
        print("=" * 50)
        print("Commands: /mode  /style  /clear  /stats  /history  /export  /quit")
        print("=" * 50 + "\n")

        asyncio.create_task(self.loop.run())

        while self.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    self.loop.set_input(user_input)
                await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as exc:
                print(f"Error: {exc}")

        # FIX: stop() is synchronous — do NOT await it
        self.loop.stop()

    async def _handle_command(self, command: str) -> None:
        cmd = command.lower().strip()

        if cmd in ("/quit", "/exit"):
            print("Shutting down…")
            self.running = False

        elif cmd == "/clear":
            self.loop.clear_context()
            print("✅ Context cleared")

        elif cmd.startswith("/mode"):
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    self.loop.set_mode(AgentMode(parts[1]))
                    print(f"✅ Mode: {self.loop.mode.value}")
                except ValueError:
                    print(f"Invalid mode. Options: {[m.value for m in AgentMode]}")
            else:
                print(f"Current mode: {self.loop.mode.value}")

        elif cmd.startswith("/style"):
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    self.loop.set_response_style(ResponseStyle(parts[1]))
                    print(f"✅ Style: {self.loop.response_style.value}")
                except ValueError:
                    print(f"Invalid style. Options: {[s.value for s in ResponseStyle]}")
            else:
                print(f"Current style: {self.loop.response_style.value}")

        elif cmd == "/stats":
            s = self.loop.get_stats()
            print("\n📊 STATISTICS")
            print("-" * 40)
            for k, v in s["session"].items():
                print(f"  {k}: {v}")
            print("-" * 40 + "\n")

        elif cmd == "/history":
            history = self.loop.get_conversation_history(limit=5)
            if not history:
                print("No conversation history")
            else:
                print("\n📜 RECENT CONVERSATION")
                print("-" * 40)
                for t in history:
                    print(f"User: {t['user_input'][:80]}")
                    print(f"AI:   {t['assistant_response'][:80]}")
                    print("-" * 20)

        elif cmd == "/export":
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.loop.export_session(filename)
            print(f"✅ Exported to {filename}")

        else:
            print(f"Unknown command: {command}")


# ══════════════════════════════════════════════════════════════════════════════
# Web server integration
# ══════════════════════════════════════════════════════════════════════════════

class EDIATHWebServer:
    def __init__(
        self,
        loop: AutonomousLoop,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        self.loop = loop
        self.host = host
        self.port = port

    async def start(self) -> None:
        # FIX: aiohttp import moved here — no module-level crash when not installed
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return

        app = web.Application()
        app.router.add_get("/",             self.handle_index)
        app.router.add_post("/api/chat",    self.handle_chat)
        app.router.add_get("/api/stats",    self.handle_stats)
        app.router.add_get("/api/history",  self.handle_history)
        app.router.add_post("/api/clear",   self.handle_clear)

        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, self.host, self.port).start()
        logger.info("Web server at http://%s:%d", self.host, self.port)
        await asyncio.Future()  # run forever

    async def handle_index(self, request):
        from aiohttp import web
        html = """<!DOCTYPE html>
<html>
<head><title>EDIATH</title>
<style>
body{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px;background:#f5f5f5}
.chat{background:#fff;border-radius:10px;padding:20px;height:500px;overflow-y:auto;margin-bottom:20px}
.msg{margin-bottom:15px;padding:10px;border-radius:5px}
.user{background:#007bff;color:#fff;text-align:right}
.ai{background:#e9ecef;color:#333}
.row{display:flex;gap:10px}
input{flex:1;padding:10px;border:1px solid #ddd;border-radius:5px}
button{padding:10px 20px;background:#007bff;color:#fff;border:none;border-radius:5px;cursor:pointer}
button:hover{background:#0056b3}
</style></head>
<body>
<h1>🔥 EDIATH AI Assistant</h1>
<div class="chat" id="chat"></div>
<div class="row">
  <input id="inp" placeholder="Type your message…"/>
  <button onclick="send()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('inp');
function add(t,u){const d=document.createElement('div');d.className='msg '+(u?'user':'ai');d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
async function send(){const t=inp.value.trim();if(!t)return;add(t,true);inp.value='';
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});
  const d=await r.json();add(d.response||d.error,false);}
inp.addEventListener('keypress',e=>{if(e.key==='Enter')send();});
</script>
</body></html>"""
        return web.Response(text=html, content_type="text/html")

    async def handle_chat(self, request):
        from aiohttp import web
        try:
            data    = await request.json()
            message = data.get("message", "").strip()
            if not message:
                return web.json_response({"error": "No message provided"}, status=400)
            response = await self.loop.process_input(message)
            return web.json_response({"response": response})
        except Exception as exc:
            logger.error("Chat error: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_stats(self, request):
        from aiohttp import web
        return web.json_response(self.loop.get_stats())

    async def handle_history(self, request):
        from aiohttp import web
        return web.json_response(self.loop.get_conversation_history(limit=50))

    async def handle_clear(self, request):
        from aiohttp import web
        self.loop.clear_context()
        return web.json_response({"status": "cleared"})


# ══════════════════════════════════════════════════════════════════════════════
# Unified brain wrapper
# ══════════════════════════════════════════════════════════════════════════════

class EDIATHUnifiedBrain:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.system = self

        self.memory = _create_memory_manager()
        self.llm = None

        self.loop = AutonomousLoop(
            system            = self,
            mode              = AgentMode(config.get("mode", "assistant")),
            response_style    = ResponseStyle(config.get("response_style", "friendly")),
            enable_emotion    = config.get("enable_emotion", True),
            enable_context    = config.get("enable_context", True),
            enable_tools      = config.get("enable_tools", True),
            max_history_turns = config.get("max_history_turns", 20),
            rate_limit_calls  = config.get("rate_limit_calls", 60),
            rate_limit_window = config.get("rate_limit_window", 60.0),
            streaming         = config.get("streaming", False),
            personality       = config.get("personality", "Helpful and friendly AI assistant"),
        )

        self.cli = EDIATHCLI(self.loop)
        self.web_server: Optional[EDIATHWebServer] = None
        if config.get("enable_web", False):
            self.web_server = EDIATHWebServer(
                self.loop,
                host=config.get("web_host", "localhost"),
                port=config.get("web_port", 8080),
            )

        self.is_running = False
        self._tasks:     List[asyncio.Task] = []
        logger.info("EDIATH Unified Brain initialized")

    async def start(self, mode: str = "cli") -> None:
        if self.is_running:
            logger.warning("Already running")
            return
        self.is_running = True

        if mode == "cli":
            await self.cli.start()
        elif mode == "web":
            if self.web_server:
                await self.web_server.start()
            else:
                logger.error("Web server not enabled in config")
        elif mode == "auto":
            try:
                import aiohttp  # noqa: F401
                tasks = [asyncio.create_task(self.cli.start())]
                if self.web_server:
                    tasks.append(asyncio.create_task(self.web_server.start()))
                await asyncio.gather(*tasks)
            except ImportError:
                await self.cli.start()
        else:
            logger.error("Unknown start mode: %s", mode)

    async def stop(self) -> None:
        self.is_running = False
        self.loop.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        logger.info("EDIATH stopped")

    async def brain_process(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if action == "task":
            response = await self.loop.process_input(params.get("task", ""))
            return {"result": response}
        if action == "think":
            return {"result": "Thinking…"}
        return {"result": "Unknown action"}

    def set_input(self, text: str) -> bool:
        return self.loop.set_input(text)

    def get_stats(self) -> Dict[str, Any]:
        return self.loop.get_stats()

    def get_conversation(self) -> List[Dict[str, Any]]:
        return self.loop.get_conversation_history()

    def clear_context(self) -> None:
        self.loop.clear_context()

    def set_mode(self, mode: str) -> None:
        try:
            self.loop.set_mode(AgentMode(mode))
        except ValueError:
            logger.error("Invalid mode: %s", mode)

    def set_style(self, style: str) -> None:
        try:
            self.loop.set_response_style(ResponseStyle(style))
        except ValueError:
            logger.error("Invalid style: %s", style)


# ══════════════════════════════════════════════════════════════════════════════
# Factory & convenience
# ══════════════════════════════════════════════════════════════════════════════

def create_ediath(config: Optional[Dict[str, Any]] = None) -> EDIATHUnifiedBrain:
    """Create and return an EDIATHUnifiedBrain instance."""
    return EDIATHUnifiedBrain(config)


async def quick_chat() -> None:
    """
    Interactive chat for quick testing.
    FIX: original set_input + sleep never worked because no loop was running.
    Now calls process_input() directly.
    """
    brain = create_ediath({
        "mode":              "chat",
        "response_style":    "friendly",
        "enable_emotion":    True,
        "enable_tools":      True,
        "max_history_turns": 10,
    })
    print("EDIATH Quick Chat (type 'quit' to exit)")
    print("-" * 40)
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            break
        if user_input:
            response = await brain.loop.process_input(user_input)
            print(f"AI: {response}")
    await brain.stop()


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="EDIATH AI Assistant")
    parser.add_argument("--mode",  choices=["cli", "web", "auto"], default="cli")
    parser.add_argument("--style", choices=["concise", "detailed", "friendly", "professional"], default="friendly")
    parser.add_argument("--port",  type=int, default=8080)
    parser.add_argument("--no-emotion", action="store_true")
    parser.add_argument("--no-tools",   action="store_true")
    args = parser.parse_args()

    brain = create_ediath({
        "mode":           "assistant",
        "response_style": args.style,
        "enable_emotion": not args.no_emotion,
        "enable_tools":   not args.no_tools,
        "enable_web":     args.mode in ("web", "auto"),
        "web_port":       args.port,
    })

    try:
        await brain.start(args.mode)
    except KeyboardInterrupt:
        print("\n\nShutting down…")
        await brain.stop()


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "AutonomousLoop",
    "EDIATHUnifiedBrain",
    "EDIATHCLI",
    "EDIATHWebServer",
    "AgentMode",
    "EmotionState",
    "ResponseStyle",
    "ConversationTurn",
    "SessionMetrics",
    "RateLimiter",
    "SentimentAnalyzer",
    "ContextManager",
    "ToolRegistry",
    "create_ediath",
    "quick_chat",
]
