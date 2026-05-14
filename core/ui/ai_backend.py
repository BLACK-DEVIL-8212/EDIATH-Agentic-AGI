import asyncio
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from kivy.clock import Clock

logger = logging.getLogger(__name__)


class AIBackend:
    """
    Proxy between the Kivy UI and the real EDIATHSystem living in main.py.
    Does NOT create its own system — it connects to main._SYSTEM_INSTANCE.

    Uses DIRECT async calls with timeouts instead of relying on _UI_CALLBACK
    or the heavy process_user_input pipeline, ensuring responses always come
    back even if memory/context/decision layers hang.
    """

    def __init__(self):
        self.system: Optional[Any] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Compatibility: some UI widgets expect `controller.backend`.
        self.backend = self

        # Support multiple callbacks (UI components may register independently)
        self._response_callbacks: List[Callable[[str], None]] = []
        self._status_callbacks: List[Callable[[str], None]] = []

        self.chat_screen = None
        self.status_bar = None
        self.task_display = None
        self.planning_panel = None

        self._ready = False
        self._processing = False
        self._shutdown = False
        self._welcomed = False
        self._events: List = []

        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # slightly longer to reduce spam

        self._initialized = False
        self._running = False

    def send_message(self, text: str, *args, **kwargs):
        """Backward-compatible alias used by older UI wiring."""
        self.send_user_message(text)

    # ─────────────────────────────────────────────────────────────────
    # CONNECT TO REAL SYSTEM  (called by main.py init thread)
    # ─────────────────────────────────────────────────────────────────
    def connect_system(self, system, loop: asyncio.AbstractEventLoop):
        """Link this proxy to the real EDIATHSystem and its event loop."""
        try:
            if not system or not loop:
                self._emit_status_safe("❌ Invalid system connection")
                return

            self.system = system
            self.loop = loop
            self._ready = True

            self._processing = False
            self._shutdown = False

            self._emit_status_safe("✅ AI Ready")

            if not self._welcomed:
                self._emit_response_safe(
                    "👋 Hello! I'm EDIATH, your intelligent assistant. How can I help you?"
                )
                self._welcomed = True

                logger.info("✅ Backend connected successfully")

        except Exception as e:
            logger.exception("❌ connect_system error: %s", e)
            self._emit_status_safe(f"❌ Connection failed: {str(e)}")
            self._ready = False

    # ─────────────────────────────────────────────────────────────────
    # UI BINDING
    # ─────────────────────────────────────────────────────────────────
    def bind_chat_screen(self, chat_screen):
        self.chat_screen = chat_screen
        self.set_response_callback(chat_screen.receive_ai_message)

    def bind_status_bar(self, status_bar):
        self.status_bar = status_bar
        self.set_status_callback(status_bar.update_status)

    def bind_task_display(self, task_display):
        self.task_display = task_display

    def bind_planning_panel(self, planning_panel):
        self.planning_panel = planning_panel

    # ─────────────────────────────────────────────────────────────────
    # SEND MESSAGE
    # ─────────────────────────────────────────────────────────────────
    def send_user_message(self, text: str):
        """Called from UI thread — safely dispatch message to backend loop."""

        if getattr(self, "_shutdown", False):
            return

        if not text or not text.strip():
            return

        # ── Guard: backend not yet connected ─────────────────────────
        if not self._ready or not self.system:
            self._emit_status_safe("⏳ AI loading...")
            return

        # ── Guard: loop missing or dead ───────────────────────────────
        if not self.loop:
            self._emit_status_safe("❌ No event loop")
            return

        if not self.loop.is_running():
            logger.warning("⚠️ Backend loop not running — attempting fallback thread")
            self._fallback_thread(text.strip())
            return

        now = time.time()

        # ── Rate limit ────────────────────────────────────────────────
        if now - self._last_request_time < self._min_request_interval:
            self._emit_status_safe("⏱️ Please wait...")
            return

        # ── Prevent stuck state (auto-reset after 20 s) ───────────────
        if self._processing:
            if hasattr(self, "_last_processing_time"):
                if now - self._last_processing_time > 20:
                    logger.warning("⚠️ Resetting stuck processing state")
                    self._processing = False
                else:
                    self._emit_status_safe("⏳ Still processing...")
                    return
            else:
                self._emit_status_safe("⏳ Still processing...")
                return

        self._last_request_time = now
        self._last_processing_time = now
        self._processing = True

        self._emit_status_safe("🧠 Thinking...")
        self._show_typing_indicator_safe(True)

        logger.info("➡ Sending to backend: %s", text)

        # ── Submit to backend event loop ──────────────────────────────
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._process_message(text.strip()),
                self.loop,
            )

            def _handle_result(fut):
                try:
                    fut.result()
                except Exception as e:
                    logger.exception("❌ Backend future error: %s", e)
                    self._emit_response_safe(f"❌ Error: {str(e)}")
                    self._emit_status_safe("❌ Failed")
                    self._show_typing_indicator_safe(False)
                    self._processing = False

            future.add_done_callback(_handle_result)

        except Exception as e:
            logger.exception("❌ Failed to submit task: %s", e)
            self._emit_response_safe(f"❌ Submission error: {str(e)}")
            self._emit_status_safe("❌ Failed")
            self._show_typing_indicator_safe(False)
            self._processing = False

    # ─────────────────────────────────────────────────────────────────
    # FALLBACK: run in a fresh thread when the main loop is dead/busy
    # ─────────────────────────────────────────────────────────────────
    def _fallback_thread(self, text: str):
        """Run brain_process in a brand-new event loop on a worker thread."""

        logger.info("🔄 Fallback thread for: %s", text[:50])
        self._emit_status_safe("🔄 Fallback mode…")
        self._show_typing_indicator_safe(True)
        self._processing = True
        self._last_processing_time = time.time()

        def _run():
            response = "🤔 Could not generate a response."
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def _call():
                    # Strategy 1 — brain_process
                    try:
                        result = await asyncio.wait_for(
                            self.system.brain_process("text", {"text": text}),
                            timeout=20.0,
                        )
                        if result and result.get("success"):
                            return str(result.get("output", "")).strip()
                    except Exception as e:
                        logger.exception("❌ Fallback brain_process error: %s", e)

                    # Strategy 2 — agent.run
                    try:
                        if self.system.agent:
                            r = await asyncio.wait_for(
                                self.system.agent.run(text), timeout=15.0
                            )
                            return str(r or "").strip()
                    except Exception as e:
                        logger.exception("❌ Fallback agent.run error: %s", e)

                    return ""

                response = loop.run_until_complete(_call()) or response
            except Exception as e:
                response = f"❌ Fallback error: {e}"
                logger.exception(response)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
                self._show_typing_indicator_safe(False)
                self._emit_response_safe(response)
                self._emit_status_safe("✅ Ready")
                self._processing = False

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────
    # CORE ASYNC PROCESSOR
    # ─────────────────────────────────────────────────────────────────
    async def _process_message(self, text: str):
        """
        Runs on the backend event loop.
        Strategy 1 → brain_process (25 s timeout)
        Strategy 2 → agent.run      (20 s timeout)
        Strategy 3 → last-resort message
        Always resets _processing via try/finally.
        """
        response = ""
        try:
            if not self.system:
                response = "⚠️ System not connected."
                return

            logger.info("📤 Processing message: %s...", text[:50])

            # ── Strategy 1: brain_process ─────────────────────────────
            try:
                result = await asyncio.wait_for(
                    self.system.brain_process("text", {"text": text}),
                    timeout=25.0,
                )

                success = result.get("success") if result else None
                logger.debug("📥 brain_process result: success=%s", success)

                if result and isinstance(result, dict):
                    if result.get("success"):
                        response = str(result.get("output", "")).strip()
                        if response:
                            logger.info(
                                "✅ brain_process response: %s...", response[:80]
                            )
                    else:
                        err = str(result.get("error", "")).strip()
                        # If error is "brain busy" or "llm not ready", try fallback
                        if err in ("brain busy", "llm not ready", "llm timeout"):
                            logger.warning(
                                "⚠️ brain_process soft error: %s — trying agent fallback",
                                err,
                            )
                            response = ""
                        else:
                            response = f"⚠️ {err}" if err else "⚠️ Processing failed."
                else:
                    response = ""

            except asyncio.TimeoutError:
                response = "⏱️ Response timed out. Please try a shorter message."
                logger.error("❌ brain_process timeout after 25 s")
            except Exception as e:
                logger.exception("❌ brain_process error: %s", e)
                response = ""
                self._emit_status_safe("🔄 Fallback mode…")

            # ── Strategy 2: agent.run ─────────────────────────────────
            if not response and self.system.agent:
                try:
                    logger.info("🔄 Falling back to agent.run...")
                    agent_result = await asyncio.wait_for(
                        self.system.agent.run(text),
                        timeout=20.0,
                    )
                    response = str(agent_result or "").strip()
                    if response:
                        logger.info("✅ agent.run response: %s...", response[:80])
                except asyncio.TimeoutError:
                    response = "⏱️ The AI is taking too long to respond."
                    logger.error("❌ agent.run timeout after 20 s")
                except Exception as e:
                    response = f"❌ Error: {str(e)}"
                    logger.exception("❌ agent.run error: %s", e)

            # ── Strategy 3: last resort ───────────────────────────────
            if not response:
                response = "🤔 I couldn't generate a response. Please try again."
                logger.warning("⚠️ No response from any strategy")

        except Exception as e:
            response = f"❌ Unexpected error: {str(e)}"
            logger.exception("❌ Unexpected error in _process_message: %s", e)
        finally:
            logger.debug(
                "⬅ Received from backend: %s",
                (response[:80] if response else "(empty)"),
            )
            self._show_typing_indicator_safe(False)
            self._emit_response_safe(response)
            self._emit_status_safe("✅ Ready")
            self._processing = False

    # ─────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────
    def initialize(self, *args, **kwargs) -> bool:
        try:
            if self._initialized:
                return True

            self._initialized = True
            self._shutdown = False
            self._processing = False
            self.system = None
            self.loop = None
            self._ready = False

            logger.info("🧠 AIBackend initialized")
            return True
        except Exception as e:
            logger.exception("❌ AIBackend initialize failed: %s", e)
            return False

    def start(self) -> bool:
        try:
            if not self._initialized:
                self.initialize()

            if self._running:
                return True

            self._running = True
            self._shutdown = False

            logger.info("🚀 AIBackend started")
            return True
        except Exception as e:
            logger.exception("❌ AIBackend start failed: %s", e)
            self._running = False
            return False

    def stop(self) -> bool:
        try:
            if not self._running:
                return True

            self._running = False
            self._shutdown = True
            self._processing = False
            self.system = None
            self.loop = None
            self._ready = False

            try:
                self._emit_status_safe("🛑 Backend stopped")
            except Exception:
                pass

            logger.info("🛑 AIBackend stopped")
            return True
        except Exception as e:
            logger.exception("❌ AIBackend stop failed: %s", e)
            return False

    def shutdown(self):
        """Disconnect proxy — real shutdown is handled by main.py."""
        self._shutdown = True
        self._ready = False
        self._processing = False
        self.system = None
        self.loop = None

        try:
            for e in self._events:
                try:
                    e.cancel()
                except Exception:
                    pass
            self._events.clear()
        except Exception:
            pass

        try:
            self._response_callbacks.clear()
            self._status_callbacks.clear()
        except Exception:
            pass

        # Use logger for disconnect messages
        logger.info("👋 AIBackend disconnected")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # SYSTEM STATS
    # ─────────────────────────────────────────────────────────────────
    def get_system_stats(self) -> Dict[str, Any]:
        if not self._ready or not self.loop or not self.system:
            return {
                "status": "initializing",
                "ready": False,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "processing": self._processing,
            }

        if not self.loop.is_running():
            return {
                "status": "loop_dead",
                "ready": False,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "processing": self._processing,
            }

        future = asyncio.run_coroutine_threadsafe(self._get_stats(), self.loop)
        try:
            stats = future.result(timeout=2)
            stats["processing"] = self._processing
            return stats
        except Exception:
            return {
                "status": "timeout",
                "ready": self._ready,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "processing": self._processing,
            }

    async def _get_stats(self) -> Dict[str, Any]:
        try:
            if self.system and hasattr(self.system, "get_system_status"):
                return await self.system.get_system_status()
            elif self.system and hasattr(self.system, "get_status"):
                return self.system.get_status()
            return {
                "status": "running",
                "ready": self._ready,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
            }
        except Exception:
            return {
                "status": "error",
                "ready": self._ready,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
            }

    # ─────────────────────────────────────────────────────────────────
    # TASK / PLANNING DISPLAY
    # ─────────────────────────────────────────────────────────────────
    def update_task(self, task_name: str):
        def _update():
            if self.task_display:
                try:
                    if hasattr(self.task_display, "set_current_task"):
                        self.task_display.set_current_task(task_name)
                    elif hasattr(self.task_display, "set_task"):
                        self.task_display.set_task(task_name)
                except Exception:
                    pass

        Clock.schedule_once(lambda dt: _update(), 0)

    def update_planning_steps(self, steps: list):
        def _update():
            if self.planning_panel and hasattr(self.planning_panel, "update_steps"):
                try:
                    self.planning_panel.update_steps(steps)
                except Exception:
                    pass

        Clock.schedule_once(lambda dt: _update(), 0)

    # ─────────────────────────────────────────────────────────────────
    # UI-THREAD SAFE HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _show_typing_indicator_safe(self, show: bool):
        def _update():
            if self.chat_screen and hasattr(self.chat_screen, "_show_typing_indicator"):
                try:
                    self.chat_screen._show_typing_indicator(show)
                except Exception:
                    pass

        Clock.schedule_once(lambda dt: _update(), 0)

    def _emit_response_safe(self, text: str):
        def _update():
            if self._response_callbacks:
                for cb in list(self._response_callbacks):
                    try:
                        logger.debug(
                            "📤 Emitting response to UI: %s",
                            (text[:100] if text else "(empty)"),
                        )
                        cb(text)
                    except Exception as e:
                        logger.exception("Response callback error: %s", e)
            else:
                logger.debug("[RESPONSE] %s", (text[:100] if text else "(empty)"))

        Clock.schedule_once(lambda dt: _update(), 0)

    def _emit_status_safe(self, text: str):
        def _update():
            if self._status_callbacks:
                for cb in list(self._status_callbacks):
                    try:
                        cb(text)
                    except Exception as e:
                        logger.exception("Status callback error: %s", e)
            else:
                logger.info("[STATUS] %s", text)

        Clock.schedule_once(lambda dt: _update(), 0)

    # ─────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────────────
    def set_response_callback(self, func):
        try:
            if func not in self._response_callbacks:
                self._response_callbacks.append(func)
        except Exception:
            pass

    def set_status_callback(self, func):
        try:
            if func not in self._status_callbacks:
                self._status_callbacks.append(func)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return (
            self._ready
            and bool(self.loop)
            and (self.loop.is_running() if self.loop else False)
        )

    def get_status(self) -> str:
        if self._processing:
            return "processing"
        elif self._ready:
            return "ready"
        return "initializing"

    def is_processing(self) -> bool:
        return self._processing


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_backend_instance: Optional[AIBackend] = None


def get_backend() -> AIBackend:
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = AIBackend()
    return _backend_instance


__all__ = ["AIBackend", "get_backend"]
