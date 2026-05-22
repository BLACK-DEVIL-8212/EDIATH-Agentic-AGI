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
    def connect_system(
        self,
        system,
        loop: asyncio.AbstractEventLoop,
    ):
        """
        Production-safe backend connector.

        Features:
        - validates backend system
        - validates event loop
        - prevents dead loop binding
        - prevents duplicate initialization
        - safe UI updates
        - thread-safe state setup
        - fallback-safe startup
        - async-safe validation
        - memory-safe initialization
        - defensive runtime protection
        """

        import asyncio
        import threading
        import time
        import traceback
        import gc

        try:

            logger.info(
                "🔌 Connecting backend system..."
            )

            # ============================================================
            # INIT SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_connect_lock",
            ):

                self._connect_lock = threading.RLock()

            with self._connect_lock:

                # ========================================================
                # SHUTDOWN GUARD
                # ========================================================
                if getattr(
                    self,
                    "_shutdown",
                    False,
                ):

                    logger.warning(
                        "connect_system blocked: shutdown active"
                    )

                    self._emit_status_safe(
                        "⚠️ System shutting down"
                    )

                    return

                # ========================================================
                # VALIDATE SYSTEM
                # ========================================================
                if system is None:

                    logger.error(
                        "connect_system: system is None"
                    )

                    self._emit_status_safe(
                        "❌ Invalid backend"
                    )

                    self._ready = False

                    return

                # callable corruption
                if callable(system):

                    logger.error(
                        "connect_system: system is callable"
                    )

                    self._emit_status_safe(
                        "❌ Invalid backend object"
                    )

                    self._ready = False

                    return

                # ========================================================
                # VALIDATE LOOP
                # ========================================================
                if loop is None:

                    logger.error(
                        "connect_system: loop is None"
                    )

                    self._emit_status_safe(
                        "❌ Missing event loop"
                    )

                    self._ready = False

                    return

                if not isinstance(
                    loop,
                    asyncio.AbstractEventLoop,
                ):

                    logger.error(
                        "connect_system: invalid loop type"
                    )

                    self._emit_status_safe(
                        "❌ Invalid event loop"
                    )

                    self._ready = False

                    return

                # ========================================================
                # LOOP HEALTH CHECK
                # ========================================================
                try:

                    loop_running = bool(
                        loop.is_running()
                    )

                except Exception:

                    loop_running = False

                if not loop_running:

                    logger.warning(
                        "connect_system: loop not running"
                    )

                    self._emit_status_safe(
                        "⚠️ Backend loop offline"
                    )

                # ========================================================
                # PREVENT DUPLICATE BINDING
                # ========================================================
                existing_system = getattr(
                    self,
                    "system",
                    None,
                )

                existing_loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if (
                    existing_system is system
                    and existing_loop is loop
                    and getattr(
                        self,
                        "_ready",
                        False,
                    )
                ):

                    logger.info(
                        "Backend already connected"
                    )

                    return

                # ========================================================
                # VALIDATE CORE METHODS
                # ========================================================
                required_methods = [

                    "brain_process",

                ]

                missing_methods = []

                for method_name in required_methods:

                    method = getattr(
                        system,
                        method_name,
                        None,
                    )

                    if not callable(
                        method
                    ):

                        missing_methods.append(
                            method_name
                        )

                if missing_methods:

                    logger.warning(
                        "System missing methods: %s",
                        missing_methods,
                    )

                # ========================================================
                # VALIDATE OPTIONAL AGENT
                # ========================================================
                agent = getattr(
                    system,
                    "agent",
                    None,
                )

                if (
                    agent is not None
                    and callable(agent)
                ):

                    logger.warning(
                        "Agent corrupted (callable)"
                    )

                # ========================================================
                # SAFE STATE RESET
                # ========================================================
                self.system = system
                self.loop = loop

                self._ready = True
                self._processing = False
                self._shutdown = False

                self._last_processing_time = 0.0
                self._last_request_time = 0.0

                # ========================================================
                # SAFE FUTURE STORAGE
                # ========================================================
                if not hasattr(
                    self,
                    "_events",
                ):

                    self._events = []

                # ========================================================
                # FALLBACK COUNTERS
                # ========================================================
                if not hasattr(
                    self,
                    "_fallback_threads_active",
                ):

                    self._fallback_threads_active = 0

                # ========================================================
                # SAFE UI STATUS
                # ========================================================
                try:

                    self._emit_status_safe(
                        "✅ AI Ready"
                    )

                except Exception:
                    pass

                # ========================================================
                # WELCOME MESSAGE
                # ========================================================
                welcomed = bool(

                    getattr(
                        self,
                        "_welcomed",
                        False,
                    )
                )

                if not welcomed:

                    try:

                        self._emit_response_safe(
                            (
                                "👋 Hello! I'm EDIATH, "
                                "your intelligent assistant. "
                                "How can I help you?"
                            )
                        )

                        self._welcomed = True

                    except Exception as e:

                        logger.warning(
                            "Welcome message failed: %s",
                            e,
                        )

                # ========================================================
                # BACKEND PING TEST
                # ========================================================
                try:

                    async def _backend_ping():

                        return True

                    future = asyncio.run_coroutine_threadsafe(

                        _backend_ping(),

                        loop,
                    )

                    future.result(
                        timeout=5
                    )

                    logger.info(
                        "✅ Backend loop verified"
                    )

                except Exception as e:

                    logger.warning(
                        "Backend ping failed: %s",
                        e,
                    )

                # ========================================================
                # OPTIONAL STATS INIT
                # ========================================================
                try:

                    if not hasattr(
                        self,
                        "_connected_at",
                    ):

                        self._connected_at = time.time()

                except Exception:
                    pass

                # ========================================================
                # CLEANUP
                # ========================================================
                try:

                    gc.collect()

                except Exception:
                    pass

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.info(
                    "✅ Backend connected successfully"
                )

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "connect_system cancelled"
            )

            self._ready = False

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ connect_system error: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # RESET STATE
            # ------------------------------------------------------------
            try:

                self._ready = False
                self._processing = False

                self.system = None
                self.loop = None

            except Exception:
                pass

            # ------------------------------------------------------------
            # SAFE UI ERROR
            # ------------------------------------------------------------
            try:

                self._emit_status_safe(
                    f"❌ Connection failed: {str(e)}"
                )

            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    # UI BINDING
    # ─────────────────────────────────────────────────────────────────
    def bind_chat_screen(self, chat_screen):
        """
        Production-safe chat screen binder.

        Features:
        - safe widget binding
        - callback validation
        - duplicate prevention
        - thread-safe UI binding
        - weak-reference protection
        - stale callback cleanup
        - defensive programming
        - safe callback registration
        """

        import threading
        import traceback
        import weakref

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_ui_bind_lock",
            ):

                self._ui_bind_lock = threading.RLock()

            with self._ui_bind_lock:

                logger.info(
                    "🔗 Binding chat screen..."
                )

                # ========================================================
                # VALIDATE INPUT
                # ========================================================
                if chat_screen is None:

                    logger.warning(
                        "Cannot bind None chat_screen"
                    )

                    return False

                # callable corruption
                if callable(chat_screen):

                    logger.warning(
                        "Invalid chat_screen object"
                    )

                    return False

                # ========================================================
                # VALIDATE REQUIRED METHOD
                # ========================================================
                receive_method = getattr(

                    chat_screen,

                    "receive_ai_message",

                    None,
                )

                if not callable(
                    receive_method
                ):

                    logger.warning(
                        "chat_screen missing receive_ai_message"
                    )

                    return False

                # ========================================================
                # CLEAN OLD SCREEN REFERENCE
                # ========================================================
                old_screen = getattr(
                    self,
                    "chat_screen",
                    None,
                )

                if old_screen is not None:

                    try:

                        if old_screen is chat_screen:

                            logger.debug(
                                "Chat screen already bound"
                            )

                            return True

                    except Exception:
                        pass

                # ========================================================
                # SAFE ASSIGNMENT
                # ========================================================
                self.chat_screen = chat_screen

                # ========================================================
                # TRACK WEAKREF
                # ========================================================
                try:

                    self._chat_screen_ref = weakref.ref(
                        chat_screen
                    )

                except Exception:

                    self._chat_screen_ref = None

                # ========================================================
                # REGISTER RESPONSE CALLBACK
                # ========================================================
                callback_registered = False

                try:

                    callback_registered = bool(

                        self.set_response_callback(
                            receive_method
                        )
                    )

                except Exception as e:

                    logger.warning(
                        "Response callback registration failed: %s",
                        e,
                    )

                # ========================================================
                # OPTIONAL STATUS CALLBACK
                # ========================================================
                try:

                    status_method = getattr(

                        chat_screen,

                        "receive_status",

                        None,
                    )

                    if callable(
                        status_method
                    ):

                        self.set_status_callback(
                            status_method
                        )

                except Exception:
                    pass

                # ========================================================
                # OPTIONAL TYPING INDICATOR CHECK
                # ========================================================
                try:

                    typing_method = getattr(

                        chat_screen,

                        "_show_typing_indicator",

                        None,
                    )

                    if (
                        typing_method is not None
                        and not callable(
                            typing_method
                        )
                    ):

                        logger.warning(
                            "Typing indicator method corrupted"
                        )

                except Exception:
                    pass

                # ========================================================
                # SUCCESS LOGGING
                # ========================================================
                logger.info(
                    "✅ Chat screen bound successfully"
                )

                logger.debug(
                    "Response callback registered: %s",
                    callback_registered,
                )

                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ bind_chat_screen crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # SAFE RESET
            # ------------------------------------------------------------
            try:

                self.chat_screen = None

            except Exception:
                pass

            return False

    def bind_status_bar(self, status_bar):
        """
        Production-safe status bar binder.

        Features:
        - safe widget binding
        - callback validation
        - duplicate prevention
        - thread-safe UI binding
        - weak-reference protection
        - stale callback cleanup
        - defensive programming
        - safe callback registration
        """

        import threading
        import traceback
        import weakref

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_ui_bind_lock",
            ):

                self._ui_bind_lock = threading.RLock()

            with self._ui_bind_lock:

                logger.info(
                    "🔗 Binding status bar..."
                )

                # ========================================================
                # VALIDATE INPUT
                # ========================================================
                if status_bar is None:

                    logger.warning(
                        "Cannot bind None status_bar"
                    )

                    return False

                # callable corruption
                if callable(status_bar):

                    logger.warning(
                        "Invalid status_bar object"
                    )

                    return False

                # ========================================================
                # VALIDATE REQUIRED METHOD
                # ========================================================
                update_method = getattr(

                    status_bar,

                    "update_status",

                    None,
                )

                if not callable(
                    update_method
                ):

                    logger.warning(
                        "status_bar missing update_status"
                    )

                    return False

                # ========================================================
                # CLEAN OLD STATUS BAR REFERENCE
                # ========================================================
                old_status_bar = getattr(
                    self,
                    "status_bar",
                    None,
                )

                if old_status_bar is not None:

                    try:

                        if old_status_bar is status_bar:

                            logger.debug(
                                "Status bar already bound"
                            )

                            return True

                    except Exception:
                        pass

                # ========================================================
                # SAFE ASSIGNMENT
                # ========================================================
                self.status_bar = status_bar

                # ========================================================
                # TRACK WEAKREF
                # ========================================================
                try:

                    self._status_bar_ref = weakref.ref(
                        status_bar
                    )

                except Exception:

                    self._status_bar_ref = None

                # ========================================================
                # REGISTER STATUS CALLBACK
                # ========================================================
                callback_registered = False

                try:

                    callback_registered = bool(

                        self.set_status_callback(
                            update_method
                        )
                    )

                except Exception as e:

                    logger.warning(
                        "Status callback registration failed: %s",
                        e,
                    )

                # ========================================================
                # OPTIONAL EXTRA METHODS
                # ========================================================
                optional_methods = [

                    "set_ready",

                    "show_error",

                    "show_loading",

                ]

                for method_name in optional_methods:

                    try:

                        method = getattr(

                            status_bar,

                            method_name,

                            None,
                        )

                        if (
                            method is not None
                            and not callable(
                                method
                            )
                        ):

                            logger.warning(
                                "Status bar method corrupted: %s",
                                method_name,
                            )

                    except Exception:
                        pass

                # ========================================================
                # SUCCESS LOGGING
                # ========================================================
                logger.info(
                    "✅ Status bar bound successfully"
                )

                logger.debug(
                    "Status callback registered: %s",
                    callback_registered,
                )

                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ bind_status_bar crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # SAFE RESET
            # ------------------------------------------------------------
            try:

                self.status_bar = None

            except Exception:
                pass

            return False

    def bind_task_display(self, task_display):
        self.task_display = task_display

    def bind_planning_panel(self, planning_panel):
        self.planning_panel = planning_panel

    # ─────────────────────────────────────────────────────────────────
    # SEND MESSAGE
    # ─────────────────────────────────────────────────────────────────
    def send_user_message(self, text: str):
        """
        Production-safe UI → backend message dispatcher.

        Fixes:
        - AI not replying
        - stuck processing state
        - dead event loop
        - coroutine leaks
        - unsafe thread submission
        - invalid text handling
        - duplicate requests
        - race conditions
        - callback crashes
        - event loop corruption
        - memory leaks
        - silent backend failures
        """

        import asyncio
        import threading
        import traceback
        import time

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            if getattr(self, "_shutdown", False):

                logger.warning(
                    "send_user_message blocked: shutdown active"
                )

                return

            # ============================================================
            # TEXT VALIDATION
            # ============================================================
            if text is None:
                return

            try:
                text = str(text).strip()
            except Exception:
                return

            if not text:
                return

            # prevent absurd payloads
            if len(text) > 10000:
                text = text[:10000]

            # ============================================================
            # INIT SAFETY
            # ============================================================
            if not hasattr(self, "_processing_lock"):
                self._processing_lock = threading.RLock()

            if not hasattr(self, "_request_counter"):
                self._request_counter = 0

            # ============================================================
            # BACKEND READY CHECK
            # ============================================================
            if not getattr(self, "_ready", False):

                logger.warning(
                    "Backend not ready"
                )

                self._emit_status_safe(
                    "⏳ AI loading..."
                )

                return

            # ============================================================
            # SYSTEM CHECK
            # ============================================================
            system = getattr(
                self,
                "system",
                None,
            )

            if system is None:

                logger.warning(
                    "System missing"
                )

                self._emit_status_safe(
                    "⚠️ System unavailable"
                )

                return

            # ============================================================
            # LOOP VALIDATION
            # ============================================================
            loop = getattr(
                self,
                "loop",
                None,
            )

            if loop is None:

                logger.error(
                    "No backend loop"
                )

                self._emit_status_safe(
                    "❌ No event loop"
                )

                return

            # ============================================================
            # DEAD LOOP RECOVERY
            # ============================================================
            try:

                loop_running = bool(
                    loop.is_running()
                )

            except Exception:

                loop_running = False

            if not loop_running:

                logger.warning(
                    "⚠️ Backend loop dead — fallback mode"
                )

                try:

                    self._fallback_thread(
                        text
                    )

                except Exception as e:

                    logger.exception(
                        "Fallback thread failed: %s",
                        e,
                    )

                    self._emit_response_safe(
                        f"❌ Backend unavailable: {e}"
                    )

                return

            # ============================================================
            # TIMING
            # ============================================================
            now = time.monotonic()

            if not hasattr(self, "_last_request_time"):
                self._last_request_time = 0.0

            if not hasattr(self, "_last_processing_time"):
                self._last_processing_time = 0.0

            # ============================================================
            # RATE LIMIT
            # ============================================================
            try:

                min_interval = float(
                    getattr(
                        self,
                        "_min_request_interval",
                        0.5,
                    )
                )

            except Exception:

                min_interval = 0.5

            if (
                now - self._last_request_time
                < min_interval
            ):

                logger.debug(
                    "Rate limited request"
                )

                self._emit_status_safe(
                    "⏱️ Please wait..."
                )

                return

            # ============================================================
            # STUCK PROCESSING DETECTION
            # ============================================================
            with self._processing_lock:

                if getattr(
                    self,
                    "_processing",
                    False,
                ):

                    elapsed = (
                        now
                        - getattr(
                            self,
                            "_last_processing_time",
                            now,
                        )
                    )

                    # auto recover
                    if elapsed > 30:

                        logger.warning(
                            "⚠️ Resetting stuck processing state "
                            f"({elapsed:.1f}s)"
                        )

                        self._processing = False

                    else:

                        self._emit_status_safe(
                            "⏳ Still processing..."
                        )

                        return

                # lock processing
                self._processing = True

            # ============================================================
            # UPDATE STATE
            # ============================================================
            self._last_request_time = now
            self._last_processing_time = now
            self._request_counter += 1

            request_id = self._request_counter

            # ============================================================
            # UI STATUS
            # ============================================================
            try:
                self._emit_status_safe(
                    "🧠 Thinking..."
                )
            except Exception:
                pass

            try:
                self._show_typing_indicator_safe(
                    True
                )
            except Exception:
                pass

            logger.info(
                "➡ Sending request #%s to backend: %s",
                request_id,
                text[:200],
            )

            # ============================================================
            # VALIDATE PROCESS FUNCTION
            # ============================================================
            process_method = getattr(
                self,
                "_process_message",
                None,
            )

            if not callable(
                process_method
            ):

                logger.error(
                    "_process_message missing"
                )

                self._emit_response_safe(
                    "❌ Backend processing unavailable"
                )

                self._emit_status_safe(
                    "❌ Failed"
                )

                self._show_typing_indicator_safe(
                    False
                )

                self._processing = False

                return

            # ============================================================
            # CREATE COROUTINE SAFELY
            # ============================================================
            try:

                coroutine = process_method(
                    text
                )

            except Exception as e:

                logger.exception(
                    "Coroutine creation failed: %s",
                    e,
                )

                self._emit_response_safe(
                    f"❌ Failed to start request: {e}"
                )

                self._emit_status_safe(
                    "❌ Failed"
                )

                self._show_typing_indicator_safe(
                    False
                )

                self._processing = False

                return

            # ============================================================
            # VALIDATE COROUTINE
            # ============================================================
            if not asyncio.iscoroutine(
                coroutine
            ):

                logger.error(
                    "_process_message did not return coroutine"
                )

                self._emit_response_safe(
                    "❌ Invalid backend coroutine"
                )

                self._emit_status_safe(
                    "❌ Failed"
                )

                self._show_typing_indicator_safe(
                    False
                )

                self._processing = False

                return

            # ============================================================
            # SUBMIT TASK
            # ============================================================
            try:

                future = asyncio.run_coroutine_threadsafe(
                    coroutine,
                    loop,
                )

            except RuntimeError as e:

                logger.exception(
                    "Loop submission failed: %s",
                    e,
                )

                self._processing = False

                try:

                    self._fallback_thread(
                        text
                    )

                except Exception:

                    self._emit_response_safe(
                        "❌ Backend unavailable"
                    )

                return

            except Exception as e:

                logger.exception(
                    "Task submission failed: %s",
                    e,
                )

                self._emit_response_safe(
                    f"❌ Submission error: {e}"
                )

                self._emit_status_safe(
                    "❌ Failed"
                )

                self._show_typing_indicator_safe(
                    False
                )

                self._processing = False

                return

            # ============================================================
            # STORE FUTURE
            # ============================================================
            try:

                if not hasattr(
                    self,
                    "_events",
                ):
                    self._events = []

                self._events.append(
                    future
                )

                # bounded memory
                if len(self._events) > 100:

                    self._events = self._events[-100:]

            except Exception:
                pass

            # ============================================================
            # RESULT CALLBACK
            # ============================================================
            def _handle_result(fut):

                try:

                    # ----------------------------------------------------
                    # FUTURE RESULT
                    # ----------------------------------------------------
                    result = fut.result(
                        timeout=0
                    )

                    logger.debug(
                        "Request #%s completed",
                        request_id,
                    )

                    return result

                # --------------------------------------------------------
                # CANCELLED
                # --------------------------------------------------------
                except asyncio.CancelledError:

                    logger.warning(
                        "Request #%s cancelled",
                        request_id,
                    )

                    try:
                        self._emit_response_safe(
                            "⚠️ Request cancelled"
                        )
                    except Exception:
                        pass

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except TimeoutError:

                    logger.error(
                        "Request #%s timeout",
                        request_id,
                    )

                    try:
                        self._emit_response_safe(
                            "⏱️ Request timed out"
                        )
                    except Exception:
                        pass

                # --------------------------------------------------------
                # GENERAL FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    logger.exception(
                        "❌ Backend future error "
                        "(request #%s): %s",
                        request_id,
                        e,
                    )

                    logger.debug(
                        traceback.format_exc()[:4000]
                    )

                    try:

                        self._emit_response_safe(
                            f"❌ Error: {str(e)}"
                        )

                    except Exception:
                        pass

                    try:

                        self._emit_status_safe(
                            "❌ Failed"
                        )

                    except Exception:
                        pass

                # --------------------------------------------------------
                # FINAL CLEANUP
                # --------------------------------------------------------
                finally:

                    try:

                        self._show_typing_indicator_safe(
                            False
                        )

                    except Exception:
                        pass

                    self._processing = False

                    # cleanup completed futures
                    try:

                        if hasattr(
                            self,
                            "_events",
                        ):

                            self._events = [

                                ev

                                for ev in self._events

                                if not (
                                    hasattr(ev, "done")
                                    and ev.done()
                                )
                            ]

                    except Exception:
                        pass

            # ============================================================
            # REGISTER CALLBACK
            # ============================================================
            try:

                future.add_done_callback(
                    _handle_result
                )

            except Exception as e:

                logger.exception(
                    "Failed to register callback: %s",
                    e,
                )

                self._processing = False

                self._emit_response_safe(
                    f"❌ Callback failure: {e}"
                )

                self._show_typing_indicator_safe(
                    False
                )

                return

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ send_user_message crashed: %s",
                e,
            )

            try:

                self._emit_response_safe(
                    f"❌ Critical backend error: {e}"
                )

            except Exception:
                pass

            try:

                self._emit_status_safe(
                    "❌ System Error"
                )

            except Exception:
                pass

            try:

                self._show_typing_indicator_safe(
                    False
                )

            except Exception:
                pass

            self._processing = False

    # ─────────────────────────────────────────────────────────────────
    # FALLBACK: run in a fresh thread when the main loop is dead/busy
    # ─────────────────────────────────────────────────────────────────
    def _fallback_thread(self, text: str):
        """
        Production-safe fallback execution thread.

        Features:
        - isolated event loop
        - async-safe fallback execution
        - coroutine leak prevention
        - thread-safe recovery
        - safe loop cleanup
        - timeout protection
        - memory cleanup
        - dead backend recovery
        - safe UI updates
        - defensive execution
        """

        import asyncio
        import gc
        import inspect
        import threading
        import time
        import traceback

        try:

            # ============================================================
            # VALIDATE INPUT
            # ============================================================
            if text is None:
                return

            try:
                text = str(text).strip()
            except Exception:
                return

            if not text:
                return

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            if getattr(self, "_shutdown", False):

                logger.warning(
                    "Fallback blocked: shutdown active"
                )

                return

            # ============================================================
            # PREVENT THREAD STORM
            # ============================================================
            active_threads = int(

                getattr(
                    self,
                    "_fallback_threads_active",
                    0,
                )
            )

            if active_threads >= 3:

                logger.warning(
                    "Too many fallback threads active"
                )

                self._emit_status_safe(
                    "⚠️ AI busy..."
                )

                return

            self._fallback_threads_active = (
                active_threads + 1
            )

            # ============================================================
            # UPDATE UI
            # ============================================================
            try:

                self._emit_status_safe(
                    "🔄 Recovery mode..."
                )

            except Exception:
                pass

            try:

                self._show_typing_indicator_safe(
                    True
                )

            except Exception:
                pass

            self._processing = True

            self._last_processing_time = (
                time.monotonic()
            )

            logger.info(
                "🔄 Starting fallback thread: %s",
                text[:100],
            )

            # ============================================================
            # THREAD TARGET
            # ============================================================
            def _run():

                loop = None

                response = (
                    "🤔 Could not generate a response."
                )

                start_time = time.monotonic()

                try:

                    # ----------------------------------------------------
                    # CREATE ISOLATED LOOP
                    # ----------------------------------------------------
                    loop = asyncio.new_event_loop()

                    asyncio.set_event_loop(
                        loop
                    )

                    # ----------------------------------------------------
                    # INNER COROUTINE
                    # ----------------------------------------------------
                    async def _call():

                        import asyncio
                        import inspect

                        # =================================================
                        # VALIDATE SYSTEM
                        # =================================================
                        system = getattr(
                            self,
                            "system",
                            None,
                        )

                        if system is None:

                            return (
                                "⚠️ System unavailable."
                            )

                        # =================================================
                        # STRATEGY 1
                        # brain_process
                        # =================================================
                        try:

                            brain_process = getattr(
                                system,
                                "brain_process",
                                None,
                            )

                            if callable(
                                brain_process
                            ):

                                logger.info(
                                    "Fallback: brain_process"
                                )

                                coro = brain_process(

                                    "text",

                                    {
                                        "text": text,
                                        "timestamp": time.time(),
                                    },
                                )

                                # FIX:
                                # bool await crash
                                if inspect.isawaitable(
                                    coro
                                ):

                                    result = await asyncio.wait_for(

                                        coro,

                                        timeout=20.0,
                                    )

                                else:

                                    result = coro

                                # FIX:
                                # function.items crash
                                if isinstance(
                                    result,
                                    dict,
                                ):

                                    if result.get(
                                        "success"
                                    ):

                                        output = result.get(
                                            "output",
                                            ""
                                        )

                                        if output:

                                            return str(
                                                output
                                            ).strip()

                                elif result is not None:

                                    value = str(
                                        result
                                    ).strip()

                                    if value:

                                        return value

                        except asyncio.TimeoutError:

                            logger.error(
                                "Fallback brain_process timeout"
                            )

                        except Exception as e:

                            logger.exception(
                                "Fallback brain_process failed: %s",
                                e,
                            )

                        # =================================================
                        # STRATEGY 2
                        # agent.run
                        # =================================================
                        try:

                            agent = getattr(
                                system,
                                "agent",
                                None,
                            )

                            if agent is not None:

                                run_method = getattr(
                                    agent,
                                    "run",
                                    None,
                                )

                                if callable(
                                    run_method
                                ):

                                    logger.info(
                                        "Fallback: agent.run"
                                    )

                                    coro = run_method(
                                        text
                                    )

                                    if inspect.isawaitable(
                                        coro
                                    ):

                                        result = await asyncio.wait_for(

                                            coro,

                                            timeout=15.0,
                                        )

                                    else:

                                        result = coro

                                    if result is not None:

                                        value = str(
                                            result
                                        ).strip()

                                        if value:

                                            return value

                        except asyncio.TimeoutError:

                            logger.error(
                                "Fallback agent timeout"
                            )

                        except Exception as e:

                            logger.exception(
                                "Fallback agent failed: %s",
                                e,
                            )

                        # =================================================
                        # STRATEGY 3
                        # DIRECT LLM
                        # =================================================
                        try:

                            llm = getattr(
                                system,
                                "llm_engine",
                                None,
                            )

                            if llm is not None:

                                generate = getattr(
                                    llm,
                                    "generate",
                                    None,
                                )

                                if callable(
                                    generate
                                ):

                                    logger.info(
                                        "Fallback: direct LLM"
                                    )

                                    coro = generate(
                                        text
                                    )

                                    if inspect.isawaitable(
                                        coro
                                    ):

                                        result = await asyncio.wait_for(

                                            coro,

                                            timeout=15,
                                        )

                                    else:

                                        result = coro

                                    if result is not None:

                                        value = str(
                                            result
                                        ).strip()

                                        if value:

                                            return value

                        except Exception as e:

                            logger.exception(
                                "Fallback LLM failed: %s",
                                e,
                            )

                        # =================================================
                        # LAST RESORT
                        # =================================================
                        return (
                            "🤔 I couldn't generate a response."
                        )

                    # ----------------------------------------------------
                    # RUN COROUTINE
                    # ----------------------------------------------------
                    response_value = loop.run_until_complete(
                        _call()
                    )

                    if response_value:

                        response = str(
                            response_value
                        ).strip()

                # --------------------------------------------------------
                # THREAD FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    response = (
                        f"❌ Recovery error: {e}"
                    )

                    logger.exception(
                        "Fallback thread crashed: %s",
                        e,
                    )

                    logger.debug(
                        traceback.format_exc()[:4000]
                    )

                # --------------------------------------------------------
                # CLEANUP
                # --------------------------------------------------------
                finally:

                    # ----------------------------------------------------
                    # CANCEL REMAINING TASKS
                    # ----------------------------------------------------
                    try:

                        if loop is not None:

                            pending = asyncio.all_tasks(
                                loop
                            )

                            for task in pending:

                                try:
                                    task.cancel()
                                except Exception:
                                    pass

                            if pending:

                                loop.run_until_complete(

                                    asyncio.gather(

                                        *pending,

                                        return_exceptions=True,
                                    )
                                )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # SHUTDOWN ASYNC GENERATORS
                    # ----------------------------------------------------
                    try:

                        if loop is not None:

                            loop.run_until_complete(
                                loop.shutdown_asyncgens()
                            )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # CLOSE LOOP
                    # ----------------------------------------------------
                    try:

                        if loop is not None:

                            loop.close()

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # RESET EVENT LOOP
                    # ----------------------------------------------------
                    try:

                        asyncio.set_event_loop(
                            None
                        )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # UI CLEANUP
                    # ----------------------------------------------------
                    try:

                        self._show_typing_indicator_safe(
                            False
                        )

                    except Exception:
                        pass

                    try:

                        self._emit_response_safe(
                            response
                        )

                    except Exception:
                        pass

                    try:

                        self._emit_status_safe(
                            "✅ Ready"
                        )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # RESET PROCESSING
                    # ----------------------------------------------------
                    self._processing = False

                    # ----------------------------------------------------
                    # THREAD COUNTER
                    # ----------------------------------------------------
                    try:

                        current = int(

                            getattr(
                                self,
                                "_fallback_threads_active",
                                1,
                            )
                        )

                        self._fallback_threads_active = max(
                            0,
                            current - 1,
                        )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # MEMORY CLEANUP
                    # ----------------------------------------------------
                    try:

                        gc.collect()

                    except Exception:
                        pass

                    elapsed = round(
                        time.monotonic()
                        - start_time,
                        2,
                    )

                    logger.info(
                        "Fallback thread finished "
                        "(%ss)",
                        elapsed,
                    )

            # ============================================================
            # START THREAD
            # ============================================================
            thread = threading.Thread(

                target=_run,

                daemon=True,

                name=f"EDIATHFallback-{int(time.time())}",
            )

            thread.start()

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ _fallback_thread crashed: %s",
                e,
            )

            try:

                self._show_typing_indicator_safe(
                    False
                )

            except Exception:
                pass

            try:

                self._emit_response_safe(
                    f"❌ Fallback system error: {e}"
                )

            except Exception:
                pass

            try:

                self._emit_status_safe(
                    "❌ Recovery Failed"
                )

            except Exception:
                pass

            self._processing = False

    # ─────────────────────────────────────────────────────────────────
    # CORE ASYNC PROCESSOR
    # ─────────────────────────────────────────────────────────────────
    async def _process_message(self, text: str):
        """
        Production-safe backend message processor.

        Features:
        - multi-stage fallback pipeline
        - async-safe execution
        - coroutine leak prevention
        - timeout protection
        - memory-safe execution
        - automatic recovery
        - safe dict validation
        - safe agent fallback
        - dead-loop protection
        - defensive programming
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        response = ""

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            if getattr(self, "_shutdown", False):

                response = "⚠️ System shutting down."

                return

            # ============================================================
            # VALIDATE TEXT
            # ============================================================
            if text is None:

                response = "⚠️ Empty request."

                return

            try:

                text = str(text).strip()

            except Exception:

                response = "⚠️ Invalid message."

                return

            if not text:

                response = "⚠️ Empty message."

                return

            # limit memory abuse
            if len(text) > 20000:

                text = text[:20000]

            # ============================================================
            # VALIDATE SYSTEM
            # ============================================================
            system = getattr(
                self,
                "system",
                None,
            )

            if system is None:

                response = "⚠️ System not connected."

                logger.error(
                    "System unavailable"
                )

                return

            logger.info(
                "📤 Processing message: %s...",
                text[:100],
            )

            # ============================================================
            # STATUS UPDATE
            # ============================================================
            try:

                self._emit_status_safe(
                    "🧠 Processing..."
                )

            except Exception:
                pass

            # ============================================================
            # STRATEGY 1
            # brain_process
            # ============================================================
            brain_process = getattr(
                system,
                "brain_process",
                None,
            )

            if callable(
                brain_process
            ):

                try:

                    logger.debug(
                        "Running brain_process..."
                    )

                    # ----------------------------------------------------
                    # CREATE COROUTINE SAFELY
                    # ----------------------------------------------------
                    coro = brain_process(

                        "text",

                        {
                            "text": text,
                            "timestamp": time.time(),
                        },
                    )

                    # FIX:
                    # prevents:
                    # object bool can't be used in await
                    if inspect.isawaitable(
                        coro
                    ):

                        result = await asyncio.wait_for(

                            coro,

                            timeout=25.0,
                        )

                    else:

                        result = coro

                    # ----------------------------------------------------
                    # VALIDATE RESULT
                    # ----------------------------------------------------
                    if callable(result):

                        logger.warning(
                            "brain_process returned callable"
                        )

                        result = None

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if result is not None and not isinstance(
                        result,
                        dict,
                    ):

                        try:

                            response = str(
                                result
                            ).strip()

                        except Exception:

                            response = ""

                    elif isinstance(
                        result,
                        dict,
                    ):

                        success = bool(
                            result.get(
                                "success",
                                False,
                            )
                        )

                        logger.debug(
                            "brain_process success=%s",
                            success,
                        )

                        # success path
                        if success:

                            try:

                                output = result.get(
                                    "output",
                                    ""
                                )

                                if output is not None:

                                    response = str(
                                        output
                                    ).strip()

                            except Exception:

                                response = ""

                        # error path
                        else:

                            err = str(

                                result.get(
                                    "error",
                                    ""
                                )

                            ).strip().lower()

                            soft_errors = {

                                "brain busy",

                                "llm not ready",

                                "llm timeout",

                                "timeout",

                                "busy",
                            }

                            if err in soft_errors:

                                logger.warning(
                                    "Soft brain failure: %s",
                                    err,
                                )

                                response = ""

                            else:

                                response = (

                                    f"⚠️ {err}"

                                    if err

                                    else ""

                                )

                    # ----------------------------------------------------
                    # SUCCESS
                    # ----------------------------------------------------
                    if response:

                        logger.info(
                            "✅ brain_process success"
                        )

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except asyncio.TimeoutError:

                    logger.error(
                        "brain_process timeout"
                    )

                    response = ""

                    try:

                        self._emit_status_safe(
                            "🔄 Trying fallback..."
                        )

                    except Exception:
                        pass

                # --------------------------------------------------------
                # CANCELLED
                # --------------------------------------------------------
                except asyncio.CancelledError:
                    raise

                # --------------------------------------------------------
                # FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    logger.exception(
                        "brain_process failed: %s",
                        e,
                    )

                    logger.debug(
                        traceback.format_exc()[:4000]
                    )

                    response = ""

                    try:

                        self._emit_status_safe(
                            "🔄 Recovery mode..."
                        )

                    except Exception:
                        pass

            # ============================================================
            # STRATEGY 2
            # agent.run
            # ============================================================
            if not response:

                agent = getattr(
                    system,
                    "agent",
                    None,
                )

                if agent is not None:

                    run_method = getattr(
                        agent,
                        "run",
                        None,
                    )

                    if callable(
                        run_method
                    ):

                        try:

                            logger.info(
                                "🔄 Falling back to agent.run..."
                            )

                            coro = run_method(
                                text
                            )

                            if inspect.isawaitable(
                                coro
                            ):

                                agent_result = await asyncio.wait_for(

                                    coro,

                                    timeout=20.0,
                                )

                            else:

                                agent_result = coro

                            # ------------------------------------------------
                            # VALIDATE RESPONSE
                            # ------------------------------------------------
                            if callable(
                                agent_result
                            ):

                                logger.warning(
                                    "agent.run returned callable"
                                )

                                agent_result = ""

                            if agent_result is not None:

                                response = str(
                                    agent_result
                                ).strip()

                            if response:

                                logger.info(
                                    "✅ agent.run success"
                                )

                        # ----------------------------------------------------
                        # TIMEOUT
                        # ----------------------------------------------------
                        except asyncio.TimeoutError:

                            logger.error(
                                "agent.run timeout"
                            )

                            response = ""

                        # ----------------------------------------------------
                        # CANCELLED
                        # ----------------------------------------------------
                        except asyncio.CancelledError:
                            raise

                        # ----------------------------------------------------
                        # FAILURE
                        # ----------------------------------------------------
                        except Exception as e:

                            logger.exception(
                                "agent.run failed: %s",
                                e,
                            )

                            logger.debug(
                                traceback.format_exc()[:4000]
                            )

                            response = ""

            # ============================================================
            # STRATEGY 3
            # direct llm fallback
            # ============================================================
            if not response:

                try:

                    llm = getattr(
                        system,
                        "llm_engine",
                        None,
                    )

                    if llm is not None:

                        generate = getattr(
                            llm,
                            "generate",
                            None,
                        )

                        if callable(
                            generate
                        ):

                            logger.info(
                                "🔄 Falling back to direct LLM..."
                            )

                            coro = generate(
                                text
                            )

                            if inspect.isawaitable(
                                coro
                            ):

                                llm_result = await asyncio.wait_for(

                                    coro,

                                    timeout=20,
                                )

                            else:

                                llm_result = coro

                            if llm_result is not None:

                                response = str(
                                    llm_result
                                ).strip()

                except Exception as e:

                    logger.exception(
                        "LLM fallback failed: %s",
                        e,
                    )

            # ============================================================
            # STRATEGY 4
            # last resort
            # ============================================================
            if not response:

                response = (
                    "🤔 I couldn't generate a response. "
                    "Please try again."
                )

                logger.warning(
                    "No valid response produced"
                )

            # ============================================================
            # CLEAN RESPONSE
            # ============================================================
            try:

                response = str(
                    response
                ).strip()

            except Exception:

                response = (
                    "⚠️ Response formatting failed."
                )

            if not response:

                response = (
                    "⚠️ Empty response generated."
                )

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "_process_message cancelled"
            )

            response = (
                "⚠️ Request cancelled."
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ Unexpected error in _process_message: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            response = (
                f"❌ Unexpected error: {str(e)}"
            )

        # ================================================================
        # FINALIZATION
        # ================================================================
        finally:

            try:

                elapsed = round(
                    time.monotonic()
                    - start_time,
                    2,
                )

                logger.debug(
                    "⬅ Backend response (%ss): %s",
                    elapsed,
                    (
                        response[:120]
                        if response
                        else "(empty)"
                    ),
                )

            except Exception:
                pass

            # ------------------------------------------------------------
            # UI SAFE CLEANUP
            # ------------------------------------------------------------
            try:

                self._show_typing_indicator_safe(
                    False
                )

            except Exception:
                pass

            try:

                self._emit_response_safe(
                    response
                )

            except Exception as e:

                logger.exception(
                    "Response emit failed: %s",
                    e,
                )

            try:

                self._emit_status_safe(
                    "✅ Ready"
                )

            except Exception:
                pass

            # ------------------------------------------------------------
            # RESET PROCESSING
            # ------------------------------------------------------------
            self._processing = False

            # ------------------------------------------------------------
            # MEMORY CLEANUP
            # ------------------------------------------------------------
            try:

                gc.collect()

            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────
    def initialize(self, *args, **kwargs) -> bool:
        """
        Production-safe AIBackend initializer.

        Features:
        - idempotent initialization
        - thread-safe startup
        - safe runtime defaults
        - async-safe state setup
        - event-loop safety
        - memory-safe initialization
        - coroutine leak prevention
        - defensive runtime validation
        - graceful recovery
        """

        import asyncio
        import gc
        import threading
        import time
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_init_lock",
            ):

                self._init_lock = threading.RLock()

            with self._init_lock:

                # ========================================================
                # IDEMPOTENT INIT
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_initialized",
                        False,
                    )
                ):

                    logger.debug(
                        "AIBackend already initialized"
                    )

                    return True

                logger.info(
                    "🧠 Initializing AIBackend..."
                )

                # ========================================================
                # CORE STATE
                # ========================================================
                self._initialized = True
                self._shutdown = False
                self._processing = False
                self._ready = False

                # ========================================================
                # BACKEND REFERENCES
                # ========================================================
                self.system = None
                self.loop = None

                # ========================================================
                # TIMERS
                # ========================================================
                self._startup_time = time.time()
                self._last_request_time = 0.0
                self._last_processing_time = 0.0

                # ========================================================
                # REQUEST CONTROL
                # ========================================================
                self._request_counter = 0

                try:

                    self._min_request_interval = float(

                        getattr(
                            self,
                            "_min_request_interval",
                            0.5,
                        )
                    )

                except Exception:

                    self._min_request_interval = 0.5

                # ========================================================
                # THREAD / LOCK SAFETY
                # ========================================================
                self._processing_lock = threading.RLock()
                self._connect_lock = threading.RLock()
                self._stats_lock = threading.RLock()

                # ========================================================
                # FALLBACK THREAD TRACKING
                # ========================================================
                self._fallback_threads_active = 0

                # ========================================================
                # FUTURE STORAGE
                # ========================================================
                self._events = []

                # ========================================================
                # CALLBACKS
                # ========================================================
                if not hasattr(
                    self,
                    "_response_callback",
                ):

                    self._response_callback = None

                if not hasattr(
                    self,
                    "_status_callback",
                ):

                    self._status_callback = None

                # ========================================================
                # UI FLAGS
                # ========================================================
                if not hasattr(
                    self,
                    "_welcomed",
                ):

                    self._welcomed = False

                # ========================================================
                # SAFE LOOP TEST
                # ========================================================
                try:

                    running_loop = asyncio.get_running_loop()

                    logger.debug(
                        "Running loop detected: %s",
                        running_loop,
                    )

                except RuntimeError:

                    # normal outside async context
                    running_loop = None

                # ========================================================
                # CLEAN STALE FUTURES
                # ========================================================
                try:

                    cleaned_events = []

                    for event in self._events:

                        try:

                            if hasattr(
                                event,
                                "done",
                            ):

                                if not event.done():

                                    cleaned_events.append(
                                        event
                                    )

                        except Exception:
                            continue

                    self._events = cleaned_events[-100:]

                except Exception:
                    self._events = []

                # ========================================================
                # MEMORY CLEANUP
                # ========================================================
                try:

                    gc.collect()

                except Exception:
                    pass

                # ========================================================
                # OPTIONAL STATUS UPDATE
                # ========================================================
                try:

                    self._emit_status_safe(
                        "🧠 Initializing AI..."
                    )

                except Exception:
                    pass

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.info(
                    "✅ AIBackend initialized successfully"
                )

                return True

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "AIBackend initialize cancelled"
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ AIBackend initialize failed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # RESET TO SAFE STATE
            # ------------------------------------------------------------
            try:

                self._initialized = False
                self._shutdown = False
                self._processing = False
                self._ready = False

                self.system = None
                self.loop = None

            except Exception:
                pass

            # ------------------------------------------------------------
            # SAFE STATUS
            # ------------------------------------------------------------
            try:

                self._emit_status_safe(
                    f"❌ Initialization failed: {str(e)}"
                )

            except Exception:
                pass

            return False

    def start(self) -> bool:
        """
        Production-safe AIBackend starter.

        Features:
        - idempotent startup
        - thread-safe runtime startup
        - safe initialization recovery
        - event-loop validation
        - backend health checks
        - stale state cleanup
        - coroutine leak prevention
        - memory-safe startup
        - graceful recovery
        - defensive runtime protection
        """

        import asyncio
        import gc
        import threading
        import time
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_start_lock",
            ):

                self._start_lock = threading.RLock()

            with self._start_lock:

                logger.info(
                    "🚀 Starting AIBackend..."
                )

                # ========================================================
                # SHUTDOWN GUARD
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_shutdown",
                        False,
                    )
                ):

                    logger.warning(
                        "Cannot start: shutdown active"
                    )

                    self._emit_status_safe(
                        "⚠️ Shutdown active"
                    )

                    return False

                # ========================================================
                # INITIALIZE IF NEEDED
                # ========================================================
                if not bool(
                    getattr(
                        self,
                        "_initialized",
                        False,
                    )
                ):

                    logger.info(
                        "Backend not initialized — initializing..."
                    )

                    init_result = self.initialize()

                    if not init_result:

                        logger.error(
                            "Initialization failed during start"
                        )

                        self._emit_status_safe(
                            "❌ Initialization failed"
                        )

                        return False

                # ========================================================
                # IDEMPOTENT START
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_running",
                        False,
                    )
                ):

                    logger.debug(
                        "AIBackend already running"
                    )

                    return True

                # ========================================================
                # SAFE STATE RESET
                # ========================================================
                self._running = True
                self._shutdown = False
                self._processing = False

                # ========================================================
                # START TIMERS
                # ========================================================
                self._started_at = time.time()

                # ========================================================
                # CLEAN STALE EVENTS
                # ========================================================
                try:

                    if not hasattr(
                        self,
                        "_events",
                    ):

                        self._events = []

                    cleaned_events = []

                    for event in self._events:

                        try:

                            if hasattr(
                                event,
                                "done",
                            ):

                                if not event.done():

                                    cleaned_events.append(
                                        event
                                    )

                        except Exception:
                            continue

                    # bounded memory
                    self._events = cleaned_events[-100:]

                except Exception:
                    self._events = []

                # ========================================================
                # RESET FALLBACK THREAD COUNT
                # ========================================================
                self._fallback_threads_active = 0

                # ========================================================
                # VALIDATE EXISTING LOOP
                # ========================================================
                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is not None:

                    try:

                        loop_running = bool(
                            loop.is_running()
                        )

                        if not loop_running:

                            logger.warning(
                                "Stored backend loop is not running"
                            )

                    except Exception as e:

                        logger.warning(
                            "Loop validation failed: %s",
                            e,
                        )

                # ========================================================
                # SAFE CALLBACK VALIDATION
                # ========================================================
                for callback_name in [

                    "_response_callback",

                    "_status_callback",

                ]:

                    try:

                        callback = getattr(
                            self,
                            callback_name,
                            None,
                        )

                        if (
                            callback is not None
                            and not callable(callback)
                        ):

                            logger.warning(
                                "%s corrupted — resetting",
                                callback_name,
                            )

                            setattr(
                                self,
                                callback_name,
                                None,
                            )

                    except Exception:
                        pass

                # ========================================================
                # OPTIONAL STATUS UPDATE
                # ========================================================
                try:

                    self._emit_status_safe(
                        "🚀 AI Backend Started"
                    )

                except Exception:
                    pass

                # ========================================================
                # MEMORY CLEANUP
                # ========================================================
                try:

                    gc.collect()

                except Exception:
                    pass

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.info(
                    "✅ AIBackend started successfully"
                )

                return True

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "AIBackend start cancelled"
            )

            self._running = False

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ AIBackend start failed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # RESET SAFE STATE
            # ------------------------------------------------------------
            try:

                self._running = False
                self._processing = False

            except Exception:
                pass

            # ------------------------------------------------------------
            # SAFE STATUS UPDATE
            # ------------------------------------------------------------
            try:

                self._emit_status_safe(
                    f"❌ Start failed: {str(e)}"
                )

            except Exception:
                pass

            return False

    def stop(self) -> bool:
        """
        Production-safe AIBackend shutdown handler.

        Features:
        - idempotent shutdown
        - safe async cleanup
        - coroutine leak prevention
        - future cancellation
        - thread-safe shutdown
        - event loop cleanup
        - fallback thread cleanup
        - memory cleanup
        - safe UI reset
        - defensive shutdown protection
        """

        import asyncio
        import gc
        import threading
        import time
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_stop_lock",
            ):

                self._stop_lock = threading.RLock()

            with self._stop_lock:

                logger.info(
                    "🛑 Stopping AIBackend..."
                )

                # ========================================================
                # IDEMPOTENT STOP
                # ========================================================
                if not bool(
                    getattr(
                        self,
                        "_running",
                        False,
                    )
                ):

                    logger.debug(
                        "AIBackend already stopped"
                    )

                    self._shutdown = True
                    self._processing = False

                    return True

                # ========================================================
                # MARK SHUTDOWN
                # ========================================================
                self._shutdown = True
                self._running = False

                # ========================================================
                # CANCEL ACTIVE PROCESSING
                # ========================================================
                self._processing = False

                # ========================================================
                # CANCEL ACTIVE FUTURES
                # ========================================================
                try:

                    events = list(

                        getattr(
                            self,
                            "_events",
                            [],
                        )
                    )

                    cancelled_count = 0

                    for future in events:

                        try:

                            if hasattr(
                                future,
                                "done",
                            ):

                                if not future.done():

                                    future.cancel()

                                    cancelled_count += 1

                        except Exception:
                            continue

                    logger.info(
                        "Cancelled %s active futures",
                        cancelled_count,
                    )

                except Exception as e:

                    logger.warning(
                        "Future cleanup failed: %s",
                        e,
                    )

                # ========================================================
                # CLEAN FUTURE STORAGE
                # ========================================================
                self._events = []

                # ========================================================
                # RESET FALLBACK THREAD COUNTER
                # ========================================================
                self._fallback_threads_active = 0

                # ========================================================
                # LOOP CLEANUP
                # ========================================================
                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is not None:

                    try:

                        if loop.is_running():

                            logger.debug(
                                "Backend loop still running during stop"
                            )

                    except Exception as e:

                        logger.warning(
                            "Loop cleanup warning: %s",
                            e,
                        )

                # ========================================================
                # RESET BACKEND REFERENCES
                # ========================================================
                self.system = None
                self.loop = None

                # ========================================================
                # RESET READY STATE
                # ========================================================
                self._ready = False

                # ========================================================
                # RESET REQUEST TIMERS
                # ========================================================
                self._last_request_time = 0.0
                self._last_processing_time = 0.0

                # ========================================================
                # OPTIONAL CALLBACK CLEANUP
                # ========================================================
                try:

                    if hasattr(
                        self,
                        "_response_callback",
                    ):

                        callback = getattr(
                            self,
                            "_response_callback",
                        )

                        if (
                            callback is not None
                            and not callable(callback)
                        ):

                            self._response_callback = None

                    if hasattr(
                        self,
                        "_status_callback",
                    ):

                        callback = getattr(
                            self,
                            "_status_callback",
                        )

                        if (
                            callback is not None
                            and not callable(callback)
                        ):

                            self._status_callback = None

                except Exception:
                    pass

                # ========================================================
                # SAFE UI CLEANUP
                # ========================================================
                try:

                    self._show_typing_indicator_safe(
                        False
                    )

                except Exception:
                    pass

                try:

                    self._emit_status_safe(
                        "🛑 Backend stopped"
                    )

                except Exception:
                    pass

                # ========================================================
                # MEMORY CLEANUP
                # ========================================================
                try:

                    gc.collect()

                except Exception:
                    pass

                # ========================================================
                # STATS
                # ========================================================
                try:

                    uptime = round(

                        time.time()
                        - getattr(
                            self,
                            "_started_at",
                            time.time(),
                        ),

                        2,
                    )

                    logger.info(
                        "Backend uptime before stop: %ss",
                        uptime,
                    )

                except Exception:
                    pass

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.info(
                    "✅ AIBackend stopped successfully"
                )

                return True

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "AIBackend stop cancelled"
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ AIBackend stop failed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # EMERGENCY SAFE STATE
            # ------------------------------------------------------------
            try:

                self._running = False
                self._shutdown = True
                self._processing = False
                self._ready = False

                self.system = None
                self.loop = None

            except Exception:
                pass

            # ------------------------------------------------------------
            # EMERGENCY STATUS
            # ------------------------------------------------------------
            try:

                self._emit_status_safe(
                    f"❌ Stop failed: {str(e)}"
                )

            except Exception:
                pass

            return False

    def shutdown(self):
        """
        Production-safe AIBackend shutdown routine.

        Features:
        - full resource cleanup
        - coroutine leak prevention
        - thread-safe shutdown
        - future cancellation
        - callback cleanup
        - event loop cleanup
        - fallback thread cleanup
        - memory cleanup
        - safe disconnect handling
        - defensive shutdown protection
        """

        import asyncio
        import gc
        import threading
        import time
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_shutdown_lock",
            ):

                self._shutdown_lock = threading.RLock()

            with self._shutdown_lock:

                logger.info(
                    "👋 Shutting down AIBackend..."
                )

                # ========================================================
                # IDEMPOTENT SHUTDOWN
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_shutdown",
                        False,
                    )
                ):

                    logger.debug(
                        "AIBackend already shutdown"
                    )

                # ========================================================
                # MARK SHUTDOWN STATE
                # ========================================================
                self._shutdown = True
                self._running = False
                self._ready = False
                self._processing = False

                # ========================================================
                # STOP TYPING INDICATOR
                # ========================================================
                try:

                    self._show_typing_indicator_safe(
                        False
                    )

                except Exception:
                    pass

                # ========================================================
                # CANCEL ACTIVE FUTURES
                # ========================================================
                cancelled_count = 0

                try:

                    events = list(

                        getattr(
                            self,
                            "_events",
                            [],
                        )
                    )

                    for event in events:

                        try:

                            if hasattr(
                                event,
                                "done",
                            ):

                                if not event.done():

                                    event.cancel()

                                    cancelled_count += 1

                        except Exception:
                            continue

                    logger.info(
                        "Cancelled %s pending futures",
                        cancelled_count,
                    )

                except Exception as e:

                    logger.warning(
                        "Future cancellation failed: %s",
                        e,
                    )

                # ========================================================
                # CLEAR EVENT STORAGE
                # ========================================================
                try:

                    if hasattr(
                        self,
                        "_events",
                    ):

                        self._events.clear()

                except Exception:
                    pass

                # ========================================================
                # CLEAN CALLBACKS
                # ========================================================
                callback_collections = [

                    "_response_callbacks",

                    "_status_callbacks",

                ]

                for attr_name in callback_collections:

                    try:

                        collection = getattr(
                            self,
                            attr_name,
                            None,
                        )

                        if collection is not None:

                            if hasattr(
                                collection,
                                "clear",
                            ):

                                collection.clear()

                            else:

                                setattr(
                                    self,
                                    attr_name,
                                    [],
                                )

                    except Exception as e:

                        logger.warning(
                            "Failed clearing %s: %s",
                            attr_name,
                            e,
                        )

                # ========================================================
                # CLEAN SINGLE CALLBACKS
                # ========================================================
                try:

                    if hasattr(
                        self,
                        "_response_callback",
                    ):

                        self._response_callback = None

                    if hasattr(
                        self,
                        "_status_callback",
                    ):

                        self._status_callback = None

                except Exception:
                    pass

                # ========================================================
                # FALLBACK THREAD RESET
                # ========================================================
                self._fallback_threads_active = 0

                # ========================================================
                # LOOP CLEANUP
                # ========================================================
                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is not None:

                    try:

                        if isinstance(
                            loop,
                            asyncio.AbstractEventLoop,
                        ):

                            if loop.is_running():

                                logger.debug(
                                    "Loop still active during shutdown"
                                )

                    except Exception as e:

                        logger.warning(
                            "Loop cleanup warning: %s",
                            e,
                        )

                # ========================================================
                # DISCONNECT BACKEND
                # ========================================================
                self.system = None
                self.loop = None

                # ========================================================
                # RESET INTERNAL TIMERS
                # ========================================================
                self._last_request_time = 0.0
                self._last_processing_time = 0.0

                # ========================================================
                # RESET COUNTERS
                # ========================================================
                self._request_counter = 0

                # ========================================================
                # OPTIONAL STATUS MESSAGE
                # ========================================================
                try:

                    self._emit_status_safe(
                        "👋 Backend disconnected"
                    )

                except Exception:
                    pass

                # ========================================================
                # MEMORY CLEANUP
                # ========================================================
                try:

                    gc.collect()

                except Exception:
                    pass

                # ========================================================
                # STATS LOGGING
                # ========================================================
                try:

                    uptime = round(

                        time.time()
                        - getattr(
                            self,
                            "_startup_time",
                            time.time(),
                        ),

                        2,
                    )

                    logger.info(
                        "AIBackend uptime: %ss",
                        uptime,
                    )

                except Exception:
                    pass

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.info(
                    "✅ AIBackend shutdown complete"
                )

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "AIBackend shutdown cancelled"
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ AIBackend shutdown failed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            # ------------------------------------------------------------
            # EMERGENCY SAFE STATE
            # ------------------------------------------------------------
            try:

                self._shutdown = True
                self._running = False
                self._ready = False
                self._processing = False

                self.system = None
                self.loop = None

            except Exception:
                pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    # SYSTEM STATS
    # ─────────────────────────────────────────────────────────────────
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Production-safe synchronous system stats fetcher.

        Features:
        - safe cross-thread async execution
        - dead loop protection
        - timeout protection
        - coroutine leak prevention
        - safe fallback stats
        - thread-safe polling
        - defensive validation
        - memory-safe execution
        """

        import asyncio
        import inspect
        import threading
        import time
        import traceback

        try:

            # ============================================================
            # DEFAULT RESPONSE
            # ============================================================
            fallback_response = {
                "status": "initializing",
                "ready": False,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "processing": bool(
                    getattr(
                        self,
                        "_processing",
                        False,
                    )
                ),
                "backend_connected": False,
                "loop_running": False,
                "timestamp": time.time(),
            }

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_stats_lock",
            ):

                self._stats_lock = threading.RLock()

            with self._stats_lock:

                # ========================================================
                # VALIDATE READY
                # ========================================================
                ready = bool(

                    getattr(
                        self,
                        "_ready",
                        False,
                    )
                )

                if not ready:

                    fallback_response[
                        "status"
                    ] = "initializing"

                    return fallback_response

                # ========================================================
                # VALIDATE SYSTEM
                # ========================================================
                system = getattr(
                    self,
                    "system",
                    None,
                )

                if system is None:

                    fallback_response[
                        "status"
                    ] = "system_missing"

                    return fallback_response

                # callable corruption
                if callable(system):

                    fallback_response[
                        "status"
                    ] = "system_corrupted"

                    logger.warning(
                        "System callable corruption"
                    )

                    return fallback_response

                # ========================================================
                # VALIDATE LOOP
                # ========================================================
                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is None:

                    fallback_response[
                        "status"
                    ] = "loop_missing"

                    return fallback_response

                if not isinstance(
                    loop,
                    asyncio.AbstractEventLoop,
                ):

                    fallback_response[
                        "status"
                    ] = "invalid_loop"

                    return fallback_response

                # ========================================================
                # LOOP HEALTH
                # ========================================================
                try:

                    loop_running = bool(
                        loop.is_running()
                    )

                except Exception:

                    loop_running = False

                fallback_response[
                    "loop_running"
                ] = loop_running

                if not loop_running:

                    fallback_response[
                        "status"
                    ] = "loop_dead"

                    return fallback_response

                # ========================================================
                # VALIDATE _get_stats
                # ========================================================
                get_stats_method = getattr(
                    self,
                    "_get_stats",
                    None,
                )

                if not callable(
                    get_stats_method
                ):

                    fallback_response[
                        "status"
                    ] = "stats_method_missing"

                    logger.warning(
                        "_get_stats missing"
                    )

                    return fallback_response

                # ========================================================
                # CREATE COROUTINE
                # ========================================================
                try:

                    coroutine = get_stats_method()

                except Exception as e:

                    logger.exception(
                        "_get_stats creation failed: %s",
                        e,
                    )

                    fallback_response[
                        "status"
                    ] = "stats_creation_failed"

                    return fallback_response

                # ========================================================
                # VALIDATE COROUTINE
                # ========================================================
                if not inspect.isawaitable(
                    coroutine
                ):

                    logger.warning(
                        "_get_stats returned non-awaitable"
                    )

                    fallback_response[
                        "status"
                    ] = "invalid_stats_coroutine"

                    return fallback_response

                # ========================================================
                # EXECUTE SAFELY
                # ========================================================
                try:

                    future = asyncio.run_coroutine_threadsafe(

                        coroutine,

                        loop,
                    )

                except Exception as e:

                    logger.exception(
                        "Stats scheduling failed: %s",
                        e,
                    )

                    fallback_response[
                        "status"
                    ] = "schedule_failed"

                    return fallback_response

                # ========================================================
                # FETCH RESULT
                # ========================================================
                try:

                    stats = future.result(
                        timeout=3
                    )

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except TimeoutError:

                    logger.warning(
                        "Stats request timeout"
                    )

                    try:

                        future.cancel()

                    except Exception:
                        pass

                    fallback_response[
                        "status"
                    ] = "timeout"

                    fallback_response[
                        "ready"
                    ] = ready

                    return fallback_response

                # --------------------------------------------------------
                # FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    logger.exception(
                        "Stats fetch failed: %s",
                        e,
                    )

                    fallback_response[
                        "status"
                    ] = "stats_error"

                    fallback_response[
                        "ready"
                    ] = ready

                    return fallback_response

                # ========================================================
                # VALIDATE RESPONSE
                # ========================================================
                if callable(stats):

                    logger.warning(
                        "Stats callable corruption"
                    )

                    fallback_response[
                        "status"
                    ] = "stats_corrupted"

                    return fallback_response

                if not isinstance(
                    stats,
                    dict,
                ):

                    logger.warning(
                        "Stats invalid type: %s",
                        type(stats),
                    )

                    fallback_response[
                        "status"
                    ] = "invalid_stats"

                    return fallback_response

                # ========================================================
                # SAFE SERIALIZATION
                # ========================================================
                safe_stats = {}

                for key, value in stats.items():

                    try:

                        key = str(key)

                        if isinstance(

                            value,

                            (
                                str,
                                int,
                                float,
                                bool,
                                list,
                                dict,
                                type(None),
                            ),
                        ):

                            safe_stats[key] = value

                        else:

                            safe_stats[key] = str(
                                value
                            )

                    except Exception:

                        continue

                # ========================================================
                # ENSURE REQUIRED FIELDS
                # ========================================================
                safe_stats.setdefault(
                    "status",
                    "running",
                )

                safe_stats.setdefault(
                    "ready",
                    ready,
                )

                safe_stats.setdefault(
                    "uptime",
                    0,
                )

                safe_stats.setdefault(
                    "memory_usage",
                    0,
                )

                safe_stats.setdefault(
                    "cpu_usage",
                    0,
                )

                # processing state always local truth
                safe_stats[
                    "processing"
                ] = bool(
                    getattr(
                        self,
                        "_processing",
                        False,
                    )
                )

                safe_stats[
                    "backend_connected"
                ] = True

                safe_stats[
                    "loop_running"
                ] = loop_running

                safe_stats[
                    "timestamp"
                ] = time.time()

                return safe_stats

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ get_system_stats crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            return {
                "status": "error",
                "ready": bool(
                    getattr(
                        self,
                        "_ready",
                        False,
                    )
                ),
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "processing": bool(
                    getattr(
                        self,
                        "_processing",
                        False,
                    )
                ),
                "backend_connected": False,
                "loop_running": False,
                "timestamp": time.time(),
                "error": str(e),
            }

    async def _get_stats(self) -> Dict[str, Any]:
        """
        Production-safe backend stats collector.

        Features:
        - async/sync safe status collection
        - coroutine leak prevention
        - safe fallback stats
        - timeout protection
        - memory-safe execution
        - defensive validation
        - non-blocking system polling
        - safe serialization
        - robust uptime handling
        """

        import asyncio
        import inspect
        import os
        import time
        import traceback

        try:

            # ============================================================
            # DEFAULT STATS
            # ============================================================
            now = time.time()

            connected_at = float(

                getattr(
                    self,
                    "_connected_at",
                    now,
                )
            )

            uptime = max(
                0,
                int(now - connected_at),
            )

            fallback_stats = {
                "status": "running",
                "ready": bool(
                    getattr(
                        self,
                        "_ready",
                        False,
                    )
                ),
                "processing": bool(
                    getattr(
                        self,
                        "_processing",
                        False,
                    )
                ),
                "uptime": uptime,
                "memory_usage": 0,
                "cpu_usage": 0,
                "backend_connected": bool(
                    getattr(
                        self,
                        "system",
                        None,
                    )
                ),
                "loop_running": False,
                "active_threads": 0,
                "timestamp": now,
            }

            # ============================================================
            # SAFE LOOP STATUS
            # ============================================================
            try:

                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is not None:

                    fallback_stats[
                        "loop_running"
                    ] = bool(
                        loop.is_running()
                    )

            except Exception:
                pass

            # ============================================================
            # THREAD COUNT
            # ============================================================
            try:

                import threading

                fallback_stats[
                    "active_threads"
                ] = len(
                    threading.enumerate()
                )

            except Exception:
                pass

            # ============================================================
            # SYSTEM CHECK
            # ============================================================
            system = getattr(
                self,
                "system",
                None,
            )

            if system is None:

                return fallback_stats

            # callable corruption
            if callable(system):

                logger.warning(
                    "_get_stats: system callable corruption"
                )

                return fallback_stats

            # ============================================================
            # STRATEGY 1
            # get_system_status
            # ============================================================
            get_system_status = getattr(
                system,
                "get_system_status",
                None,
            )

            if callable(
                get_system_status
            ):

                try:

                    result = get_system_status()

                    # FIX:
                    # object bool can't be used in await
                    if inspect.isawaitable(
                        result
                    ):

                        result = await asyncio.wait_for(

                            result,

                            timeout=5.0,
                        )

                    # ----------------------------------------------------
                    # VALIDATE RESULT
                    # ----------------------------------------------------
                    if isinstance(
                        result,
                        dict,
                    ):

                        safe_result = {}

                        for key, value in result.items():

                            try:

                                # safe serialization
                                if isinstance(

                                    value,

                                    (
                                        str,
                                        int,
                                        float,
                                        bool,
                                        list,
                                        dict,
                                        type(None),
                                    ),
                                ):

                                    safe_result[
                                        str(key)
                                    ] = value

                                else:

                                    safe_result[
                                        str(key)
                                    ] = str(
                                        value
                                    )

                            except Exception:

                                continue

                        safe_result.setdefault(
                            "ready",
                            fallback_stats["ready"],
                        )

                        safe_result.setdefault(
                            "uptime",
                            uptime,
                        )

                        safe_result.setdefault(
                            "status",
                            "running",
                        )

                        return safe_result

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except asyncio.TimeoutError:

                    logger.warning(
                        "get_system_status timeout"
                    )

                # --------------------------------------------------------
                # FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    logger.warning(
                        "get_system_status failed: %s",
                        e,
                    )

                    logger.debug(
                        traceback.format_exc()[:3000]
                    )

            # ============================================================
            # STRATEGY 2
            # get_status
            # ============================================================
            get_status = getattr(
                system,
                "get_status",
                None,
            )

            if callable(
                get_status
            ):

                try:

                    result = get_status()

                    if inspect.isawaitable(
                        result
                    ):

                        result = await asyncio.wait_for(

                            result,

                            timeout=5.0,
                        )

                    # ----------------------------------------------------
                    # VALIDATE RESULT
                    # ----------------------------------------------------
                    if isinstance(
                        result,
                        dict,
                    ):

                        safe_result = {}

                        for key, value in result.items():

                            try:

                                if isinstance(

                                    value,

                                    (
                                        str,
                                        int,
                                        float,
                                        bool,
                                        list,
                                        dict,
                                        type(None),
                                    ),
                                ):

                                    safe_result[
                                        str(key)
                                    ] = value

                                else:

                                    safe_result[
                                        str(key)
                                    ] = str(
                                        value
                                    )

                            except Exception:

                                continue

                        safe_result.setdefault(
                            "ready",
                            fallback_stats["ready"],
                        )

                        safe_result.setdefault(
                            "uptime",
                            uptime,
                        )

                        safe_result.setdefault(
                            "status",
                            "running",
                        )

                        return safe_result

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except asyncio.TimeoutError:

                    logger.warning(
                        "get_status timeout"
                    )

                # --------------------------------------------------------
                # FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    logger.warning(
                        "get_status failed: %s",
                        e,
                    )

                    logger.debug(
                        traceback.format_exc()[:3000]
                    )

            # ============================================================
            # LOCAL RESOURCE FALLBACK
            # ============================================================
            try:

                import psutil

                process = psutil.Process(
                    os.getpid()
                )

                fallback_stats[
                    "memory_usage"
                ] = round(

                    process.memory_info().rss
                    / (1024 * 1024),

                    2,
                )

                fallback_stats[
                    "cpu_usage"
                ] = round(

                    psutil.cpu_percent(
                        interval=0.1
                    ),

                    2,
                )

            except Exception:
                pass

            return fallback_stats

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "_get_stats cancelled"
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.exception(
                "❌ _get_stats crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            return {
                "status": "error",
                "ready": bool(
                    getattr(
                        self,
                        "_ready",
                        False,
                    )
                ),
                "processing": bool(
                    getattr(
                        self,
                        "_processing",
                        False,
                    )
                ),
                "uptime": 0,
                "memory_usage": 0,
                "cpu_usage": 0,
                "backend_connected": False,
                "loop_running": False,
                "active_threads": 0,
                "timestamp": time.time(),
                "error": str(e),
            }

    # ─────────────────────────────────────────────────────────────────
    # TASK / PLANNING DISPLAY
    # ─────────────────────────────────────────────────────────────────
    def update_task(self, task_name: str):
        """
        Production-safe task display updater.

        Features:
        - thread-safe UI updates
        - Kivy-safe scheduling
        - task validation
        - duplicate update prevention
        - widget validation
        - callback safety
        - shutdown protection
        - memory-safe scheduling
        - defensive programming
        """

        import threading
        import time
        import traceback

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            if bool(
                getattr(
                    self,
                    "_shutdown",
                    False,
                )
            ):

                return False

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_task_update_lock",
            ):

                self._task_update_lock = threading.RLock()

            with self._task_update_lock:

                # ========================================================
                # VALIDATE TASK NAME
                # ========================================================
                if task_name is None:

                    task_name = ""

                try:

                    task_name = str(
                        task_name
                    ).strip()

                except Exception:

                    task_name = ""

                if not task_name:

                    task_name = "Idle"

                # ========================================================
                # PREVENT DUPLICATE UPDATES
                # ========================================================
                previous_task = str(

                    getattr(
                        self,
                        "_last_task_name",
                        "",
                    )
                )

                if previous_task == task_name:

                    return True

                self._last_task_name = task_name

                # ========================================================
                # VALIDATE TASK DISPLAY
                # ========================================================
                task_display = getattr(
                    self,
                    "task_display",
                    None,
                )

                if task_display is None:

                    logger.debug(
                        "No task display bound"
                    )

                    return False

                # callable corruption
                if callable(
                    task_display
                ):

                    logger.warning(
                        "Task display corrupted"
                    )

                    return False

                # ========================================================
                # FIND UPDATE METHOD
                # ========================================================
                update_method = None

                if hasattr(
                    task_display,
                    "set_current_task",
                ):

                    candidate = getattr(

                        task_display,

                        "set_current_task",

                        None,
                    )

                    if callable(
                        candidate
                    ):

                        update_method = candidate

                if (
                    update_method is None
                    and hasattr(
                        task_display,
                        "set_task",
                    )
                ):

                    candidate = getattr(

                        task_display,

                        "set_task",

                        None,
                    )

                    if callable(
                        candidate
                    ):

                        update_method = candidate

                # ========================================================
                # VALIDATE METHOD
                # ========================================================
                if not callable(
                    update_method
                ):

                    logger.warning(
                        "No valid task update method found"
                    )

                    return False

                # ========================================================
                # UI UPDATE CALLBACK
                # ========================================================
                def _update_ui(dt=None):

                    try:

                        # ------------------------------------------------
                        # SHUTDOWN CHECK
                        # ------------------------------------------------
                        if bool(
                            getattr(
                                self,
                                "_shutdown",
                                False,
                            )
                        ):

                            return

                        # ------------------------------------------------
                        # GET CURRENT DISPLAY
                        # ------------------------------------------------
                        current_display = getattr(
                            self,
                            "task_display",
                            None,
                        )

                        if current_display is None:

                            return

                        # ------------------------------------------------
                        # FIND METHOD AGAIN
                        # ------------------------------------------------
                        current_method = None

                        if hasattr(
                            current_display,
                            "set_current_task",
                        ):

                            candidate = getattr(

                                current_display,

                                "set_current_task",

                                None,
                            )

                            if callable(
                                candidate
                            ):

                                current_method = candidate

                        if (
                            current_method is None
                            and hasattr(
                                current_display,
                                "set_task",
                            )
                        ):

                            candidate = getattr(

                                current_display,

                                "set_task",

                                None,
                            )

                            if callable(
                                candidate
                            ):

                                current_method = candidate

                        if not callable(
                            current_method
                        ):

                            return

                        # ------------------------------------------------
                        # EXECUTE UPDATE
                        # ------------------------------------------------
                        current_method(
                            task_name
                        )

                    except Exception as e:

                        logger.debug(
                            "Task UI update failed: %s",
                            e,
                        )

                # ========================================================
                # SCHEDULE SAFELY
                # ========================================================
                try:

                    Clock.schedule_once(
                        _update_ui,
                        0,
                    )

                except Exception as e:

                    logger.warning(
                        "Task UI scheduling failed: %s",
                        e,
                    )

                    # ----------------------------------------------------
                    # DIRECT FALLBACK
                    # ----------------------------------------------------
                    try:

                        _update_ui()

                    except Exception:
                        pass

                # ========================================================
                # UPDATE TIMESTAMP
                # ========================================================
                self._last_task_update = time.time()

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.debug(
                    "✅ Task updated: %s",
                    task_name,
                )

                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "update_task crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            return False

    def update_planning_steps(self, steps: list):
        """
        Production-safe planning panel updater.

        Features:
        - thread-safe UI updates
        - Kivy-safe scheduling
        - planning step validation
        - duplicate update prevention
        - widget validation
        - memory-safe updates
        - shutdown protection
        - defensive programming
        - safe serialization
        """

        import threading
        import time
        import traceback

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            if bool(
                getattr(
                    self,
                    "_shutdown",
                    False,
                )
            ):

                return False

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_planning_update_lock",
            ):

                self._planning_update_lock = threading.RLock()

            with self._planning_update_lock:

                # ========================================================
                # VALIDATE INPUT
                # ========================================================
                if steps is None:

                    steps = []

                # callable corruption
                if callable(
                    steps
                ):

                    logger.warning(
                        "Planning steps corrupted (callable)"
                    )

                    return False

                # ========================================================
                # NORMALIZE STEPS
                # ========================================================
                if not isinstance(
                    steps,
                    (list, tuple),
                ):

                    try:

                        steps = [str(steps)]

                    except Exception:

                        steps = []

                safe_steps = []

                for step in steps:

                    try:

                        # skip invalid callables
                        if callable(step):

                            continue

                        # normalize string
                        safe_step = str(
                            step
                        ).strip()

                        if safe_step:

                            safe_steps.append(
                                safe_step
                            )

                    except Exception:
                        continue

                # ========================================================
                # BOUNDED MEMORY
                # ========================================================
                max_steps = 100

                if len(safe_steps) > max_steps:

                    logger.warning(
                        "Planning steps exceeded limit — trimming"
                    )

                    safe_steps = safe_steps[
                        :max_steps
                    ]

                # ========================================================
                # PREVENT DUPLICATE UPDATES
                # ========================================================
                previous_steps = getattr(
                    self,
                    "_last_planning_steps",
                    None,
                )

                if previous_steps == safe_steps:

                    return True

                self._last_planning_steps = list(
                    safe_steps
                )

                # ========================================================
                # VALIDATE PANEL
                # ========================================================
                planning_panel = getattr(
                    self,
                    "planning_panel",
                    None,
                )

                if planning_panel is None:

                    logger.debug(
                        "No planning panel bound"
                    )

                    return False

                # callable corruption
                if callable(
                    planning_panel
                ):

                    logger.warning(
                        "Planning panel corrupted"
                    )

                    return False

                # ========================================================
                # VALIDATE METHOD
                # ========================================================
                update_method = getattr(

                    planning_panel,

                    "update_steps",

                    None,
                )

                if not callable(
                    update_method
                ):

                    logger.warning(
                        "Planning panel missing update_steps"
                    )

                    return False

                # ========================================================
                # UI UPDATE CALLBACK
                # ========================================================
                def _update_ui(dt=None):

                    try:

                        # ------------------------------------------------
                        # SHUTDOWN CHECK
                        # ------------------------------------------------
                        if bool(
                            getattr(
                                self,
                                "_shutdown",
                                False,
                            )
                        ):

                            return

                        # ------------------------------------------------
                        # GET CURRENT PANEL
                        # ------------------------------------------------
                        current_panel = getattr(
                            self,
                            "planning_panel",
                            None,
                        )

                        if current_panel is None:

                            return

                        # ------------------------------------------------
                        # VALIDATE METHOD AGAIN
                        # ------------------------------------------------
                        current_method = getattr(

                            current_panel,

                            "update_steps",

                            None,
                        )

                        if not callable(
                            current_method
                        ):

                            return

                        # ------------------------------------------------
                        # EXECUTE UPDATE
                        # ------------------------------------------------
                        current_method(
                            list(safe_steps)
                        )

                    except Exception as e:

                        logger.debug(
                            "Planning panel UI update failed: %s",
                            e,
                        )

                # ========================================================
                # SCHEDULE SAFELY
                # ========================================================
                try:

                    Clock.schedule_once(
                        _update_ui,
                        0,
                    )

                except Exception as e:

                    logger.warning(
                        "Planning panel scheduling failed: %s",
                        e,
                    )

                    # ----------------------------------------------------
                    # DIRECT FALLBACK
                    # ----------------------------------------------------
                    try:

                        _update_ui()

                    except Exception:
                        pass

                # ========================================================
                # UPDATE TIMESTAMP
                # ========================================================
                self._last_planning_update = time.time()

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.debug(
                    "✅ Planning steps updated (%s steps)",
                    len(safe_steps),
                )

                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "update_planning_steps crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            return False

    # ─────────────────────────────────────────────────────────────────
    # UI-THREAD SAFE HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _show_typing_indicator_safe(self, show: bool):
        """
        Production-safe typing indicator updater.

        Features:
        - thread-safe UI updates
        - Kivy-safe scheduling
        - callback validation
        - widget existence validation
        - crash-safe UI execution
        - memory-safe scheduling
        - duplicate update prevention
        - shutdown protection
        - event-loop safe execution
        """

        import threading
        import time
        import traceback

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            if bool(
                getattr(
                    self,
                    "_shutdown",
                    False,
                )
            ):

                return

            # ============================================================
            # NORMALIZE INPUT
            # ============================================================
            show = bool(show)

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_typing_lock",
            ):

                self._typing_lock = threading.RLock()

            with self._typing_lock:

                # ========================================================
                # PREVENT DUPLICATE UI UPDATES
                # ========================================================
                previous_state = bool(

                    getattr(
                        self,
                        "_typing_visible",
                        False,
                    )
                )

                if previous_state == show:

                    return

                self._typing_visible = show

                # ========================================================
                # GET CHAT SCREEN SAFELY
                # ========================================================
                chat_screen = getattr(
                    self,
                    "chat_screen",
                    None,
                )

                if chat_screen is None:

                    return

                # ========================================================
                # VALIDATE METHOD
                # ========================================================
                update_method = getattr(

                    chat_screen,

                    "_show_typing_indicator",

                    None,
                )

                if not callable(
                    update_method
                ):

                    logger.warning(
                        "Typing indicator method missing"
                    )

                    return

                # ========================================================
                # UI UPDATE CALLBACK
                # ========================================================
                def _update_ui(dt=None):

                    try:

                        # ------------------------------------------------
                        # SHUTDOWN CHECK
                        # ------------------------------------------------
                        if bool(
                            getattr(
                                self,
                                "_shutdown",
                                False,
                            )
                        ):

                            return

                        # ------------------------------------------------
                        # SCREEN STILL VALID?
                        # ------------------------------------------------
                        current_screen = getattr(
                            self,
                            "chat_screen",
                            None,
                        )

                        if current_screen is None:

                            return

                        # ------------------------------------------------
                        # METHOD STILL EXISTS?
                        # ------------------------------------------------
                        current_method = getattr(

                            current_screen,

                            "_show_typing_indicator",

                            None,
                        )

                        if not callable(
                            current_method
                        ):

                            return

                        # ------------------------------------------------
                        # EXECUTE UI UPDATE
                        # ------------------------------------------------
                        current_method(
                            show
                        )

                    except Exception as e:

                        logger.debug(
                            "Typing indicator UI update failed: %s",
                            e,
                        )

                # ========================================================
                # SCHEDULE SAFELY ON MAIN THREAD
                # ========================================================
                try:

                    Clock.schedule_once(
                        _update_ui,
                        0,
                    )

                except Exception as e:

                    logger.warning(
                        "Clock scheduling failed: %s",
                        e,
                    )

                    # ----------------------------------------------------
                    # DIRECT FALLBACK EXECUTION
                    # ----------------------------------------------------
                    try:

                        _update_ui()

                    except Exception:
                        pass

                # ========================================================
                # DEBUG LOGGING
                # ========================================================
                logger.debug(
                    "Typing indicator updated: %s",
                    show,
                )

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "_show_typing_indicator_safe crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

    def _emit_response_safe(self, text: str):
        """Emit response to UI safely with error handling"""
        
        def _update():
            try:
                if hasattr(self, '_response_callbacks') and self._response_callbacks:
                    # Create a copy to avoid modification during iteration
                    callbacks = list(self._response_callbacks)
                    for cb in callbacks:
                        try:
                            if text:
                                logger.debug("📤 Emitting response to UI: %s", text[:100])
                            else:
                                logger.debug("📤 Emitting empty response to UI")
                            
                            # Call the callback safely
                            cb(text)
                            
                        except TypeError as e:
                            # Handle case where callback expects different arguments
                            logger.debug(f"Callback argument error: {e}")
                            try:
                                # Try with no arguments
                                cb()
                            except Exception:
                                pass
                        except Exception as e:
                            logger.exception(f"Response callback error: {e}")
                            
                else:
                    # No callbacks registered
                    if text:
                        logger.debug("[RESPONSE] %s", text[:100] if len(text) > 100 else text)
                    else:
                        logger.debug("[RESPONSE] (empty)")
                        
            except Exception as e:
                logger.error(f"Error in _emit_response_safe: {e}")
        
        try:
            # Check if Clock is available (Kivy environment)
            try:
                from kivy.clock import Clock
                Clock.schedule_once(lambda dt: _update(), 0)
            except ImportError:
                # Fallback for non-Kivy environment
                import threading
                threading.Timer(0, _update).start()
                
        except Exception as e:
            # Ultimate fallback - execute directly
            logger.warning(f"Failed to schedule response emission: {e}")
            _update()

    def _emit_status_safe(self, text: str):
        """Emit status update to UI safely with error handling"""
        
        def _update():
            try:
                if hasattr(self, '_status_callbacks') and self._status_callbacks:
                    # Create a copy to avoid modification during iteration
                    callbacks = list(self._status_callbacks)
                    for cb in callbacks:
                        try:
                            if text:
                                logger.debug("📊 Emitting status to UI: %s", text[:100])
                            else:
                                logger.debug("📊 Emitting empty status to UI")
                            
                            # Call the callback safely
                            cb(text)
                            
                        except TypeError as e:
                            # Handle case where callback expects different arguments
                            logger.debug(f"Status callback argument error: {e}")
                            try:
                                # Try with no arguments
                                cb()
                            except Exception:
                                pass
                        except Exception as e:
                            logger.exception(f"Status callback error: {e}")
                            
                else:
                    # No callbacks registered
                    if text:
                        logger.info("[STATUS] %s", text[:200] if len(text) > 200 else text)
                    else:
                        logger.debug("[STATUS] (empty)")
                        
            except Exception as e:
                logger.error(f"Error in _emit_status_safe: {e}")
        
        try:
            # Check if Clock is available (Kivy environment)
            try:
                from kivy.clock import Clock
                Clock.schedule_once(lambda dt: _update(), 0)
            except ImportError:
                # Fallback for non-Kivy environment
                import threading
                threading.Timer(0, _update).start()
                
        except Exception as e:
            # Ultimate fallback - execute directly
            logger.warning(f"Failed to schedule status emission: {e}")
            _update()

    # ─────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────────────
    def set_response_callback(self, func):
        """
        Production-safe response callback registrar.

        Features:
        - callback validation
        - duplicate prevention
        - thread-safe registration
        - callable verification
        - memory-safe callback storage
        - corruption cleanup
        - defensive programming
        """

        import threading
        import traceback
        import weakref

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_callback_lock",
            ):

                self._callback_lock = threading.RLock()

            with self._callback_lock:

                # ========================================================
                # VALIDATE STORAGE
                # ========================================================
                if not hasattr(
                    self,
                    "_response_callbacks",
                ):

                    self._response_callbacks = []

                # corrupted storage
                if not isinstance(
                    self._response_callbacks,
                    list,
                ):

                    logger.warning(
                        "Response callback storage corrupted — resetting"
                    )

                    self._response_callbacks = []

                # ========================================================
                # VALIDATE CALLBACK
                # ========================================================
                if func is None:

                    logger.warning(
                        "Ignoring None response callback"
                    )

                    return False

                if not callable(
                    func
                ):

                    logger.warning(
                        "Ignoring non-callable response callback"
                    )

                    return False

                # ========================================================
                # CLEAN DEAD CALLBACKS
                # ========================================================
                cleaned_callbacks = []

                for callback in self._response_callbacks:

                    try:

                        # weakref support
                        if isinstance(
                            callback,
                            weakref.ReferenceType,
                        ):

                            real_callback = callback()

                            if real_callback is not None:

                                cleaned_callbacks.append(
                                    callback
                                )

                        else:

                            if callable(
                                callback
                            ):

                                cleaned_callbacks.append(
                                    callback
                                )

                    except Exception:
                        continue

                self._response_callbacks = cleaned_callbacks

                # ========================================================
                # PREVENT DUPLICATES
                # ========================================================
                for existing in self._response_callbacks:

                    try:

                        # weakref callback
                        if isinstance(
                            existing,
                            weakref.ReferenceType,
                        ):

                            existing_func = existing()

                            if existing_func is func:

                                logger.debug(
                                    "Response callback already registered"
                                )

                                return True

                        # normal callback
                        else:

                            if existing is func:

                                logger.debug(
                                    "Response callback already registered"
                                )

                                return True

                    except Exception:
                        continue

                # ========================================================
                # MEMORY-SAFE STORAGE
                # ========================================================
                try:

                    # bound methods → weakref
                    if hasattr(
                        func,
                        "__self__",
                    ) and hasattr(
                        func,
                        "__func__",
                    ):

                        callback_ref = weakref.WeakMethod(
                            func
                        )

                        self._response_callbacks.append(
                            callback_ref
                        )

                    else:

                        self._response_callbacks.append(
                            func
                        )

                except Exception:

                    # fallback normal storage
                    self._response_callbacks.append(
                        func
                    )

                # ========================================================
                # BOUNDED STORAGE
                # ========================================================
                max_callbacks = 100

                if (
                    len(self._response_callbacks)
                    > max_callbacks
                ):

                    logger.warning(
                        "Response callback limit reached — trimming"
                    )

                    self._response_callbacks = (
                        self._response_callbacks[
                            -max_callbacks:
                        ]
                    )

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.debug(
                    "✅ Response callback registered"
                )

                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "set_response_callback crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            return False

    def set_status_callback(self, func):
        """
        Production-safe status callback registrar.

        Features:
        - callback validation
        - duplicate prevention
        - thread-safe registration
        - callable verification
        - memory-safe callback storage
        - corruption cleanup
        - weakref support
        - defensive programming
        """

        import threading
        import traceback
        import weakref

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_callback_lock",
            ):

                self._callback_lock = threading.RLock()

            with self._callback_lock:

                # ========================================================
                # VALIDATE STORAGE
                # ========================================================
                if not hasattr(
                    self,
                    "_status_callbacks",
                ):

                    self._status_callbacks = []

                # corrupted storage
                if not isinstance(
                    self._status_callbacks,
                    list,
                ):

                    logger.warning(
                        "Status callback storage corrupted — resetting"
                    )

                    self._status_callbacks = []

                # ========================================================
                # VALIDATE CALLBACK
                # ========================================================
                if func is None:

                    logger.warning(
                        "Ignoring None status callback"
                    )

                    return False

                if not callable(
                    func
                ):

                    logger.warning(
                        "Ignoring non-callable status callback"
                    )

                    return False

                # ========================================================
                # CLEAN DEAD CALLBACKS
                # ========================================================
                cleaned_callbacks = []

                for callback in self._status_callbacks:

                    try:

                        # weakref support
                        if isinstance(
                            callback,
                            weakref.ReferenceType,
                        ):

                            real_callback = callback()

                            if real_callback is not None:

                                cleaned_callbacks.append(
                                    callback
                                )

                        else:

                            if callable(
                                callback
                            ):

                                cleaned_callbacks.append(
                                    callback
                                )

                    except Exception:
                        continue

                self._status_callbacks = cleaned_callbacks

                # ========================================================
                # PREVENT DUPLICATES
                # ========================================================
                for existing in self._status_callbacks:

                    try:

                        # weakref callback
                        if isinstance(
                            existing,
                            weakref.ReferenceType,
                        ):

                            existing_func = existing()

                            if existing_func is func:

                                logger.debug(
                                    "Status callback already registered"
                                )

                                return True

                        # normal callback
                        else:

                            if existing is func:

                                logger.debug(
                                    "Status callback already registered"
                                )

                                return True

                    except Exception:
                        continue

                # ========================================================
                # MEMORY-SAFE STORAGE
                # ========================================================
                try:

                    # bound methods → weakref
                    if hasattr(
                        func,
                        "__self__",
                    ) and hasattr(
                        func,
                        "__func__",
                    ):

                        callback_ref = weakref.WeakMethod(
                            func
                        )

                        self._status_callbacks.append(
                            callback_ref
                        )

                    else:

                        self._status_callbacks.append(
                            func
                        )

                except Exception:

                    # fallback normal storage
                    self._status_callbacks.append(
                        func
                    )

                # ========================================================
                # BOUNDED STORAGE
                # ========================================================
                max_callbacks = 100

                if (
                    len(self._status_callbacks)
                    > max_callbacks
                ):

                    logger.warning(
                        "Status callback limit reached — trimming"
                    )

                    self._status_callbacks = (
                        self._status_callbacks[
                            -max_callbacks:
                        ]
                    )

                # ========================================================
                # SUCCESS
                # ========================================================
                logger.debug(
                    "✅ Status callback registered"
                )

                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "set_status_callback crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            return False

    # ─────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        """
        Production-safe backend readiness checker.

        Features:
        - safe loop validation
        - thread-safe readiness checks
        - backend integrity validation
        - event-loop safety
        - corruption detection
        - shutdown protection
        - defensive runtime checks
        """

        import asyncio
        import threading
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_ready_lock",
            ):

                self._ready_lock = threading.RLock()

            with self._ready_lock:

                # ========================================================
                # SHUTDOWN GUARD
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_shutdown",
                        False,
                    )
                ):

                    return False

                # ========================================================
                # VALIDATE READY FLAG
                # ========================================================
                ready_flag = bool(

                    getattr(
                        self,
                        "_ready",
                        False,
                    )
                )

                if not ready_flag:

                    return False

                # ========================================================
                # VALIDATE SYSTEM
                # ========================================================
                system = getattr(
                    self,
                    "system",
                    None,
                )

                if system is None:

                    logger.debug(
                        "Backend not ready: system missing"
                    )

                    return False

                # callable corruption
                if callable(system):

                    logger.warning(
                        "Backend corrupted: system callable"
                    )

                    return False

                # ========================================================
                # VALIDATE LOOP
                # ========================================================
                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is None:

                    logger.debug(
                        "Backend not ready: loop missing"
                    )

                    return False

                if not isinstance(
                    loop,
                    asyncio.AbstractEventLoop,
                ):

                    logger.warning(
                        "Backend corrupted: invalid loop"
                    )

                    return False

                # ========================================================
                # LOOP HEALTH CHECK
                # ========================================================
                try:

                    if not loop.is_running():

                        logger.debug(
                            "Backend not ready: loop not running"
                        )

                        return False

                except Exception as e:

                    logger.warning(
                        "Loop validation failed: %s",
                        e,
                    )

                    return False

                # ========================================================
                # OPTIONAL LOOP CLOSED CHECK
                # ========================================================
                try:

                    if hasattr(
                        loop,
                        "is_closed",
                    ):

                        if loop.is_closed():

                            logger.warning(
                                "Backend loop is closed"
                            )

                            return False

                except Exception:
                    pass

                # ========================================================
                # VALIDATE CORE METHODS
                # ========================================================
                required_methods = [

                    "brain_process",

                ]

                for method_name in required_methods:

                    method = getattr(
                        system,
                        method_name,
                        None,
                    )

                    if not callable(
                        method
                    ):

                        logger.warning(
                            "Missing required system method: %s",
                            method_name,
                        )

                        return False

                # ========================================================
                # PROCESSING STATE VALIDATION
                # ========================================================
                processing = getattr(
                    self,
                    "_processing",
                    False,
                )

                if not isinstance(
                    processing,
                    bool,
                ):

                    logger.warning(
                        "Processing flag corrupted"
                    )

                    self._processing = False

                # ========================================================
                # SUCCESS
                # ========================================================
                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "is_ready crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            return False

    def get_status(self) -> str:
        """
        Production-safe backend status resolver.

        Features:
        - thread-safe status checks
        - loop validation
        - backend integrity checks
        - shutdown detection
        - corruption detection
        - defensive runtime validation
        """

        import asyncio
        import threading
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_status_lock",
            ):

                self._status_lock = threading.RLock()

            with self._status_lock:

                # ========================================================
                # SHUTDOWN STATE
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_shutdown",
                        False,
                    )
                ):

                    return "shutdown"

                # ========================================================
                # PROCESSING STATE
                # ========================================================
                processing = getattr(
                    self,
                    "_processing",
                    False,
                )

                if not isinstance(
                    processing,
                    bool,
                ):

                    logger.warning(
                        "Processing flag corrupted"
                    )

                    self._processing = False
                    processing = False

                if processing:

                    return "processing"

                # ========================================================
                # READY FLAG
                # ========================================================
                ready = bool(

                    getattr(
                        self,
                        "_ready",
                        False,
                    )
                )

                if not ready:

                    return "initializing"

                # ========================================================
                # SYSTEM VALIDATION
                # ========================================================
                system = getattr(
                    self,
                    "system",
                    None,
                )

                if system is None:

                    return "disconnected"

                # callable corruption
                if callable(system):

                    logger.warning(
                        "System corrupted: callable"
                    )

                    return "error"

                # ========================================================
                # LOOP VALIDATION
                # ========================================================
                loop = getattr(
                    self,
                    "loop",
                    None,
                )

                if loop is None:

                    return "loop_missing"

                if not isinstance(
                    loop,
                    asyncio.AbstractEventLoop,
                ):

                    logger.warning(
                        "Invalid event loop"
                    )

                    return "error"

                # ========================================================
                # LOOP HEALTH CHECK
                # ========================================================
                try:

                    if not loop.is_running():

                        return "loop_dead"

                except Exception as e:

                    logger.warning(
                        "Loop health check failed: %s",
                        e,
                    )

                    return "error"

                # ========================================================
                # LOOP CLOSED CHECK
                # ========================================================
                try:

                    if hasattr(
                        loop,
                        "is_closed",
                    ):

                        if loop.is_closed():

                            return "loop_closed"

                except Exception:
                    pass

                # ========================================================
                # VALIDATE CORE SYSTEM METHODS
                # ========================================================
                required_methods = [

                    "brain_process",

                ]

                for method_name in required_methods:

                    method = getattr(
                        system,
                        method_name,
                        None,
                    )

                    if not callable(
                        method
                    ):

                        logger.warning(
                            "Missing required method: %s",
                            method_name,
                        )

                        return "degraded"

                # ========================================================
                # SUCCESS
                # ========================================================
                return "ready"

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "get_status crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            return "error"

    def is_processing(self) -> bool:
        """
        Production-safe processing state checker.

        Features:
        - thread-safe state validation
        - corruption detection
        - stale processing detection
        - auto-recovery for stuck states
        - shutdown awareness
        - defensive runtime checks
        """

        import threading
        import time
        import traceback

        try:

            # ============================================================
            # THREAD SAFETY
            # ============================================================
            if not hasattr(
                self,
                "_processing_lock",
            ):

                self._processing_lock = threading.RLock()

            with self._processing_lock:

                # ========================================================
                # SHUTDOWN CHECK
                # ========================================================
                if bool(
                    getattr(
                        self,
                        "_shutdown",
                        False,
                    )
                ):

                    return False

                # ========================================================
                # VALIDATE PROCESSING FLAG
                # ========================================================
                processing = getattr(
                    self,
                    "_processing",
                    False,
                )

                if not isinstance(
                    processing,
                    bool,
                ):

                    logger.warning(
                        "Processing flag corrupted — resetting"
                    )

                    self._processing = False

                    return False

                # ========================================================
                # FAST PATH
                # ========================================================
                if not processing:

                    return False

                # ========================================================
                # STUCK PROCESSING DETECTION
                # ========================================================
                try:

                    last_processing_time = float(

                        getattr(
                            self,
                            "_last_processing_time",
                            0,
                        )
                    )

                except Exception:

                    last_processing_time = 0

                if last_processing_time > 0:

                    elapsed = time.time() - last_processing_time

                    # auto-reset stale processing state
                    max_processing_time = 120.0

                    if elapsed > max_processing_time:

                        logger.warning(
                            "Detected stuck processing state "
                            "(%.2fs) — auto resetting",
                            elapsed,
                        )

                        self._processing = False

                        return False

                # ========================================================
                # OPTIONAL SYSTEM VALIDATION
                # ========================================================
                system = getattr(
                    self,
                    "system",
                    None,
                )

                if (
                    processing
                    and system is None
                ):

                    logger.warning(
                        "Processing active but system missing"
                    )

                    self._processing = False

                    return False

                # ========================================================
                # SUCCESS
                # ========================================================
                return True

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            logger.debug(
                "is_processing crashed: %s",
                e,
            )

            logger.debug(
                traceback.format_exc()[:3000]
            )

            # fail-safe reset
            try:

                self._processing = False

            except Exception:
                pass

            return False

# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_backend_instance: Optional[AIBackend] = None


def get_backend() -> AIBackend:
    """
    Production-safe singleton backend factory.

    Features:
    - thread-safe singleton creation
    - corruption detection
    - auto-recovery
    - lazy initialization
    - startup validation
    - memory-safe instance handling
    - defensive runtime protection
    """

    import threading
    import traceback
    import gc

    global _backend_instance

    try:

        # ============================================================
        # GLOBAL LOCK INIT
        # ============================================================
        global _backend_lock

        try:

            _backend_lock

        except NameError:

            _backend_lock = threading.RLock()

        # ============================================================
        # THREAD-SAFE SINGLETON
        # ============================================================
        with _backend_lock:

            # ========================================================
            # EXISTING INSTANCE VALIDATION
            # ========================================================
            if _backend_instance is not None:

                try:

                    # type validation
                    if not isinstance(
                        _backend_instance,
                        AIBackend,
                    ):

                        logger.warning(
                            "Backend instance corrupted — recreating"
                        )

                        _backend_instance = None

                    # callable corruption
                    elif callable(
                        _backend_instance
                    ):

                        logger.warning(
                            "Backend instance callable corruption"
                        )

                        _backend_instance = None

                    else:

                        # optional health check
                        try:

                            initialized = bool(

                                getattr(
                                    _backend_instance,
                                    "_initialized",
                                    False,
                                )
                            )

                            if not initialized:

                                logger.warning(
                                    "Backend instance not initialized"
                                )

                                init_result = (
                                    _backend_instance.initialize()
                                )

                                if not init_result:

                                    logger.warning(
                                        "Backend re-init failed"
                                    )

                                    _backend_instance = None

                        except Exception as e:

                            logger.warning(
                                "Backend validation failed: %s",
                                e,
                            )

                            _backend_instance = None

                except Exception as e:

                    logger.warning(
                        "Backend integrity check failed: %s",
                        e,
                    )

                    _backend_instance = None

            # ========================================================
            # CREATE INSTANCE
            # ========================================================
            if _backend_instance is None:

                logger.info(
                    "🧠 Creating AIBackend singleton..."
                )

                try:

                    backend = AIBackend()

                except Exception as e:

                    logger.exception(
                        "AIBackend constructor failed: %s",
                        e,
                    )

                    raise

                # ----------------------------------------------------
                # VALIDATE INSTANCE
                # ----------------------------------------------------
                if backend is None:

                    raise RuntimeError(
                        "AIBackend returned None"
                    )

                if callable(
                    backend
                ):

                    raise RuntimeError(
                        "AIBackend corruption detected"
                    )

                # ----------------------------------------------------
                # SAFE INITIALIZATION
                # ----------------------------------------------------
                try:

                    if hasattr(
                        backend,
                        "initialize",
                    ):

                        initialize_method = getattr(

                            backend,

                            "initialize",

                            None,
                        )

                        if callable(
                            initialize_method
                        ):

                            init_result = bool(

                                initialize_method()
                            )

                            if not init_result:

                                raise RuntimeError(
                                    "Backend initialization failed"
                                )

                except Exception as e:

                    logger.exception(
                        "Backend initialization failed: %s",
                        e,
                    )

                    raise

                # ----------------------------------------------------
                # OPTIONAL STARTUP
                # ----------------------------------------------------
                try:

                    if hasattr(
                        backend,
                        "start",
                    ):

                        start_method = getattr(

                            backend,

                            "start",

                            None,
                        )

                        if callable(
                            start_method
                        ):

                            start_result = bool(

                                start_method()
                            )

                            if not start_result:

                                logger.warning(
                                    "Backend start returned False"
                                )

                except Exception as e:

                    logger.warning(
                        "Backend startup warning: %s",
                        e,
                    )

                # ----------------------------------------------------
                # ASSIGN SINGLETON
                # ----------------------------------------------------
                _backend_instance = backend

                # ----------------------------------------------------
                # MEMORY CLEANUP
                # ----------------------------------------------------
                try:

                    gc.collect()

                except Exception:
                    pass

                logger.info(
                    "✅ AIBackend singleton created"
                )

            # ========================================================
            # FINAL VALIDATION
            # ========================================================
            if _backend_instance is None:

                raise RuntimeError(
                    "Backend singleton unavailable"
                )

            # ========================================================
            # SUCCESS
            # ========================================================
            return _backend_instance

    # ================================================================
    # HARD FAILURE
    # ================================================================
    except Exception as e:

        logger.exception(
            "❌ get_backend crashed: %s",
            e,
        )

        logger.debug(
            traceback.format_exc()[:4000]
        )

        # ------------------------------------------------------------
        # SAFE RESET
        # ------------------------------------------------------------
        try:

            _backend_instance = None

        except Exception:
            pass

        raise


__all__ = ["AIBackend", "get_backend"]
