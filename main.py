from __future__ import annotations

import os
import sys
import asyncio
import argparse
import time
import threading
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

# ── Kivy environment flags (must be set before any kivy import) ──────────────
os.environ.setdefault("KIVY_NO_CONSOLELOG", "1")
os.environ.setdefault("KIVY_NO_FILELOG", "1")
os.environ.setdefault("KIVY_NO_ARGS", "1")

# ── stdout encoding ──────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Logging (must be configured before first use of `logger`) ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("ediath.main")

# ── Legacy-mode deprecation notice ──────────────────────────────────────────
logger.info(
    "main.py is the legacy entry point. Prefer: python -m ediath.cli --mode=ui"
)

# ── System component imports ─────────────────────────────────────────────────
from core.security.content_filter import ContentFilter
from core.brain.llm_engine import LLMEngine
from core.brain.decision_engine import DecisionEngine, DecisionType
from core.autonomy.autonomous_loop import AutonomousLoop
from core.autonomy.background_thinker import BackgroundThinker
from core.autonomy.curiosity_engine import CuriosityEngine
from core.autonomy.goal_manage import GoalManager
from core.autonomy.self_improvement import SelfImprovement
from core.autonomous import software_builder
from core.autonomous.autonomous_core import AutonomousCoreEngine
from core.system.file_manager import FileManager
from core.system.browser_controller import BrowserController
from core.system.executor import Executor
from core.system.orchestrator import EDIATHOrchestrator, OrchestratorConfig
from core.system.task_queue import (
    Task,
    TaskQueueManager,
    TaskPriority,
    QueueType,
    init_task_queue_system,
    shutdown_task_queue_system,
)
from core.system.rate_limiter import (
    RateLimiter,
    RateLimitRule,
    RateLimitStrategy,
    RateLimitScope,
    RateLimitAction,
    get_rate_limiter,
    shutdown_rate_limiter,
    rate_limited,
)
from core.security.encryption import EncryptionManager
from core.security.permission_manager import PermissionManager
from core.vision import vision_engine
from core.automation.task_scheduler import TaskScheduler as AutoTaskScheduler
from core.automation.chrome_controller import ChromeController
from core.automation.citation_manager import CitationManager
from core.automation.human_browser import HumanBrowser
from core.automation.data_extractor import DataExtractor
from core.automation.web_researcher import WebResearcher
from core.agent.agent_core import RAgent

# Optional: multi-agent wrappers requested by user
try:
    from core.agents.register_with_coordinator import register_default_agents as _register_default_agents
except Exception:
    _register_default_agents = None


# ── Optional: FilterAction for content filtering ─────────────────────────────
try:
    from core.security.content_filter import FilterAction as _FilterAction
except Exception as _e:
    logger.debug("FilterAction not importable: %s", _e)
    _FilterAction = None

# ── Optional: Kivy ───────────────────────────────────────────────────────────
try:
    from kivy.clock import Clock as KivyClock
    _KIVY_AVAILABLE = True
except Exception:
    KivyClock = None
    _KIVY_AVAILABLE = False

# ── Optional: AudioListener ──────────────────────────────────────────────────
try:
    from core.audio.listener import AudioListener
except Exception as _e:
    AudioListener = None
    logger.warning("AudioListener import failed: %s", _e)

from core.audio.speaker import Speaker
from core.audio.wakeword import Wakeword

# ── Optional: config_loader ──────────────────────────────────────────────────
try:
    from core.system import config_loader as _config_loader_module
    _CONFIG_LOADER_AVAILABLE = True
except Exception:
    _config_loader_module = None
    _CONFIG_LOADER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON LLM ENGINE - LOADED ONCE FOR ALL AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class SharedLLMEngine:
    """
    TRUE SINGLETON LLM Engine.
    - Loaded ONLY ONCE at system startup
    - Shared across ALL agents (Orchestrator, DecisionEngine, CuriosityEngine, RAgent, etc.)
    - Prevents redundant LLM instances and reduces system load
    """
    _instance: Optional[LLMEngine] = None
    _initialized: bool = False
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> Optional[LLMEngine]:
        """Get or create the single LLM engine instance (thread-safe)."""
        if cls._instance is not None:
            return cls._instance
        
        with cls._lock:
            if cls._instance is None:
                logger.info("=" * 60)
                logger.info("🚀 CREATING SINGLETON LLM ENGINE (FIRST & ONLY INSTANCE)")
                logger.info("=" * 60)
                try:
                    cls._instance = LLMEngine()
                    cls._initialized = True
                    logger.info("✅ Shared LLM Engine created successfully")
                    logger.info("   → This instance will be shared across ALL agents")
                    logger.info("   → No additional LLM instances will be created")
                except Exception as e:
                    logger.error(f"❌ Failed to create shared LLM engine: {e}")
                    cls._instance = None
                    cls._initialized = False
            return cls._instance

    @classmethod
    def is_ready(cls) -> bool:
        return cls._initialized and cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            if cls._instance:
                try:
                    # Attempt to clean up if needed
                    if hasattr(cls._instance, 'shutdown'):
                        try:
                            asyncio.create_task(cls._instance.shutdown())
                        except Exception:
                            pass
                except Exception:
                    pass
            cls._instance = None
            cls._initialized = False
            logger.info("SharedLLMEngine reset")


# ══════════════════════════════════════════════════════════════════════════════
# INJECTOR FUNCTION - Ensures all components receive the shared LLM
# ══════════════════════════════════════════════════════════════════════════════

def inject_shared_llm_into_component(component: Any, shared_llm: LLMEngine) -> None:
    """
    Recursively inject the shared LLM engine into a component and all its sub-components.
    This ensures NO component creates its own LLM instance.
    """
    if not shared_llm or not component:
        return
    
    # Direct attribute injection
    if hasattr(component, 'llm_engine'):
        component.llm_engine = shared_llm
        if hasattr(component, 'logger'):
            component.logger.debug(f"Injected shared LLM into {component.__class__.__name__}.llm_engine")
    
    if hasattr(component, 'llm'):
        component.llm = shared_llm
        if hasattr(component, 'logger'):
            component.logger.debug(f"Injected shared LLM into {component.__class__.__name__}.llm")
    
    # Method-based injection
    if hasattr(component, 'set_llm_engine') and callable(component.set_llm_engine):
        try:
            component.set_llm_engine(shared_llm)
        except Exception as e:
            logger.debug(f"set_llm_engine failed for {type(component).__name__}: {e}")
    
    if hasattr(component, 'set_llm') and callable(component.set_llm):
        try:
            component.set_llm(shared_llm)
        except Exception as e:
            logger.debug(f"set_llm failed for {type(component).__name__}: {e}")
    
    # Recursively inject into child components
    for attr_name in [
        'decision_engine', 'curiosity', 'agent', 'orchestrator',
        'memory_api', 'vision_engine_instance', 'goal_manager',
        'self_improvement', 'autonomous_core', 'autonomous_loop',
        'background_thinker', 'software_builder', 'data_extractor',
        'web_researcher', 'human_browser',
    ]:
        if hasattr(component, attr_name):
            child = getattr(component, attr_name)
            if child is not None and child is not component:
                inject_shared_llm_into_component(child, shared_llm)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config/orchestrator_config.yaml")
AI_QUEUE_NAME = "ai_tasks"
AUTOMATION_QUEUE_NAME = "automation"
LEARNING_QUEUE_NAME = "learning"
LISTENER_POLL_INTERVAL = 0.05
LISTENER_READY_TIMEOUT = 30.0
MAX_LISTENER_RESTARTS = 5
LISTENER_BACKOFF_BASE = 1.0

# Module-level system reference
_SYSTEM_INSTANCE: Optional["EDIATHSystem"] = None
_UI_CALLBACK: Optional[callable] = None


def set_ui_callback(callback: callable) -> None:
    """Set the UI callback function for sending messages to the chat interface."""
    global _UI_CALLBACK
    _UI_CALLBACK = callback


# ─────────────────────────────────────────────────────────────────────────────
# Windows-safe atomic file write (prevents WinError 5 Access Denied)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_atomic_write(
    tmp_path: str, final_path: str, retries: int = 5, delay: float = 0.3
) -> None:
    """
    Atomic rename with retry loop.
    On Windows, os.replace can raise PermissionError if another process
    still has the .tmp file open. We back off and retry before giving up.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError as exc:
            last_exc = exc
            logger.warning(
                "Persist write permission error (attempt %d): %s", attempt + 1, exc
            )
            time.sleep(delay * (attempt + 1))  # progressive back-off
        except Exception as exc:
            raise exc  # non-permission errors: surface immediately
    raise PermissionError(
        f"Could not rename {tmp_path!r} → {final_path!r} after {retries} attempts"
    ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Lazy agent module imports
# ─────────────────────────────────────────────────────────────────────────────

_agent_modules_imported = False
_intent_classifier_instance = None
_action_router_instance = None
_router_instance = None


def _import_agent_modules(shared_llm: Optional[LLMEngine] = None) -> bool:
    """Import agent modules and inject the shared LLM engine."""
    global _agent_modules_imported
    global _intent_classifier_instance, _action_router_instance, _router_instance

    if _agent_modules_imported:
        return True
    
    try:
        from core.agent.intent_classifier import IntentClassifier
        from core.agent.action_router import ActionRouter
        from core.agent.router import Router

        # Create instances with SHARED LLM (not a new one!)
        _intent_classifier_instance = IntentClassifier(llm_engine=shared_llm)
        _action_router_instance = ActionRouter()
        _router_instance = Router()
        
        # Inject shared LLM into these agents as well
        inject_shared_llm_into_component(_intent_classifier_instance, shared_llm)
        
        _agent_modules_imported = True
        logger.info("✅ Agent modules loaded with SHARED LLM engine")
        return True
    except Exception as exc:
        logger.warning(f"Agent modules unavailable: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main system class
# ─────────────────────────────────────────────────────────────────────────────

class EDIATHSystem:
    """Main EDIATH AI System with SHARED SINGLETON LLM Engine."""

    def __init__(self, orchestrator: Optional[EDIATHOrchestrator] = None) -> None:
        self._tasks: List[asyncio.Task] = []
        self._backend_loop: Optional[asyncio.AbstractEventLoop] = None
        self._brain_lock: asyncio.Lock = asyncio.Lock()
        self._last_input: Optional[str] = None
        self._listener_ready = threading.Event()
        self._listener_restart_count = 0

        self.is_running = False
        self.thinking = False
        self.llm_ready = False
        self.logger = logger

        self.orchestrator = orchestrator
        self.task_queue_manager: Optional[TaskQueueManager] = None
        self.rate_limiter: Optional[RateLimiter] = None

        # Core components
        self.file_manager = FileManager()
        self.content_filter = ContentFilter()
        self.task_scheduler = AutoTaskScheduler()
        self.goal_manager = GoalManager()
        self.curiosity = CuriosityEngine()
        self.self_improvement = SelfImprovement()

        # ═══════════════════════════════════════════════════════════════════
        # SHARED LLM ENGINE - SINGLETON (LOADED ONCE)
        # ═══════════════════════════════════════════════════════════════════
        self.shared_llm: Optional[LLMEngine] = None
        self.llm_engine: Optional[LLMEngine] = None  # Alias for compatibility
        self.llm_lock = asyncio.Lock()

        # Decision engine (will receive shared LLM)
        self.decision_engine = DecisionEngine()

        # Autonomy
        self.autonomous_core = AutonomousCoreEngine()
        self.autonomous_loop = AutonomousLoop(self)
        self.background_thinker = BackgroundThinker(self)

        # Audio
        self.listener = AudioListener() if AudioListener else None
        self.speaker = Speaker()
        self.wakeword = Wakeword()

        # Automation
        self.browser_controller = BrowserController()
        self.human_browser = HumanBrowser()
        self.chrome_controller = ChromeController()
        self.citation_manager = CitationManager()
        self.data_extractor = DataExtractor()
        self.web_researcher = WebResearcher()

        # Security
        self.permission_manager = PermissionManager()
        self.encryption_manager = EncryptionManager()

        # Vision / build
        self.vision_engine_instance = None
        self.software_builder = software_builder.SoftwareBuilder()

        # UI references
        self.ui_window = None
        self.chat_panel = None
        self.status_bar = None
        self.overlay = None

        # Optional specialised agents
        self.search_agent = None
        self.code_agent = None
        self.file_agent = None

        # Core agent (will receive shared LLM)
        self.agent = None  # Initialized after shared LLM is ready

        # Memory
        self.memory_api = None
        try:
            from core.memory.memory_manager import MemoryManager
            self.memory_api = MemoryManager()
        except Exception as exc:
            self.logger.warning("Memory API not available: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # SHARED LLM INJECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _inject_shared_llm_to_all_components(self) -> None:
        """Inject the shared LLM engine into ALL components."""
        if not self.shared_llm:
            self.logger.error("Cannot inject: shared_llm is None")
            return

        self.logger.info("=" * 60)
        self.logger.info("🔗 INJECTING SHARED LLM INTO ALL COMPONENTS")
        self.logger.info("=" * 60)

        # List of all components that need the shared LLM
        components_to_inject = [
            ("DecisionEngine", self.decision_engine),
            ("CuriosityEngine", self.curiosity),
            ("Orchestrator", self.orchestrator),
            ("AutonomousCore", self.autonomous_core),
            ("GoalManager", self.goal_manager),
            ("SelfImprovement", self.self_improvement),
            ("RAgent", self.agent),
        ]

        for name, comp in components_to_inject:
            if comp:
                inject_shared_llm_into_component(comp, self.shared_llm)
                self.logger.info(f"   ✅ Injected into {name}")

        # Inject into agent modules
        if _import_agent_modules(self.shared_llm):
            global _intent_classifier_instance, _action_router_instance, _router_instance
            for name, comp in [
                ("IntentClassifier", _intent_classifier_instance),
                ("ActionRouter", _action_router_instance),
                ("Router", _router_instance),
            ]:
                if comp:
                    inject_shared_llm_into_component(comp, self.shared_llm)
                    self.logger.info(f"   ✅ Injected into {name}")

        self.logger.info("=" * 60)
        self.logger.info("✅ ALL COMPONENTS NOW USE THE SAME LLM ENGINE")
        self.logger.info("   → No duplicate LLM instances will be created")
        self.logger.info("=" * 60)

    # ── Listener helpers ──────────────────────────────────────────────────────

    def _start_listener(self) -> bool:
        if not self.listener:
            return False
        try:
            if hasattr(self.listener, "start_listening"):
                self.listener.start_listening()
            elif hasattr(self.listener, "start"):
                self.listener.start()
            else:
                self.logger.error("Listener has no start method")
                return False
            time.sleep(0.3)
            self._listener_ready.set()
            self._listener_restart_count = 0
            self.logger.info("Listener started")
            return True
        except Exception as exc:
            self.logger.error("Listener start failed: %s", exc)
            return False

    def _restart_listener(self) -> bool:
        if self._listener_restart_count >= MAX_LISTENER_RESTARTS:
            self.logger.error("Max listener restarts reached")
            return False
        backoff = LISTENER_BACKOFF_BASE * (2**self._listener_restart_count)
        self._listener_restart_count += 1
        self.logger.warning(
            "Restarting listener (attempt %d) after %.1fs",
            self._listener_restart_count,
            backoff,
        )
        time.sleep(backoff)
        try:
            if hasattr(self.listener, "stop"):
                self.listener.stop()
        except Exception as exc:
            self.logger.debug("Listener stop during restart: %s", exc)
        self._listener_ready.clear()
        return self._start_listener()

    def _get_listener_text(self) -> Optional[str]:
        if not self.listener:
            return None
        try:
            for method in ("get_text", "get_transcript", "result"):
                if hasattr(self.listener, method):
                    return getattr(self.listener, method)() or None
        except Exception:
            pass
        return None

    def start_wakeword(self) -> None:
        if not self.wakeword or not self.listener:
            return
        try:
            if hasattr(self.wakeword, "start"):
                self.wakeword.start()
            if hasattr(self.listener, "set_wakeword"):
                self.listener.set_wakeword(self.wakeword)
            self.logger.info("Wakeword started")
        except Exception as exc:
            self.logger.warning("Wakeword start failed: %s", exc)

    # ── Initialization ────────────────────────────────────────────────────────

    async def validate_deps(self) -> Dict[str, bool]:
        deps = {}

        try:
            deps["audio_listener"] = AudioListener is not None
            deps["kivy"] = _KIVY_AVAILABLE
            deps["config_file"] = CONFIG_PATH.exists()
            deps["llm_engine"] = self.shared_llm is not None
            deps["event_loop"] = asyncio.get_running_loop() is not None

            try:
                await asyncio.wait_for(asyncio.sleep(0), timeout=1)
                deps["async_ready"] = True
            except asyncio.TimeoutError:
                deps["async_ready"] = False

            missing = [k for k, v in deps.items() if not v]
            if missing:
                self.logger.warning(
                    "⚠ Missing deps: %s → running in degraded mode", ", ".join(missing)
                )
            else:
                self.logger.info("✅ Dependency check passed")

            return deps

        except Exception as exc:
            self.logger.error("Dependency validation failed: %s", exc, exc_info=True)
            return {"error": True}

    async def initialize(self, *args, **kwargs) -> bool:
        try:
            self.logger.info("=" * 60)
            self.logger.info("EDIATH AUTONOMOUS AI SYSTEM — INITIALIZATION")
            self.logger.info("WITH SHARED SINGLETON LLM ENGINE")
            self.logger.info("=" * 60)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: CREATE SHARED LLM ENGINE (LOADED ONCE)
            # ═══════════════════════════════════════════════════════════════════
            self.logger.info("📦 STEP 1: Creating Shared Singleton LLM Engine...")
            self.shared_llm = SharedLLMEngine.get_instance()
            
            if not self.shared_llm:
                self.logger.error("❌ Failed to create shared LLM engine")
                self.llm_ready = False
                return False
            
            # Set aliases for compatibility
            self.llm_engine = self.shared_llm
            
            # Initialize the LLM engine (load model once)
            self.logger.info("⏳ Initializing LLM Engine (loading model)...")
            try:
                # Increase timeout for slow model loading on first run
                await asyncio.wait_for(self.shared_llm.initialize(), timeout=120)
                if self.shared_llm and self.shared_llm.provider:
                    self.llm_ready = True
                    self.logger.info("✅ LLM Engine ready (model loaded once)")
                else:
                    raise RuntimeError("LLM provider missing")
            except asyncio.TimeoutError:
                self.logger.error("❌ LLM initialization timed out after 120s")
                self.llm_ready = False
                # Don't return False - allow partial operation
            except Exception as exc:
                self.logger.error(f"❌ LLM initialization failed: {exc}")
                self.llm_ready = False
                # Don't return False - allow partial operation

            await self.validate_deps()

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Create RAgent with SHARED LLM
            # ═══════════════════════════════════════════════════════════════════
            self.logger.info("🤖 STEP 2: Creating RAgent with shared LLM...")
            self.agent = RAgent(self)
            inject_shared_llm_into_component(self.agent, self.shared_llm)
            self.logger.info("✅ RAgent created with shared LLM")

            # ── RATE LIMITER ──────────────────────────────────────────────────
            try:
                self.logger.info("Initializing RateLimiter…")
                self.rate_limiter = await asyncio.wait_for(
                    get_rate_limiter(), timeout=5
                )
                await self._setup_rate_limits()
                self.logger.info("RateLimiter ready")
            except Exception as exc:
                self.logger.warning("RateLimiter failed → continuing without it: %s", exc)
                self.rate_limiter = None

            # ── TASK QUEUE ────────────────────────────────────────────────────
            try:
                self.logger.info("Initializing TaskQueue…")
                self.task_queue_manager = await asyncio.wait_for(
                    init_task_queue_system(), timeout=5
                )
                await self._setup_task_queues()
                self.logger.info("TaskQueue ready")
            except Exception as exc:
                self.logger.warning("TaskQueue failed → degraded mode: %s", exc)
                self.task_queue_manager = None

            # ── DECISION ENGINE with SHARED LLM ────────────────────────────────
            self.logger.info("⚖️ STEP 3: Configuring DecisionEngine with shared LLM...")
            inject_shared_llm_into_component(self.decision_engine, self.shared_llm)
            self.logger.info("✅ DecisionEngine using shared LLM")

            # ── CURIOSITY ENGINE with SHARED LLM ───────────────────────────────
            self.logger.info("🧠 STEP 4: Configuring CuriosityEngine with shared LLM...")
            inject_shared_llm_into_component(self.curiosity, self.shared_llm)
            self.logger.info("✅ CuriosityEngine using shared LLM")

            # ── ORCHESTRATOR ──────────────────────────────────────────────────
            if self.orchestrator:
                try:
                    self.logger.info("STEP 5: Initializing Orchestrator with shared LLM...")
                    self.orchestrator.system_instance = self
                    inject_shared_llm_into_component(self.orchestrator, self.shared_llm)

                    ok = await asyncio.wait_for(
                        self.orchestrator.initialize(), timeout=30
                    )

                    if not ok:
                        raise RuntimeError("Orchestrator returned False")

                    self.logger.info("✅ Orchestrator ready with shared LLM")
                    t = asyncio.create_task(self.orchestrator.run())
                    self._tasks.append(t)

                except asyncio.TimeoutError:
                    self.logger.warning("Orchestrator init timed out — running without it")
                    self.orchestrator = None
                except Exception as exc:
                    self.logger.warning("Orchestrator disabled: %s", exc)
                    self.orchestrator = None

            # ── INJECT SHARED LLM INTO ALL OTHER COMPONENTS ────────────────────
            self._inject_shared_llm_to_all_components()

            # ── EXECUTOR ──────────────────────────────────────────────────────
            try:
                executor = Executor()

                def secure_run_file(*args, **kwargs):
                    if not self.permission_manager.check("run_file"):
                        return None
                    return self.file_manager.run_file(*args, **kwargs)

                def secure_open_file(*args, **kwargs):
                    if not self.permission_manager.check("open_file"):
                        return None
                    return self.file_manager.open_file(*args, **kwargs)

                executor.register("run_file", secure_run_file)
                executor.register("open_file", secure_open_file)
                self.executor_system = executor
                self.logger.info("Executor ready")

            except Exception as exc:
                self.logger.warning("Executor disabled: %s", exc)

            # ── AUDIO (NON-CRITICAL) ──────────────────────────────────────────
            try:
                if self.speaker and hasattr(self.speaker, "initialize"):
                    await asyncio.to_thread(self.speaker.initialize)
                    self.logger.info("Speaker ready")
            except Exception as exc:
                self.logger.warning("Audio disabled: %s", exc)

            # ── VISION ────────────────────────────────────────────────────────
            try:
                # Check system memory before allocating heavy vision resources
                # Proactively skip vision if memory is above 85% to prevent OOM kills
                if os.environ.get("EDIATH_LIGHT_MODE", "0") == "1":
                    raise RuntimeError("Light mode enabled")

                try:
                    import psutil
                    mem_pct = psutil.virtual_memory().percent
                    if mem_pct >= 85.0:
                        self.logger.warning(
                            "⚠️ System memory at %.1f%% — skipping vision (would trigger OOM)",
                            mem_pct,
                        )
                        raise RuntimeError(f"Memory pressure ({mem_pct:.1f}%)")
                except ImportError:
                    pass  # psutil unavailable — proceed with vision anyway

                self.vision_engine_instance = vision_engine.VisionEngine()
                inject_shared_llm_into_component(self.vision_engine_instance, self.shared_llm)
                await asyncio.wait_for(
                    self.vision_engine_instance.initialize(), timeout=5
                )
                await asyncio.wait_for(
                    self.vision_engine_instance.start_vision(), timeout=5
                )
                self.logger.info("✅ Vision Engine started with shared LLM")

            except Exception as exc:
                self.logger.warning("Vision disabled: %s", exc)
                self.vision_engine_instance = None

            # ── FINAL STATE ───────────────────────────────────────────────────
            set_ui_callback(self.send_ui_message)
            self.is_running = True

            self.logger.info("=" * 60)
            self.logger.info("✅ EDIATH INITIALIZATION COMPLETE")
            self.logger.info(f"   → LLM Engine: {'READY' if self.llm_ready else 'DEGRADED MODE'}")
            self.logger.info("   → ALL AGENTS SHARE THE SAME LLM INSTANCE")
            self.logger.info("   → System load optimized (no duplicate LLM models)")
            self.logger.info("=" * 60)

            if self.llm_ready:
                self.send_ui_message("SYSTEM", "✅ AI READY (Shared LLM Mode)")
            else:
                self.send_ui_message("SYSTEM", "⚠ AI INITIALIZED (LLM FAILED - Using fallback mode)")

            return True

        except Exception as exc:
            self.logger.error("💥 Initialization failed: %s", exc, exc_info=True)
            self.is_running = False
            return False

    async def _setup_rate_limits(self) -> None:
        if not self.rate_limiter:
            return

        try:
            rules = [
                RateLimitRule(
                    name="llm_api",
                    key_pattern="llm:*",
                    limit=60,
                    window_seconds=60,
                    strategy=RateLimitStrategy.TOKEN_BUCKET,
                    scope=RateLimitScope.GLOBAL,
                    action=RateLimitAction.THROTTLE,
                    burst_limit=10,
                    refill_rate=1.5,
                    queue_size=20,
                    queue_timeout_seconds=10,
                    priority=1,
                ),
                RateLimitRule(
                    name="chat_messages",
                    key_pattern="user:*:chat",
                    limit=15,
                    window_seconds=60,
                    strategy=RateLimitStrategy.SLIDING_WINDOW,
                    scope=RateLimitScope.PER_USER,
                    action=RateLimitAction.DELAY,
                    priority=2,
                ),
                RateLimitRule(
                    name="file_operations",
                    key_pattern="file:*",
                    limit=80,
                    window_seconds=60,
                    strategy=RateLimitStrategy.FIXED_WINDOW,
                    scope=RateLimitScope.GLOBAL,
                    action=RateLimitAction.BLOCK,
                    priority=3,
                ),
                RateLimitRule(
                    name="browser_automation",
                    key_pattern="browser:*",
                    limit=20,
                    window_seconds=60,
                    strategy=RateLimitStrategy.LEAKY_BUCKET,
                    scope=RateLimitScope.GLOBAL,
                    action=RateLimitAction.THROTTLE,
                    burst_limit=5,
                    priority=4,
                ),
                RateLimitRule(
                    name="code_execution",
                    key_pattern="code:*",
                    limit=10,
                    window_seconds=60,
                    strategy=RateLimitStrategy.SLIDING_LOG,
                    scope=RateLimitScope.GLOBAL,
                    action=RateLimitAction.BLOCK,
                    priority=5,
                ),
            ]

            for rule in rules:
                try:
                    self.rate_limiter.add_rule(rule)
                except Exception as exc:
                    self.logger.warning("Failed to add rule %s: %s", rule.name, exc)

            self.logger.info("✅ Rate limits configured")

        except Exception as exc:
            self.logger.error("Rate limiter setup failed: %s", exc)

    async def _setup_task_queues(self) -> None:
        if not self.task_queue_manager:
            return

        try:
            await asyncio.wait_for(
                self.task_queue_manager.create_queue(AI_QUEUE_NAME, QueueType.PRIORITY),
                timeout=3,
            )
            await asyncio.wait_for(
                self.task_queue_manager.create_queue(
                    AUTOMATION_QUEUE_NAME, QueueType.FIFO
                ),
                timeout=3,
            )
            await asyncio.wait_for(
                self.task_queue_manager.create_queue(
                    LEARNING_QUEUE_NAME, QueueType.ROUND_ROBIN
                ),
                timeout=3,
            )
        except Exception as exc:
            self.logger.warning("Queue creation issue: %s", exc)

        async def ai_handler(task: Task) -> Dict[str, Any]:
            try:
                if self.rate_limiter:
                    rr = await self.rate_limiter.check_rate_limit(
                        identifier="ai_worker",
                        context={"endpoint": "ai_task"},
                    )
                    if not rr.get("allowed"):
                        return {"error": "Rate limited"}

                prompt = str(task.data.get("prompt", "")).strip()
                if not prompt:
                    return {"error": "No prompt", "task_id": task.id}

                try:
                    result = await asyncio.wait_for(self.agent.run(prompt), timeout=10)
                except asyncio.TimeoutError:
                    return {"error": "AI timeout", "task_id": task.id}

                return {"response": result, "task_id": task.id}

            except Exception as exc:
                return {"error": str(exc), "task_id": task.id}

        async def auto_handler(task: Task) -> Dict[str, Any]:
            try:
                if self.rate_limiter:
                    rr = await self.rate_limiter.check_rate_limit(
                        identifier="automation_worker",
                        context={"endpoint": "automation"},
                    )
                    if not rr.get("allowed"):
                        return {"error": "Rate limited"}

                action = str(task.data.get("action", "")).strip()

                if action == "browse":
                    url = str(task.data.get("url", "")).strip()
                    if not url:
                        return {"error": "No URL", "task_id": task.id}
                    try:
                        result = await asyncio.wait_for(
                            self.browser_controller.open_url(url), timeout=10
                        )
                    except asyncio.TimeoutError:
                        return {"error": "Browser timeout", "task_id": task.id}
                    return {"result": result, "task_id": task.id}

                return {"result": f"{action} done", "task_id": task.id}

            except Exception as exc:
                return {"error": str(exc), "task_id": task.id}

        async def learn_handler(task: Task) -> Dict[str, Any]:
            try:
                fb = str(task.data.get("feedback", "")).strip()
                if fb:
                    try:
                        await asyncio.wait_for(
                            self.curiosity.process_feedback(fb), timeout=5
                        )
                    except asyncio.TimeoutError:
                        return {"error": "Learning timeout", "task_id": task.id}
                return {"result": "Learning done", "task_id": task.id}
            except Exception as exc:
                return {"error": str(exc), "task_id": task.id}

        try:
            await self.task_queue_manager.add_worker(
                AI_QUEUE_NAME, "ai_worker_1", ai_handler
            )
            await self.task_queue_manager.add_worker(
                AUTOMATION_QUEUE_NAME, "automation_worker_1", auto_handler
            )
            await self.task_queue_manager.add_worker(
                LEARNING_QUEUE_NAME, "learning_worker_1", learn_handler
            )
        except Exception as exc:
            self.logger.warning("Worker registration failed: %s", exc)

        self.logger.info("✅ Task queues ready (safe mode)")

    # ── Public task submission ────────────────────────────────────────────────

    async def submit_ai_task(
        self,
        prompt: str,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Optional[str]:
        if not prompt or not prompt.strip():
            return None
        if self.rate_limiter:
            rr = await self.rate_limiter.check_rate_limit(
                identifier="user_input",
                context={"endpoint": "ai_task_submit", "prompt_length": len(prompt)},
            )
            if not rr.get("allowed"):
                self.send_ui_message(
                    "SYSTEM",
                    f"Rate limit exceeded. Retry in {rr.get('reset_time', 60)}s.",
                )
                return None
        if not self.task_queue_manager:
            return None
        queue = await self.task_queue_manager.get_queue(AI_QUEUE_NAME)
        if not queue:
            return None
        task = Task(
            name="ai_processing",
            priority=priority,
            data={"prompt": prompt},
            timeout_seconds=30,
            max_retries=2,
        )
        await queue.put(task)
        return task.id

    async def submit_automation_task(
        self, action: str, data: Dict[str, Any]
    ) -> Optional[str]:
        if not action:
            return None
        if self.rate_limiter:
            rr = await self.rate_limiter.check_rate_limit(
                identifier="automation",
                context={"endpoint": "automation_submit", "action": action},
            )
            if not rr.get("allowed"):
                return None
        if not self.task_queue_manager:
            return None
        queue = await self.task_queue_manager.get_queue(AUTOMATION_QUEUE_NAME)
        if not queue:
            return None
        task = Task(
            name=f"automation_{action}",
            priority=TaskPriority.NORMAL,
            data={"action": action, **data},
            timeout_seconds=60,
        )
        await queue.put(task)
        return task.id

    async def submit_learning_task(self, feedback: str) -> Optional[str]:
        if not feedback or not feedback.strip():
            return None
        if not self.task_queue_manager:
            return None
        queue = await self.task_queue_manager.get_queue(LEARNING_QUEUE_NAME)
        if not queue:
            return None
        low = getattr(TaskPriority, "LOW", TaskPriority.NORMAL)
        task = Task(
            name="learning_feedback",
            priority=low,
            data={"feedback": feedback},
            timeout_seconds=20,
        )
        await queue.put(task)
        return task.id

    @rate_limited("chat_messages", identifier_key="user_id")
    async def rate_limited_chat(self, user_id: str, message: str) -> str:
        return await self.agent.run(message)

    async def filter_and_process(
        self, content: str, learn: bool = False
    ) -> Optional[str]:
        """Safe, fast content filtering with timeout + fallback."""
        try:
            if not content or not isinstance(content, str):
                return None

            content = content.strip()
            if not content:
                return None

            if len(content) > 2000:
                content = content[:2000]

            try:
                is_allowed, action, message = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.content_filter.filter_content, content, learn
                    ),
                    timeout=2,
                )
            except asyncio.TimeoutError:
                self.logger.warning("Content filter timeout → allowing input")
                return content
            except Exception as exc:
                self.logger.error("Content filter failure: %s", exc)
                return None

            if not is_allowed:
                self.send_ui_message("SYSTEM", f"Blocked: {message}")
                return None

            if _FilterAction is not None and action == _FilterAction.WARN:
                self.send_ui_message("SYSTEM", f"Warning: {message}")

            cleaned = content.replace("\x00", "").strip()
            return cleaned if cleaned else None

        except Exception as exc:
            self.logger.error("filter_and_process failed: %s", exc, exc_info=True)
            return None

    def process_user_input(self, user_input: str) -> None:
        """Thread-safe, non-blocking input handler with overload protection."""
        if not user_input or not user_input.strip():
            return

        user_input = user_input.strip()

        if getattr(self, "thinking", False):
            self.send_ui_message("SYSTEM", "Still thinking…")
            return

        if self._backend_loop and self._backend_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._process_user_input_async(user_input),
                    self._backend_loop,
                )
                return
            except Exception as exc:
                self.logger.warning("Backend loop failed: %s", exc)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_user_input_async(user_input))
            return
        except RuntimeError:
            pass

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    asyncio.wait_for(
                        self._process_user_input_async(user_input), timeout=15
                    )
                )
            except asyncio.TimeoutError:
                self.logger.warning("Input processing timeout")
                self.send_ui_message("SYSTEM", "Request timed out.")
            except Exception as exc:
                self.logger.error("Thread processing error: %s", exc, exc_info=True)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def set_backend_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._backend_loop = loop

    async def _process_user_input_async(self, user_input: str) -> None:
        try:
            if not self.is_running or not user_input:
                return

            if self.thinking:
                self.send_ui_message("SYSTEM", "Still thinking… please wait.")
                return

            self.thinking = True
            response = ""

            try:
                filtered = await asyncio.wait_for(
                    self.filter_and_process(user_input, learn=True), timeout=3
                )
            except asyncio.TimeoutError:
                self.send_ui_message("SYSTEM", "Input processing timeout")
                return

            if not filtered:
                return

            # Allow operation even without LLM (fallback mode)
            if not self.llm_ready:
                self.send_ui_message("SYSTEM", "⚠ AI in fallback mode (LLM not ready)")
                # Use fallback response
                response = "I'm operating in limited mode. Some features may be unavailable."
                self.send_ui_message("AI", response)
                self.thinking = False
                return

            agent = self.agent

            if _import_agent_modules(self.shared_llm):
                try:
                    intent = await asyncio.wait_for(
                        _intent_classifier_instance.classify(filtered), timeout=2
                    )
                    itype = intent.type.value if hasattr(intent, "type") else "query"
                    routed = await asyncio.wait_for(
                        _action_router_instance.route(itype, {}), timeout=2
                    )
                    agent_key = _router_instance.route({"action": routed or "unknown"})
                    agent = {
                        "search": self.search_agent or self.agent,
                        "codeexecution": self.code_agent or self.agent,
                        "file": self.file_agent or self.agent,
                        "llm": self.agent,
                    }.get(agent_key, self.agent)
                    self.logger.info("Routing → intent=%s, agent=%s", itype, agent_key)
                except Exception as exc:
                    self.logger.warning("Routing failed → fallback agent: %s", exc)

            vision_context = ""
            memory_context = ""

            try:
                if self.vision_engine_instance:
                    vdata = getattr(
                        self.vision_engine_instance, "current_frame_data", None
                    )
                    if vdata:
                        vision_context = (
                            await asyncio.wait_for(
                                self.decision_engine.process_vision(vdata), timeout=2
                            )
                            or ""
                        )
            except Exception:
                pass

            try:
                if self.memory_api and hasattr(self.memory_api, "get_recent"):
                    recent = await asyncio.wait_for(
                        self.memory_api.get_recent(limit=2), timeout=2
                    )
                    if recent:
                        memory_context = f"\nRecent: {recent}"
            except Exception:
                pass

            full_context = (
                f"{vision_context}\n{memory_context}\nUser: {filtered}".strip()
            )

            try:
                async with self.llm_lock:
                    decision_result = await asyncio.wait_for(
                        self.decision_engine.intelligent_decision(full_context),
                        timeout=10,
                    )

                if isinstance(decision_result, dict):
                    reasoning = decision_result.get("reasoning", "")
                    action = decision_result.get("action", "respond")
                    if action == "respond" and len(reasoning) > 5:
                        response = reasoning
                    else:
                        prompt = reasoning if reasoning else full_context
                        response = str(await agent.run(prompt) or "")
                else:
                    response = str(await agent.run(full_context) or "")

            except asyncio.TimeoutError:
                response = "System is busy, try again."
            except Exception as exc:
                self.logger.error("Decision error: %s", exc)
                response = "Error processing request."

            response = str(response or "No response generated.").strip()
            response = (
                response.replace("assistant:", "").replace("ASSISTANT:", "").strip()
            )

            self.send_ui_message("AI", response)

            if _UI_CALLBACK:
                try:
                    _UI_CALLBACK(response)
                except Exception:
                    pass

            if self.speaker and response:
                try:
                    asyncio.create_task(self.speaker.speak(response))
                except Exception:
                    pass

            self.logger.info("Response length: %d", len(response))

        except Exception as exc:
            self.logger.error("Processing error: %s", exc, exc_info=True)
            self.send_ui_message("SYSTEM", "Internal error occurred.")

        finally:
            self.thinking = False

    async def create_application(
        self, app_name: str, specifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if not app_name or not isinstance(app_name, str):
                return {"success": False, "error": "Invalid app name"}
            if not isinstance(specifications, dict):
                return {"success": False, "error": "Specifications must be a dict"}
            app_name = app_name.strip()
            if not app_name:
                return {"success": False, "error": "Empty app name"}
            try:
                if not self.permission_manager.check("create_application"):
                    return {"success": False, "error": "Permission denied"}
            except Exception as exc:
                self.logger.warning("Permission check failed: %s", exc)
            self.send_ui_message("SYSTEM", f"Building application: {app_name}…")
            try:
                result = await asyncio.wait_for(
                    self.software_builder.build_project(app_name, specifications),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                return {"success": False, "error": "Build timed out after 120s"}
            except Exception as exc:
                return {"success": False, "error": str(exc)}
            if not result:
                return {"success": False, "error": "Empty build result"}
            if not isinstance(result, dict):
                result = {"success": True, "data": result}
            self.send_ui_message("SYSTEM", f"App '{app_name}' created")
            return result
        except Exception as exc:
            self.logger.error("create_application failed: %s", exc)
            return {"success": False, "error": str(exc)}

    # ── UI messaging ──────────────────────────────────────────────────────────

    def send_ui_message(self, role: str, content: str) -> None:
        """Safe, non-blocking UI messaging with fallback + dedup protection."""
        if not content or not isinstance(content, str):
            return

        role = str(role).upper().strip() or "SYSTEM"
        content = content.strip()
        if not content:
            return

        if getattr(self, "_last_ui_message", None) == content:
            return
        self._last_ui_message = content

        if _UI_CALLBACK:
            try:
                # Prefer callback signature (role, content) if supported
                try:
                    _UI_CALLBACK(role, content)
                except TypeError:
                    _UI_CALLBACK(content)
            except Exception as exc:
                self.logger.debug("UI callback error: %s", exc)

        if not _KIVY_AVAILABLE or KivyClock is None:
            logger.info("[%s] %s", role, content)
            return

        def _update_ui(dt: float) -> None:
            try:
                if self.chat_panel and hasattr(self.chat_panel, "add_message"):
                    self.chat_panel.add_message(role, content)
                else:
                    print(f"[UI/{role}] {content}")
            except Exception as exc:
                self.logger.debug("ChatPanel error: %s", exc)

            try:
                if self.status_bar and hasattr(self.status_bar, "update_status"):
                    self.status_bar.update_status(content[:80])
            except Exception as exc:
                self.logger.debug("StatusBar error: %s", exc)

            try:
                if self.overlay and hasattr(self.overlay, "update_status"):
                    self.overlay.update_status(content[:100])
            except Exception as exc:
                self.logger.debug("Overlay error: %s", exc)

        try:
            KivyClock.schedule_once(_update_ui, 0)
        except Exception as exc:
            self.logger.warning("Kivy scheduling failed: %s", exc)
            logger.warning("[FALLBACK/%s] %s", role, content)

    # ── Autonomous reasoning ──────────────────────────────────────────────────

    async def run_autonomous_reasoning(self, duration: float = 10.0) -> Dict[str, Any]:
        try:
            if not isinstance(duration, (int, float)) or duration <= 0:
                return {"success": False, "error": "Invalid duration"}
            duration = float(duration)
            self.send_ui_message(
                "SYSTEM", f"Thinking autonomously for {duration:.1f}s…"
            )

            chosen, confidence = None, 0.0
            try:
                d = self.decision_engine.make_decision(
                    decision_type=DecisionType.ACTION,
                    context="choose_autonomy_subtask",
                    options=[("explore", 0.9), ("optimize", 0.7), ("rest", 0.3)],
                )
                if d and getattr(d, "options", None):
                    chosen, confidence = self.decision_engine.evaluate_decision(d)
            except Exception as exc:
                self.logger.warning("Decision phase failed: %s", exc)

            try:
                if self.autonomous_core:
                    await asyncio.wait_for(
                        self.autonomous_core.autonomous_operation(duration),
                        timeout=duration + 5,
                    )
            except (asyncio.TimeoutError, Exception) as exc:
                self.logger.warning("Autonomous operation: %s", exc)

            insights = []
            try:
                if self.curiosity:
                    insights = await asyncio.wait_for(
                        self.curiosity.autonomous_exploration(duration * 0.5),
                        timeout=(duration * 0.5) + 5,
                    )
            except (asyncio.TimeoutError, Exception) as exc:
                self.logger.warning("Curiosity exploration: %s", exc)

            count = len(insights) if isinstance(insights, (list, tuple)) else 0
            self.send_ui_message("SYSTEM", f"Generated {count} insights")
            return {
                "success": True,
                "decision": chosen,
                "confidence": confidence,
                "insights": insights,
                "insight_count": count,
            }
        except Exception as exc:
            self.logger.error("run_autonomous_reasoning failed: %s", exc)
            return {"success": False, "error": str(exc)}

    # ── System status ─────────────────────────────────────────────────────────

    async def get_system_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "is_running": self.is_running,
            "llm_ready": self.llm_ready,
            "llm_shared": SharedLLMEngine.is_ready(),
            "thinking": self.thinking,
            "timestamp": time.time(),
            "listener_ready": self._listener_ready.is_set(),
            "listener_restarts": self._listener_restart_count,
            "components": {
                "content_filter": self._safe_get_stats(self.content_filter),
                "goal_manager": self._safe_get_stats(self.goal_manager),
                "autonomous_core": self._safe_get_stats(self.autonomous_core),
                "curiosity": self._safe_get_stats(self.curiosity),
                "software_builder": self._safe_get_stats(self.software_builder),
            },
        }

        if self.memory_api:
            try:
                mem = (
                    self.memory_api.get_stats()
                    if hasattr(self.memory_api, "get_stats")
                    else {}
                )
                ov = (
                    mem.get("memory_stats", {}).get("overview", {})
                    if isinstance(mem, dict)
                    else {}
                )
                status["memory"] = {
                    "episodic": ov.get("episodic", 0),
                    "semantic": ov.get("semantic", 0),
                    "vector": ov.get("vector", 0),
                }
            except Exception as exc:
                status["memory_error"] = str(exc)
                status["memory"] = {"episodic": 0, "semantic": 0, "vector": 0}

        if self.orchestrator and hasattr(self.orchestrator, "get_status"):
            try:
                status["orchestrator"] = self.orchestrator.get_status()
            except Exception as exc:
                status["orchestrator_error"] = str(exc)

        if self.task_queue_manager:
            try:
                for attr, key in [
                    ("get_metrics", "task_queue"),
                    ("get_all_workers_stats", "workers"),
                ]:
                    if hasattr(self.task_queue_manager, attr):
                        v = getattr(self.task_queue_manager, attr)()
                        status[key] = await v if asyncio.iscoroutine(v) else v
            except Exception as exc:
                status["task_queue_error"] = str(exc)

        if self.rate_limiter:
            try:
                for attr, key in [
                    ("get_status", "rate_limiter"),
                    ("get_metrics", "rate_metrics"),
                ]:
                    if hasattr(self.rate_limiter, attr):
                        v = getattr(self.rate_limiter, attr)()
                        status[key] = await v if asyncio.iscoroutine(v) else v
            except Exception as exc:
                status["rate_limiter_error"] = str(exc)

        status["ui"] = {
            "has_chat_panel": self.chat_panel is not None,
            "has_status_bar": self.status_bar is not None,
            "has_overlay": self.overlay is not None,
        }

        return status

    @staticmethod
    def _safe_get_stats(obj: Any) -> Dict[str, Any]:
        try:
            if obj and hasattr(obj, "get_stats"):
                result = obj.get_stats()
                if asyncio.iscoroutine(result):
                    return {}  # async get_stats can't be awaited in sync context
                return result or {}
        except Exception:
            pass
        return {}

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        self.logger.info("🛑 Shutting down EDIATH system…")
        self.is_running = False

        async def safe_stop(name, obj, method, timeout=5):
            if obj and hasattr(obj, method):
                try:
                    result = getattr(obj, method)()
                    if asyncio.iscoroutine(result):
                        await asyncio.wait_for(result, timeout=timeout)
                    self.logger.info("%s stopped", name)
                except asyncio.TimeoutError:
                    self.logger.warning("%s stop timeout", name)
                except Exception as exc:
                    self.logger.warning("%s stop error: %s", name, exc)

        await safe_stop("Listener", self.listener, "stop")
        await safe_stop("Speaker", self.speaker, "stop")

        if self.rate_limiter:
            try:
                await asyncio.wait_for(shutdown_rate_limiter(), timeout=5)
                self.logger.info("RateLimiter shutdown")
            except Exception as exc:
                self.logger.warning("RateLimiter shutdown error: %s", exc)

        if self.task_queue_manager:
            try:
                await asyncio.wait_for(shutdown_task_queue_system(), timeout=5)
                self.logger.info("TaskQueue shutdown")
            except Exception as exc:
                self.logger.warning("TaskQueue shutdown error: %s", exc)

        if self.orchestrator:
            try:
                await asyncio.wait_for(self.orchestrator.shutdown(), timeout=5)
                self.logger.info("Orchestrator shutdown")
            except Exception as exc:
                self.logger.warning("Orchestrator shutdown error: %s", exc)

        if self.vision_engine_instance:
            try:
                await asyncio.wait_for(
                    self.vision_engine_instance.stop_vision(), timeout=5
                )
                self.logger.info("Vision engine stopped")
            except Exception as exc:
                self.logger.warning("Vision shutdown: %s", exc)

        to_cancel = [t for t in self._tasks if t and not t.done()]
        for t in to_cancel:
            t.cancel()
        if to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*to_cancel, return_exceptions=True), timeout=5
                )
            except asyncio.TimeoutError:
                self.logger.warning("Task cancellation timeout")

        self._tasks.clear()
        self.logger.info("Cancelled %d background task(s)", len(to_cancel))

        if self._backend_loop and not self._backend_loop.is_closed():

            def _cancel_all():
                try:
                    for t in asyncio.all_tasks(self._backend_loop):
                        if not t.done():
                            t.cancel()
                except Exception as exc:
                    self.logger.debug("cancel_all error: %s", exc)

            try:
                self._backend_loop.call_soon_threadsafe(_cancel_all)
            except Exception:
                pass

        try:
            self.send_ui_message("SYSTEM", "Shutdown complete")
        except Exception:
            print("[SYSTEM] Shutdown complete")

        self.logger.info("✅ EDIATH system shutdown complete")

    # ── Brain process ─────────────────────────────────────────────────────────

    async def brain_process(self, input_type: str, data: dict) -> Dict[str, Any]:
        # Allow operation even without LLM (fallback mode)
        if not self.llm_ready:
            return {"success": False, "error": "llm not ready - operating in fallback mode"}

        try:
            await asyncio.wait_for(self._brain_lock.acquire(), timeout=5)
        except asyncio.TimeoutError:
            return {"success": False, "error": "brain busy"}

        try:
            if not isinstance(data, dict):
                return {"success": False, "error": "invalid data"}

            try:
                context = await asyncio.wait_for(
                    self._build_brain_context(input_type, data), timeout=5
                )
            except asyncio.TimeoutError:
                context = "Input processing timeout."

            context = (context or "I could not understand that clearly.").strip()

            try:
                filtered = await asyncio.wait_for(
                    self.filter_and_process(context, learn=False), timeout=3
                )
                if filtered:
                    context = filtered
            except asyncio.TimeoutError:
                pass

            full_context = (
                "You are EDIATH, an AI assistant. "
                "Be concise, clear, and helpful.\n\n" + context
            )

            try:
                async with self.llm_lock:
                    response = await asyncio.wait_for(
                        self.llm_engine.generate(full_context), timeout=12
                    )
            except asyncio.TimeoutError:
                return {"success": False, "error": "llm timeout"}
            except Exception as exc:
                self.logger.error("LLM error: %s", exc)
                return {"success": False, "error": "llm failure"}

            if not response:
                return {"success": False, "error": "empty response"}

            if isinstance(response, dict):
                output = (
                    response.get("response")
                    or response.get("result")
                    or response.get("output")
                    or ""
                )
            else:
                output = str(response)

            output = output.strip() or "No clear response generated."

            return {"success": True, "output": output}

        except Exception as exc:
            self.logger.error("brain_process failed: %s", exc, exc_info=True)
            return {"success": False, "error": "internal error"}

        finally:
            if self._brain_lock.locked():
                try:
                    self._brain_lock.release()
                except RuntimeError:
                    pass

    async def _build_brain_context(self, input_type: str, data: dict) -> str:
        try:
            if not isinstance(data, dict):
                return "Invalid input data."

            if input_type == "voice":
                text = str(data.get("text", "")).strip()
                if not text:
                    return "No voice input detected."

                lower = text.lower()
                if any(
                    kw in lower
                    for kw in ("see me", "camera", "look", "what do you see")
                ):
                    if not self.vision_engine_instance:
                        return "I cannot access vision right now."
                    try:
                        detections = getattr(
                            self.vision_engine_instance, "current_detections", []
                        )
                        caption = getattr(
                            self.vision_engine_instance, "current_caption", None
                        )
                        vr = await asyncio.wait_for(
                            self.decision_engine.process_vision(
                                {"detections": detections, "caption": caption}
                            ),
                            timeout=3,
                        )
                        if vr:
                            return f"Camera view: {vr}"
                        if detections:
                            return f"I see: {', '.join(map(str, detections[:3]))}"
                        return "Camera is active but nothing clear detected."
                    except asyncio.TimeoutError:
                        return "Vision processing timeout."
                    except Exception:
                        return "Vision processing failed."
                return text

            if input_type == "vision":
                try:
                    detections = data.get("detections", [])
                    caption = data.get("caption")
                    vr = await asyncio.wait_for(
                        self.decision_engine.process_vision(
                            {"detections": detections, "caption": caption}
                        ),
                        timeout=3,
                    )
                    return f"Scene: {vr}" if vr else "Scene unclear."
                except asyncio.TimeoutError:
                    return "Vision timeout."
                except Exception:
                    return "Vision error."

            if input_type == "task":
                return str(data.get("task", "")).strip() or "Empty task."

            return str(data)[:500]

        except Exception as exc:
            self.logger.warning("Context build error: %s", exc)
            return "Error understanding input."

    async def wait_until_ready(self, timeout: float = 60.0) -> bool:
        """Wait until system is fully initialized with timeout + safety checks."""
        start_time = time.monotonic()
        deadline = start_time + timeout
        check_interval = 0.5

        while time.monotonic() < deadline:
            try:
                await asyncio.sleep(check_interval)

                if self.is_running:
                    self.logger.info("✅ System ready")
                    return True

                if getattr(self, "thinking", False):
                    if not hasattr(self, "_ready_wait_timer"):
                        self._ready_wait_timer = 0
                    self._ready_wait_timer += check_interval
                    if self._ready_wait_timer > 20:
                        self.logger.warning("⚠ Resetting stuck state during startup")
                        self.thinking = False
                        self._ready_wait_timer = 0
                else:
                    self._ready_wait_timer = 0

            except Exception as exc:
                self.logger.warning("wait_until_ready loop error: %s", exc)

        self.logger.error("❌ System readiness timeout")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Config helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_or_create_config() -> OrchestratorConfig:
    """Load orchestrator config from YAML or create defaults."""
    if CONFIG_PATH.exists():
        try:
            return OrchestratorConfig.from_yaml(CONFIG_PATH)
        except Exception as exc:
            logger.warning("Config load failed (%s), using defaults", exc)

    if _CONFIG_LOADER_AVAILABLE and _config_loader_module:
        try:
            _config_loader_module.load_config("system")
            _config_loader_module.load_config("orchestrator")
        except Exception as exc:
            logger.debug("config_loader optional load: %s", exc)

    config = OrchestratorConfig()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import dataclasses as _dc
        config_dict = _dc.asdict(config)
    except Exception:
        try:
            config_dict = vars(config)
        except Exception:
            config_dict = {}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info("Wrote default config to %s", CONFIG_PATH)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Voice loop
# ─────────────────────────────────────────────────────────────────────────────

async def _run_voice_loop(
    system: EDIATHSystem,
    loop: asyncio.AbstractEventLoop,
) -> None:
    if not system.listener:
        system.logger.warning("No listener — voice loop disabled")
        return

    try:
        # Use configured listener ready timeout for startup to allow slower devices
        started = await asyncio.wait_for(
            asyncio.to_thread(system._start_listener), timeout=LISTENER_READY_TIMEOUT
        )
    except asyncio.TimeoutError:
        system.logger.warning("Listener start timeout -- voice disabled, continuing")
        return

    if not started:
        system.logger.warning("Could not start listener -- voice disabled")
        return

    try:
        ready = await asyncio.wait_for(
            asyncio.to_thread(system._listener_ready.wait, LISTENER_READY_TIMEOUT),
            timeout=LISTENER_READY_TIMEOUT + 2,
        )
    except asyncio.TimeoutError:
        system.logger.warning("Listener readiness timeout -- voice disabled")
        return

    if not ready:
        system.logger.error("Listener device did not open in time")
        return

    system.logger.info("🎤 Voice loop active")
    pending: List[asyncio.Task] = []
    max_tasks = 5

    # Activate wakeword detection
    system.start_wakeword()

    while system.is_running:
        try:
            text = system._get_listener_text()
            if text:
                display = text.strip()
                clean = display.lower()

                if len(clean) > 2:
                    system.send_ui_message("YOU (voice)", display)

                    if len(pending) >= max_tasks:
                        pending = [t for t in pending if not t.done()]
                        await asyncio.sleep(0.1)
                        continue

                    async def _handle(t: str = clean):
                        # Persist voice transcript as episodic memory
                        if system.memory_api is not None:
                            try:
                                memory_key = f"voice_{int(time.time()*1000)}"
                                await system.memory_api.store_event(
                                    key=memory_key,
                                    event={
                                        "input_type": "voice",
                                        "transcript": t,
                                        "session": "default",
                                    },
                                    importance=0.6,
                                    tags=["voice", "listening"],
                                    context={"source": "voice_loop"},
                                )
                            except Exception as _mem_exc:
                                system.logger.debug("voice memory store failed: %s", _mem_exc)

                        try:
                            res = await asyncio.wait_for(
                                system.brain_process("voice", {"text": t}), timeout=12
                            )
                            if res.get("success"):
                                out = res["output"]
                                system.send_ui_message("AI", out)
                                if system.speaker:
                                    try:
                                        asyncio.create_task(system.speaker.speak(out))
                                    except Exception:
                                        pass
                        except asyncio.TimeoutError:
                            system.logger.warning("Voice processing timeout")
                        except Exception as exc:
                            system.logger.error("Voice handle error: %s", exc)

                    task = loop.create_task(_handle())
                    pending.append(task)
                    pending = [t for t in pending if not t.done()]

        except Exception as exc:
            system.logger.error("Voice poll error: %s", exc)
            try:
                restarted = await asyncio.wait_for(
                    asyncio.to_thread(system._restart_listener), timeout=5
                )
            except asyncio.TimeoutError:
                restarted = False

            if not restarted:
                system.logger.error("Voice loop terminating — listener unrecoverable")
                break

        await asyncio.sleep(LISTENER_POLL_INTERVAL)

    for t in pending:
        if not t.done():
            t.cancel()

    system.logger.info("🎤 Voice loop stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────

def run_ui_only() -> None:
    """Run a minimal Kivy UI with no backend."""
    try:
        from kivy.app import App
        from kivy.uix.label import Label
        from kivy.uix.boxlayout import BoxLayout

        class SimpleUI(App):
            def build(self):
                layout = BoxLayout(orientation="vertical")
                layout.add_widget(Label(text="EDIATH AI", font_size="24sp"))
                layout.add_widget(
                    Label(text="UI Mode — Backend not loaded", font_size="16sp")
                )
                return layout

        SimpleUI().run()
    except Exception as exc:
        logger.error("UI failed: %s", exc)


async def run_backend_only() -> None:
    """Run only the async backend with stability, restart, and timeout protection."""
    global _SYSTEM_INSTANCE

    try:
        config = _load_or_create_config()
        system = EDIATHSystem(orchestrator=EDIATHOrchestrator(config))
        _SYSTEM_INSTANCE = system

        try:
            # Increased timeout for slow model loading
            ok = await asyncio.wait_for(system.initialize(), timeout=120)
            if not ok:
                logger.error("❌ Backend initialization failed")
                return
        except asyncio.TimeoutError:
            logger.error("❌ Initialization timeout (>120s)")
            return

        loop = asyncio.get_running_loop()
        system.set_backend_loop(loop)

        if system.listener:
            try:
                task = loop.create_task(_run_voice_loop(system, loop))
                system._tasks.append(task)
            except Exception as exc:
                system.logger.warning("Voice loop failed: %s", exc)

        system.logger.info("✅ Backend running (Shared LLM Mode)")
        restart_attempts = 0

        while system.is_running:
            try:
                await asyncio.sleep(1)

                if getattr(system, "thinking", False):
                    if not hasattr(system, "_thinking_timer"):
                        system._thinking_timer = 0
                    system._thinking_timer += 1
                    if system._thinking_timer > 30:
                        system.logger.warning("⚠ Resetting stuck thinking state")
                        system.thinking = False
                        system._thinking_timer = 0
                else:
                    system._thinking_timer = 0

            except Exception as exc:
                system.logger.error("Loop error: %s", exc)
                restart_attempts += 1
                if restart_attempts > 3:
                    system.logger.error("❌ Too many errors, stopping backend")
                    break
                await asyncio.sleep(2)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("🛑 Shutdown requested")
    except Exception as exc:
        logger.error("💥 Fatal backend error: %s", exc)
    finally:
        try:
            if _SYSTEM_INSTANCE:
                await asyncio.wait_for(_SYSTEM_INSTANCE.shutdown(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("⚠ Shutdown timeout, forcing exit")
        except Exception as exc:
            logger.error("Shutdown error: %s", exc)
        logger.info("👋 Backend stopped")


def main_interactive() -> None:
    """Stable UI + backend runner with shared LLM injection."""
    global _SYSTEM_INSTANCE

    from kivy.app import App
    from kivy.core.window import Window
    from kivy.uix.screenmanager import ScreenManager
    from kivy.clock import Clock

    config = _load_or_create_config()
    system = EDIATHSystem(orchestrator=EDIATHOrchestrator(config))
    _SYSTEM_INSTANCE = system
    system.is_running = True

    from core.ui.ai_backend import AIBackend

    ui_backend = AIBackend()
    ui_backend.start()

    init_success = False
    init_error = None
    init_done = threading.Event()
    backend_loop = None

    # ─────────────────────────────────────────────────────────────────────────
    # BACKEND THREAD
    # FIX 1: Raised system.initialize() timeout 90 → 120 s for slow model loading
    # FIX 2: Use ui_backend.connect_system() (the proper API)
    # FIX 3: Signal init_done BEFORE entering the keepalive loop
    # FIX 4: Partial-connect fallback — if init times out but agent is alive
    # ─────────────────────────────────────────────────────────────────────────
    def init_in_thread():
        nonlocal init_success, init_error, backend_loop

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        backend_loop = loop
        system.set_backend_loop(loop)

        try:
            # ── INIT (generous timeout) ───────────────────────────────────────
            try:
                init_success = loop.run_until_complete(
                    asyncio.wait_for(system.initialize(), timeout=120)  # was 90, now 120
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "⚠ system.initialize() exceeded 120 s — attempting partial connect"
                )
                # Partial success: connect if agent is alive so UI works
                if system.agent:
                    system.is_running = True
                    init_success = True
                    logger.info("Partial connect — agent available, continuing")
                else:
                    init_success = False
                    init_error = "Initialization timed out and no agent available"

            # ── CONNECT BACKEND → UI ──────────────────────────────────────────
            if init_success:
                # Use the documented API, not private attributes
                # Connect after the backend loop is alive.
                # If connect_system detects loop not running, it will fall back safely.
                ui_backend.connect_system(system, loop)

                # Voice loop intentionally disabled to prevent startup crashes
                # Voice features can be re-enabled once stable
                if system.listener:
                    system.logger.info("Voice listener present but disabled for stability")


        except Exception as exc:
            init_error = exc
            init_success = False
            logger.error("💥 Backend crash: \n%s", traceback.format_exc())

        finally:
            # ── SIGNAL UI (CRITICAL FIX) ─────────────────────────────────────
            # Must happen BEFORE keepalive so check_ready() in the Kivy thread
            # is unblocked while the system is still alive.
            init_done.set()

        # ── KEEP ALIVE (runs after UI has been notified) ──────────────────────
        if init_success:

            async def keepalive():
                while system.is_running:
                    await asyncio.sleep(1)
                    if getattr(system, "thinking", False):
                        if not hasattr(system, "_thinking_timer"):
                            system._thinking_timer = 0
                        system._thinking_timer += 1
                        if system._thinking_timer > 30:
                            system.logger.warning("Reset stuck thinking state")
                            system.thinking = False
                            system._thinking_timer = 0
                    else:
                        system._thinking_timer = 0

            try:
                loop.run_until_complete(keepalive())
            except Exception as exc:
                logger.warning("Keepalive exited: %s", exc)

        # ── CLEAN SHUTDOWN ────────────────────────────────────────────────────
        try:
            loop.run_until_complete(system.shutdown())
        except Exception:
            pass

        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass

        loop.close()

    # ── UI IMPORT ─────────────────────────────────────────────────────────────
    try:
        from core.ui.dashboard_screen import DashboardScreen
    except ImportError as exc:
        logger.error("❌ UI load failed: %s", exc)
        system.is_running = False
        return

    threading.Thread(target=init_in_thread, daemon=True).start()

    # ── KIVY APP ──────────────────────────────────────────────────────────────
    class EDIATHKivyApp(App):
        def __init__(self, backend=None, **kwargs):
            super().__init__(**kwargs)
            self.backend = backend

        def build(self):
            Window.title = "EDIATH AI (Shared LLM Mode)"
            Window.size = (1280, 800)
            sm = ScreenManager()
            dashboard = DashboardScreen(name="dashboard")
            dashboard.controller = self.backend
            sm.add_widget(dashboard)
            return sm

        def on_start(self):
            def check_ready(dt):
                if not init_done.is_set():
                    Clock.schedule_once(check_ready, 0.5)
                    return

                # Dismiss loading overlay when backend ready
                try:
                    screen = getattr(self.root, "current_screen", None)
                    if screen and hasattr(screen, "hide_loading_overlay"):
                        screen.hide_loading_overlay()
                except Exception:
                    pass

                if init_success:
                    ui_backend.set_response_callback(self._on_ai_response)
                    ui_backend.set_status_callback(self._on_status_update)
                    logger.info("✅ Backend connected (Shared LLM enabled)")

                    # Emit ready message to chat
                    def send_welcome(dt):
                        try:
                            screen = getattr(self.root, "current_screen", None)
                            if screen:
                                chat = getattr(screen, "chat_screen", None)
                                if chat and hasattr(chat, "add_message"):
                                    chat.add_message("SYSTEM", "✅ EDIATH AI Ready! Type a message or speak...")
                                else:
                                    logger.warning("send_welcome: chat_screen not found on screen")
                        except Exception as exc:
                            logger.warning("send_welcome failed: %s", exc)
                    Clock.schedule_once(send_welcome, 0.3)
                else:
                    logger.error("❌ Backend failed: %s", init_error)

            Clock.schedule_once(check_ready, 0.5)

        def _on_ai_response(self, text: str):
            try:
                screen = getattr(self.root, "current_screen", None)
                if not screen:
                    return
                chat = getattr(screen, "chat_panel", None)
                if chat:
                    chat.add_message("AI", text)
                bar = getattr(screen, "status_bar", None)
                if bar:
                    bar.update_status(text[:50])
            except Exception as exc:
                logger.error("UI response error: %s", exc)

        def _on_status_update(self, text: str):
            try:
                screen = getattr(self.root, "current_screen", None)
                if not screen:
                    return
                bar = getattr(screen, "status_bar", None)
                if bar:
                    bar.update_status(text[:50])
            except Exception:
                pass

        def on_stop(self):
            system.is_running = False
            ui_backend.shutdown()
            logger.info("🛑 UI stopping")

    app = EDIATHKivyApp(backend=ui_backend)
    app.run()

    system.is_running = False
    if init_done.wait(timeout=5) is False:
        logger.warning("⚠ Backend did not shut down cleanly")
    logger.info("👋 System shutdown complete")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDIATH AI System with Shared LLM")
    parser.add_argument(
        "--mode",
        choices=["ui", "backend", "ui-only"],
        default="ui",
        help="Run mode: 'ui' (full), 'backend' (headless), 'ui-only' (no backend)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EDIATH AI - SINGLETON LLM MODE")
    print("=" * 70)
    print("✓ LLM Engine will be loaded ONLY ONCE")
    print("✓ ALL agents will share the SAME LLM instance")
    print("✓ System load optimized (no duplicate models)")
    print("=" * 70)
    print()

    if args.mode == "ui-only":
        run_ui_only()
    elif args.mode == "backend":
        try:
            asyncio.run(run_backend_only())
        except KeyboardInterrupt:
            print("\nInterrupted by user")
    else:
        main_interactive()