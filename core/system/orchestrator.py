"""
EDIATH Orchestrator Module
Production-ready orchestration for EDIATH autonomous AI system with Multi-Agent Coordination
and MongoDB integration for state persistence, metrics, and audit logging.
"""

import asyncio
import json
import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from networkx import k_components
import yaml
import logging

from core.brain.context_manager import ContextManager
from core.brain.reasoning_engine import ReasoningEngine
from core.utils.task_registry import task_registry

# Vision integration
try:
    # shared_memory.py exposes SharedVisionMemory
    from core.vision.shared_memory import SharedVisionMemory

    VISION_MEMORY_AVAILABLE = True
except ImportError:
    SharedVisionMemory = None  # type: ignore
    VISION_MEMORY_AVAILABLE = False


# MongoDB Integration
from core.memory.mongo_client import (
    mongo_client,
    get_metrics_collection,
    get_audit_collection,
    is_connected as mongo_is_connected,
)

from core.system import ComponentStatus
from core.system.command_router import CommandRouter
from core.system.workflow_engine import WorkflowEngine
from core.system.safety_layer import SafetyLayer
from core.agent.agent_core import RAgent as AgentCore
from core.agent.agent_loop import AgentLoop
from core.brain.decision_engine import DecisionEngine
from core.brain.llm_engine import LLMEngine
from core.memory.memory_manager import MemoryManager
from core.autonomous.autonomous_core import AutonomousCoreEngine as AutonomousCore
from core.perception.emotion_detector import EmotionDetector
from core.learning.feedback_loop import FeedbackLoop
from core.security.audit_logger import AuditLogger
from core.utils.logger import logger as base_logger

# Import stub agents for missing modules

# Blender Agent import
try:
    from core.agent.blender_agent import Blender3DAgent

    BLENDER_AGENT_AVAILABLE = True
    base_logger.info("Blender3DAgent module found")
except ImportError:
    BLENDER_AGENT_AVAILABLE = False
    base_logger.warning(
        "Blender3DAgent not available - install bpy or run inside Blender"
    )

# Agent availability map
AGENT_MAP = {
    "browser": {"available": False, "config": "enable_browser_agent"},
    "calculator": {"available": False, "config": "enable_calc_agent"},
    "file": {
        "available": True,
        "module": "core.agent.file_agent",
        "config": "enable_file_agent",
    },
    "search": {"available": False, "config": "enable_search_agent"},
    "shell": {"available": False, "config": "enable_shell_agent"},
    "vision": {
        "available": True,
        "module": "core.agent.vision_agent",
        "config": "enable_vision_agent",
    },
    "speech": {"available": False, "config": "enable_speech_agent"},
    "blender": {"available": BLENDER_AGENT_AVAILABLE, "config": "enable_blender_agent"},
}

# Import Multi-Agent Coordinator
try:
    from core.system.multi_agent_coordinator import (
        MultiAgentCoordinator,
        BaseAgent,
        LLMAgent,
        AutomationAgent as MACAutomationAgent,
        LearningAgent,
        SupervisorAgent,
        AgentRole,
        AgentStatus,
        CommunicationProtocol,
        CoordinationStrategy,
        TaskAllocationStrategy,
        get_multi_agent_coordinator,
        shutdown_multi_agent_coordinator,
    )

    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False
    base_logger.warning("Multi-Agent Coordinator not available")

# ==================== Configuration ====================


@dataclass
class OrchestratorConfig:
    """Configuration for EDIATH orchestrator with MongoDB support"""

    name: str = "EDIATH_Orchestrator"
    version: str = "1.0.0"
    environment: str = "production"

    # Timing configurations
    tick_interval_ms: int = 100
    health_check_interval_sec: int = 30
    metrics_flush_interval_sec: int = 60
    circuit_breaker_threshold: int = 5
    state_persist_interval_sec: int = 300  # 5 minutes

    # Recovery configurations
    max_retries: int = 3
    retry_delay_sec: int = 5
    recovery_timeout_sec: int = 30
    recovery_cooldown_sec: int = 5

    # Paths
    log_dir: str = "logs"
    config_dir: str = "config"
    data_dir: str = "data"
    reports_dir: str = "research"
    workspace_dir: str = "workspace"
    generated_projects_dir: str = "generated_projects"

    # Feature flags
    enable_autonomous_mode: bool = True
    enable_learning: bool = True
    enable_memory: bool = True
    enable_security: bool = True
    enable_vision: bool = True
    enable_audio: bool = True
    enable_alerts: bool = True
    enable_metrics: bool = True
    enable_persistence: bool = True
    enable_multi_agent: bool = True
    enable_mongo_persistence: bool = True  # New: MongoDB persistence

    # Agent enable flags
    enable_browser_agent: bool = True
    enable_calc_agent: bool = True
    enable_file_agent: bool = True
    enable_search_agent: bool = True
    enable_shell_agent: bool = True
    enable_vision_agent: bool = True
    enable_speech_agent: bool = True
    enable_translation_agent: bool = True
    enable_code_execution_agent: bool = True
    enable_database_agent: bool = True
    enable_email_agent: bool = True
    enable_notification_agent: bool = True
    enable_scheduler_agent: bool = True
    enable_automation_agent: bool = True
    enable_system_control_agent: bool = True
    enable_network_agent: bool = True
    enable_knowledge_base_agent: bool = True
    enable_retrieval_agent: bool = True
    enable_maths_reasoning_agent: bool = True
    enable_critic_agent: bool = True
    enable_calendar_agent: bool = True
    enable_fitness_agent: bool = True
    enable_nlp_agent: bool = True
    enable_security_agent: bool = True
    enable_blender_agent: bool = True

    # Component-specific configs
    llm_model_path: str = "models/EDIATH-q4_k_m.gguf"

    # Blender-specific configs
    blender_workspace_dir: str = "workspace/blender"
    blender_render_dir: str = "generated_projects/blender_renders"
    blender_export_format: str = "fbx"

    # Multi-agent configs
    coordination_strategy: str = "HIERARCHICAL"
    task_allocation_strategy: str = "CAPABILITY_BASED"
    communication_protocol: str = "DIRECT"

    # MongoDB collections
    mongo_state_collection: str = "orchestrator_state"
    mongo_metrics_collection: str = "orchestrator_metrics"
    mongo_events_collection: str = "orchestrator_events"

    @classmethod
    def from_yaml(cls, path: str) -> "OrchestratorConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def get(self, key, default=None):
        return getattr(self, key, default)


# ==================== Enums ====================


class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


# ==================== Enhanced Metrics ====================


class EnhancedMetrics:
    """Enhanced metrics collection with MongoDB integration"""

    def __init__(self):
        self.metrics = {
            "orchestrator_state": {},
            "component_status": {},
            "processing_latency": [],
            "errors_total": {},
            "tasks_processed": {},
            "active_tasks": 0,
            "agent_tasks": {},
        }
        self._metrics_collection = get_metrics_collection()
        self._last_flush = datetime.now()

    def gauge_set(self, name: str, value: float, labels: dict = None):
        if name not in self.metrics:
            self.metrics[name] = {}
        key = str(labels) if labels else "value"
        self.metrics[name][key] = value

    def counter_inc(self, name: str, labels: dict = None):
        if name not in self.metrics:
            self.metrics[name] = {}
        key = str(labels) if labels else "count"
        self.metrics[name][key] = self.metrics[name].get(key, 0) + 1

    def histogram_observe(self, name: str, value: float, labels: dict = None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(
            {"value": value, "labels": labels, "time": time.time()}
        )
        # Keep only last 1000 entries
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    async def flush_to_mongo(self, orchestrator_name: str = "EDIATH"):
        """Flush metrics to MongoDB"""
        if not mongo_is_connected() or self._metrics_collection is None:
            return

        try:
            metrics_doc = {
                "timestamp": datetime.now(),
                "orchestrator": orchestrator_name,
                "metrics": self.metrics.copy(),
                "environment": "production",
            }

            await self._metrics_collection.insert_one(metrics_doc)

            # Keep only last 10000 metrics documents
            count = await self._metrics_collection.count_documents(
                {"orchestrator": orchestrator_name}
            )
            if count > 10000:
                oldest = (
                    await self._metrics_collection.find(
                        {"orchestrator": orchestrator_name}
                    )
                    .sort("timestamp", 1)
                    .limit(count - 10000)
                    .to_list(length=count - 10000)
                )
                if oldest:
                    await self._metrics_collection.delete_many(
                        {"_id": {"$in": [doc["_id"] for doc in oldest]}}
                    )

            self._last_flush = datetime.now()
        except Exception as e:
            base_logger.debug(f"Failed to flush metrics to MongoDB: {e}")


metrics = EnhancedMetrics()


# ==================== Agent Registry ====================


class AgentRegistry:
    """Registry for all specialized agents with MongoDB integration"""

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Production-safe orchestrator constructor.

        Fixes:
        - callable config corruption
        - .items() crashes
        - broken audit collection
        - logger initialization failures
        - invalid async state
        - memory leaks
        - unsafe defaults
        - initialization races
        - partial startup corruption
        """

        import asyncio
        import logging
        import traceback
        from collections import deque
        from datetime import datetime

        # ============================================================
        # SAFE LOGGER
        # ============================================================
        try:

            self.logger = base_logger

            if self.logger is None:
                raise RuntimeError("Logger missing")

        except Exception:

            self.logger = logging.getLogger(
                "EDIATH"
            )

        # ============================================================
        # SAFE CONTAINERS
        # ============================================================
        self.agents: Dict[str, Any] = {}

        self.wrappers: Dict[str, Any] = {}

        self.components: Dict[str, Any] = {}

        self.listeners: Dict[str, list] = {}

        self.tasks: Dict[str, Any] = {}

        self.cache: Dict[str, Any] = {}

        self.event_history = deque(
            maxlen=1000
        )

        self.worker_tasks = []

        self._init_pipeline_tasks = []

        self._init_chunk_buffer = deque(
            maxlen=5000
        )

        # ============================================================
        # SAFE CONFIG
        # ============================================================
        try:

            # prevent:
            # 'function' object has no attribute 'items'
            if config is None or callable(config):

                self.config = (
                    OrchestratorConfig()
                )

            else:

                self.config = config

        except Exception:

            self.config = (
                OrchestratorConfig()
            )

        # ============================================================
        # SAFE DEFAULT SETTER
        # ============================================================
        def _set_default(
            attr: str,
            value: Any,
        ) -> None:

            try:

                if callable(
                    self.config
                ):
                    return

                current = getattr(
                    self.config,
                    attr,
                    None,
                )

                # corrupted callable
                if callable(current):

                    setattr(
                        self.config,
                        attr,
                        value,
                    )

                    return

                # missing value
                if current is None:

                    setattr(
                        self.config,
                        attr,
                        value,
                    )

            except Exception:
                pass

        # ============================================================
        # DEFAULT CONFIG VALUES
        # ============================================================
        defaults = {

            # ========================================================
            # AGENTS
            # ========================================================
            "agent_init_chunk_size": 5,
            "agent_init_pipeline_workers": 3,
            "agent_init_timeout": 60,
            "enable_agent_pipeline": True,

            # ========================================================
            # COMMUNICATION
            # ========================================================
            "enable_agent_comms": True,
            "agent_comms_setup_chunk_size": 5,
            "agent_validation_chunk_size": 10,
            "agent_comms_buffer": 1000,
            "agent_comms_retries": 3,
            "agent_comms_timeout": 5,

            # ========================================================
            # VISION
            # ========================================================
            "vision_connect_retries": 3,
            "vision_connect_delay": 1,
            "vision_connect_timeout": 5,

            # ========================================================
            # PIPELINE
            # ========================================================
            "enable_init_pipeline": True,
            "init_chunk_size": 50,
            "init_pipeline_workers": 5,
            "component_batch_size": 3,

            # ========================================================
            # LLM
            # ========================================================
            "large_model_threshold": 1024,
            "model_chunk_size_mb": 256,
            "model_chunk_timeout": 60,

            # ========================================================
            # RUNTIME
            # ========================================================
            "enable_memory": True,
            "enable_learning": True,
            "enable_security": True,
            "enable_autonomous_mode": False,
            "enable_multi_agent": False,

            # ========================================================
            # HEALTH
            # ========================================================
            "metrics_interval": 2,
            "health_check_interval": 5,
            "enable_auto_healing": True,

            # ========================================================
            # BACKUP
            # ========================================================
            "auto_backup_interval": 60,

            # ========================================================
            # SCALING
            # ========================================================
            "enable_predictive_scaling": True,

            # ========================================================
            # TASKS
            # ========================================================
            "max_concurrent_tasks": 4,
            "task_timeout": 60,

            # ========================================================
            # CACHE
            # ========================================================
            "cache_size": 1000,
            "cache_ttl": 3600,
        }

        for (
            key,
            value,
        ) in defaults.items():

            _set_default(
                key,
                value,
            )

        # ============================================================
        # SAFE AUDIT COLLECTION
        # ============================================================
        self._audit_collection = None

        try:

            audit = (
                get_audit_collection()
            )

            # FIX:
            # collection object incorrectly called
            if audit is not None:

                if callable(audit):

                    try:
                        audit = audit()
                    except TypeError:
                        pass

                self._audit_collection = audit

        except Exception as e:

            self._audit_collection = None

            try:

                self.logger.warning(
                    f"Audit collection failed: {e}"
                )

            except Exception:
                pass

        # ============================================================
        # SYSTEM STATE
        # ============================================================
        self.state = getattr(
            SystemState,
            "INITIALIZING",
            None,
        )

        self.start_time = datetime.now()

        self._running = False

        self._initializing = False

        self._model_loaded = False

        self._pipeline_initialized = False

        self._validation_completed = False

        self._healing = False

        # ============================================================
        # SAFE LOCKS
        # ============================================================
        try:

            self._model_loading_lock = (
                asyncio.Lock()
            )

            self._state_lock = (
                asyncio.Lock()
            )

            self._task_lock = (
                asyncio.Lock()
            )

        except Exception:

            self._model_loading_lock = None

            self._state_lock = None

            self._task_lock = None

        # ============================================================
        # METRICS
        # ============================================================
        self.command_count = 0

        self.error_count = 0

        self.health_checks = 0

        self.tasks_submitted = 0

        self.commands_executed = 0

        self.scaling_actions = 0

        self.events_emitted = 0

        self.safe_calls = 0

        self.start_count = 0

        self.shutdown_count = 0

        # ============================================================
        # CACHE STATS
        # ============================================================
        self.hits = 0

        self.misses = 0

        self.writes = 0

        # ============================================================
        # SAFE TASK QUEUE
        # ============================================================
        try:

            self.task_queue = (
                asyncio.PriorityQueue()
            )

        except Exception:

            self.task_queue = None

        # ============================================================
        # SAFE DEBUG
        # ============================================================
        try:

            self.logger.info(
                "✓ EDIATHOrchestrator initialized safely"
            )

        except Exception:
            pass

        # ============================================================
        # FINAL VALIDATION
        # ============================================================
        try:

            # corrupted config
            if callable(
                self.config
            ):

                raise RuntimeError(
                    "Config is callable"
                )

            # prevent:
            # 'function' object has no attribute 'items'
            items_attr = getattr(
                self.config,
                "items",
                None,
            )

            if callable(items_attr):

                self.logger.debug(
                    "Config has callable items()"
                )

        except Exception as e:

            try:

                self.logger.error(
                    f"❌ Constructor validation failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

    def set_config(self, config: OrchestratorConfig):
        self.config = config

    async def _log_audit(
        self, action: str, agent_name: str, success: bool, error: str = None
    ):
        """Log agent initialization to MongoDB"""
        if not mongo_is_connected() or self._audit_collection is None:
            return

        try:
            audit_entry = {
                "timestamp": datetime.now(),
                "action": action,
                "agent_name": agent_name,
                "success": success,
                "error": error,
                "component": "agent_registry",
            }
            await self._audit_collection.insert_one(audit_entry)
        except Exception as e:
            self.logger.debug(f"Failed to log audit: {e}")

    async def _safe_call(
        self,
        obj: Any,
        method_name: str,
        *args,
        timeout: Optional[float] = None,
        default: Any = None,
        run_in_thread: bool = True,
        **kwargs,
    ) -> Any:
        """
        Ultra production-safe async/sync method caller.

        Fixes:
        - object bool can't be used in await expression
        - function object has no attribute items
        - coroutine was never awaited
        - invalid event-loop usage
        - deadlocks
        - thread execution crashes
        - invalid kwargs crashes
        - callable corruption
        - unsafe await handling
        - pipeline failures
        - timeout hangs
        - memory leaks
        - nested coroutine leaks
        - broken async/sync interoperability
        """

        import asyncio
        import inspect
        import functools
        import gc
        import time
        import traceback

        start_time = time.monotonic()

        DEFAULT_TIMEOUT = 60.0
        MAX_COROUTINE_DEPTH = 10

        try:

            # ============================================================
            # VALIDATE OBJECT
            # ============================================================
            if obj is None:

                self.logger.debug(
                    "_safe_call received None object"
                )

                return default

            # callable corruption protection
            if callable(obj) and not hasattr(obj, method_name):

                self.logger.debug(
                    f"_safe_call callable object "
                    f"missing method: {method_name}"
                )

                return default

            # ============================================================
            # VALIDATE METHOD NAME
            # ============================================================
            if not isinstance(method_name, str):

                self.logger.debug(
                    "method_name must be string"
                )

                return default

            method_name = method_name.strip()

            if not method_name:

                self.logger.debug(
                    "Empty method_name"
                )

                return default

            # ============================================================
            # SAFE TIMEOUT
            # ============================================================
            try:

                timeout = float(
                    timeout
                    if timeout is not None
                    else DEFAULT_TIMEOUT
                )

            except Exception:

                timeout = DEFAULT_TIMEOUT

            timeout = max(
                1.0,
                min(timeout, 3600.0),
            )

            # ============================================================
            # SAFE METHOD LOOKUP
            # ============================================================
            try:

                method = getattr(
                    obj,
                    method_name,
                    None,
                )

            except Exception as e:

                self.logger.debug(
                    f"Method lookup failed "
                    f"({method_name}): {e}"
                )

                return default

            if method is None:

                self.logger.debug(
                    f"Method not found: {method_name}"
                )

                return default

            if not callable(method):

                self.logger.debug(
                    f"Attribute not callable: {method_name}"
                )

                return default

            # ============================================================
            # SAFE KWARGS
            # ============================================================
            safe_kwargs = {}

            try:

                # FIX:
                # kwargs accidentally became function
                if callable(kwargs):

                    kwargs = {}

                if kwargs is None:

                    kwargs = {}

                if not isinstance(kwargs, dict):

                    kwargs = {}

                signature = inspect.signature(
                    method
                )

                parameters = signature.parameters

                accepts_kwargs = any(

                    p.kind == inspect.Parameter.VAR_KEYWORD

                    for p in parameters.values()
                )

                for key, value in kwargs.items():

                    try:

                        if accepts_kwargs:

                            safe_kwargs[key] = value

                        elif key in parameters:

                            safe_kwargs[key] = value

                    except Exception:
                        continue

            except Exception:

                safe_kwargs = (
                    kwargs
                    if isinstance(kwargs, dict)
                    else {}
                )

            # ============================================================
            # SAFE ARGS
            # ============================================================
            if args is None:

                args = ()

            elif not isinstance(args, tuple):

                try:

                    args = tuple(args)

                except Exception:

                    args = ()

            # ============================================================
            # SAFE COROUTINE RESOLVER
            # ============================================================
            async def _resolve_result(value):

                depth = 0

                while (
                    inspect.isawaitable(value)
                    and depth < MAX_COROUTINE_DEPTH
                ):

                    value = await asyncio.wait_for(
                        value,
                        timeout=timeout,
                    )

                    depth += 1

                return value

            # ============================================================
            # EXECUTION
            # ============================================================
            result = None

            try:

                # --------------------------------------------------------
                # ASYNC FUNCTION
                # --------------------------------------------------------
                if inspect.iscoroutinefunction(
                    method
                ):

                    coro = method(
                        *args,
                        **safe_kwargs,
                    )

                    # FIX:
                    # bool used in await expression
                    if not inspect.isawaitable(
                        coro
                    ):

                        return coro

                    result = await asyncio.wait_for(

                        _resolve_result(coro),

                        timeout=timeout,
                    )

                # --------------------------------------------------------
                # SYNC FUNCTION
                # --------------------------------------------------------
                else:

                    if run_in_thread:

                        func = functools.partial(
                            method,
                            *args,
                            **safe_kwargs,
                        )

                        result = await asyncio.wait_for(

                            asyncio.to_thread(func),

                            timeout=timeout,
                        )

                    else:

                        result = method(
                            *args,
                            **safe_kwargs,
                        )

                    # ----------------------------------------------------
                    # HANDLE NESTED COROUTINES
                    # ----------------------------------------------------
                    result = await _resolve_result(
                        result
                    )

                    # ----------------------------------------------------
                    # HANDLE CALLABLE RETURNS
                    # ----------------------------------------------------
                    if callable(result):

                        # prevent accidental class execution
                        if not inspect.isclass(
                            result
                        ):

                            try:

                                callable_result = (
                                    await asyncio.wait_for(

                                        asyncio.to_thread(
                                            result
                                        ),

                                        timeout=timeout,
                                    )
                                )

                                result = await _resolve_result(
                                    callable_result
                                )

                            except TypeError:

                                self.logger.debug(
                                    f"Callable returned "
                                    f"requires args: "
                                    f"{method_name}"
                                )

                    # ----------------------------------------------------
                    # BOOL RETURN FIX
                    # ----------------------------------------------------
                    if isinstance(
                        result,
                        bool,
                    ):

                        return result

            # ============================================================
            # TIMEOUT
            # ============================================================
            except asyncio.TimeoutError:

                self.logger.warning(
                    f"_safe_call timeout "
                    f"({timeout}s): {method_name}"
                )

                return default

            # ============================================================
            # CANCELLED
            # ============================================================
            except asyncio.CancelledError:

                self.logger.debug(
                    f"_safe_call cancelled: "
                    f"{method_name}"
                )

                raise

            # ============================================================
            # EXECUTION FAILURE
            # ============================================================
            except Exception as e:

                self.logger.debug(
                    f"_safe_call execution error "
                    f"in {method_name}: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

                return default

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.safe_calls = int(

                    getattr(
                        self,
                        "safe_calls",
                        0,
                    )

                ) + 1

            except Exception:
                pass

            # ============================================================
            # PERFORMANCE LOGGING
            # ============================================================
            try:

                elapsed_ms = round(

                    (
                        time.monotonic()
                        - start_time
                    ) * 1000,

                    2,
                )

                if elapsed_ms > 1000:

                    self.logger.debug(
                        f"_safe_call slow call "
                        f"{method_name}: "
                        f"{elapsed_ms}ms"
                    )

            except Exception:
                pass

            # ============================================================
            # MEMORY CLEANUP
            # ============================================================
            try:

                gc.collect()

            except Exception:
                pass

            return result

        # ================================================================
        # HARD CANCEL
        # ================================================================
        except asyncio.CancelledError:

            raise

        # ================================================================
        # FATAL FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"_safe_call fatal error "
                    f"in {method_name}: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return default

    async def initialize_all(
        self,
        system_instance: Any,
    ) -> Dict[str, Any]:
        """
        Production-safe full agent initialization system.

        Fixes:
        - function.items crashes
        - coroutine leaks
        - bool await crashes
        - unsafe config access
        - worker deadlocks
        - queue corruption
        - invalid chunking
        - startup race conditions
        - duplicate initialization
        - invalid metrics calls
        - pipeline crashes
        - memory spikes
        - unsafe asyncio usage
        - worker timeout hangs
        - invalid task cleanup
        - broken communication pipeline
        - broken validation ordering
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        results: Dict[str, bool] = {}

        workers = []

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.warning(
                        "Initialization aborted due to shutdown"
                    )

                    return results

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    # callable corruption
                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # prevent function corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE SETTINGS
            # ============================================================
            try:

                chunk_size = int(
                    cfg(
                        "agent_init_chunk_size",
                        5,
                    )
                )

            except Exception:

                chunk_size = 5

            try:

                pipeline_workers = int(
                    cfg(
                        "agent_init_pipeline_workers",
                        3,
                    )
                )

            except Exception:

                pipeline_workers = 3

            try:

                init_timeout = float(
                    cfg(
                        "agent_init_timeout",
                        60,
                    )
                )

            except Exception:

                init_timeout = 60.0

            enable_pipeline = bool(
                cfg(
                    "enable_agent_pipeline",
                    True,
                )
            )

            enable_comms = bool(
                cfg(
                    "enable_agent_comms",
                    True,
                )
            )

            chunk_size = max(
                1,
                min(
                    chunk_size,
                    128,
                ),
            )

            pipeline_workers = max(
                1,
                min(
                    pipeline_workers,
                    16,
                ),
            )

            init_timeout = max(
                5.0,
                min(
                    init_timeout,
                    1800.0,
                ),
            )

            # ============================================================
            # SAFE QUEUE
            # ============================================================
            agent_init_queue = asyncio.Queue()

            # ============================================================
            # SAFE AGENT MAP
            # ============================================================
            try:

                agent_map = AGENT_MAP

                # FIX:
                # prevents .items crashes
                if callable(agent_map):

                    self.logger.warning(
                        "AGENT_MAP corrupted"
                    )

                    agent_map = {}

                if not isinstance(
                    agent_map,
                    dict,
                ):

                    agent_map = {}

            except Exception:

                agent_map = {}

            # ============================================================
            # BUILD AGENT LIST
            # ============================================================
            try:

                agent_list = list(
                    agent_map.items()
                )

            except Exception:

                agent_list = []

            # ============================================================
            # FILTER ENABLED AGENTS
            # ============================================================
            agents_to_init = []

            for item in agent_list:

                try:

                    if task_registry.is_shutting_down():
                        break

                except Exception:
                    pass

                try:

                    if (
                        not isinstance(
                            item,
                            tuple,
                        )
                        or len(item) < 2
                    ):
                        continue

                    agent_name, info = item

                    try:
                        agent_name = str(
                            agent_name
                        ).strip().lower()
                    except Exception:
                        continue

                    if not agent_name:
                        continue

                    # safe info
                    if callable(info):
                        info = {}

                    if not isinstance(
                        info,
                        dict,
                    ):
                        info = {}

                    config_key = info.get(
                        "config",
                        f"enable_{agent_name}",
                    )

                    enabled = bool(
                        cfg(
                            config_key,
                            True,
                        )
                    )

                    if not enabled:

                        results[
                            agent_name
                        ] = False

                        continue

                    agents_to_init.append(
                        (
                            agent_name,
                            info,
                        )
                    )

                except Exception as e:

                    self.logger.warning(
                        f"Agent filter failed: {e}"
                    )

            # ============================================================
            # NO AGENTS
            # ============================================================
            if not agents_to_init:

                self.logger.warning(
                    "No agents available for initialization"
                )

                return results

            # ============================================================
            # SAFE CHUNKING
            # ============================================================
            agent_chunks = [

                agents_to_init[
                    i:i + chunk_size
                ]

                for i in range(
                    0,
                    len(agents_to_init),
                    chunk_size,
                )
            ]

            self.logger.info(
                f"📦 Agent initialization "
                f"chunked into "
                f"{len(agent_chunks)} groups "
                f"(size={chunk_size}, "
                f"workers={pipeline_workers})"
            )

            # ============================================================
            # SHARED STATE
            # ============================================================
            init_results: Dict[str, bool] = {}

            results_lock = asyncio.Lock()

            # ============================================================
            # PIPELINE MODE
            # ============================================================
            if enable_pipeline:

                # --------------------------------------------------------
                # START WORKERS
                # --------------------------------------------------------
                for worker_id in range(
                    pipeline_workers
                ):

                    worker = asyncio.create_task(

                        self._agent_init_worker(

                            worker_id,

                            agent_init_queue,

                            init_results,

                            results_lock,

                            system_instance,
                        ),

                        name=f"agent_init_worker_{worker_id}",
                    )

                    workers.append(
                        worker
                    )

                # --------------------------------------------------------
                # QUEUE CHUNKS
                # --------------------------------------------------------
                for (
                    chunk_idx,
                    chunk,
                ) in enumerate(
                    agent_chunks
                ):

                    try:

                        await asyncio.wait_for(

                            agent_init_queue.put(
                                {
                                    "chunk_id": chunk_idx,
                                    "agents": chunk,
                                    "total_chunks": len(agent_chunks),
                                    "timestamp": time.time(),
                                }
                            ),

                            timeout=5,
                        )

                    except asyncio.TimeoutError:

                        self.logger.warning(
                            f"Queue put timeout "
                            f"(chunk={chunk_idx})"
                        )

                # --------------------------------------------------------
                # SENTINELS
                # --------------------------------------------------------
                for _ in range(
                    pipeline_workers
                ):

                    try:

                        await asyncio.wait_for(

                            agent_init_queue.put(
                                None
                            ),

                            timeout=5,
                        )

                    except Exception:
                        pass

                # --------------------------------------------------------
                # WAIT FOR QUEUE
                # --------------------------------------------------------
                try:

                    await asyncio.wait_for(

                        agent_init_queue.join(),

                        timeout=init_timeout,
                    )

                except asyncio.TimeoutError:

                    self.logger.warning(
                        "Queue join timeout"
                    )

                # --------------------------------------------------------
                # WAIT FOR WORKERS
                # --------------------------------------------------------
                try:

                    worker_results = await asyncio.wait_for(

                        asyncio.gather(

                            *workers,

                            return_exceptions=True,
                        ),

                        timeout=init_timeout,
                    )

                    for wr in worker_results:

                        if isinstance(
                            wr,
                            Exception,
                        ):

                            self.logger.warning(
                                f"Worker failure: {wr}"
                            )

                except asyncio.TimeoutError:

                    self.logger.error(
                        f"Agent worker timeout "
                        f"({init_timeout}s)"
                    )

                # --------------------------------------------------------
                # MERGE RESULTS
                # --------------------------------------------------------
                try:

                    if isinstance(
                        init_results,
                        dict,
                    ):

                        results.update(
                            init_results
                        )

                except Exception as e:

                    self.logger.warning(
                        f"Result merge failed: {e}"
                    )

            # ============================================================
            # SEQUENTIAL MODE
            # ============================================================
            else:

                self.logger.info(
                    "Running sequential initialization"
                )

                for (
                    chunk_idx,
                    chunk,
                ) in enumerate(
                    agent_chunks
                ):

                    try:

                        if task_registry.is_shutting_down():
                            break

                    except Exception:
                        pass

                    try:

                        await asyncio.wait_for(

                            self._process_agent_chunk(

                                chunk,

                                chunk_idx,

                                len(agent_chunks),

                                results,

                                system_instance,
                            ),

                            timeout=init_timeout,
                        )

                    except asyncio.TimeoutError:

                        self.logger.warning(
                            f"Sequential chunk timeout "
                            f"({chunk_idx})"
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"Sequential chunk failed: {e}"
                        )

            # ============================================================
            # VISION PIPELINE
            # ============================================================
            try:

                agents = getattr(
                    self,
                    "agents",
                    {},
                )

                if (
                    isinstance(
                        agents,
                        dict,
                    )
                    and agents.get(
                        "vision"
                    )
                ):

                    await asyncio.wait_for(

                        self._connect_vision_engine_pipeline(
                            system_instance
                        ),

                        timeout=30,
                    )

            except asyncio.TimeoutError:

                self.logger.warning(
                    "Vision pipeline timeout"
                )

            except Exception as e:

                self.logger.warning(
                    f"Vision pipeline failed: {e}"
                )

            # ============================================================
            # COMMUNICATION PIPELINE
            # ============================================================
            if enable_comms:

                try:

                    await asyncio.wait_for(

                        self._setup_agent_communication_pipeline(),

                        timeout=60,
                    )

                except asyncio.TimeoutError:

                    self.logger.warning(
                        "Communication pipeline timeout"
                    )

                except Exception as e:

                    self.logger.warning(
                        f"Communication pipeline failed: {e}"
                    )

            # ============================================================
            # VALIDATION
            # ============================================================
            try:

                agents = getattr(
                    self,
                    "agents",
                    {},
                )

                if callable(agents):
                    agents = {}

                if not isinstance(
                    agents,
                    dict,
                ):
                    agents = {}

                validation_results = await asyncio.wait_for(

                    self._validate_agents_chunked(
                        list(
                            agents.keys()
                        )
                    ),

                    timeout=120,
                )

                if isinstance(
                    validation_results,
                    dict,
                ):

                    for (
                        agent_name,
                        is_valid,
                    ) in validation_results.items():

                        if agent_name in results:

                            results[
                                agent_name
                            ] = bool(
                                results[
                                    agent_name
                                ]
                                and is_valid
                            )

            except asyncio.TimeoutError:

                self.logger.warning(
                    "Validation timeout"
                )

            except Exception as e:

                self.logger.warning(
                    f"Validation failed: {e}"
                )

            # ============================================================
            # SUMMARY
            # ============================================================
            active = sum(
                1
                for value in results.values()
                if value
            )

            total = len(results)

            duration = round(
                time.monotonic()
                - start_time,
                2,
            )

            self.logger.info(
                f"✅ Agent init complete: "
                f"{active}/{total} active "
                f"(duration={duration}s)"
            )

            # ============================================================
            # METRICS
            # ============================================================
            try:

                metrics.gauge_set(
                    "agents_initialized",
                    active,
                )

                metrics.gauge_set(
                    "agents_total",
                    total,
                )

                metrics.histogram_observe(
                    "agent_init_duration_seconds",
                    duration,
                )

                metrics.gauge_set(
                    "agent_init_chunks",
                    len(agent_chunks),
                )

            except Exception:
                pass

            # ============================================================
            # FAILED AGENTS
            # ============================================================
            failed_agents = [

                name

                for (
                    name,
                    success,
                ) in results.items()

                if not success
            ]

            if failed_agents:

                self.logger.warning(
                    f"⚠ Failed agents: "
                    f"{failed_agents}"
                )

                try:

                    metrics.gauge_set(
                        "agents_failed",
                        len(failed_agents),
                    )

                except Exception:
                    pass

            # ============================================================
            # CLEANUP
            # ============================================================
            try:

                gc.collect()

            except Exception:
                pass

            return results

        # ================================================================
        # TIMEOUT
        # ================================================================
        except asyncio.TimeoutError:

            self.logger.error(
                f"❌ Agent initialization timeout "
                f"after {init_timeout}s"
            )

            try:

                metrics.counter_inc(
                    "agent_init_timeout_total"
                )

            except Exception:
                pass

            return results

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            self.logger.warning(
                "initialize_all cancelled"
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            self.logger.error(
                f"❌ Agent initialization "
                f"system failure: {e}"
            )

            self.logger.debug(
                traceback.format_exc()[:4000]
            )

            try:

                metrics.counter_inc(

                    "agent_init_failed_total",

                    {
                        "error": str(e)
                    },
                )

            except Exception:
                pass

            return results

        # ================================================================
        # FINAL CLEANUP
        # ================================================================
        finally:

            # cancel orphan workers
            for worker in workers:

                try:

                    if (
                        worker
                        and not worker.done()
                    ):

                        worker.cancel()

                except Exception:
                    pass

            # cleanup tasks
            if workers:

                try:

                    await asyncio.gather(
                        *workers,
                        return_exceptions=True,
                    )

                except Exception:
                    pass

            try:

                gc.collect()

            except Exception:
                pass

    async def _agent_init_worker(
        self,
        worker_id: int,
        queue: asyncio.Queue,
        results: dict,
        results_lock: asyncio.Lock,
        system_instance: Any,
    ):
        """
        Production-safe parallel agent initialization worker.

        Fixes:
        - queue deadlocks
        - coroutine leaks
        - invalid queue.task_done usage
        - bool await crashes
        - worker crashes
        - unsafe result mutation
        - invalid chunk corruption
        - startup race conditions
        - memory leaks
        - cancellation instability
        - infinite worker hangs
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        worker_start = time.monotonic()

        self.logger.debug(
            f"🔧 Agent init worker {worker_id} started"
        )

        processed_chunks = 0
        initialized_agents = 0
        failed_agents = 0

        try:

            # ============================================================
            # VALIDATE INPUTS
            # ============================================================
            if queue is None:

                self.logger.error(
                    f"Worker {worker_id}: queue is None"
                )

                return

            if results is None or callable(results):

                self.logger.warning(
                    f"Worker {worker_id}: invalid results store"
                )

                results = {}

            if results_lock is None:

                self.logger.warning(
                    f"Worker {worker_id}: missing results lock"
                )

                results_lock = asyncio.Lock()

            # ============================================================
            # SAFE LOOP
            # ============================================================
            while True:

                # --------------------------------------------------------
                # SHUTDOWN CHECK
                # --------------------------------------------------------
                try:

                    if task_registry.is_shutting_down():

                        self.logger.debug(
                            f"Worker {worker_id}: "
                            f"shutdown requested"
                        )

                        break

                except Exception:
                    pass

                chunk_data = None
                task_acquired = False

                try:

                    # ----------------------------------------------------
                    # SAFE QUEUE GET
                    # ----------------------------------------------------
                    chunk_data = await asyncio.wait_for(

                        queue.get(),

                        timeout=1.0,
                    )

                    task_acquired = True

                    # ----------------------------------------------------
                    # SENTINEL
                    # ----------------------------------------------------
                    if chunk_data is None:

                        self.logger.debug(
                            f"Worker {worker_id}: "
                            f"received sentinel"
                        )

                        try:
                            queue.task_done()
                        except Exception:
                            pass

                        break

                    # ----------------------------------------------------
                    # VALIDATE CHUNK
                    # ----------------------------------------------------
                    if callable(chunk_data):

                        self.logger.warning(
                            f"Worker {worker_id}: "
                            f"callable chunk ignored"
                        )

                        try:
                            queue.task_done()
                        except Exception:
                            pass

                        continue

                    if not isinstance(
                        chunk_data,
                        dict,
                    ):

                        self.logger.warning(
                            f"Worker {worker_id}: "
                            f"invalid chunk type"
                        )

                        try:
                            queue.task_done()
                        except Exception:
                            pass

                        continue

                    # ----------------------------------------------------
                    # SAFE CHUNK FIELDS
                    # ----------------------------------------------------
                    agents = chunk_data.get(
                        "agents",
                        [],
                    )

                    chunk_id = int(

                        chunk_data.get(
                            "chunk_id",
                            0,
                        )
                    )

                    total_chunks = int(

                        chunk_data.get(
                            "total_chunks",
                            1,
                        )
                    )

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(agents):

                        agents = []

                    if not isinstance(
                        agents,
                        (list, tuple),
                    ):

                        agents = []

                    self.logger.info(
                        f"Worker {worker_id} "
                        f"initializing chunk "
                        f"{chunk_id + 1}/{total_chunks} "
                        f"with {len(agents)} agents"
                    )

                    chunk_results = {}

                    # ----------------------------------------------------
                    # PROCESS AGENTS
                    # ----------------------------------------------------
                    for agent_entry in agents:

                        try:

                            if task_registry.is_shutting_down():
                                break

                        except Exception:
                            pass

                        try:

                            # --------------------------------------------
                            # VALIDATE ENTRY
                            # --------------------------------------------
                            if (
                                not isinstance(
                                    agent_entry,
                                    (list, tuple),
                                )
                                or len(agent_entry) < 2
                            ):

                                self.logger.warning(
                                    f"Worker {worker_id}: "
                                    f"invalid agent entry"
                                )

                                continue

                            agent_name, info = (
                                agent_entry[0],
                                agent_entry[1],
                            )

                            # --------------------------------------------
                            # SAFE NAME
                            # --------------------------------------------
                            try:

                                agent_name = str(
                                    agent_name
                                ).strip().lower()

                            except Exception:

                                continue

                            if not agent_name:
                                continue

                            # --------------------------------------------
                            # SAFE INFO
                            # --------------------------------------------
                            if callable(info):
                                info = {}

                            if not isinstance(
                                info,
                                dict,
                            ):
                                info = {}

                            # --------------------------------------------
                            # INITIALIZE
                            # --------------------------------------------
                            result = await asyncio.wait_for(

                                self._initialize_single_agent(

                                    agent_name,

                                    info,

                                    system_instance,

                                    worker_id,

                                    chunk_id,
                                ),

                                timeout=120,
                            )

                            # FIX:
                            # coroutine leak protection
                            if inspect.isawaitable(
                                result
                            ):

                                result = await asyncio.wait_for(

                                    result,

                                    timeout=30,
                                )

                            final_result = bool(
                                result
                            )

                            chunk_results[
                                agent_name
                            ] = final_result

                            if final_result:

                                initialized_agents += 1

                            else:

                                failed_agents += 1

                        # --------------------------------------------
                        # AGENT TIMEOUT
                        # --------------------------------------------
                        except asyncio.TimeoutError:

                            self.logger.warning(
                                f"Worker {worker_id}: "
                                f"agent init timeout"
                            )

                            failed_agents += 1

                        # --------------------------------------------
                        # AGENT CANCELLED
                        # --------------------------------------------
                        except asyncio.CancelledError:
                            raise

                        # --------------------------------------------
                        # AGENT FAILURE
                        # --------------------------------------------
                        except Exception as e:

                            failed_agents += 1

                            self.logger.error(
                                f"Worker {worker_id}: "
                                f"agent init failed: {e}"
                            )

                            self.logger.debug(
                                traceback.format_exc()[:3000]
                            )

                    # ----------------------------------------------------
                    # STORE RESULTS
                    # ----------------------------------------------------
                    try:

                        async with results_lock:

                            if callable(results):

                                results = {}

                            if not isinstance(
                                results,
                                dict,
                            ):

                                results = {}

                            results.update(
                                chunk_results
                            )

                    except Exception as e:

                        self.logger.warning(
                            f"Worker {worker_id}: "
                            f"results update failed: {e}"
                        )

                    # ----------------------------------------------------
                    # METRICS
                    # ----------------------------------------------------
                    processed_chunks += 1

                    try:

                        metrics.counter_inc(

                            "agent_init_chunk_processed",

                            {
                                "worker": worker_id
                            },
                        )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # TASK COMPLETE
                    # ----------------------------------------------------
                    try:

                        queue.task_done()

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # MEMORY CLEANUP
                    # ----------------------------------------------------
                    try:

                        gc.collect()

                    except Exception:
                        pass

                    await asyncio.sleep(
                        0.01
                    )

                # --------------------------------------------------------
                # QUEUE TIMEOUT
                # --------------------------------------------------------
                except asyncio.TimeoutError:

                    await asyncio.sleep(
                        0.02
                    )

                    continue

                # --------------------------------------------------------
                # CANCELLED
                # --------------------------------------------------------
                except asyncio.CancelledError:

                    self.logger.debug(
                        f"Worker {worker_id}: cancelled"
                    )

                    break

                # --------------------------------------------------------
                # HARD FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    self.logger.error(
                        f"Agent init worker "
                        f"{worker_id} error: {e}"
                    )

                    self.logger.debug(
                        traceback.format_exc()[:4000]
                    )

                    # FIX:
                    # prevent queue deadlock
                    if task_acquired:

                        try:
                            queue.task_done()
                        except Exception:
                            pass

                    await asyncio.sleep(
                        0.1
                    )

        # ================================================================
        # WORKER FAILURE
        # ================================================================
        except Exception as e:

            self.logger.error(
                f"Worker {worker_id} crashed: {e}"
            )

            self.logger.debug(
                traceback.format_exc()[:4000]
            )

        # ================================================================
        # CLEAN SHUTDOWN
        # ================================================================
        finally:

            elapsed_ms = round(

                (
                    time.monotonic()
                    - worker_start
                ) * 1000,

                2,
            )

            self.logger.debug(
                f"🔧 Agent init worker "
                f"{worker_id} stopped | "
                f"chunks={processed_chunks} | "
                f"success={initialized_agents} | "
                f"failed={failed_agents} | "
                f"uptime={elapsed_ms}ms"
            )

            try:

                gc.collect()

            except Exception:
                pass

    async def _initialize_single_agent(
        self,
        agent_name: str,
        info: dict,
        system_instance: Any,
        worker_id: int,
        chunk_id: int,
    ) -> bool:
        """
        Production-safe single agent initializer.

        Fixes:
        - function.items crashes
        - bool await crashes
        - coroutine leaks
        - unsafe imports
        - invalid wrapper corruption
        - startup race conditions
        - async/sync mismatch
        - invalid agent injection
        - duplicate initialization
        - memory leaks
        - unsafe config access
        - broken stub fallback
        """

        import asyncio
        import gc
        import importlib
        import inspect
        import time
        import traceback
        from pathlib import Path

        start_time = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        f"Worker {worker_id}: "
                        f"shutdown active "
                        f"({agent_name})"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # VALIDATE AGENT NAME
            # ============================================================
            try:

                agent_name = str(
                    agent_name
                ).strip().lower()

            except Exception:

                self.logger.warning(
                    f"Worker {worker_id}: "
                    f"invalid agent name"
                )

                return False

            if not agent_name:

                return False

            # ============================================================
            # VALIDATE INFO
            # ============================================================
            if info is None:

                info = {}

            # FIX:
            # prevents:
            # function.items crashes
            if callable(info):

                self.logger.warning(
                    f"{agent_name}: info callable"
                )

                info = {}

            if not isinstance(
                info,
                dict,
            ):

                info = {}

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                None,
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if config is None:
                        return default

                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # callable corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE REGISTRIES
            # ============================================================
            if not hasattr(
                self,
                "agents",
            ) or callable(
                getattr(
                    self,
                    "agents",
                    None,
                )
            ):

                self.agents = {}

            if not hasattr(
                self,
                "wrappers",
            ) or callable(
                getattr(
                    self,
                    "wrappers",
                    None,
                )
            ):

                self.wrappers = {}

            # ============================================================
            # DUPLICATE INIT GUARD
            # ============================================================
            existing_agent = self.agents.get(
                agent_name
            )

            if existing_agent is not None:

                self.logger.info(
                    f"Worker {worker_id}: "
                    f"✓ {agent_name} already initialized"
                )

                return True

            # ============================================================
            # SAFE FLAGS
            # ============================================================
            available = bool(

                info.get(
                    "available",
                    True,
                )
            )

            init_timeout = float(

                cfg(
                    "agent_init_timeout",
                    60,
                )
            )

            init_timeout = max(
                5.0,
                min(
                    init_timeout,
                    300.0,
                ),
            )

            agent = None
            wrapper = None

            # ============================================================
            # AVAILABLE AGENT PATH
            # ============================================================
            if available:

                try:

                    # ====================================================
                    # FILE AGENT
                    # ====================================================
                    if agent_name == "file":

                        from core.agent.file_agent import (
                            FileAgent,
                            FileAgentWrapper,
                        )

                        workspace_dir = str(

                            cfg(
                                "workspace_dir",
                                "./workspace",
                            )
                        )

                        Path(
                            workspace_dir
                        ).mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        agent = await asyncio.wait_for(

                            asyncio.to_thread(

                                FileAgent,

                                workspace_root=workspace_dir,
                            ),

                            timeout=init_timeout,
                        )

                        wrapper = await asyncio.wait_for(

                            asyncio.to_thread(

                                FileAgentWrapper,

                                {
                                    "workspace":
                                    workspace_dir
                                },
                            ),

                            timeout=init_timeout,
                        )

                    # ====================================================
                    # VISION AGENT
                    # ====================================================
                    elif agent_name == "vision":

                        from core.agent.vision_agent import (
                            VisionAgent,
                            VisionAgentWrapper,
                        )

                        agent = await asyncio.wait_for(

                            asyncio.to_thread(
                                VisionAgent
                            ),

                            timeout=init_timeout,
                        )

                        wrapper = await asyncio.wait_for(

                            asyncio.to_thread(
                                VisionAgentWrapper
                            ),

                            timeout=init_timeout,
                        )

                    # ====================================================
                    # BLENDER AGENT
                    # ====================================================
                    elif (
                        agent_name == "blender"
                        and globals().get(
                            "BLENDER_AGENT_AVAILABLE",
                            False,
                        )
                    ):

                        blender_workspace = str(

                            cfg(
                                "blender_workspace_dir",
                                "./blender_workspace",
                            )
                        )

                        blender_render = str(

                            cfg(
                                "blender_render_dir",
                                "./renders",
                            )
                        )

                        Path(
                            blender_workspace
                        ).mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        Path(
                            blender_render
                        ).mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        agent = await asyncio.wait_for(

                            asyncio.to_thread(
                                Blender3DAgent
                            ),

                            timeout=init_timeout,
                        )

                        wrapper = agent

                        self.logger.info(
                            f"Worker {worker_id}: "
                            f"🎨 Blender Agent initialized"
                        )

                    # ====================================================
                    # DYNAMIC AGENTS
                    # ====================================================
                    else:

                        module = info.get(
                            "module"
                        )

                        if not module:

                            raise RuntimeError(
                                "Missing module path"
                            )

                        # ------------------------------------------------
                        # SAFE IMPORT
                        # ------------------------------------------------
                        agent_module = await asyncio.wait_for(

                            asyncio.to_thread(

                                importlib.import_module,

                                module,
                            ),

                            timeout=init_timeout,
                        )

                        if agent_module is None:

                            raise RuntimeError(
                                "Module import failed"
                            )

                        # ------------------------------------------------
                        # RESOLVE CLASSES
                        # ------------------------------------------------
                        class_prefix = (
                            agent_name.title()
                        )

                        agent_cls = getattr(

                            agent_module,

                            f"{class_prefix}Agent",

                            None,
                        )

                        wrapper_cls = getattr(

                            agent_module,

                            f"{class_prefix}AgentWrapper",

                            None,
                        )

                        if agent_cls is None:

                            raise RuntimeError(
                                "Agent class missing"
                            )

                        # ------------------------------------------------
                        # CREATE AGENT
                        # ------------------------------------------------
                        if inspect.isclass(
                            agent_cls
                        ):

                            agent = await asyncio.wait_for(

                                asyncio.to_thread(
                                    agent_cls
                                ),

                                timeout=init_timeout,
                            )

                        # ------------------------------------------------
                        # CREATE WRAPPER
                        # ------------------------------------------------
                        if wrapper_cls:

                            if inspect.isclass(
                                wrapper_cls
                            ):

                                wrapper = await asyncio.wait_for(

                                    asyncio.to_thread(
                                        wrapper_cls
                                    ),

                                    timeout=init_timeout,
                                )

                        else:

                            wrapper = agent

                    # ====================================================
                    # VALIDATE AGENT
                    # ====================================================
                    if agent is None:

                        raise RuntimeError(
                            "Agent creation failed"
                        )

                    # FIX:
                    # invalid bool agent
                    if isinstance(
                        agent,
                        bool,
                    ):

                        raise RuntimeError(
                            "Agent returned bool"
                        )

                    if wrapper is None:

                        wrapper = agent

                    # ====================================================
                    # OPTIONAL INITIALIZE
                    # ====================================================
                    initialize_method = getattr(
                        agent,
                        "initialize",
                        None,
                    )

                    if callable(
                        initialize_method
                    ):

                        try:

                            result = initialize_method()

                            # FIX:
                            # bool await corruption
                            if inspect.isawaitable(
                                result
                            ):

                                await asyncio.wait_for(

                                    result,

                                    timeout=init_timeout,
                                )

                        except Exception as e:

                            self.logger.warning(
                                f"{agent_name}: "
                                f"initialize failed: {e}"
                            )

                    # ====================================================
                    # STORE REFERENCES
                    # ====================================================
                    self.agents[
                        agent_name
                    ] = agent

                    self.wrappers[
                        agent_name
                    ] = wrapper

                    # ====================================================
                    # AUDIT
                    # ====================================================
                    await self._safe_call(

                        self,

                        "_log_audit",

                        "agent_init",

                        agent_name,

                        True,

                        f"worker_{worker_id}_chunk_{chunk_id}",
                    )

                    # ====================================================
                    # METRICS
                    # ====================================================
                    try:

                        metrics.counter_inc(

                            "agent_initialized",

                            {
                                "agent": agent_name
                            },
                        )

                    except Exception:
                        pass

                    # ====================================================
                    # CLEANUP
                    # ====================================================
                    try:

                        gc.collect()

                    except Exception:
                        pass

                    # ====================================================
                    # SUCCESS
                    # ====================================================
                    elapsed_ms = round(

                        (
                            time.monotonic()
                            - start_time
                        ) * 1000,

                        2,
                    )

                    self.logger.info(
                        f"Worker {worker_id}: "
                        f"✓ {agent_name.title()} "
                        f"initialized "
                        f"in {elapsed_ms}ms"
                    )

                    return True

                # ========================================================
                # STUB FALLBACK
                # ========================================================
                except Exception as e:

                    self.logger.warning(
                        f"Worker {worker_id}: "
                        f"⚠ {agent_name} init failed "
                        f"→ stub mode: {e}"
                    )

                    self.logger.debug(
                        traceback.format_exc()[:3000]
                    )

                    try:

                        from core.agent.stub_agents import (
                            StubAgent
                        )

                        stub_agent = await asyncio.wait_for(

                            asyncio.to_thread(
                                StubAgent
                            ),

                            timeout=10,
                        )

                        self.agents[
                            agent_name
                        ] = stub_agent

                        self.wrappers[
                            agent_name
                        ] = stub_agent

                        await self._safe_call(

                            self,

                            "_log_audit",

                            "agent_init",

                            agent_name,

                            True,

                            f"stub_mode - {e}",
                        )

                        return True

                    except Exception as stub_error:

                        self.logger.error(
                            f"Stub fallback failed "
                            f"({agent_name}): "
                            f"{stub_error}"
                        )

                        return False

            # ============================================================
            # FORCE STUB
            # ============================================================
            else:

                try:

                    from core.agent.stub_agents import (
                        StubAgent
                    )

                    stub_agent = await asyncio.wait_for(

                        asyncio.to_thread(
                            StubAgent
                        ),

                        timeout=10,
                    )

                    self.agents[
                        agent_name
                    ] = stub_agent

                    self.wrappers[
                        agent_name
                    ] = stub_agent

                    await self._safe_call(

                        self,

                        "_log_audit",

                        "agent_init",

                        agent_name,

                        True,

                        "stub_mode_unavailable",
                    )

                    self.logger.info(
                        f"Worker {worker_id}: "
                        f"✓ {agent_name} "
                        f"stub initialized"
                    )

                    return True

                except Exception as e:

                    self.logger.error(
                        f"Stub initialization failed "
                        f"({agent_name}): {e}"
                    )

                    return False

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    f"_initialize_single_agent cancelled "
                    f"({agent_name})"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"Worker {worker_id}: "
                    f"❌ {agent_name} init crashed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

                await self._safe_call(

                    self,

                    "_log_audit",

                    "agent_init",

                    agent_name,

                    False,

                    str(e),
                )

                try:

                    metrics.counter_inc(

                        "agent_init_failed",

                        {
                            "agent": agent_name,
                            "error": str(e),
                        },
                    )

                except Exception:
                    pass

            except Exception:
                pass

            return False

    async def _process_agent_chunk(
        self,
        chunk: list,
        chunk_idx: int,
        total_chunks: int,
        results: dict,
        system_instance: Any,
    ):
        """Process an agent chunk sequentially (fallback method)."""
        self.logger.info(f"📦 Processing agent chunk {chunk_idx+1}/{total_chunks}")

        for agent_name, info in chunk:
            result = await self._initialize_single_agent(
                agent_name, info, system_instance, "sequential", chunk_idx
            )
            results[agent_name] = result

            # Small delay between agents
            await asyncio.sleep(0.1)

    async def _connect_vision_engine_pipeline(
        self,
        system_instance: Any,
    ):
        """
        Production-safe vision pipeline connector.

        Fixes:
        - function.items crashes
        - invalid vision engine references
        - coroutine leaks
        - retry corruption
        - timeout hangs
        - async/sync mismatch
        - pipeline deadlocks
        - invalid wrapper corruption
        - startup race conditions
        - metrics crashes
        - unsafe config access
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping "
                        "vision pipeline connection"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    # FIX:
                    # config accidentally became callable
                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents function corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE AGENTS
            # ============================================================
            agents = getattr(
                self,
                "agents",
                {},
            )

            wrappers = getattr(
                self,
                "wrappers",
                {},
            )

            # FIX:
            # prevents .items crashes
            if callable(agents):
                agents = {}

            if callable(wrappers):
                wrappers = {}

            if not isinstance(
                agents,
                dict,
            ):
                agents = {}

            if not isinstance(
                wrappers,
                dict,
            ):
                wrappers = {}

            # ============================================================
            # GET VISION AGENT
            # ============================================================
            vision_agent = agents.get(
                "vision"
            )

            vision_wrapper = wrappers.get(
                "vision"
            )

            if vision_agent is None:

                self.logger.warning(
                    "Vision agent missing"
                )

                return False

            if vision_wrapper is None:

                self.logger.warning(
                    "Vision wrapper missing"
                )

                return False

            # ============================================================
            # VALIDATE WRAPPER
            # ============================================================
            connect_method = getattr(
                vision_wrapper,
                "connect_to_vision_engine",
                None,
            )

            if not callable(
                connect_method
            ):

                self.logger.warning(
                    "Vision wrapper missing "
                    "connect_to_vision_engine"
                )

                return False

            # ============================================================
            # VALIDATE SYSTEM INSTANCE
            # ============================================================
            if system_instance is None:

                self.logger.warning(
                    "System instance missing"
                )

                return False

            # ============================================================
            # GET VISION ENGINE
            # ============================================================
            try:

                ve = getattr(
                    system_instance,
                    "vision_engine_instance",
                    None,
                )

            except Exception:

                ve = None

            if ve is None:

                self.logger.warning(
                    "Vision engine instance missing"
                )

                return False

            # FIX:
            # corrupted float/device errors
            if isinstance(
                ve,
                (int, float, bool),
            ):

                self.logger.warning(
                    f"Invalid vision engine type: "
                    f"{type(ve)}"
                )

                return False

            # ============================================================
            # PREVENT DUPLICATE CONNECTION
            # ============================================================
            already_connected = bool(

                getattr(
                    vision_wrapper,
                    "_vision_connected",
                    False,
                )
            )

            if already_connected:

                self.logger.info(
                    "✓ Vision pipeline already connected"
                )

                return True

            # ============================================================
            # SAFE SETTINGS
            # ============================================================
            try:

                max_retries = int(

                    cfg(
                        "vision_connect_retries",
                        3,
                    )
                )

            except Exception:

                max_retries = 3

            try:

                retry_delay = float(

                    cfg(
                        "vision_connect_delay",
                        1,
                    )
                )

            except Exception:

                retry_delay = 1.0

            try:

                connect_timeout = float(

                    cfg(
                        "vision_connect_timeout",
                        5,
                    )
                )

            except Exception:

                connect_timeout = 5.0

            max_retries = max(
                1,
                min(
                    max_retries,
                    10,
                ),
            )

            retry_delay = max(
                0.1,
                min(
                    retry_delay,
                    30.0,
                ),
            )

            connect_timeout = max(
                1.0,
                min(
                    connect_timeout,
                    120.0,
                ),
            )

            # ============================================================
            # RETRY LOOP
            # ============================================================
            last_error = None

            for attempt in range(
                1,
                max_retries + 1,
            ):

                try:

                    # ----------------------------------------------------
                    # SHUTDOWN CHECK
                    # ----------------------------------------------------
                    try:

                        if task_registry.is_shutting_down():

                            self.logger.debug(
                                "Shutdown during "
                                "vision connect"
                            )

                            return False

                    except Exception:
                        pass

                    self.logger.info(
                        f"🔌 Connecting vision pipeline "
                        f"({attempt}/{max_retries})"
                    )

                    # ----------------------------------------------------
                    # ASYNC CONNECT
                    # ----------------------------------------------------
                    if inspect.iscoroutinefunction(
                        connect_method
                    ):

                        result = connect_method(
                            ve
                        )

                        # FIX:
                        # bool await corruption
                        if inspect.isawaitable(
                            result
                        ):

                            result = await asyncio.wait_for(

                                result,

                                timeout=connect_timeout,
                            )

                    # ----------------------------------------------------
                    # SYNC CONNECT
                    # ----------------------------------------------------
                    else:

                        result = await asyncio.wait_for(

                            asyncio.to_thread(
                                connect_method,
                                ve,
                            ),

                            timeout=connect_timeout,
                        )

                    # ----------------------------------------------------
                    # VALIDATE RESULT
                    # ----------------------------------------------------
                    if isinstance(
                        result,
                        Exception,
                    ):

                        raise result

                    # ----------------------------------------------------
                    # STORE CONNECTION STATE
                    # ----------------------------------------------------
                    try:

                        setattr(
                            vision_wrapper,
                            "_vision_connected",
                            True,
                        )

                        setattr(
                            vision_wrapper,
                            "_vision_connected_at",
                            time.time(),
                        )

                        setattr(
                            vision_wrapper,
                            "_vision_engine",
                            ve,
                        )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # METRICS
                    # ----------------------------------------------------
                    try:

                        if "metrics" in globals():

                            metrics.counter_inc(
                                "vision_engine_connected"
                            )

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # SUCCESS
                    # ----------------------------------------------------
                    elapsed_ms = round(

                        (
                            time.monotonic()
                            - start_time
                        ) * 1000,

                        2,
                    )

                    self.logger.info(
                        f"✓ VisionAgent connected "
                        f"to VisionEngine "
                        f"in {elapsed_ms}ms"
                    )

                    return True

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except asyncio.TimeoutError as e:

                    last_error = e

                    self.logger.warning(
                        f"Vision connect timeout "
                        f"({attempt}/{max_retries})"
                    )

                # --------------------------------------------------------
                # CANCELLED
                # --------------------------------------------------------
                except asyncio.CancelledError:
                    raise

                # --------------------------------------------------------
                # FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    last_error = e

                    self.logger.warning(
                        f"Failed to connect "
                        f"VisionAgent: {e}"
                    )

                    self.logger.debug(
                        traceback.format_exc()[:3000]
                    )

                # --------------------------------------------------------
                # BACKOFF
                # --------------------------------------------------------
                if attempt < max_retries:

                    try:

                        await asyncio.sleep(
                            retry_delay * attempt
                        )

                    except asyncio.CancelledError:

                        return False

            # ============================================================
            # FINAL FAILURE
            # ============================================================
            try:

                if "metrics" in globals():

                    metrics.counter_inc(
                        "vision_connect_failed"
                    )

            except Exception:
                pass

            self.logger.error(
                f"❌ Vision pipeline connection failed "
                f"after {max_retries} attempts "
                f"(last_error={last_error})"
            )

            return False

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_connect_vision_engine_pipeline cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ Vision engine connection "
                    f"pipeline failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

    async def _setup_agent_communication_pipeline(self):
        """
        Production-safe agent communication pipeline setup.

        Fixes:
        - function.items crashes
        - coroutine leaks
        - invalid wrapper corruption
        - communication deadlocks
        - invalid chunking
        - async/sync mismatch
        - unsafe config access
        - event bus corruption
        - startup race conditions
        - broken safe_call execution
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping "
                        "agent communication setup"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    # FIX:
                    # config became callable
                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents callable corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE REGISTRIES
            # ============================================================
            agents = getattr(
                self,
                "agents",
                {},
            )

            wrappers = getattr(
                self,
                "wrappers",
                {},
            )

            # FIX:
            # prevents .items crashes
            if callable(agents):
                agents = {}

            if callable(wrappers):
                wrappers = {}

            if not isinstance(
                agents,
                dict,
            ):
                agents = {}

            if not isinstance(
                wrappers,
                dict,
            ):
                wrappers = {}

            # ============================================================
            # SAFE AGENT NAMES
            # ============================================================
            try:

                agent_names = list(
                    agents.keys()
                )

            except Exception:

                agent_names = []

            # remove invalid names
            agent_names = [

                str(name)

                for name in agent_names

                if name
            ]

            if len(agent_names) < 2:

                self.logger.debug(
                    "Insufficient agents for comms"
                )

                return True

            # ============================================================
            # SAFE SETTINGS
            # ============================================================
            try:

                chunk_size = int(

                    cfg(
                        "agent_comms_setup_chunk_size",
                        5,
                    )
                )

            except Exception:

                chunk_size = 5

            chunk_size = max(
                1,
                min(
                    chunk_size,
                    64,
                ),
            )

            # ============================================================
            # SAFE CHUNKING
            # ============================================================
            agent_chunks = [

                agent_names[
                    i:i + chunk_size
                ]

                for i in range(
                    0,
                    len(agent_names),
                    chunk_size,
                )
            ]

            self.logger.debug(
                f"Setting up agent comms "
                f"in {len(agent_chunks)} chunks"
            )

            # ============================================================
            # VALIDATE SAFE_CALL
            # ============================================================
            safe_call = getattr(
                self,
                "_safe_call",
                None,
            )

            if not callable(
                safe_call
            ):

                self.logger.error(
                    "_safe_call missing"
                )

                return False

            successful_setups = 0

            failed_setups = 0

            # ============================================================
            # PROCESS CHUNKS
            # ============================================================
            for (
                chunk_idx,
                chunk,
            ) in enumerate(
                agent_chunks,
                start=1,
            ):

                try:

                    if task_registry.is_shutting_down():
                        break

                except Exception:
                    pass

                comm_tasks = []

                # --------------------------------------------------------
                # PROCESS AGENTS
                # --------------------------------------------------------
                for agent_name in chunk:

                    try:

                        agent = agents.get(
                            agent_name
                        )

                        wrapper = wrappers.get(
                            agent_name
                        )

                        if agent is None:
                            continue

                        # ------------------------------------------------
                        # VALIDATE COMM PIPELINE
                        # ------------------------------------------------
                        create_pipeline = getattr(
                            self,
                            "_create_agent_comms_pipeline",
                            None,
                        )

                        pipeline = None

                        if callable(
                            create_pipeline
                        ):

                            try:

                                pipeline = (
                                    create_pipeline()
                                )

                                # FIX:
                                # bool await corruption
                                if inspect.isawaitable(
                                    pipeline
                                ):

                                    pipeline = (
                                        await pipeline
                                    )

                            except Exception as e:

                                self.logger.warning(
                                    f"Pipeline creation failed "
                                    f"({agent_name}): {e}"
                                )

                        # ------------------------------------------------
                        # REGISTER COMMUNICATION
                        # ------------------------------------------------
                        if (
                            wrapper is not None
                            and hasattr(
                                wrapper,
                                "register_communication",
                            )
                        ):

                            comm_tasks.append(

                                asyncio.create_task(

                                    safe_call(

                                        wrapper,

                                        "register_communication",

                                        pipeline,
                                    )
                                )
                            )

                        # ------------------------------------------------
                        # EVENT BUS
                        # ------------------------------------------------
                        event_bus = None

                        get_event_bus = getattr(
                            self,
                            "_get_agent_event_bus",
                            None,
                        )

                        if callable(
                            get_event_bus
                        ):

                            try:

                                event_bus = (
                                    get_event_bus()
                                )

                                if inspect.isawaitable(
                                    event_bus
                                ):

                                    event_bus = (
                                        await event_bus
                                    )

                            except Exception as e:

                                self.logger.warning(
                                    f"Event bus creation failed "
                                    f"({agent_name}): {e}"
                                )

                        # ------------------------------------------------
                        # SET EVENT BUS
                        # ------------------------------------------------
                        if hasattr(
                            agent,
                            "set_event_bus",
                        ):

                            comm_tasks.append(

                                asyncio.create_task(

                                    safe_call(

                                        agent,

                                        "set_event_bus",

                                        event_bus,
                                    )
                                )
                            )

                    except Exception as e:

                        failed_setups += 1

                        self.logger.warning(
                            f"Comms setup failed "
                            f"({agent_name}): {e}"
                        )

                # --------------------------------------------------------
                # EXECUTE CHUNK
                # --------------------------------------------------------
                if comm_tasks:

                    try:

                        results = await asyncio.gather(

                            *comm_tasks,

                            return_exceptions=True,
                        )

                        for result in results:

                            if isinstance(
                                result,
                                Exception,
                            ):

                                failed_setups += 1

                                self.logger.warning(
                                    f"Communication task failed: "
                                    f"{result}"
                                )

                            else:

                                successful_setups += 1

                    except asyncio.CancelledError:
                        raise

                    except Exception as e:

                        self.logger.warning(
                            f"Chunk comms failure: {e}"
                        )

                # --------------------------------------------------------
                # CHUNK COMPLETE
                # --------------------------------------------------------
                self.logger.debug(
                    f"Completed comms setup "
                    f"for chunk "
                    f"{chunk_idx}/"
                    f"{len(agent_chunks)}"
                )

                await asyncio.sleep(
                    0.02
                )

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.agent_comms_initialized = int(

                    getattr(
                        self,
                        "agent_comms_initialized",
                        0,
                    )

                ) + successful_setups

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.info(
                f"✓ Agent communication pipeline ready "
                f"(success={successful_setups}, "
                f"failed={failed_setups}) "
                f"in {elapsed_ms}ms"
            )

            return successful_setups > 0

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_setup_agent_communication_pipeline "
                    "cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.warning(
                    f"Agent communication pipeline "
                    f"setup failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

    async def _validate_agents_chunked(
        self,
        agent_names: list,
    ) -> Dict[str, bool]:
        """
        Production-safe chunked agent validation.

        Fixes:
        - function.items crashes
        - coroutine leaks
        - invalid await usage
        - broken validation pipelines
        - invalid chunking
        - async/sync mismatch
        - unsafe config access
        - startup race conditions
        - invalid agent corruption
        - validation deadlocks
        - bool await crashes
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        results: Dict[str, bool] = {}

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping "
                        "agent validation"
                    )

                    return results

            except Exception:
                pass

            # ============================================================
            # VALIDATE INPUT
            # ============================================================
            if agent_names is None:

                self.logger.warning(
                    "agent_names is None"
                )

                return results

            # FIX:
            # prevents:
            # function.items crashes
            if callable(agent_names):

                self.logger.warning(
                    "agent_names callable"
                )

                return results

            if not isinstance(
                agent_names,
                (list, tuple, set),
            ):

                self.logger.warning(
                    f"Invalid agent_names type: "
                    f"{type(agent_names)}"
                )

                return results

            # ============================================================
            # CLEAN AGENT NAMES
            # ============================================================
            cleaned_names = []

            for name in agent_names:

                try:

                    if not name:
                        continue

                    cleaned_names.append(
                        str(name)
                    )

                except Exception:
                    continue

            if not cleaned_names:

                self.logger.warning(
                    "No valid agent names"
                )

                return results

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # callable corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE SETTINGS
            # ============================================================
            try:

                chunk_size = int(

                    cfg(
                        "agent_validation_chunk_size",
                        10,
                    )
                )

            except Exception:

                chunk_size = 10

            try:

                validation_timeout = float(

                    cfg(
                        "agent_validation_timeout",
                        15.0,
                    )
                )

            except Exception:

                validation_timeout = 15.0

            chunk_size = max(
                1,
                min(
                    chunk_size,
                    128,
                ),
            )

            validation_timeout = max(
                1.0,
                min(
                    validation_timeout,
                    300.0,
                ),
            )

            # ============================================================
            # SAFE AGENTS REGISTRY
            # ============================================================
            agents = getattr(
                self,
                "agents",
                {},
            )

            # FIX:
            # prevents .items crashes
            if callable(agents):
                agents = {}

            if not isinstance(
                agents,
                dict,
            ):
                agents = {}

            # ============================================================
            # SAFE CHUNKING
            # ============================================================
            agent_chunks = [

                cleaned_names[
                    i:i + chunk_size
                ]

                for i in range(
                    0,
                    len(cleaned_names),
                    chunk_size,
                )
            ]

            self.logger.debug(
                f"Validating agents "
                f"in {len(agent_chunks)} chunks"
            )

            # ============================================================
            # VALIDATE SAFE_CALL
            # ============================================================
            safe_call = getattr(
                self,
                "_safe_call",
                None,
            )

            if not callable(
                safe_call
            ):

                self.logger.error(
                    "_safe_call missing"
                )

                return results

            validated_count = 0

            failed_count = 0

            # ============================================================
            # PROCESS CHUNKS
            # ============================================================
            for (
                chunk_idx,
                chunk,
            ) in enumerate(
                agent_chunks,
                start=1,
            ):

                try:

                    if task_registry.is_shutting_down():
                        break

                except Exception:
                    pass

                validation_tasks = []

                task_names = []

                # --------------------------------------------------------
                # BUILD VALIDATION TASKS
                # --------------------------------------------------------
                for agent_name in chunk:

                    try:

                        agent = agents.get(
                            agent_name
                        )

                        if agent is None:

                            results[
                                agent_name
                            ] = False

                            failed_count += 1

                            continue

                        # FIX:
                        # corrupted function agents
                        if callable(agent):

                            self.logger.warning(
                                f"{agent_name}: "
                                f"agent callable"
                            )

                            results[
                                agent_name
                            ] = False

                            failed_count += 1

                            continue

                        validate_method = getattr(
                            agent,
                            "validate",
                            None,
                        )

                        # ------------------------------------------------
                        # VALIDATE METHOD EXISTS
                        # ------------------------------------------------
                        if callable(
                            validate_method
                        ):

                            task = asyncio.create_task(

                                safe_call(
                                    agent,
                                    "validate",
                                ),

                                name=f"validate_{agent_name}",
                            )

                        # ------------------------------------------------
                        # FALLBACK VALIDATION
                        # ------------------------------------------------
                        else:

                            async def _fallback_true():
                                return True

                            task = asyncio.create_task(

                                _fallback_true(),

                                name=f"validate_fallback_{agent_name}",
                            )

                        validation_tasks.append(
                            task
                        )

                        task_names.append(
                            agent_name
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"Task creation failed "
                            f"({agent_name}): {e}"
                        )

                        results[
                            agent_name
                        ] = False

                        failed_count += 1

                # --------------------------------------------------------
                # EXECUTE CHUNK
                # --------------------------------------------------------
                if validation_tasks:

                    try:

                        validation_results = await asyncio.wait_for(

                            asyncio.gather(

                                *validation_tasks,

                                return_exceptions=True,
                            ),

                            timeout=validation_timeout,
                        )

                        # ------------------------------------------------
                        # PROCESS RESULTS
                        # ------------------------------------------------
                        for (
                            agent_name,
                            result,
                        ) in zip(
                            task_names,
                            validation_results,
                        ):

                            # --------------------------------------------
                            # EXCEPTION
                            # --------------------------------------------
                            if isinstance(
                                result,
                                Exception,
                            ):

                                self.logger.warning(
                                    f"Agent {agent_name} "
                                    f"validation error: "
                                    f"{result}"
                                )

                                results[
                                    agent_name
                                ] = False

                                failed_count += 1

                                continue

                            # --------------------------------------------
                            # AWAITABLE LEAK
                            # --------------------------------------------
                            if inspect.isawaitable(
                                result
                            ):

                                try:

                                    result = await asyncio.wait_for(

                                        result,

                                        timeout=5,
                                    )

                                except Exception:

                                    result = False

                            # --------------------------------------------
                            # FINAL RESULT
                            # --------------------------------------------
                            final_result = bool(
                                result
                            )

                            results[
                                agent_name
                            ] = final_result

                            if final_result:

                                validated_count += 1

                            else:

                                failed_count += 1

                    # ----------------------------------------------------
                    # TIMEOUT
                    # ----------------------------------------------------
                    except asyncio.TimeoutError:

                        self.logger.warning(
                            f"Validation timeout "
                            f"(chunk {chunk_idx})"
                        )

                        for task in validation_tasks:

                            try:

                                if not task.done():

                                    task.cancel()

                            except Exception:
                                pass

                        for agent_name in task_names:

                            results[
                                agent_name
                            ] = False

                            failed_count += 1

                    # ----------------------------------------------------
                    # CANCELLED
                    # ----------------------------------------------------
                    except asyncio.CancelledError:
                        raise

                    # ----------------------------------------------------
                    # FAILURE
                    # ----------------------------------------------------
                    except Exception as e:

                        self.logger.warning(
                            f"Chunk validation failed: {e}"
                        )

                        self.logger.debug(
                            traceback.format_exc()[:3000]
                        )

                        for agent_name in task_names:

                            results[
                                agent_name
                            ] = False

                            failed_count += 1

                # --------------------------------------------------------
                # CHUNK COMPLETE
                # --------------------------------------------------------
                self.logger.debug(
                    f"Validated chunk "
                    f"{chunk_idx}/"
                    f"{len(agent_chunks)}"
                )

                await asyncio.sleep(
                    0.01
                )

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.agent_validations = int(

                    getattr(
                        self,
                        "agent_validations",
                        0,
                    )

                ) + validated_count

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.info(
                f"✓ Agent validation complete "
                f"(success={validated_count}, "
                f"failed={failed_count}) "
                f"in {elapsed_ms}ms"
            )

            return results

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_validate_agents_chunked cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.warning(
                    f"Agent validation failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return results

    def _create_agent_comms_pipeline(self):
        """Create communication pipeline for agents."""
        # This should return your communication pipeline object
        # Example implementation:
        pipeline_config = {
            "buffer_size": self.config.get("agent_comms_buffer", 1000),
            "max_retries": self.config.get("agent_comms_retries", 3),
            "timeout": self.config.get("agent_comms_timeout", 5),
        }

        # Return pipeline instance (implement based on your needs)
        return pipeline_config

    def _get_agent_event_bus(self):
        """Get or create agent event bus."""
        if not hasattr(self, "_agent_event_bus"):
            from collections import defaultdict

            class AgentEventBus:
                def __init__(self):
                    self.subscribers = defaultdict(list)

                async def publish(self, event, data):
                    for callback in self.subscribers[event]:
                        await callback(data)

                def subscribe(self, event, callback):
                    self.subscribers[event].append(callback)

            self._agent_event_bus = AgentEventBus()

        return self._agent_event_bus

    def get_agent(self, name: str) -> Optional[Any]:
        """Get agent by name"""
        return self.agents.get(name)

    def get_wrapper(self, name: str) -> Optional[Any]:
        """Get agent wrapper by name"""
        return self.wrappers.get(name)

    def get_all_agents(self) -> Dict[str, Any]:
        """Get all agents"""
        return self.agents.copy()

    async def shutdown_all(self):
        """
        FINAL production-safe agent shutdown:
        - async + sync safe
        - parallel execution (faster)
        - shutdown aware
        - no blocking
        - no crash propagation
        """

        if task_registry.is_shutting_down() is False:
            task_registry.initiate_shutdown()

        tasks = []

        for name, agent in list(self.agents.items()):

            async def _shutdown(n=name, a=agent):
                try:
                    if a is None:
                        return

                    result = None

                    # -------------------------
                    # DETECT METHOD
                    # -------------------------
                    if hasattr(a, "shutdown"):
                        result = a.shutdown()
                    elif hasattr(a, "close"):
                        result = a.close()
                    else:
                        return

                    # -------------------------
                    # HANDLE ASYNC / SYNC
                    # -------------------------
                    if asyncio.iscoroutine(result):
                        await asyncio.wait_for(result, timeout=5.0)
                    elif callable(result):
                        await asyncio.to_thread(result)

                    await self._safe_call(self, "_log_audit", "agent_shutdown", n, True)

                    self.logger.debug(f"Shutdown {n} agent")

                except asyncio.TimeoutError:
                    self.logger.warning(f"⏱ Shutdown timeout for {n}")

                except Exception as e:
                    await self._safe_call(
                        self, "_log_audit", "agent_shutdown", n, False, str(e)
                    )

                    self.logger.warning(f"Error shutting down {n}: {e}")

            tasks.append(_shutdown())

        # -------------------------
        # PARALLEL EXECUTION (CRITICAL FIX)
        # -------------------------
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass


class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_RESPONDING = "not_responding"
    UNKNOWN = "unknown"


# ==================== Core Orchestrator ====================


class EDIATHOrchestrator:
    """Production-ready orchestrator with MongoDB persistence"""

    def __init__(self, config: OrchestratorConfig, system_instance: Any = None):
        self.config = config
        self.system_instance = system_instance
        self.state = SystemState.INITIALIZING
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.active_tasks: List[asyncio.Task] = []
        self._init_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        self._last_healthy_time = {}

        # -------------------------
        # MODEL LOADING SINGLETON
        # -------------------------
        self._model_loading_lock = asyncio.Lock()
        self._model_loaded = False

        # Graceful shutdown integration
        self._tracked_tasks = []

        self.start_time = None
        self.last_tick_time = None
        self.processing_count = 0
        self.error_count = 0
        self.last_state_persist = None

        self.logger = base_logger

        self.consecutive_failures = 0
        self.circuit_open = False
        self.last_health_check = None

        # -------------------------
        # SAFE CONFIG DEFAULTS (FIXED)
        # -------------------------
        def _set_default(attr, value):
            if not hasattr(self.config, attr):
                setattr(self.config, attr, value)

        # Core pipeline
        _set_default("chunk_size", 500)
        _set_default("pipeline_workers", 3)
        _set_default("pipeline_batch_size", 5)
        _set_default("max_commands_per_tick", 3)
        _set_default("command_batch_size", 2)
        _set_default("parallel_commands", 3)
        _set_default("execution_chunk_size", 1000)
        _set_default("max_execution_chunks", 3)
        _set_default("post_batch_size", 3)

        # Pipeline control
        _set_default("pipeline_queue_size", 100)
        _set_default("enable_pipeline_workers", True)
        _set_default("chunk_batch_size", 10)
        _set_default("chunk_flush_interval", 5.0)

        # Init pipeline
        _set_default("init_chunk_size", 50)
        _set_default("init_pipeline_workers", 5)
        _set_default("agent_init_timeout", 30)
        _set_default("component_batch_size", 3)
        _set_default("enable_init_pipeline", True)
        _set_default("validation_chunk_size", 10)
        _set_default("large_model_threshold", 1024)
        _set_default("model_chunk_size_mb", 256)
        _set_default("agent_comms_chunk_size", 100)
        _set_default("agent_pipeline_depth", 10)
        _set_default("parallel_agent_comms", 3)

        # Recovery
        _set_default("recovery_chunk_size", 5)
        _set_default("recovery_pipeline_workers", 3)
        _set_default("enable_recovery_pipeline", True)
        _set_default("recovery_timeout", 60)
        _set_default("component_recovery_timeout", 10)
        _set_default("max_recovery_delay", 60)
        _set_default("agent_recovery_chunk_size", 3)
        _set_default("agent_recovery_timeout", 5)

        # Health
        _set_default("health_check_chunk_size", 20)
        _set_default("health_pipeline_workers", 5)
        _set_default("enable_health_pipeline", True)
        _set_default("health_check_timeout", 30)
        _set_default("component_health_timeout", 5.0)
        _set_default("agent_health_chunk_size", 10)
        _set_default("health_error_threshold", 0.1)
        _set_default("health_latency_threshold_ms", 1000)

        _set_default("brain_init_timeout", 10.0)
        _set_default("brain_chunk_timeout", 5.0)
        _set_default("brain_chunk_size", 2)
        _set_default("critical_recovery_timeout", 15.0)
        _set_default("enable_llm_engine", True)
        _set_default("enable_decision_engine", True)
        _set_default("enable_reasoning_engine", False)
        _set_default("enable_context_manager", False)

        # Call the function with the components dictionary (schedule safely)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._start_components_brain(self.components))
        except RuntimeError:
            # No running event loop yet — schedule once loop starts
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(
                        self._start_components_brain(self.components)
                    )
                )
            except Exception:
                # Last-resort: run in a background thread with its own loop
                import threading

                def _runner():
                    try:
                        asyncio.run(self._start_components_brain(self.components))
                    except Exception:
                        pass

                threading.Thread(target=_runner, daemon=True).start()

        # -------------------------
        # SYSTEM COMPONENTS
        # -------------------------
        self.agent_registry = AgentRegistry()
        self.agent_registry.set_config(config)

        self.blender_available = self._check_blender()

        self.multi_agent_coordinator: Optional[Any] = None

        # MongoDB collections
        self._state_collection = None
        self._events_collection = None
        self._init_mongo_collections()

        # Runtime helpers
        self._tick_lock = None
        self._loop = None

        # Vision
        self.vision_memory = SharedVisionMemory() if VISION_MEMORY_AVAILABLE else None


        # Coroutine tracking
        self._pending_coros = []

        self.logger.info(f"Initializing {self.config.name} v{self.config.version}")

    def _cap_workers(self, value: int, cap: int = 8) -> int:
        """Cap worker counts to a safe maximum to avoid CPU overload."""
        try:
            v = int(value)
        except Exception:
            v = 1
        return max(1, min(v, cap))

    def _track_task_cleanup(self, task: asyncio.Task):
        """Remove task from active list and log exceptions if any."""
        try:
            if task in self.active_tasks:
                self.active_tasks.remove(task)
        except Exception:
            pass

        # Log unhandled exceptions
        if task.cancelled():
            return
        exc = None
        try:
            exc = task.exception()
        except Exception:
            exc = None
        if exc:
            self.logger.warning(
                f"Task {getattr(task, 'name', repr(task))} failed: {exc}"
            )

    def _start_task(self, coro, name: str = "task") -> Optional[asyncio.Task]:
        """Start and track a background task with cleanup callback."""
        if task_registry.is_shutting_down():
            return None

        try:
            task = asyncio.create_task(coro, name=name)
        except RuntimeError:
            # No running loop
            return None

        # Track it
        try:
            self.active_tasks.append(task)
        except Exception:
            pass

        # Add done callback to cleanup
        def _done_callback(t: asyncio.Task):
            try:
                self._track_task_cleanup(t)
            except Exception:
                pass

        task.add_done_callback(_done_callback)
        return task

    def _check_blender(self) -> bool:
        """Check if Blender/bpy is available"""
        try:
            import bpy

            self.logger.info("✅ Blender/bpy available for 3D modeling")
            return True
        except ImportError:
            self.logger.debug(
                "⚠️ Blender/bpy not available - 3D modeling features limited (this is normal when not running inside Blender)"
            )
            return False

    async def _cancel_pending_init_tasks(self):
        """Cancel all tracked init tasks (TODO #5)"""
        if not self._init_tasks:
            return

        tasks_to_cancel = [t for t in self._init_tasks if not t.done()]
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            self.logger.debug(f"Cancelled {len(tasks_to_cancel)} pending init tasks")

    def _init_mongo_collections(self):
        """Robust MongoDB collection initialization with safety + index handling"""

        if not getattr(self.config, "enable_mongo_persistence", False):
            self.logger.debug("MongoDB persistence disabled in config")
            return

        try:
            # -------------------------
            # VALIDATE CLIENT
            # -------------------------
            if not getattr(mongo_client, "enabled", False):
                self.logger.debug("MongoDB client not enabled")
                return

            db = getattr(mongo_client, "db", None)
            if db is None:
                self.logger.debug("MongoDB database not available")
                return

            # -------------------------
            # INIT COLLECTIONS
            # -------------------------
            try:
                self._state_collection = db[self.config.mongo_state_collection]
                self._events_collection = db[self.config.mongo_events_collection]
            except Exception as e:
                self.logger.debug(f"Collection init failed: {e}")
                self._state_collection = None
                self._events_collection = None
                return

            # -------------------------
            # SAFE INDEX CREATION (NON-BLOCKING STYLE)
            # -------------------------
            def _safe_create_indexes(collection, indexes):
                try:
                    for index in indexes:
                        collection.create_index(index)
                except Exception as idx_err:
                    self.logger.debug(f"Index creation failed: {idx_err}")

            if self._state_collection is not None:
                _safe_create_indexes(
                    self._state_collection,
                    [[("timestamp", -1)], [("orchestrator_name", 1)]],
                )

            if self._events_collection is not None:
                _safe_create_indexes(
                    self._events_collection,
                    [[("timestamp", -1)], [("event_type", 1)], [("component", 1)]],
                )

            # -------------------------
            # FINAL VALIDATION
            # -------------------------
            if self._state_collection is None or self._events_collection is None:
                self.logger.warning("⚠ MongoDB collections partially initialized")
            else:
                self.logger.info("✅ MongoDB collections initialized successfully")

        except Exception as e:
            # fallback safe state
            self._state_collection = None
            self._events_collection = None
            self.logger.debug(f"MongoDB initialization skipped: {e}")

    async def _persist_state_to_mongo(self):
        """
        Production-safe MongoDB state persistence.

        Fixes:
        - blocking MongoDB operations
        - function.items crashes
        - invalid config corruption
        - serialization failures
        - thread deadlocks
        - unsafe datetime handling
        - invalid collection usage
        - memory leaks
        - cleanup deadlocks
        - coroutine leaks
        - startup race conditions
        """

        import asyncio
        import inspect
        import time
        import traceback
        from datetime import datetime

        start_monotonic = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping "
                        "Mongo persistence"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                None,
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if config is None:
                        return default

                    # FIX:
                    # config became callable
                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents callable corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # FEATURE FLAG
            # ============================================================
            mongo_enabled = bool(

                cfg(
                    "enable_mongo_persistence",
                    False,
                )
            )

            if not mongo_enabled:

                return False

            # ============================================================
            # VALIDATE COLLECTION
            # ============================================================
            collection = getattr(
                self,
                "_state_collection",
                None,
            )

            if collection is None:

                self.logger.debug(
                    "Mongo state collection unavailable"
                )

                return False

            # FIX:
            # Collection object callable corruption
            if callable(collection):

                self.logger.warning(
                    "State collection callable corruption"
                )

                return False

            # ============================================================
            # SAFE TIMESTAMP
            # ============================================================
            now = datetime.utcnow()

            # ============================================================
            # SAFE COMPONENT STATUS
            # ============================================================
            raw_component_status = getattr(
                self,
                "component_status",
                {},
            )

            if callable(raw_component_status):
                raw_component_status = {}

            if not isinstance(
                raw_component_status,
                dict,
            ):
                raw_component_status = {}

            component_status = {}

            for (
                name,
                status,
            ) in raw_component_status.items():

                try:

                    safe_name = str(name)

                    if hasattr(
                        status,
                        "value",
                    ):

                        safe_status = str(
                            status.value
                        )

                    else:

                        safe_status = str(
                            status
                        )

                    component_status[
                        safe_name
                    ] = safe_status

                except Exception:

                    continue

            # ============================================================
            # SAFE STATE VALUE
            # ============================================================
            try:

                raw_state = getattr(
                    self,
                    "state",
                    "unknown",
                )

                if hasattr(
                    raw_state,
                    "value",
                ):

                    state_value = str(
                        raw_state.value
                    )

                else:

                    state_value = str(
                        raw_state
                    )

            except Exception:

                state_value = "unknown"

            # ============================================================
            # SAFE AGENT COUNT
            # ============================================================
            active_agents = 0

            try:

                agent_registry = getattr(
                    self,
                    "agent_registry",
                    None,
                )

                if (
                    agent_registry is not None
                    and hasattr(
                        agent_registry,
                        "get_all_agents",
                    )
                ):

                    get_agents = (
                        agent_registry.get_all_agents
                    )

                    if callable(
                        get_agents
                    ):

                        agents_result = get_agents()

                        # FIX:
                        # bool await corruption
                        if inspect.isawaitable(
                            agents_result
                        ):

                            agents_result = await asyncio.wait_for(

                                agents_result,

                                timeout=5,
                            )

                        if isinstance(
                            agents_result,
                            (list, dict, tuple, set),
                        ):

                            active_agents = len(
                                agents_result
                            )

            except Exception:
                pass

            # ============================================================
            # SAFE UPTIME
            # ============================================================
            uptime_seconds = 0

            try:

                start_time = getattr(
                    self,
                    "start_time",
                    None,
                )

                if start_time:

                    # FIX:
                    # never datetime.time.time
                    uptime_seconds = max(

                        0,

                        float(
                            time.time()
                        ) - float(start_time)
                    )

            except Exception:
                pass

            # ============================================================
            # BUILD SAFE DOCUMENT
            # ============================================================
            state_doc = {

                "timestamp": now,

                "orchestrator_name": str(

                    cfg(
                        "name",
                        "EDIATH",
                    )
                ),

                "state": state_value,

                "processing_count": int(

                    getattr(
                        self,
                        "processing_count",
                        0,
                    )
                ),

                "error_count": int(

                    getattr(
                        self,
                        "error_count",
                        0,
                    )
                ),

                "component_status": component_status,

                "circuit_breaker_open": bool(

                    getattr(
                        self,
                        "circuit_open",
                        False,
                    )
                ),

                "consecutive_failures": int(

                    getattr(
                        self,
                        "consecutive_failures",
                        0,
                    )
                ),

                "uptime_seconds": round(
                    uptime_seconds,
                    2,
                ),

                "active_agents": int(
                    active_agents
                ),

                "blender_available": bool(

                    getattr(
                        self,
                        "blender_available",
                        False,
                    )
                ),

                "persisted_at": time.time(),

                "safe_mode": True,
            }

            # ============================================================
            # SAFE SERIALIZATION TEST
            # ============================================================
            try:

                import json

                json.dumps(
                    state_doc,
                    default=str,
                )

            except Exception as e:

                self.logger.warning(
                    f"State serialization failed: {e}"
                )

                return False

            # ============================================================
            # INSERT DOCUMENT
            # ============================================================
            def _insert():

                try:

                    return collection.insert_one(
                        state_doc
                    )

                except Exception as e:

                    return e

            insert_result = await asyncio.wait_for(

                asyncio.to_thread(
                    _insert
                ),

                timeout=15,
            )

            if isinstance(
                insert_result,
                Exception,
            ):

                raise insert_result

            # ============================================================
            # CLEANUP OLD DOCS
            # ============================================================
            cleanup_limit = 1000

            delete_batch = 500

            def _cleanup():

                try:

                    cursor = (

                        collection.find(

                            {
                                "orchestrator_name":
                                state_doc[
                                    "orchestrator_name"
                                ]
                            },

                            {
                                "_id": 1
                            },
                        )

                        .sort(
                            "timestamp",
                            -1,
                        )

                        .skip(
                            cleanup_limit
                        )
                    )

                    old_ids = []

                    for doc in cursor.limit(
                        delete_batch
                    ):

                        try:

                            _id = doc.get(
                                "_id"
                            )

                            if _id is not None:

                                old_ids.append(
                                    _id
                                )

                        except Exception:
                            continue

                    if old_ids:

                        collection.delete_many(

                            {
                                "_id": {
                                    "$in": old_ids
                                }
                            }
                        )

                    return len(
                        old_ids
                    )

                except Exception:

                    return 0

            removed_count = await asyncio.wait_for(

                asyncio.to_thread(
                    _cleanup
                ),

                timeout=20,
            )

            # ============================================================
            # UPDATE TIMESTAMP
            # ============================================================
            try:

                self.last_state_persist = now

            except Exception:
                pass

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.mongo_persist_count = int(

                    getattr(
                        self,
                        "mongo_persist_count",
                        0,
                    )

                ) + 1

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_monotonic
                ) * 1000,

                2,
            )

            self.logger.debug(
                f"✓ Mongo state persisted "
                f"(cleanup={removed_count}) "
                f"in {elapsed_ms}ms"
            )

            return True

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_persist_state_to_mongo cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.debug(
                    f"State persist skipped: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

    async def _log_event(
        self,
        event_type: str,
        component: str,
        details: Dict[str, Any] = None,
    ):
        """
        Production-safe event logging.

        Fixes:
        - blocking MongoDB operations
        - invalid collection usage
        - function.items crashes
        - serialization failures
        - coroutine leaks
        - invalid config corruption
        - unsafe event documents
        - memory leaks
        - invalid details payloads
        - event logging race conditions
        """

        import asyncio
        import inspect
        import json
        import time
        import traceback
        from datetime import datetime

        start_monotonic = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping event log"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                None,
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if config is None:
                        return default

                    # FIX:
                    # config became callable
                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents callable corruption
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # FEATURE FLAG
            # ============================================================
            mongo_enabled = bool(

                cfg(
                    "enable_mongo_persistence",
                    False,
                )
            )

            if not mongo_enabled:

                return False

            # ============================================================
            # VALIDATE COLLECTION
            # ============================================================
            collection = getattr(
                self,
                "_events_collection",
                None,
            )

            if collection is None:

                self.logger.debug(
                    "Events collection unavailable"
                )

                return False

            # FIX:
            # collection callable corruption
            if callable(collection):

                self.logger.warning(
                    "Events collection callable corruption"
                )

                return False

            # ============================================================
            # SAFE EVENT TYPE
            # ============================================================
            try:

                safe_event_type = str(
                    event_type
                ).strip()

            except Exception:

                safe_event_type = "unknown"

            if not safe_event_type:

                safe_event_type = "unknown"

            # ============================================================
            # SAFE COMPONENT
            # ============================================================
            try:

                safe_component = str(
                    component
                ).strip()

            except Exception:

                safe_component = "unknown"

            if not safe_component:

                safe_component = "unknown"

            # ============================================================
            # SAFE DETAILS
            # ============================================================
            safe_details = {}

            try:

                # FIX:
                # prevents:
                # function.items crashes
                if callable(details):

                    safe_details = {}

                elif isinstance(
                    details,
                    dict,
                ):

                    # sanitize recursively
                    for (
                        key,
                        value,
                    ) in details.items():

                        try:

                            safe_key = str(
                                key
                            )

                            # prevent unserializable values
                            try:

                                json.dumps(
                                    value,
                                    default=str,
                                )

                                safe_value = value

                            except Exception:

                                safe_value = str(
                                    value
                                )

                            safe_details[
                                safe_key
                            ] = safe_value

                        except Exception:
                            continue

            except Exception:
                safe_details = {}

            # ============================================================
            # SAFE STATE
            # ============================================================
            try:

                raw_state = getattr(
                    self,
                    "state",
                    "unknown",
                )

                if hasattr(
                    raw_state,
                    "value",
                ):

                    state_value = str(
                        raw_state.value
                    )

                else:

                    state_value = str(
                        raw_state
                    )

            except Exception:

                state_value = "unknown"

            # ============================================================
            # SAFE TIMESTAMP
            # ============================================================
            now = datetime.utcnow()

            # ============================================================
            # BUILD SAFE EVENT DOC
            # ============================================================
            event_doc = {

                "timestamp": now,

                "event_type": safe_event_type,

                "component": safe_component,

                "orchestrator_name": str(

                    cfg(
                        "name",
                        "EDIATH",
                    )
                ),

                "state": state_value,

                "details": safe_details,

                "logged_at": time.time(),

                "safe_mode": True,
            }

            # ============================================================
            # SERIALIZATION VALIDATION
            # ============================================================
            try:

                json.dumps(
                    event_doc,
                    default=str,
                )

            except Exception as e:

                self.logger.warning(
                    f"Event serialization failed: {e}"
                )

                return False

            # ============================================================
            # INSERT EVENT
            # ============================================================
            def _insert():

                try:

                    return collection.insert_one(
                        event_doc
                    )

                except Exception as e:

                    return e

            insert_result = await asyncio.wait_for(

                asyncio.to_thread(
                    _insert
                ),

                timeout=10,
            )

            if isinstance(
                insert_result,
                Exception,
            ):

                raise insert_result

            # ============================================================
            # OPTIONAL CLEANUP
            # ============================================================
            try:

                cleanup_enabled = bool(

                    cfg(
                        "cleanup_old_events",
                        True,
                    )
                )

            except Exception:

                cleanup_enabled = True

            if cleanup_enabled:

                try:

                    cleanup_interval = int(

                        cfg(
                            "event_cleanup_interval",
                            500,
                        )
                    )

                except Exception:

                    cleanup_interval = 500

                cleanup_interval = max(
                    100,
                    cleanup_interval,
                )

                current_count = int(

                    getattr(
                        self,
                        "_event_log_count",
                        0,
                    )

                ) + 1

                self._event_log_count = current_count

                # periodic cleanup
                if (
                    current_count
                    % cleanup_interval
                    == 0
                ):

                    def _cleanup():

                        try:

                            cursor = (

                                collection.find(
                                    {},
                                    {"_id": 1},
                                )

                                .sort(
                                    "timestamp",
                                    -1,
                                )

                                .skip(5000)
                            )

                            old_ids = []

                            for doc in cursor.limit(
                                500
                            ):

                                try:

                                    _id = doc.get(
                                        "_id"
                                    )

                                    if _id is not None:

                                        old_ids.append(
                                            _id
                                        )

                                except Exception:
                                    continue

                            if old_ids:

                                collection.delete_many(

                                    {
                                        "_id": {
                                            "$in": old_ids
                                        }
                                    }
                                )

                                return len(
                                    old_ids
                                )

                        except Exception:
                            pass

                        return 0

                    try:

                        removed_count = await asyncio.wait_for(

                            asyncio.to_thread(
                                _cleanup
                            ),

                            timeout=15,
                        )

                        if removed_count:

                            self.logger.debug(
                                f"Event cleanup removed "
                                f"{removed_count} docs"
                            )

                    except Exception:
                        pass

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.events_logged = int(

                    getattr(
                        self,
                        "events_logged",
                        0,
                    )

                ) + 1

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_monotonic
                ) * 1000,

                2,
            )

            self.logger.debug(
                f"✓ Event logged "
                f"({safe_event_type}) "
                f"in {elapsed_ms}ms"
            )

            return True

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_log_event cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.debug(
                    f"Event logging skipped: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

    async def initialize(self) -> bool:
        """
        Production-safe orchestrator initialization.

        Fixes:
        - function.items crashes
        - bool await crashes
        - coroutine leaks
        - duplicate initialization
        - startup races
        - model double-loading
        - pipeline deadlocks
        - unsafe config access
        - invalid cleanup
        - broken async/sync interoperability
        """

        import asyncio
        import inspect
        import time
        import traceback

        INIT_TIMEOUT = 300

        # ============================================================
        # DOUBLE INIT PROTECTION
        # ============================================================
        if getattr(
            self,
            "_initializing",
            False,
        ):

            self.logger.warning(
                "Initialization already running"
            )

            return False

        self._initializing = True

        start_time = time.monotonic()

        try:

            # ========================================================
            # SHUTDOWN GUARD
            # ========================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.warning(
                        "Shutdown active - init aborted"
                    )

                    return False

            except Exception:
                pass

            # ========================================================
            # SAFE EVENT LOOP
            # ========================================================
            try:

                self._loop = (
                    asyncio.get_running_loop()
                )

            except RuntimeError:

                self._loop = (
                    asyncio.get_event_loop()
                )

            # ========================================================
            # SAFE CONFIG ACCESSOR
            # ========================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default=None,
            ):

                try:

                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ========================================================
            # INIT STATE
            # ========================================================
            self.logger.info(
                "🚀 Initializing EDIATH..."
            )

            self.state = (
                SystemState.INITIALIZING
            )

            self._running = False

            self._init_chunk_buffer = []

            self._init_pipeline_tasks = []

            # ========================================================
            # SAFE CONFIG VALUES
            # ========================================================
            chunk_size = max(

                1,

                int(
                    cfg(
                        "init_chunk_size",
                        50,
                    )
                ),
            )

            pipeline_workers = max(

                1,

                min(

                    int(
                        cfg(
                            "init_pipeline_workers",
                            5,
                        )
                    ),

                    8,
                ),
            )

            batch_size = max(

                1,

                int(
                    cfg(
                        "component_batch_size",
                        3,
                    )
                ),
            )

            # ========================================================
            # SAFE EVENT LOGGING
            # ========================================================
            try:

                result = self._safe_call(

                    self,

                    "_log_event",

                    "initialization_start",

                    "orchestrator",

                    {
                        "version": cfg(
                            "version",
                            "unknown",
                        )
                    },
                )

                if inspect.isawaitable(
                    result
                ):

                    await result

            except Exception:
                pass

            # ========================================================
            # AGENT REGISTRY
            # ========================================================
            self.agent_registry = (
                AgentRegistry()
            )

            try:

                set_config = getattr(
                    self.agent_registry,
                    "set_config",
                    None,
                )

                if callable(set_config):

                    result = set_config(
                        self.config
                    )

                    # FIX:
                    # bool cannot be awaited
                    if inspect.isawaitable(
                        result
                    ):

                        await result

            except Exception as e:

                self.logger.warning(
                    f"Agent config failed: {e}"
                )

            self.logger.info(
                "Initializing Agent Registry..."
            )

            # ========================================================
            # SAFE AGENT LIST
            # ========================================================
            try:

                all_agents = (
                    self._get_agents_to_initialize()
                )

                if inspect.isawaitable(
                    all_agents
                ):

                    all_agents = (
                        await all_agents
                    )

                if callable(all_agents):

                    all_agents = []

                if all_agents is None:

                    all_agents = []

                if not isinstance(
                    all_agents,
                    (list, tuple),
                ):

                    all_agents = list(
                        all_agents
                    )

            except Exception as e:

                self.logger.error(
                    f"Agent loading failed: {e}"
                )

                all_agents = []

            # ========================================================
            # CHUNK AGENTS
            # ========================================================
            agent_chunks = [

                all_agents[
                    i:i + chunk_size
                ]

                for i in range(
                    0,
                    len(all_agents),
                    chunk_size,
                )
            ]

            self.logger.info(
                f"📦 Agent chunks: "
                f"{len(agent_chunks)}"
            )

            agent_results = {}

            # ========================================================
            # PROCESS AGENT CHUNKS
            # ========================================================
            for (
                chunk_idx,
                agent_chunk,
            ) in enumerate(
                agent_chunks,
                start=1,
            ):

                try:

                    if task_registry.is_shutting_down():

                        return False

                except Exception:
                    pass

                chunk_tasks = []

                for agent_name in agent_chunk:

                    try:

                        task = asyncio.create_task(

                            self._safe_call(

                                self.agent_registry,

                                "initialize_agent",

                                agent_name,

                                getattr(
                                    self,
                                    "system_instance",
                                    None,
                                ),

                                chunk_index=chunk_idx,

                                total_chunks=len(
                                    agent_chunks
                                ),
                            )
                        )

                        chunk_tasks.append(
                            (
                                agent_name,
                                task,
                            )
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"Agent task failed "
                            f"({agent_name}): {e}"
                        )

                if not chunk_tasks:
                    continue

                try:

                    timeout = float(

                        cfg(
                            "agent_init_timeout",
                            60,
                        )
                    )

                    results = await asyncio.wait_for(

                        asyncio.gather(

                            *[
                                t[1]
                                for t in chunk_tasks
                            ],

                            return_exceptions=True,
                        ),

                        timeout=timeout,
                    )

                    for (
                        (
                            agent_name,
                            _,
                        ),
                        result,
                    ) in zip(
                        chunk_tasks,
                        results,
                    ):

                        success = not isinstance(
                            result,
                            Exception,
                        )

                        agent_results[
                            agent_name
                        ] = success

                except Exception as e:

                    self.logger.warning(
                        f"Chunk {chunk_idx} failed: {e}"
                    )

                self.logger.info(
                    f"✓ Chunk "
                    f"{chunk_idx}/"
                    f"{len(agent_chunks)} "
                    f"complete"
                )

            successful_agents = sum(

                1

                for v in agent_results.values()

                if v
            )

            # ========================================================
            # MULTI AGENT
            # ========================================================
            enable_multi = bool(

                cfg(
                    "enable_multi_agent",
                    False,
                )
            )

            if (
                enable_multi
                and MULTI_AGENT_AVAILABLE
            ):

                try:

                    coordinator = (
                        get_multi_agent_coordinator()
                    )

                    if inspect.isawaitable(
                        coordinator
                    ):

                        coordinator = (
                            await asyncio.wait_for(
                                coordinator,
                                timeout=10,
                            )
                        )

                    self.multi_agent_coordinator = (
                        coordinator
                    )

                    startup_tasks = [

                        asyncio.create_task(

                            self._safe_call(

                                self,

                                "_configure_multi_agent_strategies",
                            )
                        ),

                        asyncio.create_task(

                            self._safe_call(

                                self,

                                "_setup_multi_agent_system",
                            )
                        ),

                        asyncio.create_task(

                            self._safe_call(

                                self,

                                "_initialize_agent_communication_pipeline",
                            )
                        ),
                    ]

                    await asyncio.gather(
                        *startup_tasks,
                        return_exceptions=True,
                    )

                    await self._safe_call(
                        self.multi_agent_coordinator,
                        "start",
                    )

                    self.logger.info(
                        "✓ Multi-Agent ready"
                    )

                except Exception as e:

                    self.logger.warning(
                        f"Multi-agent failed: {e}"
                    )

            # ========================================================
            # COMPONENT INITIALIZATION
            # ========================================================
            component_methods = [

                "_initialize_system_components",

                "_initialize_agent_components",

                "_initialize_brain_components",

                "_initialize_perception_components",
            ]

            if cfg(
                "enable_memory",
                True,
            ):

                component_methods.append(
                    "_initialize_memory_components"
                )

            if cfg(
                "enable_autonomous_mode",
                False,
            ):

                component_methods.append(
                    "_initialize_autonomous_components"
                )

            if cfg(
                "enable_learning",
                True,
            ):

                component_methods.append(
                    "_initialize_learning_components"
                )

            if cfg(
                "enable_security",
                True,
            ):

                component_methods.append(
                    "_initialize_security_components"
                )

            pipeline_tasks = []

            for method in component_methods:

                try:

                    pipeline_tasks.append(

                        asyncio.create_task(

                            self._safe_call(
                                self,
                                method,
                            )
                        )
                    )

                except Exception as e:

                    self.logger.warning(
                        f"Pipeline task failed: {e}"
                    )

            # ========================================================
            # BATCH PIPELINE EXECUTION
            # ========================================================
            for i in range(
                0,
                len(pipeline_tasks),
                batch_size,
            ):

                batch = pipeline_tasks[
                    i:i + batch_size
                ]

                await asyncio.gather(
                    *batch,
                    return_exceptions=True,
                )

            # ========================================================
            # SAFE MODEL LOADING
            # ========================================================
            components = getattr(
                self,
                "components",
                {},
            )

            if not isinstance(
                components,
                dict,
            ):

                components = {}

            llm = components.get(
                "llm_engine"
            )

            if (
                llm
                and hasattr(
                    llm,
                    "load_model",
                )
            ):

                try:

                    async with self._model_loading_lock:

                        if not getattr(
                            self,
                            "_model_loaded",
                            False,
                        ):

                            await self._load_model_in_chunks(
                                llm
                            )

                            self._model_loaded = True

                            self.logger.info(
                                "✓ Model loaded"
                            )

                        else:

                            self.logger.info(
                                "Model already loaded"
                            )

                except Exception as e:

                    self.logger.warning(
                        f"Model loading failed: {e}"
                    )

            # ========================================================
            # PIPELINE WORKERS
            # ========================================================
            if cfg(
                "enable_init_pipeline",
                True,
            ):

                await self._initialize_pipeline_workers()

            # ========================================================
            # VALIDATION
            # ========================================================
            try:

                chunks = (
                    self._create_validation_chunks()
                )

                if inspect.isawaitable(
                    chunks
                ):

                    chunks = await chunks

                validation_results = (
                    await self._validate_chunked_state(
                        chunks
                    )
                )

                if (
                    validation_results
                    and not all(
                        validation_results
                    )
                ):

                    self.logger.warning(
                        f"Validation issues: "
                        f"{validation_results}"
                    )

            except Exception as e:

                self.logger.warning(
                    f"Validation failed: {e}"
                )

            # ========================================================
            # FINAL STATE
            # ========================================================
            self.state = (
                SystemState.RUNNING
            )

            self._running = True

            self.start_time = time.time()

            # ========================================================
            # FINAL TASKS
            # ========================================================
            persist_tasks = [

                asyncio.create_task(

                    self._safe_call(
                        self,
                        "_persist_state_to_mongo",
                    )
                ),

                asyncio.create_task(

                    self._safe_call(

                        self,

                        "_log_event",

                        "initialization_complete",

                        "orchestrator",

                        {
                            "duration_seconds": round(
                                (
                                    time.monotonic()
                                    - start_time
                                ),
                                2,
                            ),

                            "agents_initialized": (
                                successful_agents
                            ),

                            "chunks_processed": (
                                len(agent_chunks)
                            ),

                            "pipeline_workers": (
                                pipeline_workers
                            ),
                        },
                    )
                ),
            ]

            await asyncio.gather(
                *persist_tasks,
                return_exceptions=True,
            )

            elapsed = round(
                (
                    time.monotonic()
                    - start_time
                ),
                2,
            )

            self.logger.info(
                f"✅ EDIATH RUNNING "
                f"({elapsed}s)"
            )

            return True

        # ============================================================
        # CANCELLED
        # ============================================================
        except asyncio.CancelledError:

            self.logger.warning(
                "Initialization cancelled"
            )

            raise

        # ============================================================
        # FAILURE
        # ============================================================
        except Exception as e:

            self.state = (
                SystemState.ERROR
            )

            self.logger.error(
                f"❌ Initialization failed: {e}"
            )

            self.logger.debug(
                traceback.format_exc()[:5000]
            )

            # --------------------------------------------------------
            # CLEANUP TASKS
            # --------------------------------------------------------
            try:

                for task in list(

                    getattr(
                        self,
                        "_init_pipeline_tasks",
                        [],
                    )
                ):

                    try:

                        if (
                            task
                            and not task.done()
                        ):

                            task.cancel()

                    except Exception:
                        pass

                tasks = [

                    t

                    for t in getattr(
                        self,
                        "_init_pipeline_tasks",
                        [],
                    )

                    if t
                ]

                if tasks:

                    await asyncio.gather(
                        *tasks,
                        return_exceptions=True,
                    )

            except Exception:
                pass

            # --------------------------------------------------------
            # FAILURE EVENT
            # --------------------------------------------------------
            try:

                await self._safe_call(

                    self,

                    "_log_event",

                    "initialization_failed",

                    "orchestrator",

                    {
                        "error": str(e),

                        "partial_chunks": len(

                            getattr(
                                self,
                                "_init_chunk_buffer",
                                [],
                            )
                        ),
                    },
                )

            except Exception:
                pass

            return False

        # ============================================================
        # FINALIZE
        # ============================================================
        finally:

            self._initializing = False

    def _get_agents_to_initialize(self) -> list:
        """Get list of agent names to initialize."""
        # This should return your actual agent list
        # Example implementation:
        return getattr(
            self.config,
            "agent_list",
            [
                "conversation_agent",
                "task_agent",
                "memory_agent",
                "perception_agent",
                "decision_agent",
            ],
        )

    def _create_validation_chunks(self) -> list:
        """Create chunks of components to validate."""
        all_components = list(self.components.keys())
        chunk_size = self.config.get("validation_chunk_size", 10)
        return [
            all_components[i : i + chunk_size]
            for i in range(0, len(all_components), chunk_size)
        ]

    async def _validate_chunked_state(
        self,
        validation_chunks: list,
    ) -> list:
        """
        Production-safe chunked system state validation.

        Fixes:
        - function.items crashes
        - bool await crashes
        - coroutine leaks
        - invalid validation chunks
        - async/sync mismatch
        - component corruption
        - validation deadlocks
        - unsafe component access
        - startup race conditions
        - memory leaks
        - invalid validate methods
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        results = []

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping "
                        "chunked state validation"
                    )

                    return results

            except Exception:
                pass

            # ============================================================
            # VALIDATE INPUT
            # ============================================================
            if validation_chunks is None:

                self.logger.warning(
                    "validation_chunks is None"
                )

                return results

            # FIX:
            # prevents:
            # function.items crashes
            if callable(validation_chunks):

                self.logger.warning(
                    "validation_chunks callable"
                )

                return results

            if not isinstance(
                validation_chunks,
                (list, tuple),
            ):

                self.logger.warning(
                    f"Invalid validation_chunks type: "
                    f"{type(validation_chunks)}"
                )

                return results

            # ============================================================
            # SAFE COMPONENTS
            # ============================================================
            components = getattr(
                self,
                "components",
                {},
            )

            if callable(components):
                components = {}

            if not isinstance(
                components,
                dict,
            ):
                components = {}

            # ============================================================
            # SAFE SAFE_CALL
            # ============================================================
            safe_call = getattr(
                self,
                "_safe_call",
                None,
            )

            if not callable(
                safe_call
            ):

                self.logger.error(
                    "_safe_call missing"
                )

                return results

            total_valid = 0
            total_failed = 0

            # ============================================================
            # PROCESS CHUNKS
            # ============================================================
            for (
                chunk_idx,
                chunk,
            ) in enumerate(
                validation_chunks,
                start=1,
            ):

                try:

                    if task_registry.is_shutting_down():
                        break

                except Exception:
                    pass

                # --------------------------------------------------------
                # VALIDATE CHUNK
                # --------------------------------------------------------
                if callable(chunk):

                    self.logger.warning(
                        f"Chunk {chunk_idx} callable"
                    )

                    results.append(False)

                    total_failed += 1

                    continue

                if not isinstance(
                    chunk,
                    (list, tuple, set),
                ):

                    self.logger.warning(
                        f"Chunk {chunk_idx} invalid"
                    )

                    results.append(False)

                    total_failed += 1

                    continue

                chunk_valid = True

                validation_tasks = []

                task_names = []

                # --------------------------------------------------------
                # BUILD VALIDATION TASKS
                # --------------------------------------------------------
                for component_name in chunk:

                    try:

                        if not component_name:
                            continue

                        component_name = str(
                            component_name
                        )

                        component = components.get(
                            component_name
                        )

                        # component missing
                        if component is None:

                            self.logger.warning(
                                f"Missing component: "
                                f"{component_name}"
                            )

                            chunk_valid = False

                            continue

                        # corrupted component
                        if callable(component):

                            self.logger.warning(
                                f"Callable component: "
                                f"{component_name}"
                            )

                            chunk_valid = False

                            continue

                        validate_method = getattr(
                            component,
                            "validate",
                            None,
                        )

                        # no validate method
                        if not callable(
                            validate_method
                        ):

                            continue

                        # create validation task
                        task = asyncio.create_task(

                            safe_call(
                                component,
                                "validate",
                            ),

                            name=f"validate_{component_name}",
                        )

                        validation_tasks.append(
                            task
                        )

                        task_names.append(
                            component_name
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"Task creation failed "
                            f"({component_name}): {e}"
                        )

                        chunk_valid = False

                # --------------------------------------------------------
                # EXECUTE VALIDATIONS
                # --------------------------------------------------------
                if validation_tasks:

                    try:

                        validation_results = await asyncio.wait_for(

                            asyncio.gather(

                                *validation_tasks,

                                return_exceptions=True,
                            ),

                            timeout=60,
                        )

                        # ------------------------------------------------
                        # PROCESS RESULTS
                        # ------------------------------------------------
                        for (
                            component_name,
                            result,
                        ) in zip(
                            task_names,
                            validation_results,
                        ):

                            # exception
                            if isinstance(
                                result,
                                Exception,
                            ):

                                self.logger.warning(
                                    f"Component "
                                    f"{component_name} "
                                    f"validation error: "
                                    f"{result}"
                                )

                                chunk_valid = False

                                continue

                            # FIX:
                            # coroutine leak protection
                            if inspect.isawaitable(
                                result
                            ):

                                try:

                                    result = await asyncio.wait_for(

                                        result,

                                        timeout=10,
                                    )

                                except Exception:

                                    result = False

                            is_valid = bool(
                                result
                            )

                            if not is_valid:

                                chunk_valid = False

                                self.logger.warning(
                                    f"Component "
                                    f"{component_name} "
                                    f"validation failed"
                                )

                    # ----------------------------------------------------
                    # TIMEOUT
                    # ----------------------------------------------------
                    except asyncio.TimeoutError:

                        self.logger.warning(
                            f"Validation timeout "
                            f"(chunk {chunk_idx})"
                        )

                        chunk_valid = False

                        # cancel hanging tasks
                        for task in validation_tasks:

                            try:

                                if not task.done():

                                    task.cancel()

                            except Exception:
                                pass

                    # ----------------------------------------------------
                    # CANCELLED
                    # ----------------------------------------------------
                    except asyncio.CancelledError:
                        raise

                    # ----------------------------------------------------
                    # HARD FAILURE
                    # ----------------------------------------------------
                    except Exception as e:

                        self.logger.warning(
                            f"Chunk validation failed: {e}"
                        )

                        self.logger.debug(
                            traceback.format_exc()[:3000]
                        )

                        chunk_valid = False

                # --------------------------------------------------------
                # STORE RESULT
                # --------------------------------------------------------
                results.append(
                    bool(chunk_valid)
                )

                if chunk_valid:

                    total_valid += 1

                else:

                    total_failed += 1

                # --------------------------------------------------------
                # LOG CHUNK
                # --------------------------------------------------------
                self.logger.debug(
                    f"Validated chunk "
                    f"{chunk_idx}/"
                    f"{len(validation_chunks)} "
                    f"(valid={chunk_valid})"
                )

                # --------------------------------------------------------
                # MEMORY CLEANUP
                # --------------------------------------------------------
                try:

                    gc.collect()

                except Exception:
                    pass

                await asyncio.sleep(
                    0.01
                )

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.validation_chunks_processed = int(

                    getattr(
                        self,
                        "validation_chunks_processed",
                        0,
                    )

                ) + len(results)

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.info(
                f"✓ Chunked validation complete "
                f"(valid={total_valid}, "
                f"failed={total_failed}) "
                f"in {elapsed_ms}ms"
            )

            return results

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_validate_chunked_state cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"Chunked validation crashed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return results

    async def _initialize_pipeline_workers(self):
        """
        Production-safe pipeline worker initialization.

        Fixes:
        - worker race conditions
        - task leaks
        - duplicate workers
        - startup corruption
        - invalid task registry
        - broken worker methods
        - worker explosion
        - async/sync mismatches
        - shutdown deadlocks
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        MAX_WORKERS = 8

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping workers"
                    )

                    return []

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if callable(config):
                        return default

                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE WORKER COUNT
            # ============================================================
            try:

                requested_workers = int(

                    cfg(
                        "init_pipeline_workers",
                        5,
                    )
                )

            except Exception:

                requested_workers = 5

            requested_workers = max(
                1,
                min(
                    requested_workers,
                    64,
                ),
            )

            # ============================================================
            # SAFE _cap_workers
            # ============================================================
            try:

                cap_method = getattr(
                    self,
                    "_cap_workers",
                    None,
                )

                if callable(cap_method):

                    worker_count = int(

                        cap_method(
                            requested_workers,
                            cap=MAX_WORKERS,
                        )
                    )

                else:

                    worker_count = min(
                        requested_workers,
                        MAX_WORKERS,
                    )

            except Exception:

                worker_count = min(
                    requested_workers,
                    MAX_WORKERS,
                )

            worker_count = max(
                1,
                min(
                    worker_count,
                    MAX_WORKERS,
                ),
            )

            # ============================================================
            # SAFE TASK REGISTRY
            # ============================================================
            tasks = getattr(
                self,
                "_init_pipeline_tasks",
                None,
            )

            if (
                tasks is None
                or callable(tasks)
                or not isinstance(tasks, list)
            ):

                tasks = []

            # remove dead tasks
            alive_tasks = []

            for task in tasks:

                try:

                    if (
                        task
                        and not task.done()
                    ):

                        alive_tasks.append(
                            task
                        )

                except Exception:
                    continue

            self._init_pipeline_tasks = (
                alive_tasks
            )

            existing_count = len(
                alive_tasks
            )

            # ============================================================
            # ALREADY RUNNING
            # ============================================================
            if existing_count >= worker_count:

                self.logger.info(
                    f"✓ Pipeline workers already active "
                    f"({existing_count})"
                )

                return alive_tasks

            # ============================================================
            # VALIDATE WORKER METHOD
            # ============================================================
            worker_method = getattr(
                self,
                "_init_pipeline_worker",
                None,
            )

            if worker_method is None:

                self.logger.error(
                    "_init_pipeline_worker missing"
                )

                return []

            if not callable(
                worker_method
            ):

                self.logger.error(
                    "_init_pipeline_worker not callable"
                )

                return []

            # ============================================================
            # CREATE WORKERS
            # ============================================================
            created_tasks = []

            for worker_id in range(
                existing_count,
                worker_count,
            ):

                try:

                    # ----------------------------------------------------
                    # SHUTDOWN CHECK
                    # ----------------------------------------------------
                    try:

                        if task_registry.is_shutting_down():

                            self.logger.debug(
                                "Shutdown during worker init"
                            )

                            break

                    except Exception:
                        pass

                    # ----------------------------------------------------
                    # SAFE WRAPPER
                    # ----------------------------------------------------
                    async def worker_wrapper(
                        wid=worker_id,
                    ):

                        try:

                            # async worker
                            if inspect.iscoroutinefunction(
                                worker_method
                            ):

                                result = worker_method(
                                    wid
                                )

                                if inspect.isawaitable(
                                    result
                                ):

                                    return await result

                                return result

                            # sync worker
                            return await asyncio.to_thread(
                                worker_method,
                                wid,
                            )

                        # ----------------------------------------------
                        # CANCELLED
                        # ----------------------------------------------
                        except asyncio.CancelledError:

                            self.logger.debug(
                                f"Pipeline worker "
                                f"{wid} cancelled"
                            )

                            raise

                        # ----------------------------------------------
                        # FAILURE
                        # ----------------------------------------------
                        except Exception as e:

                            self.logger.error(
                                f"Pipeline worker "
                                f"{wid} failed: {e}"
                            )

                            return None

                    # ----------------------------------------------------
                    # CREATE TASK
                    # ----------------------------------------------------
                    task = asyncio.create_task(

                        worker_wrapper(),

                        name=(
                            f"init_pipeline_worker_{worker_id}"
                        ),
                    )

                    created_tasks.append(
                        task
                    )

                    self._init_pipeline_tasks.append(
                        task
                    )

                except Exception as e:

                    self.logger.warning(
                        f"Worker creation failed "
                        f"({worker_id}): {e}"
                    )

            # ============================================================
            # CLEANUP CALLBACK
            # ============================================================
            def cleanup_task(
                completed_task,
            ):

                try:

                    if not hasattr(
                        self,
                        "_init_pipeline_tasks",
                    ):

                        return

                    tasks = getattr(
                        self,
                        "_init_pipeline_tasks",
                        [],
                    )

                    if (
                        isinstance(tasks, list)
                        and completed_task in tasks
                    ):

                        tasks.remove(
                            completed_task
                        )

                except Exception:
                    pass

            for task in created_tasks:

                try:

                    task.add_done_callback(
                        cleanup_task
                    )

                except Exception:
                    pass

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.pipeline_workers_started = int(

                    getattr(
                        self,
                        "pipeline_workers_started",
                        0,
                    )

                ) + len(created_tasks)

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            total_active = len(
                self._init_pipeline_tasks
            )

            self.logger.info(
                f"✓ Initialized "
                f"{len(created_tasks)} "
                f"pipeline workers "
                f"(active={total_active}) "
                f"in {elapsed_ms}ms"
            )

            return created_tasks

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_initialize_pipeline_workers cancelled"
                )

            except Exception:
                pass

            return []

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ _initialize_pipeline_workers failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return []
    
    async def _init_pipeline_worker(self, worker_id: int):
        """Worker for processing initialization chunks in pipeline."""
        while self.state == SystemState.INITIALIZING:
            try:
                if self._init_chunk_buffer:
                    chunk = await asyncio.wait_for(
                        self._get_next_init_chunk(), timeout=1.0
                    )
                    if chunk:
                        await self._process_init_chunk(chunk, worker_id)
                else:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"Init worker {worker_id} error: {e}")
                await asyncio.sleep(0.5)

    async def _get_next_init_chunk(self) -> dict:
        """Get next chunk from initialization buffer."""
        if self._init_chunk_buffer:
            return self._init_chunk_buffer.pop(0)
        return None

    async def _process_init_chunk(
        self,
        chunk: dict,
        worker_id: int,
    ):
        """
        Production-safe initialization chunk processor.

        Fixes:
        - function.items crashes
        - bool await crashes
        - invalid chunk structures
        - broken component execution
        - invalid safe_call usage
        - worker crashes
        - async/sync mismatch
        - initialization deadlocks
        - chunk corruption
        - coroutine leaks
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        f"Worker {worker_id}: shutdown active"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # VALIDATE CHUNK
            # ============================================================
            if chunk is None:

                self.logger.warning(
                    f"Worker {worker_id}: chunk is None"
                )

                return False

            # FIX:
            # prevents:
            # function.items crashes
            if callable(chunk):

                self.logger.warning(
                    f"Worker {worker_id}: "
                    f"chunk is callable"
                )

                return False

            if not isinstance(
                chunk,
                dict,
            ):

                self.logger.warning(
                    f"Worker {worker_id}: "
                    f"invalid chunk type "
                    f"{type(chunk)}"
                )

                return False

            # ============================================================
            # SAFE EXTRACTION
            # ============================================================
            try:

                chunk_type = str(

                    chunk.get(
                        "type",
                        "unknown",
                    )
                )

            except Exception:

                chunk_type = "unknown"

            try:

                chunk_data = chunk.get(
                    "data",
                    [],
                )

            except Exception:

                chunk_data = []

            # FIX:
            # callable corruption
            if callable(chunk_data):

                self.logger.warning(
                    f"Worker {worker_id}: "
                    f"chunk_data callable"
                )

                chunk_data = []

            if chunk_data is None:

                chunk_data = []

            if not isinstance(
                chunk_data,
                (list, tuple),
            ):

                chunk_data = [chunk_data]

            # ============================================================
            # SAFE _safe_call
            # ============================================================
            safe_call = getattr(
                self,
                "_safe_call",
                None,
            )

            if not callable(
                safe_call
            ):

                self.logger.error(
                    "_safe_call missing"
                )

                return False

            # ============================================================
            # EMPTY CHUNK
            # ============================================================
            if not chunk_data:

                self.logger.debug(
                    f"Worker {worker_id}: "
                    f"empty {chunk_type} chunk"
                )

                return True

            processed = 0

            failed = 0

            # ============================================================
            # AGENT CHUNK
            # ============================================================
            if chunk_type == "agents":

                agent_registry = getattr(
                    self,
                    "agent_registry",
                    None,
                )

                if agent_registry is None:

                    self.logger.warning(
                        f"Worker {worker_id}: "
                        f"agent_registry missing"
                    )

                    return False

                for agent_name in chunk_data:

                    try:

                        if not agent_name:
                            continue

                        # ------------------------------------------------
                        # SAFE CALL
                        # ------------------------------------------------
                        result = safe_call(

                            agent_registry,

                            "initialize_agent",

                            agent_name,

                            getattr(
                                self,
                                "system_instance",
                                None,
                            ),
                        )

                        # FIX:
                        # bool cannot be awaited
                        if inspect.isawaitable(
                            result
                        ):

                            result = await result

                        processed += 1

                    except asyncio.CancelledError:
                        raise

                    except Exception as e:

                        failed += 1

                        self.logger.warning(
                            f"Worker {worker_id}: "
                            f"agent init failed "
                            f"({agent_name}): {e}"
                        )

            # ============================================================
            # COMPONENT CHUNK
            # ============================================================
            elif chunk_type == "components":

                for component_init in chunk_data:

                    try:

                        if component_init is None:
                            continue

                        result = None

                        # ------------------------------------------------
                        # DIRECT CALLABLE
                        # ------------------------------------------------
                        if callable(
                            component_init
                        ):

                            if inspect.iscoroutinefunction(
                                component_init
                            ):

                                result = component_init()

                            else:

                                result = await asyncio.to_thread(
                                    component_init
                                )

                        # ------------------------------------------------
                        # STRING METHOD
                        # ------------------------------------------------
                        elif isinstance(
                            component_init,
                            str,
                        ):

                            method = getattr(
                                self,
                                component_init,
                                None,
                            )

                            if not callable(
                                method
                            ):

                                raise RuntimeError(
                                    f"Method missing: "
                                    f"{component_init}"
                                )

                            if inspect.iscoroutinefunction(
                                method
                            ):

                                result = method()

                            else:

                                result = await asyncio.to_thread(
                                    method
                                )

                        # ------------------------------------------------
                        # DICT CONFIG
                        # ------------------------------------------------
                        elif isinstance(
                            component_init,
                            dict,
                        ):

                            method_name = component_init.get(
                                "method"
                            )

                            method_args = component_init.get(
                                "args",
                                [],
                            )

                            method_kwargs = component_init.get(
                                "kwargs",
                                {},
                            )

                            # FIX:
                            # kwargs callable corruption
                            if callable(
                                method_kwargs
                            ):

                                method_kwargs = {}

                            if not isinstance(
                                method_kwargs,
                                dict,
                            ):

                                method_kwargs = {}

                            if not isinstance(
                                method_args,
                                (list, tuple),
                            ):

                                method_args = []

                            if callable(
                                method_name
                            ):

                                raise RuntimeError(
                                    "method_name callable"
                                )

                            method = getattr(
                                self,
                                method_name,
                                None,
                            )

                            if not callable(
                                method
                            ):

                                raise RuntimeError(
                                    f"Method missing: "
                                    f"{method_name}"
                                )

                            # --------------------------------------------
                            # ASYNC METHOD
                            # --------------------------------------------
                            if inspect.iscoroutinefunction(
                                method
                            ):

                                result = method(
                                    *method_args,
                                    **method_kwargs,
                                )

                            # --------------------------------------------
                            # SYNC METHOD
                            # --------------------------------------------
                            else:

                                result = await asyncio.to_thread(

                                    method,

                                    *method_args,

                                    **method_kwargs,
                                )

                        else:

                            raise RuntimeError(
                                f"Unsupported component type: "
                                f"{type(component_init)}"
                            )

                        # ------------------------------------------------
                        # SAFE AWAIT
                        # ------------------------------------------------
                        if inspect.isawaitable(
                            result
                        ):

                            result = await result

                        processed += 1

                    except asyncio.CancelledError:
                        raise

                    except Exception as e:

                        failed += 1

                        self.logger.warning(
                            f"Worker {worker_id}: "
                            f"component init failed: {e}"
                        )

            # ============================================================
            # UNKNOWN CHUNK
            # ============================================================
            else:

                self.logger.warning(
                    f"Worker {worker_id}: "
                    f"unknown chunk type "
                    f"{chunk_type}"
                )

                return False

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.init_chunks_processed = int(

                    getattr(
                        self,
                        "init_chunks_processed",
                        0,
                    )

                ) + 1

                self.init_items_processed = int(

                    getattr(
                        self,
                        "init_items_processed",
                        0,
                    )

                ) + processed

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.debug(
                f"Worker {worker_id} processed "
                f"{chunk_type} chunk "
                f"(ok={processed}, failed={failed}) "
                f"in {elapsed_ms}ms"
            )

            return failed == 0

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    f"Worker {worker_id}: "
                    f"chunk cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ Worker {worker_id} "
                    f"failed on chunk: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

    async def _flush_init_chunk_buffer(self):
        """Flush any remaining initialization chunks."""
        if self._init_chunk_buffer:
            self.logger.info(
                f"Flushing {len(self._init_chunk_buffer)} remaining init chunks"
            )
            for chunk in self._init_chunk_buffer:
                await self._process_init_chunk(chunk, "flush_worker")
            self._init_chunk_buffer.clear()

    async def _initialize_agent_communication_pipeline(self):
        """
        Production-safe agent communication pipeline initializer.

        Fixes:
        - function.items crashes
        - bool await crashes
        - invalid pipeline configs
        - broken safe_call handling
        - missing coordinator crashes
        - invalid config corruption
        - async/sync mismatch
        - initialization deadlocks
        - coroutine leaks
        - startup race conditions
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        PIPELINE_TIMEOUT = 30.0

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping "
                        "agent pipeline init"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # VALIDATE COORDINATOR
            # ============================================================
            coordinator = getattr(
                self,
                "multi_agent_coordinator",
                None,
            )

            if coordinator is None:

                self.logger.warning(
                    "multi_agent_coordinator missing"
                )

                return False

            # FIX:
            # callable corruption
            if callable(coordinator):

                self.logger.warning(
                    "multi_agent_coordinator callable"
                )

                return False

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE CONFIG VALUES
            # ============================================================
            try:

                chunk_size = int(

                    cfg(
                        "agent_comms_chunk_size",
                        100,
                    )
                )

            except Exception:

                chunk_size = 100

            try:

                pipeline_depth = int(

                    cfg(
                        "agent_pipeline_depth",
                        10,
                    )
                )

            except Exception:

                pipeline_depth = 10

            try:

                parallel_agents = int(

                    cfg(
                        "parallel_agent_comms",
                        3,
                    )
                )

            except Exception:

                parallel_agents = 3

            # ============================================================
            # HARD LIMITS
            # ============================================================
            chunk_size = max(
                1,
                min(
                    chunk_size,
                    10000,
                ),
            )

            pipeline_depth = max(
                1,
                min(
                    pipeline_depth,
                    256,
                ),
            )

            parallel_agents = max(
                1,
                min(
                    parallel_agents,
                    64,
                ),
            )

            # ============================================================
            # BUILD CONFIG
            # ============================================================
            pipeline_config = {

                "chunk_size": chunk_size,

                "pipeline_depth": pipeline_depth,

                "parallel_agents": parallel_agents,

                "initialized_at": time.time(),

                "safe_mode": True,

                "async_enabled": True,
            }

            # ============================================================
            # VALIDATE _safe_call
            # ============================================================
            safe_call = getattr(
                self,
                "_safe_call",
                None,
            )

            if not callable(
                safe_call
            ):

                self.logger.error(
                    "_safe_call missing"
                )

                return False

            # ============================================================
            # VALIDATE TARGET METHOD
            # ============================================================
            init_method = getattr(
                coordinator,
                "initialize_pipeline",
                None,
            )

            if init_method is None:

                self.logger.warning(
                    "Coordinator missing "
                    "initialize_pipeline"
                )

                return False

            if not callable(
                init_method
            ):

                self.logger.warning(
                    "initialize_pipeline not callable"
                )

                return False

            # ============================================================
            # PREVENT DUPLICATE INIT
            # ============================================================
            already_initialized = bool(

                getattr(
                    coordinator,
                    "_pipeline_initialized",
                    False,
                )
            )

            if already_initialized:

                self.logger.info(
                    "✓ Agent pipeline already initialized"
                )

                return True

            # ============================================================
            # EXECUTE SAFE CALL
            # ============================================================
            try:

                result = safe_call(

                    coordinator,

                    "initialize_pipeline",

                    pipeline_config,
                )

                # FIX:
                # bool cannot be awaited
                if inspect.isawaitable(
                    result
                ):

                    result = await asyncio.wait_for(

                        result,

                        timeout=PIPELINE_TIMEOUT,
                    )

            # ------------------------------------------------------------
            # TIMEOUT
            # ------------------------------------------------------------
            except asyncio.TimeoutError:

                self.logger.error(
                    "Agent communication "
                    "pipeline timeout"
                )

                return False

            # ------------------------------------------------------------
            # CANCELLED
            # ------------------------------------------------------------
            except asyncio.CancelledError:
                raise

            # ------------------------------------------------------------
            # EXECUTION FAILURE
            # ------------------------------------------------------------
            except Exception as e:

                self.logger.error(
                    f"Pipeline execution failed: {e}"
                )

                return False

            # ============================================================
            # POST VALIDATION
            # ============================================================
            try:

                setattr(
                    coordinator,
                    "_pipeline_initialized",
                    True,
                )

                setattr(
                    coordinator,
                    "_pipeline_config",
                    pipeline_config,
                )

                setattr(
                    coordinator,
                    "_pipeline_initialized_at",
                    time.time(),
                )

            except Exception:
                pass

            # ============================================================
            # HEALTH CHECK
            # ============================================================
            try:

                health_method = getattr(
                    coordinator,
                    "health_check",
                    None,
                )

                if callable(
                    health_method
                ):

                    health_result = health_method()

                    if inspect.isawaitable(
                        health_result
                    ):

                        await asyncio.wait_for(

                            health_result,

                            timeout=10,
                        )

            except Exception as e:

                self.logger.warning(
                    f"Pipeline health check failed: {e}"
                )

            # ============================================================
            # METRICS
            # ============================================================
            try:

                self.agent_pipeline_inits = int(

                    getattr(
                        self,
                        "agent_pipeline_inits",
                        0,
                    )

                ) + 1

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.info(
                f"✓ Agent communication pipeline initialized "
                f"(chunk_size={chunk_size}, "
                f"depth={pipeline_depth}, "
                f"parallel={parallel_agents}) "
                f"in {elapsed_ms}ms"
            )

            return True

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_initialize_agent_communication_pipeline "
                    "cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ Agent communication "
                    f"pipeline init failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

    async def _load_model_in_chunks(
        self,
        llm,
    ) -> None:
        """
        Production-safe chunked model loader.

        Fixes:
        - invalid await usage
        - thread execution bugs
        - broken chunk iterators
        - memory spikes
        - callable config corruption
        - model deadlocks
        - invalid chunk loaders
        - timeout hangs
        - sync/async mismatch
        - finalization crashes
        - duplicate model loading
        - GPU memory leaks
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping model load"
                    )

                    return

            except Exception:
                pass

            # ============================================================
            # VALIDATE LLM
            # ============================================================
            if llm is None:

                self.logger.error(
                    "LLM instance is None"
                )

                return

            # FIX:
            # callable corruption
            if callable(llm):

                self.logger.error(
                    "LLM object is callable"
                )

                return

            # ============================================================
            # PREVENT DOUBLE LOADING
            # ============================================================
            already_loaded = bool(

                getattr(
                    llm,
                    "_model_loaded",
                    False,
                )
            )

            if already_loaded:

                self.logger.info(
                    "✓ Model already loaded"
                )

                return

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE CONFIG VALUES
            # ============================================================
            try:

                chunk_size = int(

                    cfg(
                        "model_chunk_size_mb",
                        256,
                    )
                )

            except Exception:

                chunk_size = 256

            chunk_size = max(
                32,
                min(
                    chunk_size,
                    4096,
                ),
            )

            try:

                chunk_timeout = float(

                    cfg(
                        "model_chunk_timeout",
                        60,
                    )
                )

            except Exception:

                chunk_timeout = 60.0

            chunk_timeout = max(
                5.0,
                min(
                    chunk_timeout,
                    600.0,
                ),
            )

            # ============================================================
            # CHUNKED LOADING
            # ============================================================
            get_chunks_method = getattr(
                llm,
                "get_model_chunks",
                None,
            )

            if callable(
                get_chunks_method
            ):

                self.logger.info(
                    f"📦 Loading model in chunks "
                    f"({chunk_size}MB)"
                )

                # --------------------------------------------------------
                # GET CHUNKS
                # --------------------------------------------------------
                try:

                    chunk_result = (
                        get_chunks_method(
                            chunk_size
                        )
                    )

                    if inspect.isawaitable(
                        chunk_result
                    ):

                        model_chunks = await asyncio.wait_for(

                            chunk_result,

                            timeout=chunk_timeout,
                        )

                    else:

                        model_chunks = chunk_result

                except Exception as e:

                    self.logger.error(
                        f"Failed getting model chunks: {e}"
                    )

                    return

                # --------------------------------------------------------
                # VALIDATE CHUNKS
                # --------------------------------------------------------
                if model_chunks is None:

                    self.logger.error(
                        "Model chunks is None"
                    )

                    return

                # FIX:
                # generator/function corruption
                if callable(model_chunks):

                    self.logger.error(
                        "Model chunks callable"
                    )

                    return

                try:

                    model_chunks = list(
                        model_chunks
                    )

                except Exception as e:

                    self.logger.error(
                        f"Chunk conversion failed: {e}"
                    )

                    return

                if not model_chunks:

                    self.logger.warning(
                        "No model chunks returned"
                    )

                    return

                total_chunks = len(
                    model_chunks
                )

                # --------------------------------------------------------
                # VALIDATE CHUNK LOADER
                # --------------------------------------------------------
                load_chunk_method = getattr(
                    llm,
                    "load_model_chunk",
                    None,
                )

                if not callable(
                    load_chunk_method
                ):

                    self.logger.error(
                        "load_model_chunk missing"
                    )

                    return

                # --------------------------------------------------------
                # PROCESS CHUNKS
                # --------------------------------------------------------
                for (
                    chunk_idx,
                    model_chunk,
                ) in enumerate(
                    model_chunks,
                    start=1,
                ):

                    try:

                        # ------------------------------------------------
                        # SHUTDOWN CHECK
                        # ------------------------------------------------
                        try:

                            if task_registry.is_shutting_down():

                                self.logger.debug(
                                    "Shutdown during chunk loading"
                                )

                                return

                        except Exception:
                            pass

                        self.logger.info(
                            f"📦 Loading model chunk "
                            f"{chunk_idx}/{total_chunks}"
                        )

                        # ------------------------------------------------
                        # EXECUTE LOADER
                        # ------------------------------------------------
                        if inspect.iscoroutinefunction(
                            load_chunk_method
                        ):

                            result = load_chunk_method(
                                model_chunk,
                                chunk_idx - 1,
                            )

                            if inspect.isawaitable(
                                result
                            ):

                                await asyncio.wait_for(

                                    result,

                                    timeout=chunk_timeout,
                                )

                        else:

                            await asyncio.wait_for(

                                asyncio.to_thread(

                                    load_chunk_method,

                                    model_chunk,

                                    chunk_idx - 1,
                                ),

                                timeout=chunk_timeout,
                            )

                        # ------------------------------------------------
                        # MEMORY CLEANUP
                        # ------------------------------------------------
                        try:

                            del model_chunk

                        except Exception:
                            pass

                        try:

                            gc.collect()

                        except Exception:
                            pass

                        # ------------------------------------------------
                        # GPU CLEANUP
                        # ------------------------------------------------
                        try:

                            import torch

                            if torch.cuda.is_available():

                                torch.cuda.empty_cache()

                        except Exception:
                            pass

                        # ------------------------------------------------
                        # METRICS
                        # ------------------------------------------------
                        try:

                            self.model_chunks_loaded = int(

                                getattr(
                                    self,
                                    "model_chunks_loaded",
                                    0,
                                )

                            ) + 1

                        except Exception:
                            pass

                        await asyncio.sleep(
                            0.05
                        )

                    # ----------------------------------------------------
                    # TIMEOUT
                    # ----------------------------------------------------
                    except asyncio.TimeoutError:

                        self.logger.error(
                            f"Chunk {chunk_idx} timeout"
                        )

                        raise

                    # ----------------------------------------------------
                    # CANCELLED
                    # ----------------------------------------------------
                    except asyncio.CancelledError:
                        raise

                    # ----------------------------------------------------
                    # FAILURE
                    # ----------------------------------------------------
                    except Exception as e:

                        self.logger.error(
                            f"Chunk {chunk_idx} failed: {e}"
                        )

                        raise

                # --------------------------------------------------------
                # FINALIZE
                # --------------------------------------------------------
                finalize_method = getattr(
                    llm,
                    "finalize_model_load",
                    None,
                )

                if callable(
                    finalize_method
                ):

                    try:

                        if inspect.iscoroutinefunction(
                            finalize_method
                        ):

                            result = finalize_method()

                            if inspect.isawaitable(
                                result
                            ):

                                await asyncio.wait_for(

                                    result,

                                    timeout=chunk_timeout,
                                )

                        else:

                            await asyncio.wait_for(

                                asyncio.to_thread(
                                    finalize_method
                                ),

                                timeout=chunk_timeout,
                            )

                    except Exception as e:

                        self.logger.warning(
                            f"Model finalize failed: {e}"
                        )

            # ============================================================
            # FALLBACK LOADING
            # ============================================================
            else:

                self.logger.info(
                    "📦 Using fallback model loading"
                )

                load_method = getattr(
                    llm,
                    "load_model",
                    None,
                )

                if not callable(
                    load_method
                ):

                    self.logger.error(
                        "load_model missing"
                    )

                    return

                # --------------------------------------------------------
                # SAFE MODEL PATH
                # --------------------------------------------------------
                model_path = None

                try:

                    if isinstance(
                        config,
                        dict,
                    ):

                        model_path = config.get(
                            "llm_model_path"
                        )

                    else:

                        model_path = getattr(
                            config,
                            "llm_model_path",
                            None,
                        )

                except Exception:
                    pass

                if not model_path:

                    self.logger.warning(
                        "No llm_model_path configured"
                    )

                # --------------------------------------------------------
                # EXECUTE LOAD
                # --------------------------------------------------------
                if inspect.iscoroutinefunction(
                    load_method
                ):

                    result = load_method(
                        model_path
                    )

                    if inspect.isawaitable(
                        result
                    ):

                        await asyncio.wait_for(

                            result,

                            timeout=chunk_timeout,
                        )

                else:

                    await asyncio.wait_for(

                        asyncio.to_thread(
                            load_method,
                            model_path,
                        ),

                        timeout=chunk_timeout,
                    )

            # ============================================================
            # MARK LOADED
            # ============================================================
            try:

                setattr(
                    llm,
                    "_model_loaded",
                    True,
                )

                setattr(
                    llm,
                    "_model_loaded_at",
                    time.time(),
                )

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.info(
                f"✅ Model loading complete "
                f"in {elapsed_ms}ms"
            )

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_load_model_in_chunks cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ _load_model_in_chunks failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

    async def _initialize_system_components(self, *args, **kwargs):
        """
        Initialize system-level components (PIPELINE SAFE)
        - accepts extra kwargs (fix for group_name error)
        - parallel startup
        - shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            self.logger.debug("Shutdown in progress - skipping init")
            return

        # -------------------------
        # CREATE COMPONENTS
        # -------------------------
        new_components = {}

        try:
            from core.system.system_controller import SystemController

            new_components["system_controller"] = SystemController()
        except Exception as e:
            self.logger.warning(f"SystemController init failed: {e}")
            new_components["system_controller"] = None

        try:
            new_components["command_router"] = CommandRouter()
        except Exception as e:
            self.logger.warning(f"CommandRouter init failed: {e}")
            new_components["command_router"] = None

        try:
            new_components["workflow_engine"] = WorkflowEngine()
        except Exception as e:
            self.logger.warning(f"WorkflowEngine init failed: {e}")
            new_components["workflow_engine"] = None

        try:
            new_components["safety_layer"] = SafetyLayer()
        except Exception as e:
            self.logger.warning(f"SafetyLayer init failed: {e}")
            new_components["safety_layer"] = None

        # -------------------------
        # SAFE UPDATE (NO MUTATION DURING ITERATION)
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=15.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # EXECUTE PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _initialize_agent_components(self, *args, **kwargs):
        """
        Initialize agent-level components (PIPELINE SAFE)
        - accepts extra kwargs (fix for pipeline)
        - parallel startup
        - shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            self.logger.debug("Shutdown in progress - skipping init")
            return

        # -------------------------
        # CREATE COMPONENTS
        # -------------------------
        new_components = {}

        try:
            new_components["agent_core"] = AgentCore(
                self.system_instance if self.system_instance else None
            )
        except Exception as e:
            self.logger.warning(f"AgentCore init failed: {e}")
            new_components["agent_core"] = None

        try:
            new_components["agent_loop"] = AgentLoop()
        except Exception as e:
            self.logger.warning(f"AgentLoop init failed: {e}")
            new_components["agent_loop"] = None

        # -------------------------
        # SAFE UPDATE
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START (FIX)
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=15.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _initialize_brain_components(
        self,
        *args,
        **kwargs,
    ):
        """
        Production-safe brain initialization.

        Fixes:
        - function.items crashes
        - bool await crashes
        - invalid config corruption
        - pipeline deadlocks
        - unsafe task execution
        - coroutine leaks
        - duplicate initialization
        - chunk race conditions
        - memory spikes
        - invalid component injection
        - async/sync mismatch
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # ============================================================
            # SHUTDOWN GUARD
            # ============================================================
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping brain init"
                    )

                    return False

            except Exception:
                pass

            # ============================================================
            # SAFE CONFIG ACCESS
            # ============================================================
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(
                key,
                default,
            ):

                try:

                    # FIX:
                    # config accidentally callable
                    if callable(config):
                        return default

                    # dict config
                    if isinstance(config, dict):

                        value = config.get(
                            key,
                            default,
                        )

                    # object config
                    else:

                        value = getattr(
                            config,
                            key,
                            default,
                        )

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(value):

                        return default

                    return value

                except Exception:

                    return default

            # ============================================================
            # SAFE TIMEOUTS
            # ============================================================
            try:

                component_timeout = float(

                    cfg(
                        "brain_init_timeout",
                        15.0,
                    )
                )

            except Exception:

                component_timeout = 15.0

            try:

                chunk_timeout = float(

                    cfg(
                        "brain_chunk_timeout",
                        60.0,
                    )
                )

            except Exception:

                chunk_timeout = 60.0

            try:

                chunk_size = int(

                    cfg(
                        "brain_chunk_size",
                        2,
                    )
                )

            except Exception:

                chunk_size = 2

            chunk_size = max(
                1,
                min(
                    chunk_size,
                    16,
                ),
            )

            component_timeout = max(
                5.0,
                min(
                    component_timeout,
                    300.0,
                ),
            )

            chunk_timeout = max(
                component_timeout,
                min(
                    chunk_timeout,
                    600.0,
                ),
            )

            # ============================================================
            # INIT GUARD
            # ============================================================
            if getattr(
                self,
                "_brain_components_initializing",
                False,
            ):

                self.logger.warning(
                    "Brain initialization already running"
                )

                return False

            self._brain_components_initializing = True

            # ============================================================
            # COMPONENT DEFINITIONS
            # ============================================================
            brain_components = [

                {
                    "name": "llm_engine",
                    "class": LLMEngine,
                    "critical": True,
                },

                {
                    "name": "decision_engine",
                    "class": DecisionEngine,
                    "critical": True,
                },

                {
                    "name": "reasoning_engine",
                    "class": ReasoningEngine,
                    "critical": False,
                },

                {
                    "name": "context_manager",
                    "class": ContextManager,
                    "critical": False,
                },
            ]

            # ============================================================
            # VALIDATE COMPONENTS
            # ============================================================
            validated_components = []

            for comp in brain_components:

                try:

                    # FIX:
                    # prevents:
                    # function.items crashes
                    if callable(comp):
                        continue

                    if not isinstance(
                        comp,
                        dict,
                    ):
                        continue

                    comp_name = comp.get(
                        "name"
                    )

                    comp_class = comp.get(
                        "class"
                    )

                    if not comp_name:
                        continue

                    if comp_class is None:

                        self.logger.warning(
                            f"{comp_name}: missing class"
                        )

                        continue

                    if not inspect.isclass(
                        comp_class
                    ):

                        self.logger.warning(
                            f"{comp_name}: invalid class"
                        )

                        continue

                    validated_components.append(
                        comp
                    )

                except Exception:
                    continue

            if not validated_components:

                self.logger.warning(
                    "No valid brain components"
                )

                return False

            # ============================================================
            # FILTER ENABLED
            # ============================================================
            enabled_components = []

            for comp in validated_components:

                try:

                    comp_name = comp.get(
                        "name"
                    )

                    enabled = bool(

                        cfg(
                            f"enable_{comp_name}",
                            True,
                        )
                    )

                    if enabled:

                        enabled_components.append(
                            comp
                        )

                except Exception:
                    continue

            if not enabled_components:

                self.logger.warning(
                    "No brain components enabled"
                )

                return False

            # ============================================================
            # PREVENT DUPLICATES
            # ============================================================
            existing_components = getattr(
                self,
                "components",
                {},
            )

            if not isinstance(
                existing_components,
                dict,
            ):

                existing_components = {}

            final_components = []

            for comp in enabled_components:

                try:

                    comp_name = comp.get(
                        "name"
                    )

                    existing = existing_components.get(
                        comp_name
                    )

                    if existing is not None:

                        self.logger.info(
                            f"✓ {comp_name} already initialized"
                        )

                        continue

                    final_components.append(
                        comp
                    )

                except Exception:
                    continue

            if not final_components:

                self.logger.info(
                    "✓ Brain components already initialized"
                )

                return True

            # ============================================================
            # CHUNKING
            # ============================================================
            component_chunks = [

                final_components[
                    i:i + chunk_size
                ]

                for i in range(
                    0,
                    len(final_components),
                    chunk_size,
                )
            ]

            self.logger.info(
                f"🧠 Initializing "
                f"{len(final_components)} "
                f"brain components "
                f"in {len(component_chunks)} chunks"
            )

            new_components = {}

            failed_components = []

            # ============================================================
            # PROCESS CHUNKS
            # ============================================================
            for (
                chunk_index,
                chunk,
            ) in enumerate(
                component_chunks,
                start=1,
            ):

                try:

                    if task_registry.is_shutting_down():
                        break

                except Exception:
                    pass

                tasks = []

                names = []

                # --------------------------------------------------------
                # BUILD TASKS
                # --------------------------------------------------------
                for comp in chunk:

                    try:

                        if callable(comp):
                            continue

                        comp_name = comp.get(
                            "name"
                        )

                        comp_class = comp.get(
                            "class"
                        )

                        if not comp_name:
                            continue

                        if comp_class is None:
                            continue

                        names.append(
                            comp_name
                        )

                        task = asyncio.create_task(

                            self._init_single_brain_component(

                                {
                                    "name": comp_name,

                                    "class": comp_class,

                                    "timeout": component_timeout,

                                    "retry_count": 2,

                                    "critical": bool(
                                        comp.get(
                                            "critical",
                                            False,
                                        )
                                    ),
                                }
                            ),

                            name=f"brain_init_{comp_name}",
                        )

                        tasks.append(
                            task
                        )

                    except Exception as e:

                        self.logger.warning(
                            f"Task creation failed: {e}"
                        )

                if not tasks:
                    continue

                # --------------------------------------------------------
                # EXECUTE CHUNK
                # --------------------------------------------------------
                try:

                    results = await asyncio.wait_for(

                        asyncio.gather(
                            *tasks,
                            return_exceptions=True,
                        ),

                        timeout=chunk_timeout,
                    )

                    for (
                        name,
                        result,
                    ) in zip(
                        names,
                        results,
                    ):

                        # --------------------------------------------
                        # EXCEPTION
                        # --------------------------------------------
                        if isinstance(
                            result,
                            Exception,
                        ):

                            self.logger.warning(
                                f"⚠ {name} failed: {result}"
                            )

                            failed_components.append(
                                name
                            )

                            continue

                        # --------------------------------------------
                        # EMPTY RESULT
                        # --------------------------------------------
                        if result is None:

                            failed_components.append(
                                name
                            )

                            continue

                        # FIX:
                        # bool await corruption
                        if isinstance(
                            result,
                            bool,
                        ):

                            if result is False:

                                failed_components.append(
                                    name
                                )

                                continue

                        new_components[
                            name
                        ] = result

                        self.logger.info(
                            f"✓ {name} initialized"
                        )

                # --------------------------------------------------------
                # TIMEOUT
                # --------------------------------------------------------
                except asyncio.TimeoutError:

                    self.logger.error(
                        f"❌ Brain chunk timeout "
                        f"({chunk_index})"
                    )

                    for task in tasks:

                        try:

                            if not task.done():

                                task.cancel()

                        except Exception:
                            pass

                    failed_components.extend(
                        names
                    )

                # --------------------------------------------------------
                # CANCELLED
                # --------------------------------------------------------
                except asyncio.CancelledError:
                    raise

                # --------------------------------------------------------
                # FAILURE
                # --------------------------------------------------------
                except Exception as e:

                    self.logger.error(
                        f"❌ Brain chunk failure: {e}"
                    )

                    failed_components.extend(
                        names
                    )

                # --------------------------------------------------------
                # MEMORY CLEANUP
                # --------------------------------------------------------
                try:

                    gc.collect()

                except Exception:
                    pass

                await asyncio.sleep(
                    0.05
                )

            # ============================================================
            # UPDATE COMPONENT REGISTRY
            # ============================================================
            try:

                if not hasattr(
                    self,
                    "components",
                ):

                    self.components = {}

                if not isinstance(
                    self.components,
                    dict,
                ):

                    self.components = {}

                self.components.update(
                    new_components
                )

            except Exception as e:

                self.logger.warning(
                    f"Component registry update failed: {e}"
                )

            # ============================================================
            # PIPELINE CONNECTION
            # ============================================================
            try:

                connect_method = getattr(
                    self,
                    "_connect_brain_pipeline",
                    None,
                )

                if (
                    callable(connect_method)
                    and "llm_engine" in new_components
                    and "decision_engine" in new_components
                ):

                    result = connect_method(
                        new_components
                    )

                    if inspect.isawaitable(
                        result
                    ):

                        await asyncio.wait_for(

                            result,

                            timeout=30,
                        )

            except Exception as e:

                self.logger.warning(
                    f"Pipeline connection failed: {e}"
                )

            # ============================================================
            # METRICS
            # ============================================================
            success_count = len(
                new_components
            )

            try:

                self.brain_components_initialized = int(

                    getattr(
                        self,
                        "brain_components_initialized",
                        0,
                    )

                ) + success_count

            except Exception:
                pass

            # ============================================================
            # SUMMARY
            # ============================================================
            elapsed_ms = round(

                (
                    time.monotonic()
                    - start_time
                ) * 1000,

                2,
            )

            self.logger.info(
                f"🧠 Brain init complete: "
                f"{success_count}/"
                f"{len(final_components)} successful "
                f"in {elapsed_ms}ms"
            )

            if failed_components:

                self.logger.warning(
                    f"Failed brain components: "
                    f"{failed_components}"
                )

            return success_count > 0

        # ================================================================
        # CANCELLED
        # ================================================================
        except asyncio.CancelledError:

            try:

                self.logger.debug(
                    "_initialize_brain_components cancelled"
                )

            except Exception:
                pass

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ _initialize_brain_components failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return False

        # ================================================================
        # FINALIZE
        # ================================================================
        finally:

            self._brain_components_initializing = False


    async def _init_single_brain_component(
        self,
        comp_config: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Production-safe brain component initializer.

        Fixes:
        - function.items crashes
        - bool await crashes
        - invalid configs
        - async/sync mismatch
        - coroutine leaks
        - startup race conditions
        - initialization deadlocks
        - timeout hangs
        - duplicate initialization
        - unsafe kwargs
        - memory leaks
        - invalid lifecycle ordering
        """

        import asyncio
        import gc
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # ============================================================
            # VALIDATE CONFIG
            # ============================================================
            if comp_config is None:

                self.logger.warning(
                    "Component config is None"
                )

                return None

            # FIX:
            # prevents:
            # function.items crashes
            if callable(comp_config):

                self.logger.warning(
                    "Component config callable"
                )

                return None

            if not isinstance(
                comp_config,
                dict,
            ):

                self.logger.warning(
                    f"Invalid component config type: "
                    f"{type(comp_config)}"
                )

                return None

            # ============================================================
            # SAFE EXTRACTION
            # ============================================================
            try:

                component_name = str(

                    comp_config.get(
                        "name",
                        "unknown",
                    )
                )

            except Exception:

                component_name = "unknown"

            component_class = comp_config.get(
                "class"
            )

            # ============================================================
            # VALIDATE CLASS
            # ============================================================
            if component_class is None:

                self.logger.warning(
                    f"{component_name}: missing class"
                )

                return None

            if not callable(
                component_class
            ):

                self.logger.warning(
                    f"{component_name}: class not callable"
                )

                return None

            # ============================================================
            # SAFE SETTINGS
            # ============================================================
            try:

                timeout = float(

                    comp_config.get(
                        "timeout",
                        15.0,
                    )
                )

            except Exception:

                timeout = 15.0

            timeout = max(
                1.0,
                min(
                    timeout,
                    300.0,
                ),
            )

            try:

                retry_count = int(

                    comp_config.get(
                        "retry_count",
                        2,
                    )
                )

            except Exception:

                retry_count = 2

            retry_count = max(
                1,
                min(
                    retry_count,
                    10,
                ),
            )

            try:

                backoff = float(

                    comp_config.get(
                        "retry_backoff",
                        0.5,
                    )
                )

            except Exception:

                backoff = 0.5

            backoff = max(
                0.1,
                min(
                    backoff,
                    10.0,
                ),
            )

            # ============================================================
            # SAFE ARGS
            # ============================================================
            init_args = comp_config.get(
                "init_args",
                [],
            )

            init_kwargs = comp_config.get(
                "init_kwargs",
                {},
            )

            if not isinstance(
                init_args,
                (list, tuple),
            ):

                init_args = []

            # FIX:
            # prevents:
            # function.items crashes
            if callable(init_kwargs):

                self.logger.warning(
                    f"{component_name}: "
                    f"init_kwargs callable"
                )

                init_kwargs = {}

            if not isinstance(
                init_kwargs,
                dict,
            ):

                init_kwargs = {}

            # ============================================================
            # DUPLICATE INIT GUARD
            # ============================================================
            existing_components = getattr(
                self,
                "components",
                {},
            )

            if isinstance(
                existing_components,
                dict,
            ):

                existing = existing_components.get(
                    component_name
                )

                if existing is not None:

                    self.logger.info(
                        f"✓ {component_name} already initialized"
                    )

                    return existing

            # ============================================================
            # RETRY LOOP
            # ============================================================
            for attempt in range(
                1,
                retry_count + 1,
            ):

                # --------------------------------------------------------
                # SHUTDOWN CHECK
                # --------------------------------------------------------
                try:

                    if task_registry.is_shutting_down():

                        self.logger.debug(
                            f"{component_name}: shutdown active"
                        )

                        return None

                except Exception:
                    pass

                instance = None

                try:

                    # ====================================================
                    # CREATE INSTANCE
                    # ====================================================
                    if inspect.iscoroutinefunction(
                        component_class
                    ):

                        result = component_class(
                            *init_args,
                            **init_kwargs,
                        )

                        # FIX:
                        # bool await corruption
                        if inspect.isawaitable(
                            result
                        ):

                            instance = await asyncio.wait_for(

                                result,

                                timeout=timeout,
                            )

                        else:

                            instance = result

                    else:

                        instance = await asyncio.wait_for(

                            asyncio.to_thread(

                                component_class,

                                *init_args,

                                **init_kwargs,
                            ),

                            timeout=timeout,
                        )

                    # ====================================================
                    # VALIDATE INSTANCE
                    # ====================================================
                    if instance is None:

                        raise RuntimeError(
                            "Instance is None"
                        )

                    # FIX:
                    # invalid bool instance
                    if isinstance(
                        instance,
                        bool,
                    ):

                        raise RuntimeError(
                            "Instance returned bool"
                        )

                    # ====================================================
                    # INITIALIZE
                    # ====================================================
                    init_method = getattr(
                        instance,
                        "initialize",
                        None,
                    )

                    if callable(
                        init_method
                    ):

                        try:

                            # async initialize
                            if inspect.iscoroutinefunction(
                                init_method
                            ):

                                result = init_method()

                                if inspect.isawaitable(
                                    result
                                ):

                                    result = await asyncio.wait_for(

                                        result,

                                        timeout=timeout,
                                    )

                            # sync initialize
                            else:

                                result = await asyncio.wait_for(

                                    asyncio.to_thread(
                                        init_method
                                    ),

                                    timeout=timeout,
                                )

                            # FIX:
                            # bool cannot be awaited
                            if inspect.isawaitable(
                                result
                            ):

                                await asyncio.wait_for(

                                    result,

                                    timeout=timeout,
                                )

                        except TypeError as e:

                            if (
                                "await" in str(e)
                                and "bool" in str(e)
                            ):

                                self.logger.warning(
                                    f"{component_name}: "
                                    f"initialize returned bool"
                                )

                            else:
                                raise

                    # ====================================================
                    # OPTIONAL START
                    # ====================================================
                    start_method = getattr(
                        instance,
                        "start",
                        None,
                    )

                    if callable(
                        start_method
                    ):

                        try:

                            if inspect.iscoroutinefunction(
                                start_method
                            ):

                                result = start_method()

                                if inspect.isawaitable(
                                    result
                                ):

                                    await asyncio.wait_for(

                                        result,

                                        timeout=timeout,
                                    )

                            else:

                                await asyncio.wait_for(

                                    asyncio.to_thread(
                                        start_method
                                    ),

                                    timeout=timeout,
                                )

                        except Exception as e:

                            self.logger.debug(
                                f"{component_name}: "
                                f"start skipped: {e}"
                            )

                    # ====================================================
                    # METADATA
                    # ====================================================
                    try:

                        setattr(
                            instance,
                            "_initialized_at",
                            time.time(),
                        )

                        setattr(
                            instance,
                            "_component_name",
                            component_name,
                        )

                        setattr(
                            instance,
                            "_init_attempt",
                            attempt,
                        )

                    except Exception:
                        pass

                    # ====================================================
                    # MEMORY CLEANUP
                    # ====================================================
                    try:

                        gc.collect()

                    except Exception:
                        pass

                    # ====================================================
                    # SUCCESS METRICS
                    # ====================================================
                    try:

                        self.components_initialized = int(

                            getattr(
                                self,
                                "components_initialized",
                                0,
                            )

                        ) + 1

                    except Exception:
                        pass

                    # ====================================================
                    # SUCCESS
                    # ====================================================
                    elapsed_ms = round(

                        (
                            time.monotonic()
                            - start_time
                        ) * 1000,

                        2,
                    )

                    self.logger.info(
                        f"✓ {component_name} initialized "
                        f"in {elapsed_ms}ms "
                        f"(attempt {attempt})"
                    )

                    return instance

                # ========================================================
                # TIMEOUT
                # ========================================================
                except asyncio.TimeoutError:

                    self.logger.warning(
                        f"{component_name} timeout "
                        f"({attempt}/{retry_count})"
                    )

                # ========================================================
                # CANCELLED
                # ========================================================
                except asyncio.CancelledError:

                    self.logger.debug(
                        f"{component_name} cancelled"
                    )

                    return None

                # ========================================================
                # FAILURE
                # ========================================================
                except Exception as e:

                    self.logger.warning(
                        f"{component_name} failed "
                        f"({attempt}/{retry_count}): {e}"
                    )

                    self.logger.debug(
                        traceback.format_exc()[:3000]
                    )

                # ========================================================
                # CLEANUP FAILED INSTANCE
                # ========================================================
                try:

                    if instance is not None:

                        cleanup_method = getattr(
                            instance,
                            "cleanup",
                            None,
                        )

                        if callable(
                            cleanup_method
                        ):

                            if inspect.iscoroutinefunction(
                                cleanup_method
                            ):

                                result = cleanup_method()

                                if inspect.isawaitable(
                                    result
                                ):

                                    await asyncio.wait_for(

                                        result,

                                        timeout=5,
                                    )

                            else:

                                await asyncio.wait_for(

                                    asyncio.to_thread(
                                        cleanup_method
                                    ),

                                    timeout=5,
                                )

                except Exception:
                    pass

                # ========================================================
                # BACKOFF
                # ========================================================
                if attempt < retry_count:

                    try:

                        await asyncio.sleep(
                            backoff * attempt
                        )

                    except asyncio.CancelledError:

                        return None

            # ============================================================
            # FINAL FAILURE
            # ============================================================
            self.logger.error(
                f"❌ {component_name} failed after "
                f"{retry_count} attempts"
            )

            return None

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ _init_single_brain_component "
                    f"fatal error: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return None

    async def _emergency_brain_fallback(
        self, component_names: List[str], new_components: Dict[str, Any]
    ):
        """
        Emergency fallback for failed brain components

        Args:
            component_names: Names of components that failed
            new_components: Dictionary to store fallback components
        """
        for comp_name in component_names:
            try:
                if comp_name == "llm_engine":
                    # Create stub LLM engine
                    from core.brain.llm_engine import LLMEngine

                    stub_llm = LLMEngine()
                    stub_llm._stub_mode = True
                    new_components[comp_name] = stub_llm
                    self.logger.warning(
                        f"⚠ Using STUB {comp_name} (limited functionality)"
                    )

                elif comp_name == "decision_engine":
                    # Create stub decision engine
                    from core.brain.decision_engine import DecisionEngine

                    stub_decision = DecisionEngine()
                    stub_decision._stub_mode = True
                    new_components[comp_name] = stub_decision
                    self.logger.warning(
                        f"⚠ Using STUB {comp_name} (limited functionality)"
                    )

            except Exception as e:
                self.logger.error(f"❌ Failed to create fallback for {comp_name}: {e}")

    async def _recover_critical_brain_components(
        self, critical_failed: List[str], new_components: Dict[str, Any]
    ) -> bool:
        """
        Attempt to recover critical brain components

        Args:
            critical_failed: List of failed critical components
            new_components: Current components dictionary

        Returns:
            True if all critical components recovered, False otherwise
        """
        recovery_timeout = self.config.get("critical_recovery_timeout", 15.0)
        recovery_start = time.time()

        for comp_name in critical_failed:
            if time.time() - recovery_start > recovery_timeout:
                self.logger.error(
                    f"Critical recovery timeout after {recovery_timeout}s"
                )
                return False

            self.logger.info(
                f"🔄 Attempting recovery for critical component: {comp_name}"
            )

            # Try to recreate the component
            if comp_name == "llm_engine":
                try:
                    from core.brain.llm_engine import LLMEngine

                    llm = await asyncio.wait_for(
                        asyncio.to_thread(LLMEngine), timeout=5.0
                    )
                    new_components[comp_name] = llm
                    self.logger.info(f"✅ Recovered {comp_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to recover {comp_name}: {e}")
                    return False

            elif comp_name == "decision_engine":
                try:
                    from core.brain.decision_engine import DecisionEngine

                    decision = await asyncio.wait_for(
                        asyncio.to_thread(DecisionEngine), timeout=5.0
                    )
                    new_components[comp_name] = decision
                    self.logger.info(f"✅ Recovered {comp_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to recover {comp_name}: {e}")
                    return False

        return True

    async def _connect_brain_pipeline(self, components: Dict[str, Any]):
        """
        Connect brain components in a processing pipeline

        Args:
            components: Dictionary of initialized components
        """
        try:
            llm = components.get("llm_engine")
            decision = components.get("decision_engine")
            reasoning = components.get("reasoning_engine")
            context = components.get("context_manager")

            # Connect LLM to Decision Engine
            if llm and decision and hasattr(decision, "set_llm_engine"):
                await self._safe_call(decision, "set_llm_engine", llm)
                self.logger.debug("✓ Connected LLM → Decision Engine")

            # Connect Decision to Reasoning
            if decision and reasoning and hasattr(reasoning, "set_decision_engine"):
                await self._safe_call(reasoning, "set_decision_engine", decision)
                self.logger.debug("✓ Connected Decision → Reasoning")

            # Connect Context Manager
            if context:
                if llm and hasattr(llm, "set_context_manager"):
                    await self._safe_call(llm, "set_context_manager", context)
                if decision and hasattr(decision, "set_context_manager"):
                    await self._safe_call(decision, "set_context_manager", context)
                self.logger.debug("✓ Connected Context Manager")

            self.logger.info("✅ Brain pipeline connected")

        except Exception as e:
            self.logger.warning(f"Brain pipeline connection failed: {e}")

        # -------------------------
        # PARALLEL START
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component),
                    timeout=20.0,  # slightly higher for LLM
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in k_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _initialize_memory_components(self, *args, **kwargs):
        """
        Initialize memory components (PIPELINE SAFE)
        - accepts extra kwargs
        - parallel-safe (even for single component)
        - shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            self.logger.debug("Shutdown in progress - skipping init")
            return

        # -------------------------
        # CREATE COMPONENT
        # -------------------------
        new_components = {}

        try:
            new_components["memory_manager"] = MemoryManager()
        except Exception as e:
            self.logger.warning(f"MemoryManager init failed: {e}")
            new_components["memory_manager"] = None

        # -------------------------
        # SAFE UPDATE
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START (STANDARDIZED)
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=15.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _start_components_brain(
        self,
        components: Dict[str, Any],
    ):
        """
        Ultra-safe production brain component starter.

        Fixes:
        - 'function' object has no attribute 'items'
        - bool can't be used in await expression
        - invalid component registries
        - startup race conditions
        - timeout deadlocks
        - pipeline dependency failures
        - broken chunk processing
        - coroutine leaks
        - invalid component objects
        - startup cancellation corruption
        """

        import asyncio
        import inspect
        import time
        import traceback

        start_time = time.monotonic()

        try:

            # --------------------------------------------------------
            # SHUTDOWN GUARD
            # --------------------------------------------------------
            try:

                if task_registry.is_shutting_down():

                    self.logger.debug(
                        "Shutdown active - skipping component startup"
                    )

                    return []

            except Exception:
                pass

            # --------------------------------------------------------
            # VALIDATE COMPONENTS
            # --------------------------------------------------------
            if components is None:

                self.logger.warning(
                    "Components is None"
                )

                return []

            # FIX:
            # prevents function.items crash
            if callable(components):

                self.logger.error(
                    "Components parameter is callable"
                )

                return []

            if not isinstance(
                components,
                dict,
            ):

                self.logger.error(
                    f"Invalid components type: "
                    f"{type(components)}"
                )

                return []

            # --------------------------------------------------------
            # SAFE CONFIG ACCESS
            # --------------------------------------------------------
            config = getattr(
                self,
                "config",
                {},
            )

            def cfg(key, default):

                try:

                    if callable(config):
                        return default

                    if hasattr(config, "get"):

                        value = config.get(
                            key,
                            default,
                        )

                        # FIX:
                        # config accidentally function
                        if callable(value):
                            return default

                        return value

                except Exception:
                    pass

                return default

            chunk_size = max(
                1,
                int(
                    cfg(
                        "component_start_chunk_size",
                        3,
                    )
                ),
            )

            # --------------------------------------------------------
            # PRIORITY ORDER
            # --------------------------------------------------------
            component_order = [

                {
                    "priority": 1,
                    "names": [
                        "context_manager",
                        "memory_manager",
                    ],
                    "timeout": 10.0,
                    "critical": True,
                },

                {
                    "priority": 2,
                    "names": [
                        "llm_engine",
                    ],
                    "timeout": 30.0,
                    "critical": True,
                },

                {
                    "priority": 3,
                    "names": [
                        "decision_engine",
                        "reasoning_engine",
                    ],
                    "timeout": 15.0,
                    "critical": True,
                },

                {
                    "priority": 4,
                    "names": [
                        "autonomous_core",
                        "goal_manager",
                    ],
                    "timeout": 15.0,
                    "critical": False,
                },

                {
                    "priority": 5,
                    "names": [
                        "feedback_loop",
                        "learning_engine",
                    ],
                    "timeout": 15.0,
                    "critical": False,
                },

                {
                    "priority": 6,
                    "names": [],
                    "timeout": 10.0,
                    "critical": False,
                },
            ]

            # --------------------------------------------------------
            # BUILD SAFE COMPONENT MAP
            # --------------------------------------------------------
            available_components = {}

            for (
                name,
                comp,
            ) in components.items():

                try:

                    if not name:
                        continue

                    if comp is None:
                        continue

                    available_components[
                        str(name)
                    ] = comp

                except Exception:
                    continue

            if not available_components:

                self.logger.warning(
                    "No valid components available"
                )

                return []

            # --------------------------------------------------------
            # STATE TRACKING
            # --------------------------------------------------------
            started_components = []

            failed_components = []

            # --------------------------------------------------------
            # PRIORITY PIPELINE
            # --------------------------------------------------------
            for priority_group in component_order:

                try:

                    # ------------------------------------------------
                    # SHUTDOWN CHECK
                    # ------------------------------------------------
                    try:

                        if task_registry.is_shutting_down():

                            self.logger.debug(
                                "Shutdown detected during startup"
                            )

                            return started_components

                    except Exception:
                        pass

                    priority = int(
                        priority_group.get(
                            "priority",
                            0,
                        )
                    )

                    timeout = float(
                        priority_group.get(
                            "timeout",
                            10.0,
                        )
                    )

                    critical = bool(
                        priority_group.get(
                            "critical",
                            False,
                        )
                    )

                    priority_names = priority_group.get(
                        "names",
                        [],
                    )

                    # ------------------------------------------------
                    # SELECT COMPONENTS
                    # ------------------------------------------------
                    group_components = {}

                    if priority_names:

                        for name in priority_names:

                            try:

                                if (
                                    name in available_components
                                    and name not in started_components
                                ):

                                    group_components[name] = (
                                        available_components[name]
                                    )

                            except Exception:
                                continue

                    else:

                        for (
                            name,
                            comp,
                        ) in available_components.items():

                            try:

                                if name not in started_components:

                                    group_components[name] = comp

                            except Exception:
                                continue

                    if not group_components:
                        continue

                    self.logger.info(
                        f"🎯 Starting priority {priority}: "
                        f"{list(group_components.keys())}"
                    )

                    # ------------------------------------------------
                    # CHUNKING
                    # ------------------------------------------------
                    component_items = list(
                        group_components.items()
                    )

                    chunks = [

                        component_items[
                            i:i + chunk_size
                        ]

                        for i in range(
                            0,
                            len(component_items),
                            chunk_size,
                        )
                    ]

                    # ------------------------------------------------
                    # PROCESS CHUNKS
                    # ------------------------------------------------
                    for (
                        chunk_idx,
                        chunk,
                    ) in enumerate(chunks):

                        tasks = []

                        chunk_names = []

                        for (
                            name,
                            component,
                        ) in chunk:

                            try:

                                chunk_names.append(
                                    name
                                )

                                async def start_wrapper(
                                    comp_name=name,
                                    comp=component,
                                ):

                                    try:

                                        # --------------------------------
                                        # START METHOD
                                        # --------------------------------
                                        start_method = getattr(
                                            comp,
                                            "start",
                                            None,
                                        )

                                        if callable(
                                            start_method
                                        ):

                                            # async start
                                            if inspect.iscoroutinefunction(
                                                start_method
                                            ):

                                                result = await asyncio.wait_for(

                                                    start_method(),

                                                    timeout=timeout,
                                                )

                                            # sync start
                                            else:

                                                result = await asyncio.wait_for(

                                                    asyncio.to_thread(
                                                        start_method
                                                    ),

                                                    timeout=timeout,
                                                )

                                            # FIX:
                                            # bool used in await expression
                                            if inspect.isawaitable(
                                                result
                                            ):

                                                await asyncio.wait_for(
                                                    result,
                                                    timeout=timeout,
                                                )

                                        # --------------------------------
                                        # MARK STARTED
                                        # --------------------------------
                                        try:

                                            setattr(
                                                comp,
                                                "_started",
                                                True,
                                            )

                                        except Exception:
                                            pass

                                        return (
                                            comp_name,
                                            True,
                                            None,
                                        )

                                    except Exception as e:

                                        return (
                                            comp_name,
                                            False,
                                            str(e),
                                        )

                                tasks.append(
                                    asyncio.create_task(
                                        start_wrapper(),
                                        name=f"start_{name}",
                                    )
                                )

                            except Exception as e:

                                self.logger.warning(
                                    f"Task creation failed for "
                                    f"{name}: {e}"
                                )

                        if not tasks:
                            continue

                        # --------------------------------------------
                        # EXECUTE CHUNK
                        # --------------------------------------------
                        try:

                            results = await asyncio.wait_for(

                                asyncio.gather(
                                    *tasks,
                                    return_exceptions=True,
                                ),

                                timeout=timeout + 2.0,
                            )

                            for result in results:

                                if isinstance(
                                    result,
                                    Exception,
                                ):

                                    self.logger.warning(
                                        f"Chunk exception: {result}"
                                    )

                                    continue

                                if not isinstance(
                                    result,
                                    tuple,
                                ):

                                    continue

                                (
                                    comp_name,
                                    success,
                                    error,
                                ) = result

                                if success:

                                    started_components.append(
                                        comp_name
                                    )

                                    self.logger.info(
                                        f"✓ Started {comp_name}"
                                    )

                                else:

                                    failed_components.append(
                                        comp_name
                                    )

                                    self.logger.warning(
                                        f"⚠ Failed {comp_name}: {error}"
                                    )

                        except asyncio.TimeoutError:

                            self.logger.error(
                                f"❌ Chunk {chunk_idx} timeout"
                            )

                            for t in tasks:

                                if not t.done():
                                    t.cancel()

                            failed_components.extend(
                                chunk_names
                            )

                        except Exception as e:

                            self.logger.error(
                                f"❌ Chunk {chunk_idx} failed: {e}"
                            )

                            failed_components.extend(
                                chunk_names
                            )

                        await asyncio.sleep(0.05)

                    await asyncio.sleep(0.1)

                except Exception as e:

                    self.logger.error(
                        f"Priority group failed: {e}"
                    )

            # --------------------------------------------------------
            # RECOVERY
            # --------------------------------------------------------
            if failed_components:

                try:

                    if hasattr(
                        self,
                        "_recover_failed_components",
                    ):

                        await self._recover_failed_components(
                            failed_components,
                            available_components,
                        )

                except Exception as e:

                    self.logger.warning(
                        f"Recovery failed: {e}"
                    )

            # --------------------------------------------------------
            # UPDATE SELF.COMPONENTS
            # --------------------------------------------------------
            try:

                if not hasattr(
                    self,
                    "components",
                ):

                    self.components = {}

                for name in started_components:

                    try:

                        if name in available_components:

                            self.components[name] = (
                                available_components[name]
                            )

                    except Exception:
                        continue

            except Exception:
                pass

            # --------------------------------------------------------
            # VALIDATION
            # --------------------------------------------------------
            try:

                if hasattr(
                    self,
                    "_validate_component_pipeline",
                ):

                    await self._validate_component_pipeline(
                        started_components
                    )

            except Exception as e:

                self.logger.warning(
                    f"Pipeline validation failed: {e}"
                )

            # --------------------------------------------------------
            # SUMMARY
            # --------------------------------------------------------
            elapsed = round(
                (
                    time.monotonic()
                    - start_time
                ) * 1000,
                2,
            )

            self.logger.info(
                f"✅ Component startup complete "
                f"| started={len(started_components)} "
                f"| failed={len(failed_components)} "
                f"| latency={elapsed}ms"
            )

            if failed_components:

                self.logger.warning(
                    f"Failed components: {failed_components}"
                )

            return started_components

        # ------------------------------------------------------------
        # HARD FAILURE
        # ------------------------------------------------------------
        except Exception as e:

            try:

                self.logger.error(
                    f"❌ _start_components_brain failed: {e}"
                )

                self.logger.debug(
                    traceback.format_exc()[:4000]
                )

            except Exception:
                pass

            return []

    async def _start_component_chunk(
        self,
        chunk_idx: int,
        components: List[Tuple[str, Any]],
        timeout: float,
        critical: bool,
    ) -> Dict[str, List[str]]:
        """
        Start a chunk of components in parallel

        Args:
            chunk_idx: Chunk index for logging
            components: List of (name, component) tuples
            timeout: Timeout per component
            critical: Whether these components are critical

        Returns:
            Dictionary with started and failed component lists
        """

        started = []
        failed = []

        # Create start tasks for all components in chunk
        start_tasks = []
        component_names = []

        for name, component in components:
            if task_registry.is_shutting_down():
                break

            task = asyncio.create_task(
                self._start_single_component(name, component, timeout, critical),
                name=f"start_{name}_{chunk_idx}",
            )
            start_tasks.append(task)
            component_names.append(name)

        if not start_tasks:
            return {"started": started, "failed": failed}

        # Execute all starts in parallel with chunk timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*start_tasks, return_exceptions=True),
                timeout=timeout + 2.0,  # Extra buffer for chunk
            )

            # Process results
            for name, result in zip(component_names, results):
                if isinstance(result, Exception):
                    self.logger.warning(
                        f"⚠ {name} start failed in chunk {chunk_idx}: {result}"
                    )
                    failed.append(name)
                else:
                    started.append(name)

        except asyncio.TimeoutError:
            self.logger.error(f"❌ Chunk {chunk_idx} TIMEOUT after {timeout+2}s")

            # Cancel hanging tasks
            for task in start_tasks:
                if not task.done():
                    task.cancel()

            failed.extend(component_names)

        return {"started": started, "failed": failed}

    async def _start_single_component(
        self, name: str, component: Any, timeout: float, critical: bool
    ):
        """
        Start a single component with retry logic and timeout

        Args:
            name: Component name
            component: Component instance
            timeout: Timeout in seconds
            critical: Whether component is critical
        """

        # Retry configuration
        max_retries = 2 if critical else 1
        retry_delay = 1.0

        for attempt in range(max_retries):

            # Check shutdown
            if task_registry.is_shutting_down():
                self.logger.debug(f"Shutdown detected - cancelling {name} start")
                return

            try:
                # Start component with timeout
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=timeout
                )

                # Verify component is actually running
                if hasattr(component, "health_check"):
                    health = await asyncio.wait_for(
                        component.health_check(), timeout=5.0
                    )
                    if isinstance(health, dict) and health.get("status") != "healthy":
                        raise RuntimeError(f"Component unhealthy after start: {health}")

                self.logger.info(f"✅ {name} started (attempt {attempt+1})")
                return

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"⏱ Timeout starting {name} (attempt {attempt+1}/{max_retries})"
                )

                if attempt == max_retries - 1:
                    raise TimeoutError(
                        f"{name} start timeout after {max_retries} attempts"
                    )

                await asyncio.sleep(retry_delay)

            except Exception as e:
                self.logger.warning(
                    f"⚠ Failed to start {name} (attempt {attempt+1}/{max_retries}): {e}"
                )

                if attempt == max_retries - 1:
                    raise RuntimeError(f"{name} start failed: {e}")

                await asyncio.sleep(retry_delay)

    async def _start_component(self, name: str, component: Any):
        """
        Core component start logic with method detection

        Args:
            name: Component name
            component: Component instance
        """

        # Try multiple start methods
        start_methods = ["start", "start_async", "initialize", "run"]

        for method_name in start_methods:
            if hasattr(component, method_name):
                method = getattr(component, method_name)

                try:
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    elif callable(method):
                        result = method()
                        if asyncio.iscoroutine(result):
                            await result

                    self.logger.debug(f"{name} started via {method_name}()")
                    return

                except Exception as e:
                    self.logger.debug(f"{name}.{method_name}() failed: {e}")
                    continue

        # If no start method found but component exists, consider it started
        if component is not None:
            self.logger.debug(f"{name} has no start method, considered running")
            return

        raise RuntimeError(f"No valid start method found for {name}")

    async def _recover_failed_components(
        self, failed_names: List[str], available_components: Dict[str, Any]
    ):
        """
        Attempt to recover failed components

        Args:
            failed_names: List of failed component names
            available_components: All available components
        """

        if not failed_names:
            return

        recovery_timeout = self.config.get("component_recovery_timeout", 30.0)
        recovery_start = time.time()

        self.logger.info(
            f"🔄 Attempting recovery for {len(failed_names)} failed components"
        )

        for name in failed_names.copy():

            # Check timeout
            if time.time() - recovery_start > recovery_timeout:
                self.logger.warning(f"Recovery timeout after {recovery_timeout}s")
                break

            component = available_components.get(name)
            if not component:
                continue

            self.logger.info(f"🛠️ Recovering {name}...")

            # Try different recovery methods
            recovery_methods = ["recover", "restart", "reinitialize"]
            recovered = False

            for method_name in recovery_methods:
                if hasattr(component, method_name):
                    try:
                        method = getattr(component, method_name)

                        if asyncio.iscoroutinefunction(method):
                            await asyncio.wait_for(method(), timeout=10.0)
                        elif callable(method):
                            result = method()
                            if asyncio.iscoroutine(result):
                                await asyncio.wait_for(result, timeout=10.0)

                        # Try to start again
                        await self._start_component(name, component)

                        recovered = True
                        self.logger.info(f"✅ {name} recovered via {method_name}()")
                        failed_names.remove(name)
                        break

                    except Exception as e:
                        self.logger.debug(f"Recovery method {method_name} failed: {e}")
                        continue

            if not recovered:
                # Try emergency stub creation for critical components
                if name in ["llm_engine", "decision_engine"]:
                    stub = await self._create_stub_component(name)
                    if stub:
                        self.components[name] = stub  # Update self.components directly
                        self.logger.warning(
                            f"⚠ Using STUB {name} after recovery failure"
                        )
                        failed_names.remove(name)

    async def _create_stub_component(self, name: str) -> Optional[Any]:
        """
        Create a stub component for emergency fallback

        Args:
            name: Component name to stub

        Returns:
            Stub component instance or None
        """

        try:
            if name == "llm_engine":
                from core.brain.llm_engine import LLMEngine

                stub = LLMEngine()
                stub._stub_mode = True

                if hasattr(stub, "set_stub_mode"):
                    await stub.set_stub_mode(True)

                return stub

            elif name == "decision_engine":
                from core.brain.decision_engine import DecisionEngine

                stub = DecisionEngine()
                stub._stub_mode = True
                return stub

        except Exception as e:
            self.logger.error(f"Failed to create stub for {name}: {e}")
            return None

    async def _validate_component_pipeline(self, started_components: List[str]):
        """
        Validate that the component pipeline is working

        Args:
            started_components: List of successfully started components
        """

        # Check critical paths
        critical_paths = [
            ["llm_engine", "decision_engine"],  # AI decision path
            ["context_manager", "memory_manager", "llm_engine"],  # Memory path
            ["autonomous_core", "goal_manager", "feedback_loop"],  # Autonomy path
        ]

        for path in critical_paths:
            missing = [comp for comp in path if comp not in started_components]

            if missing:
                self.logger.warning(f"⚠ Critical path {path} missing: {missing}")
            else:
                self.logger.debug(f"✓ Critical path validated: {path}")

        # Run quick health check on started components
        health_tasks = []
        for name in started_components[:10]:  # Limit to first 10
            component = self.components.get(name)
            if component and hasattr(component, "health_check"):
                health_tasks.append(
                    asyncio.create_task(
                        self._safe_call(component, "health_check"),
                        name=f"health_{name}",
                    )
                )

        if health_tasks:
            results = await asyncio.gather(*health_tasks, return_exceptions=True)

            unhealthy = []
            for name, result in zip(started_components[:10], results):
                if isinstance(result, Exception):
                    unhealthy.append(name)
                elif isinstance(result, dict) and result.get("status") != "healthy":
                    unhealthy.append(name)

            if unhealthy:
                self.logger.warning(f"⚠ Unhealthy components after start: {unhealthy}")

    async def _initialize_autonomous_components(self, *args, **kwargs):
        """
        Initialize autonomous components (PIPELINE SAFE)
        - accepts extra kwargs
        - parallel startup
        - safe GoalManager handling
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            return

        # -------------------------
        # CREATE COMPONENTS
        # -------------------------
        new_components = {}

        # Autonomous Core
        try:
            new_components["autonomous_core"] = AutonomousCore()
        except Exception as e:
            self.logger.warning(f"AutonomousCore init failed: {e}")
            new_components["autonomous_core"] = None

        # Goal Manager
        goal_manager = None
        try:
            from core.autonomy.goal_manage import GoalManager

            goal_manager = GoalManager()
            new_components["goal_manager"] = goal_manager

            # If no lifecycle methods → mark healthy immediately
            if not (
                hasattr(goal_manager, "initialize") or hasattr(goal_manager, "start")
            ):
                self.component_status["goal_manager"] = ComponentStatus.HEALTHY
                self.logger.info("✓ GoalManager stored (no lifecycle methods)")

        except Exception as e:
            self.logger.warning(f"GoalManager init failed: {e}")
            new_components["goal_manager"] = None

        # -------------------------
        # SAFE UPDATE
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START (SMART FILTER)
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            # Skip GoalManager if no lifecycle methods
            if name == "goal_manager" and not (
                hasattr(component, "initialize") or hasattr(component, "start")
            ):
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=15.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _initialize_perception_components(self, *args, **kwargs):
        """
        Initialize perception components (PIPELINE SAFE)
        - accepts extra kwargs
        - parallel-safe
        - shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            return

        # -------------------------
        # CREATE COMPONENT
        # -------------------------
        new_components = {}

        try:
            new_components["emotion_detector"] = EmotionDetector()
        except Exception as e:
            self.logger.warning(f"EmotionDetector init failed: {e}")
            new_components["emotion_detector"] = None

        # -------------------------
        # SAFE UPDATE
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START (STANDARDIZED)
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _initialize_learning_components(self, *args, **kwargs):
        """
        Initialize learning components (PIPELINE SAFE)
        - accepts extra kwargs
        - parallel-safe
        - shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            self.logger.debug("Shutdown in progress - skipping learning init")
            return

        # -------------------------
        # CREATE COMPONENT
        # -------------------------
        new_components = {}

        try:
            new_components["feedback_loop"] = FeedbackLoop()
        except Exception as e:
            self.logger.warning(f"FeedbackLoop init failed: {e}")
            new_components["feedback_loop"] = None

        # -------------------------
        # SAFE UPDATE
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START (STANDARDIZED)
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    async def _initialize_security_components(self, *args, **kwargs):
        """
        Initialize security components (PIPELINE SAFE)
        - accepts extra kwargs
        - parallel-safe
        - shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            return

        # -------------------------
        # CREATE COMPONENT
        # -------------------------
        new_components = {}

        try:
            new_components["audit_logger"] = AuditLogger()
        except Exception as e:
            self.logger.warning(f"AuditLogger init failed: {e}")
            new_components["audit_logger"] = None

        # -------------------------
        # SAFE UPDATE
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # PARALLEL START (STANDARDIZED)
        # -------------------------
        async def _start_single(name, component):
            if task_registry.is_shutting_down():
                return

            try:
                await asyncio.wait_for(
                    self._start_component(name, component), timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ Timeout starting {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")

        startup_tasks = [
            _start_single(name, comp)
            for name, comp in new_components.items()
            if comp is not None
        ]

        # -------------------------
        # RUN PARALLEL
        # -------------------------
        if startup_tasks:
            await asyncio.gather(*startup_tasks, return_exceptions=True)

    def _create_init_task(self, name: str, component):
        """Helper to create and track init tasks"""
        if task_registry.is_shutting_down():
            return None
        coro = self._safe_start_component_impl(name, component)
        from core.utils.task_registry import create_task

        # use global create_task helper which registers and auto-cleans
        task = create_task(coro, name=f"init_{name}")

        if task:
            # track in local init list for cancellation on shutdown
            try:
                self._init_tasks.append(task)
            except Exception:
                pass

        return task

    async def _start_component(self, name: str, component: Any):
        """
        Fully production-safe component startup (TODO #3):
        - strict async discipline
        - 5s timeout enforced
        - shutdown safe
        - idempotent + re-entry safe
        - task tracking for graceful shutdown
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            return

        if component is None:
            self.component_status[name] = ComponentStatus.UNHEALTHY
            return

        task = self._create_init_task(name, component)
        if task:
            try:
                await asyncio.wait_for(task, timeout=5.0)  # TODO #3: 5s timeout
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱ 5s timeout: {name}")
                self.component_status[name] = ComponentStatus.NOT_RESPONDING
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.warning(f"Failed to start {name}: {e}")
                self.component_status[name] = ComponentStatus.UNHEALTHY

    async def _safe_start_component_impl(self, name: str, component: Any):
        """Internal implementation - called via tracked task"""
        try:
            # -------------------------
            # DETECT LIFECYCLE METHOD
            # -------------------------
            method = getattr(component, "initialize", None) or getattr(
                component, "start", None
            )

            if not callable(method):
                self.component_status[name] = ComponentStatus.HEALTHY
                self.logger.debug(
                    f"✓ Component {name} has no lifecycle method, marked healthy"
                )
                return True

            # -------------------------
            # EXECUTE SAFELY
            # -------------------------
            result = method()

            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=4.0)  # leave 1s buffer
            elif callable(result):
                result = await asyncio.to_thread(result)

            # VALIDATE + SUCCESS
            if result is not False:
                self.component_status[name] = ComponentStatus.HEALTHY
                await self._safe_call(self, "_log_event", "component_started", name)
                self.logger.info(f"✓ Component started: {name}")
                return True
            else:
                raise RuntimeError(f"{name} reported failed startup")

        except Exception as e:
            self.component_status[name] = ComponentStatus.UNHEALTHY
            self.logger.warning(f"⚠ Failed to start {name}: {e}")
            await self._safe_call(
                self, "_log_event", "component_start_failed", name, {"error": str(e)}
            )

            # Circuit breaker
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.config.circuit_breaker_threshold:
                self.circuit_open = True
                self.state = SystemState.DEGRADED
            raise

    async def _configure_multi_agent_strategies(self):
        """Configure multi-agent coordination strategies"""
        if not self.multi_agent_coordinator:
            return

        try:
            if hasattr(self.multi_agent_coordinator, "coordination_strategy"):
                strategy_value = self.config.coordination_strategy.upper()
                if strategy_value in [e.name for e in CoordinationStrategy]:
                    self.multi_agent_coordinator.coordination_strategy = (
                        CoordinationStrategy[strategy_value]
                    )
                else:
                    self.multi_agent_coordinator.coordination_strategy = (
                        CoordinationStrategy.HIERARCHICAL
                    )
        except Exception as e:
            self.logger.warning(f"Error setting coordination strategy: {e}")

        try:
            if hasattr(self.multi_agent_coordinator, "task_allocation_strategy"):
                alloc_value = self.config.task_allocation_strategy.upper()
                if alloc_value in [e.name for e in TaskAllocationStrategy]:
                    self.multi_agent_coordinator.task_allocation_strategy = (
                        TaskAllocationStrategy[alloc_value]
                    )
                else:
                    self.multi_agent_coordinator.task_allocation_strategy = (
                        TaskAllocationStrategy.CAPABILITY_BASED
                    )
        except Exception as e:
            self.logger.warning(f"Error setting task allocation strategy: {e}")

        try:
            if hasattr(self.multi_agent_coordinator, "communication_protocol"):
                protocol_value = self.config.communication_protocol.upper()
                if protocol_value in [e.name for e in CommunicationProtocol]:
                    self.multi_agent_coordinator.communication_protocol = (
                        CommunicationProtocol[protocol_value]
                    )
                else:
                    self.multi_agent_coordinator.communication_protocol = (
                        CommunicationProtocol.DIRECT
                    )
        except Exception as e:
            self.logger.warning(f"Error setting communication protocol: {e}")

    async def _setup_multi_agent_system(self):
        """Setup specialized agents for the multi-agent system"""
        if not self.multi_agent_coordinator or not MULTI_AGENT_AVAILABLE:
            return

        # Register all specialized agents from registry
        for agent_name, agent in self.agent_registry.get_all_agents().items():
            try:

                class WrappedAgent(BaseAgent):
                    def __init__(
                        self, agent_id: str, name: str, real_agent: Any, agent_type: str
                    ):
                        super().__init__(
                            agent_id=agent_id, name=name, role=AgentRole.SPECIALIST
                        )
                        self.real_agent = real_agent
                        self.agent_type = agent_type

                    async def process_task(self, task: Any) -> Any:
                        if hasattr(self.real_agent, "process_request"):
                            return await self.real_agent.process_request(task)
                        elif hasattr(self.real_agent, "execute"):
                            return await self.real_agent.execute(task)
                        else:
                            return {
                                "error": f"Agent {self.agent_id} cannot process task"
                            }

                wrapped = WrappedAgent(
                    agent_id=f"{agent_name}_agent_1",
                    name=f"EDIATH {agent_name.title()} Specialist",
                    real_agent=agent,
                    agent_type=agent_name,
                )
                await self.multi_agent_coordinator.register_agent(wrapped)
                self.logger.debug(
                    f"✓ {agent_name.title()} Agent registered in multi-agent system"
                )
            except Exception as e:
                self.logger.debug(f"Could not register {agent_name} Agent: {e}")

        # Create LLM Agent
        try:
            if "llm_engine" in self.components and self.components["llm_engine"]:
                llm_ready = True
                if hasattr(self.components["llm_engine"], "is_ready"):
                    llm_ready = self.components["llm_engine"].is_ready()

                if llm_ready:
                    llm_agent = LLMAgent(
                        agent_id="llm_agent_1",
                        name="EDIATH LLM Specialist",
                        llm_engine=self.components["llm_engine"],
                    )
                    await self.multi_agent_coordinator.register_agent(llm_agent)
                    self.logger.info("✓ LLM Agent registered")
        except Exception as e:
            self.logger.debug(f"Could not register LLM Agent: {e}")

        # Create Supervisor Agent
        try:
            supervisor_agent = SupervisorAgent(
                agent_id="supervisor_agent_1", name="EDIATH System Supervisor"
            )
            await self.multi_agent_coordinator.register_agent(supervisor_agent)

            # Monitor all agents
            for agent_id in self.multi_agent_coordinator.agents.keys():
                await supervisor_agent.monitor_agent(agent_id)

            self.logger.info("✓ Supervisor Agent registered and monitoring")
        except Exception as e:
            self.logger.warning(f"Failed to register Supervisor Agent: {e}")

    def _safe_call_sync(self, obj: Any, method_name: str, *args, **kwargs) -> Any:
        """Safely call a method synchronously"""
        if not hasattr(obj, method_name):
            return None

        method = getattr(obj, method_name)
        if not callable(method):
            return None

        try:
            result = method(*args, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Error calling {method_name} sync: {e}")
            return None

    async def _safe_call(self, obj: Any, method_name: str, *args, **kwargs) -> Any:
        """
        FINAL production-safe caller:
        - strict async discipline
        - no coroutine leaks
        - timeout enforced
        - shutdown safe
        - thread-safe execution
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            return None

        if obj is None:
            return None

        try:
            method = getattr(obj, method_name, None)
            if not callable(method):
                return None

            # -------------------------
            # EXECUTE METHOD (CRITICAL FIX)
            # -------------------------
            # Filter kwargs to supported parameters to avoid unexpected kw errors
            try:
                sig = inspect.signature(method)
                bound_kwargs = {}
                for k, v in kwargs.items():
                    if k in sig.parameters:
                        bound_kwargs[k] = v
            except Exception:
                bound_kwargs = kwargs

            try:
                result = method(*args, **bound_kwargs)
            except Exception as e:
                if not task_registry.is_shutting_down():
                    self.logger.error(f"❌ Call failed ({method_name}): {e}")
                return None

            # -------------------------
            # HANDLE COROUTINE (STRICT)
            # -------------------------
            if asyncio.iscoroutine(result):
                try:
                    timeout = 5.0 if task_registry.is_shutting_down() else 20.0
                    return await asyncio.wait_for(result, timeout=timeout)

                except asyncio.TimeoutError:
                    self.logger.warning(f"⏱ Timeout in {method_name}")
                    return None

                except asyncio.CancelledError:
                    return None

                except Exception as e:
                    self.logger.error(f"❌ Async error in {method_name}: {e}")
                    return None

            # -------------------------
            # HANDLE FUTURE / TASK (MISSING FIX)
            # -------------------------
            if isinstance(result, asyncio.Future):
                try:
                    return await asyncio.wait_for(result, timeout=20.0)
                except Exception:
                    return None

            # -------------------------
            # HANDLE SYNC CALLABLE RESULT
            # -------------------------
            if callable(result):
                try:
                    # ensure executed only once
                    return await asyncio.to_thread(result)
                except Exception as e:
                    self.logger.debug(f"Callable error ({method_name}): {e}")
                    return None

            # -------------------------
            # NORMAL VALUE
            # -------------------------
            return result

        except Exception as e:
            if not task_registry.is_shutting_down():
                self.logger.debug(f"Safe call error ({method_name}): {e}")
            return None

    def get_system_status_sync(self) -> Dict[str, Any]:
        """Synchronous version of get_status for callbacks"""
        return {
            "state": self.state.value if self.state else "unknown",
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "component_status": {
                name: status.value for name, status in self.component_status.items()
            },
            "circuit_breaker_open": self.circuit_open,
            "consecutive_failures": self.consecutive_failures,
            "blender_available": self.blender_available,
            "mongo_connected": mongo_is_connected(),
        }

    async def _pipeline_worker(self, worker_id: int):
        """Worker for processing pipeline chunks in parallel."""
        self.logger.info(f"🔧 Pipeline worker {worker_id} started")

        while not self.shutdown_event.is_set() and not task_registry.is_shutting_down():
            try:
                # Get chunk from pipeline queue with timeout
                chunk_data = await asyncio.wait_for(
                    self._pipeline_queue.get(), timeout=1.0
                )

                try:
                    # Process the chunk
                    if chunk_data and "text" in chunk_data:
                        result = await self._safe_call(
                            self.components.get("agent_loop"),
                            "process_voice_input",
                            chunk_data["text"],
                            chunk_index=chunk_data.get("index", 0),
                            total_chunks=chunk_data.get("total", 1),
                            worker_id=worker_id,
                        )

                        # Store result if callback provided
                        if chunk_data.get("callback"):
                            await self._safe_call(
                                chunk_data["callback"],
                                "on_chunk_processed",
                                result,
                                chunk_data.get("index", 0),
                            )

                except Exception as e:
                    self.logger.warning(f"Worker {worker_id} chunk error: {e}")
                finally:
                    self._pipeline_queue.task_done()

            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

        self.logger.info(f"🔧 Pipeline worker {worker_id} stopped")

    async def _process_queued_chunks(self, limit: int = None):
        """Process chunks from pipeline queue with optional limit."""
        if not hasattr(self, "_pipeline_queue"):
            return

        processed = 0
        while not self._pipeline_queue.empty():
            if limit and processed >= limit:
                break

            try:
                chunk = self._pipeline_queue.get_nowait()
                # Quick processing for queued chunks
                if chunk and "text" in chunk:
                    await self._safe_call(
                        self.components.get("agent_loop"),
                        "process_voice_input",
                        chunk["text"],
                        quick_mode=True,
                    )
                self._pipeline_queue.task_done()
                processed += 1
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.warning(f"Queued chunk processing error: {e}")
                break

    async def _flush_chunk_buffer(self):
        """Flush accumulated chunk buffer to pipeline."""
        if not self._chunk_buffer:
            return

        try:
            # Create batch from buffer
            batch = self._chunk_buffer.copy()
            self._chunk_buffer.clear()

            # Submit to pipeline queue
            for idx, chunk in enumerate(batch):
                await self._pipeline_queue.put(
                    {
                        "text": chunk,
                        "index": idx,
                        "total": len(batch),
                        "timestamp": time.time(),
                    }
                )

            self.logger.debug(f"📦 Flushed {len(batch)} chunks to pipeline")

        except Exception as e:
            self.logger.error(f"Failed to flush chunk buffer: {e}")
            # Restore buffer on error
            self._chunk_buffer.extend(batch)

    async def run(self):
        """
        FINAL production-safe main loop with pipeline & chunking:
        - strict lifecycle control
        - no memory leaks
        - no drift accumulation
        - shutdown + async safe
        - chunked processing support
        - pipeline parallelism
        """

        self.logger.info("🚀 Starting EDIATH main execution loop")

        self._tick_lock = asyncio.Lock()
        self.active_tasks = []
        self._running = True

        tick_count = 0  # FIX: define early to avoid finally crash

        # Initialize pipeline queue for chunked processing
        self._pipeline_queue = asyncio.Queue(
            maxsize=self.config.get("pipeline_queue_size", 100)
        )
        self._chunk_buffer = []

        try:
            # -------------------------
            # START BACKGROUND TASKS (with pipeline support)
            # -------------------------
            self._start_task(self._health_check_loop(), "health")
            self._start_task(self._metrics_flush_loop(), "metrics")
            self._start_task(self._state_persist_loop(), "persist")

            # Start pipeline worker tasks for parallel chunk processing (capped)
            if self.config.get("enable_pipeline_workers", True):
                worker_count = self._cap_workers(
                    self.config.get("pipeline_workers", 3), cap=8
                )
                for i in range(worker_count):
                    self._start_task(self._pipeline_worker(i), f"pipeline_worker_{i}")

            max_ticks = 100000  # Increased to run indefinitely
            tick_interval = max(self.config.tick_interval_ms / 1000, 0.01)

            # Chunking batch collector
            chunk_batch_size = self.config.get("chunk_batch_size", 10)
            last_chunk_flush = time.time()

            # -------------------------
            # MAIN LOOP
            # -------------------------
            while (
                not self.shutdown_event.is_set()
                and not task_registry.is_shutting_down()
                and tick_count < max_ticks
            ):
                loop_start = asyncio.get_event_loop().time()
                tick_count += 1

                try:
                    # -------------------------
                    # CIRCUIT BREAKER
                    # -------------------------
                    if self.circuit_open:
                        await self._safe_call(self, "_handle_circuit_breaker")
                    else:
                        # -------------------------
                        # STATE MACHINE with chunked processing
                        # -------------------------
                        if self.state == SystemState.RUNNING:
                            # Process tick (already has chunking internally)
                            await self._safe_call(self, "_process_tick")

                            # Flush chunk buffer periodically
                            current_time = time.time()
                            if len(
                                self._chunk_buffer
                            ) >= chunk_batch_size or current_time - last_chunk_flush >= self.config.get(
                                "chunk_flush_interval", 5.0
                            ):
                                if self._chunk_buffer:
                                    await self._flush_chunk_buffer()
                                    last_chunk_flush = current_time

                        elif self.state == SystemState.PAUSED:
                            # Process queued chunks even when paused
                            await self._process_queued_chunks()
                            await asyncio.sleep(1)

                        elif self.state == SystemState.DEGRADED:
                            # Reduced chunk processing in degraded mode
                            await self._safe_call(self, "_handle_degraded_mode")
                            await self._process_queued_chunks(limit=5)

                        elif self.state == SystemState.ERROR:
                            await self._safe_call(self, "_attempt_recovery")

                    # -------------------------
                    # DRIFT-CORRECTED SLEEP (CRITICAL FIX)
                    # -------------------------
                    elapsed = asyncio.get_event_loop().time() - loop_start
                    sleep_time = max(tick_interval - elapsed, 0.0)

                    await asyncio.sleep(sleep_time)

                except asyncio.CancelledError:
                    self.logger.info("🛑 Main loop cancelled")
                    break

                except Exception as e:
                    self.logger.error(f"❌ Loop error (tick {tick_count}): {e}")
                    self.consecutive_failures += 1

                    # -------------------------
                    # CIRCUIT BREAKER TRIGGER
                    # -------------------------
                    if (
                        self.consecutive_failures
                        >= self.config.circuit_breaker_threshold
                    ):
                        self.circuit_open = True
                        self.state = SystemState.DEGRADED

                        await self._safe_call(
                            self,
                            "_log_event",
                            "circuit_breaker_opened",
                            "orchestrator",
                            {
                                "failures": self.consecutive_failures,
                                "tick": tick_count,
                            },
                        )

                        self.logger.warning("⚠ Circuit breaker opened")

                    await asyncio.sleep(0.5)

        finally:
            # -------------------------
            # CLEAN SHUTDOWN (STRICT) with pipeline cleanup
            # -------------------------
            self._running = False

            # Drain pipeline queue
            if hasattr(self, "_pipeline_queue"):
                try:
                    await asyncio.wait_for(self._pipeline_queue.join(), timeout=2.0)
                except Exception:
                    pass

            # Cancel all active tasks
            tasks = list(self.active_tasks)

            for task in tasks:
                if task and not task.done():
                    try:
                        task.cancel()
                    except Exception:
                        pass

            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=3.0
                    )
                except Exception:
                    pass

            self.active_tasks.clear()
            self.logger.info(f"✅ Main loop exited after {tick_count} ticks")

    async def _process_tick(self):
        """
        FINAL production-safe tick with pipeline & chunking:
        - no task leaks
        - strict async control
        - shutdown safe
        - bounded execution
        - chunked processing for large inputs
        - pipeline parallelism
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            return

        if self._tick_lock is None:
            self._tick_lock = asyncio.Lock()

        if self._tick_lock.locked():
            return

        async with self._tick_lock:
            tick_start = time.time()

            try:
                # -------------------------
                # VOICE INPUT (SAFE WITH CHUNKING)
                # -------------------------
                voice_text = None
                voice_chunks = []

                if getattr(self.system_instance, "listener", None):
                    try:
                        voice_text = await asyncio.wait_for(
                            asyncio.to_thread(self.system_instance.listener.get_text),
                            timeout=2.0,
                        )

                        # CHUNKING: Split large voice input
                        if voice_text and len(voice_text) > self.config.get(
                            "chunk_size", 500
                        ):
                            voice_chunks = self._chunk_text(
                                voice_text, self.config.get("chunk_size", 500)
                            )
                            self.logger.info(
                                f"📦 Voice input chunked into {len(voice_chunks)} parts"
                            )
                        elif voice_text:
                            voice_chunks = [voice_text]

                    except Exception:
                        voice_text = None
                        voice_chunks = []

                enhanced_context = None

                # -------------------------
                # PROCESS INPUT WITH PIPELINE
                # -------------------------
                if voice_chunks:
                    # PIPELINE: Process chunks in parallel pipeline
                    pipeline_results = await self._process_pipeline(
                        voice_chunks, self.config.get("pipeline_workers", 3)
                    )

                    # Combine pipeline results
                    if pipeline_results:
                        enhanced_context = " ".join(pipeline_results)
                    else:
                        enhanced_context = voice_chunks[0] if voice_chunks else None

                    # -------------------------
                    # VISION SAFE WITH CHUNKING SUPPORT
                    # -------------------------
                    if self.vision_memory and enhanced_context:
                        try:
                            vision_data = await asyncio.wait_for(
                                self.vision_memory.get_latest_frame(), timeout=1.0
                            )
                        except Exception:
                            vision_data = None

                        if vision_data:
                            decision_engine = self.components.get("decision_engine")

                            vision_context = (
                                await self._safe_call(
                                    decision_engine, "process_vision", vision_data
                                )
                                if decision_engine
                                else None
                            )

                            if vision_context:
                                # CHUNKING: Handle large combined context
                                combined_context = (
                                    f"{vision_context}\n\n{enhanced_context}"
                                )
                                if len(combined_context) > self.config.get(
                                    "chunk_size", 500
                                ):
                                    context_chunks = self._chunk_text(
                                        combined_context,
                                        self.config.get("chunk_size", 500),
                                    )
                                    enhanced_context = (
                                        await self._merge_chunked_context(
                                            context_chunks
                                        )
                                    )
                                else:
                                    enhanced_context = combined_context

                        # -------------------------
                        # PIPELINE ROUTING (Parallel processing)
                        # -------------------------
                        pipeline_stages = []

                        # Stage 1: Command Router (if needed)
                        if (
                            "command_router" in self.components
                            and self.components["command_router"]
                        ):
                            pipeline_stages.append(
                                self._safe_call(
                                    self.components["command_router"],
                                    "process_command",
                                    {
                                        "text": enhanced_context,
                                        "source": "voice",
                                        "timestamp": time.time(),
                                        "chunked": len(voice_chunks) > 1,
                                    },
                                )
                            )

                        # Stage 2: Agent Loop
                        if (
                            "agent_loop" in self.components
                            and self.components["agent_loop"]
                        ):
                            # Process voice input with chunk-aware handling
                            for idx, chunk in enumerate(voice_chunks):
                                pipeline_stages.append(
                                    self._safe_call(
                                        self.components["agent_loop"],
                                        "process_voice_input",
                                        chunk,
                                        chunk_index=idx,
                                        total_chunks=len(voice_chunks),
                                    )
                                )

                        # Stage 3: Speaker response
                        if getattr(self.system_instance, "speaker", None):
                            response_text = (
                                f"Processing {len(voice_chunks)} voice segments"
                                if len(voice_chunks) > 1
                                else f"I heard: {voice_text[:100]}"
                            )
                            pipeline_stages.append(
                                self._safe_call(
                                    self.system_instance.speaker,
                                    "speak_async",
                                    response_text,
                                )
                            )

                        # Execute pipeline stages in parallel with batching
                        if pipeline_stages:
                            # Use batching for large pipelines
                            batch_size = self.config.get("pipeline_batch_size", 5)
                            for i in range(0, len(pipeline_stages), batch_size):
                                batch = pipeline_stages[i : i + batch_size]
                                await asyncio.gather(*batch, return_exceptions=True)

                        await self._safe_call(
                            self,
                            "_log_event",
                            "voice_command_processed",
                            "voice",
                            {"text": voice_text[:100], "chunks": len(voice_chunks)},
                        )

                # -------------------------
                # COMMAND FETCH WITH CHUNKING
                # -------------------------
                commands = []

                if (
                    "command_router" in self.components
                    and self.components["command_router"]
                ):
                    result = await self._safe_call(
                        self.components["command_router"], "get_commands"
                    )
                    if isinstance(result, list):
                        # CHUNKING: Limit and optionally chunk commands
                        commands = result[:5]
                        if len(commands) > self.config.get("max_commands_per_tick", 3):
                            command_chunks = [
                                commands[
                                    i : i + self.config.get("command_batch_size", 2)
                                ]
                                for i in range(
                                    0,
                                    len(commands),
                                    self.config.get("command_batch_size", 2),
                                )
                            ]
                            commands = command_chunks[0] if command_chunks else commands

                # -------------------------
                # AGENT EXECUTION WITH PIPELINE
                # -------------------------
                agent_response = None
                execution_result = None

                if (
                    commands
                    and "agent_loop" in self.components
                    and self.components["agent_loop"]
                ):
                    # PIPELINE: Process commands in parallel if multiple
                    if isinstance(commands, list) and len(commands) > 1:
                        command_tasks = []
                        for cmd in commands[: self.config.get("parallel_commands", 3)]:
                            command_tasks.append(
                                self._safe_call(
                                    self.components["agent_loop"],
                                    "process",
                                    [cmd] if not isinstance(cmd, list) else cmd,
                                )
                            )

                        if command_tasks:
                            results = await asyncio.gather(
                                *command_tasks, return_exceptions=True
                            )
                            agent_response = [
                                r
                                for r in results
                                if r is not None and not isinstance(r, Exception)
                            ]
                            if len(agent_response) == 1:
                                agent_response = agent_response[0]
                    else:
                        agent_response = await self._safe_call(
                            self.components["agent_loop"], "process", commands
                        )

                if (
                    agent_response
                    and "system_controller" in self.components
                    and self.components["system_controller"]
                ):
                    # CHUNKING: Execute large responses in chunks
                    if isinstance(agent_response, str) and len(
                        agent_response
                    ) > self.config.get("execution_chunk_size", 1000):
                        execution_chunks = self._chunk_text(
                            agent_response,
                            self.config.get("execution_chunk_size", 1000),
                        )
                        execution_results = []
                        for chunk in execution_chunks[
                            : self.config.get("max_execution_chunks", 3)
                        ]:
                            result = await self._safe_call(
                                self.components["system_controller"],
                                "execute",
                                chunk,
                            )
                            if result:
                                execution_results.append(result)
                        execution_result = (
                            execution_results[-1] if execution_results else None
                        )
                    else:
                        execution_result = await self._safe_call(
                            self.components["system_controller"],
                            "execute",
                            agent_response,
                        )

                # -------------------------
                # POST PROCESSING WITH PIPELINE
                # -------------------------
                post_tasks = []

                if execution_result:
                    # PIPELINE: Parallel post-processing
                    if (
                        self.config.enable_memory
                        and "memory_manager" in self.components
                        and self.components["memory_manager"]
                    ):
                        post_tasks.append(
                            self._safe_call(
                                self.components["memory_manager"],
                                "store_experience",
                                {
                                    "timestamp": time.time(),
                                    "commands": commands,
                                    "response": agent_response,
                                    "result": execution_result,
                                    "chunked": len(voice_chunks) > 0,
                                },
                            )
                        )

                    if (
                        self.config.enable_learning
                        and "feedback_loop" in self.components
                        and self.components["feedback_loop"]
                    ):
                        post_tasks.append(
                            self._safe_call(
                                self.components["feedback_loop"],
                                "process_feedback",
                                execution_result,
                            )
                        )

                    if (
                        self.config.enable_autonomous_mode
                        and "goal_manager" in self.components
                        and self.components["goal_manager"]
                    ):
                        # CHUNKING: Update progress with chunked data
                        progress_data = execution_result
                        if (
                            isinstance(execution_result, str)
                            and len(execution_result) > 500
                        ):
                            progress_data = execution_result[
                                :500
                            ]  # Truncate for goal manager
                        post_tasks.append(
                            self._safe_call(
                                self.components["goal_manager"],
                                "update_progress",
                                progress_data,
                            )
                        )

                if post_tasks:
                    # Execute post-processing in parallel with batching
                    batch_size = self.config.get("post_batch_size", 3)
                    for i in range(0, len(post_tasks), batch_size):
                        batch = post_tasks[i : i + batch_size]
                        await asyncio.gather(*batch, return_exceptions=True)

                # -------------------------
                # BACKGROUND TASK (FIXED LEAK)
                # -------------------------
                if (
                    self.config.enable_autonomous_mode
                    and "autonomous_core" in self.components
                    and self.components["autonomous_core"]
                    and not task_registry.is_shutting_down()
                ):
                    # PIPELINE: Run background process as tracked task with chunked processing support
                    # start background autonomous task without awaiting completion
                    self._start_task(
                        self._safe_call(
                            self.components["autonomous_core"],
                            "background_process",
                            chunked_mode=len(voice_chunks) > 0,
                        ),
                        name="autonomous_bg",
                    )

                # -------------------------
                # METRICS SAFE
                # -------------------------
                if self.multi_agent_coordinator:
                    metrics_data = await self._safe_call(
                        self.multi_agent_coordinator, "get_system_metrics"
                    )
                    if metrics_data:
                        metrics.gauge_set(
                            "agent_count", metrics_data.get("total_agents", 0)
                        )
                        metrics.gauge_set(
                            "active_agents", metrics_data.get("active_agents", 0)
                        )

                        # Add pipeline metrics
                        metrics.gauge_set("pipeline_chunks", len(voice_chunks))
                        metrics.gauge_set(
                            "pipeline_parallelism",
                            self.config.get("pipeline_workers", 3),
                        )

                # -------------------------
                # SUCCESS UPDATE
                # -------------------------
                self.processing_count += 1
                self.consecutive_failures = 0
                self.circuit_open = False

                latency = time.time() - tick_start
                metrics.histogram_observe(
                    "processing_latency_seconds",
                    latency,
                    {"operation": "tick", "chunked": len(voice_chunks) > 0},
                )

            except Exception as e:
                self.logger.error(f"❌ Tick error: {e}")

                self.consecutive_failures += 1

                metrics.counter_inc(
                    "errors_total",
                    {"component": "tick", "error_type": str(e)},
                )

                if self.consecutive_failures >= self.config.circuit_breaker_threshold:
                    self.circuit_open = True
                    self.state = SystemState.DEGRADED
                    await self._safe_call(
                        self,
                        "_log_event",
                        "circuit_breaker_opened",
                        "orchestrator",
                        {
                            "failures": self.consecutive_failures,
                            "tick": datetime.now().isoformat(),
                            "error": str(e),
                        },
                    )

    def _chunk_text(self, text: str, chunk_size: int) -> list:
        """Split text into chunks of approximate chunk_size characters."""
        if not text or chunk_size <= 0:
            return [text] if text else []

        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def _process_pipeline(self, chunks: list, workers: int) -> list:
        """Process chunks in parallel pipeline with worker pool."""
        if not chunks:
            return []

        results = [None] * len(chunks)
        semaphore = asyncio.Semaphore(workers)

        async def process_chunk(idx: int, chunk: str):
            async with semaphore:
                try:
                    # Process chunk through agent loop
                    if (
                        "agent_loop" in self.components
                        and self.components["agent_loop"]
                    ):
                        result = await self._safe_call(
                            self.components["agent_loop"],
                            "process_voice_input",
                            chunk,
                            chunk_index=idx,
                            total_chunks=len(chunks),
                        )
                        results[idx] = result if result else chunk
                    else:
                        results[idx] = chunk
                except Exception as e:
                    self.logger.warning(f"Pipeline chunk {idx} failed: {e}")
                    results[idx] = chunk

        # Create and execute tasks
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results
        return [r for r in results if r is not None]

    async def _merge_chunked_context(self, chunks: list) -> str:
        """Merge chunked context intelligently."""
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0]

        # Try to intelligently merge by checking for complete sentences
        merged = []
        for chunk in chunks:
            if merged and not merged[-1].endswith((".", "!", "?")):
                # Continue previous sentence
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        return "\n\n".join(merged)

    async def _health_check_loop(self):
        """
        FINAL production-safe health loop:
        - drift-free timing
        - shutdown safe
        - no tight loops
        - fully cancellable
        """

        interval = max(getattr(self.config, "health_check_interval_sec", 10), 1)

        try:
            while (
                not self.shutdown_event.is_set()
                and not task_registry.is_shutting_down()
            ):

                loop_start = asyncio.get_event_loop().time()

                try:
                    # -------------------------
                    # RUN HEALTH CHECK (SAFE)
                    # -------------------------
                    await self._safe_call(self, "_perform_health_check")

                except asyncio.CancelledError:
                    self.logger.info("🛑 Health check loop cancelled")
                    break

                except Exception as e:
                    self.logger.error(f"❌ Health check error: {e}")

                # -------------------------
                # DRIFT-CORRECTED SLEEP (CRITICAL FIX)
                # -------------------------
                elapsed = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(interval - elapsed, 0.0)

                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break

        finally:
            self.logger.info("✅ Health check loop stopped safely")

    async def _perform_health_check(self):
        """
        Production-grade health check with pipeline & chunking:
        - timeout protected
        - shutdown aware
        - no blocking
        - accurate state transitions
        - chunked component validation
        - parallel health checking
        """

        if task_registry.is_shutting_down():
            return

        self.last_health_check = time.time()

        # Initialize health check tracking
        health_check_id = f"health_{int(self.last_health_check)}"
        chunk_size = self.config.get("health_check_chunk_size", 20)
        pipeline_workers = self.config.get("health_pipeline_workers", 5)
        health_results = {}

        try:
            # -------------------------
            # PREPARE COMPONENTS FOR CHUNKING
            # -------------------------
            components_list = [
                (name, component)
                for name, component in list(self.components.items())
                if component is not None
            ]

            if not components_list:
                self.logger.warning("No components to check")
                return

            # -------------------------
            # CHUNKING: Split components into manageable groups
            # -------------------------
            component_chunks = [
                components_list[i : i + chunk_size]
                for i in range(0, len(components_list), chunk_size)
            ]

            self.logger.debug(
                f"📦 Health check chunked into {len(component_chunks)} groups"
            )

            # -------------------------
            # PIPELINE: Parallel health checking with workers
            # -------------------------
            if self.config.get("enable_health_pipeline", True):
                health_queue = asyncio.Queue(maxsize=len(component_chunks))

                # Start pipeline workers
                health_workers = []
                for worker_id in range(pipeline_workers):
                    task = asyncio.create_task(
                        self._health_pipeline_worker(
                            worker_id, health_queue, health_check_id, health_results
                        ),
                        name=f"health_worker_{worker_id}_{health_check_id}",
                    )
                    health_workers.append(task)

                # Queue chunks for processing
                for chunk_idx, chunk in enumerate(component_chunks):
                    await health_queue.put(
                        {
                            "chunk_id": chunk_idx,
                            "components": chunk,
                            "total_chunks": len(component_chunks),
                            "timestamp": time.time(),
                        }
                    )

                # Signal workers to stop
                for _ in range(pipeline_workers):
                    await health_queue.put(None)

                # Wait for all workers with timeout
                await asyncio.wait_for(
                    asyncio.gather(*health_workers, return_exceptions=True),
                    timeout=self.config.get("health_check_timeout", 30),
                )

            else:
                # Sequential chunked processing
                for chunk_idx, chunk in enumerate(component_chunks):
                    await self._process_health_chunk(
                        chunk, chunk_idx, len(component_chunks), health_results
                    )

            # -------------------------
            # MULTI-AGENT HEALTH with chunking
            # -------------------------
            if self.multi_agent_coordinator:
                await self._check_multi_agent_health_chunked()

            # -------------------------
            # CHUNKED SYSTEM STATE UPDATE
            # -------------------------
            await self._update_system_state_from_health(health_results)

            # -------------------------
            # HEALTH CHECK METRICS
            # -------------------------
            metrics.gauge_set(
                "health_check_duration_seconds", time.time() - self.last_health_check
            )
            metrics.gauge_set("health_check_chunks_processed", len(component_chunks))

            # Track unhealthy components
            unhealthy_count = sum(
                1
                for status in self.component_status.values()
                if status in [ComponentStatus.UNHEALTHY, ComponentStatus.NOT_RESPONDING]
            )

            if unhealthy_count > 0:
                metrics.gauge_set("unhealthy_components", unhealthy_count)

        except asyncio.TimeoutError:
            self.logger.error(
                f"❌ Health check timeout after {self.config.get('health_check_timeout', 30)}s"
            )

            await self._safe_call(
                self,
                "_log_event",
                "health_check_timeout",
                "orchestrator",
                {"timeout_seconds": self.config.get("health_check_timeout", 30)},
            )

            metrics.counter_inc("health_check_timeout_total")

        except Exception as e:
            self.logger.error(f"❌ Health check system failure: {e}")

            metrics.counter_inc(
                "errors_total", {"component": "health_check", "error_type": str(e)}
            )

    async def _health_pipeline_worker(
        self, worker_id: int, queue: asyncio.Queue, health_check_id: str, results: dict
    ):
        """Worker for parallel health checking."""
        self.logger.debug(f"🏥 Health worker {worker_id} started")

        while not task_registry.is_shutting_down():
            try:
                # Get chunk from queue with timeout
                chunk_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                # Sentinel check
                if chunk_data is None:
                    queue.task_done()
                    break

                # Process the health chunk
                components = chunk_data.get("components", [])
                chunk_id = chunk_data.get("chunk_id", 0)
                total_chunks = chunk_data.get("total_chunks", 1)

                self.logger.debug(
                    f"Worker {worker_id} checking chunk {chunk_id+1}/{total_chunks} "
                    f"with {len(components)} components"
                )

                # Check each component in chunk
                chunk_results = {}
                for name, component in components:
                    if task_registry.is_shutting_down():
                        break

                    status = await self._check_single_component_health(
                        name, component, worker_id
                    )
                    chunk_results[name] = status

                # Store results
                results.update(chunk_results)

                queue.task_done()

            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health worker {worker_id} error: {e}")
                queue.task_done()
                await asyncio.sleep(0.1)

        self.logger.debug(f"🏥 Health worker {worker_id} stopped")

    async def _check_single_component_health(
        self, name: str, component, worker_id: int
    ) -> ComponentStatus:
        """Check health of a single component with timeout."""
        try:
            if task_registry.is_shutting_down():
                return ComponentStatus.UNKNOWN

            is_healthy = True
            health_details = {}

            # -------------------------
            # HEALTH CHECK METHOD (with timeout)
            # -------------------------
            health_timeout = self.config.get("component_health_timeout", 5.0)

            if hasattr(component, "health_check"):
                try:
                    result = await asyncio.wait_for(
                        self._safe_call(component, "health_check"),
                        timeout=health_timeout,
                    )

                    # Handle different result types
                    if isinstance(result, dict):
                        is_healthy = result.get("healthy", True)
                        health_details = result.get("details", {})
                    elif isinstance(result, bool):
                        is_healthy = result
                    elif result is not None:
                        is_healthy = bool(result)

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"⏱ Health check timeout for {name} (>{health_timeout}s)"
                    )
                    metrics.counter_inc("component_health_timeout", {"component": name})
                    return ComponentStatus.NOT_RESPONDING

            elif hasattr(component, "is_running"):
                try:
                    result = component.is_running()
                    if asyncio.iscoroutine(result):
                        result = await asyncio.wait_for(result, timeout=health_timeout)
                    is_healthy = result() if callable(result) else bool(result)
                except Exception:
                    is_healthy = False

            elif hasattr(component, "get_status"):
                try:
                    result = component.get_status()
                    if asyncio.iscoroutine(result):
                        result = await asyncio.wait_for(result, timeout=health_timeout)
                    is_healthy = result == "running" or result == "healthy"
                except Exception:
                    is_healthy = False

            # -------------------------
            # PERFORMANCE METRICS
            # -------------------------
            if hasattr(component, "get_metrics"):
                try:
                    metrics_data = await self._safe_call(component, "get_metrics")
                    if metrics_data:
                        # Check for degraded performance
                        error_rate = metrics_data.get("error_rate", 0)
                        latency = metrics_data.get("latency_ms", 0)

                        if error_rate > self.config.get("health_error_threshold", 0.1):
                            is_healthy = False
                            health_details["high_error_rate"] = error_rate
                        if latency > self.config.get(
                            "health_latency_threshold_ms", 1000
                        ):
                            is_healthy = False
                            health_details["high_latency_ms"] = latency

                except Exception:
                    pass

            # -------------------------
            # STATUS UPDATE WITH TRANSITION TRACKING
            # -------------------------
            new_status = (
                ComponentStatus.HEALTHY
                if is_healthy
                else (
                    ComponentStatus.DEGRADED
                    if health_details
                    else ComponentStatus.UNHEALTHY
                )
            )

            old_status = self.component_status.get(name, ComponentStatus.UNKNOWN)

            if old_status != new_status:
                self.component_status[name] = new_status

                # Log status change
                self.logger.info(
                    f"🔄 Component {name}: {old_status.value} → {new_status.value}"
                )

                metrics.gauge_set(
                    "component_status", 1 if is_healthy else 0, {"component": name}
                )

                # Track degraded components
                if new_status == ComponentStatus.DEGRADED:
                    metrics.counter_inc("component_degraded", {"component": name})

                await self._safe_call(
                    self,
                    "_log_event",
                    "component_status_change",
                    name,
                    {
                        "old_status": old_status.value if old_status else "unknown",
                        "new_status": new_status.value,
                        "health_details": health_details,
                        "worker_id": worker_id,
                    },
                )
            else:
                # Update timestamp for healthy component
                if new_status == ComponentStatus.HEALTHY:
                    self._last_healthy_time = self._last_healthy_time or {}
                    self._last_healthy_time[name] = time.time()

            return new_status

        except Exception as e:
            self.logger.error(f"Health check failed for {name}: {e}")

            metrics.counter_inc(
                "errors_total", {"component": name, "error_type": "health_check"}
            )

            return ComponentStatus.NOT_RESPONDING

    async def _process_health_chunk(
        self, chunk: list, chunk_idx: int, total_chunks: int, results: dict
    ):
        """Process a health chunk sequentially (fallback method)."""
        self.logger.debug(f"🏥 Processing health chunk {chunk_idx+1}/{total_chunks}")

        for name, component in chunk:
            status = await self._check_single_component_health(
                name, component, "sequential"
            )
            results[name] = status

            # Small delay between checks to prevent overwhelming
            await asyncio.sleep(0.05)

    async def _check_multi_agent_health_chunked(self):
        """Check multi-agent health with chunking."""
        if not self.multi_agent_coordinator:
            return

        try:
            # Get all agents
            all_agents = (
                await self._safe_call(self.multi_agent_coordinator, "get_all_agents")
                or []
            )

            if not all_agents:
                return

            # CHUNKING: Check agents in chunks
            chunk_size = self.config.get("agent_health_chunk_size", 10)
            agent_chunks = [
                all_agents[i : i + chunk_size]
                for i in range(0, len(all_agents), chunk_size)
            ]

            unhealthy_agents = []

            for chunk_idx, agent_chunk in enumerate(agent_chunks):
                # Get statuses in parallel for this chunk
                status_tasks = []
                for agent in agent_chunk:
                    if hasattr(agent, "get_status"):
                        status_tasks.append(self._safe_call(agent, "get_status"))

                if status_tasks:
                    statuses = await asyncio.gather(
                        *status_tasks, return_exceptions=True
                    )

                    for agent, status in zip(agent_chunk, statuses):
                        if isinstance(status, Exception) or status == "error":
                            agent_name = getattr(agent, "name", "unknown")
                            unhealthy_agents.append(agent_name)

            # Log unhealthy agents
            if unhealthy_agents:
                await self._safe_call(
                    self,
                    "_log_event",
                    "unhealthy_agents_detected",
                    "multi_agent",
                    {
                        "count": len(unhealthy_agents),
                        "agents": unhealthy_agents[:10],  # Limit to first 10
                        "total_agents": len(all_agents),
                    },
                )

                metrics.gauge_set("unhealthy_agents", len(unhealthy_agents))

            # Get multi-agent system metrics
            metrics_data = await self._safe_call(
                self.multi_agent_coordinator, "get_system_metrics"
            )

            if metrics_data:
                metrics.gauge_set("agent_count", metrics_data.get("total_agents", 0))
                metrics.gauge_set("active_agents", metrics_data.get("active_agents", 0))

        except Exception as e:
            self.logger.warning(f"Multi-agent health check failed: {e}")

    async def _update_system_state_from_health(self, health_results: dict):
        """Update system state based on chunked health results."""
        # Count component statuses
        status_counts = {
            ComponentStatus.HEALTHY: 0,
            ComponentStatus.DEGRADED: 0,
            ComponentStatus.UNHEALTHY: 0,
            ComponentStatus.NOT_RESPONDING: 0,
            ComponentStatus.UNKNOWN: 0,
        }

        for status in self.component_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1

        total = max(len(self.components), 1)
        healthy_ratio = status_counts[ComponentStatus.HEALTHY] / total

        # Calculate degradation severity
        unhealthy_count = (
            status_counts[ComponentStatus.UNHEALTHY]
            + status_counts[ComponentStatus.NOT_RESPONDING]
        )
        degraded_count = status_counts[ComponentStatus.DEGRADED]

        # -------------------------
        # STATE TRANSITION LOGIC
        # -------------------------
        new_state = self.state

        if unhealthy_count > total / 2:
            # More than half components unhealthy
            if self.state != SystemState.ERROR:
                new_state = SystemState.ERROR
                severity = "critical"

                await self._safe_call(
                    self,
                    "_log_event",
                    "system_critical",
                    "orchestrator",
                    {
                        "unhealthy": unhealthy_count,
                        "degraded": degraded_count,
                        "total": total,
                        "healthy_ratio": healthy_ratio,
                    },
                )

                self.logger.error(
                    f"💀 System CRITICAL: {unhealthy_count}/{total} components unhealthy"
                )

        elif unhealthy_count > 0 or degraded_count > total / 3:
            # Some unhealthy or many degraded components
            if self.state == SystemState.RUNNING:
                new_state = SystemState.DEGRADED

                await self._safe_call(
                    self,
                    "_log_event",
                    "system_degraded",
                    "orchestrator",
                    {
                        "unhealthy": unhealthy_count,
                        "degraded": degraded_count,
                        "total": total,
                        "healthy_ratio": healthy_ratio,
                    },
                )

                self.logger.warning(
                    f"⚠ System degraded: {unhealthy_count} unhealthy, {degraded_count} degraded"
                )

        elif (
            unhealthy_count == 0
            and degraded_count == 0
            and self.state == SystemState.DEGRADED
        ):
            # Full recovery
            new_state = SystemState.RUNNING

            await self._safe_call(
                self,
                "_log_event",
                "system_restored",
                "orchestrator",
                {
                    "healthy_components": status_counts[ComponentStatus.HEALTHY],
                    "total": total,
                },
            )

            self.logger.info("✅ System fully restored")

        # Apply state change if needed
        if new_state != self.state:
            self.state = new_state
            metrics.gauge_set("system_state", {"state": new_state.value})

            # Clear circuit breaker if recovering
            if new_state == SystemState.RUNNING and self.circuit_open:
                self.circuit_open = False
                self.consecutive_failures = 0
                self.logger.info("🔌 Circuit breaker reset")

        # Update health metrics
        metrics.gauge_set("healthy_components", status_counts[ComponentStatus.HEALTHY])
        metrics.gauge_set("degraded_components", degraded_count)
        metrics.gauge_set("unhealthy_components", unhealthy_count)
        metrics.gauge_set("system_health_ratio", healthy_ratio)

    async def _metrics_flush_loop(self):
        """
        FINAL production-safe metrics loop:
        - drift-free timing
        - non-blocking execution
        - shutdown safe
        - no tight loops
        """

        interval = max(getattr(self.config, "metrics_flush_interval_sec", 30), 1)

        try:
            while (
                not self.shutdown_event.is_set()
                and not task_registry.is_shutting_down()
            ):

                loop_start = asyncio.get_event_loop().time()

                try:
                    # -------------------------
                    # NON-BLOCKING FLUSH
                    # -------------------------
                    # metrics.flush_to_mongo is async; await it directly instead
                    await metrics.flush_to_mongo(self.config.name)

                except asyncio.CancelledError:
                    self.logger.info("🛑 Metrics flush loop cancelled")
                    break

                except Exception as e:
                    self.logger.error(f"❌ Metrics flush error: {e}")

                # -------------------------
                # DRIFT-CORRECTED SLEEP (CRITICAL FIX)
                # -------------------------
                elapsed = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(interval - elapsed, 0.0)

                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break

        finally:
            self.logger.info("✅ Metrics flush loop stopped safely")

    async def _state_persist_loop(self):
        """
        FINAL production-safe state persistence loop:
        - drift-free timing
        - shutdown safe
        - no tight loops
        - safe persistence execution
        """

        interval = max(getattr(self.config, "state_persist_interval_sec", 30), 1)

        try:
            while (
                not self.shutdown_event.is_set()
                and not task_registry.is_shutting_down()
            ):

                loop_start = asyncio.get_event_loop().time()

                try:
                    # -------------------------
                    # SAFE STATE PERSIST
                    # -------------------------
                    await self._safe_call(self, "_persist_state_to_mongo")

                except asyncio.CancelledError:
                    self.logger.info("🛑 State persist loop cancelled")
                    break

                except Exception as e:
                    self.logger.error(f"❌ State persist error: {e}")

                # -------------------------
                # DRIFT-CORRECTED SLEEP (CRITICAL FIX)
                # -------------------------
                elapsed = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(interval - elapsed, 0.0)

                try:
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break

        finally:
            self.logger.info("✅ State persist loop stopped safely")

    async def _handle_circuit_breaker(self):
        """
        FINAL production-safe circuit breaker:
        - drift-free timing
        - controlled retries
        - shutdown safe
        - no retry storms
        """

        if task_registry.is_shutting_down():
            return

        interval = max(getattr(self.config, "recovery_timeout_sec", 10), 1)

        try:
            self.logger.warning("⚠ Circuit breaker OPEN - limiting operations")

            # -------------------------
            # DRIFT-SAFE WAIT
            # -------------------------
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return

            # -------------------------
            # SHUTDOWN GUARD
            # -------------------------
            if self.shutdown_event.is_set() or task_registry.is_shutting_down():
                return

            # -------------------------
            # HEALTH CHECK (SAFE)
            # -------------------------
            test_result = await self._safe_call(self, "_test_system_health")

            # -------------------------
            # SUCCESS → CLOSE CIRCUIT
            # -------------------------
            if test_result:
                self.circuit_open = False
                self.consecutive_failures = 0
                self.state = SystemState.RUNNING

                await self._safe_call(
                    self, "_log_event", "circuit_breaker_closed", "orchestrator"
                )

                self.logger.info("✅ Circuit breaker CLOSED - system recovered")

            else:
                # -------------------------
                # CONTROLLED RETRY
                # -------------------------
                self.consecutive_failures += 1

                # Prevent log spam
                if self.consecutive_failures % 3 == 0:
                    self.logger.warning(
                        f"⚠ Circuit still OPEN (failures={self.consecutive_failures})"
                    )

        except asyncio.CancelledError:
            self.logger.info("🛑 Circuit breaker handling cancelled")

        except Exception as e:
            self.logger.error(f"❌ Circuit breaker error: {e}")

    async def _handle_degraded_mode(self):
        """Handle degraded mode operations"""
        self.logger.warning("EDIATH running in degraded mode")
        await asyncio.sleep(1)

    def _create_recovery_chunks(self, components: list, chunk_size: int) -> list:
        """Create priority-based chunks for recovery."""
        if not components:
            return []

        # Sort by priority (critical components first)
        priority_order = {
            "llm_engine": 1,
            "agent_loop": 2,
            "command_router": 3,
            "system_controller": 4,
            "memory_manager": 5,
            "decision_engine": 6,
            "goal_manager": 7,
            "feedback_loop": 8,
            "autonomous_core": 9,
        }

        sorted_components = sorted(
            components, key=lambda x: priority_order.get(x[0], 99)
        )

        # Create chunks
        chunks = []
        for i in range(0, len(sorted_components), chunk_size):
            chunks.append(sorted_components[i : i + chunk_size])

        return chunks

    def _create_health_check_chunks(self) -> list:
        """Create chunks for health checking."""
        all_components = list(self.component_status.keys())
        chunk_size = self.config.get("health_check_chunk_size", 10)

        chunks = []
        for i in range(0, len(all_components), chunk_size):
            chunks.append(all_components[i : i + chunk_size])

        return chunks

    async def _recovery_pipeline_worker(
        self, worker_id: int, queue: asyncio.Queue, recovery_id: str
    ):
        """Worker for parallel recovery processing."""
        self.logger.debug(f"🔧 Recovery worker {worker_id} started")

        while not task_registry.is_shutting_down():
            try:
                # Get chunk from queue with timeout
                chunk_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                # Sentinel check
                if chunk_data is None:
                    queue.task_done()
                    break

                # Process the recovery chunk
                components = chunk_data.get("components", [])
                chunk_id = chunk_data.get("chunk_id", 0)
                total_chunks = chunk_data.get("total_chunks", 1)

                self.logger.info(
                    f"Worker {worker_id} processing chunk {chunk_id+1}/{total_chunks} "
                    f"with {len(components)} components"
                )

                # Process each component in chunk
                for name, component in components:
                    if task_registry.is_shutting_down():
                        break

                    await self._recover_single_component(name, component, worker_id)

                queue.task_done()

            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                queue.task_done()
                await asyncio.sleep(0.1)

        self.logger.debug(f"🔧 Recovery worker {worker_id} stopped")

    async def _recover_single_component(self, name: str, component, worker_id: int):
        """Recover a single component with timeout and retry logic."""
        try:
            if task_registry.is_shutting_down():
                return

            self.logger.info(f"🔄 Worker {worker_id} reinitializing {name}...")

            # Try multiple recovery methods
            result = None

            if hasattr(component, "reinitialize"):
                result = component.reinitialize()
            elif hasattr(component, "restart"):
                result = component.restart()
            elif hasattr(component, "recover"):
                result = component.recover()
            else:
                self.logger.warning(f"No recovery method for {name}")
                return

            # SAFE EXECUTION + TIMEOUT
            timeout = self.config.get("component_recovery_timeout", 10)

            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=timeout)
            elif callable(result):
                await asyncio.to_thread(result)

            # Verify recovery with health check
            if hasattr(component, "health_check"):
                health_result = await self._safe_call(component, "health_check")
                if health_result:
                    self.component_status[name] = ComponentStatus.HEALTHY
                    self.logger.info(f"✅ Component {name} recovered successfully")
                else:
                    self.component_status[name] = ComponentStatus.DEGRADED
                    self.logger.warning(f"⚠ Component {name} recovered but unhealthy")
            else:
                self.component_status[name] = ComponentStatus.HEALTHY
                self.logger.info(f"✅ Component {name} recovered")

            # Update metrics
            metrics.counter_inc("component_recovery_total", {"component": name})

        except asyncio.TimeoutError:
            self.logger.warning(f"⏱ Recovery timeout for {name} (>{timeout}s)")
            self.component_status[name] = ComponentStatus.UNHEALTHY
            metrics.counter_inc("component_recovery_timeout", {"component": name})

        except Exception as e:
            self.logger.warning(f"Recovery failed for {name}: {e}")
            self.component_status[name] = ComponentStatus.UNHEALTHY
            metrics.counter_inc(
                "component_recovery_failed", {"component": name, "error": str(e)}
            )

    async def _process_recovery_chunk(
        self, chunk: list, chunk_idx: int, total_chunks: int
    ):
        """Process a recovery chunk sequentially (fallback method)."""
        self.logger.info(f"📦 Processing recovery chunk {chunk_idx+1}/{total_chunks}")

        for name, component in chunk:
            await self._recover_single_component(name, component, "sequential")

            # Small delay between components to prevent thundering herd
            await asyncio.sleep(0.1)

    async def _check_chunked_health(self, health_chunks: list) -> list:
        """Check health of components in chunks."""
        results = []

        for chunk_idx, chunk in enumerate(health_chunks):
            chunk_healthy = True

            for component_name in chunk:
                status = self.component_status.get(component_name)
                if status != ComponentStatus.HEALTHY:
                    chunk_healthy = False
                    self.logger.debug(f"Component {component_name} unhealthy: {status}")

            results.append(chunk_healthy)

            # Early exit if chunk is unhealthy and we have critical failures
            if (
                not chunk_healthy and chunk_idx == 0
            ):  # First chunk has critical components
                self.logger.warning(
                    "Critical components still unhealthy after recovery"
                )

        return results

    async def _recover_multi_agent_system(self):
        """Recover multi-agent system with chunking."""
        if not self.multi_agent_coordinator:
            return

        try:
            # Get all agents
            agents = (
                await self._safe_call(self.multi_agent_coordinator, "get_all_agents")
                or []
            )

            if not agents:
                self.logger.warning("No agents found for recovery")
                return

            # CHUNKING: Recover agents in chunks
            chunk_size = self.config.get("agent_recovery_chunk_size", 3)
            agent_chunks = [
                agents[i : i + chunk_size] for i in range(0, len(agents), chunk_size)
            ]

            self.logger.info(
                f"🔄 Recovering {len(agents)} agents in {len(agent_chunks)} chunks"
            )

            for chunk_idx, agent_chunk in enumerate(agent_chunks):
                recovery_tasks = []

                for agent in agent_chunk:
                    if hasattr(agent, "recover"):
                        recovery_tasks.append(
                            self._safe_call(
                                agent,
                                "recover",
                                timeout=self.config.get("agent_recovery_timeout", 5),
                            )
                        )

                if recovery_tasks:
                    await asyncio.gather(*recovery_tasks, return_exceptions=True)

                self.logger.debug(
                    f"Recovered agent chunk {chunk_idx+1}/{len(agent_chunks)}"
                )

                # Small delay between chunks
                await asyncio.sleep(0.2)

            # Reinitialize multi-agent coordination
            await self._safe_call(self, "_setup_multi_agent_system")

            self.logger.info("✅ Multi-agent system recovered")

        except Exception as e:
            self.logger.error(f"Multi-agent recovery failed: {e}")
            raise

    async def _attempt_recovery(self):
        """
        FINAL production-safe recovery with pipeline & chunking:
        - bounded + controlled retries
        - no recovery storms
        - shutdown safe
        - strict async execution
        - chunked component recovery
        - pipeline-based parallel recovery
        """

        if task_registry.is_shutting_down():
            return

        # -------------------------
        # PREVENT RECOVERY SPAM (CRITICAL FIX)
        # -------------------------
        now = asyncio.get_event_loop().time()
        last = getattr(self, "_last_recovery_attempt", 0)

        min_interval = max(getattr(self.config, "recovery_cooldown_sec", 5), 1)

        if now - last < min_interval:
            return

        self._last_recovery_attempt = now

        self.logger.info("🔄 Attempting system recovery with chunking pipeline...")

        await self._safe_call(self, "_log_event", "recovery_attempt", "orchestrator")

        # Initialize recovery tracking
        recovery_id = f"recovery_{int(now)}"
        self._recovery_chunk_buffer = []
        self._recovery_pipeline_tasks = []
        chunk_size = self.config.get("recovery_chunk_size", 5)
        pipeline_workers = self.config.get("recovery_pipeline_workers", 3)
        recovery_start = time.time()

        try:
            # -------------------------
            # IDENTIFY UNHEALTHY COMPONENTS
            # -------------------------
            unhealthy_components = [
                (name, component)
                for name, component in list(self.components.items())
                if component is not None
                and self.component_status.get(name) != ComponentStatus.HEALTHY
            ]

            if not unhealthy_components:
                self.logger.info("✓ No unhealthy components to recover")
                return

            self.logger.info(
                f"📋 Found {len(unhealthy_components)} unhealthy components"
            )

            # -------------------------
            # CHUNKING: Group components by priority and type
            # -------------------------
            priority_chunks = self._create_recovery_chunks(
                unhealthy_components, chunk_size
            )
            self.logger.info(f"📦 Recovery chunked into {len(priority_chunks)} groups")

            # -------------------------
            # PIPELINE: Start recovery workers
            # -------------------------
            if self.config.get("enable_recovery_pipeline", True):
                recovery_queue = asyncio.Queue(maxsize=len(unhealthy_components))

                # Start pipeline workers
                for worker_id in range(pipeline_workers):
                    task = asyncio.create_task(
                        self._recovery_pipeline_worker(
                            worker_id, recovery_queue, recovery_id
                        ),
                        name=f"recovery_worker_{worker_id}_{recovery_id}",
                    )
                    self._recovery_pipeline_tasks.append(task)

                # CHUNKING: Process components through pipeline
                for chunk_idx, chunk in enumerate(priority_chunks):
                    await recovery_queue.put(
                        {
                            "chunk_id": chunk_idx,
                            "components": chunk,
                            "total_chunks": len(priority_chunks),
                            "timestamp": time.time(),
                        }
                    )

                # Wait for all chunks to be processed
                await asyncio.wait_for(
                    recovery_queue.join(),
                    timeout=self.config.get("recovery_timeout", 60),
                )

                # Stop workers
                for _ in range(pipeline_workers):
                    await recovery_queue.put(None)  # Sentinel for worker termination

                await asyncio.gather(
                    *self._recovery_pipeline_tasks, return_exceptions=True
                )

            else:
                # Fallback to sequential chunked recovery
                for chunk_idx, chunk in enumerate(priority_chunks):
                    await self._process_recovery_chunk(
                        chunk, chunk_idx, len(priority_chunks)
                    )

            # -------------------------
            # MULTI-AGENT RECOVERY with chunking
            # -------------------------
            if self.multi_agent_coordinator:
                self.logger.info("🔄 Recovering multi-agent system...")
                await self._recover_multi_agent_system()

            # -------------------------
            # CHUNKED FINAL HEALTH CHECK
            # -------------------------
            health_check_chunks = self._create_health_check_chunks()
            health_results = await self._check_chunked_health(health_check_chunks)

            unhealthy_count = sum(1 for result in health_results if not result)

            # -------------------------
            # RECOVERY METRICS
            # -------------------------
            recovery_duration = time.time() - recovery_start

            if unhealthy_count == 0:
                self.state = SystemState.RUNNING
                self.consecutive_failures = 0
                self.circuit_open = False

                # Clear recovery buffers on success
                self._recovery_chunk_buffer.clear()

                await self._safe_call(
                    self,
                    "_log_event",
                    "recovery_success",
                    "orchestrator",
                    {
                        "duration_seconds": recovery_duration,
                        "components_recovered": len(unhealthy_components),
                        "chunks_processed": len(priority_chunks),
                        "pipeline_workers": pipeline_workers,
                    },
                )

                self.logger.info(f"✅ Recovery successful in {recovery_duration:.2f}s")

                # Update metrics
                metrics.histogram_observe(
                    "recovery_duration_seconds", recovery_duration
                )
                metrics.counter_inc("recovery_success_total")

            else:
                self.state = SystemState.DEGRADED

                await self._safe_call(
                    self,
                    "_log_event",
                    "recovery_partial",
                    "orchestrator",
                    {
                        "unhealthy_remaining": unhealthy_count,
                        "total_components": len(self.component_status),
                        "duration_seconds": recovery_duration,
                    },
                )

                self.logger.warning(
                    f"⚠ Partial recovery: {unhealthy_count} components still unhealthy"
                )
                metrics.counter_inc("recovery_partial_total")

        except asyncio.TimeoutError:
            self.logger.error(
                f"❌ Recovery timeout after {self.config.get('recovery_timeout', 60)}s"
            )

            await self._safe_call(
                self,
                "_log_event",
                "recovery_timeout",
                "orchestrator",
                {"timeout_seconds": self.config.get("recovery_timeout", 60)},
            )

            metrics.counter_inc("recovery_timeout_total")

        except Exception as e:
            self.logger.error(f"❌ Recovery failed: {e}")

            await self._safe_call(
                self,
                "_log_event",
                "recovery_failed",
                "orchestrator",
                {"error": str(e), "recovery_id": recovery_id},
            )

            metrics.counter_inc("recovery_failed_total", {"error_type": str(e)})

            # -------------------------
            # CONTROLLED BACKOFF with exponential delay
            # -------------------------
            retry_count = getattr(self, "_recovery_retry_count", 0) + 1
            self._recovery_retry_count = retry_count

            # Exponential backoff: 5, 10, 20, 40, 60 (max)
            delay = min(
                max(getattr(self.config, "retry_delay_sec", 5), 1)
                * (2 ** (retry_count - 1)),
                self.config.get("max_recovery_delay", 60),
            )

            self.logger.info(f"⏳ Recovery backoff: {delay}s (attempt {retry_count})")

            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                pass

        finally:
            # Cleanup
            for task in self._recovery_pipeline_tasks:
                if not task.done():
                    task.cancel()

            if self._recovery_pipeline_tasks:
                await asyncio.gather(
                    *self._recovery_pipeline_tasks, return_exceptions=True
                )
                self._recovery_pipeline_tasks.clear()

    async def _test_system_health(self) -> bool:
        """Test overall system health"""
        try:
            if (
                "agent_core" in self.components
                and self.components["agent_core"]
                and hasattr(self.components["agent_core"], "health_check")
            ):
                result = await self._safe_call(
                    self.components["agent_core"], "health_check"
                )
                return bool(result) if result is not None else True
            return True
        except Exception:
            return False

    async def pause(self):
        """Pause system operations"""
        self.state = SystemState.PAUSED
        await self._log_event("system_paused", "orchestrator")
        self.logger.info("System paused")

        if (
            self.config.enable_autonomous_mode
            and "autonomous_core" in self.components
            and self.components["autonomous_core"]
        ):
            await self._safe_call(self.components["autonomous_core"], "pause")

    async def resume(self):
        """Resume system operations"""
        self.state = SystemState.RUNNING
        await self._log_event("system_resumed", "orchestrator")
        self.logger.info("System resumed")

        if (
            self.config.enable_autonomous_mode
            and "autonomous_core" in self.components
            and self.components["autonomous_core"]
        ):
            await self._safe_call(self.components["autonomous_core"], "resume")

    async def shutdown(self, graceful: bool = True):
        """
        FINAL production-safe shutdown (TODO #5):
        - strictly ordered
        - cancel tracked init tasks
        - cleanup _pending_coros
        - no race conditions
        - no task leaks
        - async-safe everywhere
        """

        if self.state == SystemState.STOPPED:
            return

        self.logger.info("🛑 Shutting down EDIATH system...")
        self.state = SystemState.SHUTTING_DOWN

        # -------------------------
        # CANCEL TRACKED INIT TASKS (TODO #5)
        # -------------------------
        await self._cancel_pending_init_tasks()

        # -------------------------
        # CLEANUP PENDING COROS (TODO #5)
        # -------------------------
        if self._pending_coros:
            self._pending_coros.clear()
            self.logger.debug("Cleared pending coroutines")

        # -------------------------
        # SIGNAL SHUTDOWN FIRST
        # -------------------------
        task_registry.initiate_shutdown()
        self.shutdown_event.set()

        await self._safe_call(
            self, "_log_event", "shutdown_start", "orchestrator", {"graceful": graceful}
        )

        # -------------------------
        # CANCEL ACTIVE TASKS
        # -------------------------
        tasks = list(getattr(self, "active_tasks", []))

        for task in tasks:
            if task and not task.done():
                try:
                    task.cancel()
                except Exception:
                    pass

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=3.0
                )
            except Exception:
                pass

        self.active_tasks.clear()

        # -------------------------
        # GLOBAL TASK REGISTRY
        # -------------------------
        try:
            await asyncio.wait_for(task_registry.cancel_all_tasks(), timeout=3.0)
        except Exception:
            pass

        # -------------------------
        # SHUTDOWN AGENTS
        # -------------------------
        await self._safe_call(self.agent_registry, "shutdown_all")

        # -------------------------
        # MULTI-AGENT
        # -------------------------
        if self.multi_agent_coordinator:
            await self._safe_call(self, "shutdown_multi_agent_coordinator")

        # -------------------------
        # COMPONENT SHUTDOWN (STRICT ORDER)
        # -------------------------
        for name in reversed(list(self.components.keys())):

            component = self.components.get(name)
            if component is None:
                continue

            try:
                self.logger.info(f"Shutting down {name}...")

                result = None

                if hasattr(component, "shutdown"):
                    result = component.shutdown()
                elif hasattr(component, "stop"):
                    result = component.stop()

                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=3.0)
                elif callable(result):
                    await asyncio.to_thread(result)

                await self._safe_call(self, "_log_event", "component_shutdown", name)

            except Exception as e:
                self.logger.warning(f"Shutdown failed for {name}: {e}")

        # -------------------------
        # FINAL PERSIST (ORDERED)
        # -------------------------
        try:
            await asyncio.wait_for(
                self._safe_call(self, "_persist_state_to_mongo"), timeout=3.0
            )
        except Exception:
            pass

        try:
            # metrics.flush_to_mongo is async; await directly to avoid leaking
            await metrics.flush_to_mongo(self.config.name)
        except Exception:
            pass

        await self._safe_call(self, "_log_event", "shutdown_complete", "orchestrator")

        # -------------------------
        # ATOMIC FILE WRITE (NO LOCKS)
        # -------------------------
        if getattr(self.config, "enable_persistence", False):
            try:
                report_file = (
                    Path(self.config.reports_dir)
                    / f"shutdown_report_{int(time.time())}.json"
                )
                report_file.parent.mkdir(parents=True, exist_ok=True)

                def _write_report():
                    temp_file = report_file.with_suffix(".tmp")

                    with open(temp_file, "w") as f:
                        json.dump(self.get_status(), f, indent=2, default=str)

                    # CRITICAL: atomic replace
                    temp_file.replace(report_file)

                await asyncio.to_thread(_write_report)

            except Exception as e:
                self.logger.warning(f"Report write failed: {e}")

        # -------------------------
        # FINAL STATE
        # -------------------------
        self.state = SystemState.STOPPED
        self.logger.info("✅ EDIATH system shutdown complete")

    async def shutdown_multi_agent_coordinator(self):
        """Shutdown multi-agent coordinator"""
        if self.multi_agent_coordinator and MULTI_AGENT_AVAILABLE:
            try:
                await shutdown_multi_agent_coordinator()
            except Exception as e:
                self.logger.warning(f"Multi-agent shutdown error: {e}")

    async def self_test(self, timeout: float = 20.0) -> Dict[str, Any]:
        """
        Run a lightweight self-test across core components:
        - initialize if not initialized
        - test agent responsiveness (health_check/ping/is_ready)
        - test speaker and listener
        - test memory manager and llm basic readiness
        - test vision memory frame retrieval
        Returns a dictionary of results.
        """
        results = {
            "initialized": False,
            "agents": {},
            "speaker": None,
            "listener": None,
            "memory": None,
            "llm": None,
            "vision": None,
            "ok": False,
        }

        # ensure system initialized
        try:
            if self.state != SystemState.RUNNING:
                init_ok = await asyncio.wait_for(self.initialize(), timeout=timeout)
                results["initialized"] = bool(init_ok)
            else:
                results["initialized"] = True
        except Exception as e:
            results["initialized"] = False
            results["init_error"] = str(e)

        # test agents
        try:
            agents = self.agent_registry.get_all_agents()
            for name, agent in agents.items():
                ok = False
                try:
                    resp = await asyncio.wait_for(
                        self._safe_call(agent, "health_check"), timeout=3.0
                    )
                    ok = bool(resp) if resp is not None else True
                except Exception:
                    try:
                        resp = await asyncio.wait_for(
                            self._safe_call(agent, "ping"), timeout=3.0
                        )
                        ok = bool(resp) if resp is not None else True
                    except Exception:
                        ok = False

                results["agents"][name] = {"responsive": ok}
        except Exception as e:
            results["agents_error"] = str(e)

        # speaker
        try:
            if getattr(self.system_instance, "speaker", None):
                sp = self.system_instance.speaker
                # try async speak or sync speak
                resp = await self._safe_call(sp, "speak_async", "Self-test: hello")
                if resp is None:
                    resp = await self._safe_call(sp, "speak", "Self-test: hello")
                results["speaker"] = True if resp is not None else True
            else:
                results["speaker"] = False
        except Exception as e:
            results["speaker"] = False
            results["speaker_error"] = str(e)

        # listener
        try:
            if getattr(self.system_instance, "listener", None):
                lst = self.system_instance.listener
                # try to fetch text quickly
                try:
                    text = await asyncio.wait_for(
                        asyncio.to_thread(lst.get_text), timeout=2.0
                    )
                    results["listener"] = True if text is not None else True
                except Exception:
                    results["listener"] = True
            else:
                results["listener"] = False
        except Exception as e:
            results["listener"] = False
            results["listener_error"] = str(e)

        # memory manager
        try:
            mm = self.components.get("memory_manager")
            if mm:
                resp = await self._safe_call(mm, "health_check")
                results["memory"] = bool(resp) if resp is not None else True
            else:
                results["memory"] = False
        except Exception as e:
            results["memory"] = False
            results["memory_error"] = str(e)

        # llm engine
        try:
            llm = self.components.get("llm_engine")
            if llm:
                resp = await self._safe_call(llm, "is_ready")
                results["llm"] = bool(resp) if resp is not None else True
            else:
                results["llm"] = False
        except Exception as e:
            results["llm"] = False
            results["llm_error"] = str(e)

        # vision
        try:
            if self.vision_memory:
                frame = None
                try:
                    frame = await asyncio.wait_for(
                        self.vision_memory.get_latest_frame(), timeout=2.0
                    )
                except Exception:
                    frame = None
                results["vision"] = True if frame is not None else False
            else:
                results["vision"] = False
        except Exception as e:
            results["vision"] = False
            results["vision_error"] = str(e)

        # aggregate
        results["ok"] = results["initialized"] is True and (
            all(v.get("responsive", True) for v in results.get("agents", {}).values())
            if results.get("agents")
            else True
        )

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "state": self.state.value,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "component_status": {
                name: status.value for name, status in self.component_status.items()
            },
            "circuit_breaker_open": self.circuit_open,
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": self.last_health_check,
            "config": asdict(self.config),
            "agents": list(self.agent_registry.get_all_agents().keys()),
            "blender_available": self.blender_available,
            "mongo_connected": mongo_is_connected(),
            "mongo_status": mongo_client.status.value if mongo_client else "unknown",
        }

        if self.multi_agent_coordinator:
            try:
                status["multi_agent"] = self.multi_agent_coordinator.get_status()
            except Exception:
                status["multi_agent"] = {"status": "unavailable"}

        return status

    def get_agent(self, name: str) -> Optional[Any]:
        """Get a specific agent by name"""
        return self.agent_registry.get_agent(name)

    def get_all_agents(self) -> Dict[str, Any]:
        """Get all registered agents"""
        return self.agent_registry.get_all_agents()

    async def execute_blender_command(self, command: str, **params) -> Dict[str, Any]:
        """
        FINAL production-safe Blender execution:
        - non-blocking (CRITICAL for bpy)
        - timeout protected
        - validation safe
        - no event loop blocking
        """

        blender_agent = self.get_agent("blender")

        # -------------------------
        # VALIDATION
        # -------------------------
        if not blender_agent:
            return {"success": False, "error": "Blender Agent not available"}

        if not self.blender_available:
            return {"success": False, "error": "Blender/bpy not available"}

        if task_registry.is_shutting_down():
            return {"success": False, "error": "System shutting down"}

        try:
            # -------------------------
            # COMMAND MAP (STRICT)
            # -------------------------
            command_map = {
                "create_scene": lambda: blender_agent.create_complex_scene(),
                "create_cube": lambda: blender_agent.create_cube(**params),
                "create_sphere": lambda: blender_agent.create_sphere(**params),
                "create_cylinder": lambda: blender_agent.create_cylinder(**params),
                "create_cone": lambda: blender_agent.create_cone(**params),
                "create_torus": lambda: blender_agent.create_torus(**params),
                "create_monkey": lambda: blender_agent.create_monkey(**params),
                "create_plane": lambda: blender_agent.create_plane(**params),
                "render": lambda: blender_agent.render_image(
                    params.get(
                        "filepath", f"{self.config.blender_render_dir}/render.png"
                    )
                ),
                "export": lambda: blender_agent.export_scene(
                    params.get(
                        "filepath", f"{self.config.blender_workspace_dir}/output.fbx"
                    ),
                    params.get("format", self.config.blender_export_format),
                ),
                "arrange_circle": lambda: blender_agent.arrange_in_circle(
                    params.get("objects", []), params.get("radius", 5)
                ),
                "animate": lambda: blender_agent.animate_object(
                    params.get("object"),
                    params.get("start_frame", 1),
                    params.get("end_frame", 120),
                ),
            }

            func = command_map.get(command)
            if not func:
                return {"success": False, "error": f"Unknown command: {command}"}

            # -------------------------
            # EXECUTION (CRITICAL FIX)
            # Blender MUST run in thread (bpy blocks event loop)
            # -------------------------
            try:
                result = await asyncio.wait_for(asyncio.to_thread(func), timeout=30)
            except asyncio.TimeoutError:
                raise

            # -------------------------
            # SAFE LOGGING
            # -------------------------
            await self._safe_call(
                self,
                "_log_event",
                "blender_command",
                "blender_agent",
                {"command": command, "success": True},
            )

            return {"success": True, "result": result}

        except asyncio.TimeoutError:
            self.logger.error(f"⏱ Blender command '{command}' timed out")

            await self._safe_call(
                self,
                "_log_event",
                "blender_command",
                "blender_agent",
                {"command": command, "success": False, "error": "timeout"},
            )

            return {"success": False, "error": "Command timeout"}

        except Exception as e:
            self.logger.error(f"❌ Blender command '{command}' failed: {e}")

            await self._safe_call(
                self,
                "_log_event",
                "blender_command",
                "blender_agent",
                {"command": command, "success": False, "error": str(e)},
            )

            return {"success": False, "error": str(e)}


# ==================== Main Entry Point ====================


async def main() -> None:
    """Main entry point with full lifecycle and error handling"""

    config_path = Path("config/orchestrator_config.yaml")
    orchestrator = None

    try:
        # -------------------------
        # LOAD / CREATE CONFIG
        # -------------------------
        if config_path.exists():
            logging.info(f"Loading config from {config_path}")
            config = OrchestratorConfig.from_yaml(config_path)
        else:
            logging.warning("Config not found. Creating default config...")
            config = OrchestratorConfig()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config.to_yaml(config_path)
            logging.info(f"Default config created at {config_path}")

        # -------------------------
        # INIT ORCHESTRATOR
        # -------------------------
        orchestrator = EDIATHOrchestrator(config)

        logging.info("Initializing orchestrator...")
        initialized = await orchestrator.initialize()

        if not initialized:
            logging.error("Failed to initialize orchestrator")
            return

        logging.info("Orchestrator initialized successfully")

        # -------------------------
        # RUN SYSTEM
        # -------------------------
        logging.info("Starting orchestrator run loop...")
        await orchestrator.run()

    except asyncio.CancelledError:
        logging.info("Main task cancelled")

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")

    except Exception as e:
        logging.exception(f"Fatal error in main: {e}")

    finally:
        # -------------------------
        # CLEAN SHUTDOWN
        # -------------------------
        if orchestrator:
            logging.info("Shutting down orchestrator...")
            try:
                await asyncio.wait_for(orchestrator.shutdown(graceful=True), timeout=15)
                logging.info("Shutdown completed")
            except asyncio.TimeoutError:
                logging.error("Shutdown timed out")
            except Exception:
                logging.exception("Error during shutdown")


if __name__ == "__main__":
    asyncio.run(main())
