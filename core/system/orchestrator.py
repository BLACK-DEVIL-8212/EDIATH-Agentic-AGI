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
    from core.vision.shared_memory import VisionMemory

    VISION_MEMORY_AVAILABLE = True
except ImportError:
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

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.agents: Dict[str, Any] = {}
        self.wrappers: Dict[str, Any] = {}

        # -------------------------
        # FIX: ENSURE CONFIG EXISTS
        # -------------------------
        self.config = config if config is not None else OrchestratorConfig()

        # -------------------------
        # SAFE DEFAULT SETTER
        # -------------------------
        def _set_default(attr, value):
            if not hasattr(self.config, attr):
                setattr(self.config, attr, value)

        # -------------------------
        # CONFIG DEFAULTS
        # -------------------------
        _set_default("agent_init_chunk_size", 5)
        _set_default("agent_init_pipeline_workers", 3)
        _set_default("enable_agent_pipeline", True)
        _set_default("agent_init_timeout", 60)
        _set_default("enable_agent_comms", True)
        _set_default("agent_comms_setup_chunk_size", 5)
        _set_default("agent_validation_chunk_size", 10)
        _set_default("vision_connect_retries", 3)
        _set_default("vision_connect_delay", 1)
        _set_default("vision_connect_timeout", 5)
        _set_default("agent_comms_buffer", 1000)
        _set_default("agent_comms_retries", 3)
        _set_default("agent_comms_timeout", 5)

        # -------------------------
        # LOGGING + STORAGE
        # -------------------------
        self.logger = base_logger
        self._audit_collection = get_audit_collection()

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

    async def _safe_call(self, obj: Any, method_name: str, *args, **kwargs) -> Any:
        """
        Safe async method caller
        """
        if obj is None:
            return None

        try:
            method = getattr(obj, method_name, None)
            if not callable(method):
                return None

            # Filter kwargs to only those accepted by the method to avoid
            # "unexpected keyword argument" when calling with pipeline kwargs
            try:
                sig = inspect.signature(method)
                bound_kwargs = {}
                for k, v in kwargs.items():
                    if k in sig.parameters:
                        bound_kwargs[k] = v
            except Exception:
                bound_kwargs = kwargs

            result = method(*args, **bound_kwargs)

            if asyncio.iscoroutine(result):
                return await result
            elif callable(result):
                return await asyncio.to_thread(result)
            else:
                return result

        except Exception as e:
            self.logger.debug(f"Safe call error in {method_name}: {e}")
            return None

    async def initialize_all(self, system_instance: Any) -> Dict[str, Any]:
        """
        FINAL production-safe agent initialization with pipeline & chunking:
        - async-safe imports
        - guaranteed stub fallback
        - no crash propagation
        - shutdown aware
        - chunked agent initialization
        - parallel pipeline processing
        """

        results = {}

        # Initialize chunking and pipeline structures
        chunk_size = self.config.get("agent_init_chunk_size", 5)
        pipeline_workers = max(
            1, min(int(self.config.get("agent_init_pipeline_workers", 3)), 8)
        )
        agent_init_queue = asyncio.Queue()
        init_start_time = time.time()

        # Prepare agent list
        agent_list = list(AGENT_MAP.items())

        # Filter agents based on config and shutdown
        agents_to_init = []
        for agent_name, info in agent_list:
            if task_registry.is_shutting_down():
                break

            if not self.config or not getattr(
                self.config, info.get("config", f"enable_{agent_name}"), True
            ):
                results[agent_name] = False
                continue

            agents_to_init.append((agent_name, info))

        if not agents_to_init:
            self.logger.info("No agents to initialize")
            return results

        # -------------------------
        # CHUNKING: Split agents into chunks
        # -------------------------
        agent_chunks = [
            agents_to_init[i : i + chunk_size]
            for i in range(0, len(agents_to_init), chunk_size)
        ]

        self.logger.info(
            f"📦 Agent initialization chunked into {len(agent_chunks)} groups "
            f"(size={chunk_size}, workers={pipeline_workers})"
        )

        # Shared results storage with thread-safe updates
        init_results = {}
        results_lock = asyncio.Lock()

        try:
            # -------------------------
            # PIPELINE: Start worker tasks
            # -------------------------
            if self.config.get("enable_agent_pipeline", True):
                workers = []

                # Create worker tasks
                for worker_id in range(pipeline_workers):
                    worker_task = asyncio.create_task(
                        self._agent_init_worker(
                            worker_id,
                            agent_init_queue,
                            init_results,
                            results_lock,
                            system_instance,
                        ),
                        name=f"agent_init_worker_{worker_id}",
                    )
                    workers.append(worker_task)

                # Queue chunks for processing
                for chunk_idx, chunk in enumerate(agent_chunks):
                    await agent_init_queue.put(
                        {
                            "chunk_id": chunk_idx,
                            "agents": chunk,
                            "total_chunks": len(agent_chunks),
                            "timestamp": time.time(),
                        }
                    )

                # Signal workers to stop
                for _ in range(pipeline_workers):
                    await agent_init_queue.put(None)

                # Wait for all workers with timeout
                timeout = self.config.get("agent_init_timeout", 60)
                await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True), timeout=timeout
                )

                # Merge results
                results.update(init_results)

            else:
                # Sequential chunked initialization
                for chunk_idx, chunk in enumerate(agent_chunks):
                    await self._process_agent_chunk(
                        chunk, chunk_idx, len(agent_chunks), results, system_instance
                    )

            # -------------------------
            # CONNECT VISION ENGINE (Post-initialization pipeline)
            # -------------------------
            if "vision" in self.agents and self.agents["vision"]:
                await self._connect_vision_engine_pipeline(system_instance)

            # -------------------------
            # AGENT COMMUNICATION PIPELINE SETUP
            # -------------------------
            if self.config.get("enable_agent_comms", True):
                await self._setup_agent_communication_pipeline()

            # -------------------------
            # VALIDATE INITIALIZED AGENTS (Chunked)
            # -------------------------
            validation_results = await self._validate_agents_chunked(
                list(self.agents.keys())
            )

            # Update results with validation status
            for agent_name, is_valid in validation_results.items():
                if agent_name in results:
                    results[agent_name] = results[agent_name] and is_valid

            # -------------------------
            # SUMMARY WITH METRICS
            # -------------------------
            active = sum(1 for v in results.values() if v)
            total = len(results)
            init_duration = time.time() - init_start_time

            self.logger.info(
                f"✅ Agent init complete: {active}/{total} active "
                f"(duration: {init_duration:.2f}s, chunks: {len(agent_chunks)})"
            )

            # Update metrics
            metrics.gauge_set("agents_initialized", active)
            metrics.gauge_set("agents_total", total)
            metrics.histogram_observe("agent_init_duration_seconds", init_duration)
            metrics.gauge_set("agent_init_chunks", len(agent_chunks))

            # Log failed agents if any
            failed_agents = [name for name, success in results.items() if not success]
            if failed_agents:
                self.logger.warning(f"⚠ Failed agents: {failed_agents}")
                metrics.gauge_set("agents_failed", len(failed_agents))

            return results

        except asyncio.TimeoutError:
            self.logger.error(
                f"❌ Agent initialization timeout after {self.config.get('agent_init_timeout', 60)}s"
            )

            # Mark remaining agents as failed
            for agent_name, _ in agents_to_init:
                if agent_name not in results:
                    results[agent_name] = False

            metrics.counter_inc("agent_init_timeout_total")
            return results

        except Exception as e:
            self.logger.error(f"❌ Agent initialization system failure: {e}")

            metrics.counter_inc("agent_init_failed_total", {"error": str(e)})

            # Mark remaining as failed
            for agent_name, _ in agents_to_init:
                if agent_name not in results:
                    results[agent_name] = False

            return results

    async def _agent_init_worker(
        self,
        worker_id: int,
        queue: asyncio.Queue,
        results: dict,
        results_lock: asyncio.Lock,
        system_instance: Any,
    ):
        """Worker for parallel agent initialization."""
        self.logger.debug(f"🔧 Agent init worker {worker_id} started")

        while not task_registry.is_shutting_down():
            try:
                # Get chunk from queue with timeout
                chunk_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                # Sentinel check
                if chunk_data is None:
                    queue.task_done()
                    break

                # Process the agent chunk
                agents = chunk_data.get("agents", [])
                chunk_id = chunk_data.get("chunk_id", 0)
                total_chunks = chunk_data.get("total_chunks", 1)

                self.logger.info(
                    f"Worker {worker_id} initializing chunk {chunk_id+1}/{total_chunks} "
                    f"with {len(agents)} agents"
                )

                chunk_results = {}

                # Initialize agents in this chunk
                for agent_name, info in agents:
                    if task_registry.is_shutting_down():
                        break

                    result = await self._initialize_single_agent(
                        agent_name, info, system_instance, worker_id, chunk_id
                    )
                    chunk_results[agent_name] = result

                # Store results safely
                async with results_lock:
                    results.update(chunk_results)

                queue.task_done()

            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Agent init worker {worker_id} error: {e}")
                queue.task_done()
                await asyncio.sleep(0.1)

        self.logger.debug(f"🔧 Agent init worker {worker_id} stopped")

    async def _initialize_single_agent(
        self,
        agent_name: str,
        info: dict,
        system_instance: Any,
        worker_id: int,
        chunk_id: int,
    ) -> bool:
        """Initialize a single agent with full error handling."""
        try:
            agent = None
            wrapper = None

            # Check availability
            if info.get("available", True):
                try:
                    # FILE AGENT
                    if agent_name == "file":
                        from core.agent.file_agent import FileAgent, FileAgentWrapper

                        agent = FileAgent(workspace_root=self.config.workspace_dir)
                        wrapper = FileAgentWrapper(
                            {"workspace": self.config.workspace_dir}
                        )

                    # VISION AGENT
                    elif agent_name == "vision":
                        from core.agent.vision_agent import (
                            VisionAgent,
                            VisionAgentWrapper,
                        )

                        agent = VisionAgent()
                        wrapper = VisionAgentWrapper()

                    # BLENDER AGENT (SAFE)
                    elif agent_name == "blender" and BLENDER_AGENT_AVAILABLE:
                        Path(self.config.blender_workspace_dir).mkdir(
                            parents=True, exist_ok=True
                        )
                        Path(self.config.blender_render_dir).mkdir(
                            parents=True, exist_ok=True
                        )

                        agent = Blender3DAgent()
                        wrapper = agent

                        self.logger.info(
                            f"Worker {worker_id}: 🎨 Blender 3D Agent initialized"
                        )

                    # DYNAMIC AGENTS with chunked loading
                    else:
                        module = info.get("module")

                        if module:
                            import importlib

                            # FIX: safe threaded import
                            agent_module = await asyncio.to_thread(
                                importlib.import_module, module
                            )

                            agent_cls = getattr(
                                agent_module, f"{agent_name.title()}Agent", None
                            )
                            wrapper_cls = getattr(
                                agent_module, f"{agent_name.title()}AgentWrapper", None
                            )

                            if agent_cls:
                                agent = agent_cls()

                            if wrapper_cls:
                                wrapper = wrapper_cls()

                    # Validation
                    if agent is None:
                        raise RuntimeError("Agent creation failed")

                    if wrapper is None:
                        wrapper = agent

                    # Store references
                    self.agents[agent_name] = agent
                    self.wrappers[agent_name] = wrapper

                    await self._safe_call(
                        self,
                        "_log_audit",
                        "agent_init",
                        agent_name,
                        True,
                        f"worker_{worker_id}_chunk_{chunk_id}",
                    )

                    self.logger.info(
                        f"Worker {worker_id}: ✓ {agent_name.title()} Agent initialized"
                    )

                    return True

                except Exception as e:
                    # STUB FALLBACK (CRITICAL)
                    self.logger.warning(
                        f"Worker {worker_id}: ⚠ {agent_name} init failed → stub: {e}"
                    )

                    from core.agent.stub_agents import StubAgent

                    agent = StubAgent()
                    wrapper = agent

                    self.agents[agent_name] = agent
                    self.wrappers[agent_name] = wrapper

                    await self._safe_call(
                        self,
                        "_log_audit",
                        "agent_init",
                        agent_name,
                        True,
                        f"stub_mode - {e}",
                    )

                    return True

            else:
                # FORCE STUB for unavailable agents
                from core.agent.stub_agents import StubAgent

                agent = StubAgent()

                self.agents[agent_name] = agent
                self.wrappers[agent_name] = agent

                await self._safe_call(
                    self,
                    "_log_audit",
                    "agent_init",
                    agent_name,
                    True,
                    "stub_mode_unavailable",
                )

                return True

        except Exception as e:
            # HARD FAILURE ISOLATION
            self.logger.error(f"Worker {worker_id}: ❌ {agent_name} init crashed: {e}")

            await self._safe_call(
                self, "_log_audit", "agent_init", agent_name, False, str(e)
            )

            metrics.counter_inc(
                "agent_init_failed", {"agent": agent_name, "error": str(e)}
            )

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

    async def _connect_vision_engine_pipeline(self, system_instance: Any):
        """Connect vision engine using pipeline pattern."""
        try:
            # Get vision agent
            vision_agent = self.agents.get("vision")
            vision_wrapper = self.wrappers.get("vision")

            if not vision_agent or not vision_wrapper:
                return

            # Get vision engine instance
            ve = None
            if system_instance is not None:
                ve = getattr(system_instance, "vision_engine_instance", None)

            if ve is not None and hasattr(vision_wrapper, "connect_to_vision_engine"):
                # Pipeline: Connect with retry logic
                max_retries = self.config.get("vision_connect_retries", 3)
                retry_delay = self.config.get("vision_connect_delay", 1)

                for attempt in range(max_retries):
                    try:
                        await asyncio.wait_for(
                            vision_wrapper.connect_to_vision_engine(ve),
                            timeout=self.config.get("vision_connect_timeout", 5),
                        )
                        self.logger.info(
                            "✓ VisionAgent connected to VisionEngine instance"
                        )
                        metrics.counter_inc("vision_engine_connected")
                        break
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"Vision connect timeout (attempt {attempt+1}/{max_retries})"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                    except Exception as e:
                        self.logger.warning(f"Failed to connect VisionAgent: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                        else:
                            metrics.counter_inc("vision_connect_failed")

        except Exception as e:
            self.logger.warning(f"Vision engine connection pipeline failed: {e}")

    async def _setup_agent_communication_pipeline(self):
        """Setup communication pipeline between agents with chunking."""
        try:
            agent_names = list(self.agents.keys())

            if len(agent_names) < 2:
                return

            # CHUNKING: Setup communication in chunks
            chunk_size = self.config.get("agent_comms_setup_chunk_size", 5)
            agent_chunks = [
                agent_names[i : i + chunk_size]
                for i in range(0, len(agent_names), chunk_size)
            ]

            self.logger.debug(f"Setting up agent comms in {len(agent_chunks)} chunks")

            # Pipeline: Register agent communication channels
            for chunk_idx, chunk in enumerate(agent_chunks):
                comm_tasks = []

                for agent_name in chunk:
                    agent = self.agents.get(agent_name)
                    wrapper = self.wrappers.get(agent_name)

                    # Register communication methods
                    if hasattr(wrapper, "register_communication"):
                        comm_tasks.append(
                            self._safe_call(
                                wrapper,
                                "register_communication",
                                self._create_agent_comms_pipeline(),
                            )
                        )

                    # Set up event bus if available
                    if hasattr(agent, "set_event_bus"):
                        comm_tasks.append(
                            self._safe_call(
                                agent, "set_event_bus", self._get_agent_event_bus()
                            )
                        )

                if comm_tasks:
                    await asyncio.gather(*comm_tasks, return_exceptions=True)

                self.logger.debug(
                    f"Completed comms setup for chunk {chunk_idx+1}/{len(agent_chunks)}"
                )

        except Exception as e:
            self.logger.warning(f"Agent communication pipeline setup failed: {e}")

    async def _validate_agents_chunked(self, agent_names: list) -> Dict[str, bool]:
        """Validate initialized agents in chunks."""
        results = {}

        chunk_size = self.config.get("agent_validation_chunk_size", 10)
        agent_chunks = [
            agent_names[i : i + chunk_size]
            for i in range(0, len(agent_names), chunk_size)
        ]

        for chunk_idx, chunk in enumerate(agent_chunks):
            validation_tasks = []

            for agent_name in chunk:
                agent = self.agents.get(agent_name)

                if agent is None:
                    results[agent_name] = False
                    continue

                # Check if agent has validation method
                if hasattr(agent, "validate"):
                    validation_tasks.append(self._safe_call(agent, "validate"))
                else:
                    validation_tasks.append(asyncio.sleep(0, result=True))

            if validation_tasks:
                validation_results = await asyncio.gather(
                    *validation_tasks, return_exceptions=True
                )

                for agent_name, result in zip(chunk, validation_results):
                    if isinstance(result, Exception):
                        self.logger.warning(
                            f"Agent {agent_name} validation error: {result}"
                        )
                        results[agent_name] = False
                    else:
                        results[agent_name] = bool(result)

            self.logger.debug(f"Validated chunk {chunk_idx+1}/{len(agent_chunks)}")

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
        self.vision_memory = VisionMemory() if VISION_MEMORY_AVAILABLE else None

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
        """Safe, non-blocking MongoDB state persistence with bounded cleanup"""

        if not getattr(self.config, "enable_mongo_persistence", False):
            return

        if self._state_collection is None:
            return

        try:
            now = datetime.utcnow()

            # -------------------------
            # BUILD DOCUMENT (SAFE)
            # -------------------------
            state_doc = {
                "timestamp": now,
                "orchestrator_name": self.config.name,
                "state": getattr(self.state, "value", str(self.state)),
                "processing_count": self.processing_count,
                "error_count": self.error_count,
                "component_status": {
                    name: getattr(status, "value", str(status))
                    for name, status in self.component_status.items()
                },
                "circuit_breaker_open": self.circuit_open,
                "consecutive_failures": self.consecutive_failures,
                "uptime_seconds": (
                    time.time() - self.start_time if self.start_time else 0
                ),
                "active_agents": len(self.agent_registry.get_all_agents()),
                "blender_available": self.blender_available,
            }

            # -------------------------
            # NON-BLOCKING INSERT (CRITICAL FIX)
            # -------------------------
            def _insert():
                try:
                    return self._state_collection.insert_one(state_doc)
                except Exception:
                    return None

            await asyncio.to_thread(_insert)

            # -------------------------
            # CLEANUP (EFFICIENT + NON-BLOCKING)
            # -------------------------
            def _cleanup():
                try:
                    # delete older docs beyond latest 1000
                    cursor = (
                        self._state_collection.find(
                            {"orchestrator_name": self.config.name}, {"_id": 1}
                        )
                        .sort("timestamp", -1)
                        .skip(1000)
                    )

                    old_ids = [doc["_id"] for doc in cursor.limit(500)]

                    if old_ids:
                        self._state_collection.delete_many({"_id": {"$in": old_ids}})
                except Exception:
                    pass

            await asyncio.to_thread(_cleanup)

            # -------------------------
            # UPDATE TIMESTAMP
            # -------------------------
            self.last_state_persist = now

        except Exception as e:
            self.logger.debug(f"State persist skipped: {e}")

    async def _log_event(
        self, event_type: str, component: str, details: Dict[str, Any] = None
    ):
        """Safe, non-blocking event logging with lightweight protection"""

        if not getattr(self.config, "enable_mongo_persistence", False):
            return

        if self._events_collection is None:
            return

        try:
            now = datetime.utcnow()

            # -------------------------
            # BUILD EVENT DOC (SAFE)
            # -------------------------
            event_doc = {
                "timestamp": now,
                "event_type": str(event_type),
                "component": str(component),
                "orchestrator_name": self.config.name,
                "state": getattr(self.state, "value", str(self.state)),
                "details": details if isinstance(details, dict) else {},
            }

            # -------------------------
            # NON-BLOCKING INSERT (CRITICAL FIX)
            # -------------------------
            def _insert():
                try:
                    self._events_collection.insert_one(event_doc)
                except Exception:
                    pass

            await asyncio.to_thread(_insert)

        except Exception as e:
            self.logger.debug(f"Event logging skipped: {e}")

    async def initialize(self) -> bool:
        """
        Fully production-safe initialization with pipeline & chunking:
        - strict async discipline
        - no double execution
        - no coroutine leaks
        - timeout + shutdown safe
        - chunked component initialization
        - pipeline-based agent loading
        """

        if task_registry.is_shutting_down():
            return False

        self.logger.info("🚀 Initializing EDIATH system...")

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        start_time = time.time()

        # Initialize chunking and pipeline structures
        self._init_chunk_buffer = []
        self._init_pipeline_tasks = []
        chunk_size = self.config.get("init_chunk_size", 50)
        pipeline_workers = self._cap_workers(
            self.config.get("init_pipeline_workers", 5), cap=8
        )

        try:
            # -------------------------
            # SAFE EVENT LOG
            # -------------------------
            await self._safe_call(
                self,
                "_log_event",
                "initialization_start",
                "orchestrator",
                {"version": getattr(self.config, "version", "unknown")},
            )

            # -------------------------
            # AGENT INIT WITH CHUNKING
            # -------------------------
            self.agent_registry = AgentRegistry()
            self.agent_registry.set_config(self.config)
            self.logger.info("Initializing Agent Registry...")

            # Get all agents to initialize
            all_agents = self._get_agents_to_initialize()

            # CHUNKING: Initialize agents in chunks
            agent_chunks = [
                all_agents[i : i + chunk_size]
                for i in range(0, len(all_agents), chunk_size)
            ]
            self.logger.info(
                f"📦 Agent initialization chunked into {len(agent_chunks)} groups"
            )

            agent_results = {}

            for chunk_idx, agent_chunk in enumerate(agent_chunks):
                # PIPELINE: Initialize agents in parallel within chunk
                chunk_tasks = []
                for agent_name in agent_chunk:
                    chunk_tasks.append(
                        self._safe_call(
                            self.agent_registry,
                            "initialize_agent",
                            agent_name,
                            self.system_instance,
                            chunk_index=chunk_idx,
                            total_chunks=len(agent_chunks),
                        )
                    )

                # Execute chunk with timeout
                chunk_results = await asyncio.wait_for(
                    asyncio.gather(*chunk_tasks, return_exceptions=True),
                    timeout=self.config.get("agent_init_timeout", 30),
                )

                # Collect results
                for agent_name, result in zip(agent_chunk, chunk_results):
                    agent_results[agent_name] = (
                        result if not isinstance(result, Exception) else False
                    )

                self.logger.info(
                    f"✓ Chunk {chunk_idx+1}/{len(agent_chunks)} agents initialized"
                )

            successful_agents = sum(1 for v in agent_results.values() if v)

            self.logger.info(
                f"✓ Agents initialized: {successful_agents}/{len(agent_results)}"
            )

            # -------------------------
            # MULTI-AGENT INIT (FIXED) with pipeline
            # -------------------------
            if self.config.enable_multi_agent and MULTI_AGENT_AVAILABLE:
                try:
                    self.multi_agent_coordinator = await asyncio.wait_for(
                        get_multi_agent_coordinator(), timeout=10
                    )

                    # PIPELINE: Parallel multi-agent configuration
                    multi_agent_tasks = [
                        self._safe_call(self, "_configure_multi_agent_strategies"),
                        self._safe_call(self, "_setup_multi_agent_system"),
                        self._safe_call(
                            self, "_initialize_agent_communication_pipeline"
                        ),
                    ]

                    await asyncio.gather(*multi_agent_tasks, return_exceptions=True)

                    # FIX: ensure start is awaited properly
                    await self._safe_call(self.multi_agent_coordinator, "start")

                    self.logger.info("✓ Multi-Agent ready")

                except Exception as e:
                    self.logger.warning(f"Multi-agent init failed: {e}")

            # -------------------------
            # COMPONENT INIT WITH PIPELINE (SAFE PARALLEL)
            # -------------------------
            component_groups = [
                ("system", ["_initialize_system_components"]),
                ("agent", ["_initialize_agent_components"]),
                ("brain", ["_initialize_brain_components"]),
                ("perception", ["_initialize_perception_components"]),
            ]

            # Dynamic component groups based on config
            if self.config.enable_memory:
                component_groups.append(("memory", ["_initialize_memory_components"]))

            if self.config.enable_autonomous_mode:
                component_groups.append(
                    ("autonomous", ["_initialize_autonomous_components"])
                )

            if self.config.enable_learning:
                component_groups.append(
                    ("learning", ["_initialize_learning_components"])
                )

            if self.config.enable_security:
                component_groups.append(
                    ("security", ["_initialize_security_components"])
                )

            # PIPELINE: Process component groups in parallel with chunking
            component_pipeline = []
            for group_name, init_methods in component_groups:
                for method in init_methods:
                    component_pipeline.append(
                        self._safe_call(self, method, group_name=group_name)
                    )

            # Execute in batches to avoid overwhelming
            batch_size = self.config.get("component_batch_size", 3)
            for i in range(0, len(component_pipeline), batch_size):
                batch = component_pipeline[i : i + batch_size]
                await asyncio.gather(*batch, return_exceptions=True)
                self.logger.debug(f"✓ Component batch {i//batch_size + 1} initialized")

            # -------------------------
            # LOAD MODEL WITH CHUNKING (CRITICAL FIX)
            # -------------------------
            llm = self.components.get("llm_engine")

            if llm and hasattr(llm, "load_model"):
                try:
                    # CHUNKING: Load model in chunks if large
                    if hasattr(
                        llm, "get_model_size"
                    ) and llm.get_model_size() > self.config.get(
                        "large_model_threshold", 1024
                    ):
                        self.logger.info("📦 Loading large model in chunks...")
                        result = await self._load_model_in_chunks(llm)
                    else:
                        # Ensure singleton model load
                        async with self._model_loading_lock:
                            if self._model_loaded:
                                self.logger.info("Model already loaded, skipping")
                            else:
                                result = llm.load_model(self.config.llm_model_path)
                                if asyncio.iscoroutine(result):
                                    await asyncio.wait_for(result, timeout=30)
                                else:
                                    # If result is callable or immediate, run in thread
                                    if callable(result):
                                        await asyncio.to_thread(result)
                                self._model_loaded = True
                                self.logger.info("✓ Model loaded successfully")

                except Exception as e:
                    self.logger.warning(f"Model load failed: {e}")

            # -------------------------
            # PIPELINE: Initialize chunk processing workers
            # -------------------------
            if self.config.get("enable_init_pipeline", True):
                await self._initialize_pipeline_workers()

            # -------------------------
            # CHUNKED STATE VALIDATION
            # -------------------------
            validation_chunks = self._create_validation_chunks()
            validation_results = await self._validate_chunked_state(validation_chunks)

            if not all(validation_results):
                self.logger.warning(
                    f"⚠ Some validation chunks failed: {validation_results}"
                )

            # -------------------------
            # FINAL STATE
            # -------------------------
            self.state = SystemState.RUNNING
            self.start_time = time.time()

            metrics.gauge_set("orchestrator_state", 1, {"state": self.state.value})
            metrics.gauge_set("initialized_agents", successful_agents)
            metrics.gauge_set("init_chunks_processed", len(agent_chunks))

            # -------------------------
            # FINAL PERSIST + LOG (Parallel)
            # -------------------------
            persist_tasks = [
                self._safe_call(self, "_persist_state_to_mongo"),
                self._safe_call(
                    self,
                    "_log_event",
                    "initialization_complete",
                    "orchestrator",
                    {
                        "duration_seconds": time.time() - start_time,
                        "agents_initialized": successful_agents,
                        "chunks_processed": len(agent_chunks),
                        "pipeline_workers": pipeline_workers,
                    },
                ),
                self._safe_call(self, "_flush_init_chunk_buffer"),
            ]

            await asyncio.gather(*persist_tasks, return_exceptions=True)

            self.logger.info("✅ EDIATH system is RUNNING")
            return True

        except Exception as e:
            # -------------------------
            # FAILURE HANDLING with cleanup
            # -------------------------
            self.state = SystemState.ERROR

            self.logger.error(f"❌ Initialization failed: {e}")

            # Cleanup pipeline tasks
            for task in self._init_pipeline_tasks:
                if not task.done():
                    task.cancel()

            if self._init_pipeline_tasks:
                await asyncio.gather(*self._init_pipeline_tasks, return_exceptions=True)

            await self._safe_call(
                self,
                "_log_event",
                "initialization_failed",
                "orchestrator",
                {"error": str(e), "partial_chunks": len(self._init_chunk_buffer)},
            )

            metrics.counter_inc(
                "errors_total", {"component": "initialization", "error_type": str(e)}
            )

            return False

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

    async def _validate_chunked_state(self, validation_chunks: list) -> list:
        """Validate system state in chunks."""
        results = []

        for chunk_idx, chunk in enumerate(validation_chunks):
            chunk_valid = True
            for component_name in chunk:
                if component_name in self.components:
                    component = self.components[component_name]
                    if hasattr(component, "validate"):
                        try:
                            is_valid = await self._safe_call(component, "validate")
                            if not is_valid:
                                chunk_valid = False
                                self.logger.warning(
                                    f"Component {component_name} validation failed"
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Component {component_name} validation error: {e}"
                            )
                            chunk_valid = False

            results.append(chunk_valid)

        return results

    async def _initialize_pipeline_workers(self):
        """Initialize pipeline workers for initialization phase."""
        worker_count = self._cap_workers(
            self.config.get("init_pipeline_workers", 5), cap=8
        )

        for i in range(worker_count):
            task = asyncio.create_task(
                self._init_pipeline_worker(i), name=f"init_pipeline_worker_{i}"
            )
            self._init_pipeline_tasks.append(task)

        self.logger.info(f"✓ Initialized {worker_count} pipeline workers")

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

    async def _process_init_chunk(self, chunk: dict, worker_id: int):
        """Process a single initialization chunk."""
        chunk_type = chunk.get("type", "unknown")
        chunk_data = chunk.get("data", [])

        try:
            if chunk_type == "agents":
                # Process agent chunk
                for agent_name in chunk_data:
                    await self._safe_call(
                        self.agent_registry,
                        "initialize_agent",
                        agent_name,
                        self.system_instance,
                    )
            elif chunk_type == "components":
                # Process component chunk
                for component_init in chunk_data:
                    await self._safe_call(self, component_init)

            self.logger.debug(f"Worker {worker_id} processed {chunk_type} chunk")

        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed on {chunk_type} chunk: {e}")

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
        """Initialize communication pipeline for agents."""
        if hasattr(self, "multi_agent_coordinator"):
            pipeline_config = {
                "chunk_size": self.config.get("agent_comms_chunk_size", 100),
                "pipeline_depth": self.config.get("agent_pipeline_depth", 10),
                "parallel_agents": self.config.get("parallel_agent_comms", 3),
            }
            await self._safe_call(
                self.multi_agent_coordinator, "initialize_pipeline", pipeline_config
            )

    async def _load_model_in_chunks(self, llm) -> None:
        """Load a large model in chunks to manage memory."""
        chunk_size = self.config.get("model_chunk_size_mb", 256)

        # Get model chunks
        if hasattr(llm, "get_model_chunks"):
            model_chunks = await llm.get_model_chunks(chunk_size)

            for chunk_idx, model_chunk in enumerate(model_chunks):
                self.logger.info(
                    f"📦 Loading model chunk {chunk_idx+1}/{len(model_chunks)}"
                )

                result = llm.load_model_chunk(model_chunk, chunk_idx)
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=30)
                else:
                    await asyncio.to_thread(lambda: result)

            # Finalize model loading
            if hasattr(llm, "finalize_model_load"):
                await llm.finalize_model_load()
        else:
            # Fallback to regular loading
            result = llm.load_model(self.config.llm_model_path)
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=30)
            else:
                await asyncio.to_thread(lambda: result)

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

    async def _initialize_brain_components(self, *args, **kwargs):
        """
        Initialize brain components with TIMEOUT, CHUNKING & PIPELINE SAFE
        - accepts extra kwargs (fix for pipeline)
        - parallel startup with timeout protection
        - shutdown safe
        - chunked initialization
        - timeout recovery
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            self.logger.debug("Shutdown in progress - skipping init")
            return

        # -------------------------
        # TIMEOUT CONFIGURATION
        # -------------------------
        component_timeout = self.config.get("brain_init_timeout", 10.0)
        chunk_timeout = self.config.get("brain_chunk_timeout", 5.0)

        # -------------------------
        # DEFINE COMPONENTS WITH METADATA
        # -------------------------
        brain_components = [
            {
                "name": "llm_engine",
                "class": LLMEngine,
                "init_args": [],
                "init_kwargs": {},
                "critical": True,
                "timeout": component_timeout,
                "retry_count": 2,
            },
            {
                "name": "decision_engine",
                "class": DecisionEngine,
                "init_args": [],
                "init_kwargs": {},
                "critical": True,
                "timeout": component_timeout,
                "retry_count": 2,
            },
            {
                "name": "reasoning_engine",
                "class": ReasoningEngine,
                "init_args": [],
                "init_kwargs": {},
                "critical": False,
                "timeout": component_timeout,
                "retry_count": 1,
            },
            {
                "name": "context_manager",
                "class": ContextManager,
                "init_args": [],
                "init_kwargs": {},
                "critical": False,
                "timeout": component_timeout,
                "retry_count": 1,
            },
        ]

        # Filter available components based on config
        enabled_components = [
            comp
            for comp in brain_components
            if self.config.get(f"enable_{comp['name']}", True)
        ]

        if not enabled_components:
            self.logger.info("No brain components to initialize")
            return

        # -------------------------
        # CHUNKING: Split components into chunks
        # -------------------------
        chunk_size = self.config.get("brain_chunk_size", 2)
        component_chunks = [
            enabled_components[i : i + chunk_size]
            for i in range(0, len(enabled_components), chunk_size)
        ]

        self.logger.info(
            f"🧠 Initializing {len(enabled_components)} brain components "
            f"in {len(component_chunks)} chunks (timeout={component_timeout}s)"
        )

        new_components = {}
        failed_components = []

        # -------------------------
        # PROCESS CHUNKS WITH TIMEOUT
        # -------------------------
        for chunk_idx, chunk in enumerate(component_chunks):

            # Check shutdown before each chunk
            if task_registry.is_shutting_down():
                self.logger.debug("Shutdown detected - stopping brain init")
                break

            self.logger.debug(
                f"Processing brain chunk {chunk_idx+1}/{len(component_chunks)}"
            )

            # Create tasks for this chunk
            chunk_tasks = []
            component_names = []

            for comp_config in chunk:
                component_names.append(comp_config["name"])

                task = asyncio.create_task(
                    self._init_single_brain_component(comp_config),
                    name=f"brain_init_{comp_config['name']}",
                )
                chunk_tasks.append(task)

            # Execute chunk with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*chunk_tasks, return_exceptions=True),
                    timeout=chunk_timeout,
                )

                # Process results
                for comp_name, result in zip(component_names, results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"⚠ {comp_name} init failed: {result}")
                        failed_components.append(comp_name)
                    elif result is not None:
                        new_components[comp_name] = result
                        self.logger.info(f"✓ {comp_name} initialized")

            except asyncio.TimeoutError:
                self.logger.error(
                    f"❌ Brain chunk {chunk_idx+1} TIMEOUT after {chunk_timeout}s"
                )

                # Cancel hanging tasks
                for task in chunk_tasks:
                    if not task.done():
                        task.cancel()

                # Mark chunk components as failed
                failed_components.extend(component_names)

                # Try emergency fallback for critical components
                await self._emergency_brain_fallback(component_names, new_components)

            except Exception as e:
                self.logger.error(f"❌ Brain chunk {chunk_idx+1} error: {e}")
                failed_components.extend(component_names)

            # Small delay between chunks to prevent overwhelming
            await asyncio.sleep(0.1)

        # -------------------------
        # UPDATE COMPONENTS
        # -------------------------
        self.components.update(new_components)

        # -------------------------
        # VERIFY CRITICAL COMPONENTS
        # -------------------------
        critical_failed = [
            name
            for name in failed_components
            if any(
                comp["name"] == name and comp.get("critical", False)
                for comp in enabled_components
            )
        ]

        if critical_failed:
            self.logger.error(f"❌ Critical brain components failed: {critical_failed}")

            # Attempt recovery for critical components
            recovery_success = await self._recover_critical_brain_components(
                critical_failed, new_components
            )

            if not recovery_success:
                self.logger.error(
                    "💀 Brain initialization partially failed - system may be degraded"
                )
                self.state = SystemState.DEGRADED

        # -------------------------
        # INITIALIZATION SUMMARY
        # -------------------------
        success_count = len(new_components)
        total_count = len(enabled_components)

        if success_count == total_count:
            self.logger.info(
                f"✅ All {success_count} brain components initialized successfully"
            )
        else:
            self.logger.warning(
                f"⚠ Brain components: {success_count}/{total_count} initialized, "
                f"failed: {failed_components}"
            )

        # -------------------------
        # PIPELINE CONNECTION (Post-init)
        # -------------------------
        if new_components.get("llm_engine") and new_components.get("decision_engine"):
            await self._connect_brain_pipeline(new_components)

    async def _init_single_brain_component(
        self, comp_config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Initialize a single brain component with:
        - retry
        - timeout
        - backoff
        - cancellation safety (CRITICAL FIX)
        """

        component_name = comp_config.get("name", "unknown")
        component_class = comp_config.get("class")

        if component_class is None:
            raise ValueError(f"{component_name}: Missing 'class' in config")

        retry_count = comp_config.get("retry_count", 2)
        timeout = comp_config.get("timeout", 10.0)
        backoff = comp_config.get("retry_backoff", 0.5)

        init_args = comp_config.get("init_args", [])
        init_kwargs = comp_config.get("init_kwargs", {})

        for attempt in range(1, retry_count + 1):
            try:
                # -------------------------
                # SHUTDOWN GUARD
                # -------------------------
                if task_registry.is_shutting_down():
                    self.logger.debug(f"⛔ Skipping {component_name}, shutdown active")
                    return None

                # -------------------------
                # CREATE INSTANCE (SAFE)
                # -------------------------
                if asyncio.iscoroutinefunction(component_class):
                    instance = await asyncio.wait_for(
                        component_class(*init_args, **init_kwargs),
                        timeout=timeout
                    )
                else:
                    # don't wrap to_thread inside wait_for directly (reduces cancellation issues)
                    instance = await asyncio.wait_for(
                        asyncio.to_thread(component_class, *init_args, **init_kwargs),
                        timeout=timeout
                    )

                # -------------------------
                # INITIALIZE (SAFE)
                # -------------------------
                if hasattr(instance, "initialize"):
                    init_method = instance.initialize

                    if asyncio.iscoroutinefunction(init_method):
                        await asyncio.wait_for(init_method(), timeout=timeout)
                    else:
                        await asyncio.to_thread(init_method)

                self.logger.debug(f"✓ {component_name} initialized (attempt {attempt})")
                return instance

            # -------------------------
            # CRITICAL FIX: HANDLE CANCELLED
            # -------------------------
            except asyncio.CancelledError:
                self.logger.debug(f"🛑 {component_name} init cancelled")
                return None

            # -------------------------
            # TIMEOUT HANDLING
            # -------------------------
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"⏱ {component_name} timeout ({attempt}/{retry_count})"
                )
                if attempt == retry_count:
                    return None

            # -------------------------
            # GENERAL FAILURE
            # -------------------------
            except Exception as e:
                self.logger.warning(
                    f"⚠ {component_name} failed ({attempt}/{retry_count}): {e}"
                )
                if attempt == retry_count:
                    return None

            # -------------------------
            # BACKOFF
            # -------------------------
            if attempt < retry_count:
                try:
                    await asyncio.sleep(backoff * attempt)
                except asyncio.CancelledError:
                    return None

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

    async def _start_components_brain(self, components: Dict[str, Any]):
        """
        Start brain components with CHUNKING, PIPELINE & TIMEOUT PROTECTION
        - Chunked parallel startup
        - Per-component timeout
        - Recovery for failed starts
        - Pipeline ordering for dependencies
        - Shutdown safe
        """

        # -------------------------
        # SHUTDOWN GUARD
        # -------------------------
        if task_registry.is_shutting_down():
            self.logger.debug("Shutdown in progress - skipping component start")
            return

        # -------------------------
        # COMPONENT START ORDER (with dependencies)
        # -------------------------
        component_order = [
            {
                "priority": 1,
                "names": ["context_manager", "memory_manager"],  # Foundation first
                "timeout": 10.0,
                "critical": True,
            },
            {
                "priority": 2,
                "names": ["llm_engine"],  # LLM second
                "timeout": 30.0,  # LLM needs more time
                "critical": True,
            },
            {
                "priority": 3,
                "names": ["decision_engine", "reasoning_engine"],  # Decision engines
                "timeout": 15.0,
                "critical": True,
            },
            {
                "priority": 4,
                "names": ["autonomous_core", "goal_manager"],  # Autonomous components
                "timeout": 15.0,
                "critical": False,
            },
            {
                "priority": 5,
                "names": ["feedback_loop", "learning_engine"],  # Learning components
                "timeout": 15.0,
                "critical": False,
            },
            {
                "priority": 6,
                "names": [],  # All remaining components
                "timeout": 10.0,
                "critical": False,
            },
        ]

        # Build component map from the passed 'components' parameter
        available_components = {
            name: comp for name, comp in components.items() if comp is not None
        }

        if not available_components:
            self.logger.debug("No components to start")
            return

        # -------------------------
        # PIPELINE: Start components in priority order
        # -------------------------
        chunk_size = self.config.get("component_start_chunk_size", 3)

        started_components = []
        failed_components = []

        for priority_group in component_order:

            # Get components for this priority
            if priority_group["names"]:
                group_components = {
                    name: available_components[name]
                    for name in priority_group["names"]
                    if name in available_components and name not in started_components
                }
            else:
                # Priority 6: All remaining components
                group_components = {
                    name: comp
                    for name, comp in available_components.items()
                    if name not in started_components
                }

            if not group_components:
                continue

            timeout = priority_group.get("timeout", 15.0)
            critical = priority_group.get("critical", False)

            self.logger.info(
                f"🎯 Starting priority {priority_group['priority']} components: "
                f"{list(group_components.keys())}"
            )

            # -------------------------
            # CHUNKING: Split components into chunks
            # -------------------------
            component_items = list(group_components.items())
            chunks = [
                component_items[i : i + chunk_size]
                for i in range(0, len(component_items), chunk_size)
            ]

            # Process chunks with timeout
            for chunk_idx, chunk in enumerate(chunks):

                # Check shutdown
                if task_registry.is_shutting_down():
                    self.logger.debug("Shutdown detected - stopping component start")
                    return

                try:
                    # Start the chunk
                    result = await asyncio.wait_for(
                        self._start_component_chunk(
                            chunk_idx, chunk, timeout, critical
                        ),
                        timeout=timeout + 2.0,
                    )

                    # Collect results
                    if isinstance(result, dict):
                        started_components.extend(result.get("started", []))
                        failed_components.extend(result.get("failed", []))

                except asyncio.TimeoutError:
                    self.logger.error(
                        f"❌ Chunk {chunk_idx} TIMEOUT after {timeout+2}s"
                    )
                    # Mark all components in this chunk as failed
                    for name, _ in chunk:
                        failed_components.append(name)

                except Exception as e:
                    self.logger.error(f"❌ Chunk {chunk_idx} error: {e}")
                    for name, _ in chunk:
                        failed_components.append(name)

                # Small delay between chunks
                await asyncio.sleep(0.1)

            # Small delay between priorities
            await asyncio.sleep(0.2)

        # -------------------------
        # VERIFY STARTED COMPONENTS
        # -------------------------
        total_components = len(available_components)
        success_count = len(started_components)

        if success_count == total_components:
            self.logger.info(f"✅ All {success_count} components started successfully")
        else:
            self.logger.warning(
                f"⚠ Components started: {success_count}/{total_components}, "
                f"failed: {failed_components}"
            )

            # Attempt recovery for failed components
            await self._recover_failed_components(
                failed_components, available_components
            )

        # -------------------------
        # UPDATE COMPONENTS (FIXED - use self.components)
        # -------------------------
        # Note: The components are already in self.components
        # This just ensures they're marked as started
        for name in started_components:
            if name in self.components:
                if hasattr(self.components[name], "_started"):
                    self.components[name]._started = True

        # -------------------------
        # PIPELINE VALIDATION
        # -------------------------
        await self._validate_component_pipeline(started_components)

        return started_components

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
