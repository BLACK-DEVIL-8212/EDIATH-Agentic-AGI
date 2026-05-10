"""
EDIATH Multi-Agent Coordinator Module
Production-grade multi-agent coordination and orchestration for autonomous AI system
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from core.utils.logger import logger
from core.system.task_queue import Task, TaskPriority, TaskStatus
from core.system.rate_limiter import get_rate_limiter

# #==================== Enums #====================


class AgentRole(Enum):
    """Agent roles in the multi-agent system"""

    MASTER = "master"
    WORKER = "worker"
    SPECIALIST = "specialist"
    SUPERVISOR = "supervisor"
    ORCHESTRATOR = "orchestrator"


class AgentStatus(Enum):
    """Agent status enumeration"""

    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class CommunicationProtocol(Enum):
    """Communication protocols between agents"""

    DIRECT = "direct"
    BROADCAST = "broadcast"
    PUBLISH_SUBSCRIBE = "pubsub"
    ROUTED = "routed"
    HIERARCHICAL = "hierarchical"


class CoordinationStrategy(Enum):
    """Coordination strategies"""

    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    SWARM = "swarm"


class TaskAllocationStrategy(Enum):
    """Task allocation strategies"""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    PRIORITY_BASED = "priority_based"
    AUCTION_BASED = "auction_based"


# #==================== Data Classes #====================


@dataclass
class AgentCapability:
    """Agent capability definition"""

    name: str
    version: str
    actions: List[str]
    confidence: float = 1.0
    max_concurrent_tasks: int = 5
    priority: int = 0


@dataclass
class AgentInfo:
    """Agent information"""

    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[AgentCapability] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    total_tasks_processed: int = 0
    success_rate: float = 1.0
    average_response_time: float = 0.0
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterAgentMessage:
    """Message between agents"""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = "request"
    content: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    requires_response: bool = True
    response_received: bool = False
    response: Any = None
    timeout_seconds: int = 30


@dataclass
class CollaborativeTask:
    """Collaborative task involving multiple agents"""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    assigned_agents: List[str] = field(default_factory=list)
    completed_agents: Set[str] = field(default_factory=set)
    failed_agents: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    requires_all_agents: bool = True
    timeout_seconds: int = 300


# #==================== Agent Base Class #====================


class BaseAgent:
    """Base class for all agents in the multi-agent system"""

    def __init__(self, agent_id: str, name: str, role: AgentRole = AgentRole.WORKER):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.capabilities: List[AgentCapability] = []
        self.current_tasks: List[str] = []
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.response_times: List[float] = []
        self.coordinator: Optional["MultiAgentCoordinator"] = None
        self.message_queue: deque = deque()
        self.is_running = False
        self._task_lock = asyncio.Lock()

        logger.info(f"Agent {agent_id} ({name}) initialized with role {role.value}")

    def add_capability(self, capability: AgentCapability):
        """Add a capability to the agent"""
        self.capabilities.append(capability)
        logger.debug(f"Agent {self.agent_id} gained capability: {capability.name}")

    def can_handle(self, action: str) -> bool:
        """Check if agent can handle a specific action"""
        for capability in self.capabilities:
            if action in capability.actions:
                return True
        return False

    def get_capability_score(self, action: str) -> float:
        """Get capability score for an action"""
        for capability in self.capabilities:
            if action in capability.actions:
                return capability.confidence
        return 0.0

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement execute_task")

    async def send_message(self, message: InterAgentMessage) -> Optional[Any]:
        """Send message to another agent"""
        if not self.coordinator:
            logger.error(f"Agent {self.agent_id} not registered with coordinator")
            return None

        message.sender_id = self.agent_id
        return await self.coordinator.route_message(message)

    async def receive_message(self, message: InterAgentMessage) -> Any:
        """Receive and process a message"""
        logger.debug(
            f"Agent {self.agent_id} received message from {message.sender_id}: {message.message_type}"
        )

        if message.message_type == "heartbeat":
            return {"status": "alive", "agent_id": self.agent_id}
        elif message.message_type == "status_request":
            return self.get_status()
        elif message.message_type == "capability_request":
            return [c.__dict__ for c in self.capabilities]
        elif message.message_type == "task_offer":
            return await self._handle_task_offer(message.content)

        return {"received": True, "message_id": message.message_id}

    async def _handle_task_offer(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task offer from coordinator"""
        if self.status != AgentStatus.IDLE:
            return {"accepted": False, "reason": "busy"}

        action = task_data.get("action", "")
        if not self.can_handle(action):
            return {"accepted": False, "reason": "cannot_handle"}

        # Calculate estimated time
        estimated_time = task_data.get("estimated_time", 5)

        return {
            "accepted": True,
            "estimated_time": estimated_time,
            "confidence": self.get_capability_score(action),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "current_tasks": self.current_tasks,
            "total_tasks": self.total_tasks_processed,
            "success_rate": self.successful_tasks / max(1, self.total_tasks_processed),
            "avg_response_time": sum(self.response_times)
            / max(1, len(self.response_times)),
            "capabilities": [c.name for c in self.capabilities],
        }

    async def start(self):
        """Start the agent"""
        self.is_running = True
        logger.info(f"Agent {self.agent_id} started")

    async def stop(self):
        """Stop the agent"""
        self.is_running = False
        logger.info(f"Agent {self.agent_id} stopped")

    async def health_check(self) -> bool:
        """Health check for the agent"""
        return self.is_running and self.status != AgentStatus.ERROR


# #==================== Specialized Agents #====================


class LLMAgent(BaseAgent):
    """Specialized agent for LLM operations"""

    def __init__(self, agent_id: str, name: str, llm_engine):
        super().__init__(agent_id, name, AgentRole.SPECIALIST)
        self.llm_engine = llm_engine

        # Add capabilities
        self.add_capability(
            AgentCapability(
                name="text_generation",
                version="1.0",
                actions=["generate", "chat", "complete", "summarize"],
                confidence=0.95,
            )
        )
        self.add_capability(
            AgentCapability(
                name="code_generation",
                version="1.0",
                actions=["write_code", "explain_code", "debug"],
                confidence=0.90,
            )
        )

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute LLM task"""
        start_time = time.time()
        self.status = AgentStatus.BUSY

        try:
            action = task.data.get("action", "generate")
            prompt = task.data.get("prompt", "")

            if action == "generate":
                result = await self.llm_engine.generate(prompt)
            elif action == "chat":
                context = task.data.get("context", "")
                result = await self.llm_engine.chat(prompt, context)
            elif action == "summarize":
                text = task.data.get("text", "")
                result = await self.llm_engine.summarize(text)
            else:
                result = await self.llm_engine.generate(prompt)

            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self.total_tasks_processed += 1
            self.successful_tasks += 1

            return {
                "success": True,
                "result": result,
                "response_time_ms": response_time,
            }

        except Exception as e:
            self.failed_tasks += 1
            self.total_tasks_processed += 1
            return {"success": False, "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE


class AutomationAgent(BaseAgent):
    """Specialized agent for automation tasks"""

    def __init__(self, agent_id: str, name: str, browser_controller):
        super().__init__(agent_id, name, AgentRole.WORKER)
        self.browser_controller = browser_controller

        self.add_capability(
            AgentCapability(
                name="web_automation",
                version="1.0",
                actions=["browse", "scrape", "click", "fill_form", "navigate"],
                confidence=0.85,
            )
        )

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute automation task"""
        start_time = time.time()
        self.status = AgentStatus.BUSY

        try:
            action = task.data.get("action", "")
            url = task.data.get("url", "")

            if action == "browse":
                result = await self.browser_controller.open_url(url)
            elif action == "scrape":
                selector = task.data.get("selector", "")
                result = await self.browser_controller.scrape(selector)
            else:
                result = {"status": "action_not_recognized"}

            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self.total_tasks_processed += 1
            self.successful_tasks += 1

            return {
                "success": True,
                "result": result,
                "response_time_ms": response_time,
            }

        except Exception as e:
            self.failed_tasks += 1
            self.total_tasks_processed += 1
            return {"success": False, "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE


class LearningAgent(BaseAgent):
    """Specialized agent for learning and adaptation"""

    def __init__(self, agent_id: str, name: str, curiosity_engine):
        super().__init__(agent_id, name, AgentRole.SPECIALIST)
        self.curiosity_engine = curiosity_engine

        self.add_capability(
            AgentCapability(
                name="learning",
                version="1.0",
                actions=["learn", "adapt", "explore", "optimize"],
                confidence=0.80,
            )
        )

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute learning task"""
        start_time = time.time()
        self.status = AgentStatus.BUSY

        try:
            action = task.data.get("action", "")
            feedback = task.data.get("feedback", "")

            if action == "learn" and feedback:
                result = await self.curiosity_engine.process_feedback(feedback)
            elif action == "explore":
                duration = task.data.get("duration", 10)
                result = await self.curiosity_engine.autonomous_exploration(duration)
            else:
                result = {"status": "learning_pending"}

            response_time = (time.time() - start_time) * 1000
            self.response_times.append(response_time)
            self.total_tasks_processed += 1
            self.successful_tasks += 1

            return {
                "success": True,
                "result": result,
                "response_time_ms": response_time,
            }

        except Exception as e:
            self.failed_tasks += 1
            self.total_tasks_processed += 1
            return {"success": False, "error": str(e)}
        finally:
            self.status = AgentStatus.IDLE


class SupervisorAgent(BaseAgent):
    """Supervisor agent for monitoring and coordination"""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.SUPERVISOR)
        self.monitored_agents: Set[str] = set()
        self.alerts: List[Dict[str, Any]] = []

        self.add_capability(
            AgentCapability(
                name="supervision",
                version="1.0",
                actions=["monitor", "alert", "coordinate", "resolve"],
                confidence=0.95,
            )
        )

    async def monitor_agent(self, agent_id: str):
        """Start monitoring an agent"""
        self.monitored_agents.add(agent_id)
        logger.info(f"Supervisor {self.agent_id} now monitoring agent {agent_id}")

    async def check_agent_health(self, agent_info: AgentInfo) -> bool:
        """Check health of monitored agent"""
        if agent_info.status == AgentStatus.ERROR:
            await self.raise_alert(agent_info.agent_id, "agent_error", agent_info)
            return False

        time_since_heartbeat = (
            datetime.now() - agent_info.last_heartbeat
        ).total_seconds()
        if time_since_heartbeat > 60:
            await self.raise_alert(
                agent_info.agent_id,
                "heartbeat_missing",
                {"seconds": time_since_heartbeat},
            )
            return False

        return True

    async def raise_alert(self, agent_id: str, alert_type: str, details: Any):
        """Raise an alert for an agent"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "type": alert_type,
            "details": details,
        }
        self.alerts.append(alert)
        logger.warning(
            f"Alert from supervisor {self.agent_id}: {alert_type} for agent {agent_id}"
        )

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute supervision task"""
        start_time = time.time()

        try:
            action = task.data.get("action", "")

            if action == "get_alerts":
                return {"alerts": self.alerts[-50:]}
            elif action == "get_monitored_agents":
                return {"agents": list(self.monitored_agents)}
            elif action == "clear_alerts":
                self.alerts.clear()
                return {"cleared": True}

            return {"status": "supervisor_ready"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute supervisor action"""
        if action == "get_alerts":
            return {"alerts": self.alerts[-50:]}
        elif action == "get_monitored_agents":
            return {"agents": list(self.monitored_agents)}
        return {"status": "unknown_action"}


# #==================== Multi-Agent Coordinator #====================


class MultiAgentCoordinator:
    """
    Production-grade multi-agent coordinator
    Manages agent lifecycle, task allocation, and inter-agent communication
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_info: Dict[str, AgentInfo] = {}
        self.collaborative_tasks: Dict[str, CollaborativeTask] = {}
        self.message_bus: deque = deque()
        self.pending_messages: Dict[str, InterAgentMessage] = {}

        # Coordination settings
        self.coordination_strategy = CoordinationStrategy.HIERARCHICAL
        self.task_allocation_strategy = TaskAllocationStrategy.CAPABILITY_BASED
        self.communication_protocol = CommunicationProtocol.DIRECT

        # Task queues by priority
        self.task_queues: Dict[TaskPriority, deque] = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque(),
            TaskPriority.BACKGROUND: deque(),
        }

        # Locks and state
        self._lock = asyncio.Lock()
        self.is_running = False
        self._coordinator_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Metrics
        self.total_tasks_allocated = 0
        self.total_messages_sent = 0
        self.collaborative_tasks_completed = 0
        self.agent_performance_history: Dict[str, List[float]] = {}

        # Rate limiter
        self.rate_limiter = None

        logger.info("MultiAgentCoordinator initialized")

    async def initialize(self, *args, **kwargs):
        """Initialize the coordinator"""
        self.rate_limiter = await get_rate_limiter()
        logger.info("MultiAgentCoordinator initialized successfully")
        return True

    async def register_agent(self, agent: BaseAgent) -> bool:
        """
        Register an agent with the coordinator (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if agent is None:
                raise ValueError("Agent cannot be None")

            if not hasattr(agent, "agent_id") or not agent.agent_id:
                raise ValueError("Agent must have a valid agent_id")

            agent_id = str(agent.agent_id).strip()

            if not agent_id:
                raise ValueError("Agent ID cannot be empty")

            # ------------------------
            # 🔥 SAFE LOCK
            # ------------------------
            async with self._lock:

                # ------------------------
                # 🔥 DUPLICATE CHECK
                # ------------------------
                if agent_id in self.agents:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Agent already registered: {agent_id}")
                    except Exception:
                        pass
                    return False

                # ------------------------
                # 🔥 SAFE REGISTRY INIT
                # ------------------------
                if not isinstance(getattr(self, "agents", None), dict):
                    self.agents = {}

                if not isinstance(getattr(self, "agent_info", None), dict):
                    self.agent_info = {}

                # ------------------------
                # 🔥 REGISTER AGENT
                # ------------------------
                self.agents[agent_id] = agent
                agent.coordinator = self

                # ------------------------
                # 🔥 SAFE AGENT INFO BUILD
                # ------------------------
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    name=getattr(agent, "name", "unknown"),
                    role=getattr(agent, "role", AgentRole.WORKER),
                    capabilities=list(getattr(agent, "capabilities", [])),
                )

                self.agent_info[agent_id] = agent_info

                # ------------------------
                # 🔥 METRICS (SAFE)
                # ------------------------
                try:
                    self.total_agents = len(self.agents)
                except Exception:
                    pass

                # ------------------------
                # 🔥 SAFE LOGGING (FIXED)
                # ------------------------
                try:
                    if hasattr(self, "logger"):
                        self.logger.info(
                            f"✅ Agent registered: {agent_id} "
                            f"({agent_info.name}, role={agent_info.role.value})"
                        )
                except Exception:
                    pass

                return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ register_agent failed: {e}")
            except Exception:
                pass

            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the coordinator (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not agent_id:
                raise ValueError("Agent ID cannot be empty")

            agent_id = str(agent_id).strip()

            if not agent_id:
                raise ValueError("Invalid agent ID")

            # ------------------------
            # 🔥 SAFE LOCK
            # ------------------------
            async with self._lock:

                # ------------------------
                # 🔥 SAFE REGISTRY ACCESS
                # ------------------------
                agents = getattr(self, "agents", {})
                agent_info = getattr(self, "agent_info", {})

                if agent_id not in agents:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Agent not found: {agent_id}")
                    except Exception:
                        pass
                    return False

                agent = agents.get(agent_id)

                # ------------------------
                # 🔥 SAFE STOP (FIXED)
                # ------------------------
                try:
                    if agent and hasattr(agent, "stop"):
                        result = agent.stop()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception as e:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"Agent stop failed: {agent_id} - {e}")
                    except Exception:
                        pass

                # ------------------------
                # 🔥 REMOVE AGENT
                # ------------------------
                try:
                    agents.pop(agent_id, None)
                    agent_info.pop(agent_id, None)
                except Exception:
                    pass

                # ------------------------
                # 🔥 METRICS (SAFE)
                # ------------------------
                try:
                    self.total_agents = len(agents)
                except Exception:
                    pass

                # ------------------------
                # 🔥 SAFE LOGGING (FIXED)
                # ------------------------
                try:
                    if hasattr(self, "logger"):
                        self.logger.info(f"🛑 Agent unregistered: {agent_id}")
                except Exception:
                    pass

                return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ unregister_agent failed: {e}")
            except Exception:
                pass

            return False

    async def allocate_task(self, task: Task) -> Optional[str]:
        """
        Allocate a task to the most suitable agent (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if task is None:
                raise ValueError("Task cannot be None")

            task_id = getattr(task, "id", None)
            if not task_id:
                raise ValueError("Task must have a valid id")

            # ------------------------
            # 🔥 SAFE LOCK
            # ------------------------
            async with self._lock:

                # ------------------------
                # 🔥 SAFE AGENT REGISTRY
                # ------------------------
                agents = getattr(self, "agents", {}) or {}
                agent_info = getattr(self, "agent_info", {}) or {}

                # ------------------------
                # 🔥 FIND SUITABLE AGENTS
                # ------------------------
                try:
                    suitable_agents = self._find_suitable_agents(task) or []
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Agent selection failed: {e}")
                    return None

                if not suitable_agents:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.warning(
                                f"No suitable agent for task: {task_id}"
                            )
                    except Exception:
                        pass
                    return None

                # ------------------------
                # 🔥 SELECT BEST AGENT
                # ------------------------
                try:
                    selected_agent = await self._select_agent(task, suitable_agents)
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Agent selection error: {e}")
                    return None

                if not selected_agent or selected_agent not in agents:
                    return None

                agent = agents.get(selected_agent)

                # ------------------------
                # 🔥 SAFE TASK ASSIGNMENT
                # ------------------------
                try:
                    if not hasattr(agent, "current_tasks"):
                        agent.current_tasks = []

                    if selected_agent not in agent_info:
                        agent_info[selected_agent] = type("AgentInfo", (), {})()
                        agent_info[selected_agent].current_tasks = []

                    if not hasattr(agent_info[selected_agent], "current_tasks"):
                        agent_info[selected_agent].current_tasks = []

                    agent.current_tasks.append(task_id)
                    agent_info[selected_agent].current_tasks.append(task_id)

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Task assignment failed: {e}")
                    return None

                # ------------------------
                # 🔥 METRICS (SAFE)
                # ------------------------
                try:
                    self.total_tasks_allocated = (
                        int(getattr(self, "total_tasks_allocated", 0)) + 1
                    )
                except Exception:
                    pass

                # ------------------------
                # 🔥 EXECUTE ASYNC (SAFE)
                # ------------------------
                try:
                    asyncio.create_task(self._execute_task_on_agent(agent, task))
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.error(f"Task execution dispatch failed: {e}")

                # ------------------------
                # 🔥 SAFE LOGGING
                # ------------------------
                try:
                    if hasattr(self, "logger"):
                        self.logger.info(
                            f"📌 Task {task_id} allocated to agent {selected_agent}"
                        )
                except Exception:
                    pass

                return selected_agent

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ allocate_task failed: {e}")
            except Exception:
                pass

            return None

    def _find_suitable_agents(self, task: Task) -> List[str]:
        """Find agents that can handle the task"""
        action = task.data.get("action", "")
        suitable = []

        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.IDLE and agent.can_handle(action):
                suitable.append(agent_id)

        return suitable

    async def _select_agent(
        self, task: Task, suitable_agents: List[str]
    ) -> Optional[str]:
        """
        Select the best agent based on strategy (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not suitable_agents:
                return None

            agents = getattr(self, "agents", {}) or {}

            strategy = getattr(self, "task_allocation_strategy", None)

            # ------------------------
            # 🔁 ROUND ROBIN
            # ------------------------
            if strategy == TaskAllocationStrategy.ROUND_ROBIN:
                return suitable_agents[0]

            # ------------------------
            # ⚖️ LOAD BALANCED
            # ------------------------
            elif strategy == TaskAllocationStrategy.LOAD_BALANCED:
                try:
                    return min(
                        suitable_agents,
                        key=lambda a: len(getattr(agents.get(a), "current_tasks", [])),
                    )
                except Exception:
                    return suitable_agents[0]

            # ------------------------
            # 🧠 CAPABILITY BASED
            # ------------------------
            elif strategy == TaskAllocationStrategy.CAPABILITY_BASED:
                action = ""
                try:
                    action = str(getattr(task, "data", {}).get("action", ""))
                except Exception:
                    pass

                def score(a):
                    agent = agents.get(a)
                    if agent and hasattr(agent, "get_capability_score"):
                        try:
                            return float(agent.get_capability_score(action))
                        except Exception:
                            return 0.0
                    return 0.0

                try:
                    return max(suitable_agents, key=score)
                except Exception:
                    return suitable_agents[0]

            # ------------------------
            # 🔥 PRIORITY BASED (FIXED)
            # ------------------------
            elif strategy == TaskAllocationStrategy.PRIORITY_BASED:
                try:
                    priority = getattr(getattr(task, "priority", None), "value", 0)
                    return max(suitable_agents, key=lambda _: priority)
                except Exception:
                    return suitable_agents[0]

            # ------------------------
            # 🔁 DEFAULT FALLBACK
            # ------------------------
            return suitable_agents[0]

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"_select_agent failed: {e}")
            except Exception:
                pass

            return suitable_agents[0] if suitable_agents else None

    async def _execute_task_on_agent(self, agent: BaseAgent, task: Task):
        """
        Execute task on selected agent (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not agent or not task:
                return

            agent_id = getattr(agent, "agent_id", None)
            if not agent_id:
                return

            # ------------------------
            # 🔥 SAFE REGISTRY ACCESS
            # ------------------------
            agent_info = getattr(self, "agent_info", {}).get(agent_id)
            if not agent_info:
                return

            # ------------------------
            # 🔥 EXECUTION
            # ------------------------
            start_time = time.time()

            try:
                result = await agent.execute_task(task)
            except Exception as e:
                result = {"success": False, "error": str(e)}

            execution_time = (time.time() - start_time) * 1000

            success = bool(result.get("success", False))

            # ------------------------
            # 🔥 METRICS UPDATE (SAFE ORDER FIX)
            # ------------------------
            try:
                agent_info.total_tasks_processed = (
                    int(getattr(agent_info, "total_tasks_processed", 0)) + 1
                )

                if success:
                    agent_info.successful_tasks = (
                        int(getattr(agent_info, "successful_tasks", 0)) + 1
                    )
                else:
                    agent_info.failed_tasks = (
                        int(getattr(agent_info, "failed_tasks", 0)) + 1
                    )

                # 🔥 FIX: correct success rate calculation
                total = agent_info.total_tasks_processed
                if total > 0:
                    agent_info.success_rate = agent_info.successful_tasks / total

            except Exception:
                pass

            # ------------------------
            # 🔥 PERFORMANCE TRACKING (SAFE)
            # ------------------------
            try:
                if not hasattr(self, "agent_performance_history"):
                    self.agent_performance_history = {}

                history = self.agent_performance_history.setdefault(agent_id, [])
                history.append(execution_time)

                # keep last 100
                if len(history) > 100:
                    self.agent_performance_history[agent_id] = history[-100:]

            except Exception:
                pass

            # ------------------------
            # 🔥 CLEANUP CURRENT TASKS (SAFE)
            # ------------------------
            try:
                if hasattr(agent, "current_tasks") and task.id in agent.current_tasks:
                    agent.current_tasks.remove(task.id)

                if (
                    hasattr(agent_info, "current_tasks")
                    and task.id in agent_info.current_tasks
                ):
                    agent_info.current_tasks.remove(task.id)

            except Exception:
                pass

            # ------------------------
            # 🔥 UPDATE TASK STATE
            # ------------------------
            try:
                if success:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.get("error", "Unknown error")
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(
                        f"✅ Task {task.id} completed on agent {agent_id} "
                        f"({'success' if success else 'failed'}) in {round(execution_time,2)}ms"
                    )
            except Exception:
                pass

        except Exception as e:
            # ------------------------
            # 🔥 HARD FAILSAFE
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"❌ Task execution failed on agent {agent_id}: {e}"
                    )
            except Exception:
                pass

            try:
                task.status = TaskStatus.FAILED
                task.error = str(e)
            except Exception:
                pass

    async def create_collaborative_task(
        self, name: str, agent_ids: List[str], requires_all: bool = True
    ) -> str:
        """Create a collaborative task involving multiple agents"""
        async with self._lock:
            collab_task = CollaborativeTask(
                name=name, assigned_agents=agent_ids, requires_all_agents=requires_all
            )
            self.collaborative_tasks[collab_task.task_id] = collab_task
            logger.info(
                f"Collaborative task {collab_task.task_id} created with {len(agent_ids)} agents"
            )
            return collab_task.task_id

    async def submit_collaborative_subtask(
        self,
        task_id: str,
        agent_id: str,
        result: Any = None,
        error: Optional[str] = None,
    ):
        """Submit result from a collaborative subtask"""
        async with self._lock:
            if task_id not in self.collaborative_tasks:
                logger.warning(f"Collaborative task {task_id} not found")
                return

            collab_task = self.collaborative_tasks[task_id]

            if error:
                collab_task.failed_agents.add(agent_id)
            else:
                collab_task.completed_agents.add(agent_id)

            logger.info(f"Subtask from {agent_id} completed for task {task_id}")

            # Check if task is complete
            await self._check_collaborative_completion(collab_task)

    async def _check_collaborative_completion(self, collab_task: CollaborativeTask):
        """Check if collaborative task is complete"""
        if collab_task.requires_all_agents:
            all_completed = len(collab_task.completed_agents) == len(
                collab_task.assigned_agents
            )
        else:
            all_completed = len(collab_task.completed_agents) > 0

        if all_completed:
            collab_task.status = TaskStatus.COMPLETED
            self.collaborative_tasks_completed += 1
            logger.info(f"Collaborative task {collab_task.task_id} completed")

            # Aggregate results
            results = {}
            # This would aggregate results from all agents
            collab_task.result = results

    async def route_message(self, message: InterAgentMessage) -> Optional[Any]:
        """
        Route a message between agents (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if message is None:
                return None

            sender_id = getattr(message, "sender_id", None)
            recipient_id = getattr(message, "recipient_id", None)

            # ------------------------
            # 🔥 SAFE METRICS
            # ------------------------
            try:
                self.total_messages_sent = (
                    int(getattr(self, "total_messages_sent", 0)) + 1
                )
            except Exception:
                pass

            agents = getattr(self, "agents", {}) or {}

            # ------------------------
            # 🔥 RATE LIMIT (SAFE)
            # ------------------------
            if getattr(self, "rate_limiter", None):
                try:
                    rate_result = await self.rate_limiter.check_rate_limit(
                        identifier="message_routing",
                        context={"sender": sender_id, "recipient": recipient_id},
                    )

                    if not rate_result.get("allowed", True):
                        if hasattr(self, "logger"):
                            self.logger.warning("⚠ Message routing rate limited")
                        return None
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Rate limiter failed: {e}")

            # ------------------------
            # 🔥 DIRECT MESSAGE
            # ------------------------
            if recipient_id and recipient_id in agents:
                recipient = agents.get(recipient_id)

                try:
                    response = await recipient.receive_message(message)
                except Exception as e:
                    response = {"success": False, "error": str(e)}

                if getattr(message, "requires_response", False):
                    message.response = response
                    message.response_received = True

                return response

            # ------------------------
            # 🔥 BROADCAST MESSAGE
            # ------------------------
            elif not recipient_id or recipient_id == "all":
                responses = {}

                for agent_id, agent in agents.items():
                    if agent_id == sender_id:
                        continue

                    try:
                        response = await agent.receive_message(message)
                    except Exception as e:
                        response = {"success": False, "error": str(e)}

                    responses[agent_id] = response

                if getattr(message, "requires_response", False):
                    return responses

                return responses

            # ------------------------
            # 🔥 STORE FOR LATER (SAFE)
            # ------------------------
            else:
                try:
                    if not hasattr(self, "pending_messages") or not isinstance(
                        self.pending_messages, dict
                    ):
                        self.pending_messages = {}

                    if not hasattr(self, "message_bus") or not isinstance(
                        self.message_bus, list
                    ):
                        self.message_bus = []

                    message_id = getattr(message, "message_id", None)

                    if message_id:
                        self.pending_messages[message_id] = message

                    self.message_bus.append(message)

                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"Message storage failed: {e}")

                return None

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ route_message failed: {e}")
            except Exception:
                pass

            return None

    async def broadcast(
        self, message_type: str, content: Any, sender_id: str
    ) -> Dict[str, Any]:
        """Broadcast a message to all agents"""
        message = InterAgentMessage(
            sender_id=sender_id,
            recipient_id="all",
            message_type=message_type,
            content=content,
        )
        return await self.route_message(message)

    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent"""
        if agent_id in self.agents:
            return self.agents[agent_id].get_status()
        return None

    async def get_all_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        status = {}
        for agent_id, agent in self.agents.items():
            status[agent_id] = agent.get_status()
        return status

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics (optimized + production-grade)"""

        try:
            async with self._lock:
                agents = getattr(self, "agents", {}) or {}
                agent_info = getattr(self, "agent_info", {}) or {}

                # ------------------------
                # 🔥 SAFE TOTAL CAPABILITIES
                # ------------------------
                total_capabilities = sum(
                    len(getattr(a, "capabilities", [])) for a in agents.values()
                )

                # ------------------------
                # 🔥 SAFE AVG RESPONSE TIME (CRITICAL FIX)
                # ------------------------
                response_times = []

                for info in agent_info.values():
                    times = getattr(info, "response_times", [])
                    if isinstance(times, list) and times:
                        response_times.append(sum(times) / len(times))

                avg_response_time = (
                    sum(response_times) / len(response_times) if response_times else 0.0
                )

                # ------------------------
                # 🔥 AGENT COUNTS (SAFE)
                # ------------------------
                total_agents = len(agents)

                active_agents = sum(
                    1
                    for a in agents.values()
                    if getattr(a, "status", None) == AgentStatus.IDLE
                )

                busy_agents = sum(
                    1
                    for a in agents.values()
                    if getattr(a, "status", None) == AgentStatus.BUSY
                )

                # ------------------------
                # 🔥 SAFE PERFORMANCE METRICS
                # ------------------------
                performance = {}
                for aid, perf in getattr(self, "agent_performance_history", {}).items():
                    if isinstance(perf, list) and perf:
                        performance[aid] = sum(perf) / len(perf)
                    else:
                        performance[aid] = 0.0

                # ------------------------
                # 🔥 FINAL OUTPUT
                # ------------------------
                return {
                    "total_agents": total_agents,
                    "active_agents": active_agents,
                    "busy_agents": busy_agents,
                    "total_tasks_allocated": int(
                        getattr(self, "total_tasks_allocated", 0)
                    ),
                    "total_messages_sent": int(getattr(self, "total_messages_sent", 0)),
                    "collaborative_tasks_completed": int(
                        getattr(self, "collaborative_tasks_completed", 0)
                    ),
                    "pending_messages": len(getattr(self, "pending_messages", {})),
                    "total_capabilities": total_capabilities,
                    "average_response_time_ms": round(avg_response_time, 3),
                    "agent_performance": performance,
                }

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ get_system_metrics failed: {e}")
            except Exception:
                pass

            return {
                "status": "error",
                "total_agents": 0,
                "average_response_time_ms": 0.0,
            }

    async def start(self):
        """Start the coordinator"""
        if self.is_running:
            return

        self.is_running = True

        # Start all agents
        for agent in self.agents.values():
            await agent.start()

        # Start coordinator tasks
        self._coordinator_task = asyncio.create_task(self._coordination_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info("MultiAgentCoordinator started")

    async def _coordination_loop(self):
        """
        Main coordination loop (optimized + production-grade)
        """

        import asyncio

        try:
            while getattr(self, "is_running", False):

                try:
                    # ------------------------
                    # 🔥 SAFE MESSAGE PROCESSING
                    # ------------------------
                    message_bus = getattr(self, "message_bus", None)

                    if message_bus:
                        while message_bus:
                            try:
                                message = message_bus.popleft()
                            except Exception:
                                break

                            try:
                                await self.route_message(message)
                            except Exception as e:
                                if hasattr(self, "logger"):
                                    self.logger.warning(f"Message routing failed: {e}")

                    # ------------------------
                    # 🔥 SAFE TASK QUEUE PROCESSING
                    # ------------------------
                    task_queues = getattr(self, "task_queues", {}) or {}

                    for priority in (
                        TaskPriority.CRITICAL,
                        TaskPriority.HIGH,
                        TaskPriority.NORMAL,
                        TaskPriority.LOW,
                        TaskPriority.BACKGROUND,
                    ):
                        queue = task_queues.get(priority)

                        if not queue:
                            continue

                        try:
                            if len(queue) > 0:
                                task = queue.popleft()
                            else:
                                continue
                        except Exception:
                            continue

                        try:
                            await self.allocate_task(task)
                        except Exception as e:
                            if hasattr(self, "logger"):
                                self.logger.error(f"Task allocation failed: {e}")

                        # 🔥 process ONE task per loop (prevents starvation)
                        break

                    # ------------------------
                    # 🔥 CONTROLLED LOOP DELAY
                    # ------------------------
                    await asyncio.sleep(0.1)

                except asyncio.CancelledError:
                    # 🔥 graceful shutdown
                    break

                except Exception as e:
                    # ------------------------
                    # 🔥 SAFE ERROR HANDLING
                    # ------------------------
                    try:
                        if hasattr(self, "logger"):
                            self.logger.error(f"Coordination loop error: {e}")
                    except Exception:
                        pass

                    # 🔥 prevent tight crash loop
                    await asyncio.sleep(1)

        finally:
            # ------------------------
            # 🔥 CLEAN EXIT STATE
            # ------------------------
            try:
                self.is_running = False
            except Exception:
                pass

    async def _health_check_loop(self):
        """
        Health check loop for agents (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        try:
            while getattr(self, "is_running", False):

                try:
                    # ------------------------
                    # 🔥 CONTROLLED INTERVAL
                    # ------------------------
                    await asyncio.sleep(30)

                    agents = getattr(self, "agents", {}) or {}
                    agent_info_map = getattr(self, "agent_info", {}) or {}

                    # ------------------------
                    # 🔥 ITERATE SAFELY (COPY)
                    # ------------------------
                    for agent_id, agent in list(agents.items()):

                        info = agent_info_map.get(agent_id)
                        if not agent or not info:
                            continue

                        # ------------------------
                        # 🔥 SAFE HEALTH CHECK
                        # ------------------------
                        try:
                            is_healthy = await agent.health_check()
                        except Exception as e:
                            is_healthy = False
                            if hasattr(self, "logger"):
                                self.logger.warning(
                                    f"Health check error for {agent_id}: {e}"
                                )

                        # ------------------------
                        # ❌ UNHEALTHY AGENT
                        # ------------------------
                        if not is_healthy:
                            try:
                                info.status = AgentStatus.ERROR
                            except Exception:
                                pass

                            try:
                                if hasattr(self, "logger"):
                                    self.logger.warning(
                                        f"⚠ Agent unhealthy: {agent_id}"
                                    )
                            except Exception:
                                pass

                            # ------------------------
                            # 🔁 RESTART LOGIC (SAFE)
                            # ------------------------
                            try:
                                if hasattr(agent, "stop"):
                                    res = agent.stop()
                                    if asyncio.iscoroutine(res):
                                        await res

                                if hasattr(agent, "start"):
                                    res = agent.start()
                                    if asyncio.iscoroutine(res):
                                        await res

                                info.status = AgentStatus.IDLE

                                if hasattr(self, "logger"):
                                    self.logger.info(f"🔄 Agent restarted: {agent_id}")

                            except Exception as e:
                                try:
                                    info.status = AgentStatus.ERROR
                                except Exception:
                                    pass

                                if hasattr(self, "logger"):
                                    self.logger.error(
                                        f"❌ Restart failed for {agent_id}: {e}"
                                    )

                        # ------------------------
                        # ✅ HEALTHY AGENT
                        # ------------------------
                        else:
                            try:
                                info.last_heartbeat = datetime.now()

                                if getattr(info, "status", None) == AgentStatus.ERROR:
                                    info.status = AgentStatus.IDLE

                            except Exception:
                                pass

                except asyncio.CancelledError:
                    # 🔥 graceful shutdown
                    break

                except Exception as e:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.error(f"Health check loop error: {e}")
                    except Exception:
                        pass

                    # prevent tight failure loop
                    await asyncio.sleep(5)

        finally:
            # ------------------------
            # 🔥 CLEAN EXIT
            # ------------------------
            try:
                self.is_running = False
            except Exception:
                pass

    async def shutdown(self):
        """
        Shutdown the coordinator and all agents (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            if hasattr(self, "logger"):
                self.logger.info("🛑 Shutting down MultiAgentCoordinator...")

            # ------------------------
            # 🔥 STOP MAIN LOOP
            # ------------------------
            self.is_running = False

            # ------------------------
            # 🔥 CANCEL BACKGROUND TASKS (SAFE)
            # ------------------------
            tasks = [
                getattr(self, "_coordinator_task", None),
                getattr(self, "_health_task", None),
            ]

            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            # ------------------------
            # 🔥 SHUTDOWN AGENTS (SAFE)
            # ------------------------
            agents = list(getattr(self, "agents", {}).values())

            for agent in agents:
                try:
                    if hasattr(agent, "stop"):
                        res = agent.stop()
                        if asyncio.iscoroutine(res):
                            await res
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"Agent shutdown failed: {getattr(agent, 'agent_id', 'unknown')} - {e}"
                        )

            # ------------------------
            # 🔥 FINAL CLEANUP
            # ------------------------
            try:
                self._coordinator_task = None
                self._health_task = None
            except Exception:
                pass

            # ------------------------
            # 🔥 FINAL LOG
            # ------------------------
            if hasattr(self, "logger"):
                self.logger.info("✅ MultiAgentCoordinator shutdown complete")

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Shutdown failed: {e}")
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get coordinator status (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 SAFE ATTRIBUTE ACCESS
            # ------------------------
            is_running = bool(getattr(self, "is_running", False))

            coordination_strategy = getattr(
                getattr(self, "coordination_strategy", None), "value", "unknown"
            )

            task_allocation_strategy = getattr(
                getattr(self, "task_allocation_strategy", None), "value", "unknown"
            )

            communication_protocol = getattr(
                getattr(self, "communication_protocol", None), "value", "unknown"
            )

            # ------------------------
            # 🔥 OPTIONAL METRICS
            # ------------------------
            total_agents = len(getattr(self, "agents", {}) or {})
            total_messages = int(getattr(self, "total_messages_sent", 0))
            total_tasks = int(getattr(self, "total_tasks_allocated", 0))

            # ------------------------
            # 🔥 BUILD RESPONSE
            # ------------------------
            return {
                "is_running": is_running,
                "coordination_strategy": coordination_strategy,
                "task_allocation_strategy": task_allocation_strategy,
                "communication_protocol": communication_protocol,
                # 🔥 EXTRA INSIGHT (UPGRADE)
                "total_agents": total_agents,
                "total_messages": total_messages,
                "total_tasks_allocated": total_tasks,
            }

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ get_status failed: {e}")
            except Exception:
                pass

            return {"is_running": False, "status": "error"}


# #==================== Global Instance #====================

_multi_agent_coordinator: Optional[MultiAgentCoordinator] = None
_coordinator_lock = asyncio.Lock()


async def get_multi_agent_coordinator() -> MultiAgentCoordinator:
    """Get global multi-agent coordinator instance"""
    global _multi_agent_coordinator
    async with _coordinator_lock:
        if _multi_agent_coordinator is None:
            _multi_agent_coordinator = MultiAgentCoordinator()
            await _multi_agent_coordinator.initialize()
    return _multi_agent_coordinator


async def shutdown_multi_agent_coordinator():
    """Shutdown the multi-agent coordinator"""
    global _multi_agent_coordinator
    if _multi_agent_coordinator:
        await _multi_agent_coordinator.shutdown()
        _multi_agent_coordinator = None


# #==================== Export #====================

__all__ = [
    "MultiAgentCoordinator",
    "BaseAgent",
    "LLMAgent",
    "AutomationAgent",
    "LearningAgent",
    "SupervisorAgent",
    "AgentRole",
    "AgentStatus",
    "CommunicationProtocol",
    "CoordinationStrategy",
    "TaskAllocationStrategy",
    "AgentCapability",
    "AgentInfo",
    "InterAgentMessage",
    "CollaborativeTask",
    "get_multi_agent_coordinator",
    "shutdown_multi_agent_coordinator",
]

# #==================== Example Usage #====================


async def example():
    """Example usage of multi-agent coordinator"""
    coordinator = await get_multi_agent_coordinator()

    # Start coordinator
    await coordinator.start()

    # Get metrics
    metrics = await coordinator.get_system_metrics()
    print(f"System metrics: {json.dumps(metrics, indent=2)}")

    # Shutdown
    await shutdown_multi_agent_coordinator()
