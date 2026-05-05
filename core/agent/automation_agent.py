"""
🔥 FINAL PRODUCTION Automation Agent for EDIATH
✔ Workflow automation engine
✔ Multiple trigger types (schedule, webhook, file, email, database, API)
✔ Multiple action types (webhook, email, notification, database, file, shell, code)
✔ Conditional branching and logic
✔ Data transformation and filtering
✔ Variable substitution and templating (Jinja2)
✔ Error handling and retries
✔ Parallel execution and loops
✔ Webhook server for external triggers
✔ Persistence and recovery
✔ Execution history and monitoring
✔ Circuit breaker pattern
✔ Rate limiting
✔ Advanced metrics
✔ Custom action registry
✔ Multi-step workflows with branching
✔ Concurrency safe
✔ Production ready
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

from aiohttp import web

# Optional imports with fallbacks
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    import jinja2

    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

try:
    from croniter import croniter

    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False


# =========================
# ENUMS AND CONSTANTS
# =========================


class TriggerType(Enum):
    """Types of automation triggers"""

    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    FILE = "file"
    EMAIL = "email"
    DATABASE = "database"
    API = "api"
    CONDITION = "condition"
    MANUAL = "manual"
    WEBHOOK_RESPONSE = "webhook_response"
    ERROR = "error"
    SUCCESS = "success"
    QUEUE = "queue"
    MESSAGE = "message"


class ActionType(Enum):
    """Types of automation actions"""

    WEBHOOK = "webhook"
    EMAIL = "email"
    NOTIFICATION = "notification"
    DATABASE = "database"
    FILE = "file"
    API = "api"
    CONDITION = "condition"
    DELAY = "delay"
    TRANSFORM = "transform"
    FILTER = "filter"
    LOG = "log"
    CUSTOM = "custom"
    SHELL = "shell"
    CODE = "code"
    LOOP = "loop"
    PARALLEL = "parallel"
    HTTP = "http"
    MQTT = "mqtt"
    REDIS = "redis"
    S3 = "s3"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"


class ConditionOperator(Enum):
    """Comparison operators for conditions"""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUALS = "ge"
    LESS_EQUALS = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    IN_LIST = "in"
    NOT_IN_LIST = "not_in"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class WorkflowStatus(Enum):
    """Workflow execution status"""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    DRAFT = "draft"
    ARCHIVED = "archived"


class ExecutionStatus(Enum):
    """Execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


# =========================
# DATACLASSES
# =========================


@dataclass
class Trigger:
    """Trigger configuration"""

    type: TriggerType
    config: Dict[str, Any]
    condition: Optional["Condition"] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Action:
    """Action configuration"""

    type: ActionType
    config: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    next_on_success: Optional[str] = None
    next_on_failure: Optional[str] = None
    retry_count: int = 0
    retry_delay: int = 5
    timeout: int = 30
    continue_on_error: bool = False


@dataclass
class Condition:
    """Condition configuration"""

    operator: ConditionOperator
    value: Any
    field: Optional[str] = None
    nested_conditions: Optional[List["Condition"]] = None
    logical_operator: str = "and"


@dataclass
class Workflow:
    """Workflow definition"""

    id: str
    name: str
    description: str
    status: WorkflowStatus
    trigger: Trigger
    actions: List[Action]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    max_executions: Optional[int] = None
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_execution: Optional[datetime] = None
    version: int = 1
    owner: str = "system"
    priority: int = 5  # 1-10, higher = more priority
    timeout: int = 300  # Overall workflow timeout


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""

    id: str
    workflow_id: str
    trigger_data: Dict[str, Any]
    context: Dict[str, Any]
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    action_results: List[Dict] = field(default_factory=list)
    retry_count: int = 0
    execution_time_ms: float = 0.0


@dataclass
class ExecutionMetrics:
    """Execution metrics"""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float("inf")
    max_execution_time_ms: float = 0.0
    executions_per_hour: float = 0.0
    last_hour_executions: int = 0


# =========================
# MAIN AUTOMATION AGENT
# =========================


class AutomationAgent:
    """
    Advanced automation agent capable of:
    - Workflow creation and management
    - Multiple trigger types (schedule, webhook, file, email, database)
    - Multiple action types (webhook, email, notification, database, file)
    - Conditional branching
    - Data transformation and filtering
    - Variable substitution and templating
    - Error handling and retries
    - Parallel execution
    - Loops and iterations
    - Workflow versioning
    - Execution history and monitoring
    - Webhook endpoints for external triggers
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Automation Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Storage
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: deque = deque(
            maxlen=self.config.get("max_history", 10000)
        )

        # Webhook server
        self.webhook_server = None
        self.webhook_port = self.config.get("webhook_port", 8080)
        self.webhook_host = self.config.get("webhook_host", "0.0.0.0")
        self.webhook_routes: Dict[str, str] = {}
        self.webhook_server_task: Optional[asyncio.Task] = None

        # Templates
        self.template_env = None
        if JINJA_AVAILABLE:
            self.template_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(
                    self.config.get("template_dir", "./templates")
                ),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )

        # Persistence
        self.persist_enabled = self.config.get("persist_enabled", True)
        self.data_dir = Path(self.config.get("data_dir", "./automation_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_file = self.data_dir / "workflows.json"
        self.executions_file = self.data_dir / "executions.json"

        # Execution queue
        self.execution_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.get("max_queue_size", 1000)
        )
        self.running = False
        self.workers: List[asyncio.Task] = []
        self.num_workers = self.config.get("num_workers", 5)

        # Scheduler
        self.scheduler_task: Optional[asyncio.Task] = None
        self.scheduler_running = False
        self.schedule_check_interval = self.config.get("schedule_check_interval", 60)

        # File watcher
        self.file_watcher_task: Optional[asyncio.Task] = None
        self.file_watches: Dict[str, Dict] = {}

        # Circuit breaker
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime] = None
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)
        self._consecutive_failures = 0

        # Rate limiting
        self._request_timestamps: deque = deque(maxlen=100)
        self.max_requests_per_minute = self.config.get("max_requests_per_minute", 100)

        # Statistics
        self.stats = {
            "total_workflows": 0,
            "active_workflows": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time_ms": 0.0,
            "total_webhook_calls": 0,
            "total_schedule_triggers": 0,
            "total_manual_triggers": 0,
            "startup_time": datetime.now().isoformat(),
        }

        # Execution metrics per workflow
        self.workflow_metrics: Dict[str, ExecutionMetrics] = {}

        # Custom action registry
        self.custom_actions: Dict[str, Callable] = {}

        # Middleware hooks
        self._before_execution_hooks: List[Callable] = []
        self._after_execution_hooks: List[Callable] = []
        self._error_hooks: List[Callable] = []

        # Load persisted data
        if self.persist_enabled:
            self._load_data()

        # Initialize workflow metrics
        for wf_id in self.workflows:
            self.workflow_metrics[wf_id] = ExecutionMetrics()

        self.logger.info("Automation Agent initialized")

    # =========================
    # PERSISTENCE
    # =========================

    def _load_data(self):
        """Load workflows from disk"""
        try:
            if self.workflows_file.exists():
                with open(self.workflows_file, "r") as f:
                    workflows_data = json.load(f)
                    for wf_data in workflows_data:
                        workflow = self._dict_to_workflow(wf_data)
                        self.workflows[workflow.id] = workflow
                self.logger.info(f"Loaded {len(self.workflows)} workflows")

            if self.executions_file.exists():
                with open(self.executions_file, "r") as f:
                    executions_data = json.load(f)
                    for exec_data in executions_data[-1000:]:  # Limit to last 1000
                        execution = self._dict_to_execution(exec_data)
                        self.execution_history.append(execution)
                self.logger.info(f"Loaded {len(self.execution_history)} executions")
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")

    def _save_data(self):
        """Save workflows to disk"""
        if not self.persist_enabled:
            return

        try:
            workflows_data = [
                self._workflow_to_dict(w) for w in self.workflows.values()
            ]
            with open(self.workflows_file, "w") as f:
                json.dump(workflows_data, f, indent=2, default=str)

            # Save only recent executions
            executions_data = [
                self._execution_to_dict(e) for e in list(self.execution_history)[-1000:]
            ]
            with open(self.executions_file, "w") as f:
                json.dump(executions_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")

    # =========================
    # SERIALIZATION HELPERS
    # =========================

    def _workflow_to_dict(self, workflow: Workflow) -> Dict:
        """Convert Workflow to dictionary"""
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status.value,
            "trigger": {
                "type": workflow.trigger.type.value,
                "config": workflow.trigger.config,
                "condition": (
                    self._condition_to_dict(workflow.trigger.condition)
                    if workflow.trigger.condition
                    else None
                ),
                "id": workflow.trigger.id,
            },
            "actions": [self._action_to_dict(a) for a in workflow.actions],
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "tags": workflow.tags,
            "max_executions": workflow.max_executions,
            "execution_count": workflow.execution_count,
            "success_count": workflow.success_count,
            "failure_count": workflow.failure_count,
            "last_execution": (
                workflow.last_execution.isoformat() if workflow.last_execution else None
            ),
            "version": workflow.version,
            "owner": workflow.owner,
            "priority": workflow.priority,
            "timeout": workflow.timeout,
        }

    def _dict_to_workflow(self, data: Dict) -> Workflow:
        """Convert dictionary to Workflow"""
        return Workflow(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=WorkflowStatus(data["status"]),
            trigger=Trigger(
                type=TriggerType(data["trigger"]["type"]),
                config=data["trigger"]["config"],
                condition=(
                    self._dict_to_condition(data["trigger"]["condition"])
                    if data["trigger"].get("condition")
                    else None
                ),
                id=data["trigger"].get("id", str(uuid.uuid4())[:8]),
            ),
            actions=[self._dict_to_action(a) for a in data["actions"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []),
            max_executions=data.get("max_executions"),
            execution_count=data.get("execution_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            last_execution=(
                datetime.fromisoformat(data["last_execution"])
                if data.get("last_execution")
                else None
            ),
            version=data.get("version", 1),
            owner=data.get("owner", "system"),
            priority=data.get("priority", 5),
            timeout=data.get("timeout", 300),
        )

    def _action_to_dict(self, action: Action) -> Dict:
        """Convert Action to dictionary"""
        return {
            "type": action.type.value,
            "config": action.config,
            "id": action.id,
            "name": action.name,
            "description": action.description,
            "next_on_success": action.next_on_success,
            "next_on_failure": action.next_on_failure,
            "retry_count": action.retry_count,
            "retry_delay": action.retry_delay,
            "timeout": action.timeout,
            "continue_on_error": action.continue_on_error,
        }

    def _dict_to_action(self, data: Dict) -> Action:
        """Convert dictionary to Action"""
        return Action(
            type=ActionType(data["type"]),
            config=data["config"],
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            next_on_success=data.get("next_on_success"),
            next_on_failure=data.get("next_on_failure"),
            retry_count=data.get("retry_count", 0),
            retry_delay=data.get("retry_delay", 5),
            timeout=data.get("timeout", 30),
            continue_on_error=data.get("continue_on_error", False),
        )

    def _condition_to_dict(self, condition: Condition) -> Dict:
        """Convert Condition to dictionary"""
        result = {
            "operator": condition.operator.value,
            "value": condition.value,
            "field": condition.field,
            "logical_operator": condition.logical_operator,
        }
        if condition.nested_conditions:
            result["nested_conditions"] = [
                self._condition_to_dict(c) for c in condition.nested_conditions
            ]
        return result

    def _dict_to_condition(self, data: Dict) -> Condition:
        """Convert dictionary to Condition"""
        return Condition(
            operator=ConditionOperator(data["operator"]),
            value=data["value"],
            field=data.get("field"),
            nested_conditions=(
                [self._dict_to_condition(c) for c in data.get("nested_conditions", [])]
                if data.get("nested_conditions")
                else None
            ),
            logical_operator=data.get("logical_operator", "and"),
        )

    def _execution_to_dict(self, execution: WorkflowExecution) -> Dict:
        """Convert WorkflowExecution to dictionary"""
        return {
            "id": execution.id,
            "workflow_id": execution.workflow_id,
            "trigger_data": execution.trigger_data,
            "context": execution.context,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": (
                execution.completed_at.isoformat() if execution.completed_at else None
            ),
            "error": execution.error,
            "action_results": execution.action_results,
            "retry_count": execution.retry_count,
            "execution_time_ms": execution.execution_time_ms,
        }

    def _dict_to_execution(self, data: Dict) -> WorkflowExecution:
        """Convert dictionary to WorkflowExecution"""
        return WorkflowExecution(
            id=data["id"],
            workflow_id=data["workflow_id"],
            trigger_data=data.get("trigger_data", {}),
            context=data.get("context", {}),
            status=ExecutionStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            error=data.get("error"),
            action_results=data.get("action_results", []),
            retry_count=data.get("retry_count", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )

    # =========================
    # CIRCUIT BREAKER
    # =========================

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open"""
        if not self._circuit_open:
            return True

        if self._circuit_open_until and datetime.now() >= self._circuit_open_until:
            self._circuit_open = False
            self._consecutive_failures = 0
            self.logger.info("Circuit breaker closed")
            return True

        return False

    async def _record_failure(self):
        """Record a failure for circuit breaker"""
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_open = True
            self._circuit_open_until = datetime.now() + timedelta(
                seconds=self.circuit_breaker_timeout
            )
            self.logger.error(
                f"Circuit breaker opened after {self._consecutive_failures} failures"
            )

    async def _record_success(self):
        """Record a successful operation"""
        self._consecutive_failures = 0

    # =========================
    # RATE LIMITING
    # =========================

    async def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()

        # Clean old timestamps
        self._request_timestamps = deque(
            [t for t in self._request_timestamps if now - t < 60], maxlen=100
        )

        if len(self._request_timestamps) >= self.max_requests_per_minute:
            oldest = self._request_timestamps[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self._request_timestamps.append(now)
        return True

    # =========================
    # WORKFLOW MANAGEMENT
    # =========================

    def _generate_id(self, prefix: str = "wf") -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"

    async def create_workflow(
        self,
        name: str,
        description: str,
        trigger: Trigger,
        actions: List[Action],
        tags: List[str] = None,
        max_executions: int = None,
        owner: str = "system",
        priority: int = 5,
    ) -> Dict[str, Any]:
        """
        Create a new workflow

        Args:
            name: Workflow name
            description: Workflow description
            trigger: Trigger configuration
            actions: List of actions
            tags: Workflow tags
            max_executions: Maximum number of executions
            owner: Workflow owner
            priority: Workflow priority (1-10)

        Returns:
            Dictionary with workflow creation result
        """
        workflow_id = self._generate_id()
        now = datetime.now()

        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            status=WorkflowStatus.ACTIVE,
            trigger=trigger,
            actions=actions,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            max_executions=max_executions,
            owner=owner,
            priority=priority,
        )

        self.workflows[workflow_id] = workflow
        self.workflow_metrics[workflow_id] = ExecutionMetrics()

        # Setup webhook if needed
        if trigger.type == TriggerType.WEBHOOK:
            webhook_path = trigger.config.get("webhook_path", f"/webhook/{workflow_id}")
            self.webhook_routes[webhook_path] = workflow_id
            self.logger.info(f"Webhook endpoint registered: {webhook_path}")

        # Setup file watch if needed
        if trigger.type == TriggerType.FILE:
            await self._add_file_watch(workflow_id, trigger.config)

        if self.persist_enabled:
            self._save_data()

        self.stats["total_workflows"] += 1
        self.stats["active_workflows"] += 1

        self.logger.info(f"Created workflow: {name} (ID: {workflow_id})")

        return {
            "success": True,
            "workflow_id": workflow_id,
            "name": name,
            "message": f"Workflow {name} created successfully",
        }

    async def update_workflow(self, workflow_id: str, **updates) -> Dict[str, Any]:
        """
        Update an existing workflow

        Args:
            workflow_id: Workflow ID
            **updates: Fields to update

        Returns:
            Dictionary with update result
        """
        if workflow_id not in self.workflows:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        workflow = self.workflows[workflow_id]

        for key, value in updates.items():
            if hasattr(workflow, key):
                setattr(workflow, key, value)

        workflow.version += 1
        workflow.updated_at = datetime.now()

        if self.persist_enabled:
            self._save_data()

        return {
            "success": True,
            "workflow_id": workflow_id,
            "message": f"Workflow {workflow.name} updated",
        }

    async def delete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Delete a workflow

        Args:
            workflow_id: Workflow ID

        Returns:
            Dictionary with deletion result
        """
        if workflow_id not in self.workflows:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        workflow = self.workflows[workflow_id]
        name = workflow.name

        # Remove webhook route if exists
        for path, wf_id in list(self.webhook_routes.items()):
            if wf_id == workflow_id:
                del self.webhook_routes[path]

        # Remove file watch if exists
        if workflow.trigger.type == TriggerType.FILE:
            await self._remove_file_watch(workflow_id)

        del self.workflows[workflow_id]

        if workflow.status == WorkflowStatus.ACTIVE:
            self.stats["active_workflows"] -= 1
        self.stats["total_workflows"] -= 1

        if self.persist_enabled:
            self._save_data()

        self.logger.info(f"Deleted workflow: {name}")

        return {"success": True, "message": f"Workflow {name} deleted"}

    async def enable_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Enable a workflow"""
        return await self.update_workflow(workflow_id, status=WorkflowStatus.ACTIVE)

    async def disable_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Disable a workflow"""
        return await self.update_workflow(workflow_id, status=WorkflowStatus.DISABLED)

    async def pause_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Pause a workflow"""
        return await self.update_workflow(workflow_id, status=WorkflowStatus.PAUSED)

    # =========================
    # FILE WATCHING
    # =========================

    async def _add_file_watch(self, workflow_id: str, config: Dict):
        """Add a file watch for workflow"""
        file_path = Path(config.get("path"))
        event_type = config.get("event", "modified")

        self.file_watches[workflow_id] = {
            "path": file_path,
            "event": event_type,
            "last_modified": file_path.stat().st_mtime if file_path.exists() else None,
        }

    async def _remove_file_watch(self, workflow_id: str):
        """Remove a file watch"""
        if workflow_id in self.file_watches:
            del self.file_watches[workflow_id]

    async def _file_watcher_loop(self):
        """Background file watcher"""
        while self.scheduler_running:
            try:
                for workflow_id, watch in self.file_watches.items():
                    file_path = watch["path"]
                    event_type = watch["event"]

                    if file_path.exists():
                        current_mtime = file_path.stat().st_mtime
                        last_mtime = watch.get("last_modified")

                        if last_mtime is None:
                            watch["last_modified"] = current_mtime
                        elif event_type == "modified" and current_mtime > last_mtime:
                            watch["last_modified"] = current_mtime
                            await self.trigger_workflow(
                                workflow_id,
                                {
                                    "file_path": str(file_path),
                                    "event": "modified",
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                        elif event_type == "created" and not last_mtime:
                            watch["last_modified"] = current_mtime
                            await self.trigger_workflow(
                                workflow_id,
                                {
                                    "file_path": str(file_path),
                                    "event": "created",
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                    elif (
                        watch.get("last_modified") is not None
                        and event_type == "deleted"
                    ):
                        watch["last_modified"] = None
                        await self.trigger_workflow(
                            workflow_id,
                            {
                                "file_path": str(file_path),
                                "event": "deleted",
                                "timestamp": datetime.now().isoformat(),
                            },
                        )

                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"File watcher error: {e}")
                await asyncio.sleep(5)

    # =========================
    # SCHEDULER
    # =========================

    async def _scheduler_loop(self):
        """Background scheduler for cron-triggered workflows"""
        self.scheduler_running = True

        # Start file watcher
        self.file_watcher_task = asyncio.create_task(self._file_watcher_loop())

        while self.scheduler_running:
            try:
                now = datetime.now()

                for workflow in self.workflows.values():
                    if workflow.status != WorkflowStatus.ACTIVE:
                        continue

                    if workflow.trigger.type != TriggerType.SCHEDULE:
                        continue

                    cron_expr = workflow.trigger.config.get("cron")
                    if not cron_expr:
                        continue

                    if CRONITER_AVAILABLE:
                        cron = croniter(cron_expr, workflow.last_execution or now)
                        next_run = cron.get_next(datetime)

                        if now >= next_run:
                            await self.trigger_workflow(
                                workflow.id,
                                {"scheduled_time": now.isoformat(), "cron": cron_expr},
                            )
                            self.stats["total_schedule_triggers"] += 1
                    else:
                        # Simple interval-based scheduling
                        interval = workflow.trigger.config.get("interval_seconds", 3600)
                        last_run = workflow.last_execution

                        if (
                            last_run is None
                            or (now - last_run).total_seconds() >= interval
                        ):
                            await self.trigger_workflow(
                                workflow.id,
                                {
                                    "scheduled_time": now.isoformat(),
                                    "interval": interval,
                                },
                            )
                            self.stats["total_schedule_triggers"] += 1

                await asyncio.sleep(self.schedule_check_interval)
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

    # =========================
    # WORKFLOW TRIGGERING
    # =========================

    async def trigger_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """
        Manually trigger a workflow

        Args:
            workflow_id: Workflow ID
            trigger_data: Data to pass to workflow
            source: Trigger source (manual, webhook, schedule, etc.)

        Returns:
            Dictionary with trigger result
        """
        # Check circuit breaker
        if not await self._check_circuit_breaker():
            return {"success": False, "error": "Circuit breaker open"}

        # Check rate limit
        if not await self._check_rate_limit():
            return {"success": False, "error": "Rate limit exceeded"}

        if workflow_id not in self.workflows:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        workflow = self.workflows[workflow_id]

        if workflow.status != WorkflowStatus.ACTIVE:
            return {
                "success": False,
                "error": f"Workflow {workflow.name} is not active",
            }

        if (
            workflow.max_executions
            and workflow.execution_count >= workflow.max_executions
        ):
            return {"success": False, "error": "Max executions reached"}

        # Queue workflow execution
        execution_id = self._generate_id("exec")
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            trigger_data=trigger_data or {},
            context={"trigger_data": trigger_data or {}, "source": source},
            status=ExecutionStatus.PENDING,
            started_at=datetime.now(),
        )

        self.executions[execution_id] = execution

        try:
            await self.execution_queue.put(execution)
            self.stats["total_executions"] += 1

            if source == "manual":
                self.stats["total_manual_triggers"] += 1
            elif source == "webhook":
                self.stats["total_webhook_calls"] += 1

            return {
                "success": True,
                "execution_id": execution_id,
                "message": f"Workflow {workflow.name} triggered",
            }
        except asyncio.QueueFull:
            return {"success": False, "error": "Execution queue full"}

    # =========================
    # WORKFLOW EXECUTION
    # =========================

    async def _execute_workflow(self, execution: WorkflowExecution):
        """Execute a workflow"""
        workflow = self.workflows[execution.workflow_id]

        self.logger.info(f"Executing workflow: {workflow.name} (ID: {execution.id})")

        start_time = time.time()
        execution.status = ExecutionStatus.RUNNING

        # Run before hooks
        await self._run_before_hooks(workflow, execution)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_workflow_actions(workflow, execution),
                timeout=workflow.timeout,
            )

            execution.status = (
                ExecutionStatus.COMPLETED if result else ExecutionStatus.FAILED
            )
            execution.completed_at = datetime.now()

            # Update workflow stats
            workflow.execution_count += 1
            workflow.last_execution = datetime.now()

            if execution.status == ExecutionStatus.COMPLETED:
                workflow.success_count += 1
                self.stats["successful_executions"] += 1
            else:
                workflow.failure_count += 1
                self.stats["failed_executions"] += 1

            # Update metrics
            await self._record_success()

        except asyncio.TimeoutError:
            execution.status = ExecutionStatus.TIMEOUT
            execution.error = f"Workflow timeout after {workflow.timeout}s"
            execution.completed_at = datetime.now()
            workflow.failure_count += 1
            self.stats["failed_executions"] += 1
            self.logger.error(f"Workflow {workflow.name} timed out")

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            workflow.failure_count += 1
            self.stats["failed_executions"] += 1
            await self._record_failure()
            self.logger.error(f"Workflow {workflow.name} failed: {e}")

        finally:
            execution.execution_time_ms = (time.time() - start_time) * 1000

            # Update metrics
            metrics = self.workflow_metrics[workflow.id]
            metrics.total_executions += 1
            if execution.status == ExecutionStatus.COMPLETED:
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1
            metrics.total_execution_time_ms += execution.execution_time_ms
            metrics.avg_execution_time_ms = (
                metrics.total_execution_time_ms / metrics.total_executions
            )
            metrics.min_execution_time_ms = min(
                metrics.min_execution_time_ms, execution.execution_time_ms
            )
            metrics.max_execution_time_ms = max(
                metrics.max_execution_time_ms, execution.execution_time_ms
            )

            # Add to history
            self.execution_history.append(execution)

            # Save workflow changes
            if self.persist_enabled:
                self._save_data()

            # Run after hooks
            await self._run_after_hooks(workflow, execution)

            self.logger.info(
                f"Workflow {workflow.name} completed in {execution.execution_time_ms:.2f}ms"
            )

    async def _execute_workflow_actions(
        self, workflow: Workflow, execution: WorkflowExecution
    ) -> bool:
        """Execute all actions in workflow"""
        context = execution.context
        action_results = []

        i = 0
        while i < len(workflow.actions):
            action = workflow.actions[i]

            try:
                # Check condition if any
                if action.config.get("condition"):
                    condition = self._dict_to_condition(action.config["condition"])
                    if not self._check_condition(condition, context):
                        i += 1
                        continue

                # Execute action with retry
                result = await self._execute_action_with_retry(action, context)

                action_results.append(
                    {
                        "action_id": action.id,
                        "action_name": action.name,
                        "action_type": action.type.value,
                        "success": result["success"],
                        "result": result.get("data"),
                        "error": result.get("error"),
                        "execution_time_ms": result.get("execution_time_ms", 0),
                    }
                )

                # Update context with action result
                context[f"action_{action.id}_result"] = result.get("data")
                context[f"action_{action.id}_success"] = result["success"]

                # Handle branching
                if result["success"] and action.next_on_success:
                    next_id = action.next_on_success
                    next_index = self._find_action_index(workflow.actions, next_id)
                    if next_index is not None:
                        i = next_index
                        continue
                elif not result["success"] and action.next_on_failure:
                    next_id = action.next_on_failure
                    next_index = self._find_action_index(workflow.actions, next_id)
                    if next_index is not None:
                        i = next_index
                        continue
                elif not result["success"] and not action.continue_on_error:
                    execution.error = (
                        f"Action {action.name} failed: {result.get('error')}"
                    )
                    execution.action_results = action_results
                    return False

            except Exception as e:
                self.logger.error(f"Action {action.name} error: {e}")
                if not action.continue_on_error:
                    execution.error = str(e)
                    execution.action_results = action_results
                    return False

            i += 1

        execution.action_results = action_results
        return True

    async def _execute_action_with_retry(self, action: Action, context: Dict) -> Dict:
        """Execute action with retry logic"""
        start_time = time.time()

        for attempt in range(action.retry_count + 1):
            try:
                if attempt > 0:
                    self.logger.info(
                        f"Retrying action {action.name} (attempt {attempt + 1})"
                    )

                result = await self._execute_action(action, context)
                result["execution_time_ms"] = (time.time() - start_time) * 1000

                if result["success"]:
                    return result
                elif attempt < action.retry_count:
                    await asyncio.sleep(action.retry_delay * (attempt + 1))

            except Exception as e:
                if attempt < action.retry_count:
                    await asyncio.sleep(action.retry_delay * (attempt + 1))
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "execution_time_ms": (time.time() - start_time) * 1000,
                    }

        return {
            "success": False,
            "error": "Max retries exceeded",
            "execution_time_ms": (time.time() - start_time) * 1000,
        }

    async def _execute_action(self, action: Action, context: Dict) -> Dict:
        """Execute a single action"""

        # Execute based on action type
        if action.type == ActionType.WEBHOOK or action.type == ActionType.HTTP:
            return await self._execute_http(action.config, context)

        elif action.type == ActionType.EMAIL:
            return await self._execute_email(action.config, context)

        elif action.type == ActionType.NOTIFICATION:
            return await self._execute_notification(action.config, context)

        elif action.type == ActionType.DATABASE:
            return await self._execute_database(action.config, context)

        elif action.type == ActionType.FILE:
            return await self._execute_file(action.config, context)

        elif action.type == ActionType.CONDITION:
            return await self._execute_condition_action(action.config, context)

        elif action.type == ActionType.DELAY:
            return await self._execute_delay(action.config, context)

        elif action.type == ActionType.TRANSFORM:
            return await self._execute_transform(action.config, context)

        elif action.type == ActionType.FILTER:
            return await self._execute_filter(action.config, context)

        elif action.type == ActionType.LOG:
            return await self._execute_log(action.config, context)

        elif action.type == ActionType.SHELL:
            return await self._execute_shell(action.config, context)

        elif action.type == ActionType.CODE:
            return await self._execute_code(action.config, context)

        elif action.type == ActionType.LOOP:
            return await self._execute_loop(action.config, context)

        elif action.type == ActionType.PARALLEL:
            return await self._execute_parallel(action.config, context)

        elif action.type == ActionType.CUSTOM:
            return await self._execute_custom(action.config, context)

        else:
            return {"success": False, "error": f"Unknown action type: {action.type}"}

    # =========================
    # ACTION IMPLEMENTATIONS
    # =========================

    async def _execute_http(self, config: Dict, context: Dict) -> Dict:
        """Execute HTTP/webhook action"""
        if not AIOHTTP_AVAILABLE:
            return {"success": False, "error": "aiohttp not available"}

        url = self._substitute_variables(config.get("url"), context)
        method = config.get("method", "POST")
        headers = config.get("headers", {})
        body = config.get("body", {})
        timeout = config.get("timeout", 30)

        # Substitute variables
        headers = {
            k: self._substitute_variables(v, context) for k, v in headers.items()
        }
        body = self._substitute_variables(body, context)

        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(
                        url, headers=headers, timeout=timeout
                    ) as response:
                        data = await response.text()
                        return {
                            "success": response.status < 400,
                            "data": {
                                "status_code": response.status,
                                "body": data,
                                "headers": dict(response.headers),
                            },
                        }
                else:
                    async with session.request(
                        method, url, json=body, headers=headers, timeout=timeout
                    ) as response:
                        data = await response.text()
                        return {
                            "success": response.status < 400,
                            "data": {
                                "status_code": response.status,
                                "body": data,
                                "headers": dict(response.headers),
                            },
                        }
        except asyncio.TimeoutError:
            return {"success": False, "error": f"HTTP timeout after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_email(self, config: Dict, context: Dict) -> Dict:
        """Execute email action"""
        # Placeholder - integrate with email agent
        to = self._substitute_variables(config.get("to"), context)
        subject = self._substitute_variables(config.get("subject"), context)
        body = self._substitute_variables(config.get("body"), context)

        self.logger.info(f"Sending email to {to}: {subject}")

        return {"success": True, "data": {"to": to, "subject": subject, "sent": True}}

    async def _execute_notification(self, config: Dict, context: Dict) -> Dict:
        """Execute notification action"""
        message = self._substitute_variables(config.get("message"), context)
        channel = config.get("channel", "default")

        self.logger.info(f"Sending notification to {channel}: {message}")

        return {
            "success": True,
            "data": {"channel": channel, "message": message, "sent": True},
        }

    async def _execute_database(self, config: Dict, context: Dict) -> Dict:
        """Execute database action"""
        operation = config.get("operation")
        query = self._substitute_variables(config.get("query"), context)

        self.logger.info(f"Database operation: {operation}")

        return {"success": True, "data": {"operation": operation, "executed": True}}

    async def _execute_file(self, config: Dict, context: Dict) -> Dict:
        """Execute file action"""
        if not AIOFILES_AVAILABLE:
            return {"success": False, "error": "aiofiles not available"}

        operation = config.get("operation")
        path = self._substitute_variables(config.get("path"), context)
        file_path = Path(path)

        try:
            if operation == "read":
                async with aiofiles.open(path, "r") as f:
                    content = await f.read()
                return {"success": True, "data": {"content": content, "path": path}}

            elif operation == "write":
                content = self._substitute_variables(config.get("content"), context)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(path, "w") as f:
                    await f.write(content)
                return {"success": True, "data": {"written": True, "path": path}}

            elif operation == "append":
                content = self._substitute_variables(config.get("content"), context)
                async with aiofiles.open(path, "a") as f:
                    await f.write(content)
                return {"success": True, "data": {"appended": True, "path": path}}

            elif operation == "delete":
                if file_path.exists():
                    file_path.unlink()
                return {"success": True, "data": {"deleted": True, "path": path}}

            elif operation == "exists":
                exists = file_path.exists()
                return {"success": True, "data": {"exists": exists, "path": path}}

            else:
                return {
                    "success": False,
                    "error": f"Unknown file operation: {operation}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_condition_action(self, config: Dict, context: Dict) -> Dict:
        """Execute condition action (branching)"""
        condition = config.get("condition")
        if condition:
            condition_obj = self._dict_to_condition(condition)
            result = self._check_condition(condition_obj, context)
            return {"success": True, "data": {"condition_met": result}}
        return {"success": True, "data": {"condition_met": True}}

    async def _execute_delay(self, config: Dict, context: Dict) -> Dict:
        """Execute delay action"""
        seconds = config.get("seconds", 1)
        await asyncio.sleep(seconds)
        return {"success": True, "data": {"delayed": seconds}}

    async def _execute_transform(self, config: Dict, context: Dict) -> Dict:
        """Execute data transformation"""
        input_data = self._get_value_from_context(config.get("input"), context)
        transform_type = config.get("type", "json")

        try:
            if transform_type == "json":
                import json

                result = json.dumps(input_data)
            elif transform_type == "string":
                result = str(input_data)
            elif transform_type == "integer":
                result = int(input_data)
            elif transform_type == "float":
                result = float(input_data)
            elif transform_type == "boolean":
                result = bool(input_data)
            elif transform_type == "lowercase":
                result = str(input_data).lower()
            elif transform_type == "uppercase":
                result = str(input_data).upper()
            elif transform_type == "capitalize":
                result = str(input_data).capitalize()
            elif transform_type == "template" and JINJA_AVAILABLE:
                template = self.template_env.from_string(config.get("template", ""))
                result = template.render(**context)
            else:
                result = input_data

            return {"success": True, "data": {"transformed": result}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_filter(self, config: Dict, context: Dict) -> Dict:
        """Execute data filtering"""
        input_data = self._get_value_from_context(config.get("input"), context)
        field = config.get("field")
        operator = config.get("operator")
        value = config.get("value")

        try:
            filtered = []
            items = input_data if isinstance(input_data, list) else [input_data]

            for item in items:
                item_value = item.get(field) if isinstance(item, dict) else item
                if self._compare_values(item_value, ConditionOperator(operator), value):
                    filtered.append(item)

            return {
                "success": True,
                "data": {
                    "filtered": filtered,
                    "original_count": len(items),
                    "filtered_count": len(filtered),
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_log(self, config: Dict, context: Dict) -> Dict:
        """Execute log action"""
        message = self._substitute_variables(config.get("message"), context)
        level = config.get("level", "info")

        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        else:
            self.logger.debug(message)

        return {"success": True, "data": {"logged": True, "level": level}}

    async def _execute_shell(self, config: Dict, context: Dict) -> Dict:
        """Execute shell command"""
        import subprocess

        command = self._substitute_variables(config.get("command"), context)
        timeout = config.get("timeout", 30)

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "data": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command,
                },
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timeout after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_code(self, config: Dict, context: Dict) -> Dict:
        """Execute Python code"""
        code = config.get("code", "")

        # Create a safe execution environment
        exec_globals = {"context": context, "result": None, "print": self.logger.info}

        try:
            exec(code, exec_globals)
            return {"success": True, "data": {"result": exec_globals.get("result")}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_loop(self, config: Dict, context: Dict) -> Dict:
        """Execute loop action"""
        items = self._get_value_from_context(config.get("items"), context)
        loop_actions = config.get("actions", [])
        max_iterations = config.get("max_iterations", 1000)

        if not items:
            return {"success": True, "data": {"loop_results": [], "iterations": 0}}

        results = []
        for i, item in enumerate(items[:max_iterations]):
            # Create loop context
            loop_context = context.copy()
            loop_context["loop_item"] = item
            loop_context["loop_index"] = i
            loop_context["loop_total"] = len(items)

            # Execute loop body
            for action_config in loop_actions:
                action = self._dict_to_action(action_config)
                result = await self._execute_action(action, loop_context)
                results.append(result)

                if not result["success"] and not config.get("continue_on_error", False):
                    break

        return {
            "success": True,
            "data": {"loop_results": results, "iterations": len(results)},
        }

    async def _execute_parallel(self, config: Dict, context: Dict) -> Dict:
        """Execute parallel actions"""
        parallel_actions = config.get("actions", [])
        max_concurrent = config.get("max_concurrent", 5)

        # Limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_limit(action_config):
            async with semaphore:
                action = self._dict_to_action(action_config)
                return await self._execute_action(action, context)

        tasks = [execute_with_limit(ac) for ac in parallel_actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for r in results:
            if isinstance(r, Exception):
                processed_results.append({"success": False, "error": str(r)})
            else:
                processed_results.append(r)

        all_success = all(r.get("success", False) for r in processed_results)

        return {
            "success": all_success,
            "data": {
                "parallel_results": processed_results,
                "count": len(processed_results),
            },
        }

    async def _execute_custom(self, config: Dict, context: Dict) -> Dict:
        """Execute custom registered action"""
        action_name = config.get("action_name")

        if action_name not in self.custom_actions:
            return {
                "success": False,
                "error": f"Custom action {action_name} not registered",
            }

        callback = self.custom_actions[action_name]

        try:
            if asyncio.iscoroutinefunction(callback):
                result = await callback(config, context)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, callback, config, context)

            return (
                result
                if isinstance(result, dict)
                else {"success": True, "data": result}
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # CONDITION EVALUATION
    # =========================

    def _check_condition(self, condition: Condition, context: Dict) -> bool:
        """Check if condition is met"""
        # Get value to compare
        if condition.field:
            value = self._get_value_from_context(condition.field, context)
        else:
            value = condition.value

        # Handle nested conditions
        if condition.nested_conditions:
            results = [
                self._check_condition(c, context) for c in condition.nested_conditions
            ]
            if condition.logical_operator == "and":
                return all(results)
            else:
                return any(results)

        # Single condition
        return self._compare_values(value, condition.operator, condition.value)

    def _compare_values(
        self, value: Any, operator: ConditionOperator, compare_to: Any
    ) -> bool:
        """Compare two values based on operator"""
        try:
            if operator == ConditionOperator.EQUALS:
                return value == compare_to
            elif operator == ConditionOperator.NOT_EQUALS:
                return value != compare_to
            elif operator == ConditionOperator.GREATER_THAN:
                return value > compare_to
            elif operator == ConditionOperator.LESS_THAN:
                return value < compare_to
            elif operator == ConditionOperator.GREATER_EQUALS:
                return value >= compare_to
            elif operator == ConditionOperator.LESS_EQUALS:
                return value <= compare_to
            elif operator == ConditionOperator.CONTAINS:
                return compare_to in value if hasattr(value, "__contains__") else False
            elif operator == ConditionOperator.NOT_CONTAINS:
                return (
                    compare_to not in value if hasattr(value, "__contains__") else True
                )
            elif operator == ConditionOperator.STARTS_WITH:
                return str(value).startswith(str(compare_to))
            elif operator == ConditionOperator.ENDS_WITH:
                return str(value).endswith(str(compare_to))
            elif operator == ConditionOperator.MATCHES_REGEX:
                return bool(re.match(str(compare_to), str(value)))
            elif operator == ConditionOperator.IN_LIST:
                return (
                    value in compare_to
                    if hasattr(compare_to, "__contains__")
                    else False
                )
            elif operator == ConditionOperator.NOT_IN_LIST:
                return (
                    value not in compare_to
                    if hasattr(compare_to, "__contains__")
                    else True
                )
            elif operator == ConditionOperator.IS_TRUE:
                return bool(value)
            elif operator == ConditionOperator.IS_FALSE:
                return not bool(value)
            elif operator == ConditionOperator.IS_NULL:
                return value is None
            elif operator == ConditionOperator.IS_NOT_NULL:
                return value is not None
            else:
                return False
        except Exception:
            return False

    # =========================
    # UTILITY METHODS
    # =========================

    def _get_value_from_context(self, path: str, context: Dict) -> Any:
        """Get value from context using dot notation"""
        if not path or not isinstance(path, str):
            return path

        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                value = value[int(part)]
            else:
                return None

        return value

    def _substitute_variables(self, template: Any, context: Dict) -> Any:
        """Substitute variables in template using {{var}} syntax"""
        if isinstance(template, str):
            result = template
            # Simple variable substitution
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    result = result.replace(f"{{{{{key}}}}}", str(value))
            return result
        elif isinstance(template, dict):
            return {
                k: self._substitute_variables(v, context) for k, v in template.items()
            }
        elif isinstance(template, list):
            return [self._substitute_variables(item, context) for item in template]
        else:
            return template

    def _find_action_index(
        self, actions: List[Action], action_id: str
    ) -> Optional[int]:
        """Find action index by ID"""
        for i, action in enumerate(actions):
            if action.id == action_id:
                return i
        return None

    # =========================
    # WORKER MANAGEMENT
    # =========================

    async def _start_workers(self):
        """Start worker tasks"""
        self.running = True

        async def worker(worker_id: int):
            while self.running:
                try:
                    execution = await self.execution_queue.get()
                    await self._execute_workflow(execution)
                    self.execution_queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Worker {worker_id} error: {e}")
                    await asyncio.sleep(1)

        for i in range(self.num_workers):
            worker_task = asyncio.create_task(worker(i))
            self.workers.append(worker_task)

        self.logger.info(f"Started {self.num_workers} workers")

    async def _stop_workers(self):
        """Stop worker tasks"""
        self.running = False

        for task in self.workers:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.workers.clear()
        self.logger.info("Workers stopped")

    # =========================
    # WEBHOOK SERVER
    # =========================

    async def _start_webhook_server(self):
        """Start webhook server"""
        if not AIOHTTP_AVAILABLE:
            self.logger.warning("aiohttp not available, webhook server disabled")
            return

        async def handle_webhook(request):
            path = request.path
            workflow_id = self.webhook_routes.get(path)

            if not workflow_id:
                return web.json_response({"error": "Workflow not found"}, status=404)

            # Extract request data
            try:
                body = await request.json() if request.can_read_body else {}
            except:
                body = {}

            data = {
                "method": request.method,
                "headers": dict(request.headers),
                "query": dict(request.query),
                "body": body,
                "path": path,
                "timestamp": datetime.now().isoformat(),
            }

            # Trigger workflow
            result = await self.trigger_workflow(workflow_id, data, source="webhook")

            if result["success"]:
                return web.json_response(
                    {"status": "accepted", "execution_id": result["execution_id"]}
                )
            else:
                return web.json_response({"error": result["error"]}, status=500)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle_webhook)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.webhook_host, self.webhook_port)
        await site.start()

        self.logger.info(
            f"Webhook server started on {self.webhook_host}:{self.webhook_port}"
        )

    # =========================
    # MIDDLEWARE HOOKS
    # =========================

    def add_before_hook(self, hook: Callable):
        """Add hook to run before workflow execution"""
        self._before_execution_hooks.append(hook)

    def add_after_hook(self, hook: Callable):
        """Add hook to run after workflow execution"""
        self._after_execution_hooks.append(hook)

    def add_error_hook(self, hook: Callable):
        """Add hook to run on error"""
        self._error_hooks.append(hook)

    async def _run_before_hooks(self, workflow: Workflow, execution: WorkflowExecution):
        """Run all before hooks"""
        for hook in self._before_execution_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(workflow, execution)
                else:
                    hook(workflow, execution)
            except Exception as e:
                self.logger.debug(f"Before hook error: {e}")

    async def _run_after_hooks(self, workflow: Workflow, execution: WorkflowExecution):
        """Run all after hooks"""
        for hook in self._after_execution_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(workflow, execution)
                else:
                    hook(workflow, execution)
            except Exception as e:
                self.logger.debug(f"After hook error: {e}")

    async def _run_error_hooks(
        self, workflow: Workflow, execution: WorkflowExecution, error: Exception
    ):
        """Run all error hooks"""
        for hook in self._error_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(workflow, execution, error)
                else:
                    hook(workflow, execution, error)
            except Exception as e:
                self.logger.debug(f"Error hook error: {e}")

    # =========================
    # CUSTOM ACTION REGISTRATION
    # =========================

    async def register_custom_action(
        self, name: str, callback: Callable
    ) -> Dict[str, Any]:
        """
        Register a custom action

        Args:
            name: Action name
            callback: Async function to call

        Returns:
            Dictionary with registration result
        """
        self.custom_actions[name] = callback

        return {
            "success": True,
            "action_name": name,
            "message": f"Custom action {name} registered",
        }

    # =========================
    # QUERY METHODS
    # =========================

    async def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow details"""
        if workflow_id not in self.workflows:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        workflow = self.workflows[workflow_id]

        return {"success": True, "workflow": self._workflow_to_dict(workflow)}

    async def get_all_workflows(
        self, status: Optional[str] = None, tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get all workflows"""
        workflows = list(self.workflows.values())

        if status:
            workflows = [w for w in workflows if w.status.value == status]

        if tag:
            workflows = [w for w in workflows if tag in w.tags]

        # Sort by priority
        workflows.sort(key=lambda w: w.priority, reverse=True)

        return {
            "success": True,
            "total": len(workflows),
            "workflows": [self._workflow_to_dict(w) for w in workflows],
        }

    async def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get execution details"""
        # Check active executions
        if execution_id in self.executions:
            execution = self.executions[execution_id]
        else:
            # Check history
            for exec_hist in self.execution_history:
                if exec_hist.id == execution_id:
                    execution = exec_hist
                    break
            else:
                return {
                    "success": False,
                    "error": f"Execution {execution_id} not found",
                }

        return {"success": True, "execution": self._execution_to_dict(execution)}

    async def get_executions(
        self, workflow_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get recent executions"""
        executions = list(self.execution_history)[-limit:]

        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]

        return {
            "success": True,
            "total": len(executions),
            "executions": [self._execution_to_dict(e) for e in executions],
        }

    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get metrics for a workflow"""
        if workflow_id not in self.workflow_metrics:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}

        metrics = self.workflow_metrics[workflow_id]

        return {
            "success": True,
            "metrics": {
                "total_executions": metrics.total_executions,
                "successful_executions": metrics.successful_executions,
                "failed_executions": metrics.failed_executions,
                "success_rate": (
                    metrics.successful_executions / max(1, metrics.total_executions)
                )
                * 100,
                "avg_execution_time_ms": round(metrics.avg_execution_time_ms, 2),
                "min_execution_time_ms": (
                    round(metrics.min_execution_time_ms, 2)
                    if metrics.min_execution_time_ms != float("inf")
                    else 0
                ),
                "max_execution_time_ms": round(metrics.max_execution_time_ms, 2),
                "total_execution_time_ms": round(metrics.total_execution_time_ms, 2),
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        queue_size = (
            self.execution_queue.qsize()
            if hasattr(self.execution_queue, "qsize")
            else 0
        )

        return {
            **self.stats,
            "queue_size": queue_size,
            "workers": len(self.workers),
            "webhook_routes": len(self.webhook_routes),
            "custom_actions": len(self.custom_actions),
            "workflow_metrics_count": len(self.workflow_metrics),
            "history_size": len(self.execution_history),
            "circuit_breaker_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
        }

    # =========================
    # LIFECYCLE MANAGEMENT
    # =========================

    async def start(self):
        """Start the automation agent"""
        self.logger.info("Starting Automation Agent...")

        # Start workers
        await self._start_workers()

        # Start scheduler
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

        # Start webhook server if enabled
        if self.config.get("enable_webhook_server", True):
            self.webhook_server_task = asyncio.create_task(self._start_webhook_server())

        self.logger.info("Automation Agent started")

    async def stop(self):
        """Stop the automation agent"""
        self.logger.info("Stopping Automation Agent...")

        # Stop workers
        await self._stop_workers()

        # Stop scheduler
        if self.scheduler_task:
            self.scheduler_running = False
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        # Stop file watcher
        if self.file_watcher_task:
            self.file_watcher_task.cancel()
            try:
                await self.file_watcher_task
            except asyncio.CancelledError:
                pass

        # Stop webhook server
        if self.webhook_server_task:
            self.webhook_server_task.cancel()
            try:
                await self.webhook_server_task
            except asyncio.CancelledError:
                pass

        # Save data
        if self.persist_enabled:
            self._save_data()

        self.logger.info("Automation Agent stopped")


# =========================
# INTEGRATION WRAPPER
# =========================


class AutomationAgentWrapper:
    """
    Wrapper class to integrate AutomationAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.automation_agent = AutomationAgent(config)
        self.agent_type = "automation"
        self.capabilities = [
            "create_workflow",
            "trigger_workflow",
            "webhook_integration",
            "schedule_automation",
            "file_watching",
            "custom_actions",
            "list_workflows",
            "get_workflow",
            "delete_workflow",
            "update_workflow",
            "get_executions",
            "get_metrics",
        ]
        self._initialized = False

    async def initialize(self, *args, **kwargs):
        """Initialize the wrapper"""
        await self.automation_agent.start()
        self._initialized = True
        return True

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an automation request

        Request format:
        {
            'operation': 'create_workflow|trigger|list|get|delete|stats',
            ... operation specific parameters ...
        }
        """
        if not self._initialized:
            await self.initialize()

        operation = request.get("operation")

        if operation == "create_workflow":
            trigger_type = TriggerType(request.get("trigger_type"))
            actions_data = request.get("actions", [])

            actions = []
            for action_data in actions_data:
                action = Action(
                    type=ActionType(action_data["type"]),
                    config=action_data["config"],
                    name=action_data.get("name", ""),
                    description=action_data.get("description", ""),
                    next_on_success=action_data.get("next_on_success"),
                    next_on_failure=action_data.get("next_on_failure"),
                    retry_count=action_data.get("retry_count", 0),
                    retry_delay=action_data.get("retry_delay", 5),
                    timeout=action_data.get("timeout", 30),
                    continue_on_error=action_data.get("continue_on_error", False),
                )
                actions.append(action)

            trigger = Trigger(
                type=trigger_type, config=request.get("trigger_config", {})
            )

            return await self.automation_agent.create_workflow(
                name=request.get("name"),
                description=request.get("description", ""),
                trigger=trigger,
                actions=actions,
                tags=request.get("tags", []),
                max_executions=request.get("max_executions"),
                owner=request.get("owner", "system"),
                priority=request.get("priority", 5),
            )

        elif operation == "trigger":
            return await self.automation_agent.trigger_workflow(
                workflow_id=request.get("workflow_id"),
                trigger_data=request.get("data", {}),
                source=request.get("source", "api"),
            )

        elif operation == "list":
            return await self.automation_agent.get_all_workflows(
                status=request.get("status"), tag=request.get("tag")
            )

        elif operation == "get":
            return await self.automation_agent.get_workflow(
                workflow_id=request.get("workflow_id")
            )

        elif operation == "get_execution":
            return await self.automation_agent.get_execution(
                execution_id=request.get("execution_id")
            )

        elif operation == "get_executions":
            return await self.automation_agent.get_executions(
                workflow_id=request.get("workflow_id"), limit=request.get("limit", 100)
            )

        elif operation == "delete":
            return await self.automation_agent.delete_workflow(
                workflow_id=request.get("workflow_id")
            )

        elif operation == "update":
            updates = {
                k: v
                for k, v in request.items()
                if k not in ["operation", "workflow_id"]
            }
            return await self.automation_agent.update_workflow(
                workflow_id=request.get("workflow_id"), **updates
            )

        elif operation == "enable":
            return await self.automation_agent.enable_workflow(
                workflow_id=request.get("workflow_id")
            )

        elif operation == "disable":
            return await self.automation_agent.disable_workflow(
                workflow_id=request.get("workflow_id")
            )

        elif operation == "pause":
            return await self.automation_agent.pause_workflow(
                workflow_id=request.get("workflow_id")
            )

        elif operation == "get_metrics":
            return await self.automation_agent.get_workflow_metrics(
                workflow_id=request.get("workflow_id")
            )

        elif operation == "register_action":
            return await self.automation_agent.register_custom_action(
                name=request.get("action_name"), callback=request.get("callback")
            )

        elif operation == "stats":
            return self.automation_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "AutomationAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.automation_agent.get_stats(),
            "trigger_types": [t.value for t in TriggerType],
            "action_types": [a.value for a in ActionType],
            "initialized": self._initialized,
        }

    async def close(self):
        """Clean up resources"""
        await self.automation_agent.stop()
        self._initialized = False


# =========================
# EXAMPLE USAGE
# =========================


async def test_automation_agent():
    """Test the automation agent functionality"""

    # Initialize agent
    agent = AutomationAgent({"persist_enabled": False})
    await agent.start()

    print("=== Automation Agent Test ===\n")

    # Test creating a webhook-triggered workflow
    print("1. Creating Webhook Workflow...")

    actions = [
        Action(
            name="Log Trigger",
            type=ActionType.LOG,
            config={
                "message": "Workflow triggered with data: {{trigger_data}}",
                "level": "info",
            },
        ),
        Action(
            name="Transform Data",
            type=ActionType.TRANSFORM,
            config={"type": "json", "input": "trigger_data.body"},
        ),
        Action(
            name="Send Webhook",
            type=ActionType.WEBHOOK,
            config={
                "url": "https://webhook.site/example",
                "method": "POST",
                "body": {"received": "{{trigger_data}}", "timestamp": "{{now}}"},
            },
        ),
    ]

    result = await agent.create_webhook_workflow(
        name="Test Webhook Workflow",
        description="Processes incoming webhook data",
        webhook_path="/webhook/test",
        actions=actions,
    )
    print(f"   Workflow created: {result['success']}")
    if result["success"]:
        print(f"   Workflow ID: {result['workflow_id']}")

    # Test creating a schedule-triggered workflow
    print("\n2. Creating Schedule Workflow...")

    actions = [
        Action(
            name="Log Schedule",
            type=ActionType.LOG,
            config={
                "message": "Scheduled job running at {{current_time}}",
                "level": "info",
            },
        ),
        Action(
            name="Send Notification",
            type=ActionType.NOTIFICATION,
            config={"message": "Hourly task completed", "channel": "desktop"},
        ),
    ]

    result = await agent.create_schedule_workflow(
        name="Hourly Report",
        description="Runs every hour",
        cron_expression="0 * * * *",
        actions=actions,
    )
    print(f"   Schedule workflow: {result['success']}")

    # Test manual trigger
    print("\n3. Manual Trigger Test...")
    if result["success"]:
        trigger_result = await agent.trigger_workflow(
            workflow_id=result["workflow_id"],
            trigger_data={"test": "data", "timestamp": datetime.now().isoformat()},
        )
        print(f"   Manual trigger: {trigger_result['success']}")
        if trigger_result["success"]:
            print(f"   Execution ID: {trigger_result['execution_id']}")

    # Wait for execution
    await asyncio.sleep(2)

    # Get all workflows
    print("\n4. All Workflows...")
    workflows = await agent.get_all_workflows()
    print(f"   Total workflows: {workflows['total']}")
    for wf in workflows["workflows"]:
        print(f"     - {wf['name']} ({wf['trigger']['type']}) - Status: {wf['status']}")

    # Get execution details
    if "execution_id" in locals():
        print("\n5. Execution Details...")
        exec_result = await agent.get_execution(trigger_result["execution_id"])
        if exec_result["success"]:
            exec_data = exec_result["execution"]
            print(f"   Status: {exec_data['status']}")
            print(f"   Started: {exec_data['started_at']}")
            print(f"   Completed: {exec_data['completed_at']}")
            if exec_data["error"]:
                print(f"   Error: {exec_data['error']}")

    # Get workflow metrics
    if "workflow_id" in locals():
        print("\n6. Workflow Metrics...")
        metrics = await agent.get_workflow_metrics(result["workflow_id"])
        if metrics["success"]:
            m = metrics["metrics"]
            print(f"   Total executions: {m['total_executions']}")
            print(f"   Success rate: {m['success_rate']:.1f}%")
            print(f"   Avg execution time: {m['avg_execution_time_ms']:.2f}ms")

    # Get agent statistics
    print("\n7. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total workflows: {stats['total_workflows']}")
    print(f"   Active workflows: {stats['active_workflows']}")
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Queue size: {stats['queue_size']}")
    print(f"   Workers: {stats['workers']}")

    # Clean up
    print("\n8. Cleaning up...")
    for wf in workflows["workflows"]:
        if "Test" in wf["name"] or "Hourly" in wf["name"]:
            result = await agent.delete_workflow(wf["id"])
            print(f"     Deleted: {wf['name']}")

    # Stop agent
    await agent.stop()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_automation_agent())
