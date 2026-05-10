"""
Advanced Workflow Engine - DAG Based Execution
Enhanced with persistence, monitoring, parallel execution, and advanced error handling
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback

from ..utils.logger import logger


class WorkflowStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING = "pending"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DAG = "dag"
    HYBRID = "hybrid"


class ErrorHandlingStrategy(Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    RETRY = "retry"
    FALLBACK = "fallback"


@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""

    execution_id: str
    workflow_id: str
    workflow_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    steps_total: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    retry_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class WorkflowEvent:
    """Event for workflow monitoring"""

    event_id: str
    workflow_id: str
    event_type: str
    step_name: Optional[str]
    message: str
    timestamp: datetime
    data: Optional[Dict] = None


class WorkflowStep:
    """Enhanced workflow step with advanced features"""

    def __init__(
        self,
        name: str,
        action: Callable,
        depends_on: Optional[List[str]] = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        fallback: Optional[Callable] = None,
        condition: Optional[Callable] = None,
        parallel_group: Optional[str] = None,
        priority: int = 0,
        max_concurrency: int = 1,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.name = name
        self.action = action
        self.depends_on = depends_on or []
        self.retries = retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.fallback = fallback
        self.condition = condition
        self.parallel_group = parallel_group
        self.priority = priority
        self.max_concurrency = max_concurrency
        self.tags = tags or []
        self.metadata = metadata or {}

        # Runtime attributes
        self.status: StepStatus = StepStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.retry_count: int = 0
        self.execution_id: Optional[str] = None

    async def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute step with enhanced safety, retries, timeout, and monitoring
        """

        import asyncio
        from datetime import datetime
        import uuid

        self.status = StepStatus.RUNNING
        self.start_time = datetime.now()
        self.execution_id = str(uuid.uuid4())
        self.error = None

        try:
            # ------------------------
            # 🔍 CONDITION CHECK
            # ------------------------
            if self.condition:
                try:
                    should_execute = (
                        await self.condition(context)
                        if asyncio.iscoroutinefunction(self.condition)
                        else self.condition(context)
                    )

                    if not should_execute:
                        self.status = StepStatus.SKIPPED
                        self.result = None
                        self.end_time = datetime.now()
                        return None

                except Exception as e:
                    logger.warning(f"Condition check failed for {self.name}: {e}")

            # ------------------------
            # 🔁 RETRY LOOP
            # ------------------------
            last_error = None

            for attempt in range(self.retries + 1):
                try:
                    coro = self._execute_action(context)

                    result = (
                        await asyncio.wait_for(coro, timeout=self.timeout)
                        if self.timeout
                        else await coro
                    )

                    # ✅ SUCCESS
                    self.result = result
                    self.status = StepStatus.COMPLETED
                    self.end_time = datetime.now()
                    return result

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {self.timeout}s"
                    self.status = StepStatus.TIMEOUT

                except Exception as e:
                    last_error = str(e)
                    self.retry_count = attempt + 1

                    if attempt < self.retries:
                        try:
                            logger.warning(
                                f"Step {self.name} failed "
                                f"(attempt {attempt + 1}/{self.retries + 1}): {e}"
                            )
                        except Exception:
                            pass

                        await asyncio.sleep(max(0.1, self.retry_delay))
                    else:
                        break

            # ------------------------
            # 🔄 FALLBACK HANDLING
            # ------------------------
            self.error = last_error
            self.end_time = datetime.now()

            if self.fallback:
                try:
                    logger.info(f"Executing fallback for {self.name}")

                    result = (
                        await self.fallback(context, last_error)
                        if asyncio.iscoroutinefunction(self.fallback)
                        else self.fallback(context, last_error)
                    )

                    self.result = result
                    self.status = StepStatus.COMPLETED
                    return result

                except Exception as e:
                    self.error = f"Fallback failed: {e}"

            # ------------------------
            # ❌ FINAL FAILURE
            # ------------------------
            self.status = StepStatus.FAILED

            raise RuntimeError(
                f"Step {self.name} failed after {self.retries + 1} attempts: {last_error}"
            )

        finally:
            # ------------------------
            # 🔒 GUARANTEE END TIME
            # ------------------------
            if not getattr(self, "end_time", None):
                self.end_time = datetime.now()

    async def _execute_action(self, context: Dict[str, Any]) -> Any:
        """
        Execute action safely with smart context injection (production-grade)
        """

        import asyncio
        import inspect

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not callable(self.action):
                raise TypeError(f"Action for {self.name} is not callable")

            # ------------------------
            # 🔍 SIGNATURE INSPECTION (SAFE)
            # ------------------------
            try:
                sig = inspect.signature(self.action)
                params = sig.parameters
            except Exception:
                # fallback if signature fails (e.g. C functions)
                params = {}

            # ------------------------
            # 🔁 DETERMINE ARGUMENTS
            # ------------------------
            args = ()
            kwargs = {}

            if params:
                # try smart injection
                first_param = next(iter(params.values()))

                if first_param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    args = (context,)
                elif first_param.kind == inspect.Parameter.VAR_KEYWORD:
                    kwargs = context if isinstance(context, dict) else {}
                else:
                    args = (context,)

            # ------------------------
            # ⚡ EXECUTION
            # ------------------------
            if asyncio.iscoroutinefunction(self.action):
                return await self.action(*args, **kwargs)

            # run sync safely in thread
            return await asyncio.to_thread(self.action, *args, **kwargs)

        except Exception as e:
            try:
                logger.error(f"❌ Action execution error in {self.name}: {e}")
            except Exception:
                pass
            raise

    def get_duration_ms(self) -> float:
        """Get step execution duration in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict:
        """Convert step to dictionary for serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.get_duration_ms(),
            "retry_count": self.retry_count,
            "depends_on": self.depends_on,
            "priority": self.priority,
            "tags": self.tags,
        }


class Workflow:
    """Enhanced workflow with persistence and monitoring"""

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        timeout: Optional[float] = None,
        max_parallel: int = 10,
        error_strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.DAG,
        persist_results: bool = True,
        checkpoint_interval: int = 5,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version
        self.timeout = timeout
        self.max_parallel = max_parallel
        self.error_strategy = error_strategy
        self.execution_strategy = execution_strategy
        self.persist_results = persist_results
        self.checkpoint_interval = checkpoint_interval

        self.steps: Dict[str, WorkflowStep] = {}
        self.step_order: List[str] = []

        self.status = WorkflowStatus.CREATED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.paused_at: Optional[datetime] = None
        self.resumed_at: Optional[datetime] = None

        self._paused = False
        self._cancelled = False
        self._context: Dict[str, Any] = {}
        self._checkpoint_count = 0
        self._events: List[WorkflowEvent] = []

        # Metrics
        self.metrics = WorkflowMetrics(
            execution_id=self.id,
            workflow_id=self.id,
            workflow_name=name,
            start_time=self.created_at,
        )

        # Callbacks
        self.on_step_start: Optional[Callable] = None
        self.on_step_complete: Optional[Callable] = None
        self.on_step_failed: Optional[Callable] = None
        self.on_workflow_complete: Optional[Callable] = None
        self.on_workflow_failed: Optional[Callable] = None

    def add_step(
        self,
        name: str,
        action: Callable,
        depends_on: Optional[List[str]] = None,
        retries: int = 0,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        fallback: Optional[Callable] = None,
        condition: Optional[Callable] = None,
        parallel_group: Optional[str] = None,
        priority: int = 0,
        max_concurrency: int = 1,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> WorkflowStep:
        """
        Add a workflow step safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not isinstance(name, str):
                raise ValueError("Step name must be a non-empty string")

            name = name.strip()

            if not callable(action):
                raise TypeError(f"Action for step '{name}' must be callable")

            steps = getattr(self, "steps", {})
            if name in steps:
                raise ValueError(f"Step '{name}' already exists")

            # ------------------------
            # 🔁 NORMALIZATION
            # ------------------------
            depends_on = list(set(depends_on or []))

            retries = max(0, int(retries))
            retry_delay = max(0.0, float(retry_delay))
            timeout = float(timeout) if timeout is not None else None
            priority = int(priority)
            max_concurrency = max(1, int(max_concurrency))

            tags = list(set(tags or []))
            metadata = metadata or {}

            # ------------------------
            # 🔍 DEPENDENCY VALIDATION (CRITICAL FIX)
            # ------------------------
            for dep in depends_on:
                if dep not in steps:
                    raise ValueError(f"Dependency '{dep}' does not exist")

            # ------------------------
            # 🔥 CREATE STEP
            # ------------------------
            step = WorkflowStep(
                name=name,
                action=action,
                depends_on=depends_on,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                fallback=fallback,
                condition=condition,
                parallel_group=parallel_group,
                priority=priority,
                max_concurrency=max_concurrency,
                tags=tags,
                metadata=metadata,
            )

            # ------------------------
            # 📦 STORE STEP
            # ------------------------
            self.steps[name] = step

            order = getattr(self, "step_order", [])
            order.append(name)
            self.step_order = order

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.steps_added = getattr(self, "steps_added", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 📢 LOG
            # ------------------------
            try:
                logger.debug(f"➕ Step added: {name}")
            except Exception:
                pass

            return step

        except Exception as e:
            try:
                logger.error(f"❌ Failed to add step '{name}': {e}")
            except Exception:
                pass
            raise

    def add_parallel_steps(
        self,
        steps: List[Tuple[str, Callable]],
        depends_on: Optional[List[str]] = None,
        group_name: Optional[str] = None,
    ) -> List[WorkflowStep]:
        """Add multiple steps that can execute in parallel"""
        group = group_name or f"parallel_group_{uuid.uuid4().hex[:8]}"
        added_steps = []

        for name, action in steps:
            step = self.add_step(
                name=name, action=action, depends_on=depends_on, parallel_group=group
            )
            added_steps.append(step)

        return added_steps

    def get_step(self, name: str) -> Optional[WorkflowStep]:
        """Get a step by name"""
        return self.steps.get(name)

    def get_steps_by_tag(self, tag: str) -> List[WorkflowStep]:
        """Get all steps with a specific tag"""
        return [step for step in self.steps.values() if tag in step.tags]

    def get_steps_by_status(self, status: StepStatus) -> List[WorkflowStep]:
        """Get all steps with a specific status"""
        return [step for step in self.steps.values() if step.status == status]

    def pause(self):
        """Pause workflow execution"""
        self._paused = True
        self.status = WorkflowStatus.PAUSED
        self.paused_at = datetime.now()
        self._add_event("workflow_paused", None, "Workflow paused")

    def resume(self):
        """Resume workflow execution"""
        self._paused = False
        self.status = WorkflowStatus.RUNNING
        self.resumed_at = datetime.now()
        self._add_event("workflow_resumed", None, "Workflow resumed")

    def cancel(self):
        """Cancel workflow execution"""
        self._cancelled = True
        self.status = WorkflowStatus.CANCELLED
        self.completed_at = datetime.now()
        self._add_event("workflow_cancelled", None, "Workflow cancelled")

    def set_context(self, key: str, value: Any):
        """Set context value for workflow"""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value from workflow"""
        return self._context.get(key, default)

    def _add_event(
        self,
        event_type: str,
        step_name: Optional[str],
        message: str,
        data: Optional[Dict] = None,
    ):
        """Add monitoring event"""
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=self.id,
            event_type=event_type,
            step_name=step_name,
            message=message,
            timestamp=datetime.now(),
            data=data,
        )
        self._events.append(event)

        # Keep only last 1000 events
        if len(self._events) > 1000:
            self._events = self._events[-1000:]

    def get_events(self, limit: int = 100) -> List[WorkflowEvent]:
        """Get workflow events"""
        return self._events[-limit:]

    def get_duration_ms(self) -> float:
        """Get workflow execution duration in milliseconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def get_progress(self) -> float:
        """Get workflow completion progress (0-100)"""
        if not self.steps:
            return 0.0

        completed = sum(
            1
            for step in self.steps.values()
            if step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
        )
        return (completed / len(self.steps)) * 100

    def to_dict(self) -> Dict:
        """Convert workflow to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_ms": self.get_duration_ms(),
            "progress": self.get_progress(),
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "metrics": asdict(self.metrics),
            "context_keys": list(self._context.keys()),
        }

    def save_checkpoint(self, checkpoint_dir: Path) -> bool:
        """Save workflow checkpoint to disk"""
        if not self.persist_results:
            return False

        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = (
                checkpoint_dir / f"{self.id}_checkpoint_{self._checkpoint_count}.json"
            )

            checkpoint_data = {
                "workflow_id": self.id,
                "checkpoint_id": self._checkpoint_count,
                "timestamp": datetime.now().isoformat(),
                "status": self.status.value,
                "completed_steps": [
                    name
                    for name, step in self.steps.items()
                    if step.status == StepStatus.COMPLETED
                ],
                "results": {
                    name: str(step.result)
                    for name, step in self.steps.items()
                    if step.result is not None
                },
                "context": {k: str(v) for k, v in self._context.items()},
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            self._checkpoint_count += 1
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False


class WorkflowEngine:
    """Advanced workflow engine with persistence, monitoring, and optimization"""

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_concurrent_workflows: int = 10,
        enable_metrics: bool = True,
        enable_persistence: bool = True,
    ):
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, asyncio.Task] = {}
        self.execution_history: List[WorkflowMetrics] = []
        self.checkpoint_dir = checkpoint_dir or Path("data/workflow_checkpoints")
        self.max_concurrent_workflows = max_concurrent_workflows
        self.enable_metrics = enable_metrics
        self.enable_persistence = enable_persistence

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_workflows)

        # Statistics
        self.execution_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.cancelled_count = 0

        # Event bus for monitoring
        self._event_listeners: List[Callable] = []

        # Create checkpoint directory
        if self.enable_persistence:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🚀 WorkflowEngine initialized with advanced features")

    def create_workflow(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        timeout: Optional[float] = None,
        error_strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.DAG,
    ) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            name=name,
            description=description,
            version=version,
            timeout=timeout,
            error_strategy=error_strategy,
            execution_strategy=execution_strategy,
            persist_results=self.enable_persistence,
        )
        self.workflows[workflow.id] = workflow
        logger.info(f"✅ Workflow created: {name} (id: {workflow.id})")
        return workflow

    async def execute_workflow(
        self, workflow: Workflow, initial_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow safely with DAG-based parallel execution (production-grade)
        """

        import asyncio
        from datetime import datetime

        async with self._semaphore:
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            workflow.metrics.start_time = workflow.started_at

            try:
                # ------------------------
                # 🔥 CONTEXT INIT (SAFE)
                # ------------------------
                if initial_context and isinstance(initial_context, dict):
                    for key, value in initial_context.items():
                        try:
                            workflow.set_context(key, value)
                        except Exception:
                            pass

                workflow._add_event(
                    "workflow_started", None, "Workflow execution started"
                )

                try:
                    logger.info(f"▶️ Executing workflow: {workflow.name}")
                except Exception:
                    pass

                results = {}

                # ------------------------
                # 🔥 BUILD + VALIDATE DAG
                # ------------------------
                dependency_graph = self._build_dependency_graph(workflow)

                if not self._validate_dag(dependency_graph):
                    raise RuntimeError("Circular dependency detected in workflow")

                # ------------------------
                # 🔁 EXECUTION STRATEGY
                # ------------------------
                if workflow.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                    results = await self._execute_sequential(workflow)

                elif workflow.execution_strategy == ExecutionStrategy.PARALLEL:
                    results = (
                        await asyncio.wait_for(
                            self._execute_parallel_all(workflow),
                            timeout=getattr(workflow, "timeout", None),
                        )
                        if getattr(workflow, "timeout", None)
                        else await self._execute_parallel_all(workflow)
                    )

                else:  # DAG / HYBRID
                    results = (
                        await asyncio.wait_for(
                            self._execute_dag(workflow, dependency_graph),
                            timeout=getattr(workflow, "timeout", None),
                        )
                        if getattr(workflow, "timeout", None)
                        else await self._execute_dag(workflow, dependency_graph)
                    )

                # ------------------------
                # 📊 METRICS (SAFE)
                # ------------------------
                try:
                    steps = list(workflow.steps.values())

                    workflow.metrics.steps_completed = sum(
                        1 for s in steps if s.status == StepStatus.COMPLETED
                    )
                    workflow.metrics.steps_failed = sum(
                        1 for s in steps if s.status == StepStatus.FAILED
                    )
                    workflow.metrics.steps_skipped = sum(
                        1 for s in steps if s.status == StepStatus.SKIPPED
                    )
                except Exception:
                    pass

                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = datetime.now()
                workflow.metrics.end_time = workflow.completed_at
                workflow.metrics.duration_ms = workflow.get_duration_ms()

                self.completed_count = getattr(self, "completed_count", 0) + 1

                workflow._add_event(
                    "workflow_completed", None, "Workflow completed successfully"
                )

                try:
                    logger.info(
                        f"✅ Workflow completed: {workflow.name} in {workflow.get_duration_ms():.2f}ms"
                    )
                except Exception:
                    pass

                # ------------------------
                # 🔔 CALLBACK (SAFE)
                # ------------------------
                try:
                    if workflow.on_workflow_complete:
                        if asyncio.iscoroutinefunction(workflow.on_workflow_complete):
                            await workflow.on_workflow_complete(results)
                        else:
                            await asyncio.to_thread(
                                workflow.on_workflow_complete, results
                            )
                except Exception as e:
                    logger.warning(f"Workflow completion callback failed: {e}")

                # ------------------------
                # 💾 CHECKPOINT (SAFE)
                # ------------------------
                if getattr(self, "enable_persistence", False):
                    try:
                        workflow.save_checkpoint(self.checkpoint_dir)
                    except Exception:
                        pass

                return results

            except asyncio.CancelledError:
                workflow.status = WorkflowStatus.CANCELLED
                workflow.completed_at = datetime.now()
                self.cancelled_count = getattr(self, "cancelled_count", 0) + 1
                workflow._add_event("workflow_cancelled", None, "Workflow cancelled")
                raise

            except Exception as e:
                workflow.status = WorkflowStatus.FAILED
                workflow.completed_at = datetime.now()
                workflow.metrics.end_time = workflow.completed_at

                self.failed_count = getattr(self, "failed_count", 0) + 1

                error_msg = f"Workflow failed: {str(e)}\n{traceback.format_exc()}"
                workflow._add_event("workflow_failed", None, error_msg)

                try:
                    logger.error(f"❌ Workflow failed: {workflow.name} - {e}")
                except Exception:
                    pass

                # ------------------------
                # 🔔 FAILURE CALLBACK (SAFE)
                # ------------------------
                try:
                    if workflow.on_workflow_failed:
                        if asyncio.iscoroutinefunction(workflow.on_workflow_failed):
                            await workflow.on_workflow_failed(e)
                        else:
                            await asyncio.to_thread(workflow.on_workflow_failed, e)
                except Exception:
                    pass

                raise

            finally:
                # ------------------------
                # 📊 GLOBAL METRICS
                # ------------------------
                self.execution_count = getattr(self, "execution_count", 0) + 1

                if getattr(self, "enable_metrics", False):
                    try:
                        self.execution_history = getattr(self, "execution_history", [])
                        self.execution_history.append(workflow.metrics)

                        if len(self.execution_history) > 1000:
                            self.execution_history = self.execution_history[-1000:]
                    except Exception:
                        pass

    def _build_dependency_graph(self, workflow: Workflow) -> Dict[str, Set[str]]:
        """Build dependency graph for workflow"""
        graph = {}
        for step_name, step in workflow.steps.items():
            graph[step_name] = set(step.depends_on)
        return graph

    def _validate_dag(self, graph: Dict[str, Set[str]]) -> bool:
        """Validate that graph has no cycles"""
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return False

        return True

    async def _execute_sequential(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute workflow sequentially (optimized + production-grade)
        """

        import asyncio

        results = {}

        for step_name in list(getattr(workflow, "step_order", [])):

            try:
                # ------------------------
                # 🔥 CANCEL CHECK
                # ------------------------
                if getattr(workflow, "_cancelled", False):
                    raise RuntimeError("Workflow cancelled")

                # ------------------------
                # ⏸️ PAUSE HANDLING (SAFE)
                # ------------------------
                while getattr(workflow, "_paused", False):
                    await asyncio.sleep(0.1)

                step = workflow.steps.get(step_name)
                if not step:
                    continue

                # ------------------------
                # 🔁 DEPENDENCY CHECK (SAFE)
                # ------------------------
                if getattr(step, "depends_on", None):
                    deps_met = all(
                        workflow.steps.get(dep)
                        and workflow.steps[dep].status == StepStatus.COMPLETED
                        for dep in step.depends_on
                    )
                    if not deps_met:
                        continue

                # ------------------------
                # 🔔 STEP START CALLBACK (SAFE)
                # ------------------------
                try:
                    if workflow.on_step_start:
                        if asyncio.iscoroutinefunction(workflow.on_step_start):
                            await workflow.on_step_start(step)
                        else:
                            await asyncio.to_thread(workflow.on_step_start, step)
                except Exception as e:
                    logger.warning(f"Step start callback failed: {e}")

                # ------------------------
                # ⚡ EXECUTE STEP (WITH TIMEOUT)
                # ------------------------
                try:
                    result = (
                        await asyncio.wait_for(
                            step.execute(workflow._context),
                            timeout=getattr(step, "timeout", None),
                        )
                        if getattr(step, "timeout", None)
                        else await step.execute(workflow._context)
                    )

                except asyncio.TimeoutError:
                    raise RuntimeError(f"Step '{step_name}' timed out")

                results[step_name] = result

                # ------------------------
                # 🧠 CONTEXT UPDATE (SAFE)
                # ------------------------
                try:
                    workflow.set_context(step_name, result)
                except Exception:
                    pass

                # ------------------------
                # 🔔 STEP COMPLETE CALLBACK (SAFE)
                # ------------------------
                try:
                    if workflow.on_step_complete:
                        if asyncio.iscoroutinefunction(workflow.on_step_complete):
                            await workflow.on_step_complete(step, result)
                        else:
                            await asyncio.to_thread(
                                workflow.on_step_complete, step, result
                            )
                except Exception as e:
                    logger.warning(f"Step complete callback failed: {e}")

                workflow._add_event(
                    "step_completed", step_name, "Step completed successfully"
                )

            except Exception as e:
                # ------------------------
                # ❌ FAILURE HANDLING
                # ------------------------
                workflow._add_event("step_failed", step_name, str(e))

                try:
                    if workflow.on_step_failed:
                        if asyncio.iscoroutinefunction(workflow.on_step_failed):
                            await workflow.on_step_failed(step, e)
                        else:
                            await asyncio.to_thread(workflow.on_step_failed, step, e)
                except Exception:
                    pass

                strategy = getattr(workflow, "error_strategy", None)

                if strategy == ErrorHandlingStrategy.FAIL_FAST:
                    raise

                elif strategy == ErrorHandlingStrategy.CONTINUE:
                    logger.warning(f"Step {step_name} failed but continuing: {e}")
                    continue

                else:
                    raise

        return results

    async def _execute_parallel_all(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute all steps in parallel safely (optimized + production-grade)
        """

        import asyncio

        try:
            tasks = []
            step_names = list(workflow.steps.keys())

            # ------------------------
            # 🔥 CREATE TASKS (SAFE)
            # ------------------------
            for step_name in step_names:
                step = workflow.steps.get(step_name)
                if not step:
                    continue

                coro = self._execute_step_safe(workflow, step, workflow._context)

                # wrap with timeout if defined
                if getattr(step, "timeout", None):
                    coro = asyncio.wait_for(coro, timeout=step.timeout)

                tasks.append(coro)

            # ------------------------
            # ⚡ EXECUTE ALL
            # ------------------------
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            results = {}

            # ------------------------
            # 🔁 PROCESS RESULTS
            # ------------------------
            for step_name, result in zip(step_names, results_list):

                if isinstance(result, Exception):
                    if workflow.error_strategy == ErrorHandlingStrategy.FAIL_FAST:
                        raise result

                    try:
                        logger.warning(f"Step {step_name} failed: {result}")
                    except Exception:
                        pass

                    continue

                # ------------------------
                # 🧠 SAFE CONTEXT UPDATE
                # ------------------------
                results[step_name] = result

                try:
                    workflow.set_context(step_name, result)
                except Exception:
                    pass

            return results

        except Exception as e:
            try:
                logger.error(f"❌ Parallel execution failed: {e}")
            except Exception:
                pass
            raise

    async def _execute_dag(
        self, workflow: Workflow, dependency_graph: Dict[str, Set[str]]
    ) -> Dict[str, Any]:
        """
        Execute workflow using DAG-based parallel execution (production-grade)
        """

        import asyncio
        from datetime import datetime

        results = {}
        completed = set()
        in_progress = set()

        # ------------------------
        # 🔥 INITIAL READY SET
        # ------------------------
        ready = {step_name for step_name, deps in dependency_graph.items() if not deps}

        try:
            while len(completed) < len(workflow.steps):

                # ------------------------
                # ❌ CANCEL CHECK
                # ------------------------
                if getattr(workflow, "_cancelled", False):
                    raise RuntimeError("Workflow cancelled")

                # ------------------------
                # ⏸️ PAUSE HANDLING
                # ------------------------
                while getattr(workflow, "_paused", False):
                    await asyncio.sleep(0.1)

                # ------------------------
                # 🔒 DEADLOCK DETECTION
                # ------------------------
                if not ready:
                    remaining = set(workflow.steps.keys()) - completed
                    if remaining:
                        raise RuntimeError(
                            f"Deadlock detected. Remaining steps: {remaining}"
                        )
                    break

                # ------------------------
                # 🔥 CONCURRENCY CONTROL
                # ------------------------
                max_parallel = max(1, getattr(workflow, "max_parallel", 1))
                batch = list(ready)[:max_parallel]

                in_progress.update(batch)
                ready -= set(batch)

                # ------------------------
                # ⚡ CREATE TASKS (SAFE + TIMEOUT)
                # ------------------------
                tasks = []
                for step_name in batch:
                    step = workflow.steps.get(step_name)
                    if not step:
                        continue

                    coro = self._execute_step_with_deps(
                        workflow, step, workflow._context, results
                    )

                    if getattr(step, "timeout", None):
                        coro = asyncio.wait_for(coro, timeout=step.timeout)

                    tasks.append(coro)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # ------------------------
                # 🔁 PROCESS RESULTS
                # ------------------------
                for step_name, result in zip(batch, batch_results):

                    in_progress.discard(step_name)

                    if isinstance(result, Exception):

                        if workflow.error_strategy == ErrorHandlingStrategy.FAIL_FAST:
                            raise result

                        try:
                            logger.warning(f"Step {step_name} failed: {result}")
                        except Exception:
                            pass

                        step = workflow.steps.get(step_name)
                        if step:
                            step.status = StepStatus.FAILED
                            step.error = str(result)

                        continue

                    # ------------------------
                    # ✅ SUCCESS
                    # ------------------------
                    results[step_name] = result

                    try:
                        workflow.set_context(step_name, result)
                    except Exception:
                        pass

                    completed.add(step_name)

                    # ------------------------
                    # 🔁 UPDATE READY SET
                    # ------------------------
                    for step, deps in dependency_graph.items():
                        if step not in completed and step not in in_progress:
                            if deps.issubset(completed):
                                ready.add(step)

                # ------------------------
                # ⏱️ GLOBAL TIMEOUT CHECK
                # ------------------------
                if getattr(workflow, "timeout", None):
                    elapsed = (datetime.now() - workflow.started_at).total_seconds()
                    if elapsed > workflow.timeout:
                        raise TimeoutError(
                            f"Workflow timeout after {workflow.timeout}s"
                        )

            return results

        except Exception as e:
            try:
                logger.error(f"❌ DAG execution failed: {e}")
            except Exception:
                pass
            raise

    async def _execute_step_with_deps(
        self,
        workflow: Workflow,
        step: WorkflowStep,
        context: Dict[str, Any],
        results: Dict[str, Any],
    ) -> Any:
        """
        Execute step with strict dependency validation (production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not step:
                raise ValueError("Step is None")

            # ------------------------
            # 🔁 DEPENDENCY CHECK (SAFE)
            # ------------------------
            missing_deps = [
                dep for dep in getattr(step, "depends_on", []) if dep not in results
            ]

            if missing_deps:
                raise RuntimeError(
                    f"Missing dependencies for step '{step.name}': {missing_deps}"
                )

            # ------------------------
            # ⚡ EXECUTION (WITH TIMEOUT)
            # ------------------------
            coro = step.execute(context)

            if getattr(step, "timeout", None):
                result = await asyncio.wait_for(coro, timeout=step.timeout)
            else:
                result = await coro

            return result

        except asyncio.TimeoutError:
            try:
                logger.error(f"⏱️ Step '{step.name}' timed out")
            except Exception:
                pass
            raise

        except Exception as e:
            try:
                logger.error(f"❌ Step '{step.name}' dependency execution failed: {e}")
            except Exception:
                pass
            raise

    async def _execute_step_safe(
        self, workflow: Workflow, step: WorkflowStep, context: Dict[str, Any]
    ) -> Any:
        """
        Execute step safely with structured error handling (production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not step:
                raise ValueError("Step is None")

            # ------------------------
            # ⚡ EXECUTION (WITH TIMEOUT)
            # ------------------------
            coro = step.execute(context)

            if getattr(step, "timeout", None):
                result = await asyncio.wait_for(coro, timeout=step.timeout)
            else:
                result = await coro

            return result

        except asyncio.TimeoutError:
            try:
                logger.warning(f"⏱️ Step '{step.name}' timed out")
            except Exception:
                pass

            # return structured error instead of raw exception
            return RuntimeError(f"Timeout in step '{step.name}'")

        except Exception as e:
            try:
                logger.warning(f"⚠️ Step '{step.name}' failed: {e}")
            except Exception:
                pass

            # return exception (non-breaking for parallel execution)
            return e

    async def execute_workflows_batch(
        self, workflows: List[Workflow], max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple workflows in batch safely (production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not workflows or not isinstance(workflows, list):
                return []

            limit = max_concurrent or getattr(self, "max_concurrent_workflows", 5)
            limit = max(1, int(limit))

            semaphore = asyncio.Semaphore(limit)

            results: List[Any] = []

            # ------------------------
            # 🔁 SAFE EXECUTOR
            # ------------------------
            async def execute_with_limit(workflow: Workflow) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        # optional workflow-level timeout
                        timeout = getattr(workflow, "timeout", None)

                        if timeout:
                            return await asyncio.wait_for(
                                self.execute_workflow(workflow), timeout=timeout
                            )

                        return await self.execute_workflow(workflow)

                    except asyncio.TimeoutError:
                        return {
                            "error": "timeout",
                            "workflow": getattr(workflow, "name", "unknown"),
                        }

                    except Exception as e:
                        return {
                            "error": str(e),
                            "workflow": getattr(workflow, "name", "unknown"),
                        }

            # ------------------------
            # ⚡ CREATE TASKS
            # ------------------------
            tasks = [
                asyncio.create_task(execute_with_limit(wf))
                for wf in workflows
                if wf is not None
            ]

            # ------------------------
            # 🔁 EXECUTE ALL
            # ------------------------
            results = await asyncio.gather(*tasks, return_exceptions=False)

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.batch_executions = getattr(self, "batch_executions", 0) + 1
            except Exception:
                pass

            return results

        except Exception as e:
            try:
                logger.error(f"❌ Batch execution failed: {e}")
            except Exception:
                pass

            return []

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)

    def get_workflow_by_name(self, name: str) -> Optional[Workflow]:
        """Get workflow by name"""
        for workflow in self.workflows.values():
            if workflow.name == name:
                return workflow
        return None

    def get_active_workflows(self) -> List[Workflow]:
        """Get all active (running/paused) workflows"""
        return [
            wf
            for wf in self.workflows.values()
            if wf.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get workflow engine statistics safely (optimized + production-grade)
        """

        try:
            workflows = getattr(self, "workflows", {})
            history = getattr(self, "execution_history", [])

            execution_count = int(getattr(self, "execution_count", 0))
            completed = int(getattr(self, "completed_count", 0))
            failed = int(getattr(self, "failed_count", 0))
            cancelled = int(getattr(self, "cancelled_count", 0))

            # ------------------------
            # 📊 SAFE SUCCESS RATE
            # ------------------------
            success_rate = (
                (completed / execution_count * 100) if execution_count > 0 else 0.0
            )

            # ------------------------
            # 📊 LAST 100 METRICS
            # ------------------------
            recent = history[-100:] if history else []

            avg_duration = (
                sum(getattr(m, "duration_ms", 0) for m in recent) / len(recent)
                if recent
                else 0.0
            )

            total_steps = sum(getattr(m, "steps_total", 0) for m in history)
            total_retries = sum(getattr(m, "retry_count", 0) for m in history)

            # ------------------------
            # 📊 ACTIVE WORKFLOWS (SAFE)
            # ------------------------
            try:
                active = len(self.get_active_workflows())
            except Exception:
                active = 0

            return {
                "total_workflows": len(workflows),
                "executions": execution_count,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "success_rate": round(success_rate, 2),
                "active_workflows": active,
                "max_concurrent": getattr(self, "max_concurrent_workflows", 1),
                "avg_duration_ms": round(avg_duration, 2),
                "total_steps_executed": total_steps,
                "total_retries": total_retries,
                "throughput_per_min": (
                    round(execution_count / max(1, len(history) / 60), 2)
                    if history
                    else 0
                ),
                "timestamp": __import__("time").time(),
            }

        except Exception as e:
            try:
                logger.error(f"❌ Stats generation failed: {e}")
            except Exception:
                pass

            return {"error": "stats_failed", "timestamp": __import__("time").time()}

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        return {
            "engine": self.get_stats(),
            "recent_executions": [
                {
                    "workflow_name": m.workflow_name,
                    "duration_ms": m.duration_ms,
                    "steps_completed": m.steps_completed,
                    "steps_failed": m.steps_failed,
                    "timestamp": m.start_time.isoformat(),
                }
                for m in self.execution_history[-50:]
            ],
        }

    def add_event_listener(self, callback: Callable):
        """Add event listener for monitoring"""
        self._event_listeners.append(callback)

    def remove_event_listener(self, callback: Callable):
        """Remove event listener"""
        if callback in self._event_listeners:
            self._event_listeners.remove(callback)

    async def _broadcast_event(self, event: WorkflowEvent):
        """Broadcast event to all listeners"""
        for listener in self._event_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")

    async def clear_completed_workflows(self, older_than_hours: int = 24):
        """Clear completed workflows older than specified hours"""
        cutoff = datetime.now() - timedelta(hours=older_than_hours)

        to_remove = [
            wf_id
            for wf_id, wf in self.workflows.items()
            if wf.status
            in [
                WorkflowStatus.COMPLETED,
                WorkflowStatus.FAILED,
                WorkflowStatus.CANCELLED,
            ]
            and wf.completed_at
            and wf.completed_at < cutoff
        ]

        for wf_id in to_remove:
            del self.workflows[wf_id]

        logger.info(f"Cleared {len(to_remove)} old workflows")
        return len(to_remove)

    async def _wait_for_workflows_to_finish(self, workflows):
        """Wait until all workflows are no longer active"""
        import asyncio

        while True:
            still_running = [
                wf
                for wf in workflows
                if getattr(wf, "status", None) == WorkflowStatus.RUNNING
            ]

            if not still_running:
                break

            await asyncio.sleep(0.2)

    async def shutdown(self) -> None:
        """
        Shutdown workflow engine safely (optimized + production-grade)
        """

        import asyncio

        try:
            try:
                logger.info("🛑 Shutting down WorkflowEngine...")
            except Exception:
                pass

            # ------------------------
            # 🔥 GET ACTIVE WORKFLOWS (SAFE)
            # ------------------------
            try:
                active = list(self.get_active_workflows() or [])
            except Exception:
                active = []

            # ------------------------
            # ❌ CANCEL ALL WORKFLOWS
            # ------------------------
            for workflow in active:
                try:
                    workflow.cancel()
                except Exception:
                    pass

            # ------------------------
            # ⏱️ WAIT FOR CLEAN EXIT (WITH TIMEOUT)
            # ------------------------
            if active:
                try:
                    await asyncio.wait_for(
                        self._wait_for_workflows_to_finish(active), timeout=5
                    )
                except asyncio.TimeoutError:
                    try:
                        logger.warning("⚠️ Shutdown timeout — forcing exit")
                    except Exception:
                        pass

            # ------------------------
            # 🧹 CLEAN STATE
            # ------------------------
            try:
                self.workflows = {}
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.shutdown_count = getattr(self, "shutdown_count", 0) + 1
            except Exception:
                pass

            try:
                logger.info("✅ WorkflowEngine shutdown complete")
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Shutdown failed: {e}")
            except Exception:
                pass
