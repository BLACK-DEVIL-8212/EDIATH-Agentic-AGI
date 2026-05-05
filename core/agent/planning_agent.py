"""
Planning Agent for EDIATH
Specialized planning: hierarchical planning, resource allocation, dependency management, timeline optimization
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import defaultdict

try:
    from ..utils.logger import logger
except ImportError:
    logger = None


class PlanningHorizon(Enum):
    """Planning time horizons"""

    IMMEDIATE = "immediate"  # Next 1-2 actions (seconds to minutes)
    SHORT_TERM = "short_term"  # Next 5-10 actions (minutes to hours)
    MEDIUM_TERM = "medium_term"  # Today to this week (hours to days)
    LONG_TERM = "long_term"  # This week to this month (days to weeks)
    STRATEGIC = "strategic"  # Months to years


class PlanType(Enum):
    """Types of plans"""

    REACTIVE = "reactive"  # Respond to immediate input
    TACTICAL = "tactical"  # Short-term goal oriented
    OPERATIONAL = "operational"  # Medium-term execution
    STRATEGIC = "strategic"  # Long-term vision
    EMERGENCY = "emergency"  # Crisis response
    CONTINGENCY = "contingency"  # Backup plans


class ResourceType(Enum):
    """Resource types for planning"""

    TIME = "time"
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"
    API = "api"
    HUMAN = "human"
    FILE = "file"
    TOOL = "tool"


class DependencyType(Enum):
    """Dependency relationships"""

    FINISH_TO_START = "fs"  # Task B depends on task A finishing
    START_TO_START = "ss"  # Task B depends on task A starting
    FINISH_TO_FINISH = "ff"  # Task B depends on task A finishing
    START_TO_FINISH = "sf"  # Task B depends on task A starting


@dataclass
class Resource:
    """Resource definition"""

    type: ResourceType
    name: str
    quantity: float = 1.0
    unit: str = "units"
    cost: float = 0.0


@dataclass
class Dependency:
    """Task dependency"""

    depends_on: str  # Task ID
    type: DependencyType = DependencyType.FINISH_TO_START
    lag: int = 0  # Lag time in seconds


@dataclass
class Task:
    """Planning task with metadata"""

    id: str
    name: str
    description: str
    horizon: PlanningHorizon
    estimated_duration: int  # seconds
    priority: int  # 1-10, lower is higher priority
    dependencies: List[Dependency]
    resources: List[Resource]
    required_tools: List[str]
    expected_output: str
    fallback_task_id: Optional[str] = None
    deadline: Optional[datetime] = None
    status: str = "pending"
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Milestone:
    """Project milestone"""

    id: str
    name: str
    target_date: datetime
    tasks: List[str]  # Task IDs
    completed: bool = False
    completed_at: Optional[datetime] = None


@dataclass
class Plan:
    """Complete plan with all components"""

    id: str
    name: str
    description: str
    type: PlanType
    horizon: PlanningHorizon
    created_at: datetime
    updated_at: datetime
    tasks: List[Task]
    milestones: List[Milestone]
    resources: Dict[str, Resource]
    status: str = "active"  # active, paused, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanningAgent:
    """
    Specialized planning agent capable of:
    - Hierarchical planning (strategic → operational → tactical)
    - Resource allocation and optimization
    - Dependency management and critical path analysis
    - Timeline estimation and scheduling
    - Contingency planning
    - Multi-agent coordination
    - Plan validation and simulation
    - Progress tracking and reporting
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Planning Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Planning limits
        self.max_tasks_per_plan = self.config.get("max_tasks_per_plan", 100)
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 10)
        self.default_task_timeout = self.config.get(
            "default_task_timeout", 3600
        )  # 1 hour

        # Resource pools
        self.resource_pools: Dict[str, Dict] = {}
        self._init_resource_pools()

        # Active plans
        self.active_plans: Dict[str, Plan] = {}
        self.plan_history: List[Plan] = []
        self.max_history = self.config.get("max_history", 100)

        # Task execution
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_callbacks: Dict[str, List[callable]] = {}

        # Statistics
        self.stats = {
            "plans_created": 0,
            "plans_completed": 0,
            "plans_failed": 0,
            "tasks_executed": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "average_planning_time": 0.0,
        }

        # Cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.plan_cache: Dict[str, tuple] = {}

        self.logger.info("Planning Agent initialized")

    def _init_resource_pools(self):
        """Initialize resource pools from config"""
        self.resource_pools = {
            "compute": {
                "available": self.config.get("compute_units", 10),
                "max": 10,
                "unit": "cores",
            },
            "memory": {
                "available": self.config.get("memory_units", 16),
                "max": 16,
                "unit": "GB",
            },
            "api": {
                "available": self.config.get("api_calls", 1000),
                "max": 1000,
                "unit": "calls/hour",
            },
            "time": {
                "available": self.config.get("time_budget", 3600),
                "max": 3600,
                "unit": "seconds",
            },
        }

    def _generate_id(self, prefix: str = "plan") -> str:
        """Generate unique ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    # ============
    # Plan Creation and Management
    # ============

    async def create_plan(
        self,
        name: str,
        description: str,
        plan_type: PlanType,
        horizon: PlanningHorizon,
        goals: List[str],
        constraints: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Create a new plan from goals

        Args:
            name: Plan name
            description: Plan description
            plan_type: Type of plan
            horizon: Planning horizon
            goals: List of goals to achieve
            constraints: Resource and time constraints

        Returns:
            Dictionary with plan creation result
        """
        start_time = datetime.now()

        try:
            # Generate plan ID
            plan_id = self._generate_id("plan")

            # Decompose goals into tasks
            tasks = await self._decompose_goals(goals, horizon, constraints)

            # Identify dependencies
            tasks = self._identify_dependencies(tasks)

            # Calculate timeline
            tasks = self._calculate_timeline(tasks)

            # Allocate resources
            tasks, resources = self._allocate_resources(tasks, constraints)

            # Create milestones
            milestones = self._create_milestones(tasks, horizon)

            # Create plan
            plan = Plan(
                id=plan_id,
                name=name,
                description=description,
                type=plan_type,
                horizon=horizon,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tasks=tasks,
                milestones=milestones,
                resources=resources,
                metadata={
                    "goals": goals,
                    "constraints": constraints or {},
                    "planning_time": (datetime.now() - start_time).total_seconds(),
                },
            )

            # Validate plan
            validation = await self.validate_plan(plan)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Plan validation failed: {validation['issues']}",
                    "plan_id": plan_id,
                }

            # Store plan
            self.active_plans[plan_id] = plan
            self.stats["plans_created"] += 1

            # Update statistics
            planning_time = (datetime.now() - start_time).total_seconds()
            self.stats["average_planning_time"] = (
                self.stats["average_planning_time"] * (self.stats["plans_created"] - 1)
                + planning_time
            ) / self.stats["plans_created"]

            self.logger.info(f"Plan created: {name} (ID: {plan_id})")

            return {
                "success": True,
                "plan_id": plan_id,
                "name": name,
                "total_tasks": len(tasks),
                "total_milestones": len(milestones),
                "estimated_duration": sum(t.estimated_duration for t in tasks),
                "planning_time": planning_time,
            }

        except Exception as e:
            self.logger.error(f"Plan creation error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _decompose_goals(
        self, goals: List[str], horizon: PlanningHorizon, constraints: Optional[Dict]
    ) -> List[Task]:
        """Decompose high-level goals into executable tasks"""
        tasks = []

        for i, goal in enumerate(goals):
            # Create root task for goal
            task_id = self._generate_id("task")

            # Estimate duration based on horizon
            duration_map = {
                PlanningHorizon.IMMEDIATE: (5, 30),
                PlanningHorizon.SHORT_TERM: (30, 300),
                PlanningHorizon.MEDIUM_TERM: (300, 3600),
                PlanningHorizon.LONG_TERM: (3600, 86400),
                PlanningHorizon.STRATEGIC: (86400, 604800),
            }
            min_dur, max_dur = duration_map.get(horizon, (30, 300))
            estimated_duration = (min_dur + max_dur) // 2

            # Determine required tools based on goal content
            required_tools = self._infer_tools(goal)

            task = Task(
                id=task_id,
                name=f"Task_{i+1}: {goal[:50]}",
                description=goal,
                horizon=horizon,
                estimated_duration=estimated_duration,
                priority=5,  # Default medium priority
                dependencies=[],
                resources=[],
                required_tools=required_tools,
                expected_output=f"Completion of: {goal[:50]}",
            )
            tasks.append(task)

            # Further decompose if needed
            if horizon in [PlanningHorizon.LONG_TERM, PlanningHorizon.STRATEGIC]:
                sub_tasks = await self._decompose_goal(goal, horizon)
                tasks.extend(sub_tasks)

        return tasks

    async def _decompose_goal(self, goal: str, horizon: PlanningHorizon) -> List[Task]:
        """Decompose a single goal into sub-tasks"""
        # This would use LLM or rules to decompose
        # For now, create generic sub-tasks
        sub_tasks = []

        # Simple decomposition based on keywords
        if "research" in goal.lower() or "learn" in goal.lower():
            sub_tasks.append(
                Task(
                    id=self._generate_id("task"),
                    name="Gather information",
                    description=f"Research phase for: {goal}",
                    horizon=horizon,
                    estimated_duration=300,
                    priority=5,
                    dependencies=[],
                    resources=[],
                    required_tools=["search", "browser"],
                    expected_output="Research findings",
                )
            )
            sub_tasks.append(
                Task(
                    id=self._generate_id("task"),
                    name="Synthesize findings",
                    description=f"Synthesize research for: {goal}",
                    horizon=horizon,
                    estimated_duration=180,
                    priority=5,
                    dependencies=[Dependency(depends_on=sub_tasks[0].id)],
                    resources=[],
                    required_tools=["llm"],
                    expected_output="Synthesized insights",
                )
            )

        elif "build" in goal.lower() or "create" in goal.lower():
            sub_tasks.append(
                Task(
                    id=self._generate_id("task"),
                    name="Plan build",
                    description=f"Planning phase for: {goal}",
                    horizon=horizon,
                    estimated_duration=120,
                    priority=5,
                    dependencies=[],
                    resources=[],
                    required_tools=["planner"],
                    expected_output="Build plan",
                )
            )
            sub_tasks.append(
                Task(
                    id=self._generate_id("task"),
                    name="Execute build",
                    description=f"Execution phase for: {goal}",
                    horizon=horizon,
                    estimated_duration=600,
                    priority=4,
                    dependencies=[Dependency(depends_on=sub_tasks[0].id)],
                    resources=[],
                    required_tools=["code", "file"],
                    expected_output="Built artifact",
                )
            )

        return sub_tasks

    def _infer_tools(self, goal: str) -> List[str]:
        """Infer required tools from goal description"""
        tools = []
        goal_lower = goal.lower()

        tool_keywords = {
            "search": ["search", "find", "look up", "research"],
            "browser": ["browse", "web", "website", "url", "http"],
            "file": ["file", "document", "read", "write", "save"],
            "code": ["code", "program", "script", "function", "class"],
            "llm": ["analyze", "summarize", "explain", "describe"],
            "database": ["database", "sql", "query", "store"],
            "email": ["email", "send", "message"],
            "notification": ["notify", "alert", "remind"],
            "shell": ["terminal", "command", "run", "execute"],
        }

        for tool, keywords in tool_keywords.items():
            if any(kw in goal_lower for kw in keywords):
                tools.append(tool)

        return tools or ["llm"]  # Default to LLM

    def _identify_dependencies(self, tasks: List[Task]) -> List[Task]:
        """Identify dependencies between tasks"""
        # Simple dependency inference based on task names and descriptions
        task_map = {task.id: task for task in tasks}

        for i, task in enumerate(tasks):
            for j, other in enumerate(tasks):
                if i != j:
                    # Check if task depends on other
                    if self._has_dependency(task.description, other.description):
                        task.dependencies.append(
                            Dependency(
                                depends_on=other.id, type=DependencyType.FINISH_TO_START
                            )
                        )

        # Remove duplicates
        for task in tasks:
            unique_deps = {}
            for dep in task.dependencies:
                unique_deps[dep.depends_on] = dep
            task.dependencies = list(unique_deps.values())

        return tasks

    def _has_dependency(self, task_desc: str, other_desc: str) -> bool:
        """Check if task depends on other based on descriptions"""
        task_lower = task_desc.lower()
        other_lower = other_desc.lower()

        # Keywords indicating dependency
        dependency_indicators = [
            "based on",
            "using",
            "from",
            "after",
            "then",
            "subsequently",
            "requires",
            "needs",
            "depends on",
        ]

        for indicator in dependency_indicators:
            if indicator in task_lower and other_lower.split()[0] in task_lower:
                return True

        return False

    def _calculate_timeline(self, tasks: List[Task]) -> List[Task]:
        """Calculate start and end times based on dependencies"""
        task_map = {task.id: task for task in tasks}

        # Topological sort
        visited = set()
        order = []

        def dfs(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map[task_id]
            for dep in task.dependencies:
                if dep.depends_on in task_map:
                    dfs(dep.depends_on)
            order.append(task_id)

        for task in tasks:
            if task.id not in visited:
                dfs(task.id)

        # Calculate times
        task_times = {}
        for task_id in order:
            task = task_map[task_id]
            max_dep_time = 0
            for dep in task.dependencies:
                if dep.depends_on in task_times:
                    dep_end = task_times[dep.depends_on][1]
                    max_dep_time = max(max_dep_time, dep_end + dep.lag)

            start_time = max_dep_time
            end_time = start_time + task.estimated_duration
            task_times[task_id] = (start_time, end_time)

        # Store times in task metadata
        for task in tasks:
            if task.id in task_times:
                task.metadata["start_time"] = task_times[task.id][0]
                task.metadata["end_time"] = task_times[task.id][1]

        return tasks

    def _allocate_resources(
        self, tasks: List[Task], constraints: Optional[Dict]
    ) -> Tuple[List[Task], Dict[str, Resource]]:
        """Allocate resources to tasks"""
        resources = {}

        # Track resource usage over time
        resource_usage = defaultdict(list)

        for task in tasks:
            # Allocate resources based on task type
            task_resources = []

            # Time resource
            task_resources.append(
                Resource(
                    type=ResourceType.TIME,
                    name="time",
                    quantity=task.estimated_duration,
                    unit="seconds",
                )
            )

            # Compute resource
            if "code" in task.required_tools:
                task_resources.append(
                    Resource(
                        type=ResourceType.COMPUTE,
                        name="cpu",
                        quantity=1.0,
                        unit="cores",
                    )
                )

            # API resource
            if any(t in task.required_tools for t in ["search", "browser"]):
                task_resources.append(
                    Resource(
                        type=ResourceType.API,
                        name="api_calls",
                        quantity=10,
                        unit="calls",
                    )
                )

            task.resources = task_resources

            # Add to resource pool
            for res in task_resources:
                key = f"{res.type.value}_{res.name}"
                if key not in resources:
                    resources[key] = res

        return tasks, resources

    def _create_milestones(
        self, tasks: List[Task], horizon: PlanningHorizon
    ) -> List[Milestone]:
        """Create milestones from tasks"""
        milestones = []

        # Create milestone for every 25% of tasks
        milestone_interval = max(1, len(tasks) // 4)

        for i in range(1, 5):
            milestone_idx = min(i * milestone_interval, len(tasks)) - 1
            if milestone_idx >= 0:
                milestone_tasks = [t.id for t in tasks[: milestone_idx + 1]]
                milestone = Milestone(
                    id=self._generate_id("ms"),
                    name=f"Milestone {i}: {i*25}% Complete",
                    target_date=datetime.now()
                    + timedelta(
                        seconds=sum(
                            t.estimated_duration for t in tasks[: milestone_idx + 1]
                        )
                    ),
                    tasks=milestone_tasks,
                )
                milestones.append(milestone)

        return milestones

    async def validate_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Validate a plan for feasibility

        Args:
            plan: Plan to validate

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        # Check task count
        if len(plan.tasks) > self.max_tasks_per_plan:
            issues.append(
                f"Too many tasks: {len(plan.tasks)} > {self.max_tasks_per_plan}"
            )

        # Check for circular dependencies
        if self._has_circular_dependencies(plan.tasks):
            issues.append("Circular dependencies detected")

        # Check resource availability
        resource_shortages = self._check_resources(plan)
        if resource_shortages:
            warnings.extend(resource_shortages)

        # Check deadlines
        missed_deadlines = self._check_deadlines(plan)
        if missed_deadlines:
            warnings.extend(missed_deadlines)

        # Check tool availability
        missing_tools = self._check_tools(plan)
        if missing_tools:
            warnings.append(f"Missing tools: {missing_tools}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "severity": "error" if issues else ("warning" if warnings else "success"),
        }

    def _has_circular_dependencies(self, tasks: List[Task]) -> bool:
        """Check for circular dependencies in task graph"""
        task_map = {task.id: task for task in tasks}
        visited = set()
        recursion_stack = set()

        def has_cycle(task_id):
            visited.add(task_id)
            recursion_stack.add(task_id)

            task = task_map.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep.depends_on not in task_map:
                        continue
                    if dep.depends_on not in visited:
                        if has_cycle(dep.depends_on):
                            return True
                    elif dep.depends_on in recursion_stack:
                        return True

            recursion_stack.remove(task_id)
            return False

        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    return True

        return False

    def _check_resources(self, plan: Plan) -> List[str]:
        """Check resource availability"""
        shortages = []

        # Group resources by type
        resource_demand = defaultdict(float)
        for task in plan.tasks:
            for resource in task.resources:
                key = f"{resource.type.value}_{resource.name}"
                resource_demand[key] += resource.quantity

        # Check against pools
        for key, demand in resource_demand.items():
            if key in self.resource_pools:
                available = self.resource_pools[key]["available"]
                if demand > available:
                    shortages.append(f"Resource {key}: need {demand}, have {available}")

        return shortages

    def _check_deadlines(self, plan: Plan) -> List[str]:
        """Check if deadlines are achievable"""
        missed = []
        total_duration = sum(t.estimated_duration for t in plan.tasks)

        for milestone in plan.milestones:
            milestone_duration = sum(
                t.estimated_duration for t in plan.tasks if t.id in milestone.tasks
            )
            estimated_completion = datetime.now() + timedelta(
                seconds=milestone_duration
            )

            if estimated_completion > milestone.target_date:
                missed.append(
                    f"Milestone '{milestone.name}' may be missed: "
                    f"estimated {estimated_completion}, target {milestone.target_date}"
                )

        return missed

    def _check_tools(self, plan: Plan) -> List[str]:
        """Check if required tools are available"""
        # This would check against available agents/tools
        # For now, assume all tools are available
        return []

    # ============
    # Plan Execution
    # ============

    async def execute_plan(
        self, plan_id: str, callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan

        Args:
            plan_id: ID of plan to execute
            callback: Progress callback function

        Returns:
            Dictionary with execution result
        """
        if plan_id not in self.active_plans:
            return {"success": False, "error": f"Plan {plan_id} not found"}

        plan = self.active_plans[plan_id]

        if plan.status != "active":
            return {"success": False, "error": f"Plan {plan_id} is {plan.status}"}

        self.logger.info(f"Executing plan: {plan.name}")

        # Get execution order
        execution_order = self._get_execution_order(plan.tasks)

        results = []
        completed_tasks = set()

        for task_id in execution_order:
            task = next((t for t in plan.tasks if t.id == task_id), None)
            if not task:
                continue

            # Check if dependencies are met
            deps_met = all(d.depends_on in completed_tasks for d in task.dependencies)
            if not deps_met:
                continue

            # Execute task
            self.logger.info(f"Executing task: {task.name}")
            task.status = "running"
            task.started_at = datetime.now()

            try:
                result = await self._execute_task(task)
                task.status = "completed"
                task.completed_at = datetime.now()
                task.result = result
                results.append({"task_id": task_id, "success": True, "result": result})
                completed_tasks.add(task_id)

                self.stats["tasks_executed"] += 1
                self.stats["tasks_succeeded"] += 1

            except Exception as e:
                self.logger.error(f"Task {task.name} failed: {str(e)}")
                task.status = "failed"
                task.error = str(e)
                results.append({"task_id": task_id, "success": False, "error": str(e)})
                self.stats["tasks_failed"] += 1

                # Check for fallback
                if task.fallback_task_id:
                    fallback = next(
                        (t for t in plan.tasks if t.id == task.fallback_task_id), None
                    )
                    if fallback:
                        self.logger.info(f"Executing fallback task: {fallback.name}")
                        fallback_result = await self._execute_task(fallback)
                        results.append(
                            {
                                "task_id": task.fallback_task_id,
                                "success": True,
                                "result": fallback_result,
                            }
                        )
                        completed_tasks.add(task.fallback_task_id)

            # Update progress
            if callback:
                progress = len(completed_tasks) / len(execution_order) * 100
                if asyncio.iscoroutinefunction(callback):
                    await callback(plan_id, progress, task.name, task.status)
                else:
                    callback(plan_id, progress, task.name, task.status)

            # Check if plan should continue
            if task.status == "failed" and not task.fallback_task_id:
                plan.status = "failed"
                self.stats["plans_failed"] += 1
                break

        # Check if all tasks completed
        if all(t.status == "completed" for t in plan.tasks):
            plan.status = "completed"
            self.stats["plans_completed"] += 1

            # Update milestones
            for milestone in plan.milestones:
                milestone_tasks = set(milestone.tasks)
                completed_milestone_tasks = {
                    t.id for t in plan.tasks if t.status == "completed"
                }
                if milestone_tasks.issubset(completed_milestone_tasks):
                    milestone.completed = True
                    milestone.completed_at = datetime.now()

        plan.updated_at = datetime.now()

        # Move to history
        self.plan_history.append(plan)
        if len(self.plan_history) > self.max_history:
            self.plan_history = self.plan_history[-self.max_history :]

        # Remove from active if completed or failed
        if plan.status in ["completed", "failed"]:
            del self.active_plans[plan_id]

        return {
            "success": plan.status == "completed",
            "plan_id": plan_id,
            "plan_name": plan.name,
            "status": plan.status,
            "tasks_completed": len([t for t in plan.tasks if t.status == "completed"]),
            "total_tasks": len(plan.tasks),
            "results": results,
        }

    def _get_execution_order(self, tasks: List[Task]) -> List[str]:
        """Get topological order for task execution"""
        task_map = {task.id: task for task in tasks}
        in_degree = defaultdict(int)
        adj = defaultdict(list)

        for task in tasks:
            for dep in task.dependencies:
                if dep.depends_on in task_map:
                    adj[dep.depends_on].append(task.id)
                    in_degree[task.id] += 1

        # Kahn's algorithm
        queue = [task.id for task in tasks if in_degree[task.id] == 0]
        order = []

        while queue:
            # Sort by priority (higher priority = lower number)
            queue.sort(key=lambda tid: task_map[tid].priority)
            task_id = queue.pop(0)
            order.append(task_id)

            for neighbor in adj[task_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    async def _execute_task(self, task: Task) -> Any:
        """Execute a single task"""
        # This would delegate to appropriate agent
        # For now, simulate execution
        await asyncio.sleep(
            min(task.estimated_duration, 5)
        )  # Cap at 5 seconds for testing

        return {
            "task_id": task.id,
            "task_name": task.name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        }

    # ============
    # Plan Analysis and Reporting
    # ============

    def get_critical_path(self, plan_id: str) -> Dict[str, Any]:
        """Get critical path of a plan"""
        if plan_id not in self.active_plans:
            plan = next((p for p in self.plan_history if p.id == plan_id), None)
            if not plan:
                return {"success": False, "error": f"Plan {plan_id} not found"}
        else:
            plan = self.active_plans[plan_id]

        # Calculate earliest start and finish times
        task_map = {task.id: task for task in plan.tasks}
        earliest_start = {task.id: 0 for task in plan.tasks}
        earliest_finish = {task.id: 0 for task in plan.tasks}

        order = self._get_execution_order(plan.tasks)

        for task_id in order:
            task = task_map[task_id]
            max_dep_finish = 0
            for dep in task.dependencies:
                if dep.depends_on in earliest_finish:
                    max_dep_finish = max(
                        max_dep_finish, earliest_finish[dep.depends_on] + dep.lag
                    )
            earliest_start[task_id] = max_dep_finish
            earliest_finish[task_id] = earliest_start[task_id] + task.estimated_duration

        # Calculate latest start and finish times (backward pass)
        latest_finish = {task.id: max(earliest_finish.values()) for task in plan.tasks}
        latest_start = {task.id: 0 for task in plan.tasks}

        for task_id in reversed(order):
            task = task_map[task_id]
            min_successor_start = latest_finish[task_id]
            for other in plan.tasks:
                for dep in other.dependencies:
                    if dep.depends_on == task_id:
                        successor_start = latest_start[other.id] - dep.lag
                        min_successor_start = min(min_successor_start, successor_start)
            latest_finish[task_id] = min_successor_start
            latest_start[task_id] = latest_finish[task_id] - task.estimated_duration

        # Identify critical path (tasks where slack = 0)
        critical_tasks = []
        for task in plan.tasks:
            slack = latest_start[task.id] - earliest_start[task.id]
            if slack == 0:
                critical_tasks.append(
                    {
                        "id": task.id,
                        "name": task.name,
                        "duration": task.estimated_duration,
                        "start": earliest_start[task.id],
                        "finish": earliest_finish[task.id],
                    }
                )

        return {
            "success": True,
            "plan_id": plan_id,
            "total_duration": max(earliest_finish.values()),
            "critical_path_length": len(critical_tasks),
            "critical_tasks": critical_tasks,
        }

    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """Get progress of a plan"""
        if plan_id not in self.active_plans:
            plan = next((p for p in self.plan_history if p.id == plan_id), None)
            if not plan:
                return {"success": False, "error": f"Plan {plan_id} not found"}
        else:
            plan = self.active_plans[plan_id]

        completed_tasks = [t for t in plan.tasks if t.status == "completed"]
        failed_tasks = [t for t in plan.tasks if t.status == "failed"]
        running_tasks = [t for t in plan.tasks if t.status == "running"]

        progress = len(completed_tasks) / len(plan.tasks) * 100 if plan.tasks else 0

        # Calculate estimated completion
        remaining_duration = sum(
            t.estimated_duration
            for t in plan.tasks
            if t.status not in ["completed", "failed"]
        )
        estimated_completion = datetime.now() + timedelta(seconds=remaining_duration)

        return {
            "success": True,
            "plan_id": plan_id,
            "plan_name": plan.name,
            "status": plan.status,
            "progress_percent": progress,
            "tasks": {
                "total": len(plan.tasks),
                "completed": len(completed_tasks),
                "running": len(running_tasks),
                "failed": len(failed_tasks),
                "pending": len(plan.tasks)
                - len(completed_tasks)
                - len(running_tasks)
                - len(failed_tasks),
            },
            "milestones": [
                {
                    "name": m.name,
                    "completed": m.completed,
                    "target_date": m.target_date.isoformat(),
                    "completed_at": (
                        m.completed_at.isoformat() if m.completed_at else None
                    ),
                }
                for m in plan.milestones
            ],
            "estimated_completion": estimated_completion.isoformat(),
        }

    def get_resource_utilization(self, plan_id: str) -> Dict[str, Any]:
        """Get resource utilization for a plan"""
        if plan_id not in self.active_plans:
            plan = next((p for p in self.plan_history if p.id == plan_id), None)
            if not plan:
                return {"success": False, "error": f"Plan {plan_id} not found"}
        else:
            plan = self.active_plans[plan_id]

        utilization = {}

        for resource_key, resource in plan.resources.items():
            # Calculate total demand
            total_demand = 0
            for task in plan.tasks:
                for task_resource in task.resources:
                    task_key = f"{task_resource.type.value}_{task_resource.name}"
                    if task_key == resource_key:
                        total_demand += task_resource.quantity

            # Calculate used (for completed tasks)
            used = 0
            for task in plan.tasks:
                if task.status == "completed":
                    for task_resource in task.resources:
                        task_key = f"{task_resource.type.value}_{task_resource.name}"
                        if task_key == resource_key:
                            used += task_resource.quantity

            utilization[resource_key] = {
                "type": resource.type.value,
                "name": resource.name,
                "total_demand": total_demand,
                "used": used,
                "remaining": total_demand - used,
                "unit": resource.unit,
            }

        return {"success": True, "plan_id": plan_id, "utilization": utilization}

    def get_schedule(self, plan_id: str) -> Dict[str, Any]:
        """Get detailed schedule for a plan"""
        if plan_id not in self.active_plans:
            plan = next((p for p in self.plan_history if p.id == plan_id), None)
            if not plan:
                return {"success": False, "error": f"Plan {plan_id} not found"}
        else:
            plan = self.active_plans[plan_id]

        schedule = []

        for task in plan.tasks:
            schedule.append(
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority,
                    "estimated_duration": task.estimated_duration,
                    "start_time": task.metadata.get("start_time"),
                    "end_time": task.metadata.get("end_time"),
                    "dependencies": [d.depends_on for d in task.dependencies],
                    "assigned_agent": task.assigned_agent,
                    "started_at": (
                        task.started_at.isoformat() if task.started_at else None
                    ),
                    "completed_at": (
                        task.completed_at.isoformat() if task.completed_at else None
                    ),
                    "error": task.error,
                }
            )

        return {"success": True, "plan_id": plan_id, "schedule": schedule}

    # ============
    # Contingency Planning
    # ============

    async def create_contingency_plan(
        self, plan_id: str, risk_scenario: str
    ) -> Dict[str, Any]:
        """
        Create a contingency plan for a risk scenario

        Args:
            plan_id: Original plan ID
            risk_scenario: Description of the risk

        Returns:
            Dictionary with contingency plan
        """
        if plan_id not in self.active_plans:
            return {"success": False, "error": f"Plan {plan_id} not found"}

        original_plan = self.active_plans[plan_id]

        # Identify affected tasks
        affected_tasks = []
        for task in original_plan.tasks:
            if self._is_task_affected(task, risk_scenario):
                affected_tasks.append(task)

        # Create contingency tasks
        contingency_tasks = []
        for task in affected_tasks:
            fallback_task = Task(
                id=self._generate_id("ctask"),
                name=f"Contingency: {task.name}",
                description=f"Fallback for {task.name} due to {risk_scenario}",
                horizon=task.horizon,
                estimated_duration=task.estimated_duration
                * 1.5,  # Fallback takes longer
                priority=task.priority + 1,  # Lower priority
                dependencies=task.dependencies.copy(),
                resources=task.resources.copy(),
                required_tools=task.required_tools.copy(),
                expected_output=task.expected_output,
                fallback_task_id=None,
            )
            contingency_tasks.append(fallback_task)
            task.fallback_task_id = fallback_task.id

        contingency_plan = Plan(
            id=self._generate_id("cplan"),
            name=f"Contingency: {original_plan.name}",
            description=f"Contingency plan for: {risk_scenario}",
            type=PlanType.CONTINGENCY,
            horizon=original_plan.horizon,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tasks=contingency_tasks,
            milestones=[],
            resources=original_plan.resources,
            metadata={
                "original_plan_id": plan_id,
                "risk_scenario": risk_scenario,
                "affected_tasks": [t.id for t in affected_tasks],
            },
        )

        return {
            "success": True,
            "contingency_plan_id": contingency_plan.id,
            "name": contingency_plan.name,
            "affected_tasks": len(affected_tasks),
            "contingency_tasks": len(contingency_tasks),
        }

    def _is_task_affected(self, task: Task, risk_scenario: str) -> bool:
        """Check if a task is affected by a risk scenario"""
        risk_lower = risk_scenario.lower()
        task_desc = task.description.lower()

        risk_keywords = {
            "timeout": ["timeout", "slow", "delay"],
            "resource": ["memory", "cpu", "resource", "out of memory"],
            "network": ["network", "connection", "offline", "api"],
            "data": ["data", "corrupt", "missing", "invalid"],
            "tool": ["tool", "agent", "unavailable", "failed"],
        }

        for risk_type, keywords in risk_keywords.items():
            if risk_type in risk_lower:
                if any(kw in task_desc for kw in keywords):
                    return True

        return False

    # ============
    # Utility Methods
    # ============

    def get_all_plans(self, include_history: bool = False) -> Dict[str, Any]:
        """Get all plans"""
        plans = []

        for plan in self.active_plans.values():
            plans.append(
                {
                    "id": plan.id,
                    "name": plan.name,
                    "type": plan.type.value,
                    "status": plan.status,
                    "created_at": plan.created_at.isoformat(),
                    "tasks_count": len(plan.tasks),
                }
            )

        if include_history:
            for plan in self.plan_history:
                plans.append(
                    {
                        "id": plan.id,
                        "name": plan.name,
                        "type": plan.type.value,
                        "status": plan.status,
                        "created_at": plan.created_at.isoformat(),
                        "completed_at": plan.updated_at.isoformat(),
                        "tasks_count": len(plan.tasks),
                    }
                )

        return {"success": True, "total_plans": len(plans), "plans": plans}

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        success_rate = (
            self.stats["tasks_succeeded"] / max(1, self.stats["tasks_executed"])
        ) * 100

        return {
            **self.stats,
            "task_success_rate": success_rate,
            "active_plans": len(self.active_plans),
            "history_size": len(self.plan_history),
            "resource_pools": self.resource_pools,
        }

    async def pause_plan(self, plan_id: str) -> Dict[str, Any]:
        """Pause an active plan"""
        if plan_id not in self.active_plans:
            return {"success": False, "error": f"Plan {plan_id} not found"}

        plan = self.active_plans[plan_id]
        plan.status = "paused"
        plan.updated_at = datetime.now()

        return {
            "success": True,
            "plan_id": plan_id,
            "message": f"Plan {plan.name} paused",
        }

    async def resume_plan(self, plan_id: str) -> Dict[str, Any]:
        """Resume a paused plan"""
        if plan_id not in self.active_plans:
            return {"success": False, "error": f"Plan {plan_id} not found"}

        plan = self.active_plans[plan_id]
        if plan.status != "paused":
            return {"success": False, "error": f"Plan {plan_id} is not paused"}

        plan.status = "active"
        plan.updated_at = datetime.now()

        # Resume execution
        asyncio.create_task(self.execute_plan(plan_id))

        return {
            "success": True,
            "plan_id": plan_id,
            "message": f"Plan {plan.name} resumed",
        }

    def clear_history(self):
        """Clear plan history"""
        self.plan_history.clear()
        self.logger.info("Plan history cleared")


# Integration wrapper for EDIATH
class PlanningAgentWrapper:
    """
    Wrapper class to integrate PlanningAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.planning_agent = PlanningAgent(config)
        self.agent_type = "planning"
        self.capabilities = [
            "create_plan",
            "execute_plan",
            "get_plan_progress",
            "get_critical_path",
            "create_contingency_plan",
            "get_schedule",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a planning request

        Request format:
        {
            'operation': 'create|execute|progress|critical_path|schedule|contingency|list|stats',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "create":
            plan_type = PlanType(request.get("plan_type", "tactical"))
            horizon = PlanningHorizon(request.get("horizon", "short_term"))

            return await self.planning_agent.create_plan(
                name=request.get("name"),
                description=request.get("description", ""),
                plan_type=plan_type,
                horizon=horizon,
                goals=request.get("goals", []),
                constraints=request.get("constraints"),
            )

        elif operation == "execute":
            return await self.planning_agent.execute_plan(
                plan_id=request.get("plan_id"),
                callback=None,  # Callbacks not supported via wrapper
            )

        elif operation == "progress":
            return self.planning_agent.get_plan_progress(plan_id=request.get("plan_id"))

        elif operation == "critical_path":
            return self.planning_agent.get_critical_path(plan_id=request.get("plan_id"))

        elif operation == "schedule":
            return self.planning_agent.get_schedule(plan_id=request.get("plan_id"))

        elif operation == "resources":
            return self.planning_agent.get_resource_utilization(
                plan_id=request.get("plan_id")
            )

        elif operation == "contingency":
            return await self.planning_agent.create_contingency_plan(
                plan_id=request.get("plan_id"),
                risk_scenario=request.get("risk_scenario"),
            )

        elif operation == "pause":
            return await self.planning_agent.pause_plan(plan_id=request.get("plan_id"))

        elif operation == "resume":
            return await self.planning_agent.resume_plan(plan_id=request.get("plan_id"))

        elif operation == "list":
            return self.planning_agent.get_all_plans(
                include_history=request.get("include_history", False)
            )

        elif operation == "stats":
            return self.planning_agent.get_stats()

        elif operation == "clear_history":
            self.planning_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "PlanningAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.planning_agent.get_stats(),
            "plan_types": [t.value for t in PlanType],
            "horizons": [h.value for h in PlanningHorizon],
        }


# Example usage and testing
async def test_planning_agent():
    """Test the planning agent functionality"""

    # Initialize agent
    agent = PlanningAgent()

    print("=== Planning Agent Test ===\n")

    # Test creating a plan
    print("1. Creating a Plan...")
    result = await agent.create_plan(
        name="Build EDIATH Assistant",
        description="Complete development of EDIATH AI assistant",
        plan_type=PlanType.OPERATIONAL,
        horizon=PlanningHorizon.MEDIUM_TERM,
        goals=[
            "Implement core agent system",
            "Add memory and persistence",
            "Create user interface",
            "Integrate all agents",
            "Testing and deployment",
        ],
    )

    if result["success"]:
        plan_id = result["plan_id"]
        print(f"   Plan created: {result['name']}")
        print(f"   Total tasks: {result['total_tasks']}")
        print(f"   Estimated duration: {result['estimated_duration']}s")

    # Test getting plan details
    print("\n2. Getting Plan Details...")
    progress = agent.get_plan_progress(plan_id)
    if progress["success"]:
        print(f"   Plan: {progress['plan_name']}")
        print(f"   Status: {progress['status']}")
        print(f"   Progress: {progress['progress_percent']:.1f}%")
        print(f"   Tasks: {progress['tasks']}")

    # Test critical path analysis
    print("\n3. Critical Path Analysis...")
    critical = agent.get_critical_path(plan_id)
    if critical["success"]:
        print(f"   Total duration: {critical['total_duration']}s")
        print(f"   Critical path length: {critical['critical_path_length']} tasks")
        for task in critical["critical_tasks"][:3]:
            print(f"     - {task['name']} ({task['duration']}s)")

    # Test schedule
    print("\n4. Schedule...")
    schedule = agent.get_schedule(plan_id)
    if schedule["success"]:
        print(f"   Total scheduled tasks: {len(schedule['schedule'])}")
        for task in schedule["schedule"][:3]:
            print(f"     - {task['name']}: Priority {task['priority']}")

    # Test contingency plan
    print("\n5. Creating Contingency Plan...")
    contingency = await agent.create_contingency_plan(
        plan_id, risk_scenario="resource shortage"
    )
    if contingency["success"]:
        print(f"   Contingency plan: {contingency['name']}")
        print(f"   Affected tasks: {contingency['affected_tasks']}")

    # Test resource utilization
    print("\n6. Resource Utilization...")
    resources = agent.get_resource_utilization(plan_id)
    if resources["success"]:
        print("   Resources:")
        for key, util in resources["utilization"].items():
            print(f"     - {key}: {util['used']}/{util['total_demand']} {util['unit']}")

    # Get statistics
    print("\n7. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Plans created: {stats['plans_created']}")
    print(f"   Tasks executed: {stats['tasks_executed']}")
    print(f"   Task success rate: {stats['task_success_rate']:.1f}%")
    print(f"   Active plans: {stats['active_plans']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_planning_agent())
