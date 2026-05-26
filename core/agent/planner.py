"""
Production Planner for EDIATH R_AGENT (FINAL)
✔ Input validation
✔ Structured multi-step planning
✔ Tool-aware steps
✔ Priority handling
✔ Safe fallback
✔ Future extensibility (task chains, multi-agent)
✔ Logging support
✔ Planning Hierarchy - Long-term vs Short-term planning layers
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

try:
    from ..utils.logger import logger
except:
    logger = None


class PlanningHorizon(Enum):
    """Planning time horizons"""

    IMMEDIATE = "immediate"  # Next 1-2 actions
    SHORT_TERM = "short_term"  # Next 5-10 minutes
    MEDIUM_TERM = "medium_term"  # Next hour to day
    LONG_TERM = "long_term"  # Days to weeks
    STRATEGIC = "strategic"  # Weeks to months


class PlanType(Enum):
    """Types of plans"""

    REACTIVE = "reactive"  # Respond to immediate input
    TACTICAL = "tactical"  # Short-term goal oriented
    OPERATIONAL = "operational"  # Medium-term execution
    STRATEGIC = "strategic"  # Long-term vision
    EMERGENCY = "emergency"  # Crisis response


class PlanStep:
    """Individual planning step with metadata"""

    def __init__(
        self,
        action: str,
        step_type: str = "llm",
        horizon: PlanningHorizon = PlanningHorizon.IMMEDIATE,
        dependencies: List[str] = None,
        estimated_duration: int = 5,
        priority: int = 5,
    ):
        self.action = action
        self.step_type = step_type
        self.horizon = horizon
        self.dependencies = dependencies or []
        self.estimated_duration = estimated_duration  # seconds
        self.priority = priority  # 1-10, lower is higher priority
        self.status = "pending"
        self.result = None
        self.error = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "type": self.step_type,
            "horizon": self.horizon.value,
            "dependencies": self.dependencies,
            "estimated_duration": self.estimated_duration,
            "priority": self.priority,
            "status": self.status,
        }


class LongTermGoal:
    """Long-term goal tracking"""

    def __init__(
        self,
        name: str,
        description: str,
        target_date: datetime = None,
        priority: int = 5,
    ):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.target_date = target_date or (datetime.now() + timedelta(days=7))
        self.priority = priority
        self.progress = 0.0  # 0-100
        self.sub_goals: List[Dict[str, Any]] = []
        self.completed = False
        self.metrics: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "target_date": self.target_date.isoformat(),
            "priority": self.priority,
            "progress": self.progress,
            "sub_goals": self.sub_goals,
            "completed": self.completed,
        }


class Planner:
    def __init__(self):
        try:
            # 🔥 safety limits
            self.max_steps = 10
            self.max_context_length = 500
            self.max_long_term_goals = 10
            self.max_short_term_tasks = 20

            # 🔥 planning modes (future-ready)
            self.enable_multi_step = True
            self.enable_tool_chaining = True
            self.enable_hierarchical_planning = True  # NEW

            # 🔥 priority levels
            self.default_priority = "normal"
            self.default_route = "llm"  # FIXED: Added missing default_route attribute

            # 🔥 Planning Hierarchy (NEW)
            self.long_term_goals: List[LongTermGoal] = []
            self.short_term_tasks: List[PlanStep] = []
            self.active_plan: Optional[Dict[str, Any]] = None
            self.plan_history: List[Dict[str, Any]] = []

            # 🔥 Time horizons
            self.current_horizon = PlanningHorizon.IMMEDIATE
            self.strategic_direction: Optional[str] = None

            # 🔥 Memory integration
            self.memory_cache: Dict[str, Any] = {}

            # 🔥 Performance tracking
            self.plans_executed = 0
            self.successful_plans = 0
            self.failed_plans = 0

            # 🔥 debug flag
            self.debug = False

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Planner Init Error] {e}")

            # 🔥 fallback safe defaults
            self.max_steps = 5
            self.max_context_length = 500
            self.enable_multi_step = False
            self.enable_tool_chaining = False
            self.enable_hierarchical_planning = False
            self.default_priority = "normal"
            self.default_route = "llm"  # FIXED: Added missing default_route attribute
            self.debug = False
            self.long_term_goals = []
            self.short_term_tasks = []
            self.plan_history = []
            self.current_horizon = PlanningHorizon.IMMEDIATE

    # ------------------------
    # HIERARCHICAL PLANNING (NEW)
    # ------------------------
    def add_long_term_goal(
        self,
        name: str,
        description: str,
        target_date: datetime = None,
        priority: int = 5,
    ) -> bool:
        """Add a long-term strategic goal"""
        try:
            if len(self.long_term_goals) >= self.max_long_term_goals:
                # Remove lowest priority completed goal
                self.long_term_goals.sort(key=lambda x: x.priority)
                if self.long_term_goals and self.long_term_goals[0].completed:
                    self.long_term_goals.pop(0)
                else:
                    return False

            goal = LongTermGoal(name, description, target_date, priority)
            self.long_term_goals.append(goal)

            if logger:
                logger.info(f"[Planner] Long-term goal added: {name}")
            return True

        except Exception as e:
            if logger:
                logger.warning(f"[Planner] Add long-term goal error: {e}")
            return False

    def update_long_term_progress(self, goal_name: str, progress: float):
        """Update progress on a long-term goal"""
        for goal in self.long_term_goals:
            if goal.name == goal_name:
                goal.progress = min(100.0, max(0.0, progress))
                if goal.progress >= 100.0:
                    goal.completed = True
                    if logger:
                        logger.info(f"[Planner] Long-term goal completed: {goal_name}")
                break

    def add_short_term_task(self, step: PlanStep) -> bool:
        """Add a short-term task to the queue"""
        try:
            if len(self.short_term_tasks) >= self.max_short_term_tasks:
                # Remove oldest pending task
                self.short_term_tasks.pop(0)

            self.short_term_tasks.append(step)

            # Sort by priority (lower number = higher priority)
            self.short_term_tasks.sort(key=lambda x: x.priority)

            return True
        except Exception as e:
            if logger:
                logger.warning(f"[Planner] Add short-term task error: {e}")
            return False

    def get_next_task(self) -> Optional[PlanStep]:
        """Get the next highest priority task"""
        if not self.short_term_tasks:
            return None

        # Check dependencies
        for i, task in enumerate(self.short_term_tasks):
            deps_met = True
            for dep in task.dependencies:
                # Check if dependency is completed
                dep_found = False
                for completed in self.plan_history:
                    if completed.get("steps"):
                        for step in completed["steps"]:
                            if (
                                step.get("action") == dep
                                and step.get("status") == "completed"
                            ):
                                dep_found = True
                                break
                if not dep_found:
                    deps_met = False
                    break

            if deps_met:
                return self.short_term_tasks.pop(i)

        return None

    def get_long_term_goals_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all long-term goals"""
        return [g.to_dict() for g in self.long_term_goals]

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get active (non-completed) long-term goals"""
        return [g.to_dict() for g in self.long_term_goals if not g.completed]

    def get_strategic_plan(self) -> Dict[str, Any]:
        """Generate strategic plan based on long-term goals"""
        active_goals = [g for g in self.long_term_goals if not g.completed]

        if not active_goals:
            return {"has_strategy": False, "message": "No active long-term goals"}

        # Prioritize goals
        active_goals.sort(key=lambda x: x.priority)

        # Calculate overall progress
        total_progress = sum(g.progress for g in active_goals) / len(active_goals)

        # Generate next steps
        next_steps = []
        for goal in active_goals[:3]:
            if goal.progress < 100:
                next_steps.append(
                    {
                        "goal": goal.name,
                        "current_progress": goal.progress,
                        "remaining": 100 - goal.progress,
                        "target_date": goal.target_date.isoformat(),
                    }
                )

        return {
            "has_strategy": True,
            "active_goals_count": len(active_goals),
            "overall_progress": round(total_progress, 1),
            "next_steps": next_steps,
            "strategic_direction": self.strategic_direction,
        }

    # ------------------------
    # MAIN PLAN FUNCTION (ENHANCED)
    # ------------------------
    def plan(
        self, route: str, context: str, horizon: PlanningHorizon = None
    ) -> Dict[str, Any]:
        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not isinstance(route, str) or not route.strip():
                return self._fallback("Invalid route")

            if not isinstance(context, str) or not context.strip():
                return self._fallback("Invalid context")

            route = route.strip().lower()
            context = context.strip()[: getattr(self, "max_context_length", 500)]

            # Set planning horizon
            if horizon:
                self.current_horizon = horizon
            elif self.enable_hierarchical_planning:
                # Auto-detect horizon based on context
                self.current_horizon = self._detect_horizon(context)

            # ------------------------
            # 🔥 BUILD STEPS WITH HIERARCHY
            # ------------------------
            steps = self._build_steps_with_hierarchy(route, context)

            if not isinstance(steps, list) or not steps:
                return self._fallback("Invalid steps")

            # 🔥 enforce safety limit
            steps = steps[: getattr(self, "max_steps", 10)]

            # ------------------------
            # 🔥 META INFO WITH HIERARCHY
            # ------------------------
            meta = {
                "step_count": len(steps),
                "priority": self._get_priority(route),
                "created_by": "planner_v4_hierarchical",
                "safe": True,
                "horizon": self.current_horizon.value,
                "plan_type": self._determine_plan_type(route, steps),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }

            # Add strategic context if available
            if self.long_term_goals:
                meta["strategic_context"] = {
                    "active_goals": len(
                        [g for g in self.long_term_goals if not g.completed]
                    ),
                    "overall_progress": sum(g.progress for g in self.long_term_goals)
                    / len(self.long_term_goals),
                }

            # ------------------------
            # 🔥 CREATE PLAN
            # ------------------------
            plan = {"route": route, "steps": steps, "meta": meta}

            # Store active plan
            self.active_plan = plan

            # Add to history
            self.plan_history.append(plan)
            if len(self.plan_history) > 50:
                self.plan_history = self.plan_history[-50:]

            self.plans_executed += 1

            # ------------------------
            # 🔥 DEBUG LOG
            # ------------------------
            if getattr(self, "debug", False) and "logger" in globals() and logger:
                logger.debug(
                    f"[Planner] Route={route} Horizon={self.current_horizon.value} Steps={len(steps)}"
                )

            return plan

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Planner Error] {e}")
            return self._fallback(f"Planner error: {str(e)}")

    def _detect_horizon(self, context: str) -> PlanningHorizon:
        """Auto-detect planning horizon from context"""
        context_lower = context.lower()

        # Strategic/long-term indicators
        strategic_keywords = [
            "strategy",
            "vision",
            "mission",
            "quarter",
            "year",
            "month",
            "long term",
        ]
        long_term_keywords = [
            "project",
            "build",
            "develop",
            "create",
            "learn",
            "master",
        ]
        medium_term_keywords = ["today", "this week", "complete", "finish", "implement"]
        short_term_keywords = ["now", "immediate", "urgent", "quick", "fast", "soon"]

        if any(k in context_lower for k in strategic_keywords):
            return PlanningHorizon.STRATEGIC
        elif any(k in context_lower for k in long_term_keywords):
            return PlanningHorizon.LONG_TERM
        elif any(k in context_lower for k in medium_term_keywords):
            return PlanningHorizon.MEDIUM_TERM
        elif any(k in context_lower for k in short_term_keywords):
            return PlanningHorizon.SHORT_TERM
        else:
            return PlanningHorizon.IMMEDIATE

    def _determine_plan_type(self, route: str, steps: List[Dict]) -> str:
        """Determine plan type based on route and steps"""
        if route == "emergency":
            return PlanType.EMERGENCY.value
        elif len(steps) > 5:
            return PlanType.STRATEGIC.value
        elif len(steps) > 2:
            return PlanType.OPERATIONAL.value
        elif any(step.get("type") == "tool" for step in steps):
            return PlanType.TACTICAL.value
        else:
            return PlanType.REACTIVE.value

    # ------------------------
    # STEP BUILDER WITH HIERARCHY (ENHANCED)
    # ------------------------
    def _build_steps_with_hierarchy(
        self, route: str, context: str
    ) -> List[Dict[str, Any]]:
        """Build steps with hierarchical planning awareness"""
        try:
            import re

            route = str(route).strip().lower()
            context = str(context).strip()
            lower = context.lower()

            steps: List[Dict[str, Any]] = []

            # Check for long-term goal creation
            if "goal" in lower or "plan to" in lower or "want to" in lower:
                # Extract potential goal name
                goal_match = re.search(
                    r"(?:goal|plan to|want to) (?:is to|:)?\s*(.+?)(?:[.!?]|$)", context
                )
                if goal_match:
                    goal_name = goal_match.group(1).strip()[:50]
                    if goal_name:
                        self.add_long_term_goal(
                            name=goal_name, description=context[:200], priority=3
                        )
                        steps.append(
                            {
                                "type": "system",
                                "action": "acknowledge_goal",
                                "input": f"Long-term goal added: {goal_name}",
                            }
                        )

            # Check for strategic planning
            if self.current_horizon in [
                PlanningHorizon.LONG_TERM,
                PlanningHorizon.STRATEGIC,
            ]:
                steps.append(
                    {
                        "type": "planning",
                        "action": "strategic_assessment",
                        "horizon": self.current_horizon.value,
                    }
                )
                steps.append({"type": "planning", "action": "break_down_goals"})

            # Check for tactical planning
            if self.current_horizon == PlanningHorizon.MEDIUM_TERM:
                steps.append(
                    {
                        "type": "planning",
                        "action": "tactical_planning",
                        "horizon": "medium_term",
                    }
                )

            # Immediate execution for short-term
            if self.current_horizon in [
                PlanningHorizon.IMMEDIATE,
                PlanningHorizon.SHORT_TERM,
            ]:
                # Add immediate action steps
                pass

            # ------------------------
            # 🔥 EXECUTOR COMMANDS
            # ------------------------
            file_match = re.findall(r"[a-zA-Z0-9_\-./\\]+", context)
            file_target = file_match[-1] if file_match else ""

            if "run" in lower and file_target:
                steps.append(
                    {
                        "type": "tool",
                        "action": "run_file",
                        "input": file_target,
                        "horizon": "immediate",
                    }
                )

            elif "open" in lower and file_target:
                steps.append(
                    {
                        "type": "tool",
                        "action": "open_file",
                        "input": file_target,
                        "horizon": "immediate",
                    }
                )

            # ------------------------
            # 🔥 VISION FLOW
            # ------------------------
            elif route == "vision":
                steps.extend(
                    [
                        {
                            "type": "tool",
                            "action": "capture_frame",
                            "horizon": "immediate",
                        },
                        {
                            "type": "tool",
                            "action": "analyze_scene",
                            "input": context,
                            "horizon": "short_term",
                        },
                    ]
                )

            # ------------------------
            # 🔥 MEMORY FLOW
            # ------------------------
            elif route == "memory":
                steps.extend(
                    [
                        {
                            "type": "tool",
                            "action": "fetch_memory",
                            "horizon": "immediate",
                        },
                        {
                            "type": "llm",
                            "action": "summarize_memory",
                            "horizon": "short_term",
                        },
                    ]
                )

            # ------------------------
            # 🔥 CODE FLOW
            # ------------------------
            elif route == "code":
                steps.extend(
                    [
                        {
                            "type": "tool",
                            "action": "generate_code",
                            "input": context,
                            "horizon": "medium_term",
                        },
                        {
                            "type": "llm",
                            "action": "explain_code",
                            "horizon": "short_term",
                        },
                        {
                            "type": "tool",
                            "action": "test_code",
                            "horizon": "short_term",
                        },
                    ]
                )

            # ------------------------
            # 🔥 SYSTEM FLOW
            # ------------------------
            elif route == "system":
                steps.append(
                    {"type": "system", "action": "status_check", "horizon": "immediate"}
                )

                # Add health check for long-term monitoring
                if self.long_term_goals:
                    steps.append(
                        {
                            "type": "system",
                            "action": "monitor_goals",
                            "horizon": "long_term",
                        }
                    )

            # ------------------------
            # 🔥 DEFAULT → LLM
            # ------------------------
            else:
                # Check if this relates to an existing goal
                related_goal = None
                for goal in self.long_term_goals:
                    if not goal.completed and goal.name.lower() in lower:
                        related_goal = goal
                        break

                if related_goal:
                    steps.append(
                        {
                            "type": "llm",
                            "action": "respond_with_goal_context",
                            "input": context,
                            "goal_context": related_goal.name,
                            "horizon": "long_term",
                        }
                    )
                else:
                    steps.append(
                        {
                            "type": "llm",
                            "action": "respond",
                            "input": context,
                            "horizon": "immediate",
                        }
                    )

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            validated_steps = []

            for step in steps:
                if not isinstance(step, dict):
                    continue

                step_type = str(step.get("type", "")).strip().lower()
                action = str(step.get("action", "")).strip()

                if not step_type or not action:
                    continue

                validated_step = {
                    "type": step_type,
                    "action": action,
                    "horizon": step.get("horizon", self.current_horizon.value),
                }

                if "input" in step and step["input"]:
                    validated_step["input"] = str(step["input"])[
                        : getattr(self, "max_context_length", 500)
                    ]

                if "goal_context" in step:
                    validated_step["goal_context"] = step["goal_context"]

                validated_steps.append(validated_step)

            if not validated_steps:
                return [
                    {
                        "type": "llm",
                        "action": "respond",
                        "input": context,
                        "horizon": "immediate",
                    }
                ]

            return validated_steps

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Planner BuildSteps Error] {e}")

            return [
                {
                    "type": "llm",
                    "action": "respond",
                    "input": context,
                    "horizon": "immediate",
                }
            ]

    # ------------------------
    # PRIORITY SYSTEM (ENHANCED)
    # ------------------------
    def _get_priority(self, route: str) -> str:
        try:
            if not route:
                return getattr(self, "default_priority", "normal")

            route = str(route).strip().lower()

            # 🔥 priority mapping with horizon consideration
            priority_map = {
                "system": "high",
                "code": "high",
                "emergency": "critical",
                "vision": "medium",
                "memory": "medium",
                "llm": "normal",
                "planning": "high",
            }

            # Check if any long-term goals are urgent
            for goal in self.long_term_goals:
                if not goal.completed and goal.target_date < datetime.now() + timedelta(
                    days=2
                ):
                    return "critical"

            priority = priority_map.get(
                route, getattr(self, "default_priority", "normal")
            )

            # 🔥 validate output
            valid_priorities = {"critical", "high", "medium", "normal", "low"}
            if priority not in valid_priorities:
                priority = getattr(self, "default_priority", "normal")

            return priority

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Planner Priority Error] {e}")

            return getattr(self, "default_priority", "normal")

    # ------------------------
    # PLAN EXECUTION TRACKING (NEW)
    # ------------------------
    def mark_step_completed(self, step_index: int, result: Any = None):
        """Mark a plan step as completed"""
        if self.active_plan and step_index < len(self.active_plan["steps"]):
            self.active_plan["steps"][step_index]["status"] = "completed"
            if result:
                self.active_plan["steps"][step_index]["result"] = str(result)[:200]

    def mark_plan_completed(self, success: bool = True):
        """Mark the current plan as completed"""
        if self.active_plan:
            self.active_plan["meta"]["completed_at"] = datetime.now().isoformat()
            self.active_plan["meta"]["success"] = success

            if success:
                self.successful_plans += 1
            else:
                self.failed_plans += 1

            self.active_plan = None

    def get_plan_progress(self) -> Dict[str, Any]:
        """Get progress of current plan"""
        if not self.active_plan:
            return {"has_active_plan": False}

        steps = self.active_plan.get("steps", [])
        completed = sum(1 for s in steps if s.get("status") == "completed")

        return {
            "has_active_plan": True,
            "total_steps": len(steps),
            "completed_steps": completed,
            "progress_percent": (completed / len(steps) * 100) if steps else 0,
            "current_step": steps[completed] if completed < len(steps) else None,
        }

    # ------------------------
    # FALLBACK (ENHANCED)
    # ------------------------
    def _fallback(self, reason: str) -> Dict[str, Any]:
        try:
            if logger:
                logger.warning(f"[Planner Fallback] {reason}")

            self.failed_plans += 1

            reason = str(reason).lower() if reason else ""

            # 🔥 context-aware input message
            if "route" in reason:
                fallback_input = "Invalid route detected. Respond safely."

            elif "context" in reason:
                fallback_input = "Input was invalid. Generate a safe response."

            elif "steps" in reason:
                fallback_input = "Planning failed. Provide a basic response."

            elif "error" in reason:
                fallback_input = "An internal error occurred. Respond safely."

            else:
                fallback_input = "fallback"

            # 🔥 safe structured fallback with hierarchy
            return {
                "route": getattr(self, "default_route", "llm"),
                "steps": [
                    {
                        "type": "llm",
                        "action": "respond",
                        "input": fallback_input[
                            : getattr(self, "max_context_length", 500)
                        ],
                        "horizon": "immediate",
                    }
                ],
                "meta": {
                    "step_count": 1,
                    "priority": "low",
                    "created_by": "planner_fallback_v3",
                    "safe": False,
                    "horizon": "immediate",
                    "plan_type": "reactive",
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                },
            }

        except Exception as e:
            if logger:
                logger.warning(f"[Planner Fallback Error] {e}")

            # 🔥 hard fail-safe fallback
            return {
                "route": "llm",
                "steps": [
                    {
                        "type": "llm",
                        "action": "respond",
                        "input": "fallback",
                        "horizon": "immediate",
                    }
                ],
                "meta": {
                    "step_count": 1,
                    "priority": "low",
                    "created_by": "planner_fallback_safe",
                    "safe": False,
                    "horizon": "immediate",
                },
            }

    # ------------------------
    # DEBUG / INFO (ENHANCED)
    # ------------------------
    def get_limits(self):
        try:
            return {
                # 🔥 core limits
                "max_steps": getattr(self, "max_steps", 10),
                "max_context_length": getattr(self, "max_context_length", 500),
                "max_long_term_goals": getattr(self, "max_long_term_goals", 10),
                "max_short_term_tasks": getattr(self, "max_short_term_tasks", 20),
                # 🔥 feature flags
                "enable_multi_step": getattr(self, "enable_multi_step", False),
                "enable_tool_chaining": getattr(self, "enable_tool_chaining", False),
                "enable_hierarchical_planning": getattr(
                    self, "enable_hierarchical_planning", False
                ),
                # 🔥 defaults
                "default_priority": getattr(self, "default_priority", "normal"),
                "default_route": getattr(self, "default_route", "llm"),
                "current_horizon": (
                    self.current_horizon.value
                    if hasattr(self, "current_horizon")
                    else "immediate"
                ),
            }

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Planner GetLimits Error] {e}")

            # 🔥 safe fallback
            return {
                "max_steps": 10,
                "max_context_length": 500,
                "max_long_term_goals": 10,
                "max_short_term_tasks": 20,
                "enable_multi_step": False,
                "enable_tool_chaining": False,
                "enable_hierarchical_planning": False,
                "default_priority": "normal",
                "default_route": "llm",
                "current_horizon": "immediate",
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            "plans_executed": self.plans_executed,
            "successful_plans": self.successful_plans,
            "failed_plans": self.failed_plans,
            "success_rate": (self.successful_plans / max(1, self.plans_executed)) * 100,
            "active_long_term_goals": len(
                [g for g in self.long_term_goals if not g.completed]
            ),
            "total_long_term_goals": len(self.long_term_goals),
            "pending_short_term_tasks": len(self.short_term_tasks),
            "has_active_plan": self.active_plan is not None,
            "current_horizon": (
                self.current_horizon.value
                if hasattr(self, "current_horizon")
                else "immediate"
            ),
        }

    def validate_plan(self, plan: Dict[str, Any]) -> bool:
        try:
            # ------------------------
            # 🔥 BASIC STRUCTURE
            # ------------------------
            if not isinstance(plan, dict):
                return False

            route = plan.get("route")
            steps = plan.get("steps")
            meta = plan.get("meta")

            # ------------------------
            # 🔥 ROUTE VALIDATION
            # ------------------------
            if not isinstance(route, str) or not route.strip():
                return False

            # ------------------------
            # 🔥 STEPS VALIDATION
            # ------------------------
            if not isinstance(steps, list) or not steps:
                return False

            for step in steps:
                if not isinstance(step, dict):
                    return False

                step_type = step.get("type")
                action = step.get("action")

                if not isinstance(step_type, str) or not step_type:
                    return False

                if not isinstance(action, str) or not action:
                    return False

                # optional input validation
                if "input" in step and step["input"] is not None:
                    if not isinstance(step["input"], str):
                        return False

            # ------------------------
            # 🔥 META VALIDATION (OPTIONAL)
            # ------------------------
            if meta is not None:
                if not isinstance(meta, dict):
                    return False

                if "step_count" in meta and not isinstance(meta["step_count"], int):
                    return False

                if "priority" in meta and meta["priority"] not in {
                    "critical",
                    "high",
                    "medium",
                    "normal",
                    "low",
                }:
                    return False

                if "horizon" in meta:
                    valid_horizons = {
                        "immediate",
                        "short_term",
                        "medium_term",
                        "long_term",
                        "strategic",
                    }
                    if meta["horizon"] not in valid_horizons:
                        return False

            return True

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Planner ValidatePlan Error] {e}")
            return False


__all__ = ["Planner", "PlanningHorizon", "PlanType", "PlanStep", "LongTermGoal"]
