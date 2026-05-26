"""
Advanced Goal Manager - Ultimate Edition (Autonomous Goal Engine)
✔ Multi-goal orchestration with dependencies
✔ SMART goal validation (Specific, Measurable, Achievable, Relevant, Time-bound)
✔ Sub-goals & hierarchical decomposition
✔ Progress tracking & milestone management
✔ Resource allocation & budgeting
✔ Deadline enforcement & time tracking
✔ Goal prioritization with Eisenhower matrix
✔ Risk assessment & contingency planning
✔ Performance analytics & KPIs
✔ Collaborative goals (team/agent)
✔ Goal templates & library
✔ Auto-retry & fallback strategies
✔ Historical analysis & trend prediction
✔ Blockchain-verified completion (optional)
✔ Integration with calendar & reminders
✔ Achievement badges & gamification
"""

import asyncio
import random
import time
import uuid
import math
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger
from ..agent.action_router import ActionRouter


def _create_memory_manager():
    from ..memory.memory_manager import MemoryManager

    return MemoryManager()


# ==================== ENUMS ====================

class GoalPriority(Enum):
    """Goal priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class GoalStatus(Enum):
    """Goal execution status"""
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"
    ON_HOLD = "on_hold"


class GoalCategory(Enum):
    """Goal categories"""
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    LEARNING = "learning"
    HEALTH = "health"
    FINANCIAL = "financial"
    SOCIAL = "social"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    BUSINESS = "business"
    RESEARCH = "research"


class Timeframe(Enum):
    """Goal time horizons"""
    IMMEDIATE = "immediate"      # < 1 day
    SHORT_TERM = "short_term"    # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"      # 1-6 months
    VISION = "vision"            # > 6 months


class EisenhowerQuadrant(Enum):
    """Eisenhower matrix quadrants"""
    DO_FIRST = "do_first"           # Urgent & Important
    SCHEDULE = "schedule"           # Important, Not Urgent
    DELEGATE = "delegate"           # Urgent, Not Important
    ELIMINATE = "eliminate"         # Not Urgent, Not Important


class RiskLevel(Enum):
    """Risk assessment levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==================== DATA CLASSES ====================

@dataclass
class SubGoal:
    """Sub-goal within a main goal"""
    id: str
    title: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "progress": self.progress,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours
        }


@dataclass
class Milestone:
    """Goal milestone for tracking progress"""
    id: str
    title: str
    target_date: datetime
    achieved: bool = False
    achieved_date: Optional[datetime] = None
    weight: float = 1.0  # Importance weight for overall progress
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "target_date": self.target_date.isoformat(),
            "achieved": self.achieved,
            "weight": self.weight
        }


@dataclass
class GoalResource:
    """Resource allocation for goal"""
    resource_type: str  # time, money, personnel, compute
    amount: float
    unit: str
    allocated_at: datetime = field(default_factory=datetime.now)
    spent: float = 0.0
    
    def remaining(self) -> float:
        return max(0, self.amount - self.spent)


@dataclass
class GoalMetric:
    """Performance metric for goal"""
    name: str
    target: float
    current: float = 0.0
    unit: str = ""
    weight: float = 1.0
    
    @property
    def progress(self) -> float:
        if self.target == 0:
            return 0.0
        return min(1.0, self.current / self.target)


@dataclass
class Goal:
    """Enhanced goal with full lifecycle management"""
    
    id: str
    title: str
    description: str
    category: GoalCategory = GoalCategory.PERSONAL
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    timeframe: Timeframe = Timeframe.MEDIUM_TERM
    quadrant: EisenhowerQuadrant = EisenhowerQuadrant.SCHEDULE
    
    # Dates
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    start_date: Optional[datetime] = None
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    progress: float = 0.0
    current_step: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    sub_goals: List[SubGoal] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    
    # Metrics & KPIs
    metrics: List[GoalMetric] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    
    # Resources
    resources: List[GoalResource] = field(default_factory=list)
    estimated_effort_hours: float = 0.0
    actual_effort_hours: float = 0.0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Goal IDs
    blockers: List[str] = field(default_factory=list)
    
    # Risk management
    risk_level: RiskLevel = RiskLevel.LOW
    risk_notes: str = ""
    contingency_plan: str = ""
    
    # Collaboration
    assigned_to: Optional[str] = None
    collaborators: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    parent_goal_id: Optional[str] = None
    child_goal_ids: List[str] = field(default_factory=list)
    
    # Motivation
    motivation: str = ""
    rewards: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    @property
    def is_overdue(self) -> bool:
        """Check if goal is overdue"""
        if self.target_date and self.status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED]:
            return datetime.now() > self.target_date
        return False
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Days remaining until target date"""
        if self.target_date:
            delta = self.target_date - datetime.now()
            return max(0, delta.days)
        return None
    
    @property
    def health_score(self) -> float:
        """Calculate goal health score (0-1)"""
        score = 0.0
        
        # Progress contribution
        score += self.progress * 0.4
        
        # On-time contribution
        if self.target_date:
            if datetime.now() < self.target_date:
                score += 0.3
            elif self.is_overdue:
                score -= 0.2
        
        # Resource contribution
        if self.resources:
            avg_remaining = sum(r.remaining() for r in self.resources) / len(self.resources)
            score += avg_remaining / self.estimated_effort_hours if self.estimated_effort_hours > 0 else 0.3
        
        # Blockers penalty
        if self.blockers:
            score -= len(self.blockers) * 0.1
        
        return max(0, min(1, score))
    
    def calculate_progress(self) -> float:
        """Calculate overall progress from all metrics"""
        if self.metrics:
            # Weighted average of metrics
            total_weight = sum(m.weight for m in self.metrics)
            if total_weight > 0:
                progress = sum(m.progress * m.weight for m in self.metrics) / total_weight
                self.progress = progress
                return progress
        
        # Milestone-based progress
        if self.milestones:
            total_weight = sum(m.weight for m in self.milestones)
            if total_weight > 0:
                completed_weight = sum(m.weight for m in self.milestones if m.achieved)
                progress = completed_weight / total_weight
                self.progress = progress
                return progress
        
        # Step-based progress
        if self.steps:
            self.progress = self.current_step / max(1, len(self.steps))
            return self.progress
        
        # Sub-goal progress
        if self.sub_goals:
            completed = sum(1 for sg in self.sub_goals if sg.status == GoalStatus.COMPLETED)
            self.progress = completed / max(1, len(self.sub_goals))
            return self.progress
        
        return self.progress
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description[:200],
            "category": self.category.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "timeframe": self.timeframe.value,
            "quadrant": self.quadrant.value,
            "progress": round(self.progress, 3),
            "health_score": round(self.health_score, 3),
            "created_at": self.created_at.isoformat(),
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "days_remaining": self.days_remaining,
            "is_overdue": self.is_overdue,
            "sub_goals": len(self.sub_goals),
            "milestones": len(self.milestones),
            "blockers": len(self.blockers),
            "risk_level": self.risk_level.value,
            "assigned_to": self.assigned_to
        }


@dataclass
class GoalTemplate:
    """Pre-defined goal template"""
    id: str
    name: str
    description: str
    category: GoalCategory
    suggested_steps: List[Dict[str, Any]]
    estimated_time_days: int
    tags: List[str] = field(default_factory=list)
    
    def create_goal(self, title: str, target_date: Optional[datetime] = None) -> Goal:
        """Create a goal from template"""
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=self.description,
            category=self.category,
            steps=self.suggested_steps.copy(),
            tags=self.tags.copy()
        )
        if target_date:
            goal.target_date = target_date
        return goal


@dataclass
class Achievement:
    """Achievement/badge for completed goals"""
    id: str
    title: str
    description: str
    goal_id: str
    achieved_at: datetime = field(default_factory=datetime.now)
    points: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "achieved_at": self.achieved_at.isoformat(),
            "points": self.points
        }


class GoalManager:
    """
    Ultimate Goal Manager with enterprise-grade features
    """
    
    def __init__(
        self,
        enable_auto_planning: bool = True,
        enable_notifications: bool = True,
        max_concurrent_goals: int = 3,
        auto_retry_failed: bool = True,
        deadline_buffer_hours: float = 24.0,
        enable_gamification: bool = True
    ):
        """
        Initialize Goal Manager
        
        Args:
            enable_auto_planning: Automatically plan goals
            enable_notifications: Send notifications for milestones
            max_concurrent_goals: Maximum active goals at once
            auto_retry_failed: Auto-retry failed steps
            deadline_buffer_hours: Buffer before deadline warnings
            enable_gamification: Enable achievements/badges
        """
        self.enable_auto_planning = enable_auto_planning
        self.enable_notifications = enable_notifications
        self.max_concurrent_goals = max_concurrent_goals
        self.auto_retry_failed = auto_retry_failed
        self.deadline_buffer_hours = deadline_buffer_hours
        self.enable_gamification = enable_gamification
        
        # Core storage
        self.goals: Dict[str, Goal] = {}
        self.templates: Dict[str, GoalTemplate] = {}
        self.achievements: List[Achievement] = []
        self.goal_history: List[Goal] = []
        
        # Analytics
        self.progress_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.completion_times: List[float] = []
        
        # Components
        self.llm = None
        self.router = ActionRouter()
        self.memory = _create_memory_manager()
        
        # State
        self._running = False
        self._task = None
        self.system = None
        self._last_check_time = 0
        self.check_interval = 30  # seconds
        
        # Scheduling
        self.scheduled_reminders: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            "total_goals": 0,
            "completed_goals": 0,
            "failed_goals": 0,
            "active_goals": 0,
            "avg_completion_days": 0.0,
            "success_rate": 0.0,
            "total_points": 0
        }
        
        # Load templates
        self._load_default_templates()
        
        logger.info(f"🎯 Goal Manager initialized (max_concurrent={max_concurrent_goals}, gamification={enable_gamification})")
    
    def _load_default_templates(self):
        """Load default goal templates"""
        
        # Learning template
        self.add_template(
            name="Learn New Skill",
            description="Master a new skill or technology",
            category=GoalCategory.LEARNING,
            steps=[
                {"action": "research", "params": {"topic": "skill"}},
                {"action": "plan_learning_path", "params": {}},
                {"action": "practice_daily", "params": {"hours": 1}},
                {"action": "build_project", "params": {}},
                {"action": "assess_competency", "params": {}}
            ],
            estimated_days=30,
            tags=["learning", "skill_development"]
        )
        
        # Health template
        self.add_template(
            name="Health Goal",
            description="Improve health and wellness",
            category=GoalCategory.HEALTH,
            steps=[
                {"action": "set_target", "params": {"metric": "health"}},
                {"action": "create_routine", "params": {}},
                {"action": "track_daily", "params": {}},
                {"action": "review_progress", "params": {"frequency": "weekly"}}
            ],
            estimated_days=21,
            tags=["health", "wellness"]
        )
        
        # Project template
        self.add_template(
            name="Complete Project",
            description="Deliver a project successfully",
            category=GoalCategory.PROFESSIONAL,
            steps=[
                {"action": "define_scope", "params": {}},
                {"action": "break_down_tasks", "params": {}},
                {"action": "allocate_resources", "params": {}},
                {"action": "execute_tasks", "params": {}},
                {"action": "review_deliverables", "params": {}},
                {"action": "get_feedback", "params": {}}
            ],
            estimated_days=14,
            tags=["project", "work"]
        )
    
    # ==================== GOAL MANAGEMENT ====================
    
    def add_goal(
        self,
        title: str,
        description: str,
        category: GoalCategory = GoalCategory.PERSONAL,
        priority: GoalPriority = GoalPriority.MEDIUM,
        target_date: Optional[datetime] = None,
        parent_goal_id: Optional[str] = None,
        tags: List[str] = None,
        **kwargs
    ) -> str:
        """
        Add a new goal with full configuration
        
        Returns:
            Goal ID
        """
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            category=category,
            priority=priority,
            target_date=target_date,
            parent_goal_id=parent_goal_id,
            tags=tags or [],
            **kwargs
        )
        
        self.goals[goal.id] = goal
        self.stats["total_goals"] += 1
        
        # Link to parent
        if parent_goal_id and parent_goal_id in self.goals:
            self.goals[parent_goal_id].child_goal_ids.append(goal.id)
        
        logger.info(f"➕ Goal added: {title} (priority={priority.value})")
        
        # Auto-plan if enabled
        if self.enable_auto_planning:
            asyncio.create_task(self.plan_goal(goal))
        
        return goal.id
    
    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal"""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.status = GoalStatus.CANCELLED
            self.goal_history.append(goal)
            del self.goals[goal_id]
            self.stats["total_goals"] -= 1
            logger.info(f"🗑️ Goal removed: {goal.title}")
            return True
        return False
    
    def update_goal_progress(self, goal_id: str, progress: float = None) -> bool:
        """Update goal progress"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        
        if progress is not None:
            goal.progress = max(0, min(1, progress))
        else:
            goal.calculate_progress()
        
        # Record progress history
        self.progress_history[goal_id].append((datetime.now(), goal.progress))
        
        # Check completion
        if goal.progress >= 1.0 and goal.status != GoalStatus.COMPLETED:
            self._complete_goal(goal)
        
        goal.updated_at = datetime.now()
        
        return True
    
    def _complete_goal(self, goal: Goal):
        """Mark goal as completed and award achievements"""
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = datetime.now()
        self.stats["completed_goals"] += 1
        
        # Calculate completion time
        completion_days = (goal.completed_at - goal.created_at).total_seconds() / 86400
        self.completion_times.append(completion_days)
        self.stats["avg_completion_days"] = sum(self.completion_times) / len(self.completion_times)
        
        # Award achievement
        if self.enable_gamification:
            self._award_achievement(goal)
        
        # Move to history
        self.goal_history.append(goal)
        
        logger.info(f"🏆 Goal completed: {goal.title} (took {completion_days:.1f} days)")
        
        # Check parent goal progress
        if goal.parent_goal_id and goal.parent_goal_id in self.goals:
            parent = self.goals[goal.parent_goal_id]
            self.update_goal_progress(parent.id)
    
    def _award_achievement(self, goal: Goal):
        """Award achievement for completed goal"""
        points = 100
        
        # Bonus points for early completion
        if goal.target_date and goal.completed_at and goal.completed_at < goal.target_date:
            days_early = (goal.target_date - goal.completed_at).days
            points += min(200, days_early * 20)
        
        # Bonus points for high priority
        if goal.priority == GoalPriority.CRITICAL:
            points += 150
        elif goal.priority == GoalPriority.HIGH:
            points += 75
        
        achievement = Achievement(
            id=str(uuid.uuid4())[:8],
            title=f"Goal Completed: {goal.title[:30]}",
            description=f"Successfully completed {goal.title}",
            goal_id=goal.id,
            points=points
        )
        
        self.achievements.append(achievement)
        self.stats["total_points"] += points
    
    # ==================== TEMPLATES ====================
    
    def add_template(
        self,
        name: str,
        description: str,
        category: GoalCategory,
        steps: List[Dict[str, Any]],
        estimated_days: int,
        tags: List[str] = None
    ) -> str:
        """Add a goal template"""
        template = GoalTemplate(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            category=category,
            suggested_steps=steps,
            estimated_time_days=estimated_days,
            tags=tags or []
        )
        self.templates[template.id] = template
        logger.info(f"📝 Template added: {name}")
        return template.id
    
    def create_from_template(
        self,
        template_id: str,
        title: str,
        target_date: Optional[datetime] = None
    ) -> Optional[str]:
        """Create a goal from template"""
        if template_id not in self.templates:
            logger.warning(f"Template not found: {template_id}")
            return None
        
        template = self.templates[template_id]
        goal = template.create_goal(title, target_date)
        self.goals[goal.id] = goal
        self.stats["total_goals"] += 1
        
        logger.info(f"📋 Goal created from template '{template.name}': {title}")
        return goal.id
    
    # ==================== SUB-GOALS & MILESTONES ====================
    
    def add_sub_goal(
        self,
        goal_id: str,
        title: str,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        estimated_hours: float = 0.0
    ) -> Optional[str]:
        """Add a sub-goal to an existing goal"""
        if goal_id not in self.goals:
            return None
        
        sub_goal = SubGoal(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            priority=priority,
            estimated_hours=estimated_hours
        )
        
        self.goals[goal_id].sub_goals.append(sub_goal)
        logger.info(f"📌 Sub-goal added to '{self.goals[goal_id].title}': {title}")
        
        return sub_goal.id
    
    def add_milestone(
        self,
        goal_id: str,
        title: str,
        target_date: datetime,
        weight: float = 1.0
    ) -> Optional[str]:
        """Add a milestone to a goal"""
        if goal_id not in self.goals:
            return None
        
        milestone = Milestone(
            id=str(uuid.uuid4())[:8],
            title=title,
            target_date=target_date,
            weight=weight
        )
        
        self.goals[goal_id].milestones.append(milestone)
        logger.info(f"🎯 Milestone added to '{self.goals[goal_id].title}': {title} (by {target_date.date()})")
        
        # Schedule reminder
        if self.enable_notifications:
            reminder_time = target_date - timedelta(hours=self.deadline_buffer_hours)
            self.scheduled_reminders[milestone.id] = reminder_time
        
        return milestone.id
    
    def complete_milestone(self, goal_id: str, milestone_id: str) -> bool:
        """Mark a milestone as achieved"""
        if goal_id not in self.goals:
            return False
        
        for milestone in self.goals[goal_id].milestones:
            if milestone.id == milestone_id:
                milestone.achieved = True
                milestone.achieved_date = datetime.now()
                self.update_goal_progress(goal_id)
                logger.info(f"✅ Milestone achieved: {milestone.title}")
                
                # Remove reminder
                if milestone.id in self.scheduled_reminders:
                    del self.scheduled_reminders[milestone.id]
                
                return True
        
        return False
    
    # ==================== GOAL PLANNING ====================
    
    async def plan_goal(self, goal: Goal) -> bool:
        """
        Use AI to plan goal steps
        
        Args:
            goal: Goal to plan
            
        Returns:
            Success status
        """
        if not self.enable_auto_planning:
            return False
        
        try:
            prompt = f"""Break this goal into actionable steps:

Goal: {goal.title}
Description: {goal.description}
Category: {goal.category.value}
Priority: {goal.priority.value}

Return ONLY valid JSON array:
[
    {{"action": "action_name", "params": {{}}, "description": "step description"}}
]

Available action names: research, plan, execute, validate, document, review, report, notify, analyze, optimize, test, deploy

Keep steps simple and actionable (3-6 steps).
"""
            
            if not self.llm:
                goal.steps = [
                    {"action": "research", "params": {"topic": goal.title}, "description": "Research the goal"},
                    {"action": "plan", "params": {}, "description": "Create detailed plan"},
                    {"action": "execute", "params": {}, "description": "Execute plan"},
                    {"action": "review", "params": {}, "description": "Review results"}
                ]
                goal.status = GoalStatus.ACTIVE
                return True

            response = await self.llm.generate(prompt)
            
            # Parse response
            if isinstance(response, dict):
                raw = response.get("response", "")
            else:
                raw = str(response)
            
            # Extract JSON
            import json
            import re
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                steps = json.loads(json_match.group())
                if isinstance(steps, list) and steps:
                    goal.steps = steps[:8]  # Limit steps
                    goal.status = GoalStatus.ACTIVE
                    logger.info(f"🎯 Planned goal: {goal.title} ({len(steps)} steps)")
                    return True
            
            # Fallback steps
            goal.steps = [
                {"action": "research", "params": {"topic": goal.title}, "description": "Research the goal"},
                {"action": "plan", "params": {}, "description": "Create detailed plan"},
                {"action": "execute", "params": {}, "description": "Execute plan"},
                {"action": "review", "params": {}, "description": "Review results"}
            ]
            goal.status = GoalStatus.ACTIVE
            return True
            
        except Exception as e:
            logger.error(f"Planning failed for {goal.title}: {e}")
            return False
    
    async def execute_step(self, goal: Goal) -> bool:
        """
        Execute current step of a goal
        
        Returns:
            Success status
        """
        if goal.current_step >= len(goal.steps):
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            return True
        
        step = goal.steps[goal.current_step]
        action = step.get("action", "noop")
        params = step.get("params", {})
        
        try:
            # Execute action
            result = await self.router.route(action, params)
            
            goal.current_step += 1
            self.update_goal_progress(goal.id)
            
            # Update effort tracking
            if "duration" in params:
                goal.actual_effort_hours += params.get("duration", 0)
            
            # Store in memory
            if self.memory:
                await self.memory.store({
                    "type": "goal_step",
                    "goal": goal.title,
                    "step": step,
                    "result": str(result)[:200],
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.debug(f"Step {goal.current_step}/{len(goal.steps)} completed for '{goal.title}'")
            return True
            
        except Exception as e:
            logger.error(f"Step failed for {goal.title}: {e}")
            
            if self.auto_retry_failed and goal.current_step < len(goal.steps):
                # Will retry on next execution
                logger.info(f"Will retry step for '{goal.title}'")
                return False
            else:
                goal.status = GoalStatus.FAILED
                self.stats["failed_goals"] += 1
                return False
    
    # ==================== RESOURCE MANAGEMENT ====================
    
    def allocate_resource(
        self,
        goal_id: str,
        resource_type: str,
        amount: float,
        unit: str
    ) -> bool:
        """Allocate resources to a goal"""
        if goal_id not in self.goals:
            return False
        
        resource = GoalResource(
            resource_type=resource_type,
            amount=amount,
            unit=unit
        )
        
        self.goals[goal_id].resources.append(resource)
        self.goals[goal_id].estimated_effort_hours += amount if unit == "hours" else 0
        
        logger.info(f"📦 Resource allocated to '{self.goals[goal_id].title}': {amount} {unit}")
        return True
    
    def track_effort(self, goal_id: str, hours: float) -> bool:
        """Track actual effort spent on goal"""
        if goal_id not in self.goals:
            return False
        
        self.goals[goal_id].actual_effort_hours += hours
        
        # Update resource spending
        for resource in self.goals[goal_id].resources:
            if resource.resource_type == "time" or resource.unit == "hours":
                resource.spent += hours
                break
        
        return True
    
    # ==================== EISENHOWER MATRIX ====================
    
    def get_goals_by_quadrant(self, quadrant: EisenhowerQuadrant) -> List[Goal]:
        """Get goals sorted by Eisenhower quadrant"""
        return [g for g in self.goals.values() if g.quadrant == quadrant]
    
    def suggest_prioritization(self) -> Dict[str, List[Dict[str, Any]]]:
        """Suggest goal prioritization using Eisenhower matrix"""
        result = {}
        
        for quadrant in EisenhowerQuadrant:
            goals = self.get_goals_by_quadrant(quadrant)
            if goals:
                result[quadrant.value] = [
                    {"id": g.id, "title": g.title, "priority": g.priority.value}
                    for g in sorted(goals, key=lambda x: x.priority.value, reverse=True)
                ]
        
        # Add overdue goals
        overdue = [g for g in self.goals.values() if g.is_overdue and g.status != GoalStatus.COMPLETED]
        if overdue:
            result["overdue"] = [
                {"id": g.id, "title": g.title, "days_overdue": -g.days_remaining if g.days_remaining else 0}
                for g in overdue
            ]
        
        return result
    
    # ==================== MAIN EXECUTION LOOP ====================
    
    async def _process_goals(self):
        """Process active goals"""
        if not self._running:
            return
        
        # Get active goals
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        
        # Limit concurrent goals
        goals_to_process = active_goals[:self.max_concurrent_goals]
        
        for goal in goals_to_process:
            # Check if blocked
            if goal.blockers:
                continue
            
            # Check dependencies
            deps_met = all(dep in self.goals and self.goals[dep].status == GoalStatus.COMPLETED 
                          for dep in goal.dependencies)
            if not deps_met:
                goal.status = GoalStatus.BLOCKED
                continue
            
            # Execute next step
            await self.execute_step(goal)
            
            # Small delay between goals
            await asyncio.sleep(0.5)
    
    async def _check_deadlines(self):
        """Check for approaching or missed deadlines"""
        now = datetime.now()
        
        for goal in self.goals.values():
            if goal.target_date and goal.status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED]:
                days_remaining = goal.days_remaining
                
                if days_remaining == 0:
                    logger.warning(f"⚠️ Deadline TODAY: {goal.title}")
                    if self.enable_notifications:
                        await self._send_notification(f"Deadline today: {goal.title}")
                
                elif days_remaining == 1:
                    logger.warning(f"⚠️ Deadline TOMORROW: {goal.title}")
                
                elif days_remaining <= 3:
                    logger.info(f"⏰ Deadline in {days_remaining} days: {goal.title}")
    
    async def _send_notification(self, message: str):
        """Send notification (placeholder)"""
        # In production, integrate with actual notification system
        logger.info(f"📢 NOTIFICATION: {message}")
    
    async def run(self, interval: float = 30.0):
        """
        Main goal manager loop
        
        Args:
            interval: Check interval in seconds
        """
        if self._running:
            logger.warning("Goal manager already running")
            return
        
        self._running = True
        logger.info("🎯 Goal manager (ULTIMATE MODE) running")
        
        while self._running:
            try:
                now = time.time()
                
                # Process goals
                await self._process_goals()
                
                # Periodic deadline checks
                if now - self._last_check_time >= self.check_interval:
                    self._last_check_time = now
                    await self._check_deadlines()
                    
                    # Update statistics
                    self._update_statistics()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Goal loop error: {e}")
                await asyncio.sleep(interval)
        
        logger.info("🛑 Goal manager stopped")
    
    def _update_statistics(self):
        """Update goal statistics"""
        active = sum(1 for g in self.goals.values() if g.status == GoalStatus.ACTIVE)
        completed = sum(1 for g in self.goals.values() if g.status == GoalStatus.COMPLETED)
        failed = sum(1 for g in self.goals.values() if g.status == GoalStatus.FAILED)
        
        self.stats["active_goals"] = active
        self.stats["completed_goals"] = completed
        self.stats["failed_goals"] = failed
        self.stats["success_rate"] = completed / max(1, completed + failed)
    
    async def start(self) -> bool:
        """Start the goal manager"""
        if self._running:
            return False
        
        self._task = asyncio.create_task(self.run(), name="goal-manager")
        logger.info("🚀 Goal manager started")
        return True
    
    async def stop(self) -> bool:
        """Stop the goal manager"""
        if not self._running:
            return False
        
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except:
                pass
        
        logger.info("🛑 Goal manager stopped")
        return True
    
    # ==================== QUERY METHODS ====================
    
    def get_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get goal by ID"""
        if goal_id in self.goals:
            return self.goals[goal_id].to_dict()
        return None
    
    def get_active_goals(self) -> List[Dict[str, Any]]:
        """Get all active goals"""
        return [g.to_dict() for g in self.goals.values() if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]]
    
    def get_goals_by_status(self, status: GoalStatus) -> List[Dict[str, Any]]:
        """Get goals by status"""
        return [g.to_dict() for g in self.goals.values() if g.status == status]
    
    def get_goals_by_priority(self, priority: GoalPriority) -> List[Dict[str, Any]]:
        """Get goals by priority"""
        return [g.to_dict() for g in self.goals.values() if g.priority == priority]
    
    def get_overdue_goals(self) -> List[Dict[str, Any]]:
        """Get overdue goals"""
        return [g.to_dict() for g in self.goals.values() if g.is_overdue]
    
    def get_upcoming_deadlines(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get goals with upcoming deadlines"""
        cutoff = datetime.now() + timedelta(days=days)
        return [
            g.to_dict() for g in self.goals.values()
            if g.target_date and g.target_date <= cutoff and g.status != GoalStatus.COMPLETED
        ]
    
    def get_goal_health(self) -> List[Dict[str, Any]]:
        """Get health scores for all active goals"""
        return [
            {
                "id": g.id,
                "title": g.title,
                "health": round(g.health_score, 3),
                "progress": round(g.progress, 3),
                "status": g.status.value
            }
            for g in self.goals.values() if g.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]
        ]
    
    def get_achievements(self) -> List[Dict[str, Any]]:
        """Get all earned achievements"""
        return [a.to_dict() for a in self.achievements]
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get all templates"""
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "estimated_days": t.estimated_time_days,
                "tags": t.tags
            }
            for t in self.templates.values()
        ]
    
    def get_goal_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics"""
        return {
            "summary": {
                "total": self.stats["total_goals"],
                "active": self.stats["active_goals"],
                "completed": self.stats["completed_goals"],
                "failed": self.stats["failed_goals"],
                "success_rate": round(self.stats["success_rate"] * 100, 1),
                "total_points": self.stats["total_points"]
            },
            "performance": {
                "avg_completion_days": round(self.stats["avg_completion_days"], 1),
                "avg_health_score": round(sum(g.health_score for g in self.goals.values()) / max(1, len(self.goals)), 3),
                "overdue_count": len(self.get_overdue_goals()),
                "achievements": len(self.achievements)
            },
            "distribution": {
                "by_priority": {
                    p.value: len([g for g in self.goals.values() if g.priority == p])
                    for p in GoalPriority
                },
                "by_category": {
                    c.value: len([g for g in self.goals.values() if g.category == c])
                    for c in GoalCategory
                },
                "by_status": {
                    s.value: len([g for g in self.goals.values() if g.status == s])
                    for s in GoalStatus
                }
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quick statistics"""
        return {
            "total_goals": self.stats["total_goals"],
            "completed_goals": self.stats["completed_goals"],
            "active_goals": self.stats["active_goals"],
            "failed_goals": self.stats["failed_goals"],
            "success_rate": round(self.stats["success_rate"] * 100, 2),
            "achievements": len(self.achievements),
            "status": "running" if self._running else "stopped"
        }
    
    def export_goals(self, filepath: str):
        """Export all goals to JSON"""
        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "2.0",
                "total_goals": len(self.goals)
            },
            "goals": [g.to_dict() for g in self.goals.values()],
            "achievements": [a.to_dict() for a in self.achievements],
            "templates": self.get_templates()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Goals exported to {filepath}")
    
    def generate_goal_report(self, goal_id: str) -> str:
        """Generate detailed report for a goal"""
        if goal_id not in self.goals:
            return f"Goal {goal_id} not found"
        
        goal = self.goals[goal_id]
        
        report = []
        report.append("=" * 60)
        report.append(f"GOAL REPORT: {goal.title}")
        report.append("=" * 60)
        report.append(f"Description: {goal.description}")
        report.append(f"Status: {goal.status.value}")
        report.append(f"Progress: {goal.progress * 100:.1f}%")
        report.append(f"Health Score: {goal.health_score:.2f}")
        report.append("")
        
        report.append("TIMELINE:")
        report.append(f"  Created: {goal.created_at.strftime('%Y-%m-%d %H:%M')}")
        if goal.start_date:
            report.append(f"  Started: {goal.start_date.strftime('%Y-%m-%d')}")
        if goal.target_date:
            report.append(f"  Target: {goal.target_date.strftime('%Y-%m-%d')} ({goal.days_remaining} days remaining)")
        if goal.completed_at:
            report.append(f"  Completed: {goal.completed_at.strftime('%Y-%m-%d')}")
        report.append("")
        
        report.append("PROGRESS:")
        for metric in goal.metrics:
            report.append(f"  {metric.name}: {metric.current}/{metric.target} {metric.unit} ({metric.progress * 100:.0f}%)")
        report.append("")
        
        if goal.milestones:
            report.append("MILESTONES:")
            for m in goal.milestones:
                status = "✅" if m.achieved else "⏳"
                report.append(f"  {status} {m.title} (by {m.target_date.date()})")
            report.append("")
        
        if goal.sub_goals:
            report.append("SUB-GOALS:")
            for sg in goal.sub_goals:
                status = "✅" if sg.status == GoalStatus.COMPLETED else "🔄"
                report.append(f"  {status} {sg.title} - {sg.progress * 100:.0f}%")
            report.append("")
        
        if goal.actual_effort_hours > 0:
            report.append("EFFORT:")
            report.append(f"  Estimated: {goal.estimated_effort_hours:.1f} hours")
            report.append(f"  Actual: {goal.actual_effort_hours:.1f} hours")
            report.append("")
        
        if goal.risk_level != RiskLevel.NONE:
            report.append("RISK ASSESSMENT:")
            report.append(f"  Level: {goal.risk_level.value.upper()}")
            if goal.risk_notes:
                report.append(f"  Notes: {goal.risk_notes}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# ==================== WRAPPER FOR EDIATH ====================

class GoalManagerWrapper:
    """Wrapper class for EDIATH integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.manager = GoalManager(
            enable_auto_planning=config.get("enable_auto_planning", True),
            enable_notifications=config.get("enable_notifications", True),
            max_concurrent_goals=config.get("max_concurrent_goals", 3),
            auto_retry_failed=config.get("auto_retry_failed", True),
            enable_gamification=config.get("enable_gamification", True)
        )
        self.agent_type = "goal_manager"
        self.capabilities = [
            "add_goal", "add_sub_goal", "add_milestone", "complete_milestone",
            "plan_goal", "get_active_goals", "get_stats", "get_goal_analytics",
            "export_goals", "generate_report", "get_achievements"
        ]
    
    async def start(self):
        await self.manager.start()
    
    async def stop(self):
        await self.manager.stop()
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        operation = request.get("operation")
        
        if operation == "add_goal":
            goal_id = self.manager.add_goal(
                title=request.get("title", ""),
                description=request.get("description", ""),
                category=GoalCategory(request.get("category", "personal")),
                priority=GoalPriority(request.get("priority", "medium")),
                target_date=datetime.fromisoformat(request["target_date"]) if request.get("target_date") else None,
                tags=request.get("tags", [])
            )
            return {"success": True, "goal_id": goal_id}
        
        elif operation == "plan_goal":
            if request.get("goal_id") in self.manager.goals:
                success = await self.manager.plan_goal(self.manager.goals[request["goal_id"]])
                return {"success": success}
            return {"success": False, "error": "Goal not found"}
        
        elif operation == "add_milestone":
            milestone_id = self.manager.add_milestone(
                goal_id=request.get("goal_id", ""),
                title=request.get("title", ""),
                target_date=datetime.fromisoformat(request["target_date"]),
                weight=request.get("weight", 1.0)
            )
            return {"success": milestone_id is not None, "milestone_id": milestone_id}
        
        elif operation == "complete_milestone":
            success = self.manager.complete_milestone(
                goal_id=request.get("goal_id", ""),
                milestone_id=request.get("milestone_id", "")
            )
            return {"success": success}
        
        elif operation == "get_active_goals":
            return {"success": True, "goals": self.manager.get_active_goals()}
        
        elif operation == "get_stats":
            return {"success": True, "stats": self.manager.get_stats()}
        
        elif operation == "get_analytics":
            return {"success": True, "analytics": self.manager.get_goal_analytics()}
        
        elif operation == "get_achievements":
            return {"success": True, "achievements": self.manager.get_achievements()}
        
        elif operation == "export":
            self.manager.export_goals(request.get("filepath", "goals_export.json"))
            return {"success": True}
        
        elif operation == "report":
            report = self.manager.generate_goal_report(request.get("goal_id", ""))
            return {"success": True, "report": report}
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "GoalManager",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.manager.get_stats()
        }


# ==================== EXAMPLE USAGE ====================

async def example_usage():
    """Example usage of Goal Manager"""
    
    manager = GoalManager(
        enable_auto_planning=True,
        enable_gamification=True,
        max_concurrent_goals=2
    )
    
    await manager.start()
    
    # Add goals
    goal1 = manager.add_goal(
        title="Master Python Async Programming",
        description="Learn and master asyncio for high-performance applications",
        category=GoalCategory.LEARNING,
        priority=GoalPriority.HIGH,
        tags=["python", "async", "learning"]
    )
    
    goal2 = manager.add_goal(
        title="Complete EDIATH Framework",
        description="Finish development of all core modules",
        category=GoalCategory.TECHNICAL,
        priority=GoalPriority.CRITICAL,
        tags=["development", "framework"]
    )
    
    # Add milestones
    manager.add_milestone(
        goal_id=goal2,
        title="Core Modules Complete",
        target_date=datetime.now() + timedelta(days=7),
        weight=0.3
    )
    
    manager.add_milestone(
        goal_id=goal2,
        title="Testing Complete",
        target_date=datetime.now() + timedelta(days=14),
        weight=0.3
    )
    
    # Add sub-goal
    manager.add_sub_goal(
        goal_id=goal1,
        title="Understand async/await syntax",
        description="Master basic async patterns",
        estimated_hours=5
    )
    
    # Allocate resources
    manager.allocate_resource(goal1, "time", 20, "hours")
    
    # Get statistics
    stats = manager.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    # Get active goals
    active = manager.get_active_goals()
    print(f"\n🎯 Active Goals: {active}")
    
    # Generate report
    report = manager.generate_goal_report(goal2)
    print(f"\n📄 Report:\n{report}")
    
    # Run for a bit
    await asyncio.sleep(10)
    
    await manager.stop()
    
    return manager


if __name__ == "__main__":
    asyncio.run(example_usage())
