"""Autonomy modules for autonomous behavior."""

from .autonomous_loop import AutonomousLoop
from .background_thinker import (
    BackgroundThinker,
    ThinkingTask,
    ThinkingMode,
    TaskPriority,
    ThinkingPattern,
)
from .curiosity_engine import (
    CuriosityEngine,
    CuriosityItem,
    CuriosityType,
    CuriositySuggestion,
    SuggestionPriority,
)
from .goal_manage import GoalManager, Goal, GoalPriority, GoalStatus
from .self_improvement import SelfImprovement, ImprovementMetric, ImprovementArea

__all__ = [
    "AutonomousLoop",
    "BackgroundThinker",
    "ThinkingTask",
    "ThinkingMode",
    "TaskPriority",
    "ThinkingPattern",
    "CuriosityEngine",
    "CuriosityItem",
    "CuriosityType",
    "CuriositySuggestion",
    "SuggestionPriority",
    "GoalManager",
    "Goal",
    "GoalPriority",
    "GoalStatus",
    "SelfImprovement",
    "ImprovementMetric",
    "ImprovementArea",
]
