"""
Advanced Goal Manager - Autonomous Goal Engine (SAFE VERSION)
"""

import asyncio
import random
import time
from typing import Dict, List, Any
from enum import Enum
from datetime import datetime
import uuid

from ..utils.logger import logger
from ..brain.llm_engine import LLMEngine
from ..agent.action_router import ActionRouter
from ..memory import MemoryManager


class GoalPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class Goal:
    def __init__(self, title: str, description: str, priority=GoalPriority.MEDIUM):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.priority = priority

        self.status = GoalStatus.PENDING
        self.created_at = datetime.now()

        self.steps: List[Dict[str, Any]] = []
        self.current_step = 0
        self.progress = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "progress": self.progress,
            "steps": len(self.steps),
        }


class GoalManager:
    def __init__(self):
        self.goals: Dict[str, Goal] = {}

        self.llm = LLMEngine()
        self.router = ActionRouter()
        self.memory = MemoryManager()

        self._running = False
        self.system = None  # 🔥 for idle check

    # ------------------------
    # ADD GOAL
    # ------------------------
    def add_goal(self, title: str, description: str, priority=GoalPriority.MEDIUM):
        goal = Goal(title, description, priority)
        self.goals[goal.id] = goal
        return goal.id

    # ------------------------
    # PLAN GOAL 🔥 (SAFE)
    # ------------------------
    async def plan_goal(self, goal: Goal):
        try:
            prompt = f"""
Break this goal into simple steps:

Goal: {goal.description}

Return JSON:
[{{"action": "...", "params": {{}}}}]
"""

            response = await self.llm.generate(prompt)

            import json

            raw = (
                response.get("response", "")
                if isinstance(response, dict)
                else str(response)
            )
            steps = json.loads(raw or "[]")

            # 🔥 LIMIT STEPS (prevent overload)
            goal.steps = steps[:3]
            goal.status = GoalStatus.ACTIVE

            logger.info(f"🎯 Planned goal: {goal.title}")

        except Exception as e:
            logger.error(f"Planning failed: {e}")

    # ------------------------
    # EXECUTE STEP 🔥 (SAFE)
    # ------------------------
    async def execute_step(self, goal: Goal):
        if goal.current_step >= len(goal.steps):
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            return

        step = goal.steps[goal.current_step]

        try:
            result = await self.router.route(step.get("action"), step.get("params", {}))

            goal.current_step += 1
            goal.progress = goal.current_step / max(1, len(goal.steps))

            # 🔥 SAFE MEMORY
            try:
                if self.memory:
                    await self.memory.store(
                        {"goal": goal.title, "step": step, "result": result}
                    )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Step failed: {e}")
            goal.status = GoalStatus.FAILED

    # ------------------------
    # MAIN LOOP 🔥 (FIXED)
    # ------------------------
    async def run(self, interval=5.0):
        """SAFE goal loop (NO SPAM MODE)"""

        if self._running:
            return

        self._running = True
        logger.info("🎯 Goal engine (SAFE MODE) running")

        last_run = 0
        COOLDOWN = 10

        while self._running:
            try:
                now = time.time()

                # ------------------------
                # 🔥 ONLY RUN IF IDLE
                # ------------------------
                if self.system and hasattr(self.system, "idle_controller"):
                    if not self.system.idle_controller.is_idle():
                        await asyncio.sleep(2)
                        continue

                # ------------------------
                # 🔥 COOLDOWN
                # ------------------------
                if now - last_run < COOLDOWN:
                    await asyncio.sleep(2)
                    continue

                # ------------------------
                # 🔥 RANDOM TRIGGER
                # ------------------------
                if random.random() > 0.01:
                    await asyncio.sleep(interval)
                    continue

                last_run = now

                # ------------------------
                # 🔥 PROCESS ONLY ONE GOAL
                # ------------------------
                for goal in list(self.goals.values())[:1]:

                    if goal.status == GoalStatus.PENDING:
                        await self.plan_goal(goal)

                    elif goal.status == GoalStatus.ACTIVE:
                        await self.execute_step(goal)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Goal loop error: {e}")
                await asyncio.sleep(2)

        logger.info("🛑 Goal engine stopped")

    # ------------------------
    # CONTROL
    # ------------------------
    def stop(self):
        self._running = False

    # ------------------------
    # GET NEXT
    # ------------------------
    def get_active_goals(self):
        return [
            g.to_dict() for g in self.goals.values() if g.status == GoalStatus.ACTIVE
        ]

    # ------------------------
    # STATS (FIXED BUG)
    # ------------------------
    def get_stats(self):
        try:
            goals = list(self.goals.values())

            return {
                "total_goals": len(goals),
                "completed_goals": len(
                    [g for g in goals if g.status == GoalStatus.COMPLETED]
                ),
                "active_goals": len(
                    [g for g in goals if g.status == GoalStatus.ACTIVE]
                ),
                "status": "active",
            }

        except Exception as e:
            logger.error(f"GoalManager stats error: {e}")

            return {
                "total_goals": 0,
                "completed_goals": 0,
                "active_goals": 0,
                "status": "error",
            }


__all__ = ["GoalManager", "Goal", "GoalPriority", "GoalStatus"]
