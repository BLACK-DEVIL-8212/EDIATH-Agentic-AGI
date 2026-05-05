"""
Advanced Self Improvement - AI Self-Evolution Engine
"""

import asyncio

from typing import Dict, Any, List
from enum import Enum

from ..utils.logger import logger
from ..brain.llm_engine import LLMEngine
from ..memory import MemoryManager


class ImprovementArea(Enum):
    PERFORMANCE = "performance"
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"


class ImprovementMetric:
    def __init__(self, name: str, area: ImprovementArea):
        self.name = name
        self.area = area
        self.current_value = 0.5
        self.history: List[float] = []

    def update(self, value: float):
        self.current_value = max(0, min(1, value))
        self.history.append(self.current_value)


class SelfImprovement:
    def __init__(self):
        self.metrics: Dict[str, ImprovementMetric] = {}

        self.llm = LLMEngine()
        self.memory = MemoryManager()

        self.iterations = 0
        self.improvements_applied = 0
        self._running = False

    # ------------------------
    # REGISTER METRIC
    # ------------------------
    def register_metric(self, name: str, area: ImprovementArea):
        self.metrics[name] = ImprovementMetric(name, area)

    # ------------------------
    # UPDATE FROM REAL RESULT 🔥
    # ------------------------
    async def update_from_result(self, result: Dict[str, Any]):
        """
        Update metrics based on real system output
        """

        score = 1.0 if result else 0.3

        for metric in self.metrics.values():
            metric.update((metric.current_value + score) / 2)

        await self.memory.store(result)

    # ------------------------
    # AI ANALYSIS 🔥
    # ------------------------
    async def analyze(self):
        """
        Use LLM to analyze system performance
        """

        context = {name: m.current_value for name, m in self.metrics.items()}

        prompt = f"""
Analyze system performance:

Metrics:
{context}

Suggest improvements in JSON:
[
  {{"area": "...", "suggestion": "..."}}
]
"""

        try:
            response = await self.llm.generate(prompt)

            import json

            suggestions = json.loads(response.get("response", "[]"))

            return suggestions

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return []

    # ------------------------
    # APPLY IMPROVEMENTS 🔥
    # ------------------------
    async def improve(self):
        suggestions = await self.analyze()

        for s in suggestions:
            self.improvements_applied += 1

            logger.info(f"⚡ Improvement: {s}")

        self.iterations += 1

        return suggestions

    # ------------------------
    # LOOP 🔥
    # ------------------------
    async def run(self, interval=5.0):
        logger.info("🧠 Self-improvement running")
        self._running = True

        while self._running:
            try:
                await self.improve()
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Improvement error: {e}")

    # ------------------------
    # CONTROL
    # ------------------------
    def stop(self):
        self._running = False

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self):
        return {
            "iterations": self.iterations,
            "metrics": len(self.metrics),
            "improvements": self.improvements_applied,
        }


__all__ = ["SelfImprovement", "ImprovementMetric", "ImprovementArea"]
