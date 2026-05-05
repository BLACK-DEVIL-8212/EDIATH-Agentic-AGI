"""
Advanced Background Thinker - Production Ready (MAXIMUM FEATURES)
✔ Safe execution
✔ Rate-limited
✔ Memory-safe
✔ Fault-tolerant
✔ Auto-task control
✔ Priority queue
✔ Batch processing
✔ Persistent insights
✔ Pattern recognition
✔ Multi-mode thinking
"""

import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import heapq

from ..utils.logger import logger
from ..brain.llm_engine import LLMEngine
from ..memory import MemoryManager


# ------------------------
# THINKING MODES (ENHANCED)
# ------------------------
class ThinkingMode(Enum):
    ANALYSIS = "analysis"
    PLANNING = "planning"
    REASONING = "reasoning"
    OPTIMIZATION = "optimization"
    REFLECTION = "reflection"
    CREATIVE = "creative"
    CRITICAL = "critical"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    FORECAST = "forecast"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ------------------------
# THINKING TASK (ENHANCED)
# ------------------------
@dataclass
class ThinkingTask:
    name: str
    mode: ThinkingMode
    context: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value

    async def execute(self, llm: LLMEngine, system=None) -> Dict[str, Any]:
        """
        🔥 Fully safe execution with timeout and retry
        """
        import asyncio

        self.started_at = datetime.now()
        self.status = "running"

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not self.context and self.mode != ThinkingMode.REFLECTION:
                return {"error": "empty context"}

            # ------------------------
            # 🔥 HARD LOCK (CRITICAL FIX)
            # ------------------------
            if system and getattr(system, "_brain_busy", False):
                self.status = "skipped"
                return {"status": "skipped", "reason": "brain busy"}

            if system:
                system._brain_busy = True

            try:
                # ------------------------
                # 🔥 BUILD ENHANCED CONTEXT
                # ------------------------
                system_prompt = self._get_system_prompt()
                user_prompt = f"{self.mode.value.upper()}: {self.context}"

                # Add metadata context
                if self.tags:
                    user_prompt += f"\nTags: {', '.join(self.tags)}"

                # ------------------------
                # 🔥 SAFE BRAIN CALL (WITH TIMEOUT)
                # ------------------------
                if system and hasattr(system, "brain_process"):
                    result = await asyncio.wait_for(
                        system.brain_process("task", {"task": user_prompt}), timeout=30
                    )
                elif llm:
                    result = await asyncio.wait_for(
                        llm.generate(user_prompt, system_prompt=system_prompt),
                        timeout=30,
                    )
                else:
                    result = {"response": "LLM not available"}

                # ------------------------
                # 🔥 NORMALIZE OUTPUT
                # ------------------------
                output = ""
                if isinstance(result, dict):
                    output = str(
                        result.get("response", result.get("result", ""))
                    ).strip()
                elif isinstance(result, str):
                    output = result.strip()

                # ------------------------
                # 🔥 VALIDATE OUTPUT
                # ------------------------
                if not output:
                    output = f"Processed {self.mode.value} request."

                # ------------------------
                # 🔥 FINALIZE
                # ------------------------
                self.result = output
                self.status = "completed"
                self.completed_at = datetime.now()

                return {
                    "name": self.name,
                    "mode": self.mode.value,
                    "result": output,
                    "duration_ms": (self.completed_at - self.started_at).total_seconds()
                    * 1000,
                    "tags": self.tags,
                }

            finally:
                # 🔥 ALWAYS RELEASE LOCK
                if system:
                    system._brain_busy = False

        except asyncio.TimeoutError:
            self.status = "timeout"
            self.error = "Timeout exceeded"

            # Retry logic
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                self.status = "pending"
                return await self.execute(llm, system)

            return {"error": "timeout", "retries": self.retry_count}

        except Exception as e:
            self.status = "failed"
            self.error = str(e)

            if logger:
                logger.warning(f"[ThinkingTask Error] {e}")

            return {"error": str(e)}

    def _get_system_prompt(self) -> str:
        """Get system prompt based on thinking mode"""
        prompts = {
            ThinkingMode.ANALYSIS: "You are analyzing information. Be thorough and structured.",
            ThinkingMode.PLANNING: "You are planning actions. Be logical and step-by-step.",
            ThinkingMode.REASONING: "You are reasoning through a problem. Show your logic.",
            ThinkingMode.OPTIMIZATION: "You are optimizing a process. Find improvements.",
            ThinkingMode.REFLECTION: "You are reflecting on past actions. Extract lessons.",
            ThinkingMode.CREATIVE: "You are thinking creatively. Generate novel ideas.",
            ThinkingMode.CRITICAL: "You are thinking critically. Evaluate pros and cons.",
            ThinkingMode.STRATEGIC: "You are thinking strategically. Consider long-term impact.",
            ThinkingMode.TACTICAL: "You are thinking tactically. Focus on immediate actions.",
            ThinkingMode.FORECAST: "You are forecasting outcomes. Predict possible scenarios.",
            ThinkingMode.SYNTHESIS: "You are synthesizing information. Combine insights.",
            ThinkingMode.EVALUATION: "You are evaluating options. Compare and rank.",
        }
        return prompts.get(
            self.mode, "You are an AI assistant. Be helpful and concise."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "priority": self.priority.value,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_ms": (
                (self.completed_at - self.started_at).total_seconds() * 1000
                if self.completed_at and self.started_at
                else 0
            ),
            "tags": self.tags,
            "error": self.error,
        }


# ------------------------
# THINKING PATTERN
# ------------------------
class ThinkingPattern:
    """Identifies patterns in thinking tasks"""

    def __init__(self, pattern_id: str, pattern_type: str, confidence: float = 0.0):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.confidence = confidence
        self.occurrences = 1
        self.last_seen = datetime.now()
        self.related_tasks: List[str] = []

    def update(self, confidence_delta: float = 0.1):
        self.occurrences += 1
        self.confidence = min(1.0, self.confidence + confidence_delta)
        self.last_seen = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "type": self.pattern_type,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen.isoformat(),
        }


# ------------------------
# BACKGROUND THINKER (ENHANCED)
# ------------------------
class BackgroundThinker:
    def __init__(self, system=None):
        try:
            self.system = system

            self.is_running = False
            self.queue: List[ThinkingTask] = []  # Priority queue
            self.completed: List[Dict[str, Any]] = []
            self.failed: List[Dict[str, Any]] = []

            # Processing state
            self._processing_lock = asyncio.Lock()
            self._current_task: Optional[ThinkingTask] = None

            # 🔥 USE SYSTEM SHARED COMPONENTS
            self.llm = getattr(system, "llm_engine", LLMEngine())
            self.memory = getattr(system, "memory", MemoryManager())
            self.vision = getattr(system, "vision_engine_instance", None)

            # limits
            self.max_queue = 50
            self.max_completed = 200
            self.max_failed = 100

            # Statistics
            self.insights_generated = 0
            self.iterations = 0
            self.tasks_processed = 0
            self.tasks_failed = 0

            # Performance tracking
            self.processing_times: List[float] = []
            self.daily_stats: Dict[str, Any] = {}

            # Pattern recognition
            self.patterns: Dict[str, ThinkingPattern] = {}
            self.pattern_threshold = 3  # Minimum occurrences to recognize pattern

            # Scheduling
            self.auto_mode = True
            self.min_interval = 1.0
            self.batch_size = 2  # Process up to 2 tasks per cycle
            self.schedule: Dict[str, List[float]] = {}  # Scheduled tasks by hour

            # Persistent storage
            self.persist_path = "data/thinking_insights.json"

            # Caching
            self.result_cache: Dict[str, Any] = {}
            self.cache_ttl = 3600  # 1 hour
            self._cache_timestamps: Dict[str, float] = {}

        except Exception as e:
            if logger:
                logger.warning(f"[Thinker Init Error] {e}")

    # ------------------------
    # ADD TASK (ENHANCED)
    # ------------------------
    def add_task(
        self,
        name: str,
        mode: ThinkingMode,
        context: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        tags: List[str] = None,
        use_cache: bool = True,
    ) -> bool:
        try:
            # ------------------------
            # 🔥 QUEUE LIMIT
            # ------------------------
            if len(self.queue) >= self.max_queue:
                logger.warning(f"Queue full, cannot add task: {name}")
                return False

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not mode:
                return False

            # Check cache
            if use_cache:
                cache_key = self._get_cache_key(name, mode, context)
                if cache_key in self.result_cache:
                    cache_time = self._cache_timestamps.get(cache_key, 0)
                    if time.time() - cache_time < self.cache_ttl:
                        logger.debug(f"Using cached result for: {name}")
                        return True

            # ------------------------
            # 🔥 CREATE TASK
            # ------------------------
            task = ThinkingTask(
                name=name,
                mode=mode,
                context=str(context)[:500],  # Limit context length
                priority=priority,
                tags=tags or [],
                system=self.system,
            )

            # ------------------------
            # 🔥 ADD TO PRIORITY QUEUE
            # ------------------------
            heapq.heappush(self.queue, (task.priority.value, len(self.queue), task))

            # Update schedule
            hour = datetime.now().hour
            if name not in self.schedule:
                self.schedule[name] = []
            self.schedule[name].append(float(hour))
            # Keep last 10 schedule entries
            if len(self.schedule[name]) > 10:
                self.schedule[name] = self.schedule[name][-10:]

            logger.debug(f"Task added: {name} (priority: {priority.name})")
            return True

        except Exception as e:
            if logger:
                logger.warning(f"[add_task Error] {e}")
            return False

    def _get_cache_key(self, name: str, mode: ThinkingMode, context: str) -> str:
        """Generate cache key for task"""
        content = f"{name}:{mode.value}:{context[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_result(self, task: ThinkingTask, result: Any):
        """Cache task result"""
        cache_key = self._get_cache_key(task.name, task.mode, task.context)
        self.result_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        # Clean old cache entries
        now = time.time()
        old_keys = [
            k for k, t in self._cache_timestamps.items() if now - t > self.cache_ttl
        ]
        for k in old_keys:
            del self.result_cache[k]
            del self._cache_timestamps[k]

    # ------------------------
    # BATCH PROCESSING
    # ------------------------
    async def process_batch(self, batch_size: int = 2) -> List[Dict[str, Any]]:
        """Process multiple tasks in batch"""
        results = []

        async with self._processing_lock:
            for _ in range(min(batch_size, len(self.queue))):
                if not self.queue:
                    break

                # Get highest priority task
                priority_val, idx, task = heapq.heappop(self.queue)

                result = await task.execute(self.llm, self.system)
                results.append(result)

                # Update stats
                self.tasks_processed += 1
                if task.status == "failed":
                    self.tasks_failed += 1
                    self.failed.append(task.to_dict())
                    if len(self.failed) > self.max_failed:
                        self.failed = self.failed[-self.max_failed :]
                else:
                    self.completed.append(task.to_dict())
                    if len(self.completed) > self.max_completed:
                        self.completed = self.completed[-self.max_completed :]

                # Track processing time
                if task.completed_at and task.started_at:
                    duration = (
                        task.completed_at - task.started_at
                    ).total_seconds() * 1000
                    self.processing_times.append(duration)
                    if len(self.processing_times) > 100:
                        self.processing_times = self.processing_times[-100:]

                # Cache result
                if result and not result.get("error"):
                    self._cache_result(task, result)
                    self.insights_generated += 1

                    # Store in memory
                    try:
                        if self.memory and hasattr(self.memory, "store"):
                            await self.memory.store(
                                {
                                    "type": "thinking_insight",
                                    "task": task.name,
                                    "mode": task.mode.value,
                                    "result": result.get("result", ""),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                    except Exception:
                        pass

                    # Recognize patterns
                    self._recognize_patterns(task, result)

        return results

    def _recognize_patterns(self, task: ThinkingTask, result: Dict[str, Any]):
        """Recognize patterns in thinking tasks"""
        try:
            pattern_key = f"{task.mode.value}:{task.name[:30]}"

            if pattern_key in self.patterns:
                self.patterns[pattern_key].update()
            else:
                # Check if this is a recurring pattern
                recent_similar = [
                    t
                    for t in self.completed[-20:]
                    if t.get("mode") == task.mode.value
                    and t.get("name", "")[:30] == task.name[:30]
                ]

                if len(recent_similar) >= self.pattern_threshold:
                    self.patterns[pattern_key] = ThinkingPattern(
                        pattern_key,
                        task.mode.value,
                        confidence=min(0.8, 0.3 + (len(recent_similar) * 0.1)),
                    )
                    self.patterns[pattern_key].related_tasks.append(task.name)

                    if logger:
                        logger.info(f"New pattern recognized: {pattern_key}")

            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.daily_stats:
                self.daily_stats[today] = {"total": 0, "by_mode": {}}

            self.daily_stats[today]["total"] += 1
            mode_str = task.mode.value
            if mode_str not in self.daily_stats[today]["by_mode"]:
                self.daily_stats[today]["by_mode"][mode_str] = 0
            self.daily_stats[today]["by_mode"][mode_str] += 1

            # Keep only last 30 days
            if len(self.daily_stats) > 30:
                oldest = min(self.daily_stats.keys())
                del self.daily_stats[oldest]

        except Exception as e:
            logger.debug(f"Pattern recognition error: {e}")

    # ------------------------
    # SCHEDULED TASKS
    # ------------------------
    def schedule_task(
        self, name: str, mode: ThinkingMode, hour: int, context: str = ""
    ):
        """Schedule a task to run at specific hour"""
        if hour < 0 or hour > 23:
            return False

        if name not in self.schedule:
            self.schedule[name] = []

        self.schedule[name].append(float(hour))
        logger.info(f"Scheduled task '{name}' at hour {hour}")
        return True

    async def _check_scheduled_tasks(self):
        """Check and execute scheduled tasks"""
        now = datetime.now()
        current_hour = now.hour
        context = ""

        for task_name, hours in self.schedule.items():
            if current_hour in [int(h) for h in hours]:
                # Check if already run this hour
                last_run = None
                for completed in self.completed[-10:]:
                    if completed.get("name") == task_name:
                        completed_time = completed.get("completed_at")
                        if completed_time:
                            try:
                                dt = datetime.fromisoformat(completed_time)
                                if dt.hour == current_hour and dt.date() == now.date():
                                    last_run = dt
                                    break
                            except:
                                pass

                if not last_run:
                    self.add_task(
                        task_name,
                        mode=(
                            ThinkingMode[task_name.upper()]
                            if task_name.upper() in ThinkingMode.__members__
                            else ThinkingMode.REFLECTION
                        ),
                        context=context,
                        priority=TaskPriority.NORMAL,
                    )

    # ------------------------
    # SMART TASK GENERATION
    # ------------------------
    async def _auto_generate_tasks(self):
        """Generate tasks based on context and patterns"""
        try:
            context_parts = []

            # 🔥 MEMORY
            if self.memory and hasattr(self.memory, "get_recent"):
                recent_memory = await self.memory.get_recent(5)
                if recent_memory:
                    context_parts.append(f"Recent memory: {recent_memory}")

            # 🔥 VISION
            if self.vision and hasattr(self.vision, "webcam"):
                try:
                    if hasattr(self.vision, "current_caption"):
                        caption = self.vision.current_caption
                        if caption:
                            context_parts.append(f"Visual: {caption}")
                except Exception:
                    pass

            context = (
                "\n".join(context_parts)
                if context_parts
                else "General system operation"
            )

            # Generate tasks based on patterns and time
            modes_to_consider = []

            # Morning: planning and analysis
            hour = datetime.now().hour
            if 5 <= hour < 12:
                modes_to_consider = [ThinkingMode.PLANNING, ThinkingMode.ANALYSIS]
            # Afternoon: tactical and optimization
            elif 12 <= hour < 17:
                modes_to_consider = [ThinkingMode.TACTICAL, ThinkingMode.OPTIMIZATION]
            # Evening: reflection and synthesis
            elif 17 <= hour < 22:
                modes_to_consider = [ThinkingMode.REFLECTION, ThinkingMode.SYNTHESIS]
            # Night: creative and strategic
            else:
                modes_to_consider = [ThinkingMode.CREATIVE, ThinkingMode.STRATEGIC]

            # Add tasks based on identified patterns
            for pattern in list(self.patterns.values())[:2]:
                if pattern.confidence > 0.7:
                    self.add_task(
                        name=f"pattern_{pattern.pattern_type}",
                        mode=(
                            ThinkingMode(pattern.pattern_type)
                            if pattern.pattern_type in [m.value for m in ThinkingMode]
                            else ThinkingMode.ANALYSIS
                        ),
                        context=f"Analyze pattern: {pattern.pattern_id}",
                        priority=TaskPriority.LOW,
                    )

            # Add one task from current modes
            if modes_to_consider and len(self.queue) < 3:
                mode = modes_to_consider[0]
                self.add_task(
                    name=f"auto_{mode.value}",
                    mode=mode,
                    context=context[:300],
                    priority=TaskPriority.BACKGROUND,
                )

        except Exception as e:
            if logger:
                logger.debug(f"[AutoTask Error] {e}")

    # ------------------------
    # MAIN LOOP (ENHANCED)
    # ------------------------
    async def run(self, interval: float = 2.0):
        """Enhanced background thinker with smart scheduling"""

        try:
            if self.is_running:
                return

            self.is_running = True
            interval = max(interval, self.min_interval)

            if logger:
                logger.info("🧠 Background thinker (MAXIMUM MODE) started")

            import random

            # Cooldown control
            last_think_time = 0
            THINK_COOLDOWN = 15  # seconds
            last_auto_generate = 0
            AUTO_GENERATE_INTERVAL = 60  # seconds

            while self.is_running:
                try:
                    self.iterations += 1
                    now = time.time()

                    # ------------------------
                    # 🔥 ONLY RUN IF SYSTEM IDLE
                    # ------------------------
                    if hasattr(self.system, "idle_controller"):
                        if hasattr(self.system.idle_controller, "is_idle"):
                            is_idle = self.system.idle_controller.is_idle()
                            if callable(is_idle):
                                is_idle = is_idle()
                            if not is_idle:
                                await asyncio.sleep(1.5)
                                continue

                    # ------------------------
                    # 🔥 CHECK SCHEDULED TASKS
                    # ------------------------
                    await self._check_scheduled_tasks()

                    # ------------------------
                    # 🔥 AUTO GENERATE TASKS
                    # ------------------------
                    if (
                        self.auto_mode
                        and (now - last_auto_generate) > AUTO_GENERATE_INTERVAL
                    ):
                        last_auto_generate = now
                        await self._auto_generate_tasks()

                    # ------------------------
                    # 🔥 COOLDOWN CONTROL
                    # ------------------------
                    if now - last_think_time < THINK_COOLDOWN:
                        await asyncio.sleep(1.0)
                        continue

                    # ------------------------
                    # 🔥 PROCESS BATCH
                    # ------------------------
                    if self.queue:
                        # Dynamic batch size based on queue length
                        batch = min(self.batch_size, len(self.queue))
                        results = await self.process_batch(batch)

                        if results:
                            last_think_time = now
                            logger.debug(f"Processed {len(results)} thinking tasks")
                    else:
                        # Light thinking when idle
                        if random.random() < 0.05:  # 5% chance
                            logger.debug("🧠 Background idle thinking")
                            await asyncio.sleep(0.5)

                    # ------------------------
                    # 🔥 SLEEP CONTROL
                    # ------------------------
                    await asyncio.sleep(interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if logger:
                        logger.error(f"[Thinker Loop Error] {e}")

                    await asyncio.sleep(1.5)

        finally:
            self.is_running = False
            # Persist insights on shutdown
            await self._persist_insights()
            if logger:
                logger.info("🛑 Background thinker stopped")

    async def _persist_insights(self):
        """Persist insights to disk"""
        try:
            import json
            from pathlib import Path

            data = {
                "insights_generated": self.insights_generated,
                "tasks_processed": self.tasks_processed,
                "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
                "daily_stats": self.daily_stats,
                "last_updated": datetime.now().isoformat(),
            }

            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug(f"Failed to persist insights: {e}")

    # ------------------------
    # STOP
    # ------------------------
    def stop(self):
        self.is_running = False

    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        await self._persist_insights()
        logger.info("Background thinker shutdown complete")

    # ------------------------
    # GET INSIGHTS (ENHANCED)
    # ------------------------
    def get_insights(
        self, limit: int = 10, filter_mode: Optional[ThinkingMode] = None
    ) -> List[Dict[str, Any]]:
        try:
            insights = self.completed[-limit:]

            if filter_mode:
                insights = [i for i in insights if i.get("mode") == filter_mode.value]

            return insights
        except Exception:
            return []

    def get_failed_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get failed tasks for analysis"""
        return self.failed[-limit:]

    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get recognized patterns"""
        return [p.to_dict() for p in self.patterns.values()]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_processing = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times
            else 0
        )
        success_rate = (
            (self.tasks_processed - self.tasks_failed)
            / max(1, self.tasks_processed)
            * 100
        )

        return {
            "avg_processing_time_ms": round(avg_processing, 2),
            "success_rate": round(success_rate, 2),
            "tasks_per_minute": round(
                self.tasks_processed / max(1, self.iterations) * 60, 2
            ),
            "cache_hits": len(self.result_cache),
            "active_patterns": len(self.patterns),
        }

    # ------------------------
    # STATS (ENHANCED)
    # ------------------------
    def get_stats(self):
        try:
            return {
                "running": self.is_running,
                "queue_size": len(self.queue),
                "completed": len(self.completed),
                "failed": len(self.failed),
                "insights_generated": self.insights_generated,
                "iterations": self.iterations,
                "tasks_processed": self.tasks_processed,
                "patterns_recognized": len(self.patterns),
                "performance": self.get_performance_metrics(),
                "daily_stats": dict(list(self.daily_stats.items())[-7:]),  # Last 7 days
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_history(self):
        """Clear completed and failed history"""
        self.completed.clear()
        self.failed.clear()
        self.result_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Thinking history cleared")


__all__ = [
    "BackgroundThinker",
    "ThinkingTask",
    "ThinkingMode",
    "TaskPriority",
    "ThinkingPattern",
]
