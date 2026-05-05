"""
Advanced Autonomous Core - Real AI Decision Engine (FINAL STABLE)
with Chunking, Pipeline Processing & Fixed Auto-Healing
"""

import asyncio
import json
import re
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field

from ..utils.logger import logger
from ..brain.llm_engine import LLMEngine
from ..agent.action_router import ActionRouter
from ..agent.self_reflection import SelfReflection
from ..memory import MemoryManager


class AutonomyLevel(Enum):
    MANUAL = 0
    SUGGESTION = 1
    SEMI_AUTO = 2
    FULL_AUTO = 3


@dataclass
class Decision:
    """Decision data class with execution tracking"""

    action: str
    reasoning: str
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    executed: bool = False
    result: Any = None
    execution_time_ms: float = 0.0


@dataclass
class DecisionChunk:
    """Chunk of decisions for batch processing"""

    chunk_id: int
    decisions: List[Decision]
    total_chunks: int
    timestamp: float = field(default_factory=time.time)


class AutonomousCoreEngine:
    """Advanced Autonomous Core with chunking and pipeline processing"""

    def __init__(self, autonomy_level=AutonomyLevel.SEMI_AUTO):
        self.autonomy_level = autonomy_level

        self.llm = LLMEngine()
        self.router = ActionRouter()
        self.memory = MemoryManager()
        self.reflection = SelfReflection()

        self.decisions: List[Decision] = []
        self.executed_actions = 0
        self.decision_queue: asyncio.Queue = None
        self.result_queue: asyncio.Queue = None

        # Chunking configuration
        self.chunk_size = 10
        self.pipeline_workers = 4
        self.decision_chunks: List[DecisionChunk] = []

        # State management
        self._running = False
        self._paused = False
        self._initialized = False
        self._shutdown_event = asyncio.Event()

        # Auto-healing protection (FIXED)
        self._last_healing_attempt = 0
        self._healing_cooldown = 30  # seconds between healing attempts
        self._healing_consecutive = 0
        self._healing_lock = asyncio.Lock()
        self._last_cache_cleanup = 0
        self._cache_cleanup_interval = 60  # seconds

        # Pipeline workers
        self._pipeline_tasks: List[asyncio.Task] = []

        # Metrics
        self.metrics = {
            "decisions_made": 0,
            "actions_executed": 0,
            "chunks_processed": 0,
            "avg_decision_time_ms": 0,
            "errors": 0,
            "healing_attempts": 0,
            "healing_successes": 0,
        }

    # ------------------------
    # INITIALIZATION
    # ------------------------
    async def initialize(self, *args, **kwargs) -> bool:
        """
        Initialize AutonomousCoreEngine safely with pipeline setup.
        """
        try:
            if getattr(self, "_initialized", False):
                return True

            # Initialize queues
            self.decision_queue = asyncio.Queue(maxsize=100)
            self.result_queue = asyncio.Queue(maxsize=100)

            # Optional: validate subcomponents
            for comp in (self.llm, self.router, self.memory, self.reflection):
                if hasattr(comp, "initialize"):
                    try:
                        await comp.initialize()
                    except Exception:
                        pass

            self._initialized = True
            self._running = False
            self._paused = False

            logger.info("🧠 AutonomousCore initialized with chunking pipeline")
            return True

        except Exception as e:
            logger.error(f"❌ AutonomousCore initialize failed: {e}")
            return False

    # ------------------------
    # START
    # ------------------------
    def start(self) -> bool:
        """
        Start AutonomousCoreEngine with pipeline workers.
        """
        try:
            if not getattr(self, "_initialized", False):
                asyncio.create_task(self.initialize())

            if self._running:
                logger.warning("⚠ AutonomousCore already running")
                return True

            self._running = True
            self._paused = False
            self._shutdown_event.clear()

            # Start pipeline workers
            for worker_id in range(self.pipeline_workers):
                task = asyncio.create_task(
                    self._decision_pipeline_worker(worker_id),
                    name=f"autonomous_worker_{worker_id}",
                )
                self._pipeline_tasks.append(task)

            # Start chunk processor
            task = asyncio.create_task(
                self._chunk_processor(), name="autonomous_chunk_processor"
            )
            self._pipeline_tasks.append(task)

            # Start metrics collector
            task = asyncio.create_task(
                self._collect_metrics(), name="autonomous_metrics"
            )
            self._pipeline_tasks.append(task)

            # Start subcomponents if needed
            for comp in (self.llm, self.router, self.memory, self.reflection):
                if hasattr(comp, "start"):
                    try:
                        comp.start()
                    except Exception:
                        pass

            logger.info(
                f"🚀 AutonomousCore started with {self.pipeline_workers} workers"
            )
            return True

        except Exception as e:
            logger.error(f"❌ AutonomousCore start failed: {e}")
            self._running = False
            return False

    # ------------------------
    # STOP
    # ------------------------
    def stop(self) -> bool:
        """
        Stop AutonomousCoreEngine safely.
        """
        try:
            if not self._running:
                return True

            self._running = False
            self._shutdown_event.set()

            # Cancel pipeline workers
            for task in self._pipeline_tasks:
                if not task.done():
                    task.cancel()

            if self._pipeline_tasks:
                asyncio.create_task(self._cleanup_tasks())

            # Stop subcomponents
            for comp in (self.llm, self.router, self.memory, self.reflection):
                if hasattr(comp, "stop"):
                    try:
                        comp.stop()
                    except Exception:
                        pass

            logger.info("🛑 AutonomousCore stopped")
            return True

        except Exception as e:
            logger.error(f"❌ AutonomousCore stop failed: {e}")
            return False

    async def _cleanup_tasks(self):
        """Clean up pipeline tasks"""
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pipeline_tasks, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Pipeline task cleanup timeout")
        finally:
            self._pipeline_tasks.clear()

    # ------------------------
    # DECISION PIPELINE WORKER
    # ------------------------
    async def _decision_pipeline_worker(self, worker_id: int):
        """Worker for processing decisions in parallel"""
        logger.debug(f"Decision worker {worker_id} started")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Get decision chunk from queue
                chunk = await asyncio.wait_for(self.decision_queue.get(), timeout=1.0)

                if chunk is None:
                    break

                # Process each decision in chunk
                for decision in chunk.decisions:
                    if not self._running:
                        break

                    start_time = time.time()

                    # Execute decision
                    result = await self.execute_decision(decision)

                    # Calculate execution time
                    execution_time = (time.time() - start_time) * 1000
                    decision.execution_time_ms = execution_time

                    # Store result
                    await self.result_queue.put(
                        {
                            "worker_id": worker_id,
                            "decision": decision,
                            "result": result,
                            "timestamp": time.time(),
                        }
                    )

                self.decision_queue.task_done()
                self.metrics["chunks_processed"] += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.metrics["errors"] += 1
                await asyncio.sleep(0.1)

        logger.debug(f"Decision worker {worker_id} stopped")

    # ------------------------
    # CHUNK PROCESSOR
    # ------------------------
    async def _chunk_processor(self):
        """Process decisions in chunks"""
        decision_buffer = []
        last_chunk_time = time.time()
        chunk_interval = 1.0  # Process chunks every second

        while self._running and not self._shutdown_event.is_set():
            try:
                # Build context
                context = await self._build_context()

                # Evaluate situation
                decision = await self.evaluate_situation(context)

                if decision:
                    decision_buffer.append(decision)
                    self.metrics["decisions_made"] += 1

                # Create chunk when buffer is full or time expired
                current_time = time.time()
                if len(decision_buffer) >= self.chunk_size or (
                    decision_buffer and current_time - last_chunk_time >= chunk_interval
                ):

                    if decision_buffer:
                        chunk = DecisionChunk(
                            chunk_id=len(self.decision_chunks),
                            decisions=decision_buffer.copy(),
                            total_chunks=len(self.decision_chunks) + 1,
                            timestamp=current_time,
                        )

                        self.decision_chunks.append(chunk)
                        await self.decision_queue.put(chunk)

                        decision_buffer.clear()
                        last_chunk_time = current_time

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Chunk processor error: {e}")
                await asyncio.sleep(0.5)

    # ------------------------
    # AI DECISION (FIXED)
    # ------------------------
    async def evaluate_situation(self, context: Dict[str, Any]) -> Optional[Decision]:
        """
        Evaluate situation and make decision with chunking support
        """

        if self.autonomy_level == AutonomyLevel.MANUAL:
            return None

        data: Dict[str, Any] = {}
        raw: str = ""

        try:
            # ------------------------
            # CONTEXT SAFE with chunking
            # ------------------------
            try:
                context_str = json.dumps(context)
            except Exception:
                context_str = str(context)

            # Chunk large contexts
            if len(context_str) > 3000:
                context_str = context_str[:3000] + "..."

            # ------------------------
            # PROMPT with structured output
            # ------------------------
            prompt = f"""
Return ONLY JSON.

Context:
{context_str}

Format:
{{
    "action": "string",
    "params": {{}},
    "confidence": 0.0,
    "reasoning": "string"
}}

Available actions: explore, learn, act, wait, communicate, system_status, noop
"""

            # ------------------------
            # LLM CALL with timeout
            # ------------------------
            try:
                response = await asyncio.wait_for(
                    self.llm.generate(prompt), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Decision generation timeout")
                return Decision("explore", "timeout fallback", 0.4, {})

            # ------------------------
            # SAFE EXTRACTION
            # ------------------------
            if isinstance(response, dict):
                raw = response.get("response", "")
            else:
                raw = response

            raw = str(raw).strip()

            # ------------------------
            # PARSE JSON (SAFE)
            # ------------------------
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None

            if isinstance(parsed, dict):
                data = parsed
            else:
                # Extract JSON block
                matches = re.findall(r"\{[\s\S]*?\}", raw)

                for m in reversed(matches):
                    try:
                        temp = json.loads(m)
                        if isinstance(temp, dict) and "action" in temp:
                            data = temp
                            break
                    except Exception:
                        continue

            # ------------------------
            # FINAL VALIDATION
            # ------------------------
            if not isinstance(data, dict) or "action" not in data:
                return Decision("explore", "fallback", 0.5, {})

            # ------------------------
            # CLEAN VALUES
            # ------------------------
            action = str(data.get("action", "noop")).lower().strip()
            reasoning = str(data.get("reasoning", ""))

            try:
                confidence = float(data.get("confidence", 0.5))
            except Exception:
                confidence = 0.5

            confidence = max(0.0, min(1.0, confidence))

            params = data.get("params", {})
            if not isinstance(params, dict):
                params = {}

            # ------------------------
            # CREATE DECISION
            # ------------------------
            decision = Decision(action, reasoning, confidence, params)

            self.decisions.append(decision)

            # Keep only last 100 decisions
            if len(self.decisions) > 100:
                self.decisions = self.decisions[-100:]

            return decision

        except Exception as e:
            logger.error(f"Decision failed: {e}")
            self.metrics["errors"] += 1

            # NEVER CRASH
            return Decision("noop", "error fallback", 0.3, {})

    # ------------------------
    # EXECUTION
    # ------------------------
    async def execute_decision(self, decision: Decision) -> Dict[str, Any]:
        """
        Execute decision safely with approval checks
        """

        # Check if approval needed
        if self.autonomy_level == AutonomyLevel.SEMI_AUTO and decision.confidence < 0.6:
            return {
                "status": "requires_approval",
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            }

        try:
            action = decision.action.lower().strip()

            # Built-in actions
            if action == "system_status":
                result = {
                    "status": "running",
                    "autonomy": self.autonomy_level.name,
                    "decisions": len(self.decisions),
                    "executed": self.executed_actions,
                    "metrics": self.metrics,
                }

            elif action == "pause":
                self.pause()
                result = {"status": "paused", "action": action}

            elif action == "resume":
                self.resume()
                result = {"status": "resumed", "action": action}

            else:
                try:
                    result = await self.router.route(action, decision.params)
                except Exception as router_error:
                    logger.warning(f"Router failed for '{action}': {router_error}")
                    result = {
                        "status": "ignored",
                        "action": action,
                        "message": f"Unknown action handled safely: {router_error}",
                    }

            decision.executed = True
            decision.result = result
            self.executed_actions += 1
            self.metrics["actions_executed"] += 1

            # Update average decision time
            total_time = self.metrics["avg_decision_time_ms"] * (
                self.executed_actions - 1
            )
            self.metrics["avg_decision_time_ms"] = (
                total_time + decision.execution_time_ms
            ) / self.executed_actions

            # Reflect on result
            try:
                self.reflection.reflect(
                    topic=decision.action,
                    observation=decision.reasoning,
                    result=result,
                    success=result is not None,
                )
            except Exception as e:
                logger.debug(f"Reflection failed: {e}")

            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.metrics["errors"] += 1
            return {"status": "error", "message": str(e)}

    # ------------------------
    # CONTROL METHODS
    # ------------------------
    def pause(self):
        """Pause autonomous operations"""
        self._paused = True
        logger.info("🛑 Autonomous core paused")

    def resume(self):
        """Resume autonomous operations"""
        self._paused = False
        logger.info("▶️ Autonomous core resumed")

    # ------------------------
    # AUTO-HEALING (FIXED - NO SPAM)
    # ------------------------
    async def auto_heal(self):
        """
        Production-safe auto-healing with cooldown and circuit breaker
        """
        # ------------------------
        # PREVENT RECURSIVE HEALING (CRITICAL FIX)
        # ------------------------
        async with self._healing_lock:
            now = time.time()

            # Cooldown check - prevent spam
            if now - self._last_healing_attempt < self._healing_cooldown:
                return

            # Circuit breaker - if too many failures, stop trying
            if self._healing_consecutive >= 5:
                logger.warning(
                    "⚠ Auto-healing circuit breaker OPEN - too many attempts"
                )
                return

            self._last_healing_attempt = now
            self.metrics["healing_attempts"] += 1

            try:
                # Check if healing is actually needed
                if not await self._needs_healing():
                    self._healing_consecutive = 0
                    return

                logger.info("🔄 Running auto-healing procedures")

                # Perform healing actions
                success = await self._perform_healing()

                if success:
                    self._healing_consecutive = 0
                    self.metrics["healing_successes"] += 1
                    logger.info("✅ Auto-healing successful")
                else:
                    self._healing_consecutive += 1
                    logger.warning(
                        f"⚠ Auto-healing failed (attempt {self._healing_consecutive}/5)"
                    )

            except Exception as e:
                self._healing_consecutive += 1
                logger.error(f"❌ Auto-healing error: {e}")

    async def _needs_healing(self) -> bool:
        """Check if system actually needs healing"""
        try:
            # Check component health
            components_healthy = True

            for comp_name, comp in [
                ("LLM", self.llm),
                ("Router", self.router),
                ("Memory", self.memory),
                ("Reflection", self.reflection),
            ]:
                if hasattr(comp, "health_check"):
                    try:
                        health = await comp.health_check()
                        if (
                            isinstance(health, dict)
                            and health.get("status") != "healthy"
                        ):
                            components_healthy = False
                            logger.debug(f"{comp_name} unhealthy: {health}")
                    except Exception:
                        components_healthy = False

            # Check error rate
            total_ops = max(1, self.metrics["decisions_made"])
            error_rate = self.metrics["errors"] / total_ops

            # Only heal if error rate > 20% or components unhealthy
            needs_heal = error_rate > 0.2 or not components_healthy

            if not needs_heal:
                logger.debug(f"System healthy (error rate: {error_rate:.1%})")

            return needs_heal

        except Exception as e:
            logger.debug(f"Health check error: {e}")
            return False

    async def _perform_healing(self) -> bool:
        """Perform actual healing operations"""
        success_count = 0
        total_attempts = 0

        try:
            # Heal components
            for comp in [self.llm, self.router, self.memory, self.reflection]:
                total_attempts += 1
                if hasattr(comp, "heal"):
                    try:
                        healed = await comp.heal()
                        if healed:
                            success_count += 1
                    except Exception as e:
                        logger.debug(f"Heal failed for {comp}: {e}")

            # Clear old decisions if memory is high
            if len(self.decisions) > 50:
                self.decisions = self.decisions[-50:]
                success_count += 1

            # Reset error counter if healing partially successful
            if success_count > 0:
                self.metrics["errors"] = max(0, self.metrics["errors"] - 5)

            return success_count > 0

        except Exception as e:
            logger.error(f"Healing execution error: {e}")
            return False

    # ------------------------
    # CACHE CLEANUP (RATE LIMITED)
    # ------------------------
    async def _rate_limited_cache_cleanup(self):
        """Clean cache with rate limiting to prevent spam"""
        now = time.time()

        # Only cleanup if enough time has passed
        if now - self._last_cache_cleanup < self._cache_cleanup_interval:
            return

        # Check if memory is actually high
        if not self._is_memory_high():
            return

        self._last_cache_cleanup = now

        # Perform cleanup
        try:
            # Clear old decision results
            for decision in self.decisions:
                if decision.execution_time_ms > 1000:  # Old decisions
                    decision.result = None  # Free memory

            logger.info("🧹 Cache cleared (rate-limited)")
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")

    def _is_memory_high(self) -> bool:
        """Check if memory usage is actually high"""
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > 0.9  # 90% threshold
        except ImportError:
            return False  # Don't clean if can't check

    # ------------------------
    # METRICS COLLECTOR
    # ------------------------
    async def _collect_metrics(self):
        """Collect and log metrics periodically"""
        while self._running and not self._shutdown_event.is_set():
            await asyncio.sleep(30)  # Every 30 seconds

            if self.executed_actions > 0:
                logger.info(
                    f"📊 Autonomous Metrics: decisions={self.metrics['decisions_made']}, "
                    f"executed={self.metrics['actions_executed']}, "
                    f"chunks={self.metrics['chunks_processed']}, "
                    f"avg_time={self.metrics['avg_decision_time_ms']:.2f}ms, "
                    f"errors={self.metrics['errors']}"
                )

            # Run rate-limited cache cleanup
            await self._rate_limited_cache_cleanup()

            # Run auto-healing (rate-limited)
            await self.auto_heal()

    # ------------------------
    # LOOP METHODS
    # ------------------------
    async def autonomous_loop(self, interval=2.0):
        """
        Main autonomous loop with chunking support
        """
        logger.info("🧠 Autonomous core running with pipeline")

        self._running = True

        while self._running and not self._shutdown_event.is_set():
            if self._paused:
                await asyncio.sleep(interval)
                continue

            try:
                # Build context
                context = await self._build_context()

                # Evaluate and queue decision
                decision = await self.evaluate_situation(context)

                if decision:
                    # Process through pipeline
                    chunk = DecisionChunk(
                        chunk_id=len(self.decision_chunks),
                        decisions=[decision],
                        total_chunks=1,
                    )
                    await self.decision_queue.put(chunk)

                    # Get result
                    try:
                        result = await asyncio.wait_for(
                            self.result_queue.get(), timeout=5.0
                        )
                        logger.debug(f"Decision executed: {decision.action}")
                    except asyncio.TimeoutError:
                        logger.warning("Result timeout")

            except Exception as e:
                logger.error(f"Loop error: {e}")

            await asyncio.sleep(interval)

        logger.info("🛑 Autonomous loop stopped")

    async def autonomous_operation(self, duration: int = 30, interval: float = 2.0):
        """
        Run autonomous operation for specified duration
        """
        logger.info(f"🚀 Starting autonomous operation for {duration}s")

        self._running = True
        start_time = time.time()

        while self._running and not self._shutdown_event.is_set():
            if time.time() - start_time >= duration:
                break

            if self._paused:
                await asyncio.sleep(interval)
                continue

            try:
                context = await self._build_context()
                decision = await self.evaluate_situation(context)

                if decision:
                    await self.execute_decision(decision)

            except Exception as e:
                logger.error(f"Operation error: {e}")

            await asyncio.sleep(interval)

        self._running = False
        logger.info("🛑 Autonomous operation completed")

    # ------------------------
    # CONTEXT BUILDING
    # ------------------------
    async def _build_context(self) -> Dict[str, Any]:
        """Build context with chunked memory access"""
        memory_data = []

        if hasattr(self.memory, "get_recent"):
            try:
                memory_data = await self.memory.get_recent()
            except Exception as e:
                logger.debug(f"Memory access error: {e}")

        if isinstance(memory_data, list):
            memory_data = memory_data[-20:]  # Keep only recent

        return {
            "memory": memory_data,
            "timestamp": time.time(),
            "autonomy_level": self.autonomy_level.value,
            "stats": self.get_stats(),
        }

    # ------------------------
    # STATISTICS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "decisions": len(self.decisions),
            "executed": self.executed_actions,
            "autonomy": self.autonomy_level.name,
            "chunks_processed": self.metrics["chunks_processed"],
            "avg_decision_time_ms": round(self.metrics["avg_decision_time_ms"], 2),
            "errors": self.metrics["errors"],
            "healing_attempts": self.metrics["healing_attempts"],
            "healing_successes": self.metrics["healing_successes"],
            "pipeline_workers": len(self._pipeline_tasks),
            "queue_size": self.decision_queue.qsize() if self.decision_queue else 0,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the autonomous core"""
        return {
            "status": "healthy" if self._running else "stopped",
            "initialized": self._initialized,
            "running": self._running,
            "paused": self._paused,
            "stats": self.get_stats(),
            "metrics": self.metrics,
            "healing_cooldown_remaining": max(
                0, self._healing_cooldown - (time.time() - self._last_healing_attempt)
            ),
        }

    async def background_process(self, chunked_mode: bool = False):
        """Background process for autonomous operations"""
        if chunked_mode:
            await self.autonomous_loop()
        else:
            await self.autonomous_operation(duration=60)


__all__ = ["AutonomousCoreEngine", "AutonomyLevel", "Decision"]
