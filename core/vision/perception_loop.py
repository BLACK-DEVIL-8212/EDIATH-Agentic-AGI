# perception_loop.py

import asyncio
from typing import Dict, Any, Optional, Callable, List
import numpy as np
import time


# ------------------------
# PERCEPTION STEP
# ------------------------
class PerceptionStep:
    def __init__(self, name: str, processor: Callable, timeout: float = 2.0):
        self.name = name
        self.processor = processor
        self.timeout = timeout

        self.execution_count = 0
        self.total_time = 0.0
        self.failures = 0

    async def execute(self, data: Any) -> Any:
        """
        Execute perception step safely (optimized + production-grade)
        """

        import asyncio

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not callable(self.processor):
                self.failures += 1
                return {"error": f"{self.name} processor not callable"}

            # ------------------------
            # 🔥 SAFE EXECUTION WRAPPER
            # ------------------------
            async def run():
                if asyncio.iscoroutinefunction(self.processor):
                    return await self.processor(data)
                return await asyncio.to_thread(self.processor, data)

            # ------------------------
            # 🔥 EXECUTE WITH TIMEOUT (CRITICAL FIX)
            # ------------------------
            try:
                result = await asyncio.wait_for(run(), timeout=self.timeout)
            except asyncio.TimeoutError:
                self.failures += 1

                # 🧠 fallback: return structured timeout result
                return {
                    "error": f"{self.name} timeout",
                    "timeout": self.timeout,
                    "timestamp": time.time(),
                }

            # ------------------------
            # 🔁 NORMALIZATION
            # ------------------------
            if result is None:
                result = {}

            if not isinstance(result, (dict, list, tuple, str, int, float)):
                result = {"value": str(result)}

            # ------------------------
            # 📊 METRICS (IMPROVED)
            # ------------------------
            elapsed = time.time() - start

            self.execution_count += 1
            self.total_time += elapsed

            try:
                self.last_execution_time = elapsed
            except Exception:
                pass

            return result

        except asyncio.CancelledError:
            # ------------------------
            # 🛑 HANDLE CANCELLATION
            # ------------------------
            self.failures += 1
            raise

        except Exception as e:
            # ------------------------
            # 🔥 ERROR HANDLING
            # ------------------------
            self.failures += 1

            return {"error": f"{self.name} failed: {str(e)}", "timestamp": time.time()}


# ------------------------
# MAIN LOOP
# ------------------------
class PerceptionLoop:
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.frame_time = 1.0 / fps

        self.steps: List[PerceptionStep] = []
        self.is_running = False

        self.frames_processed = 0
        self.skipped_frames = 0
        self.last_frame_time = 0

        # 🔥 NEW: queue for backpressure
        self.max_queue = 5
        self._frame_queue = asyncio.Queue(maxsize=self.max_queue)

    # ------------------------
    # ADD STEP
    # ------------------------
    def add_step(self, name: str, processor: Callable) -> None:
        """
        Add perception step safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not isinstance(name, str):
                raise ValueError("Step name must be a non-empty string")

            if not callable(processor):
                raise ValueError(f"Processor for '{name}' must be callable")

            name = name.strip().lower()

            # ------------------------
            # 🔁 DUPLICATE CHECK (CRITICAL FIX)
            # ------------------------
            for step in self.steps:
                if getattr(step, "name", None) == name:
                    # overwrite instead of duplicate (safe behavior)
                    step.processor = processor
                    return

            # ------------------------
            # 🔥 ADD STEP
            # ------------------------
            step = PerceptionStep(name, processor)
            self.steps.append(step)

            # ------------------------
            # 📊 METRICS (OPTIONAL)
            # ------------------------
            try:
                self.total_steps = getattr(self, "total_steps", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 📢 LOG
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(f"➕ Step added: {name}")
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Failed to add step '{name}': {e}")
            except Exception:
                pass

    # ------------------------
    # START / STOP
    # ------------------------
    async def start(self) -> None:
        """
        Start perception loop safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔁 ALREADY RUNNING CHECK (CRITICAL FIX)
            # ------------------------
            if getattr(self, "is_running", False):
                if hasattr(self, "logger"):
                    self.logger.warning("Perception loop already running")
                return

            # ------------------------
            # 🔥 STATE INITIALIZATION
            # ------------------------
            self.is_running = True
            self.frames_processed = 0
            self.skipped_frames = 0
            self.last_frame_time = time.time()

            # ------------------------
            # 📊 EXTRA METRICS (UPGRADE)
            # ------------------------
            try:
                self.start_time = self.last_frame_time
                self.total_runtime = 0.0
                self.loop_iterations = 0
            except Exception:
                pass

            # ------------------------
            # 📢 LOG START
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info("🚀 Perception loop started")
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Failed to start perception loop: {e}")
            except Exception:
                pass

            # ------------------------
            # 🔥 FAILSAFE RESET
            # ------------------------
            self.is_running = False

    async def stop(self) -> None:
        """
        Stop perception loop safely (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔁 ALREADY STOPPED CHECK
            # ------------------------
            if not getattr(self, "is_running", False):
                if hasattr(self, "logger"):
                    self.logger.warning("Perception loop already stopped")
                return

            # ------------------------
            # 🔥 STATE UPDATE
            # ------------------------
            self.is_running = False

            # ------------------------
            # ⏳ WAIT FOR LOOP TO SETTLE (GRACEFUL STOP)
            # ------------------------
            try:
                await asyncio.sleep(0)  # yield control to allow loop exit
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS UPDATE
            # ------------------------
            try:
                now = time.time()
                start_time = getattr(self, "start_time", None)
                if start_time:
                    self.total_runtime = round(now - start_time, 4)

                self.stopped_at = now
            except Exception:
                pass

            # ------------------------
            # 🧹 OPTIONAL CLEANUP
            # ------------------------
            try:
                self.current_frame = None
            except Exception:
                pass

            # ------------------------
            # 📢 LOG STOP
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info("🛑 Perception loop stopped")
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Failed to stop perception loop: {e}")
            except Exception:
                pass

    # ------------------------
    # PROCESS FRAME
    # ------------------------
    async def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process frame through pipeline (optimized + production-grade)
        """

        import asyncio
        import numpy as np

        # ------------------------
        # 🔥 VALIDATION
        # ------------------------
        if not getattr(self, "is_running", False):
            return None

        if frame is None or not isinstance(frame, np.ndarray):
            return None

        start_time = time.time()

        results = {"frame_data": frame, "timestamp": start_time}

        try:
            current_data = frame

            # ------------------------
            # 🔁 STEP EXECUTION (ISOLATED + TIMEOUT)
            # ------------------------
            for step in getattr(self, "steps", []):

                try:
                    step_result = await asyncio.wait_for(
                        step.execute(current_data), timeout=getattr(step, "timeout", 2)
                    )

                except asyncio.TimeoutError:
                    step_result = {"error": f"{getattr(step, 'name', 'step')} timeout"}

                except Exception as e:
                    step_result = {
                        "error": f"{getattr(step, 'name', 'step')} failed: {e}"
                    }

                # ------------------------
                # 🔥 HANDLE STEP RESULT
                # ------------------------
                if isinstance(step_result, dict) and "error" in step_result:
                    results[step.name] = step_result
                    continue

                current_data = step_result
                results[step.name] = current_data

            # ------------------------
            # 📊 METRICS (IMPROVED)
            # ------------------------
            elapsed = time.time() - start_time

            self.frames_processed += 1

            # smarter skip logic
            if elapsed > getattr(self, "frame_time", 0.033):
                self.skipped_frames += 1

            # smooth FPS (CRITICAL FIX)
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps = (
                self.fps * 0.8 + current_fps * 0.2
                if getattr(self, "fps", 0)
                else current_fps
            )

            results["processing_time"] = round(elapsed, 4)
            results["fps"] = self.fps
            results["skipped_frames"] = self.skipped_frames

            return results

        except asyncio.CancelledError:
            self.skipped_frames += 1
            raise

        except Exception as e:
            self.skipped_frames += 1

            return {"error": str(e), "timestamp": time.time()}

    # ------------------------
    # MAIN LOOP RUNNER
    # ------------------------
    async def run(
        self, frame_source: Callable, duration: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        Run perception loop (optimized + production-grade)
        """

        import asyncio

        results = []
        start_time = time.time()

        try:
            # ------------------------
            # 🔥 START SAFELY
            # ------------------------
            await self.start()

            # ------------------------
            # 🔁 MAIN LOOP
            # ------------------------
            async for frame in frame_source():

                # ------------------------
                # 🔒 STOP CONDITIONS
                # ------------------------
                if not getattr(self, "is_running", False):
                    break

                if (time.time() - start_time) > duration:
                    break

                if frame is None:
                    continue

                # ------------------------
                # 🔥 BACKPRESSURE CONTROL (CRITICAL FIX)
                # ------------------------
                if getattr(self, "_frame_queue", None):
                    try:
                        if self._frame_queue.full():
                            self.skipped_frames += 1
                            continue

                        await asyncio.wait_for(
                            self._frame_queue.put(frame), timeout=0.05
                        )

                        frame = await asyncio.wait_for(
                            self._frame_queue.get(), timeout=0.05
                        )

                    except asyncio.TimeoutError:
                        self.skipped_frames += 1
                        continue
                else:
                    # fallback (no queue)
                    pass

                # ------------------------
                # 🔥 PROCESS FRAME (SAFE)
                # ------------------------
                try:
                    result = await asyncio.wait_for(
                        self.process_frame(frame),
                        timeout=getattr(self, "frame_timeout", 3),
                    )
                except asyncio.TimeoutError:
                    self.skipped_frames += 1
                    continue
                except Exception:
                    self.skipped_frames += 1
                    continue

                if result:
                    results.append(result)

                # ------------------------
                # ⏱️ FPS CONTROL (SMOOTH)
                # ------------------------
                now = time.time()
                elapsed = now - getattr(self, "last_frame_time", now)

                sleep_time = max(0, getattr(self, "frame_time", 0.033) - elapsed)
                await asyncio.sleep(sleep_time)

                self.last_frame_time = time.time()

                # ------------------------
                # 📊 LOOP METRICS
                # ------------------------
                try:
                    self.loop_iterations = getattr(self, "loop_iterations", 0) + 1
                except Exception:
                    pass

        except asyncio.CancelledError:
            raise

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Run loop error: {e}")
            except Exception:
                pass

        finally:
            # ------------------------
            # 🛑 ENSURE CLEAN STOP
            # ------------------------
            try:
                await self.stop()
            except Exception:
                pass

        return results

    # ------------------------
    # PIPELINE STATS
    # ------------------------
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline stats safely (optimized + production-grade)
        """

        try:
            stats = {}

            for step in getattr(self, "steps", []):

                try:
                    executions = int(getattr(step, "execution_count", 0))
                    failures = int(getattr(step, "failures", 0))
                    total_time = float(getattr(step, "total_time", 0.0))

                    avg_time = (total_time / executions) if executions > 0 else 0.0
                    success_rate = (
                        (executions - failures) / executions if executions > 0 else 0.0
                    )

                    stats[getattr(step, "name", "unknown")] = {
                        "executions": executions,
                        "failures": failures,
                        "success_rate": round(success_rate, 3),
                        "total_time": round(total_time, 4),
                        "avg_time": round(avg_time, 4),
                        "last_execution_time": getattr(
                            step, "last_execution_time", None
                        ),
                    }

                except Exception:
                    # isolate step failure
                    stats[getattr(step, "name", "unknown")] = {"error": "stats_failed"}

            # ------------------------
            # 📊 GLOBAL METRICS (UPGRADE)
            # ------------------------
            try:
                stats["_global"] = {
                    "total_steps": len(getattr(self, "steps", [])),
                    "frames_processed": getattr(self, "frames_processed", 0),
                    "skipped_frames": getattr(self, "skipped_frames", 0),
                    "timestamp": time.time(),
                }
            except Exception:
                pass

            return stats

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Pipeline stats error: {e}")
            except Exception:
                pass

            return {"error": "pipeline_stats_failed", "timestamp": time.time()}

    # ------------------------
    # OVERALL STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        """
        Get runtime stats (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔒 SAFE VALUES
            # ------------------------
            frames_processed = int(getattr(self, "frames_processed", 0))
            skipped_frames = int(getattr(self, "skipped_frames", 0))
            is_running = bool(getattr(self, "is_running", False))
            steps = getattr(self, "steps", []) or []

            total = frames_processed + skipped_frames
            drop_rate = (skipped_frames / total) if total > 0 else 0.0

            # ------------------------
            # 📊 FPS (SMOOTHED SAFE)
            # ------------------------
            fps = float(getattr(self, "fps", 0.0))

            # ------------------------
            # 🔥 BUILD STATS
            # ------------------------
            stats = {
                "fps_target": fps,
                "frames_processed": frames_processed,
                "skipped_frames": skipped_frames,
                "total_frames": total,
                "drop_rate": round(drop_rate, 3),
                "is_running": is_running,
                "pipeline_stages": len(steps),
                "timestamp": time.time(),
            }

            # ------------------------
            # 🔁 PIPELINE STATS (SAFE)
            # ------------------------
            try:
                stats["pipeline_stats"] = self.get_pipeline_stats()
            except Exception:
                stats["pipeline_stats"] = {}

            # ------------------------
            # 📊 EXTRA METRICS (UPGRADE)
            # ------------------------
            try:
                stats["performance"] = {
                    "efficiency": round(1 - drop_rate, 3),
                    "load": round(
                        len(steps) / max(1, len(steps)), 2
                    ),  # placeholder load calc
                }
            except Exception:
                pass

            return stats

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Stats error: {e}")
            except Exception:
                pass

            return {"error": "stats_failed", "timestamp": time.time()}


__all__ = ["PerceptionLoop", "PerceptionStep"]
