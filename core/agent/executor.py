"""
Production Executor for EDIATH R_AGENT (FINAL)
✔ Structured step execution (planner v2 compatible)
✔ Tool + LLM routing
✔ Timeout protection
✔ Error isolation
✔ Logging
✔ Safe fallback
"""

import asyncio
from typing import Dict, Any, List

from core.agent.tools import Tools

try:
    from ..utils.logger import logger
except:
    logger = None


class Executor:
    def __init__(self, system=None):
        try:
            # 🔥 core system
            self.system = system

            # 🔥 FIXED: Allow None system for initialization
            if not self.system:
                if logger:
                    # Changed from warning to info - this is expected during startup
                    logger.info(
                        "Executor initialized without system - will connect when system is ready"
                    )
                # Continue with limited functionality instead of failing

            # 🔥 tools layer
            try:
                if self.system:
                    self.tools = Tools(system)
                else:
                    self.tools = None
                    if logger:
                        logger.debug("Tools not initialized because system is None")
            except Exception as e:
                if logger:
                    logger.debug(f"Tools initialization pending: {e}")
                self.tools = None

            # 🔥 execution limits
            self.timeout = 15
            self.max_steps = 5

            # 🔥 retry config
            self.max_retries = 1

            # 🔥 feature flags (check system availability)
            self.enable_tools = self.tools is not None and self.system is not None
            self.enable_llm = (
                hasattr(self.system, "llm_engine") if self.system else False
            )

            # 🔥 debug
            self.debug = False

            if logger and self.system:
                logger.info("Executor initialized successfully with system")
            elif logger:
                logger.debug("Executor initialized in standby mode")

        except Exception as e:
            if logger:
                logger.warning(f"[Executor Init Error] {e}")

            # 🔥 fallback safe init (always works)
            self.system = system  # Can be None
            self.tools = None
            self.timeout = 15
            self.max_steps = 5
            self.max_retries = 0
            self.enable_tools = False
            self.enable_llm = False
            self.debug = False

    # ------------------------
    # MAIN EXECUTE
    # ------------------------
    async def execute(self, plan: Dict[str, Any]) -> str:
        try:
            # ------------------------
            # 🔥 VALIDATE PLAN
            # ------------------------
            if not isinstance(plan, dict):
                return self._fallback("Invalid plan")

            route = str(plan.get("route", "llm")).lower()
            steps = plan.get("steps", [])

            if not isinstance(steps, list) or not steps:
                return self._fallback("No steps provided")

            # 🔥 limit steps (safety)
            steps = steps[: getattr(self, "max_steps", 5)]

            # ------------------------
            # 🔥 EXECUTION WITH TIMEOUT
            # ------------------------
            result = await asyncio.wait_for(
                self._execute_steps(route, steps), timeout=getattr(self, "timeout", 15)
            )

            # ------------------------
            # 🔥 VALIDATE OUTPUT
            # ------------------------
            if not result:
                return self._fallback("Empty execution result")

            return str(result)

        except asyncio.TimeoutError:
            return self._fallback("Execution timeout")

        except Exception as e:
            if logger:
                logger.warning(f"[Executor Error] {e}")
            return self._fallback(f"Execution error: {str(e)}")

    # ------------------------
    # STEP EXECUTION (UPDATED)
    # ------------------------
    async def _execute_steps(self, route: str, steps: List[Dict[str, Any]]) -> str:
        try:
            result = ""
            step_count = 0

            for step in steps:
                try:
                    # 🔥 limit execution
                    step_count += 1
                    if step_count > getattr(self, "max_steps", 5):
                        break

                    if not isinstance(step, dict):
                        continue

                    step_type = str(step.get("type", "")).lower()
                    action = str(step.get("action", "")).strip()
                    data = step.get("input", "")

                    if not step_type or not action:
                        continue

                    # ------------------------
                    # 🔥 TOOL EXECUTION
                    # ------------------------
                    if step_type == "tool" and getattr(self, "enable_tools", False):
                        result = await self._run_tool(action, data)

                    # ------------------------
                    # 🔥 SYSTEM EXECUTION
                    # ------------------------
                    elif step_type == "system":
                        result = self._system_status()

                    # ------------------------
                    # 🔥 LLM EXECUTION
                    # ------------------------
                    elif step_type == "llm" and getattr(self, "enable_llm", False):
                        result = await self._call_llm(data)

                    else:
                        continue

                    # 🔥 validate result
                    if result:
                        result = str(result).strip()

                    # 🔥 continue chain if multi-step enabled
                    if result and not getattr(self, "enable_tools", True):
                        break

                    # 🔥 break on meaningful result
                    if result and len(result) > 2:
                        break

                except Exception as e:
                    if logger:
                        logger.debug(f"[Executor Step Error] {e}")
                    continue

            return result if result else "No result generated."

        except Exception as e:
            if logger:
                logger.warning(f"[Executor Core Error] {e}")

            return self._fallback("Step execution failed")

    # ------------------------
    # TOOL HANDLER
    # ------------------------
    async def _run_tool(self, action: str, data: str) -> str:
        try:
            # ------------------------
            # 🔥 VALIDATE
            # ------------------------
            if not action:
                return ""

            if not getattr(self, "enable_tools", False) or not self.tools:
                return ""

            action = str(action).strip().lower()
            data = str(data) if data is not None else ""

            # ------------------------
            # 🔥 TOOL MAP (SCALABLE)
            # ------------------------
            tool_map = {
                "capture_frame": self.tools.vision,
                "analyze_scene": self.tools.vision,
                "vision": self.tools.vision,
                "fetch_memory": self.tools.memory,
                "memory": self.tools.memory,
                "generate_code": lambda: self.tools.code(data),
                "code": lambda: self.tools.code(data),
            }

            tool_func = tool_map.get(action)

            if not tool_func:
                if logger:
                    logger.debug(f"[Tool] Unknown action: {action}")
                return ""

            # ------------------------
            # 🔥 EXECUTE TOOL
            # ------------------------
            result = (
                await tool_func()
                if asyncio.iscoroutinefunction(tool_func)
                else tool_func()
            )

            # 🔥 VALIDATE OUTPUT
            if result:
                return str(result).strip()

            return ""

        except Exception as e:
            if logger:
                logger.debug(f"[Tool Execution Error] {e}")

            return ""

    # ------------------------
    # LLM CALL
    # ------------------------
    async def _call_llm(self, prompt: str) -> str:
        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            if not prompt:
                return ""

            if not getattr(self, "enable_llm", False):
                return ""

            if not self.system or not hasattr(self.system, "llm_engine"):
                return ""

            # 🔥 normalize (NO LIMIT as per your requirement)
            prompt = str(prompt).strip()

            if not prompt:
                return ""

            # ------------------------
            # 🔥 CALL LLM (QUEUE SAFE)
            # ------------------------
            # Check if queued_generate exists, otherwise use generate
            if hasattr(self.system.llm_engine, "queued_generate"):
                result = await self.system.llm_engine.queued_generate(
                    prompt=prompt, system_prompt="You are EDIATH."
                )
            else:
                result = await self.system.llm_engine.generate(prompt)

            # ------------------------
            # 🔥 VALIDATE OUTPUT
            # ------------------------
            if not result:
                return ""

            # Handle different return types
            if isinstance(result, dict):
                response = result.get("response", "")
            elif isinstance(result, str):
                response = result
            else:
                response = str(result)

            if not response:
                return ""

            response = str(response).strip()

            # 🔥 basic cleanup
            response = (
                response.replace("assistant:", "").replace("ASSISTANT:", "").strip()
            )

            return response

        except Exception as e:
            if logger:
                logger.debug(f"[LLM Error] {e}")

            return ""

    # ------------------------
    # SYSTEM STATUS
    # ------------------------
    def _system_status(self) -> str:
        try:
            status = {
                # 🔥 core system
                "system": (
                    "running"
                    if (self.system and getattr(self.system, "is_running", False))
                    else "stopped"
                ),
                # 🔥 components
                "llm": (
                    "available" if getattr(self, "enable_llm", False) else "unavailable"
                ),
                "tools": (
                    "available"
                    if getattr(self, "enable_tools", False)
                    else "unavailable"
                ),
                # 🔥 subsystems (safe checks)
                "vision": (
                    "ready"
                    if (
                        self.system
                        and getattr(self.system, "vision_engine_instance", None)
                    )
                    else "unavailable"
                ),
                "memory": (
                    "ready"
                    if (self.system and getattr(self.system, "memory", None))
                    else "unavailable"
                ),
                # 🔥 executor info
                "max_steps": getattr(self, "max_steps", 5),
                "timeout": getattr(self, "timeout", 15),
            }

            # 🔥 format output (clean string)
            return " | ".join(f"{k}:{v}" for k, v in status.items())

        except Exception as e:
            if logger:
                logger.debug(f"[System Status Error] {e}")

            return "System status unavailable."

    # ------------------------
    # FALLBACK
    # ------------------------
    def _fallback(self, reason: str) -> str:
        try:
            if logger:
                logger.debug(f"[Executor Fallback] {reason}")

            reason = str(reason).lower() if reason else ""

            # ------------------------
            # 🔥 CONTEXT-AWARE RESPONSES
            # ------------------------
            if "invalid plan" in reason:
                return "I couldn't understand the request structure."

            if "no steps" in reason:
                return "There was nothing to execute for your request."

            if "timeout" in reason:
                return "The request took too long to process."

            if "llm" in reason:
                return "I couldn't generate a response at the moment."

            if "tool" in reason:
                return "I couldn't execute the required operation."

            if "error" in reason:
                return "Something went wrong while processing your request."

            if "empty" in reason:
                return "I didn't get a valid result from the execution."

            # ------------------------
            # 🔥 DEFAULT FALLBACK
            # ------------------------
            return "Sorry, I couldn't complete the request."

        except Exception as e:
            if logger:
                logger.debug(f"[Executor Fallback Error] {e}")

            return "Sorry, something went wrong."

    # ------------------------
    # CONNECT TO SYSTEM (FOR LATE BINDING)
    # ------------------------
    def connect_system(self, system):
        """Connect or reconnect to the EDIATH system"""
        try:
            self.system = system

            # Re-initialize tools with the new system
            if self.system:
                try:
                    self.tools = Tools(system)
                    self.enable_tools = self.tools is not None
                    self.enable_llm = hasattr(self.system, "llm_engine")

                    if logger:
                        logger.info("Executor successfully connected to system")
                except Exception as e:
                    if logger:
                        logger.warning(
                            f"Failed to initialize tools after connection: {e}"
                        )
                    self.enable_tools = False
                    self.enable_llm = False
            else:
                if logger:
                    logger.warning("Executor: Cannot connect to None system")

        except Exception as e:
            if logger:
                logger.warning(f"Executor connection error: {e}")
