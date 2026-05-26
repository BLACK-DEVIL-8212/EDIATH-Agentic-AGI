"""
Production Tools for EDIATH R_AGENT
✔ Safe execution
✔ Timeout protection
✔ Output limiting
✔ Error handling
✔ Logging support
"""

import asyncio

# core.system imported via package resolution
pass

try:
    from ..utils.logger import logger
except:
    logger = None


class Tools:
    def __init__(self, system):
        try:
            # 🔥 core system reference
            self.system = system

            # 🔥 validate system (important safety)
            if not self.system:
                raise ValueError("Tools initialized without system reference")

            # 🔥 timeouts
            self.timeout = 10
            self.max_output_length = 300

            # 🔥 tool availability flags
            self.enable_vision = hasattr(system, "vision_engine_instance")
            self.enable_memory = hasattr(system, "memory")
            self.enable_code = True

            # 🔥 retry config (future safe)
            self.max_retries = 1

            # 🔥 debug flag
            self.debug = False

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Tools Init Error] {e}")

            # 🔥 fallback safe defaults
            self.system = system
            self.timeout = 10
            self.max_output_length = 300
            self.enable_vision = False
            self.enable_memory = False
            self.enable_code = False
            self.max_retries = 0
            self.debug = False

    # ------------------------
    # VISION TOOL
    # ------------------------
    async def vision(self) -> str:
        try:
            # 🔥 check feature enabled
            if not getattr(self, "enable_vision", False):
                return "Vision is disabled."

            ve = getattr(self.system, "vision_engine_instance", None)

            if not ve:
                return "Vision not available"

            # 🔥 retry mechanism (safe)
            for attempt in range(getattr(self, "max_retries", 1) + 1):
                try:
                    result = await asyncio.wait_for(
                        self._safe_vision(ve), timeout=self.timeout
                    )

                    if result:
                        return str(result)[: self.max_output_length]

                except asyncio.TimeoutError:
                    if logger:
                        logger.warning(f"[Vision Timeout] Attempt {attempt+1}")
                    continue

                except Exception as e:
                    if logger:
                        logger.warning(f"[Vision Attempt Error] {e}")
                    continue

            # 🔥 all retries failed
            return self._fallback("Vision failed after retries")

        except Exception as e:
            if logger:
                logger.warning(f"[Vision Fatal Error] {e}")
            return self._fallback("Vision fatal error")

    async def _safe_vision(self, ve):
        try:
            # 🔥 validate engine
            if not ve or not hasattr(ve, "webcam"):
                return "Vision engine unavailable"

            # ------------------------
            # 🔥 CAPTURE FRAME
            # ------------------------
            frame = await ve.webcam.capture_frame()

            if frame is None:
                return "No visual data"

            # 🔥 basic frame validation
            if hasattr(frame, "size") and frame.size == 0:
                return "Invalid frame data"

            # ------------------------
            # 🔥 PROCESS FRAME
            # ------------------------
            data = await ve.process_single_frame(frame)

            if not data:
                return "No objects detected"

            # ------------------------
            # 🔥 CLEAN + LIMIT OUTPUT
            # ------------------------
            result = str(data)

            # remove excessive whitespace
            import re

            result = re.sub(r"\s+", " ", result)

            # limit output size
            result = result[: getattr(self, "max_output_length", 300)]

            return result.strip()

        except Exception as e:
            if logger:
                logger.warning(f"[Safe Vision Error] {e}")

            return "Vision processing failed"

    # ------------------------
    # MEMORY TOOL
    # ------------------------
    async def memory(self) -> str:
        try:
            # 🔥 feature check
            if not getattr(self, "enable_memory", False):
                return "Memory is disabled."

            # 🔥 retry mechanism
            for attempt in range(getattr(self, "max_retries", 1) + 1):
                try:
                    result = await asyncio.wait_for(
                        self._safe_memory(), timeout=self.timeout
                    )

                    if result:
                        return str(result)[: self.max_output_length]

                except asyncio.TimeoutError:
                    if logger:
                        logger.warning(f"[Memory Timeout] Attempt {attempt+1}")
                    continue

                except Exception as e:
                    if logger:
                        logger.warning(f"[Memory Attempt Error] {e}")
                    continue

            # 🔥 all retries failed
            return self._fallback("Memory failed after retries")

        except Exception as e:
            if logger:
                logger.warning(f"[Memory Fatal Error] {e}")
            return self._fallback("Memory fatal error")

    async def _safe_memory(self):
        try:
            # 🔥 validate memory system
            memory = getattr(self.system, "memory", None)

            if not memory:
                return "Memory not available"

            if not hasattr(memory, "get_recent"):
                return "Memory interface invalid"

            # ------------------------
            # 🔥 FETCH DATA
            # ------------------------
            data = await memory.get_recent(5)

            if not data:
                return "No memory data"

            # ------------------------
            # 🔥 CLEAN OUTPUT
            # ------------------------
            result = str(data)

            import re

            # remove excessive whitespace
            result = re.sub(r"\s+", " ", result)

            # remove overly long repeating patterns
            result = re.sub(r"(.)\1{5,}", r"\1\1", result)

            # limit size safely
            result = result[: getattr(self, "max_output_length", 300)]

            return result.strip()

        except Exception as e:
            if logger:
                logger.warning(f"[Safe Memory Error] {e}")

            return "Memory processing failed"

    # ------------------------
    # CODE TOOL
    # ------------------------
    async def code(self, context: str) -> str:
        try:
            # 🔥 feature check
            if not getattr(self, "enable_code", True):
                return "Code execution is disabled."

            # 🔥 validate + normalize input
            if not context:
                return "No code context provided."

            if not isinstance(context, str):
                context = str(context)

            context = context.strip()[:300]

            if not context:
                return "Invalid code input."

            lower = context.lower()

            # ------------------------
            # 🔥 RESTRICTED CODE FILTER (CRITICAL)
            # ------------------------
            restricted_keywords = [
                "ddos",
                "attack",
                "hack",
                "exploit",
                "malware",
                "ransomware",
                "virus",
            ]

            for keyword in restricted_keywords:
                if keyword in lower:
                    if logger:
                        logger.warning(
                            f"[Code Tool] Blocked restricted keyword: {keyword}"
                        )
                    return "I can't assist with that request."

            # ------------------------
            # 🔥 SAFE EXECUTION (SIMULATION MODE)
            # ------------------------
            result = f"Code task processed safely: {context}"

            # 🔥 limit output
            return result[: getattr(self, "max_output_length", 300)]

        except Exception as e:
            if logger:
                logger.warning(f"[Code Tool Error] {e}")
            return self._fallback("Code execution failed")

    # ------------------------
    # FALLBACK
    # ------------------------
    def _fallback(self, reason: str) -> str:
        try:
            if logger:
                logger.warning(f"[Tools Fallback] {reason}")

            reason = str(reason).lower()

            # ------------------------
            # 🔥 CONTEXT-AWARE RESPONSES
            # ------------------------
            if "vision" in reason:
                return "I couldn't process visual data right now."

            if "memory" in reason:
                return "I couldn't retrieve memory data."

            if "code" in reason:
                return "I couldn't safely process that code request."

            if "timeout" in reason:
                return "The request took too long to process."

            if "failed" in reason:
                return "The operation couldn't be completed."

            if "error" in reason:
                return "Something went wrong while executing the tool."

            # ------------------------
            # 🔥 DEFAULT FALLBACK
            # ------------------------
            return "Tool execution failed."

        except Exception as e:
            if logger:
                logger.warning(f"[Tools Fallback Error] {e}")

            return "Tool execution failed."
