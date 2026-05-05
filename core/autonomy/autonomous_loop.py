"""
🔥 EDIATH FINAL (Voice + RAgent Unified Brain)
✔ Single intelligence (RAgent)
✔ No duplicate LLM calls
✔ No loops / no overflow
✔ Stable + production ready
"""

import asyncio
import time
from datetime import datetime

from ..utils.logger import logger
from ..memory.memory_manager import MemoryManager


class AutonomousLoop:
    def __init__(self, system):
        self.system = system  # 🔥 CONNECT SYSTEM

        self.memory = MemoryManager()

        self.is_running = False

        # 🔥 INPUT CONTROL
        self.user_input = None
        self.is_speaking = False
        self.processing = False

        self._last_input_time = 0
        self.input_cooldown = 1.5

        self._last_response = ""

        self.ignore_words = ["you", "uh", "hmm", "noise", ""]

    # ------------------------
    # 🎤 INPUT
    # ------------------------
    def set_input(self, text: str) -> None:
        """
        Safely set user input (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not text or not isinstance(text, str):
                return

            text = text.strip().lower()

            if not text or len(text) < 3:
                return

            # ------------------------
            # 🔁 IGNORE FILTER
            # ------------------------
            ignore = set(getattr(self, "ignore_words", []))
            if text in ignore:
                return

            # ------------------------
            # ⏱️ COOLDOWN (CRITICAL FIX)
            # ------------------------
            now = time.time()
            last_time = getattr(self, "_last_input_time", 0)

            if now - last_time < getattr(self, "input_cooldown", 1.5):
                return

            # ------------------------
            # 🔁 DUPLICATE PREVENTION (NEW FIX)
            # ------------------------
            if text == getattr(self, "user_input", None):
                return

            # ------------------------
            # 🚫 BLOCK WHEN BUSY
            # ------------------------
            if getattr(self, "is_speaking", False) or getattr(
                self, "processing", False
            ):
                return

            # ------------------------
            # 🔥 SET INPUT
            # ------------------------
            self._last_input_time = now
            self.user_input = text

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.inputs_received = getattr(self, "inputs_received", 0) + 1
            except Exception:
                pass

            # ------------------------
            # 📢 LOG
            # ------------------------
            try:
                logger.debug(f"🎤 Input accepted: {text}")
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Input handling failed: {e}")
            except Exception:
                pass

    # ------------------------
    # 🔊 SPEAK
    # ------------------------
    async def speak(self, text: str) -> None:
        """
        Speak safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not text or not isinstance(text, str):
                return

            text = text.strip()
            if not text:
                return

            # ------------------------
            # 🔁 PREVENT OVERLAP (CRITICAL FIX)
            # ------------------------
            if getattr(self, "is_speaking", False):
                return

            # ------------------------
            # 🔥 SET STATE
            # ------------------------
            self.is_speaking = True

            # ------------------------
            # 📢 OUTPUT
            # ------------------------
            try:
                print(f"🔊 {text}")
            except Exception:
                pass

            # ------------------------
            # ⏱️ NON-BLOCKING DELAY (SMART)
            # ------------------------
            duration = min(2.0, max(0.4, len(text) * 0.02))
            await asyncio.sleep(duration)

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.messages_spoken = getattr(self, "messages_spoken", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Speak failed: {e}")
            except Exception:
                pass

        finally:
            # ------------------------
            # 🔒 ALWAYS RESET STATE
            # ------------------------
            self.is_speaking = False

    # ------------------------
    # 🚀 MAIN LOOP (FIXED)
    # ------------------------
    async def run(self, duration: float = None) -> None:
        """
        Main autonomous loop (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔁 PREVENT DOUBLE START
            # ------------------------
            if getattr(self, "is_running", False):
                return

            self.is_running = True

            try:
                logger.info("🎤 EDIATH Started (Unified Brain Mode)")
            except Exception:
                pass

            start_time = time.time()

            # ------------------------
            # 🔄 MAIN LOOP
            # ------------------------
            while getattr(self, "is_running", False):

                try:
                    now = time.time()

                    # ------------------------
                    # ⏱️ DURATION LIMIT
                    # ------------------------
                    if duration and (now - start_time) > duration:
                        break

                    # ------------------------
                    # 💤 NO INPUT
                    # ------------------------
                    if not getattr(self, "user_input", None):
                        await asyncio.sleep(0.1)
                        continue

                    # ------------------------
                    # 🚫 SYSTEM BUSY CHECK
                    # ------------------------
                    if getattr(self.system, "thinking", False):
                        await asyncio.sleep(0.2)
                        continue

                    # ------------------------
                    # 🔁 LOCK PROCESSING
                    # ------------------------
                    if getattr(self, "processing", False):
                        await asyncio.sleep(0.1)
                        continue

                    self.processing = True

                    user_text = self.user_input
                    self.user_input = None

                    try:
                        logger.info(f"🎤 User: {user_text}")
                    except Exception:
                        pass

                    # ------------------------
                    # 🧠 CENTRAL BRAIN (WITH TIMEOUT)
                    # ------------------------
                    try:
                        result = await asyncio.wait_for(
                            self.system.brain_process("task", {"task": user_text}),
                            timeout=15,
                        )
                    except asyncio.TimeoutError:
                        result = {"result": "Sorry, I took too long to respond."}
                    except Exception as e:
                        result = {"result": str(e)}

                    # ------------------------
                    # 🔥 EXTRACT RESPONSE
                    # ------------------------
                    response = ""

                    if isinstance(result, dict):
                        response = str(result.get("result", "")).strip()
                    elif isinstance(result, str):
                        response = result.strip()

                    if not response:
                        self.processing = False
                        continue

                    # ------------------------
                    # 🔁 DUPLICATE PREVENTION
                    # ------------------------
                    if response == getattr(self, "_last_response", ""):
                        self.processing = False
                        continue

                    self._last_response = response

                    try:
                        logger.info(f"🤖 AI: {response}")
                    except Exception:
                        pass

                    # ------------------------
                    # 🔊 SPEAK (SAFE)
                    # ------------------------
                    try:
                        await self.speak(response)
                    except Exception:
                        pass

                    # ------------------------
                    # 🧠 MEMORY STORE (NON-BLOCKING)
                    # ------------------------
                    try:
                        asyncio.create_task(
                            self.memory.store(
                                {
                                    "input": user_text,
                                    "response": response[:200],
                                    "time": datetime.now().isoformat(),
                                }
                            )
                        )
                    except Exception:
                        pass

                    # ------------------------
                    # 📊 METRICS
                    # ------------------------
                    try:
                        self.processed_inputs = getattr(self, "processed_inputs", 0) + 1
                    except Exception:
                        pass

                    self.processing = False

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    try:
                        logger.error(f"🔥 Loop crash recovered: {e}")
                    except Exception:
                        pass

                    self.processing = False
                    await asyncio.sleep(0.2)

        finally:
            # ------------------------
            # 🛑 CLEAN SHUTDOWN
            # ------------------------
            self.is_running = False

            try:
                logger.info("🛑 Assistant loop stopped")
            except Exception:
                pass

    # ------------------------
    def stop(self) -> bool:
        """
        Stop autonomous loop safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔁 ALREADY STOPPED CHECK
            # ------------------------
            if not getattr(self, "is_running", False):
                try:
                    logger.warning("⚠ Assistant already stopped")
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 UPDATE STATE
            # ------------------------
            self.is_running = False

            # ------------------------
            # 🔒 RESET FLAGS
            # ------------------------
            try:
                self.processing = False
                self.is_speaking = False
            except Exception:
                pass

            # ------------------------
            # 🧵 CANCEL RUN TASK (IF TRACKED)
            # ------------------------
            try:
                task = getattr(self, "_task", None)
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.stopped_at = time.time()
            except Exception:
                pass

            # ------------------------
            # 📢 LOG
            # ------------------------
            try:
                logger.info("🛑 Assistant stopped")
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                logger.error(f"❌ Stop failed: {e}")
            except Exception:
                pass

            return False
