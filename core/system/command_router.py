"""
Advanced Command Router - Autonomous Execution Engine
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio
import heapq
import time
import uuid

from core.utils.task_registry import task_registry

from ..utils.logger import logger


class CommandType(Enum):
    EXECUTE = "execute"
    QUERY = "query"
    CONFIG = "config"
    WORKFLOW = "workflow"
    SYSTEM = "system"

    # ------------------------
    # 🔥 SAFE PARSE FROM STRING
    # ------------------------
    @classmethod
    def from_string(cls, value: Optional[str]) -> "CommandType":
        """
        Safely parse string into CommandType (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 EMPTY / NONE HANDLING
            # ------------------------
            if not value:
                return cls.QUERY

            # ------------------------
            # 🔥 NORMALIZE INPUT (CRITICAL FIX)
            # ------------------------
            value = str(value).strip().lower()

            # ------------------------
            # 🔥 DIRECT MATCH (FAST PATH)
            # ------------------------
            try:
                return cls(value)
            except Exception:
                pass

            # ------------------------
            # 🔥 ALIAS MAP (OPTIMIZED)
            # ------------------------
            alias_map = {
                # execute
                "run": cls.EXECUTE,
                "exec": cls.EXECUTE,
                "start": cls.EXECUTE,
                # query
                "ask": cls.QUERY,
                "question": cls.QUERY,
                "search": cls.QUERY,
                "find": cls.QUERY,
                "what": cls.QUERY,
                "how": cls.QUERY,
                # config
                "config": cls.CONFIG,
                "settings": cls.CONFIG,
                "setup": cls.CONFIG,
                # workflow
                "flow": cls.WORKFLOW,
                "pipeline": cls.WORKFLOW,
                # system
                "system": cls.SYSTEM,
                "admin": cls.SYSTEM,
                "core": cls.SYSTEM,
            }

            if value in alias_map:
                return alias_map[value]

            # ------------------------
            # 🔥 PARTIAL MATCH (CONTROLLED)
            # ------------------------
            for item in cls:
                iv = item.value
                if value == iv or value.startswith(iv) or iv in value:
                    return item

            # ------------------------
            # 🔥 HEURISTIC MATCH (ORDERED PRIORITY)
            # ------------------------
            heuristic_map = [
                (cls.EXECUTE, ["run", "exec", "start"]),
                (cls.QUERY, ["ask", "find", "search", "what", "how"]),
                (cls.CONFIG, ["config", "setting", "setup"]),
                (cls.WORKFLOW, ["flow", "process", "pipeline"]),
                (cls.SYSTEM, ["system", "admin", "core"]),
            ]

            for cmd_type, keywords in heuristic_map:
                if any(k in value for k in keywords):
                    return cmd_type

            # ------------------------
            # 🔥 DEFAULT FALLBACK
            # ------------------------
            return cls.QUERY

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE (NEVER BREAK SYSTEM)
            # ------------------------
            return cls.QUERY

    # ------------------------
    # 🔥 VALIDATE INPUT
    # ------------------------
    @classmethod
    def is_valid(cls, value: Optional[str]) -> bool:
        """
        Validate if input maps to a valid CommandType (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 NONE / EMPTY CHECK (CRITICAL FIX)
            # ------------------------
            if not value:
                return False

            value = str(value).strip().lower()

            # ------------------------
            # 🔥 DIRECT MATCH (FAST PATH)
            # ------------------------
            if any(value == item.value for item in cls):
                return True

            # ------------------------
            # 🔥 ALIAS SET (OPTIMIZED)
            # ------------------------
            alias_set = {
                "run",
                "exec",
                "start",
                "ask",
                "query",
                "search",
                "find",
                "what",
                "how",
                "config",
                "settings",
                "setup",
                "flow",
                "workflow",
                "pipeline",
                "process",
                "system",
                "admin",
                "core",
            }

            if value in alias_set:
                return True

            # ------------------------
            # 🔥 PARTIAL MATCH (CONTROLLED)
            # ------------------------
            for item in cls:
                iv = item.value
                if value == iv or value.startswith(iv) or iv in value:
                    return True

            # ------------------------
            # 🔥 HEURISTIC MATCH (PRIORITIZED)
            # ------------------------
            heuristic_groups = [
                ["run", "exec", "start"],
                ["ask", "find", "search", "what", "how"],
                ["config", "setting", "setup"],
                ["flow", "process", "pipeline"],
                ["system", "admin", "core"],
            ]

            for group in heuristic_groups:
                if any(k in value for k in group):
                    return True

            # ------------------------
            # 🔥 FINAL FAIL
            # ------------------------
            return False

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return False

    # ------------------------
    # 🔥 LIST ALL VALUES
    # ------------------------
    @classmethod
    def list(
        cls, include_names: bool = False, include_metadata: bool = False
    ) -> List[Any]:
        """
        Return available command types (optimized + production-grade)
        """

        try:
            result = []

            for item in cls:
                # ------------------------
                # 🔥 BASE STRUCTURE
                # ------------------------
                if not include_names and not include_metadata:
                    result.append(item.value)
                    continue

                entry = {}

                # ------------------------
                # 🔥 NAME + VALUE
                # ------------------------
                if include_names or include_metadata:
                    entry["name"] = item.name
                    entry["value"] = item.value

                # ------------------------
                # 🔥 METADATA (SAFE)
                # ------------------------
                if include_metadata:
                    try:
                        entry["is_critical"] = item.is_critical()
                    except Exception:
                        entry["is_critical"] = False

                    try:
                        entry["priority"] = item.priority()
                    except Exception:
                        entry["priority"] = 0

                result.append(entry)

            return result

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return []

    # ------------------------
    # 🔥 CHECK TYPE HELPERS
    # ------------------------
    def is_execution(self) -> bool:
        """
        Check if command is execution type (production-grade)
        """

        try:
            # ------------------------
            # 🔥 DIRECT CHECK (FAST PATH)
            # ------------------------
            if self is CommandType.EXECUTE:
                return True

            # ------------------------
            # 🔥 FALLBACK SAFETY
            # ------------------------
            if getattr(self, "value", None) == CommandType.EXECUTE.value:
                return True

            # ------------------------
            # 🔥 FAIL
            # ------------------------
            return False

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return False

    def is_query(self) -> bool:
        """
        Check if command is query type (production-grade)
        """
        try:
            if self is CommandType.QUERY:
                return True

            if getattr(self, "value", None) == CommandType.QUERY.value:
                return True

            return False

        except Exception:
            return False

    def is_config(self) -> bool:
        """
        Check if command is config type (production-grade)
        """
        try:
            if self is CommandType.CONFIG:
                return True

            if getattr(self, "value", None) == CommandType.CONFIG.value:
                return True

            return False

        except Exception:
            return False

    def is_workflow(self) -> bool:
        """
        Check if command is workflow type (production-grade)
        """
        try:
            if self is CommandType.WORKFLOW:
                return True

            if getattr(self, "value", None) == CommandType.WORKFLOW.value:
                return True

            return False

        except Exception:
            return False

    def is_system(self) -> bool:
        """
        Check if command is system type (production-grade)
        """
        try:
            if self is CommandType.SYSTEM:
                return True

            if getattr(self, "value", None) == CommandType.SYSTEM.value:
                return True

            return False

        except Exception:
            return False

    # ------------------------
    # 🔥 PRIORITY LEVEL (FOR SCHEDULING)
    # ------------------------
    def priority(self) -> int:
        """
        Get priority level of command (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 STATIC PRIORITY MAP (CRITICAL FIX)
            # ------------------------
            priority_map = {
                CommandType.SYSTEM: 5,
                CommandType.EXECUTE: 4,
                CommandType.WORKFLOW: 3,
                CommandType.CONFIG: 2,
                CommandType.QUERY: 1,
            }

            # ------------------------
            # 🔥 DIRECT LOOKUP (FAST PATH)
            # ------------------------
            priority = priority_map.get(self)
            if priority is not None:
                return priority

            # ------------------------
            # 🔥 SAFE VALUE FALLBACK (SIMPLIFIED)
            # ------------------------
            value = getattr(self, "value", None)
            if value:
                for key, val in priority_map.items():
                    if key.value == value:
                        return val

            # ------------------------
            # 🔥 DEFAULT PRIORITY
            # ------------------------
            return 0

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return 0

    # ------------------------
    # 🔥 IS CRITICAL COMMAND
    # ------------------------
    def is_critical(self) -> bool:
        """
        Check if command is critical (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 STATIC CRITICAL SET (CRITICAL FIX)
            # ------------------------
            critical_set = {CommandType.SYSTEM, CommandType.EXECUTE}

            # ------------------------
            # 🔥 DIRECT CHECK (FAST PATH)
            # ------------------------
            if self in critical_set:
                return True

            # ------------------------
            # 🔥 SAFE VALUE FALLBACK (SIMPLIFIED)
            # ------------------------
            value = getattr(self, "value", None)
            if value:
                return any(item.value == value for item in critical_set)

            # ------------------------
            # 🔥 DEFAULT
            # ------------------------
            return False

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return False

    # ------------------------
    # 🔥 STRING REPRESENTATION
    # ------------------------
    def __str__(self) -> str:
        """
        Human-readable string representation (production-grade)
        """

        try:
            value = getattr(self, "value", None)

            if value is None:
                return "unknown"

            return str(value)

        except Exception:
            return "unknown"

    def __repr__(self) -> str:
        """
        Developer/debug representation (production-grade)
        """

        try:
            name = getattr(self, "name", "UNKNOWN")
            value = getattr(self, "value", "unknown")

            return f"<CommandType.{name}: '{value}'>"

        except Exception:
            return "<CommandType.UNKNOWN>"


class Command:
    def __init__(
        self,
        cmd_type: CommandType,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        async_execution: bool = False,
        retries: int = 0,
        timeout: Optional[int] = None,
    ):
        """
        Production-grade Command initialization
        """

        try:
            # ------------------------
            # 🔥 VALIDATE COMMAND TYPE
            # ------------------------
            if not isinstance(cmd_type, CommandType):
                cmd_type = CommandType.from_string(cmd_type)

            self.cmd_type = cmd_type

            # ------------------------
            # 🔥 VALIDATE ACTION
            # ------------------------
            if not action or not isinstance(action, str):
                raise ValueError("Action must be a non-empty string")

            self.action = action.strip()

            # ------------------------
            # 🔥 PAYLOAD (SAFE)
            # ------------------------
            if payload is None:
                payload = {}
            elif not isinstance(payload, dict):
                raise ValueError("Payload must be a dictionary")

            self.payload = payload

            # ------------------------
            # 🔥 PRIORITY (AUTO FIX)
            # ------------------------
            try:
                priority = int(priority)
            except Exception:
                priority = 0

            if priority <= 0:
                priority = self.cmd_type.priority()

            self.priority = priority

            # ------------------------
            # 🔥 EXECUTION FLAGS
            # ------------------------
            self.async_execution = bool(async_execution)

            try:
                self.retries = max(0, int(retries))
            except Exception:
                self.retries = 0

            try:
                self.timeout = int(timeout) if timeout else None
            except Exception:
                self.timeout = None

            # ------------------------
            # 🔥 METADATA
            # ------------------------
            self.created_at = time.time()
            self.command_id = getattr(self, "command_id", None) or str(uuid.uuid4())

            # ------------------------
            # 🔥 STATE TRACKING
            # ------------------------
            self.status = "pending"
            self.attempts = 0
            self.last_error = None
            self.result = None

            # ------------------------
            # 🔥 TAGS / EXTENSIBILITY
            # ------------------------
            self.tags = []
            self.metadata = {}

        except Exception as e:
            raise RuntimeError(f"Command initialization failed: {e}")

    def __lt__(self, other) -> bool:
        """
        Priority queue sorting (higher priority first, production-grade)
        """

        try:
            # ------------------------
            # 🔥 TYPE CHECK (CRITICAL FIX)
            # ------------------------
            if not isinstance(other, self.__class__):
                return NotImplemented

            # ------------------------
            # 🔥 SAFE PRIORITY FETCH (FIXED)
            # ------------------------
            try:
                self_priority = (
                    self.priority()
                    if callable(getattr(self, "priority", None))
                    else getattr(self, "priority", 0)
                )
            except Exception:
                self_priority = 0

            try:
                other_priority = (
                    other.priority()
                    if callable(getattr(other, "priority", None))
                    else getattr(other, "priority", 0)
                )
            except Exception:
                other_priority = 0

            # ------------------------
            # 🔥 PRIMARY SORT (HIGHER FIRST)
            # ------------------------
            if self_priority != other_priority:
                return self_priority > other_priority

            # ------------------------
            # 🔥 SECONDARY SORT (OLDER FIRST)
            # ------------------------
            self_time = getattr(self, "created_at", 0)
            other_time = getattr(other, "created_at", 0)

            try:
                return self_time < other_time
            except Exception:
                return False

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return False


class CommandHandler(ABC):
    """
    Advanced Command Handler
    Supports:
    - Async + Sync execution
    - Validation
    - Retry
    - Timeout
    - Pre/Post hooks
    """

    def __init__(self):
        """
        Production-grade initialization
        """

        try:
            # ------------------------
            # 🔥 TIMESTAMP (SAFE + CONSISTENT)
            # ------------------------
            try:
                self.created_at = datetime.utcnow()
            except Exception:
                self.created_at = None

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            self.updated_at = self.created_at
            self.access_count = 0
            self.error_count = 0

            # ------------------------
            # 🔥 STATE FLAGS
            # ------------------------
            self.is_active = True
            self.is_initialized = True

            # ------------------------
            # 🔥 IDENTIFIER (OPTIONAL)
            # ------------------------
            try:
                self.instance_id = str(uuid.uuid4())
            except Exception:
                self.instance_id = None

            # ------------------------
            # 🔥 METADATA
            # ------------------------
            self.metadata = {}

            # ------------------------
            # 🔥 LOGGING
            # ------------------------
            try:
                logger.debug(f"Instance initialized at {self.created_at}")
            except:
                pass

        except Exception as e:
            try:
                logger.error(f"Initialization failed: {e}")
            except:
                pass

            raise RuntimeError(f"Initialization failed: {e}")

    # ------------------------
    # MAIN ENTRY 🔥
    # ------------------------
    async def __call__(self, command: "Command") -> Any:
        """
        Execute command with full lifecycle (optimized + production-grade)
        """

        import asyncio
        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise ValueError("Command cannot be None")

            # ------------------------
            # 🔥 SAFE DEFAULTS
            # ------------------------
            retries = int(getattr(command, "retries", 0) or 0)
            timeout = getattr(command, "timeout", None)

            # ------------------------
            # 🔥 INITIAL STATE
            # ------------------------
            command.status = "running"
            command.attempts = int(getattr(command, "attempts", 0)) + 1

            # ------------------------
            # 🔥 PRE HANDLE (SAFE)
            # ------------------------
            try:
                if hasattr(self, "pre_handle"):
                    await self.pre_handle(command)
            except Exception:
                pass

            last_error = None

            # ------------------------
            # 🔥 EXECUTION LOOP (FIXED)
            # ------------------------
            for attempt in range(retries + 1):
                try:

                    async def _run():
                        return await self._execute(command)

                    if timeout and timeout > 0:
                        result = await asyncio.wait_for(_run(), timeout=timeout)
                    else:
                        result = await _run()

                    # ------------------------
                    # 🔥 SUCCESS
                    # ------------------------
                    command.status = "completed"
                    command.result = result

                    # ------------------------
                    # 🔥 POST HANDLE (SAFE)
                    # ------------------------
                    try:
                        if hasattr(self, "post_handle"):
                            await self.post_handle(command, result)
                    except Exception:
                        pass

                    # ------------------------
                    # 🔥 METRICS (SAFE)
                    # ------------------------
                    try:
                        self.execution_count = (
                            int(getattr(self, "execution_count", 0)) + 1
                        )
                    except Exception:
                        pass

                    return result

                except asyncio.TimeoutError:
                    last_error = "timeout"

                except Exception as e:
                    last_error = str(e)

                # ------------------------
                # 🔥 RETRY BACKOFF (IMPROVED)
                # ------------------------
                if attempt < retries:
                    await asyncio.sleep(min(1.0, 0.2 + attempt * 0.3))

            # ------------------------
            # 🔥 FAILURE AFTER RETRIES
            # ------------------------
            command.status = "failed"
            command.last_error = last_error or "unknown error"

            raise RuntimeError(command.last_error)

        except asyncio.CancelledError:
            # ------------------------
            # 🔥 CANCEL HANDLING (CRITICAL FIX)
            # ------------------------
            command.status = "cancelled"
            command.last_error = "cancelled"
            raise

        except Exception as e:
            # ------------------------
            # 🔥 ERROR HANDLING (FIXED)
            # ------------------------
            command.status = "error"
            command.last_error = str(e)

            try:
                self.failed_count = int(getattr(self, "failed_count", 0)) + 1
            except Exception:
                pass

            try:
                if hasattr(self, "on_error"):
                    return await self.on_error(command, e)
            except Exception:
                pass

            return {
                "success": False,
                "error": str(e),
                "execution_time": round(time.time() - start, 4),
            }

    # ------------------------
    # EXECUTION WRAPPER
    # ------------------------
    async def _execute(self, command: "Command") -> Any:
        """
        Internal execution handler (optimized + production-grade)
        Supports async + sync handlers safely
        """

        import asyncio
        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise ValueError("Command cannot be None")

            handler = getattr(self, "handle", None)

            if not callable(handler):
                raise RuntimeError("Handler not defined or not callable")

            # ------------------------
            # 🔥 EXECUTION PATH (FIXED)
            # ------------------------
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(command)
                else:
                    result = await asyncio.to_thread(handler, command)
            except Exception as e:
                raise RuntimeError(f"Handler execution failed: {e}")

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.execution_count = int(getattr(self, "execution_count", 0)) + 1
                self.last_execution_time = round(time.time() - start, 4)
            except Exception:
                pass

            return result

        except asyncio.CancelledError:
            # ------------------------
            # 🔥 CANCEL HANDLING (CRITICAL FIX)
            # ------------------------
            try:
                command.status = "cancelled"
                command.last_error = "cancelled"
            except Exception:
                pass
            raise

        except Exception as e:
            # ------------------------
            # 🔥 ERROR TRACKING (IMPROVED)
            # ------------------------
            try:
                command.last_error = str(e)
            except Exception:
                pass

            try:
                self.error_count = int(getattr(self, "error_count", 0)) + 1
            except Exception:
                pass

            raise

    # ------------------------
    # CORE METHOD (YOU IMPLEMENT THIS)
    # ------------------------
    @abstractmethod
    async def handle(self, command: "Command") -> Any:
        """
        Abstract handler for command execution (production-grade)

        Implement this in subclasses to define actual execution logic.
        Must return a result or raise an exception on failure.
        """

        # ------------------------
        # 🔥 BASIC VALIDATION (OPTIONAL ENFORCEMENT)
        # ------------------------
        if command is None:
            raise ValueError("Command cannot be None")

        # ------------------------
        # 🔥 CONTRACT NOTES
        # ------------------------
        # Implementations should:
        # - Respect command.timeout (if applicable)
        # - Use command.payload safely
        # - Return structured result (dict / object)
        # - Raise exceptions for failure (handled upstream)
        # - Avoid blocking (use async or thread offloading)

        raise NotImplementedError("Handle method must be implemented by subclass")

    # ------------------------
    # HOOKS 🔥
    # ------------------------
    async def pre_handle(self, command: "Command") -> None:
        """
        Pre-execution hook (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise ValueError("Command cannot be None")

            # ------------------------
            # 🔥 STATE UPDATE (SAFE)
            # ------------------------
            try:
                command.status = "preparing"
                command.started_at = time.time()
            except Exception:
                pass

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.pre_execution_count = (
                    int(getattr(self, "pre_execution_count", 0)) + 1
                )
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE ATTRIBUTE EXTRACTION
            # ------------------------
            action = getattr(command, "action", "unknown")
            cmd_type = getattr(getattr(command, "cmd_type", None), "value", "unknown")

            try:
                priority = (
                    command.priority()
                    if callable(getattr(command, "priority", None))
                    else getattr(command, "priority", 0)
                )
            except Exception:
                priority = 0

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"➡ Handling command: {action} "
                        f"(type={cmd_type}, priority={priority})"
                    )
            except Exception:
                pass

            # ------------------------
            # 🔥 EXTENSION POINT
            # ------------------------
            # subclasses can extend safely

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.warning(f"⚠ pre_handle error: {e}")
            except Exception:
                pass

    async def post_handle(self, command: "Command", result: Any) -> None:
        """
        Post-execution hook (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise ValueError("Command cannot be None")

            # ------------------------
            # 🔥 STATE UPDATE (SAFE)
            # ------------------------
            try:
                command.status = "completed"
                command.result = result
                command.completed_at = time.time()
            except Exception:
                pass

            # ------------------------
            # 🔥 EXECUTION TIME (SAFE)
            # ------------------------
            try:
                start_time = getattr(command, "started_at", None)
                if isinstance(start_time, (int, float)):
                    command.execution_time = round(command.completed_at - start_time, 4)
                else:
                    command.execution_time = None
            except Exception:
                command.execution_time = None

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.post_execution_count = (
                    int(getattr(self, "post_execution_count", 0)) + 1
                )
                self.last_execution_time = getattr(command, "execution_time", None)
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE ATTRIBUTE EXTRACTION
            # ------------------------
            action = getattr(command, "action", "unknown")
            status = getattr(command, "status", "unknown")
            exec_time = getattr(command, "execution_time", "n/a")

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"✅ Completed: {action} "
                        f"(status={status}, time={exec_time}s)"
                    )
            except Exception:
                pass

            # ------------------------
            # 🔥 EXTENSION POINT
            # ------------------------
            # subclasses can extend safely

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.warning(f"⚠ post_handle error: {e}")
            except Exception:
                pass

    async def on_error(self, command: "Command", error: Exception) -> Any:
        """
        Error handling + retry (optimized + production-grade)
        """

        import asyncio
        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise RuntimeError("Invalid command in error handler")

            error_str = str(error)

            # ------------------------
            # 🔥 STATE UPDATE (SAFE)
            # ------------------------
            command.status = "error"
            command.last_error = error_str

            try:
                command.error_count = int(getattr(command, "error_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.error_count = int(getattr(self, "error_count", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE ATTRIBUTE EXTRACTION
            # ------------------------
            action = getattr(command, "action", "unknown")
            attempts = int(getattr(command, "attempts", 1))

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Error in {action}: {error_str}")
            except Exception:
                pass

            # ------------------------
            # 🔥 RETRY LOGIC (FIXED + SAFE)
            # ------------------------
            retries_left = int(getattr(command, "retries", 0))

            if retries_left > 0:
                command.retries = retries_left - 1
                command.status = "retrying"

                # capped exponential backoff
                delay = min(2**attempts, 5)

                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"🔁 Retrying ({command.retries} left): {action} after {delay}s"
                        )
                except Exception:
                    pass

                await asyncio.sleep(delay)

                # 🔥 IMPORTANT FIX: avoid infinite recursion explosion
                return await self.__call__(command)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            command.status = "failed"

            try:
                command.completed_at = time.time()
            except Exception:
                pass

            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Final failure: {action}")
            except Exception:
                pass

            return {"success": False, "error": error_str, "action": action}

        except asyncio.CancelledError:
            raise

        except Exception as e:
            # ------------------------
            # 🔥 HARD FAILSAFE
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"Critical error handler failure: {e}")
            except Exception:
                pass

            return {"success": False, "error": "Unhandled system error"}


class CommandRouter:
    def __init__(self):
        """
        Production-grade Command Router initialization
        """
        try:
            # ------------------------
            # 🔥 HANDLER REGISTRY
            # ------------------------
            self.handlers: Dict[CommandType, Dict[str, CommandHandler]] = {
                cmd_type: {} for cmd_type in CommandType
            }

            # ------------------------
            # 🔥 QUEUE SYSTEM
            # ------------------------
            self.queue: List[Command] = []
            self.max_queue_size = 1000

            # ------------------------
            # 🔥 EXECUTION HISTORY
            # ------------------------
            self.execution_history: List[Dict[str, Any]] = []
            self.max_history_size = 500

            # ------------------------
            # 🔥 STATE FLAGS
            # ------------------------
            self._running = False
            self._paused = False
            self._shutdown = False

            # ------------------------
            # 🔥 CONCURRENCY CONTROL
            # ------------------------
            self._lock = asyncio.Lock()
            self._worker_task = None

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            self.total_commands = 0
            self.completed_commands = 0
            self.failed_commands = 0
            self.active_commands = 0

            self.created_at = time.time()
            self.last_execution_time = None

            # ------------------------
            # 🔥 CONFIG
            # ------------------------
            self.max_workers = 4
            self.retry_delay = 0.5

            # ------------------------
            # 🔥 LOGGING
            # ------------------------
            try:
                logger.info("🧠 Command Router initialized")
            except:
                pass

        except Exception as e:
            try:
                logger.error(f"CommandRouter init failed: {e}")
            except:
                pass

            raise RuntimeError(f"CommandRouter initialization failed: {e}")

    # ------------------------
    # REGISTER
    # ------------------------
    def register_handler(
        self,
        cmd_type: CommandType,
        action: str,
        handler: CommandHandler,
        overwrite: bool = False,
    ) -> bool:
        """
        Register handler safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if not isinstance(cmd_type, CommandType):
                cmd_type = CommandType.from_string(cmd_type)

            if not cmd_type:
                raise ValueError("Invalid command type")

            if not action or not isinstance(action, str):
                raise ValueError("Invalid action")

            if not callable(handler):
                raise ValueError("Handler must be callable")

            action = action.strip().lower()

            # ------------------------
            # 🔥 SAFE HANDLER STORAGE
            # ------------------------
            if not hasattr(self, "handlers") or not isinstance(self.handlers, dict):
                self.handlers = {}

            if cmd_type not in self.handlers:
                self.handlers[cmd_type] = {}

            # ------------------------
            # 🔥 DUPLICATE CHECK (SAFE)
            # ------------------------
            if action in self.handlers[cmd_type] and not overwrite:
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"⚠ Handler already exists: {cmd_type.value}/{action}"
                        )
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 REGISTER HANDLER
            # ------------------------
            self.handlers[cmd_type][action] = handler

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.total_handlers = int(getattr(self, "total_handlers", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"✅ Handler registered: {cmd_type.value}/{action}"
                    )
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"❌ Handler registration failed: {str(action)} - {e}"
                    )
            except Exception:
                pass

            return False

    # ------------------------
    # DIRECT ROUTE
    # ------------------------
    async def route(self, command: Command) -> Any:
        """
        Route command to appropriate handler (optimized + production-grade)
        """

        import asyncio
        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise ValueError("Command cannot be None")

            cmd_type = getattr(command, "cmd_type", None)
            action = str(getattr(command, "action", "")).strip().lower()

            if not action:
                raise ValueError("Invalid command action")

            # normalize cmd_type
            if not isinstance(cmd_type, CommandType):
                cmd_type = CommandType.from_string(cmd_type)

            # ------------------------
            # 🔥 SAFE HANDLER FETCH
            # ------------------------
            handlers = getattr(self, "handlers", {}) or {}
            handler = handlers.get(cmd_type, {}).get(action)

            if not handler:
                raise ValueError(
                    f"No handler for {getattr(cmd_type, 'value', 'unknown')}/{action}"
                )

            # ------------------------
            # 🔥 STATE + METRICS
            # ------------------------
            command.status = "routing"
            command.attempts = int(getattr(command, "attempts", 0)) + 1

            try:
                self.total_commands = int(getattr(self, "total_commands", 0)) + 1
                self.active_commands = int(getattr(self, "active_commands", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(f"➡ Executing: {cmd_type.value}/{action}")
            except Exception:
                pass

            # ------------------------
            # 🔥 EXECUTION (SAFE CALL)
            # ------------------------
            try:

                async def _run():
                    if hasattr(handler, "handle") and callable(handler.handle):
                        return await handler.handle(command)
                    elif callable(handler):
                        return await handler(command)
                    else:
                        raise RuntimeError("Invalid handler")

                timeout = getattr(command, "timeout", None)

                if timeout and timeout > 0:
                    result = await asyncio.wait_for(_run(), timeout=timeout)
                else:
                    result = await _run()

                # ------------------------
                # 🔥 SUCCESS
                # ------------------------
                command.status = "completed"
                command.result = result
                command.completed_at = time.time()

                try:
                    self.completed_commands = (
                        int(getattr(self, "completed_commands", 0)) + 1
                    )
                except Exception:
                    pass

                try:
                    if hasattr(self, "_log_success"):
                        self._log_success(command, result)
                except Exception:
                    pass

                return result

            except asyncio.TimeoutError:
                error = RuntimeError("Command timeout")

            except Exception as e:
                error = e

            # ------------------------
            # 🔥 ERROR HANDLING
            # ------------------------
            command.last_error = str(error)
            command.status = "error"

            try:
                self.failed_commands = int(getattr(self, "failed_commands", 0)) + 1
            except Exception:
                pass

            try:
                if hasattr(self, "_log_error"):
                    self._log_error(command, error)
            except Exception:
                pass

            # ------------------------
            # 🔥 RETRY LOGIC (SAFE)
            # ------------------------
            retries_left = int(getattr(command, "retries", 0))

            if retries_left > 0:
                command.retries = retries_left - 1
                command.status = "retrying"

                delay = min(2**command.attempts, 5)

                try:
                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"🔁 Retrying ({command.retries} left): {action} after {delay}s"
                        )
                except Exception:
                    pass

                await asyncio.sleep(delay)

                # 🔥 IMPORTANT FIX: prevent recursion overflow
                return await self.route(command)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            raise error

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Routing failed: {e}")
            except Exception:
                pass

            raise

        finally:
            # ------------------------
            # 🔥 CLEANUP ACTIVE COUNT (SAFE)
            # ------------------------
            try:
                self.active_commands = max(
                    int(getattr(self, "active_commands", 1)) - 1, 0
                )
            except Exception:
                pass

    # ------------------------
    # QUEUE SYSTEM 🔥
    # ------------------------
    def enqueue(self, command: Command) -> bool:
        """
        Add command to priority queue safely (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                raise ValueError("Command cannot be None")

            if not hasattr(command, "__lt__"):
                raise ValueError("Command must support priority comparison")

            # ------------------------
            # 🔥 SAFE QUEUE INIT
            # ------------------------
            if not hasattr(self, "queue") or not isinstance(self.queue, list):
                self.queue = []

            # ------------------------
            # 🔥 QUEUE SIZE LIMIT
            # ------------------------
            max_size = int(getattr(self, "max_queue_size", 1000) or 1000)

            if len(self.queue) >= max_size:
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning("⚠ Queue full, rejecting command")
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 STATE UPDATE
            # ------------------------
            try:
                command.status = "queued"
                command.queued_at = time.time()
            except Exception:
                pass

            # ------------------------
            # 🔥 PRIORITY SAFE FETCH
            # ------------------------
            try:
                priority = (
                    command.priority()
                    if callable(getattr(command, "priority", None))
                    else getattr(command, "priority", 0)
                )
            except Exception:
                priority = 0

            # ------------------------
            # 🔥 PUSH TO PRIORITY QUEUE
            # ------------------------
            heapq.heappush(self.queue, command)

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.total_commands = int(getattr(self, "total_commands", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"📥 Command queued: {getattr(command, 'action', 'unknown')} "
                        f"(priority={priority})"
                    )
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Enqueue failed: {e}")
            except Exception:
                pass

            return False

    async def process_queue(self) -> None:
        """
        Process command queue with workers (optimized + production-grade)
        """

        import asyncio
        import time

        try:
            # ------------------------
            # 🔥 SAFE INIT (CRITICAL FIX)
            # ------------------------
            if not hasattr(self, "queue") or not isinstance(self.queue, list):
                self.queue = []

            self._running = True
            self._shutdown = False

            try:
                if hasattr(self, "logger"):
                    self.logger.info("🚀 Command processor started")
            except Exception:
                pass

            # ------------------------
            # 🔥 WORKER LOOP (SAFE)
            # ------------------------
            while self._running and not getattr(self, "_shutdown", False):

                try:
                    # ------------------------
                    # 🔥 GLOBAL SHUTDOWN CHECK
                    # ------------------------
                    try:
                        if (
                            "task_registry" in globals()
                            and task_registry.is_shutting_down()
                        ):
                            break
                    except Exception:
                        pass

                    # ------------------------
                    # 🔥 PAUSE SUPPORT
                    # ------------------------
                    if getattr(self, "_paused", False):
                        await asyncio.sleep(0.2)
                        continue

                    # ------------------------
                    # 🔥 EMPTY QUEUE (NON-BLOCKING)
                    # ------------------------
                    if not self.queue:
                        await asyncio.sleep(0.05)
                        continue

                    # ------------------------
                    # 🔥 POP COMMAND (THREAD-SAFE STYLE)
                    # ------------------------
                    try:
                        command = heapq.heappop(self.queue)
                    except IndexError:
                        continue

                    # ------------------------
                    # 🔥 STATE UPDATE
                    # ------------------------
                    try:
                        command.status = "processing"
                        command.started_at = time.time()
                    except Exception:
                        pass

                    # ------------------------
                    # 🔥 EXECUTION DISPATCH (SAFE)
                    # ------------------------
                    try:
                        if getattr(command, "async_execution", False):
                            asyncio.create_task(self._safe_execute(command))
                        else:
                            await self._safe_execute(command)
                    except Exception as e:
                        try:
                            if hasattr(self, "logger"):
                                self.logger.error(f"Execution dispatch failed: {e}")
                        except Exception:
                            pass

                    # ------------------------
                    # 🔥 METRICS (SAFE)
                    # ------------------------
                    try:
                        self.last_execution_time = time.time()
                    except Exception:
                        pass

                except asyncio.CancelledError:
                    break

                except Exception as e:
                    try:
                        if hasattr(self, "logger"):
                            self.logger.error(f"❌ Queue processing error: {e}")
                    except Exception:
                        pass

                    await asyncio.sleep(0.1)

            # ------------------------
            # 🔥 STOPPED STATE
            # ------------------------
            self._running = False

            try:
                if hasattr(self, "logger"):
                    self.logger.info("🛑 Command processor stopped")
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Critical queue failure: {e}")
            except Exception:
                pass

    # ------------------------
    # 🔥 SAFE EXECUTION WRAPPER
    # ------------------------
    async def _safe_execute(self, command: Command):
        """
        Safe execution wrapper to isolate failures (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                return

            # ------------------------
            # 🔥 EXECUTION
            # ------------------------
            await self.route(command)

        except asyncio.CancelledError:
            # ------------------------
            # 🔥 CANCEL HANDLING
            # ------------------------
            try:
                command.status = "cancelled"
                command.last_error = "cancelled"
            except Exception:
                pass
            raise

        except Exception as e:
            error_str = str(e)

            # ------------------------
            # 🔥 STATE UPDATE
            # ------------------------
            try:
                command.status = "failed"
                command.last_error = error_str
                command.completed_at = time.time()
            except Exception:
                pass

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.failed_commands = int(getattr(self, "failed_commands", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"❌ Command execution failed: "
                        f"{getattr(command, 'action', 'unknown')} - {error_str}"
                    )
            except Exception:
                pass

            # ------------------------
            # 🔥 OPTIONAL ERROR HOOK
            # ------------------------
            try:
                if hasattr(self, "on_error"):
                    await self.on_error(command, e)
            except Exception:
                pass

    def stop(self, force: bool = False) -> bool:
        """
        Stop command processor safely (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 SAFE STATE CHECK (CRITICAL FIX)
            # ------------------------
            if not getattr(self, "_running", False):
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning("⚠ Command processor already stopped")
                except Exception:
                    pass
                return False

            # ------------------------
            # 🔥 STATE UPDATE
            # ------------------------
            self._running = False
            self._shutdown = True

            # ------------------------
            # 🔥 SAFE QUEUE ACCESS
            # ------------------------
            queue = getattr(self, "queue", None)

            # ------------------------
            # 🔥 FORCE CLEAR (SAFE)
            # ------------------------
            if force and isinstance(queue, list):
                try:
                    cleared = len(queue)
                    queue.clear()

                    if hasattr(self, "logger"):
                        self.logger.warning(
                            f"⚠ Force stop: cleared {cleared} queued commands"
                        )
                except Exception:
                    pass

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.stopped_at = time.time()
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info("🛑 Command processor stopped")
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Stop failed: {e}")
            except Exception:
                pass

            return False

    # ------------------------
    # LOGGING
    # ------------------------
    def _log_success(self, command: Command, result: Any) -> None:
        """
        Log successful execution (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                return

            # ------------------------
            # 🔥 SAFE HISTORY INIT
            # ------------------------
            if not hasattr(self, "execution_history") or not isinstance(
                self.execution_history, list
            ):
                self.execution_history = []

            # ------------------------
            # 🔥 SAFE ATTRIBUTE EXTRACTION
            # ------------------------
            cmd_type = getattr(getattr(command, "cmd_type", None), "value", "unknown")
            action = getattr(command, "action", "unknown")

            # ------------------------
            # 🔥 BUILD ENTRY
            # ------------------------
            entry = {
                "type": cmd_type,
                "action": action,
                "status": "success",
                "result_type": type(result).__name__,
                "timestamp": time.time(),
            }

            # ------------------------
            # 🔥 OPTIONAL METADATA (SAFE)
            # ------------------------
            exec_time = getattr(command, "execution_time", None)
            if isinstance(exec_time, (int, float)):
                entry["execution_time"] = exec_time

            command_id = getattr(command, "command_id", None)
            if command_id:
                entry["command_id"] = command_id

            # ------------------------
            # 🔥 APPEND HISTORY
            # ------------------------
            self.execution_history.append(entry)

            # ------------------------
            # 🔥 HISTORY SIZE LIMIT (SAFE)
            # ------------------------
            max_size = int(getattr(self, "max_history_size", 500) or 500)

            if len(self.execution_history) > max_size:
                self.execution_history = self.execution_history[-max_size:]

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.completed_commands = (
                    int(getattr(self, "completed_commands", 0)) + 1
                )
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"✅ Success: {cmd_type}/{action} ({entry['result_type']})"
                    )
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.warning(f"⚠ Success logging failed: {e}")
            except Exception:
                pass

    def _log_error(self, command: Command, error: Exception) -> None:
        """
        Log failed execution (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 VALIDATION (CRITICAL FIX)
            # ------------------------
            if command is None:
                return

            error_str = str(error)

            # ------------------------
            # 🔥 SAFE HISTORY INIT
            # ------------------------
            if not hasattr(self, "execution_history") or not isinstance(
                self.execution_history, list
            ):
                self.execution_history = []

            # ------------------------
            # 🔥 SAFE ATTRIBUTE EXTRACTION
            # ------------------------
            cmd_type = getattr(getattr(command, "cmd_type", None), "value", "unknown")
            action = getattr(command, "action", "unknown")

            # ------------------------
            # 🔥 BUILD ENTRY
            # ------------------------
            entry = {
                "type": cmd_type,
                "action": action,
                "status": "error",
                "error": error_str,
                "error_type": type(error).__name__,
                "timestamp": time.time(),
            }

            # ------------------------
            # 🔥 OPTIONAL METADATA (SAFE)
            # ------------------------
            command_id = getattr(command, "command_id", None)
            if command_id:
                entry["command_id"] = command_id

            exec_time = getattr(command, "execution_time", None)
            if isinstance(exec_time, (int, float)):
                entry["execution_time"] = exec_time

            attempts = getattr(command, "attempts", None)
            if isinstance(attempts, int):
                entry["attempts"] = attempts

            # ------------------------
            # 🔥 APPEND HISTORY
            # ------------------------
            self.execution_history.append(entry)

            # ------------------------
            # 🔥 HISTORY SIZE LIMIT (SAFE)
            # ------------------------
            max_size = int(getattr(self, "max_history_size", 500) or 500)

            if len(self.execution_history) > max_size:
                self.execution_history = self.execution_history[-max_size:]

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.failed_commands = int(getattr(self, "failed_commands", 0)) + 1
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"❌ Command failed: {cmd_type}/{action} -> {error_str}"
                    )
            except Exception:
                pass

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.warning(f"⚠ Error logging failed: {e}")
            except Exception:
                pass

    # ------------------------
    # INFO
    # ------------------------
    def get_available_commands(
        self,
        cmd_type: Optional[CommandType] = None,
        include_empty: bool = False,
        include_count: bool = False,
    ) -> Dict[str, Any]:
        """
        Get available commands safely (optimized + production-grade)
        """

        try:
            result: Dict[str, Any] = {}

            # ------------------------
            # 🔥 SAFE HANDLERS INIT (CRITICAL FIX)
            # ------------------------
            handlers = getattr(self, "handlers", {}) or {}

            # ------------------------
            # 🔥 NORMALIZE INPUT
            # ------------------------
            if cmd_type and not isinstance(cmd_type, CommandType):
                cmd_type = CommandType.from_string(cmd_type)

            # ------------------------
            # 🔥 SINGLE TYPE MODE
            # ------------------------
            if cmd_type:
                actions = handlers.get(cmd_type, {}) or {}

                if actions or include_empty:
                    result[cmd_type.value] = list(actions.keys())

                    if include_count:
                        result[f"{cmd_type.value}_count"] = len(actions)

                return result

            # ------------------------
            # 🔥 ALL TYPES MODE
            # ------------------------
            total_commands = 0

            for ctype, actions in handlers.items():
                actions = actions or {}

                if not actions and not include_empty:
                    continue

                result[ctype.value] = list(actions.keys())

                if include_count:
                    count = len(actions)
                    result[f"{ctype.value}_count"] = count
                    total_commands += count

            # ------------------------
            # 🔥 METADATA
            # ------------------------
            if include_count:
                result["total_types"] = len(handlers)
                result["total_commands"] = total_commands

            return result

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ get_available_commands failed: {e}")
            except Exception:
                pass

            return {}

    def clear_history(self, archive: bool = False) -> bool:
        """
        Clear execution history safely (optimized + production-grade)
        """

        import time

        try:
            # ------------------------
            # 🔥 SAFE HISTORY ACCESS (CRITICAL FIX)
            # ------------------------
            history = getattr(self, "execution_history", None)

            if not isinstance(history, list):
                return False

            history_size = len(history)

            # ------------------------
            # 🔥 EMPTY CHECK
            # ------------------------
            if history_size == 0:
                try:
                    if hasattr(self, "logger"):
                        self.logger.warning("⚠ History already empty")
                except Exception:
                    pass
                return True

            # ------------------------
            # 🔥 OPTIONAL ARCHIVE (SAFE)
            # ------------------------
            if archive:
                try:
                    if not hasattr(self, "_history_archive") or not isinstance(
                        self._history_archive, list
                    ):
                        self._history_archive = []

                    # extend safely
                    self._history_archive.extend(history[:])
                except Exception:
                    pass

            # ------------------------
            # 🔥 CLEAR HISTORY
            # ------------------------
            try:
                history.clear()
            except Exception:
                self.execution_history = []

            # ------------------------
            # 🔥 METRICS (SAFE)
            # ------------------------
            try:
                self.history_cleared_at = time.time()
                self.last_cleared_count = history_size
            except Exception:
                pass

            # ------------------------
            # 🔥 SAFE LOGGING (FIXED)
            # ------------------------
            try:
                if hasattr(self, "logger"):
                    self.logger.info(f"🧹 History cleared ({history_size} entries)")
            except Exception:
                pass

            return True

        except Exception as e:
            try:
                if hasattr(self, "logger"):
                    self.logger.error(f"❌ Clear history failed: {e}")
            except Exception:
                pass

            return False
