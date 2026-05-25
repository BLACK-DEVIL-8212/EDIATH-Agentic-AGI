from __future__ import annotations

from typing import Any, Dict, Optional, Set, List
from datetime import datetime
from collections import deque
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import asyncio
import sys
import ast
import tracemalloc
import threading
import time
import warnings

# Platform-specific imports
if sys.platform != "win32":
    import resource
    import signal
else:
    resource = None  # type: ignore[assignment]
    signal = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SandboxException(Exception):
    """Raised when sandbox execution fails."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class SandboxPolicy:
    """Enhanced security policy for sandbox execution."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_memory_mb: int = 256,
        max_cpu_time: float = 5.0,
        max_output_size_kb: int = 1024,
        allowed_imports: Optional[Set[str]] = None,
        disallowed_functions: Optional[Set[str]] = None,
        disallowed_modules: Optional[Set[str]] = None,
        disallowed_attributes: Optional[Set[str]] = None,
        allowed_paths: Optional[Set[str]] = None,
        allow_network: bool = False,
        allow_filesystem: bool = False,
        allow_subprocess: bool = False,
        max_stack_depth: int = 100,
        restrict_builtins: bool = True,
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_output_size_kb = max_output_size_kb
        self.allow_network = allow_network
        self.allow_filesystem = allow_filesystem
        self.allow_subprocess = allow_subprocess
        self.max_stack_depth = max_stack_depth
        self.restrict_builtins = restrict_builtins

        self.allowed_imports: Set[str] = allowed_imports or {
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "string",
            "re",
            "json",
            "typing",
        }

        self.disallowed_functions: Set[str] = disallowed_functions or {
            "exec",
            "eval",
            "__import__",
            "compile",
            "globals",
            "locals",
            "vars",
            "dir",
            "help",
            "open",
            "input",
            "breakpoint",
            "execfile",
            "reload",
            "__builtins__",
            "__dict__",
            "__globals__",
            "__locals__",
            "__call__",
            "__subclasses__",
            "__bases__",
            "__mro__",
            "__code__",
            "__func__",
        }

        self.disallowed_modules: Set[str] = disallowed_modules or {
            "os",
            "sys",
            "subprocess",
            "socket",
            "requests",
            "urllib",
            "shutil",
            "pathlib",
            "importlib",
            "inspect",
            "ctypes",
            "multiprocessing",
            "threading",
            "asyncio",
            "sysconfig",
            "distutils",
            "setuptools",
            "pickle",
            "shelve",
            "marshal",
            "sqlite3",
            "dbm",
            "crypt",
            "ssl",
            "hashlib",
            "hmac",
        }

        # Windows does not have the resource module
        if sys.platform == "win32":
            self.disallowed_modules.discard("resource")

        self.disallowed_attributes: Set[str] = disallowed_attributes or {
            "__import__",
            "__loader__",
            "__spec__",
            "__builtins__",
            "__class__",
            "__bases__",
            "__subclasses__",
            "__del__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
            "__reduce__",
            "__reduce_ex__",
            "__getstate__",
            "__mro__",
            "__dict__",
            "__globals__",
            "__code__",
        }

        self.allowed_paths: Set[str] = allowed_paths or set()

    def check_code(self, code: str) -> tuple[bool, str]:
        """
        Check whether code is safe to execute.

        Returns (True, "OK") if safe, or (False, reason) if not.

        FIX: removed the naive substring-based scan that ran *after* the AST walk.
             Substring matching (e.g. ``"import os" in code``) is defeated by
             aliasing, multi-line strings, or encoding tricks.  The AST visitor is
             the authoritative check.
        """
        if "\x00" in code:
            return False, "Null bytes detected in code"

        if len(code) > 100_000:
            return False, "Code exceeds maximum size (100 KB)"

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        visitor = SecurityVisitor(self)
        try:
            visitor.visit(tree)
        except Exception as exc:
            return False, f"Security check failed: {exc}"

        if visitor.violation:
            return False, visitor.violation

        return True, "OK"


# ---------------------------------------------------------------------------
# AST Security Visitor
# ---------------------------------------------------------------------------


class SecurityVisitor(ast.NodeVisitor):
    """AST visitor that enforces SandboxPolicy rules."""

    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy
        self.violation: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _check_attribute_chain(self, node: ast.expr) -> Optional[str]:
        """
        Recursively inspect an attribute chain such as
        ``().__class__.__bases__[0].__subclasses__()``
        and return a violation string if a disallowed dunder is found.

        FIX: the original visitor only checked the *immediate* attribute name,
             so chained dunder accesses like ``x.__class__.__subclasses__``
             were not caught.
        """
        if isinstance(node, ast.Attribute):
            if node.attr in self.policy.disallowed_attributes:
                return f"Disallowed attribute access: {node.attr}"
            return self._check_attribute_chain(node.value)
        return None

    def _abort_if_violated(self) -> bool:
        return self.violation is not None

    # ------------------------------------------------------------------ #
    # Visitors                                                             #
    # ------------------------------------------------------------------ #

    def visit_Import(self, node: ast.Import) -> None:
        if self._abort_if_violated():
            return
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in self.policy.disallowed_modules:
                self.violation = f"Disallowed import: {module_name}"
                return
            if module_name not in self.policy.allowed_imports:
                self.violation = f"Unauthorized import: {module_name}"
                return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._abort_if_violated():
            return
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name in self.policy.disallowed_modules:
                self.violation = f"Disallowed import: {module_name}"
                return
            if module_name not in self.policy.allowed_imports:
                self.violation = f"Unauthorized import: {module_name}"
                return
        for alias in node.names:
            if alias.name in self.policy.disallowed_functions:
                self.violation = f"Disallowed import of name: {alias.name}"
                return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._abort_if_violated():
            return
        if isinstance(node.func, ast.Name):
            if node.func.id in self.policy.disallowed_functions:
                self.violation = f"Disallowed function call: {node.func.id}"
                return
        elif isinstance(node.func, ast.Attribute):
            violation = self._check_attribute_chain(node.func)
            if violation:
                self.violation = violation
                return
            if node.func.attr in self.policy.disallowed_functions:
                self.violation = f"Disallowed method call: {node.func.attr}"
                return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._abort_if_violated():
            return
        # FIX: check the full chain, not just the leaf attribute
        violation = self._check_attribute_chain(node)
        if violation:
            self.violation = violation
            return
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


class ExecutionResult:
    """Result of a sandboxed execution."""

    def __init__(self) -> None:
        self.success: bool = False
        self.output: Any = None
        self.stdout: str = ""
        self.stderr: str = ""
        self.error: Optional[str] = None
        self.execution_time: float = 0.0
        self.cpu_time: float = 0.0
        self.memory_used: int = 0
        self.peak_memory: int = 0
        self.exit_code: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": str(self.output) if self.output is not None else None,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "execution_time": self.execution_time,
            "cpu_time": self.cpu_time,
            "memory_used": self.memory_used,
            "peak_memory": self.peak_memory,
            "exit_code": self.exit_code,
        }


# ---------------------------------------------------------------------------
# Resource limiter
# ---------------------------------------------------------------------------


class ResourceLimiter:
    """Resource limiting utilities (cross-platform)."""

    @staticmethod
    def set_limits(memory_mb: int, cpu_time: float) -> None:
        """Set resource limits for the current process."""
        if resource is None:
            ResourceLimiter._set_windows_limits(memory_mb, cpu_time)
            return

        try:
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            cpu_seconds = int(cpu_time)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))

            resource.setrlimit(
                resource.RLIMIT_STACK, (8 * 1024 * 1024, 8 * 1024 * 1024)
            )
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

        except Exception as exc:
            warnings.warn(f"Could not set resource limits: {exc}")

    @staticmethod
    def _set_windows_limits(memory_mb: int, cpu_time: float) -> None:
        """Attempt Windows job-object based memory limiting."""
        try:
            import win32job
            import win32process

            h_process = win32process.GetCurrentProcess()
            h_job = win32job.CreateJobObject(None, "")

            job_info = win32job.QueryInformationJobObject(
                h_job, win32job.JobObjectExtendedLimitInformation
            )
            job_info.ProcessMemoryLimit = memory_mb * 1024 * 1024
            job_info.BasicLimitInformation.LimitFlags = (
                win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            )
            win32job.SetInformationJobObject(
                h_job, win32job.JobObjectExtendedLimitInformation, job_info
            )
            win32job.AssignProcessToJobObject(h_job, h_process)
        except Exception as exc:
            # win32job is optional; soft-limit only
            warnings.warn(f"Windows job-object limits not available: {exc}")

    @staticmethod
    def get_memory_usage() -> int:
        """Return current process RSS memory in bytes, or 0 on failure."""
        try:
            import psutil

            return psutil.Process().memory_info().rss
        except Exception:
            # FIX: was bare `except:` which swallowed KeyboardInterrupt etc.
            return 0

    @staticmethod
    def get_cpu_time() -> float:
        """Return CPU time consumed so far in seconds."""
        if resource is not None:
            return resource.getrusage(resource.RUSAGE_SELF).ru_utime
        # Windows fallback
        return time.process_time()


# ---------------------------------------------------------------------------
# Timeout executor
# ---------------------------------------------------------------------------


class TimeoutExecutor:
    """
    Execute a callable in a daemon thread with a wall-clock timeout.

    NOTE: Python threads cannot be forcibly killed.  If the executed function
    enters an infinite loop the thread will remain alive after the timeout
    (though the caller will receive a TimeoutError immediately).  The outer
    asyncio.wait_for guard in Sandbox.execute provides an additional layer of
    protection at the coroutine level.
    """

    @staticmethod
    def run_with_timeout(func, timeout: float, *args, **kwargs):
        """Run *func* in a daemon thread; raise TimeoutError if it exceeds *timeout*."""
        result: Dict[str, Any] = {"value": None, "error": None}
        done_event = threading.Event()

        def target():
            try:
                result["value"] = func(*args, **kwargs)
            except Exception as exc:
                result["error"] = exc
            finally:
                done_event.set()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        finished = done_event.wait(timeout=timeout)

        if not finished:
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        if result["error"] is not None:
            raise result["error"]

        return result["value"]


# ---------------------------------------------------------------------------
# Code executor
# ---------------------------------------------------------------------------


class CodeExecutor:
    """Execute code in a restricted namespace."""

    @staticmethod
    def execute_safe(
        code: str,
        timeout: float,
        max_memory_mb: int,
        max_output_kb: int,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute *code* with resource limits, output capture, and a locked-down namespace."""
        result = ExecutionResult()
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            tracemalloc.start()
            ResourceLimiter.set_limits(max_memory_mb, timeout)

            safe_globals = CodeExecutor._create_safe_globals()
            if variables:
                safe_globals.update(variables)

            def execute_code() -> bool:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # FIX: exec is called with an explicit, locked-down globals dict.
                    # __builtins__ is set to the restricted dict (not the real builtins
                    # module) so attribute-chain escapes like
                    # ().__class__.__bases__[0].__subclasses__() cannot reach
                    # dangerous built-ins.
                    exec(code, safe_globals)  # noqa: S102
                return True

            try:
                TimeoutExecutor.run_with_timeout(execute_code, timeout)
                result.success = True
                result.exit_code = 0
            except TimeoutError as exc:
                result.error = str(exc)
                result.exit_code = 1
            except Exception as exc:
                result.error = str(exc)
                result.exit_code = 1

            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            max_bytes = max_output_kb * 1024
            if len(result.stdout) > max_bytes:
                result.stdout = result.stdout[:max_bytes] + "\n... [TRUNCATED]"
            if len(result.stderr) > max_bytes:
                result.stderr = result.stderr[:max_bytes] + "\n... [TRUNCATED]"

            current, peak = tracemalloc.get_traced_memory()
            result.memory_used = current
            result.peak_memory = peak

        except Exception as exc:
            result.error = str(exc)
            result.stderr = stderr_capture.getvalue()
            result.exit_code = 1
        finally:
            tracemalloc.stop()

        return result

    @staticmethod
    def _create_safe_globals() -> Dict[str, Any]:
        """
        Return a locked-down global namespace for code execution.

        FIX: __builtins__ is set to a *dict* of safe callables, not to the real
             builtins module.  When __builtins__ is a dict Python uses it as the
             only source of built-in names, preventing access to dangerous
             callables via ``__builtins__.__dict__``.
        """
        safe_builtins: Dict[str, Any] = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            # Exceptions users legitimately raise
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "StopIteration": StopIteration,
            "NotImplementedError": NotImplementedError,
            "RuntimeError": RuntimeError,
            "AssertionError": AssertionError,
        }

        safe_modules: Dict[str, Any] = {}
        for module_name in [
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "string",
            "re",
            "json",
        ]:
            try:
                safe_modules[module_name] = __import__(module_name)
            except ImportError:
                pass

        globals_dict: Dict[str, Any] = {
            "__builtins__": safe_builtins,  # dict, NOT the builtins module
            "__name__": "__sandbox__",
            "__doc__": None,
        }
        globals_dict.update(safe_modules)
        return globals_dict


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    """Secure sandbox for async and sync code execution."""

    def __init__(self, policy: Optional[SandboxPolicy] = None) -> None:
        self.is_enabled: bool = True
        self.policy: SandboxPolicy = policy or SandboxPolicy()
        self.execution_count: int = 0
        self.failed_executions: int = 0
        # FIX: use deque for O(1) append/trim instead of list.pop(0) which is O(n)
        self._history: deque[ExecutionResult] = deque(maxlen=100)
        self._executor = CodeExecutor()

    # ------------------------------------------------------------------ #
    # Public – async execution                                             #
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        variables: Optional[Dict[str, Any]] = None,
        sandbox_globals: Optional[Dict[str, Any]] = None,  # ignored for security
        capture_stdout: bool = True,
    ) -> ExecutionResult:
        """
        Execute *code* inside the sandbox.

        Args:
            code: Source code string to run.
            timeout: Wall-clock timeout in seconds (overrides policy default).
            variables: Extra variables injected into the execution namespace.
            sandbox_globals: Ignored — present for API compatibility only.
            capture_stdout: Whether stdout/stderr are captured (always True).

        Returns:
            ExecutionResult populated with outcome details.
        """
        result = ExecutionResult()

        if not self.is_enabled:
            result.error = "Sandbox is disabled"
            return result

        is_safe, error_msg = self.policy.check_code(code)
        if not is_safe:
            result.error = error_msg
            self.failed_executions += 1
            return result

        effective_timeout = timeout if timeout is not None else self.policy.timeout

        try:
            start_time = datetime.now()
            start_cpu = ResourceLimiter.get_cpu_time()

            loop = asyncio.get_event_loop()

            exec_task = asyncio.create_task(
                loop.run_in_executor(
                    None,
                    self._executor.execute_safe,
                    code,
                    effective_timeout,
                    self.policy.max_memory_mb,
                    self.policy.max_output_size_kb,
                    variables or {},
                )
            )

            try:
                result = await asyncio.wait_for(
                    exec_task, timeout=effective_timeout + 1
                )
            except asyncio.TimeoutError:
                result.error = f"Execution timeout exceeded ({effective_timeout}s)"
                self.failed_executions += 1

            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.cpu_time = ResourceLimiter.get_cpu_time() - start_cpu

            self.execution_count += 1
            self._history.append(result)
            return result

        except Exception as exc:
            result.error = f"Execution failed: {exc}"
            self.failed_executions += 1
            self.execution_count += 1
            return result

    # ------------------------------------------------------------------ #
    # Public – sync execution                                              #
    # ------------------------------------------------------------------ #

    def execute_sync(
        self,
        code: str,
        timeout: Optional[float] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Synchronous wrapper around :meth:`execute`.

        FIX: original code referenced ``loop`` in the ``finally`` block without
             guarding against the case where ``asyncio.new_event_loop()`` itself
             raises, which would cause an ``UnboundLocalError``.
        """
        loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.execute(code, timeout, variables))
        except Exception as exc:
            result = ExecutionResult()
            result.error = f"Sync execution failed: {exc}"
            return result
        finally:
            if loop is not None and not loop.is_closed():
                loop.close()

    # ------------------------------------------------------------------ #
    # Convenience properties / methods                                     #
    # ------------------------------------------------------------------ #

    @property
    def execution_history(self) -> List[ExecutionResult]:
        """Return history as a plain list (read-only snapshot)."""
        return list(self._history)

    def enable(self) -> None:
        """Enable the sandbox."""
        self.is_enabled = True

    def disable(self) -> None:
        """Disable the sandbox (use with extreme caution)."""
        self.is_enabled = False

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return sandbox statistics."""
        success_rate = 0.0
        if self.execution_count > 0:
            success_rate = (
                self.execution_count - self.failed_executions
            ) / self.execution_count
        return {
            "enabled": self.is_enabled,
            "executions": self.execution_count,
            "failures": self.failed_executions,
            "success_rate": success_rate,
            "timeout": self.policy.timeout,
            "max_memory_mb": self.policy.max_memory_mb,
            "max_cpu_time": self.policy.max_cpu_time,
            "allow_network": self.policy.allow_network,
            "allow_filesystem": self.policy.allow_filesystem,
            "allow_subprocess": self.policy.allow_subprocess,
            "history_size": len(self._history),
            "platform": sys.platform,
        }

    def get_last_execution(self) -> Optional[ExecutionResult]:
        """Return the most recent ExecutionResult, or None."""
        return self._history[-1] if self._history else None

    def test_safety(self, code: str) -> Dict[str, Any]:
        """Check code safety without executing it."""
        is_safe, error_msg = self.policy.check_code(code)
        return {
            "safe": is_safe,
            "error": error_msg if not is_safe else None,
            "code_length": len(code),
            "line_count": len(code.split("\n")),
        }


# ---------------------------------------------------------------------------
# Pre-configured subclasses
# ---------------------------------------------------------------------------


class SecureSandbox(Sandbox):
    """Maximum-security sandbox — recommended for untrusted code."""

    def __init__(self) -> None:
        super().__init__(
            SandboxPolicy(
                timeout=5.0,
                max_memory_mb=128,
                max_cpu_time=3.0,
                allow_network=False,
                allow_filesystem=False,
                allow_subprocess=False,
                restrict_builtins=True,
            )
        )


class TrustedSandbox(Sandbox):
    """Less restrictive sandbox for trusted/internal code."""

    def __init__(self) -> None:
        super().__init__(
            SandboxPolicy(
                timeout=30.0,
                max_memory_mb=512,
                max_cpu_time=20.0,
                allow_network=True,
                allow_filesystem=True,
                allow_subprocess=False,
                restrict_builtins=False,
            )
        )


# ---------------------------------------------------------------------------
# Example / smoke test (disabled in production)
# ---------------------------------------------------------------------------

if False:  # Disabled: unsafe demo code should not run in production

    import logging

    logging.basicConfig(level=logging.INFO)

    sandbox = SecureSandbox()

    # Safe code
    result = sandbox.execute_sync("print('Hello, World!')")
    print(f"[safe]   success={result.success}  stdout={result.stdout.strip()!r}")

    # Unsafe – blocked import
    result = sandbox.execute_sync("import os; os.system('ls')")
    print(f"[unsafe] success={result.success}  error={result.error!r}")

    # Variable injection
    result = sandbox.execute_sync(
        "result = x + y\nprint(result)", variables={"x": 10, "y": 20}
    )
    print(f"[vars]   success={result.success}  stdout={result.stdout.strip()!r}")

    # Dunder chain escape attempt — should be blocked
    result = sandbox.execute_sync("().__class__.__bases__[0].__subclasses__()")
    print(f"[dunder] success={result.success}  error={result.error!r}")

    # Stats
    print(f"\nStats: {sandbox.get_stats()}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Sandbox",
    "SandboxPolicy",
    "ExecutionResult",
    "SandboxException",
    "SecureSandbox",
    "TrustedSandbox",
    "CodeExecutor",
    "ResourceLimiter",
    "SecurityVisitor",
    "TimeoutExecutor",
]
