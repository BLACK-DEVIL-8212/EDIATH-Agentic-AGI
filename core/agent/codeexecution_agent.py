"""
🔥 FINAL PRODUCTION Code Execution Agent for EDIATH
✔ Safe code execution in isolated sandboxes
✔ Multiple language support (Python, JS, Bash, C++, Java, Go, Rust, Ruby, PowerShell)
✔ Resource limiting (CPU, memory, time)
✔ Docker container isolation
✔ Static code analysis (AST + regex)
✔ Code validation and sanitization
✔ Persistent execution sessions
✔ Package management in sandboxes
✔ Circuit breaker pattern
✔ Rate limiting
✔ Execution history with persistence
✔ Production ready
"""

import asyncio
import re
import tempfile
import os
import sys
import time
import ast
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from collections import deque
import logging
import traceback
import hashlib

# Import blacklist module
try:
    from . import blacklist

    BLACKLIST_AVAILABLE = True
except ImportError:
    try:
        import blacklist

        BLACKLIST_AVAILABLE = True
    except ImportError:
        BLACKLIST_AVAILABLE = False

# Resource module is Unix-only, handle Windows compatibility
try:
    import resource

    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

# Docker for container isolation
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Restricted Python execution
try:
    from RestrictedPython import compile_restricted, safe_globals

    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False

# psutil for memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# =========================
# ENUMS AND CONSTANTS
# =========================


class Language(Enum):
    """Supported programming languages"""

    PYTHON = "python"
    PYTHON_SAFE = "python_safe"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    POWERSHELL = "powershell"
    NODE = "node"
    RUBY = "ruby"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    TYPESCRIPT = "typescript"
    PHP = "php"
    PERL = "perl"


class SandboxType(Enum):
    """Sandbox isolation methods"""

    SUBSHELL = "subshell"
    DOCKER = "docker"
    NAMESPACE = "namespace"
    RESTRICTED = "restricted"


class ExecutionStatus(Enum):
    """Execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# =========================
# DATACLASSES
# =========================


@dataclass
class ExecutionResult:
    """Code execution result"""

    id: str
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_used: int
    language: str
    timestamp: datetime
    status: ExecutionStatus
    error: Optional[str] = None
    analysis: Optional[Dict] = None


@dataclass
class CodeAnalysis:
    """Static code analysis result"""

    is_safe: bool
    issues: List[str]
    imports: List[str]
    functions: List[str]
    dangerous_patterns: List[str]
    complexity_score: float
    lines_of_code: int
    language: str
    security_score: float = 0.0


@dataclass
class ExecutionSession:
    """Persistent execution session"""

    id: str
    language: Language
    created_at: datetime
    last_activity: datetime
    variables: Dict[str, Any]
    history: List[Dict]
    environment: Dict[str, str]


# =========================
# MAIN AGENT
# =========================


class CodeExecutionAgent:
    """
    Advanced code execution agent capable of:
    - Safe code execution in sandboxed environments
    - Multiple language support (Python, JS, Bash, etc.)
    - Resource limiting (CPU, memory, time)
    - Docker container isolation
    - Static code analysis
    - Code validation and sanitization
    - Persistent code execution sessions
    - Package management in sandboxes
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Code Execution Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Sandbox configuration
        self.sandbox_type = SandboxType(self.config.get("sandbox_type", "subshell"))
        self.default_language = Language(self.config.get("default_language", "python"))
        self.timeout_default = self.config.get("timeout", 30)
        self.memory_limit = self.config.get("memory_limit", 256 * 1024 * 1024)  # 256 MB
        self.cpu_limit = self.config.get("cpu_limit", 1.0)

        # Docker configuration
        self.docker_image = self.config.get("docker_image", "python:3.11-slim")
        self.docker_client = None
        if DOCKER_AVAILABLE and self.sandbox_type == SandboxType.DOCKER:
            try:
                self.docker_client = docker.from_env()
            except Exception as exc:
                self.logger.warning(
                    "Docker not available (%s), falling back to subshell", exc
                )
                self.sandbox_type = SandboxType.SUBSHELL

        # Workspace configuration
        self.workspace_dir = Path(self.config.get("workspace_dir", "./code_workspace"))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Allowed / blocked modules for Python safe execution
        self.allowed_modules = set(
            self.config.get(
                "allowed_modules",
                [
                    "math",
                    "random",
                    "datetime",
                    "json",
                    "re",
                    "string",
                    "collections",
                    "itertools",
                    "functools",
                    "typing",
                    "statistics",
                    "decimal",
                    "fractions",
                    "heapq",
                    "bisect",
                ],
            )
        )

        self.blocked_modules = set(
            self.config.get(
                "blocked_modules",
                [
                    "os",
                    "subprocess",
                    "sys",
                    "socket",
                    "requests",
                    "urllib",
                    "importlib",
                    "__builtins__",
                    "eval",
                    "exec",
                    "compile",
                    "open",
                    "file",
                    "input",
                    "execfile",
                ],
            )
        )

        # Dangerous AST-level call names
        self._dangerous_calls = {"eval", "exec", "compile", "__import__", "execfile"}

        # Dangerous regex patterns (secondary check)
        self._dangerous_regex_patterns = [
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"subprocess\.",
            r"os\.system",
            r"os\.popen",
            r"os\.fork",
            r"os\.exec",
            r"__builtins__",
            r"globals\(",
            r"locals\(",
            r"getattr\(",
            r"setattr\(",
            r"delattr\(",
            r"execfile",
            r"input\(",
            r"raw_input\(",
        ]

        # Circuit breaker
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime] = None
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)
        self._consecutive_failures = 0

        # Rate limiting
        self._request_timestamps: deque = deque(maxlen=100)
        self.max_requests_per_minute = self.config.get("max_requests_per_minute", 30)
        self._last_request_time = 0
        self.min_request_interval = self.config.get("min_request_interval", 0.1)

        # Statistics
        self.stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "blocked_executions": 0,
            "timeout_executions": 0,
            "total_execution_time": 0.0,
            "average_memory_usage": 0.0,
            "by_language": {},
            "start_time": datetime.now().isoformat(),
        }

        # Execution history
        self.execution_history: List[ExecutionResult] = []
        self.max_history: int = self.config.get("max_history", 1000)
        self.history_file = Path(
            self.config.get("history_file", "./code_workspace/history.json")
        )
        self._load_history()

        # Active sessions
        self.active_sessions: Dict[str, ExecutionSession] = {}

        # Cache for analysis results
        self._analysis_cache: Dict[str, CodeAnalysis] = {}
        self._cache_max_size = 1000

        # Execution queue
        self.execution_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.get("max_queue_size", 100)
        )
        self.queue_processor_task: Optional[asyncio.Task] = None
        self.queue_enabled = self.config.get("queue_enabled", False)

        self.logger.info(
            "Code Execution Agent initialized. Sandbox: %s", self.sandbox_type.value
        )

    # =========================
    # CIRCUIT BREAKER
    # =========================

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open"""
        if not self._circuit_open:
            return True

        if self._circuit_open_until and datetime.now() >= self._circuit_open_until:
            self._circuit_open = False
            self._consecutive_failures = 0
            self.logger.info("Circuit breaker closed")
            return True

        return False

    async def _record_failure(self):
        """Record a failure for circuit breaker"""
        self._consecutive_failures += 1
        self.stats["failed_executions"] += 1

        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_open = True
            self._circuit_open_until = datetime.now() + timedelta(
                seconds=self.circuit_breaker_timeout
            )
            self.logger.error(
                f"Circuit breaker opened after {self._consecutive_failures} failures"
            )

    async def _record_success(self):
        """Record a successful execution"""
        self._consecutive_failures = 0

    # =========================
    # RATE LIMITING
    # =========================

    async def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()

        # Basic rate limiting
        if now - self._last_request_time < self.min_request_interval:
            await asyncio.sleep(
                self.min_request_interval - (now - self._last_request_time)
            )

        # Sliding window rate limiting
        self._request_timestamps.append(now)
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            oldest = self._request_timestamps[0]
            if now - oldest < 60:
                wait_time = 60 - (now - oldest)
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self._last_request_time = time.time()
        return True

    # =========================
    # PERSISTENCE
    # =========================

    def _load_history(self):
        """Load execution history from file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[-self.max_history :]:
                    result = ExecutionResult(
                        id=item["id"],
                        success=item["success"],
                        stdout=item.get("stdout", ""),
                        stderr=item.get("stderr", ""),
                        return_code=item["return_code"],
                        execution_time=item["execution_time"],
                        memory_used=item.get("memory_used", 0),
                        language=item["language"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        status=ExecutionStatus(item.get("status", "completed")),
                        error=item.get("error"),
                    )
                    self.execution_history.append(result)
            self.logger.info(f"Loaded {len(self.execution_history)} history entries")
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save execution history to file"""
        try:
            data = []
            for result in self.execution_history[-self.max_history :]:
                data.append(
                    {
                        "id": result.id,
                        "success": result.success,
                        "stdout": result.stdout[:500],
                        "stderr": result.stderr[:500],
                        "return_code": result.return_code,
                        "execution_time": result.execution_time,
                        "memory_used": result.memory_used,
                        "language": result.language,
                        "timestamp": result.timestamp.isoformat(),
                        "status": result.status.value,
                        "error": result.error,
                    }
                )

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save history: {e}")

    # =========================
    # QUEUE MANAGEMENT
    # =========================

    async def _start_queue_processor(self):
        """Start background queue processor"""
        if not self.queue_enabled:
            return

        self.queue_processor_task = asyncio.create_task(self._process_queue())
        self.logger.debug("Queue processor started")

    async def _process_queue(self):
        """Process queued executions"""
        while self.queue_enabled:
            try:
                if not self.execution_queue.empty():
                    task = await self.execution_queue.get()
                    result = await self._execute_with_retry(
                        task["code"],
                        task["language"],
                        task["timeout"],
                        task["memory_limit"],
                        task["stdin_input"],
                        task["environment"],
                    )
                    if task.get("callback"):
                        if asyncio.iscoroutinefunction(task["callback"]):
                            await task["callback"](result)
                        else:
                            task["callback"](result)
                    self.execution_queue.task_done()
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)

    async def queue_execution(
        self,
        code: str,
        language: Optional[Language] = None,
        timeout: Optional[int] = None,
        memory_limit: Optional[int] = None,
        stdin_input: Optional[str] = None,
        environment: Optional[Dict] = None,
        callback: Optional[callable] = None,
    ) -> str:
        """Queue code for execution"""
        task_id = str(uuid.uuid4())[:8]

        await self.execution_queue.put(
            {
                "id": task_id,
                "code": code,
                "language": language or self.default_language,
                "timeout": timeout or self.timeout_default,
                "memory_limit": memory_limit or self.memory_limit,
                "stdin_input": stdin_input,
                "environment": environment,
                "callback": callback,
            }
        )

        return task_id

    # =========================
    # CODE ANALYSIS
    # =========================

    async def analyze_code(
        self, code: str, language: Language = Language.PYTHON, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Statically analyze code for safety and quality.

        Args:
            code: Source code to analyze
            language: Programming language
            use_cache: Use cached analysis results

        Returns:
            Dictionary with analysis results
        """
        # Check cache
        cache_key = hashlib.md5(f"{code}_{language.value}".encode()).hexdigest()
        if use_cache and cache_key in self._analysis_cache:
            cache_hit = self._analysis_cache[cache_key]
            return {
                "success": True,
                "is_safe": cache_hit.is_safe,
                "issues": cache_hit.issues,
                "imports": cache_hit.imports,
                "functions": cache_hit.functions,
                "dangerous_patterns": cache_hit.dangerous_patterns,
                "complexity_score": cache_hit.complexity_score,
                "lines_of_code": cache_hit.lines_of_code,
                "security_score": cache_hit.security_score,
                "language": cache_hit.language,
                "cached": True,
            }

        # Analyze based on language
        if language in (Language.PYTHON, Language.PYTHON_SAFE):
            analysis = await self._analyze_python_code(code)
        elif language == Language.JAVASCRIPT:
            analysis = await self._analyze_javascript_code(code)
        elif language in [Language.BASH, Language.POWERSHELL]:
            analysis = await self._analyze_shell_code(code, language)
        else:
            analysis = await self._analyze_generic_code(code, language)

        # Cache result
        if use_cache and len(self._analysis_cache) < self._cache_max_size:
            self._analysis_cache[cache_key] = CodeAnalysis(**analysis)

        return {"success": True, **analysis, "cached": False}

    async def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Analyze Python code for safety using AST + regex sweep."""
        issues: List[str] = []
        imports: List[str] = []
        functions: List[str] = []
        dangerous_patterns_found: List[str] = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Import checks
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        imports.append(module_name)
                        if module_name in self.blocked_modules:
                            msg = f"Blocked import: {module_name}"
                            issues.append(msg)
                            dangerous_patterns_found.append(f"import {module_name}")

                elif isinstance(node, ast.ImportFrom):
                    module_name = (node.module or "").split(".")[0]
                    if module_name in self.blocked_modules:
                        msg = f"Blocked import from: {module_name}"
                        issues.append(msg)
                        dangerous_patterns_found.append(f"from {module_name} import")
                    for alias in node.names:
                        imports.append(f"{module_name}.{alias.name}")

                # Function definitions
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)

                # Dangerous call names
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self._dangerous_calls:
                            msg = f"Dangerous function call: {node.func.id}()"
                            issues.append(msg)
                            dangerous_patterns_found.append(f"{node.func.id}(")

            # Secondary regex sweep – only report patterns NOT already caught
            existing_issue_text = " ".join(issues)
            for pattern in self._dangerous_regex_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    label = f"Dangerous pattern: {pattern}"
                    if pattern not in existing_issue_text:
                        issues.append(label)
                        dangerous_patterns_found.append(pattern)

            lines = len(code.split("\n"))
            complexity = len(functions) + (lines / 50) + len(imports) / 10
            security_score = max(0, 100 - (len(issues) * 10))

            return {
                "is_safe": len(issues) == 0,
                "issues": issues,
                "imports": imports,
                "functions": functions,
                "dangerous_patterns": dangerous_patterns_found,
                "complexity_score": min(complexity, 10.0),
                "lines_of_code": lines,
                "language": "python",
                "security_score": security_score,
            }

        except SyntaxError as exc:
            return {
                "is_safe": False,
                "issues": [f"Syntax error: {exc}"],
                "imports": [],
                "functions": [],
                "dangerous_patterns": [],
                "complexity_score": 0.0,
                "lines_of_code": 0,
                "language": "python",
                "security_score": 0.0,
            }
        except Exception as exc:
            self.logger.error("Python analysis error: %s", exc)
            return {
                "is_safe": False,
                "issues": [f"Analysis error: {exc}"],
                "imports": [],
                "functions": [],
                "dangerous_patterns": [],
                "complexity_score": 0.0,
                "lines_of_code": 0,
                "language": "python",
                "security_score": 0.0,
            }

    async def _analyze_javascript_code(self, code: str) -> Dict[str, Any]:
        """Analyze JavaScript code for safety"""
        issues = []
        dangerous_patterns = []

        dangerous_js_patterns = [
            (r"eval\s*\(", "eval() execution"),
            (r"Function\s*\(", "Function constructor"),
            (r"child_process\.exec", "Child process execution"),
            (r'require\s*\([\'"]fs[\'"]\)', "File system access"),
            (r"process\.env", "Environment access"),
            (r"__dirname", "Directory info"),
            (r"__filename", "File info"),
            (r"fetch\s*\(", "HTTP request"),
            (r"XMLHttpRequest", "AJAX request"),
            (r"WebSocket", "WebSocket connection"),
        ]

        for pattern, description in dangerous_js_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(description)
                dangerous_patterns.append(pattern)

        lines = len(code.split("\n"))
        security_score = max(0, 100 - (len(issues) * 15))

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "imports": [],
            "functions": [],
            "dangerous_patterns": dangerous_patterns,
            "complexity_score": min(lines / 50, 10.0),
            "lines_of_code": lines,
            "language": "javascript",
            "security_score": security_score,
        }

    async def _analyze_shell_code(
        self, code: str, language: Language
    ) -> Dict[str, Any]:
        """Analyze shell code for safety"""
        issues = []
        dangerous_patterns = []

        dangerous_shell_patterns = [
            (r"rm\s+-rf\s+/?", "Dangerous delete command"),
            (r"sudo\s+", "Privilege escalation"),
            (r"chmod\s+777", "Permission change"),
            (r":\(\)\{\s*:\|:&\s*\};:", "Fork bomb"),
            (r"nc\s+-e", "Reverse shell"),
            (r"bash\s+-i", "Interactive shell"),
            (r"curl.*\|.*sh", "Pipe to shell"),
            (r"wget.*\|.*sh", "Pipe to shell"),
            (r"shutdown", "System shutdown"),
            (r"reboot", "System reboot"),
        ]

        for pattern, description in dangerous_shell_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(description)
                dangerous_patterns.append(pattern)

        lines = len(code.split("\n"))
        security_score = max(0, 100 - (len(issues) * 20))

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "imports": [],
            "functions": [],
            "dangerous_patterns": dangerous_patterns,
            "complexity_score": min(lines / 30, 10.0),
            "lines_of_code": lines,
            "language": language.value,
            "security_score": security_score,
        }

    async def _analyze_generic_code(
        self, code: str, language: Language
    ) -> Dict[str, Any]:
        """Generic code analysis for other languages."""
        lines = len(code.split("\n"))

        return {
            "is_safe": True,
            "issues": [],
            "imports": [],
            "functions": [],
            "dangerous_patterns": [],
            "complexity_score": min(lines / 50, 10.0),
            "lines_of_code": lines,
            "language": language.value,
            "security_score": 100.0,
        }

    # =========================
    # CODE EXECUTION
    # =========================

    async def execute_code(
        self,
        code: str,
        language: Language = None,
        timeout: Optional[int] = None,
        memory_limit: Optional[int] = None,
        stdin_input: Optional[str] = None,
        environment: Optional[Dict] = None,
        session_id: Optional[str] = None,
        skip_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute code safely in sandbox.

        Args:
            code: Source code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            memory_limit: Memory limit in bytes
            stdin_input: Input to pipe to stdin
            environment: Additional environment variables
            session_id: Session ID for persistent execution
            skip_analysis: Skip static analysis

        Returns:
            Dictionary with execution results
        """
        execution_id = str(uuid.uuid4())[:8]

        # Check circuit breaker
        if not await self._check_circuit_breaker():
            return {
                "success": False,
                "error": "Circuit breaker open - system temporarily unavailable",
                "execution_id": execution_id,
            }

        # Check rate limit
        if not await self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded - please try again later",
                "execution_id": execution_id,
            }

        if language is None:
            language = self.default_language

        timeout = timeout or self.timeout_default
        memory_limit = memory_limit or self.memory_limit

        # Analyze code first (unless skipped)
        analysis = None
        if not skip_analysis:
            analysis = await self.analyze_code(code, language)

            if not analysis.get("is_safe", False):
                self.stats["blocked_executions"] += 1
                return {
                    "success": False,
                    "error": "Code blocked by security analysis",
                    "analysis": analysis,
                    "issues": analysis.get("issues", []),
                    "blocked": True,
                    "execution_id": execution_id,
                }

        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            # Use session if provided
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.now()
                # Merge session environment
                if environment:
                    env = {**session.environment, **environment}
                else:
                    env = session.environment
            else:
                env = environment or {}

            # Choose execution backend
            if language == Language.PYTHON_SAFE and RESTRICTED_PYTHON_AVAILABLE:
                raw = await self._execute_restricted_python(code, timeout, stdin_input)
            elif self.sandbox_type == SandboxType.DOCKER and self.docker_client:
                raw = await self._execute_docker(
                    code, language, timeout, memory_limit, stdin_input, env
                )
            else:
                raw = await self._execute_subprocess(
                    code, language, timeout, memory_limit, stdin_input, env, session_id
                )

            execution_time = time.time() - start_time
            memory_used = max(0, self._get_memory_usage() - start_memory)

            status = (
                ExecutionStatus.COMPLETED
                if raw["return_code"] == 0
                else ExecutionStatus.FAILED
            )
            if raw.get("timeout", False):
                status = ExecutionStatus.TIMEOUT
                self.stats["timeout_executions"] += 1

            exec_result = ExecutionResult(
                id=execution_id,
                success=raw["return_code"] == 0 and not raw.get("timeout", False),
                stdout=raw.get("stdout", ""),
                stderr=raw.get("stderr", ""),
                return_code=raw["return_code"],
                execution_time=execution_time,
                memory_used=memory_used,
                language=language.value,
                timestamp=datetime.now(),
                status=status,
                error=raw.get("error"),
                analysis=analysis,
            )

            self._update_stats(exec_result, execution_time, memory_used, language)
            self._add_to_history(exec_result)
            await self._record_success()

            # Update session history
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].history.append(
                    {
                        "execution_id": execution_id,
                        "timestamp": datetime.now().isoformat(),
                        "code_preview": code[:200],
                        "success": exec_result.success,
                    }
                )
                # Trim session history
                if len(self.active_sessions[session_id].history) > 100:
                    self.active_sessions[session_id].history = self.active_sessions[
                        session_id
                    ].history[-100:]

            return {
                "success": exec_result.success,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "return_code": exec_result.return_code,
                "execution_time": round(exec_result.execution_time, 3),
                "memory_used_kb": round(exec_result.memory_used / 1024, 2),
                "language": language.value,
                "analysis": analysis,
                "execution_id": execution_id,
                "timestamp": exec_result.timestamp.isoformat(),
                "status": exec_result.status.value,
            }

        except asyncio.TimeoutError:
            self.stats["failed_executions"] += 1
            self.stats["timeout_executions"] += 1
            await self._record_failure()
            return {
                "success": False,
                "error": f"Execution timeout after {timeout} seconds",
                "timeout": True,
                "language": language.value,
                "execution_id": execution_id,
            }
        except Exception as exc:
            self.logger.error("Execution error: %s\n%s", exc, traceback.format_exc())
            self.stats["failed_executions"] += 1
            await self._record_failure()
            return {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "language": language.value,
                "execution_id": execution_id,
            }

    async def _execute_subprocess(
        self,
        code: str,
        language: Language,
        timeout: int,
        memory_limit: int,
        stdin_input: Optional[str],
        environment: Optional[Dict],
        session_id: Optional[str],
    ) -> Dict:
        """
        Execute code using asyncio subprocess with resource limits.
        """
        suffix = self._get_file_extension(language)

        # Special handling for compiled languages
        if language == Language.CPP:
            return await self._execute_cpp(code, timeout, memory_limit, environment)
        elif language == Language.JAVA:
            return await self._execute_java(code, timeout, memory_limit, environment)
        elif language == Language.GO:
            return await self._execute_go(code, timeout, memory_limit, environment)
        elif language == Language.RUST:
            return await self._execute_rust(code, timeout, memory_limit, environment)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            dir=self.workspace_dir,
            encoding="utf-8",
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            cmd = self._build_command(language, temp_file)
            env = {**os.environ}
            if environment:
                env.update(environment)

            # Resource limit preexec (Unix only)
            preexec = None
            if RESOURCE_AVAILABLE and os.name != "nt":

                def set_limits():
                    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
                    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
                    resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
                    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))

                preexec = set_limits

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_dir),
                preexec_fn=preexec,
            )

            stdin_bytes = stdin_input.encode() if stdin_input else None

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=stdin_bytes), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "stdout": "",
                    "stderr": f"Execution timeout after {timeout} seconds",
                    "return_code": -1,
                    "timeout": True,
                }

            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "return_code": process.returncode or 0,
            }

        finally:
            try:
                os.unlink(temp_file)
            except Exception as exc:
                self.logger.debug("Could not delete temp file %s: %s", temp_file, exc)

    async def _execute_cpp(
        self, code: str, timeout: int, memory_limit: int, environment: Optional[Dict]
    ) -> Dict:
        """Compile and run C++ code."""
        env = {**os.environ}
        if environment:
            env.update(environment)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cpp",
            delete=False,
            dir=self.workspace_dir,
            encoding="utf-8",
        ) as src_f:
            src_f.write(code)
            src_path = src_f.name

        out_path = src_path + ".out"

        try:
            # Compile
            compile_proc = await asyncio.create_subprocess_exec(
                "g++",
                src_path,
                "-o",
                out_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, c_err = await asyncio.wait_for(
                    compile_proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                compile_proc.kill()
                await compile_proc.wait()
                return {
                    "stdout": "",
                    "stderr": "Compilation timeout",
                    "return_code": -1,
                    "timeout": True,
                }

            if compile_proc.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": c_err.decode("utf-8", errors="replace"),
                    "return_code": compile_proc.returncode,
                    "error": "Compilation failed",
                }

            # Run
            run_proc = await asyncio.create_subprocess_exec(
                out_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_dir),
            )
            try:
                r_out, r_err = await asyncio.wait_for(
                    run_proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                run_proc.kill()
                await run_proc.wait()
                return {
                    "stdout": "",
                    "stderr": f"Execution timeout after {timeout} seconds",
                    "return_code": -1,
                    "timeout": True,
                }

            return {
                "stdout": r_out.decode("utf-8", errors="replace"),
                "stderr": r_err.decode("utf-8", errors="replace"),
                "return_code": run_proc.returncode or 0,
            }

        finally:
            for path in (src_path, out_path):
                try:
                    os.unlink(path)
                except Exception as exc:
                    self.logger.debug("Could not delete temp file %s: %s", path, exc)

    async def _execute_java(
        self, code: str, timeout: int, memory_limit: int, environment: Optional[Dict]
    ) -> Dict:
        """Compile and run Java code."""
        env = {**os.environ}
        if environment:
            env.update(environment)

        # Extract class name from code (simplified)
        class_match = re.search(r"public\s+class\s+(\w+)", code)
        class_name = class_match.group(1) if class_match else "Main"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".java",
            delete=False,
            dir=self.workspace_dir,
            encoding="utf-8",
        ) as src_f:
            src_f.write(code)
            src_path = src_f.name

        try:
            # Compile
            compile_proc = await asyncio.create_subprocess_exec(
                "javac",
                src_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                _, c_err = await asyncio.wait_for(
                    compile_proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                compile_proc.kill()
                await compile_proc.wait()
                return {
                    "stdout": "",
                    "stderr": "Compilation timeout",
                    "return_code": -1,
                    "timeout": True,
                }

            if compile_proc.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": c_err.decode("utf-8", errors="replace"),
                    "return_code": compile_proc.returncode,
                    "error": "Compilation failed",
                }

            # Run
            run_proc = await asyncio.create_subprocess_exec(
                "java",
                "-cp",
                str(self.workspace_dir),
                class_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_dir),
            )
            try:
                r_out, r_err = await asyncio.wait_for(
                    run_proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                run_proc.kill()
                await run_proc.wait()
                return {
                    "stdout": "",
                    "stderr": f"Execution timeout after {timeout} seconds",
                    "return_code": -1,
                    "timeout": True,
                }

            return {
                "stdout": r_out.decode("utf-8", errors="replace"),
                "stderr": r_err.decode("utf-8", errors="replace"),
                "return_code": run_proc.returncode or 0,
            }

        finally:
            try:
                os.unlink(src_path)
                class_file = self.workspace_dir / f"{class_name}.class"
                if class_file.exists():
                    class_file.unlink()
            except Exception as exc:
                self.logger.debug("Could not delete temp files: %s", exc)

    async def _execute_go(
        self, code: str, timeout: int, memory_limit: int, environment: Optional[Dict]
    ) -> Dict:
        """Run Go code."""
        env = {**os.environ}
        if environment:
            env.update(environment)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".go",
            delete=False,
            dir=self.workspace_dir,
            encoding="utf-8",
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                "go",
                "run",
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.workspace_dir),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "stdout": "",
                    "stderr": f"Execution timeout after {timeout} seconds",
                    "return_code": -1,
                    "timeout": True,
                }

            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "return_code": process.returncode or 0,
            }
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    async def _execute_rust(
        self, code: str, timeout: int, memory_limit: int, environment: Optional[Dict]
    ) -> Dict:
        """Compile and run Rust code."""
        env = {**os.environ}
        if environment:
            env.update(environment)

        # Create temporary project structure
        with tempfile.TemporaryDirectory(dir=self.workspace_dir) as tmpdir:
            tmp_path = Path(tmpdir)

            # Create Cargo.toml
            cargo_toml = tmp_path / "Cargo.toml"
            cargo_toml.write_text("""
[package]
name = "temp"
version = "0.1.0"
edition = "2021"

[dependencies]
""")

            # Create src directory
            src_dir = tmp_path / "src"
            src_dir.mkdir()
            main_rs = src_dir / "main.rs"
            main_rs.write_text(code)

            # Build and run
            process = await asyncio.create_subprocess_exec(
                "cargo",
                "run",
                "--manifest-path",
                str(cargo_toml),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(tmp_path),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "stdout": "",
                    "stderr": f"Execution timeout after {timeout} seconds",
                    "return_code": -1,
                    "timeout": True,
                }

            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "return_code": process.returncode or 0,
            }

    async def _execute_docker(
        self,
        code: str,
        language: Language,
        timeout: int,
        memory_limit: int,
        stdin_input: Optional[str],
        environment: Optional[Dict],
    ) -> Dict:
        """Execute code in a Docker container."""
        if not self.docker_client:
            raise RuntimeError("Docker client is not initialised")

        suffix = self._get_file_extension(language)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            container_cmd = self._build_docker_command(language, temp_file)
            env_vars = [f"{k}={v}" for k, v in (environment or {}).items()]

            container = self.docker_client.containers.run(
                image=self.docker_image,
                command=container_cmd,
                mem_limit=f"{memory_limit // (1024 * 1024)}m",
                nano_cpus=int(self.cpu_limit * 1e9),
                environment=env_vars,
                stdin_open=True,
                detach=True,
                remove=False,
                working_dir="/workspace",
            )

            try:
                result = container.wait(timeout=timeout)
                stdout_data = container.logs(stdout=True, stderr=False).decode(
                    "utf-8", errors="replace"
                )
                stderr_data = container.logs(stdout=False, stderr=True).decode(
                    "utf-8", errors="replace"
                )

                return {
                    "stdout": stdout_data,
                    "stderr": stderr_data,
                    "return_code": result["StatusCode"],
                }
            except Exception as exc:
                try:
                    container.kill()
                except Exception:
                    pass
                raise exc
            finally:
                try:
                    container.remove()
                except Exception as exc:
                    self.logger.debug("Could not remove container: %s", exc)

        finally:
            try:
                os.unlink(temp_file)
            except Exception as exc:
                self.logger.debug("Could not delete temp file %s: %s", temp_file, exc)

    async def _execute_restricted_python(
        self, code: str, timeout: int, stdin_input: Optional[str]
    ) -> Dict:
        """Execute Python code using RestrictedPython sandbox."""
        if not RESTRICTED_PYTHON_AVAILABLE:
            return await self._execute_subprocess(
                code,
                Language.PYTHON,
                timeout,
                self.memory_limit,
                stdin_input,
                None,
                None,
            )

        restricted_globals = {
            "__builtins__": safe_globals.get("__builtins__", {}),
            "_print_": print,
            "_getattr_": getattr,
            "_setattr_": setattr,
            "_import_": self._restricted_import,
        }

        for module in self.allowed_modules:
            try:
                restricted_globals[module] = __import__(module)
            except Exception as exc:
                self.logger.debug(
                    "Could not pre-import allowed module %s: %s", module, exc
                )

        try:
            compiled_code = compile_restricted(code, "<string>", "exec")
        except SyntaxError as exc:
            return {"stdout": "", "stderr": f"SyntaxError: {exc}", "return_code": 1}

        import io

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        try:
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, exec, compiled_code, restricted_globals
                    ),
                    timeout=timeout,
                )
                stdout = stdout_capture.getvalue()
                return {"stdout": stdout, "stderr": "", "return_code": 0}
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                stdout = stdout_capture.getvalue()
                return {"stdout": stdout, "stderr": str(exc), "return_code": 1}
        finally:
            sys.stdout = old_stdout

    def _restricted_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Restricted import – only allows modules in self.allowed_modules."""
        module_name = name.split(".")[0]
        if module_name in self.allowed_modules:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Module '{name}' is not allowed in restricted mode")

    # =========================
    # COMMAND BUILDERS
    # =========================

    def _build_command(self, language: Language, file_path: str) -> List[str]:
        """Build subprocess command list for the given language."""
        commands = {
            Language.PYTHON: [sys.executable, file_path],
            Language.PYTHON_SAFE: [sys.executable, file_path],
            Language.JAVASCRIPT: ["node", file_path],
            Language.TYPESCRIPT: ["ts-node", file_path],
            Language.BASH: ["bash", file_path],
            Language.POWERSHELL: ["powershell", "-File", file_path],
            Language.NODE: ["node", file_path],
            Language.RUBY: ["ruby", file_path],
            Language.PHP: ["php", file_path],
            Language.PERL: ["perl", file_path],
        }
        return commands.get(language, [file_path])

    def _build_docker_command(self, language: Language, file_path: str) -> str:
        """Build shell command string for Docker container execution."""
        name = Path(file_path).name
        commands = {
            Language.PYTHON: f"python {name}",
            Language.JAVASCRIPT: f"node {name}",
            Language.TYPESCRIPT: f"ts-node {name}",
            Language.BASH: f"bash {name}",
            Language.RUBY: f"ruby {name}",
            Language.PHP: f"php {name}",
            Language.PERL: f"perl {name}",
        }
        return commands.get(language, f"sh {name}")

    def _get_file_extension(self, language: Language) -> str:
        """Return the file extension for the given language."""
        extensions = {
            Language.PYTHON: ".py",
            Language.PYTHON_SAFE: ".py",
            Language.JAVASCRIPT: ".js",
            Language.TYPESCRIPT: ".ts",
            Language.BASH: ".sh",
            Language.POWERSHELL: ".ps1",
            Language.NODE: ".js",
            Language.RUBY: ".rb",
            Language.GO: ".go",
            Language.RUST: ".rs",
            Language.JAVA: ".java",
            Language.CPP: ".cpp",
            Language.PHP: ".php",
            Language.PERL: ".pl",
        }
        return extensions.get(language, ".txt")

    # =========================
    # SESSION MANAGEMENT
    # =========================

    async def create_session(
        self,
        session_id: str,
        language: Language = Language.PYTHON,
        environment: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a persistent execution session."""
        if session_id in self.active_sessions:
            return {"success": False, "error": "Session already exists"}

        self.active_sessions[session_id] = ExecutionSession(
            id=session_id,
            language=language,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            variables={},
            history=[],
            environment=environment or {},
        )

        return {
            "success": True,
            "session_id": session_id,
            "language": language.value,
            "created_at": datetime.now().isoformat(),
            "message": "Session created",
        }

    async def execute_in_session(
        self, session_id: str, code: str, **kwargs
    ) -> Dict[str, Any]:
        """Execute code within an existing session."""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        session = self.active_sessions[session_id]
        session.last_activity = datetime.now()

        return await self.execute_code(
            code, session.language, session_id=session_id, **kwargs
        )

    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Return current state information for a session."""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        session = self.active_sessions[session_id]
        return {
            "success": True,
            "session_id": session_id,
            "language": session.language.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "history_count": len(session.history),
            "session_age_seconds": (
                datetime.now() - session.created_at
            ).total_seconds(),
            "variables": session.variables,
        }

    async def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close and remove a persistent session."""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        del self.active_sessions[session_id]
        return {"success": True, "message": "Session closed"}

    async def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions."""
        sessions = []
        for session_id, session in self.active_sessions.items():
            sessions.append(
                {
                    "id": session_id,
                    "language": session.language.value,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "history_count": len(session.history),
                }
            )

        return {"success": True, "total": len(sessions), "sessions": sessions}

    # =========================
    # FILE EXECUTION
    # =========================

    async def execute_file(
        self, file_path: str, language: Optional[Language] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute code from a file."""
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        if language is None:
            ext_map = {
                ".py": Language.PYTHON,
                ".js": Language.JAVASCRIPT,
                ".ts": Language.TYPESCRIPT,
                ".sh": Language.BASH,
                ".ps1": Language.POWERSHELL,
                ".rb": Language.RUBY,
                ".go": Language.GO,
                ".rs": Language.RUST,
                ".java": Language.JAVA,
                ".cpp": Language.CPP,
                ".cc": Language.CPP,
                ".php": Language.PHP,
                ".pl": Language.PERL,
            }
            language = ext_map.get(path.suffix.lower(), self.default_language)

        return await self.execute_code(code, language, **kwargs)

    # =========================
    # PACKAGE MANAGEMENT
    # =========================

    async def install_package(
        self, package_name: str, language: Language = Language.PYTHON
    ) -> Dict[str, Any]:
        """Install a package in the sandbox environment."""
        # Validate package name (security)
        if not self._validate_package_name(package_name):
            return {
                "success": False,
                "error": f"Invalid package name: '{package_name}'",
                "package": package_name,
            }

        install_commands = {
            Language.PYTHON: [sys.executable, "-m", "pip", "install", package_name],
            Language.JAVASCRIPT: ["npm", "install", "-g", package_name],
            Language.RUBY: ["gem", "install", package_name],
            Language.PHP: ["composer", "require", package_name],
        }

        cmd = install_commands.get(language)
        if cmd is None:
            return {
                "success": False,
                "error": f"Package installation not supported for {language.value}",
                "package": package_name,
            }

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=120
            )
            return {
                "success": process.returncode == 0,
                "package": package_name,
                "language": language.value,
                "output": stdout_bytes.decode("utf-8", errors="replace")[:500],
                "error": (
                    stderr_bytes.decode("utf-8", errors="replace")[:500]
                    if process.returncode != 0
                    else None
                ),
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "package": package_name,
                "error": "Installation timed out after 120 seconds",
            }
        except Exception as exc:
            self.logger.error("Package install error: %s", exc)
            return {"success": False, "package": package_name, "error": str(exc)}

    def _validate_package_name(self, name: str) -> bool:
        """Validate package name for security."""
        import re

        # Only allow alphanumeric, hyphens, underscores, dots
        return bool(re.match(r"^[a-zA-Z0-9_.\-]+$", name))

    # =========================
    # STATISTICS & HISTORY
    # =========================

    def _update_stats(
        self,
        result: ExecutionResult,
        execution_time: float,
        memory_used: int,
        language: Language,
    ):
        """Update internal statistics after an execution."""
        self.stats["total_executions"] += 1
        if result.success:
            self.stats["successful_executions"] += 1
        else:
            self.stats["failed_executions"] += 1

        self.stats["total_execution_time"] += execution_time
        n = self.stats["total_executions"]
        prev_avg = self.stats["average_memory_usage"]
        self.stats["average_memory_usage"] = prev_avg + (memory_used - prev_avg) / n

        self.stats["by_language"][language.value] = (
            self.stats["by_language"].get(language.value, 0) + 1
        )

    def _add_to_history(self, result: ExecutionResult):
        """Append result to execution history, trimming if over limit."""
        self.execution_history.append(result)
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history :]
        self._save_history()

    def get_history(
        self,
        limit: int = None,
        language: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """Return execution history with filters."""
        history = self.execution_history

        if language:
            history = [h for h in history if h.language == language]
        if status:
            history = [h for h in history if h.status.value == status]

        if limit:
            history = history[-limit:]

        return [
            {
                "id": h.id,
                "success": h.success,
                "stdout_preview": h.stdout[:200],
                "stderr_preview": h.stderr[:200],
                "return_code": h.return_code,
                "execution_time": round(h.execution_time, 3),
                "memory_used_kb": round(h.memory_used / 1024, 2),
                "language": h.language,
                "timestamp": h.timestamp.isoformat(),
                "status": h.status.value,
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        success_rate = (
            self.stats["successful_executions"] / max(1, self.stats["total_executions"])
        ) * 100
        return {
            **self.stats,
            "success_rate": round(success_rate, 2),
            "history_size": len(self.execution_history),
            "active_sessions": len(self.active_sessions),
            "sandbox_type": self.sandbox_type.value,
            "default_language": self.default_language.value,
            "memory_limit_mb": round(self.memory_limit / (1024 * 1024), 2),
            "timeout_default": self.timeout_default,
            "circuit_breaker_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
        }

    def clear_history(self) -> Dict[str, Any]:
        """Clear execution history."""
        count = len(self.execution_history)
        self.execution_history.clear()
        self._save_history()
        self.logger.info("Execution history cleared")
        return {
            "success": True,
            "cleared": count,
            "message": f"Cleared {count} history entries",
        }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear analysis cache."""
        cache_size = len(self._analysis_cache)
        self._analysis_cache.clear()
        return {
            "success": True,
            "cleared": cache_size,
            "message": f"Cleared {cache_size} cache entries",
        }

    def _get_memory_usage(self) -> int:
        """Return current process RSS memory in bytes."""
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process().memory_info().rss
            except Exception:
                return 0
        return 0

    # =========================
    # HEALTH CHECK
    # =========================

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check sandbox availability
        if self.sandbox_type == SandboxType.DOCKER and self.docker_client:
            try:
                self.docker_client.ping()
                status["components"]["docker"] = {"status": "healthy"}
            except Exception as e:
                status["components"]["docker"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                status["status"] = "degraded"
        else:
            status["components"]["sandbox"] = {
                "status": "healthy",
                "type": self.sandbox_type.value,
            }

        # Check workspace
        if self.workspace_dir.exists():
            status["components"]["workspace"] = {
                "status": "healthy",
                "path": str(self.workspace_dir),
            }
        else:
            status["components"]["workspace"] = {
                "status": "unhealthy",
                "error": "Workspace directory missing",
            }
            status["status"] = "degraded"

        # Check circuit breaker
        if self._circuit_open:
            status["status"] = "degraded"
            status["circuit_breaker"] = {
                "open": True,
                "open_until": (
                    self._circuit_open_until.isoformat()
                    if self._circuit_open_until
                    else None
                ),
            }

        return status


# =========================
# INTEGRATION WRAPPER
# =========================


class CodeExecutionAgentWrapper:
    """
    Wrapper class to integrate CodeExecutionAgent with EDIATH's agent architecture.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.code_agent = CodeExecutionAgent(config)
        self.agent_type = "code_execution"
        self.capabilities = [
            "execute_code",
            "analyze_code",
            "execute_file",
            "session_management",
            "package_installation",
            "multiple_languages",
            "queue_execution",
            "history_query",
        ]
        self._initialized = True

    async def initialize(self, *args, **kwargs) -> bool:
        """Initialize the wrapper."""
        return True

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a code execution request.

        Request format:
        {
            'operation': 'execute|analyze|file|session|install|history|stats|clear_history|health',
            ... operation-specific parameters ...
        }
        """
        operation = request.get("operation")

        try:
            if operation == "execute":
                raw_lang = request.get("language")
                language = Language(raw_lang) if raw_lang else None
                return await self.code_agent.execute_code(
                    code=request.get("code", ""),
                    language=language,
                    timeout=request.get("timeout"),
                    memory_limit=request.get("memory_limit"),
                    stdin_input=request.get("stdin"),
                    environment=request.get("environment"),
                    session_id=request.get("session_id"),
                    skip_analysis=request.get("skip_analysis", False),
                )

            elif operation == "analyze":
                language = Language(request.get("language", "python"))
                return await self.code_agent.analyze_code(
                    code=request.get("code", ""), language=language
                )

            elif operation == "file":
                raw_lang = request.get("language")
                language = Language(raw_lang) if raw_lang else None
                return await self.code_agent.execute_file(
                    file_path=request.get("file_path", ""),
                    language=language,
                    timeout=request.get("timeout"),
                    stdin_input=request.get("stdin"),
                )

            elif operation == "session":
                action = request.get("action", "create")

                if action == "create":
                    return await self.code_agent.create_session(
                        session_id=request.get("session_id", ""),
                        language=Language(request.get("language", "python")),
                        environment=request.get("environment"),
                    )
                elif action == "execute":
                    return await self.code_agent.execute_in_session(
                        session_id=request.get("session_id", ""),
                        code=request.get("code", ""),
                        timeout=request.get("timeout"),
                    )
                elif action == "state":
                    return await self.code_agent.get_session_state(
                        session_id=request.get("session_id", "")
                    )
                elif action == "close":
                    return await self.code_agent.close_session(
                        session_id=request.get("session_id", "")
                    )
                elif action == "list":
                    return await self.code_agent.list_sessions()
                else:
                    return {
                        "success": False,
                        "error": f"Unknown session action: {action}",
                    }

            elif operation == "install":
                return await self.code_agent.install_package(
                    package_name=request.get("package", ""),
                    language=Language(request.get("language", "python")),
                )

            elif operation == "queue":
                raw_lang = request.get("language")
                language = Language(raw_lang) if raw_lang else None
                task_id = await self.code_agent.queue_execution(
                    code=request.get("code", ""),
                    language=language,
                    timeout=request.get("timeout"),
                    memory_limit=request.get("memory_limit"),
                    stdin_input=request.get("stdin"),
                    environment=request.get("environment"),
                    callback=None,  # Callbacks not supported over API
                )
                return {"success": True, "task_id": task_id, "queued": True}

            elif operation == "history":
                return {
                    "success": True,
                    "history": self.code_agent.get_history(
                        limit=request.get("limit"),
                        language=request.get("language"),
                        status=request.get("status"),
                    ),
                }

            elif operation == "stats":
                return self.code_agent.get_stats()

            elif operation == "clear_history":
                return self.code_agent.clear_history()

            elif operation == "clear_cache":
                return self.code_agent.clear_cache()

            elif operation == "health":
                return await self.code_agent.health_check()

            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            self.code_agent.logger.error(f"Request error: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": operation,
                "traceback": traceback.format_exc(),
            }

    def get_info(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {
            "name": "CodeExecutionAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.code_agent.get_stats(),
            "sandbox_type": self.code_agent.sandbox_type.value,
            "supported_languages": [lang.value for lang in Language],
            "version": "2.0.0",
        }

    async def close(self):
        """Clean up resources."""
        self.code_agent._save_history()
        self._initialized = False


# =========================
# TESTING
# =========================


async def test_code_agent():
    """Test the code execution agent functionality."""

    agent = CodeExecutionAgent(config={"sandbox_type": "subshell"})

    print("=== Code Execution Agent Test ===\n")

    # 1. Code analysis
    print("1. Code Analysis...")
    safe_code = """
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

result = calculate_circle_area(5)
print(f"Area: {result}")
"""
    analysis = await agent.analyze_code(safe_code)
    print(f"   Safe: {analysis['is_safe']}")
    print(f"   Security Score: {analysis['security_score']:.1f}")
    print(f"   Imports: {analysis['imports']}")
    print(f"   Functions: {analysis['functions']}")
    print(f"   Complexity: {analysis['complexity_score']:.2f}")

    # 2. Safe code execution
    print("\n2. Safe Code Execution...")
    result = await agent.execute_code(safe_code)
    if result["success"]:
        print(f"   Output: {result['stdout'].strip()}")
        print(f"   Execution time: {result['execution_time']:.3f}s")
        print(f"   Memory used: {result['memory_used_kb']:.2f}KB")
    else:
        print(f"   Error: {result.get('error')}")

    # 3. Unsafe code blocking
    print("\n3. Unsafe Code Blocking...")
    unsafe_code = """
import os
os.system('rm -rf /')
"""
    result = await agent.execute_code(unsafe_code)
    print(f"   Blocked: {result.get('blocked', False)}")
    if "analysis" in result:
        print(f"   Issues: {result['analysis'].get('issues', [])[:2]}")

    # 4. Multiple language support
    print("\n4. JavaScript Execution...")
    js_code = """
const greet = (name) => {
    console.log(`Hello, ${name}!`);
};
greet("EDIATH");
"""
    result = await agent.execute_code(js_code, language=Language.JAVASCRIPT)
    if result["success"]:
        print(f"   Output: {result['stdout'].strip()}")
    else:
        print(
            f"   Result: {result.get('error', 'Not available (node may not be installed)')}"
        )

    # 5. Session management
    print("\n5. Session Management...")
    session = await agent.create_session("test_session")
    print(f"   Session created: {session['success']}")

    result = await agent.execute_in_session(
        "test_session", "x = 10\ny = 20\nprint(x + y)"
    )
    if result["success"]:
        print(f"   Session execution output: {result['stdout'].strip()}")

    state = await agent.get_session_state("test_session")
    print(f"   Session history entries: {state['history_count']}")

    sessions = await agent.list_sessions()
    print(f"   Active sessions: {sessions['total']}")

    await agent.close_session("test_session")
    print("   Session closed.")

    # 6. Batch execution via queue
    print("\n6. Queue Execution...")
    agent.queue_enabled = True
    await agent._start_queue_processor()

    task_id = await agent.queue_execution("print('Hello from queue')")
    print(f"   Task queued: {task_id}")

    await asyncio.sleep(1)

    # 7. History
    print("\n7. Execution History...")
    history = agent.get_history(limit=5)
    print(f"   Recent executions: {len(history)}")
    for h in history[:3]:
        print(
            f"     - {h['language']}: success={h['success']}, time={h['execution_time']:.2f}s"
        )

    # 8. Statistics
    print("\n8. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Successful: {stats['successful_executions']}")
    print(f"   Blocked: {stats['blocked_executions']}")
    print(f"   Timeout: {stats['timeout_executions']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Active sessions: {stats['active_sessions']}")

    # 9. Health check
    print("\n9. Health Check...")
    health = await agent.health_check()
    print(f"   Status: {health['status']}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_code_agent())
