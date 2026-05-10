"""
Advanced Code Runner - Production-Grade Multi-Language Execution
Supports Python, JavaScript, C, C++, Java, Go, Ruby, PHP, Rust, and more
Windows/Linux/Mac cross-platform compatible
"""

import uuid
import asyncio
import subprocess
import time
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Cross-platform resource management
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Memory monitoring disabled.")

# Unix-only module
try:
    import resource

    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    # Windows compatibility

import traceback

from ..utils.logger import logger


class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"
    MEMORY_ERROR = "memory_error"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED = "unsupported"


class LanguageConfig(Enum):
    """Configuration for supported languages"""

    PYTHON = {
        "ext": ".py",
        "cmd": ["python3", "{file}"],
        "compile": None,
        "version_cmd": ["python3", "--version"],
    }
    PYTHON2 = {
        "ext": ".py",
        "cmd": ["python2", "{file}"],
        "compile": None,
        "version_cmd": ["python2", "--version"],
    }
    JAVASCRIPT = {
        "ext": ".js",
        "cmd": ["node", "{file}"],
        "compile": None,
        "version_cmd": ["node", "--version"],
    }
    TYPESCRIPT = {
        "ext": ".ts",
        "cmd": ["ts-node", "{file}"],
        "compile": ["tsc", "{file}"],
        "version_cmd": ["ts-node", "--version"],
    }
    C = {
        "ext": ".c",
        "cmd": ["./{exe}"],
        "compile": ["gcc", "{file}", "-o", "{exe}", "-lm"],
        "version_cmd": ["gcc", "--version"],
    }
    CPP = {
        "ext": ".cpp",
        "cmd": ["./{exe}"],
        "compile": ["g++", "{file}", "-o", "{exe}", "-std=c++17", "-O2"],
        "version_cmd": ["g++", "--version"],
    }
    JAVA = {
        "ext": ".java",
        "cmd": ["java", "{class_name}"],
        "compile": ["javac", "{file}"],
        "version_cmd": ["javac", "-version"],
    }
    GO = {
        "ext": ".go",
        "cmd": ["./{exe}"],
        "compile": ["go", "build", "-o", "{exe}", "{file}"],
        "version_cmd": ["go", "version"],
    }
    RUBY = {
        "ext": ".rb",
        "cmd": ["ruby", "{file}"],
        "compile": None,
        "version_cmd": ["ruby", "--version"],
    }
    PHP = {
        "ext": ".php",
        "cmd": ["php", "{file}"],
        "compile": None,
        "version_cmd": ["php", "--version"],
    }
    RUST = {
        "ext": ".rs",
        "cmd": ["./{exe}"],
        "compile": ["rustc", "{file}", "-o", "{exe}"],
        "version_cmd": ["rustc", "--version"],
    }
    CSHARP = {
        "ext": ".cs",
        "cmd": ["dotnet", "run"],
        "compile": ["dotnet", "new", "console", "-n", "{project_name}"],
        "version_cmd": ["dotnet", "--version"],
    }
    SWIFT = {
        "ext": ".swift",
        "cmd": ["./{exe}"],
        "compile": ["swiftc", "{file}", "-o", "{exe}"],
        "version_cmd": ["swiftc", "--version"],
    }
    KOTLIN = {
        "ext": ".kt",
        "cmd": ["kotlin", "{class_name}Kt"],
        "compile": ["kotlinc", "{file}", "-include-runtime", "-d", "{exe}.jar"],
        "version_cmd": ["kotlinc", "-version"],
    }
    R = {
        "ext": ".r",
        "cmd": ["Rscript", "{file}"],
        "compile": None,
        "version_cmd": ["Rscript", "--version"],
    }
    PERL = {
        "ext": ".pl",
        "cmd": ["perl", "{file}"],
        "compile": None,
        "version_cmd": ["perl", "--version"],
    }
    LUA = {
        "ext": ".lua",
        "cmd": ["lua", "{file}"],
        "compile": None,
        "version_cmd": ["lua", "-v"],
    }
    BASH = {
        "ext": ".sh",
        "cmd": ["bash", "{file}"],
        "compile": None,
        "version_cmd": ["bash", "--version"],
    }


@dataclass
class ExecutionResult:
    """Structured execution result"""

    status: ExecutionStatus
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    exit_code: int = 0
    file_id: str = ""
    language: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.status == ExecutionStatus.SUCCESS,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "memory_used_mb": self.memory_used_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "exit_code": self.exit_code,
            "language": self.language,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CodeRunnerConfig:
    """Configuration for code runner"""

    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 512  # MB
    max_output_size: int = 10 * 1024 * 1024  # 10MB
    enable_network: bool = False
    enable_file_system: bool = True
    enable_compilation: bool = True
    auto_cleanup: bool = True
    persist_code: bool = False
    log_executions: bool = True
    resource_limits: bool = True
    docker_sandbox: bool = False
    docker_image: str = "code-runner-sandbox:latest"


class CodeRunner:
    """
    Production-grade multi-language code runner with resource limits,
    sandboxing, and comprehensive error handling
    Cross-platform: Windows, Linux, macOS
    """

    def __init__(
        self,
        workspace: Optional[str] = "./workspace",
        config: Optional[CodeRunnerConfig] = None,
        allowed_languages: Optional[List[str]] = None,
    ):
        """
        Initialize the code runner with workspace and configuration

        Args:
            workspace: Workspace directory for code execution
            config: Configuration object
            allowed_languages: List of allowed languages (None = all supported)
        """
        self.config = config or CodeRunnerConfig()
        self.allowed_languages = allowed_languages or [
            lang.name.lower() for lang in LanguageConfig
        ]

        # Initialize workspace
        self._init_workspace(workspace)

        # Statistics
        self.execution_count = 0
        self.error_count = 0
        self.success_count = 0
        self.created_at = time.time()
        self.execution_history: List[ExecutionResult] = []

        # File tracking
        self.created_files: List[Path] = []
        self.created_dirs: List[Path] = []

        # Locks for async safety
        self._execution_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()

        # Check available languages
        self.available_languages = self._check_available_languages()

        # Log initialization
        logger.info(f"🚀 CodeRunner initialized: {self.workspace}")
        logger.info(f"   Platform: {sys.platform}")
        logger.info(f"   Available languages: {', '.join(self.available_languages)}")
        logger.info(f"   Max execution time: {self.config.max_execution_time}s")
        logger.info(f"   Max memory: {self.config.max_memory_mb}MB")
        if not PSUTIL_AVAILABLE:
            logger.warning("   psutil not installed - memory monitoring disabled")

    def _init_workspace(self, workspace: Optional[str]):
        """Initialize workspace with security checks"""
        try:
            if workspace is None:
                workspace = "./workspace"

            if not isinstance(workspace, str):
                raise ValueError("Workspace must be a string path")

            workspace = workspace.strip()
            self.workspace = Path(workspace).expanduser().resolve()

            # Create workspace directory
            self.workspace.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            self.temp_dir = self.workspace / "temp"
            self.temp_dir.mkdir(exist_ok=True)

            self.logs_dir = self.workspace / "logs"
            self.logs_dir.mkdir(exist_ok=True)

            # Security check
            if not self.workspace.exists() or not self.workspace.is_dir():
                raise RuntimeError("Workspace initialization failed")

            # Set resource limits for the process (Unix only)
            if (
                self.config.resource_limits
                and RESOURCE_AVAILABLE
                and sys.platform != "win32"
            ):
                try:
                    resource.setrlimit(
                        resource.RLIMIT_AS,
                        (
                            self.config.max_memory_mb * 1024 * 1024,
                            self.config.max_memory_mb * 1024 * 1024,
                        ),
                    )
                    logger.debug("Resource limits configured")
                except Exception as e:
                    logger.warning(f"Could not set memory limit: {e}")
            elif self.config.resource_limits and sys.platform == "win32":
                logger.debug("Resource limits skipped on Windows (use psutil instead)")

        except Exception as e:
            logger.error(f"Workspace init failed: {e}")
            raise RuntimeError(f"Workspace initialization failed: {e}")

    def _check_available_languages(self) -> List[str]:
        """Check which languages are available on the system"""
        available = []

        for lang in LanguageConfig:
            lang_name = lang.name.lower()
            config = lang.value

            # Check if language is allowed
            if lang_name not in [l.lower() for l in self.allowed_languages]:
                continue

            # Check if command exists
            cmd = config["version_cmd"]
            try:
                # On Windows, use where command for some executables
                if sys.platform == "win32" and cmd[0] in [
                    "gcc",
                    "g++",
                    "javac",
                    "go",
                    "rustc",
                ]:
                    result = subprocess.run(
                        ["where", cmd[0]], capture_output=True, timeout=5
                    )
                else:
                    result = subprocess.run(cmd, capture_output=True, timeout=5)

                if result.returncode == 0:
                    available.append(lang_name)
                    logger.debug(f"✓ {lang_name.upper()} available")
                else:
                    logger.debug(f"✗ {lang_name.upper()} not available")
            except Exception:
                logger.debug(f"✗ {lang_name.upper()} not available")

        return available

    def is_language_available(self, language: str) -> bool:
        """Check if a specific language is available"""
        return language.lower() in self.available_languages

    async def run(
        self,
        language: str,
        code: str,
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
        memory_limit: Optional[int] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute code safely with resource limits

        Args:
            language: Programming language (python, javascript, c, cpp, etc.)
            code: Source code to execute
            stdin: Standard input to provide to the program
            timeout: Execution timeout in seconds (overrides config)
            memory_limit: Memory limit in MB (overrides config)
            args: Command line arguments for the program
            env: Environment variables for the program

        Returns:
            Dictionary with execution results
        """

        async with self._execution_lock:
            start_time = time.time()
            file_id = uuid.uuid4().hex[:8]
            file_path = None
            exe_path = None
            process = None

            # Track memory usage
            process_info = None

            try:
                # Validate inputs
                validation_error = self._validate_inputs(language, code)
                if validation_error:
                    return validation_error

                language = language.lower().strip()
                lang_config = self._get_language_config(language)

                if not lang_config:
                    return {
                        "success": False,
                        "error": f"Unsupported language: {language}",
                        "status": ExecutionStatus.UNSUPPORTED.value,
                    }

                # Create code file
                file_path, exe_path, compile_cmd, run_cmd = (
                    await self._prepare_execution(
                        language, code, file_id, lang_config, args
                    )
                )

                # Compile if needed
                if compile_cmd:
                    compile_result = await self._compile(
                        compile_cmd,
                        lang_config,
                        timeout or self.config.max_execution_time,
                    )
                    if not compile_result["success"]:
                        return compile_result

                # Execute the code
                execution_timeout = timeout or self.config.max_execution_time
                memory_mb = memory_limit or self.config.max_memory_mb

                # Start process
                process = await asyncio.create_subprocess_exec(
                    *run_cmd,
                    cwd=self.workspace,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, **(env or {})},
                )

                # Get initial process info for monitoring
                if PSUTIL_AVAILABLE and process.pid:
                    try:
                        process_info = psutil.Process(process.pid)
                    except:
                        pass

                # Monitor memory usage
                memory_task = None
                if process_info and self.config.resource_limits and PSUTIL_AVAILABLE:
                    memory_task = asyncio.create_task(
                        self._monitor_memory(process_info, memory_mb)
                    )

                # Execute with timeout
                try:
                    out, err = await asyncio.wait_for(
                        process.communicate(input=stdin.encode() if stdin else None),
                        timeout=execution_timeout,
                    )

                    # Cancel memory monitoring
                    if memory_task:
                        memory_task.cancel()

                    output = out.decode("utf-8", errors="replace")
                    error = err.decode("utf-8", errors="replace")

                    # Truncate output if too large
                    if len(output) > self.config.max_output_size:
                        output = (
                            output[: self.config.max_output_size]
                            + "\n... (output truncated)"
                        )
                    if len(error) > self.config.max_output_size:
                        error = (
                            error[: self.config.max_output_size]
                            + "\n... (error truncated)"
                        )

                    execution_time = time.time() - start_time

                    # Get resource usage
                    memory_used = 0.0
                    cpu_usage = 0.0
                    if process_info and PSUTIL_AVAILABLE:
                        try:
                            memory_used = process_info.memory_info().rss / (1024 * 1024)
                            cpu_usage = process_info.cpu_percent()
                        except:
                            pass

                    # Determine status
                    status = (
                        ExecutionStatus.SUCCESS
                        if process.returncode == 0
                        else ExecutionStatus.RUNTIME_ERROR
                    )

                    result = ExecutionResult(
                        status=status,
                        output=output,
                        error=error if process.returncode != 0 else "",
                        execution_time=execution_time,
                        memory_used_mb=memory_used,
                        cpu_usage_percent=cpu_usage,
                        exit_code=process.returncode,
                        file_id=file_id,
                        language=language,
                    )

                    # Update statistics
                    self.execution_count += 1
                    if result.status == ExecutionStatus.SUCCESS:
                        self.success_count += 1
                    else:
                        self.error_count += 1

                    # Log execution
                    if self.config.log_executions:
                        await self._log_execution(result, code)

                    # Store in history
                    if self.config.log_executions:
                        self.execution_history.append(result)
                        if len(self.execution_history) > 1000:
                            self.execution_history = self.execution_history[-1000:]

                    return result.to_dict()

                except asyncio.TimeoutError:
                    # Kill process on timeout
                    if process:
                        try:
                            process.kill()
                            await process.wait()
                        except:
                            pass

                    return {
                        "success": False,
                        "error": f"Execution timeout after {execution_timeout} seconds",
                        "status": ExecutionStatus.TIMEOUT.value,
                        "execution_time": execution_timeout,
                        "exit_code": -1,
                    }

            except Exception as e:
                self.error_count += 1
                logger.error(f"Execution error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "status": ExecutionStatus.ERROR.value,
                    "execution_time": time.time() - start_time,
                }

            finally:
                # Cleanup files
                if self.config.auto_cleanup:
                    await self._cleanup_files(file_path, exe_path)

    def _validate_inputs(self, language: str, code: str) -> Optional[Dict]:
        """Validate inputs before execution"""
        if not language or not isinstance(language, str):
            return {
                "success": False,
                "error": "Invalid language",
                "status": ExecutionStatus.ERROR.value,
            }

        if code is None or not isinstance(code, str):
            return {
                "success": False,
                "error": "Invalid code",
                "status": ExecutionStatus.ERROR.value,
            }

        if len(code) > self.config.max_file_size:
            return {
                "success": False,
                "error": f"Code too large. Max size: {self.config.max_file_size / 1024 / 1024}MB",
                "status": ExecutionStatus.ERROR.value,
            }

        language = language.lower().strip()
        if language not in [l.lower() for l in self.allowed_languages]:
            return {
                "success": False,
                "error": f"Language '{language}' not allowed. Allowed: {', '.join(self.allowed_languages)}",
                "status": ExecutionStatus.PERMISSION_DENIED.value,
            }

        if language not in [lang.name.lower() for lang in LanguageConfig]:
            return {
                "success": False,
                "error": f"Language '{language}' not supported",
                "status": ExecutionStatus.UNSUPPORTED.value,
            }

        if not self.is_language_available(language):
            return {
                "success": False,
                "error": f"Language '{language}' is not available on this system. Please install required runtime.",
                "status": ExecutionStatus.UNSUPPORTED.value,
            }

        # Security checks
        dangerous_patterns = [
            "__import__('os')",
            "subprocess.",
            "eval(",
            "exec(",
            "open(",
            "file(",
            "__builtins__",
            "globals()",
            "locals()",
        ]

        if language == "python":
            for pattern in dangerous_patterns:
                if pattern in code:
                    logger.warning(f"Dangerous pattern detected: {pattern}")
                    # Allow but log - could block if needed

        return None

    def _get_language_config(self, language: str) -> Optional[Dict]:
        """Get configuration for a language"""
        for lang in LanguageConfig:
            if lang.name.lower() == language:
                return lang.value
        return None

    async def _prepare_execution(
        self,
        language: str,
        code: str,
        file_id: str,
        lang_config: Dict,
        args: Optional[List[str]] = None,
    ) -> Tuple[Path, Optional[Path], Optional[List[str]], List[str]]:
        """Prepare files and commands for execution"""

        try:
            # Create code file
            ext = lang_config.get("ext", ".txt")
            file_path = self.temp_dir / f"{file_id}{ext}"

            # Write code with proper encoding
            file_path.write_text(code, encoding="utf-8")
            self.created_files.append(file_path)
            logger.debug(f"Created code file: {file_path}")

            # Prepare executable path (for compiled languages)
            exe_path = None
            compile_cmd = None

            # Handle compilation if needed
            compile_config = lang_config.get("compile")
            if compile_config and self.config.enable_compilation:
                exe_path = self.temp_dir / file_id
                if sys.platform == "win32":
                    exe_path = self.temp_dir / f"{file_id}.exe"

                # Format compile command safely (avoid using 'class' as keyword)
                compile_cmd = []
                for cmd_template in compile_config:
                    try:
                        formatted_cmd = cmd_template.format(
                            file=str(file_path),
                            exe=str(exe_path),
                            class_name=file_id,
                            project_name=f"project_{file_id}",
                            filename=file_path.stem,
                        )
                        compile_cmd.append(formatted_cmd)
                    except KeyError as e:
                        # Handle missing format keys gracefully
                        logger.warning(f"Missing format key in compile command: {e}")
                        formatted_cmd = cmd_template
                        compile_cmd.append(formatted_cmd)
                    except Exception as e:
                        logger.error(f"Error formatting compile command: {e}")
                        compile_cmd.append(cmd_template)

            # Prepare run command
            run_cmd = []
            run_config = lang_config.get("cmd", [])

            for cmd_template in run_config:
                try:
                    # Prepare format arguments safely
                    format_args = {
                        "file": str(file_path),
                        "exe": str(exe_path) if exe_path else str(file_path),
                        "class_name": file_id,
                        "project_name": f"project_{file_id}",
                        "filename": file_path.stem,
                        "dirname": str(self.temp_dir),
                    }

                    # Only include exe in format if exe_path exists
                    if not exe_path:
                        format_args["exe"] = str(file_path)

                    formatted_cmd = cmd_template.format(**format_args)
                    run_cmd.append(formatted_cmd)

                except KeyError as e:
                    # Handle missing format keys by using basic replacement
                    logger.warning(f"Missing format key in run command: {e}")
                    formatted_cmd = cmd_template
                    # Simple replacement as fallback
                    formatted_cmd = formatted_cmd.replace("{file}", str(file_path))
                    if exe_path:
                        formatted_cmd = formatted_cmd.replace("{exe}", str(exe_path))
                    formatted_cmd = formatted_cmd.replace("{class_name}", file_id)
                    formatted_cmd = formatted_cmd.replace(
                        "{project_name}", f"project_{file_id}"
                    )
                    formatted_cmd = formatted_cmd.replace("{filename}", file_path.stem)
                    formatted_cmd = formatted_cmd.replace(
                        "{dirname}", str(self.temp_dir)
                    )
                    run_cmd.append(formatted_cmd)

                except Exception as e:
                    logger.error(f"Error formatting run command: {e}")
                    run_cmd.append(cmd_template)

            # Add arguments if provided
            if args:
                run_cmd.extend(args)

            # Validate commands
            if not run_cmd:
                raise ValueError(f"No run command configured for language: {language}")

            # Log preparation
            logger.debug(
                f"Prepared execution - Language: {language}, File: {file_path.name}"
            )
            if compile_cmd:
                logger.debug(f"Compile command: {' '.join(compile_cmd)}")
            logger.debug(f"Run command: {' '.join(run_cmd)}")

            return file_path, exe_path, compile_cmd, run_cmd

        except Exception as e:
            logger.error(f"Failed to prepare execution: {e}")
            # Return minimal viable configuration
            file_path = self.temp_dir / f"{file_id}.txt"
            file_path.write_text(code, encoding="utf-8")
            self.created_files.append(file_path)
            return (
                file_path,
                None,
                None,
                (
                    [sys.executable, str(file_path)]
                    if sys.executable
                    else ["python", str(file_path)]
                ),
            )

    async def _compile(
        self, compile_cmd: List[str], lang_config: Dict, timeout: int
    ) -> Dict:
        """Compile the code with proper error handling"""

        if not compile_cmd:
            return {"success": True}

        try:
            # Ensure compile_cmd is a list of strings
            if isinstance(compile_cmd, str):
                compile_cmd = [compile_cmd]

            # Filter out empty strings
            compile_cmd = [
                str(cmd).strip() for cmd in compile_cmd if cmd and str(cmd).strip()
            ]

            if not compile_cmd:
                return {"success": True}

            # Run compilation
            compile_process = await asyncio.create_subprocess_exec(
                *compile_cmd,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=False,  # Security: avoid shell injection
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    compile_process.communicate(), timeout=timeout
                )

                # Decode outputs
                out_str = stdout.decode("utf-8", errors="replace") if stdout else ""
                err_str = stderr.decode("utf-8", errors="replace") if stderr else ""

                # Log compilation output for debugging
                if out_str:
                    logger.debug(f"Compilation stdout: {out_str[:200]}")
                if err_str:
                    logger.debug(f"Compilation stderr: {err_str[:200]}")

                if compile_process.returncode != 0:
                    error_msg = f"Compilation failed (exit code: {compile_process.returncode})\n"
                    if err_str:
                        error_msg += f"Error: {err_str}\n"
                    if out_str:
                        error_msg += f"Output: {out_str}"

                    return {
                        "success": False,
                        "error": error_msg,
                        "status": ExecutionStatus.COMPILATION_ERROR.value,
                        "output": out_str,
                        "error_detail": err_str,
                    }

                return {
                    "success": True,
                    "output": out_str,
                    "error": err_str if err_str else "",
                }

            except asyncio.TimeoutError:
                try:
                    compile_process.kill()
                    await compile_process.wait()
                except:
                    pass

                return {
                    "success": False,
                    "error": f"Compilation timeout after {timeout} seconds",
                    "status": ExecutionStatus.TIMEOUT.value,
                }

        except FileNotFoundError as e:
            compiler = compile_cmd[0] if compile_cmd else "unknown"
            return {
                "success": False,
                "error": f"Compiler '{compiler}' not found. Please install required build tools.",
                "status": ExecutionStatus.COMPILATION_ERROR.value,
                "error_detail": str(e),
            }

        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return {
                "success": False,
                "error": f"Compilation error: {str(e)}",
                "status": ExecutionStatus.COMPILATION_ERROR.value,
                "error_detail": (
                    traceback.format_exc()
                    if hasattr(traceback, "format_exc")
                    else str(e)
                ),
            }

    async def _monitor_memory(self, process, memory_limit_mb: int):
        """Monitor process memory usage (psutil required)"""
        if not PSUTIL_AVAILABLE:
            return

        try:
            while True:
                try:
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)

                    if memory_mb > memory_limit_mb:
                        logger.warning(
                            f"Memory limit exceeded: {memory_mb:.2f}MB > {memory_limit_mb}MB"
                        )
                        process.kill()
                        break

                    await asyncio.sleep(0.5)
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.debug(f"Memory monitoring error: {e}")
                    break
        except Exception as e:
            logger.debug(f"Memory monitor task error: {e}")

    async def _cleanup_files(self, file_path: Optional[Path], exe_path: Optional[Path]):
        """Clean up temporary files"""
        async with self._cleanup_lock:
            try:
                if file_path and file_path.exists():
                    file_path.unlink()

                if exe_path and exe_path.exists():
                    exe_path.unlink()

                # Clean up any other generated files
                for pattern in ["*.class", "*.jar", "*.exe", "*.o", "*.pyc"]:
                    for file in self.temp_dir.glob(pattern):
                        try:
                            file.unlink()
                        except:
                            pass

            except Exception as e:
                logger.debug(f"Cleanup error: {e}")

    async def _log_execution(self, result: ExecutionResult, code: str):
        """Log execution details"""
        try:
            log_file = (
                self.logs_dir / f"{datetime.now().strftime('%Y%m%d')}_executions.jsonl"
            )

            log_entry = {
                "timestamp": result.timestamp.isoformat(),
                "language": result.language,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "memory_used_mb": result.memory_used_mb,
                "exit_code": result.exit_code,
                "output_length": len(result.output),
                "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.debug(f"Failed to log execution: {e}")

    async def execute_file(
        self,
        language: str,
        file_path: str,
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute code from a file"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            code = path.read_text(encoding="utf-8")
            return await self.run(language, code, stdin, timeout)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_batch(
        self, executions: List[Tuple[str, str]], max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute multiple code snippets in batch"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_limit(language: str, code: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.run(language, code)

        tasks = [run_with_limit(lang, code) for lang, code in executions]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_executions": self.execution_count,
            "successful": self.success_count,
            "failed": self.error_count,
            "success_rate": (
                (self.success_count / self.execution_count * 100)
                if self.execution_count > 0
                else 0
            ),
            "available_languages": self.available_languages,
            "workspace": str(self.workspace),
            "platform": sys.platform,
            "uptime_seconds": time.time() - self.created_at,
            "files_created": len(self.created_files),
            "psutil_available": PSUTIL_AVAILABLE,
            "resource_available": RESOURCE_AVAILABLE,
            "config": {
                "max_file_size_mb": self.config.max_file_size / 1024 / 1024,
                "max_execution_time": self.config.max_execution_time,
                "max_memory_mb": self.config.max_memory_mb,
                "resource_limits": self.config.resource_limits,
            },
        }

    async def cleanup_old_files(self, older_than_hours: int = 24):
        """Clean up temporary files older than specified hours"""
        cutoff = time.time() - (older_than_hours * 3600)

        cleaned = 0
        for file in self.temp_dir.glob("*"):
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
                    cleaned += 1
            except:
                pass

        logger.info(f"Cleaned up {cleaned} old files")
        return cleaned

    async def shutdown(self):
        """Shutdown the code runner gracefully"""
        logger.info("Shutting down CodeRunner...")

        # Clean up all temporary files
        if self.config.auto_cleanup:
            for file in self.created_files:
                try:
                    if file.exists():
                        file.unlink()
                except:
                    pass

        # Save statistics
        stats_file = self.workspace / "stats.json"
        try:
            with open(stats_file, "w") as f:
                json.dump(self.get_stats(), f, indent=2)
        except:
            pass

        logger.info(
            f"✅ CodeRunner shutdown complete. Total executions: {self.execution_count}"
        )

    def get_language_info(self, language: str) -> Optional[Dict]:
        """Get information about a specific language"""
        config = self._get_language_config(language)
        if not config:
            return None

        return {
            "name": language,
            "extension": config["ext"],
            "compiled": config["compile"] is not None,
            "available": self.is_language_available(language),
            "version": self._get_language_version(language),
        }

    def _get_language_version(self, language: str) -> Optional[str]:
        """Get version of a language interpreter/compiler"""
        config = self._get_language_config(language)
        if not config:
            return None

        try:
            cmd = config["version_cmd"]
            # On Windows, use where for some commands
            if sys.platform == "win32" and cmd[0] in [
                "gcc",
                "g++",
                "javac",
                "go",
                "rustc",
            ]:
                result = subprocess.run(
                    ["where", cmd[0]], capture_output=True, timeout=5, text=True
                )
            else:
                result = subprocess.run(cmd, capture_output=True, timeout=5, text=True)

            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except:
            pass

        return None

    def get_all_languages_info(self) -> Dict[str, Dict]:
        """Get information about all supported languages"""
        return {
            lang: self.get_language_info(lang)
            for lang in [l.name.lower() for l in LanguageConfig]
            if self.get_language_info(lang)
        }
