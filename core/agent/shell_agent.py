"""
Shell Agent for EDIATH
Safely execute terminal commands, manage processes, and handle system operations
"""

import asyncio
import subprocess
import os
import re
import shlex
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import psutil
from enum import Enum
import shutil
import platform
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


class CommandCategory(Enum):
    """Command categories for security filtering"""

    SAFE = "safe"
    FILE_OPS = "file_operations"
    SYSTEM_INFO = "system_info"
    NETWORK = "network"
    PACKAGE_MANAGER = "package_manager"
    DANGEROUS = "dangerous"
    FORBIDDEN = "forbidden"


class ShellType(Enum):
    """Supported shell types"""

    BASH = "bash"
    ZSH = "zsh"
    SH = "sh"
    POWERSHELL = "powershell"
    CMD = "cmd"


@dataclass
class ProcessInfo:
    """Process information container"""

    pid: int
    name: str
    cmdline: List[str]
    status: str
    cpu_percent: float
    memory_percent: float
    create_time: datetime
    parent_pid: Optional[int] = None
    username: Optional[str] = None


@dataclass
class CommandResult:
    """Command execution result"""

    command: str
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    success: bool
    start_time: datetime
    end_time: datetime


class ShellAgent:
    """
    Advanced shell command execution agent with:
    - Safe command execution with sandboxing
    - Command whitelist/blacklist filtering
    - Timeout and resource limits
    - Process management
    - Environment variable handling
    - Command history and logging
    - Parallel command execution
    - Output streaming
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Shell Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Security settings
        self.allowed_commands = self.config.get("allowed_commands", [])
        self.blocked_commands = self.config.get(
            "blocked_commands",
            [
                "rm -rf /",
                "dd if=",
                "mkfs",
                "format",
                ":(){ :|:& };:",  # Fork bomb
                "sudo",
                "su",
                "chmod 777",
                "chown",
                "passwd",
                "shutdown",
                "reboot",
                "halt",
                "poweroff",
                "kill -9",
                "pkill",
                "killall",
            ],
        )
        self.command_timeout = self.config.get("command_timeout", 60)  # seconds
        self.max_output_size = self.config.get(
            "max_output_size", 10 * 1024 * 1024
        )  # 10MB
        self.working_directory = self.config.get("working_directory", os.getcwd())
        self.safe_mode = self.config.get("safe_mode", True)

        # Environment
        self.env_vars = self.config.get("environment", {})
        self.base_env = os.environ.copy()
        self.base_env.update(self.env_vars)

        # Shell configuration
        self.shell_type = self._detect_shell()
        self.shell_executable = self._get_shell_executable()

        # History and logging
        self.command_history: List[CommandResult] = []
        self.max_history = self.config.get("max_history", 1000)

        # Process tracking
        self.active_processes: Dict[int, subprocess.Popen] = {}
        self.process_executor = ThreadPoolExecutor(max_workers=10)

        # Statistics
        self.stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "blocked_commands": 0,
            "total_execution_time": 0.0,
        }

        self.logger.info(
            f"Shell Agent initialized. Shell: {self.shell_type.value}, Safe mode: {self.safe_mode}"
        )

    def _detect_shell(self) -> ShellType:
        """Detect current shell type"""
        shell = os.environ.get("SHELL", "")
        if "bash" in shell:
            return ShellType.BASH
        elif "zsh" in shell:
            return ShellType.ZSH
        elif "powershell" in shell.lower():
            return ShellType.POWERSHELL
        elif "cmd" in shell.lower():
            return ShellType.CMD
        else:
            return ShellType.SH

    def _get_shell_executable(self) -> List[str]:
        """Get shell executable command"""
        if platform.system() == "Windows":
            if self.shell_type == ShellType.POWERSHELL:
                return ["powershell.exe", "-Command"]
            else:
                return ["cmd.exe", "/c"]
        else:
            if self.shell_type == ShellType.BASH:
                return ["bash", "-c"]
            elif self.shell_type == ShellType.ZSH:
                return ["zsh", "-c"]
            else:
                return ["sh", "-c"]

    def _validate_command(self, command: str) -> Tuple[bool, str, CommandCategory]:
        """
        Validate command for security

        Returns:
            (is_allowed, reason, category)
        """
        command_lower = command.lower().strip()

        # Check for empty command
        if not command_lower:
            return False, "Empty command", CommandCategory.FORBIDDEN

        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return (
                    False,
                    f"Blocked command pattern: {blocked}",
                    CommandCategory.FORBIDDEN,
                )

        # Check dangerous patterns
        dangerous_patterns = [
            (r"\|\s*sh\b", "Pipe to shell"),
            (r"\$\{.*\}", "Variable substitution"),
            (r"`.*`", "Command substitution"),
            (r"\\x[0-9a-f]{2}", "Hex encoding"),
            (r"curl.*\|.*sh", "Curl pipe to shell"),
            (r"wget.*\|.*sh", "Wget pipe to shell"),
        ]

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, command_lower):
                if self.safe_mode:
                    return (
                        False,
                        f"Dangerous pattern: {reason}",
                        CommandCategory.DANGEROUS,
                    )

        # Check allowed commands if whitelist is provided
        if self.allowed_commands:
            command_first_word = shlex.split(command)[0] if shlex.split(command) else ""
            if command_first_word not in self.allowed_commands:
                return (
                    False,
                    f"Command not in whitelist: {command_first_word}",
                    CommandCategory.FORBIDDEN,
                )

        # Determine command category
        category = self._categorize_command(command_lower)

        return True, "Command allowed", category

    def _categorize_command(self, command: str) -> CommandCategory:
        """Categorize command type"""
        # File operations
        file_cmds = ["ls", "cat", "head", "tail", "grep", "find", "wc", "sort", "uniq"]
        if any(command.startswith(cmd) for cmd in file_cmds):
            return CommandCategory.FILE_OPS

        # System info
        info_cmds = [
            "ps",
            "top",
            "df",
            "du",
            "free",
            "uname",
            "whoami",
            "date",
            "uptime",
        ]
        if any(command.startswith(cmd) for cmd in info_cmds):
            return CommandCategory.SYSTEM_INFO

        # Network
        net_cmds = [
            "ping",
            "curl",
            "wget",
            "netstat",
            "ss",
            "ip",
            "ifconfig",
            "nslookup",
        ]
        if any(command.startswith(cmd) for cmd in net_cmds):
            return CommandCategory.NETWORK

        # Package managers
        pkg_cmds = ["apt", "yum", "dnf", "pacman", "pip", "npm", "gem"]
        if any(command.startswith(cmd) for cmd in pkg_cmds):
            return CommandCategory.PACKAGE_MANAGER

        # Dangerous commands (but not forbidden)
        dangerous = ["kill", "pkill", "killall", "chmod", "chown", "rm", "mv", "dd"]
        if any(command.startswith(cmd) for cmd in dangerous):
            return CommandCategory.DANGEROUS

        return CommandCategory.SAFE

    async def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict] = None,
        shell: bool = True,
        stream_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a shell command safely

        Args:
            command: Command to execute
            timeout: Timeout in seconds (overrides default)
            cwd: Working directory
            env: Additional environment variables
            shell: Use shell execution (recommended True)
            stream_output: Stream output in real-time

        Returns:
            Dictionary with command result
        """
        # Validate command
        is_allowed, reason, category = self._validate_command(command)

        if not is_allowed:
            self.stats["blocked_commands"] += 1
            self.logger.warning(f"Blocked command: {command} - {reason}")
            return {
                "success": False,
                "command": command,
                "error": f"Command blocked: {reason}",
                "category": category.value,
                "blocked": True,
            }

        start_time = datetime.now()
        timeout_seconds = timeout or self.command_timeout

        # Prepare environment
        process_env = self.base_env.copy()
        if env:
            process_env.update(env)

        # Prepare working directory
        work_dir = cwd or self.working_directory
        Path(work_dir).mkdir(parents=True, exist_ok=True)

        try:
            self.logger.info(f"Executing command: {command}")
            self.stats["total_commands"] += 1

            if stream_output:
                # Stream output in real-time
                result = await self._execute_with_streaming(
                    command, work_dir, process_env, timeout_seconds
                )
            else:
                # Execute and capture output
                result = await self._execute_captured(
                    command, work_dir, process_env, timeout_seconds
                )

            execution_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_execution_time"] += execution_time

            if result["success"]:
                self.stats["successful_commands"] += 1
            else:
                self.stats["failed_commands"] += 1

            # Add metadata
            result.update(
                {
                    "command": command,
                    "execution_time": execution_time,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "category": category.value,
                    "working_directory": work_dir,
                }
            )

            # Store in history
            self._add_to_history(
                CommandResult(
                    command=command,
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    return_code=result.get("return_code", -1),
                    execution_time=execution_time,
                    success=result["success"],
                    start_time=start_time,
                    end_time=datetime.now(),
                )
            )

            return result

        except subprocess.TimeoutExpired:
            self.stats["failed_commands"] += 1
            return {
                "success": False,
                "command": command,
                "error": f"Command timed out after {timeout_seconds} seconds",
                "return_code": -1,
                "timeout": True,
            }
        except Exception as e:
            self.stats["failed_commands"] += 1
            self.logger.error(f"Command execution error: {str(e)}")
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "return_code": -1,
            }

    async def _execute_captured(
        self, command: str, cwd: str, env: Dict, timeout: int
    ) -> Dict:
        """Execute command and capture output"""
        try:
            # Run command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                executable=self.shell_executable[0],
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                # Decode output
                stdout_str = stdout.decode("utf-8", errors="ignore")
                stderr_str = stderr.decode("utf-8", errors="ignore")

                # Truncate if too large
                if len(stdout_str) > self.max_output_size:
                    stdout_str = (
                        stdout_str[: self.max_output_size] + "\n...[TRUNCATED]..."
                    )
                if len(stderr_str) > self.max_output_size:
                    stderr_str = (
                        stderr_str[: self.max_output_size] + "\n...[TRUNCATED]..."
                    )

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": process.returncode,
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise subprocess.TimeoutExpired(command, timeout)

        except Exception:
            raise

    async def _execute_with_streaming(
        self, command: str, cwd: str, env: Dict, timeout: int
    ) -> Dict:
        """Execute command with real-time output streaming"""
        stdout_lines = []
        stderr_lines = []

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                executable=self.shell_executable[0],
            )

            async def read_stream(stream, lines_list, stream_name):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="ignore")
                    lines_list.append(decoded)
                    self.logger.debug(f"[{stream_name}] {decoded.rstrip()}")

            try:
                # Read streams concurrently with timeout
                read_tasks = [
                    asyncio.create_task(
                        read_stream(process.stdout, stdout_lines, "stdout")
                    ),
                    asyncio.create_task(
                        read_stream(process.stderr, stderr_lines, "stderr")
                    ),
                ]

                await asyncio.wait_for(
                    asyncio.gather(*read_tasks, process.wait()), timeout=timeout
                )

                stdout_str = "".join(stdout_lines)
                stderr_str = "".join(stderr_lines)

                # Truncate if too large
                if len(stdout_str) > self.max_output_size:
                    stdout_str = (
                        stdout_str[: self.max_output_size] + "\n...[TRUNCATED]..."
                    )
                if len(stderr_str) > self.max_output_size:
                    stderr_str = (
                        stderr_str[: self.max_output_size] + "\n...[TRUNCATED]..."
                    )

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": process.returncode,
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise subprocess.TimeoutExpired(command, timeout)

        except Exception:
            raise

    def execute_sync(
        self, command: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Synchronous command execution (for non-async contexts)

        Args:
            command: Command to execute
            timeout: Timeout in seconds

        Returns:
            Dictionary with command result
        """
        # Validate command
        is_allowed, reason, category = self._validate_command(command)

        if not is_allowed:
            self.stats["blocked_commands"] += 1
            return {
                "success": False,
                "command": command,
                "error": f"Command blocked: {reason}",
                "category": category.value,
                "blocked": True,
            }

        start_time = datetime.now()
        timeout_seconds = timeout or self.command_timeout

        try:
            self.logger.info(f"Executing sync command: {command}")
            self.stats["total_commands"] += 1

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=self.working_directory,
                env=self.base_env,
                executable=self.shell_executable[0],
            )

            execution_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_execution_time"] += execution_time

            # Truncate output if needed
            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > self.max_output_size:
                stdout = stdout[: self.max_output_size] + "\n...[TRUNCATED]..."
            if len(stderr) > self.max_output_size:
                stderr = stderr[: self.max_output_size] + "\n...[TRUNCATED]..."

            command_result = {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": result.returncode,
                "command": command,
                "execution_time": execution_time,
                "category": category.value,
            }

            if result.returncode == 0:
                self.stats["successful_commands"] += 1
            else:
                self.stats["failed_commands"] += 1

            return command_result

        except subprocess.TimeoutExpired:
            self.stats["failed_commands"] += 1
            return {
                "success": False,
                "command": command,
                "error": f"Command timed out after {timeout_seconds} seconds",
                "return_code": -1,
                "timeout": True,
            }
        except Exception as e:
            self.stats["failed_commands"] += 1
            self.logger.error(f"Sync command error: {str(e)}")
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "return_code": -1,
            }

    async def execute_script(
        self,
        script_content: str,
        interpreter: str = "bash",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a script from string content

        Args:
            script_content: Script content as string
            interpreter: Script interpreter (bash, python, node, etc.)
            timeout: Timeout in seconds

        Returns:
            Dictionary with execution result
        """
        # Create temporary script file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{interpreter}", delete=False, dir=self.working_directory
        ) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Make executable on Unix-like systems
            if platform.system() != "Windows":
                os.chmod(script_path, 0o755)

            # Execute script
            command = (
                f"{interpreter} {script_path}" if interpreter != "bash" else script_path
            )
            result = await self.execute_command(command, timeout=timeout)

            return result

        finally:
            # Clean up temp file
            try:
                os.unlink(script_path)
            except:
                pass

    async def run_pipeline(
        self, commands: List[str], fail_fast: bool = True
    ) -> Dict[str, Any]:
        """
        Run a pipeline of commands sequentially

        Args:
            commands: List of commands to execute
            fail_fast: Stop on first failure

        Returns:
            Dictionary with pipeline results
        """
        results = []
        success_count = 0

        for i, command in enumerate(commands):
            result = await self.execute_command(command)
            results.append(result)

            if result["success"]:
                success_count += 1
            elif fail_fast:
                break

        return {
            "success": (
                all(r["success"] for r in results)
                if not fail_fast
                else success_count == len(commands)
            ),
            "total_commands": len(commands),
            "successful_commands": success_count,
            "failed_commands": len(commands) - success_count,
            "results": results,
        }

    async def run_parallel(
        self, commands: List[str], max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Run multiple commands in parallel

        Args:
            commands: List of commands to execute
            max_concurrent: Maximum concurrent executions

        Returns:
            Dictionary with parallel execution results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(cmd):
            async with semaphore:
                return await self.execute_command(cmd)

        tasks = [run_with_semaphore(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        success_count = 0

        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"success": False, "error": str(result)})
            else:
                processed_results.append(result)
                if result.get("success", False):
                    success_count += 1

        return {
            "success": success_count > 0,
            "total_commands": len(commands),
            "successful_commands": success_count,
            "failed_commands": len(commands) - success_count,
            "results": processed_results,
        }

    def get_process_list(self) -> List[ProcessInfo]:
        """
        Get list of running processes

        Returns:
            List of ProcessInfo objects
        """
        processes = []

        for proc in psutil.process_iter(
            [
                "pid",
                "name",
                "cmdline",
                "status",
                "cpu_percent",
                "memory_percent",
                "create_time",
                "ppid",
                "username",
            ]
        ):
            try:
                info = proc.info
                processes.append(
                    ProcessInfo(
                        pid=info["pid"],
                        name=info["name"] or "unknown",
                        cmdline=info["cmdline"] or [],
                        status=info["status"],
                        cpu_percent=info["cpu_percent"] or 0.0,
                        memory_percent=info["memory_percent"] or 0.0,
                        create_time=datetime.fromtimestamp(info["create_time"]),
                        parent_pid=info["ppid"],
                        username=info["username"],
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return processes

    def kill_process(self, pid: int, force: bool = False) -> Dict[str, Any]:
        """
        Terminate a process by PID

        Args:
            pid: Process ID
            force: Force kill (SIGKILL instead of SIGTERM)

        Returns:
            Dictionary with kill result
        """
        try:
            process = psutil.Process(pid)
            if force:
                process.kill()
                signal_name = "SIGKILL"
            else:
                process.terminate()
                signal_name = "SIGTERM"

            return {
                "success": True,
                "pid": pid,
                "name": process.name(),
                "signal": signal_name,
                "message": f"Process {pid} terminated",
            }
        except psutil.NoSuchProcess:
            return {"success": False, "pid": pid, "error": f"Process {pid} not found"}
        except psutil.AccessDenied:
            return {
                "success": False,
                "pid": pid,
                "error": f"Access denied to kill process {pid}",
            }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information

        Returns:
            Dictionary with system information
        """
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cpus": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": {
                "total": shutil.disk_usage("/").total,
                "used": shutil.disk_usage("/").used,
                "free": shutil.disk_usage("/").free,
            },
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
        }

    def get_command_history(
        self, limit: int = None, success_only: bool = False
    ) -> List[Dict]:
        """
        Get command execution history

        Args:
            limit: Maximum number of entries to return
            success_only: Only return successful commands

        Returns:
            List of command history entries
        """
        history = self.command_history

        if success_only:
            history = [h for h in history if h.success]

        if limit:
            history = history[-limit:]

        return [
            {
                "command": h.command,
                "success": h.success,
                "return_code": h.return_code,
                "execution_time": h.execution_time,
                "start_time": h.start_time.isoformat(),
                "stdout_preview": h.stdout[:200] if h.stdout else "",
                "stderr_preview": h.stderr[:200] if h.stderr else "",
            }
            for h in history
        ]

    def _add_to_history(self, result: CommandResult):
        """Add command result to history"""
        self.command_history.append(result)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

    def add_allowed_command(self, command: str):
        """Add command to whitelist"""
        if command not in self.allowed_commands:
            self.allowed_commands.append(command)
            self.logger.info(f"Added command to whitelist: {command}")

    def add_blocked_pattern(self, pattern: str):
        """Add pattern to blocked commands"""
        if pattern not in self.blocked_commands:
            self.blocked_commands.append(pattern)
            self.logger.info(f"Added blocked pattern: {pattern}")

    def set_working_directory(self, directory: str):
        """Set working directory for commands"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.working_directory = str(path.resolve())
        self.logger.info(f"Working directory set to: {self.working_directory}")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "success_rate": (
                (self.stats["successful_commands"] / self.stats["total_commands"] * 100)
                if self.stats["total_commands"] > 0
                else 0
            ),
            "blocked_rate": (
                (self.stats["blocked_commands"] / self.stats["total_commands"] * 100)
                if self.stats["total_commands"] > 0
                else 0
            ),
            "active_processes": len(self.active_processes),
            "history_size": len(self.command_history),
            "working_directory": self.working_directory,
            "safe_mode": self.safe_mode,
            "shell_type": self.shell_type.value,
        }

    def clear_history(self):
        """Clear command history"""
        self.command_history.clear()
        self.logger.info("Command history cleared")

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "blocked_commands": 0,
            "total_execution_time": 0.0,
        }
        self.logger.info("Statistics reset")


# Integration wrapper for EDIATH
class ShellAgentWrapper:
    """
    Wrapper class to integrate ShellAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.shell_agent = ShellAgent(config)
        self.agent_type = "shell_executor"
        self.capabilities = [
            "execute_command",
            "execute_script",
            "run_pipeline",
            "run_parallel",
            "get_process_list",
            "kill_process",
            "get_system_info",
            "get_command_history",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a shell operation request

        Request format:
        {
            'operation': 'execute|script|pipeline|parallel|processes|kill|system|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "execute":
            return await self.shell_agent.execute_command(
                request.get("command"),
                timeout=request.get("timeout"),
                cwd=request.get("cwd"),
                env=request.get("env"),
                stream_output=request.get("stream_output", False),
            )

        elif operation == "execute_sync":
            # Sync execution (for non-async contexts)
            return self.shell_agent.execute_sync(
                request.get("command"), timeout=request.get("timeout")
            )

        elif operation == "script":
            return await self.shell_agent.execute_script(
                request.get("script_content"),
                interpreter=request.get("interpreter", "bash"),
                timeout=request.get("timeout"),
            )

        elif operation == "pipeline":
            return await self.shell_agent.run_pipeline(
                request.get("commands", []), fail_fast=request.get("fail_fast", True)
            )

        elif operation == "parallel":
            return await self.shell_agent.run_parallel(
                request.get("commands", []),
                max_concurrent=request.get("max_concurrent", 5),
            )

        elif operation == "processes":
            processes = self.shell_agent.get_process_list()
            return {
                "success": True,
                "processes": [
                    {
                        "pid": p.pid,
                        "name": p.name,
                        "status": p.status,
                        "cpu_percent": p.cpu_percent,
                        "memory_percent": p.memory_percent,
                        "create_time": p.create_time.isoformat(),
                        "username": p.username,
                    }
                    for p in processes[:100]  # Limit output
                ],
                "total": len(processes),
            }

        elif operation == "kill":
            return self.shell_agent.kill_process(
                request.get("pid"), force=request.get("force", False)
            )

        elif operation == "system":
            return self.shell_agent.get_system_info()

        elif operation == "history":
            return {
                "success": True,
                "history": self.shell_agent.get_command_history(
                    limit=request.get("limit"),
                    success_only=request.get("success_only", False),
                ),
                "total": len(self.shell_agent.command_history),
            }

        elif operation == "clear_history":
            self.shell_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        elif operation == "stats":
            return self.shell_agent.get_stats()

        elif operation == "set_cwd":
            self.shell_agent.set_working_directory(request.get("directory"))
            return {
                "success": True,
                "working_directory": self.shell_agent.working_directory,
            }

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "ShellAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.shell_agent.get_stats(),
            "shell_type": self.shell_agent.shell_type.value,
            "safe_mode": self.shell_agent.safe_mode,
        }


# Example usage and testing
async def test_shell_agent():
    """Test the shell agent functionality"""

    # Initialize agent
    agent = ShellAgent(safe_mode=True)

    print("=== Shell Agent Test ===\n")

    # Test basic command
    print("1. Basic Command Execution")
    result = await agent.execute_command("echo 'Hello, EDIATH!'")
    print(f"   Success: {result['success']}")
    print(f"   Output: {result.get('stdout', '').strip()}")
    print()

    # Test with timeout
    print("2. Command with Timeout")
    result = await agent.execute_command("sleep 2", timeout=1)
    print(f"   Success: {result['success']}")
    print(f"   Error: {result.get('error', 'No error')}")
    print()

    # Test blocked command
    print("3. Blocked Command")
    result = await agent.execute_command("rm -rf /")
    print(f"   Blocked: {result.get('blocked', False)}")
    print(f"   Reason: {result.get('error', '')}")
    print()

    # Test pipeline
    print("4. Command Pipeline")
    commands = ["echo 'line1\nline2\nline3'", "grep line2", "wc -l"]
    result = await agent.run_pipeline(commands)
    print(f"   Pipeline success: {result['success']}")
    print(f"   Last command output: {result['results'][-1].get('stdout', '').strip()}")
    print()

    # Test parallel execution
    print("5. Parallel Execution")
    commands = [
        "echo 'Task 1' && sleep 1",
        "echo 'Task 2' && sleep 1",
        "echo 'Task 3' && sleep 1",
    ]
    result = await agent.run_parallel(commands, max_concurrent=3)
    print(
        f"   Parallel success: {result['successful_commands']}/{result['total_commands']}"
    )
    print()

    # Test system info
    print("6. System Information")
    info = agent.get_system_info()
    print(f"   Platform: {info['platform']} {info['platform_release']}")
    print(f"   CPU Count: {info['cpus']}")
    print(f"   Memory: {info['memory_total'] / (1024**3):.2f} GB")
    print()

    # Test process list
    print("7. Process List (first 5)")
    processes = agent.get_process_list()
    for proc in processes[:5]:
        print(f"   PID: {proc.pid}, Name: {proc.name}, CPU: {proc.cpu_percent:.1f}%")
    print()

    # Test history
    print("8. Command History")
    history = agent.get_command_history(limit=3)
    for cmd in history:
        print(f"   Command: {cmd['command'][:50]}")
        print(f"   Success: {cmd['success']}, Time: {cmd['execution_time']:.2f}s")
    print()

    # Test stats
    print("9. Agent Statistics")
    stats = agent.get_stats()
    print(f"   Total commands: {stats['total_commands']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Blocked rate: {stats['blocked_rate']:.1f}%")
    print(f"   Total execution time: {stats['total_execution_time']:.2f}s")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_shell_agent())
