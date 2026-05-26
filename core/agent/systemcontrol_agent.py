"""
System Control Agent for EDIATH
OS-level control: process management, system monitoring, resource control, hardware info
"""

import asyncio
import os
import sys
import platform
import subprocess
import psutil
import socket
import getpass
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import logging
import re

# Hardware monitoring
try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import netifaces

    NETIFACES_AVAILABLE = True
except ImportError:
    NETIFACES_AVAILABLE = False


class SystemControlAgent:
    """
    Advanced system control agent capable of:
    - Process management (start, stop, list, monitor)
    - System resource monitoring (CPU, memory, disk, network)
    - Service management (start, stop, restart)
    - User session management
    - System information gathering
    - Hardware information (CPU, RAM, GPU, disks)
    - Network configuration and monitoring
    - Power management (shutdown, reboot, sleep)
    - File system operations (mount, unmount, disk usage)
    - Environment variables management
    - System logs access
    - Performance tuning
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize System Control Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Security settings
        self.allow_power_management = self.config.get("allow_power_management", False)
        self.allow_service_management = self.config.get(
            "allow_service_management", False
        )
        self.allowed_commands = self.config.get("allowed_commands", [])
        self.blocked_commands = self.config.get(
            "blocked_commands",
            ["rm -rf /", "dd if=", "mkfs", "format", ":(){ :|:& };:"],
        )

        # Process tracking
        self.monitored_processes: Dict[int, Dict] = {}
        self.process_history: List[Dict] = []

        # System thresholds
        self.cpu_threshold = self.config.get("cpu_threshold", 90)  # percentage
        self.memory_threshold = self.config.get("memory_threshold", 90)
        self.disk_threshold = self.config.get("disk_threshold", 85)

        # Statistics
        self.stats = {
            "processes_started": 0,
            "processes_stopped": 0,
            "commands_executed": 0,
            "alerts_triggered": 0,
            "uptime_start": datetime.now(),
        }

        # Alert callbacks
        self.alert_callbacks: List[callable] = []

        self.logger.info(
            f"System Control Agent initialized on {platform.system()} {platform.release()}"
        )

    # ============
    # System Information
    # ============

    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information

        Returns:
            Dictionary with system information
        """
        try:
            # Basic system info
            system_info = {
                "os": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "hostname": socket.gethostname(),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                },
                "user": {
                    "username": getpass.getuser(),
                    "home": str(Path.home()),
                    "cwd": os.getcwd(),
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "path": sys.path,
                },
            }

            return {
                "success": True,
                "system_info": system_info,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"System info error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information and usage

        Returns:
            Dictionary with CPU information
        """
        try:
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "current_frequency": (
                    psutil.cpu_freq().current if psutil.cpu_freq() else None
                ),
                "min_frequency": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "usage_per_core": psutil.cpu_percent(percpu=True, interval=1),
                "total_usage": psutil.cpu_percent(interval=1),
                "load_average": (
                    psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                ),
                "times": {
                    "user": psutil.cpu_times().user,
                    "system": psutil.cpu_times().system,
                    "idle": psutil.cpu_times().idle,
                },
            }

            # Check threshold
            if cpu_info["total_usage"] > self.cpu_threshold:
                await self._trigger_alert(
                    "high_cpu", f"CPU usage at {cpu_info['total_usage']}%"
                )

            return {
                "success": True,
                "cpu_info": cpu_info,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"CPU info error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory information and usage

        Returns:
            Dictionary with memory information
        """
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()

            memory_info = {
                "virtual": {
                    "total": virtual_memory.total,
                    "available": virtual_memory.available,
                    "used": virtual_memory.used,
                    "free": virtual_memory.free,
                    "percentage": virtual_memory.percent,
                },
                "swap": {
                    "total": swap_memory.total,
                    "used": swap_memory.used,
                    "free": swap_memory.free,
                    "percentage": swap_memory.percent,
                },
            }

            # Check threshold
            if memory_info["virtual"]["percentage"] > self.memory_threshold:
                await self._trigger_alert(
                    "high_memory",
                    f"Memory usage at {memory_info['virtual']['percentage']}%",
                )

            return {
                "success": True,
                "memory_info": memory_info,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Memory info error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_disk_info(self, path: str = "/") -> Dict[str, Any]:
        """
        Get disk information and usage

        Args:
            path: Path to check disk usage

        Returns:
            Dictionary with disk information
        """
        try:
            disk_usage = psutil.disk_usage(path)
            disk_partitions = []

            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    disk_partitions.append(
                        {
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total": partition_usage.total,
                            "used": partition_usage.used,
                            "free": partition_usage.free,
                            "percentage": partition_usage.percent,
                        }
                    )
                except:
                    continue

            disk_info = {
                "current_path": {
                    "path": path,
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percentage": disk_usage.percent,
                },
                "all_partitions": disk_partitions,
                "io_counters": (
                    {
                        "read_count": (
                            psutil.disk_io_counters().read_count
                            if psutil.disk_io_counters()
                            else 0
                        ),
                        "write_count": (
                            psutil.disk_io_counters().write_count
                            if psutil.disk_io_counters()
                            else 0
                        ),
                        "read_bytes": (
                            psutil.disk_io_counters().read_bytes
                            if psutil.disk_io_counters()
                            else 0
                        ),
                        "write_bytes": (
                            psutil.disk_io_counters().write_bytes
                            if psutil.disk_io_counters()
                            else 0
                        ),
                    }
                    if psutil.disk_io_counters()
                    else {}
                ),
            }

            # Check threshold
            if disk_info["current_path"]["percentage"] > self.disk_threshold:
                await self._trigger_alert(
                    "high_disk",
                    f"Disk usage at {disk_info['current_path']['percentage']}% on {path}",
                )

            return {
                "success": True,
                "disk_info": disk_info,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Disk info error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_network_info(self) -> Dict[str, Any]:
        """
        Get network information and statistics

        Returns:
            Dictionary with network information
        """
        try:
            network_info = {"interfaces": {}, "connections": [], "io_counters": {}}

            # Get network interfaces
            for interface, addrs in psutil.net_if_addrs().items():
                network_info["interfaces"][interface] = []
                for addr in addrs:
                    network_info["interfaces"][interface].append(
                        {
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                            "broadcast": addr.broadcast,
                        }
                    )

            # Get network connections
            for conn in psutil.net_connections(kind="inet"):
                network_info["connections"].append(
                    {
                        "fd": conn.fd,
                        "family": conn.family,
                        "type": conn.type,
                        "laddr": (
                            f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None
                        ),
                        "raddr": (
                            f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None
                        ),
                        "status": conn.status,
                        "pid": conn.pid,
                    }
                )

            # Get IO counters
            io_counters = psutil.net_io_counters()
            if io_counters:
                network_info["io_counters"] = {
                    "bytes_sent": io_counters.bytes_sent,
                    "bytes_recv": io_counters.bytes_recv,
                    "packets_sent": io_counters.packets_sent,
                    "packets_recv": io_counters.packets_recv,
                    "errin": io_counters.errin,
                    "errout": io_counters.errout,
                    "dropin": io_counters.dropin,
                    "dropout": io_counters.dropout,
                }

            return {
                "success": True,
                "network_info": network_info,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Network info error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information (if available)

        Returns:
            Dictionary with GPU information
        """
        if not GPU_AVAILABLE:
            return {
                "success": False,
                "error": "GPUtil not installed. Install with: pip install gputil",
            }

        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []

            for gpu in gpus:
                gpu_info.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "memory_util": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                        "uuid": gpu.uuid,
                    }
                )

            return {
                "success": True,
                "gpu_info": gpu_info,
                "total_gpus": len(gpu_info),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"GPU info error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # Process Management
    # ============

    async def list_processes(
        self, filter_by: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        """
        List running processes

        Args:
            filter_by: Filter by process name
            limit: Maximum number of processes to return

        Returns:
            Dictionary with process list
        """
        try:
            processes = []

            for proc in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "cpu_percent",
                    "memory_percent",
                    "status",
                    "create_time",
                    "username",
                ]
            ):
                try:
                    info = proc.info

                    if filter_by and filter_by.lower() not in info["name"].lower():
                        continue

                    processes.append(
                        {
                            "pid": info["pid"],
                            "name": info["name"],
                            "cpu_percent": info["cpu_percent"],
                            "memory_percent": info["memory_percent"],
                            "status": info["status"],
                            "create_time": datetime.fromtimestamp(
                                info["create_time"]
                            ).isoformat(),
                            "username": info["username"],
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage
            processes.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)

            return {
                "success": True,
                "total_processes": len(processes),
                "processes": processes[:limit],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"List processes error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_process_details(self, pid: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific process

        Args:
            pid: Process ID

        Returns:
            Dictionary with process details
        """
        try:
            proc = psutil.Process(pid)

            process_info = {
                "pid": proc.pid,
                "name": proc.name(),
                "exe": proc.exe(),
                "cwd": proc.cwd(),
                "status": proc.status(),
                "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
                "cpu_percent": proc.cpu_percent(interval=0.5),
                "memory_percent": proc.memory_percent(),
                "memory_info": {
                    "rss": proc.memory_info().rss,
                    "vms": proc.memory_info().vms,
                },
                "connections": [],
                "open_files": [],
                "threads": proc.num_threads(),
                "username": proc.username(),
                "parent_pid": proc.ppid(),
                "children": [c.pid for c in proc.children()],
            }

            # Get connections
            try:
                for conn in proc.connections(kind="inet"):
                    process_info["connections"].append(
                        {
                            "fd": conn.fd,
                            "family": conn.family,
                            "type": conn.type,
                            "laddr": (
                                f"{conn.laddr.ip}:{conn.laddr.port}"
                                if conn.laddr
                                else None
                            ),
                            "raddr": (
                                f"{conn.raddr.ip}:{conn.raddr.port}"
                                if conn.raddr
                                else None
                            ),
                            "status": conn.status,
                        }
                    )
            except:
                pass

            # Get open files
            try:
                for file in proc.open_files():
                    process_info["open_files"].append(
                        {"path": file.path, "fd": file.fd}
                    )
            except:
                pass

            return {
                "success": True,
                "process": process_info,
                "timestamp": datetime.now().isoformat(),
            }

        except psutil.NoSuchProcess:
            return {"success": False, "error": f"Process {pid} not found"}
        except Exception as e:
            self.logger.error(f"Process details error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def kill_process(self, pid: int, force: bool = False) -> Dict[str, Any]:
        """
        Terminate a process

        Args:
            pid: Process ID
            force: Force kill (SIGKILL instead of SIGTERM)

        Returns:
            Dictionary with kill result
        """
        try:
            proc = psutil.Process(pid)
            process_name = proc.name()

            if force:
                proc.kill()
                signal_name = "SIGKILL"
            else:
                proc.terminate()
                # Wait for process to terminate
                gone, alive = psutil.wait_procs([proc], timeout=5)
                if proc in alive:
                    proc.kill()
                signal_name = "SIGTERM"

            self.stats["processes_stopped"] += 1

            self.logger.info(
                f"Killed process {pid} ({process_name}) with {signal_name}"
            )

            return {
                "success": True,
                "pid": pid,
                "name": process_name,
                "signal": signal_name,
                "message": f"Process {pid} terminated",
            }

        except psutil.NoSuchProcess:
            return {"success": False, "error": f"Process {pid} not found"}
        except psutil.AccessDenied:
            return {"success": False, "error": f"Access denied to kill process {pid}"}
        except Exception as e:
            self.logger.error(f"Kill process error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def start_process(
        self,
        command: Union[str, List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict] = None,
        background: bool = True,
    ) -> Dict[str, Any]:
        """
        Start a new process

        Args:
            command: Command to execute (string or list)
            cwd: Working directory
            env: Environment variables
            background: Run in background (non-blocking)

        Returns:
            Dictionary with process start result
        """
        # Security check
        if isinstance(command, str):
            command_str = command
        else:
            command_str = " ".join(command)

        if not self._is_command_allowed(command_str):
            return {"success": False, "error": "Command not allowed", "blocked": True}

        try:
            if background:
                # Run in background
                if isinstance(command, str):
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        cwd=cwd,
                        env={**os.environ, **(env or {})},
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                else:
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env={**os.environ, **(env or {})},
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                self.stats["processes_started"] += 1
                self.monitored_processes[process.pid] = {
                    "pid": process.pid,
                    "command": command_str,
                    "started_at": datetime.now(),
                    "process": process,
                }

                return {
                    "success": True,
                    "pid": process.pid,
                    "command": command_str,
                    "background": True,
                    "message": f"Process started with PID {process.pid}",
                }
            else:
                # Run synchronously and wait
                result = subprocess.run(
                    command if isinstance(command, list) else command,
                    shell=isinstance(command, str),
                    cwd=cwd,
                    env={**os.environ, **(env or {})},
                    capture_output=True,
                    text=True,
                    timeout=self.config.get("process_timeout", 300),
                )

                self.stats["processes_started"] += 1

                return {
                    "success": result.returncode == 0,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": command_str,
                    "message": "Process completed",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Process timeout",
                "command": command_str,
            }
        except Exception as e:
            self.logger.error(f"Start process error: {str(e)}")
            return {"success": False, "error": str(e), "command": command_str}

    async def monitor_process(
        self, pid: int, callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Monitor a process for changes

        Args:
            pid: Process ID to monitor
            callback: Callback function on process state change

        Returns:
            Dictionary with monitoring result
        """
        try:
            proc = psutil.Process(pid)

            monitor_info = {
                "pid": pid,
                "name": proc.name(),
                "monitoring": True,
                "started_at": datetime.now().isoformat(),
            }

            self.monitored_processes[pid] = {
                **monitor_info,
                "callback": callback,
                "last_status": proc.status(),
            }

            # Start monitoring task
            asyncio.create_task(self._monitor_process_task(pid, callback))

            return {"success": True, "pid": pid, "message": f"Monitoring process {pid}"}

        except psutil.NoSuchProcess:
            return {"success": False, "error": f"Process {pid} not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _monitor_process_task(self, pid: int, callback: Optional[callable]):
        """Background task to monitor process"""
        while pid in self.monitored_processes:
            try:
                proc = psutil.Process(pid)
                current_status = proc.status()
                last_status = self.monitored_processes[pid].get("last_status")

                if current_status != last_status and callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(pid, current_status, last_status)
                    else:
                        callback(pid, current_status, last_status)

                self.monitored_processes[pid]["last_status"] = current_status
                await asyncio.sleep(1)

            except psutil.NoSuchProcess:
                # Process ended
                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(pid, "terminated", None)
                    else:
                        callback(pid, "terminated", None)

                del self.monitored_processes[pid]
                break
            except Exception as e:
                self.logger.error(f"Monitor task error: {str(e)}")
                await asyncio.sleep(5)

    # ============
    # Service Management
    # ============

    async def service_control(self, service_name: str, action: str) -> Dict[str, Any]:
        """
        Control system services (Linux systemd)

        Args:
            service_name: Name of the service
            action: Action to perform (start, stop, restart, status, enable, disable)

        Returns:
            Dictionary with service control result
        """
        if not self.allow_service_management:
            return {"success": False, "error": "Service management not allowed"}

        if platform.system() != "Linux":
            return {
                "success": False,
                "error": "Service management only supported on Linux",
            }

        allowed_actions = ["start", "stop", "restart", "status", "enable", "disable"]
        if action not in allowed_actions:
            return {
                "success": False,
                "error": f"Invalid action. Allowed: {allowed_actions}",
            }

        try:
            result = subprocess.run(
                ["systemctl", action, service_name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            return {
                "success": result.returncode == 0,
                "service": service_name,
                "action": action,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Service control error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # Power Management
    # ============

    async def shutdown(self, delay: int = 0, force: bool = False) -> Dict[str, Any]:
        """
        Shutdown the system

        Args:
            delay: Delay in seconds
            force: Force shutdown

        Returns:
            Dictionary with shutdown result
        """
        if not self.allow_power_management:
            return {"success": False, "error": "Power management not allowed"}

        try:
            system = platform.system()

            if system == "Windows":
                cmd = f"shutdown /s /t {delay}"
                if force:
                    cmd += " /f"
            elif system == "Linux" or system == "Darwin":
                cmd = f"shutdown -h +{delay // 60}"
                if force:
                    cmd = "shutdown -h now"
            else:
                return {"success": False, "error": f"Unsupported OS: {system}"}

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            self.logger.warning(f"System shutdown initiated with delay {delay}s")

            return {
                "success": result.returncode == 0,
                "action": "shutdown",
                "delay": delay,
                "force": force,
                "message": f"System will shutdown in {delay} seconds",
            }

        except Exception as e:
            self.logger.error(f"Shutdown error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def reboot(self, delay: int = 0, force: bool = False) -> Dict[str, Any]:
        """
        Reboot the system

        Args:
            delay: Delay in seconds
            force: Force reboot

        Returns:
            Dictionary with reboot result
        """
        if not self.allow_power_management:
            return {"success": False, "error": "Power management not allowed"}

        try:
            system = platform.system()

            if system == "Windows":
                cmd = f"shutdown /r /t {delay}"
                if force:
                    cmd += " /f"
            elif system == "Linux" or system == "Darwin":
                cmd = f"shutdown -r +{delay // 60}"
                if force:
                    cmd = "shutdown -r now"
            else:
                return {"success": False, "error": f"Unsupported OS: {system}"}

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            self.logger.warning(f"System reboot initiated with delay {delay}s")

            return {
                "success": result.returncode == 0,
                "action": "reboot",
                "delay": delay,
                "force": force,
                "message": f"System will reboot in {delay} seconds",
            }

        except Exception as e:
            self.logger.error(f"Reboot error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def sleep(self) -> Dict[str, Any]:
        """
        Put system to sleep/suspend

        Returns:
            Dictionary with sleep result
        """
        if not self.allow_power_management:
            return {"success": False, "error": "Power management not allowed"}

        try:
            system = platform.system()

            if system == "Windows":
                cmd = "rundll32.exe powrprof.dll,SetSuspendState 0,1,0"
            elif system == "Linux":
                cmd = "systemctl suspend"
            elif system == "Darwin":
                cmd = "pmset sleepnow"
            else:
                return {"success": False, "error": f"Unsupported OS: {system}"}

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            self.logger.info("System going to sleep")

            return {
                "success": result.returncode == 0,
                "action": "sleep",
                "message": "System going to sleep",
            }

        except Exception as e:
            self.logger.error(f"Sleep error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # Environment Management
    # ============

    async def get_environment_variables(
        self, filter_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get environment variables

        Args:
            filter_pattern: Regex pattern to filter variables

        Returns:
            Dictionary with environment variables
        """
        try:
            env_vars = dict(os.environ)

            if filter_pattern:
                pattern = re.compile(filter_pattern, re.IGNORECASE)
                env_vars = {k: v for k, v in env_vars.items() if pattern.search(k)}

            return {
                "success": True,
                "variables": env_vars,
                "count": len(env_vars),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Get env error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def set_environment_variable(
        self, name: str, value: str, permanent: bool = False
    ) -> Dict[str, Any]:
        """
        Set environment variable

        Args:
            name: Variable name
            value: Variable value
            permanent: Make permanent (system-wide)

        Returns:
            Dictionary with set result
        """
        try:
            # Set for current process
            os.environ[name] = value

            if permanent:
                # Try to set permanently (platform specific)
                system = platform.system()

                if system == "Windows":
                    # Use setx command
                    subprocess.run(
                        f'setx {name} "{value}"', shell=True, capture_output=True
                    )
                elif system == "Linux" or system == "Darwin":
                    # Add to shell profile
                    profile_path = (
                        Path.home() / ".bashrc"
                        if system == "Linux"
                        else Path.home() / ".zshrc"
                    )
                    with open(profile_path, "a") as f:
                        f.write(f'\nexport {name}="{value}"\n')
                else:
                    return {
                        "success": False,
                        "error": f"Permanent env not supported on {system}",
                    }

            return {
                "success": True,
                "name": name,
                "value": value,
                "permanent": permanent,
                "message": f"Environment variable {name} set",
            }

        except Exception as e:
            self.logger.error(f"Set env error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # System Logs
    # ============

    async def get_system_logs(
        self, lines: int = 100, service: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get system logs

        Args:
            lines: Number of lines to retrieve
            service: Specific service to get logs from (Linux only)

        Returns:
            Dictionary with system logs
        """
        try:
            system = platform.system()

            if system == "Linux":
                if service:
                    cmd = f"journalctl -u {service} -n {lines} --no-pager"
                else:
                    cmd = f"journalctl -n {lines} --no-pager"
            elif system == "Darwin":
                cmd = f"log show --last 1h --predicate \"process == 'kernel'\" | head -n {lines}"
            else:
                return {
                    "success": False,
                    "error": f"System logs not supported on {system}",
                }

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )

            return {
                "success": True,
                "logs": result.stdout,
                "lines": len(result.stdout.splitlines()),
                "service": service,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Get logs error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # File System Operations
    # ============

    async def get_disk_usage(self, path: str = "/") -> Dict[str, Any]:
        """Get disk usage for a path"""
        return await self.get_disk_info(path)

    async def mount_filesystem(
        self, device: str, mountpoint: str, fstype: str = "auto"
    ) -> Dict[str, Any]:
        """
        Mount a filesystem (Linux only)

        Args:
            device: Device to mount
            mountpoint: Mount point directory
            fstype: Filesystem type

        Returns:
            Dictionary with mount result
        """
        if platform.system() != "Linux":
            return {"success": False, "error": "Mount only supported on Linux"}

        try:
            # Create mountpoint if it doesn't exist
            Path(mountpoint).mkdir(parents=True, exist_ok=True)

            cmd = (
                f"mount -t {fstype} {device} {mountpoint}"
                if fstype != "auto"
                else f"mount {device} {mountpoint}"
            )
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            return {
                "success": result.returncode == 0,
                "device": device,
                "mountpoint": mountpoint,
                "fstype": fstype,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Mount error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def unmount_filesystem(self, mountpoint: str) -> Dict[str, Any]:
        """
        Unmount a filesystem (Linux only)

        Args:
            mountpoint: Mount point to unmount

        Returns:
            Dictionary with unmount result
        """
        if platform.system() != "Linux":
            return {"success": False, "error": "Unmount only supported on Linux"}

        try:
            result = subprocess.run(
                f"umount {mountpoint}", shell=True, capture_output=True, text=True
            )

            return {
                "success": result.returncode == 0,
                "mountpoint": mountpoint,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }

        except Exception as e:
            self.logger.error(f"Unmount error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # System Monitoring
    # ============

    async def start_system_monitor(
        self, interval: int = 60, callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Start continuous system monitoring

        Args:
            interval: Monitoring interval in seconds
            callback: Callback function for alerts

        Returns:
            Dictionary with monitoring result
        """
        if callback:
            self.alert_callbacks.append(callback)

        # Start monitoring task
        asyncio.create_task(self._system_monitor_task(interval))

        return {
            "success": True,
            "interval": interval,
            "message": "System monitoring started",
        }

    async def _system_monitor_task(self, interval: int):
        """Background task for system monitoring"""
        while True:
            try:
                # Check CPU
                cpu_info = await self.get_cpu_info()
                if (
                    cpu_info["success"]
                    and cpu_info["cpu_info"]["total_usage"] > self.cpu_threshold
                ):
                    await self._trigger_alert(
                        "high_cpu", f"CPU: {cpu_info['cpu_info']['total_usage']}%"
                    )

                # Check Memory
                mem_info = await self.get_memory_info()
                if (
                    mem_info["success"]
                    and mem_info["memory_info"]["virtual"]["percentage"]
                    > self.memory_threshold
                ):
                    await self._trigger_alert(
                        "high_memory",
                        f"Memory: {mem_info['memory_info']['virtual']['percentage']}%",
                    )

                # Check Disk
                disk_info = await self.get_disk_info()
                if (
                    disk_info["success"]
                    and disk_info["disk_info"]["current_path"]["percentage"]
                    > self.disk_threshold
                ):
                    await self._trigger_alert(
                        "high_disk",
                        f"Disk: {disk_info['disk_info']['current_path']['percentage']}%",
                    )

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Monitor task error: {str(e)}")
                await asyncio.sleep(interval)

    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger alert callbacks"""
        self.stats["alerts_triggered"] += 1

        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, message)
                else:
                    callback(alert_type, message)
            except Exception as e:
                self.logger.error(f"Alert callback error: {str(e)}")

    # ============
    # Performance Tuning
    # ============

    async def set_cpu_affinity(self, pid: int, cpu_cores: List[int]) -> Dict[str, Any]:
        """
        Set CPU affinity for a process

        Args:
            pid: Process ID
            cpu_cores: List of CPU core indices

        Returns:
            Dictionary with affinity set result
        """
        try:
            proc = psutil.Process(pid)
            proc.cpu_affinity(cpu_cores)

            return {
                "success": True,
                "pid": pid,
                "cpu_cores": cpu_cores,
                "message": f"CPU affinity set to cores {cpu_cores}",
            }

        except Exception as e:
            self.logger.error(f"Set affinity error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def set_process_priority(self, pid: int, priority: int) -> Dict[str, Any]:
        """
        Set process priority (nice value)

        Args:
            pid: Process ID
            priority: Priority value (-20 to 19, lower = higher priority)

        Returns:
            Dictionary with priority set result
        """
        try:
            proc = psutil.Process(pid)

            if platform.system() == "Windows":
                # Windows priority classes
                priority_map = {
                    -20: psutil.IDLE_PRIORITY_CLASS,
                    -10: psutil.BELOW_NORMAL_PRIORITY_CLASS,
                    0: psutil.NORMAL_PRIORITY_CLASS,
                    10: psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                    20: psutil.HIGH_PRIORITY_CLASS,
                }
                # Find closest priority
                for p_val, p_class in sorted(priority_map.items()):
                    if priority <= p_val:
                        proc.nice(p_class)
                        break
            else:
                # Unix nice value
                proc.nice(priority)

            return {
                "success": True,
                "pid": pid,
                "priority": priority,
                "message": f"Process priority set to {priority}",
            }

        except Exception as e:
            self.logger.error(f"Set priority error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # Utility Methods
    # ============

    def _is_command_allowed(self, command: str) -> bool:
        """Check if command is allowed to execute"""
        command_lower = command.lower()

        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return False

        # Check allowed commands if whitelist is provided
        if self.allowed_commands:
            cmd_first = command_lower.split()[0] if command_lower.split() else ""
            if cmd_first not in self.allowed_commands:
                return False

        return True

    async def register_alert_callback(self, callback: callable) -> Dict[str, Any]:
        """
        Register a callback for system alerts

        Args:
            callback: Callback function (async or sync)

        Returns:
            Dictionary with registration result
        """
        self.alert_callbacks.append(callback)

        return {"success": True, "message": "Alert callback registered"}

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        uptime = (datetime.now() - self.stats["uptime_start"]).total_seconds()

        return {
            **self.stats,
            "uptime_seconds": uptime,
            "monitored_processes": len(self.monitored_processes),
            "alert_callbacks": len(self.alert_callbacks),
            "platform": platform.system(),
            "python_version": sys.version,
        }

    async def shutdown_agent(self):
        """Shutdown the system control agent"""
        self.logger.info("Shutting down System Control Agent")

        # Stop all monitoring tasks
        for pid in list(self.monitored_processes.keys()):
            if pid in self.monitored_processes:
                del self.monitored_processes[pid]

        self.logger.info("System Control Agent shutdown complete")


# Integration wrapper for EDIATH
class SystemControlAgentWrapper:
    """
    Wrapper class to integrate SystemControlAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.system_agent = SystemControlAgent(config)
        self.agent_type = "system_control"
        self.capabilities = [
            "system_info",
            "process_management",
            "resource_monitoring",
            "service_control",
            "power_management",
            "environment_management",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a system control request

        Request format:
        {
            'operation': 'system_info|cpu|memory|disk|processes|kill|start|...',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "system_info":
            return await self.system_agent.get_system_info()

        elif operation == "cpu_info":
            return await self.system_agent.get_cpu_info()

        elif operation == "memory_info":
            return await self.system_agent.get_memory_info()

        elif operation == "disk_info":
            return await self.system_agent.get_disk_info(path=request.get("path", "/"))

        elif operation == "network_info":
            return await self.system_agent.get_network_info()

        elif operation == "gpu_info":
            return await self.system_agent.get_gpu_info()

        elif operation == "list_processes":
            return await self.system_agent.list_processes(
                filter_by=request.get("filter"), limit=request.get("limit", 50)
            )

        elif operation == "process_details":
            return await self.system_agent.get_process_details(pid=request.get("pid"))

        elif operation == "kill_process":
            return await self.system_agent.kill_process(
                pid=request.get("pid"), force=request.get("force", False)
            )

        elif operation == "start_process":
            return await self.system_agent.start_process(
                command=request.get("command"),
                cwd=request.get("cwd"),
                env=request.get("env"),
                background=request.get("background", True),
            )

        elif operation == "monitor_process":
            # Callback not supported via wrapper
            return await self.system_agent.monitor_process(
                pid=request.get("pid"), callback=None
            )

        elif operation == "service_control":
            return await self.system_agent.service_control(
                service_name=request.get("service"), action=request.get("action")
            )

        elif operation == "shutdown":
            return await self.system_agent.shutdown(
                delay=request.get("delay", 0), force=request.get("force", False)
            )

        elif operation == "reboot":
            return await self.system_agent.reboot(
                delay=request.get("delay", 0), force=request.get("force", False)
            )

        elif operation == "sleep":
            return await self.system_agent.sleep()

        elif operation == "get_env":
            return await self.system_agent.get_environment_variables(
                filter_pattern=request.get("filter")
            )

        elif operation == "set_env":
            return await self.system_agent.set_environment_variable(
                name=request.get("name"),
                value=request.get("value"),
                permanent=request.get("permanent", False),
            )

        elif operation == "system_logs":
            return await self.system_agent.get_system_logs(
                lines=request.get("lines", 100), service=request.get("service")
            )

        elif operation == "start_monitor":
            return await self.system_agent.start_system_monitor(
                interval=request.get("interval", 60), callback=None
            )

        elif operation == "stats":
            return self.system_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "SystemControlAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.system_agent.get_stats(),
            "platform": platform.system(),
            "allow_power_management": self.system_agent.allow_power_management,
            "allow_service_management": self.system_agent.allow_service_management,
        }

    async def close(self):
        """Clean up resources"""
        await self.system_agent.shutdown_agent()


# Example usage and testing
async def test_system_agent():
    """Test the system control agent functionality"""

    # Initialize agent
    agent = SystemControlAgent(allow_power_management=False)

    print("=== System Control Agent Test ===\n")

    # Test system info
    print("1. System Information...")
    result = await agent.get_system_info()
    if result["success"]:
        info = result["system_info"]
        print(f"   OS: {info['os']['system']} {info['os']['release']}")
        print(f"   Hostname: {info['os']['hostname']}")
        print(f"   User: {info['user']['username']}")

    # Test CPU info
    print("\n2. CPU Information...")
    result = await agent.get_cpu_info()
    if result["success"]:
        cpu = result["cpu_info"]
        print(
            f"   Cores: {cpu['physical_cores']} physical, {cpu['logical_cores']} logical"
        )
        print(f"   Usage: {cpu['total_usage']}%")
        print(f"   Load average: {cpu['load_average']}")

    # Test memory info
    print("\n3. Memory Information...")
    result = await agent.get_memory_info()
    if result["success"]:
        mem = result["memory_info"]
        total_gb = mem["virtual"]["total"] / (1024**3)
        used_gb = mem["virtual"]["used"] / (1024**3)
        print(f"   Total: {total_gb:.2f} GB")
        print(f"   Used: {used_gb:.2f} GB ({mem['virtual']['percentage']}%)")

    # Test disk info
    print("\n4. Disk Information...")
    result = await agent.get_disk_info()
    if result["success"]:
        disk = result["disk_info"]["current_path"]
        total_gb = disk["total"] / (1024**3)
        free_gb = disk["free"] / (1024**3)
        print(f"   Total: {total_gb:.2f} GB")
        print(f"   Free: {free_gb:.2f} GB ({disk['percentage']}% used)")

    # Test process list
    print("\n5. Process List (top 5)...")
    result = await agent.list_processes(limit=5)
    if result["success"]:
        print(f"   Total processes: {result['total_processes']}")
        for proc in result["processes"]:
            print(
                f"     PID {proc['pid']:6d}: {proc['name'][:30]:30s} CPU: {proc['cpu_percent']:5.1f}%"
            )

    # Test environment variables
    print("\n6. Environment Variables (filtered)...")
    result = await agent.get_environment_variables(filter_pattern="PATH")
    if result["success"]:
        print(f"   Found {result['count']} matching variables")
        for name, value in list(result["variables"].items())[:3]:
            print(f"     {name}={value[:50]}...")

    # Test network info
    print("\n7. Network Information...")
    result = await agent.get_network_info()
    if result["success"]:
        net = result["network_info"]
        print(f"   Interfaces: {len(net['interfaces'])}")
        print(f"   Connections: {len(net['connections'])}")
        if net["io_counters"]:
            sent_mb = net["io_counters"]["bytes_sent"] / (1024**2)
            recv_mb = net["io_counters"]["bytes_recv"] / (1024**2)
            print(f"   Data: {sent_mb:.1f} MB sent, {recv_mb:.1f} MB received")

    # Get statistics
    print("\n8. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Processes started: {stats['processes_started']}")
    print(f"   Processes stopped: {stats['processes_stopped']}")
    print(f"   Commands executed: {stats['commands_executed']}")
    print(f"   Alerts triggered: {stats['alerts_triggered']}")
    print(f"   Uptime: {stats['uptime_seconds']:.0f} seconds")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_system_agent())
