"""
Advanced System Controller - EDIATH Core Kernel
Maximum Features Edition - Enterprise Grade
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import psutil
import json
import hashlib
from collections import deque
from contextlib import asynccontextmanager
import aiofiles
import aiohttp
import numpy as np

from core.system import ComponentStatus

from ..utils.logger import logger
from .executor import Executor
from .file_manager import FileManager
from .command_router import CommandRouter, Command, CommandType, CommandHandler
from .safety_layer import SafetyLayer
from .idle_controller import IdleController
from .code_runner import CodeRunner


class Priority(Enum):
    """Command priority levels"""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task descriptor for tracking"""

    id: str
    name: str
    state: TaskState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    priority: Priority = Priority.NORMAL


class MetricsCollector:
    """Collects system metrics"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.disk_history = deque(maxlen=max_history)
        self.network_history = deque(maxlen=max_history)
        self.command_count = 0
        self.error_count = 0
        self.start_time = datetime.now()

    def record_metrics(self) -> None:
        """
        Record current system metrics safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # ⚡ CPU (NON-BLOCKING)
            # ------------------------
            try:
                cpu = psutil.cpu_percent(interval=None)  # FIX: non-blocking
            except Exception:
                cpu = 0.0
            self.cpu_history.append(cpu)

            # ------------------------
            # 🧠 MEMORY
            # ------------------------
            try:
                memory = psutil.virtual_memory().percent
            except Exception:
                memory = 0.0
            self.memory_history.append(memory)

            # ------------------------
            # 💾 DISK
            # ------------------------
            try:
                disk = psutil.disk_usage("/").percent
            except Exception:
                disk = 0.0
            self.disk_history.append(disk)

            # ------------------------
            # 🌐 NETWORK
            # ------------------------
            try:
                net_io = psutil.net_io_counters()
                net_data = {
                    "bytes_sent": getattr(net_io, "bytes_sent", 0),
                    "bytes_recv": getattr(net_io, "bytes_recv", 0),
                }
            except Exception:
                net_data = {"bytes_sent": 0, "bytes_recv": 0}

            self.network_history.append(net_data)

        except Exception as e:
            try:
                logger.debug(f"Metrics recording error: {e}")
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical analysis of metrics (optimized + production-grade)
        """

        from datetime import datetime

        try:
            cpu_hist = list(getattr(self, "cpu_history", []))
            mem_hist = list(getattr(self, "memory_history", []))

            # ------------------------
            # 🛡️ EMPTY CHECK
            # ------------------------
            if not cpu_hist and not mem_hist:
                return {}

            # ------------------------
            # 📊 CPU STATS (SAFE)
            # ------------------------
            cpu_stats = {
                "current": cpu_hist[-1] if cpu_hist else 0.0,
                "average": float(np.mean(cpu_hist)) if cpu_hist else 0.0,
                "max": max(cpu_hist) if cpu_hist else 0.0,
                "min": min(cpu_hist) if cpu_hist else 0.0,
                "trend": self._calculate_trend(cpu_hist) if cpu_hist else 0.0,
            }

            # ------------------------
            # 🧠 MEMORY STATS (SAFE)
            # ------------------------
            mem_stats = {
                "current": mem_hist[-1] if mem_hist else 0.0,
                "average": float(np.mean(mem_hist)) if mem_hist else 0.0,
                "max": max(mem_hist) if mem_hist else 0.0,
                "min": min(mem_hist) if mem_hist else 0.0,
            }

            # ------------------------
            # ⏱️ UPTIME (SAFE)
            # ------------------------
            try:
                uptime = (datetime.now() - self.start_time).total_seconds()
                uptime = max(0.0, uptime)
            except Exception:
                uptime = 0.0

            # ------------------------
            # 📊 COMMAND METRICS
            # ------------------------
            commands = int(getattr(self, "command_count", 0))
            errors = int(getattr(self, "error_count", 0))

            success_rate = (commands - errors) / commands * 100 if commands > 0 else 0.0

            # ------------------------
            # 🚀 BUILD RESULT
            # ------------------------
            return {
                "cpu": cpu_stats,
                "memory": mem_stats,
                "uptime_seconds": round(uptime, 2),
                "uptime_minutes": round(uptime / 60, 2) if uptime > 0 else 0,
                "total_commands": commands,
                "total_errors": errors,
                "success_rate": round(success_rate, 2),
                "health_score": round(
                    (100 - cpu_stats["average"]) * 0.4
                    + (100 - mem_stats["average"]) * 0.4
                    + success_rate * 0.2,
                    2,
                ),
                "timestamp": __import__("time").time(),
            }

        except Exception as e:
            try:
                logger.error(f"❌ Statistics generation failed: {e}")
            except Exception:
                pass

            return {
                "error": "statistics_failed",
                "timestamp": __import__("time").time(),
            }

    def _calculate_trend(self, data: List[float]) -> str:
        """
        Calculate trend direction safely (optimized + production-grade)
        """

        try:
            # ------------------------
            # 🛡️ VALIDATION
            # ------------------------
            if not data or len(data) < 2:
                return "stable"

            # Remove invalid values
            clean_data = [float(x) for x in data if isinstance(x, (int, float))]

            if len(clean_data) < 2:
                return "stable"

            # ------------------------
            # 🔁 SPLIT DATA
            # ------------------------
            mid = len(clean_data) // 2

            first_half = clean_data[:mid]
            second_half = clean_data[mid:]

            if not first_half or not second_half:
                return "stable"

            # ------------------------
            # 📊 MEAN CALCULATION
            # ------------------------
            first_avg = float(np.mean(first_half))
            second_avg = float(np.mean(second_half))

            # ------------------------
            # ⚡ DELTA CALCULATION (MORE STABLE)
            # ------------------------
            delta = second_avg - first_avg

            # Normalize change
            base = max(abs(first_avg), 1e-6)  # avoid division by zero
            change_ratio = delta / base

            # ------------------------
            # 🎯 TREND LOGIC (IMPROVED)
            # ------------------------
            if change_ratio > 0.1:
                return "increasing"
            elif change_ratio < -0.1:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            try:
                logger.debug(f"Trend calculation error: {e}")
            except Exception:
                pass
            return "stable"

    def increment_commands(self):
        self.command_count += 1

    def increment_errors(self):
        self.error_count += 1


class CacheManager:
    """Advanced caching system"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache safely (optimized + production-grade)
        """

        from datetime import datetime

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not key or not isinstance(key, str):
                return None

            cache = getattr(self, "cache", {})

            # ------------------------
            # 🔍 LOOKUP
            # ------------------------
            item = cache.get(key)
            if not item:
                self.misses = getattr(self, "misses", 0) + 1
                return None

            value, timestamp = item

            # ------------------------
            # ⏱️ TTL CHECK (SAFE)
            # ------------------------
            try:
                ttl = float(getattr(self, "ttl", 0))
                age = (datetime.now() - timestamp).total_seconds()
            except Exception:
                ttl = 0
                age = 0

            if ttl > 0 and age >= ttl:
                # expired → remove safely
                try:
                    cache.pop(key, None)
                except Exception:
                    pass

                self.misses = getattr(self, "misses", 0) + 1
                return None

            # ------------------------
            # ✅ HIT
            # ------------------------
            self.hits = getattr(self, "hits", 0) + 1
            return value

        except Exception as e:
            try:
                logger.debug(f"Cache get error: {e}")
            except Exception:
                pass

            self.misses = getattr(self, "misses", 0) + 1
            return None

    async def set(self, key: str, value: Any) -> None:
        """
        Set value in cache safely (optimized + production-grade)
        """

        from datetime import datetime

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not key or not isinstance(key, str):
                return

            cache = getattr(self, "cache", {})
            max_size = max(1, int(getattr(self, "max_size", 1000)))

            # ------------------------
            # 🔁 EVICTION (SAFE LRU-ish)
            # ------------------------
            if len(cache) >= max_size:
                try:
                    # find oldest safely
                    oldest_key = min(
                        cache.keys(),
                        key=lambda k: cache[k][1] if k in cache else datetime.now(),
                    )
                    cache.pop(oldest_key, None)
                except Exception:
                    # fallback: clear one arbitrary item
                    try:
                        cache.pop(next(iter(cache)), None)
                    except Exception:
                        pass

            # ------------------------
            # ⚡ SET VALUE
            # ------------------------
            cache[key] = (value, datetime.now())

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.writes = getattr(self, "writes", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.debug(f"Cache set error: {e}")
            except Exception:
                pass

    async def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total * 100) if total > 0 else 0,
        }


class EventBus:
    """Event bus for inter-component communication"""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.event_history: deque = deque(maxlen=1000)

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    async def emit(self, event_type: str, data: Any = None) -> None:
        """
        Emit an event safely (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not event_type or not isinstance(event_type, str):
                return

            # ------------------------
            # 📦 CREATE EVENT
            # ------------------------
            event = {"type": event_type, "data": data, "timestamp": datetime.now()}

            # ------------------------
            # 🧠 STORE EVENT (SAFE)
            # ------------------------
            try:
                history = getattr(self, "event_history", [])
                history.append(event)

                # limit history size (prevent memory leak)
                if len(history) > 1000:
                    self.event_history = history[-1000:]
            except Exception:
                pass

            # ------------------------
            # 🔔 GET LISTENERS
            # ------------------------
            listeners = getattr(self, "listeners", {}).get(event_type, [])
            if not listeners:
                return

            # ------------------------
            # ⚡ EXECUTE LISTENERS (ISOLATED)
            # ------------------------
            tasks = []

            for callback in list(listeners):  # copy to avoid mutation issues
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(asyncio.create_task(callback(data)))
                    else:
                        # run sync safely in thread
                        tasks.append(asyncio.to_thread(callback, data))
                except Exception as e:
                    try:
                        logger.error(f"Listener scheduling error: {e}")
                    except Exception:
                        pass

            # ------------------------
            # 🔁 RUN ALL (SAFE)
            # ------------------------
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # log failures only (non-blocking)
                for r in results:
                    if isinstance(r, Exception):
                        try:
                            logger.error(f"Event handler error: {r}")
                        except Exception:
                            pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.events_emitted = getattr(self, "events_emitted", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Emit failed: {e}")
            except Exception:
                pass

    def get_history(self, event_type: Optional[str] = None) -> List[Dict]:
        """Get event history"""
        if event_type:
            return [e for e in self.event_history if e["type"] == event_type]
        return list(self.event_history)


class Plugin:
    """Base plugin class"""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.enabled = True

    async def on_load(self):
        """Called when plugin is loaded"""
        pass

    async def on_unload(self):
        """Called when plugin is unloaded"""
        pass

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute plugin functionality"""
        raise NotImplementedError


class PluginManager:
    """Manages plugins and extensions"""

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {}

    def register_plugin(self, plugin: Plugin):
        """Register a plugin"""
        self.plugins[plugin.name] = plugin
        logger.info(f"Plugin registered: {plugin.name} v{plugin.version}")

    async def load_plugin(self, name: str):
        """Load a plugin"""
        if name in self.plugins:
            await self.plugins[name].on_load()
            logger.info(f"Plugin loaded: {name}")

    async def unload_plugin(self, name: str):
        """Unload a plugin"""
        if name in self.plugins:
            await self.plugins[name].on_unload()
            logger.info(f"Plugin unloaded: {name}")

    async def execute_hook(self, hook_name: str, context: Dict[str, Any]) -> Any:
        """
        Execute all hooks safely (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not hook_name or not isinstance(hook_name, str):
                return None

            hooks = getattr(self, "plugin_hooks", {}).get(hook_name)
            if not hooks:
                return None

            tasks = []

            # ------------------------
            # ⚡ SCHEDULE HOOKS (PARALLEL)
            # ------------------------
            for hook in list(hooks):  # copy to avoid mutation issues
                try:
                    if asyncio.iscoroutinefunction(hook):
                        tasks.append(asyncio.create_task(hook(context)))
                    else:
                        tasks.append(asyncio.to_thread(hook, context))
                except Exception as e:
                    try:
                        logger.error(f"Hook scheduling error: {e}")
                    except Exception:
                        pass

            if not tasks:
                return None

            # ------------------------
            # 🔁 EXECUTE ALL (SAFE)
            # ------------------------
            results = await asyncio.gather(*tasks, return_exceptions=True)

            output = []

            for r in results:
                if isinstance(r, Exception):
                    try:
                        logger.error(f"Hook execution error: {r}")
                    except Exception:
                        pass
                    continue

                if r is not None:
                    output.append(r)

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.hooks_executed = getattr(self, "hooks_executed", 0) + len(output)
            except Exception:
                pass

            return output

        except Exception as e:
            try:
                logger.error(f"❌ execute_hook failed: {e}")
            except Exception:
                pass
            return None

    def get_plugins_info(self) -> List[Dict]:
        """Get information about all plugins"""
        return [
            {"name": p.name, "version": p.version, "enabled": p.enabled}
            for p in self.plugins.values()
        ]


class HealthChecker:
    """Advanced health monitoring system"""

    def __init__(self):
        self.health_status = {}
        self.check_functions: Dict[str, Callable] = {}

    def register_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.check_functions[name] = check_func

    async def run_checks(self) -> Dict[str, Any]:
        """
        Run all health checks safely (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        try:
            checks = getattr(self, "check_functions", {})
            if not checks:
                return {"overall": "unknown", "checks": {}, "timestamp": datetime.now()}

            tasks = {}
            results = {}

            # ------------------------
            # ⚡ SCHEDULE CHECKS (PARALLEL)
            # ------------------------
            for name, check_func in checks.items():
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        coro = check_func()
                    else:
                        coro = asyncio.to_thread(check_func)

                    # optional timeout per check
                    timeout = getattr(self, "check_timeout", 5)
                    tasks[name] = asyncio.create_task(
                        asyncio.wait_for(coro, timeout=timeout)
                    )

                except Exception as e:
                    results[name] = {"status": "unhealthy", "error": str(e)}

            # ------------------------
            # 🔁 EXECUTE ALL
            # ------------------------
            if tasks:
                completed = await asyncio.gather(
                    *tasks.values(), return_exceptions=True
                )

                for (name, _), result in zip(tasks.items(), completed):

                    if isinstance(result, Exception):
                        results[name] = {"status": "unhealthy", "error": str(result)}
                    else:
                        # normalize result
                        if not isinstance(result, dict):
                            result = {"status": "unknown", "value": result}

                        result.setdefault("status", "unknown")
                        results[name] = result

            # ------------------------
            # 📊 OVERALL HEALTH
            # ------------------------
            overall_health = "healthy"

            for r in results.values():
                status = r.get("status")

                if status == "unhealthy":
                    overall_health = "unhealthy"
                    break
                elif status == "degraded" and overall_health == "healthy":
                    overall_health = "degraded"

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.health_checks_run = getattr(self, "health_checks_run", 0) + 1
            except Exception:
                pass

            return {
                "overall": overall_health,
                "checks": results,
                "total_checks": len(results),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            try:
                logger.error(f"❌ Health checks failed: {e}")
            except Exception:
                pass

            return {
                "overall": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(),
            }


class ResourceMonitor:
    """Resource usage monitoring and limiting"""

    def __init__(self):
        self.resource_limits = {
            "cpu_percent": 80,
            "memory_percent": 80,
            "disk_percent": 85,
            "network_mbps": 100,
        }
        self.violations = deque(maxlen=100)

    def set_limit(self, resource: str, limit: float):
        """Set resource limit"""
        if resource in self.resource_limits:
            self.resource_limits[resource] = limit

    async def check_resources(self) -> Dict[str, bool]:
        """
        Check if resources are within limits safely (optimized + production-grade)
        """

        from datetime import datetime
        import psutil

        status = {}

        try:
            limits = getattr(self, "resource_limits", {})

            # ------------------------
            # ⚡ CPU (NON-BLOCKING)
            # ------------------------
            try:
                cpu = psutil.cpu_percent(interval=None)
                cpu_limit = limits.get("cpu_percent", 100)
                status["cpu"] = cpu < cpu_limit
            except Exception:
                cpu = 0.0
                status["cpu"] = True

            # ------------------------
            # 🧠 MEMORY
            # ------------------------
            try:
                memory = psutil.virtual_memory().percent
                mem_limit = limits.get("memory_percent", 100)
                status["memory"] = memory < mem_limit
            except Exception:
                memory = 0.0
                status["memory"] = True

            # ------------------------
            # 💾 DISK
            # ------------------------
            try:
                disk = psutil.disk_usage("/").percent
                disk_limit = limits.get("disk_percent", 100)
                status["disk"] = disk < disk_limit
            except Exception:
                disk = 0.0
                status["disk"] = True

            # ------------------------
            # 🚨 RECORD VIOLATIONS (SAFE)
            # ------------------------
            for resource, is_ok in status.items():
                if not is_ok:
                    try:
                        value = {"cpu": cpu, "memory": memory, "disk": disk}.get(
                            resource, None
                        )

                        self.violations.append(
                            {
                                "resource": resource,
                                "timestamp": datetime.now(),
                                "value": value,
                            }
                        )
                    except Exception:
                        pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.resource_checks = getattr(self, "resource_checks", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Resource check error: {e}")
            except Exception:
                pass

        return status

    def get_violations_report(self) -> Dict[str, Any]:
        """Get violations report"""
        if not self.violations:
            return {"has_violations": False}

        return {
            "has_violations": True,
            "total_violations": len(self.violations),
            "recent_violations": list(self.violations)[-10:],
        }


class BackupManager:
    """Manages system backups"""

    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = backup_dir
        self.backup_history = deque(maxlen=50)

    async def create_backup(self, name: str, data: Any) -> Optional[str]:
        """
        Create a backup safely (optimized + production-grade)
        """

        import os
        from datetime import datetime

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not isinstance(name, str):
                return None

            backup_dir = getattr(self, "backup_dir", "./backups")

            # ------------------------
            # 📁 ENSURE DIRECTORY
            # ------------------------
            try:
                os.makedirs(backup_dir, exist_ok=True)
            except Exception:
                pass

            # ------------------------
            # 🕒 SAFE FILENAME
            # ------------------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))

            filename = os.path.join(backup_dir, f"{safe_name}_{timestamp}.json")

            # ------------------------
            # 📦 SERIALIZATION (SAFE)
            # ------------------------
            try:
                content = json.dumps(data, default=str, indent=2)
            except Exception:
                # fallback serialization
                content = json.dumps({"data": str(data)}, indent=2)

            # ------------------------
            # 💾 WRITE FILE (ASYNC)
            # ------------------------
            async with aiofiles.open(filename, "w") as f:
                await f.write(content)

            # ------------------------
            # 🧠 TRACK HISTORY (SAFE)
            # ------------------------
            try:
                history = getattr(self, "backup_history", [])
                history.append(
                    {"name": safe_name, "filename": filename, "timestamp": timestamp}
                )

                # prevent memory growth
                if len(history) > 500:
                    self.backup_history = history[-500:]
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.backups_created = getattr(self, "backups_created", 0) + 1
            except Exception:
                pass

            return filename

        except Exception as e:
            try:
                logger.error(f"❌ Backup creation error: {e}")
            except Exception:
                pass

            return None

    async def restore_backup(self, filename: str) -> Any:
        """
        Restore from backup safely (optimized + production-grade)
        """

        import os

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not filename or not isinstance(filename, str):
                return None

            if not os.path.exists(filename):
                raise FileNotFoundError(f"Backup file not found: {filename}")

            # ------------------------
            # 📂 READ FILE (ASYNC)
            # ------------------------
            async with aiofiles.open(filename, "r") as f:
                content = await f.read()

            if not content:
                return None

            # ------------------------
            # 📦 SAFE PARSE
            # ------------------------
            try:
                data = json.loads(content)
            except Exception:
                # fallback for corrupted JSON
                data = {"raw": content}

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.backups_restored = getattr(self, "backups_restored", 0) + 1
            except Exception:
                pass

            return data

        except Exception as e:
            try:
                logger.error(f"❌ Backup restore error: {e}")
            except Exception:
                pass

            return None


class SystemController:
    """Maximum features System Controller"""

    def __init__(self):
        # Core components
        self.executor = Executor()
        self.file_manager = FileManager()
        self.command_router = CommandRouter()
        self.safety_layer = SafetyLayer()
        self.idle_controller = IdleController()
        self.code_runner = CodeRunner()

        # Advanced components
        self.metrics_collector = MetricsCollector()
        self.cache_manager = CacheManager()
        self.event_bus = EventBus()
        self.plugin_manager = PluginManager()
        self.health_checker = HealthChecker()
        self.resource_monitor = ResourceMonitor()
        self.backup_manager = BackupManager()

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.Queue()
        self.worker_tasks = []

        # State
        self.started_at = datetime.now()
        self._running = False
        self._main_loop_task = None
        self._metrics_task = None
        self._health_task = None
        self._backup_task = None
        # component status tracking
        self.component_status: Dict[str, Any] = {}

        # Configuration
        self.config = {
            "auto_backup_interval": 3600,  # 1 hour
            "metrics_interval": 5,  # 5 seconds
            "health_check_interval": 30,  # 30 seconds
            "max_concurrent_tasks": 10,
            "enable_auto_healing": True,
            "enable_predictive_scaling": True,
        }

        # Setup
        self._setup_safety()
        asyncio.create_task(self._setup_idle_behavior())
        self._register_core_handlers()
        self._register_health_checks()
        self._setup_event_listeners()

        logger.info("🧠 System Controller initialized (Maximum Features Edition)")

    # ------------------------
    # SETUP METHODS
    # ------------------------

    def _setup_safety(self):
        """Setup safety layer with maximum protection"""
        self.safety_layer.register_rate_limit("llm_call", 100, 60)
        self.safety_layer.register_rate_limit("file_operation", 1000, 60)
        self.safety_layer.register_rate_limit("api_call", 500, 60)
        self.safety_layer.register_rate_limit("code_execution", 100, 60)
        self.safety_layer.register_rate_limit("network_request", 200, 60)
        self.safety_layer.register_rate_limit("database_query", 500, 60)

        self.safety_layer.allow_operation("file_read")
        self.safety_layer.allow_operation("file_write")
        self.safety_layer.allow_operation("browser_open")
        self.safety_layer.allow_operation("code_execution")
        self.safety_layer.allow_operation("network_request")
        self.safety_layer.allow_operation("database_operation")

    async def _setup_idle_behavior(self):
        """Setup idle behavior with advanced features"""

        async def on_idle():
            logger.info("🧠 System idle → autonomous thinking")
            await self.autonomous_tick()

        async def on_deep_idle():
            logger.info("🔮 System deep idle → optimization mode")
            await self.optimize_system()

        # idle_controller.add_idle_listener is synchronous and returns bool
        try:
            added = self.idle_controller.add_idle_listener(on_idle)

            # add_deep_idle_listener is async; await it here (we are inside an async setup)
            try:
                deep_added = await self.idle_controller.add_deep_idle_listener(
                    on_deep_idle
                )
            except Exception:
                deep_added = False

            # update component status safely
            try:
                self.component_status["idle_controller"] = ComponentStatus.HEALTHY
            except Exception:
                pass

            if not (added or deep_added):
                logger.debug(
                    "Idle listeners registration returned False for both listeners"
                )

        except Exception as e:
            logger.warning(f"Failed to register idle listeners: {e}")

    def _register_health_checks(self):
        """Register health check functions"""
        self.health_checker.register_check("system", self._system_health_check)
        self.health_checker.register_check("resources", self._resource_health_check)
        self.health_checker.register_check("components", self._components_health_check)
        self.health_checker.register_check(
            "performance", self._performance_health_check
        )

    def _setup_event_listeners(self):
        """Setup event bus listeners"""

        async def on_command_executed(data):
            self.metrics_collector.increment_commands()

        async def on_error_occurred(data):
            self.metrics_collector.increment_errors()

        async def on_system_unhealthy(data):
            if self.config["enable_auto_healing"]:
                await self.auto_heal()

        self.event_bus.subscribe("command_executed", on_command_executed)
        self.event_bus.subscribe("error_occurred", on_error_occurred)
        self.event_bus.subscribe("system_unhealthy", on_system_unhealthy)

    # ------------------------
    # COMMAND HANDLERS
    # ------------------------

    def _register_core_handlers(self):
        """Register all command handlers with maximum features"""

        # Code Handler
        class CodeHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                await self.system.safety_layer.check_permission("code_execution")
                await self.system.safety_layer.check_rate_limit("code_execution")

                lang = command.payload.get("language")
                code = command.payload.get("code")

                if not lang or not code:
                    raise ValueError("Missing language or code")

                # Cache check
                cache_key = f"code_{hashlib.md5(f'{lang}:{code}'.encode()).hexdigest()}"
                cached_result = await self.system.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result

                result = await self.system.code_runner.run(lang, code)
                await self.system.cache_manager.set(cache_key, result)

                return result

        # File Read Handler
        class FileReadHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                await self.system.safety_layer.check_permission("file_read")
                path = command.payload.get("path")

                if not path:
                    raise ValueError("Missing file path")

                # Cache check
                cache_key = f"file_{hashlib.md5(path.encode()).hexdigest()}"
                cached = await self.system.cache_manager.get(cache_key)
                if cached:
                    return cached

                result = await self.system.file_manager.read_file(path)
                await self.system.cache_manager.set(cache_key, result)

                return result

        # File Write Handler
        class FileWriteHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                await self.system.safety_layer.check_permission("file_write")

                path = command.payload.get("path")
                content = command.payload.get("content")

                if not path:
                    raise ValueError("Missing file path")

                result = await self.system.file_manager.write_file(path, content)

                # Invalidate cache
                cache_key = f"file_{hashlib.md5(path.encode()).hexdigest()}"
                await self.system.cache_manager.set(cache_key, content)

                return result

        # Network Request Handler
        class NetworkHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                await self.system.safety_layer.check_permission("network_request")
                await self.system.safety_layer.check_rate_limit("network_request")

                url = command.payload.get("url")
                method = command.payload.get("method", "GET")
                data = command.payload.get("data")

                if not url:
                    raise ValueError("Missing URL")

                cache_key = f"net_{hashlib.md5(f'{method}:{url}'.encode()).hexdigest()}"

                if method == "GET":
                    cached = await self.system.cache_manager.get(cache_key)
                    if cached:
                        return cached

                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, json=data) as response:
                        result = {
                            "status": response.status,
                            "data": await response.json(),
                        }

                        if method == "GET" and response.status == 200:
                            await self.system.cache_manager.set(cache_key, result)

                        return result

        # Database Handler
        class DatabaseHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                await self.system.safety_layer.check_permission("database_operation")
                await self.system.safety_layer.check_rate_limit("database_query")

                operation = command.payload.get("operation")
                collection = command.payload.get("collection")
                data = command.payload.get("data")

                # Mock database operations (replace with actual DB)
                if operation == "query":
                    return {"result": "mock_query_result"}
                elif operation == "insert":
                    return {"result": "mock_insert_result"}
                elif operation == "update":
                    return {"result": "mock_update_result"}
                else:
                    raise ValueError(f"Unknown database operation: {operation}")

        # Status Handler
        class StatusHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                detailed = command.payload.get("detailed", False)

                status = self.system.get_all_status()

                if detailed:
                    status["metrics"] = self.system.metrics_collector.get_statistics()
                    status["cache"] = self.system.cache_manager.get_stats()
                    status["events"] = self.system.event_bus.get_history()
                    status["plugins"] = self.system.plugin_manager.get_plugins_info()
                    status["health"] = await self.system.health_checker.run_checks()
                    status["resources"] = (
                        await self.system.resource_monitor.check_resources()
                    )

                return status

        # Task Manager Handler
        class TaskHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                action = command.payload.get("action", "list")

                if action == "list":
                    return self.system.get_all_tasks()
                elif action == "cancel":
                    task_id = command.payload.get("task_id")
                    return await self.system.cancel_task(task_id)
                elif action == "status":
                    task_id = command.payload.get("task_id")
                    return self.system.get_task_status(task_id)

        # Backup Handler
        class BackupHandler(CommandHandler):
            def __init__(self, system):
                self.system = system

            async def handle(self, command):
                action = command.payload.get("action", "create")

                if action == "create":
                    data = command.payload.get("data", self.system.get_all_status())
                    return await self.system.backup_manager.create_backup(
                        command.payload.get("name", "auto_backup"), data
                    )
                elif action == "restore":
                    filename = command.payload.get("filename")
                    return await self.system.backup_manager.restore_backup(filename)
                elif action == "list":
                    return list(self.system.backup_manager.backup_history)

        # Register all handlers
        handlers = [
            ("run_code", CodeHandler(self)),
            ("read_file", FileReadHandler(self)),
            ("write_file", FileWriteHandler(self)),
            ("network_request", NetworkHandler(self)),
            ("database_query", DatabaseHandler(self)),
            ("status_check", StatusHandler(self)),
            ("task_management", TaskHandler(self)),
            ("backup_operation", BackupHandler(self)),
        ]

        for handler_name, handler in handlers:
            self.command_router.register_handler(
                CommandType.EXECUTE, handler_name, handler
            )

    # ------------------------
    # HEALTH CHECKS
    # ------------------------

    async def _system_health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return {
            "status": "healthy" if self._running else "unhealthy",
            "running": self._running,
            "uptime": (datetime.now() - self.started_at).total_seconds(),
        }

    async def _resource_health_check(self) -> Dict[str, Any]:
        """Check resource health"""
        resources = await self.resource_monitor.check_resources()
        all_ok = all(resources.values())

        return {
            "status": "healthy" if all_ok else "degraded",
            "resources": resources,
            "violations": self.resource_monitor.get_violations_report(),
        }

    async def _components_health_check(self) -> Dict[str, Any]:
        """Check components health"""
        components_status = {}

        # Check each component
        components = [
            ("executor", self.executor),
            ("file_manager", self.file_manager),
            ("command_router", self.command_router),
            ("safety_layer", self.safety_layer),
            ("idle_controller", self.idle_controller),
            ("code_runner", self.code_runner),
        ]

        for name, component in components:
            try:
                if hasattr(component, "health_check"):
                    status = await component.health_check()
                else:
                    status = "healthy"
                components_status[name] = status
            except Exception as e:
                components_status[name] = f"unhealthy: {str(e)}"

        return {
            "status": (
                "healthy"
                if all("unhealthy" not in str(v) for v in components_status.values())
                else "degraded"
            ),
            "components": components_status,
        }

    async def _performance_health_check(self) -> Dict[str, Any]:
        """Check performance health"""
        stats = self.metrics_collector.get_statistics()

        performance_issues = []
        if stats.get("cpu", {}).get("current", 0) > 80:
            performance_issues.append("high_cpu")
        if stats.get("memory", {}).get("current", 0) > 80:
            performance_issues.append("high_memory")

        return {
            "status": "degraded" if performance_issues else "healthy",
            "statistics": stats,
            "issues": performance_issues,
        }

    # ------------------------
    # AUTO HEALING
    # ------------------------

    async def auto_heal(self) -> None:
        """
        Automatic healing procedures (optimized + production-grade)
        """

        import asyncio
        import psutil
        from datetime import datetime

        try:
            try:
                logger.warning("🔄 Running auto-healing procedures")
            except Exception:
                pass

            # ------------------------
            # 🧠 MEMORY CHECK (SAFE)
            # ------------------------
            try:
                mem = psutil.virtual_memory().percent
            except Exception:
                mem = 0

            if mem > 85:
                try:
                    if hasattr(self, "cache_manager") and hasattr(
                        self.cache_manager, "clear"
                    ):
                        await self.cache_manager.clear()
                        logger.info("🧹 Cache cleared (high memory)")
                except Exception as e:
                    logger.debug(f"Cache clear failed: {e}")

            # ------------------------
            # ⚡ CPU CHECK (NON-BLOCKING)
            # ------------------------
            try:
                cpu = psutil.cpu_percent(interval=None)
            except Exception:
                cpu = 0

            if cpu > 90:
                try:
                    process = psutil.Process()
                    for child in process.children(recursive=True):
                        try:
                            child_cpu = child.cpu_percent(interval=0.0)
                            if child_cpu > 50:
                                child.terminate()
                                logger.info(
                                    f"⚠️ Terminated high-CPU process: {child.pid}"
                                )
                        except Exception:
                            continue
                except Exception as e:
                    logger.debug(f"Process handling failed: {e}")

            # ------------------------
            # 🔁 RESTART COMPONENTS (SAFE)
            # ------------------------
            try:
                router = getattr(self, "command_router", None)

                if router and not getattr(router, "_running", False):
                    if hasattr(router, "start"):
                        if asyncio.iscoroutinefunction(router.start):
                            await router.start()
                        else:
                            await asyncio.to_thread(router.start)

                        logger.info("🔁 Command router restarted")
            except Exception as e:
                logger.debug(f"Router restart failed: {e}")

            # ------------------------
            # 📡 EMIT EVENT (SAFE)
            # ------------------------
            try:
                if hasattr(self, "event_bus") and hasattr(self.event_bus, "emit"):
                    await self.event_bus.emit(
                        "healing_completed",
                        {"timestamp": datetime.now(), "cpu": cpu, "memory": mem},
                    )
            except Exception as e:
                logger.debug(f"Event emit failed: {e}")

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.heal_runs = getattr(self, "heal_runs", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Auto-healing error: {e}")
            except Exception:
                pass

    async def optimize_system(self) -> None:
        """
        System optimization routines (optimized + production-grade)
        """

        import asyncio
        import psutil
        from datetime import datetime

        try:
            try:
                logger.info("🔧 Running system optimization")
            except Exception:
                pass

            # ------------------------
            # 🧠 CACHE OPTIMIZATION (SAFE)
            # ------------------------
            try:
                if hasattr(self, "cache_manager"):
                    stats = self.cache_manager.get_stats()
                    if stats.get("size", 0) > 500:
                        await self.cache_manager.clear()
                        logger.info("🧹 Cleared large cache")
            except Exception as e:
                logger.debug(f"Cache optimization failed: {e}")

            # ------------------------
            # 🗑️ CLEAN OLD TASKS (SAFE)
            # ------------------------
            try:
                now = datetime.now()
                tasks = getattr(self, "tasks", {})

                old_tasks = []
                for task_id, task in list(tasks.items()):
                    try:
                        if (
                            task.state
                            in [
                                TaskState.COMPLETED,
                                TaskState.FAILED,
                                TaskState.CANCELLED,
                            ]
                            and getattr(task, "completed_at", None)
                            and (now - task.completed_at).total_seconds() > 3600
                        ):
                            old_tasks.append(task_id)
                    except Exception:
                        continue

                for task_id in old_tasks:
                    tasks.pop(task_id, None)

            except Exception as e:
                logger.debug(f"Task cleanup failed: {e}")

            # ------------------------
            # ⚡ WORKER OPTIMIZATION
            # ------------------------
            try:
                cpu_count = psutil.cpu_count() or 1
                max_tasks = getattr(self, "config", {}).get(
                    "max_concurrent_tasks", cpu_count
                )

                optimal_workers = max(1, min(cpu_count, max_tasks))

                worker_tasks = getattr(self, "worker_tasks", [])

                # Scale up
                while len(worker_tasks) < optimal_workers:
                    try:
                        worker = asyncio.create_task(self._task_worker())
                        worker_tasks.append(worker)
                    except Exception:
                        break

                # Scale down
                while len(worker_tasks) > optimal_workers:
                    worker = worker_tasks.pop()
                    try:
                        worker.cancel()
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"Worker optimization failed: {e}")

            # ------------------------
            # 📡 EMIT EVENT (SAFE)
            # ------------------------
            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "optimization_completed",
                        {
                            "timestamp": datetime.now(),
                            "workers": len(getattr(self, "worker_tasks", [])),
                        },
                    )
            except Exception as e:
                logger.debug(f"Event emit failed: {e}")

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.optimizations = getattr(self, "optimizations", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Optimization error: {e}")
            except Exception:
                pass

    # ------------------------
    # TASK MANAGEMENT
    # ------------------------

    async def submit_task(
        self, name: str, coroutine, priority: Priority = Priority.NORMAL
    ) -> Optional[str]:
        """
        Submit a task for execution safely (optimized + production-grade)
        """

        import hashlib
        from datetime import datetime

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not name or not isinstance(name, str):
                return None

            if coroutine is None:
                return None

            if not isinstance(priority, Priority):
                priority = Priority.NORMAL

            # ------------------------
            # 🆔 UNIQUE TASK ID (STRONGER)
            # ------------------------
            raw_id = f"{name}-{datetime.utcnow().isoformat()}-{id(coroutine)}"
            task_id = hashlib.sha256(raw_id.encode()).hexdigest()[:12]

            # ------------------------
            # 📦 CREATE TASK OBJECT
            # ------------------------
            task = Task(
                id=task_id,
                name=name,
                state=TaskState.PENDING,
                created_at=datetime.utcnow(),
                priority=priority,
            )

            # ------------------------
            # 🧠 STORE TASK (SAFE)
            # ------------------------
            tasks = getattr(self, "tasks", {})
            tasks[task_id] = task

            # ------------------------
            # 📥 QUEUE TASK (SAFE)
            # ------------------------
            try:
                await self.task_queue.put((priority.value, task_id, coroutine))
            except Exception as e:
                # rollback on failure
                tasks.pop(task_id, None)
                raise RuntimeError(f"Queue failed: {e}")

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.tasks_submitted = getattr(self, "tasks_submitted", 0) + 1
            except Exception:
                pass

            return task_id

        except Exception as e:
            try:
                logger.error(f"❌ Task submission error: {e}")
            except Exception:
                pass

            return None

    async def _task_worker(self) -> None:
        """
        Worker process for tasks (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        while getattr(self, "_running", False):
            try:
                # ------------------------
                # 📥 GET TASK
                # ------------------------
                priority, task_id, coroutine = await self.task_queue.get()

                task = getattr(self, "tasks", {}).get(task_id)
                if not task:
                    self.task_queue.task_done()
                    continue

                # ------------------------
                # 🔄 MARK RUNNING
                # ------------------------
                try:
                    task.state = TaskState.RUNNING
                    task.started_at = datetime.utcnow()
                except Exception:
                    pass

                # ------------------------
                # ⚡ EXECUTE WITH TIMEOUT
                # ------------------------
                try:
                    timeout = getattr(self, "task_timeout", None)

                    if timeout:
                        result = await asyncio.wait_for(coroutine, timeout=timeout)
                    else:
                        result = await coroutine

                    task.state = TaskState.COMPLETED
                    task.result = result

                except asyncio.TimeoutError:
                    task.state = TaskState.FAILED
                    task.error = "timeout"

                except Exception as e:
                    task.state = TaskState.FAILED
                    task.error = str(e)

                finally:
                    try:
                        task.completed_at = datetime.utcnow()
                    except Exception:
                        pass

                    # mark queue done
                    try:
                        self.task_queue.task_done()
                    except Exception:
                        pass

            # ------------------------
            # 🛑 CANCEL HANDLING
            # ------------------------
            except asyncio.CancelledError:
                break

            # ------------------------
            # ❌ WORKER ERROR
            # ------------------------
            except Exception as e:
                try:
                    logger.error(f"❌ Task worker error: {e}")
                except Exception:
                    pass

                await asyncio.sleep(0.1)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task safely (optimized + production-grade)
        """

        from datetime import datetime

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not task_id or not isinstance(task_id, str):
                return False

            task = getattr(self, "tasks", {}).get(task_id)
            if not task:
                return False

            # ------------------------
            # 🛑 CANCEL PENDING TASK
            # ------------------------
            if task.state == TaskState.PENDING:
                task.state = TaskState.CANCELLED
                task.completed_at = datetime.utcnow()
                return True

            # ------------------------
            # ⚡ CANCEL RUNNING TASK
            # ------------------------
            if task.state == TaskState.RUNNING:
                try:
                    # if task has underlying asyncio Task
                    if hasattr(task, "asyncio_task") and task.asyncio_task:
                        task.asyncio_task.cancel()
                except Exception:
                    pass

                task.state = TaskState.CANCELLED
                task.completed_at = datetime.utcnow()
                return True

            # ------------------------
            # ❌ ALREADY FINISHED
            # ------------------------
            return False

        except Exception as e:
            try:
                logger.error(f"❌ Cancel task error: {e}")
            except Exception:
                pass
            return False

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                "id": task.id,
                "name": task.name,
                "state": task.state.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "result": task.result,
                "error": task.error,
            }
        return None

    def get_all_tasks(self) -> Dict[str, Any]:
        """Get all tasks status"""
        return {task_id: self.get_task_status(task_id) for task_id in self.tasks}

    # ------------------------
    # MAIN LOOP
    # ------------------------

    async def start(self) -> bool:
        """
        Start full system safely (optimized + production-grade)
        """

        import asyncio
        import psutil
        from datetime import datetime

        try:
            # ------------------------
            # 🔥 PRE-CHECK
            # ------------------------
            if getattr(self, "_running", False):
                logger.warning("System already running")
                return True

            self._running = True

            try:
                logger.info("🚀 Starting EDIATH system (Maximum Features Edition)...")
            except Exception:
                pass

            # ------------------------
            # 🔁 START SUBSYSTEMS (SAFE)
            # ------------------------
            try:
                # Command Router
                router = getattr(self, "command_router", None)
                if router:
                    if hasattr(router, "start"):
                        if asyncio.iscoroutinefunction(router.start):
                            await router.start()
                        else:
                            await asyncio.to_thread(router.start)
                    elif hasattr(router, "process_queue"):
                        asyncio.create_task(router.process_queue())

                # Idle Controller
                idle = getattr(self, "idle_controller", None)
                if idle:
                    if hasattr(idle, "start"):
                        if asyncio.iscoroutinefunction(idle.start):
                            await idle.start()
                        else:
                            await asyncio.to_thread(idle.start)
                    elif hasattr(idle, "start_monitoring"):
                        asyncio.create_task(idle.start_monitoring())

            except Exception as e:
                logger.error(f"Subsystem start error: {e}")

            # ------------------------
            # ⚡ START WORKERS (SAFE)
            # ------------------------
            try:
                cpu_count = psutil.cpu_count() or 1
                max_tasks = getattr(self, "config", {}).get(
                    "max_concurrent_tasks", cpu_count
                )

                worker_count = max(1, min(cpu_count, max_tasks))

                self.worker_tasks = getattr(self, "worker_tasks", [])

                for _ in range(worker_count):
                    try:
                        worker = asyncio.create_task(self._task_worker())
                        self.worker_tasks.append(worker)
                    except Exception:
                        continue

            except Exception as e:
                logger.error(f"Worker start error: {e}")

            # ------------------------
            # 🔁 BACKGROUND LOOPS (SAFE)
            # ------------------------
            try:
                self._main_loop_task = asyncio.create_task(self._main_loop())
                self._metrics_task = asyncio.create_task(self._metrics_loop())
                self._health_task = asyncio.create_task(self._health_loop())
                self._backup_task = asyncio.create_task(self._backup_loop())
            except Exception as e:
                logger.error(f"Background loop error: {e}")

            # ------------------------
            # 📡 EMIT EVENT
            # ------------------------
            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "system_started", {"timestamp": datetime.now()}
                    )
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.start_count = getattr(self, "start_count", 0) + 1
            except Exception:
                pass

            logger.info("✅ System controller started successfully")
            return True

        except Exception as e:
            try:
                logger.error(f"❌ System start failed: {e}")
            except Exception:
                pass

            self._running = False
            return False

    async def _main_loop(self) -> None:
        """
        Internal main loop (optimized + production-grade)
        """

        import asyncio
        import random

        LOOP_DELAY = 3.0

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        while getattr(self, "_running", False):
            try:
                start_time = loop.time()

                # ------------------------
                # 🛡️ HEALTH CHECK (SAFE)
                # ------------------------
                try:
                    await self.health_check()
                except Exception as e:
                    logger.debug(f"Health check failed: {e}")

                # ------------------------
                # ⚡ RESOURCE CHECK
                # ------------------------
                try:
                    resources = await self.resource_monitor.check_resources()

                    if not all(resources.values()):
                        if getattr(self, "config", {}).get(
                            "enable_auto_healing", False
                        ):
                            await self.auto_heal()
                except Exception as e:
                    logger.debug(f"Resource check failed: {e}")

                # ------------------------
                # 💤 IDLE DETECTION (SAFE)
                # ------------------------
                is_idle = False
                try:
                    idle_ctrl = getattr(self, "idle_controller", None)

                    if idle_ctrl and hasattr(idle_ctrl, "is_idle"):
                        idle_check = idle_ctrl.is_idle

                        if callable(idle_check):
                            if asyncio.iscoroutinefunction(idle_check):
                                is_idle = await idle_check()
                            else:
                                is_idle = idle_check()
                        else:
                            is_idle = bool(idle_check)

                except Exception as e:
                    logger.debug(f"Idle check failed: {e}")

                if not is_idle:
                    await asyncio.sleep(0.2)
                    continue

                # ------------------------
                # 🤖 AUTONOMOUS TASKS
                # ------------------------
                try:
                    if random.random() < 0.05:
                        if hasattr(self, "autonomous_tick"):
                            if asyncio.iscoroutinefunction(self.autonomous_tick):
                                await self.autonomous_tick()
                            else:
                                await asyncio.to_thread(self.autonomous_tick)
                except Exception as e:
                    logger.debug(f"Autonomous tick failed: {e}")

                # ------------------------
                # 📈 PREDICTIVE SCALING
                # ------------------------
                try:
                    if getattr(self, "config", {}).get(
                        "enable_predictive_scaling", False
                    ):
                        if hasattr(self, "_predictive_scaling"):
                            await self._predictive_scaling()
                except Exception as e:
                    logger.debug(f"Scaling failed: {e}")

                # ------------------------
                # ⏱️ LOOP TIMING CONTROL
                # ------------------------
                elapsed = loop.time() - start_time
                sleep_time = max(0.5, LOOP_DELAY - elapsed)

                await asyncio.sleep(sleep_time)

            # ------------------------
            # 🛑 CANCEL HANDLING
            # ------------------------
            except asyncio.CancelledError:
                logger.info("🛑 System loop cancelled")
                break

            # ------------------------
            # ❌ ERROR HANDLING
            # ------------------------
            except Exception as e:
                try:
                    logger.error(f"❌ Main loop error: {e}")
                except Exception:
                    pass

                await asyncio.sleep(1.0)

    async def _metrics_loop(self) -> None:
        """
        Metrics collection loop (optimized + production-grade)
        """

        import asyncio

        while getattr(self, "_running", False):
            try:
                # ------------------------
                # 📊 RECORD METRICS (SAFE)
                # ------------------------
                try:
                    if hasattr(self, "metrics_collector"):
                        self.metrics_collector.record_metrics()
                except Exception as e:
                    logger.debug(f"Metrics collection failed: {e}")

                # ------------------------
                # ⏱️ SAFE INTERVAL
                # ------------------------
                interval = max(
                    0.5, float(getattr(self, "config", {}).get("metrics_interval", 2))
                )

                await asyncio.sleep(interval)

            # ------------------------
            # 🛑 CANCEL HANDLING
            # ------------------------
            except asyncio.CancelledError:
                break

            # ------------------------
            # ❌ ERROR HANDLING
            # ------------------------
            except Exception as e:
                try:
                    logger.error(f"❌ Metrics loop error: {e}")
                except Exception:
                    pass

                await asyncio.sleep(1.0)

    async def _health_loop(self) -> None:
        """
        Health check loop (optimized + production-grade)
        """

        import asyncio

        while getattr(self, "_running", False):
            try:
                # ------------------------
                # 🛡️ RUN HEALTH CHECKS (SAFE)
                # ------------------------
                try:
                    if hasattr(self, "health_checker"):
                        health = await self.health_checker.run_checks()
                    else:
                        health = {"overall": "unknown"}
                except Exception as e:
                    logger.debug(f"Health check failed: {e}")
                    health = {"overall": "unhealthy", "error": str(e)}

                # ------------------------
                # 🚨 HANDLE UNHEALTHY STATE
                # ------------------------
                try:
                    if health.get("overall") == "unhealthy":
                        if hasattr(self, "event_bus"):
                            await self.event_bus.emit("system_unhealthy", health)
                except Exception as e:
                    logger.debug(f"Health event emit failed: {e}")

                # ------------------------
                # ⏱️ SAFE INTERVAL
                # ------------------------
                interval = max(
                    1.0,
                    float(getattr(self, "config", {}).get("health_check_interval", 5)),
                )

                await asyncio.sleep(interval)

            # ------------------------
            # 🛑 CANCEL HANDLING
            # ------------------------
            except asyncio.CancelledError:
                break

            # ------------------------
            # ❌ ERROR HANDLING
            # ------------------------
            except Exception as e:
                try:
                    logger.error(f"❌ Health loop error: {e}")
                except Exception:
                    pass

                await asyncio.sleep(1.0)

    async def _backup_loop(self) -> None:
        """
        Automatic backup loop (optimized + production-grade)
        """

        import asyncio

        while getattr(self, "_running", False):
            try:
                # ------------------------
                # ⏱️ SAFE INTERVAL
                # ------------------------
                interval = max(
                    5.0,
                    float(getattr(self, "config", {}).get("auto_backup_interval", 60)),
                )

                await asyncio.sleep(interval)

                # ------------------------
                # 💾 CREATE BACKUP (SAFE)
                # ------------------------
                try:
                    if hasattr(self, "backup_manager") and hasattr(
                        self, "get_all_status"
                    ):
                        data = self.get_all_status()

                        await self.backup_manager.create_backup("auto_backup", data)

                        logger.info("💾 Auto backup completed")

                except Exception as e:
                    logger.error(f"Backup creation failed: {e}")

            # ------------------------
            # 🛑 CANCEL HANDLING
            # ------------------------
            except asyncio.CancelledError:
                break

            # ------------------------
            # ❌ ERROR HANDLING
            # ------------------------
            except Exception as e:
                try:
                    logger.error(f"❌ Backup loop error: {e}")
                except Exception:
                    pass

                await asyncio.sleep(2.0)

    async def _predictive_scaling(self) -> None:
        """
        Predictive scaling based on metrics trends (optimized + production-grade)
        """

        import asyncio
        import psutil

        try:
            # ------------------------
            # 📊 GET STATS (SAFE)
            # ------------------------
            try:
                stats = self.metrics_collector.get_statistics()
            except Exception:
                stats = {}

            cpu_stats = stats.get("cpu", {})
            cpu_trend = cpu_stats.get("trend", "stable")
            current_cpu = float(cpu_stats.get("current", 0))

            worker_tasks = getattr(self, "worker_tasks", [])

            # ------------------------
            # ⚡ LIMITS
            # ------------------------
            cpu_count = psutil.cpu_count() or 1
            max_workers = cpu_count * 2
            min_workers = 1

            # ------------------------
            # 📈 SCALE UP
            # ------------------------
            if cpu_trend == "increasing" and current_cpu > 70:
                target = min(len(worker_tasks) + 2, max_workers)

                while len(worker_tasks) < target:
                    try:
                        worker = asyncio.create_task(self._task_worker())
                        worker_tasks.append(worker)
                    except Exception:
                        break

                logger.info(f"📈 Scaled up workers → {len(worker_tasks)}")

            # ------------------------
            # 📉 SCALE DOWN
            # ------------------------
            elif (
                cpu_trend == "decreasing"
                and current_cpu < 30
                and len(worker_tasks) > min_workers
            ):
                remove_count = min(
                    len(worker_tasks) // 2, len(worker_tasks) - min_workers
                )

                for _ in range(remove_count):
                    try:
                        worker = worker_tasks.pop()
                        worker.cancel()
                    except Exception:
                        continue

                logger.info(f"📉 Scaled down workers → {len(worker_tasks)}")

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.scaling_actions = getattr(self, "scaling_actions", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.error(f"❌ Predictive scaling error: {e}")
            except Exception:
                pass

    # ------------------------
    # CORE FUNCTIONALITY
    # ------------------------

    async def autonomous_tick(self):
        """Single autonomous cycle"""
        logger.debug("⚡ Autonomous tick running")

        cmd = Command(
            CommandType.SYSTEM,
            "status_check",
            priority=5,
            async_execution=True,
            payload={"detailed": False},
        )

        self.command_router.enqueue(cmd)

    async def health_check(self) -> None:
        """
        Health check with auto-healing (optimized + production-grade)
        """

        import psutil

        try:
            # ------------------------
            # ⚡ NON-BLOCKING METRICS
            # ------------------------
            try:
                cpu = psutil.cpu_percent(interval=None)
            except Exception:
                cpu = 0.0

            try:
                mem = psutil.virtual_memory().percent
            except Exception:
                mem = 0.0

            enable_heal = getattr(self, "config", {}).get("enable_auto_healing", False)

            # ------------------------
            # 🚨 CPU CHECK
            # ------------------------
            if cpu > 90:
                logger.warning(f"⚠ High CPU usage: {cpu}%")

                if enable_heal and not getattr(self, "_healing", False):
                    try:
                        self._healing = True
                        await self.auto_heal()
                    finally:
                        self._healing = False

            # ------------------------
            # 🚨 MEMORY CHECK
            # ------------------------
            if mem > 90:
                logger.warning(f"⚠ High memory usage: {mem}%")

                if enable_heal and not getattr(self, "_healing", False):
                    try:
                        self._healing = True
                        await self.auto_heal()
                    finally:
                        self._healing = False

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.health_checks = getattr(self, "health_checks", 0) + 1
            except Exception:
                pass

        except Exception as e:
            try:
                logger.debug(f"Health check error: {e}")
            except Exception:
                pass

    # ------------------------
    # STATUS AND INFO
    # ------------------------

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "cpu_count": psutil.cpu_count(),
                "memory_percent": memory.percent,
                "memory_available": memory.available // (1024 * 1024),
                "memory_total": memory.total // (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free": disk.free // (1024 * 1024),
                "disk_total": disk.total // (1024 * 1024),
                "uptime": (datetime.now() - self.started_at).total_seconds(),
                "uptime_human": str(
                    timedelta(
                        seconds=(datetime.now() - self.started_at).total_seconds()
                    )
                ),
            }
        except Exception:
            return {
                "cpu_percent": 0,
                "cpu_count": 0,
                "memory_percent": 0,
                "memory_available": 0,
                "memory_total": 0,
                "disk_percent": 0,
                "disk_free": 0,
                "disk_total": 0,
                "uptime": 0,
                "uptime_human": "0:00:00",
            }

    def get_process_info(self) -> Dict[str, Any]:
        """Get detailed process information"""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "name": process.name(),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_percent": process.memory_percent(),
                "memory_rss": process.memory_info().rss // (1024 * 1024),
                "memory_vms": process.memory_info().vms // (1024 * 1024),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
                "create_time": datetime.fromtimestamp(
                    process.create_time()
                ).isoformat(),
            }
        except Exception:
            return {
                "pid": 0,
                "name": "unknown",
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_rss": 0,
                "memory_vms": 0,
                "threads": 0,
                "open_files": 0,
                "connections": 0,
                "create_time": None,
            }

    def get_all_status(self) -> Dict[str, Any]:
        """Get all system status information"""
        return {
            "system": self.get_system_status(),
            "process": self.get_process_info(),
            "executor": (
                self.executor.get_stats() if hasattr(self.executor, "get_stats") else {}
            ),
            "violations": (
                len(self.safety_layer.violation_log)
                if hasattr(self.safety_layer, "violation_log")
                else 0
            ),
            "tasks": len(self.tasks),
            "workers": len(self.worker_tasks),
            "running": self._running,
        }

    # ------------------------
    # COMMAND EXECUTION
    # ------------------------

    async def execute(self, command: Any) -> Any:
        """
        Execute a command safely (optimized + production-grade)
        """

        import asyncio

        try:
            # ------------------------
            # 📡 EMIT RECEIVED EVENT (SAFE)
            # ------------------------
            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "command_received", {"command": str(command)[:200]}
                    )
            except Exception:
                pass

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if command is None:
                return None

            # ------------------------
            # 🧠 NORMALIZE COMMAND
            # ------------------------
            try:
                if isinstance(command, str):
                    cmd = Command(CommandType.EXECUTE, command, async_execution=True)
                else:
                    cmd = command
            except Exception as e:
                raise RuntimeError(f"Command creation failed: {e}")

            # ------------------------
            # 📥 ENQUEUE COMMAND (SAFE)
            # ------------------------
            try:
                future = self.command_router.enqueue(cmd)
            except Exception as e:
                raise RuntimeError(f"Command enqueue failed: {e}")

            if not future:
                return None

            # ------------------------
            # ⚡ EXECUTION HANDLING
            # ------------------------
            try:
                if asyncio.isfuture(future) or hasattr(future, "__await__"):
                    result = await future
                else:
                    # sync fallback
                    result = future
            except Exception as e:
                raise RuntimeError(f"Command execution failed: {e}")

            # ------------------------
            # 📡 EMIT SUCCESS EVENT
            # ------------------------
            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "command_executed", {"result": str(result)[:200]}
                    )
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.commands_executed = getattr(self, "commands_executed", 0) + 1
            except Exception:
                pass

            return result

        except Exception as e:
            # ------------------------
            # 🚨 ERROR HANDLING
            # ------------------------
            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "error_occurred",
                        {"error": str(e), "command": str(command)[:200]},
                    )
            except Exception:
                pass

            try:
                logger.error(f"❌ Execute error: {e}")
            except Exception:
                pass

            return None

    # ------------------------
    # PLUGIN MANAGEMENT
    # ------------------------

    async def load_plugin(self, plugin: Plugin):
        """Load a plugin"""
        self.plugin_manager.register_plugin(plugin)
        await self.plugin_manager.load_plugin(plugin.name)
        await self.event_bus.emit("plugin_loaded", {"plugin": plugin.name})

    async def unload_plugin(self, plugin_name: str):
        """Unload a plugin"""
        await self.plugin_manager.unload_plugin(plugin_name)
        await self.event_bus.emit("plugin_unloaded", {"plugin": plugin_name})

    # ------------------------
    # CONFIGURATION
    # ------------------------

    def update_config(self, updates: Dict[str, Any]):
        """Update system configuration"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Updated config: {key} = {value}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

    # ------------------------
    # SHUTDOWN
    # ------------------------

    async def stop(self):
        """Stop the system controller"""
        await self.shutdown()

    async def shutdown(self) -> None:
        """
        Graceful shutdown with full cleanup (optimized + production-grade)
        """

        import asyncio
        from datetime import datetime

        try:
            logger.info("🛑 Shutting down system...")

            # ------------------------
            # 🔥 STOP FLAG
            # ------------------------
            self._running = False

            # ------------------------
            # 📡 EMIT SHUTDOWN EVENT (SAFE)
            # ------------------------
            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "system_shutting_down", {"timestamp": datetime.now()}
                    )
            except Exception:
                pass

            # ------------------------
            # ❌ CANCEL MAIN TASKS (SAFE)
            # ------------------------
            tasks_to_cancel = [
                getattr(self, "_main_loop_task", None),
                getattr(self, "_metrics_task", None),
                getattr(self, "_health_task", None),
                getattr(self, "_backup_task", None),
            ]

            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()

            # wait with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[t for t in tasks_to_cancel if t], return_exceptions=True
                    ),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ Main tasks shutdown timeout")

            # ------------------------
            # ⚡ CANCEL WORKERS
            # ------------------------
            worker_tasks = getattr(self, "worker_tasks", [])

            for worker in worker_tasks:
                try:
                    worker.cancel()
                except Exception:
                    pass

            if worker_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*worker_tasks, return_exceptions=True), timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Worker shutdown timeout")

            # ------------------------
            # 🔁 STOP SUBSYSTEMS (SAFE)
            # ------------------------
            try:
                router = getattr(self, "command_router", None)
                if router and hasattr(router, "stop"):
                    if asyncio.iscoroutinefunction(router.stop):
                        await router.stop()
                    else:
                        await asyncio.to_thread(router.stop)
            except Exception as e:
                logger.debug(f"Router stop failed: {e}")

            try:
                idle = getattr(self, "idle_controller", None)
                if idle and hasattr(idle, "stop"):
                    if asyncio.iscoroutinefunction(idle.stop):
                        await idle.stop()
                    else:
                        await asyncio.to_thread(idle.stop)
            except Exception as e:
                logger.debug(f"Idle stop failed: {e}")

            try:
                executor = getattr(self, "executor", None)
                if executor and hasattr(executor, "clear_tasks"):
                    executor.clear_tasks()
            except Exception:
                pass

            # ------------------------
            # 💾 FINAL BACKUP (SAFE)
            # ------------------------
            try:
                if hasattr(self, "backup_manager") and hasattr(self, "get_all_status"):
                    await self.backup_manager.create_backup(
                        "final_backup", self.get_all_status()
                    )
                    logger.info("💾 Final backup created")
            except Exception as e:
                logger.error(f"Final backup failed: {e}")

            # ------------------------
            # 🧹 CLEAR CACHE (SAFE)
            # ------------------------
            try:
                if hasattr(self, "cache_manager"):
                    await self.cache_manager.clear()
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.shutdown_count = getattr(self, "shutdown_count", 0) + 1
            except Exception:
                pass

            logger.info("✅ System shutdown complete")

        except Exception as e:
            try:
                logger.error(f"❌ Shutdown error: {e}")
            except Exception:
                pass

    # ------------------------
    # UTILITY METHODS
    # ------------------------

    @asynccontextmanager
    async def operation_context(self, operation_name: str):
        """
        Context manager for tracking operations (optimized + production-grade)
        """

        from datetime import datetime

        start_time = datetime.utcnow()

        try:
            # ------------------------
            # ▶ EXECUTION
            # ------------------------
            yield

            # ------------------------
            # ✅ SUCCESS EVENT
            # ------------------------
            duration = (datetime.utcnow() - start_time).total_seconds()

            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "operation_completed",
                        {
                            "name": str(operation_name),
                            "duration": duration,
                            "success": True,
                        },
                    )
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.operations_success = getattr(self, "operations_success", 0) + 1
            except Exception:
                pass

        except Exception as e:
            # ------------------------
            # ❌ FAILURE EVENT
            # ------------------------
            duration = (datetime.utcnow() - start_time).total_seconds()

            try:
                if hasattr(self, "event_bus"):
                    await self.event_bus.emit(
                        "operation_failed",
                        {
                            "name": str(operation_name),
                            "duration": duration,
                            "error": str(e),
                        },
                    )
            except Exception:
                pass

            # ------------------------
            # 📊 METRICS
            # ------------------------
            try:
                self.operations_failed = getattr(self, "operations_failed", 0) + 1
            except Exception:
                pass

            raise

        finally:
            # ------------------------
            # 🧠 TRACK TOTAL OPERATIONS
            # ------------------------
            try:
                self.operations_total = getattr(self, "operations_total", 0) + 1
            except Exception:
                pass
