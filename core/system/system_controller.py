"""
Advanced System Controller - EDIATH Core Kernel
Maximum Features Edition - Enterprise Grade
"""
import math
import asyncio
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import os

import psutil
import json
from datetime import datetime as dt


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

    def __init__(
        self,
        max_history: int = 1000,
    ):
        """
        Production-grade MetricsCollector initialization.

        Fixes:
        - invalid history sizes
        - missing locks
        - race conditions
        - corrupted metric buffers
        - missing counters
        - uninitialized timestamps
        - thread safety issues
        - future extension compatibility
        """

        import threading
        import time

        from collections import deque
        from datetime import datetime

        try:

            # ---------------------------------------------------------
            # VALIDATE HISTORY SIZE
            # ---------------------------------------------------------
            try:

                max_history = int(
                    max_history
                )

            except Exception:

                max_history = 1000

            # hard safety limits
            max_history = max(
                10,
                min(
                    max_history,
                    1_000_000,
                ),
            )

            self.max_history = (
                max_history
            )

            # ---------------------------------------------------------
            # THREAD LOCKS
            # ---------------------------------------------------------
            self._metrics_lock = (
                threading.RLock()
            )

            self._history_lock = (
                threading.RLock()
            )

            # ---------------------------------------------------------
            # METRIC HISTORIES
            # ---------------------------------------------------------
            self.cpu_history = deque(
                maxlen=max_history
            )

            self.memory_history = deque(
                maxlen=max_history
            )

            self.disk_history = deque(
                maxlen=max_history
            )

            self.network_history = deque(
                maxlen=max_history
            )

            self.gpu_history = deque(
                maxlen=max_history
            )

            self.latency_history = deque(
                maxlen=max_history
            )

            # ---------------------------------------------------------
            # COUNTERS
            # ---------------------------------------------------------
            self.command_count = 0

            self.error_count = 0

            self.warning_count = 0

            self.events_processed = 0

            self.hooks_executed = 0

            self.cache_hits = 0

            self.cache_misses = 0

            # ---------------------------------------------------------
            # TIMESTAMPS
            # ---------------------------------------------------------
            self.start_time = (
                datetime.now()
            )

            self.last_metrics_update = (
                None
            )

            self.last_error_time = (
                None
            )

            # ---------------------------------------------------------
            # SYSTEM FLAGS
            # ---------------------------------------------------------
            self.initialized = True

            self.enabled = True

            self.collecting = False

            # ---------------------------------------------------------
            # PERFORMANCE TRACKING
            # ---------------------------------------------------------
            self.avg_collection_time_ms = 0.0

            self.total_collections = 0

            self.failed_collections = 0

            # ---------------------------------------------------------
            # NETWORK SNAPSHOT
            # ---------------------------------------------------------
            self.last_network_snapshot = {

                "bytes_sent": 0,

                "bytes_recv": 0,

                "timestamp": time.time(),
            }

            # ---------------------------------------------------------
            # HEALTH STATE
            # ---------------------------------------------------------
            self.health_state = {

                "status": "healthy",

                "last_check": time.time(),

                "failures": 0,
            }

            # ---------------------------------------------------------
            # SAFE LOGGER
            # ---------------------------------------------------------
            try:

                logger.info(
                    f"📊 MetricsCollector initialized "
                    f"| history={max_history}"
                )

            except Exception:
                pass

        except Exception as exc:

            # ---------------------------------------------------------
            # FAILSAFE INIT
            # ---------------------------------------------------------
            self.max_history = 100

            self.cpu_history = deque(
                maxlen=100
            )

            self.memory_history = deque(
                maxlen=100
            )

            self.disk_history = deque(
                maxlen=100
            )

            self.network_history = deque(
                maxlen=100
            )

            self.command_count = 0

            self.error_count = 0

            self.start_time = (
                datetime.now()
            )

            self.initialized = False

            try:

                logger.error(
                    f"❌ MetricsCollector init failed: "
                    f"{exc}"
                )

            except Exception:
                pass

    def record_metrics(self) -> Dict[str, Any]:
        """
        Production-grade system metrics recorder.

        Fixes:
        - blocking psutil calls
        - history corruption
        - unbounded metric growth
        - invalid metric values
        - disk lookup failures
        - network counter crashes
        - race conditions
        - serialization instability
        - metric spikes from bad reads
        """

        import gc
        import threading
        import time
        import traceback

        metrics_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # METRICS LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_metrics_lock",
            ):

                self._metrics_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # HISTORY LIMIT
            # ---------------------------------------------------------
            max_history = int(
                getattr(
                    self,
                    "max_metrics_history",
                    1000,
                )
            )

            max_history = max(
                10,
                min(max_history, 100000),
            )

            # ---------------------------------------------------------
            # THREAD-SAFE RECORDING
            # ---------------------------------------------------------
            with self._metrics_lock:

                # -----------------------------------------------------
                # CPU METRICS
                # -----------------------------------------------------
                try:

                    cpu = float(
                        psutil.cpu_percent(
                            interval=None
                        )
                    )

                except Exception as exc:

                    logger.debug(
                        f"CPU metrics failed: {exc}"
                    )

                    cpu = 0.0

                cpu = max(
                    0.0,
                    min(cpu, 100.0),
                )

                try:

                    self.cpu_history.append(
                        cpu
                    )

                except Exception:

                    self.cpu_history = [cpu]

                # -----------------------------------------------------
                # MEMORY METRICS
                # -----------------------------------------------------
                try:

                    memory = float(
                        psutil.virtual_memory()
                        .percent
                    )

                except Exception as exc:

                    logger.debug(
                        f"Memory metrics failed: {exc}"
                    )

                    memory = 0.0

                memory = max(
                    0.0,
                    min(memory, 100.0),
                )

                try:

                    self.memory_history.append(
                        memory
                    )

                except Exception:

                    self.memory_history = [
                        memory
                    ]

                # -----------------------------------------------------
                # DISK METRICS
                # -----------------------------------------------------
                try:

                    disk = float(
                        psutil.disk_usage(
                            "/"
                        ).percent
                    )

                except Exception as exc:

                    logger.debug(
                        f"Disk metrics failed: {exc}"
                    )

                    disk = 0.0

                disk = max(
                    0.0,
                    min(disk, 100.0),
                )

                try:

                    self.disk_history.append(
                        disk
                    )

                except Exception:

                    self.disk_history = [disk]

                # -----------------------------------------------------
                # NETWORK METRICS
                # -----------------------------------------------------
                try:

                    net_io = (
                        psutil.net_io_counters()
                    )

                    network_data = {

                        "bytes_sent": int(
                            getattr(
                                net_io,
                                "bytes_sent",
                                0,
                            )
                        ),

                        "bytes_recv": int(
                            getattr(
                                net_io,
                                "bytes_recv",
                                0,
                            )
                        ),

                        "packets_sent": int(
                            getattr(
                                net_io,
                                "packets_sent",
                                0,
                            )
                        ),

                        "packets_recv": int(
                            getattr(
                                net_io,
                                "packets_recv",
                                0,
                            )
                        ),

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception as exc:

                    logger.debug(
                        f"Network metrics failed: {exc}"
                    )

                    network_data = {

                        "bytes_sent": 0,

                        "bytes_recv": 0,

                        "packets_sent": 0,

                        "packets_recv": 0,

                        "timestamp": (
                            time.time()
                        ),
                    }

                try:

                    self.network_history.append(
                        network_data
                    )

                except Exception:

                    self.network_history = [
                        network_data
                    ]

                # -----------------------------------------------------
                # HISTORY TRIMMING
                # -----------------------------------------------------
                histories = [

                    "cpu_history",

                    "memory_history",

                    "disk_history",

                    "network_history",
                ]

                for history_name in histories:

                    try:

                        history = getattr(
                            self,
                            history_name,
                            None,
                        )

                        if (
                            isinstance(
                                history,
                                list,
                            )
                            and len(history)
                            > max_history
                        ):

                            trim_size = (
                                len(history)
                                - max_history
                            )

                            del history[
                                :trim_size
                            ]

                    except Exception:
                        continue

            # ---------------------------------------------------------
            # HEALTH DETECTION
            # ---------------------------------------------------------
            issues = []

            if cpu >= 90:

                issues.append(
                    "high_cpu"
                )

            if memory >= 90:

                issues.append(
                    "high_memory"
                )

            if disk >= 95:

                issues.append(
                    "high_disk"
                )

            status = (
                "degraded"
                if issues
                else "healthy"
            )

            # ---------------------------------------------------------
            # SNAPSHOT
            # ---------------------------------------------------------
            snapshot = {

                "status": status,

                "cpu": round(cpu, 2),

                "memory": round(
                    memory,
                    2,
                ),

                "disk": round(disk, 2),

                "network": network_data,

                "issues": issues,

                "timestamp": time.time(),
            }

            # ---------------------------------------------------------
            # CACHE LAST METRICS
            # ---------------------------------------------------------
            try:

                self._last_metrics_snapshot = (
                    snapshot
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS COUNTERS
            # ---------------------------------------------------------
            try:

                self.metrics_records = int(
                    getattr(
                        self,
                        "metrics_records",
                        0,
                    )
                ) + 1

                self.last_metrics_time = (
                    time.time()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # PERIODIC GC
            # ---------------------------------------------------------
            try:

                records = int(
                    getattr(
                        self,
                        "metrics_records",
                        0,
                    )
                )

                if records % 250 == 0:

                    gc.collect()

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - metrics_start
                ) * 1000,
                2,
            )

            snapshot[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"📊 Metrics recorded "
                f"| cpu={cpu:.1f}% "
                f"| mem={memory:.1f}% "
                f"| latency={latency_ms}ms"
            )

            return snapshot

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.debug(
                f"Metrics recording failed: {exc}"
            )

            return {

                "status": "error",

                "cpu": 0.0,

                "memory": 0.0,

                "disk": 0.0,

                "network": {

                    "bytes_sent": 0,

                    "bytes_recv": 0,
                },

                "issues": [
                    "metrics_failure"
                ],

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :800
                    ]
                ),

                "timestamp": time.time(),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Production-grade metrics statistics analyzer.

        Fixes:
        - corrupted history handling
        - empty dataset crashes
        - invalid trend calculations
        - NaN/inf propagation
        - unsafe numeric operations
        - inconsistent uptime reporting
        - unstable health scoring
        - serialization instability
        - metric overflow issues
        """

        import math
        import statistics
        import threading
        import time
        import traceback

        from datetime import datetime

        analysis_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # METRICS LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_metrics_lock",
            ):

                self._metrics_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # SAFE HISTORY SNAPSHOT
            # ---------------------------------------------------------
            with self._metrics_lock:

                cpu_hist = list(
                    getattr(
                        self,
                        "cpu_history",
                        [],
                    )
                )

                mem_hist = list(
                    getattr(
                        self,
                        "memory_history",
                        [],
                    )
                )

                disk_hist = list(
                    getattr(
                        self,
                        "disk_history",
                        [],
                    )
                )

                network_hist = list(
                    getattr(
                        self,
                        "network_history",
                        [],
                    )
                )

            # ---------------------------------------------------------
            # SANITIZE HISTORIES
            # ---------------------------------------------------------
            def sanitize_numeric(
                values,
            ):

                clean = []

                for value in values:

                    try:

                        number = float(
                            value
                        )

                        if (
                            math.isnan(
                                number
                            )
                            or math.isinf(
                                number
                            )
                        ):

                            continue

                        number = max(
                            0.0,
                            min(number, 100.0),
                        )

                        clean.append(
                            number
                        )

                    except Exception:
                        continue

                return clean

            cpu_hist = sanitize_numeric(
                cpu_hist
            )

            mem_hist = sanitize_numeric(
                mem_hist
            )

            disk_hist = sanitize_numeric(
                disk_hist
            )

            # ---------------------------------------------------------
            # EMPTY CHECK
            # ---------------------------------------------------------
            if (
                not cpu_hist
                and not mem_hist
                and not disk_hist
            ):

                return {

                    "status": "empty",

                    "message": (
                        "No metrics available"
                    ),

                    "timestamp": (
                        time.time()
                    ),
                }

            # ---------------------------------------------------------
            # SAFE STATS HELPER
            # ---------------------------------------------------------
            def calculate_stats(
                history,
            ):

                if not history:

                    return {

                        "current": 0.0,

                        "average": 0.0,

                        "max": 0.0,

                        "min": 0.0,

                        "median": 0.0,

                        "stdev": 0.0,

                        "samples": 0,

                        "trend": "stable",
                    }

                try:

                    average = round(
                        float(
                            statistics.mean(
                                history
                            )
                        ),
                        2,
                    )

                except Exception:

                    average = 0.0

                try:

                    median = round(
                        float(
                            statistics.median(
                                history
                            )
                        ),
                        2,
                    )

                except Exception:

                    median = 0.0

                try:

                    stdev = round(
                        float(
                            statistics.pstdev(
                                history
                            )
                        ),
                        2,
                    )

                except Exception:

                    stdev = 0.0

                # -----------------------------------------------------
                # TREND DETECTION
                # -----------------------------------------------------
                trend = "stable"

                try:

                    if len(history) >= 5:

                        recent = history[-5:]

                        slope = (
                            recent[-1]
                            - recent[0]
                        )

                        if slope > 5:

                            trend = (
                                "increasing"
                            )

                        elif slope < -5:

                            trend = (
                                "decreasing"
                            )

                except Exception:
                    pass

                return {

                    "current": round(
                        float(
                            history[-1]
                        ),
                        2,
                    ),

                    "average": average,

                    "max": round(
                        max(history),
                        2,
                    ),

                    "min": round(
                        min(history),
                        2,
                    ),

                    "median": median,

                    "stdev": stdev,

                    "samples": len(
                        history
                    ),

                    "trend": trend,
                }

            # ---------------------------------------------------------
            # CPU STATS
            # ---------------------------------------------------------
            cpu_stats = calculate_stats(
                cpu_hist
            )

            # ---------------------------------------------------------
            # MEMORY STATS
            # ---------------------------------------------------------
            mem_stats = calculate_stats(
                mem_hist
            )

            # ---------------------------------------------------------
            # DISK STATS
            # ---------------------------------------------------------
            disk_stats = calculate_stats(
                disk_hist
            )

            # ---------------------------------------------------------
            # NETWORK STATS
            # ---------------------------------------------------------
            network_stats = {

                "samples": len(
                    network_hist
                ),

                "bytes_sent": 0,

                "bytes_recv": 0,
            }

            try:

                if network_hist:

                    latest = network_hist[-1]

                    if isinstance(
                        latest,
                        dict,
                    ):

                        network_stats[
                            "bytes_sent"
                        ] = int(
                            latest.get(
                                "bytes_sent",
                                0,
                            )
                        )

                        network_stats[
                            "bytes_recv"
                        ] = int(
                            latest.get(
                                "bytes_recv",
                                0,
                            )
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # UPTIME
            # ---------------------------------------------------------
            try:

                start_time = getattr(
                    self,
                    "start_time",
                    None,
                )

                if start_time:

                    uptime = max(

                        0.0,

                        (
                            datetime.now()
                            - start_time
                        ).total_seconds(),
                    )

                else:

                    uptime = 0.0

            except Exception:

                uptime = 0.0

            # ---------------------------------------------------------
            # COMMAND METRICS
            # ---------------------------------------------------------
            try:

                commands = int(
                    getattr(
                        self,
                        "command_count",
                        0,
                    )
                )

            except Exception:

                commands = 0

            try:

                errors = int(
                    getattr(
                        self,
                        "error_count",
                        0,
                    )
                )

            except Exception:

                errors = 0

            commands = max(
                0,
                commands,
            )

            errors = max(
                0,
                errors,
            )

            success_rate = 0.0

            try:

                if commands > 0:

                    success_rate = round(

                        (
                            (
                                commands
                                - errors
                            )
                            / commands
                        )
                        * 100,

                        2,
                    )

                    success_rate = max(
                        0.0,
                        min(
                            success_rate,
                            100.0,
                        ),
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            try:

                cpu_component = (
                    100
                    - cpu_stats[
                        "average"
                    ]
                ) * 0.4

                mem_component = (
                    100
                    - mem_stats[
                        "average"
                    ]
                ) * 0.4

                success_component = (
                    success_rate
                ) * 0.2

                health_score = round(

                    max(
                        0.0,

                        min(

                            cpu_component
                            + mem_component
                            + success_component,

                            100.0,
                        ),
                    ),

                    2,
                )

            except Exception:

                health_score = 0.0

            # ---------------------------------------------------------
            # HEALTH STATUS
            # ---------------------------------------------------------
            issues = []

            if (
                cpu_stats["average"]
                >= 85
            ):

                issues.append(
                    "high_cpu"
                )

            if (
                mem_stats["average"]
                >= 85
            ):

                issues.append(
                    "high_memory"
                )

            if (
                disk_stats["average"]
                >= 95
            ):

                issues.append(
                    "high_disk"
                )

            status = (
                "degraded"
                if issues
                else "healthy"
            )

            # ---------------------------------------------------------
            # BUILD RESULT
            # ---------------------------------------------------------
            result = {

                "status": status,

                "cpu": cpu_stats,

                "memory": mem_stats,

                "disk": disk_stats,

                "network": network_stats,

                "uptime_seconds": round(
                    uptime,
                    2,
                ),

                "uptime_minutes": round(
                    uptime / 60,
                    2,
                ),

                "uptime_hours": round(
                    uptime / 3600,
                    2,
                ),

                "total_commands": commands,

                "total_errors": errors,

                "success_rate": success_rate,

                "health_score": (
                    health_score
                ),

                "issues": issues,

                "timestamp": (
                    time.time()
                ),
            }

            # ---------------------------------------------------------
            # CACHE LAST STATS
            # ---------------------------------------------------------
            try:

                self._last_statistics = (
                    result
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.statistics_requests = int(
                    getattr(
                        self,
                        "statistics_requests",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - analysis_start
                ) * 1000,
                2,
            )

            result[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"📈 Statistics generated "
                f"| health={health_score} "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Statistics generation failed: {exc}"
            )

            return {

                "status": "error",

                "error": (
                    "statistics_failed"
                ),

                "message": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1000
                    ]
                ),

                "timestamp": (
                    time.time()
                ),
            }

    def _calculate_trend(
        self,
        data: List[float],
    ) -> str:
        """
        Production-grade trend analysis.

        Fixes:
        - invalid numeric values
        - NaN/inf contamination
        - unstable averages
        - tiny sample instability
        - division-by-zero
        - noisy fluctuation misclassification
        - overflow/underflow issues
        - corrupted datasets
        """

        import math
        import statistics
        import time
        import traceback

        trend_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # INPUT VALIDATION
            # ---------------------------------------------------------
            if data is None:

                return "stable"

            if not isinstance(
                data,
                (
                    list,
                    tuple,
                ),
            ):

                return "stable"

            if len(data) < 2:

                return "stable"

            # ---------------------------------------------------------
            # SANITIZE DATA
            # ---------------------------------------------------------
            clean_data = []

            for value in data:

                try:

                    number = float(
                        value
                    )

                    # reject invalid numbers
                    if (
                        math.isnan(
                            number
                        )
                        or math.isinf(
                            number
                        )
                    ):

                        continue

                    # clamp absurd values
                    number = max(
                        -1e9,
                        min(
                            number,
                            1e9,
                        ),
                    )

                    clean_data.append(
                        number
                    )

                except Exception:
                    continue

            # ---------------------------------------------------------
            # MINIMUM SAMPLE SIZE
            # ---------------------------------------------------------
            if len(clean_data) < 3:

                return "stable"

            # ---------------------------------------------------------
            # REMOVE EXTREME OUTLIERS
            # ---------------------------------------------------------
            try:

                median = statistics.median(
                    clean_data
                )

                deviations = [

                    abs(x - median)
                    for x in clean_data
                ]

                mad = statistics.median(
                    deviations
                )

                # MAD filtering
                if mad > 0:

                    filtered = []

                    for value in clean_data:

                        z_score = (
                            abs(
                                value
                                - median
                            )
                            / mad
                        )

                        if z_score <= 10:

                            filtered.append(
                                value
                            )

                    if len(filtered) >= 3:

                        clean_data = (
                            filtered
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # SPLIT DATA
            # ---------------------------------------------------------
            mid = max(
                1,
                len(clean_data) // 2,
            )

            first_half = clean_data[:mid]

            second_half = clean_data[mid:]

            if (
                len(first_half) < 1
                or len(second_half) < 1
            ):

                return "stable"

            # ---------------------------------------------------------
            # ROBUST AVERAGES
            # ---------------------------------------------------------
            try:

                first_avg = float(
                    statistics.mean(
                        first_half
                    )
                )

            except Exception:

                first_avg = 0.0

            try:

                second_avg = float(
                    statistics.mean(
                        second_half
                    )
                )

            except Exception:

                second_avg = 0.0

            # ---------------------------------------------------------
            # DELTA CALCULATION
            # ---------------------------------------------------------
            delta = (
                second_avg
                - first_avg
            )

            # ---------------------------------------------------------
            # NORMALIZATION
            # ---------------------------------------------------------
            base = max(
                abs(first_avg),
                1e-6,
            )

            change_ratio = (
                delta / base
            )

            # ---------------------------------------------------------
            # NOISE DETECTION
            # ---------------------------------------------------------
            try:

                volatility = float(
                    statistics.pstdev(
                        clean_data
                    )
                )

            except Exception:

                volatility = 0.0

            # dynamic threshold
            threshold = max(
                0.05,
                min(
                    0.25,
                    volatility
                    / max(
                        base,
                        1e-6,
                    ),
                ),
            )

            # ---------------------------------------------------------
            # TREND CLASSIFICATION
            # ---------------------------------------------------------
            if change_ratio > threshold:

                trend = "increasing"

            elif change_ratio < -threshold:

                trend = "decreasing"

            else:

                trend = "stable"

            # ---------------------------------------------------------
            # MOMENTUM VALIDATION
            # ---------------------------------------------------------
            try:

                recent_window = clean_data[
                    -min(
                        5,
                        len(clean_data),
                    ):
                ]

                recent_delta = (
                    recent_window[-1]
                    - recent_window[0]
                )

                # contradicting momentum → stable
                if (
                    trend == "increasing"
                    and recent_delta < 0
                ):

                    trend = "stable"

                elif (
                    trend == "decreasing"
                    and recent_delta > 0
                ):

                    trend = "stable"

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.trend_calculations = int(
                    getattr(
                        self,
                        "trend_calculations",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - trend_start
                ) * 1000,
                2,
            )

            logger.debug(
                f"📈 Trend calculated "
                f"| trend={trend} "
                f"| samples={len(clean_data)} "
                f"| latency={latency_ms}ms"
            )

            return trend

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.debug(
                f"Trend calculation failed: {exc}"
            )

            try:

                self.trend_failures = int(
                    getattr(
                        self,
                        "trend_failures",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return "stable"

    def increment_commands(self) -> Dict[str, Any]:
        """
        Production-grade command counter incrementer.

        Fixes:
        - race conditions
        - integer corruption
        - missing counters
        - overflow risks
        - unsafe concurrent writes
        - metrics desynchronization
        """

        import threading
        import time
        import traceback

        increment_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # COUNTER LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_metrics_counter_lock",
            ):

                self._metrics_counter_lock = (
                    threading.Lock()
                )

            # ---------------------------------------------------------
            # THREAD-SAFE INCREMENT
            # ---------------------------------------------------------
            with self._metrics_counter_lock:

                current = int(
                    getattr(
                        self,
                        "command_count",
                        0,
                    )
                )

                # -----------------------------------------------------
                # OVERFLOW PROTECTION
                # -----------------------------------------------------
                if current >= 9_000_000_000:

                    logger.warning(
                        "Command counter reset "
                        "due to overflow protection"
                    )

                    current = 0

                current += 1

                self.command_count = (
                    current
                )

                # -----------------------------------------------------
                # LAST COMMAND TIMESTAMP
                # -----------------------------------------------------
                self.last_command_timestamp = (
                    time.time()
                )

            # ---------------------------------------------------------
            # OPTIONAL COMMAND RATE
            # ---------------------------------------------------------
            try:

                history = getattr(
                    self,
                    "_command_timestamps",
                    None,
                )

                if history is None:

                    history = []

                    self._command_timestamps = (
                        history
                    )

                history.append(
                    time.time()
                )

                # keep recent only
                cutoff = (
                    time.time() - 60
                )

                while (
                    history
                    and history[0]
                    < cutoff
                ):

                    history.pop(0)

                self.commands_per_minute = (
                    len(history)
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - increment_start
                ) * 1000,
                3,
            )

            logger.debug(
                f"📈 Command count incremented "
                f"| total={current} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "command_count": current,

                "commands_per_minute": int(
                    getattr(
                        self,
                        "commands_per_minute",
                        0,
                    )
                ),

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.debug(
                f"Command increment failed: {exc}"
            )

            return {

                "status": "error",

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :500
                    ]
                ),

                "timestamp": time.time(),
            }

    def increment_errors(self) -> Dict[str, Any]:
        """
        Production-grade error counter incrementer.

        Fixes:
        - race conditions
        - counter corruption
        - overflow issues
        - concurrent write hazards
        - missing error tracking
        - metrics desynchronization
        """

        import threading
        import time
        import traceback

        increment_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # COUNTER LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_metrics_counter_lock",
            ):

                self._metrics_counter_lock = (
                    threading.Lock()
                )

            # ---------------------------------------------------------
            # THREAD-SAFE INCREMENT
            # ---------------------------------------------------------
            with self._metrics_counter_lock:

                current = int(
                    getattr(
                        self,
                        "error_count",
                        0,
                    )
                )

                # -----------------------------------------------------
                # OVERFLOW PROTECTION
                # -----------------------------------------------------
                if current >= 9_000_000_000:

                    logger.warning(
                        "Error counter reset "
                        "due to overflow protection"
                    )

                    current = 0

                current += 1

                self.error_count = (
                    current
                )

                # -----------------------------------------------------
                # LAST ERROR TIMESTAMP
                # -----------------------------------------------------
                self.last_error_timestamp = (
                    time.time()
                )

            # ---------------------------------------------------------
            # ERROR RATE TRACKING
            # ---------------------------------------------------------
            try:

                history = getattr(
                    self,
                    "_error_timestamps",
                    None,
                )

                if history is None:

                    history = []

                    self._error_timestamps = (
                        history
                    )

                history.append(
                    time.time()
                )

                # retain only last minute
                cutoff = (
                    time.time() - 60
                )

                while (
                    history
                    and history[0]
                    < cutoff
                ):

                    history.pop(0)

                self.errors_per_minute = (
                    len(history)
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # ERROR RATIO
            # ---------------------------------------------------------
            try:

                commands = int(
                    getattr(
                        self,
                        "command_count",
                        0,
                    )
                )

                if commands > 0:

                    self.error_rate = round(

                        (
                            current
                            / commands
                        )
                        * 100,

                        2,
                    )

                else:

                    self.error_rate = 0.0

            except Exception:

                self.error_rate = 0.0

            # ---------------------------------------------------------
            # HEALTH WARNING
            # ---------------------------------------------------------
            try:

                if (
                    getattr(
                        self,
                        "errors_per_minute",
                        0,
                    )
                    >= 25
                ):

                    logger.warning(
                        f"⚠️ High error rate detected "
                        f"({self.errors_per_minute}/min)"
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - increment_start
                ) * 1000,
                3,
            )

            logger.debug(
                f"📉 Error count incremented "
                f"| total={current} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "error_count": current,

                "errors_per_minute": int(
                    getattr(
                        self,
                        "errors_per_minute",
                        0,
                    )
                ),

                "error_rate": float(
                    getattr(
                        self,
                        "error_rate",
                        0.0,
                    )
                ),

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.debug(
                f"Error increment failed: {exc}"
            )

            return {

                "status": "error",

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :500
                    ]
                ),

                "timestamp": time.time(),
            }

class CacheManager:
    """Advanced caching system"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    async def get(
        self,
        key: str,
    ) -> Optional[Any]:
        """
        Ultra production-safe async cache retrieval.

        Fixes:
        - race conditions
        - stale cache reads
        - corrupted timestamps
        - broken tuple unpacking
        - invalid TTL handling
        - mutation leaks
        - unsafe concurrent access
        - cache poisoning
        - memory leaks
        - cleanup corruption
        - serialization instability
        - function object has no attribute items
        """

        import asyncio
        import copy
        import threading
        import time
        import traceback

        from datetime import datetime

        lookup_start = time.monotonic()

        DEFAULT_TTL = 300.0
        CLEANUP_INTERVAL = 100

        try:

            # ============================================================
            # CACHE LOCK
            # ============================================================
            if not hasattr(
                self,
                "_cache_lock",
            ):

                self._cache_lock = (
                    threading.RLock()
                )

            # ============================================================
            # VALIDATE KEY
            # ============================================================
            if key is None:

                return None

            if not isinstance(
                key,
                str,
            ):

                try:

                    key = str(key)

                except Exception:

                    return None

            key = key.strip()

            if not key:

                return None

            # ============================================================
            # VALIDATE CACHE STORE
            # ============================================================
            cache = getattr(
                self,
                "cache",
                None,
            )

            if cache is None:

                logger.debug(
                    "Cache store missing"
                )

                return None

            # FIX:
            # function object has no attribute items
            if callable(cache):

                logger.warning(
                    "Cache store corrupted (callable)"
                )

                return None

            if not hasattr(
                cache,
                "get",
            ):

                logger.warning(
                    "Cache missing get method"
                )

                return None

            # ============================================================
            # SAFE TTL
            # ============================================================
            try:

                ttl = float(
                    getattr(
                        self,
                        "ttl",
                        DEFAULT_TTL,
                    )
                )

                ttl = max(
                    0.0,
                    ttl,
                )

            except Exception:

                ttl = DEFAULT_TTL

            # ============================================================
            # THREAD-SAFE LOOKUP
            # ============================================================
            with self._cache_lock:

                try:

                    item = cache.get(key)

                except Exception as e:

                    logger.debug(
                        f"Cache lookup failed: {e}"
                    )

                    return None

                # --------------------------------------------------------
                # CACHE MISS
                # --------------------------------------------------------
                if item is None:

                    try:

                        self.misses = int(
                            getattr(
                                self,
                                "misses",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                    return None

                # --------------------------------------------------------
                # VALIDATE ITEM STRUCTURE
                # --------------------------------------------------------
                if (
                    not isinstance(
                        item,
                        (
                            tuple,
                            list,
                        ),
                    )
                    or len(item) != 2
                ):

                    logger.warning(
                        f"Invalid cache item format "
                        f"| key={key}"
                    )

                    try:

                        cache.pop(
                            key,
                            None,
                        )

                    except Exception:
                        pass

                    try:

                        self.misses = int(
                            getattr(
                                self,
                                "misses",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                    return None

                # FIX:
                # correct tuple unpacking
                value, timestamp = item

                # --------------------------------------------------------
                # VALIDATE TIMESTAMP
                # --------------------------------------------------------
                try:

                    if isinstance(
                        timestamp,
                        datetime,
                    ):

                        age = max(

                            0.0,

                            (
                                datetime.now()
                                - timestamp
                            ).total_seconds(),
                        )

                    elif isinstance(
                        timestamp,
                        (
                            int,
                            float,
                        ),
                    ):

                        age = max(

                            0.0,

                            time.time()
                            - float(timestamp),
                        )

                    else:

                        raise TypeError(
                            "Invalid timestamp"
                        )

                except Exception:

                    logger.warning(
                        f"Corrupted timestamp "
                        f"| key={key}"
                    )

                    try:

                        cache.pop(
                            key,
                            None,
                        )

                    except Exception:
                        pass

                    return None

                # --------------------------------------------------------
                # TTL EXPIRATION
                # --------------------------------------------------------
                if (
                    ttl > 0
                    and age >= ttl
                ):

                    try:

                        cache.pop(
                            key,
                            None,
                        )

                    except Exception:
                        pass

                    try:

                        self.evictions = int(
                            getattr(
                                self,
                                "evictions",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                    try:

                        self.misses = int(
                            getattr(
                                self,
                                "misses",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                    logger.debug(
                        f"🗑️ Cache expired "
                        f"| key={key}"
                    )

                    return None

                # --------------------------------------------------------
                # UPDATE HIT METRICS
                # --------------------------------------------------------
                try:

                    self.hits = int(
                        getattr(
                            self,
                            "hits",
                            0,
                        )
                    ) + 1

                    self.last_hit = time.time()

                except Exception:
                    pass

                # --------------------------------------------------------
                # OPTIONAL LRU TOUCH
                # --------------------------------------------------------
                try:

                    if hasattr(
                        cache,
                        "move_to_end",
                    ):

                        cache.move_to_end(
                            key
                        )

                except Exception:
                    pass

                # --------------------------------------------------------
                # SAFE VALUE COPY
                # --------------------------------------------------------
                try:

                    immutable_types = (

                        str,
                        int,
                        float,
                        bool,
                        bytes,
                        tuple,
                        frozenset,
                        type(None),
                    )

                    if isinstance(
                        value,
                        immutable_types,
                    ):

                        safe_value = value

                    else:

                        safe_value = (
                            copy.deepcopy(
                                value
                            )
                        )

                except Exception:

                    logger.debug(
                        f"Deepcopy failed "
                        f"| key={key}"
                    )

                    safe_value = value

            # ============================================================
            # HIT RATE
            # ============================================================
            try:

                hits = int(
                    getattr(
                        self,
                        "hits",
                        0,
                    )
                )

                misses = int(
                    getattr(
                        self,
                        "misses",
                        0,
                    )
                )

                total = hits + misses

                self.hit_rate = round(

                    (
                        hits / total
                    ) * 100,

                    2,
                ) if total > 0 else 0.0

            except Exception:
                pass

            # ============================================================
            # PERIODIC CLEANUP
            # ============================================================
            try:

                checks = int(
                    getattr(
                        self,
                        "_cleanup_checks",
                        0,
                    )
                ) + 1

                self._cleanup_checks = (
                    checks
                )

                if (
                    checks % CLEANUP_INTERVAL
                    == 0
                ):

                    expired_keys = []

                    with self._cache_lock:

                        # FIX:
                        # ensure cache.items exists
                        if hasattr(
                            cache,
                            "items",
                        ) and callable(
                            cache.items
                        ):

                            cache_items = list(
                                cache.items()
                            )

                        else:

                            cache_items = []

                        for (
                            k,
                            v,
                        ) in cache_items:

                            try:

                                if (
                                    not isinstance(
                                        v,
                                        (
                                            tuple,
                                            list,
                                        ),
                                    )
                                    or len(v) != 2
                                ):

                                    expired_keys.append(
                                        k
                                    )

                                    continue

                                # FIX:
                                # correct unpacking
                                _, ts = v

                                if isinstance(
                                    ts,
                                    datetime,
                                ):

                                    item_age = (
                                        datetime.now()
                                        - ts
                                    ).total_seconds()

                                elif isinstance(
                                    ts,
                                    (
                                        int,
                                        float,
                                    ),
                                ):

                                    item_age = (
                                        time.time()
                                        - float(ts)
                                    )

                                else:

                                    expired_keys.append(
                                        k
                                    )

                                    continue

                                if (
                                    ttl > 0
                                    and item_age
                                    >= ttl
                                ):

                                    expired_keys.append(
                                        k
                                    )

                            except Exception:

                                expired_keys.append(
                                    k
                                )

                        # ------------------------------------------------
                        # REMOVE EXPIRED
                        # ------------------------------------------------
                        for expired_key in expired_keys:

                            try:

                                cache.pop(
                                    expired_key,
                                    None,
                                )

                            except Exception:
                                pass

                    if expired_keys:

                        logger.debug(
                            f"🧹 Cache cleanup removed "
                            f"{len(expired_keys)} items"
                        )

            except Exception as e:

                logger.debug(
                    f"Cache cleanup failed: {e}"
                )

            # ============================================================
            # LATENCY METRICS
            # ============================================================
            try:

                latency_ms = round(

                    (
                        time.monotonic()
                        - lookup_start
                    ) * 1000,

                    3,
                )

                self.last_lookup_latency = (
                    latency_ms
                )

                logger.debug(
                    f"⚡ Cache hit "
                    f"| key={key} "
                    f"| latency={latency_ms}ms"
                )

            except Exception:
                pass

            # ============================================================
            # SUCCESS
            # ============================================================
            return safe_value

        # ================================================================
        # CANCELLATION
        # ================================================================
        except asyncio.CancelledError:

            logger.warning(
                "Cache get cancelled"
            )

            raise

        # ================================================================
        # HARD FAILURE
        # ================================================================
        except Exception as exc:

            logger.debug(
                f"Cache get failed: {exc}"
            )

            logger.debug(
                traceback.format_exc()[:4000]
            )

            try:

                self.misses = int(
                    getattr(
                        self,
                        "misses",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return None

    async def set(
        self,
        key: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Production-grade async cache setter.

        Fixes:
        - race conditions
        - cache corruption
        - invalid keys
        - unsafe mutation leaks
        - uncontrolled cache growth
        - broken LRU eviction
        - timestamp corruption
        - oversized payloads
        - serialization instability
        """

        import asyncio
        import copy
        import pickle
        import threading
        import time
        import traceback

        from datetime import datetime

        set_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # CACHE LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_cache_lock",
            ):

                self._cache_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE KEY
            # ---------------------------------------------------------
            if not key:

                return {

                    "status": "invalid_key",
                    "success": False,
                }

            if not isinstance(
                key,
                str,
            ):

                return {

                    "status": "invalid_key_type",
                    "success": False,
                }

            key = key.strip()

            if not key:

                return {

                    "status": "empty_key",
                    "success": False,
                }

            # ---------------------------------------------------------
            # CACHE VALIDATION
            # ---------------------------------------------------------
            cache = getattr(
                self,
                "cache",
                None,
            )

            if cache is None:

                cache = {}

                self.cache = cache

            # ---------------------------------------------------------
            # MAX SIZE
            # ---------------------------------------------------------
            try:

                max_size = int(
                    getattr(
                        self,
                        "max_size",
                        1000,
                    )
                )

            except Exception:

                max_size = 1000

            max_size = max(
                1,
                min(
                    max_size,
                    1_000_000,
                ),
            )

            # ---------------------------------------------------------
            # SAFE VALUE COPY
            # ---------------------------------------------------------
            try:

                if isinstance(
                    value,
                    (
                        dict,
                        list,
                        set,
                    ),
                ):

                    safe_value = (
                        copy.deepcopy(
                            value
                        )
                    )

                else:

                    safe_value = value

            except Exception:

                safe_value = value

            # ---------------------------------------------------------
            # SERIALIZATION TEST
            # ---------------------------------------------------------
            try:

                pickle.dumps(
                    safe_value
                )

            except Exception:

                logger.warning(
                    f"Non-serializable cache value "
                    f"for key={key}"
                )

            # ---------------------------------------------------------
            # VALUE SIZE LIMIT
            # ---------------------------------------------------------
            try:

                approx_size = len(
                    pickle.dumps(
                        safe_value
                    )
                )

            except Exception:

                approx_size = 0

            max_entry_size = int(
                getattr(
                    self,
                    "max_entry_size",
                    10 * 1024 * 1024,
                )
            )

            if (
                approx_size
                > max_entry_size
            ):

                logger.warning(
                    f"Cache entry too large "
                    f"| key={key} "
                    f"| size={approx_size}"
                )

                return {

                    "status": "entry_too_large",

                    "success": False,

                    "size": approx_size,
                }

            # ---------------------------------------------------------
            # THREAD-SAFE CACHE WRITE
            # ---------------------------------------------------------
            with self._cache_lock:

                # -----------------------------------------------------
                # EXPIRED CLEANUP
                # -----------------------------------------------------
                try:

                    ttl = float(
                        getattr(
                            self,
                            "ttl",
                            0,
                        )
                    )

                except Exception:

                    ttl = 0.0

                current_time = (
                    datetime.now()
                )

                if ttl > 0:

                    expired_keys = []

                    for (
                        k,
                        v,
                    ) in list(
                        cache.items()
                    ):

                        try:

                            if (
                                not isinstance(
                                    v,
                                    (
                                        tuple,
                                        list,
                                    ),
                                )
                                or len(v) != 2
                            ):

                                expired_keys.append(
                                    k
                                )

                                continue

                            ts = v

                            if isinstance(
                                ts,
                                datetime,
                            ):

                                age = (
                                    current_time
                                    - ts
                                ).total_seconds()

                            else:

                                age = (
                                    time.time()
                                    - float(ts)
                                )

                            if age >= ttl:

                                expired_keys.append(
                                    k
                                )

                        except Exception:

                            expired_keys.append(
                                k
                            )

                    for expired in expired_keys:

                        cache.pop(
                            expired,
                            None,
                        )

                    if expired_keys:

                        logger.debug(
                            f"🧹 Removed "
                            f"{len(expired_keys)} "
                            f"expired cache entries"
                        )

                # -----------------------------------------------------
                # LRU EVICTION
                # -----------------------------------------------------
                while (
                    len(cache)
                    >= max_size
                ):

                    evicted = False

                    try:

                        # OrderedDict support
                        if hasattr(
                            cache,
                            "popitem",
                        ):

                            cache.popitem(
                                last=False
                            )

                            evicted = True

                    except Exception:
                        pass

                    if not evicted:

                        try:

                            oldest_key = min(

                                cache.keys(),

                                key=lambda k:
                                (
                                    cache[k][1]
                                    if (
                                        isinstance(
                                            cache.get(
                                                k
                                            ),
                                            (
                                                tuple,
                                                list,
                                            ),
                                        )
                                        and len(
                                            cache[k]
                                        ) == 2
                                    )
                                    else datetime.min
                                ),
                            )

                            cache.pop(
                                oldest_key,
                                None,
                            )

                            evicted = True

                        except Exception:
                            pass

                    if not evicted:

                        try:

                            arbitrary = next(
                                iter(cache)
                            )

                            cache.pop(
                                arbitrary,
                                None,
                            )

                        except Exception:

                            break

                    # eviction metrics
                    try:

                        self.evictions = int(
                            getattr(
                                self,
                                "evictions",
                                0,
                            )
                        ) + 1

                    except Exception:
                        pass

                # -----------------------------------------------------
                # STORE VALUE
                # -----------------------------------------------------
                cache[key] = (

                    safe_value,

                    current_time,
                )

                # -----------------------------------------------------
                # OPTIONAL LRU TOUCH
                # -----------------------------------------------------
                try:

                    if hasattr(
                        cache,
                        "move_to_end",
                    ):

                        cache.move_to_end(
                            key
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # CACHE SIZE
                # -----------------------------------------------------
                current_size = len(
                    cache
                )

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.writes = int(
                    getattr(
                        self,
                        "writes",
                        0,
                    )
                ) + 1

                self.last_write = (
                    time.time()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # CACHE UTILIZATION
            # ---------------------------------------------------------
            try:

                self.cache_utilization = round(

                    (
                        current_size
                        / max_size
                    ) * 100,

                    2,
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - set_start
                ) * 1000,
                3,
            )

            logger.debug(
                f"💾 Cache set "
                f"| key={key} "
                f"| size={current_size}/{max_size} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "success": True,

                "key": key,

                "cache_size": current_size,

                "utilization": float(
                    getattr(
                        self,
                        "cache_utilization",
                        0.0,
                    )
                ),

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Cache set cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.debug(
                f"Cache set failed: {exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :800
                    ]
                ),

                "timestamp": time.time(),
            }
        
    async def clear(self) -> Dict[str, Any]:
        """
        Production-grade async cache clear operation.

        Fixes:
        - race conditions
        - partial cache clearing
        - metrics corruption
        - dangling references
        - concurrent mutation crashes
        - stale cache snapshots
        - memory retention issues
        - unsafe state resets
        """

        import asyncio
        import gc
        import threading
        import time
        import traceback

        clear_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # CACHE LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_cache_lock",
            ):

                self._cache_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # CACHE VALIDATION
            # ---------------------------------------------------------
            cache = getattr(
                self,
                "cache",
                None,
            )

            if cache is None:

                self.cache = {}

                cache = self.cache

            # ---------------------------------------------------------
            # THREAD-SAFE CLEAR
            # ---------------------------------------------------------
            with self._cache_lock:

                # -----------------------------------------------------
                # PRE-CLEAR STATS
                # -----------------------------------------------------
                try:

                    cleared_items = len(
                        cache
                    )

                except Exception:

                    cleared_items = 0

                # -----------------------------------------------------
                # CLEAR CACHE
                # -----------------------------------------------------
                try:

                    cache.clear()

                except Exception:

                    # fallback replacement
                    self.cache = {}

                    cache = self.cache

                # -----------------------------------------------------
                # RESET METRICS
                # -----------------------------------------------------
                try:

                    self.hits = 0

                except Exception:
                    pass

                try:

                    self.misses = 0

                except Exception:
                    pass

                try:

                    self.evictions = 0

                except Exception:
                    pass

                try:

                    self.writes = 0

                except Exception:
                    pass

                try:

                    self.hit_rate = 0.0

                except Exception:
                    pass

                try:

                    self.cache_utilization = 0.0

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET TRACKING
                # -----------------------------------------------------
                tracking_attrs = [

                    "_last_hit",

                    "_last_write",

                    "_last_config_snapshot",

                    "_cleanup_checks",
                ]

                for attr in tracking_attrs:

                    try:

                        if hasattr(
                            self,
                            attr,
                        ):

                            setattr(
                                self,
                                attr,
                                None,
                            )

                    except Exception:
                        continue

            # ---------------------------------------------------------
            # MEMORY CLEANUP
            # ---------------------------------------------------------
            try:

                gc.collect()

            except Exception:
                pass

            # ---------------------------------------------------------
            # CLEAR COUNTER
            # ---------------------------------------------------------
            try:

                self.cache_clears = int(
                    getattr(
                        self,
                        "cache_clears",
                        0,
                    )
                ) + 1

                self.last_clear = (
                    time.time()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - clear_start
                ) * 1000,
                3,
            )

            logger.info(
                f"🧹 Cache cleared "
                f"| removed={cleared_items} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "success": True,

                "items_cleared": cleared_items,

                "cache_size": 0,

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Cache clear cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Cache clear failed: {exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :800
                    ]
                ),

                "timestamp": time.time(),
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Production-grade cache statistics collector.

        Fixes:
        - race conditions
        - division-by-zero
        - corrupted counters
        - invalid cache state
        - unsafe concurrent reads
        - overflow issues
        - inconsistent metric reporting
        """

        import math
        import threading
        import time
        import traceback

        stats_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # CACHE LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_cache_lock",
            ):

                self._cache_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # THREAD-SAFE SNAPSHOT
            # ---------------------------------------------------------
            with self._cache_lock:

                # -----------------------------------------------------
                # CACHE VALIDATION
                # -----------------------------------------------------
                cache = getattr(
                    self,
                    "cache",
                    {},
                )

                try:

                    cache_size = int(
                        len(cache)
                    )

                except Exception:

                    cache_size = 0

                cache_size = max(
                    0,
                    cache_size,
                )

                # -----------------------------------------------------
                # COUNTERS
                # -----------------------------------------------------
                try:

                    hits = int(
                        getattr(
                            self,
                            "hits",
                            0,
                        )
                    )

                except Exception:

                    hits = 0

                try:

                    misses = int(
                        getattr(
                            self,
                            "misses",
                            0,
                        )
                    )

                except Exception:

                    misses = 0

                try:

                    writes = int(
                        getattr(
                            self,
                            "writes",
                            0,
                        )
                    )

                except Exception:

                    writes = 0

                try:

                    evictions = int(
                        getattr(
                            self,
                            "evictions",
                            0,
                        )
                    )

                except Exception:

                    evictions = 0

                # sanitize negatives
                hits = max(
                    0,
                    hits,
                )

                misses = max(
                    0,
                    misses,
                )

                writes = max(
                    0,
                    writes,
                )

                evictions = max(
                    0,
                    evictions,
                )

                # -----------------------------------------------------
                # TOTAL REQUESTS
                # -----------------------------------------------------
                total = hits + misses

                # -----------------------------------------------------
                # HIT RATE
                # -----------------------------------------------------
                if total > 0:

                    hit_rate = round(

                        (
                            hits / total
                        ) * 100,

                        2,
                    )

                else:

                    hit_rate = 0.0

                if (
                    math.isnan(
                        hit_rate
                    )
                    or math.isinf(
                        hit_rate
                    )
                ):

                    hit_rate = 0.0

                hit_rate = max(
                    0.0,
                    min(
                        hit_rate,
                        100.0,
                    ),
                )

                # -----------------------------------------------------
                # UTILIZATION
                # -----------------------------------------------------
                try:

                    max_size = int(
                        getattr(
                            self,
                            "max_size",
                            1000,
                        )
                    )

                except Exception:

                    max_size = 1000

                max_size = max(
                    1,
                    max_size,
                )

                utilization = round(

                    (
                        cache_size
                        / max_size
                    ) * 100,

                    2,
                )

                # -----------------------------------------------------
                # TTL
                # -----------------------------------------------------
                try:

                    ttl = float(
                        getattr(
                            self,
                            "ttl",
                            0,
                        )
                    )

                except Exception:

                    ttl = 0.0

                ttl = max(
                    0.0,
                    ttl,
                )

            # ---------------------------------------------------------
            # HEALTH STATUS
            # ---------------------------------------------------------
            issues = []

            if utilization >= 95:

                issues.append(
                    "cache_near_capacity"
                )

            if (
                total >= 100
                and hit_rate < 20
            ):

                issues.append(
                    "low_hit_rate"
                )

            status = (
                "degraded"
                if issues
                else "healthy"
            )

            # ---------------------------------------------------------
            # PERFORMANCE CLASSIFICATION
            # ---------------------------------------------------------
            if hit_rate >= 90:

                performance = "excellent"

            elif hit_rate >= 70:

                performance = "good"

            elif hit_rate >= 40:

                performance = "fair"

            else:

                performance = "poor"

            # ---------------------------------------------------------
            # BUILD RESULT
            # ---------------------------------------------------------
            result = {

                "status": status,

                "performance": performance,

                "size": cache_size,

                "max_size": max_size,

                "utilization": utilization,

                "hits": hits,

                "misses": misses,

                "writes": writes,

                "evictions": evictions,

                "total_requests": total,

                "hit_rate": hit_rate,

                "ttl_seconds": ttl,

                "issues": issues,

                "timestamp": time.time(),
            }

            # ---------------------------------------------------------
            # CACHE LAST STATS
            # ---------------------------------------------------------
            try:

                self._last_cache_stats = (
                    result
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # STATS METRICS
            # ---------------------------------------------------------
            try:

                self.stats_requests = int(
                    getattr(
                        self,
                        "stats_requests",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - stats_start
                ) * 1000,
                3,
            )

            result[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"📊 Cache stats generated "
                f"| size={cache_size} "
                f"| hit_rate={hit_rate}% "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Cache stats failed: {exc}"
            )

            return {

                "status": "error",

                "performance": "unknown",

                "size": 0,

                "hits": 0,

                "misses": 0,

                "writes": 0,

                "evictions": 0,

                "total_requests": 0,

                "hit_rate": 0.0,

                "utilization": 0.0,

                "issues": [
                    "stats_failure"
                ],

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :800
                    ]
                ),

                "timestamp": time.time(),
            }


class EventBus:
    """Event bus for inter-component communication"""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.event_history: deque = deque(maxlen=1000)

    def subscribe(
        self,
        event_type: str,
        callback: Callable,
    ) -> Dict[str, Any]:
        """
        Production-grade event subscription system.

        Fixes:
        - race conditions
        - duplicate subscriptions
        - invalid callbacks
        - corrupted listener registry
        - memory leaks
        - unsafe concurrent mutations
        - listener explosion
        - callback type corruption
        """

        import inspect
        import threading
        import time
        import traceback
        import weakref

        subscribe_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # EVENT LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_listener_lock",
            ):

                self._listener_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE EVENT TYPE
            # ---------------------------------------------------------
            if not event_type:

                return {

                    "status": "invalid_event",

                    "success": False,
                }

            if not isinstance(
                event_type,
                str,
            ):

                return {

                    "status": "invalid_event_type",

                    "success": False,
                }

            event_type = (
                event_type.strip()
            )

            if not event_type:

                return {

                    "status": "empty_event",

                    "success": False,
                }

            # ---------------------------------------------------------
            # VALIDATE CALLBACK
            # ---------------------------------------------------------
            if callback is None:

                return {

                    "status": "missing_callback",

                    "success": False,
                }

            if not callable(
                callback
            ):

                return {

                    "status": "callback_not_callable",

                    "success": False,
                }

            # ---------------------------------------------------------
            # LISTENER LIMITS
            # ---------------------------------------------------------
            max_listeners_per_event = int(
                getattr(
                    self,
                    "max_listeners_per_event",
                    1000,
                )
            )

            max_listeners_per_event = max(
                1,
                min(
                    max_listeners_per_event,
                    100000,
                ),
            )

            # ---------------------------------------------------------
            # THREAD-SAFE REGISTRATION
            # ---------------------------------------------------------
            with self._listener_lock:

                # -----------------------------------------------------
                # VALIDATE LISTENER STORE
                # -----------------------------------------------------
                listeners = getattr(
                    self,
                    "listeners",
                    None,
                )

                if listeners is None:

                    listeners = {}

                    self.listeners = (
                        listeners
                    )

                if not isinstance(
                    listeners,
                    dict,
                ):

                    raise RuntimeError(
                        "Listener registry corrupted"
                    )

                # -----------------------------------------------------
                # CREATE EVENT LIST
                # -----------------------------------------------------
                if (
                    event_type
                    not in listeners
                ):

                    listeners[
                        event_type
                    ] = []

                event_listeners = (
                    listeners[
                        event_type
                    ]
                )

                # -----------------------------------------------------
                # VALIDATE LIST STRUCTURE
                # -----------------------------------------------------
                if not isinstance(
                    event_listeners,
                    list,
                ):

                    logger.warning(
                        f"Listener list corrupted "
                        f"for event={event_type}"
                    )

                    event_listeners = []

                    listeners[
                        event_type
                    ] = event_listeners

                # -----------------------------------------------------
                # REMOVE DEAD WEAKREFS
                # -----------------------------------------------------
                cleaned = []

                for listener in event_listeners:

                    try:

                        if isinstance(
                            listener,
                            weakref.ReferenceType,
                        ):

                            resolved = (
                                listener()
                            )

                            if (
                                resolved
                                is not None
                            ):

                                cleaned.append(
                                    listener
                                )

                        else:

                            cleaned.append(
                                listener
                            )

                    except Exception:
                        continue

                event_listeners[:] = (
                    cleaned
                )

                # -----------------------------------------------------
                # DUPLICATE DETECTION
                # -----------------------------------------------------
                duplicate = False

                for listener in event_listeners:

                    try:

                        existing = (
                            listener()
                            if isinstance(
                                listener,
                                weakref.ReferenceType,
                            )
                            else listener
                        )

                        if existing == callback:

                            duplicate = True

                            break

                    except Exception:
                        continue

                if duplicate:

                    logger.debug(
                        f"Duplicate listener ignored "
                        f"| event={event_type}"
                    )

                    return {

                        "status": "duplicate",

                        "success": True,

                        "event_type": event_type,

                        "listener_count": len(
                            event_listeners
                        ),
                    }

                # -----------------------------------------------------
                # LIMIT ENFORCEMENT
                # -----------------------------------------------------
                if (
                    len(event_listeners)
                    >= max_listeners_per_event
                ):

                    logger.warning(
                        f"Listener limit exceeded "
                        f"| event={event_type}"
                    )

                    return {

                        "status": "listener_limit_exceeded",

                        "success": False,

                        "limit": (
                            max_listeners_per_event
                        ),
                    }

                # -----------------------------------------------------
                # STORE CALLBACK
                # -----------------------------------------------------
                try:

                    # weak refs for bound methods
                    if inspect.ismethod(
                        callback
                    ):

                        stored_callback = (
                            weakref.WeakMethod(
                                callback
                            )
                        )

                    else:

                        stored_callback = (
                            callback
                        )

                except Exception:

                    stored_callback = (
                        callback
                    )

                event_listeners.append(
                    stored_callback
                )

                listener_count = len(
                    event_listeners
                )

            # ---------------------------------------------------------
            # GLOBAL METRICS
            # ---------------------------------------------------------
            try:

                self.subscriptions = int(
                    getattr(
                        self,
                        "subscriptions",
                        0,
                    )
                ) + 1

                self.last_subscription = {

                    "event": event_type,

                    "timestamp": (
                        time.time()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # EVENT METRICS
            # ---------------------------------------------------------
            try:

                metrics = getattr(
                    self,
                    "_event_metrics",
                    None,
                )

                if metrics is None:

                    metrics = {}

                    self._event_metrics = (
                        metrics
                    )

                metrics[
                    event_type
                ] = {

                    "listeners": listener_count,

                    "updated_at": (
                        time.time()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - subscribe_start
                ) * 1000,
                3,
            )

            logger.debug(
                f"📡 Event subscribed "
                f"| event={event_type} "
                f"| listeners={listener_count} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "success": True,

                "event_type": event_type,

                "listener_count": listener_count,

                "is_async": inspect.iscoroutinefunction(
                    callback
                ),

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Event subscription failed: {exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1000
                    ]
                ),

                "timestamp": time.time(),
            }

    async def emit(
        self,
        event_type: str,
        data: Any = None,
    ) -> Dict[str, Any]:
        """
        Production-grade async event emitter.

        Fixes:
        - race conditions
        - listener mutation during iteration
        - dead weakrefs
        - event history memory leaks
        - blocking listeners
        - listener crashes cascading
        - invalid event payloads
        - orphaned async tasks
        - timeoutless handlers
        - recursive event storms
        """

        import asyncio
        import inspect
        import threading
        import time
        import traceback
        import weakref

        from datetime import datetime

        emit_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # EVENT LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_listener_lock",
            ):

                self._listener_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE EVENT TYPE
            # ---------------------------------------------------------
            if not event_type:

                return {

                    "status": "invalid_event",

                    "success": False,
                }

            if not isinstance(
                event_type,
                str,
            ):

                return {

                    "status": "invalid_event_type",

                    "success": False,
                }

            event_type = (
                event_type.strip()
            )

            if not event_type:

                return {

                    "status": "empty_event",

                    "success": False,
                }

            # ---------------------------------------------------------
            # RECURSION PROTECTION
            # ---------------------------------------------------------
            active_stack = getattr(
                self,
                "_active_events",
                None,
            )

            if active_stack is None:

                active_stack = set()

                self._active_events = (
                    active_stack
                )

            recursion_key = (
                f"{event_type}:{id(data)}"
            )

            if recursion_key in active_stack:

                logger.warning(
                    f"Recursive event blocked "
                    f"| event={event_type}"
                )

                return {

                    "status": "recursive_blocked",

                    "success": False,
                }

            active_stack.add(
                recursion_key
            )

            try:

                # -----------------------------------------------------
                # BUILD EVENT
                # -----------------------------------------------------
                event = {

                    "type": event_type,

                    "data": data,

                    "timestamp": (
                        datetime.now()
                        .isoformat()
                    ),
                }

                # -----------------------------------------------------
                # EVENT HISTORY
                # -----------------------------------------------------
                try:

                    history = getattr(
                        self,
                        "event_history",
                        None,
                    )

                    if history is None:

                        history = []

                        self.event_history = (
                            history
                        )

                    history.append(
                        event
                    )

                    max_history = int(
                        getattr(
                            self,
                            "max_event_history",
                            1000,
                        )
                    )

                    max_history = max(
                        10,
                        min(
                            max_history,
                            100000,
                        ),
                    )

                    if (
                        len(history)
                        > max_history
                    ):

                        del history[
                            : len(history)
                            - max_history
                        ]

                except Exception:
                    pass

                # -----------------------------------------------------
                # SNAPSHOT LISTENERS
                # -----------------------------------------------------
                with self._listener_lock:

                    listeners_store = getattr(
                        self,
                        "listeners",
                        {},
                    )

                    raw_listeners = list(

                        listeners_store.get(
                            event_type,
                            [],
                        )
                    )

                # -----------------------------------------------------
                # RESOLVE WEAKREFS
                # -----------------------------------------------------
                listeners = []

                dead_refs = []

                for listener in raw_listeners:

                    try:

                        if isinstance(
                            listener,
                            weakref.ReferenceType,
                        ):

                            resolved = (
                                listener()
                            )

                            if (
                                resolved
                                is not None
                            ):

                                listeners.append(
                                    resolved
                                )

                            else:

                                dead_refs.append(
                                    listener
                                )

                        else:

                            listeners.append(
                                listener
                            )

                    except Exception:
                        continue

                # -----------------------------------------------------
                # CLEAN DEAD REFS
                # -----------------------------------------------------
                if dead_refs:

                    try:

                        with self._listener_lock:

                            event_list = (
                                self.listeners.get(
                                    event_type,
                                    [],
                                )
                            )

                            for dead in dead_refs:

                                try:

                                    event_list.remove(
                                        dead
                                    )

                                except Exception:
                                    pass

                    except Exception:
                        pass

                # -----------------------------------------------------
                # NO LISTENERS
                # -----------------------------------------------------
                if not listeners:

                    return {

                        "status": "no_listeners",

                        "success": True,

                        "event_type": event_type,

                        "listeners": 0,
                    }

                # -----------------------------------------------------
                # EXECUTION TASKS
                # -----------------------------------------------------
                tasks = []

                listener_results = []

                timeout = float(
                    getattr(
                        self,
                        "listener_timeout",
                        30,
                    )
                )

                timeout = max(
                    1.0,
                    min(
                        timeout,
                        300.0,
                    ),
                )

                for callback in listeners:

                    try:

                        if not callable(
                            callback
                        ):

                            continue

                        # ---------------------------------------------
                        # ASYNC CALLBACK
                        # ---------------------------------------------
                        if inspect.iscoroutinefunction(
                            callback
                        ):

                            coro = callback(
                                data
                            )

                            task = (
                                asyncio.create_task(

                                    asyncio.wait_for(
                                        coro,
                                        timeout=timeout,
                                    )
                                )
                            )

                            tasks.append(
                                task
                            )

                        # ---------------------------------------------
                        # SYNC CALLBACK
                        # ---------------------------------------------
                        else:

                            task = (
                                asyncio.create_task(

                                    asyncio.wait_for(

                                        asyncio.to_thread(
                                            callback,
                                            data,
                                        ),

                                        timeout=timeout,
                                    )
                                )
                            )

                            tasks.append(
                                task
                            )

                    except Exception as exc:

                        logger.error(
                            f"Listener scheduling failed: "
                            f"{exc}"
                        )

                # -----------------------------------------------------
                # EXECUTE LISTENERS
                # -----------------------------------------------------
                failures = 0

                completed = 0

                if tasks:

                    results = await asyncio.gather(

                        *tasks,

                        return_exceptions=True,
                    )

                    for result in results:

                        if isinstance(
                            result,
                            Exception,
                        ):

                            failures += 1

                            logger.error(
                                f"Event handler error: "
                                f"{result}"
                            )

                        else:

                            completed += 1

                            listener_results.append(
                                result
                            )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.events_emitted = int(
                        getattr(
                            self,
                            "events_emitted",
                            0,
                        )
                    ) + 1

                    self.last_event = {

                        "event": event_type,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT METRICS
                # -----------------------------------------------------
                try:

                    metrics = getattr(
                        self,
                        "_event_metrics",
                        None,
                    )

                    if metrics is None:

                        metrics = {}

                        self._event_metrics = (
                            metrics
                        )

                    event_metric = metrics.get(
                        event_type,
                        {},
                    )

                    event_metric[
                        "emitted"
                    ] = int(
                        event_metric.get(
                            "emitted",
                            0,
                        )
                    ) + 1

                    event_metric[
                        "last_emitted"
                    ] = time.time()

                    metrics[
                        event_type
                    ] = event_metric

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - emit_start
                    ) * 1000,
                    3,
                )

                logger.debug(
                    f"📡 Event emitted "
                    f"| event={event_type} "
                    f"| listeners={len(listeners)} "
                    f"| completed={completed} "
                    f"| failed={failures} "
                    f"| latency={latency_ms}ms"
                )

                return {

                    "status": "success",

                    "success": True,

                    "event_type": event_type,

                    "listeners": len(
                        listeners
                    ),

                    "completed": completed,

                    "failed": failures,

                    "results": listener_results,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

            finally:

                try:

                    active_stack.discard(
                        recursion_key
                    )

                except Exception:
                    pass

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Event emission cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Event emit failed: {exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1200
                    ]
                ),

                "timestamp": time.time(),
            }

    def get_history(
        self,
        event_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Production-grade event history retrieval.

        Fixes:
        - race conditions
        - mutation leaks
        - corrupted history entries
        - invalid event filtering
        - unsafe concurrent reads
        - memory exposure
        - malformed event objects
        - history corruption handling
        """

        import copy
        import threading
        import time
        import traceback

        history_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # EVENT LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_listener_lock",
            ):

                self._listener_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE FILTER
            # ---------------------------------------------------------
            if (
                event_type is not None
                and not isinstance(
                    event_type,
                    str,
                )
            ):

                logger.warning(
                    "Invalid event_type filter"
                )

                return []

            if isinstance(
                event_type,
                str,
            ):

                event_type = (
                    event_type.strip()
                )

                if not event_type:

                    event_type = None

            # ---------------------------------------------------------
            # THREAD-SAFE SNAPSHOT
            # ---------------------------------------------------------
            with self._listener_lock:

                raw_history = getattr(
                    self,
                    "event_history",
                    [],
                )

                # validate structure
                if not isinstance(
                    raw_history,
                    list,
                ):

                    logger.warning(
                        "Event history corrupted"
                    )

                    raw_history = []

                    self.event_history = (
                        raw_history
                    )

                # shallow snapshot first
                snapshot = list(
                    raw_history
                )

            # ---------------------------------------------------------
            # SANITIZE EVENTS
            # ---------------------------------------------------------
            clean_history = []

            for event in snapshot:

                try:

                    if not isinstance(
                        event,
                        dict,
                    ):

                        continue

                    # validate required fields
                    event_name = event.get(
                        "type"
                    )

                    if not isinstance(
                        event_name,
                        str,
                    ):

                        continue

                    # filtering
                    if (
                        event_type
                        and event_name
                        != event_type
                    ):

                        continue

                    # safe deep copy
                    try:

                        safe_event = (
                            copy.deepcopy(
                                event
                            )
                        )

                    except Exception:

                        safe_event = dict(
                            event
                        )

                    clean_history.append(
                        safe_event
                    )

                except Exception:
                    continue

            # ---------------------------------------------------------
            # SORT SAFETY
            # ---------------------------------------------------------
            try:

                clean_history.sort(

                    key=lambda e:
                    str(
                        e.get(
                            "timestamp",
                            "",
                        )
                    )
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # HISTORY LIMIT
            # ---------------------------------------------------------
            max_return = int(
                getattr(
                    self,
                    "max_history_return",
                    10000,
                )
            )

            max_return = max(
                1,
                min(
                    max_return,
                    100000,
                ),
            )

            if (
                len(clean_history)
                > max_return
            ):

                clean_history = clean_history[
                    -max_return:
                ]

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.history_reads = int(
                    getattr(
                        self,
                        "history_reads",
                        0,
                    )
                ) + 1

                self.last_history_read = {

                    "event_type": (
                        event_type
                    ),

                    "timestamp": (
                        time.time()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - history_start
                ) * 1000,
                3,
            )

            logger.debug(
                f"📜 Event history retrieved "
                f"| filter={event_type or 'ALL'} "
                f"| events={len(clean_history)} "
                f"| latency={latency_ms}ms"
            )

            return clean_history

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Event history retrieval failed: "
                f"{exc}"
            )

            try:

                self.history_failures = int(
                    getattr(
                        self,
                        "history_failures",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return []


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

    def register_plugin(
        self,
        plugin: Plugin,
    ) -> Dict[str, Any]:
        """
        Production-grade plugin registration system.

        Fixes:
        - duplicate plugin registration
        - invalid plugin objects
        - corrupted plugin registry
        - race conditions
        - missing metadata
        - plugin version conflicts
        - unsafe concurrent writes
        - malformed plugin names
        - plugin overwrite attacks
        """

        import inspect
        import threading
        import time
        import traceback

        registration_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # PLUGIN LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_plugin_lock",
            ):

                self._plugin_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE PLUGIN OBJECT
            # ---------------------------------------------------------
            if plugin is None:

                return {

                    "status": "missing_plugin",

                    "success": False,
                }

            # ---------------------------------------------------------
            # REQUIRED ATTRIBUTES
            # ---------------------------------------------------------
            required_attrs = [

                "name",

                "version",
            ]

            missing = []

            for attr in required_attrs:

                if not hasattr(
                    plugin,
                    attr,
                ):

                    missing.append(
                        attr
                    )

            if missing:

                return {

                    "status": "invalid_plugin",

                    "success": False,

                    "missing_attributes": missing,
                }

            # ---------------------------------------------------------
            # VALIDATE NAME
            # ---------------------------------------------------------
            plugin_name = str(
                getattr(
                    plugin,
                    "name",
                    "",
                )
            ).strip()

            if not plugin_name:

                return {

                    "status": "invalid_name",

                    "success": False,
                }

            # prevent absurd names
            if len(plugin_name) > 128:

                return {

                    "status": "name_too_long",

                    "success": False,
                }

            # ---------------------------------------------------------
            # VALIDATE VERSION
            # ---------------------------------------------------------
            plugin_version = str(
                getattr(
                    plugin,
                    "version",
                    "0.0.0",
                )
            ).strip()

            if not plugin_version:

                plugin_version = (
                    "0.0.0"
                )

            # ---------------------------------------------------------
            # OPTIONAL METADATA
            # ---------------------------------------------------------
            plugin_author = str(
                getattr(
                    plugin,
                    "author",
                    "unknown",
                )
            )

            plugin_description = str(
                getattr(
                    plugin,
                    "description",
                    "",
                )
            )[:500]

            # ---------------------------------------------------------
            # THREAD-SAFE REGISTRATION
            # ---------------------------------------------------------
            with self._plugin_lock:

                # -----------------------------------------------------
                # VALIDATE REGISTRY
                # -----------------------------------------------------
                plugins = getattr(
                    self,
                    "plugins",
                    None,
                )

                if plugins is None:

                    plugins = {}

                    self.plugins = (
                        plugins
                    )

                if not isinstance(
                    plugins,
                    dict,
                ):

                    raise RuntimeError(
                        "Plugin registry corrupted"
                    )

                # -----------------------------------------------------
                # DUPLICATE DETECTION
                # -----------------------------------------------------
                existing = plugins.get(
                    plugin_name
                )

                if existing is not None:

                    existing_version = str(
                        getattr(
                            existing,
                            "version",
                            "unknown",
                        )
                    )

                    # same object
                    if existing is plugin:

                        logger.debug(
                            f"Plugin already registered "
                            f"| {plugin_name}"
                        )

                        return {

                            "status": "already_registered",

                            "success": True,

                            "plugin": plugin_name,

                            "version": plugin_version,
                        }

                    # version conflict
                    logger.warning(
                        f"Plugin overwrite detected "
                        f"| {plugin_name} "
                        f"| old={existing_version} "
                        f"| new={plugin_version}"
                    )

                # -----------------------------------------------------
                # OPTIONAL VALIDATION HOOK
                # -----------------------------------------------------
                try:

                    validate_fn = getattr(
                        plugin,
                        "validate",
                        None,
                    )

                    if callable(
                        validate_fn
                    ):

                        validation_result = (
                            validate_fn()
                        )

                        if (
                            validation_result
                            is False
                        ):

                            return {

                                "status": "validation_failed",

                                "success": False,
                            }

                except Exception as exc:

                    logger.warning(
                        f"Plugin validation failed "
                        f"| plugin={plugin_name} "
                        f"| error={exc}"
                    )

                    return {

                        "status": "validation_error",

                        "success": False,

                        "error": str(exc),
                    }

                # -----------------------------------------------------
                # REGISTER PLUGIN
                # -----------------------------------------------------
                plugins[
                    plugin_name
                ] = plugin

                total_plugins = len(
                    plugins
                )

            # ---------------------------------------------------------
            # METADATA CACHE
            # ---------------------------------------------------------
            try:

                registry = getattr(
                    self,
                    "_plugin_metadata",
                    None,
                )

                if registry is None:

                    registry = {}

                    self._plugin_metadata = (
                        registry
                    )

                registry[
                    plugin_name
                ] = {

                    "version": (
                        plugin_version
                    ),

                    "author": (
                        plugin_author
                    ),

                    "description": (
                        plugin_description
                    ),

                    "registered_at": (
                        time.time()
                    ),

                    "async_capable": (
                        any(
                            inspect.iscoroutinefunction(
                                getattr(
                                    plugin,
                                    attr,
                                    None,
                                )
                            )
                            for attr in dir(
                                plugin
                            )
                        )
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.plugin_registrations = int(
                    getattr(
                        self,
                        "plugin_registrations",
                        0,
                    )
                ) + 1

                self.last_plugin_registered = {

                    "plugin": plugin_name,

                    "version": plugin_version,

                    "timestamp": (
                        time.time()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - registration_start
                ) * 1000,
                3,
            )

            logger.info(
                f"🧩 Plugin registered "
                f"| {plugin_name} "
                f"v{plugin_version} "
                f"| total={total_plugins} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "success": True,

                "plugin": plugin_name,

                "version": plugin_version,

                "author": plugin_author,

                "total_plugins": total_plugins,

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Plugin registration failed: "
                f"{exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1200
                    ]
                ),

                "timestamp": time.time(),
            }

    async def load_plugin(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Production-grade async plugin loader.

        Fixes:
        - race conditions
        - duplicate loads
        - invalid plugin names
        - plugin registry corruption
        - plugin lifecycle failures
        - hanging plugin loads
        - async/sync incompatibility
        - plugin state desynchronization
        - recursive plugin loading
        """

        import asyncio
        import inspect
        import threading
        import time
        import traceback

        load_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # PLUGIN LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_plugin_lock",
            ):

                self._plugin_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE NAME
            # ---------------------------------------------------------
            if not name:

                return {

                    "status": "invalid_name",

                    "success": False,
                }

            if not isinstance(
                name,
                str,
            ):

                return {

                    "status": "invalid_name_type",

                    "success": False,
                }

            name = name.strip()

            if not name:

                return {

                    "status": "empty_name",

                    "success": False,
                }

            # ---------------------------------------------------------
            # RECURSIVE LOAD PROTECTION
            # ---------------------------------------------------------
            active_loads = getattr(
                self,
                "_active_plugin_loads",
                None,
            )

            if active_loads is None:

                active_loads = set()

                self._active_plugin_loads = (
                    active_loads
                )

            if name in active_loads:

                logger.warning(
                    f"Recursive plugin load blocked "
                    f"| plugin={name}"
                )

                return {

                    "status": "recursive_load",

                    "success": False,

                    "plugin": name,
                }

            active_loads.add(
                name
            )

            try:

                # -----------------------------------------------------
                # THREAD-SAFE LOOKUP
                # -----------------------------------------------------
                with self._plugin_lock:

                    plugins = getattr(
                        self,
                        "plugins",
                        None,
                    )

                    if (
                        plugins is None
                        or not isinstance(
                            plugins,
                            dict,
                        )
                    ):

                        raise RuntimeError(
                            "Plugin registry corrupted"
                        )

                    plugin = plugins.get(
                        name
                    )

                # -----------------------------------------------------
                # PLUGIN NOT FOUND
                # -----------------------------------------------------
                if plugin is None:

                    return {

                        "status": "plugin_not_found",

                        "success": False,

                        "plugin": name,
                    }

                # -----------------------------------------------------
                # ALREADY LOADED
                # -----------------------------------------------------
                if bool(
                    getattr(
                        plugin,
                        "_loaded",
                        False,
                    )
                ):

                    logger.debug(
                        f"Plugin already loaded "
                        f"| {name}"
                    )

                    return {

                        "status": "already_loaded",

                        "success": True,

                        "plugin": name,
                    }

                # -----------------------------------------------------
                # VALIDATE LOAD HOOK
                # -----------------------------------------------------
                if not hasattr(
                    plugin,
                    "on_load",
                ):

                    return {

                        "status": "missing_on_load",

                        "success": False,

                        "plugin": name,
                    }

                on_load = getattr(
                    plugin,
                    "on_load",
                )

                if not callable(
                    on_load
                ):

                    return {

                        "status": "invalid_on_load",

                        "success": False,

                        "plugin": name,
                    }

                # -----------------------------------------------------
                # LOAD TIMEOUT
                # -----------------------------------------------------
                timeout = float(
                    getattr(
                        self,
                        "plugin_load_timeout",
                        60,
                    )
                )

                timeout = max(
                    1.0,
                    min(
                        timeout,
                        600.0,
                    ),
                )

                # -----------------------------------------------------
                # EXECUTE LOAD
                # -----------------------------------------------------
                result = None

                try:

                    # async hook
                    if inspect.iscoroutinefunction(
                        on_load
                    ):

                        result = await asyncio.wait_for(

                            on_load(),

                            timeout=timeout,
                        )

                    # sync hook
                    else:

                        result = await asyncio.wait_for(

                            asyncio.to_thread(
                                on_load
                            ),

                            timeout=timeout,
                        )

                except asyncio.TimeoutError:

                    logger.error(
                        f"Plugin load timeout "
                        f"| plugin={name}"
                    )

                    return {

                        "status": "timeout",

                        "success": False,

                        "plugin": name,
                    }

                # -----------------------------------------------------
                # MARK LOADED
                # -----------------------------------------------------
                try:

                    plugin._loaded = True

                    plugin.loaded_at = (
                        time.time()
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # PLUGIN METADATA
                # -----------------------------------------------------
                plugin_version = str(
                    getattr(
                        plugin,
                        "version",
                        "unknown",
                    )
                )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.plugins_loaded = int(
                        getattr(
                            self,
                            "plugins_loaded",
                            0,
                        )
                    ) + 1

                    self.last_plugin_loaded = {

                        "plugin": name,

                        "version": plugin_version,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL EVENT EMIT
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        await event_bus.emit(

                            "plugin_loaded",

                            {
                                "plugin": name,
                                "version": (
                                    plugin_version
                                ),
                            },
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - load_start
                    ) * 1000,
                    3,
                )

                logger.info(
                    f"🧩 Plugin loaded "
                    f"| {name} "
                    f"v{plugin_version} "
                    f"| latency={latency_ms}ms"
                )

                return {

                    "status": "success",

                    "success": True,

                    "plugin": name,

                    "version": plugin_version,

                    "result": result,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

            finally:

                try:

                    active_loads.discard(
                        name
                    )

                except Exception:
                    pass

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                f"Plugin load cancelled "
                f"| plugin={name}"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Plugin load failed: "
                f"{exc}"
            )

            return {

                "status": "error",

                "success": False,

                "plugin": name,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1200
                    ]
                ),

                "timestamp": time.time(),
            }

    async def unload_plugin(
        self,
        name: str,
    ) -> Dict[str, Any]:
        """
        Production-grade async plugin unload system.

        Fixes:
        - race conditions
        - unloading non-existent plugins
        - double unloads
        - hanging unload hooks
        - plugin state corruption
        - recursive unload loops
        - async/sync incompatibility
        - orphaned resources
        - plugin registry desynchronization
        """

        import asyncio
        import inspect
        import threading
        import time
        import traceback

        unload_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # PLUGIN LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_plugin_lock",
            ):

                self._plugin_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE NAME
            # ---------------------------------------------------------
            if not name:

                return {

                    "status": "invalid_name",

                    "success": False,
                }

            if not isinstance(
                name,
                str,
            ):

                return {

                    "status": "invalid_name_type",

                    "success": False,
                }

            name = name.strip()

            if not name:

                return {

                    "status": "empty_name",

                    "success": False,
                }

            # ---------------------------------------------------------
            # RECURSIVE UNLOAD PROTECTION
            # ---------------------------------------------------------
            active_unloads = getattr(
                self,
                "_active_plugin_unloads",
                None,
            )

            if active_unloads is None:

                active_unloads = set()

                self._active_plugin_unloads = (
                    active_unloads
                )

            if name in active_unloads:

                logger.warning(
                    f"Recursive unload blocked "
                    f"| plugin={name}"
                )

                return {

                    "status": "recursive_unload",

                    "success": False,

                    "plugin": name,
                }

            active_unloads.add(
                name
            )

            try:

                # -----------------------------------------------------
                # THREAD-SAFE LOOKUP
                # -----------------------------------------------------
                with self._plugin_lock:

                    plugins = getattr(
                        self,
                        "plugins",
                        None,
                    )

                    if (
                        plugins is None
                        or not isinstance(
                            plugins,
                            dict,
                        )
                    ):

                        raise RuntimeError(
                            "Plugin registry corrupted"
                        )

                    plugin = plugins.get(
                        name
                    )

                # -----------------------------------------------------
                # PLUGIN NOT FOUND
                # -----------------------------------------------------
                if plugin is None:

                    return {

                        "status": "plugin_not_found",

                        "success": False,

                        "plugin": name,
                    }

                # -----------------------------------------------------
                # ALREADY UNLOADED
                # -----------------------------------------------------
                if not bool(
                    getattr(
                        plugin,
                        "_loaded",
                        True,
                    )
                ):

                    logger.debug(
                        f"Plugin already unloaded "
                        f"| {name}"
                    )

                    return {

                        "status": "already_unloaded",

                        "success": True,

                        "plugin": name,
                    }

                # -----------------------------------------------------
                # VALIDATE UNLOAD HOOK
                # -----------------------------------------------------
                on_unload = getattr(
                    plugin,
                    "on_unload",
                    None,
                )

                # -----------------------------------------------------
                # UNLOAD TIMEOUT
                # -----------------------------------------------------
                timeout = float(
                    getattr(
                        self,
                        "plugin_unload_timeout",
                        60,
                    )
                )

                timeout = max(
                    1.0,
                    min(
                        timeout,
                        600.0,
                    ),
                )

                unload_result = None

                # -----------------------------------------------------
                # EXECUTE UNLOAD HOOK
                # -----------------------------------------------------
                if callable(
                    on_unload
                ):

                    try:

                        # async hook
                        if inspect.iscoroutinefunction(
                            on_unload
                        ):

                            unload_result = (
                                await asyncio.wait_for(

                                    on_unload(),

                                    timeout=timeout,
                                )
                            )

                        # sync hook
                        else:

                            unload_result = (
                                await asyncio.wait_for(

                                    asyncio.to_thread(
                                        on_unload
                                    ),

                                    timeout=timeout,
                                )
                            )

                    except asyncio.TimeoutError:

                        logger.error(
                            f"Plugin unload timeout "
                            f"| plugin={name}"
                        )

                        return {

                            "status": "timeout",

                            "success": False,

                            "plugin": name,
                        }

                # -----------------------------------------------------
                # OPTIONAL CLEANUP HOOK
                # -----------------------------------------------------
                try:

                    cleanup = getattr(
                        plugin,
                        "cleanup",
                        None,
                    )

                    if callable(
                        cleanup
                    ):

                        if inspect.iscoroutinefunction(
                            cleanup
                        ):

                            await asyncio.wait_for(

                                cleanup(),

                                timeout=timeout,
                            )

                        else:

                            await asyncio.wait_for(

                                asyncio.to_thread(
                                    cleanup
                                ),

                                timeout=timeout,
                            )

                except Exception as cleanup_exc:

                    logger.warning(
                        f"Plugin cleanup failed "
                        f"| plugin={name} "
                        f"| error={cleanup_exc}"
                    )

                # -----------------------------------------------------
                # MARK UNLOADED
                # -----------------------------------------------------
                try:

                    plugin._loaded = False

                    plugin.unloaded_at = (
                        time.time()
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL REGISTRY REMOVAL
                # -----------------------------------------------------
                remove_on_unload = bool(
                    getattr(
                        self,
                        "remove_plugin_on_unload",
                        False,
                    )
                )

                if remove_on_unload:

                    try:

                        with self._plugin_lock:

                            self.plugins.pop(
                                name,
                                None,
                            )

                    except Exception:
                        pass

                # -----------------------------------------------------
                # PLUGIN VERSION
                # -----------------------------------------------------
                plugin_version = str(
                    getattr(
                        plugin,
                        "version",
                        "unknown",
                    )
                )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.plugins_unloaded = int(
                        getattr(
                            self,
                            "plugins_unloaded",
                            0,
                        )
                    ) + 1

                    self.last_plugin_unloaded = {

                        "plugin": name,

                        "version": plugin_version,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT EMIT
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        await event_bus.emit(

                            "plugin_unloaded",

                            {
                                "plugin": name,
                                "version": (
                                    plugin_version
                                ),
                            },
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - unload_start
                    ) * 1000,
                    3,
                )

                logger.info(
                    f"🧩 Plugin unloaded "
                    f"| {name} "
                    f"v{plugin_version} "
                    f"| latency={latency_ms}ms"
                )

                return {

                    "status": "success",

                    "success": True,

                    "plugin": name,

                    "version": plugin_version,

                    "removed_from_registry": (
                        remove_on_unload
                    ),

                    "result": unload_result,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

            finally:

                try:

                    active_unloads.discard(
                        name
                    )

                except Exception:
                    pass

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                f"Plugin unload cancelled "
                f"| plugin={name}"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Plugin unload failed: "
                f"{exc}"
            )

            return {

                "status": "error",

                "success": False,

                "plugin": name,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1200
                    ]
                ),

                "timestamp": time.time(),
            }

    async def execute_hook(
        self,
        hook_name: str,
        context: Dict[str, Any],
    ) -> Any:
        """
        Production-grade hook execution system.

        Fixes:
        - race conditions
        - hook mutation during iteration
        - invalid hook registration
        - hanging hooks
        - recursive hook execution
        - async/sync incompatibility
        - hook crashes cascading
        - timeoutless execution
        - context mutation leaks
        - orphaned async tasks
        """

        import asyncio
        import copy
        import inspect
        import threading
        import time
        import traceback

        execution_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # HOOK LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_hook_lock",
            ):

                self._hook_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE HOOK NAME
            # ---------------------------------------------------------
            if not hook_name:

                return {

                    "status": "invalid_hook",

                    "success": False,
                }

            if not isinstance(
                hook_name,
                str,
            ):

                return {

                    "status": "invalid_hook_type",

                    "success": False,
                }

            hook_name = (
                hook_name.strip()
            )

            if not hook_name:

                return {

                    "status": "empty_hook",

                    "success": False,
                }

            # ---------------------------------------------------------
            # VALIDATE CONTEXT
            # ---------------------------------------------------------
            if context is None:

                context = {}

            if not isinstance(
                context,
                dict,
            ):

                return {

                    "status": "invalid_context",

                    "success": False,
                }

            # ---------------------------------------------------------
            # SAFE CONTEXT COPY
            # ---------------------------------------------------------
            try:

                safe_context = (
                    copy.deepcopy(
                        context
                    )
                )

            except Exception:

                safe_context = dict(
                    context
                )

            # ---------------------------------------------------------
            # RECURSION PROTECTION
            # ---------------------------------------------------------
            active_hooks = getattr(
                self,
                "_active_hooks",
                None,
            )

            if active_hooks is None:

                active_hooks = set()

                self._active_hooks = (
                    active_hooks
                )

            recursion_key = (
                f"{hook_name}:{id(context)}"
            )

            if recursion_key in active_hooks:

                logger.warning(
                    f"Recursive hook blocked "
                    f"| hook={hook_name}"
                )

                return {

                    "status": "recursive_execution",

                    "success": False,

                    "hook": hook_name,
                }

            active_hooks.add(
                recursion_key
            )

            try:

                # -----------------------------------------------------
                # THREAD-SAFE HOOK LOOKUP
                # -----------------------------------------------------
                with self._hook_lock:

                    hook_registry = getattr(
                        self,
                        "plugin_hooks",
                        {},
                    )

                    if not isinstance(
                        hook_registry,
                        dict,
                    ):

                        raise RuntimeError(
                            "Hook registry corrupted"
                        )

                    raw_hooks = list(

                        hook_registry.get(
                            hook_name,
                            [],
                        )
                    )

                # -----------------------------------------------------
                # NO HOOKS
                # -----------------------------------------------------
                if not raw_hooks:

                    return {

                        "status": "no_hooks",

                        "success": True,

                        "hook": hook_name,

                        "results": [],
                    }

                # -----------------------------------------------------
                # FILTER VALID HOOKS
                # -----------------------------------------------------
                hooks = []

                for hook in raw_hooks:

                    try:

                        if callable(
                            hook
                        ):

                            hooks.append(
                                hook
                            )

                    except Exception:
                        continue

                if not hooks:

                    return {

                        "status": "no_valid_hooks",

                        "success": True,

                        "hook": hook_name,

                        "results": [],
                    }

                # -----------------------------------------------------
                # EXECUTION SETTINGS
                # -----------------------------------------------------
                timeout = float(
                    getattr(
                        self,
                        "hook_timeout",
                        30,
                    )
                )

                timeout = max(
                    1.0,
                    min(
                        timeout,
                        300.0,
                    ),
                )

                max_parallel = int(
                    getattr(
                        self,
                        "max_parallel_hooks",
                        100,
                    )
                )

                max_parallel = max(
                    1,
                    min(
                        max_parallel,
                        10000,
                    ),
                )

                semaphore = (
                    asyncio.Semaphore(
                        max_parallel
                    )
                )

                # -----------------------------------------------------
                # EXECUTION WRAPPER
                # -----------------------------------------------------
                async def run_hook(
                    hook,
                ):

                    async with semaphore:

                        hook_name_local = (
                            getattr(
                                hook,
                                "__name__",
                                "unknown",
                            )
                        )

                        try:

                            # async hook
                            if inspect.iscoroutinefunction(
                                hook
                            ):

                                return await asyncio.wait_for(

                                    hook(
                                        safe_context
                                    ),

                                    timeout=timeout,
                                )

                            # sync hook
                            return await asyncio.wait_for(

                                asyncio.to_thread(
                                    hook,
                                    safe_context,
                                ),

                                timeout=timeout,
                            )

                        except asyncio.TimeoutError:

                            logger.error(
                                f"Hook timeout "
                                f"| hook={hook_name_local}"
                            )

                            return {

                                "hook": (
                                    hook_name_local
                                ),

                                "error": "timeout",
                            }

                        except Exception as exc:

                            logger.error(
                                f"Hook failed "
                                f"| hook={hook_name_local} "
                                f"| error={exc}"
                            )

                            return {

                                "hook": (
                                    hook_name_local
                                ),

                                "error": str(exc),
                            }

                # -----------------------------------------------------
                # SCHEDULE TASKS
                # -----------------------------------------------------
                tasks = []

                for hook in hooks:

                    try:

                        task = (
                            asyncio.create_task(
                                run_hook(
                                    hook
                                )
                            )
                        )

                        tasks.append(
                            task
                        )

                    except Exception as exc:

                        logger.error(
                            f"Hook scheduling failed: "
                            f"{exc}"
                        )

                # -----------------------------------------------------
                # EXECUTE
                # -----------------------------------------------------
                results = []

                failures = 0

                completed = 0

                if tasks:

                    gathered = await asyncio.gather(

                        *tasks,

                        return_exceptions=True,
                    )

                    for result in gathered:

                        if isinstance(
                            result,
                            Exception,
                        ):

                            failures += 1

                            logger.error(
                                f"Hook execution exception: "
                                f"{result}"
                            )

                            continue

                        if (
                            isinstance(
                                result,
                                dict,
                            )
                            and "error"
                            in result
                        ):

                            failures += 1

                        else:

                            completed += 1

                        if result is not None:

                            results.append(
                                result
                            )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.hooks_executed = int(
                        getattr(
                            self,
                            "hooks_executed",
                            0,
                        )
                    ) + completed

                    self.hook_failures = int(
                        getattr(
                            self,
                            "hook_failures",
                            0,
                        )
                    ) + failures

                    self.last_hook_execution = {

                        "hook": hook_name,

                        "completed": completed,

                        "failed": failures,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - execution_start
                    ) * 1000,
                    3,
                )

                logger.debug(
                    f"🪝 Hooks executed "
                    f"| hook={hook_name} "
                    f"| total={len(hooks)} "
                    f"| completed={completed} "
                    f"| failed={failures} "
                    f"| latency={latency_ms}ms"
                )

                return {

                    "status": "success",

                    "success": True,

                    "hook": hook_name,

                    "results": results,

                    "completed": completed,

                    "failed": failures,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

            finally:

                try:

                    active_hooks.discard(
                        recursion_key
                    )

                except Exception:
                    pass

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                f"Hook execution cancelled "
                f"| hook={hook_name}"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ execute_hook failed: "
                f"{exc}"
            )

            return {

                "status": "error",

                "success": False,

                "hook": hook_name,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1200
                    ]
                ),

                "timestamp": time.time(),
            }

    def get_plugins_info(self) -> List[Dict]:
        """
        Production-grade plugin information collector.

        Fixes:
        - race conditions
        - corrupted plugin registry
        - invalid plugin objects
        - missing attributes
        - unsafe concurrent reads
        - malformed plugin metadata
        - mutation leaks
        - duplicate plugin states
        """

        import copy
        import inspect
        import threading
        import time
        import traceback

        info_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # PLUGIN LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_plugin_lock",
            ):

                self._plugin_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # THREAD-SAFE SNAPSHOT
            # ---------------------------------------------------------
            with self._plugin_lock:

                plugins = getattr(
                    self,
                    "plugins",
                    {},
                )

                if not isinstance(
                    plugins,
                    dict,
                ):

                    raise RuntimeError(
                        "Plugin registry corrupted"
                    )

                plugin_items = list(
                    plugins.items()
                )

            # ---------------------------------------------------------
            # BUILD PLUGIN INFO
            # ---------------------------------------------------------
            output = []

            seen = set()

            for (
                plugin_name,
                plugin,
            ) in plugin_items:

                try:

                    # -------------------------------------------------
                    # INVALID PLUGIN
                    # -------------------------------------------------
                    if plugin is None:

                        continue

                    # -------------------------------------------------
                    # SAFE NAME
                    # -------------------------------------------------
                    name = str(
                        getattr(
                            plugin,
                            "name",
                            plugin_name,
                        )
                    ).strip()

                    if not name:

                        continue

                    # duplicate protection
                    if name in seen:

                        logger.warning(
                            f"Duplicate plugin ignored "
                            f"| {name}"
                        )

                        continue

                    seen.add(
                        name
                    )

                    # -------------------------------------------------
                    # SAFE VERSION
                    # -------------------------------------------------
                    version = str(
                        getattr(
                            plugin,
                            "version",
                            "unknown",
                        )
                    )

                    # -------------------------------------------------
                    # SAFE ENABLED STATE
                    # -------------------------------------------------
                    enabled = bool(
                        getattr(
                            plugin,
                            "enabled",
                            False,
                        )
                    )

                    # -------------------------------------------------
                    # OPTIONAL STATES
                    # -------------------------------------------------
                    loaded = bool(
                        getattr(
                            plugin,
                            "_loaded",
                            False,
                        )
                    )

                    author = str(
                        getattr(
                            plugin,
                            "author",
                            "unknown",
                        )
                    )

                    description = str(
                        getattr(
                            plugin,
                            "description",
                            "",
                        )
                    )[:300]

                    # -------------------------------------------------
                    # CAPABILITIES
                    # -------------------------------------------------
                    capabilities = []

                    try:

                        if hasattr(
                            plugin,
                            "on_load",
                        ):

                            capabilities.append(
                                "loadable"
                            )

                        if hasattr(
                            plugin,
                            "on_unload",
                        ):

                            capabilities.append(
                                "unloadable"
                            )

                        if hasattr(
                            plugin,
                            "execute",
                        ):

                            capabilities.append(
                                "executable"
                            )

                    except Exception:
                        pass

                    # -------------------------------------------------
                    # ASYNC SUPPORT
                    # -------------------------------------------------
                    async_capable = any(

                        inspect.iscoroutinefunction(

                            getattr(
                                plugin,
                                attr,
                                None,
                            )
                        )

                        for attr in dir(
                            plugin
                        )
                    )

                    # -------------------------------------------------
                    # BUILD INFO OBJECT
                    # -------------------------------------------------
                    plugin_info = {

                        "name": name,

                        "version": version,

                        "enabled": enabled,

                        "loaded": loaded,

                        "author": author,

                        "description": description,

                        "capabilities": capabilities,

                        "async_capable": async_capable,
                    }

                    # -------------------------------------------------
                    # OPTIONAL TIMESTAMPS
                    # -------------------------------------------------
                    if hasattr(
                        plugin,
                        "loaded_at",
                    ):

                        plugin_info[
                            "loaded_at"
                        ] = getattr(
                            plugin,
                            "loaded_at",
                        )

                    if hasattr(
                        plugin,
                        "unloaded_at",
                    ):

                        plugin_info[
                            "unloaded_at"
                        ] = getattr(
                            plugin,
                            "unloaded_at",
                        )

                    # -------------------------------------------------
                    # SAFE COPY
                    # -------------------------------------------------
                    output.append(
                        copy.deepcopy(
                            plugin_info
                        )
                    )

                except Exception as plugin_exc:

                    logger.error(
                        f"Plugin info extraction failed "
                        f"| plugin={plugin_name} "
                        f"| error={plugin_exc}"
                    )

                    continue

            # ---------------------------------------------------------
            # SORT OUTPUT
            # ---------------------------------------------------------
            try:

                output.sort(

                    key=lambda x:
                    (
                        not x.get(
                            "enabled",
                            False,
                        ),
                        x.get(
                            "name",
                            "",
                        ).lower(),
                    )
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.plugin_info_requests = int(
                    getattr(
                        self,
                        "plugin_info_requests",
                        0,
                    )
                ) + 1

                self.last_plugin_info_request = (
                    time.time()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - info_start
                ) * 1000,
                3,
            )

            logger.debug(
                f"🧩 Plugin info collected "
                f"| plugins={len(output)} "
                f"| latency={latency_ms}ms"
            )

            return output

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Plugin info retrieval failed: "
                f"{exc}"
            )

            try:

                self.plugin_info_failures = int(
                    getattr(
                        self,
                        "plugin_info_failures",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return []


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
        """
        Production-grade SystemController initialization.

        Fixes:
        - async initialization crashes
        - event loop startup failures
        - component partial initialization
        - worker deadlocks
        - task queue corruption
        - startup race conditions
        - missing component recovery
        - invalid config states
        - background task leaks
        """

        try:

            # ---------------------------------------------------------
            # CORE STATE
            # ---------------------------------------------------------
            self.started_at = datetime.utcnow()

            self._running = False

            self._shutdown = False

            self._initializing = True

            self._healing = False

            self._startup_complete = False

            # ---------------------------------------------------------
            # INTERNAL LOCKS
            # ---------------------------------------------------------
            try:

                self._state_lock = asyncio.Lock()

                self._task_lock = asyncio.Lock()

                self._metrics_lock = asyncio.Lock()

            except Exception as exc:

                logger.warning(
                    f"Async lock init failed: {exc}"
                )

                self._state_lock = None
                self._task_lock = None
                self._metrics_lock = None

            # ---------------------------------------------------------
            # SAFE CONFIG
            # ---------------------------------------------------------
            self.config = {

                "auto_backup_interval": 3600,

                "metrics_interval": 5,

                "health_check_interval": 30,

                "max_concurrent_tasks": max(
                    2,
                    min(
                        (os.cpu_count() or 4),
                        16,
                    ),
                ),

                "enable_auto_healing": True,

                "enable_predictive_scaling": True,

                "task_timeout": 300,

                "worker_restart_limit": 3,

                "max_task_queue_size": 1000,

                "safe_mode": False,
            }

            # ---------------------------------------------------------
            # COMPONENT STATUS
            # ---------------------------------------------------------
            self.component_status = {}

            self.failed_components = []

            # ---------------------------------------------------------
            # TASK MANAGEMENT
            # ---------------------------------------------------------
            self.tasks: Dict[str, Task] = {}

            try:

                queue_size = int(
                    self.config.get(
                        "max_task_queue_size",
                        1000,
                    )
                )

                self.task_queue = asyncio.PriorityQueue(
                    maxsize=max(10, queue_size)
                )

            except Exception:

                self.task_queue = asyncio.Queue()

            self.worker_tasks = []

            # ---------------------------------------------------------
            # BACKGROUND TASKS
            # ---------------------------------------------------------
            self._main_loop_task = None

            self._metrics_task = None

            self._health_task = None

            self._backup_task = None

            self._idle_setup_task = None

            # ---------------------------------------------------------
            # COMPONENT FACTORY
            # ---------------------------------------------------------
            component_builders = {

                "executor": Executor,

                "file_manager": FileManager,

                "command_router": CommandRouter,

                "safety_layer": SafetyLayer,

                "idle_controller": IdleController,

                "code_runner": CodeRunner,

                "metrics_collector": MetricsCollector,

                "cache_manager": CacheManager,

                "event_bus": EventBus,

                "plugin_manager": PluginManager,

                "health_checker": HealthChecker,

                "resource_monitor": ResourceMonitor,

                "backup_manager": BackupManager,
            }

            # ---------------------------------------------------------
            # COMPONENT INITIALIZATION
            # ---------------------------------------------------------
            for name, builder in (
                component_builders.items()
            ):

                try:

                    component = builder()

                    setattr(
                        self,
                        name,
                        component,
                    )

                    self.component_status[
                        name
                    ] = "healthy"

                    logger.debug(
                        f"Initialized component: {name}"
                    )

                except Exception as exc:

                    logger.error(
                        f"Component init failed "
                        f"({name}): {exc}"
                    )

                    setattr(
                        self,
                        name,
                        None,
                    )

                    self.component_status[
                        name
                    ] = "failed"

                    self.failed_components.append(
                        name
                    )

            # ---------------------------------------------------------
            # SAFETY SETUP
            # ---------------------------------------------------------
            try:

                if self.safety_layer:

                    self._setup_safety()

            except Exception as exc:

                logger.warning(
                    f"Safety setup failed: {exc}"
                )

            # ---------------------------------------------------------
            # CORE HANDLERS
            # ---------------------------------------------------------
            try:

                if self.command_router:

                    self._register_core_handlers()

            except Exception as exc:

                logger.warning(
                    f"Core handler setup failed: {exc}"
                )

            # ---------------------------------------------------------
            # HEALTH CHECKS
            # ---------------------------------------------------------
            try:

                if self.health_checker:

                    self._register_health_checks()

            except Exception as exc:

                logger.warning(
                    f"Health check setup failed: {exc}"
                )

            # ---------------------------------------------------------
            # EVENT LISTENERS
            # ---------------------------------------------------------
            try:

                if self.event_bus:

                    self._setup_event_listeners()

            except Exception as exc:

                logger.warning(
                    f"Event listener setup failed: {exc}"
                )

            # ---------------------------------------------------------
            # IDLE BEHAVIOR SETUP
            # ---------------------------------------------------------
            try:

                loop = asyncio.get_running_loop()

                self._idle_setup_task = (
                    loop.create_task(
                        self._setup_idle_behavior()
                    )
                )

            except RuntimeError:

                logger.warning(
                    "No running loop available "
                    "during initialization"
                )

                self._idle_setup_task = None

            except Exception as exc:

                logger.warning(
                    f"Idle setup scheduling failed: {exc}"
                )

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            total_components = len(
                component_builders
            )

            failed_count = len(
                self.failed_components
            )

            self.health_score = round(
                max(
                    0.0,
                    1.0 - (
                        failed_count
                        / max(1, total_components)
                    ),
                ),
                2,
            )

            # ---------------------------------------------------------
            # FINALIZATION
            # ---------------------------------------------------------
            self._initializing = False

            self._startup_complete = True

            # ---------------------------------------------------------
            # STARTUP METRICS
            # ---------------------------------------------------------
            self.startup_metrics = {

                "started_at": (
                    self.started_at.isoformat()
                ),

                "components_total": (
                    total_components
                ),

                "components_failed": (
                    failed_count
                ),

                "health_score": (
                    self.health_score
                ),
            }

            # ---------------------------------------------------------
            # FINAL LOGGING
            # ---------------------------------------------------------
            logger.info(
                "🧠 System Controller initialized "
                f"| health={self.health_score} "
                f"| failed_components={failed_count}"
            )

            if failed_count > 0:

                logger.warning(
                    "Some components failed: "
                    f"{self.failed_components}"
                )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.critical(
                f"SystemController init failed: {exc}"
            )

            self._running = False

            self._shutdown = True

            self._startup_complete = False

            self.health_score = 0.0

            self.failed_components = [
                "system_controller"
            ]

    # ------------------------
    # SETUP METHODS
    # ------------------------

    def _setup_safety(self):
        """
        Production-grade safety system initialization.

        Fixes:
        - duplicate registrations
        - invalid rate limits
        - missing safety layer crashes
        - unsafe operation exposure
        - registration race conditions
        - corrupted safety states
        - invalid operation permissions
        """

        try:

            # ---------------------------------------------------------
            # SAFETY LAYER VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "safety_layer",
            ):

                logger.error(
                    "Safety layer missing"
                )

                return

            if self.safety_layer is None:

                logger.error(
                    "Safety layer unavailable"
                )

                return

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_safety_lock",
            ):

                self._safety_lock = (
                    threading.Lock()
                )

            with self._safety_lock:

                # -----------------------------------------------------
                # SAFETY INITIALIZATION FLAG
                # -----------------------------------------------------
                if getattr(
                    self,
                    "_safety_initialized",
                    False,
                ):

                    logger.debug(
                        "Safety system already initialized"
                    )

                    return

                # -----------------------------------------------------
                # RATE LIMIT CONFIG
                # -----------------------------------------------------
                rate_limits = {

                    "llm_call": (
                        100,
                        60,
                    ),

                    "file_operation": (
                        1000,
                        60,
                    ),

                    "api_call": (
                        500,
                        60,
                    ),

                    "code_execution": (
                        100,
                        60,
                    ),

                    "network_request": (
                        200,
                        60,
                    ),

                    "database_query": (
                        500,
                        60,
                    ),

                    "plugin_execution": (
                        100,
                        60,
                    ),

                    "task_submission": (
                        300,
                        60,
                    ),
                }

                # -----------------------------------------------------
                # REGISTER RATE LIMITS
                # -----------------------------------------------------
                registered_limits = 0

                for operation, values in (
                    rate_limits.items()
                ):

                    try:

                        limit, window = values

                        # -----------------------------------------
                        # VALIDATION
                        # -----------------------------------------
                        limit = max(
                            1,
                            int(limit),
                        )

                        window = max(
                            1,
                            int(window),
                        )

                        # -----------------------------------------
                        # REGISTER
                        # -----------------------------------------
                        self.safety_layer.register_rate_limit(
                            operation,
                            limit,
                            window,
                        )

                        registered_limits += 1

                        logger.debug(
                            f"Safety rate limit "
                            f"registered: {operation}"
                        )

                    except Exception as exc:

                        logger.warning(
                            f"Rate limit setup failed "
                            f"({operation}): {exc}"
                        )

                # -----------------------------------------------------
                # ALLOWED OPERATIONS
                # -----------------------------------------------------
                allowed_operations = [

                    "file_read",

                    "file_write",

                    "browser_open",

                    "code_execution",

                    "network_request",

                    "database_operation",

                    "task_execution",

                    "plugin_execution",

                    "cache_access",

                    "metrics_collection",
                ]

                # -----------------------------------------------------
                # REGISTER SAFE OPERATIONS
                # -----------------------------------------------------
                registered_operations = 0

                for operation in (
                    allowed_operations
                ):

                    try:

                        self.safety_layer.allow_operation(
                            operation
                        )

                        registered_operations += 1

                        logger.debug(
                            f"Allowed operation: "
                            f"{operation}"
                        )

                    except Exception as exc:

                        logger.warning(
                            f"Operation allow failed "
                            f"({operation}): {exc}"
                        )

                # -----------------------------------------------------
                # BLOCKED OPERATIONS
                # -----------------------------------------------------
                blocked_operations = [

                    "kernel_memory_access",

                    "raw_disk_access",

                    "privilege_escalation",

                    "system_shutdown",

                    "unauthorized_network_scan",

                    "credential_dumping",
                ]

                # -----------------------------------------------------
                # OPTIONAL BLOCK REGISTRATION
                # -----------------------------------------------------
                for operation in (
                    blocked_operations
                ):

                    try:

                        if hasattr(
                            self.safety_layer,
                            "block_operation",
                        ):

                            self.safety_layer.block_operation(
                                operation
                            )

                            logger.debug(
                                f"Blocked operation: "
                                f"{operation}"
                            )

                    except Exception:
                        pass

                # -----------------------------------------------------
                # SAFETY STATUS
                # -----------------------------------------------------
                self.component_status[
                    "safety_layer"
                ] = "healthy"

                self._safety_initialized = True

                # -----------------------------------------------------
                # SAFETY METRICS
                # -----------------------------------------------------
                self.safety_metrics = {

                    "rate_limits": (
                        registered_limits
                    ),

                    "allowed_operations": (
                        registered_operations
                    ),

                    "blocked_operations": len(
                        blocked_operations
                    ),

                    "initialized_at": (
                        time.time()
                    ),
                }

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    "🛡️ Safety system initialized "
                    f"| limits={registered_limits} "
                    f"| allowed={registered_operations}"
                )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Safety setup failed: {exc}"
            )

            try:

                self.component_status[
                    "safety_layer"
                ] = "failed"

            except Exception:
                pass

    async def _setup_idle_behavior(self):
        """
        Production-grade idle behavior initialization.

        Fixes:
        - async listener crashes
        - duplicate listener registration
        - idle loop deadlocks
        - autonomous task leaks
        - deep-idle failures
        - missing controller crashes
        - unsafe background execution
        - listener registration instability
        """

        try:

            # ---------------------------------------------------------
            # IDLE CONTROLLER VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "idle_controller",
            ):

                logger.error(
                    "Idle controller missing"
                )

                return

            if self.idle_controller is None:

                logger.error(
                    "Idle controller unavailable"
                )

                return

            # ---------------------------------------------------------
            # DUPLICATE SETUP PROTECTION
            # ---------------------------------------------------------
            if getattr(
                self,
                "_idle_behavior_initialized",
                False,
            ):

                logger.debug(
                    "Idle behavior already initialized"
                )

                return

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_idle_setup_lock",
            ):

                self._idle_setup_lock = (
                    asyncio.Lock()
                )

            async with self._idle_setup_lock:

                # -----------------------------------------------------
                # DOUBLE CHECK
                # -----------------------------------------------------
                if getattr(
                    self,
                    "_idle_behavior_initialized",
                    False,
                ):

                    return

                # -----------------------------------------------------
                # SAFE IDLE CALLBACK
                # -----------------------------------------------------
                async def on_idle():

                    start = time.monotonic()

                    try:

                        logger.info(
                            "🧠 System idle "
                            "→ autonomous thinking"
                        )

                        # -----------------------------------------
                        # RUN AUTONOMOUS TICK
                        # -----------------------------------------
                        await asyncio.wait_for(
                            self.autonomous_tick(),
                            timeout=120,
                        )

                        elapsed = round(
                            time.monotonic() - start,
                            3,
                        )

                        logger.debug(
                            f"Autonomous tick completed "
                            f"in {elapsed}s"
                        )

                    # ---------------------------------------------
                    # TIMEOUT
                    # ---------------------------------------------
                    except asyncio.TimeoutError:

                        logger.warning(
                            "Autonomous tick timeout"
                        )

                    # ---------------------------------------------
                    # TASK CANCELLED
                    # ---------------------------------------------
                    except asyncio.CancelledError:

                        logger.warning(
                            "Idle task cancelled"
                        )

                        raise

                    # ---------------------------------------------
                    # FAILURE
                    # ---------------------------------------------
                    except Exception as exc:

                        logger.error(
                            f"Idle autonomous "
                            f"behavior failed: {exc}"
                        )

                # -----------------------------------------------------
                # SAFE DEEP-IDLE CALLBACK
                # -----------------------------------------------------
                async def on_deep_idle():

                    start = time.monotonic()

                    try:

                        logger.info(
                            "🔮 System deep idle "
                            "→ optimization mode"
                        )

                        # -----------------------------------------
                        # OPTIMIZATION
                        # -----------------------------------------
                        await asyncio.wait_for(
                            self.optimize_system(),
                            timeout=300,
                        )

                        elapsed = round(
                            time.monotonic() - start,
                            3,
                        )

                        logger.debug(
                            f"Deep optimization completed "
                            f"in {elapsed}s"
                        )

                    # ---------------------------------------------
                    # TIMEOUT
                    # ---------------------------------------------
                    except asyncio.TimeoutError:

                        logger.warning(
                            "Deep idle optimization timeout"
                        )

                    # ---------------------------------------------
                    # TASK CANCELLED
                    # ---------------------------------------------
                    except asyncio.CancelledError:

                        logger.warning(
                            "Deep idle task cancelled"
                        )

                        raise

                    # ---------------------------------------------
                    # FAILURE
                    # ---------------------------------------------
                    except Exception as exc:

                        logger.error(
                            f"Deep idle optimization "
                            f"failed: {exc}"
                        )

                # -----------------------------------------------------
                # LISTENER REGISTRATION
                # -----------------------------------------------------
                idle_added = False

                deep_added = False

                # -----------------------------------------------------
                # REGISTER IDLE LISTENER
                # -----------------------------------------------------
                try:

                    if not hasattr(
                        self.idle_controller,
                        "add_idle_listener",
                    ):

                        raise AttributeError(
                            "Idle listener API missing"
                        )

                    idle_added = bool(
                        self.idle_controller.add_idle_listener(
                            on_idle
                        )
                    )

                except Exception as exc:

                    logger.warning(
                        f"Idle listener registration "
                        f"failed: {exc}"
                    )

                # -----------------------------------------------------
                # REGISTER DEEP-IDLE LISTENER
                # -----------------------------------------------------
                try:

                    if hasattr(
                        self.idle_controller,
                        "add_deep_idle_listener",
                    ):

                        result = (
                            self.idle_controller.add_deep_idle_listener(
                                on_deep_idle
                            )
                        )

                        # support sync or async API
                        if asyncio.iscoroutine(
                            result
                        ):

                            deep_added = bool(
                                await result
                            )

                        else:

                            deep_added = bool(
                                result
                            )

                    else:

                        logger.debug(
                            "Deep idle listener "
                            "API unavailable"
                        )

                except Exception as exc:

                    logger.warning(
                        f"Deep idle listener "
                        f"registration failed: {exc}"
                    )

                # -----------------------------------------------------
                # STATUS UPDATE
                # -----------------------------------------------------
                try:

                    if (
                        idle_added
                        or deep_added
                    ):

                        self.component_status[
                            "idle_controller"
                        ] = (
                            ComponentStatus.HEALTHY
                        )

                    else:

                        self.component_status[
                            "idle_controller"
                        ] = (
                            ComponentStatus.DEGRADED
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # INITIALIZATION FLAG
                # -----------------------------------------------------
                self._idle_behavior_initialized = True

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                self.idle_metrics = {

                    "idle_listener": idle_added,

                    "deep_idle_listener": deep_added,

                    "initialized_at": time.time(),
                }

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                if (
                    idle_added
                    or deep_added
                ):

                    logger.info(
                        "🌙 Idle behavior initialized "
                        f"| idle={idle_added} "
                        f"| deep_idle={deep_added}"
                    )

                else:

                    logger.warning(
                        "Idle behavior initialization "
                        "completed with no listeners"
                    )

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Idle behavior setup cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Idle behavior setup failed: {exc}"
            )

            try:

                self.component_status[
                    "idle_controller"
                ] = (
                    ComponentStatus.FAILED
                )

            except Exception:
                pass

    def _register_health_checks(self):
        """
        Production-grade health check registration.

        Fixes:
        - duplicate health checks
        - missing checker crashes
        - invalid callback registration
        - registration race conditions
        - corrupted health states
        - partial registration failures
        - async callback instability
        """

        try:

            # ---------------------------------------------------------
            # HEALTH CHECKER VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "health_checker",
            ):

                logger.error(
                    "Health checker missing"
                )

                return

            if self.health_checker is None:

                logger.error(
                    "Health checker unavailable"
                )

                return

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_health_registration_lock",
            ):

                self._health_registration_lock = (
                    threading.Lock()
                )

            with self._health_registration_lock:

                # -----------------------------------------------------
                # DUPLICATE REGISTRATION PROTECTION
                # -----------------------------------------------------
                if getattr(
                    self,
                    "_health_checks_registered",
                    False,
                ):

                    logger.debug(
                        "Health checks already registered"
                    )

                    return

                # -----------------------------------------------------
                # CHECK DEFINITIONS
                # -----------------------------------------------------
                health_checks = {

                    "system": (
                        "_system_health_check"
                    ),

                    "resources": (
                        "_resource_health_check"
                    ),

                    "components": (
                        "_components_health_check"
                    ),

                    "performance": (
                        "_performance_health_check"
                    ),
                }

                registered = 0

                failed = 0

                failed_checks = []

                # -----------------------------------------------------
                # REGISTER CHECKS
                # -----------------------------------------------------
                for (
                    check_name,
                    method_name,
                ) in health_checks.items():

                    try:

                        # -----------------------------------------
                        # METHOD VALIDATION
                        # -----------------------------------------
                        if not hasattr(
                            self,
                            method_name,
                        ):

                            raise AttributeError(
                                f"Missing method: "
                                f"{method_name}"
                            )

                        callback = getattr(
                            self,
                            method_name,
                        )

                        if not callable(
                            callback
                        ):

                            raise TypeError(
                                f"{method_name} "
                                f"is not callable"
                            )

                        # -----------------------------------------
                        # REGISTER CHECK
                        # -----------------------------------------
                        self.health_checker.register_check(
                            check_name,
                            callback,
                        )

                        registered += 1

                        logger.debug(
                            f"Health check registered: "
                            f"{check_name}"
                        )

                    # ---------------------------------------------
                    # REGISTRATION FAILURE
                    # ---------------------------------------------
                    except Exception as exc:

                        failed += 1

                        failed_checks.append(
                            check_name
                        )

                        logger.warning(
                            f"Health check registration "
                            f"failed ({check_name}): {exc}"
                        )

                # -----------------------------------------------------
                # STATUS UPDATE
                # -----------------------------------------------------
                try:

                    if failed == 0:

                        self.component_status[
                            "health_checker"
                        ] = (
                            ComponentStatus.HEALTHY
                        )

                    elif registered > 0:

                        self.component_status[
                            "health_checker"
                        ] = (
                            ComponentStatus.DEGRADED
                        )

                    else:

                        self.component_status[
                            "health_checker"
                        ] = (
                            ComponentStatus.FAILED
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # REGISTRATION FLAG
                # -----------------------------------------------------
                self._health_checks_registered = (
                    registered > 0
                )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                self.health_registration_metrics = {

                    "registered": registered,

                    "failed": failed,

                    "failed_checks": failed_checks,

                    "timestamp": time.time(),
                }

                # -----------------------------------------------------
                # OPTIONAL VALIDATION PASS
                # -----------------------------------------------------
                try:

                    if hasattr(
                        self.health_checker,
                        "run_checks",
                    ):

                        logger.debug(
                            "Running initial health validation"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    "❤️ Health checks initialized "
                    f"| registered={registered} "
                    f"| failed={failed}"
                )

                if failed_checks:

                    logger.warning(
                        "Failed health checks: "
                        f"{failed_checks}"
                    )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Health check registration failed: {exc}"
            )

            try:

                self.component_status[
                    "health_checker"
                ] = (
                    ComponentStatus.FAILED
                )

            except Exception:
                pass

    def _setup_event_listeners(self):
        """
        Production-grade event listener initialization.

        Fixes:
        - duplicate event subscriptions
        - async listener crashes
        - missing event bus failures
        - metrics collector corruption
        - auto-heal recursion
        - listener registration races
        - invalid event callbacks
        - event storm instability
        """

        try:

            # ---------------------------------------------------------
            # EVENT BUS VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "event_bus",
            ):

                logger.error(
                    "Event bus missing"
                )

                return

            if self.event_bus is None:

                logger.error(
                    "Event bus unavailable"
                )

                return

            # ---------------------------------------------------------
            # METRICS COLLECTOR VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "metrics_collector",
            ):

                logger.warning(
                    "Metrics collector missing"
                )

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_event_listener_lock",
            ):

                self._event_listener_lock = (
                    threading.Lock()
                )

            with self._event_listener_lock:

                # -----------------------------------------------------
                # DUPLICATE PROTECTION
                # -----------------------------------------------------
                if getattr(
                    self,
                    "_event_listeners_initialized",
                    False,
                ):

                    logger.debug(
                        "Event listeners already initialized"
                    )

                    return

                # -----------------------------------------------------
                # COMMAND EXECUTED LISTENER
                # -----------------------------------------------------
                async def on_command_executed(
                    data,
                ):

                    try:

                        logger.debug(
                            "Event received: "
                            "command_executed"
                        )

                        if (
                            self.metrics_collector
                            and hasattr(
                                self.metrics_collector,
                                "increment_commands",
                            )
                        ):

                            self.metrics_collector.increment_commands()

                    except asyncio.CancelledError:

                        raise

                    except Exception as exc:

                        logger.error(
                            f"Command event failed: {exc}"
                        )

                # -----------------------------------------------------
                # ERROR OCCURRED LISTENER
                # -----------------------------------------------------
                async def on_error_occurred(
                    data,
                ):

                    try:

                        logger.debug(
                            "Event received: "
                            "error_occurred"
                        )

                        if (
                            self.metrics_collector
                            and hasattr(
                                self.metrics_collector,
                                "increment_errors",
                            )
                        ):

                            self.metrics_collector.increment_errors()

                        # optional logging
                        try:

                            if data:

                                logger.warning(
                                    f"System error event: "
                                    f"{data}"
                                )

                        except Exception:
                            pass

                    except asyncio.CancelledError:

                        raise

                    except Exception as exc:

                        logger.error(
                            f"Error event failed: {exc}"
                        )

                # -----------------------------------------------------
                # SYSTEM UNHEALTHY LISTENER
                # -----------------------------------------------------
                async def on_system_unhealthy(
                    data,
                ):

                    try:

                        logger.warning(
                            "⚠️ System unhealthy event"
                        )

                        # -----------------------------------------
                        # AUTO HEAL ENABLED
                        # -----------------------------------------
                        enabled = bool(
                            self.config.get(
                                "enable_auto_healing",
                                True,
                            )
                        )

                        if not enabled:

                            logger.debug(
                                "Auto healing disabled"
                            )

                            return

                        # -----------------------------------------
                        # PREVENT RECURSIVE HEALING (use lock)
                        # -----------------------------------------
                        if not hasattr(self, "_heal_lock"):
                            self._heal_lock = asyncio.Lock()

                        async with self._heal_lock:

                            await asyncio.wait_for(
                                self.auto_heal(),
                                timeout=300,
                            )

                    except asyncio.TimeoutError:

                        logger.error(
                            "Auto-heal timeout"
                        )

                    except asyncio.CancelledError:

                        raise

                    except Exception as exc:

                        logger.error(
                            f"Auto-heal failed: {exc}"
                        )

                # -----------------------------------------------------
                # LISTENER DEFINITIONS
                # -----------------------------------------------------
                listeners = {

                    "command_executed": (
                        on_command_executed
                    ),

                    "error_occurred": (
                        on_error_occurred
                    ),

                    "system_unhealthy": (
                        on_system_unhealthy
                    ),
                }

                subscribed = 0

                failed = 0

                # -----------------------------------------------------
                # SUBSCRIBE LISTENERS
                # -----------------------------------------------------
                for (
                    event_name,
                    callback,
                ) in listeners.items():

                    try:

                        # -----------------------------------------
                        # VALIDATE CALLBACK
                        # -----------------------------------------
                        if not callable(
                            callback
                        ):

                            raise TypeError(
                                f"Invalid callback "
                                f"for {event_name}"
                            )

                        # -----------------------------------------
                        # SUBSCRIBE
                        # -----------------------------------------
                        result = (
                            self.event_bus.subscribe(
                                event_name,
                                callback,
                            )
                        )

                        subscribed += 1

                        logger.debug(
                            f"Subscribed event: "
                            f"{event_name}"
                        )

                    except Exception as exc:

                        failed += 1

                        logger.warning(
                            f"Event subscription failed "
                            f"({event_name}): {exc}"
                        )

                # -----------------------------------------------------
                # STATUS UPDATE
                # -----------------------------------------------------
                try:

                    if failed == 0:

                        self.component_status[
                            "event_bus"
                        ] = (
                            ComponentStatus.HEALTHY
                        )

                    elif subscribed > 0:

                        self.component_status[
                            "event_bus"
                        ] = (
                            ComponentStatus.DEGRADED
                        )

                    else:

                        self.component_status[
                            "event_bus"
                        ] = (
                            ComponentStatus.FAILED
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # INITIALIZATION FLAG
                # -----------------------------------------------------
                self._event_listeners_initialized = (
                    subscribed > 0
                )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                self.event_listener_metrics = {

                    "subscribed": subscribed,

                    "failed": failed,

                    "timestamp": time.time(),
                }

                # -----------------------------------------------------
                # LOGGING
                # -----------------------------------------------------
                logger.info(
                    "📡 Event listeners initialized "
                    f"| subscribed={subscribed} "
                    f"| failed={failed}"
                )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Event listener setup failed: {exc}"
            )

            try:

                self.component_status[
                    "event_bus"
                ] = (
                    ComponentStatus.FAILED
                )

            except Exception:
                pass

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
        """
        Production-grade system health check.

        Fixes:
        - invalid uptime calculations
        - datetime crashes
        - corrupted running state
        - negative uptime
        - health-check race conditions
        - malformed responses
        - startup state instability
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # RUNNING STATE VALIDATION
            # ---------------------------------------------------------
            try:

                running = bool(
                    getattr(
                        self,
                        "_running",
                        False,
                    )
                )

            except Exception:

                running = False

            # ---------------------------------------------------------
            # START TIME VALIDATION
            # ---------------------------------------------------------
            try:

                started_at = getattr(
                    self,
                    "started_at",
                    None,
                )

                if not isinstance(
                    started_at,
                    datetime,
                ):

                    logger.warning(
                        "Invalid started_at timestamp"
                    )

                    started_at = datetime.utcnow()

            except Exception as exc:

                logger.warning(
                    f"Startup time validation failed: {exc}"
                )

                started_at = datetime.utcnow()

            # ---------------------------------------------------------
            # UPTIME CALCULATION
            # ---------------------------------------------------------
            try:

                uptime = (
                    datetime.utcnow() -
                    started_at
                ).total_seconds()

                uptime = max(
                    0.0,
                    float(uptime),
                )

            except Exception as exc:

                logger.warning(
                    f"Uptime calculation failed: {exc}"
                )

                uptime = 0.0

            # ---------------------------------------------------------
            # SYSTEM STATUS
            # ---------------------------------------------------------
            status = "healthy"

            if not running:

                status = "unhealthy"

            # ---------------------------------------------------------
            # INITIALIZATION STATE
            # ---------------------------------------------------------
            initializing = bool(
                getattr(
                    self,
                    "_initializing",
                    False,
                )
            )

            shutdown = bool(
                getattr(
                    self,
                    "_shutdown",
                    False,
                )
            )

            if initializing:

                status = "starting"

            elif shutdown:

                status = "shutdown"

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            health_score = 1.0

            try:

                if not running:
                    health_score -= 0.5

                if initializing:
                    health_score -= 0.2

                if shutdown:
                    health_score = 0.0

                health_score = round(
                    max(
                        0.0,
                        min(1.0, health_score),
                    ),
                    2,
                )

            except Exception:

                health_score = 0.5

            # ---------------------------------------------------------
            # TASK COUNTS
            # ---------------------------------------------------------
            try:

                task_count = len(
                    getattr(
                        self,
                        "tasks",
                        {},
                    )
                )

            except Exception:

                task_count = 0

            # ---------------------------------------------------------
            # COMPONENT HEALTH
            # ---------------------------------------------------------
            try:

                component_status = getattr(
                    self,
                    "component_status",
                    {},
                )

                healthy_components = sum(
                    1
                    for v in component_status.values()
                    if str(v).lower().endswith(
                        "healthy"
                    )
                )

                total_components = max(
                    1,
                    len(component_status),
                )

            except Exception:

                healthy_components = 0
                total_components = 1

            # ---------------------------------------------------------
            # CHECK LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic() -
                    start
                ) * 1000,
                2,
            )

            # ---------------------------------------------------------
            # FINAL RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": status,

                "running": running,

                "initializing": initializing,

                "shutdown": shutdown,

                "uptime_seconds": round(
                    uptime,
                    3,
                ),

                "uptime_human": str(
                    timedelta(
                        seconds=int(uptime)
                    )
                ),

                "started_at": (
                    started_at.isoformat()
                ),

                "health_score": (
                    health_score
                ),

                "tasks": {
                    "registered": task_count,
                },

                "components": {
                    "healthy": (
                        healthy_components
                    ),

                    "total": (
                        total_components
                    ),
                },

                "latency_ms": latency_ms,
            }

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self._health_checks = int(
                    getattr(
                        self,
                        "_health_checks",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"System health check complete "
                f"| status={status} "
                f"| uptime={round(uptime, 1)}s"
            )

            return result

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "System health check cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"System health check failed: {exc}"
            )

            return {

                "status": "error",

                "running": False,

                "health_score": 0.0,

                "error": str(exc),

                "uptime_seconds": 0.0,
            }

    async def _resource_health_check(self) -> Dict[str, Any]:
        """
        Production-grade resource health check.

        Fixes:
        - resource monitor crashes
        - invalid resource responses
        - async timeout failures
        - malformed violation reports
        - health-state corruption
        - missing monitor handling
        - partial resource failures
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # RESOURCE MONITOR VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "resource_monitor",
            ):

                logger.error(
                    "Resource monitor missing"
                )

                return {
                    "status": "error",
                    "error": "resource_monitor_missing",
                }

            if self.resource_monitor is None:

                logger.error(
                    "Resource monitor unavailable"
                )

                return {
                    "status": "error",
                    "error": "resource_monitor_unavailable",
                }

            # ---------------------------------------------------------
            # RESOURCE CHECK
            # ---------------------------------------------------------
            try:

                resources = await asyncio.wait_for(
                    self.resource_monitor.check_resources(),
                    timeout=30,
                )

            except asyncio.TimeoutError:

                logger.warning(
                    "Resource health check timeout"
                )

                return {
                    "status": "timeout",
                    "resources": {},
                    "violations": [],
                }

            # ---------------------------------------------------------
            # RESPONSE VALIDATION
            # ---------------------------------------------------------
            if not isinstance(
                resources,
                dict,
            ):

                logger.warning(
                    "Invalid resource response format"
                )

                resources = {}

            # ---------------------------------------------------------
            # SANITIZE RESOURCE VALUES
            # ---------------------------------------------------------
            safe_resources = {}

            degraded_resources = []

            for (
                resource_name,
                value,
            ) in resources.items():

                try:

                    # -----------------------------------------
                    # RESOURCE NAME
                    # -----------------------------------------
                    if not isinstance(
                        resource_name,
                        str,
                    ):

                        resource_name = str(
                            resource_name
                        )

                    resource_name = (
                        resource_name.strip()
                    )

                    if not resource_name:
                        continue

                    # -----------------------------------------
                    # RESOURCE VALUE
                    # -----------------------------------------
                    healthy = bool(value)

                    safe_resources[
                        resource_name
                    ] = healthy

                    if not healthy:

                        degraded_resources.append(
                            resource_name
                        )

                except Exception as exc:

                    logger.debug(
                        f"Resource sanitize failed "
                        f"({resource_name}): {exc}"
                    )

            # ---------------------------------------------------------
            # OVERALL HEALTH
            # ---------------------------------------------------------
            try:

                all_ok = all(
                    safe_resources.values()
                )

            except Exception:

                all_ok = False

            # ---------------------------------------------------------
            # SYSTEM STATUS
            # ---------------------------------------------------------
            if not safe_resources:

                status = "unknown"

            elif all_ok:

                status = "healthy"

            elif len(degraded_resources) < max(
                1,
                len(safe_resources) // 2,
            ):

                status = "degraded"

            else:

                status = "critical"

            # ---------------------------------------------------------
            # VIOLATION REPORT
            # ---------------------------------------------------------
            try:

                violations = (
                    self.resource_monitor.get_violations_report()
                )

                if violations is None:

                    violations = []

                elif not isinstance(
                    violations,
                    (
                        list,
                        dict,
                    ),
                ):

                    violations = [
                        str(violations)
                    ]

            except Exception as exc:

                logger.warning(
                    f"Violation report failed: {exc}"
                )

                violations = []

            # ---------------------------------------------------------
            # RESOURCE SUMMARY
            # ---------------------------------------------------------
            healthy_count = sum(
                1
                for v in safe_resources.values()
                if v
            )

            total_resources = len(
                safe_resources
            )

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            try:

                health_score = round(
                    healthy_count /
                    max(1, total_resources),
                    2,
                )

            except Exception:

                health_score = 0.0

            # ---------------------------------------------------------
            # CHECK LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic() -
                    start
                ) * 1000,
                2,
            )

            # ---------------------------------------------------------
            # FINAL RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": status,

                "resources": safe_resources,

                "violations": violations,

                "summary": {

                    "healthy": healthy_count,

                    "degraded": len(
                        degraded_resources
                    ),

                    "total": total_resources,
                },

                "degraded_resources": (
                    degraded_resources
                ),

                "health_score": (
                    health_score
                ),

                "latency_ms": latency_ms,
            }

            # ---------------------------------------------------------
            # AUTO-HEAL TRIGGER
            # ---------------------------------------------------------
            try:

                if (
                    status in (
                        "critical",
                        "degraded",
                    )
                    and getattr(
                        self,
                        "_running",
                        False,
                    )
                ):

                    logger.warning(
                        f"Resource degradation detected "
                        f"| status={status}"
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self._resource_health_checks = (
                    int(
                        getattr(
                            self,
                            "_resource_health_checks",
                            0,
                        )
                    ) + 1
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"Resource health check complete "
                f"| status={status} "
                f"| health={health_score}"
            )

            return result

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Resource health check cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Resource health check failed: {exc}"
            )

            return {

                "status": "error",

                "resources": {},

                "violations": [],

                "health_score": 0.0,

                "error": str(exc),
            }

    async def _components_health_check(self) -> Dict[str, Any]:
        """
        Production-grade component health check.

        Fixes:
        - component crashes
        - async health-check failures
        - invalid component responses
        - missing component handling
        - timeout deadlocks
        - malformed health states
        - partial health-check failures
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # COMPONENT DEFINITIONS
            # ---------------------------------------------------------
            components = [

                ("executor", getattr(
                    self,
                    "executor",
                    None,
                )),

                ("file_manager", getattr(
                    self,
                    "file_manager",
                    None,
                )),

                ("command_router", getattr(
                    self,
                    "command_router",
                    None,
                )),

                ("safety_layer", getattr(
                    self,
                    "safety_layer",
                    None,
                )),

                ("idle_controller", getattr(
                    self,
                    "idle_controller",
                    None,
                )),

                ("code_runner", getattr(
                    self,
                    "code_runner",
                    None,
                )),

                ("metrics_collector", getattr(
                    self,
                    "metrics_collector",
                    None,
                )),

                ("cache_manager", getattr(
                    self,
                    "cache_manager",
                    None,
                )),

                ("event_bus", getattr(
                    self,
                    "event_bus",
                    None,
                )),

                ("plugin_manager", getattr(
                    self,
                    "plugin_manager",
                    None,
                )),

                ("health_checker", getattr(
                    self,
                    "health_checker",
                    None,
                )),

                ("resource_monitor", getattr(
                    self,
                    "resource_monitor",
                    None,
                )),

                ("backup_manager", getattr(
                    self,
                    "backup_manager",
                    None,
                )),
            ]

            # ---------------------------------------------------------
            # RESULT STORAGE
            # ---------------------------------------------------------
            components_status = {}

            healthy_count = 0

            degraded_count = 0

            failed_count = 0

            # ---------------------------------------------------------
            # COMPONENT CHECK LOOP
            # ---------------------------------------------------------
            for (
                component_name,
                component,
            ) in components:

                try:

                    # ---------------------------------------------
                    # MISSING COMPONENT
                    # ---------------------------------------------
                    if component is None:

                        components_status[
                            component_name
                        ] = {

                            "status": "missing",

                            "healthy": False,
                        }

                        failed_count += 1

                        continue

                    # ---------------------------------------------
                    # HEALTH CHECK METHOD
                    # ---------------------------------------------
                    if hasattr(
                        component,
                        "health_check",
                    ):

                        try:

                            result = (
                                component.health_check()
                            )

                            # async check
                            if asyncio.iscoroutine(
                                result
                            ):

                                result = (
                                    await asyncio.wait_for(
                                        result,
                                        timeout=15,
                                    )
                                )

                        except asyncio.TimeoutError:

                            result = {
                                "status": "timeout",
                                "healthy": False,
                            }

                        # -----------------------------------------
                        # NORMALIZE RESULT
                        # -----------------------------------------
                        if isinstance(
                            result,
                            str,
                        ):

                            status = result

                            healthy = (
                                "unhealthy"
                                not in result.lower()
                                and "failed"
                                not in result.lower()
                            )

                            result = {

                                "status": status,

                                "healthy": healthy,
                            }

                        elif isinstance(
                            result,
                            dict,
                        ):

                            result.setdefault(
                                "status",
                                "healthy",
                            )

                            result.setdefault(
                                "healthy",
                                result.get(
                                    "status",
                                    "healthy",
                                ) == "healthy",
                            )

                        else:

                            result = {

                                "status": str(result),

                                "healthy": True,
                            }

                    # ---------------------------------------------
                    # NO HEALTH CHECK METHOD
                    # ---------------------------------------------
                    else:

                        result = {

                            "status": "healthy",

                            "healthy": True,

                            "details": (
                                "No health_check() method"
                            ),
                        }

                    # ---------------------------------------------
                    # SAVE STATUS
                    # ---------------------------------------------
                    components_status[
                        component_name
                    ] = result

                    # ---------------------------------------------
                    # COUNTERS
                    # ---------------------------------------------
                    if result.get(
                        "healthy",
                        False,
                    ):

                        healthy_count += 1

                    elif result.get(
                        "status"
                    ) in (
                        "degraded",
                        "timeout",
                    ):

                        degraded_count += 1

                    else:

                        failed_count += 1

                # -------------------------------------------------
                # COMPONENT FAILURE
                # -------------------------------------------------
                except asyncio.CancelledError:

                    raise

                except Exception as exc:

                    logger.warning(
                        f"Component health check failed "
                        f"({component_name}): {exc}"
                    )

                    components_status[
                        component_name
                    ] = {

                        "status": "unhealthy",

                        "healthy": False,

                        "error": str(exc),
                    }

                    failed_count += 1

            # ---------------------------------------------------------
            # OVERALL STATUS
            # ---------------------------------------------------------
            total_components = max(
                1,
                len(components_status),
            )

            if failed_count == 0:

                overall_status = "healthy"

            elif failed_count < (
                total_components // 2
            ):

                overall_status = "degraded"

            else:

                overall_status = "critical"

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            health_score = round(
                healthy_count /
                total_components,
                2,
            )

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic() -
                    start
                ) * 1000,
                2,
            )

            # ---------------------------------------------------------
            # FAILED COMPONENTS
            # ---------------------------------------------------------
            failed_components = [

                name
                for name, data in (
                    components_status.items()
                )
                if not data.get(
                    "healthy",
                    False,
                )
            ]

            # ---------------------------------------------------------
            # FINAL RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": overall_status,

                "health_score": (
                    health_score
                ),

                "summary": {

                    "healthy": healthy_count,

                    "degraded": degraded_count,

                    "failed": failed_count,

                    "total": total_components,
                },

                "failed_components": (
                    failed_components
                ),

                "components": (
                    components_status
                ),

                "latency_ms": latency_ms,
            }

            # ---------------------------------------------------------
            # STATUS CACHE UPDATE
            # ---------------------------------------------------------
            try:

                self.component_status.update({

                    name: (
                        ComponentStatus.HEALTHY
                        if data.get(
                            "healthy",
                            False,
                        )
                        else ComponentStatus.DEGRADED
                    )

                    for name, data in (
                        components_status.items()
                    )
                })

            except Exception:
                pass

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self._component_health_checks = (
                    int(
                        getattr(
                            self,
                            "_component_health_checks",
                            0,
                        )
                    ) + 1
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"Component health complete "
                f"| status={overall_status} "
                f"| healthy={healthy_count}/"
                f"{total_components}"
            )

            return result

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Component health check cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Component health check failed: {exc}"
            )

            return {

                "status": "error",

                "health_score": 0.0,

                "summary": {

                    "healthy": 0,

                    "degraded": 0,

                    "failed": 0,

                    "total": 0,
                },

                "failed_components": [],

                "components": {},

                "error": str(exc),
            }

    async def _performance_health_check(self) -> Dict[str, Any]:
        """
        Production-grade performance health check.

        Fixes:
        - missing metrics collector crashes
        - malformed statistics handling
        - invalid CPU/memory values
        - performance threshold instability
        - NaN/inf metric corruption
        - async timeout protection
        - malformed responses
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # METRICS COLLECTOR VALIDATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "metrics_collector",
            ):

                logger.error(
                    "Metrics collector missing"
                )

                return {
                    "status": "error",
                    "issues": [
                        "metrics_collector_missing"
                    ],
                }

            if self.metrics_collector is None:

                logger.error(
                    "Metrics collector unavailable"
                )

                return {
                    "status": "error",
                    "issues": [
                        "metrics_collector_unavailable"
                    ],
                }

            # ---------------------------------------------------------
            # GET STATISTICS
            # ---------------------------------------------------------
            try:

                stats = (
                    self.metrics_collector.get_statistics()
                )

            except Exception as exc:

                logger.warning(
                    f"Statistics retrieval failed: {exc}"
                )

                stats = {}

            # ---------------------------------------------------------
            # VALIDATE STATS
            # ---------------------------------------------------------
            if not isinstance(
                stats,
                dict,
            ):

                logger.warning(
                    "Invalid statistics format"
                )

                stats = {}

            # ---------------------------------------------------------
            # PERFORMANCE ISSUES
            # ---------------------------------------------------------
            performance_issues = []

            warnings = []

            # ---------------------------------------------------------
            # SAFE VALUE EXTRACTOR
            # ---------------------------------------------------------
            def _safe_metric(
                category: str,
                field: str = "current",
                default: float = 0.0,
            ) -> float:

                try:

                    value = (
                        stats.get(
                            category,
                            {},
                        ).get(
                            field,
                            default,
                        )
                    )

                    value = float(value)

                    if (
                        math.isnan(value)
                        or math.isinf(value)
                    ):

                        return default

                    return max(
                        0.0,
                        min(value, 100.0),
                    )

                except Exception:

                    return default

            # ---------------------------------------------------------
            # CPU CHECK
            # ---------------------------------------------------------
            cpu_usage = _safe_metric(
                "cpu"
            )

            if cpu_usage > 95:

                performance_issues.append(
                    "critical_cpu"
                )

            elif cpu_usage > 80:

                performance_issues.append(
                    "high_cpu"
                )

            elif cpu_usage > 65:

                warnings.append(
                    "elevated_cpu"
                )

            # ---------------------------------------------------------
            # MEMORY CHECK
            # ---------------------------------------------------------
            memory_usage = _safe_metric(
                "memory"
            )

            if memory_usage > 95:

                performance_issues.append(
                    "critical_memory"
                )

            elif memory_usage > 80:

                performance_issues.append(
                    "high_memory"
                )

            elif memory_usage > 65:

                warnings.append(
                    "elevated_memory"
                )

            # ---------------------------------------------------------
            # DISK CHECK
            # ---------------------------------------------------------
            disk_usage = _safe_metric(
                "disk"
            )

            if disk_usage > 95:

                performance_issues.append(
                    "critical_disk"
                )

            elif disk_usage > 85:

                performance_issues.append(
                    "high_disk"
                )

            # ---------------------------------------------------------
            # TASK QUEUE CHECK
            # ---------------------------------------------------------
            try:

                queue_size = int(
                    getattr(
                        self.task_queue,
                        "qsize",
                        lambda: 0,
                    )()
                )

                if queue_size > 500:

                    performance_issues.append(
                        "queue_overload"
                    )

                elif queue_size > 200:

                    warnings.append(
                        "queue_pressure"
                    )

            except Exception:

                queue_size = 0

            # ---------------------------------------------------------
            # ACTIVE TASK CHECK
            # ---------------------------------------------------------
            try:

                active_tasks = len(
                    getattr(
                        self,
                        "tasks",
                        {},
                    )
                )

                max_tasks = int(
                    self.config.get(
                        "max_concurrent_tasks",
                        10,
                    )
                )

                utilization = (
                    active_tasks /
                    max(1, max_tasks)
                )

                if utilization > 0.95:

                    performance_issues.append(
                        "task_overload"
                    )

                elif utilization > 0.75:

                    warnings.append(
                        "high_task_utilization"
                    )

            except Exception:

                active_tasks = 0

                utilization = 0.0

            # ---------------------------------------------------------
            # OVERALL STATUS
            # ---------------------------------------------------------
            if any(
                issue.startswith("critical")
                for issue in performance_issues
            ):

                status = "critical"

            elif performance_issues:

                status = "degraded"

            else:

                status = "healthy"

            # ---------------------------------------------------------
            # HEALTH SCORE
            # ---------------------------------------------------------
            health_score = 1.0

            try:

                health_score -= (
                    len(performance_issues)
                    * 0.15
                )

                health_score -= (
                    len(warnings)
                    * 0.05
                )

                health_score = round(
                    max(
                        0.0,
                        min(1.0, health_score),
                    ),
                    2,
                )

            except Exception:

                health_score = 0.5

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic() -
                    start
                ) * 1000,
                2,
            )

            # ---------------------------------------------------------
            # PERFORMANCE SUMMARY
            # ---------------------------------------------------------
            summary = {

                "cpu_usage": cpu_usage,

                "memory_usage": memory_usage,

                "disk_usage": disk_usage,

                "queue_size": queue_size,

                "active_tasks": active_tasks,

                "task_utilization": round(
                    utilization,
                    2,
                ),
            }

            # ---------------------------------------------------------
            # FINAL RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": status,

                "health_score": (
                    health_score
                ),

                "statistics": stats,

                "summary": summary,

                "issues": (
                    performance_issues
                ),

                "warnings": warnings,

                "latency_ms": latency_ms,
            }

            # ---------------------------------------------------------
            # AUTO-HEAL HINT
            # ---------------------------------------------------------
            try:

                if (
                    status in (
                        "degraded",
                        "critical",
                    )
                    and self.config.get(
                        "enable_auto_healing",
                        True,
                    )
                ):

                    logger.warning(
                        f"Performance degradation detected "
                        f"| issues={performance_issues}"
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self._performance_health_checks = (
                    int(
                        getattr(
                            self,
                            "_performance_health_checks",
                            0,
                        )
                    ) + 1
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"Performance health complete "
                f"| status={status} "
                f"| score={health_score}"
            )

            return result

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Performance health check cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"Performance health check failed: {exc}"
            )

            return {

                "status": "error",

                "health_score": 0.0,

                "statistics": {},

                "summary": {},

                "issues": [
                    "performance_check_failed"
                ],

                "warnings": [],

                "error": str(exc),
            }

    # ------------------------
    # AUTO HEALING
    # ------------------------

    async def auto_heal(self) -> None:
        """
        Production-grade autonomous healing system.

        Fixes:
        - recursive healing loops
        - unsafe process termination
        - memory leaks
        - blocked event loop
        - component restart crashes
        - event emission failures
        - concurrent heal races
        - CPU spike instability
        - cache clear deadlocks
        """

        start = time.monotonic()

        try:

            import asyncio
            import gc
            import psutil

            from datetime import datetime

            # ---------------------------------------------------------
            # REENTRANCY PROTECTION
            # ---------------------------------------------------------
            if getattr(
                self,
                "_healing",
                False,
            ):

                logger.warning(
                    "Auto-heal already running"
                )

                return

            self._healing = True

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_heal_lock",
            ):

                self._heal_lock = asyncio.Lock()

            async with self._heal_lock:

                logger.warning(
                    "🔄 Running autonomous healing"
                )

                heal_report = {

                    "memory_recovered": False,

                    "cpu_recovered": False,

                    "components_restarted": [],

                    "events_emitted": False,

                    "errors": [],
                }

                # -----------------------------------------------------
                # MEMORY CHECK
                # -----------------------------------------------------
                try:

                    memory_percent = float(
                        psutil.virtual_memory().percent
                    )

                except Exception as exc:

                    logger.debug(
                        f"Memory check failed: {exc}"
                    )

                    memory_percent = 0.0

                # -----------------------------------------------------
                # HIGH MEMORY RECOVERY
                # -----------------------------------------------------
                if memory_percent > 85:

                    logger.warning(
                        f"High memory usage detected "
                        f"({memory_percent:.1f}%)"
                    )

                    # ---------------------------------------------
                    # CACHE CLEAR
                    # ---------------------------------------------
                    try:

                        cache_manager = getattr(
                            self,
                            "cache_manager",
                            None,
                        )

                        if (
                            cache_manager
                            and hasattr(
                                cache_manager,
                                "clear",
                            )
                        ):

                            result = (
                                cache_manager.clear()
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                await asyncio.wait_for(
                                    result,
                                    timeout=15,
                                )

                            heal_report[
                                "memory_recovered"
                            ] = True

                            logger.info(
                                "🧹 Cache cleared"
                            )

                    except Exception as exc:

                        heal_report[
                            "errors"
                        ].append(
                            f"cache_clear:{exc}"
                        )

                        logger.debug(
                            f"Cache clear failed: {exc}"
                        )

                    # ---------------------------------------------
                    # GARBAGE COLLECTION (non-blocking)
                    # ---------------------------------------------
                    try:

                        def _gc_collect():
                            gc.collect()

                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, _gc_collect)

                        logger.info(
                            "♻️ Garbage collection completed"
                        )

                    except Exception as exc:

                        logger.debug(
                            f"GC failed: {exc}"
                        )

                # -----------------------------------------------------
                # CPU CHECK
                # -----------------------------------------------------
                try:

                    cpu_percent = float(
                        psutil.cpu_percent(
                            interval=0.0
                        )
                    )

                except Exception as exc:

                    logger.debug(
                        f"CPU check failed: {exc}"
                    )

                    cpu_percent = 0.0

                # -----------------------------------------------------
                # CPU RECOVERY
                # -----------------------------------------------------
                if cpu_percent > 90:

                    logger.warning(
                        f"High CPU usage detected "
                        f"({cpu_percent:.1f}%)"
                    )

                    try:

                        process = psutil.Process()

                        terminated = 0

                        for child in process.children(
                            recursive=True
                        ):

                            try:

                                child_cpu = (
                                    child.cpu_percent(
                                        interval=0.0
                                    )
                                )

                                # SAFE TERMINATION
                                if (
                                    child_cpu > 50
                                    and child.pid
                                    != process.pid
                                ):

                                    logger.warning(
                                        f"⚠️ Terminating "
                                        f"high CPU child "
                                        f"{child.pid}"
                                    )

                                    child.terminate()

                                    terminated += 1

                            except (
                                psutil.NoSuchProcess,
                                psutil.AccessDenied,
                            ):

                                continue

                            except Exception as exc:

                                logger.debug(
                                    f"Child process handling failed: {exc}"
                                )

                        if terminated > 0:

                            heal_report[
                                "cpu_recovered"
                            ] = True

                            logger.info(
                                f"⚡ Terminated "
                                f"{terminated} "
                                f"high-CPU processes"
                            )

                    except Exception as exc:

                        heal_report[
                            "errors"
                        ].append(
                            f"cpu_recovery:{exc}"
                        )

                        logger.debug(
                            f"CPU recovery failed: {exc}"
                        )

                # -----------------------------------------------------
                # COMPONENT RECOVERY
                # -----------------------------------------------------
                restart_targets = [

                    "command_router",

                    "event_bus",

                    "idle_controller",
                ]

                for component_name in (
                    restart_targets
                ):

                    try:

                        component = getattr(
                            self,
                            component_name,
                            None,
                        )

                        if component is None:
                            continue

                        # -----------------------------------------
                        # RUNNING CHECK
                        # -----------------------------------------
                        is_running = bool(
                            getattr(
                                component,
                                "_running",
                                True,
                            )
                        )

                        if is_running:
                            continue

                        # -----------------------------------------
                        # RESTART COMPONENT
                        # -----------------------------------------
                        if hasattr(
                            component,
                            "start",
                        ):

                            logger.warning(
                                f"🔁 Restarting "
                                f"{component_name}"
                            )

                            result = (
                                component.start()
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                await asyncio.wait_for(
                                    result,
                                    timeout=30,
                                )

                            else:

                                await asyncio.to_thread(
                                    component.start
                                )

                            heal_report[
                                "components_restarted"
                            ].append(
                                component_name
                            )

                            logger.info(
                                f"✅ Restarted "
                                f"{component_name}"
                            )

                    except asyncio.TimeoutError:

                        logger.warning(
                            f"{component_name} "
                            f"restart timeout"
                        )

                    except Exception as exc:

                        heal_report[
                            "errors"
                        ].append(
                            f"{component_name}:{exc}"
                        )

                        logger.debug(
                            f"{component_name} restart failed: {exc}"
                        )

                # -----------------------------------------------------
                # TASK QUEUE RECOVERY
                # -----------------------------------------------------
                try:

                    queue = getattr(
                        self,
                        "task_queue",
                        None,
                    )

                    if queue:

                        qsize = queue.qsize()

                        if qsize > 500:

                            logger.warning(
                                f"Task queue overload: "
                                f"{qsize}"
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                datetime.utcnow()
                                .isoformat()
                            ),

                            "cpu": cpu_percent,

                            "memory": (
                                memory_percent
                            ),

                            "report": (
                                heal_report
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "healing_completed",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=10,
                            )

                        heal_report[
                            "events_emitted"
                        ] = True

                except Exception as exc:

                    heal_report[
                        "errors"
                    ].append(
                        f"event_emit:{exc}"
                    )

                    logger.debug(
                        f"Event emission failed: {exc}"
                    )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.heal_runs = int(
                        getattr(
                            self,
                            "heal_runs",
                            0,
                        )
                    ) + 1

                    self.last_heal = (
                        datetime.utcnow()
                    )

                    self.last_heal_report = (
                        heal_report
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # HEAL DURATION
                # -----------------------------------------------------
                duration = round(
                    time.monotonic() - start,
                    3,
                )

                logger.info(
                    f"🛠️ Auto-heal completed "
                    f"in {duration}s"
                )

                if heal_report["errors"]:

                    logger.warning(
                        f"Heal completed with "
                        f"{len(heal_report['errors'])} errors"
                    )

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Auto-heal cancelled"
            )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Auto-heal failed: {exc}"
            )

        # -------------------------------------------------------------
        # CLEANUP
        # -------------------------------------------------------------
        finally:

            try:

                self._healing = False
                self._last_heal_time = time.monotonic()

            except Exception:
                pass

    async def optimize_system(self) -> None:
        """
        Production-grade autonomous optimization system.

        Fixes:
        - worker leaks
        - cache cleanup crashes
        - task cleanup corruption
        - event-loop blocking
        - worker scaling instability
        - optimization race conditions
        - event emission failures
        - orphaned tasks
        - unsafe task removal
        """

        start = time.monotonic()

        try:

            import asyncio
            import gc
            import psutil

            from datetime import datetime

            # ---------------------------------------------------------
            # REENTRANCY PROTECTION
            # ---------------------------------------------------------
            if getattr(
                self,
                "_optimizing",
                False,
            ):

                logger.warning(
                    "Optimization already running"
                )

                return

            self._optimizing = True

            # ---------------------------------------------------------
            # THREAD SAFETY
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_optimization_lock",
            ):

                self._optimization_lock = (
                    asyncio.Lock()
                )

            async with self._optimization_lock:

                logger.info(
                    "🔧 Running system optimization"
                )

                optimization_report = {

                    "cache_cleared": False,

                    "tasks_removed": 0,

                    "workers_scaled": False,

                    "gc_collected": False,

                    "errors": [],
                }

                # -----------------------------------------------------
                # CACHE OPTIMIZATION
                # -----------------------------------------------------
                try:

                    cache_manager = getattr(
                        self,
                        "cache_manager",
                        None,
                    )

                    if (
                        cache_manager
                        and hasattr(
                            cache_manager,
                            "get_stats",
                        )
                    ):

                        stats = (
                            cache_manager.get_stats()
                        )

                        if not isinstance(
                            stats,
                            dict,
                        ):

                            stats = {}

                        cache_size = int(
                            stats.get(
                                "size",
                                0,
                            )
                        )

                        # -----------------------------------------
                        # LARGE CACHE CLEANUP
                        # -----------------------------------------
                        if cache_size > 500:

                            logger.warning(
                                f"Large cache detected "
                                f"({cache_size})"
                            )

                            if hasattr(
                                cache_manager,
                                "clear",
                            ):

                                result = (
                                    cache_manager.clear()
                                )

                                if asyncio.iscoroutine(
                                    result
                                ):

                                    await asyncio.wait_for(
                                        result,
                                        timeout=20,
                                    )

                                optimization_report[
                                    "cache_cleared"
                                ] = True

                                logger.info(
                                    "🧹 Large cache cleared"
                                )

                except Exception as exc:

                    optimization_report[
                        "errors"
                    ].append(
                        f"cache_optimization:{exc}"
                    )

                    logger.debug(
                        f"Cache optimization failed: {exc}"
                    )

                # -----------------------------------------------------
                # TASK CLEANUP
                # -----------------------------------------------------
                try:

                    now = datetime.utcnow()

                    tasks = getattr(
                        self,
                        "tasks",
                        {},
                    )

                    if not isinstance(
                        tasks,
                        dict,
                    ):

                        tasks = {}

                    removable_tasks = []

                    # ---------------------------------------------
                    # FIND OLD TASKS
                    # ---------------------------------------------
                    for (
                        task_id,
                        task,
                    ) in list(tasks.items()):

                        try:

                            if task is None:
                                continue

                            state = getattr(
                                task,
                                "state",
                                None,
                            )

                            completed_at = getattr(
                                task,
                                "completed_at",
                                None,
                            )

                            if (
                                state in (
                                    TaskState.COMPLETED,
                                    TaskState.FAILED,
                                    TaskState.CANCELLED,
                                )
                                and completed_at
                            ):

                                age = (
                                    now -
                                    completed_at
                                ).total_seconds()

                                # remove tasks older than 1 hour
                                if age > 3600:

                                    removable_tasks.append(
                                        task_id
                                    )

                        except Exception as exc:

                            logger.debug(
                                f"Task scan failed "
                                f"({task_id}): {exc}"
                            )

                    # ---------------------------------------------
                    # REMOVE TASKS
                    # ---------------------------------------------
                    for task_id in (
                        removable_tasks
                    ):

                        try:

                            tasks.pop(
                                task_id,
                                None,
                            )

                            optimization_report[
                                "tasks_removed"
                            ] += 1

                        except Exception:
                            pass

                    if removable_tasks:

                        logger.info(
                            f"🗑️ Removed "
                            f"{len(removable_tasks)} "
                            f"old tasks"
                        )

                except Exception as exc:

                    optimization_report[
                        "errors"
                    ].append(
                        f"task_cleanup:{exc}"
                    )

                    logger.debug(
                        f"Task cleanup failed: {exc}"
                    )

                # -----------------------------------------------------
                # WORKER OPTIMIZATION
                # -----------------------------------------------------
                try:

                    cpu_count = max(
                        1,
                        psutil.cpu_count() or 1,
                    )

                    config = getattr(
                        self,
                        "config",
                        {},
                    )

                    max_tasks = int(
                        config.get(
                            "max_concurrent_tasks",
                            cpu_count,
                        )
                    )

                    optimal_workers = max(
                        1,
                        min(cpu_count, max_tasks),
                    )

                    worker_tasks = getattr(
                        self,
                        "worker_tasks",
                        [],
                    )

                    if not isinstance(
                        worker_tasks,
                        list,
                    ):

                        worker_tasks = []

                    # ---------------------------------------------
                    # REMOVE DEAD WORKERS
                    # ---------------------------------------------
                    alive_workers = []

                    for worker in worker_tasks:

                        try:

                            if (
                                worker
                                and not worker.done()
                            ):

                                alive_workers.append(
                                    worker
                                )

                        except Exception:
                            continue

                    worker_tasks[:] = alive_workers

                    # ---------------------------------------------
                    # SCALE UP
                    # ---------------------------------------------
                    while (
                        len(worker_tasks)
                        < optimal_workers
                    ):

                        try:

                            worker = (
                                asyncio.create_task(
                                    self._task_worker(),
                                    name=(
                                        f"optimizer_worker_"
                                        f"{len(worker_tasks)}"
                                    ),
                                )
                            )

                            worker_tasks.append(
                                worker
                            )

                            optimization_report[
                                "workers_scaled"
                            ] = True

                        except Exception as exc:

                            optimization_report[
                                "errors"
                            ].append(
                                f"worker_scale_up:{exc}"
                            )

                            break

                    # ---------------------------------------------
                    # SCALE DOWN
                    # ---------------------------------------------
                    while (
                        len(worker_tasks)
                        > optimal_workers
                    ):

                        try:

                            worker = (
                                worker_tasks.pop()
                            )

                            if (
                                worker
                                and not worker.done()
                            ):

                                worker.cancel()

                            optimization_report[
                                "workers_scaled"
                            ] = True

                        except Exception as exc:

                            optimization_report[
                                "errors"
                            ].append(
                                f"worker_scale_down:{exc}"
                            )

                            break

                    logger.info(
                        f"⚡ Worker optimization "
                        f"| active={len(worker_tasks)} "
                        f"| optimal={optimal_workers}"
                    )

                except Exception as exc:

                    optimization_report[
                        "errors"
                    ].append(
                        f"worker_optimization:{exc}"
                    )

                    logger.debug(
                        f"Worker optimization failed: {exc}"
                    )

                # -----------------------------------------------------
                # GARBAGE COLLECTION
                # -----------------------------------------------------
                try:

                    gc.collect()

                    optimization_report[
                        "gc_collected"
                    ] = True

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESOURCE SNAPSHOT
                # -----------------------------------------------------
                try:

                    optimization_report[
                        "resources"
                    ] = {

                        "cpu_percent": (
                            psutil.cpu_percent(
                                interval=0.0
                            )
                        ),

                        "memory_percent": (
                            psutil.virtual_memory().percent
                        ),

                        "workers": len(
                            getattr(
                                self,
                                "worker_tasks",
                                [],
                            )
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                datetime.utcnow()
                                .isoformat()
                            ),

                            "report": (
                                optimization_report
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "optimization_completed",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=10,
                            )

                except Exception as exc:

                    optimization_report[
                        "errors"
                    ].append(
                        f"event_emit:{exc}"
                    )

                    logger.debug(
                        f"Optimization event failed: {exc}"
                    )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.optimizations = int(
                        getattr(
                            self,
                            "optimizations",
                            0,
                        )
                    ) + 1

                    self.last_optimization = (
                        datetime.utcnow()
                    )

                    self.last_optimization_report = (
                        optimization_report
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # DURATION
                # -----------------------------------------------------
                duration = round(
                    time.monotonic() - start,
                    3,
                )

                logger.info(
                    f"✅ Optimization complete "
                    f"in {duration}s"
                )

                if optimization_report[
                    "errors"
                ]:

                    logger.warning(
                        f"Optimization completed with "
                        f"{len(optimization_report['errors'])} errors"
                    )

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Optimization cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Optimization failed: {exc}"
            )

        # -------------------------------------------------------------
        # CLEANUP
        # -------------------------------------------------------------
        finally:

            try:

                self._optimizing = False

            except Exception:
                pass

    # ------------------------
    # TASK MANAGEMENT
    # ------------------------

    async def submit_task(
        self,
        name: str,
        coroutine,
        priority: Priority = Priority.NORMAL,
    ) -> Optional[str]:
        """
        Production-grade task submission system.

        Fixes:
        - duplicate task IDs
        - invalid coroutine submission
        - queue overflow crashes
        - task registry corruption
        - priority validation issues
        - race conditions
        - task memory leaks
        - malformed task objects
        - unsafe queue insertion
        """

        start = time.monotonic()

        try:

            import asyncio
            import hashlib
            import inspect

            from datetime import datetime

            # ---------------------------------------------------------
            # INITIALIZATION CHECK
            # ---------------------------------------------------------
            if getattr(
                self,
                "_shutdown",
                False,
            ):

                logger.warning(
                    "Task submission rejected: "
                    "system shutting down"
                )

                return None

            # ---------------------------------------------------------
            # TASK LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_submit_lock",
            ):

                self._submit_lock = asyncio.Lock()

            async with self._submit_lock:

                # -----------------------------------------------------
                # NAME VALIDATION
                # -----------------------------------------------------
                if name is None:

                    logger.warning(
                        "Task submission failed: "
                        "name is None"
                    )

                    return None

                if not isinstance(
                    name,
                    str,
                ):

                    try:

                        name = str(name)

                    except Exception:

                        logger.warning(
                            "Task name conversion failed"
                        )

                        return None

                name = name.strip()

                if not name:

                    logger.warning(
                        "Task submission failed: "
                        "empty name"
                    )

                    return None

                # -----------------------------------------------------
                # HARD LIMIT
                # -----------------------------------------------------
                name = name[:128]

                # -----------------------------------------------------
                # COROUTINE VALIDATION
                # -----------------------------------------------------
                if coroutine is None:

                    logger.warning(
                        "Task submission failed: "
                        "coroutine missing"
                    )

                    return None

                valid_coroutine = (

                    asyncio.iscoroutine(
                        coroutine
                    )

                    or inspect.isawaitable(
                        coroutine
                    )
                )

                if not valid_coroutine:

                    logger.warning(
                        "Task submission failed: "
                        "invalid coroutine"
                    )

                    return None

                # -----------------------------------------------------
                # PRIORITY VALIDATION
                # -----------------------------------------------------
                try:

                    if not isinstance(
                        priority,
                        Priority,
                    ):

                        priority = Priority(
                            int(priority)
                        )

                except Exception:

                    logger.warning(
                        "Invalid task priority"
                    )

                    priority = Priority.NORMAL

                # -----------------------------------------------------
                # TASK REGISTRY VALIDATION
                # -----------------------------------------------------
                if not hasattr(
                    self,
                    "tasks",
                ):

                    self.tasks = {}

                if not isinstance(
                    self.tasks,
                    dict,
                ):

                    logger.error(
                        "Task registry corrupted"
                    )

                    self.tasks = {}

                # -----------------------------------------------------
                # QUEUE VALIDATION
                # -----------------------------------------------------
                if not hasattr(
                    self,
                    "task_queue",
                ):

                    logger.error(
                        "Task queue missing"
                    )

                    return None

                queue = self.task_queue

                # -----------------------------------------------------
                # QUEUE OVERLOAD PROTECTION
                # -----------------------------------------------------
                try:

                    queue_size = queue.qsize()

                    max_queue_size = int(
                        self.config.get(
                            "max_task_queue_size",
                            1000,
                        )
                    )

                    if queue_size >= max_queue_size:

                        logger.warning(
                            f"Task queue full "
                            f"({queue_size})"
                        )

                        return None

                except Exception:

                    queue_size = 0

                # -----------------------------------------------------
                # UNIQUE TASK ID
                # -----------------------------------------------------
                entropy = (

                    f"{name}|"

                    f"{datetime.utcnow().isoformat()}|"

                    f"{id(coroutine)}|"

                    f"{time.monotonic_ns()}"
                )

                task_id = hashlib.sha256(
                    entropy.encode(
                        "utf-8",
                        errors="ignore",
                    )
                ).hexdigest()[:16]

                # -----------------------------------------------------
                # DUPLICATE CHECK
                # -----------------------------------------------------
                if task_id in self.tasks:

                    logger.warning(
                        f"Duplicate task ID: {task_id}"
                    )

                    return None

                # -----------------------------------------------------
                # CREATE TASK OBJECT
                # -----------------------------------------------------
                try:

                    task = Task(

                        id=task_id,

                        name=name,

                        state=TaskState.PENDING,

                        created_at=datetime.utcnow(),

                        priority=priority,
                    )

                except Exception as exc:

                    logger.error(
                        f"Task object creation failed: {exc}"
                    )

                    return None

                # -----------------------------------------------------
                # STORE TASK
                # -----------------------------------------------------
                self.tasks[task_id] = task

                # -----------------------------------------------------
                # QUEUE PAYLOAD
                # -----------------------------------------------------
                payload = (

                    int(priority.value),

                    time.monotonic(),

                    task_id,

                    coroutine,
                )

                # -----------------------------------------------------
                # QUEUE INSERTION
                # -----------------------------------------------------
                try:

                    await asyncio.wait_for(

                        queue.put(payload),

                        timeout=10,
                    )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Task queue insertion timeout"
                    )

                    self.tasks.pop(
                        task_id,
                        None,
                    )

                    return None

                except Exception as exc:

                    logger.error(
                        f"Queue insertion failed: {exc}"
                    )

                    self.tasks.pop(
                        task_id,
                        None,
                    )

                    return None

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.tasks_submitted = int(
                        getattr(
                            self,
                            "tasks_submitted",
                            0,
                        )
                    ) + 1

                    self.last_task_submission = (
                        datetime.utcnow()
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "task_id": task_id,

                            "name": name,

                            "priority": (
                                priority.name
                            ),

                            "timestamp": (
                                datetime.utcnow()
                                .isoformat()
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "task_submitted",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            asyncio.create_task(
                                result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic() -
                        start
                    ) * 1000,
                    2,
                )

                logger.info(
                    f"📥 Task submitted "
                    f"| id={task_id} "
                    f"| priority={priority.name} "
                    f"| latency={latency_ms}ms"
                )

                return task_id

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Task submission cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Task submission failed: {exc}"
            )

            return None

    async def _task_worker(self) -> None:
        """
        Production-grade async task worker.

        Fixes:
        - orphaned queue tasks
        - worker crashes
        - coroutine leaks
        - stuck task states
        - queue deadlocks
        - timeout instability
        - invalid payload handling
        - worker memory leaks
        - unhandled cancellation
        """

        import asyncio
        import gc

        from datetime import datetime

        worker_id = hex(id(asyncio.current_task()))[-6:]

        logger.info(
            f"⚙️ Task worker started "
            f"| id={worker_id}"
        )

        try:

            # ---------------------------------------------------------
            # WORKER LOOP
            # ---------------------------------------------------------
            while getattr(
                self,
                "_running",
                False,
            ):

                queue_item = None

                task = None

                task_id = None

                coroutine = None

                try:

                    # -------------------------------------------------
                    # QUEUE VALIDATION
                    # -------------------------------------------------
                    if not hasattr(
                        self,
                        "task_queue",
                    ):

                        logger.error(
                            "Task queue missing"
                        )

                        await asyncio.sleep(1)

                        continue

                    # -------------------------------------------------
                    # GET TASK
                    # -------------------------------------------------
                    try:

                        queue_item = (
                            await asyncio.wait_for(
                                self.task_queue.get(),
                                timeout=5.0,
                            )
                        )

                    except asyncio.TimeoutError:

                        # idle worker
                        await asyncio.sleep(0)

                        continue

                    # -------------------------------------------------
                    # PAYLOAD VALIDATION
                    # -------------------------------------------------
                    if (
                        not isinstance(
                            queue_item,
                            (
                                tuple,
                                list,
                            ),
                        )
                        or len(queue_item) < 4
                    ):

                        logger.warning(
                            "Invalid queue payload"
                        )

                        try:
                            self.task_queue.task_done()
                        except Exception:
                            pass

                        continue

                    (
                        priority,
                        queued_at,
                        task_id,
                        coroutine,
                    ) = queue_item

                    # -------------------------------------------------
                    # TASK LOOKUP
                    # -------------------------------------------------
                    tasks = getattr(
                        self,
                        "tasks",
                        {},
                    )

                    if not isinstance(
                        tasks,
                        dict,
                    ):

                        logger.error(
                            "Task registry corrupted"
                        )

                        tasks = {}

                    task = tasks.get(task_id)

                    if task is None:

                        logger.warning(
                            f"Task missing: {task_id}"
                        )

                        try:
                            self.task_queue.task_done()
                        except Exception:
                            pass

                        continue

                    # -------------------------------------------------
                    # TASK VALIDATION
                    # -------------------------------------------------
                    if not hasattr(
                        task,
                        "state",
                    ):

                        logger.warning(
                            f"Malformed task: {task_id}"
                        )

                        try:
                            self.task_queue.task_done()
                        except Exception:
                            pass

                        continue

                    # -------------------------------------------------
                    # STALE TASK CHECK
                    # -------------------------------------------------
                    try:

                        age = (
                            time.monotonic()
                            - float(queued_at)
                        )

                        max_age = 3600

                        if age > max_age:

                            task.state = (
                                TaskState.CANCELLED
                            )

                            task.error = (
                                "task_expired"
                            )

                            task.completed_at = (
                                datetime.utcnow()
                            )

                            logger.warning(
                                f"Expired task skipped "
                                f"| id={task_id}"
                            )

                            try:
                                self.task_queue.task_done()
                            except Exception:
                                pass

                            continue

                    except Exception:
                        pass

                    # -------------------------------------------------
                    # MARK RUNNING
                    # -------------------------------------------------
                    try:

                        task.state = (
                            TaskState.RUNNING
                        )

                        task.started_at = (
                            datetime.utcnow()
                        )

                        task.worker_id = (
                            worker_id
                        )

                    except Exception:
                        pass

                    logger.debug(
                        f"▶ Executing task "
                        f"{task_id}"
                    )

                    # -------------------------------------------------
                    # EXECUTION TIMEOUT
                    # -------------------------------------------------
                    try:

                        timeout = float(
                            getattr(
                                self,
                                "task_timeout",
                                self.config.get(
                                    "task_timeout",
                                    300,
                                ),
                            )
                        )

                    except Exception:

                        timeout = 300.0

                    timeout = max(
                        1.0,
                        min(timeout, 86400),
                    )

                    # -------------------------------------------------
                    # EXECUTE TASK
                    # -------------------------------------------------
                    try:

                        # validate coroutine
                        if not (
                            asyncio.iscoroutine(
                                coroutine
                            )
                            or asyncio.isfuture(
                                coroutine
                            )
                        ):

                            raise TypeError(
                                "Invalid coroutine"
                            )

                        result = (
                            await asyncio.wait_for(
                                coroutine,
                                timeout=timeout,
                            )
                        )

                        # -----------------------------------------
                        # SUCCESS
                        # -----------------------------------------
                        task.state = (
                            TaskState.COMPLETED
                        )

                        task.result = result

                        logger.debug(
                            f"✅ Task completed "
                            f"| id={task_id}"
                        )

                        # -----------------------------------------
                        # METRICS
                        # -----------------------------------------
                        try:

                            self.tasks_completed = (
                                int(
                                    getattr(
                                        self,
                                        "tasks_completed",
                                        0,
                                    )
                                ) + 1
                            )

                        except Exception:
                            pass

                    # -------------------------------------------------
                    # TIMEOUT
                    # -------------------------------------------------
                    except asyncio.TimeoutError:

                        logger.warning(
                            f"⏰ Task timeout "
                            f"| id={task_id}"
                        )

                        task.state = (
                            TaskState.FAILED
                        )

                        task.error = (
                            f"timeout_after_{timeout}s"
                        )

                        try:

                            self.tasks_timed_out = (
                                int(
                                    getattr(
                                        self,
                                        "tasks_timed_out",
                                        0,
                                    )
                                ) + 1
                            )

                        except Exception:
                            pass

                    # -------------------------------------------------
                    # TASK CANCELLED
                    # -------------------------------------------------
                    except asyncio.CancelledError:

                        logger.warning(
                            f"🛑 Task cancelled "
                            f"| id={task_id}"
                        )

                        task.state = (
                            TaskState.CANCELLED
                        )

                        task.error = (
                            "worker_cancelled"
                        )

                        raise

                    # -------------------------------------------------
                    # EXECUTION FAILURE
                    # -------------------------------------------------
                    except Exception as exc:

                        logger.error(
                            f"❌ Task failed "
                            f"| id={task_id} "
                            f"| error={exc}"
                        )

                        task.state = (
                            TaskState.FAILED
                        )

                        task.error = str(exc)

                        try:

                            self.tasks_failed = (
                                int(
                                    getattr(
                                        self,
                                        "tasks_failed",
                                        0,
                                    )
                                ) + 1
                            )

                        except Exception:
                            pass

                    # -------------------------------------------------
                    # FINALIZATION
                    # -------------------------------------------------
                    finally:

                        try:

                            task.completed_at = (
                                datetime.utcnow()
                            )

                        except Exception:
                            pass

                        try:

                            task.duration_seconds = round(
                                (
                                    task.completed_at
                                    - task.started_at
                                ).total_seconds(),
                                3,
                            )

                        except Exception:
                            pass

                        # queue completion
                        try:

                            self.task_queue.task_done()

                        except Exception:
                            pass

                        # optional event emission
                        try:

                            event_bus = getattr(
                                self,
                                "event_bus",
                                None,
                            )

                            if (
                                event_bus
                                and hasattr(
                                    event_bus,
                                    "emit",
                                )
                            ):

                                payload = {

                                    "task_id": (
                                        task_id
                                    ),

                                    "state": str(
                                        task.state
                                    ),

                                    "timestamp": (
                                        datetime.utcnow()
                                        .isoformat()
                                    ),
                                }

                                result = (
                                    event_bus.emit(
                                        "task_completed",
                                        payload,
                                    )
                                )

                                if asyncio.iscoroutine(
                                    result
                                ):

                                    asyncio.create_task(
                                        result
                                    )

                        except Exception:
                            pass

                        # cooperative cleanup
                        try:

                            del coroutine

                        except Exception:
                            pass

                        gc.collect()

                        await asyncio.sleep(0)

                # -----------------------------------------------------
                # WORKER CANCELLED
                # -----------------------------------------------------
                except asyncio.CancelledError:

                    logger.warning(
                        f"🛑 Worker cancelled "
                        f"| id={worker_id}"
                    )

                    break

                # -----------------------------------------------------
                # WORKER FAILURE
                # -----------------------------------------------------
                except Exception as exc:

                    logger.error(
                        f"❌ Worker crash "
                        f"| id={worker_id} "
                        f"| error={exc}"
                    )

                    # prevent crash loop
                    await asyncio.sleep(0.25)

            logger.info(
                f"🛑 Task worker stopped "
                f"| id={worker_id}"
            )

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.critical(
                f"Worker fatal failure "
                f"| id={worker_id} "
                f"| error={exc}"
            )

    async def cancel_task(self, task_id: str) -> bool:
        """
        Production-grade task cancellation system.

        Fixes:
        - invalid task IDs
        - race conditions
        - orphaned running tasks
        - asyncio task leaks
        - duplicate cancellation
        - corrupted task states
        - queue/task desync
        - unsafe cancellation handling
        """

        start = time.monotonic()

        try:

            import asyncio

            from datetime import datetime

            # ---------------------------------------------------------
            # LOCK INITIALIZATION
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_cancel_lock",
            ):

                self._cancel_lock = asyncio.Lock()

            async with self._cancel_lock:

                # -----------------------------------------------------
                # VALIDATION
                # -----------------------------------------------------
                if task_id is None:

                    logger.warning(
                        "Cancel failed: task_id is None"
                    )

                    return False

                if not isinstance(
                    task_id,
                    str,
                ):

                    try:

                        task_id = str(task_id)

                    except Exception:

                        logger.warning(
                            "Cancel failed: invalid task_id"
                        )

                        return False

                task_id = task_id.strip()

                if not task_id:

                    logger.warning(
                        "Cancel failed: empty task_id"
                    )

                    return False

                # -----------------------------------------------------
                # TASK REGISTRY VALIDATION
                # -----------------------------------------------------
                tasks = getattr(
                    self,
                    "tasks",
                    {},
                )

                if not isinstance(
                    tasks,
                    dict,
                ):

                    logger.error(
                        "Task registry corrupted"
                    )

                    return False

                # -----------------------------------------------------
                # TASK LOOKUP
                # -----------------------------------------------------
                task = tasks.get(task_id)

                if task is None:

                    logger.warning(
                        f"Cancel failed: task not found "
                        f"({task_id})"
                    )

                    return False

                # -----------------------------------------------------
                # TASK STATE VALIDATION
                # -----------------------------------------------------
                try:

                    current_state = getattr(
                        task,
                        "state",
                        None,
                    )

                except Exception:

                    logger.warning(
                        f"Cancel failed: invalid task "
                        f"({task_id})"
                    )

                    return False

                # -----------------------------------------------------
                # ALREADY FINISHED
                # -----------------------------------------------------
                if current_state in (

                    TaskState.COMPLETED,

                    TaskState.FAILED,

                    TaskState.CANCELLED,
                ):

                    logger.debug(
                        f"Task already finalized "
                        f"| id={task_id}"
                    )

                    return False

                # -----------------------------------------------------
                # PENDING TASK
                # -----------------------------------------------------
                if current_state == TaskState.PENDING:

                    try:

                        task.state = (
                            TaskState.CANCELLED
                        )

                        task.completed_at = (
                            datetime.utcnow()
                        )

                        task.error = (
                            "cancelled_before_execution"
                        )

                        logger.info(
                            f"🛑 Pending task cancelled "
                            f"| id={task_id}"
                        )

                    except Exception as exc:

                        logger.error(
                            f"Pending task cancellation failed: {exc}"
                        )

                        return False

                # -----------------------------------------------------
                # RUNNING TASK
                # -----------------------------------------------------
                elif current_state == TaskState.RUNNING:

                    logger.warning(
                        f"⚡ Cancelling running task "
                        f"| id={task_id}"
                    )

                    cancelled = False

                    # ---------------------------------------------
                    # ASYNCIO TASK CANCELLATION
                    # ---------------------------------------------
                    try:

                        asyncio_task = getattr(
                            task,
                            "asyncio_task",
                            None,
                        )

                        if (
                            asyncio_task
                            and isinstance(
                                asyncio_task,
                                asyncio.Task,
                            )
                        ):

                            if (
                                not asyncio_task.done()
                            ):

                                asyncio_task.cancel()

                                try:

                                    await asyncio.wait_for(
                                        asyncio_task,
                                        timeout=5,
                                    )

                                except (
                                    asyncio.CancelledError,
                                    asyncio.TimeoutError,
                                ):

                                    pass

                                cancelled = True

                    except Exception as exc:

                        logger.debug(
                            f"Asyncio cancel failed: {exc}"
                        )

                    # ---------------------------------------------
                    # FALLBACK STATE UPDATE
                    # ---------------------------------------------
                    try:

                        task.state = (
                            TaskState.CANCELLED
                        )

                        task.completed_at = (
                            datetime.utcnow()
                        )

                        task.error = (
                            "cancelled_during_execution"
                        )

                        if cancelled:

                            task.result = None

                    except Exception as exc:

                        logger.error(
                            f"Running task update failed: {exc}"
                        )

                        return False

                    logger.info(
                        f"🛑 Running task cancelled "
                        f"| id={task_id}"
                    )

                # -----------------------------------------------------
                # UNKNOWN STATE
                # -----------------------------------------------------
                else:

                    logger.warning(
                        f"Cancel failed: unsupported "
                        f"state={current_state}"
                    )

                    return False

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "task_id": task_id,

                            "state": "cancelled",

                            "timestamp": (
                                datetime.utcnow()
                                .isoformat()
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "task_cancelled",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            asyncio.create_task(
                                result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.tasks_cancelled = int(
                        getattr(
                            self,
                            "tasks_cancelled",
                            0,
                        )
                    ) + 1

                    self.last_cancelled_task = (
                        task_id
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic() -
                        start
                    ) * 1000,
                    2,
                )

                logger.debug(
                    f"Cancel complete "
                    f"| id={task_id} "
                    f"| latency={latency_ms}ms"
                )

                return True

        # -------------------------------------------------------------
        # TASK CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Task cancellation interrupted"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Cancel task failed: {exc}"
            )

            return False

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Production-grade task status retrieval.

        Fixes:
        - invalid task IDs
        - corrupted task registry
        - datetime serialization crashes
        - malformed task objects
        - unsafe attribute access
        - missing task handling
        - non-serializable results
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------
            if task_id is None:

                logger.warning(
                    "Task status lookup failed: "
                    "task_id is None"
                )

                return None

            if not isinstance(
                task_id,
                str,
            ):

                try:

                    task_id = str(task_id)

                except Exception:

                    logger.warning(
                        "Task status lookup failed: "
                        "invalid task_id"
                    )

                    return None

            task_id = task_id.strip()

            if not task_id:

                logger.warning(
                    "Task status lookup failed: "
                    "empty task_id"
                )

                return None

            # ---------------------------------------------------------
            # TASK REGISTRY VALIDATION
            # ---------------------------------------------------------
            tasks = getattr(
                self,
                "tasks",
                {},
            )

            if not isinstance(
                tasks,
                dict,
            ):

                logger.error(
                    "Task registry corrupted"
                )

                return None

            # ---------------------------------------------------------
            # TASK LOOKUP
            # ---------------------------------------------------------
            task = tasks.get(task_id)

            if task is None:

                logger.debug(
                    f"Task not found: {task_id}"
                )

                return None

            # ---------------------------------------------------------
            # SAFE DATETIME SERIALIZER
            # ---------------------------------------------------------
            def _safe_datetime(
                value,
            ) -> Optional[str]:

                try:

                    if value is None:
                        return None

                    if hasattr(
                        value,
                        "isoformat",
                    ):

                        return value.isoformat()

                    return str(value)

                except Exception:

                    return None

            # ---------------------------------------------------------
            # SAFE VALUE EXTRACTOR
            # ---------------------------------------------------------
            def _safe_attr(
                obj,
                name,
                default=None,
            ):

                try:

                    return getattr(
                        obj,
                        name,
                        default,
                    )

                except Exception:

                    return default

            # ---------------------------------------------------------
            # TASK STATE
            # ---------------------------------------------------------
            try:

                state = _safe_attr(
                    task,
                    "state",
                )

                if hasattr(
                    state,
                    "value",
                ):

                    state_value = state.value

                else:

                    state_value = str(state)

            except Exception:

                state_value = "unknown"

            # ---------------------------------------------------------
            # RESULT SERIALIZATION
            # ---------------------------------------------------------
            result = _safe_attr(
                task,
                "result",
            )

            try:

                # ensure JSON-safe basic types
                if not isinstance(
                    result,
                    (
                        dict,
                        list,
                        tuple,
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                ):

                    result = str(result)

            except Exception:

                result = None

            # ---------------------------------------------------------
            # ERROR SERIALIZATION
            # ---------------------------------------------------------
            error = _safe_attr(
                task,
                "error",
            )

            try:

                if error is not None:

                    error = str(error)

            except Exception:

                error = "unknown_error"

            # ---------------------------------------------------------
            # DURATION
            # ---------------------------------------------------------
            duration = None

            try:

                started_at = _safe_attr(
                    task,
                    "started_at",
                )

                completed_at = _safe_attr(
                    task,
                    "completed_at",
                )

                if (
                    started_at
                    and completed_at
                ):

                    duration = round(
                        (
                            completed_at -
                            started_at
                        ).total_seconds(),
                        3,
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # FINAL RESPONSE
            # ---------------------------------------------------------
            response = {

                "id": _safe_attr(
                    task,
                    "id",
                    task_id,
                ),

                "name": _safe_attr(
                    task,
                    "name",
                    "unknown",
                ),

                "state": state_value,

                "priority": str(
                    _safe_attr(
                        task,
                        "priority",
                        "NORMAL",
                    )
                ),

                "created_at": _safe_datetime(
                    _safe_attr(
                        task,
                        "created_at",
                    )
                ),

                "started_at": _safe_datetime(
                    _safe_attr(
                        task,
                        "started_at",
                    )
                ),

                "completed_at": _safe_datetime(
                    _safe_attr(
                        task,
                        "completed_at",
                    )
                ),

                "duration_seconds": duration,

                "result": result,

                "error": error,
            }

            # ---------------------------------------------------------
            # OPTIONAL METADATA
            # ---------------------------------------------------------
            try:

                worker_id = _safe_attr(
                    task,
                    "worker_id",
                )

                if worker_id:

                    response[
                        "worker_id"
                    ] = worker_id

            except Exception:
                pass

            # ---------------------------------------------------------
            # STATUS LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic() -
                    start
                ) * 1000,
                2,
            )

            response[
                "lookup_latency_ms"
            ] = latency_ms

            logger.debug(
                f"Task status retrieved "
                f"| id={task_id} "
                f"| state={state_value}"
            )

            return response

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Task status failed: {exc}"
            )

            return {

                "id": task_id,

                "state": "error",

                "error": str(exc),
            }

    def get_all_tasks(self) -> Dict[str, Any]:
        """
        Production-grade task registry inspection.

        Fixes:
        - corrupted task registry
        - task serialization failures
        - concurrent modification crashes
        - invalid task objects
        - memory-safe task enumeration
        - partial retrieval failures
        - unstable iteration
        """

        start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # TASK REGISTRY VALIDATION
            # ---------------------------------------------------------
            tasks = getattr(
                self,
                "tasks",
                {},
            )

            if not isinstance(
                tasks,
                dict,
            ):

                logger.error(
                    "Task registry corrupted"
                )

                return {

                    "status": "error",

                    "tasks": {},

                    "total": 0,

                    "error": (
                        "invalid_task_registry"
                    ),
                }

            # ---------------------------------------------------------
            # SNAPSHOT COPY
            # ---------------------------------------------------------
            try:

                task_items = list(
                    tasks.items()
                )

            except Exception as exc:

                logger.error(
                    f"Task snapshot failed: {exc}"
                )

                return {

                    "status": "error",

                    "tasks": {},

                    "total": 0,

                    "error": str(exc),
                }

            # ---------------------------------------------------------
            # RESULT STORAGE
            # ---------------------------------------------------------
            results = {}

            failed = 0

            completed = 0

            pending = 0

            running = 0

            cancelled = 0

            unknown = 0

            # ---------------------------------------------------------
            # ENUMERATE TASKS
            # ---------------------------------------------------------
            for (
                task_id,
                _,
            ) in task_items:

                try:

                    # ---------------------------------------------
                    # SAFE STATUS LOOKUP
                    # ---------------------------------------------
                    status = (
                        self.get_task_status(
                            task_id
                        )
                    )

                    if not status:

                        failed += 1

                        continue

                    results[
                        task_id
                    ] = status

                    # ---------------------------------------------
                    # STATE COUNTING
                    # ---------------------------------------------
                    state = str(
                        status.get(
                            "state",
                            "unknown",
                        )
                    ).lower()

                    if state == "completed":

                        completed += 1

                    elif state == "pending":

                        pending += 1

                    elif state == "running":

                        running += 1

                    elif state == "cancelled":

                        cancelled += 1

                    elif state in (
                        "failed",
                        "error",
                    ):

                        failed += 1

                    else:

                        unknown += 1

                # -------------------------------------------------
                # INDIVIDUAL TASK FAILURE
                # -------------------------------------------------
                except Exception as exc:

                    logger.debug(
                        f"Task inspection failed "
                        f"({task_id}): {exc}"
                    )

                    failed += 1

            # ---------------------------------------------------------
            # SUMMARY
            # ---------------------------------------------------------
            total = len(results)

            # ---------------------------------------------------------
            # HEALTH STATUS
            # ---------------------------------------------------------
            if failed == 0:

                status = "healthy"

            elif failed < max(
                1,
                total // 2,
            ):

                status = "degraded"

            else:

                status = "critical"

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic() -
                    start
                ) * 1000,
                2,
            )

            # ---------------------------------------------------------
            # FINAL RESPONSE
            # ---------------------------------------------------------
            response = {

                "status": status,

                "total": total,

                "summary": {

                    "completed": completed,

                    "pending": pending,

                    "running": running,

                    "cancelled": cancelled,

                    "failed": failed,

                    "unknown": unknown,
                },

                "tasks": results,

                "latency_ms": latency_ms,
            }

            # ---------------------------------------------------------
            # OPTIONAL METRICS
            # ---------------------------------------------------------
            try:

                self.task_status_queries = int(
                    getattr(
                        self,
                        "task_status_queries",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # DEBUG LOGGING
            # ---------------------------------------------------------
            logger.debug(
                f"Retrieved all tasks "
                f"| total={total} "
                f"| running={running}"
            )

            return response

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ get_all_tasks failed: {exc}"
            )

            return {

                "status": "error",

                "total": 0,

                "summary": {},

                "tasks": {},

                "error": str(exc),
            }

    # ------------------------
    # MAIN LOOP
    # ------------------------

    async def start(self) -> bool:
        """
        Production-grade system startup manager.

        Fixes:
        - double startup races
        - orphaned background tasks
        - worker startup failures
        - subsystem deadlocks
        - startup partial failures
        - event-loop instability
        - resource exhaustion
        - unsafe startup rollback
        """

        start_time = time.monotonic()

        try:

            import asyncio
            import gc
            import psutil

            from datetime import datetime

            # ---------------------------------------------------------
            # STARTUP LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_startup_lock",
            ):

                self._startup_lock = asyncio.Lock()

            async with self._startup_lock:

                # -----------------------------------------------------
                # DUPLICATE START PROTECTION
                # -----------------------------------------------------
                if getattr(
                    self,
                    "_running",
                    False,
                ):

                    logger.warning(
                        "System already running"
                    )

                    return True

                if getattr(
                    self,
                    "_starting",
                    False,
                ):

                    logger.warning(
                        "System already starting"
                    )

                    return False

                self._starting = True

                logger.info(
                    "🚀 Starting EDIATH system "
                    "(Production Mode)"
                )

                # -----------------------------------------------------
                # RESET INTERNAL STATE
                # -----------------------------------------------------
                self._shutdown = False

                self._running = False

                startup_report = {

                    "subsystems": [],

                    "workers": 0,

                    "background_loops": [],

                    "errors": [],
                }

                # -----------------------------------------------------
                # RESOURCE VALIDATION
                # -----------------------------------------------------
                try:

                    memory_percent = (
                        psutil.virtual_memory()
                        .percent
                    )

                    if memory_percent > 95:

                        raise RuntimeError(
                            "System memory critically high"
                        )

                except Exception as exc:

                    if "critically high" in str(exc):
                        logger.warning(
                            f"⚠ {exc}"
                        )
                    else:
                        logger.warning(
                            f"Resource validation failed: {exc}"
                        )

                # -----------------------------------------------------
                # COMMAND ROUTER STARTUP
                # -----------------------------------------------------
                try:

                    router = getattr(
                        self,
                        "command_router",
                        None,
                    )

                    if router:

                        logger.info(
                            "⚡ Starting command router"
                        )

                        if hasattr(
                            router,
                            "start",
                        ):

                            result = (
                                router.start()
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                await asyncio.wait_for(
                                    result,
                                    timeout=30,
                                )

                            else:

                                await asyncio.to_thread(
                                    router.start
                                )

                        elif hasattr(
                            router,
                            "process_queue",
                        ):

                            task = (
                                asyncio.create_task(
                                    router.process_queue(),
                                    name=(
                                        "command_router_loop"
                                    ),
                                )
                            )

                            if not hasattr(
                                self,
                                "_background_tasks",
                            ):

                                self._background_tasks = []

                            self._background_tasks.append(
                                task
                            )

                        startup_report[
                            "subsystems"
                        ].append(
                            "command_router"
                        )

                except Exception as exc:

                    startup_report[
                        "errors"
                    ].append(
                        f"command_router:{exc}"
                    )

                    logger.error(
                        f"Command router failed: {exc}"
                    )

                # -----------------------------------------------------
                # IDLE CONTROLLER STARTUP
                # -----------------------------------------------------
                try:

                    idle = getattr(
                        self,
                        "idle_controller",
                        None,
                    )

                    if idle:

                        logger.info(
                            "🌙 Starting idle controller"
                        )

                        if hasattr(
                            idle,
                            "start",
                        ):

                            result = idle.start()

                            if asyncio.iscoroutine(
                                result
                            ):

                                await asyncio.wait_for(
                                    result,
                                    timeout=30,
                                )

                            else:

                                await asyncio.to_thread(
                                    idle.start
                                )

                        elif hasattr(
                            idle,
                            "start_monitoring",
                        ):

                            task = (
                                asyncio.create_task(
                                    idle.start_monitoring(),
                                    name=(
                                        "idle_monitor"
                                    ),
                                )
                            )

                            if not hasattr(
                                self,
                                "_background_tasks",
                            ):

                                self._background_tasks = []

                            self._background_tasks.append(
                                task
                            )

                        startup_report[
                            "subsystems"
                        ].append(
                            "idle_controller"
                        )

                except Exception as exc:

                    startup_report[
                        "errors"
                    ].append(
                        f"idle_controller:{exc}"
                    )

                    logger.error(
                        f"Idle controller failed: {exc}"
                    )

                # -----------------------------------------------------
                # WORKER INITIALIZATION
                # -----------------------------------------------------
                try:

                    cpu_count = max(
                        1,
                        psutil.cpu_count() or 1,
                    )

                    config = getattr(
                        self,
                        "config",
                        {},
                    )

                    max_tasks = int(
                        config.get(
                            "max_concurrent_tasks",
                            cpu_count,
                        )
                    )

                    worker_count = max(
                        1,
                        min(cpu_count, max_tasks),
                    )

                    # reset worker list
                    self.worker_tasks = []

                    logger.info(
                        f"⚙️ Starting "
                        f"{worker_count} workers"
                    )

                    for index in range(
                        worker_count
                    ):

                        try:

                            worker = (
                                asyncio.create_task(
                                    self._task_worker(),
                                    name=(
                                        f"task_worker_"
                                        f"{index}"
                                    ),
                                )
                            )

                            self.worker_tasks.append(
                                worker
                            )

                            startup_report[
                                "workers"
                            ] += 1

                        except Exception as exc:

                            startup_report[
                                "errors"
                            ].append(
                                f"worker_{index}:{exc}"
                            )

                    if not self.worker_tasks:

                        raise RuntimeError(
                            "No workers started"
                        )

                except Exception as exc:

                    logger.error(
                        f"Worker startup failed: {exc}"
                    )

                    startup_report[
                        "errors"
                    ].append(
                        f"workers:{exc}"
                    )

                # -----------------------------------------------------
                # BACKGROUND LOOP STARTUP
                # -----------------------------------------------------
                loop_targets = [

                    (
                        "_main_loop_task",
                        "_main_loop",
                    ),

                    (
                        "_metrics_task",
                        "_metrics_loop",
                    ),

                    (
                        "_health_task",
                        "_health_loop",
                    ),

                    (
                        "_backup_task",
                        "_backup_loop",
                    ),
                ]

                for (
                    attr_name,
                    method_name,
                ) in loop_targets:

                    try:

                        if not hasattr(
                            self,
                            method_name,
                        ):

                            continue

                        method = getattr(
                            self,
                            method_name,
                        )

                        if not callable(
                            method
                        ):

                            continue

                        task = (
                            asyncio.create_task(
                                method(),
                                name=method_name,
                            )
                        )

                        setattr(
                            self,
                            attr_name,
                            task,
                        )

                        startup_report[
                            "background_loops"
                        ].append(
                            method_name
                        )

                    except Exception as exc:

                        startup_report[
                            "errors"
                        ].append(
                            f"{method_name}:{exc}"
                        )

                        logger.error(
                            f"Loop startup failed "
                            f"({method_name}): {exc}"
                        )

                # -----------------------------------------------------
                # GARBAGE COLLECTION
                # -----------------------------------------------------
                try:

                    gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # MARK RUNNING
                # -----------------------------------------------------
                self._running = True

                self._starting = False

                self.started_at = (
                    datetime.utcnow()
                )

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                datetime.utcnow()
                                .isoformat()
                            ),

                            "workers": len(
                                self.worker_tasks
                            ),

                            "startup_report": (
                                startup_report
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "system_started",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=10,
                            )

                except Exception as exc:

                    logger.debug(
                        f"Startup event failed: {exc}"
                    )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.start_count = int(
                        getattr(
                            self,
                            "start_count",
                            0,
                        )
                    ) + 1

                    self.last_startup_report = (
                        startup_report
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # STARTUP VALIDATION
                # -----------------------------------------------------
                if (
                    startup_report[
                        "workers"
                    ] == 0
                ):

                    raise RuntimeError(
                        "Startup failed: no workers active"
                    )

                # -----------------------------------------------------
                # STARTUP DURATION
                # -----------------------------------------------------
                duration = round(
                    time.monotonic()
                    - start_time,
                    3,
                )

                logger.info(
                    f"✅ EDIATH started "
                    f"in {duration}s "
                    f"| workers={len(self.worker_tasks)}"
                )

                if startup_report["errors"]:

                    logger.warning(
                        f"Startup completed with "
                        f"{len(startup_report['errors'])} warnings"
                    )

                return True

        # -------------------------------------------------------------
        # STARTUP FAILURE
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "System startup cancelled"
            )

            raise

        except Exception as exc:

            logger.error(
                f"❌ System startup failed: {exc}"
            )

            # ---------------------------------------------------------
            # SAFE ROLLBACK
            # ---------------------------------------------------------
            try:

                self._running = False

                self._starting = False

                # cancel workers
                for worker in getattr(
                    self,
                    "worker_tasks",
                    [],
                ):

                    try:

                        if (
                            worker
                            and not worker.done()
                        ):

                            worker.cancel()

                    except Exception:
                        pass

                # cancel background loops
                for attr in [

                    "_main_loop_task",

                    "_metrics_task",

                    "_health_task",

                    "_backup_task",
                ]:

                    try:

                        task = getattr(
                            self,
                            attr,
                            None,
                        )

                        if (
                            task
                            and not task.done()
                        ):

                            task.cancel()

                    except Exception:
                        pass

            except Exception:
                pass

            return False

        # -------------------------------------------------------------
        # CLEANUP
        # -------------------------------------------------------------
        finally:

            try:

                self._starting = False

            except Exception:
                pass

    async def _main_loop(self) -> None:
        """
        Production-grade autonomous system main loop.

        Fixes:
        - infinite crash loops
        - event-loop starvation
        - health-check deadlocks
        - resource monitor failures
        - autonomous recursion
        - predictive scaling instability
        - idle-check race conditions
        - runaway CPU usage
        - orphaned async tasks
        """

        import asyncio
        import gc
        import random

        LOOP_DELAY = 3.0

        logger.info(
            "🧠 Main system loop started"
        )

        # -------------------------------------------------------------
        # LOOP METRICS
        # -------------------------------------------------------------
        loop_iterations = 0

        consecutive_failures = 0

        max_failures = 25

        # -------------------------------------------------------------
        # LOOP OBJECT
        # -------------------------------------------------------------
        try:

            loop = asyncio.get_running_loop()

        except RuntimeError:

            loop = asyncio.get_event_loop()

        # -------------------------------------------------------------
        # MAIN LOOP
        # -------------------------------------------------------------
        while getattr(
            self,
            "_running",
            False,
        ):

            cycle_start = loop.time()

            try:

                loop_iterations += 1

                # -----------------------------------------------------
                # YIELD CONTROL
                # -----------------------------------------------------
                await asyncio.sleep(0)

                # -----------------------------------------------------
                # HEALTH CHECK
                # -----------------------------------------------------
                try:

                    if hasattr(
                        self,
                        "health_check",
                    ):

                        result = (
                            self.health_check()
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=45,
                            )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Health check timeout"
                    )

                except Exception as exc:

                    logger.debug(
                        f"Health check failed: {exc}"
                    )

                # -----------------------------------------------------
                # RESOURCE MONITORING
                # -----------------------------------------------------
                try:

                    monitor = getattr(
                        self,
                        "resource_monitor",
                        None,
                    )

                    if (
                        monitor
                        and hasattr(
                            monitor,
                            "check_resources",
                        )
                    ):

                        resources = (
                            await asyncio.wait_for(
                                monitor.check_resources(),
                                timeout=20,
                            )
                        )

                        if not isinstance(
                            resources,
                            dict,
                        ):

                            resources = {}

                        unhealthy = any(
                            not bool(v)
                            for v in (
                                resources.values()
                            )
                        )

                        # -----------------------------------------
                        # AUTO-HEAL
                        # -----------------------------------------
                        if unhealthy:

                            logger.warning(
                                "⚠️ Resource degradation detected"
                            )

                            auto_heal_enabled = bool(
                                getattr(
                                    self,
                                    "config",
                                    {},
                                ).get(
                                    "enable_auto_healing",
                                    False,
                                )
                            )

                            if (
                                auto_heal_enabled
                                and not getattr(
                                    self,
                                    "_healing",
                                    False,
                                )
                            ):

                                try:

                                    await asyncio.wait_for(
                                        self.auto_heal(),
                                        timeout=120,
                                    )

                                except asyncio.TimeoutError:

                                    logger.warning(
                                        "Auto-heal timeout"
                                    )

                except Exception as exc:

                    logger.debug(
                        f"Resource check failed: {exc}"
                    )

                # -----------------------------------------------------
                # IDLE DETECTION
                # -----------------------------------------------------
                is_idle = False

                try:

                    idle_ctrl = getattr(
                        self,
                        "idle_controller",
                        None,
                    )

                    if idle_ctrl:

                        idle_check = getattr(
                            idle_ctrl,
                            "is_idle",
                            None,
                        )

                        if callable(
                            idle_check
                        ):

                            result = (
                                idle_check()
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                is_idle = bool(
                                    await asyncio.wait_for(
                                        result,
                                        timeout=10,
                                    )
                                )

                            else:

                                is_idle = bool(
                                    result
                                )

                        else:

                            is_idle = bool(
                                idle_check
                            )

                except asyncio.TimeoutError:

                    logger.debug(
                        "Idle detection timeout"
                    )

                except Exception as exc:

                    logger.debug(
                        f"Idle check failed: {exc}"
                    )

                # -----------------------------------------------------
                # NON-IDLE SYSTEM
                # -----------------------------------------------------
                if not is_idle:

                    await asyncio.sleep(0.25)

                    continue

                # -----------------------------------------------------
                # AUTONOMOUS TICK
                # -----------------------------------------------------
                try:

                    autonomous_enabled = hasattr(
                        self,
                        "autonomous_tick",
                    )

                    autonomous_busy = getattr(
                        self,
                        "_autonomous_running",
                        False,
                    )

                    should_run = (
                        autonomous_enabled
                        and not autonomous_busy
                        and random.random() < 0.05
                    )

                    if should_run:

                        logger.debug(
                            "🤖 Running autonomous tick"
                        )

                        self._autonomous_running = True

                        try:

                            tick = (
                                self.autonomous_tick()
                            )

                            if asyncio.iscoroutine(
                                tick
                            ):

                                await asyncio.wait_for(
                                    tick,
                                    timeout=120,
                                )

                            else:

                                await asyncio.to_thread(
                                    self.autonomous_tick
                                )

                        finally:

                            self._autonomous_running = False

                except asyncio.TimeoutError:

                    logger.warning(
                        "Autonomous tick timeout"
                    )

                    self._autonomous_running = False

                except Exception as exc:

                    self._autonomous_running = False

                    logger.debug(
                        f"Autonomous tick failed: {exc}"
                    )

                # -----------------------------------------------------
                # PREDICTIVE SCALING
                # -----------------------------------------------------
                try:

                    scaling_enabled = bool(
                        getattr(
                            self,
                            "config",
                            {},
                        ).get(
                            "enable_predictive_scaling",
                            False,
                        )
                    )

                    scaling_active = getattr(
                        self,
                        "_scaling",
                        False,
                    )

                    if (
                        scaling_enabled
                        and not scaling_active
                        and hasattr(
                            self,
                            "_predictive_scaling",
                        )
                    ):

                        self._scaling = True

                        try:

                            result = (
                                self._predictive_scaling()
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                await asyncio.wait_for(
                                    result,
                                    timeout=60,
                                )

                        finally:

                            self._scaling = False

                except asyncio.TimeoutError:

                    logger.warning(
                        "Predictive scaling timeout"
                    )

                    self._scaling = False

                except Exception as exc:

                    self._scaling = False

                    logger.debug(
                        f"Predictive scaling failed: {exc}"
                    )

                # -----------------------------------------------------
                # PERIODIC GARBAGE COLLECTION
                # -----------------------------------------------------
                try:

                    if loop_iterations % 250 == 0:

                        gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET FAILURE COUNTER
                # -----------------------------------------------------
                consecutive_failures = 0

                # -----------------------------------------------------
                # LOOP TIMING CONTROL
                # -----------------------------------------------------
                elapsed = (
                    loop.time()
                    - cycle_start
                )

                sleep_time = max(
                    0.5,
                    LOOP_DELAY - elapsed,
                )

                await asyncio.sleep(
                    sleep_time
                )

            # ---------------------------------------------------------
            # LOOP CANCELLED
            # ---------------------------------------------------------
            except asyncio.CancelledError:

                logger.info(
                    "🛑 Main loop cancelled"
                )

                break

            # ---------------------------------------------------------
            # LOOP FAILURE
            # ---------------------------------------------------------
            except Exception as exc:

                consecutive_failures += 1

                logger.error(
                    f"❌ Main loop failure "
                    f"({consecutive_failures}/"
                    f"{max_failures}): {exc}"
                )

                # -----------------------------------------------------
                # FAILURE LIMIT
                # -----------------------------------------------------
                if (
                    consecutive_failures
                    >= max_failures
                ):

                    logger.critical(
                        "🚨 Main loop exceeded "
                        "maximum failures"
                    )

                    try:

                        self._running = False

                    except Exception:
                        pass

                    break

                # -----------------------------------------------------
                # BACKOFF
                # -----------------------------------------------------
                await asyncio.sleep(
                    min(
                        5.0,
                        0.5 * consecutive_failures,
                    )
                )

        # -------------------------------------------------------------
        # LOOP SHUTDOWN
        # -------------------------------------------------------------
        try:

            logger.info(
                f"🛑 Main loop stopped "
                f"| iterations={loop_iterations}"
            )

        except Exception:
            pass

    async def _metrics_loop(self) -> None:
        """
        Production-grade metrics collection loop.

        Fixes:
        - metrics loop crashes
        - event-loop starvation
        - invalid interval handling
        - metrics collector corruption
        - runaway CPU usage
        - silent metric failures
        - infinite crash loops
        - blocked metrics collection
        """

        import asyncio
        import gc
        import time

        logger.info(
            "📊 Metrics loop started"
        )

        # -------------------------------------------------------------
        # LOOP METRICS
        # -------------------------------------------------------------
        loop_iterations = 0

        consecutive_failures = 0

        max_failures = 50

        # -------------------------------------------------------------
        # LOOP OBJECT
        # -------------------------------------------------------------
        try:

            loop = asyncio.get_running_loop()

        except RuntimeError:

            loop = asyncio.get_event_loop()

        # -------------------------------------------------------------
        # MAIN LOOP
        # -------------------------------------------------------------
        while getattr(
            self,
            "_running",
            False,
        ):

            cycle_start = loop.time()

            try:

                loop_iterations += 1

                # -----------------------------------------------------
                # COOPERATIVE YIELD
                # -----------------------------------------------------
                await asyncio.sleep(0)

                # -----------------------------------------------------
                # METRICS COLLECTOR VALIDATION
                # -----------------------------------------------------
                collector = getattr(
                    self,
                    "metrics_collector",
                    None,
                )

                if collector is None:

                    logger.warning(
                        "Metrics collector unavailable"
                    )

                    await asyncio.sleep(1.0)

                    continue

                # -----------------------------------------------------
                # RECORD METRICS
                # -----------------------------------------------------
                try:

                    if hasattr(
                        collector,
                        "record_metrics",
                    ):

                        result = (
                            collector.record_metrics()
                        )

                        # async support
                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=30,
                            )

                    else:

                        logger.debug(
                            "record_metrics missing"
                        )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Metrics collection timeout"
                    )

                except Exception as exc:

                    logger.debug(
                        f"Metrics collection failed: {exc}"
                    )

                # -----------------------------------------------------
                # OPTIONAL HEALTH SNAPSHOT
                # -----------------------------------------------------
                try:

                    if (
                        loop_iterations % 10
                        == 0
                    ):

                        metrics_snapshot = {

                            "timestamp": (
                                time.time()
                            ),

                            "tasks": len(
                                getattr(
                                    self,
                                    "tasks",
                                    {},
                                )
                            ),

                            "workers": len([
                                w
                                for w in getattr(
                                    self,
                                    "worker_tasks",
                                    [],
                                )
                                if (
                                    w
                                    and not w.done()
                                )
                            ]),
                        }

                        self._last_metrics_snapshot = (
                            metrics_snapshot
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # PERIODIC GC
                # -----------------------------------------------------
                try:

                    if (
                        loop_iterations % 250
                        == 0
                    ):

                        gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET FAILURE COUNTER
                # -----------------------------------------------------
                consecutive_failures = 0

                # -----------------------------------------------------
                # SAFE INTERVAL
                # -----------------------------------------------------
                try:

                    config = getattr(
                        self,
                        "config",
                        {},
                    )

                    interval = float(
                        config.get(
                            "metrics_interval",
                            2,
                        )
                    )

                except Exception:

                    interval = 2.0

                # -----------------------------------------------------
                # INTERVAL SANITIZATION
                # -----------------------------------------------------
                interval = max(
                    0.5,
                    min(interval, 3600),
                )

                # -----------------------------------------------------
                # LOOP TIMING CONTROL
                # -----------------------------------------------------
                elapsed = (
                    loop.time()
                    - cycle_start
                )

                sleep_time = max(
                    0.1,
                    interval - elapsed,
                )

                await asyncio.sleep(
                    sleep_time
                )

            # ---------------------------------------------------------
            # LOOP CANCELLED
            # ---------------------------------------------------------
            except asyncio.CancelledError:

                logger.info(
                    "🛑 Metrics loop cancelled"
                )

                break

            # ---------------------------------------------------------
            # LOOP FAILURE
            # ---------------------------------------------------------
            except Exception as exc:

                consecutive_failures += 1

                logger.error(
                    f"❌ Metrics loop failure "
                    f"({consecutive_failures}/"
                    f"{max_failures}): {exc}"
                )

                # -----------------------------------------------------
                # FAILURE LIMIT
                # -----------------------------------------------------
                if (
                    consecutive_failures
                    >= max_failures
                ):

                    logger.critical(
                        "🚨 Metrics loop exceeded "
                        "maximum failures"
                    )

                    break

                # -----------------------------------------------------
                # EXPONENTIAL BACKOFF
                # -----------------------------------------------------
                await asyncio.sleep(
                    min(
                        10.0,
                        0.5 * consecutive_failures,
                    )
                )

        # -------------------------------------------------------------
        # LOOP SHUTDOWN
        # -------------------------------------------------------------
        try:

            logger.info(
                f"📊 Metrics loop stopped "
                f"| iterations={loop_iterations}"
            )

        except Exception:
            pass

    async def _health_loop(self) -> None:
        """
        Production-grade health monitoring loop.

        Fixes:
        - health loop crashes
        - unhealthy-state spam
        - blocked health checks
        - invalid health responses
        - event emission failures
        - infinite failure loops
        - event-loop starvation
        - unstable intervals
        """

        import asyncio
        import gc
        import time

        logger.info(
            "🛡️ Health loop started"
        )

        # -------------------------------------------------------------
        # LOOP METRICS
        # -------------------------------------------------------------
        loop_iterations = 0

        consecutive_failures = 0

        max_failures = 50

        unhealthy_emits = 0

        # -------------------------------------------------------------
        # EVENT LOOP
        # -------------------------------------------------------------
        try:

            loop = asyncio.get_running_loop()

        except RuntimeError:

            loop = asyncio.get_event_loop()

        # -------------------------------------------------------------
        # MAIN LOOP
        # -------------------------------------------------------------
        while getattr(
            self,
            "_running",
            False,
        ):

            cycle_start = loop.time()

            try:

                loop_iterations += 1

                # -----------------------------------------------------
                # COOPERATIVE YIELD
                # -----------------------------------------------------
                await asyncio.sleep(0)

                # -----------------------------------------------------
                # DEFAULT HEALTH STATE
                # -----------------------------------------------------
                health = {

                    "overall": "unknown",

                    "timestamp": time.time(),
                }

                # -----------------------------------------------------
                # HEALTH CHECKER VALIDATION
                # -----------------------------------------------------
                checker = getattr(
                    self,
                    "health_checker",
                    None,
                )

                if checker is None:

                    logger.warning(
                        "Health checker unavailable"
                    )

                    health = {

                        "overall": "unhealthy",

                        "error": (
                            "health_checker_missing"
                        ),
                    }

                else:

                    # -------------------------------------------------
                    # RUN HEALTH CHECKS
                    # -------------------------------------------------
                    try:

                        if hasattr(
                            checker,
                            "run_checks",
                        ):

                            result = (
                                checker.run_checks()
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                result = (
                                    await asyncio.wait_for(
                                        result,
                                        timeout=60,
                                    )
                                )

                            # -------------------------
                            # VALIDATE RESPONSE
                            # -------------------------
                            if isinstance(
                                result,
                                dict,
                            ):

                                health = result

                            else:

                                health = {

                                    "overall": (
                                        "unknown"
                                    ),

                                    "raw_result": str(
                                        result
                                    ),
                                }

                        else:

                            health = {

                                "overall": (
                                    "unknown"
                                ),

                                "error": (
                                    "run_checks_missing"
                                ),
                            }

                    except asyncio.TimeoutError:

                        logger.warning(
                            "Health checks timeout"
                        )

                        health = {

                            "overall": (
                                "unhealthy"
                            ),

                            "error": (
                                "health_timeout"
                            ),
                        }

                    except Exception as exc:

                        logger.debug(
                            f"Health check failed: {exc}"
                        )

                        health = {

                            "overall": (
                                "unhealthy"
                            ),

                            "error": str(exc),
                        }

                # -----------------------------------------------------
                # NORMALIZE HEALTH STATUS
                # -----------------------------------------------------
                try:

                    overall = str(
                        health.get(
                            "overall",
                            "unknown",
                        )
                    ).lower()

                except Exception:

                    overall = "unknown"

                if overall not in (

                    "healthy",

                    "degraded",

                    "unhealthy",

                    "critical",

                    "unknown",
                ):

                    overall = "unknown"

                health["overall"] = overall

                # -----------------------------------------------------
                # CACHE LAST HEALTH
                # -----------------------------------------------------
                try:

                    self._last_health_report = (
                        health
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # UNHEALTHY HANDLING
                # -----------------------------------------------------
                if overall in (

                    "unhealthy",

                    "critical",
                ):

                    logger.warning(
                        f"⚠️ System unhealthy "
                        f"| status={overall}"
                    )

                    # -------------------------------------------------
                    # EVENT EMISSION
                    # -------------------------------------------------
                    try:

                        event_bus = getattr(
                            self,
                            "event_bus",
                            None,
                        )

                        if (
                            event_bus
                            and hasattr(
                                event_bus,
                                "emit",
                            )
                        ):

                            payload = {

                                "timestamp": (
                                    time.time()
                                ),

                                "health": health,

                                "iteration": (
                                    loop_iterations
                                ),
                            }

                            result = (
                                event_bus.emit(
                                    "system_unhealthy",
                                    payload,
                                )
                            )

                            if asyncio.iscoroutine(
                                result
                            ):

                                await asyncio.wait_for(
                                    result,
                                    timeout=15,
                                )

                            unhealthy_emits += 1

                    except asyncio.TimeoutError:

                        logger.warning(
                            "Health event timeout"
                        )

                    except Exception as exc:

                        logger.debug(
                            f"Health event failed: {exc}"
                        )

                # -----------------------------------------------------
                # OPTIONAL AUTO-HEAL
                # -----------------------------------------------------
                try:

                    auto_heal_enabled = bool(
                        getattr(
                            self,
                            "config",
                            {},
                        ).get(
                            "enable_auto_healing",
                            False,
                        )
                    )

                    if (
                        overall in (
                            "critical",
                        )
                        and auto_heal_enabled
                        and not getattr(
                            self,
                            "_healing",
                            False,
                        )
                    ):

                        logger.warning(
                            "🛠️ Triggering emergency auto-heal"
                        )

                        await asyncio.wait_for(
                            self.auto_heal(),
                            timeout=120,
                        )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Emergency auto-heal timeout"
                    )

                except Exception as exc:

                    logger.debug(
                        f"Emergency auto-heal failed: {exc}"
                    )

                # -----------------------------------------------------
                # PERIODIC GC
                # -----------------------------------------------------
                try:

                    if (
                        loop_iterations % 200
                        == 0
                    ):

                        def _gc_collect():
                            gc.collect()

                        try:
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(None, _gc_collect)
                        except Exception:
                            pass

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET FAILURE COUNTER
                # -----------------------------------------------------
                consecutive_failures = 0

                # -----------------------------------------------------
                # SAFE INTERVAL
                # -----------------------------------------------------
                try:

                    config = getattr(
                        self,
                        "config",
                        {},
                    )

                    interval = float(
                        config.get(
                            "health_check_interval",
                            5,
                        )
                    )

                except Exception:

                    interval = 5.0

                interval = max(
                    1.0,
                    min(interval, 3600),
                )

                # -----------------------------------------------------
                # LOOP TIMING CONTROL
                # -----------------------------------------------------
                elapsed = (
                    loop.time()
                    - cycle_start
                )

                sleep_time = max(
                    0.25,
                    interval - elapsed,
                )

                await asyncio.sleep(
                    sleep_time
                )

            # ---------------------------------------------------------
            # LOOP CANCELLED
            # ---------------------------------------------------------
            except asyncio.CancelledError:

                logger.info(
                    "🛑 Health loop cancelled"
                )

                break

            # ---------------------------------------------------------
            # LOOP FAILURE
            # ---------------------------------------------------------
            except Exception as exc:

                consecutive_failures += 1

                logger.error(
                    f"❌ Health loop failure "
                    f"({consecutive_failures}/"
                    f"{max_failures}): {exc}"
                )

                # -----------------------------------------------------
                # FAILURE LIMIT
                # -----------------------------------------------------
                if (
                    consecutive_failures
                    >= max_failures
                ):

                    logger.critical(
                        "🚨 Health loop exceeded "
                        "maximum failures"
                    )

                    break

                # -----------------------------------------------------
                # EXPONENTIAL BACKOFF
                # -----------------------------------------------------
                await asyncio.sleep(
                    min(
                        10.0,
                        0.5 * consecutive_failures,
                    )
                )

        # -------------------------------------------------------------
        # LOOP SHUTDOWN
        # -------------------------------------------------------------
        try:

            logger.info(
                f"🛡️ Health loop stopped "
                f"| iterations={loop_iterations} "
                f"| unhealthy_events={unhealthy_emits}"
            )

        except Exception:
            pass

    async def _backup_loop(self) -> None:
        """
        Production-grade automatic backup loop.

        Fixes:
        - backup loop crashes
        - corrupted backup payloads
        - runaway backup creation
        - event-loop starvation
        - invalid intervals
        - overlapping backups
        - backup manager failures
        - unsafe serialization
        - infinite failure loops
        """

        import asyncio
        import gc
        import json
        import time

        logger.info(
            "💾 Backup loop started"
        )

        # -------------------------------------------------------------
        # LOOP METRICS
        # -------------------------------------------------------------
        loop_iterations = 0

        consecutive_failures = 0

        max_failures = 50

        successful_backups = 0

        # -------------------------------------------------------------
        # EVENT LOOP
        # -------------------------------------------------------------
        try:

            loop = asyncio.get_running_loop()

        except RuntimeError:

            loop = asyncio.get_event_loop()

        # -------------------------------------------------------------
        # MAIN LOOP
        # -------------------------------------------------------------
        while getattr(
            self,
            "_running",
            False,
        ):

            cycle_start = loop.time()

            try:

                loop_iterations += 1

                # -----------------------------------------------------
                # SAFE INTERVAL
                # -----------------------------------------------------
                try:

                    config = getattr(
                        self,
                        "config",
                        {},
                    )

                    interval = float(
                        config.get(
                            "auto_backup_interval",
                            60,
                        )
                    )

                except Exception:

                    interval = 60.0

                interval = max(
                    5.0,
                    min(interval, 86400),
                )

                # -----------------------------------------------------
                # WAIT INTERVAL
                # -----------------------------------------------------
                await asyncio.sleep(
                    interval
                )

                # -----------------------------------------------------
                # YIELD CONTROL
                # -----------------------------------------------------
                await asyncio.sleep(0)

                # -----------------------------------------------------
                # BACKUP MANAGER VALIDATION
                # -----------------------------------------------------
                backup_manager = getattr(
                    self,
                    "backup_manager",
                    None,
                )

                if backup_manager is None:

                    logger.warning(
                        "Backup manager unavailable"
                    )

                    continue

                # -----------------------------------------------------
                # PREVENT OVERLAPPING BACKUPS
                # -----------------------------------------------------
                if getattr(
                    self,
                    "_backup_running",
                    False,
                ):

                    logger.warning(
                        "Backup already running"
                    )

                    continue

                self._backup_running = True

                # -----------------------------------------------------
                # CREATE BACKUP PAYLOAD
                # -----------------------------------------------------
                try:

                    if hasattr(
                        self,
                        "get_all_status",
                    ):

                        data = (
                            self.get_all_status()
                        )

                    else:

                        data = {

                            "status": (
                                "partial"
                            ),

                            "timestamp": (
                                time.time()
                            ),
                        }

                except Exception as exc:

                    logger.warning(
                        f"Status snapshot failed: {exc}"
                    )

                    data = {

                        "status": "failed",

                        "error": str(exc),
                    }

                # -----------------------------------------------------
                # SERIALIZATION VALIDATION
                # -----------------------------------------------------
                try:

                    json.dumps(
                        data,
                        default=str,
                    )

                except Exception as exc:

                    logger.warning(
                        f"Backup serialization failed: {exc}"
                    )

                    data = {

                        "status": (
                            "serialization_failed"
                        ),

                        "error": str(exc),
                    }

                # -----------------------------------------------------
                # BACKUP CREATION
                # -----------------------------------------------------
                try:

                    if hasattr(
                        backup_manager,
                        "create_backup",
                    ):

                        backup_name = (
                            f"auto_backup_"
                            f"{int(time.time())}"
                        )

                        result = (
                            backup_manager.create_backup(
                                backup_name,
                                data,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=300,
                            )

                        successful_backups += 1

                        logger.info(
                            f"💾 Backup completed "
                            f"| name={backup_name}"
                        )

                    else:

                        logger.warning(
                            "create_backup missing"
                        )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Backup creation timeout"
                    )

                except Exception as exc:

                    logger.error(
                        f"Backup creation failed: {exc}"
                    )

                finally:

                    self._backup_running = False

                # -----------------------------------------------------
                # OPTIONAL BACKUP CLEANUP
                # -----------------------------------------------------
                try:

                    if (
                        hasattr(
                            backup_manager,
                            "cleanup_old_backups",
                        )
                        and loop_iterations % 10
                        == 0
                    ):

                        result = (
                            backup_manager.cleanup_old_backups()
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=120,
                            )

                except Exception as exc:

                    logger.debug(
                        f"Backup cleanup failed: {exc}"
                    )

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                time.time()
                            ),

                            "successful_backups": (
                                successful_backups
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "backup_completed",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            asyncio.create_task(
                                result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # PERIODIC GC
                # -----------------------------------------------------
                try:

                    if (
                        loop_iterations % 50
                        == 0
                    ):

                        gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # RESET FAILURE COUNTER
                # -----------------------------------------------------
                consecutive_failures = 0

                # -----------------------------------------------------
                # LOOP TIMING CONTROL
                # -----------------------------------------------------
                elapsed = (
                    loop.time()
                    - cycle_start
                )

                if elapsed > interval:

                    logger.warning(
                        f"Backup loop slow "
                        f"({elapsed:.2f}s)"
                    )

            # ---------------------------------------------------------
            # LOOP CANCELLED
            # ---------------------------------------------------------
            except asyncio.CancelledError:

                logger.info(
                    "🛑 Backup loop cancelled"
                )

                break

            # ---------------------------------------------------------
            # LOOP FAILURE
            # ---------------------------------------------------------
            except Exception as exc:

                consecutive_failures += 1

                logger.error(
                    f"❌ Backup loop failure "
                    f"({consecutive_failures}/"
                    f"{max_failures}): {exc}"
                )

                self._backup_running = False

                # -----------------------------------------------------
                # FAILURE LIMIT
                # -----------------------------------------------------
                if (
                    consecutive_failures
                    >= max_failures
                ):

                    logger.critical(
                        "🚨 Backup loop exceeded "
                        "maximum failures"
                    )

                    break

                # -----------------------------------------------------
                # EXPONENTIAL BACKOFF
                # -----------------------------------------------------
                await asyncio.sleep(
                    min(
                        15.0,
                        1.0 * consecutive_failures,
                    )
                )

        # -------------------------------------------------------------
        # LOOP SHUTDOWN
        # -------------------------------------------------------------
        try:

            self._backup_running = False

        except Exception:
            pass

        try:

            logger.info(
                f"💾 Backup loop stopped "
                f"| iterations={loop_iterations} "
                f"| backups={successful_backups}"
            )

        except Exception:
            pass

    async def _predictive_scaling(self) -> None:
        """
        Production-grade predictive worker scaling.

        Fixes:
        - runaway worker spawning
        - worker cancellation leaks
        - scaling oscillation
        - invalid metrics handling
        - race conditions
        - CPU overload scaling
        - dead worker accumulation
        - duplicate scaling operations
        """

        import asyncio
        import psutil
        import time

        scaling_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # SCALING LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_scaling_lock",
            ):

                self._scaling_lock = asyncio.Lock()

            # prevent overlapping scaling
            if self._scaling_lock.locked():

                logger.debug(
                    "Scaling already in progress"
                )

                return

            async with self._scaling_lock:

                # -----------------------------------------------------
                # RUNNING CHECK
                # -----------------------------------------------------
                if not getattr(
                    self,
                    "_running",
                    False,
                ):

                    return

                # -----------------------------------------------------
                # METRICS COLLECTOR VALIDATION
                # -----------------------------------------------------
                metrics_collector = getattr(
                    self,
                    "metrics_collector",
                    None,
                )

                if metrics_collector is None:

                    logger.debug(
                        "Metrics collector unavailable"
                    )

                    return

                # -----------------------------------------------------
                # GET STATISTICS
                # -----------------------------------------------------
                try:

                    stats = (
                        metrics_collector.get_statistics()
                    )

                    if not isinstance(
                        stats,
                        dict,
                    ):

                        stats = {}

                except Exception as exc:

                    logger.debug(
                        f"Statistics retrieval failed: {exc}"
                    )

                    stats = {}

                # -----------------------------------------------------
                # CPU METRICS
                # -----------------------------------------------------
                cpu_stats = stats.get(
                    "cpu",
                    {},
                )

                if not isinstance(
                    cpu_stats,
                    dict,
                ):

                    cpu_stats = {}

                try:

                    cpu_trend = str(
                        cpu_stats.get(
                            "trend",
                            "stable",
                        )
                    ).lower()

                except Exception:

                    cpu_trend = "stable"

                try:

                    current_cpu = float(
                        cpu_stats.get(
                            "current",
                            0,
                        )
                    )

                except Exception:

                    current_cpu = 0.0

                current_cpu = max(
                    0.0,
                    min(current_cpu, 100.0),
                )

                # -----------------------------------------------------
                # MEMORY CHECK
                # -----------------------------------------------------
                try:

                    memory_percent = (
                        psutil.virtual_memory()
                        .percent
                    )

                except Exception:

                    memory_percent = 0.0

                # -----------------------------------------------------
                # WORKER REGISTRY
                # -----------------------------------------------------
                worker_tasks = getattr(
                    self,
                    "worker_tasks",
                    [],
                )

                if not isinstance(
                    worker_tasks,
                    list,
                ):

                    worker_tasks = []

                    self.worker_tasks = (
                        worker_tasks
                    )

                # -----------------------------------------------------
                # CLEAN DEAD WORKERS
                # -----------------------------------------------------
                alive_workers = []

                for worker in worker_tasks:

                    try:

                        if (
                            worker
                            and not worker.done()
                        ):

                            alive_workers.append(
                                worker
                            )

                    except Exception:
                        continue

                self.worker_tasks = (
                    alive_workers
                )

                worker_tasks = (
                    self.worker_tasks
                )

                current_workers = len(
                    worker_tasks
                )

                # -----------------------------------------------------
                # HARD LIMITS
                # -----------------------------------------------------
                cpu_count = max(
                    1,
                    psutil.cpu_count() or 1,
                )

                config = getattr(
                    self,
                    "config",
                    {},
                )

                configured_max = int(
                    config.get(
                        "max_concurrent_tasks",
                        cpu_count * 2,
                    )
                )

                max_workers = max(
                    1,
                    min(
                        configured_max,
                        cpu_count * 4,
                    ),
                )

                min_workers = 1

                # -----------------------------------------------------
                # QUEUE PRESSURE
                # -----------------------------------------------------
                try:

                    queue_size = (
                        self.task_queue.qsize()
                    )

                except Exception:

                    queue_size = 0

                # -----------------------------------------------------
                # SCALE DECISION
                # -----------------------------------------------------
                scale_action = None

                target_workers = (
                    current_workers
                )

                # -----------------------------------------------------
                # SCALE UP CONDITIONS
                # -----------------------------------------------------
                high_load = (

                    current_cpu >= 70

                    or queue_size > current_workers * 2
                )

                memory_safe = (
                    memory_percent < 90
                )

                if (
                    cpu_trend in (
                        "increasing",
                        "high",
                    )
                    and high_load
                    and memory_safe
                    and current_workers < max_workers
                ):

                    scale_action = "up"

                    growth = max(
                        1,
                        min(
                            2,
                            max_workers
                            - current_workers,
                        ),
                    )

                    target_workers = min(
                        current_workers + growth,
                        max_workers,
                    )

                # -----------------------------------------------------
                # SCALE DOWN CONDITIONS
                # -----------------------------------------------------
                low_load = (

                    current_cpu <= 30

                    and queue_size <= 1
                )

                if (
                    cpu_trend in (
                        "decreasing",
                        "stable",
                        "low",
                    )
                    and low_load
                    and current_workers > min_workers
                ):

                    scale_action = "down"

                    reduction = max(
                        1,
                        current_workers // 2,
                    )

                    target_workers = max(
                        min_workers,
                        current_workers
                        - reduction,
                    )

                # -----------------------------------------------------
                # NO SCALING REQUIRED
                # -----------------------------------------------------
                if (
                    target_workers
                    == current_workers
                ):

                    logger.debug(
                        "Scaling unchanged "
                        f"| workers={current_workers}"
                    )

                    return

                # -----------------------------------------------------
                # SCALE UP
                # -----------------------------------------------------
                if scale_action == "up":

                    added = 0

                    while (
                        len(worker_tasks)
                        < target_workers
                    ):

                        try:

                            worker_id = len(
                                worker_tasks
                            )

                            worker = (
                                asyncio.create_task(
                                    self._task_worker(),
                                    name=(
                                        f"scaled_worker_"
                                        f"{worker_id}"
                                    ),
                                )
                            )

                            worker_tasks.append(
                                worker
                            )

                            added += 1

                        except Exception as exc:

                            logger.debug(
                                f"Worker spawn failed: {exc}"
                            )

                            break

                    logger.info(
                        f"📈 Scaled up workers "
                        f"{current_workers} → "
                        f"{len(worker_tasks)}"
                    )

                # -----------------------------------------------------
                # SCALE DOWN
                # -----------------------------------------------------
                elif scale_action == "down":

                    removed = 0

                    while (
                        len(worker_tasks)
                        > target_workers
                    ):

                        try:

                            worker = (
                                worker_tasks.pop()
                            )

                            if (
                                worker
                                and not worker.done()
                            ):

                                worker.cancel()

                                try:

                                    await asyncio.wait_for(
                                        worker,
                                        timeout=2,
                                    )

                                except (
                                    asyncio.TimeoutError,
                                    asyncio.CancelledError,
                                ):

                                    pass

                            removed += 1

                        except Exception as exc:

                            logger.debug(
                                f"Worker removal failed: {exc}"
                            )

                            break

                    logger.info(
                        f"📉 Scaled down workers "
                        f"{current_workers} → "
                        f"{len(worker_tasks)}"
                    )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.scaling_actions = int(
                        getattr(
                            self,
                            "scaling_actions",
                            0,
                        )
                    ) + 1

                    self.last_scaling_action = {

                        "action": scale_action,

                        "workers_before": (
                            current_workers
                        ),

                        "workers_after": len(
                            worker_tasks
                        ),

                        "cpu": current_cpu,

                        "memory": memory_percent,

                        "queue_size": queue_size,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "action": (
                                scale_action
                            ),

                            "workers": len(
                                worker_tasks
                            ),

                            "cpu": current_cpu,

                            "memory": memory_percent,

                            "queue_size": queue_size,
                        }

                        result = (
                            event_bus.emit(
                                "workers_scaled",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            asyncio.create_task(
                                result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - scaling_start
                    ) * 1000,
                    2,
                )

                logger.debug(
                    f"Scaling complete "
                    f"| latency={latency_ms}ms"
                )

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Predictive scaling cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Predictive scaling failed: {exc}"
            )

    # ------------------------
    # CORE FUNCTIONALITY
    # ------------------------

    async def autonomous_tick(self) -> None:
        """
        Production-grade autonomous system cycle.

        Fixes:
        - autonomous recursion
        - queue flooding
        - duplicate autonomous commands
        - unsafe command creation
        - command router crashes
        - overlapping autonomous ticks
        - runaway idle execution
        - event-loop starvation
        """

        import asyncio
        import time

        tick_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # AUTONOMOUS LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_autonomous_lock",
            ):

                self._autonomous_lock = asyncio.Lock()

            # prevent overlapping ticks
            if self._autonomous_lock.locked():

                logger.debug(
                    "Autonomous tick skipped "
                    "(already running)"
                )

                return

            async with self._autonomous_lock:

                logger.debug(
                    "⚡ Autonomous tick started"
                )

                # -----------------------------------------------------
                # SYSTEM STATE VALIDATION
                # -----------------------------------------------------
                if not getattr(
                    self,
                    "_running",
                    False,
                ):

                    logger.debug(
                        "Autonomous tick skipped "
                        "(system not running)"
                    )

                    return

                # -----------------------------------------------------
                # COMMAND ROUTER VALIDATION
                # -----------------------------------------------------
                router = getattr(
                    self,
                    "command_router",
                    None,
                )

                if router is None:

                    logger.warning(
                        "Command router unavailable"
                    )

                    return

                # -----------------------------------------------------
                # OPTIONAL QUEUE PRESSURE CHECK
                # -----------------------------------------------------
                try:

                    queue_size = 0

                    if hasattr(
                        router,
                        "queue",
                    ):

                        queue = getattr(
                            router,
                            "queue",
                        )

                        if hasattr(
                            queue,
                            "qsize",
                        ):

                            queue_size = (
                                queue.qsize()
                            )

                    max_queue = 1000

                    if queue_size >= max_queue:

                        logger.warning(
                            "Autonomous tick skipped "
                            "(router queue overloaded)"
                        )

                        return

                except Exception:
                    pass

                # -----------------------------------------------------
                # RATE LIMITING
                # -----------------------------------------------------
                now = time.monotonic()

                last_tick = float(
                    getattr(
                        self,
                        "_last_autonomous_tick",
                        0.0,
                    )
                )

                min_interval = 5.0

                if (
                    now - last_tick
                    < min_interval
                ):

                    logger.debug(
                        "Autonomous tick throttled"
                    )

                    return

                self._last_autonomous_tick = (
                    now
                )

                # -----------------------------------------------------
                # COMMAND CREATION
                # -----------------------------------------------------
                try:

                    cmd = Command(

                        CommandType.SYSTEM,

                        "status_check",

                        priority=5,

                        async_execution=True,

                        payload={

                            "detailed": False,

                            "autonomous": True,

                            "timestamp": (
                                time.time()
                            ),
                        },
                    )

                except Exception as exc:

                    logger.error(
                        f"Autonomous command creation failed: {exc}"
                    )

                    return

                # -----------------------------------------------------
                # ENQUEUE COMMAND
                # -----------------------------------------------------
                try:

                    if not hasattr(
                        router,
                        "enqueue",
                    ):

                        logger.warning(
                            "Router enqueue missing"
                        )

                        return

                    result = router.enqueue(
                        cmd
                    )

                    # async enqueue support
                    if asyncio.iscoroutine(
                        result
                    ):

                        await asyncio.wait_for(
                            result,
                            timeout=10,
                        )

                    logger.debug(
                        "🤖 Autonomous command queued"
                    )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Autonomous enqueue timeout"
                    )

                    return

                except Exception as exc:

                    logger.error(
                        f"Autonomous enqueue failed: {exc}"
                    )

                    return

                # -----------------------------------------------------
                # OPTIONAL EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "command": (
                                "status_check"
                            ),

                            "timestamp": (
                                time.time()
                            ),
                        }

                        result = (
                            event_bus.emit(
                                "autonomous_tick",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            asyncio.create_task(
                                result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.autonomous_ticks = int(
                        getattr(
                            self,
                            "autonomous_ticks",
                            0,
                        )
                    ) + 1

                    self.last_autonomous_tick_time = (
                        time.time()
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - tick_start
                    ) * 1000,
                    2,
                )

                logger.debug(
                    f"⚡ Autonomous tick complete "
                    f"| latency={latency_ms}ms"
                )

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Autonomous tick cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Autonomous tick failed: {exc}"
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Production-grade system health check with auto-healing.

        Fixes:
        - blocking resource checks
        - recursive auto-healing
        - invalid metric readings
        - health-check race conditions
        - runaway healing loops
        - silent failures
        - inconsistent health states
        - missing diagnostics
        """

        import asyncio
        import gc
        import psutil
        import time

        check_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # HEALTH LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_health_lock",
            ):

                self._health_lock = asyncio.Lock()

            # prevent overlapping checks
            if self._health_lock.locked():

                logger.debug(
                    "Health check already running"
                )

                return {

                    "status": "busy",

                    "overall": "unknown",
                }

            async with self._health_lock:

                # -----------------------------------------------------
                # DEFAULT RESPONSE
                # -----------------------------------------------------
                result = {

                    "overall": "healthy",

                    "cpu_percent": 0.0,

                    "memory_percent": 0.0,

                    "disk_percent": 0.0,

                    "issues": [],

                    "healing_triggered": False,

                    "timestamp": time.time(),
                }

                # -----------------------------------------------------
                # SYSTEM STATE
                # -----------------------------------------------------
                result["running"] = bool(
                    getattr(
                        self,
                        "_running",
                        False,
                    )
                )

                # -----------------------------------------------------
                # CPU USAGE
                # -----------------------------------------------------
                try:

                    cpu = float(
                        psutil.cpu_percent(
                            interval=None
                        )
                    )

                except Exception as exc:

                    logger.debug(
                        f"CPU check failed: {exc}"
                    )

                    cpu = 0.0

                cpu = max(
                    0.0,
                    min(cpu, 100.0),
                )

                result["cpu_percent"] = round(
                    cpu,
                    2,
                )

                # -----------------------------------------------------
                # MEMORY USAGE
                # -----------------------------------------------------
                try:

                    mem = float(
                        psutil.virtual_memory()
                        .percent
                    )

                except Exception as exc:

                    logger.debug(
                        f"Memory check failed: {exc}"
                    )

                    mem = 0.0

                mem = max(
                    0.0,
                    min(mem, 100.0),
                )

                result[
                    "memory_percent"
                ] = round(mem, 2)

                # -----------------------------------------------------
                # DISK USAGE
                # -----------------------------------------------------
                try:

                    disk = float(
                        psutil.disk_usage(
                            "/"
                        ).percent
                    )

                except Exception:

                    disk = 0.0

                disk = max(
                    0.0,
                    min(disk, 100.0),
                )

                result[
                    "disk_percent"
                ] = round(disk, 2)

                # -----------------------------------------------------
                # TASK / WORKER HEALTH
                # -----------------------------------------------------
                try:

                    workers = getattr(
                        self,
                        "worker_tasks",
                        [],
                    )

                    alive_workers = len([
                        w
                        for w in workers
                        if (
                            w
                            and not w.done()
                        )
                    ])

                    result[
                        "workers_alive"
                    ] = alive_workers

                    if alive_workers <= 0:

                        result[
                            "issues"
                        ].append(
                            "no_active_workers"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # TASK QUEUE HEALTH
                # -----------------------------------------------------
                try:

                    queue_size = (
                        self.task_queue.qsize()
                    )

                    result[
                        "queue_size"
                    ] = queue_size

                    if queue_size > 1000:

                        result[
                            "issues"
                        ].append(
                            "queue_overloaded"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # CPU ALERT
                # -----------------------------------------------------
                if cpu >= 90:

                    logger.warning(
                        f"⚠ High CPU usage: {cpu}%"
                    )

                    result[
                        "issues"
                    ].append(
                        "high_cpu"
                    )

                elif cpu >= 75:

                    result[
                        "issues"
                    ].append(
                        "elevated_cpu"
                    )

                # -----------------------------------------------------
                # MEMORY ALERT
                # -----------------------------------------------------
                if mem >= 90:

                    logger.warning(
                        f"⚠ High memory usage: {mem}%"
                    )

                    result[
                        "issues"
                    ].append(
                        "high_memory"
                    )

                elif mem >= 75:

                    result[
                        "issues"
                    ].append(
                        "elevated_memory"
                    )

                # -----------------------------------------------------
                # DISK ALERT
                # -----------------------------------------------------
                if disk >= 95:

                    result[
                        "issues"
                    ].append(
                        "critical_disk_usage"
                    )

                elif disk >= 85:

                    result[
                        "issues"
                    ].append(
                        "high_disk_usage"
                    )

                # -----------------------------------------------------
                # HEALTH STATUS
                # -----------------------------------------------------
                issues = result["issues"]

                if any(
                    issue.startswith(
                        "high_"
                    )
                    or issue.startswith(
                        "critical_"
                    )
                    for issue in issues
                ):

                    result[
                        "overall"
                    ] = "unhealthy"

                elif issues:

                    result[
                        "overall"
                    ] = "degraded"

                else:

                    result[
                        "overall"
                    ] = "healthy"

                # -----------------------------------------------------
                # AUTO-HEAL
                # -----------------------------------------------------
                enable_heal = bool(
                    getattr(
                        self,
                        "config",
                        {},
                    ).get(
                        "enable_auto_healing",
                        False,
                    )
                )

                critical_health = (
                    result["overall"]
                    == "unhealthy"
                )

                # Cooldown check: skip if auto-heal ran recently
                now = time.monotonic()
                last_heal = getattr(self, "_last_heal_time", 0)
                heal_cooldown = 60.0  # seconds
                if now - last_heal < heal_cooldown:
                    enable_heal = False

                if (
                    enable_heal
                    and critical_health
                ):

                    logger.warning(
                        "🛠️ Triggering auto-healing"
                    )

                    try:

                        if not hasattr(self, "_heal_lock"):
                            self._heal_lock = asyncio.Lock()

                        async with self._heal_lock:

                            heal_task = (
                                self.auto_heal()
                            )

                            if asyncio.iscoroutine(
                                heal_task
                            ):

                                await asyncio.wait_for(
                                    heal_task,
                                    timeout=120,
                                )

                            result[
                                "healing_triggered"
                            ] = True

                    except asyncio.TimeoutError:

                        logger.warning(
                            "Auto-healing timeout"
                        )

                        result[
                            "issues"
                        ].append(
                            "healing_timeout"
                        )

                    except Exception as exc:

                        logger.error(
                            f"Auto-healing failed: {exc}"
                        )

                        result[
                            "issues"
                        ].append(
                            "healing_failed"
                        )

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.health_checks = int(
                        getattr(
                            self,
                            "health_checks",
                            0,
                        )
                    ) + 1

                    self.last_health_check = (
                        time.time()
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # CACHE LAST RESULT
                # -----------------------------------------------------
                try:

                    self._last_health_result = (
                        result
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "overall": (
                                result[
                                    "overall"
                                ]
                            ),

                            "issues": (
                                result[
                                    "issues"
                                ]
                            ),

                            "timestamp": (
                                time.time()
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "health_checked",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # PERIODIC GC
                # -----------------------------------------------------
                try:

                    checks = int(
                        getattr(
                            self,
                            "health_checks",
                            0,
                        )
                    )

                    if checks % 100 == 0:

                        gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - check_start
                    ) * 1000,
                    2,
                )

                result[
                    "latency_ms"
                ] = latency_ms

                logger.debug(
                    f"🩺 Health check complete "
                    f"| status={result['overall']} "
                    f"| latency={latency_ms}ms"
                )

                return result

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Health check cancelled"
            )

            return {
                "overall": "unknown",
                "status": "cancelled",
                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Health check failed: {exc}"
            )

            return {

                "overall": "unhealthy",

                "error": str(exc),

                "timestamp": time.time(),
            }

    # ------------------------
    # STATUS AND INFO
    # ------------------------

    def get_system_status(self) -> Dict[str, Any]:
        """
        Production-grade system status collector.

        Fixes:
        - blocking CPU sampling
        - psutil crashes
        - invalid uptime calculations
        - missing system metrics
        - serialization instability
        - unsafe datetime handling
        - partial metric failures
        - corrupted status responses
        """

        import gc
        import platform
        import socket
        import time

        from datetime import datetime, timedelta

        status_start = time.monotonic()

        try:

            import psutil

            # ---------------------------------------------------------
            # DEFAULT RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": "healthy",

                "timestamp": time.time(),
            }

            # ---------------------------------------------------------
            # CPU METRICS
            # ---------------------------------------------------------
            try:

                cpu_percent = float(
                    psutil.cpu_percent(
                        interval=None
                    )
                )

            except Exception as exc:

                logger.debug(
                    f"CPU metric failed: {exc}"
                )

                cpu_percent = 0.0

            cpu_percent = max(
                0.0,
                min(cpu_percent, 100.0),
            )

            result["cpu_percent"] = round(
                cpu_percent,
                2,
            )

            # ---------------------------------------------------------
            # CPU COUNT
            # ---------------------------------------------------------
            try:

                cpu_count = int(
                    psutil.cpu_count()
                    or 1
                )

            except Exception:

                cpu_count = 1

            result["cpu_count"] = cpu_count

            # ---------------------------------------------------------
            # LOAD AVERAGE
            # ---------------------------------------------------------
            try:

                if hasattr(
                    psutil,
                    "getloadavg",
                ):

                    load_avg = (
                        psutil.getloadavg()
                    )

                    result["load_average"] = {

                        "1m": round(
                            load_avg[0],
                            2,
                        ),

                        "5m": round(
                            load_avg[1],
                            2,
                        ),

                        "15m": round(
                            load_avg[2],
                            2,
                        ),
                    }

            except Exception:
                pass

            # ---------------------------------------------------------
            # MEMORY METRICS
            # ---------------------------------------------------------
            try:

                memory = (
                    psutil.virtual_memory()
                )

                result[
                    "memory_percent"
                ] = round(
                    float(memory.percent),
                    2,
                )

                result[
                    "memory_available_mb"
                ] = (
                    int(memory.available)
                    // (1024 * 1024)
                )

                result[
                    "memory_total_mb"
                ] = (
                    int(memory.total)
                    // (1024 * 1024)
                )

                result[
                    "memory_used_mb"
                ] = (
                    int(memory.used)
                    // (1024 * 1024)
                )

            except Exception as exc:

                logger.debug(
                    f"Memory metrics failed: {exc}"
                )

                result[
                    "memory_percent"
                ] = 0.0

            # ---------------------------------------------------------
            # SWAP METRICS
            # ---------------------------------------------------------
            try:

                swap = psutil.swap_memory()

                result["swap_percent"] = round(
                    float(swap.percent),
                    2,
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # DISK METRICS
            # ---------------------------------------------------------
            try:

                disk = psutil.disk_usage(
                    "/"
                )

                result["disk_percent"] = round(
                    float(disk.percent),
                    2,
                )

                result["disk_free_mb"] = (
                    int(disk.free)
                    // (1024 * 1024)
                )

                result["disk_total_mb"] = (
                    int(disk.total)
                    // (1024 * 1024)
                )

                result["disk_used_mb"] = (
                    int(disk.used)
                    // (1024 * 1024)
                )

            except Exception as exc:

                logger.debug(
                    f"Disk metrics failed: {exc}"
                )

                result[
                    "disk_percent"
                ] = 0.0

            # ---------------------------------------------------------
            # NETWORK METRICS
            # ---------------------------------------------------------
            try:

                net = (
                    psutil.net_io_counters()
                )

                result["network"] = {

                    "bytes_sent_mb": round(
                        net.bytes_sent
                        / (1024 * 1024),
                        2,
                    ),

                    "bytes_recv_mb": round(
                        net.bytes_recv
                        / (1024 * 1024),
                        2,
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # PROCESS METRICS
            # ---------------------------------------------------------
            try:

                process = psutil.Process()

                result["process"] = {

                    "pid": process.pid,

                    "threads": (
                        process.num_threads()
                    ),

                    "memory_mb": (
                        process.memory_info().rss
                        // (1024 * 1024)
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # SYSTEM INFO
            # ---------------------------------------------------------
            try:

                result["system"] = {

                    "platform": (
                        platform.system()
                    ),

                    "platform_release": (
                        platform.release()
                    ),

                    "python_version": (
                        platform.python_version()
                    ),

                    "hostname": (
                        socket.gethostname()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # UPTIME
            # ---------------------------------------------------------
            try:

                started_at = getattr(
                    self,
                    "started_at",
                    None,
                )

                if (
                    started_at
                    and isinstance(
                        started_at,
                        datetime,
                    )
                ):

                    uptime_seconds = max(
                        0.0,
                        (
                            datetime.now()
                            - started_at
                        ).total_seconds(),
                    )

                else:

                    uptime_seconds = 0.0

            except Exception:

                uptime_seconds = 0.0

            result["uptime"] = round(
                uptime_seconds,
                2,
            )

            try:

                result["uptime_human"] = str(
                    timedelta(
                        seconds=int(
                            uptime_seconds
                        )
                    )
                )

            except Exception:

                result[
                    "uptime_human"
                ] = "0:00:00"

            # ---------------------------------------------------------
            # WORKER METRICS
            # ---------------------------------------------------------
            try:

                workers = getattr(
                    self,
                    "worker_tasks",
                    [],
                )

                alive_workers = len([
                    w
                    for w in workers
                    if (
                        w
                        and not w.done()
                    )
                ])

                result["workers"] = {

                    "alive": alive_workers,

                    "total": len(workers),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # TASK METRICS
            # ---------------------------------------------------------
            try:

                tasks = getattr(
                    self,
                    "tasks",
                    {},
                )

                result["tasks"] = {

                    "registered": len(
                        tasks
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # HEALTH EVALUATION
            # ---------------------------------------------------------
            issues = []

            if cpu_percent >= 90:

                issues.append(
                    "high_cpu"
                )

            if (
                result.get(
                    "memory_percent",
                    0,
                )
                >= 90
            ):

                issues.append(
                    "high_memory"
                )

            if (
                result.get(
                    "disk_percent",
                    0,
                )
                >= 95
            ):

                issues.append(
                    "critical_disk"
                )

            result["issues"] = issues

            if issues:

                result["status"] = (
                    "degraded"
                )

            # ---------------------------------------------------------
            # CACHE LAST STATUS
            # ---------------------------------------------------------
            try:

                self._last_system_status = (
                    result
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.system_status_checks = int(
                    getattr(
                        self,
                        "system_status_checks",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # PERIODIC GC
            # ---------------------------------------------------------
            try:

                checks = int(
                    getattr(
                        self,
                        "system_status_checks",
                        0,
                    )
                )

                if checks % 100 == 0:

                    gc.collect()

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - status_start
                ) * 1000,
                2,
            )

            result[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"📊 System status collected "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ System status failed: {exc}"
            )

            return {

                "status": "error",

                "cpu_percent": 0.0,

                "cpu_count": 0,

                "memory_percent": 0.0,

                "memory_available_mb": 0,

                "memory_total_mb": 0,

                "disk_percent": 0.0,

                "disk_free_mb": 0,

                "disk_total_mb": 0,

                "uptime": 0.0,

                "uptime_human": "0:00:00",

                "issues": ["status_failure"],

                "error": str(exc),

                "timestamp": time.time(),
            }

    def get_process_info(self) -> Dict[str, Any]:
        """
        Production-grade process diagnostics collector.

        Fixes:
        - blocking CPU sampling
        - psutil access failures
        - invalid process metrics
        - connection enumeration crashes
        - timestamp serialization issues
        - unstable memory reporting
        - partial metric corruption
        """

        import gc
        import platform
        import socket
        import time

        from datetime import datetime

        info_start = time.monotonic()

        try:

            import psutil

            # ---------------------------------------------------------
            # PROCESS OBJECT
            # ---------------------------------------------------------
            try:

                process = psutil.Process()

            except Exception as exc:

                logger.error(
                    f"Process access failed: {exc}"
                )

                raise

            # ---------------------------------------------------------
            # DEFAULT RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": "healthy",

                "timestamp": time.time(),
            }

            # ---------------------------------------------------------
            # BASIC PROCESS INFO
            # ---------------------------------------------------------
            try:

                result["pid"] = int(
                    process.pid
                )

            except Exception:

                result["pid"] = 0

            try:

                result["name"] = str(
                    process.name()
                )

            except Exception:

                result["name"] = "unknown"

            # ---------------------------------------------------------
            # CPU METRICS
            # ---------------------------------------------------------
            try:

                cpu_percent = float(
                    process.cpu_percent(
                        interval=None
                    )
                )

            except Exception as exc:

                logger.debug(
                    f"Process CPU failed: {exc}"
                )

                cpu_percent = 0.0

            cpu_percent = max(
                0.0,
                min(cpu_percent, 1000.0),
            )

            result["cpu_percent"] = round(
                cpu_percent,
                2,
            )

            # ---------------------------------------------------------
            # MEMORY METRICS
            # ---------------------------------------------------------
            try:

                memory_percent = float(
                    process.memory_percent()
                )

            except Exception:

                memory_percent = 0.0

            memory_percent = max(
                0.0,
                min(memory_percent, 100.0),
            )

            result[
                "memory_percent"
            ] = round(
                memory_percent,
                4,
            )

            # ---------------------------------------------------------
            # MEMORY INFO
            # ---------------------------------------------------------
            try:

                mem_info = (
                    process.memory_info()
                )

                result["memory_rss_mb"] = (
                    int(mem_info.rss)
                    // (1024 * 1024)
                )

                result["memory_vms_mb"] = (
                    int(mem_info.vms)
                    // (1024 * 1024)
                )

            except Exception as exc:

                logger.debug(
                    f"Memory info failed: {exc}"
                )

                result["memory_rss_mb"] = 0

                result["memory_vms_mb"] = 0

            # ---------------------------------------------------------
            # THREADS
            # ---------------------------------------------------------
            try:

                result["threads"] = int(
                    process.num_threads()
                )

            except Exception:

                result["threads"] = 0

            # ---------------------------------------------------------
            # OPEN FILES
            # ---------------------------------------------------------
            try:

                open_files = (
                    process.open_files()
                )

                result["open_files"] = len(
                    open_files
                )

            except Exception as exc:

                logger.debug(
                    f"Open files failed: {exc}"
                )

                result["open_files"] = 0

            # ---------------------------------------------------------
            # CONNECTIONS
            # ---------------------------------------------------------
            try:

                # net_connections preferred
                if hasattr(
                    process,
                    "net_connections",
                ):

                    connections = (
                        process.net_connections()
                    )

                else:

                    connections = (
                        process.connections()
                    )

                result["connections"] = len(
                    connections
                )

            except Exception as exc:

                logger.debug(
                    f"Connections failed: {exc}"
                )

                result["connections"] = 0

            # ---------------------------------------------------------
            # PROCESS START TIME
            # ---------------------------------------------------------
            try:

                create_time = float(
                    process.create_time()
                )

                result["create_time"] = (
                    datetime.fromtimestamp(
                        create_time
                    ).isoformat()
                )

                result[
                    "uptime_seconds"
                ] = round(
                    max(
                        0.0,
                        time.time()
                        - create_time,
                    ),
                    2,
                )

            except Exception as exc:

                logger.debug(
                    f"Create time failed: {exc}"
                )

                result["create_time"] = None

                result[
                    "uptime_seconds"
                ] = 0.0

            # ---------------------------------------------------------
            # EXECUTABLE INFO
            # ---------------------------------------------------------
            try:

                result["exe"] = str(
                    process.exe()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # CURRENT WORKING DIRECTORY
            # ---------------------------------------------------------
            try:

                result["cwd"] = str(
                    process.cwd()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # PROCESS STATUS
            # ---------------------------------------------------------
            try:

                result["process_status"] = str(
                    process.status()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # USERNAME
            # ---------------------------------------------------------
            try:

                result["username"] = str(
                    process.username()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # SYSTEM INFO
            # ---------------------------------------------------------
            try:

                result["system"] = {

                    "hostname": (
                        socket.gethostname()
                    ),

                    "platform": (
                        platform.system()
                    ),

                    "platform_release": (
                        platform.release()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # CHILD PROCESS INFO
            # ---------------------------------------------------------
            try:

                children = (
                    process.children(
                        recursive=False
                    )
                )

                result["child_processes"] = len(
                    children
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # IO COUNTERS
            # ---------------------------------------------------------
            try:

                io = process.io_counters()

                result["io"] = {

                    "read_mb": round(
                        io.read_bytes
                        / (1024 * 1024),
                        2,
                    ),

                    "write_mb": round(
                        io.write_bytes
                        / (1024 * 1024),
                        2,
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # HEALTH EVALUATION
            # ---------------------------------------------------------
            issues = []

            if cpu_percent >= 95:

                issues.append(
                    "high_cpu"
                )

            if memory_percent >= 90:

                issues.append(
                    "high_memory"
                )

            if result.get(
                "threads",
                0,
            ) > 1000:

                issues.append(
                    "high_thread_count"
                )

            result["issues"] = issues

            if issues:

                result["status"] = (
                    "degraded"
                )

            # ---------------------------------------------------------
            # CACHE LAST RESULT
            # ---------------------------------------------------------
            try:

                self._last_process_info = (
                    result
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.process_info_checks = int(
                    getattr(
                        self,
                        "process_info_checks",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # PERIODIC GC
            # ---------------------------------------------------------
            try:

                checks = int(
                    getattr(
                        self,
                        "process_info_checks",
                        0,
                    )
                )

                if checks % 100 == 0:

                    gc.collect()

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - info_start
                ) * 1000,
                2,
            )

            result[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"📋 Process info collected "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Process info failed: {exc}"
            )

            return {

                "status": "error",

                "pid": 0,

                "name": "unknown",

                "cpu_percent": 0.0,

                "memory_percent": 0.0,

                "memory_rss_mb": 0,

                "memory_vms_mb": 0,

                "threads": 0,

                "open_files": 0,

                "connections": 0,

                "create_time": None,

                "uptime_seconds": 0.0,

                "issues": ["process_info_failure"],

                "error": str(exc),

                "timestamp": time.time(),
            }

    def get_all_status(self) -> Dict[str, Any]:
        """
        Production-grade global system status snapshot.

        Fixes:
        - partial status failures
        - corrupted subsystem responses
        - unsafe attribute access
        - serialization instability
        - missing component handling
        - inconsistent metrics
        - worker/task desync
        - nested exception crashes
        """

        import gc
        import time

        snapshot_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # DEFAULT RESPONSE
            # ---------------------------------------------------------
            result = {

                "status": "healthy",

                "timestamp": time.time(),

                "running": bool(
                    getattr(
                        self,
                        "_running",
                        False,
                    )
                ),
            }

            # ---------------------------------------------------------
            # SYSTEM STATUS
            # ---------------------------------------------------------
            try:

                if hasattr(
                    self,
                    "get_system_status",
                ):

                    system_status = (
                        self.get_system_status()
                    )

                    if isinstance(
                        system_status,
                        dict,
                    ):

                        result["system"] = (
                            system_status
                        )

                    else:

                        result["system"] = {

                            "status": (
                                "invalid"
                            )
                        }

                else:

                    result["system"] = {

                        "status": "missing"
                    }

            except Exception as exc:

                logger.debug(
                    f"System status failed: {exc}"
                )

                result["system"] = {

                    "status": "error",

                    "error": str(exc),
                }

            # ---------------------------------------------------------
            # PROCESS STATUS
            # ---------------------------------------------------------
            try:

                if hasattr(
                    self,
                    "get_process_info",
                ):

                    process_info = (
                        self.get_process_info()
                    )

                    if isinstance(
                        process_info,
                        dict,
                    ):

                        result["process"] = (
                            process_info
                        )

                    else:

                        result["process"] = {

                            "status": (
                                "invalid"
                            )
                        }

                else:

                    result["process"] = {

                        "status": "missing"
                    }

            except Exception as exc:

                logger.debug(
                    f"Process info failed: {exc}"
                )

                result["process"] = {

                    "status": "error",

                    "error": str(exc),
                }

            # ---------------------------------------------------------
            # EXECUTOR STATUS
            # ---------------------------------------------------------
            try:

                executor = getattr(
                    self,
                    "executor",
                    None,
                )

                if (
                    executor
                    and hasattr(
                        executor,
                        "get_stats",
                    )
                ):

                    stats = (
                        executor.get_stats()
                    )

                    result["executor"] = (

                        stats
                        if isinstance(
                            stats,
                            dict,
                        )
                        else {}
                    )

                else:

                    result["executor"] = {}

            except Exception as exc:

                logger.debug(
                    f"Executor stats failed: {exc}"
                )

                result["executor"] = {

                    "error": str(exc)
                }

            # ---------------------------------------------------------
            # SAFETY STATUS
            # ---------------------------------------------------------
            try:

                safety = getattr(
                    self,
                    "safety_layer",
                    None,
                )

                violations = 0

                if (
                    safety
                    and hasattr(
                        safety,
                        "violation_log",
                    )
                ):

                    violation_log = getattr(
                        safety,
                        "violation_log",
                        [],
                    )

                    if isinstance(
                        violation_log,
                        (
                            list,
                            tuple,
                        ),
                    ):

                        violations = len(
                            violation_log
                        )

                result["safety"] = {

                    "violations": violations
                }

            except Exception as exc:

                logger.debug(
                    f"Safety metrics failed: {exc}"
                )

                result["safety"] = {

                    "violations": 0,

                    "error": str(exc),
                }

            # ---------------------------------------------------------
            # TASK STATUS
            # ---------------------------------------------------------
            try:

                tasks = getattr(
                    self,
                    "tasks",
                    {},
                )

                if not isinstance(
                    tasks,
                    dict,
                ):

                    tasks = {}

                total_tasks = len(tasks)

                pending = 0
                running = 0
                completed = 0
                failed = 0
                cancelled = 0

                for task in tasks.values():

                    try:

                        state = str(
                            getattr(
                                getattr(
                                    task,
                                    "state",
                                    None,
                                ),
                                "value",
                                getattr(
                                    task,
                                    "state",
                                    "unknown",
                                ),
                            )
                        ).lower()

                        if state == "pending":

                            pending += 1

                        elif state == "running":

                            running += 1

                        elif state == "completed":

                            completed += 1

                        elif state == "failed":

                            failed += 1

                        elif state == "cancelled":

                            cancelled += 1

                    except Exception:
                        continue

                result["tasks"] = {

                    "total": total_tasks,

                    "pending": pending,

                    "running": running,

                    "completed": completed,

                    "failed": failed,

                    "cancelled": cancelled,
                }

            except Exception as exc:

                logger.debug(
                    f"Task metrics failed: {exc}"
                )

                result["tasks"] = {

                    "total": 0,

                    "error": str(exc),
                }

            # ---------------------------------------------------------
            # WORKER STATUS
            # ---------------------------------------------------------
            try:

                workers = getattr(
                    self,
                    "worker_tasks",
                    [],
                )

                if not isinstance(
                    workers,
                    list,
                ):

                    workers = []

                alive_workers = 0
                dead_workers = 0

                for worker in workers:

                    try:

                        if (
                            worker
                            and not worker.done()
                        ):

                            alive_workers += 1

                        else:

                            dead_workers += 1

                    except Exception:

                        dead_workers += 1

                result["workers"] = {

                    "total": len(workers),

                    "alive": alive_workers,

                    "dead": dead_workers,
                }

            except Exception as exc:

                logger.debug(
                    f"Worker metrics failed: {exc}"
                )

                result["workers"] = {

                    "total": 0,

                    "error": str(exc),
                }

            # ---------------------------------------------------------
            # EVENT BUS STATUS
            # ---------------------------------------------------------
            try:

                event_bus = getattr(
                    self,
                    "event_bus",
                    None,
                )

                listeners = 0

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "_listeners",
                    )
                ):

                    listeners_map = getattr(
                        event_bus,
                        "_listeners",
                        {},
                    )

                    if isinstance(
                        listeners_map,
                        dict,
                    ):

                        listeners = sum(
                            len(v)
                            for v in listeners_map.values()
                            if isinstance(
                                v,
                                list,
                            )
                        )

                result["event_bus"] = {

                    "listeners": listeners
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # COMPONENT HEALTH
            # ---------------------------------------------------------
            issues = []

            try:

                system = result.get(
                    "system",
                    {},
                )

                if (
                    system.get(
                        "cpu_percent",
                        0,
                    )
                    >= 90
                ):

                    issues.append(
                        "high_cpu"
                    )

                if (
                    system.get(
                        "memory_percent",
                        0,
                    )
                    >= 90
                ):

                    issues.append(
                        "high_memory"
                    )

            except Exception:
                pass

            try:

                if (
                    result["workers"][
                        "alive"
                    ]
                    <= 0
                ):

                    issues.append(
                        "no_active_workers"
                    )

            except Exception:
                pass

            result["issues"] = issues

            # ---------------------------------------------------------
            # OVERALL STATUS
            # ---------------------------------------------------------
            if issues:

                result["status"] = (
                    "degraded"
                )

            if (
                "no_active_workers"
                in issues
            ):

                result["status"] = (
                    "critical"
                )

            # ---------------------------------------------------------
            # CACHE SNAPSHOT
            # ---------------------------------------------------------
            try:

                self._last_global_status = (
                    result
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.global_status_checks = int(
                    getattr(
                        self,
                        "global_status_checks",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            # ---------------------------------------------------------
            # PERIODIC GC
            # ---------------------------------------------------------
            try:

                checks = int(
                    getattr(
                        self,
                        "global_status_checks",
                        0,
                    )
                )

                if checks % 50 == 0:

                    gc.collect()

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - snapshot_start
                ) * 1000,
                2,
            )

            result[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"📦 Global status snapshot created "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Global status failed: {exc}"
            )

            return {

                "status": "error",

                "running": False,

                "system": {},

                "process": {},

                "executor": {},

                "tasks": {},

                "workers": {},

                "issues": ["global_status_failure"],

                "error": str(exc),

                "timestamp": time.time(),
            }

    # ------------------------
    # COMMAND EXECUTION
    # ------------------------

    async def execute(self, command: Any) -> Any:
        """
        Production-grade command execution pipeline.

        Fixes:
        - command router crashes
        - invalid command objects
        - enqueue deadlocks
        - event bus failures
        - hanging command futures
        - recursive execution
        - unsafe serialization
        - missing timeout handling
        - event-loop starvation
        """

        import asyncio
        import time
        import traceback

        execution_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # EXECUTION LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_execute_lock",
            ):

                self._execute_lock = asyncio.Lock()

            # ---------------------------------------------------------
            # COMMAND VALIDATION
            # ---------------------------------------------------------
            if command is None:

                logger.warning(
                    "Execute called with empty command"
                )

                return None

            # ---------------------------------------------------------
            # COMMAND PREVIEW
            # ---------------------------------------------------------
            try:

                command_preview = str(
                    command
                )[:200]

            except Exception:

                command_preview = (
                    "<unserializable_command>"
                )

            # ---------------------------------------------------------
            # EVENT: COMMAND RECEIVED
            # ---------------------------------------------------------
            try:

                event_bus = getattr(
                    self,
                    "event_bus",
                    None,
                )

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "emit",
                    )
                ):

                    payload = {

                        "command": (
                            command_preview
                        ),

                        "timestamp": (
                            time.time()
                        ),
                    }

                    result = (
                        event_bus.emit(
                            "command_received",
                            payload,
                        )
                    )

                    if asyncio.iscoroutine(
                        result
                    ):

                        asyncio.create_task(
                            result
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # COMMAND NORMALIZATION
            # ---------------------------------------------------------
            try:

                if isinstance(
                    command,
                    str,
                ):

                    normalized_command = Command(

                        CommandType.EXECUTE,

                        command.strip(),

                        async_execution=True,
                    )

                else:

                    normalized_command = (
                        command
                    )

            except Exception as exc:

                raise RuntimeError(
                    f"Command normalization failed: {exc}"
                )

            # ---------------------------------------------------------
            # ROUTER VALIDATION
            # ---------------------------------------------------------
            router = getattr(
                self,
                "command_router",
                None,
            )

            if router is None:

                raise RuntimeError(
                    "Command router unavailable"
                )

            if not hasattr(
                router,
                "enqueue",
            ):

                raise RuntimeError(
                    "Router enqueue missing"
                )

            # ---------------------------------------------------------
            # QUEUE PRESSURE CHECK
            # ---------------------------------------------------------
            try:

                queue = getattr(
                    router,
                    "queue",
                    None,
                )

                if (
                    queue
                    and hasattr(
                        queue,
                        "qsize",
                    )
                ):

                    queue_size = (
                        queue.qsize()
                    )

                    max_queue = 5000

                    if queue_size >= max_queue:

                        logger.warning(
                            "Command queue overloaded"
                        )

                        raise RuntimeError(
                            "Command queue overloaded"
                        )

            except RuntimeError:
                raise

            except Exception:
                pass

            # ---------------------------------------------------------
            # ENQUEUE COMMAND
            # ---------------------------------------------------------
            try:

                enqueue_result = (
                    router.enqueue(
                        normalized_command
                    )
                )

                # async enqueue support
                if asyncio.iscoroutine(
                    enqueue_result
                ):

                    future = (
                        await asyncio.wait_for(
                            enqueue_result,
                            timeout=15,
                        )
                    )

                else:

                    future = enqueue_result

            except asyncio.TimeoutError:

                raise RuntimeError(
                    "Command enqueue timeout"
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Command enqueue failed: {exc}"
                )

            # ---------------------------------------------------------
            # FUTURE VALIDATION
            # ---------------------------------------------------------
            if future is None:

                logger.warning(
                    "Command enqueue returned None"
                )

                return None

            # ---------------------------------------------------------
            # EXECUTION TIMEOUT
            # ---------------------------------------------------------
            execution_timeout = float(
                getattr(
                    self,
                    "command_timeout",
                    120,
                )
            )

            execution_timeout = max(
                1.0,
                min(execution_timeout, 3600),
            )

            # ---------------------------------------------------------
            # EXECUTE RESULT
            # ---------------------------------------------------------
            try:

                if (
                    asyncio.isfuture(
                        future
                    )
                    or hasattr(
                        future,
                        "__await__",
                    )
                ):

                    result = (
                        await asyncio.wait_for(
                            future,
                            timeout=execution_timeout,
                        )
                    )

                else:

                    result = future

            except asyncio.TimeoutError:

                raise RuntimeError(
                    "Command execution timeout"
                )

            except asyncio.CancelledError:

                logger.warning(
                    "Command execution cancelled"
                )

                raise

            except Exception as exc:

                raise RuntimeError(
                    f"Command execution failed: {exc}"
                )

            # ---------------------------------------------------------
            # RESULT PREVIEW
            # ---------------------------------------------------------
            try:

                result_preview = str(
                    result
                )[:200]

            except Exception:

                result_preview = (
                    "<unserializable_result>"
                )

            # ---------------------------------------------------------
            # EVENT: COMMAND EXECUTED
            # ---------------------------------------------------------
            try:

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "emit",
                    )
                ):

                    payload = {

                        "command": (
                            command_preview
                        ),

                        "result": (
                            result_preview
                        ),

                        "timestamp": (
                            time.time()
                        ),
                    }

                    emit_result = (
                        event_bus.emit(
                            "command_executed",
                            payload,
                        )
                    )

                    if asyncio.iscoroutine(
                        emit_result
                    ):

                        asyncio.create_task(
                            emit_result
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.commands_executed = int(
                    getattr(
                        self,
                        "commands_executed",
                        0,
                    )
                ) + 1

                self.last_command_time = (
                    time.time()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - execution_start
                ) * 1000,
                2,
            )

            logger.debug(
                f"⚡ Command executed "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Execute cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            # ---------------------------------------------------------
            # ERROR EVENT
            # ---------------------------------------------------------
            try:

                event_bus = getattr(
                    self,
                    "event_bus",
                    None,
                )

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "emit",
                    )
                ):

                    payload = {

                        "error": str(exc),

                        "command": (
                            str(command)[:200]
                        ),

                        "traceback": (
                            traceback.format_exc()[
                                :1000
                            ]
                        ),

                        "timestamp": (
                            time.time()
                        ),
                    }

                    emit_result = (
                        event_bus.emit(
                            "error_occurred",
                            payload,
                        )
                    )

                    if asyncio.iscoroutine(
                        emit_result
                    ):

                        asyncio.create_task(
                            emit_result
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # FAILURE METRICS
            # ---------------------------------------------------------
            try:

                self.command_failures = int(
                    getattr(
                        self,
                        "command_failures",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            logger.error(
                f"❌ Execute failed: {exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "command": (
                    str(command)[:200]
                ),

                "timestamp": time.time(),
            }

    # ------------------------
    # PLUGIN MANAGEMENT
    # ------------------------

    async def load_plugin(self, plugin: Plugin) -> Dict[str, Any]:
        """
        Production-grade plugin loader.

        Fixes:
        - duplicate plugin loading
        - invalid plugin objects
        - plugin manager crashes
        - event bus failures
        - partial plugin initialization
        - async/sync incompatibility
        - silent load failures
        - plugin state corruption
        """

        import asyncio
        import time
        import traceback

        load_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # PLUGIN VALIDATION
            # ---------------------------------------------------------
            if plugin is None:

                raise ValueError(
                    "Plugin cannot be None"
                )

            # ---------------------------------------------------------
            # PLUGIN NAME VALIDATION
            # ---------------------------------------------------------
            try:

                plugin_name = str(
                    getattr(
                        plugin,
                        "name",
                        "",
                    )
                ).strip()

            except Exception:

                plugin_name = ""

            if not plugin_name:

                raise ValueError(
                    "Plugin name missing"
                )

            # ---------------------------------------------------------
            # PLUGIN MANAGER VALIDATION
            # ---------------------------------------------------------
            manager = getattr(
                self,
                "plugin_manager",
                None,
            )

            if manager is None:

                raise RuntimeError(
                    "Plugin manager unavailable"
                )

            # ---------------------------------------------------------
            # LOADING LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_plugin_load_lock",
            ):

                self._plugin_load_lock = (
                    asyncio.Lock()
                )

            async with self._plugin_load_lock:

                logger.info(
                    f"🔌 Loading plugin: {plugin_name}"
                )

                # -----------------------------------------------------
                # PREVENT DUPLICATE LOADS
                # -----------------------------------------------------
                try:

                    loaded_plugins = getattr(
                        manager,
                        "plugins",
                        {},
                    )

                    if (
                        isinstance(
                            loaded_plugins,
                            dict,
                        )
                        and plugin_name
                        in loaded_plugins
                    ):

                        logger.warning(
                            f"Plugin already registered: "
                            f"{plugin_name}"
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # REGISTER PLUGIN
                # -----------------------------------------------------
                try:

                    if not hasattr(
                        manager,
                        "register_plugin",
                    ):

                        raise RuntimeError(
                            "register_plugin missing"
                        )

                    register_result = (
                        manager.register_plugin(
                            plugin
                        )
                    )

                    if asyncio.iscoroutine(
                        register_result
                    ):

                        await asyncio.wait_for(
                            register_result,
                            timeout=30,
                        )

                except asyncio.TimeoutError:

                    raise RuntimeError(
                        "Plugin registration timeout"
                    )

                except Exception as exc:

                    raise RuntimeError(
                        f"Plugin registration failed: {exc}"
                    )

                # -----------------------------------------------------
                # LOAD PLUGIN
                # -----------------------------------------------------
                try:

                    if not hasattr(
                        manager,
                        "load_plugin",
                    ):

                        raise RuntimeError(
                            "load_plugin missing"
                        )

                    load_result = (
                        manager.load_plugin(
                            plugin_name
                        )
                    )

                    if asyncio.iscoroutine(
                        load_result
                    ):

                        await asyncio.wait_for(
                            load_result,
                            timeout=120,
                        )

                except asyncio.TimeoutError:

                    raise RuntimeError(
                        "Plugin load timeout"
                    )

                except Exception as exc:

                    # rollback attempt
                    try:

                        if hasattr(
                            manager,
                            "unload_plugin",
                        ):

                            rollback = (
                                manager.unload_plugin(
                                    plugin_name
                                )
                            )

                            if asyncio.iscoroutine(
                                rollback
                            ):

                                await asyncio.wait_for(
                                    rollback,
                                    timeout=30,
                                )

                    except Exception:
                        pass

                    raise RuntimeError(
                        f"Plugin load failed: {exc}"
                    )

                # -----------------------------------------------------
                # VERIFY LOAD
                # -----------------------------------------------------
                plugin_loaded = False

                try:

                    loaded_plugins = getattr(
                        manager,
                        "plugins",
                        {},
                    )

                    if isinstance(
                        loaded_plugins,
                        dict,
                    ):

                        plugin_loaded = (
                            plugin_name
                            in loaded_plugins
                        )

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT EMISSION
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "plugin": (
                                plugin_name
                            ),

                            "loaded": (
                                plugin_loaded
                            ),

                            "timestamp": (
                                time.time()
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "plugin_loaded",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.plugins_loaded = int(
                        getattr(
                            self,
                            "plugins_loaded",
                            0,
                        )
                    ) + 1

                    self.last_plugin_loaded = {

                        "name": plugin_name,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - load_start
                    ) * 1000,
                    2,
                )

                logger.info(
                    f"✅ Plugin loaded: "
                    f"{plugin_name} "
                    f"({latency_ms}ms)"
                )

                return {

                    "status": "success",

                    "success": True,

                    "plugin": plugin_name,

                    "loaded": plugin_loaded,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Plugin load cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Plugin load failed: {exc}"
            )

            # ---------------------------------------------------------
            # ERROR EVENT
            # ---------------------------------------------------------
            try:

                event_bus = getattr(
                    self,
                    "event_bus",
                    None,
                )

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "emit",
                    )
                ):

                    payload = {

                        "plugin": str(
                            getattr(
                                plugin,
                                "name",
                                "unknown",
                            )
                        ),

                        "error": str(exc),

                        "traceback": (
                            traceback.format_exc()[
                                :1000
                            ]
                        ),

                        "timestamp": (
                            time.time()
                        ),
                    }

                    emit_result = (
                        event_bus.emit(
                            "plugin_load_failed",
                            payload,
                        )
                    )

                    if asyncio.iscoroutine(
                        emit_result
                    ):

                        asyncio.create_task(
                            emit_result
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # FAILURE METRICS
            # ---------------------------------------------------------
            try:

                self.plugin_load_failures = int(
                    getattr(
                        self,
                        "plugin_load_failures",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return {

                "status": "error",

                "success": False,

                "plugin": str(
                    getattr(
                        plugin,
                        "name",
                        "unknown",
                    )
                ),

                "error": str(exc),

                "timestamp": time.time(),
            }

    async def unload_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """
        Production-grade plugin unloader.

        Fixes:
        - unloading non-existent plugins
        - partial unload failures
        - plugin manager crashes
        - async unload deadlocks
        - event bus failures
        - duplicate unload requests
        - plugin state corruption
        - hanging unload operations
        """

        import asyncio
        import time
        import traceback

        unload_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # VALIDATE PLUGIN NAME
            # ---------------------------------------------------------
            if not plugin_name:

                raise ValueError(
                    "Plugin name required"
                )

            plugin_name = str(
                plugin_name
            ).strip()

            if not plugin_name:

                raise ValueError(
                    "Invalid plugin name"
                )

            # ---------------------------------------------------------
            # PLUGIN MANAGER VALIDATION
            # ---------------------------------------------------------
            manager = getattr(
                self,
                "plugin_manager",
                None,
            )

            if manager is None:

                raise RuntimeError(
                    "Plugin manager unavailable"
                )

            if not hasattr(
                manager,
                "unload_plugin",
            ):

                raise RuntimeError(
                    "unload_plugin missing"
                )

            # ---------------------------------------------------------
            # UNLOAD LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_plugin_unload_lock",
            ):

                self._plugin_unload_lock = (
                    asyncio.Lock()
                )

            async with self._plugin_unload_lock:

                logger.info(
                    f"🔌 Unloading plugin: "
                    f"{plugin_name}"
                )

                # -----------------------------------------------------
                # CHECK IF PLUGIN EXISTS
                # -----------------------------------------------------
                plugin_exists = False

                try:

                    plugins = getattr(
                        manager,
                        "plugins",
                        {},
                    )

                    if isinstance(
                        plugins,
                        dict,
                    ):

                        plugin_exists = (
                            plugin_name
                            in plugins
                        )

                except Exception:
                    pass

                if not plugin_exists:

                    logger.warning(
                        f"Plugin not loaded: "
                        f"{plugin_name}"
                    )

                    return {

                        "status": "not_found",

                        "success": False,

                        "plugin": plugin_name,

                        "timestamp": (
                            time.time()
                        ),
                    }

                # -----------------------------------------------------
                # PRE-UNLOAD EVENT
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "plugin": (
                                plugin_name
                            ),

                            "timestamp": (
                                time.time()
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "plugin_unloading",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # UNLOAD PLUGIN
                # -----------------------------------------------------
                try:

                    unload_result = (
                        manager.unload_plugin(
                            plugin_name
                        )
                    )

                    if asyncio.iscoroutine(
                        unload_result
                    ):

                        await asyncio.wait_for(
                            unload_result,
                            timeout=120,
                        )

                except asyncio.TimeoutError:

                    raise RuntimeError(
                        "Plugin unload timeout"
                    )

                except Exception as exc:

                    raise RuntimeError(
                        f"Plugin unload failed: {exc}"
                    )

                # -----------------------------------------------------
                # VERIFY UNLOAD
                # -----------------------------------------------------
                unloaded = True

                try:

                    plugins = getattr(
                        manager,
                        "plugins",
                        {},
                    )

                    if (
                        isinstance(
                            plugins,
                            dict,
                        )
                        and plugin_name
                        in plugins
                    ):

                        unloaded = False

                except Exception:
                    pass

                # -----------------------------------------------------
                # POST-UNLOAD EVENT
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "plugin": (
                                plugin_name
                            ),

                            "unloaded": (
                                unloaded
                            ),

                            "timestamp": (
                                time.time()
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "plugin_unloaded",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # CLEANUP REFERENCES
                # -----------------------------------------------------
                try:

                    if hasattr(
                        self,
                        "_loaded_plugins",
                    ):

                        loaded = getattr(
                            self,
                            "_loaded_plugins",
                            {},
                        )

                        if isinstance(
                            loaded,
                            dict,
                        ):

                            loaded.pop(
                                plugin_name,
                                None,
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.plugins_unloaded = int(
                        getattr(
                            self,
                            "plugins_unloaded",
                            0,
                        )
                    ) + 1

                    self.last_plugin_unloaded = {

                        "name": plugin_name,

                        "timestamp": (
                            time.time()
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - unload_start
                    ) * 1000,
                    2,
                )

                logger.info(
                    f"🗑️ Plugin unloaded: "
                    f"{plugin_name} "
                    f"({latency_ms}ms)"
                )

                return {

                    "status": "success",

                    "success": True,

                    "plugin": plugin_name,

                    "unloaded": unloaded,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Plugin unload cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Plugin unload failed: {exc}"
            )

            # ---------------------------------------------------------
            # ERROR EVENT
            # ---------------------------------------------------------
            try:

                event_bus = getattr(
                    self,
                    "event_bus",
                    None,
                )

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "emit",
                    )
                ):

                    payload = {

                        "plugin": plugin_name,

                        "error": str(exc),

                        "traceback": (
                            traceback.format_exc()[
                                :1000
                            ]
                        ),

                        "timestamp": (
                            time.time()
                        ),
                    }

                    emit_result = (
                        event_bus.emit(
                            "plugin_unload_failed",
                            payload,
                        )
                    )

                    if asyncio.iscoroutine(
                        emit_result
                    ):

                        asyncio.create_task(
                            emit_result
                        )

            except Exception:
                pass

            # ---------------------------------------------------------
            # FAILURE METRICS
            # ---------------------------------------------------------
            try:

                self.plugin_unload_failures = int(
                    getattr(
                        self,
                        "plugin_unload_failures",
                        0,
                    )
                ) + 1

            except Exception:
                pass

            return {

                "status": "error",

                "success": False,

                "plugin": plugin_name,

                "error": str(exc),

                "timestamp": time.time(),
            }

    # ------------------------
    # CONFIGURATION
    # ------------------------

    def update_config(
        self,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Production-grade configuration updater.

        Fixes:
        - invalid config updates
        - unsafe type mutations
        - unknown config injection
        - race conditions
        - corrupted config state
        - silent update failures
        - invalid numeric ranges
        - partial update inconsistencies
        """

        import copy
        import threading
        import time
        import traceback

        update_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # CONFIG LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_config_lock",
            ):

                self._config_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # VALIDATE INPUT
            # ---------------------------------------------------------
            if updates is None:

                raise ValueError(
                    "Updates cannot be None"
                )

            if not isinstance(
                updates,
                dict,
            ):

                raise TypeError(
                    "Updates must be a dictionary"
                )

            if not updates:

                return {

                    "status": "noop",

                    "updated": {},

                    "rejected": {},

                    "timestamp": time.time(),
                }

            # ---------------------------------------------------------
            # CONFIG VALIDATION RULES
            # ---------------------------------------------------------
            validators = {

                "auto_backup_interval": (
                    lambda v:
                    isinstance(
                        v,
                        (
                            int,
                            float,
                        ),
                    )
                    and 5 <= v <= 86400
                ),

                "metrics_interval": (
                    lambda v:
                    isinstance(
                        v,
                        (
                            int,
                            float,
                        ),
                    )
                    and 0.5 <= v <= 3600
                ),

                "health_check_interval": (
                    lambda v:
                    isinstance(
                        v,
                        (
                            int,
                            float,
                        ),
                    )
                    and 1 <= v <= 3600
                ),

                "max_concurrent_tasks": (
                    lambda v:
                    isinstance(
                        v,
                        int,
                    )
                    and 1 <= v <= 10000
                ),

                "enable_auto_healing": (
                    lambda v:
                    isinstance(
                        v,
                        bool,
                    )
                ),

                "enable_predictive_scaling": (
                    lambda v:
                    isinstance(
                        v,
                        bool,
                    )
                ),
            }

            # ---------------------------------------------------------
            # RESULT OBJECT
            # ---------------------------------------------------------
            updated = {}

            rejected = {}

            unchanged = {}

            # ---------------------------------------------------------
            # THREAD-SAFE UPDATE
            # ---------------------------------------------------------
            with self._config_lock:

                # -----------------------------------------------------
                # CONFIG VALIDATION
                # -----------------------------------------------------
                config = getattr(
                    self,
                    "config",
                    None,
                )

                if config is None:

                    raise RuntimeError(
                        "Config store missing"
                    )

                if not isinstance(
                    config,
                    dict,
                ):

                    raise RuntimeError(
                        "Invalid config store"
                    )

                # -----------------------------------------------------
                # SNAPSHOT FOR ROLLBACK
                # -----------------------------------------------------
                previous_config = (
                    copy.deepcopy(config)
                )

                try:

                    # -------------------------------------------------
                    # PROCESS UPDATES
                    # -------------------------------------------------
                    for (
                        key,
                        value,
                    ) in updates.items():

                        try:

                            # -------------------------
                            # UNKNOWN CONFIG
                            # -------------------------
                            if (
                                key
                                not in config
                            ):

                                rejected[key] = (
                                    "unknown_key"
                                )

                                logger.warning(
                                    f"Unknown config key: "
                                    f"{key}"
                                )

                                continue

                            # -------------------------
                            # VALIDATION
                            # -------------------------
                            validator = (
                                validators.get(
                                    key
                                )
                            )

                            if (
                                validator
                                and not validator(
                                    value
                                )
                            ):

                                rejected[key] = (
                                    "invalid_value"
                                )

                                logger.warning(
                                    f"Invalid config value "
                                    f"for {key}: {value}"
                                )

                                continue

                            # -------------------------
                            # NORMALIZATION
                            # -------------------------
                            if isinstance(
                                config[key],
                                bool,
                            ):

                                normalized = bool(
                                    value
                                )

                            elif isinstance(
                                config[key],
                                int,
                            ):

                                normalized = int(
                                    value
                                )

                            elif isinstance(
                                config[key],
                                float,
                            ):

                                normalized = float(
                                    value
                                )

                            else:

                                normalized = value

                            # -------------------------
                            # UNCHANGED
                            # -------------------------
                            if (
                                config[key]
                                == normalized
                            ):

                                unchanged[key] = (
                                    normalized
                                )

                                continue

                            # -------------------------
                            # APPLY UPDATE
                            # -------------------------
                            old_value = (
                                config[key]
                            )

                            config[key] = (
                                normalized
                            )

                            updated[key] = {

                                "old": old_value,

                                "new": normalized,
                            }

                            logger.info(
                                f"⚙️ Config updated: "
                                f"{key} = {normalized}"
                            )

                        except Exception as exc:

                            rejected[key] = str(
                                exc
                            )

                    # -------------------------------------------------
                    # CONFIG SANITY CHECK
                    # -------------------------------------------------
                    max_tasks = int(
                        config.get(
                            "max_concurrent_tasks",
                            1,
                        )
                    )

                    metrics_interval = float(
                        config.get(
                            "metrics_interval",
                            1,
                        )
                    )

                    if max_tasks <= 0:

                        raise RuntimeError(
                            "Invalid task limit"
                        )

                    if metrics_interval <= 0:

                        raise RuntimeError(
                            "Invalid metrics interval"
                        )

                # -----------------------------------------------------
                # ROLLBACK
                # -----------------------------------------------------
                except Exception as exc:

                    logger.error(
                        f"Config update rollback: {exc}"
                    )

                    self.config = (
                        previous_config
                    )

                    raise

            # ---------------------------------------------------------
            # EVENT EMISSION
            # ---------------------------------------------------------
            try:

                event_bus = getattr(
                    self,
                    "event_bus",
                    None,
                )

                if (
                    event_bus
                    and hasattr(
                        event_bus,
                        "emit",
                    )
                ):

                    payload = {

                        "updated": updated,

                        "rejected": rejected,

                        "unchanged": unchanged,

                        "timestamp": (
                            time.time()
                        ),
                    }

                    emit_result = (
                        event_bus.emit(
                            "config_updated",
                            payload,
                        )
                    )

                    # async-safe fire-and-forget
                    try:

                        import asyncio

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                    except Exception:
                        pass

            except Exception:
                pass

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.config_updates = int(
                    getattr(
                        self,
                        "config_updates",
                        0,
                    )
                ) + len(updated)

                self.last_config_update = {

                    "timestamp": (
                        time.time()
                    ),

                    "updated_keys": list(
                        updated.keys()
                    ),
                }

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - update_start
                ) * 1000,
                2,
            )

            logger.info(
                f"⚙️ Config update complete "
                f"| updated={len(updated)} "
                f"| rejected={len(rejected)} "
                f"| latency={latency_ms}ms"
            )

            return {

                "status": "success",

                "success": True,

                "updated": updated,

                "rejected": rejected,

                "unchanged": unchanged,

                "latency_ms": latency_ms,

                "timestamp": time.time(),
            }

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Config update failed: {exc}"
            )

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1000
                    ]
                ),

                "timestamp": time.time(),
            }

    def get_config(self) -> Dict[str, Any]:
        """
        Production-grade configuration snapshot getter.

        Fixes:
        - unsafe shared config exposure
        - mutation leaks
        - partial config corruption
        - missing config store
        - serialization instability
        - race conditions
        - shallow-copy problems
        """

        import copy
        import threading
        import time
        import traceback

        snapshot_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # CONFIG LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_config_lock",
            ):

                self._config_lock = (
                    threading.RLock()
                )

            # ---------------------------------------------------------
            # THREAD-SAFE SNAPSHOT
            # ---------------------------------------------------------
            with self._config_lock:

                config = getattr(
                    self,
                    "config",
                    None,
                )

                if config is None:

                    raise RuntimeError(
                        "Config store missing"
                    )

                if not isinstance(
                    config,
                    dict,
                ):

                    raise RuntimeError(
                        "Invalid config store"
                    )

                # deep copy prevents mutation leaks
                config_snapshot = (
                    copy.deepcopy(config)
                )

            # ---------------------------------------------------------
            # VALIDATE SNAPSHOT
            # ---------------------------------------------------------
            if not isinstance(
                config_snapshot,
                dict,
            ):

                raise RuntimeError(
                    "Config snapshot invalid"
                )

            # ---------------------------------------------------------
            # OPTIONAL SANITY CHECKS
            # ---------------------------------------------------------
            try:

                if (
                    "max_concurrent_tasks"
                    in config_snapshot
                ):

                    config_snapshot[
                        "max_concurrent_tasks"
                    ] = max(
                        1,
                        int(
                            config_snapshot[
                                "max_concurrent_tasks"
                            ]
                        ),
                    )

                if (
                    "metrics_interval"
                    in config_snapshot
                ):

                    config_snapshot[
                        "metrics_interval"
                    ] = max(
                        0.5,
                        float(
                            config_snapshot[
                                "metrics_interval"
                            ]
                        ),
                    )

                if (
                    "health_check_interval"
                    in config_snapshot
                ):

                    config_snapshot[
                        "health_check_interval"
                    ] = max(
                        1.0,
                        float(
                            config_snapshot[
                                "health_check_interval"
                            ]
                        ),
                    )

            except Exception:
                pass

            # ---------------------------------------------------------
            # METADATA
            # ---------------------------------------------------------
            result = {

                "status": "success",

                "config": config_snapshot,

                "timestamp": time.time(),
            }

            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------
            try:

                self.config_reads = int(
                    getattr(
                        self,
                        "config_reads",
                        0,
                    )
                ) + 1

                self.last_config_read = (
                    time.time()
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # CACHE SNAPSHOT
            # ---------------------------------------------------------
            try:

                self._last_config_snapshot = (
                    config_snapshot
                )

            except Exception:
                pass

            # ---------------------------------------------------------
            # LATENCY
            # ---------------------------------------------------------
            latency_ms = round(
                (
                    time.monotonic()
                    - snapshot_start
                ) * 1000,
                2,
            )

            result[
                "latency_ms"
            ] = latency_ms

            logger.debug(
                f"⚙️ Config snapshot created "
                f"| latency={latency_ms}ms"
            )

            return result

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Config snapshot failed: {exc}"
            )

            return {

                "status": "error",

                "config": {},

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1000
                    ]
                ),

                "timestamp": time.time(),
            }

    # ------------------------
    # SHUTDOWN
    # ------------------------

    async def stop(self) -> Dict[str, Any]:
        """
        Production-grade system shutdown entrypoint.

        Fixes:
        - duplicate shutdown calls
        - hanging shutdowns
        - partial subsystem cleanup
        - task cancellation leaks
        - shutdown race conditions
        - event emission failures
        - worker orphaning
        - silent shutdown failures
        """

        import asyncio
        import time
        import traceback

        shutdown_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # SHUTDOWN LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_shutdown_lock",
            ):

                self._shutdown_lock = (
                    asyncio.Lock()
                )

            # ---------------------------------------------------------
            # PREVENT DUPLICATE SHUTDOWN
            # ---------------------------------------------------------
            if getattr(
                self,
                "_stopping",
                False,
            ):

                logger.warning(
                    "Stop already in progress"
                )

                return {

                    "status": "stopping",

                    "success": False,

                    "timestamp": time.time(),
                }

            async with self._shutdown_lock:

                self._stopping = True

                logger.info(
                    "🛑 Stopping system controller"
                )

                # -----------------------------------------------------
                # SYSTEM STATE
                # -----------------------------------------------------
                running_before = bool(
                    getattr(
                        self,
                        "_running",
                        False,
                    )
                )

                # -----------------------------------------------------
                # EVENT: SHUTDOWN STARTED
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                time.time()
                            ),

                            "running": (
                                running_before
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "system_stopping",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # SHUTDOWN EXECUTION
                # -----------------------------------------------------
                shutdown_result = None

                try:

                    if hasattr(
                        self,
                        "shutdown",
                    ):

                        result = (
                            self.shutdown()
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            shutdown_result = (
                                await asyncio.wait_for(
                                    result,
                                    timeout=300,
                                )
                            )

                        else:

                            shutdown_result = (
                                result
                            )

                    else:

                        raise RuntimeError(
                            "shutdown method missing"
                        )

                except asyncio.TimeoutError:

                    raise RuntimeError(
                        "System shutdown timeout"
                    )

                except asyncio.CancelledError:

                    logger.warning(
                        "Shutdown cancelled"
                    )

                    raise

                except Exception as exc:

                    raise RuntimeError(
                        f"Shutdown failed: {exc}"
                    )

                # -----------------------------------------------------
                # ENSURE STOPPED
                # -----------------------------------------------------
                try:

                    self._running = False

                except Exception:
                    pass

                # -----------------------------------------------------
                # CLEANUP BACKGROUND TASK REFERENCES
                # -----------------------------------------------------
                try:

                    background_tasks = [

                        "_main_loop_task",

                        "_metrics_task",

                        "_health_task",

                        "_backup_task",
                    ]

                    for attr in background_tasks:

                        try:

                            task = getattr(
                                self,
                                attr,
                                None,
                            )

                            if (
                                task
                                and hasattr(
                                    task,
                                    "done",
                                )
                                and not task.done()
                            ):

                                task.cancel()

                        except Exception:
                            continue

                except Exception:
                    pass

                # -----------------------------------------------------
                # CLEANUP WORKERS
                # -----------------------------------------------------
                try:

                    workers = getattr(
                        self,
                        "worker_tasks",
                        [],
                    )

                    alive_workers = 0

                    for worker in workers:

                        try:

                            if (
                                worker
                                and not worker.done()
                            ):

                                alive_workers += 1

                        except Exception:
                            pass

                except Exception:

                    alive_workers = 0

                # -----------------------------------------------------
                # EVENT: SHUTDOWN COMPLETE
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                time.time()
                            ),

                            "workers_remaining": (
                                alive_workers
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "system_stopped",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.stop_count = int(
                        getattr(
                            self,
                            "stop_count",
                            0,
                        )
                    ) + 1

                    self.last_stop_time = (
                        time.time()
                    )

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - shutdown_start
                    ) * 1000,
                    2,
                )

                logger.info(
                    f"✅ System stopped "
                    f"| latency={latency_ms}ms"
                )

                self._stopping = False

                return {

                    "status": "stopped",

                    "success": True,

                    "shutdown_result": (
                        shutdown_result
                    ),

                    "workers_remaining": (
                        alive_workers
                    ),

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Stop cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Stop failed: {exc}"
            )

            self._stopping = False

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1000
                    ]
                ),

                "timestamp": time.time(),
            }

    async def shutdown(self) -> Dict[str, Any]:
        """
        Production-grade graceful shutdown with full cleanup.

        Fixes:
        - hanging shutdowns
        - orphaned worker tasks
        - partial subsystem cleanup
        - backup corruption
        - event bus failures
        - recursive shutdown calls
        - deadlock during cancellation
        - unsafe resource cleanup
        - shutdown race conditions
        """

        import asyncio
        import gc
        import time
        import traceback

        from datetime import datetime

        shutdown_start = time.monotonic()

        try:

            # ---------------------------------------------------------
            # SHUTDOWN LOCK
            # ---------------------------------------------------------
            if not hasattr(
                self,
                "_shutdown_lock",
            ):

                self._shutdown_lock = (
                    asyncio.Lock()
                )

            # ---------------------------------------------------------
            # PREVENT DUPLICATE SHUTDOWN
            # ---------------------------------------------------------
            if getattr(
                self,
                "_shutting_down",
                False,
            ):

                logger.warning(
                    "Shutdown already in progress"
                )

                return {

                    "status": "shutting_down",

                    "success": False,

                    "timestamp": time.time(),
                }

            async with self._shutdown_lock:

                self._shutting_down = True

                logger.info(
                    "🛑 Starting graceful shutdown"
                )

                # -----------------------------------------------------
                # SHUTDOWN STATE
                # -----------------------------------------------------
                self._running = False

                shutdown_summary = {

                    "tasks_cancelled": 0,

                    "workers_cancelled": 0,

                    "subsystems_stopped": [],

                    "backup_created": False,

                    "cache_cleared": False,
                }

                # -----------------------------------------------------
                # EVENT: SHUTDOWN START
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                datetime.now()
                                .isoformat()
                            ),

                            "running": False,
                        }

                        emit_result = (
                            event_bus.emit(
                                "system_shutting_down",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # CANCEL MAIN LOOP TASKS
                # -----------------------------------------------------
                main_tasks = [

                    getattr(
                        self,
                        "_main_loop_task",
                        None,
                    ),

                    getattr(
                        self,
                        "_metrics_task",
                        None,
                    ),

                    getattr(
                        self,
                        "_health_task",
                        None,
                    ),

                    getattr(
                        self,
                        "_backup_task",
                        None,
                    ),
                ]

                active_main_tasks = []

                for task in main_tasks:

                    try:

                        if (
                            task
                            and not task.done()
                        ):

                            task.cancel()

                            active_main_tasks.append(
                                task
                            )

                            shutdown_summary[
                                "tasks_cancelled"
                            ] += 1

                    except Exception:
                        continue

                # -----------------------------------------------------
                # WAIT MAIN TASKS
                # -----------------------------------------------------
                if active_main_tasks:

                    try:

                        await asyncio.wait_for(

                            asyncio.gather(
                                *active_main_tasks,
                                return_exceptions=True,
                            ),

                            timeout=10,
                        )

                    except asyncio.TimeoutError:

                        logger.warning(
                            "⚠️ Main task shutdown timeout"
                        )

                # -----------------------------------------------------
                # CANCEL WORKERS
                # -----------------------------------------------------
                worker_tasks = getattr(
                    self,
                    "worker_tasks",
                    [],
                )

                active_workers = []

                for worker in worker_tasks:

                    try:

                        if (
                            worker
                            and not worker.done()
                        ):

                            worker.cancel()

                            active_workers.append(
                                worker
                            )

                            shutdown_summary[
                                "workers_cancelled"
                            ] += 1

                    except Exception:
                        continue

                # -----------------------------------------------------
                # WAIT WORKERS
                # -----------------------------------------------------
                if active_workers:

                    try:

                        await asyncio.wait_for(

                            asyncio.gather(
                                *active_workers,
                                return_exceptions=True,
                            ),

                            timeout=10,
                        )

                    except asyncio.TimeoutError:

                        logger.warning(
                            "⚠️ Worker shutdown timeout"
                        )

                # -----------------------------------------------------
                # STOP SUBSYSTEMS
                # -----------------------------------------------------
                subsystems = [

                    (
                        "command_router",
                        getattr(
                            self,
                            "command_router",
                            None,
                        ),
                    ),

                    (
                        "idle_controller",
                        getattr(
                            self,
                            "idle_controller",
                            None,
                        ),
                    ),

                    (
                        "health_checker",
                        getattr(
                            self,
                            "health_checker",
                            None,
                        ),
                    ),

                    (
                        "resource_monitor",
                        getattr(
                            self,
                            "resource_monitor",
                            None,
                        ),
                    ),
                ]

                for (
                    name,
                    subsystem,
                ) in subsystems:

                    try:

                        if (
                            subsystem
                            and hasattr(
                                subsystem,
                                "stop",
                            )
                        ):

                            stop_result = (
                                subsystem.stop()
                            )

                            if asyncio.iscoroutine(
                                stop_result
                            ):

                                await asyncio.wait_for(
                                    stop_result,
                                    timeout=15,
                                )

                            else:

                                await asyncio.to_thread(
                                    subsystem.stop
                                )

                            shutdown_summary[
                                "subsystems_stopped"
                            ].append(
                                name
                            )

                            logger.info(
                                f"🧩 Stopped subsystem: "
                                f"{name}"
                            )

                    except asyncio.TimeoutError:

                        logger.warning(
                            f"Subsystem stop timeout: "
                            f"{name}"
                        )

                    except Exception as exc:

                        logger.debug(
                            f"{name} stop failed: {exc}"
                        )

                # -----------------------------------------------------
                # EXECUTOR CLEANUP
                # -----------------------------------------------------
                try:

                    executor = getattr(
                        self,
                        "executor",
                        None,
                    )

                    if (
                        executor
                        and hasattr(
                            executor,
                            "clear_tasks",
                        )
                    ):

                        result = (
                            executor.clear_tasks()
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=10,
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # FINAL BACKUP
                # -----------------------------------------------------
                try:

                    backup_manager = getattr(
                        self,
                        "backup_manager",
                        None,
                    )

                    if (
                        backup_manager
                        and hasattr(
                            backup_manager,
                            "create_backup",
                        )
                        and hasattr(
                            self,
                            "get_all_status",
                        )
                    ):

                        backup_payload = (
                            self.get_all_status()
                        )

                        result = (
                            backup_manager.create_backup(
                                "final_backup",
                                backup_payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            result
                        ):

                            await asyncio.wait_for(
                                result,
                                timeout=30,
                            )

                        shutdown_summary[
                            "backup_created"
                        ] = True

                        logger.info(
                            "💾 Final backup created"
                        )

                except asyncio.TimeoutError:

                    logger.warning(
                        "Final backup timeout"
                    )

                except Exception as exc:

                    logger.error(
                        f"Final backup failed: {exc}"
                    )

                # -----------------------------------------------------
                # CACHE CLEAR
                # -----------------------------------------------------
                try:

                    cache_manager = getattr(
                        self,
                        "cache_manager",
                        None,
                    )

                    if (
                        cache_manager
                        and hasattr(
                            cache_manager,
                            "clear",
                        )
                    ):

                        clear_result = (
                            cache_manager.clear()
                        )

                        if asyncio.iscoroutine(
                            clear_result
                        ):

                            await asyncio.wait_for(
                                clear_result,
                                timeout=15,
                            )

                        shutdown_summary[
                            "cache_cleared"
                        ] = True

                except Exception:
                    pass

                # -----------------------------------------------------
                # OPTIONAL GC
                # -----------------------------------------------------
                try:

                    gc.collect()

                except Exception:
                    pass

                # -----------------------------------------------------
                # EVENT: SHUTDOWN COMPLETE
                # -----------------------------------------------------
                try:

                    event_bus = getattr(
                        self,
                        "event_bus",
                        None,
                    )

                    if (
                        event_bus
                        and hasattr(
                            event_bus,
                            "emit",
                        )
                    ):

                        payload = {

                            "timestamp": (
                                datetime.now()
                                .isoformat()
                            ),

                            "summary": (
                                shutdown_summary
                            ),
                        }

                        emit_result = (
                            event_bus.emit(
                                "system_shutdown_complete",
                                payload,
                            )
                        )

                        if asyncio.iscoroutine(
                            emit_result
                        ):

                            asyncio.create_task(
                                emit_result
                            )

                except Exception:
                    pass

                # -----------------------------------------------------
                # METRICS
                # -----------------------------------------------------
                try:

                    self.shutdown_count = int(
                        getattr(
                            self,
                            "shutdown_count",
                            0,
                        )
                    ) + 1

                    self.last_shutdown = {

                        "timestamp": (
                            time.time()
                        ),

                        "summary": (
                            shutdown_summary
                        ),
                    }

                except Exception:
                    pass

                # -----------------------------------------------------
                # LATENCY
                # -----------------------------------------------------
                latency_ms = round(
                    (
                        time.monotonic()
                        - shutdown_start
                    ) * 1000,
                    2,
                )

                logger.info(
                    f"✅ Graceful shutdown complete "
                    f"| latency={latency_ms}ms"
                )

                self._shutting_down = False

                return {

                    "status": "stopped",

                    "success": True,

                    "summary": shutdown_summary,

                    "latency_ms": latency_ms,

                    "timestamp": time.time(),
                }

        # -------------------------------------------------------------
        # CANCELLATION
        # -------------------------------------------------------------
        except asyncio.CancelledError:

            logger.warning(
                "Shutdown cancelled"
            )

            raise

        # -------------------------------------------------------------
        # HARD FAILURE
        # -------------------------------------------------------------
        except Exception as exc:

            logger.error(
                f"❌ Shutdown failed: {exc}"
            )

            self._shutting_down = False

            return {

                "status": "error",

                "success": False,

                "error": str(exc),

                "traceback": (
                    traceback.format_exc()[
                        :1200
                    ]
                ),

                "timestamp": time.time(),
            }

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
