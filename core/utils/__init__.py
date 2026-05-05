"""
EDIATH Core Utilities Package
Auto-imports for graceful shutdown & task management
"""

# Core shutdown & task management
from .shutdown import (
    ShutdownManager,
    shutdown_manager,
    check_shutdown,
    managed_shutdown,
    immediate_shutdown,
)

from .task_registry import GlobalTaskRegistry, task_registry, tracked_task, tracked

__all__ = [
    # Shutdown
    "ShutdownManager",
    "shutdown_manager",
    "check_shutdown",
    "managed_shutdown",
    "immediate_shutdown",
    # Task registry
    "GlobalTaskRegistry",
    "task_registry",
    "tracked_task",
    "tracked",
]
