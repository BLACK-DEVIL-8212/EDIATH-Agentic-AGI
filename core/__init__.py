"""Core module initialization."""

# Core package init - NO FORWARD IMPORTS (fixes circulars)
# Submodules export their own APIs

__version__ = "0.1.0"


# Lazy getters (no instantiation here - defer to orchestrator)
def get_system_controller():
    from .system.orchestrator import SystemController

    return SystemController


def get_memory_manager():
    from .memory.memory_manager import MemoryManager

    return MemoryManager


def get_logger():
    from .utils.logger import logger

    return logger


__all__ = [
    "__version__",
    "get_system_controller",
    "get_memory_manager",
    "get_logger",
]
