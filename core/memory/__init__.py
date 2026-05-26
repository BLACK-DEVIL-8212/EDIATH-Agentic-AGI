"""Memory systems for EDIATH."""

from importlib import import_module


_LAZY_EXPORTS = {
    "BaseMemory": (".memory_base", "BaseMemory"),
    "MemoryEntry": (".memory_base", "MemoryEntry"),
    "MemoryType": (".memory_base", "MemoryType"),
    "EpisodicMemory": (".episodic_memory", "EpisodicMemory"),
    "SemanticMemory": (".semantic_memory", "SemanticMemory"),
    "VectorMemory": (".vector_memory", "VectorMemory"),
    "MemoryManager": (".memory_manager", "MemoryManager"),
    "memory_manager": (".memory_manager", "memory_manager"),
    "MemoryLoader": (".memory_loader", "MemoryLoader"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BaseMemory",
    "MemoryEntry",
    "MemoryType",
    "EpisodicMemory",
    "SemanticMemory",
    "VectorMemory",
    "MemoryManager",
    "memory_manager",
    "MemoryLoader",
]
