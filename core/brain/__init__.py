"""Brain modules for EDIATH.

The brain package contains heavy optional systems. Keep package import light and
load individual components only when callers request them.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "ContextManager": (".context_manager", "ContextManager"),
    "ContextEntry": (".context_manager", "ContextEntry"),
    "LLMEngine": (".llm_engine", "LLMEngine"),
    "LLMProvider": (".llm_engine", "LLMProvider"),
    "ReasoningEngine": (".reasoning_engine", "ReasoningEngine"),
    "ReasoningChain": (".reasoning_engine", "ReasoningChain"),
    "ReasoningStrategy": (".reasoning_engine", "ReasoningStrategy"),
    "ReasoningStep": (".reasoning_engine", "ReasoningStep"),
    "DecisionEngine": (".decision_engine", "DecisionEngine"),
    "Decision": (".decision_engine", "Decision"),
    "DecisionType": (".decision_engine", "DecisionType"),
    "Option": (".decision_engine", "Option"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module_name, __name__), attr_name)
    except Exception:
        value = None

    globals()[name] = value
    return value


__all__ = [
    "ContextManager",
    "ContextEntry",
    "LLMEngine",
    "LLMProvider",
    "ReasoningEngine",
    "ReasoningChain",
    "ReasoningStrategy",
    "ReasoningStep",
    "DecisionEngine",
    "Decision",
    "DecisionType",
    "Option",
]
