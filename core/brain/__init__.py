"""
Brain modules - cognitive systems for EDIATH.
(UPDATED - FULLY SAFE, NO REMOVALS)
"""

# ------------------------
# CONTEXT MANAGER
# ------------------------
try:
    from .context_manager import ContextManager, ContextEntry
except Exception:
    ContextManager = None
    ContextEntry = None


# ------------------------
# LLM ENGINE
# ------------------------
try:
    from ..brain.llm_engine import LLMEngine, LLMProvider
except Exception:
    LLMEngine = None
    LLMProvider = None


# ------------------------
# REASONING ENGINE
# ------------------------
try:
    from .reasoning_engine import (
        ReasoningEngine,
        ReasoningChain,
        ReasoningStrategy,
        ReasoningStep,
    )
except Exception:
    ReasoningEngine = None
    ReasoningChain = None
    ReasoningStrategy = None
    ReasoningStep = None


# ------------------------
# DECISION ENGINE (CRITICAL SAFE IMPORT)
# ------------------------
try:
    from .decision_engine import DecisionEngine, Decision, DecisionType, Option
except Exception:
    # 🔥 DO NOT CRASH SYSTEM
    DecisionEngine = None
    Decision = None
    DecisionType = None
    Option = None


# ------------------------
# EXPORTS (UNCHANGED)
# ------------------------
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
