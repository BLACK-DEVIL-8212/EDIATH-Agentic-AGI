"""
EDIATH Agent Package

✔ Production-safe imports
✔ Modular + fault-tolerant
✔ Supports RAgent + AgentLoop
✔ Centralized exports
✔ Optional debug logging
"""

import os

DEBUG_IMPORTS = os.getenv("EDIATH_DEBUG_IMPORTS", "0") == "1"


def _safe_import(path, name):
    """
    Safely import a symbol from a module.
    Returns None if import fails.
    """
    try:
        module = __import__(path, fromlist=[name])
        return getattr(module, name)
    except Exception as e:
        if DEBUG_IMPORTS:
            print(f"[IMPORT ERROR] {path}.{name} -> {e}")
        return None


# ------------------------
# CORE AGENT
# ------------------------
RAgent = _safe_import(".agent_core", "RAgent")


# ------------------------
# PIPELINE COMPONENTS
# ------------------------
Router = _safe_import(".router", "Router")
Planner = _safe_import(".planner", "Planner")
Executor = _safe_import(".executor", "Executor")
Tools = _safe_import(".tools", "Tools")
Validator = _safe_import(".validator", "Validator")


# ------------------------
# ADVANCED SYSTEMS
# ------------------------
AgentLoop = _safe_import(".agent_loop", "AgentLoop")
ActionRouter = _safe_import(".action_router", "ActionRouter")


# ------------------------
# INTENT SYSTEM
# ------------------------
IntentClassifier = _safe_import(".intent_classifier", "IntentClassifier")
IntentType = _safe_import(".intent_classifier", "IntentType")
Intent = _safe_import(".intent_classifier", "Intent")


# ------------------------
# LEARNING / REFLECTION
# ------------------------
SelfReflection = _safe_import(".self_reflection", "SelfReflection")
ReflectionEntry = _safe_import(".self_reflection", "ReflectionEntry")


# ------------------------
# OPTIONAL AGENTS (AUTO-LOAD)
# ------------------------
# This keeps your system extensible without breaking if files change

OPTIONAL_AGENTS = [
    "browse_agent",
    "search_agent",
    "research_agent",
    "retrieve_agent",
    "knowledge_base_agent",
    "maths_reasoning_agent",
    "codeexecution_agent",
    "file_agent",
    "email_agent",
    "calendar_agent",
    "notification_agent",
    "speech_agent",
    "vision_agent",
    "translation_agent",
    "wikipedia_agent",
    "fitness_agent",
    "network_agent",
    "security_agent",
    "systemcontrol_agent",
    "shell_agent",
    "nlp_agent",
    "databse_agent",
    "blender_agent",
]

_loaded_optional = {}

for agent in OPTIONAL_AGENTS:
    cls_name = "".join([part.capitalize() for part in agent.split("_")])
    imported = _safe_import(f".{agent}", cls_name)
    if imported:
        _loaded_optional[cls_name] = imported


# Inject optional agents into globals
globals().update(_loaded_optional)


# ------------------------
# EXPORTS
# ------------------------
__all__ = [
    # Core
    "RAgent",
    # Pipeline
    "Router",
    "Planner",
    "Executor",
    "Tools",
    "Validator",
    # Autonomous
    "AgentLoop",
    "ActionRouter",
    # Intent
    "IntentClassifier",
    "IntentType",
    "Intent",
    # Learning
    "SelfReflection",
    "ReflectionEntry",
] + list(_loaded_optional.keys())
