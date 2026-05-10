"""EDIATH agent wrappers.

This package provides a set of high-level agent classes (requested by user)
that can be registered into the existing MultiAgentCoordinator at runtime.

Note: These wrappers are intentionally lightweight; they map each agent name
to a capability/action keyword set and execute using a shared, safe
execution strategy compatible with the current codebase.
"""

from .register_with_coordinator import register_default_agents

__all__ = ["register_default_agents"]

