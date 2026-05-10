from __future__ import annotations

import uuid
from typing import Any, Optional

from core.agent.action_router import ActionRouter

from core.system.multi_agent_coordinator import MultiAgentCoordinator

from .registry import build_agent_classes


def register_default_agents(
    coordinator: MultiAgentCoordinator,
    action_router: Optional[ActionRouter] = None,
) -> None:
    """Register the requested agents into an existing MultiAgentCoordinator.

    Safe to call multiple times; duplicates are avoided by agent_id.
    """

    classes = build_agent_classes()

    # Deterministic agent ids so duplicates don't happen across restarts.
    def agent_id_for(key: str) -> str:
        # short, stable id derived from key
        return f"agent_{key.lower()}"

    # Register synchronously (coordinator.register_agent is async)
    import asyncio

    async def _run():
        for key, cls in classes.items():
            aid = agent_id_for(key)
            if getattr(coordinator, "agents", None) and aid in coordinator.agents:
                continue

            agent = cls(agent_id=aid, action_router=action_router)
            await coordinator.register_agent(agent)

    # If already in event loop, schedule; else run.
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(_run())
    except RuntimeError:
        asyncio.run(_run())

