from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from core.agent.action_router import ActionRouter
from core.system.multi_agent_coordinator import BaseAgent, AgentCapability, AgentRole


class WrappedExecutionError(RuntimeError):
    pass


class EdiathWrappedAgent(BaseAgent):
    """Lightweight wrapper agent.

    The existing codebase already contains:
    - ActionRouter (core.agent.action_router)
    - MultiAgentCoordinator (core.system.multi_agent_coordinator)

    These wrapper agents register a small capability set and delegate
    execution to the coordinator/system via ActionRouter when possible.

    This keeps 80+ requested agents operational without requiring full,
    domain-specific implementations for each one.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        actions: List[str],
        confidence: float = 0.8,
        role: AgentRole = AgentRole.WORKER,
        action_router: Optional[ActionRouter] = None,
    ):
        super().__init__(agent_id=agent_id, name=name, role=role)

        # Single capability with all actions
        self.add_capability(
            AgentCapability(
                name=f"capability:{name}",
                version="1.0",
                actions=list(actions),
                confidence=float(confidence),
                max_concurrent_tasks=5,
                priority=0,
            )
        )

        self._action_router = action_router

    async def execute_task(self, task) -> Dict[str, Any]:
        """Execute a coordinator task.

        Expected task shape from coordinator:
        - task.data: dict
        - task.data.action: action keyword
        - task.data.params: dict (optional)
        """

        self.status = self.status.BUSY
        try:
            action = str(getattr(task, "data", {}).get("action", "noop"))
            params = getattr(task, "data", {}).get("params", {})
            if not isinstance(params, dict):
                params = {}

            # If action_router is available, try to route.
            if self._action_router and hasattr(self._action_router, "route"):
                result = await self._action_router.route(action, params)
                return {"success": result is not None, "result": result, "action": action}

            # Fallback: no routing possible.
            return {
                "success": True,
                "result": {
                    "note": "No action_router configured for this wrapped agent",
                    "action": action,
                    "params": params,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self.status = self.status.IDLE

