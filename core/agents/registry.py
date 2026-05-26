from __future__ import annotations

from typing import Dict, List, Optional, Type

from core.agent.action_router import ActionRouter
from core.system.multi_agent_coordinator import AgentRole

from .agent_base import EdiathWrappedAgent
from .agent_specs import AGENT_SPECS


def build_agent_classes() -> Dict[str, Type[EdiathWrappedAgent]]:
    """Create classes dynamically so each spec has a distinct class name."""

    classes: Dict[str, Type[EdiathWrappedAgent]] = {}

    for class_key, spec in AGENT_SPECS.items():
        # class_key already acts as a unique class name
        agent_class_name = class_key

        def _make(agent_key: str, spec_obj):
            class _A(EdiathWrappedAgent):
                def __init__(
                    self,
                    agent_id: str,
                    action_router: Optional[ActionRouter] = None,
                ):
                    super().__init__(
                        agent_id=agent_id,
                        name=spec_obj.agent_name,
                        actions=spec_obj.actions,
                        confidence=spec_obj.confidence,
                        role=AgentRole.WORKER,
                        action_router=action_router,
                    )

            _A.__name__ = agent_class_name
            return _A

        classes[class_key] = _make(class_key, spec)

    return classes

