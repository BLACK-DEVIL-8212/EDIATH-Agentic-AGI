"""
Stub implementations for missing critical agents in orchestrator.py.
Prevents ImportError spam, provides basic functionality.
"""


class StubAgent:
    """Base stub for unavailable agents"""

    def __init__(self, config=None):
        self.config = config or {}

    async def execute(self, task):
        return {
            "success": False,
            "error": f"Stub agent (real {type(self).__name__} unavailable)",
        }

    async def process_request(self, request):
        return {"response": "Stub response", "status": "stub"}


class StubFileAgent(StubAgent):
    pass


class StubVisionAgent(StubAgent):
    async def process_vision(self, data):
        return "VISION: Stub vision processing"


class StubBrowserAgent(StubAgent):
    pass


# Add more stubs as needed
