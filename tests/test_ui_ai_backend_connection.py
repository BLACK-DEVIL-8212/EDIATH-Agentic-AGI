import asyncio
import threading
import time

import core.ui.ai_backend as ai_backend_module
from core.ui.ai_backend import AIBackend


class _FakeAgent:
    async def run(self, text):
        return f"agent:{text}"


class _FakeSystem:
    def __init__(self):
        self.agent = _FakeAgent()

    async def brain_process(self, kind, payload):
        return {"success": True, "output": f"echo:{payload['text']}"}


def test_ui_backend_message_roundtrip():
    # Kivy UI loop is not running in tests; dispatch callbacks immediately.
    original_schedule_once = ai_backend_module.Clock.schedule_once
    ai_backend_module.Clock.schedule_once = lambda cb, dt=0: cb(0)

    backend = AIBackend()
    backend.start()

    responses = []
    statuses = []
    backend.set_response_callback(responses.append)
    backend.set_status_callback(statuses.append)

    loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()

    try:
        backend.connect_system(_FakeSystem(), loop)
        backend.send_user_message("hello")

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if any(msg.startswith("echo:") for msg in responses):
                break
            time.sleep(0.02)

        assert backend.is_ready()
        assert any(msg.startswith("echo:hello") for msg in responses)
        assert any("thinking" in status.lower() for status in statuses)
        assert any("ready" in status.lower() for status in statuses)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        ai_backend_module.Clock.schedule_once = original_schedule_once
