from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


@dataclass
class UIMessage:
    id: str
    role: str
    content: str
    ts: float


class WebUIServer:
    """Small FastAPI + WebSocket server to stream UI updates.

    This server is UI-only. It does not start/own the AI backend.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        static_dir: str = "core/ui/web",
    ) -> None:
        self.host = host
        self.port = port
        self.static_dir = static_dir

        self.app = FastAPI(title="EDIATH Web UI")

        self._connections: List[WebSocket] = []
        self._connections_lock = threading.RLock()

        self._message_history: List[UIMessage] = []
        self._history_lock = threading.RLock()
        self._max_history = 200

        # Backend callback: called when browser sends a chat prompt.
        # Signature: on_user_prompt(text: str) -> None
        self.on_user_prompt = None  # type: Optional[Any]


        self._setup_routes()

        self._server_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    def _setup_routes(self) -> None:
        # Serve static assets
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            with open(f"{self.static_dir}/index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())

        @self.app.post("/chat")
        async def chat(payload: Dict[str, Any]) -> JSONResponse:
            """Receive user text from browser.

            Expected payload: {"text": "..."}
            """
            text = str(payload.get("text", "")).strip()
            if not text:
                return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)

            if self.on_user_prompt is None:
                return JSONResponse({"ok": False, "error": "backend not connected"}, status_code=500)

            # Call backend in a thread-safe way.
            # The backend connector decides how to run the async pipeline.
            try:
                self.on_user_prompt(text)
            except Exception as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

            return JSONResponse({"ok": True})

        @self.app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse({"ok": True, "ts": time.time()})

        @self.app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            with self._connections_lock:
                self._connections.append(ws)

            try:
                # Send history immediately
                with self._history_lock:
                    history = [m.__dict__ for m in self._message_history]

                await ws.send_text(json.dumps({"type": "history", "messages": history}))

                while True:
                    # Keep-alive / ignore incoming messages (client may send pings)
                    _ = await ws.receive_text()
                    # no-op
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                with self._connections_lock:
                    if ws in self._connections:
                        self._connections.remove(ws)

    def start_background(self) -> None:
        """Start uvicorn in a background thread."""

        if self._server_thread and self._server_thread.is_alive():
            return

        def _run() -> None:
            import uvicorn

            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
            )
            server = uvicorn.Server(config)

            # Stop condition handled via self._shutdown_event by polling.
            async def _serve() -> None:
                while not self._shutdown_event.is_set():
                    await asyncio.sleep(0.2)
                server.should_exit = True

            threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True).start()
            server.run()

        self._server_thread = threading.Thread(target=_run, daemon=True, name="ediath-webui")
        self._server_thread.start()

    def shutdown(self) -> None:
        self._shutdown_event.set()

    def push_message(self, role: str, content: str) -> None:
        msg = UIMessage(id=str(uuid.uuid4()), role=str(role).upper().strip(), content=str(content), ts=time.time())

        with self._history_lock:
            self._message_history.append(msg)
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history :]

        # Broadcast to WS clients
        data = json.dumps({"type": "message", "message": msg.__dict__})

        with self._connections_lock:
            conns = list(self._connections)

        async def _broadcast() -> None:
            for ws in conns:
                try:
                    await ws.send_text(data)
                except Exception:
                    pass

        # Run broadcast safely.
        # If we're already inside the server's event loop thread, schedule directly.
        # Otherwise, spawn a short-lived event loop to avoid scheduling on an
        # arbitrary (caller) loop.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_broadcast())
        except RuntimeError:
            asyncio.run(_broadcast())


