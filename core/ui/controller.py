import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List


class Controller:
    """Async-safe bridge between Kivy UI and the orchestrator.

    All backend calls run in a ThreadPoolExecutor so the UI thread never blocks.
    The controller is defensive: if the orchestrator is not importable or
    methods are missing, it returns structured errors instead of raising.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 4):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, Any], None]] = []
        self._orchestrator = None
        self._available = False

        # try to import orchestrator lazily
        try:
            from core.system.orchestrator import EDIATHOrchestrator

            # try to get a running instance or create one
            try:
                # some codebases expose a singleton getter
                self._orchestrator = EDIATHOrchestrator.get_instance()
            except Exception:
                try:
                    self._orchestrator = EDIATHOrchestrator()
                except Exception:
                    self._orchestrator = None
            self._available = self._orchestrator is not None
        except Exception:
            self._available = False

    def _safe_call(self, fn_name: str, *args, **kwargs):
        if not self._available or self._orchestrator is None:
            return {"error": "orchestrator_unavailable"}
        try:
            fn = getattr(self._orchestrator, fn_name, None)
            if fn is None:
                return {"error": "method_missing", "method": fn_name}
            return fn(*args, **kwargs)
        except Exception as e:
            return {
                "error": "exception",
                "detail": str(e),
                "traceback": traceback.format_exc(),
            }

    # Public API called from UI (these schedule work on executor)
    def initialize(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "initialize")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def run(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "run")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def shutdown(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "shutdown")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def send_command(self, text: str, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "execute", text)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def upload_file(self, path: str, callback: Callable[[Any], None] = None):
        # safe best-effort upload/ingest call
        future = self._executor.submit(self._safe_call, "ingest_file", path)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def get_status(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "status")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def list_agents(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "list_agents")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def get_logs(self, since: float = 0.0, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "get_logs", since)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def get_tasks(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "get_active_tasks")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def send_chat(self, text: str, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "send_chat_message", text)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def get_chat_history(self, callback: Callable[[Any], None] = None):
        future = self._executor.submit(self._safe_call, "get_chat_history")
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def add_listener(self, cb: Callable[[str, Any], None]):
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[str, Any], None]):
        with self._lock:
            try:
                self._listeners.remove(cb)
            except ValueError:
                pass

    def _notify(self, kind: str, payload: Any):
        with self._lock:
            for cb in list(self._listeners):
                try:
                    cb(kind, payload)
                except Exception:
                    pass

    # A convenience polling method used by the UI (runs non-blocking)
    def poll(self, callback: Callable[[Dict[str, Any]], None]):
        """Poll status/logs/tasks and call `callback` with aggregated result."""

        def _work():
            res = {"status": None, "logs": None, "tasks": None, "agents": None}
            try:
                res["status"] = self._safe_call("status")
            except Exception:
                res["status"] = {"error": "status_error"}
            try:
                res["logs"] = self._safe_call("get_logs", 0.0)
            except Exception:
                res["logs"] = {"error": "logs_error"}
            try:
                res["tasks"] = self._safe_call("get_active_tasks")
            except Exception:
                res["tasks"] = {"error": "tasks_error"}
            try:
                res["agents"] = self._safe_call("list_agents")
            except Exception:
                res["agents"] = {"error": "agents_error"}
            return res

        future = self._executor.submit(_work)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future


controller = Controller()
