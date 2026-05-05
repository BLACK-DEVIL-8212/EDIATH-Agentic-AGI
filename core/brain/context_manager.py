"""
Advanced Context Manager - AI Context Intelligence Engine
"""

from typing import Any, Dict, List, Optional, Deque
from collections import deque
from datetime import datetime

from tenacity import asyncio

from ..utils.logger import logger
from ..memory import MemoryManager


class ContextEntry:
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
        self.access_count = 0

    def is_expired(self):
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

    def access(self):
        self.access_count += 1
        return self.value


class ContextManager:
    def __init__(self, max_history: int = 100):
        self.context: Dict[str, ContextEntry] = {}
        self.history: Deque = deque(maxlen=max_history)

        self.memory = MemoryManager()

        self.current_user = None
        self.session_id = None

    # ------------------------
    # BASIC CONTEXT
    # ------------------------
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.context[key] = ContextEntry(key, value, ttl)

    def get(self, key: str, default=None):
        if key in self.context:
            entry = self.context[key]
            if not entry.is_expired():
                return entry.access()
            else:
                del self.context[key]
        return default

    # ------------------------
    # HISTORY
    # ------------------------
    def add_message(self, role: str, content: str):
        self.history.append(
            {"role": role, "content": content, "time": datetime.now().isoformat()}
        )

    def get_recent(self, limit=10):
        return list(self.history)[-limit:]

    # ------------------------
    # SMART CONTEXT 🔥
    # ------------------------
    async def build_context(self, user_input: str) -> List[Dict[str, str]]:
        """
        Build LLM-ready context (safe, robust, non-blocking)
        """

        context = []

        # 🔥 SYSTEM PROMPT
        context.append(
            {
                "role": "system",
                "content": "You are an intelligent AI assistant. Be clear, helpful, and concise.",
            }
        )

        # 🔥 RECENT HISTORY (safe)
        try:
            recent = self.get_recent(10) or []
        except Exception as e:
            logger.exception("Context error (recent): %s", e)
            recent = []

        # 🔥 MEMORY SEARCH (safe + timeout)
        memory = []
        try:
            if self.memory and hasattr(self.memory, "search"):
                memory = await asyncio.wait_for(
                    self.memory.search(user_input), timeout=3
                )
                if not isinstance(memory, list):
                    memory = []
        except Exception as e:
            logger.exception("Context error (memory): %s", e)
            memory = []

        # 🔥 ADD MEMORY (limit + clean)
        for m in memory[:5]:  # limit memory size
            if m:
                context.append(
                    {"role": "system", "content": f"Relevant memory: {str(m)[:200]}"}
                )

        # 🔥 ADD CHAT HISTORY (clean)
        for msg in recent:
            try:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if content:
                    context.append({"role": role, "content": str(content)[:1000]})
            except Exception:
                continue

        # 🔥 FALLBACK (important)
        if len(context) <= 1:
            context.append({"role": "user", "content": user_input})

        return context

    # ------------------------
    # USER / SESSION
    # ------------------------
    def set_user(self, user_id: str):
        self.current_user = user_id

    def set_session(self, session_id: str):
        self.session_id = session_id

    # ------------------------
    # CLEANUP
    # ------------------------
    def cleanup(self):
        expired = [k for k, v in self.context.items() if v.is_expired()]
        for k in expired:
            del self.context[k]

    # ------------------------
    # SUMMARY
    # ------------------------
    def get_summary(self):
        return {
            "context": len(self.context),
            "history": len(self.history),
            "user": self.current_user,
            "session": self.session_id,
        }


__all__ = ["ContextManager", "ContextEntry"]
