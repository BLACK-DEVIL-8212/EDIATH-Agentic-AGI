"""
Production Router for EDIATH R_AGENT (FINAL STABLE)
✔ Supports intent + decision
✔ Fixes unknown action spam
✔ Smart default routing
✔ Strict validation
✔ Debug logging
✔ Extendable
"""

from typing import Dict, Optional

try:
    from ..utils.logger import logger
except:
    logger = None


class Router:
    def __init__(self):
        try:
            # 🔥 FULLY ALIGNED WITH IntentClassifier
            self.route_map = {
                # ------------------------
                # LLM / CHAT
                # ------------------------
                "respond": "llm",
                "think": "llm",
                "learn": "llm",
                "query": "llm",
                "conversation": "llm",
                "clarification": "llm",
                "feedback": "llm",
                "unknown": "llm",
                "llm": "llm",  # direct LLM routing from ActionRouter
                # ------------------------
                # COMMANDS
                # ------------------------
                "command": "code",
                # ------------------------
                # TOOLS
                # ------------------------
                "explore": "vision",
                "vision": "vision",
                "read_memory": "memory",
                "memory_check": "memory",
                "code": "code",
                # ------------------------
                # SYSTEM
                # ------------------------
                "system_status": "system",
                "system": "system",
            }

            # 🔥 DEFAULT ROUTE
            self.default_route = "llm"

            # 🔥 NORMALIZATION (CRITICAL FIX)
            self.route_map = {self._normalize(k): v for k, v in self.route_map.items()}

            # 🔥 VALID ROUTES SET
            self.valid_routes = {"llm", "vision", "memory", "code", "system"}

            # 🔥 DEBUG FLAG
            self.debug = False

        except Exception as e:
            if "logger" in globals() and logger:
                logger.warning(f"[Router Init Error] {e}")

            # 🔥 SAFE FALLBACK
            self.route_map = {"unknown": "llm"}
            self.default_route = "llm"
            self.valid_routes = {"llm"}
            self.debug = False

    # ------------------------
    # MAIN ROUTE
    # ------------------------
    def route(self, decision: Dict) -> str:
        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            if not isinstance(decision, dict):
                return self._fallback("Invalid decision format")

            # ------------------------
            # 🔥 EXTRACT ACTION
            # ------------------------
            action = self._extract_action(decision)

            if not action:
                return self._fallback("No action found")

            # ------------------------
            # 🔥 NORMALIZE
            # ------------------------
            action = self._normalize(action)

            # ------------------------
            # 🔥 PRIMARY LOOKUP
            # ------------------------
            route = self.route_map.get(action)

            # ------------------------
            # 🔥 SMART FALLBACK
            # ------------------------
            if not route:
                route = self._smart_fallback(action)

                if logger:
                    logger.warning(
                        f"[Router] Unknown action '{action}' → fallback '{route}'"
                    )

            # ------------------------
            # 🔥 VALIDATE ROUTE (CRITICAL)
            # ------------------------
            if route not in getattr(self, "valid_routes", {"llm"}):
                if logger:
                    logger.warning(f"[Router] Invalid route '{route}', using default")
                route = self.default_route

            # ------------------------
            # 🔥 DEBUG LOG
            # ------------------------
            if logger and getattr(self, "debug", False):
                logger.debug(f"[Router] Final: {action} → {route}")

            return route

        except Exception as e:
            return self._fallback(f"Router error: {str(e)}")

    # ------------------------
    # ACTION EXTRACTOR
    # ------------------------
    def _extract_action(self, decision: Dict) -> Optional[str]:
        try:
            if not isinstance(decision, dict):
                return None

            # ------------------------
            # 🔥 PRIMARY FIELDS
            # ------------------------
            action = decision.get("action")
            if action:
                return str(action)

            action_type = decision.get("type")
            if action_type:
                return str(action_type)

            # ------------------------
            # 🔥 NESTED SUPPORT (future-safe)
            # ------------------------
            meta = decision.get("meta")
            if isinstance(meta, dict):
                nested_action = meta.get("action") or meta.get("type")
                if nested_action:
                    return str(nested_action)

            # ------------------------
            # 🔥 FALLBACK KEYS (optional expansion)
            # ------------------------
            for key in ["intent", "category"]:
                value = decision.get(key)
                if value:
                    return str(value)

            return None

        except Exception as e:
            if logger:
                logger.warning(f"[Router ExtractAction Error] {e}")
            return None

    # ------------------------
    # NORMALIZATION
    # ------------------------
    def _normalize(self, action: str) -> str:
        try:
            if not action:
                return ""

            # 🔥 ensure string
            if not isinstance(action, str):
                action = str(action)

            # 🔥 basic normalize
            action = action.strip().lower()

            # 🔥 replace spaces + hyphens with underscore
            import re

            action = re.sub(r"[ \-]+", "_", action)

            # 🔥 remove non-alphanumeric (except underscore)
            action = re.sub(r"[^a-z0-9_]", "", action)

            # 🔥 collapse multiple underscores
            action = re.sub(r"_+", "_", action)

            # 🔥 trim underscores
            action = action.strip("_")

            return action

        except Exception as e:
            if logger:
                logger.warning(f"[Router Normalize Error] {e}")
            return ""

    # ------------------------
    # SMART FALLBACK (FIX)
    # ------------------------
    def _smart_fallback(self, action: str) -> str:
        try:
            if not action:
                return self.default_route

            # 🔥 normalize again (safety)
            action = self._normalize(action)

            # ------------------------
            # 🔥 COMMON INTENT FALLBACKS
            # ------------------------
            llm_actions = {
                "unknown",
                "feedback",
                "clarification",
                "conversation",
                "query",
                "think",
                "respond",
            }

            code_actions = {"command", "execute", "run", "open"}

            vision_actions = {"explore", "scan", "detect", "look"}

            memory_actions = {"remember", "recall", "history", "memory"}

            system_actions = {"status", "health", "system"}

            # ------------------------
            # 🔥 ROUTE SELECTION
            # ------------------------
            if action in llm_actions:
                route = "llm"

            elif action in code_actions:
                route = "code"

            elif action in vision_actions:
                route = "vision"

            elif action in memory_actions:
                route = "memory"

            elif action in system_actions:
                route = "system"

            else:
                route = self.default_route

            # ------------------------
            # 🔥 VALIDATE ROUTE
            # ------------------------
            if route not in getattr(self, "valid_routes", {"llm"}):
                route = self.default_route

            if logger:
                logger.info(f"[Router SmartFallback] {action} → {route}")

            return route

        except Exception as e:
            if logger:
                logger.warning(f"[Router SmartFallback Error] {e}")
            return self.default_route

    # ------------------------
    # HARD FALLBACK
    # ------------------------
    def _fallback(self, reason: str) -> str:
        try:
            if logger:
                logger.warning(f"[Router Fallback] {reason}")

            reason = str(reason).lower() if reason else ""

            # ------------------------
            # 🔥 CONTEXT-AWARE FALLBACK
            # ------------------------
            if "invalid decision" in reason:
                route = "llm"

            elif "no action" in reason:
                route = "llm"

            elif "error" in reason:
                route = "llm"

            else:
                route = self.default_route

            # ------------------------
            # 🔥 VALIDATE ROUTE
            # ------------------------
            if route not in getattr(self, "valid_routes", {"llm"}):
                route = self.default_route

            return route

        except Exception as e:
            if logger:
                logger.warning(f"[Router Fallback Error] {e}")

            return getattr(self, "default_route", "llm")

    # ------------------------
    # DEBUG / INFO
    # ------------------------
    def get_routes(self):
        try:
            routes = getattr(self, "route_map", {}).values()

            # 🔥 unique + sorted for consistency
            unique_routes = sorted(set(routes))

            return unique_routes

        except Exception as e:
            if logger:
                logger.warning(f"[Router GetRoutes Error] {e}")

            # 🔥 safe fallback
            return list(getattr(self, "valid_routes", {"llm"}))

    def get_actions(self):
        try:
            actions = getattr(self, "route_map", {}).keys()

            # 🔥 unique + sorted for consistency
            unique_actions = sorted(set(actions))

            return unique_actions

        except Exception as e:
            if logger:
                logger.warning(f"[Router GetActions Error] {e}")

            # 🔥 safe fallback
            return []

    def add_route(self, action: str, route: str):
        try:
            # ------------------------
            # 🔥 VALIDATE INPUT
            # ------------------------
            if not action or not route:
                return False

            # 🔥 normalize inputs
            action = self._normalize(action)
            route = str(route).strip().lower()

            if not action or not route:
                return False

            # ------------------------
            # 🔥 VALIDATE ROUTE
            # ------------------------
            valid_routes = getattr(self, "valid_routes", {"llm"})

            if route not in valid_routes:
                if logger:
                    logger.warning(f"[Router] Invalid route '{route}' rejected")
                return False

            # ------------------------
            # 🔥 ADD / UPDATE ROUTE
            # ------------------------
            self.route_map[action] = route

            if logger:
                logger.info(f"[Router] Added route: {action} → {route}")

            return True

        except Exception as e:
            if logger:
                logger.warning(f"[Router AddRoute Error] {e}")
            return False
