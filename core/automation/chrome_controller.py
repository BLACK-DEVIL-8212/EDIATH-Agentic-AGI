"""
Advanced Chrome Controller - Stable + Smart Automation (Playwright)
(FULLY UPGRADED - PRODUCTION READY + EDIATH COMPATIBLE)
"""

import asyncio
from typing import Dict, Any, List
from enum import Enum
from datetime import datetime

from ..utils.logger import logger

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None


class BrowserAction(Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    SUBMIT = "submit"


class ChromeController:
    def __init__(self, headless: bool = False, timeout: float = 30000):
        self.headless = headless
        self.timeout = timeout

        self.browser = None
        self.page = None
        self._playwright = None

        self.is_connected = False
        self.current_url = ""
        self.actions_performed = 0
        self.action_history: List[Dict[str, Any]] = []

        # Safety + system state
        self.last_action = None
        self.fail_count = 0

    # ------------------------
    # SAFE START 🔥
    # ------------------------
    async def start(self):
        if async_playwright is None:
            raise ImportError("Install playwright: pip install playwright")

        if self.is_connected:
            return

        try:
            self._playwright = await async_playwright().start()
            self.browser = await self._playwright.chromium.launch(
                headless=self.headless
            )
            self.page = await self.browser.new_page()

            self.is_connected = True
            logger.info("🌐 Browser started")

        except Exception as e:
            logger.error(f"Browser start failed: {e}")
            await self.stop()

    async def stop(self):
        try:
            if self.page:
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Browser stop warning: {e}")

        self.browser = None
        self.page = None
        self._playwright = None
        self.is_connected = False

        logger.info("❌ Browser stopped")

    # ------------------------
    # ENSURE CONNECTION 🔥
    # ------------------------
    async def _ensure(self):
        if not self.is_connected or self.page is None:
            await self.start()

    # ------------------------
    # RETRY WRAPPER 🔥
    # ------------------------
    async def _retry(self, func, *args, retries=2, timeout=None, **kwargs):
        for attempt in range(retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    if timeout:
                        return await asyncio.wait_for(
                            func(*args, **kwargs), timeout=timeout
                        )
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
                await asyncio.sleep(1)

        self.fail_count += 1
        raise RuntimeError("Action failed after retries")

    # ------------------------
    # BROWSER ACTIONS 🔥
    # ------------------------
    async def navigate(self, url: str):
        await self._ensure()

        if not url:
            return "Invalid URL"

        if not url.startswith("http"):
            url = "https://" + url

        await self._retry(self.page.goto, url, timeout=self.timeout)

        self.current_url = url
        self._log(BrowserAction.NAVIGATE, {"url": url})
        return f"Navigated to {url}"

    async def click(self, selector: str):
        await self._ensure()
        if not selector:
            return "Invalid selector"

        await self._retry(self.page.click, selector)
        self._log(BrowserAction.CLICK, {"selector": selector})
        return f"Clicked {selector}"

    async def type_text(self, selector: str, text: str):
        await self._ensure()
        await self._retry(self.page.fill, selector, text)
        self._log(BrowserAction.TYPE, {"selector": selector, "text": text})
        return f"Typed into {selector}"

    async def wait_for_element(self, selector: str):
        await self._ensure()
        await self._retry(self.page.wait_for_selector, selector, timeout=self.timeout)
        self._log(BrowserAction.WAIT, {"selector": selector})
        return f"Element found: {selector}"

    async def scroll(self, direction="down"):
        await self._ensure()
        await self.page.mouse.wheel(0, 500 if direction == "down" else -500)
        self._log(BrowserAction.SCROLL, {"direction": direction})
        return f"Scrolled {direction}"

    async def submit_form(self, selector: str):
        await self._ensure()
        await self.page.locator(selector).press("Enter")
        self._log(BrowserAction.SUBMIT, {"selector": selector})
        return f"Form submitted: {selector}"

    async def screenshot(self, path="screenshot.png"):
        await self._ensure()
        await self.page.screenshot(path=path)
        self._log(BrowserAction.SCREENSHOT, {"path": path})
        return f"Screenshot saved: {path}"

    async def get_content(self):
        await self._ensure()
        return await self.page.content()

    # ------------------------
    # SMART ACTIONS (AI READY)
    # ------------------------
    async def search_google(self, query: str):
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return await self.navigate(url)

    async def open_youtube(self):
        return await self.navigate("https://youtube.com")

    async def open_website(self, name: str):
        return await self.navigate(f"https://{name}.com")

    # ------------------------
    # 🔥 MISSING METHODS (for AI dispatcher)
    # ------------------------
    def explore(self, params: dict = None) -> dict:
        """AI explore action"""
        return {
            "status": "exploring",
            "message": "Browser is ready for exploration",
            "url": self.current_url,
            "connected": self.is_connected,
        }

    def think(self, params: dict = None) -> dict:
        """AI think action"""
        return {
            "status": "thinking",
            "message": "Processing context...",
            "timestamp": datetime.now().isoformat(),
        }

    def speak(self, params: dict = None) -> dict:
        """AI speak action"""
        text = params.get("text", "") if params else ""
        return {
            "status": "spoken",
            "message": f"Would speak: {text[:50]}" if text else "Nothing to speak",
            "output": text,
        }

    # ------------------------
    # SYSTEM STATUS (for EDIATH)
    # ------------------------
    def system_status(self):
        return {
            "connected": self.is_connected,
            "current_url": self.current_url,
            "actions_performed": self.actions_performed,
            "failures": self.fail_count,
            "status": "running" if self.is_connected else "stopped",
        }

    # ------------------------
    # AI ACTION DISPATCHER (UPGRADED)
    # ------------------------
    async def execute_action(self, action: str, params: dict = None) -> dict:
        """
        Unified action dispatcher for AI-driven automation.
        Supports both browser actions and cognitive actions.
        """
        if params is None:
            params = {}

        action = action.lower().strip()
        logger.debug(f"Executing action: {action}")

        try:
            # Cognitive / system actions
            if action == "explore":
                return self.explore(params)
            if action == "think":
                return self.think(params)
            if action == "speak":
                return self.speak(params)
            if action == "idle":
                return {"status": "idle", "message": "System idle"}
            if action == "system_status":
                return self.system_status()

            # Browser actions
            if action == "navigate":
                url = params.get("url", "")
                result = await self.navigate(url)
                return {"status": "success", "message": result}
            if action == "click":
                selector = params.get("selector", "")
                result = await self.click(selector)
                return {"status": "success", "message": result}
            if action == "type":
                selector = params.get("selector", "")
                text = params.get("text", "")
                result = await self.type_text(selector, text)
                return {"status": "success", "message": result}
            if action == "wait":
                selector = params.get("selector", "")
                result = await self.wait_for_element(selector)
                return {"status": "success", "message": result}
            if action == "scroll":
                direction = params.get("direction", "down")
                result = await self.scroll(direction)
                return {"status": "success", "message": result}
            if action == "screenshot":
                path = params.get("path", "screenshot.png")
                result = await self.screenshot(path)
                return {"status": "success", "message": result}
            if action == "search":
                query = params.get("query", "")
                result = await self.search_google(query)
                return {"status": "success", "message": result}

            # Fallback
            return {
                "status": "ignored",
                "message": f"Unknown action '{action}' safely ignored",
            }

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {"status": "error", "message": str(e)}

    # ------------------------
    # LOGGING
    # ------------------------
    def _log(self, action: BrowserAction, params: Dict[str, Any]):
        self.actions_performed += 1
        self.action_history.append(
            {
                "action": action.value,
                "params": params,
                "time": datetime.now().isoformat(),
            }
        )

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self):
        return {
            "connected": self.is_connected,
            "url": self.current_url,
            "actions": self.actions_performed,
            "history": len(self.action_history),
            "failures": self.fail_count,
        }
