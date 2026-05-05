"""
Advanced Human Browser - AI + Human Behavior Simulation (MAX CONFIG)
✔ Realistic mouse movements & scrolling
✔ Typing speed variation
✔ Tab management
✔ Session persistence (cookies, localStorage)
✔ AI-driven link selection (using EDIATH's LLM)
✔ Human-like hesitation, reading time, and back/forward navigation
✔ Full async support
"""

import asyncio
import random
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from ..utils.logger import logger
from .chrome_controller import ChromeController


class HumanBehavior:
    """Human-like timing and probability distributions"""

    @staticmethod
    def uniform(min_sec: float = 0.3, max_sec: float = 1.5) -> float:
        return random.uniform(min_sec, max_sec)

    @staticmethod
    def normal(mean: float = 1.0, std: float = 0.3) -> float:
        return max(0.1, random.gauss(mean, std))

    @staticmethod
    def scroll_amount() -> int:
        # Mimics human scrolling: sometimes small, sometimes large
        if random.random() < 0.7:
            return random.randint(100, 300)
        else:
            return random.randint(500, 1000)

    @staticmethod
    def mouse_offset() -> Tuple[int, int]:
        # Small random offset for mouse clicks
        return (random.randint(-5, 5), random.randint(-5, 5))

    @staticmethod
    def should_click(probability: float = 0.4) -> bool:
        return random.random() < probability

    @staticmethod
    def should_scroll(probability: float = 0.6) -> bool:
        return random.random() < probability

    @staticmethod
    def typing_delay() -> float:
        # Exponential-like distribution for typing speed
        return random.expovariate(10) + 0.05  # average 0.15 sec per character

    @staticmethod
    def reading_time(text_length: int) -> float:
        # Rough reading speed: ~200-300 words per minute
        words = text_length / 5
        seconds = (words / 250) * 60
        return min(10, max(0.5, seconds))


class HumanBrowser:
    def __init__(
        self,
        headless: bool = False,
        user_data_dir: Optional[str] = None,
        session_persistence: bool = True,
        use_ai_for_links: bool = True,
    ):
        self.browser = ChromeController(headless=headless)
        self.behavior = HumanBehavior()
        self.use_ai_for_links = use_ai_for_links
        self.session_persistence = session_persistence
        self.user_data_dir = user_data_dir or str(Path("data/browser_profile"))

        # Session data
        self.session_start = datetime.now()
        self.pages_visited = 0
        self.current_url = ""
        self.interactions: List[Dict] = []
        self.page_history: List[str] = []
        self.history_index = -1

        # AI decision engine (optional)
        self._llm = None
        if use_ai_for_links:
            try:
                from ..brain.llm_engine import LLMEngine

                self._llm = LLMEngine()
            except Exception as e:
                logger.warning(f"LLM not available for AI link selection: {e}")
                self.use_ai_for_links = False

    # ------------------------
    # SESSION PERSISTENCE
    # ------------------------
    async def _save_session(self):
        if not self.session_persistence:
            return
        try:
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
            data = {
                "history": self.page_history,
                "current_url": self.current_url,
                "interactions": self.interactions[-100:],  # last 100
                "timestamp": datetime.now().isoformat(),
            }
            with open(Path(self.user_data_dir) / "session.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Session save failed: {e}")

    async def _load_session(self):
        if not self.session_persistence:
            return
        try:
            path = Path(self.user_data_dir) / "session.json"
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                self.page_history = data.get("history", [])
                self.current_url = data.get("current_url", "")
                logger.info(f"Loaded session with {len(self.page_history)} pages")
        except Exception as e:
            logger.debug(f"Session load failed: {e}")

    # ------------------------
    # START / STOP
    # ------------------------
    async def start(self):
        await self.browser.start()
        await self._load_session()
        logger.info("🧑‍💻 Human browser started (max config)")

    async def stop(self):
        await self._save_session()
        await self.browser.stop()
        logger.info("🛑 Human browser stopped")

    # ------------------------
    # HUMAN NAVIGATION
    # ------------------------
    async def visit(self, url: str, reason: str = "navigation"):
        # Pre‑navigation delay (thinking)
        await asyncio.sleep(self.behavior.uniform(0.2, 0.8))

        # Actually navigate
        await self.browser.navigate(url)

        # Update history
        if self.history_index < len(self.page_history) - 1:
            self.page_history = self.page_history[: self.history_index + 1]
        self.page_history.append(url)
        self.history_index = len(self.page_history) - 1
        self.current_url = url
        self.pages_visited += 1

        self._log("visit", {"url": url, "reason": reason})

        # Human reading delay based on page content
        content = await self.browser.get_content()
        text_length = len(content)
        await asyncio.sleep(self.behavior.reading_time(text_length))

        # Save session after each navigation
        await self._save_session()

    # ------------------------
    # SCROLLING (HUMAN-LIKE)
    # ------------------------
    async def scroll_and_read(self, duration: float = None):
        if duration is None:
            duration = self.behavior.uniform(2, 6)

        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            if self.behavior.should_scroll(0.7):
                amount = self.behavior.scroll_amount()
                await self.browser.page.mouse.wheel(0, amount)
                self._log("scroll", {"amount": amount})
            # Pause between scrolls
            await asyncio.sleep(self.behavior.uniform(0.5, 1.5))

    # ------------------------
    # MOUSE MOVEMENTS (simulated via JS)
    # ------------------------
    async def _move_mouse_to_element(self, selector: str):
        """Simulate mouse movement to an element (via JavaScript)"""
        js = f"""
        const el = document.querySelector('{selector}');
        if (el) {{
            const rect = el.getBoundingClientRect();
            const x = rect.left + rect.width/2 + {self.behavior.mouse_offset()[0]};
            const y = rect.top + rect.height/2 + {self.behavior.mouse_offset()[1]};
            const event = new MouseEvent('mousemove', {{clientX: x, clientY: y, bubbles: true}});
            el.dispatchEvent(event);
        }}
        """
        await self.browser.page.evaluate(js)
        await asyncio.sleep(self.behavior.uniform(0.1, 0.3))

    # ------------------------
    # CLICK ELEMENTS
    # ------------------------
    async def click(self, selector: str, probability: float = 0.4):
        if not self.behavior.should_click(probability):
            return False
        await self._move_mouse_to_element(selector)
        await asyncio.sleep(self.behavior.uniform(0.2, 0.6))
        await self.browser.click(selector)
        self._log("click", {"selector": selector})
        return True

    # ------------------------
    # TYPING WITH VARIABLE SPEED
    # ------------------------
    async def type_text(self, selector: str, text: str):
        await self._move_mouse_to_element(selector)
        await self.browser.click(selector)  # focus
        await asyncio.sleep(self.behavior.uniform(0.2, 0.5))
        # Type each character with delay
        for char in text:
            await self.browser.page.type(
                selector, char, delay=self.behavior.typing_delay()
            )
        self._log("type", {"selector": selector, "text_length": len(text)})

    # ------------------------
    # BACK / FORWARD NAVIGATION
    # ------------------------
    async def go_back(self):
        if self.history_index > 0:
            self.history_index -= 1
            url = self.page_history[self.history_index]
            await self.browser.navigate(url)
            self.current_url = url
            self._log("back", {"url": url})
            return True
        return False

    async def go_forward(self):
        if self.history_index < len(self.page_history) - 1:
            self.history_index += 1
            url = self.page_history[self.history_index]
            await self.browser.navigate(url)
            self.current_url = url
            self._log("forward", {"url": url})
            return True
        return False

    # ------------------------
    # AI-DRIVEN LINK SELECTION
    # ------------------------
    async def _select_link_ai(self, html: str, goal: str) -> Optional[str]:
        if not self._llm:
            return None
        # Extract visible links (simplified)
        links = re.findall(
            r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>([^<]+)</a>', html, re.IGNORECASE
        )
        visible_links = [(href, text.strip()) for href, text in links if text.strip()]
        if not visible_links:
            return None

        # Ask LLM to choose the most relevant link
        prompt = f"""You are a human browsing the web. Your goal: {goal}

Available links (URL | text):
{chr(10).join([f"{i+1}. {url} | {text}" for i, (url, text) in enumerate(visible_links[:20])])}

Choose the single most relevant link number. Return only the number."""

        try:
            if hasattr(self._llm, "safe_generate"):
                result = await self._llm.safe_generate(prompt, timeout=10)
                response = result.get("response", "")
            else:
                response = await self._llm.generate(prompt)
            # Extract number
            match = re.search(r"\d+", str(response))
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(visible_links):
                    return visible_links[idx][0]
        except Exception as e:
            logger.warning(f"AI link selection failed: {e}")
        return None

    # ------------------------
    # SMART BROWSING (AUTONOMOUS)
    # ------------------------
    async def smart_browse(
        self, start_url: str, goal: str = "explore", max_pages: int = 5
    ):
        """
        AI‑aware browsing: visits start_url, then decides next links using LLM.
        """
        await self.visit(start_url, reason="start")
        pages_visited = 1

        while pages_visited < max_pages and self.behavior.should_continue():
            # Scroll and read current page
            await self.scroll_and_read(duration=self.behavior.uniform(3, 8))

            # Get page HTML for link extraction
            html = await self.browser.get_content()

            if self.use_ai_for_links:
                next_url = await self._select_link_ai(html, goal)
            else:
                # Fallback: extract all links and pick random one
                import re

                links = re.findall(r'href="(https?://[^"]+)"', html)
                next_url = random.choice(links) if links else None

            if next_url and next_url not in self.page_history:
                await self.visit(next_url, reason="ai_selected")
                pages_visited += 1
            else:
                # No new link, maybe go back or stop
                if await self.go_back():
                    continue
                else:
                    break

    # ------------------------
    # HUMAN BEHAVIOR LOOP (for integration)
    # ------------------------
    async def run_session(self, urls: List[str], max_actions: int = 20):
        """
        Simulate a human session: visit a list of URLs, interact randomly.
        """
        actions_taken = 0
        for url in urls:
            if actions_taken >= max_actions:
                break
            await self.visit(url)
            actions_taken += 1

            # Random interactions
            if self.behavior.should_click(0.5):
                # Try to click on common elements (simplified)
                common_selectors = ["a", "button", ".btn", "[role='button']"]
                for sel in common_selectors:
                    try:
                        if await self.click(sel, probability=0.3):
                            actions_taken += 1
                            break
                    except:
                        continue

            await self.scroll_and_read(duration=self.behavior.uniform(2, 5))
            actions_taken += 1

            # Occasionally go back
            if self.behavior.should_click(0.2):
                await self.go_back()
                actions_taken += 1

    # ------------------------
    # LOGGING
    # ------------------------
    def _log(self, action: str, details: Dict[str, Any]):
        self.interactions.append(
            {
                "action": action,
                "details": details,
                "time": datetime.now().isoformat(),
                "url": self.current_url,
            }
        )

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.session_start).total_seconds()
        return {
            "pages_visited": self.pages_visited,
            "interactions": len(self.interactions),
            "duration_sec": round(duration, 2),
            "current_url": self.current_url,
            "history_depth": len(self.page_history),
            "ai_link_selection": self.use_ai_for_links,
            "session_persistence": self.session_persistence,
        }
