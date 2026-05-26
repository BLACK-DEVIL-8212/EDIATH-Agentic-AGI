"""
Advanced Browser Controller for EDIATH
"""

import time

from ..utils.logger import logger
from typing import Optional

try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None


class BrowserController:
    def __init__(self, headless: bool = True, default_timeout: int = 30000):
        """
        Production-grade Browser Controller initialization
        """

        import asyncio

        try:
            # ------------------------
            # 🔥 CORE STATE
            # ------------------------
            self.browser = None
            self.page = None
            self.context = None
            self._playwright = None

            self.is_connected = False
            self.headless = bool(headless)

            # ------------------------
            # 🔥 CONFIG
            # ------------------------
            self.default_timeout = int(default_timeout)  # ms
            self.max_retries = 2

            # ------------------------
            # 🔥 PERFORMANCE / METRICS
            # ------------------------
            self.actions_count = 0
            self.errors = 0
            self.last_action_time = None
            self.connected_at = None

            # ------------------------
            # 🔥 LOCK (IMPORTANT)
            # ------------------------
            self._lock = asyncio.Lock()

            # ------------------------
            # 🔥 SESSION STATE
            # ------------------------
            self.current_url = None
            self.history = []

            # ------------------------
            # 🔥 SAFETY FLAGS
            # ------------------------
            self.allow_navigation = True
            self.allow_external_urls = True

            # ------------------------
            # 🔥 DEBUG / LOGGING
            # ------------------------
            try:
                logger.info("🌐 BrowserController initialized")
            except:
                pass

        except Exception as e:
            try:
                logger.error(f"BrowserController init failed: {e}")
            except:
                pass

            raise RuntimeError(f"BrowserController initialization failed: {e}")

    async def connect(self, headless: Optional[bool] = None) -> bool:
        """
        Start browser using Playwright (production-grade)
        """

        import time
        import asyncio

        start = time.time()

        # ------------------------
        # 🔥 RESOLVE CONFIG
        # ------------------------
        headless = self.headless if headless is None else bool(headless)

        # ------------------------
        # 🔥 DEPENDENCY CHECK
        # ------------------------
        if async_playwright is None:
            raise ImportError("Playwright not installed. Run: pip install playwright")

        # ------------------------
        # 🔥 PREVENT DOUBLE CONNECT
        # ------------------------
        if self.is_connected and self.browser:
            try:
                logger.warning("Browser already connected")
            except:
                pass
            return True

        # ------------------------
        # 🔥 LOCK (THREAD SAFE)
        # ------------------------
        async with self._lock:

            for attempt in range(self.max_retries + 1):
                try:
                    # ------------------------
                    # 🔥 START PLAYWRIGHT
                    # ------------------------
                    self._playwright = await async_playwright().start()

                    # ------------------------
                    # 🔥 LAUNCH BROWSER
                    # ------------------------
                    self.browser = await self._playwright.chromium.launch(
                        headless=headless,
                        args=[
                            "--no-sandbox",
                            "--disable-dev-shm-usage",
                            "--disable-blink-features=AutomationControlled",
                        ],
                    )

                    # ------------------------
                    # 🔥 CREATE CONTEXT (IMPORTANT 🔥)
                    # ------------------------
                    self.context = await self.browser.new_context()

                    # ------------------------
                    # 🔥 CREATE PAGE
                    # ------------------------
                    self.page = await self.context.new_page()

                    # ------------------------
                    # 🔥 SET TIMEOUT
                    # ------------------------
                    try:
                        self.page.set_default_timeout(self.default_timeout)
                    except:
                        pass

                    # ------------------------
                    # 🔥 STATE UPDATE
                    # ------------------------
                    self.is_connected = True
                    self.connected_at = time.time()
                    self.last_action_time = self.connected_at

                    # ------------------------
                    # 🔥 LOG SUCCESS
                    # ------------------------
                    try:
                        logger.info(
                            f"🌐 Browser connected (headless={headless}) "
                            f"in {round(time.time() - start, 2)}s"
                        )
                    except:
                        pass

                    return True

                except Exception as e:
                    self.errors = getattr(self, "errors", 0) + 1

                    try:
                        logger.warning(f"Browser connect retry {attempt}: {e}")
                    except:
                        pass

                    # cleanup partial state
                    try:
                        if self.browser:
                            await self.browser.close()
                    except:
                        pass

                    try:
                        if self._playwright:
                            await self._playwright.stop()
                    except:
                        pass

                    self.browser = None
                    self.page = None
                    self.context = None
                    self._playwright = None
                    self.is_connected = False

                    # retry delay
                    await asyncio.sleep(0.5 + attempt * 0.5)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            try:
                logger.error("❌ Browser connection failed after retries")
            except:
                pass

            raise RuntimeError("Browser connection failed")

    async def navigate(
        self, url: str, wait_until: str = "load", timeout: Optional[int] = None
    ) -> bool:
        """
        Navigate to a URL safely (production-grade)
        """

        import time
        import asyncio
        from urllib.parse import urlparse

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not url or not isinstance(url, str):
                raise ValueError("Invalid URL")

            url = url.strip()

            parsed = urlparse(url)

            if not parsed.scheme:
                url = "https://" + url  # auto-fix

            if not parsed.netloc and "." not in url:
                raise ValueError("Invalid URL format")

            # ------------------------
            # 🔥 CONNECTION CHECK
            # ------------------------
            if not self.is_connected or not self.page:
                raise RuntimeError("Browser not connected")

            # ------------------------
            # 🔥 SAFETY CHECK
            # ------------------------
            if not getattr(self, "allow_navigation", True):
                raise PermissionError("Navigation disabled")

            # ------------------------
            # 🔥 TIMEOUT
            # ------------------------
            timeout = timeout or self.default_timeout

            # ------------------------
            # 🔥 RETRY LOOP
            # ------------------------
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self.page.goto(
                        url, wait_until=wait_until, timeout=timeout
                    )

                    # ------------------------
                    # 🔥 RESPONSE CHECK
                    # ------------------------
                    status = response.status if response else None

                    if status and status >= 400:
                        raise RuntimeError(f"HTTP {status}")

                    # ------------------------
                    # 🔥 STATE UPDATE
                    # ------------------------
                    self.current_url = url
                    self.last_action_time = time.time()
                    self.actions_count += 1

                    if hasattr(self, "history"):
                        self.history.append(url)
                        self.history = self.history[-50:]  # limit history

                    # ------------------------
                    # 🔥 LOG SUCCESS
                    # ------------------------
                    try:
                        logger.info(
                            f"➡ Navigated to {url} "
                            f"({round(time.time() - start, 2)}s)"
                        )
                    except:
                        pass

                    return True

                except Exception as e:
                    self.errors += 1

                    try:
                        logger.warning(f"Navigate retry {attempt}: {e}")
                    except:
                        pass

                    await asyncio.sleep(0.5 + attempt * 0.5)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            raise RuntimeError(f"Navigation failed after retries: {url}")

        except Exception as e:
            # ------------------------
            # 🔥 LOG ERROR
            # ------------------------
            try:
                logger.error(f"❌ Navigation failed: {url} - {e}")
            except:
                pass

            return False

    async def click(
        self,
        selector: str,
        timeout: Optional[int] = None,
        wait_for: Optional[str] = None,
    ) -> bool:
        """
        Click element safely (production-grade)
        """

        import time
        import asyncio

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not selector or not isinstance(selector, str):
                raise ValueError("Invalid selector")

            selector = selector.strip()

            # ------------------------
            # 🔥 CONNECTION CHECK
            # ------------------------
            if not self.is_connected or not self.page:
                raise RuntimeError("Browser not connected")

            # ------------------------
            # 🔥 TIMEOUT
            # ------------------------
            timeout = timeout or self.default_timeout

            # ------------------------
            # 🔥 RETRY LOOP
            # ------------------------
            for attempt in range(self.max_retries + 1):
                try:
                    # ------------------------
                    # 🔥 WAIT FOR ELEMENT
                    # ------------------------
                    await self.page.wait_for_selector(
                        selector, timeout=timeout, state="visible"
                    )

                    # ------------------------
                    # 🔥 CLICK
                    # ------------------------
                    await self.page.click(selector, timeout=timeout)

                    # ------------------------
                    # 🔥 OPTIONAL WAIT AFTER CLICK
                    # ------------------------
                    if wait_for:
                        await self.page.wait_for_selector(wait_for, timeout=timeout)

                    # ------------------------
                    # 🔥 METRICS
                    # ------------------------
                    self.actions_count += 1
                    self.last_action_time = time.time()

                    # ------------------------
                    # 🔥 LOG SUCCESS
                    # ------------------------
                    try:
                        logger.info(
                            f"🖱 Clicked: {selector} "
                            f"({round(time.time() - start, 2)}s)"
                        )
                    except:
                        pass

                    return True

                except Exception as e:
                    self.errors = getattr(self, "errors", 0) + 1

                    try:
                        logger.warning(f"Click retry {attempt}: {e}")
                    except:
                        pass

                    await asyncio.sleep(0.3 + attempt * 0.4)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            raise RuntimeError(f"Click failed after retries: {selector}")

        except Exception as e:
            try:
                logger.error(f"❌ Click failed: {selector} - {e}")
            except:
                pass

            return False

    async def type(
        self,
        selector: str,
        text: str,
        timeout: Optional[int] = None,
        clear: bool = True,
        human_like: bool = False,
    ) -> bool:
        """
        Type text into input safely (production-grade)
        """

        import time
        import asyncio

        start = time.time()

        try:
            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if not selector or not isinstance(selector, str):
                raise ValueError("Invalid selector")

            if text is None:
                raise ValueError("Text cannot be None")

            selector = selector.strip()
            text = str(text)

            # ------------------------
            # 🔥 CONNECTION CHECK
            # ------------------------
            if not self.is_connected or not self.page:
                raise RuntimeError("Browser not connected")

            # ------------------------
            # 🔥 TIMEOUT
            # ------------------------
            timeout = timeout or self.default_timeout

            # ------------------------
            # 🔥 RETRY LOOP
            # ------------------------
            for attempt in range(self.max_retries + 1):
                try:
                    # ------------------------
                    # 🔥 WAIT FOR INPUT
                    # ------------------------
                    await self.page.wait_for_selector(
                        selector, timeout=timeout, state="visible"
                    )

                    # ------------------------
                    # 🔥 CLEAR FIELD (OPTIONAL)
                    # ------------------------
                    if clear:
                        await self.page.fill(selector, "", timeout=timeout)

                    # ------------------------
                    # 🔥 TYPE TEXT
                    # ------------------------
                    if human_like:
                        for char in text:
                            await self.page.type(selector, char, delay=50)
                    else:
                        await self.page.fill(selector, text, timeout=timeout)

                    # ------------------------
                    # 🔥 METRICS
                    # ------------------------
                    self.actions_count += 1
                    self.last_action_time = time.time()

                    # ------------------------
                    # 🔥 LOG SUCCESS
                    # ------------------------
                    try:
                        logger.info(
                            f"⌨ Typed into: {selector} "
                            f"({round(time.time() - start, 2)}s)"
                        )
                    except:
                        pass

                    return True

                except Exception as e:
                    self.errors = getattr(self, "errors", 0) + 1

                    try:
                        logger.warning(f"Type retry {attempt}: {e}")
                    except:
                        pass

                    await asyncio.sleep(0.3 + attempt * 0.4)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            raise RuntimeError(f"Typing failed after retries: {selector}")

        except Exception as e:
            try:
                logger.error(f"❌ Typing failed: {selector} - {e}")
            except:
                pass

            return False

    async def get_content(
        self, timeout: Optional[int] = None, max_length: Optional[int] = None
    ) -> str:
        """
        Get page HTML content safely (production-grade)
        """

        import time
        import asyncio

        start = time.time()

        try:
            # ------------------------
            # 🔥 CONNECTION CHECK
            # ------------------------
            if not self.is_connected or not self.page:
                raise RuntimeError("Browser not connected")

            # ------------------------
            # 🔥 TIMEOUT
            # ------------------------
            timeout = timeout or self.default_timeout

            # ------------------------
            # 🔥 RETRY LOOP
            # ------------------------
            for attempt in range(self.max_retries + 1):
                try:
                    # ------------------------
                    # 🔥 WAIT FOR PAGE READY
                    # ------------------------
                    await self.page.wait_for_load_state("load", timeout=timeout)

                    # ------------------------
                    # 🔥 GET CONTENT
                    # ------------------------
                    content = await self.page.content()

                    if not content:
                        raise ValueError("Empty page content")

                    # ------------------------
                    # 🔥 SIZE CONTROL
                    # ------------------------
                    if max_length and len(content) > max_length:
                        content = content[:max_length]

                    # ------------------------
                    # 🔥 METRICS
                    # ------------------------
                    self.actions_count += 1
                    self.last_action_time = time.time()

                    # ------------------------
                    # 🔥 LOG SUCCESS
                    # ------------------------
                    try:
                        logger.info(
                            f"📄 Page content fetched "
                            f"({len(content)} chars, {round(time.time() - start, 2)}s)"
                        )
                    except:
                        pass

                    return content

                except Exception as e:
                    self.errors = getattr(self, "errors", 0) + 1

                    try:
                        logger.warning(f"Content fetch retry {attempt}: {e}")
                    except:
                        pass

                    await asyncio.sleep(0.4 + attempt * 0.5)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            raise RuntimeError("Failed to fetch page content after retries")

        except Exception as e:
            try:
                logger.error(f"❌ Get content failed: {e}")
            except:
                pass

            return ""

    async def screenshot(
        self,
        path: str = "screenshot.png",
        full_page: bool = False,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """
        Take screenshot safely (production-grade)
        """

        import time
        import asyncio
        from pathlib import Path

        start = time.time()

        try:
            # ------------------------
            # 🔥 CONNECTION CHECK
            # ------------------------
            if not self.is_connected or not self.page:
                raise RuntimeError("Browser not connected")

            # ------------------------
            # 🔥 VALIDATE PATH
            # ------------------------
            if not path or not isinstance(path, str):
                raise ValueError("Invalid path")

            path = path.strip()

            file_path = Path(path).expanduser().resolve()

            # ensure directory exists
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except:
                pass

            # ------------------------
            # 🔥 TIMEOUT
            # ------------------------
            timeout = timeout or self.default_timeout

            # ------------------------
            # 🔥 RETRY LOOP
            # ------------------------
            for attempt in range(self.max_retries + 1):
                try:
                    # ------------------------
                    # 🔥 WAIT FOR PAGE READY
                    # ------------------------
                    await self.page.wait_for_load_state("load", timeout=timeout)

                    # ------------------------
                    # 🔥 TAKE SCREENSHOT
                    # ------------------------
                    await self.page.screenshot(
                        path=str(file_path), full_page=full_page, timeout=timeout
                    )

                    # ------------------------
                    # 🔥 METRICS
                    # ------------------------
                    self.actions_count += 1
                    self.last_action_time = time.time()

                    # ------------------------
                    # 🔥 LOG SUCCESS
                    # ------------------------
                    try:
                        logger.info(
                            f"📸 Screenshot saved: {file_path} "
                            f"({round(time.time() - start, 2)}s)"
                        )
                    except:
                        pass

                    return str(file_path)

                except Exception as e:
                    self.errors = getattr(self, "errors", 0) + 1

                    try:
                        logger.warning(f"Screenshot retry {attempt}: {e}")
                    except:
                        pass

                    await asyncio.sleep(0.4 + attempt * 0.5)

            # ------------------------
            # 🔥 FINAL FAILURE
            # ------------------------
            raise RuntimeError("Screenshot failed after retries")

        except Exception as e:
            try:
                logger.error(f"❌ Screenshot failed: {e}")
            except:
                pass

            return None

    async def disconnect(self, force: bool = False) -> bool:
        """
        Safely disconnect browser and cleanup resources (production-grade)
        """

        import time

        start = time.time()

        try:
            # ------------------------
            # 🔥 LOCK (THREAD SAFE)
            # ------------------------
            if hasattr(self, "_lock"):
                async with self._lock:
                    return await self._disconnect_internal(force, start)
            else:
                return await self._disconnect_internal(force, start)

        except Exception as e:
            try:
                logger.error(f"❌ Disconnect failed: {e}")
            except:
                pass

            return False

    async def _disconnect_internal(self, force: bool, start: float) -> bool:
        """
        Internal cleanup logic
        """

        try:
            # ------------------------
            # 🔥 CLOSE PAGE
            # ------------------------
            try:
                if self.page:
                    await self.page.close()
            except:
                pass

            # ------------------------
            # 🔥 CLOSE CONTEXT
            # ------------------------
            try:
                if getattr(self, "context", None):
                    await self.context.close()
            except:
                pass

            # ------------------------
            # 🔥 CLOSE BROWSER
            # ------------------------
            try:
                if self.browser:
                    await self.browser.close()
            except:
                pass

            # ------------------------
            # 🔥 STOP PLAYWRIGHT
            # ------------------------
            try:
                if self._playwright:
                    await self._playwright.stop()
            except:
                pass

            # ------------------------
            # 🔥 RESET STATE
            # ------------------------
            self.browser = None
            self.page = None
            self.context = None
            self._playwright = None
            self.is_connected = False

            # ------------------------
            # 🔥 METRICS UPDATE
            # ------------------------
            try:
                self.last_action_time = time.time()
            except:
                pass

            # ------------------------
            # 🔥 LOG SUCCESS
            # ------------------------
            try:
                logger.info(
                    f"❌ Browser disconnected " f"({round(time.time() - start, 2)}s)"
                )
            except:
                pass

            return True

        except Exception as e:
            try:
                logger.error(f"❌ Internal disconnect error: {e}")
            except:
                pass

            # force cleanup fallback
            self.browser = None
            self.page = None
            self.context = None
            self._playwright = None
            self.is_connected = False

            return False
