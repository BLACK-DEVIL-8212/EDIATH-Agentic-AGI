"""
🔥 FINAL PRODUCTION Browser Agent for EDIATH
✔ Advanced browser automation
✔ Multiple browser support (Chromium, Firefox, WebKit, Chrome, Edge)
✔ Page navigation and history management
✔ Element interaction (click, type, hover, select, drag-drop)
✔ Data extraction (text, attributes, HTML, screenshots)
✔ Form filling and submission
✔ JavaScript execution
✔ Network request interception
✔ Cookie management
✔ Multiple tab/window handling
✔ File upload/download
✔ Responsive viewport testing
✔ Authentication handling
✔ Screenshot and PDF generation
✔ Production ready
"""

import asyncio
import json
import base64
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import logging
from enum import Enum
from dataclasses import dataclass, field
import aiofiles

try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        Page,
        ElementHandle,
        Response,
    )
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    import logging

    logging.getLogger(__name__).warning(
        "Warning: playwright not installed. Install with: pip install playwright && playwright install"
    )


# =========================
# ENUMS AND CONSTANTS
# =========================


class BrowserType(Enum):
    """Supported browser types"""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    CHROME = "chrome"
    EDGE = "msedge"


class WaitCondition(Enum):
    """Conditions for waiting"""

    LOAD = "load"
    DOM_CONTENT_LOADED = "domcontentloaded"
    NETWORK_IDLE = "networkidle"
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ATTACHED = "attached"
    DETACHED = "detached"


class MouseButton(Enum):
    """Mouse buttons"""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class KeyModifier(Enum):
    """Keyboard modifiers"""

    ALT = "Alt"
    CONTROL = "Control"
    META = "Meta"
    SHIFT = "Shift"


class ElementState(Enum):
    """Element state for waiting"""

    VISIBLE = "visible"
    HIDDEN = "hidden"
    ENABLED = "enabled"
    DISABLED = "disabled"
    EDITABLE = "editable"


# =========================
# DATACLASSES
# =========================


@dataclass
class ElementInfo:
    """Information about a DOM element"""

    tag: str
    text: str
    html: str
    attributes: Dict[str, str]
    visible: bool
    enabled: bool
    position: Dict[str, int]
    size: Dict[str, int]
    xpath: str
    css_path: str
    inner_text: str = ""
    outer_html: str = ""


@dataclass
class PageInfo:
    """Information about a web page"""

    url: str
    title: str
    content_length: int
    status_code: int
    load_time: float
    timestamp: datetime
    screenshot: Optional[str] = None
    cookies_count: int = 0
    resources_count: int = 0


@dataclass
class NetworkRequest:
    """Network request information"""

    id: str
    url: str
    method: str
    resource_type: str
    timestamp: datetime
    status: Optional[int] = None
    status_text: Optional[str] = None
    duration_ms: Optional[float] = None
    headers: Dict[str, str] = field(default_factory=dict)
    post_data: Optional[str] = None


# =========================
# BROWSER AGENT
# =========================


class BrowserAgent:
    """
    Advanced browser automation agent capable of:
    - Launching and managing browser instances
    - Navigating to URLs and handling navigation
    - Clicking, typing, hovering, and scrolling
    - Extracting data from pages (text, attributes, HTML)
    - Taking screenshots and generating PDFs
    - Handling multiple tabs and windows
    - Waiting for elements and conditions
    - Executing JavaScript in page context
    - Handling authentication and cookies
    - Network request interception
    - Form filling and submission
    - Drag and drop operations
    - File uploads and downloads
    - Responsive design testing
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Browser Agent

        Args:
            config: Configuration dictionary
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required. Install with: pip install playwright && playwright install"
            )

        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Browser configuration
        self.browser_type = BrowserType(self.config.get("browser_type", "chromium"))
        self.headless = self.config.get("headless", False)
        self.viewport = self.config.get("viewport", {"width": 1920, "height": 1080})
        self.user_agent = self.config.get("user_agent")
        self.timeout = self.config.get("timeout", 30000)  # milliseconds
        self.default_wait_time = self.config.get(
            "default_wait_time", 5000
        )  # milliseconds
        self.slow_mo = self.config.get("slow_mo", 0)  # milliseconds between operations

        # Browser instance
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context = None
        self.page: Optional[Page] = None
        self.current_page_info: Optional[PageInfo] = None

        # Page history for navigation
        self.page_history: List[str] = []
        self.history_index = -1

        # Cookies management
        self.cookies_file = self.config.get("cookies_file", "browser_cookies.json")

        # Screenshot directory
        self.screenshot_dir = Path(self.config.get("screenshot_dir", "screenshots"))
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Download directory
        self.download_dir = Path(self.config.get("download_dir", "downloads"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "pages_visited": 0,
            "actions_performed": 0,
            "screenshots_taken": 0,
            "errors": 0,
            "navigation_time": 0.0,
            "start_time": None,
            "total_requests": 0,
            "total_responses": 0,
        }

        # Network request tracking
        self.requests: List[NetworkRequest] = []
        self.responses: List[Dict] = []
        self._request_counter = 0

        # Console messages
        self.console_messages: List[Dict] = []

        # Download tracking
        self.downloads: List[Dict] = []

        # Authentication
        self.auth_credentials: Optional[Dict] = None

        # Interception
        self.route_handlers: Dict[str, callable] = {}

        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {}

        self.logger.info(
            f"Browser Agent initialized. Browser: {self.browser_type.value}, Headless: {self.headless}"
        )

    # =========================
    # INITIALIZATION
    # =========================

    async def start(self, headless: Optional[bool] = None) -> Dict[str, Any]:
        """
        Start browser instance

        Args:
            headless: Override headless setting

        Returns:
            Dictionary with startup result
        """
        try:
            self.stats["start_time"] = datetime.now().isoformat()

            self.playwright = await async_playwright().start()

            # Browser launch options
            launch_options = {
                "headless": headless if headless is not None else self.headless,
                "slow_mo": self.slow_mo,
            }

            # Select browser type
            if self.browser_type == BrowserType.CHROMIUM:
                self.browser = await self.playwright.chromium.launch(**launch_options)
            elif self.browser_type == BrowserType.FIREFOX:
                self.browser = await self.playwright.firefox.launch(**launch_options)
            elif self.browser_type == BrowserType.WEBKIT:
                self.browser = await self.playwright.webkit.launch(**launch_options)
            elif self.browser_type == BrowserType.CHROME:
                self.browser = await self.playwright.chromium.launch(
                    channel="chrome", **launch_options
                )
            elif self.browser_type == BrowserType.EDGE:
                self.browser = await self.playwright.chromium.launch(
                    channel="msedge", **launch_options
                )
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")

            # Create browser context
            context_options = {
                "viewport": self.viewport,
                "ignore_https_errors": self.config.get("ignore_https_errors", True),
                "accept_downloads": True,
                "java_script_enabled": self.config.get("javascript_enabled", True),
                "bypass_csp": self.config.get("bypass_csp", False),
                "locale": self.config.get("locale", "en-US"),
                "timezone_id": self.config.get("timezone_id", "America/New_York"),
            }

            if self.user_agent:
                context_options["user_agent"] = self.user_agent

            # Set download path
            context_options["accept_downloads"] = True

            self.context = await self.browser.new_context(**context_options)

            # Set default timeout
            self.context.set_default_timeout(self.timeout)

            # Setup download handler
            await self.context.expose_function("onDownload", self._on_download)

            # Create new page
            self.page = await self.context.new_page()

            # Setup event handlers
            await self._setup_event_handlers()

            # Load cookies if available
            if self.config.get("load_cookies_on_start", False):
                await self.load_cookies()

            # Setup authentication if provided
            if self.auth_credentials:
                await self.setup_authentication(self.auth_credentials)

            self.logger.info(f"Browser started successfully: {self.browser_type.value}")

            return {
                "success": True,
                "browser_type": self.browser_type.value,
                "headless": headless if headless is not None else self.headless,
                "viewport": self.viewport,
                "message": "Browser started successfully",
            }

        except Exception as e:
            self.logger.error(f"Failed to start browser: {str(e)}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start browser",
            }

    async def _setup_event_handlers(self):
        """Setup page event handlers"""
        if not self.page:
            return

        # Track navigation
        self.page.on("framenavigated", self._on_navigation)

        # Track requests
        self.page.on("request", self._on_request)

        # Track responses
        self.page.on("response", self._on_response)

        # Track console messages
        self.page.on("console", self._on_console)

        # Track dialogs
        self.page.on("dialog", self._on_dialog)

        # Track page errors
        self.page.on("pageerror", self._on_page_error)

        # Track download
        self.page.on("download", self._on_download_start)

    def _on_navigation(self, frame):
        """Handle navigation events"""
        if frame == self.page.main_frame:
            self.logger.debug(f"Navigation to: {frame.url}")

    def _on_request(self, request):
        """Track network requests"""
        self._request_counter += 1
        req = NetworkRequest(
            id=str(self._request_counter),
            url=request.url,
            method=request.method,
            resource_type=request.resource_type,
            timestamp=datetime.now(),
            headers=dict(request.headers),
            post_data=request.post_data,
        )
        self.requests.append(req)
        self.stats["total_requests"] += 1

        # Limit history size
        if len(self.requests) > 1000:
            self.requests = self.requests[-500:]

    def _on_response(self, response):
        """Track network responses"""
        # Find matching request
        for req in reversed(self.requests):
            if req.url == response.url and req.method == response.request.method:
                req.status = response.status
                req.status_text = response.status_text
                req.duration_ms = (
                    datetime.now() - req.timestamp
                ).total_seconds() * 1000
                break

        self.responses.append(
            {
                "url": response.url,
                "status": response.status,
                "status_text": response.status_text,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.stats["total_responses"] += 1

        # Limit history size
        if len(self.responses) > 1000:
            self.responses = self.responses[-500:]

    def _on_console(self, msg):
        """Handle console messages"""
        self.console_messages.append(
            {
                "type": msg.type,
                "text": msg.text,
                "location": msg.location,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Limit history
        if len(self.console_messages) > 1000:
            self.console_messages = self.console_messages[-500:]

        self.logger.debug(f"Console [{msg.type}]: {msg.text}")

    async def _on_dialog(self, dialog):
        """Handle page dialogs"""
        self.logger.info(f"Dialog: {dialog.message}")

        # Auto-dismiss or auto-accept based on config
        if self.config.get("auto_dismiss_dialogs", True):
            await dialog.dismiss()
        elif self.config.get("auto_accept_dialogs", False):
            await dialog.accept()

    def _on_page_error(self, error):
        """Handle page errors"""
        self.logger.error(f"Page error: {error}")
        self.stats["errors"] += 1

    def _on_download_start(self, download):
        """Handle download start"""
        download_info = {
            "url": download.url,
            "suggested_filename": download.suggested_filename,
            "timestamp": datetime.now().isoformat(),
        }
        self.downloads.append(download_info)
        self.logger.info(f"Download started: {download.suggested_filename}")

    async def _on_download(self, download_info):
        """Handle download completion"""
        pass

    # =========================
    # AUTHENTICATION
    # =========================

    async def setup_authentication(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """
        Setup HTTP authentication

        Args:
            credentials: Dictionary with 'username' and 'password'

        Returns:
            Dictionary with result
        """
        self.auth_credentials = credentials

        if self.context:
            await self.context.set_http_credentials(
                username=credentials.get("username"),
                password=credentials.get("password"),
            )
            return {"success": True, "message": "Authentication set"}

        return {
            "success": True,
            "message": "Authentication will be applied on next context",
        }

    # =========================
    # NETWORK INTERCEPTION
    # =========================

    async def intercept_route(
        self, url_pattern: str, handler: callable
    ) -> Dict[str, Any]:
        """
        Intercept network requests matching pattern

        Args:
            url_pattern: URL pattern to intercept
            handler: Async function to handle request

        Returns:
            Dictionary with result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.route(url_pattern, handler)
            self.route_handlers[url_pattern] = handler
            return {"success": True, "message": f"Route {url_pattern} intercepted"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def block_resources(self, resource_types: List[str]) -> Dict[str, Any]:
        """
        Block specific resource types (images, stylesheets, etc.)

        Args:
            resource_types: List of resource types to block

        Returns:
            Dictionary with result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        async def block_route(route):
            if route.request.resource_type in resource_types:
                await route.abort()
            else:
                await route.continue_()

        await self.page.route("**/*", block_route)
        return {"success": True, "blocked_types": resource_types}

    # =========================
    # NAVIGATION
    # =========================

    async def navigate(
        self,
        url: str,
        wait_until: WaitCondition = WaitCondition.NETWORK_IDLE,
        timeout: Optional[int] = None,
        referer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Navigate to a URL

        Args:
            url: URL to navigate to
            wait_until: Condition to wait for
            timeout: Navigation timeout in milliseconds
            referer: Referer header value

        Returns:
            Dictionary with navigation result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        start_time = datetime.now()

        try:
            self.logger.info(f"Navigating to: {url}")

            # Navigate to URL
            response = await self.page.goto(
                url,
                wait_until=wait_until.value,
                timeout=timeout or self.timeout,
                referer=referer,
            )

            # Update history
            self.page_history.append(url)
            self.history_index = len(self.page_history) - 1

            # Get page info
            page_info = await self.get_page_info()
            page_info["load_time"] = (datetime.now() - start_time).total_seconds()
            page_info["status_code"] = response.status if response else 200

            self.current_page_info = PageInfo(**page_info)

            # Update statistics
            self.stats["pages_visited"] += 1
            self.stats["navigation_time"] += page_info["load_time"]

            return {
                "success": True,
                "url": url,
                "status_code": response.status if response else 200,
                "load_time": page_info["load_time"],
                "title": page_info["title"],
                "message": f"Successfully navigated to {url}",
            }

        except PlaywrightTimeoutError:
            self.logger.error(f"Navigation timeout: {url}")
            self.stats["errors"] += 1
            return {
                "success": False,
                "url": url,
                "error": f"Navigation timeout after {timeout or self.timeout}ms",
                "timeout": True,
            }
        except Exception as e:
            self.logger.error(f"Navigation error: {str(e)}")
            self.stats["errors"] += 1
            return {"success": False, "url": url, "error": str(e)}

    async def go_back(self) -> Dict[str, Any]:
        """Navigate back in history"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.go_back()
            return {
                "success": True,
                "current_url": self.page.url,
                "message": "Navigated back",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def go_forward(self) -> Dict[str, Any]:
        """Navigate forward in history"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.go_forward()
            return {
                "success": True,
                "current_url": self.page.url,
                "message": "Navigated forward",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def reload(
        self, wait_until: WaitCondition = WaitCondition.NETWORK_IDLE
    ) -> Dict[str, Any]:
        """Reload current page"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.reload(wait_until=wait_until.value)
            return {
                "success": True,
                "current_url": self.page.url,
                "message": "Page reloaded",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # ELEMENT INTERACTION
    # =========================

    async def click(
        self,
        selector: str,
        button: MouseButton = MouseButton.LEFT,
        click_count: int = 1,
        force: bool = False,
        modifiers: List[KeyModifier] = None,
        position: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Click on an element

        Args:
            selector: CSS selector or XPath
            button: Mouse button to use
            click_count: Number of clicks (1 = single, 2 = double)
            force: Force click even if element is not visible
            modifiers: Keyboard modifiers to hold
            position: Click position relative to element
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with click result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Wait for element
            element = await self.page.wait_for_selector(
                selector,
                state="visible" if not force else "attached",
                timeout=timeout or self.default_wait_time,
            )

            if not element:
                return {"success": False, "error": f"Element not found: {selector}"}

            # Prepare click options
            click_options = {
                "button": button.value,
                "click_count": click_count,
                "force": force,
            }

            if modifiers:
                click_options["modifiers"] = [m.value for m in modifiers]

            if position:
                click_options["position"] = position

            # Perform click
            await element.click(**click_options)

            self.stats["actions_performed"] += 1

            return {
                "success": True,
                "selector": selector,
                "button": button.value,
                "click_count": click_count,
                "message": f"Clicked element: {selector}",
            }

        except PlaywrightTimeoutError:
            return {
                "success": False,
                "selector": selector,
                "error": f"Element not found within timeout: {selector}",
            }
        except Exception as e:
            self.logger.error(f"Click error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def type_text(
        self,
        selector: str,
        text: str,
        delay: int = 100,
        clear_first: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Type text into an input field

        Args:
            selector: CSS selector or XPath
            text: Text to type
            delay: Delay between keystrokes in milliseconds
            clear_first: Clear existing text first
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with typing result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Wait for element
            element = await self.page.wait_for_selector(
                selector, state="visible", timeout=timeout or self.default_wait_time
            )

            if not element:
                return {"success": False, "error": f"Element not found: {selector}"}

            # Clear if requested
            if clear_first:
                await element.fill("")

            # Type text
            await element.type(text, delay=delay)

            self.stats["actions_performed"] += 1

            return {
                "success": True,
                "selector": selector,
                "text_length": len(text),
                "message": f"Typed text into {selector}",
            }

        except Exception as e:
            self.logger.error(f"Type error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def fill_form(
        self,
        form_data: Dict[str, str],
        selector_prefix: str = "",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fill multiple form fields

        Args:
            form_data: Dictionary mapping field names/selectors to values
            selector_prefix: Prefix to add to field names (e.g., '#', '.')
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with form filling result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        filled_fields = []
        failed_fields = []

        for field_selector, value in form_data.items():
            # Build full selector
            if not field_selector.startswith(
                ("#", ".", "[", "input", "textarea", "select")
            ):
                field_selector = f"{selector_prefix}{field_selector}"

            try:
                # Wait for element
                element = await self.page.wait_for_selector(
                    field_selector,
                    state="visible",
                    timeout=timeout or self.default_wait_time,
                )

                if element:
                    await element.fill(str(value))
                    filled_fields.append(field_selector)
                else:
                    failed_fields.append(field_selector)

            except Exception:
                failed_fields.append(field_selector)

        self.stats["actions_performed"] += len(filled_fields)

        return {
            "success": len(failed_fields) == 0,
            "filled_fields": filled_fields,
            "failed_fields": failed_fields,
            "total_fields": len(form_data),
            "success_count": len(filled_fields),
            "failure_count": len(failed_fields),
        }

    async def submit_form(
        self, selector: str = "form", timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Submit a form

        Args:
            selector: Form selector
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with submission result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Find form
            form = await self.page.wait_for_selector(
                selector, state="visible", timeout=timeout or self.default_wait_time
            )

            if not form:
                return {"success": False, "error": f"Form not found: {selector}"}

            # Submit form
            await form.press("Enter")

            # Wait for navigation
            await self.page.wait_for_load_state("networkidle")

            self.stats["actions_performed"] += 1

            return {
                "success": True,
                "selector": selector,
                "message": "Form submitted successfully",
            }

        except Exception as e:
            self.logger.error(f"Form submission error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def hover(
        self, selector: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Hover over element

        Args:
            selector: CSS selector or XPath
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with hover result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            element = await self.page.wait_for_selector(
                selector, timeout=timeout or self.default_wait_time
            )

            if element:
                await element.hover()
                self.stats["actions_performed"] += 1
                return {
                    "success": True,
                    "selector": selector,
                    "message": f"Hovered over element: {selector}",
                }
            else:
                return {"success": False, "error": f"Element not found: {selector}"}

        except Exception as e:
            self.logger.error(f"Hover error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def select_option(
        self, selector: str, value: Union[str, List[str]], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select option from dropdown

        Args:
            selector: Select element selector
            value: Option value(s) to select
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with selection result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            element = await self.page.wait_for_selector(
                selector, timeout=timeout or self.default_wait_time
            )

            if element:
                await element.select_option(value)
                self.stats["actions_performed"] += 1
                return {
                    "success": True,
                    "selector": selector,
                    "selected": value,
                    "message": f"Selected option(s) in {selector}",
                }
            else:
                return {"success": False, "error": f"Element not found: {selector}"}

        except Exception as e:
            self.logger.error(f"Select error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def drag_and_drop(
        self, source_selector: str, target_selector: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Drag and drop element

        Args:
            source_selector: Source element selector
            target_selector: Target element selector
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with drag-drop result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.drag_and_drop(
                source_selector, target_selector, timeout=timeout or self.timeout
            )

            self.stats["actions_performed"] += 1

            return {
                "success": True,
                "source": source_selector,
                "target": target_selector,
                "message": "Drag and drop completed",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def check(
        self, selector: str, state: bool = True, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check or uncheck a checkbox

        Args:
            selector: Checkbox selector
            state: True to check, False to uncheck
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            element = await self.page.wait_for_selector(
                selector, timeout=timeout or self.default_wait_time
            )

            if element:
                if state:
                    await element.check()
                else:
                    await element.uncheck()

                self.stats["actions_performed"] += 1
                return {
                    "success": True,
                    "selector": selector,
                    "checked": state,
                    "message": f"Checkbox {selector} set to {state}",
                }
            else:
                return {"success": False, "error": f"Element not found: {selector}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # DATA EXTRACTION
    # =========================

    async def get_text(
        self, selector: str, all_matches: bool = False, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get text content from element(s)

        Args:
            selector: CSS selector or XPath
            all_matches: Get text from all matching elements
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with text content
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            if all_matches:
                elements = await self.page.query_selector_all(selector)
                texts = []
                for elem in elements:
                    text = await elem.text_content()
                    if text:
                        texts.append(text.strip())
                return {
                    "success": True,
                    "selector": selector,
                    "texts": texts,
                    "count": len(texts),
                }
            else:
                element = await self.page.wait_for_selector(
                    selector, timeout=timeout or self.default_wait_time
                )
                if element:
                    text = await element.text_content()
                    return {
                        "success": True,
                        "selector": selector,
                        "text": text.strip() if text else "",
                        "found": True,
                    }
                else:
                    return {
                        "success": False,
                        "selector": selector,
                        "error": "Element not found",
                    }

        except Exception as e:
            self.logger.error(f"Get text error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def get_attribute(
        self, selector: str, attribute: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get attribute value from element

        Args:
            selector: CSS selector or XPath
            attribute: Attribute name
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with attribute value
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            element = await self.page.wait_for_selector(
                selector, timeout=timeout or self.default_wait_time
            )

            if element:
                value = await element.get_attribute(attribute)
                return {
                    "success": True,
                    "selector": selector,
                    "attribute": attribute,
                    "value": value,
                    "found": True,
                }
            else:
                return {
                    "success": False,
                    "selector": selector,
                    "error": "Element not found",
                }

        except Exception as e:
            self.logger.error(f"Get attribute error: {str(e)}")
            return {"success": False, "selector": selector, "error": str(e)}

    async def get_html(
        self, selector: Optional[str] = None, outer_html: bool = True
    ) -> Dict[str, Any]:
        """
        Get HTML content

        Args:
            selector: Optional selector for specific element
            outer_html: Get outer HTML (True) or inner HTML (False)

        Returns:
            Dictionary with HTML content
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    if outer_html:
                        html = await element.evaluate("el => el.outerHTML")
                    else:
                        html = await element.evaluate("el => el.innerHTML")
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                html = await self.page.content()

            return {
                "success": True,
                "html": html,
                "length": len(html),
                "selector": selector if selector else "full_page",
            }

        except Exception as e:
            self.logger.error(f"Get HTML error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_element_info(
        self, selector: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about an element

        Args:
            selector: CSS selector or XPath
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with element information
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            element = await self.page.wait_for_selector(
                selector, timeout=timeout or self.default_wait_time
            )

            if not element:
                return {"success": False, "error": f"Element not found: {selector}"}

            # Get element info
            info = await element.evaluate("""
                (el) => ({
                    tag: el.tagName.toLowerCase(),
                    text: el.textContent || '',
                    innerText: el.innerText || '',
                    outerHTML: el.outerHTML,
                    innerHTML: el.innerHTML,
                    attributes: Array.from(el.attributes).reduce((obj, attr) => {
                        obj[attr.name] = attr.value;
                        return obj;
                    }, {}),
                    visible: el.offsetParent !== null,
                    enabled: !el.disabled,
                    rect: el.getBoundingClientRect()
                })
            """)

            return {"success": True, "selector": selector, "info": info}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # SCREENSHOT & PDF
    # =========================

    async def take_screenshot(
        self,
        filename: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
        quality: Optional[int] = None,
        type: str = "png",
    ) -> Dict[str, Any]:
        """
        Take screenshot of page or element

        Args:
            filename: Output filename (without extension)
            full_page: Capture full scrollable page
            selector: Optional selector for element screenshot
            quality: JPEG quality (0-100)
            type: Image type ('png' or 'jpeg')

        Returns:
            Dictionary with screenshot info
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Generate filename if not provided
            if not filename:
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            filepath = self.screenshot_dir / f"{filename}.{type}"

            # Take screenshot
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    await element.screenshot(path=str(filepath))
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                screenshot_options = {"path": str(filepath), "full_page": full_page}
                if quality and type == "jpeg":
                    screenshot_options["quality"] = quality
                await self.page.screenshot(**screenshot_options)

            self.stats["screenshots_taken"] += 1

            return {
                "success": True,
                "path": str(filepath),
                "filename": f"{filename}.{type}",
                "full_page": full_page,
                "selector": selector if selector else "full_page",
                "size_bytes": filepath.stat().st_size,
            }

        except Exception as e:
            self.logger.error(f"Screenshot error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_screenshot_base64(
        self,
        full_page: bool = False,
        selector: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get screenshot as base64 string

        Args:
            full_page: Capture full scrollable page
            selector: Optional selector for element screenshot
            quality: JPEG quality (0-100)

        Returns:
            Dictionary with base64 screenshot
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    screenshot_bytes = await element.screenshot()
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                screenshot_options = {"full_page": full_page}
                if quality:
                    screenshot_options["quality"] = quality
                screenshot_bytes = await self.page.screenshot(**screenshot_options)

            base64_str = base64.b64encode(screenshot_bytes).decode("utf-8")

            return {
                "success": True,
                "base64": base64_str,
                "format": "png",
                "size_bytes": len(screenshot_bytes),
                "full_page": full_page,
            }

        except Exception as e:
            self.logger.error(f"Screenshot base64 error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_pdf(
        self,
        filename: Optional[str] = None,
        format: str = "A4",
        landscape: bool = False,
        margin: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate PDF of current page

        Args:
            filename: Output filename
            format: Paper format (A4, Letter, etc.)
            landscape: Landscape orientation
            margin: Page margins

        Returns:
            Dictionary with PDF info
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            # Generate filename if not provided
            if not filename:
                filename = f"page_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            filepath = self.screenshot_dir / f"{filename}.pdf"

            # PDF options
            pdf_options = {
                "path": str(filepath),
                "format": format,
                "landscape": landscape,
            }

            if margin:
                pdf_options["margin"] = margin

            # Generate PDF
            await self.page.pdf(**pdf_options)

            return {
                "success": True,
                "path": str(filepath),
                "filename": f"{filename}.pdf",
                "format": format,
                "landscape": landscape,
                "size_bytes": filepath.stat().st_size,
            }

        except Exception as e:
            self.logger.error(f"PDF generation error: {str(e)}")
            return {"success": False, "error": str(e)}

    # =========================
    # JAVASCRIPT EXECUTION
    # =========================

    async def execute_script(self, script: str, *args) -> Dict[str, Any]:
        """
        Execute JavaScript in page context

        Args:
            script: JavaScript code to execute
            *args: Arguments to pass to script

        Returns:
            Dictionary with script result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            result = await self.page.evaluate(script, *args)

            self.stats["actions_performed"] += 1

            return {
                "success": True,
                "result": result,
                "script": script[:100] + "..." if len(script) > 100 else script,
            }

        except Exception as e:
            self.logger.error(f"Script execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "script": script[:100] + "..." if len(script) > 100 else script,
            }

    # =========================
    # WAITING
    # =========================

    async def wait_for_element(
        self, selector: str, state: str = "visible", timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wait for element to appear/disappear

        Args:
            selector: CSS selector or XPath
            state: Element state ('visible', 'hidden', 'attached', 'detached')
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with wait result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.wait_for_selector(
                selector, state=state, timeout=timeout or self.default_wait_time
            )

            return {
                "success": True,
                "selector": selector,
                "state": state,
                "message": f"Element {selector} is {state}",
            }

        except PlaywrightTimeoutError:
            return {
                "success": False,
                "selector": selector,
                "error": f"Element not {state} within timeout",
            }
        except Exception as e:
            return {"success": False, "selector": selector, "error": str(e)}

    async def wait_for_timeout(self, milliseconds: int) -> Dict[str, Any]:
        """
        Wait for specified time

        Args:
            milliseconds: Time to wait in milliseconds

        Returns:
            Dictionary with result
        """
        await asyncio.sleep(milliseconds / 1000)
        return {"success": True, "waited_ms": milliseconds}

    async def wait_for_url(
        self, url_pattern: str, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wait for URL to match pattern

        Args:
            url_pattern: URL pattern (string or regex)
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.wait_for_url(url_pattern, timeout=timeout or self.timeout)
            return {"success": True, "url": self.page.url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_function(
        self, script: str, *args, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wait for JavaScript function to return truthy value

        Args:
            script: JavaScript function to evaluate
            *args: Arguments to pass
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            result = await self.page.wait_for_function(
                script, timeout=timeout or self.timeout, *args
            )
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # SCROLLING
    # =========================

    async def scroll_to(
        self,
        selector: Optional[str] = None,
        x: int = 0,
        y: int = 0,
        smooth: bool = False,
    ) -> Dict[str, Any]:
        """
        Scroll to element or position

        Args:
            selector: Scroll to element (if provided)
            x: Horizontal scroll position
            y: Vertical scroll position
            smooth: Smooth scrolling

        Returns:
            Dictionary with scroll result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            behavior = "smooth" if smooth else "auto"

            if selector:
                element = await self.page.wait_for_selector(selector)
                if element:
                    await element.scroll_into_view_if_needed()
                    message = f"Scrolled to element: {selector}"
                else:
                    return {"success": False, "error": f"Element not found: {selector}"}
            else:
                await self.page.evaluate(
                    f'window.scrollTo({{ left: {x}, top: {y}, behavior: "{behavior}" }})'
                )
                message = f"Scrolled to position ({x}, {y})"

            self.stats["actions_performed"] += 1

            return {
                "success": True,
                "message": message,
                "selector": selector if selector else None,
                "position": {"x": x, "y": y} if not selector else None,
            }

        except Exception as e:
            self.logger.error(f"Scroll error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def scroll_to_bottom(self, smooth: bool = False) -> Dict[str, Any]:
        """Scroll to bottom of page"""
        return await self.scroll_to(y=999999, smooth=smooth)

    async def scroll_to_top(self, smooth: bool = False) -> Dict[str, Any]:
        """Scroll to top of page"""
        return await self.scroll_to(y=0, smooth=smooth)

    # =========================
    # TAB MANAGEMENT
    # =========================

    async def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Open new tab

        Args:
            url: URL to open in new tab (optional)

        Returns:
            Dictionary with new tab info
        """
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            new_page = await self.context.new_page()

            if url:
                await new_page.goto(url)

            self.page = new_page
            await self._setup_event_handlers()

            return {
                "success": True,
                "url": url if url else "about:blank",
                "tabs_count": len(self.context.pages),
                "message": "New tab created",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def close_tab(self, page_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Close current tab or tab by index

        Args:
            page_index: Index of tab to close (None for current)

        Returns:
            Dictionary with result
        """
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            pages = self.context.pages

            if page_index is not None and 0 <= page_index < len(pages):
                page_to_close = pages[page_index]
            else:
                page_to_close = self.page

            await page_to_close.close()

            # Switch to first available page
            if self.context.pages:
                self.page = self.context.pages[0]
                await self._setup_event_handlers()
                return {
                    "success": True,
                    "message": "Tab closed, switched to existing tab",
                    "tabs_remaining": len(self.context.pages),
                }
            else:
                # Create new page if none exists
                self.page = await self.context.new_page()
                await self._setup_event_handlers()
                return {
                    "success": True,
                    "message": "Tab closed, created new tab",
                    "tabs_remaining": 1,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def switch_tab(self, index: int) -> Dict[str, Any]:
        """
        Switch to tab by index

        Args:
            index: Tab index

        Returns:
            Dictionary with result
        """
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            pages = self.context.pages
            if 0 <= index < len(pages):
                self.page = pages[index]
                await self._setup_event_handlers()
                return {
                    "success": True,
                    "index": index,
                    "url": self.page.url,
                    "total_tabs": len(pages),
                    "message": f"Switched to tab {index}",
                }
            else:
                return {"success": False, "error": f"Invalid tab index: {index}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_tabs(self) -> Dict[str, Any]:
        """Get list of open tabs"""
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        tabs = []
        for i, page in enumerate(self.context.pages):
            tabs.append(
                {
                    "index": i,
                    "url": page.url,
                    "title": await page.title(),
                    "is_active": page == self.page,
                }
            )

        return {
            "success": True,
            "tabs": tabs,
            "count": len(tabs),
            "active_index": next((i for i, t in enumerate(tabs) if t["is_active"]), -1),
        }

    # =========================
    # COOKIE MANAGEMENT
    # =========================

    async def save_cookies(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Save cookies to file"""
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            cookies = await self.context.cookies()
            save_path = filepath or self.cookies_file

            async with aiofiles.open(save_path, "w") as f:
                await f.write(json.dumps(cookies, indent=2))

            return {
                "success": True,
                "cookies_file": save_path,
                "cookies_count": len(cookies),
                "message": f"Cookies saved to {save_path}",
            }
        except Exception as e:
            self.logger.error(f"Save cookies error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def load_cookies(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load cookies from file"""
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            load_path = filepath or self.cookies_file
            cookie_file = Path(load_path)
            if not cookie_file.exists():
                return {"success": True, "message": "No cookies file found"}

            async with aiofiles.open(load_path, "r") as f:
                content = await f.read()
                cookies = json.loads(content)

            await self.context.add_cookies(cookies)

            return {
                "success": True,
                "cookies_loaded": len(cookies),
                "message": f"Loaded {len(cookies)} cookies",
            }
        except Exception as e:
            self.logger.error(f"Load cookies error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_cookies(self, urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get current cookies"""
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            cookies = await self.context.cookies(urls)
            return {"success": True, "cookies": cookies, "count": len(cookies)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_cookies(self) -> Dict[str, Any]:
        """Clear all cookies"""
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            await self.context.clear_cookies()
            return {"success": True, "message": "All cookies cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_cookie(self, cookie: Dict[str, Any]) -> Dict[str, Any]:
        """Set a single cookie"""
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            await self.context.add_cookies([cookie])
            return {
                "success": True,
                "cookie": cookie,
                "message": f'Cookie set: {cookie.get("name")}',
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # LOCAL STORAGE
    # =========================

    async def get_local_storage(self, key: str) -> Dict[str, Any]:
        """Get local storage item"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            value = await self.page.evaluate(f'localStorage.getItem("{key}")')
            return {"success": True, "key": key, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def set_local_storage(self, key: str, value: str) -> Dict[str, Any]:
        """Set local storage item"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.evaluate(f'localStorage.setItem("{key}", "{value}")')
            return {"success": True, "key": key, "value": value}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_local_storage(self) -> Dict[str, Any]:
        """Clear all local storage"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.evaluate("localStorage.clear()")
            return {"success": True, "message": "Local storage cleared"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # PAGE INFORMATION
    # =========================

    async def get_page_info(self) -> Dict[str, Any]:
        """
        Get current page information

        Returns:
            Dictionary with page info
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            url = self.page.url
            title = await self.page.title()
            content = await self.page.content()
            cookies = await self.context.cookies() if self.context else []

            return {
                "url": url,
                "title": title,
                "content_length": len(content),
                "cookies_count": len(cookies),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Get page info error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get page performance metrics"""
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            metrics = await self.page.evaluate("""() => {
                const perf = performance.getEntriesByType('navigation')[0];
                return {
                    dom_content_loaded: perf.domContentLoadedEventEnd,
                    load_complete: perf.loadEventEnd,
                    dom_interactive: perf.domInteractive,
                    first_paint: performance.getEntriesByType('paint')[0]?.startTime || 0,
                    first_contentful_paint: performance.getEntriesByType('paint')[1]?.startTime || 0
                };
            }""")

            return {"success": True, "metrics": metrics}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # NETWORK REQUESTS
    # =========================

    async def get_network_requests(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get captured network requests

        Args:
            limit: Maximum number of requests to return

        Returns:
            Dictionary with network requests
        """
        requests_data = []
        for req in self.requests[-limit:]:
            requests_data.append(
                {
                    "id": req.id,
                    "url": req.url,
                    "method": req.method,
                    "resource_type": req.resource_type,
                    "status": req.status,
                    "status_text": req.status_text,
                    "duration_ms": req.duration_ms,
                    "timestamp": req.timestamp.isoformat(),
                }
            )

        return {"success": True, "requests": requests_data, "total": len(self.requests)}

    async def get_console_messages(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get captured console messages

        Args:
            limit: Maximum number of messages to return

        Returns:
            Dictionary with console messages
        """
        return {
            "success": True,
            "messages": self.console_messages[-limit:],
            "total": len(self.console_messages),
        }

    # =========================
    # VIEWPORT & RESPONSIVE
    # =========================

    async def set_viewport(self, width: int, height: int) -> Dict[str, Any]:
        """
        Set viewport size

        Args:
            width: Viewport width
            height: Viewport height

        Returns:
            Dictionary with result
        """
        if not self.page:
            return {"success": False, "error": "Browser not started"}

        try:
            await self.page.set_viewport_size({"width": width, "height": height})
            self.viewport = {"width": width, "height": height}
            return {
                "success": True,
                "viewport": self.viewport,
                "message": f"Viewport set to {width}x{height}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def emulate_device(self, device_name: str) -> Dict[str, Any]:
        """
        Emulate a mobile device

        Args:
            device_name: Device name (e.g., 'iPhone 12', 'Pixel 5')

        Returns:
            Dictionary with result
        """
        if not self.context:
            return {"success": False, "error": "Browser context not available"}

        try:
            # Common device presets
            devices = {
                "iPhone 12": {
                    "width": 390,
                    "height": 844,
                    "is_mobile": True,
                    "has_touch": True,
                },
                "Pixel 5": {
                    "width": 393,
                    "height": 851,
                    "is_mobile": True,
                    "has_touch": True,
                },
                "iPad Pro": {
                    "width": 1024,
                    "height": 1366,
                    "is_mobile": True,
                    "has_touch": True,
                },
                "Desktop 1080p": {
                    "width": 1920,
                    "height": 1080,
                    "is_mobile": False,
                    "has_touch": False,
                },
            }

            if device_name not in devices:
                return {"success": False, "error": f"Unknown device: {device_name}"}

            device = devices[device_name]
            await self.set_viewport(device["width"], device["height"])

            return {
                "success": True,
                "device": device_name,
                "viewport": {"width": device["width"], "height": device["height"]},
                "message": f"Emulating {device_name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================
    # STATISTICS & UTILITIES
    # =========================

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        uptime = None
        if self.stats["start_time"]:
            start = datetime.fromisoformat(self.stats["start_time"])
            uptime = (datetime.now() - start).total_seconds()

        return {
            **self.stats,
            "browser_type": self.browser_type.value,
            "headless": self.headless,
            "viewport": self.viewport,
            "pages_in_history": len(self.page_history),
            "requests_captured": len(self.requests),
            "responses_captured": len(self.responses),
            "console_messages": len(self.console_messages),
            "downloads": len(self.downloads),
            "uptime_seconds": round(uptime, 2) if uptime else None,
            "is_running": self.browser is not None and self.browser.is_connected(),
        }

    async def stop(self) -> Dict[str, Any]:
        """
        Stop browser and clean up

        Returns:
            Dictionary with shutdown result
        """
        try:
            if self.page:
                await self.page.close()

            if self.context:
                await self.context.close()

            if self.browser:
                await self.browser.close()

            if self.playwright:
                await self.playwright.stop()

            self.logger.info("Browser stopped successfully")

            return {
                "success": True,
                "message": "Browser stopped successfully",
                "stats": self.get_stats(),
            }

        except Exception as e:
            self.logger.error(f"Stop error: {str(e)}")
            return {"success": False, "error": str(e)}


# =========================
# INTEGRATION WRAPPER
# =========================


class BrowserAgentWrapper:
    """
    Wrapper class to integrate BrowserAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.browser_agent = BrowserAgent(config)
        self.agent_type = "browser_automation"
        self.capabilities = [
            "navigate",
            "click",
            "type_text",
            "fill_form",
            "get_text",
            "get_html",
            "take_screenshot",
            "execute_script",
            "wait_for_element",
            "scroll_to",
            "hover",
            "select_option",
            "go_back_forward",
            "tab_management",
            "cookie_management",
            "drag_and_drop",
            "pdf_generation",
            "network_monitoring",
        ]
        self._initialized = False

    async def initialize(self, *args, **kwargs) -> bool:
        """Initialize the wrapper"""
        result = await self.browser_agent.start()
        self._initialized = result["success"]
        return self._initialized

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a browser automation request

        Request format:
        {
            'operation': 'navigate|click|type|fill|get_text|screenshot|script|...',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        # Ensure browser is started for operations that need it
        if operation not in ["start", "stop", "stats", "initialize"]:
            if not self.browser_agent.browser:
                start_result = await self.browser_agent.start()
                if not start_result["success"]:
                    return start_result

        if operation == "start":
            return await self.browser_agent.start(headless=request.get("headless"))

        elif operation == "stop":
            return await self.browser_agent.stop()

        elif operation == "initialize":
            success = await self.initialize()
            return {"success": success}

        elif operation == "navigate":
            wait_until = request.get("wait_until", "networkidle")
            return await self.browser_agent.navigate(
                url=request.get("url"),
                wait_until=WaitCondition(wait_until),
                timeout=request.get("timeout"),
                referer=request.get("referer"),
            )

        elif operation == "click":
            button = request.get("button", "left")
            modifiers = (
                [KeyModifier(m) for m in request.get("modifiers", [])]
                if request.get("modifiers")
                else None
            )
            return await self.browser_agent.click(
                selector=request.get("selector"),
                button=MouseButton(button),
                click_count=request.get("click_count", 1),
                force=request.get("force", False),
                modifiers=modifiers,
                position=request.get("position"),
                timeout=request.get("timeout"),
            )

        elif operation == "type":
            return await self.browser_agent.type_text(
                selector=request.get("selector"),
                text=request.get("text"),
                delay=request.get("delay", 100),
                clear_first=request.get("clear_first", True),
                timeout=request.get("timeout"),
            )

        elif operation == "fill_form":
            return await self.browser_agent.fill_form(
                form_data=request.get("form_data", {}),
                selector_prefix=request.get("selector_prefix", ""),
                timeout=request.get("timeout"),
            )

        elif operation == "get_text":
            return await self.browser_agent.get_text(
                selector=request.get("selector"),
                all_matches=request.get("all_matches", False),
                timeout=request.get("timeout"),
            )

        elif operation == "get_html":
            return await self.browser_agent.get_html(
                selector=request.get("selector"),
                outer_html=request.get("outer_html", True),
            )

        elif operation == "get_element_info":
            return await self.browser_agent.get_element_info(
                selector=request.get("selector"), timeout=request.get("timeout")
            )

        elif operation == "screenshot":
            return await self.browser_agent.take_screenshot(
                filename=request.get("filename"),
                full_page=request.get("full_page", False),
                selector=request.get("selector"),
                quality=request.get("quality"),
                type=request.get("type", "png"),
            )

        elif operation == "screenshot_base64":
            return await self.browser_agent.get_screenshot_base64(
                full_page=request.get("full_page", False),
                selector=request.get("selector"),
                quality=request.get("quality"),
            )

        elif operation == "pdf":
            return await self.browser_agent.generate_pdf(
                filename=request.get("filename"),
                format=request.get("format", "A4"),
                landscape=request.get("landscape", False),
                margin=request.get("margin"),
            )

        elif operation == "execute_script":
            return await self.browser_agent.execute_script(
                script=request.get("script"), *request.get("args", [])
            )

        elif operation == "wait":
            state = request.get("state", "visible")
            return await self.browser_agent.wait_for_element(
                selector=request.get("selector"),
                state=state,
                timeout=request.get("timeout"),
            )

        elif operation == "wait_timeout":
            return await self.browser_agent.wait_for_timeout(
                milliseconds=request.get("milliseconds", 1000)
            )

        elif operation == "wait_url":
            return await self.browser_agent.wait_for_url(
                url_pattern=request.get("url_pattern"), timeout=request.get("timeout")
            )

        elif operation == "scroll":
            return await self.browser_agent.scroll_to(
                selector=request.get("selector"),
                x=request.get("x", 0),
                y=request.get("y", 0),
                smooth=request.get("smooth", False),
            )

        elif operation == "scroll_bottom":
            return await self.browser_agent.scroll_to_bottom(
                smooth=request.get("smooth", False)
            )

        elif operation == "scroll_top":
            return await self.browser_agent.scroll_to_top(
                smooth=request.get("smooth", False)
            )

        elif operation == "hover":
            return await self.browser_agent.hover(
                selector=request.get("selector"), timeout=request.get("timeout")
            )

        elif operation == "select":
            return await self.browser_agent.select_option(
                selector=request.get("selector"),
                value=request.get("value"),
                timeout=request.get("timeout"),
            )

        elif operation == "drag_drop":
            return await self.browser_agent.drag_and_drop(
                source_selector=request.get("source_selector"),
                target_selector=request.get("target_selector"),
                timeout=request.get("timeout"),
            )

        elif operation == "check":
            return await self.browser_agent.check(
                selector=request.get("selector"),
                state=request.get("state", True),
                timeout=request.get("timeout"),
            )

        elif operation == "go_back":
            return await self.browser_agent.go_back()

        elif operation == "go_forward":
            return await self.browser_agent.go_forward()

        elif operation == "reload":
            return await self.browser_agent.reload()

        elif operation == "new_tab":
            return await self.browser_agent.new_tab(url=request.get("url"))

        elif operation == "close_tab":
            return await self.browser_agent.close_tab(
                page_index=request.get("page_index")
            )

        elif operation == "switch_tab":
            return await self.browser_agent.switch_tab(index=request.get("index"))

        elif operation == "get_tabs":
            return await self.browser_agent.get_tabs()

        elif operation == "save_cookies":
            return await self.browser_agent.save_cookies(
                filepath=request.get("filepath")
            )

        elif operation == "load_cookies":
            return await self.browser_agent.load_cookies(
                filepath=request.get("filepath")
            )

        elif operation == "get_cookies":
            return await self.browser_agent.get_cookies(urls=request.get("urls"))

        elif operation == "clear_cookies":
            return await self.browser_agent.clear_cookies()

        elif operation == "set_cookie":
            return await self.browser_agent.set_cookie(cookie=request.get("cookie"))

        elif operation == "get_local_storage":
            return await self.browser_agent.get_local_storage(key=request.get("key"))

        elif operation == "set_local_storage":
            return await self.browser_agent.set_local_storage(
                key=request.get("key"), value=request.get("value")
            )

        elif operation == "clear_local_storage":
            return await self.browser_agent.clear_local_storage()

        elif operation == "get_page_info":
            return await self.browser_agent.get_page_info()

        elif operation == "get_performance":
            return await self.browser_agent.get_performance_metrics()

        elif operation == "get_network":
            return await self.browser_agent.get_network_requests(
                limit=request.get("limit", 100)
            )

        elif operation == "get_console":
            return await self.browser_agent.get_console_messages(
                limit=request.get("limit", 100)
            )

        elif operation == "set_viewport":
            return await self.browser_agent.set_viewport(
                width=request.get("width"), height=request.get("height")
            )

        elif operation == "emulate_device":
            return await self.browser_agent.emulate_device(
                device_name=request.get("device_name")
            )

        elif operation == "block_resources":
            return await self.browser_agent.block_resources(
                resource_types=request.get("resource_types", [])
            )

        elif operation == "stats":
            return self.browser_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "BrowserAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.browser_agent.get_stats(),
            "browser_type": self.browser_agent.browser_type.value,
            "headless": self.browser_agent.headless,
            "initialized": self._initialized,
        }

    async def close(self):
        """Clean up resources"""
        await self.browser_agent.stop()
        self._initialized = False


# =========================
# TESTING
# =========================


async def test_browser_agent():
    """Test the browser agent functionality"""

    # Initialize agent
    agent = BrowserAgent({"headless": True})

    import logging

    logger = logging.getLogger(__name__)

    logger.info("=== Browser Agent Test ===\n")

    # Start browser
    logger.info("1. Starting browser...")
    result = await agent.start()
    logger.info("   %s", result.get("message"))

    # Navigate to a page
    logger.info("\n2. Navigating to example.com...")
    result = await agent.navigate("https://example.com")
    if result["success"]:
        logger.info("   Load time: %.2fs", result.get("load_time", 0.0))
        logger.info("   Title: %s", result.get("title"))

    # Get page info
    logger.info("\n3. Getting page info...")
    info = await agent.get_page_info()
    logger.info("   URL: %s", info.get("url"))
    logger.info("   Title: %s", info.get("title"))
    logger.info("   Content length: %s chars", info.get("content_length", 0))

    # Execute script
    logger.info("\n4. Executing JavaScript...")
    result = await agent.execute_script("return document.title")
    if result["success"]:
        logger.info("   Script result: %s", result.get("result"))

    # Get text
    logger.info("\n5. Getting text from page...")
    result = await agent.get_text("h1")
    if result["success"]:
        logger.info("   H1 text: %s", result.get("text", "N/A"))

    # Take screenshot
    logger.info("\n6. Taking screenshot...")
    result = await agent.take_screenshot("example_page")
    if result["success"]:
        logger.info("   Screenshot saved: %s", result.get("path"))
        logger.info("   Size: %s bytes", result.get("size_bytes"))

    # Get performance metrics
    logger.info("\n7. Performance metrics...")
    metrics = await agent.get_performance_metrics()
    if metrics["success"]:
        m = metrics["metrics"]
        logger.info("   DOM Content Loaded: %.0fms", m.get("dom_content_loaded", 0))
        logger.info("   Load Complete: %.0fms", m.get("load_complete", 0))

    # Get network requests
    logger.info("\n8. Network requests...")
    requests = await agent.get_network_requests(limit=5)
    logger.info("   Total requests: %s", requests.get("total"))
    for req in requests.get("requests", [])[:3]:
        logger.info(
            "     - %s %s... (%s)",
            req.get("method"),
            req.get("url")[:50],
            req.get("status"),
        )

    # Get stats
    logger.info("\n9. Agent statistics...")
    stats = agent.get_stats()
    logger.info("   Pages visited: %s", stats.get("pages_visited"))
    logger.info("   Actions performed: %s", stats.get("actions_performed"))
    logger.info("   Screenshots taken: %s", stats.get("screenshots_taken"))
    logger.info("   Total requests: %s", stats.get("total_requests"))

    # Stop browser
    logger.info("\n10. Stopping browser...")
    result = await agent.stop()
    logger.info("   %s", result.get("message"))

    logger.info("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_browser_agent())
