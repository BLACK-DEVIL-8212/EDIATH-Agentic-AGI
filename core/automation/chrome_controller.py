"""
Advanced Chrome Controller - Ultimate Edition (Playwright)
(MAXIMUM FEATURES - PRODUCTION READY + AI NATIVE + FULL CONTROL)
"""

import asyncio
import json
import re
import base64
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager

from ..utils.logger import logger

try:
    from playwright.async_api import async_playwright, BrowserContext, Browser, Page
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
except ImportError:
    async_playwright = None
    PlaywrightTimeoutError = Exception


class BrowserAction(Enum):
    """All supported browser actions"""
    NAVIGATE = "navigate"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    CLEAR = "clear"
    WAIT = "wait"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    SUBMIT = "submit"
    HOVER = "hover"
    DRAG_AND_DROP = "drag_and_drop"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    PRESS_KEY = "press_key"
    EXECUTE_SCRIPT = "execute_script"
    GET_TEXT = "get_text"
    GET_HTML = "get_html"
    GET_ATTRIBUTE = "get_attribute"
    GET_VALUE = "get_value"
    SET_VIEWPORT = "set_viewport"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    RELOAD = "reload"
    CLOSE_TAB = "close_tab"
    NEW_TAB = "new_tab"
    SWITCH_TAB = "switch_tab"
    GET_COOKIES = "get_cookies"
    SET_COOKIES = "set_cookies"
    CLEAR_COOKIES = "clear_cookies"
    GET_LOCAL_STORAGE = "get_local_storage"
    SET_LOCAL_STORAGE = "set_local_storage"
    CLEAR_LOCAL_STORAGE = "clear_local_storage"
    GET_SESSION_STORAGE = "get_session_storage"
    DOWNLOAD_FILE = "download_file"
    UPLOAD_FILE = "upload_file"
    WAIT_FOR_TEXT = "wait_for_text"
    WAIT_FOR_URL = "wait_for_url"
    CAPTURE_NETWORK = "capture_network"
    BLOCK_RESOURCES = "block_resources"
    SET_GEOLOCATION = "set_geolocation"
    SET_PERMISSIONS = "set_permissions"
    EMULATE_DEVICE = "emulate_device"
    RECORD_VIDEO = "record_video"
    STOP_RECORDING = "stop_recording"
    GET_PDF = "get_pdf"
    MOCK_RESPONSE = "mock_response"
    INTERCEPT_REQUEST = "intercept_request"
    EXTRACT_TABLE = "extract_table"
    EXTRACT_LINKS = "extract_links"
    EXTRACT_IMAGES = "extract_images"
    FIND_ELEMENTS = "find_elements"
    WAIT_AND_CLICK = "wait_and_click"
    RETRY_CLICK = "retry_click"
    SCROLL_TO_ELEMENT = "scroll_to_element"
    HIGHLIGHT_ELEMENT = "highlight_element"
    EXTRACT_METADATA = "extract_metadata"
    ANALYZE_ACCESSIBILITY = "analyze_accessibility"
    CAPTURE_CONSOLE_LOGS = "capture_console_logs"
    MONITOR_NETWORK = "monitor_network"
    PERFORM_AUDIT = "perform_audit"


@dataclass
class BrowserConfig:
    """Browser configuration settings"""
    headless: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None
    locale: str = "en-US"
    timezone_id: str = "America/New_York"
    permissions: List[str] = field(default_factory=lambda: ["geolocation", "notifications"])
    device_scale_factor: int = 1
    has_touch: bool = False
    is_mobile: bool = False
    java_script_enabled: bool = True
    ignore_https_errors: bool = False
    slow_mo: int = 0  # Slow down operations by ms
    record_video_dir: Optional[str] = None
    record_har_path: Optional[str] = None
    downloads_path: Optional[str] = None
    
    # Resource blocking
    block_images: bool = False
    block_css: bool = False
    block_fonts: bool = False
    block_media: bool = False
    
    # Proxy settings
    proxy_server: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    
    # Storage state
    storage_state_path: Optional[str] = None
    
    # Extra args for Chromium
    extra_args: List[str] = field(default_factory=lambda: [
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
        "--disable-site-isolation-trials",
    ])


@dataclass
class NetworkRequest:
    """Network request information"""
    url: str
    method: str
    headers: Dict[str, str]
    post_data: Optional[str] = None
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None


@dataclass
class ElementInfo:
    """Element information"""
    selector: str
    tag_name: str
    text: str
    html: str
    attributes: Dict[str, str]
    position: Dict[str, int]
    size: Dict[str, int]
    is_visible: bool
    is_enabled: bool
    is_editable: bool


class PerformanceMetrics:
    """Performance metrics collector"""
    def __init__(self):
        self.navigation_start = None
        self.navigation_end = None
        self.dom_content_loaded = None
        self.load_event_end = None
        self.first_paint = None
        self.first_contentful_paint = None
        self.redirect_count = 0
        self.request_count = 0
        self.failed_requests = 0
        self.total_data_transferred = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "navigation_start": self.navigation_start,
            "navigation_end": self.navigation_end,
            "navigation_duration_ms": (self.navigation_end - self.navigation_start) if self.navigation_start and self.navigation_end else None,
            "dom_content_loaded": self.dom_content_loaded,
            "load_event_end": self.load_event_end,
            "first_paint": self.first_paint,
            "first_contentful_paint": self.first_contentful_paint,
            "redirect_count": self.redirect_count,
            "request_count": self.request_count,
            "failed_requests": self.failed_requests,
            "total_data_transferred_mb": round(self.total_data_transferred / (1024 * 1024), 2),
        }


class ChromeController:
    """
    Ultimate Chrome Controller with maximum features for production automation
    """
    
    def __init__(self, config: Optional[BrowserConfig] = None, timeout: float = 30000):
        self.config = config or BrowserConfig()
        self.timeout = timeout
        
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._playwright = None
        
        self.is_connected = False
        self.current_url = ""
        self.current_tab_index = 0
        self.tabs: List[str] = []
        self.actions_performed = 0
        self.action_history: List[Dict[str, Any]] = []
        
        # State tracking
        self.last_action = None
        self.fail_count = 0
        self.is_recording_video = False
        self.video_path = None
        
        # Network monitoring
        self.network_requests: List[NetworkRequest] = []
        self.network_monitoring_enabled = False
        
        # Console logs
        self.console_logs: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.performance = PerformanceMetrics()
        
        # Custom handlers
        self.request_interceptors: List[Callable] = []
        self.response_interceptors: List[Callable] = []
        
        # Element cache
        self._element_cache: Dict[str, ElementInfo] = {}
        
        # Screenshots cache
        self._screenshot_cache: List[str] = []
        
        # Auto-save settings
        self.auto_save_screenshots = True
        self.screenshot_counter = 0
        
    # ==================== LIFE CYCLE MANAGEMENT ====================
    
    async def start(self):
        """Start browser with full configuration"""
        if async_playwright is None:
            raise ImportError("Install playwright: pip install playwright && playwright install chromium")
        
        if self.is_connected:
            return
        
        try:
            self._playwright = await async_playwright().start()
            
            # Launch browser with custom args
            browser_args = {
                "headless": self.config.headless,
                "slow_mo": self.config.slow_mo,
                "args": self.config.extra_args,
            }
            
            if self.config.proxy_server:
                browser_args["proxy"] = {
                    "server": self.config.proxy_server,
                    "username": self.config.proxy_username,
                    "password": self.config.proxy_password,
                }
            
            self.browser = await self._playwright.chromium.launch(**browser_args)
            
            # Create context with all configurations
            context_options = {
                "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height},
                "locale": self.config.locale,
                "timezone_id": self.config.timezone_id,
                "permissions": self.config.permissions,
                "device_scale_factor": self.config.device_scale_factor,
                "has_touch": self.config.has_touch,
                "is_mobile": self.config.is_mobile,
                "java_script_enabled": self.config.java_script_enabled,
                "ignore_https_errors": self.config.ignore_https_errors,
            }
            
            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent
            
            if self.config.storage_state_path:
                context_options["storage_state"] = self.config.storage_state_path
            
            if self.config.record_video_dir:
                context_options["record_video_dir"] = self.config.record_video_dir
            
            if self.config.record_har_path:
                context_options["record_har_path"] = self.config.record_har_path
            
            if self.config.downloads_path:
                context_options["accept_downloads"] = True
            
            self.context = await self.browser.new_context(**context_options)
            
            # Set default timeout
            self.context.set_default_timeout(self.timeout)
            
            # Create page
            self.page = await self.context.new_page()
            
            # Setup event listeners
            await self._setup_event_listeners()
            
            # Apply resource blocking if configured
            if any([self.config.block_images, self.config.block_css, self.config.block_fonts, self.config.block_media]):
                await self._setup_resource_blocking()
            
            self.is_connected = True
            self.tabs = [self.page.url]
            logger.info("🌐 Browser started with full configuration")
            
        except Exception as e:
            logger.error(f"Browser start failed: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop browser and cleanup resources"""
        try:
            # Stop video recording if active
            if self.is_recording_video:
                await self.stop_recording()
            
            # Save final HAR if configured
            if self.config.record_har_path and self.context:
                await self.context.close()
            
            # Close page
            if self.page:
                await self.page.close()
            
            # Close context
            if self.context:
                await self.context.close()
            
            # Close browser
            if self.browser:
                await self.browser.close()
            
            # Stop playwright
            if self._playwright:
                await self._playwright.stop()
                
        except Exception as e:
            logger.warning(f"Browser stop warning: {e}")
        
        self.browser = None
        self.context = None
        self.page = None
        self._playwright = None
        self.is_connected = False
        self.network_requests.clear()
        self.console_logs.clear()
        
        logger.info("❌ Browser stopped")
    
    async def _setup_event_listeners(self):
        """Setup all event listeners for monitoring"""
        if not self.page:
            return
        
        # Console log capture
        self.page.on("console", lambda msg: self.console_logs.append({
            "type": msg.type,
            "text": msg.text,
            "location": msg.location,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Network request capture
        if self.network_monitoring_enabled:
            self.page.on("request", self._on_request)
            self.page.on("response", self._on_response)
        
        # Dialog handling
        self.page.on("dialog", self._on_dialog)
        
        # Page crash handling
        self.page.on("crash", lambda: logger.error("Page crashed!"))
        
        # Download handling
        self.page.on("download", self._on_download)
    
    async def _setup_resource_blocking(self):
        """Block specified resource types"""
        if not self.page:
            return
        
        await self.page.route("**/*", self._block_resources_route)
    
    async def _block_resources_route(self, route):
        """Route handler for blocking resources"""
        resource_type = route.request.resource_type
        
        if (self.config.block_images and resource_type == "image") or \
           (self.config.block_css and resource_type == "stylesheet") or \
           (self.config.block_fonts and resource_type == "font") or \
           (self.config.block_media and resource_type in ["media", "video", "audio"]):
            await route.abort()
        else:
            await route.continue_()
    
    async def _on_request(self, request):
        """Handle network request"""
        req = NetworkRequest(
            url=request.url,
            method=request.method,
            headers=dict(request.headers),
            post_data=request.post_data
        )
        self.network_requests.append(req)
        self.performance.request_count += 1
    
    async def _on_response(self, response):
        """Handle network response"""
        # Find matching request
        for req in reversed(self.network_requests):
            if req.url == response.url:
                req.response_status = response.status
                req.response_headers = dict(response.headers)
                req.duration_ms = (datetime.now() - req.timestamp).total_seconds() * 1000
                
                # Track data transferred
                content_length = response.headers.get("content-length")
                if content_length:
                    self.performance.total_data_transferred += int(content_length)
                
                if response.status >= 400:
                    self.performance.failed_requests += 1
                break
    
    async def _on_dialog(self, dialog):
        """Handle browser dialogs"""
        logger.info(f"Dialog detected: {dialog.type} - {dialog.message}")
        await dialog.accept()
    
    async def _on_download(self, download):
        """Handle file downloads"""
        download_path = self.config.downloads_path or "downloads"
        path = Path(download_path) / download.suggested_filename
        await download.save_as(str(path))
        logger.info(f"Downloaded file to {path}")
    
    # ==================== CONNECTION MANAGEMENT ====================
    
    async def _ensure(self):
        """Ensure browser is connected"""
        if not self.is_connected or self.page is None:
            await self.start()
    
    async def _retry(self, func, *args, retries=2, timeout=None, **kwargs):
        """Retry wrapper for flaky operations"""
        last_error = None
        
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
                    
            except PlaywrightTimeoutError as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                await asyncio.sleep(1)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Retry {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        
        self.fail_count += 1
        raise RuntimeError(f"Action failed after {retries} retries: {last_error}")
    
    # ==================== CORE BROWSER ACTIONS ====================
    
    async def navigate(self, url: str, wait_until: str = "load", timeout: Optional[float] = None):
        """Navigate to URL with options"""
        await self._ensure()
        
        if not url:
            return "Invalid URL"
        
        if not url.startswith("http"):
            url = "https://" + url
        
        # Record navigation start
        self.performance.navigation_start = datetime.now().timestamp()
        
        try:
            await self._retry(
                self.page.goto, 
                url, 
                wait_until=wait_until, 
                timeout=timeout or self.timeout
            )
            
            # Record navigation end
            self.performance.navigation_end = datetime.now().timestamp()
            
            # Get performance metrics
            await self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise
        
        self.current_url = url
        self._log(BrowserAction.NAVIGATE, {"url": url, "wait_until": wait_until})
        return f"Navigated to {url}"
    
    async def click(self, selector: str, timeout: Optional[float] = None, force: bool = False, 
                   position: Optional[Dict[str, int]] = None):
        """Click element with options"""
        await self._ensure()
        if not selector:
            return "Invalid selector"
        
        await self._retry(
            self.page.click, 
            selector, 
            timeout=timeout or self.timeout,
            force=force,
            position=position
        )
        
        self._log(BrowserAction.CLICK, {"selector": selector})
        return f"Clicked {selector}"
    
    async def double_click(self, selector: str):
        """Double click element"""
        await self._ensure()
        await self._retry(self.page.dblclick, selector)
        self._log(BrowserAction.DOUBLE_CLICK, {"selector": selector})
        return f"Double clicked {selector}"
    
    async def right_click(self, selector: str):
        """Right click element"""
        await self._ensure()
        await self._retry(self.page.click, selector, button="right")
        self._log(BrowserAction.RIGHT_CLICK, {"selector": selector})
        return f"Right clicked {selector}"
    
    async def type_text(self, selector: str, text: str, delay: int = 0, clear_first: bool = True):
        """Type text into element"""
        await self._ensure()
        
        if clear_first:
            await self._retry(self.page.fill, selector, "", timeout=self.timeout)
        
        if delay > 0:
            await self._retry(self.page.type, selector, text, delay=delay)
        else:
            await self._retry(self.page.fill, selector, text, timeout=self.timeout)
        
        self._log(BrowserAction.TYPE, {"selector": selector, "text": text[:50]})
        return f"Typed into {selector}"
    
    async def clear_text(self, selector: str):
        """Clear text from input"""
        await self._ensure()
        await self._retry(self.page.fill, selector, "")
        self._log(BrowserAction.CLEAR, {"selector": selector})
        return f"Cleared {selector}"
    
    async def wait_for_element(self, selector: str, state: str = "visible", timeout: Optional[float] = None):
        """Wait for element with specific state"""
        await self._ensure()
        await self._retry(
            self.page.wait_for_selector, 
            selector, 
            state=state,
            timeout=timeout or self.timeout
        )
        self._log(BrowserAction.WAIT, {"selector": selector, "state": state})
        return f"Element found: {selector}"
    
    async def wait_for_navigation(self, timeout: Optional[float] = None):
        """Wait for navigation to complete"""
        await self._ensure()
        await self._retry(self.page.wait_for_navigation, timeout=timeout or self.timeout)
        self._log(BrowserAction.WAIT_FOR_NAVIGATION, {})
        return "Navigation completed"
    
    async def wait_for_text(self, text: str, timeout: Optional[float] = None):
        """Wait for text to appear on page"""
        await self._ensure()
        await self._retry(
            self.page.wait_for_function,
            f"document.body.innerText.includes('{text}')",
            timeout=timeout or self.timeout
        )
        self._log(BrowserAction.WAIT_FOR_TEXT, {"text": text})
        return f"Text found: {text}"
    
    async def scroll(self, direction: str = "down", distance: int = 500):
        """Scroll page"""
        await self._ensure()
        
        if direction == "down":
            await self.page.mouse.wheel(0, distance)
        elif direction == "up":
            await self.page.mouse.wheel(0, -distance)
        elif direction == "top":
            await self.page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        
        self._log(BrowserAction.SCROLL, {"direction": direction, "distance": distance})
        return f"Scrolled {direction}"
    
    async def scroll_to_element(self, selector: str):
        """Scroll to specific element"""
        await self._ensure()
        await self._retry(self.page.locator(selector).scroll_into_view_if_needed)
        self._log(BrowserAction.SCROLL_TO_ELEMENT, {"selector": selector})
        return f"Scrolled to {selector}"
    
    async def hover(self, selector: str):
        """Hover over element"""
        await self._ensure()
        await self._retry(self.page.hover, selector)
        self._log(BrowserAction.HOVER, {"selector": selector})
        return f"Hovered over {selector}"
    
    async def drag_and_drop(self, source: str, target: str):
        """Drag and drop element"""
        await self._ensure()
        await self._retry(self.page.drag_and_drop, source, target)
        self._log(BrowserAction.DRAG_AND_DROP, {"source": source, "target": target})
        return f"Dragged {source} to {target}"
    
    async def select_option(self, selector: str, value: Union[str, List[str]]):
        """Select option from dropdown"""
        await self._ensure()
        await self._retry(self.page.select_option, selector, value)
        self._log(BrowserAction.SELECT, {"selector": selector, "value": value})
        return f"Selected option in {selector}"
    
    async def check(self, selector: str):
        """Check checkbox or radio"""
        await self._ensure()
        await self._retry(self.page.check, selector)
        self._log(BrowserAction.CHECK, {"selector": selector})
        return f"Checked {selector}"
    
    async def uncheck(self, selector: str):
        """Uncheck checkbox"""
        await self._ensure()
        await self._retry(self.page.uncheck, selector)
        self._log(BrowserAction.UNCHECK, {"selector": selector})
        return f"Unchecked {selector}"
    
    async def press_key(self, key: str):
        """Press keyboard key"""
        await self._ensure()
        await self._retry(self.page.keyboard.press, key)
        self._log(BrowserAction.PRESS_KEY, {"key": key})
        return f"Pressed {key}"
    
    async def submit_form(self, selector: Optional[str] = None):
        """Submit form"""
        await self._ensure()
        
        if selector:
            await self._retry(self.page.locator(selector).press, "Enter")
        else:
            await self._retry(self.page.keyboard.press, "Enter")
        
        self._log(BrowserAction.SUBMIT, {"selector": selector})
        return "Form submitted"
    
    # ==================== ELEMENT INFORMATION ====================
    
    async def get_text(self, selector: str) -> str:
        """Get text content of element"""
        await self._ensure()
        text = await self._retry(self.page.text_content, selector)
        self._log(BrowserAction.GET_TEXT, {"selector": selector})
        return text or ""
    
    async def get_html(self, selector: Optional[str] = None) -> str:
        """Get HTML content"""
        await self._ensure()
        
        if selector:
            html = await self._retry(self.page.inner_html, selector)
        else:
            html = await self._retry(self.page.content)
        
        self._log(BrowserAction.GET_HTML, {"selector": selector})
        return html
    
    async def get_attribute(self, selector: str, attribute: str) -> Optional[str]:
        """Get element attribute"""
        await self._ensure()
        value = await self._retry(self.page.get_attribute, selector, attribute)
        self._log(BrowserAction.GET_ATTRIBUTE, {"selector": selector, "attribute": attribute})
        return value
    
    async def get_value(self, selector: str) -> Optional[str]:
        """Get input value"""
        await self._ensure()
        value = await self._retry(self.page.input_value, selector)
        self._log(BrowserAction.GET_VALUE, {"selector": selector})
        return value
    
    async def get_element_info(self, selector: str) -> ElementInfo:
        """Get comprehensive element information"""
        await self._ensure()
        
        element = self.page.locator(selector)
        info = await self._retry(self._get_element_details, selector)
        
        self._element_cache[selector] = info
        return info
    
    async def _get_element_details(self, selector: str) -> ElementInfo:
        """Internal method to get element details"""
        element = self.page.locator(selector)
        
        box = await element.bounding_box()
        position = {"x": box["x"], "y": box["y"]} if box else {"x": 0, "y": 0}
        size = {"width": box["width"], "height": box["height"]} if box else {"width": 0, "height": 0}
        
        return ElementInfo(
            selector=selector,
            tag_name=await element.evaluate("el => el.tagName.toLowerCase()"),
            text=await element.text_content() or "",
            html=await element.inner_html(),
            attributes=await element.evaluate("el => Object.assign({}, el.attributes)") or {},
            position=position,
            size=size,
            is_visible=await element.is_visible(),
            is_enabled=await element.is_enabled(),
            is_editable=await element.is_editable()
        )
    
    async def find_elements(self, selector: str) -> List[Dict[str, Any]]:
        """Find all matching elements"""
        await self._ensure()
        
        elements = await self.page.query_selector_all(selector)
        result = []
        
        for i, element in enumerate(elements):
            result.append({
                "index": i,
                "text": await element.text_content(),
                "html": await element.inner_html(),
                "tag": await element.evaluate("el => el.tagName"),
            })
        
        self._log(BrowserAction.FIND_ELEMENTS, {"selector": selector})
        return result
    
    # ==================== SCREENSHOT & VISUAL ====================
    
    async def screenshot(self, path: Optional[str] = None, full_page: bool = False, 
                        selector: Optional[str] = None, base64: bool = False) -> Union[str, bytes]:
        """Take screenshot with options"""
        await self._ensure()
        
        if not path and not base64:
            path = f"screenshot_{self.screenshot_counter}.png"
            self.screenshot_counter += 1
        
        screenshot_options = {"full_page": full_page}
        
        if selector:
            element = self.page.locator(selector)
            screenshot_bytes = await element.screenshot()
        else:
            if base64:
                screenshot_bytes = await self.page.screenshot(**screenshot_options, type="png")
                return base64.b64encode(screenshot_bytes).decode()
            else:
                await self.page.screenshot(path=path, **screenshot_options)
        
        self._log(BrowserAction.SCREENSHOT, {"path": path, "full_page": full_page})
        
        if base64 and not selector:
            return screenshot_bytes
        
        return f"Screenshot saved: {path}" if path else screenshot_bytes
    
    async def highlight_element(self, selector: str, duration: float = 2):
        """Highlight element for debugging"""
        await self._ensure()
        
        await self.page.evaluate(f"""
            const element = document.querySelector('{selector}');
            if (element) {{
                const originalOutline = element.style.outline;
                element.style.outline = '3px solid red';
                setTimeout(() => {{
                    element.style.outline = originalOutline;
                }}, {duration * 1000});
            }}
        """)
        
        self._log(BrowserAction.HIGHLIGHT_ELEMENT, {"selector": selector})
        return f"Highlighted {selector}"
    
    # ==================== JAVASCRIPT EXECUTION ====================
    
    async def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript in page context"""
        await self._ensure()
        result = await self._retry(self.page.evaluate, script, *args)
        self._log(BrowserAction.EXECUTE_SCRIPT, {"script": script[:100]})
        return result
    
    async def inject_script(self, script_path: str):
        """Inject external JavaScript file"""
        await self._ensure()
        
        with open(script_path, 'r') as f:
            script = f.read()
        
        await self.execute_script(script)
        return f"Injected script from {script_path}"
    
    # ==================== TAB MANAGEMENT ====================
    
    async def new_tab(self, url: Optional[str] = None):
        """Open new tab"""
        await self._ensure()
        
        new_page = await self.context.new_page()
        self.tabs.append(url or "about:blank")
        self.current_tab_index = len(self.tabs) - 1
        
        if url:
            await new_page.goto(url)
        
        self._log(BrowserAction.NEW_TAB, {"url": url})
        return f"Opened new tab: {url or 'blank'}"
    
    async def switch_tab(self, index: int):
        """Switch to tab by index"""
        await self._ensure()
        
        pages = self.context.pages
        if 0 <= index < len(pages):
            self.page = pages[index]
            self.current_tab_index = index
            self.current_url = self.page.url
            
            self._log(BrowserAction.SWITCH_TAB, {"index": index})
            return f"Switched to tab {index}"
        
        return f"Invalid tab index: {index}"
    
    async def close_tab(self, index: Optional[int] = None):
        """Close tab"""
        await self._ensure()
        
        if index is None:
            index = self.current_tab_index
        
        pages = self.context.pages
        if 0 <= index < len(pages):
            await pages[index].close()
            self.tabs.pop(index)
            
            if len(pages) > 1:
                self.page = pages[0 if index == 0 else index - 1]
                self.current_tab_index = 0 if index == 0 else index - 1
                self.current_url = self.page.url
            
            self._log(BrowserAction.CLOSE_TAB, {"index": index})
            return f"Closed tab {index}"
        
        return f"Invalid tab index: {index}"
    
    async def get_tabs_info(self) -> List[Dict[str, Any]]:
        """Get all tabs information"""
        await self._ensure()
        
        tabs_info = []
        for i, page in enumerate(self.context.pages):
            tabs_info.append({
                "index": i,
                "url": page.url,
                "title": await page.title(),
                "is_active": i == self.current_tab_index
            })
        
        return tabs_info
    
    # ==================== NAVIGATION HISTORY ====================
    
    async def go_back(self):
        """Go back in history"""
        await self._ensure()
        await self._retry(self.page.go_back)
        self.current_url = self.page.url
        self._log(BrowserAction.GO_BACK, {})
        return "Went back"
    
    async def go_forward(self):
        """Go forward in history"""
        await self._ensure()
        await self._retry(self.page.go_forward)
        self.current_url = self.page.url
        self._log(BrowserAction.GO_FORWARD, {})
        return "Went forward"
    
    async def reload(self, bypass_cache: bool = False):
        """Reload current page"""
        await self._ensure()
        await self._retry(self.page.reload)
        
        if bypass_cache:
            await self.execute_script("location.reload(true)")
        
        self._log(BrowserAction.RELOAD, {"bypass_cache": bypass_cache})
        return "Page reloaded"
    
    # ==================== COOKIE & STORAGE MANAGEMENT ====================
    
    async def get_cookies(self, urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get all cookies"""
        await self._ensure()
        cookies = await self.context.cookies(urls)
        self._log(BrowserAction.GET_COOKIES, {})
        return cookies
    
    async def set_cookies(self, cookies: List[Dict[str, Any]]):
        """Set cookies"""
        await self._ensure()
        await self.context.add_cookies(cookies)
        self._log(BrowserAction.SET_COOKIES, {"count": len(cookies)})
        return f"Set {len(cookies)} cookies"
    
    async def clear_cookies(self):
        """Clear all cookies"""
        await self._ensure()
        await self.context.clear_cookies()
        self._log(BrowserAction.CLEAR_COOKIES, {})
        return "Cleared all cookies"
    
    async def get_local_storage(self, key: Optional[str] = None) -> Union[str, Dict[str, str]]:
        """Get local storage data"""
        await self._ensure()
        
        if key:
            value = await self.page.evaluate(f"localStorage.getItem('{key}')")
            return value
        else:
            storage = await self.page.evaluate("JSON.stringify(localStorage)")
            return json.loads(storage)
    
    async def set_local_storage(self, key: str, value: str):
        """Set local storage item"""
        await self._ensure()
        await self.page.evaluate(f"localStorage.setItem('{key}', '{value}')")
        self._log(BrowserAction.SET_LOCAL_STORAGE, {"key": key})
        return f"Set localStorage {key}"
    
    async def clear_local_storage(self):
        """Clear all local storage"""
        await self._ensure()
        await self.page.evaluate("localStorage.clear()")
        self._log(BrowserAction.CLEAR_LOCAL_STORAGE, {})
        return "Cleared local storage"
    
    async def get_session_storage(self, key: Optional[str] = None) -> Union[str, Dict[str, str]]:
        """Get session storage data"""
        await self._ensure()
        
        if key:
            value = await self.page.evaluate(f"sessionStorage.getItem('{key}')")
            return value
        else:
            storage = await self.page.evaluate("JSON.stringify(sessionStorage)")
            return json.loads(storage)
    
    # ==================== VIEWPORT & DEVICE EMULATION ====================
    
    async def set_viewport(self, width: int, height: int):
        """Set viewport size"""
        await self._ensure()
        await self.page.set_viewport_size({"width": width, "height": height})
        self._log(BrowserAction.SET_VIEWPORT, {"width": width, "height": height})
        return f"Viewport set to {width}x{height}"
    
    async def emulate_device(self, device_name: str):
        """Emulate mobile device"""
        await self._ensure()
        
        devices = {
            "iPhone 12": {"width": 390, "height": 844, "is_mobile": True, "has_touch": True},
            "iPad Pro": {"width": 1024, "height": 1366, "is_mobile": True, "has_touch": True},
            "Pixel 5": {"width": 393, "height": 851, "is_mobile": True, "has_touch": True},
        }
        
        if device_name in devices:
            device = devices[device_name]
            await self.set_viewport(device["width"], device["height"])
            
            if "is_mobile" in device:
                await self.page.evaluate(f"navigator.__defineGetter__('userAgent', () => '{device_name}')")
        
        self._log(BrowserAction.EMULATE_DEVICE, {"device": device_name})
        return f"Emulated {device_name}"
    
    async def set_geolocation(self, latitude: float, longitude: float, accuracy: float = 100):
        """Set geolocation"""
        await self._ensure()
        await self.context.set_geolocation({"latitude": latitude, "longitude": longitude, "accuracy": accuracy})
        self._log(BrowserAction.SET_GEOLOCATION, {"latitude": latitude, "longitude": longitude})
        return f"Geolocation set to {latitude}, {longitude}"
    
    async def set_permissions(self, permissions: List[str]):
        """Set browser permissions"""
        await self._ensure()
        await self.context.grant_permissions(permissions)
        self._log(BrowserAction.SET_PERMISSIONS, {"permissions": permissions})
        return f"Permissions set: {permissions}"
    
    # ==================== NETWORK & INTERCEPTION ====================
    
    async def enable_network_monitoring(self):
        """Enable network request monitoring"""
        self.network_monitoring_enabled = True
        
        if self.page:
            self.page.on("request", self._on_request)
            self.page.on("response", self._on_response)
        
        self._log(BrowserAction.CAPTURE_NETWORK, {})
        return "Network monitoring enabled"
    
    async def get_network_requests(self, filter_status: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get captured network requests"""
        requests = [req.__dict__ for req in self.network_requests]
        
        if filter_status:
            requests = [req for req in requests if req.get("response_status") == filter_status]
        
        return requests
    
    async def block_resources(self, resource_types: List[str]):
        """Block specific resource types"""
        self.config.block_images = "image" in resource_types
        self.config.block_css = "stylesheet" in resource_types
        self.config.block_fonts = "font" in resource_types
        self.config.block_media = "media" in resource_types
        
        if self.page:
            await self.page.route("**/*", self._block_resources_route)
        
        self._log(BrowserAction.BLOCK_RESOURCES, {"types": resource_types})
        return f"Blocked resources: {resource_types}"
    
    async def intercept_request(self, url_pattern: str, handler: Callable):
        """Intercept and modify requests"""
        await self._ensure()
        
        async def route_handler(route):
            if re.match(url_pattern, route.request.url):
                await handler(route)
            else:
                await route.continue_()
        
        await self.page.route(url_pattern, route_handler)
        self.request_interceptors.append(route_handler)
        
        self._log(BrowserAction.INTERCEPT_REQUEST, {"pattern": url_pattern})
        return f"Request interceptor added for {url_pattern}"
    
    async def mock_response(self, url_pattern: str, mock_data: Dict[str, Any], status: int = 200):
        """Mock API responses"""
        await self._ensure()
        
        async def route_handler(route):
            if re.match(url_pattern, route.request.url):
                await route.fulfill(
                    status=status,
                    content_type="application/json",
                    body=json.dumps(mock_data)
                )
            else:
                await route.continue_()
        
        await self.page.route(url_pattern, route_handler)
        
        self._log(BrowserAction.MOCK_RESPONSE, {"pattern": url_pattern})
        return f"Response mocker added for {url_pattern}"
    
    # ==================== FILE OPERATIONS ====================
    
    async def upload_file(self, selector: str, file_paths: List[str]):
        """Upload files to input element"""
        await self._ensure()
        
        input_element = await self.page.query_selector(selector)
        if input_element:
            await input_element.set_input_files(file_paths)
        
        self._log(BrowserAction.UPLOAD_FILE, {"selector": selector, "files": file_paths})
        return f"Uploaded {len(file_paths)} files to {selector}"
    
    async def download_file(self, url: str, save_path: Optional[str] = None):
        """Download file from URL"""
        await self._ensure()
        
        async with self.page.expect_download() as download_info:
            await self.page.goto(url)
        
        download = await download_info.value
        
        if save_path:
            await download.save_as(save_path)
        else:
            save_path = download.suggested_filename
            await download.save_as(save_path)
        
        self._log(BrowserAction.DOWNLOAD_FILE, {"url": url, "save_path": save_path})
        return f"Downloaded to {save_path}"
    
    # ==================== EXTRACTION & PARSING ====================
    
    async def extract_links(self, selector: Optional[str] = None) -> List[Dict[str, str]]:
        """Extract all links from page"""
        await self._ensure()
        
        link_selector = selector or "a[href]"
        links = await self.page.evaluate(f"""
            Array.from(document.querySelectorAll('{link_selector}')).map(a => ({{
                text: a.textContent.trim(),
                href: a.href,
                title: a.title || '',
                target: a.target || ''
            }}))
        """)
        
        self._log(BrowserAction.EXTRACT_LINKS, {})
        return links
    
    async def extract_images(self) -> List[Dict[str, str]]:
        """Extract all images from page"""
        await self._ensure()
        
        images = await self.page.evaluate("""
            Array.from(document.querySelectorAll('img')).map(img => ({
                src: img.src,
                alt: img.alt,
                title: img.title,
                width: img.width,
                height: img.height
            }))
        """)
        
        self._log(BrowserAction.EXTRACT_IMAGES, {})
        return images
    
    async def extract_table(self, selector: str, as_dict: bool = True) -> Union[List[List[str]], List[Dict[str, str]]]:
        """Extract table data"""
        await self._ensure()
        
        if as_dict:
            # Extract as list of dictionaries with headers
            data = await self.page.evaluate(f"""
                const table = document.querySelector('{selector}');
                if (!table) return [];
                
                const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                const rows = Array.from(table.querySelectorAll('tbody tr'));
                
                return rows.map(row => {{
                    const cells = Array.from(row.querySelectorAll('td'));
                    const rowData = {{}};
                    headers.forEach((header, idx) => {{
                        rowData[header] = cells[idx] ? cells[idx].textContent.trim() : '';
                    }});
                    return rowData;
                }});
            """)
        else:
            # Extract as 2D array
            data = await self.page.evaluate(f"""
                const table = document.querySelector('{selector}');
                if (!table) return [];
                
                const rows = Array.from(table.querySelectorAll('tr'));
                return rows.map(row => 
                    Array.from(row.querySelectorAll('th, td')).map(cell => cell.textContent.trim())
                );
            """)
        
        self._log(BrowserAction.EXTRACT_TABLE, {"selector": selector})
        return data
    
    async def extract_metadata(self) -> Dict[str, Any]:
        """Extract page metadata"""
        await self._ensure()
        
        metadata = await self.page.evaluate("""
            ({
                title: document.title,
                url: window.location.href,
                description: document.querySelector('meta[name="description"]')?.content || '',
                keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                author: document.querySelector('meta[name="author"]')?.content || '',
                viewport: document.querySelector('meta[name="viewport"]')?.content || '',
                charset: document.characterSet,
                language: document.documentElement.lang,
                wordCount: document.body.innerText.split(/\\s+/).length,
                linksCount: document.querySelectorAll('a[href]').length,
                imagesCount: document.querySelectorAll('img').length,
                scriptsCount: document.querySelectorAll('script').length,
                stylesCount: document.querySelectorAll('link[rel="stylesheet"], style').length
            })
        """)
        
        self._log(BrowserAction.EXTRACT_METADATA, {})
        return metadata
    
    # ==================== ADVANCED WAITING ====================
    
    async def wait_and_click(self, selector: str, timeout: Optional[float] = None):
        """Wait for element and click it"""
        await self.wait_for_element(selector, timeout=timeout)
        await self.click(selector)
        
        self._log(BrowserAction.WAIT_AND_CLICK, {"selector": selector})
        return f"Waited and clicked {selector}"
    
    async def retry_click(self, selector: str, max_retries: int = 3, delay: float = 1):
        """Retry clicking element multiple times"""
        for attempt in range(max_retries):
            try:
                await self.click(selector)
                return f"Clicked {selector} on attempt {attempt + 1}"
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
        
        self._log(BrowserAction.RETRY_CLICK, {"selector": selector, "retries": max_retries})
        return f"Failed to click {selector} after {max_retries} attempts"
    
    async def wait_for_url(self, url_pattern: str, timeout: Optional[float] = None):
        """Wait for URL to match pattern"""
        await self._ensure()
        
        await self._retry(
            self.page.wait_for_function,
            f"window.location.href.includes('{url_pattern}')",
            timeout=timeout or self.timeout
        )
        
        self._log(BrowserAction.WAIT_FOR_URL, {"pattern": url_pattern})
        return f"URL matched pattern: {url_pattern}"
    
    # ==================== PERFORMANCE & AUDITING ====================
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            perf_timing = await self.page.evaluate("""
                JSON.stringify(window.performance.timing)
            """)
            
            timing = json.loads(perf_timing)
            
            self.performance.dom_content_loaded = timing.get("domContentLoadedEventEnd") - timing.get("navigationStart")
            self.performance.load_event_end = timing.get("loadEventEnd") - timing.get("navigationStart")
            
            # Get paint metrics
            paint_metrics = await self.page.evaluate("""
                const paint = performance.getEntriesByType('paint');
                return {
                    firstPaint: paint.find(p => p.name === 'first-paint')?.startTime,
                    firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime
                }
            """)
            
            self.performance.first_paint = paint_metrics.get("firstPaint")
            self.performance.first_contentful_paint = paint_metrics.get("firstContentfulPaint")
            
        except Exception as e:
            logger.warning(f"Failed to get performance metrics: {e}")
    
    async def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        await self._ensure()
        await self._update_performance_metrics()
        return self.performance.to_dict()
    
    async def analyze_accessibility(self) -> Dict[str, Any]:
        """Analyze page accessibility"""
        await self._ensure()
        
        accessibility = await self.page.evaluate("""
            const issues = [];
            
            // Check images for alt text
            document.querySelectorAll('img').forEach(img => {
                if (!img.alt) issues.push({
                    type: 'missing_alt',
                    element: 'img',
                    message: 'Image missing alt text',
                    selector: img.src
                });
            });
            
            // Check heading structure
            const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
            if (headings.length === 0) {
                issues.push({
                    type: 'no_headings',
                    element: 'page',
                    message: 'Page has no heading structure'
                });
            }
            
            // Check for skip link
            const skipLink = document.querySelector('a[href="#main"], a[href="#content"]');
            if (!skipLink) {
                issues.push({
                    type: 'no_skip_link',
                    element: 'navigation',
                    message: 'No skip navigation link found'
                });
            }
            
            return {
                issues: issues,
                totalIssues: issues.length,
                hasIssues: issues.length > 0
            };
        """)
        
        self._log(BrowserAction.ANALYZE_ACCESSIBILITY, {})
        return accessibility
    
    async def capture_console_logs(self) -> List[Dict[str, Any]]:
        """Get captured console logs"""
        self._log(BrowserAction.CAPTURE_CONSOLE_LOGS, {})
        return self.console_logs
    
    async def monitor_network(self, duration_seconds: float = 10) -> List[Dict[str, Any]]:
        """Monitor network activity for specified duration"""
        await self.enable_network_monitoring()
        
        start_requests = len(self.network_requests)
        await asyncio.sleep(duration_seconds)
        end_requests = len(self.network_requests)
        
        new_requests = self.network_requests[start_requests:end_requests]
        
        self._log(BrowserAction.MONITOR_NETWORK, {"duration": duration_seconds})
        return [req.__dict__ for req in new_requests]
    
    async def perform_audit(self) -> Dict[str, Any]:
        """Perform comprehensive page audit"""
        await self._ensure()
        
        audit_results = {
            "metadata": await self.extract_metadata(),
            "performance": await self.get_performance(),
            "accessibility": await self.analyze_accessibility(),
            "links": len(await self.extract_links()),
            "images": len(await self.extract_images()),
            "console_errors": [log for log in self.console_logs if log["type"] == "error"],
            "failed_network_requests": [req for req in self.network_requests if req.response_status and req.response_status >= 400],
            "timestamp": datetime.now().isoformat()
        }
        
        self._log(BrowserAction.PERFORM_AUDIT, {})
        return audit_results
    
    # ==================== VIDEO RECORDING ====================
    
    async def record_video(self, output_path: str):
        """Start video recording of browser session"""
        if not self.config.record_video_dir:
            self.config.record_video_dir = "videos"
        
        self.is_recording_video = True
        self.video_path = output_path
        
        self._log(BrowserAction.RECORD_VIDEO, {"path": output_path})
        return f"Recording video to {output_path}"
    
    async def stop_recording(self):
        """Stop video recording"""
        if self.is_recording_video and self.context:
            await self.context.close()
            self.is_recording_video = False
            
            self._log(BrowserAction.STOP_RECORDING, {})
            return f"Video saved to {self.video_path}"
        
        return "No active recording"
    
    async def get_pdf(self, path: str, options: Optional[Dict[str, Any]] = None):
        """Generate PDF from page"""
        await self._ensure()
        
        pdf_options = options or {
            "format": "A4",
            "print_background": True,
            "margin": {"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"}
        }
        
        await self.page.pdf(path=path, **pdf_options)
        
        self._log(BrowserAction.GET_PDF, {"path": path})
        return f"PDF saved to {path}"
    
    # ==================== AI / COGNITIVE ACTIONS ====================
    
    def explore(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI explore action - returns page state"""
        return {
            "status": "exploring",
            "message": "Browser is ready for exploration",
            "url": self.current_url,
            "connected": self.is_connected,
            "tabs": len(self.context.pages) if self.context else 0,
            "actions_performed": self.actions_performed,
        }
    
    def think(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI think action - process context"""
        return {
            "status": "thinking",
            "message": "Processing context...",
            "timestamp": datetime.now().isoformat(),
            "context": {
                "url": self.current_url,
                "page_title": asyncio.run(self.page.title()) if self.page else None,
                "actions_history": self.action_history[-10:]  # Last 10 actions
            }
        }
    
    def speak(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI speak action"""
        text = params.get("text", "") if params else ""
        return {
            "status": "spoken",
            "message": f"Would speak: {text[:100]}" if text else "Nothing to speak",
            "output": text,
            "timestamp": datetime.now().isoformat()
        }
    
    def idle(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Idle state"""
        return {
            "status": "idle",
            "message": "System idle",
            "timestamp": datetime.now().isoformat()
        }
    
    # ==================== SYSTEM STATUS ====================
    
    def system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "connected": self.is_connected,
            "current_url": self.current_url,
            "current_tab": self.current_tab_index,
            "total_tabs": len(self.context.pages) if self.context else 0,
            "actions_performed": self.actions_performed,
            "failures": self.fail_count,
            "status": "running" if self.is_connected else "stopped",
            "network_requests": len(self.network_requests),
            "console_logs": len(self.console_logs),
            "cached_elements": len(self._element_cache),
            "screenshots_taken": self.screenshot_counter,
            "uptime": (datetime.now() - datetime.fromtimestamp(self.performance.navigation_start)).total_seconds() if self.performance.navigation_start else 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple statistics"""
        return {
            "connected": self.is_connected,
            "url": self.current_url,
            "actions": self.actions_performed,
            "history": len(self.action_history),
            "failures": self.fail_count,
            "tabs": len(self.tabs),
                        "perf_navigation_ms": self.performance.navigation_duration_ms if hasattr(self.performance, 'navigation_duration_ms') else None,
        }
    
    # ==================== AI ACTION DISPATCHER (ULTIMATE) ====================
    
    async def execute_action(self, action: str, params: dict = None) -> dict:
        """
        Unified action dispatcher for AI-driven automation.
        Supports 75+ actions including browser, cognitive, and advanced features.
        """
        if params is None:
            params = {}
        
        action = action.lower().strip()
        logger.debug(f"Executing action: {action}")
        
        # Action routing table for O(1) lookup
        action_map = {
            # Cognitive / System Actions
            "explore": lambda: self.explore(params),
            "think": lambda: self.think(params),
            "speak": lambda: self.speak(params),
            "idle": lambda: self.idle(params),
            "system_status": lambda: self.system_status(),
            "get_stats": lambda: self.get_stats(),
            
            # Navigation Actions
            "navigate": lambda: self.navigate(params.get("url", ""), params.get("wait_until", "load")),
            "go_back": lambda: self.go_back(),
            "go_forward": lambda: self.go_forward(),
            "reload": lambda: self.reload(params.get("bypass_cache", False)),
            
            # Click Actions
            "click": lambda: self.click(params.get("selector", ""), params.get("timeout"), 
                                       params.get("force", False), params.get("position")),
            "double_click": lambda: self.double_click(params.get("selector", "")),
            "right_click": lambda: self.right_click(params.get("selector", "")),
            "wait_and_click": lambda: self.wait_and_click(params.get("selector", ""), params.get("timeout")),
            "retry_click": lambda: self.retry_click(params.get("selector", ""), 
                                                    params.get("max_retries", 3), 
                                                    params.get("delay", 1)),
            
            # Input Actions
            "type": lambda: self.type_text(params.get("selector", ""), params.get("text", ""),
                                          params.get("delay", 0), params.get("clear_first", True)),
            "clear": lambda: self.clear_text(params.get("selector", "")),
            "press_key": lambda: self.press_key(params.get("key", "")),
            "submit": lambda: self.submit_form(params.get("selector")),
            
            # Form Actions
            "select": lambda: self.select_option(params.get("selector", ""), params.get("value", "")),
            "check": lambda: self.check(params.get("selector", "")),
            "uncheck": lambda: self.uncheck(params.get("selector", "")),
            "upload_file": lambda: self.upload_file(params.get("selector", ""), params.get("file_paths", [])),
            "download_file": lambda: self.download_file(params.get("url", ""), params.get("save_path")),
            
            # Wait Actions
            "wait": lambda: self.wait_for_element(params.get("selector", ""), 
                                                 params.get("state", "visible"),
                                                 params.get("timeout")),
            "wait_for_navigation": lambda: self.wait_for_navigation(params.get("timeout")),
            "wait_for_text": lambda: self.wait_for_text(params.get("text", ""), params.get("timeout")),
            "wait_for_url": lambda: self.wait_for_url(params.get("pattern", ""), params.get("timeout")),
            
            # Scroll Actions
            "scroll": lambda: self.scroll(params.get("direction", "down"), params.get("distance", 500)),
            "scroll_to": lambda: self.scroll_to_element(params.get("selector", "")),
            "hover": lambda: self.hover(params.get("selector", "")),
            "drag_and_drop": lambda: self.drag_and_drop(params.get("source", ""), params.get("target", "")),
            
            # Screenshot & Visual Actions
            "screenshot": lambda: self.screenshot(params.get("path"), params.get("full_page", False),
                                                 params.get("selector"), params.get("base64", False)),
            "highlight": lambda: self.highlight_element(params.get("selector", ""), params.get("duration", 2)),
            "get_pdf": lambda: self.get_pdf(params.get("path", "page.pdf"), params.get("options")),
            
            # Get Information Actions
            "get_text": lambda: self.get_text(params.get("selector", "")),
            "get_html": lambda: self.get_html(params.get("selector")),
            "get_attribute": lambda: self.get_attribute(params.get("selector", ""), params.get("attribute", "")),
            "get_value": lambda: self.get_value(params.get("selector", "")),
            "get_element_info": lambda: self.get_element_info(params.get("selector", "")),
            "find_elements": lambda: self.find_elements(params.get("selector", "")),
            
            # Extraction Actions
            "extract_links": lambda: self.extract_links(params.get("selector")),
            "extract_images": lambda: self.extract_images(),
            "extract_table": lambda: self.extract_table(params.get("selector", ""), params.get("as_dict", True)),
            "extract_metadata": lambda: self.extract_metadata(),
            
            # JavaScript Actions
            "execute_script": lambda: self.execute_script(params.get("script", ""), *params.get("args", [])),
            "inject_script": lambda: self.inject_script(params.get("path", "")),
            
            # Tab Management
            "new_tab": lambda: self.new_tab(params.get("url")),
            "switch_tab": lambda: self.switch_tab(params.get("index", 0)),
            "close_tab": lambda: self.close_tab(params.get("index")),
            "get_tabs": lambda: self.get_tabs_info(),
            
            # Cookie & Storage
            "get_cookies": lambda: self.get_cookies(params.get("urls")),
            "set_cookies": lambda: self.set_cookies(params.get("cookies", [])),
            "clear_cookies": lambda: self.clear_cookies(),
            "get_local_storage": lambda: self.get_local_storage(params.get("key")),
            "set_local_storage": lambda: self.set_local_storage(params.get("key", ""), params.get("value", "")),
            "clear_local_storage": lambda: self.clear_local_storage(),
            "get_session_storage": lambda: self.get_session_storage(params.get("key")),
            
            # Viewport & Emulation
            "set_viewport": lambda: self.set_viewport(params.get("width", 1920), params.get("height", 1080)),
            "emulate_device": lambda: self.emulate_device(params.get("device_name", "")),
            "set_geolocation": lambda: self.set_geolocation(params.get("latitude", 0), params.get("longitude", 0),
                                                            params.get("accuracy", 100)),
            "set_permissions": lambda: self.set_permissions(params.get("permissions", [])),
            
            # Network Actions
            "enable_network_monitoring": lambda: self.enable_network_monitoring(),
            "get_network_requests": lambda: self.get_network_requests(params.get("filter_status")),
            "block_resources": lambda: self.block_resources(params.get("resource_types", [])),
            "intercept_request": lambda: self.intercept_request(params.get("url_pattern", ""), 
                                                                params.get("handler")),
            "mock_response": lambda: self.mock_response(params.get("url_pattern", ""), 
                                                        params.get("mock_data", {}),
                                                        params.get("status", 200)),
            "monitor_network": lambda: self.monitor_network(params.get("duration", 10)),
            
            # Performance & Auditing
            "get_performance": lambda: self.get_performance(),
            "analyze_accessibility": lambda: self.analyze_accessibility(),
            "capture_console_logs": lambda: self.capture_console_logs(),
            "perform_audit": lambda: self.perform_audit(),
            
            # Video Recording
            "record_video": lambda: self.record_video(params.get("path", "recording.webm")),
            "stop_recording": lambda: self.stop_recording(),
            
            # Search & Smart Actions
            "search": lambda: self.search_google(params.get("query", "")),
            "search_google": lambda: self.search_google(params.get("query", "")),
            "open_youtube": lambda: self.open_youtube(),
            "open_website": lambda: self.open_website(params.get("name", "")),
        }
        
        try:
            # Execute action from map
            if action in action_map:
                result = await action_map[action]() if asyncio.iscoroutinefunction(action_map[action]) else action_map[action]()
                
                # Standardize response format
                if isinstance(result, str):
                    return {"status": "success", "message": result}
                elif isinstance(result, dict):
                    if "status" not in result:
                        result["status"] = "success"
                    return result
                else:
                    return {"status": "success", "data": result}
            else:
                # Try direct method call for dynamic actions
                method_name = action.replace(" ", "_").lower()
                if hasattr(self, method_name) and callable(getattr(self, method_name)):
                    method = getattr(self, method_name)
                    result = await method(**params) if asyncio.iscoroutinefunction(method) else method(**params)
                    return {"status": "success", "data": result}
                
                # Action not found
                return {
                    "status": "ignored",
                    "message": f"Unknown action '{action}' safely ignored",
                    "available_actions": list(action_map.keys())[:20]  # Show first 20 for brevity
                }
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "action": action,
                "params": params
            }
    
    # ==================== SMART ACTIONS (AI READY) ====================
    
    async def search_google(self, query: str):
        """Search Google with query"""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return await self.navigate(url)
    
    async def search_bing(self, query: str):
        """Search Bing with query"""
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
        return await self.navigate(url)
    
    async def search_duckduckgo(self, query: str):
        """Search DuckDuckGo with query"""
        url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
        return await self.navigate(url)
    
    async def open_youtube(self):
        """Open YouTube"""
        return await self.navigate("https://youtube.com")
    
    async def open_github(self):
        """Open GitHub"""
        return await self.navigate("https://github.com")
    
    async def open_website(self, name: str):
        """Open website by name"""
        return await self.navigate(f"https://{name}.com")
    
    async def fill_form(self, form_data: Dict[str, str]):
        """Fill multiple form fields at once"""
        results = []
        for selector, value in form_data.items():
            result = await self.type_text(selector, value)
            results.append(result)
        return f"Filled {len(results)} form fields"
    
    async def extract_all_data(self) -> Dict[str, Any]:
        """Extract all possible data from current page"""
        await self._ensure()
        
        return {
            "metadata": await self.extract_metadata(),
            "links": await self.extract_links(),
            "images": await self.extract_images(),
            "html": await self.get_html(),
            "text": await self.get_text("body"),
            "cookies": await self.get_cookies(),
            "local_storage": await self.get_local_storage(),
            "session_storage": await self.get_session_storage(),
            "performance": await self.get_performance(),
            "timestamp": datetime.now().isoformat()
        }
    
    # ==================== CHAINING & BATCH OPERATIONS ====================
    
    async def execute_sequence(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a sequence of actions in order
        Each action: {"action": "click", "params": {"selector": "#button"}}
        """
        results = []
        
        for action_item in actions:
            action_name = action_item.get("action")
            params = action_item.get("params", {})
            
            result = await self.execute_action(action_name, params)
            results.append({
                "action": action_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Stop on critical error
            if result.get("status") == "error" and action_item.get("stop_on_error", True):
                break
        
        return results
    
    @asynccontextmanager
    async def session(self):
        """Context manager for browser session"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    # ==================== EXPORT & SAVE DATA ====================
    
    async def save_session_state(self, path: str):
        """Save browser session state (cookies, storage)"""
        await self._ensure()
        
        state = {
            "cookies": await self.get_cookies(),
            "local_storage": await self.get_local_storage(),
            "session_storage": await self.get_session_storage(),
            "url": self.current_url,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        return f"Session state saved to {path}"
    
    async def load_session_state(self, path: str):
        """Load browser session state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        if "cookies" in state:
            await self.set_cookies(state["cookies"])
        
        if "local_storage" in state:
            for key, value in state["local_storage"].items():
                await self.set_local_storage(key, value)
        
        if "url" in state:
            await self.navigate(state["url"])
        
        return f"Session state loaded from {path}"
    
    async def export_action_history(self, path: str):
        """Export action history to file"""
        with open(path, 'w') as f:
            json.dump(self.action_history, f, indent=2)
        
        return f"Action history exported to {path}"
    
    # ==================== MONITORING & OBSERVABILITY ====================
    
    async def watch_for_changes(self, selector: str, callback: Callable, interval: float = 1.0):
        """Watch element for changes and execute callback"""
        await self._ensure()
        
        last_value = await self.get_text(selector)
        
        while True:
            await asyncio.sleep(interval)
            current_value = await self.get_text(selector)
            
            if current_value != last_value:
                await callback(selector, last_value, current_value)
                last_value = current_value
    
    async def take_smart_screenshot(self, name: str = None):
        """Take screenshot with automatic naming and metadata"""
        if not name:
            name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        path = f"screenshots/{name}.png"
        
        # Ensure directory exists
        Path("screenshots").mkdir(exist_ok=True)
        
        # Take screenshot
        await self.screenshot(path)
        
        # Save metadata
        metadata = {
            "name": name,
            "path": path,
            "url": self.current_url,
            "timestamp": datetime.now().isoformat(),
            "viewport": {"width": self.config.viewport_width, "height": self.config.viewport_height}
        }
        
        with open(f"screenshots/{name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return path
    
    # ==================== ERROR HANDLING & RECOVERY ====================
    
    async def safe_execute(self, action: str, params: dict = None, fallback: callable = None) -> dict:
        """Execute action with automatic error recovery"""
        try:
            result = await self.execute_action(action, params)
            
            if result.get("status") == "error" and fallback:
                logger.warning(f"Action failed, executing fallback: {action}")
                return await fallback()
            
            return result
            
        except Exception as e:
            logger.error(f"Safe execute failed: {e}")
            
            # Attempt recovery by restarting browser
            if self.fail_count > 3:
                await self.stop()
                await self.start()
                self.fail_count = 0
            
            return {"status": "error", "message": str(e), "recovered": False}
    
    # ==================== HELPER METHODS ====================
    
    def _log(self, action: BrowserAction, params: Dict[str, Any]):
        """Log action to history"""
        self.actions_performed += 1
        self.action_history.append({
            "action": action.value,
            "params": params,
            "time": datetime.now().isoformat(),
            "url": self.current_url
        })
        
        # Keep history manageable (last 1000 actions)
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    def __repr__(self) -> str:
        return f"ChromeController(connected={self.is_connected}, url={self.current_url[:50]}, actions={self.actions_performed})"


# ==================== FACTORY FUNCTION ====================

async def create_chrome_controller(config: Optional[BrowserConfig] = None, 
                                   headless: bool = False,
                                   timeout: float = 30000) -> ChromeController:
    """
    Factory function to create and start a ChromeController instance
    """
    if config is None:
        config = BrowserConfig(headless=headless)
    
    controller = ChromeController(config, timeout)
    await controller.start()
    return controller
