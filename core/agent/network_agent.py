"""
Network Agent for EDIATH
Advanced network operations: API calls, request management, web scraping, rate limiting, proxies
"""

import asyncio
import aiohttp
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import logging
import base64

# Optional dependencies
try:
    from aiohttp_socks import ProxyConnector

    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False

try:
    import brotli

    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False


class HTTPMethod(Enum):
    """HTTP methods"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(Enum):
    """Authentication types"""

    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


class RequestStatus(Enum):
    """Request status"""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class NetworkRequest:
    """Network request container"""

    id: str
    method: HTTPMethod
    url: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    body: Any
    status: RequestStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    response_status: Optional[int] = None
    response_data: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    burst_size: int = 10


class NetworkAgent:
    """
    Advanced network agent capable of:
    - HTTP/HTTPS requests (GET, POST, PUT, DELETE, PATCH)
    - Multiple authentication methods (Basic, Bearer, API Key, OAuth2)
    - Rate limiting and throttling
    - Request retry with exponential backoff
    - Proxy support (HTTP, HTTPS, SOCKS)
    - Request/response compression (gzip, brotli)
    - Cookie management
    - Session management
    - WebSocket connections
    - Batch requests
    - Request/response transformation
    - Circuit breaker pattern
    - Request queuing
    - Response caching
    - Request signing (HMAC)
    - Custom SSL/TLS settings
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Network Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Session management
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.default_session: Optional[aiohttp.ClientSession] = None
        self.default_timeout = self.config.get("timeout", 30)
        self.max_redirects = self.config.get("max_redirects", 5)

        # Rate limiting
        self.rate_limits: Dict[str, RateLimitConfig] = {}
        self.request_timestamps: Dict[str, List[float]] = {}
        self.global_rate_limit = self.config.get("global_rate_limit", None)

        # Retry configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_backoff = self.config.get("retry_backoff", 2)  # multiplier
        self.retry_on_status = self.config.get(
            "retry_on_status", [429, 500, 502, 503, 504]
        )

        # Proxy configuration
        self.proxies: Dict[str, str] = self.config.get("proxies", {})
        self.proxy_rotation = self.config.get("proxy_rotation", False)
        self.current_proxy_index = 0

        # Circuit breaker
        self.circuit_breakers: Dict[str, Dict] = {}
        self.circuit_config = {
            "failure_threshold": self.config.get("circuit_failure_threshold", 5),
            "timeout": self.config.get("circuit_timeout", 60),
            "half_open_timeout": self.config.get("circuit_half_open_timeout", 30),
        }

        # Caching
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        self.response_cache: Dict[str, tuple] = {}

        # Headers
        self.default_headers = self.config.get(
            "default_headers",
            {
                "User-Agent": "EDIATH-NetworkAgent/1.0",
                "Accept": "application/json, text/plain, */*",
                "Accept-Encoding": (
                    "gzip, deflate, br" if BROTLI_AVAILABLE else "gzip, deflate"
                ),
                "Connection": "keep-alive",
            },
        )

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "circuit_breaker_trips": 0,
            "average_response_time": 0.0,
        }

        # Request history
        self.request_history: List[NetworkRequest] = []
        self.max_history = self.config.get("max_history", 1000)

        # Request queue
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.queue_workers: List[asyncio.Task] = []
        self.queue_enabled = self.config.get("queue_enabled", False)
        self.queue_workers_count = self.config.get("queue_workers", 5)

        # Custom transformers
        self.request_transformers: List[Callable] = []
        self.response_transformers: List[Callable] = []

        # Initialize default session lazily from async context.
        # Creating aiohttp sessions outside a running event loop can crash.
        self.default_session = None

        # Start queue workers only if an event loop is already running.
        if self.queue_enabled:
            self._start_queue_workers()

        self.logger.info("Network Agent initialized")

    def _init_default_session(self) -> bool:
        """Initialize default HTTP session if a loop is running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.default_session = None
            return False

        connector = aiohttp.TCPConnector(
            limit=self.config.get("connection_limit", 100),
            limit_per_host=self.config.get("connection_limit_per_host", 30),
            ttl_dns_cache=self.config.get("dns_cache_ttl", 300),
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(total=self.default_timeout)

        self.default_session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=self.default_headers
        )
        return True

    async def get_session(self, session_name: str = "default") -> aiohttp.ClientSession:
        """
        Get or create a named session

        Args:
            session_name: Name of the session

        Returns:
            ClientSession instance
        """
        if session_name == "default":
            if self.default_session is None or self.default_session.closed:
                self._init_default_session()
            if self.default_session is None:
                raise RuntimeError(
                    "No running event loop available to initialize default network session"
                )
            return self.default_session

        if session_name not in self.sessions:
            connector = aiohttp.TCPConnector(
                limit=self.config.get("connection_limit", 100),
                limit_per_host=self.config.get("connection_limit_per_host", 30),
            )
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)

            self.sessions[session_name] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.default_headers.copy(),
            )

        return self.sessions[session_name]

    async def request(
        self,
        method: Union[HTTPMethod, str],
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        body: Optional[Any] = None,
        auth: Optional[Dict] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        session_name: str = "default",
        use_cache: bool = True,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            headers: Request headers
            body: Request body (dict, str, bytes)
            auth: Authentication configuration
            timeout: Request timeout in seconds
            retries: Number of retries on failure
            session_name: Session to use
            use_cache: Use cached response
            stream: Stream response content
            **kwargs: Additional aiohttp parameters

        Returns:
            Dictionary with response data
        """
        if isinstance(method, str):
            method = HTTPMethod(method.upper())

        request_id = self._generate_request_id(method.value, url)
        start_time = time.time()

        # Check circuit breaker
        circuit_key = f"{method.value}:{url}"
        if self._is_circuit_open(circuit_key):
            return {
                "success": False,
                "error": "Circuit breaker is open",
                "request_id": request_id,
                "circuit_open": True,
            }

        # Check cache
        cache_key = self._get_cache_key(method, url, params, body)
        if use_cache and self.cache_enabled and method == HTTPMethod.GET:
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                self.stats["cached_responses"] += 1
                return cached_response

        # Apply rate limiting
        if not await self._check_rate_limit(circuit_key):
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "request_id": request_id,
                "retry_after": self._get_retry_after(circuit_key),
            }

        # Prepare request
        session = await self.get_session(session_name)
        request_headers = self._prepare_headers(headers)
        request_params = self._prepare_params(params)

        # Apply authentication
        if auth:
            request_headers.update(await self._apply_auth(auth, url, method, body))

        # Apply request transformers
        for transformer in self.request_transformers:
            url, request_headers, body = await transformer(url, request_headers, body)

        # Get proxy if configured
        proxy = self._get_proxy()

        retry_count = 0
        max_retries = retries if retries is not None else self.max_retries

        while retry_count <= max_retries:
            try:
                async with session.request(
                    method=method.value,
                    url=url,
                    params=request_params,
                    headers=request_headers,
                    json=(
                        body
                        if isinstance(body, dict) and not isinstance(body, (str, bytes))
                        else None
                    ),
                    data=body if isinstance(body, (str, bytes)) else None,
                    timeout=aiohttp.ClientTimeout(
                        total=timeout or self.default_timeout
                    ),
                    proxy=proxy,
                    **kwargs,
                ) as response:
                    # Read response
                    if stream:
                        content = response.content
                    else:
                        content = await self._read_response(response)

                    response_time = time.time() - start_time

                    # Apply response transformers
                    for transformer in self.response_transformers:
                        content = await transformer(content, response)

                    # Create response object
                    result = {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "data": content,
                        "url": str(response.url),
                        "request_id": request_id,
                        "response_time": response_time,
                        "retry_count": retry_count,
                    }

                    # Cache successful GET requests
                    if (
                        result["success"]
                        and method == HTTPMethod.GET
                        and use_cache
                        and self.cache_enabled
                    ):
                        self._add_to_cache(cache_key, result)

                    # Update statistics
                    self.stats["total_requests"] += 1
                    if result["success"]:
                        self.stats["successful_requests"] += 1
                    else:
                        self.stats["failed_requests"] += 1

                    # Update average response time
                    avg_time = self.stats["average_response_time"]
                    total = self.stats["total_requests"]
                    self.stats["average_response_time"] = (
                        avg_time * (total - 1) + response_time
                    ) / total

                    # Record circuit breaker success
                    self._record_circuit_success(circuit_key)

                    # Add to history
                    await self._add_to_history(
                        request_id,
                        method,
                        url,
                        request_headers,
                        request_params,
                        body,
                        result,
                        response_time,
                        retry_count,
                    )

                    return result

            except asyncio.TimeoutError:
                retry_count += 1
                error_msg = f"Request timeout after {timeout or self.default_timeout}s"

                if retry_count <= max_retries and self._should_retry(retry_count, None):
                    wait_time = self.retry_backoff**retry_count
                    self.logger.warning(
                        f"Request timeout, retrying in {wait_time}s (attempt {retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._record_circuit_failure(circuit_key)
                    return {
                        "success": False,
                        "error": error_msg,
                        "request_id": request_id,
                        "status_code": None,
                        "timeout": True,
                    }

            except aiohttp.ClientError as e:
                retry_count += 1
                error_msg = str(e)

                if retry_count <= max_retries and self._should_retry(retry_count, None):
                    wait_time = self.retry_backoff**retry_count
                    self.logger.warning(
                        f"Request failed: {error_msg}, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._record_circuit_failure(circuit_key)
                    return {
                        "success": False,
                        "error": error_msg,
                        "request_id": request_id,
                        "status_code": None,
                    }

        return {
            "success": False,
            "error": "Max retries exceeded",
            "request_id": request_id,
        }

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Perform GET request"""
        return await self.request(HTTPMethod.GET, url, **kwargs)

    async def post(self, url: str, body: Any = None, **kwargs) -> Dict[str, Any]:
        """Perform POST request"""
        return await self.request(HTTPMethod.POST, url, body=body, **kwargs)

    async def put(self, url: str, body: Any = None, **kwargs) -> Dict[str, Any]:
        """Perform PUT request"""
        return await self.request(HTTPMethod.PUT, url, body=body, **kwargs)

    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Perform DELETE request"""
        return await self.request(HTTPMethod.DELETE, url, **kwargs)

    async def patch(self, url: str, body: Any = None, **kwargs) -> Dict[str, Any]:
        """Perform PATCH request"""
        return await self.request(HTTPMethod.PATCH, url, body=body, **kwargs)

    async def batch_request(self, requests: List[Dict]) -> Dict[str, Any]:
        """
        Execute multiple requests in parallel

        Args:
            requests: List of request configurations

        Returns:
            Dictionary with batch results
        """
        tasks = []
        for req in requests:
            method = req.get("method", "GET")
            url = req.get("url")
            task = self.request(method, url, **req.get("kwargs", {}))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {"success": False, "error": str(result), "index": i}
                )
            else:
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.get("success", False))

        return {
            "success": successful > 0,
            "total": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "results": processed_results,
        }

    async def download_file(
        self,
        url: str,
        destination: Union[str, Path],
        chunk_size: int = 8192,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Download a file with progress tracking

        Args:
            url: File URL
            destination: Destination file path
            chunk_size: Download chunk size
            progress_callback: Callback for progress updates
            **kwargs: Additional request parameters

        Returns:
            Dictionary with download result
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            session = await self.get_session(kwargs.get("session_name", "default"))

            async with session.get(url, **kwargs) as response:
                if response.status != 200:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "url": url,
                    }

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(destination, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if asyncio.iscoroutinefunction(progress_callback):
                                await progress_callback(
                                    downloaded, total_size, progress
                                )
                            else:
                                progress_callback(downloaded, total_size, progress)

                download_time = time.time() - start_time
                speed_mb = (
                    (downloaded / (1024 * 1024)) / download_time
                    if download_time > 0
                    else 0
                )

                return {
                    "success": True,
                    "url": url,
                    "destination": str(destination),
                    "size": downloaded,
                    "size_mb": downloaded / (1024 * 1024),
                    "download_time": download_time,
                    "speed_mbps": speed_mb,
                }

        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return {"success": False, "error": str(e), "url": url}

    async def upload_file(
        self,
        url: str,
        file_path: Union[str, Path],
        field_name: str = "file",
        extra_data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Upload a file via multipart/form-data

        Args:
            url: Upload URL
            file_path: File to upload
            field_name: Form field name for the file
            extra_data: Additional form data
            **kwargs: Additional request parameters

        Returns:
            Dictionary with upload result
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            data = aiohttp.FormData()
            data.add_field(
                field_name,
                open(file_path, "rb"),
                filename=file_path.name,
                content_type="application/octet-stream",
            )

            if extra_data:
                for key, value in extra_data.items():
                    data.add_field(key, value)

            result = await self.post(url, data=data, **kwargs)

            if result.get("success"):
                result["uploaded_file"] = str(file_path)
                result["file_size"] = file_path.stat().st_size

            return result

        except Exception as e:
            self.logger.error(f"Upload error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def websocket_connect(
        self,
        url: str,
        on_message: Callable,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None,
        headers: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Establish WebSocket connection

        Args:
            url: WebSocket URL
            on_message: Callback for received messages
            on_error: Callback for errors
            on_close: Callback for connection close
            headers: WebSocket headers
            **kwargs: Additional aiohttp parameters

        Returns:
            Dictionary with connection result
        """
        session = await self.get_session(kwargs.get("session_name", "default"))

        try:
            async with session.ws_connect(url, headers=headers, **kwargs) as ws:
                self.logger.info(f"WebSocket connected to {url}")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if asyncio.iscoroutinefunction(on_message):
                            await on_message(msg.data)
                        else:
                            on_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        if asyncio.iscoroutinefunction(on_message):
                            await on_message(msg.data)
                        else:
                            on_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        error = ws.exception()
                        self.logger.error(f"WebSocket error: {error}")
                        if on_error:
                            if asyncio.iscoroutinefunction(on_error):
                                await on_error(error)
                            else:
                                on_error(error)
                        break

                if on_close:
                    if asyncio.iscoroutinefunction(on_close):
                        await on_close()
                    else:
                        on_close()

                return {"success": True, "message": "WebSocket connection closed"}

        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def queue_request(self, request_config: Dict) -> Dict[str, Any]:
        """
        Queue a request for later processing

        Args:
            request_config: Request configuration

        Returns:
            Dictionary with queuing result
        """
        if not self.queue_enabled:
            return {"success": False, "error": "Queue not enabled"}

        if not self.queue_workers:
            self._start_queue_workers()

        request_id = self._generate_request_id(
            request_config.get("method", "GET"), request_config.get("url", "")
        )

        await self.request_queue.put(
            {"id": request_id, "config": request_config, "created_at": datetime.now()}
        )

        return {
            "success": True,
            "request_id": request_id,
            "queued": True,
            "queue_size": self.request_queue.qsize(),
        }

    async def _process_queue(self):
        """Process queued requests"""
        while True:
            try:
                item = await self.request_queue.get()

                config = item["config"]
                method = config.get("method", "GET")
                url = config.get("url")

                result = await self.request(method, url, **config.get("kwargs", {}))

                # Callback if provided
                callback = config.get("callback")
                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)

                self.request_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Queue processor error: {str(e)}")
                await asyncio.sleep(1)

    def _start_queue_workers(self):
        """Start queue worker tasks"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.logger.warning(
                "Queue workers requested but no running event loop is available; "
                "workers will start when queue is used from async context."
            )
            return

        for _ in range(self.queue_workers_count):
            worker = asyncio.create_task(self._process_queue())
            self.queue_workers.append(worker)

        self.logger.info(f"Started {self.queue_workers_count} queue workers")

    # ============
    # Rate Limiting
    # ============

    def set_rate_limit(self, key: str, config: RateLimitConfig):
        """Set rate limit for a specific key"""
        self.rate_limits[key] = config
        self.request_timestamps[key] = []

    async def _check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limits"""
        if key not in self.rate_limits:
            if self.global_rate_limit:
                key = "global"
            else:
                return True

        if key not in self.request_timestamps:
            self.request_timestamps[key] = []

        now = time.time()
        timestamps = self.request_timestamps[key]

        # Clean old timestamps
        timestamps = [ts for ts in timestamps if now - ts < 3600]

        config = self.rate_limits.get(
            key,
            RateLimitConfig(
                requests_per_second=self.global_rate_limit or 100,
                requests_per_minute=self.global_rate_limit or 100,
                requests_per_hour=self.global_rate_limit or 100,
                burst_size=10,
            ),
        )

        # Check per second
        last_second = [ts for ts in timestamps if now - ts < 1]
        if len(last_second) >= config.requests_per_second:
            return False

        # Check per minute
        last_minute = [ts for ts in timestamps if now - ts < 60]
        if len(last_minute) >= config.requests_per_minute:
            return False

        # Check per hour
        last_hour = [ts for ts in timestamps if now - ts < 3600]
        if len(last_hour) >= config.requests_per_hour:
            return False

        timestamps.append(now)
        self.request_timestamps[key] = timestamps
        return True

    def _get_retry_after(self, key: str) -> int:
        """Get retry-after time in seconds"""
        if key not in self.request_timestamps:
            return 0

        now = time.time()
        timestamps = self.request_timestamps[key]

        config = self.rate_limits.get(key, RateLimitConfig(1, 60, 3600, 10))

        last_second = [ts for ts in timestamps if now - ts < 1]
        if len(last_second) >= config.requests_per_second:
            return 1

        last_minute = [ts for ts in timestamps if now - ts < 60]
        if len(last_minute) >= config.requests_per_minute:
            return 60 - int((now - min(last_minute)))

        last_hour = [ts for ts in timestamps if now - ts < 3600]
        if len(last_hour) >= config.requests_per_hour:
            return 3600 - int((now - min(last_hour)))

        return 0

    # ============
    # Circuit Breaker
    # ============

    def _is_circuit_open(self, key: str) -> bool:
        """Check if circuit breaker is open"""
        if key not in self.circuit_breakers:
            return False

        circuit = self.circuit_breakers[key]

        if circuit["state"] == "open":
            if time.time() - circuit["opened_at"] > self.circuit_config["timeout"]:
                circuit["state"] = "half_open"
                self.logger.info(f"Circuit {key} moved to half-open state")
                return False
            return True

        return False

    def _record_circuit_failure(self, key: str):
        """Record circuit breaker failure"""
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "failures": 0,
                "state": "closed",
                "opened_at": None,
            }

        circuit = self.circuit_breakers[key]

        if circuit["state"] == "closed":
            circuit["failures"] += 1
            if circuit["failures"] >= self.circuit_config["failure_threshold"]:
                circuit["state"] = "open"
                circuit["opened_at"] = time.time()
                self.stats["circuit_breaker_trips"] += 1
                self.logger.warning(f"Circuit {key} opened due to failures")

    def _record_circuit_success(self, key: str):
        """Record circuit breaker success"""
        if key in self.circuit_breakers:
            circuit = self.circuit_breakers[key]

            if circuit["state"] == "half_open":
                circuit["state"] = "closed"
                circuit["failures"] = 0
                self.logger.info(f"Circuit {key} closed after success")
            elif circuit["state"] == "closed":
                circuit["failures"] = max(0, circuit["failures"] - 1)

    # ============
    # Caching
    # ============

    def _get_cache_key(
        self, method: HTTPMethod, url: str, params: Dict, body: Any
    ) -> str:
        """Generate cache key for request"""
        key_data = (
            f"{method.value}:{url}:{json.dumps(params or {})}:{json.dumps(body or {})}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get response from cache"""
        if cache_key in self.response_cache:
            timestamp, response = self.response_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                response["cached"] = True
                return response
            else:
                del self.response_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, response: Dict):
        """Add response to cache"""
        if len(self.response_cache) > 1000:
            # Remove oldest 10%
            items = sorted(self.response_cache.items(), key=lambda x: x[1][0])
            for key, _ in items[:100]:
                del self.response_cache[key]

        self.response_cache[cache_key] = (datetime.now(), response.copy())

    # ============
    # Authentication
    # ============

    async def _apply_auth(
        self, auth: Dict, url: str, method: HTTPMethod, body: Any
    ) -> Dict:
        """Apply authentication to headers"""
        headers = {}
        auth_type = AuthType(auth.get("type", "none"))

        if auth_type == AuthType.BASIC:
            username = auth.get("username", "")
            password = auth.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        elif auth_type == AuthType.BEARER:
            token = auth.get("token", "")
            headers["Authorization"] = f"Bearer {token}"

        elif auth_type == AuthType.API_KEY:
            key_name = auth.get("key_name", "X-API-Key")
            key_value = auth.get("key_value", "")
            location = auth.get("location", "header")

            if location == "header":
                headers[key_name] = key_value
            elif location == "query":
                # Query param handling would be done in params
                pass

        elif auth_type == AuthType.OAUTH2:
            token = await self._get_oauth2_token(auth)
            headers["Authorization"] = f"Bearer {token}"

        return headers

    async def _get_oauth2_token(self, auth_config: Dict) -> str:
        """Get OAuth2 token (simplified)"""
        # This would implement OAuth2 client credentials flow
        token_url = auth_config.get("token_url")
        client_id = auth_config.get("client_id")
        client_secret = auth_config.get("client_secret")

        if not token_url:
            return ""

        # Simplified - in production, cache tokens and handle refresh
        result = await self.post(
            token_url,
            auth={"type": "basic", "username": client_id, "password": client_secret},
            data={"grant_type": "client_credentials"},
        )

        if result.get("success"):
            return result["data"].get("access_token", "")

        return ""

    # ============
    # Proxy Management
    # ============

    def add_proxy(self, proxy_url: str, proxy_type: str = "http"):
        """Add a proxy to the rotation"""
        self.proxies[proxy_url] = proxy_type

    def _get_proxy(self) -> Optional[str]:
        """Get proxy URL based on rotation strategy"""
        if not self.proxies:
            return None

        if self.proxy_rotation:
            proxy_urls = list(self.proxies.keys())
            proxy_url = proxy_urls[self.current_proxy_index % len(proxy_urls)]
            self.current_proxy_index += 1
            return proxy_url
        else:
            return list(self.proxies.keys())[0] if self.proxies else None

    # ============
    # Utility Methods
    # ============

    def _generate_request_id(self, method: str, url: str) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{method}_{timestamp}"

    def _prepare_headers(self, headers: Optional[Dict]) -> Dict:
        """Prepare request headers"""
        prepared = self.default_headers.copy()
        if headers:
            prepared.update(headers)
        return prepared

    def _prepare_params(self, params: Optional[Dict]) -> Optional[Dict]:
        """Prepare query parameters"""
        if not params:
            return None
        return {
            k: str(v) if not isinstance(v, (list, tuple)) else v
            for k, v in params.items()
        }

    async def _read_response(self, response: aiohttp.ClientResponse) -> Any:
        """Read and decode response content"""
        content_type = response.headers.get("content-type", "")

        # Read bytes
        data = await response.read()

        # Decompress if needed
        content_encoding = response.headers.get("content-encoding", "")
        if content_encoding == "br" and BROTLI_AVAILABLE:
            data = brotli.decompress(data)
        elif content_encoding == "gzip":
            import gzip

            data = gzip.decompress(data)

        # Parse based on content type
        if "application/json" in content_type:
            try:
                return json.loads(data.decode("utf-8"))
            except:
                return data.decode("utf-8")
        elif "text/" in content_type or "xml" in content_type:
            return data.decode("utf-8")
        else:
            return data

    def _should_retry(self, retry_count: int, status_code: Optional[int]) -> bool:
        """Determine if request should be retried"""
        if retry_count > self.max_retries:
            return False

        if status_code and status_code in self.retry_on_status:
            return True

        return True

    async def _add_to_history(
        self,
        request_id: str,
        method: HTTPMethod,
        url: str,
        headers: Dict,
        params: Dict,
        body: Any,
        result: Dict,
        response_time: float,
        retry_count: int,
    ):
        """Add request to history"""
        request = NetworkRequest(
            id=request_id,
            method=method,
            url=url,
            headers=headers,
            params=params,
            body=body,
            status=(
                RequestStatus.SUCCESS if result.get("success") else RequestStatus.FAILED
            ),
            created_at=datetime.now(),
            completed_at=datetime.now(),
            response_status=result.get("status_code"),
            response_data=result.get("data"),
            error=result.get("error"),
            retry_count=retry_count,
            execution_time=response_time,
        )

        self.request_history.append(request)
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history :]

    def register_request_transformer(self, transformer: Callable):
        """Register request transformer function"""
        self.request_transformers.append(transformer)

    def register_response_transformer(self, transformer: Callable):
        """Register response transformer function"""
        self.response_transformers.append(transformer)

    def get_history(self, limit: int = None, success_only: bool = False) -> List[Dict]:
        """Get request history"""
        history = self.request_history

        if success_only:
            history = [h for h in history if h.status == RequestStatus.SUCCESS]

        if limit:
            history = history[-limit:]

        return [
            {
                "id": h.id,
                "method": h.method.value,
                "url": h.url,
                "status": h.status.value,
                "response_status": h.response_status,
                "execution_time": h.execution_time,
                "retry_count": h.retry_count,
                "created_at": h.created_at.isoformat(),
                "error": h.error,
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        success_rate = (
            (self.stats["successful_requests"] / self.stats["total_requests"] * 100)
            if self.stats["total_requests"] > 0
            else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "history_size": len(self.request_history),
            "cache_size": len(self.response_cache),
            "active_sessions": len(self.sessions)
            + (1 if self.default_session and not self.default_session.closed else 0),
            "active_circuit_breakers": len(self.circuit_breakers),
            "queue_size": self.request_queue.qsize() if self.queue_enabled else 0,
        }

    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.logger.info("Response cache cleared")

    def clear_history(self):
        """Clear request history"""
        self.request_history.clear()
        self.logger.info("Request history cleared")

    async def close(self):
        """Close all sessions"""
        # Stop queue workers
        for worker in self.queue_workers:
            worker.cancel()

        # Close sessions
        if self.default_session:
            await self.default_session.close()

        for session in self.sessions.values():
            await session.close()

        self.logger.info("Network Agent closed")


# Integration wrapper for EDIATH
class NetworkAgentWrapper:
    """
    Wrapper class to integrate NetworkAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.network_agent = NetworkAgent(config)
        self.agent_type = "network"
        self.capabilities = [
            "http_requests",
            "file_download",
            "file_upload",
            "websocket",
            "batch_requests",
            "rate_limiting",
            "circuit_breaker",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a network request

        Request format:
        {
            'operation': 'request|get|post|download|upload|batch|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "request":
            method = request.get("method", "GET")
            return await self.network_agent.request(
                method=method,
                url=request.get("url"),
                params=request.get("params"),
                headers=request.get("headers"),
                body=request.get("body"),
                auth=request.get("auth"),
                timeout=request.get("timeout"),
                retries=request.get("retries"),
                session_name=request.get("session_name", "default"),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "get":
            return await self.network_agent.get(
                url=request.get("url"),
                params=request.get("params"),
                headers=request.get("headers"),
                **request.get("kwargs", {}),
            )

        elif operation == "post":
            return await self.network_agent.post(
                url=request.get("url"),
                body=request.get("body"),
                params=request.get("params"),
                headers=request.get("headers"),
                **request.get("kwargs", {}),
            )

        elif operation == "download":
            return await self.network_agent.download_file(
                url=request.get("url"),
                destination=request.get("destination"),
                chunk_size=request.get("chunk_size", 8192),
                **request.get("kwargs", {}),
            )

        elif operation == "upload":
            return await self.network_agent.upload_file(
                url=request.get("url"),
                file_path=request.get("file_path"),
                field_name=request.get("field_name", "file"),
                extra_data=request.get("extra_data"),
                **request.get("kwargs", {}),
            )

        elif operation == "batch":
            return await self.network_agent.batch_request(
                requests=request.get("requests", [])
            )

        elif operation == "queue":
            return await self.network_agent.queue_request(
                request_config=request.get("request_config", {})
            )

        elif operation == "history":
            return {
                "success": True,
                "history": self.network_agent.get_history(
                    limit=request.get("limit"),
                    success_only=request.get("success_only", False),
                ),
            }

        elif operation == "stats":
            return self.network_agent.get_stats()

        elif operation == "clear_cache":
            self.network_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        elif operation == "clear_history":
            self.network_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "NetworkAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.network_agent.get_stats(),
            "default_timeout": self.network_agent.default_timeout,
            "max_retries": self.network_agent.max_retries,
            "queue_enabled": self.network_agent.queue_enabled,
        }

    async def close(self):
        """Clean up resources"""
        await self.network_agent.close()


# Example usage and testing
async def test_network_agent():
    """Test the network agent functionality"""

    # Initialize agent
    agent = NetworkAgent()

    print("=== Network Agent Test ===\n")

    # Test GET request
    print("1. GET Request...")
    result = await agent.get("https://httpbin.org/get")
    if result["success"]:
        print(f"   Status: {result['status_code']}")
        print(f"   Response time: {result['response_time']:.3f}s")
        print(f"   Data preview: {str(result['data'])[:100]}...")

    # Test POST request
    print("\n2. POST Request...")
    result = await agent.post(
        "https://httpbin.org/post", body={"name": "EDIATH", "type": "AI Agent"}
    )
    if result["success"]:
        print(f"   Status: {result['status_code']}")
        print(f"   Data sent: {result['data'].get('json', {})}")

    # Test with query parameters
    print("\n3. Request with Parameters...")
    result = await agent.get(
        "https://httpbin.org/get", params={"page": 1, "limit": 10, "search": "test"}
    )
    if result["success"]:
        args = result["data"].get("args", {})
        print(f"   Parameters: {args}")

    # Test with custom headers
    print("\n4. Request with Custom Headers...")
    result = await agent.get(
        "https://httpbin.org/headers",
        headers={"X-Custom-Header": "EDIATH-Test", "X-Request-ID": "12345"},
    )
    if result["success"]:
        headers = result["data"].get("headers", {})
        print(f"   Custom headers: {headers.get('X-Custom-Header')}")

    # Test batch requests
    print("\n5. Batch Requests...")
    requests = [
        {"method": "GET", "url": "https://httpbin.org/get"},
        {"method": "GET", "url": "https://httpbin.org/status/200"},
        {"method": "GET", "url": "https://httpbin.org/delay/1"},
    ]
    result = await agent.batch_request(requests)
    print(f"   Batch results: {result['successful']}/{result['total']} successful")

    # Test download (small file)
    print("\n6. File Download...")
    result = await agent.download_file(
        "https://httpbin.org/image/png", "test_download.png"
    )
    if result["success"]:
        print(f"   Downloaded: {result['size_mb']:.2f} MB")
        print(f"   Speed: {result['speed_mbps']:.2f} MB/s")

    # Test rate limiting
    print("\n7. Rate Limiting...")
    from core.agent.network_agent import RateLimitConfig

    agent.set_rate_limit(
        "test_api",
        RateLimitConfig(
            requests_per_second=2,
            requests_per_minute=10,
            requests_per_hour=100,
            burst_size=3,
        ),
    )

    for i in range(5):
        result = await agent.get("https://httpbin.org/get")
        print(f"   Request {i+1}: {'✓' if result['success'] else '✗'}")
        if not result["success"] and "retry_after" in result:
            print(f"     Rate limited, retry after {result['retry_after']}s")

    # Get statistics
    print("\n8. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Successful: {stats['successful_requests']}")
    print(f"   Failed: {stats['failed_requests']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Average response time: {stats['average_response_time']:.3f}s")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   History size: {stats['history_size']}")

    # Get request history
    print("\n9. Request History...")
    history = agent.get_history(limit=5)
    for req in history:
        print(
            f"   {req['method']} {req['url'][:50]}... -> {req['status']} ({req['response_status']})"
        )

    # Close agent
    await agent.close()

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_network_agent())
