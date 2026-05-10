"""
Search Agent for EDIATH
Performs web searches, Google searches, and web scraping with safety and rate limiting
"""

import aiohttp
import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from bs4 import BeautifulSoup
import random


class SearchProvider(Enum):
    """Supported search providers"""

    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    BRAVE = "brave"
    CUSTOM = "custom"


class SearchType(Enum):
    """Types of search operations"""

    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    SCHOLAR = "scholar"
    CODE = "code"


@dataclass
class SearchResult:
    """Individual search result"""

    title: str
    url: str
    snippet: str
    source: str
    rank: int
    timestamp: datetime
    content_type: str = "webpage"
    score: float = 0.0
    cached: bool = False


@dataclass
class SearchQuery:
    """Search query with metadata"""

    query: str
    search_type: SearchType
    provider: SearchProvider
    timestamp: datetime
    results_count: int
    execution_time: float
    cached: bool = False


class SearchAgent:
    """
    Advanced web search agent capable of:
    - Multiple search providers (Google, Bing, DuckDuckGo, Brave)
    - Web scraping with caching
    - Rate limiting and polite crawling
    - Result deduplication and ranking
    - Search history and analytics
    - Custom search with API keys
    - Parallel search queries
    - Content extraction from search results
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Search Agent

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # API Configuration
        self.api_keys = {
            SearchProvider.GOOGLE: self.config.get("google_api_key", ""),
            SearchProvider.BING: self.config.get("bing_api_key", ""),
            SearchProvider.BRAVE: self.config.get("brave_api_key", ""),
        }

        # Search engine URLs
        self.search_urls = {
            SearchProvider.GOOGLE: "https://www.google.com/search",
            SearchProvider.BING: "https://www.bing.com/search",
            SearchProvider.DUCKDUCKGO: "https://html.duckduckgo.com/html/",
            SearchProvider.BRAVE: "https://api.search.brave.com/res/v1/web/search",
        }

        # Rate limiting
        self.rate_limit_delay = self.config.get(
            "rate_limit_delay", 1.0
        )  # seconds between requests
        self.last_request_time = 0
        self.request_count = 0
        self.request_window = 60  # seconds
        self.max_requests_per_window = self.config.get("max_requests_per_minute", 30)

        # Caching
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour default
        self.cache: Dict[str, Tuple[datetime, List[SearchResult]]] = {}

        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        ]

        # Headers for requests
        self.default_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Search history
        self.search_history: List[SearchQuery] = []
        self.max_history = self.config.get("max_history", 500)

        # Statistics
        self.stats = {
            "total_searches": 0,
            "cached_searches": 0,
            "failed_searches": 0,
            "total_results": 0,
            "api_calls": 0,
            "rate_limit_hits": 0,
        }

        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None

        self.logger.info("Search Agent initialized")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # Reset counter if window expired
        if current_time - self.last_request_time > self.request_window:
            self.request_count = 0

        # Check if we've exceeded rate limit
        if self.request_count >= self.max_requests_per_window:
            self.stats["rate_limit_hits"] += 1
            wait_time = self.request_window - (current_time - self.last_request_time)
            self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.request_count = 0

        # Ensure minimum delay between requests
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()
        self.request_count += 1

    def _get_cache_key(
        self, query: str, search_type: SearchType, provider: SearchProvider, **kwargs
    ) -> str:
        """Generate cache key for search query"""
        key_data = f"{query}:{search_type.value}:{provider.value}"
        for k, v in sorted(kwargs.items()):
            key_data += f":{k}={v}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Retrieve results from cache if valid"""
        if not self.cache_enabled:
            return None

        if cache_key in self.cache:
            cached_time, results = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                self.logger.debug(f"Cache hit for key: {cache_key}")
                self.stats["cached_searches"] += 1
                return results

        return None

    def _add_to_cache(self, cache_key: str, results: List[SearchResult]):
        """Add results to cache"""
        if self.cache_enabled:
            self.cache[cache_key] = (datetime.now(), results)
            # Clean old cache entries
            self._clean_cache()

    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = [
            key
            for key, (cached_time, _) in self.cache.items()
            if current_time - cached_time > timedelta(seconds=self.cache_ttl)
        ]
        for key in expired_keys:
            del self.cache[key]

    def _get_random_user_agent(self) -> str:
        """Get random user agent for rotation"""
        return random.choice(self.user_agents)

    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.WEB,
        provider: SearchProvider = SearchProvider.DUCKDUCKGO,
        num_results: int = 10,
        safe_search: bool = True,
        country: str = "us",
        language: str = "en",
        use_cache: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform web search

        Args:
            query: Search query string
            search_type: Type of search (web, news, images, etc.)
            provider: Search provider to use
            num_results: Number of results to return
            safe_search: Enable safe search filtering
            country: Country code for localized results
            language: Language code for results
            use_cache: Use cached results if available
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._get_cache_key(
            query,
            search_type,
            provider,
            num_results=num_results,
            safe_search=safe_search,
            country=country,
            language=language,
            **kwargs,
        )

        # Check cache
        if use_cache:
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                self.stats["total_searches"] += 1
                self.stats["total_results"] += len(cached_results)

                return {
                    "success": True,
                    "query": query,
                    "search_type": search_type.value,
                    "provider": provider.value,
                    "results": [
                        self._result_to_dict(r) for r in cached_results[:num_results]
                    ],
                    "total_results": len(cached_results),
                    "execution_time": time.time() - start_time,
                    "cached": True,
                    "timestamp": datetime.now().isoformat(),
                }

        # Perform search based on provider
        try:
            if provider == SearchProvider.GOOGLE:
                results = await self._search_google(
                    query,
                    search_type,
                    num_results,
                    safe_search,
                    country,
                    language,
                    **kwargs,
                )
            elif provider == SearchProvider.BING:
                results = await self._search_bing(
                    query,
                    search_type,
                    num_results,
                    safe_search,
                    country,
                    language,
                    **kwargs,
                )
            elif provider == SearchProvider.DUCKDUCKGO:
                results = await self._search_duckduckgo(
                    query, num_results, safe_search, **kwargs
                )
            elif provider == SearchProvider.BRAVE:
                results = await self._search_brave(
                    query, num_results, safe_search, **kwargs
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Add to cache
            self._add_to_cache(cache_key, results)

            # Update statistics
            self.stats["total_searches"] += 1
            self.stats["total_results"] += len(results)
            self.stats["api_calls"] += 1

            # Add to history
            self._add_to_history(
                SearchQuery(
                    query=query,
                    search_type=search_type,
                    provider=provider,
                    timestamp=datetime.now(),
                    results_count=len(results),
                    execution_time=time.time() - start_time,
                    cached=False,
                )
            )

            return {
                "success": True,
                "query": query,
                "search_type": search_type.value,
                "provider": provider.value,
                "results": [self._result_to_dict(r) for r in results[:num_results]],
                "total_results": len(results),
                "execution_time": time.time() - start_time,
                "cached": False,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            self.stats["failed_searches"] += 1

            return {
                "success": False,
                "query": query,
                "search_type": search_type.value,
                "provider": provider.value,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

    async def _search_duckduckgo(
        self, query: str, num_results: int = 10, safe_search: bool = True, **kwargs
    ) -> List[SearchResult]:
        """Search using DuckDuckGo HTML interface"""
        await self._rate_limit()

        params = {
            "q": query,
            "kl": "us-en",
            "kp": "-2" if not safe_search else "-1",
        }

        headers = self.default_headers.copy()
        headers["User-Agent"] = self._get_random_user_agent()

        session = await self._get_session()

        try:
            async with session.get(
                self.search_urls[SearchProvider.DUCKDUCKGO],
                params=params,
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []
                result_elements = soup.find_all("div", class_="result")

                for idx, element in enumerate(result_elements[:num_results]):
                    title_elem = element.find("a", class_="result__a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = title_elem.get("href", "")

                    # Clean URL
                    if url.startswith("/"):
                        url = f"https://duckduckgo.com{url}"

                    snippet_elem = element.find("a", class_="result__snippet")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append(
                        SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="DuckDuckGo",
                            rank=idx + 1,
                            timestamp=datetime.now(),
                            content_type="webpage",
                        )
                    )

                return results

        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {str(e)}")
            raise

    async def _search_google(
        self,
        query: str,
        search_type: SearchType,
        num_results: int,
        safe_search: bool,
        country: str,
        language: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Search using Google (requires API key)"""
        if not self.api_keys[SearchProvider.GOOGLE]:
            # Fallback to web scraping if no API key
            return await self._scrape_google(query, num_results, safe_search)

        # Use Google Custom Search API
        await self._rate_limit()

        params = {
            "key": self.api_keys[SearchProvider.GOOGLE],
            "q": query,
            "num": min(num_results, 10),
            "safe": "active" if safe_search else "off",
            "gl": country,
            "hl": language,
        }

        # Add search type specific parameters
        if search_type == SearchType.NEWS:
            params["cx"] = kwargs.get("news_cx", "news")
        elif search_type == SearchType.IMAGES:
            params["searchType"] = "image"
        elif search_type == SearchType.VIDEOS:
            params["videoDuration"] = kwargs.get("video_duration", "any")

        session = await self._get_session()
        api_url = "https://www.googleapis.com/customsearch/v1"

        try:
            async with session.get(api_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google API error: {response.status}")

                data = await response.json()

                results = []
                for idx, item in enumerate(data.get("items", [])):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            source="Google",
                            rank=idx + 1,
                            timestamp=datetime.now(),
                            content_type=search_type.value,
                        )
                    )

                return results

        except Exception as e:
            self.logger.error(f"Google search error: {str(e)}")
            raise

    async def _scrape_google(
        self, query: str, num_results: int = 10, safe_search: bool = True
    ) -> List[SearchResult]:
        """Scrape Google search results (fallback when no API key)"""
        await self._rate_limit()

        params = {
            "q": query,
            "num": num_results,
            "safe": "active" if safe_search else "off",
        }

        headers = self.default_headers.copy()
        headers["User-Agent"] = self._get_random_user_agent()

        session = await self._get_session()

        try:
            async with session.get(
                self.search_urls[SearchProvider.GOOGLE], params=params, headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []
                result_divs = soup.find_all("div", class_="g")

                for idx, div in enumerate(result_divs[:num_results]):
                    title_elem = div.find("h3")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = div.find("a")
                    url = link_elem.get("href", "") if link_elem else ""

                    snippet_elem = div.find("div", class_="VwiC3b")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append(
                        SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="Google (scraped)",
                            rank=idx + 1,
                            timestamp=datetime.now(),
                            content_type="webpage",
                        )
                    )

                return results

        except Exception as e:
            self.logger.error(f"Google scraping error: {str(e)}")
            raise

    async def _search_bing(
        self,
        query: str,
        search_type: SearchType,
        num_results: int,
        safe_search: bool,
        country: str,
        language: str,
        **kwargs,
    ) -> List[SearchResult]:
        """Search using Bing"""
        await self._rate_limit()

        params = {
            "q": query,
            "count": num_results,
            "safeSearch": "Strict" if safe_search else "Off",
            "mkt": f"{country}-{language}",
        }

        headers = self.default_headers.copy()
        headers["User-Agent"] = self._get_random_user_agent()
        headers["Ocp-Apim-Subscription-Key"] = self.api_keys.get(
            SearchProvider.BING, ""
        )

        session = await self._get_session()

        try:
            async with session.get(
                self.search_urls[SearchProvider.BING], params=params, headers=headers
            ) as response:
                if response.status != 200:
                    # Fallback to scraping if API key not available
                    return await self._scrape_bing(query, num_results, safe_search)

                data = await response.json()

                results = []
                for idx, item in enumerate(
                    data.get("webPages", {}).get("value", [])[:num_results]
                ):
                    results.append(
                        SearchResult(
                            title=item.get("name", ""),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", ""),
                            source="Bing",
                            rank=idx + 1,
                            timestamp=datetime.now(),
                            content_type="webpage",
                        )
                    )

                return results

        except Exception as e:
            self.logger.error(f"Bing search error: {str(e)}")
            raise

    async def _scrape_bing(
        self, query: str, num_results: int = 10, safe_search: bool = True
    ) -> List[SearchResult]:
        """Scrape Bing search results"""
        await self._rate_limit()

        params = {
            "q": query,
            "count": num_results,
        }

        headers = self.default_headers.copy()
        headers["User-Agent"] = self._get_random_user_agent()

        session = await self._get_session()

        try:
            async with session.get(
                self.search_urls[SearchProvider.BING], params=params, headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []
                result_items = soup.find_all("li", class_="b_algo")

                for idx, item in enumerate(result_items[:num_results]):
                    title_elem = item.find("h2")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = title_elem.find("a")
                    url = link_elem.get("href", "") if link_elem else ""

                    snippet_elem = item.find("p")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append(
                        SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="Bing (scraped)",
                            rank=idx + 1,
                            timestamp=datetime.now(),
                            content_type="webpage",
                        )
                    )

                return results

        except Exception as e:
            self.logger.error(f"Bing scraping error: {str(e)}")
            raise

    async def _search_brave(
        self, query: str, num_results: int = 10, safe_search: bool = True, **kwargs
    ) -> List[SearchResult]:
        """Search using Brave Search API"""
        if not self.api_keys[SearchProvider.BRAVE]:
            raise Exception("Brave API key required")

        await self._rate_limit()

        params = {
            "q": query,
            "count": num_results,
            "safesearch": "strict" if safe_search else "moderate",
        }

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_keys[SearchProvider.BRAVE],
        }

        session = await self._get_session()

        try:
            async with session.get(
                self.search_urls[SearchProvider.BRAVE], params=params, headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Brave API error: {response.status}")

                data = await response.json()

                results = []
                for idx, item in enumerate(
                    data.get("web", {}).get("results", [])[:num_results]
                ):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("description", ""),
                            source="Brave",
                            rank=idx + 1,
                            timestamp=datetime.now(),
                            content_type="webpage",
                            score=item.get("score", 0),
                        )
                    )

                return results

        except Exception as e:
            self.logger.error(f"Brave search error: {str(e)}")
            raise

    async def search_multiple(
        self,
        queries: List[str],
        search_type: SearchType = SearchType.WEB,
        provider: SearchProvider = SearchProvider.DUCKDUCKGO,
        max_concurrent: int = 3,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Perform multiple searches in parallel

        Args:
            queries: List of search queries
            search_type: Type of search
            provider: Search provider
            max_concurrent: Maximum concurrent searches
            **kwargs: Additional search parameters

        Returns:
            List of search results for each query
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_with_semaphore(query):
            async with semaphore:
                return await self.search(query, search_type, provider, **kwargs)

        tasks = [search_with_semaphore(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {"success": False, "query": queries[i], "error": str(result)}
                )
            else:
                processed_results.append(result)

        return processed_results

    async def get_page_content(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Fetch and extract content from a webpage

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Dictionary with page content and metadata
        """
        await self._rate_limit()

        headers = self.default_headers.copy()
        headers["User-Agent"] = self._get_random_user_agent()

        session = await self._get_session()

        try:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    return {
                        "success": False,
                        "url": url,
                        "error": f"HTTP {response.status}",
                    }

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                # Extract metadata
                title = soup.find("title")
                title = title.get_text() if title else ""

                meta_desc = soup.find("meta", attrs={"name": "description"})
                description = meta_desc.get("content", "") if meta_desc else ""

                return {
                    "success": True,
                    "url": url,
                    "title": title,
                    "description": description,
                    "content": text[:10000],  # Limit content size
                    "content_length": len(text),
                    "status_code": response.status,
                }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "url": url,
                "error": f"Timeout after {timeout} seconds",
            }
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}

    async def search_and_fetch(
        self, query: str, num_results: int = 5, fetch_content: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Search and fetch content from top results

        Args:
            query: Search query
            num_results: Number of results to fetch
            fetch_content: Whether to fetch full page content
            **kwargs: Additional search parameters

        Returns:
            Dictionary with search results and fetched content
        """
        # Perform search
        search_result = await self.search(query, num_results=num_results, **kwargs)

        if not search_result["success"]:
            return search_result

        # Fetch content for each result
        if fetch_content:
            for result in search_result["results"]:
                content = await self.get_page_content(result["url"])
                result["page_content"] = content

        return search_result

    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to dictionary"""
        return {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "source": result.source,
            "rank": result.rank,
            "timestamp": result.timestamp.isoformat(),
            "content_type": result.content_type,
            "score": result.score,
            "cached": result.cached,
        }

    def _add_to_history(self, query: SearchQuery):
        """Add search to history"""
        self.search_history.append(query)
        if len(self.search_history) > self.max_history:
            self.search_history.pop(0)

    def get_search_history(self, limit: int = None) -> List[Dict]:
        """Get search history"""
        history = self.search_history
        if limit:
            history = history[-limit:]

        return [
            {
                "query": h.query,
                "search_type": h.search_type.value,
                "provider": h.provider.value,
                "timestamp": h.timestamp.isoformat(),
                "results_count": h.results_count,
                "execution_time": h.execution_time,
                "cached": h.cached,
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        cache_size = len(self.cache)
        cache_memory = (
            sum(len(str(v)) for v in self.cache.values()) if self.cache else 0
        )

        return {
            **self.stats,
            "cache_size": cache_size,
            "cache_memory_mb": cache_memory / (1024 * 1024),
            "history_size": len(self.search_history),
            "success_rate": (
                (
                    (self.stats["total_searches"] - self.stats["failed_searches"])
                    / self.stats["total_searches"]
                    * 100
                )
                if self.stats["total_searches"] > 0
                else 0
            ),
            "cache_hit_rate": (
                (self.stats["cached_searches"] / self.stats["total_searches"] * 100)
                if self.stats["total_searches"] > 0
                else 0
            ),
        }

    def clear_cache(self):
        """Clear search cache"""
        self.cache.clear()
        self.logger.info("Search cache cleared")

    def clear_history(self):
        """Clear search history"""
        self.search_history.clear()
        self.logger.info("Search history cleared")

    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


# Integration wrapper for EDIATH
class SearchAgentWrapper:
    """
    Wrapper class to integrate SearchAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.search_agent = SearchAgent(config)
        self.agent_type = "web_search"
        self.capabilities = [
            "web_search",
            "news_search",
            "image_search",
            "video_search",
            "get_page_content",
            "search_and_fetch",
            "batch_search",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a search request

        Request format:
        {
            'operation': 'search|search_multi|fetch|search_fetch|history|stats',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "search":
            search_type = request.get("search_type", "web")
            provider = request.get("provider", "duckduckgo")

            return await self.search_agent.search(
                query=request.get("query"),
                search_type=SearchType(search_type),
                provider=SearchProvider(provider),
                num_results=request.get("num_results", 10),
                safe_search=request.get("safe_search", True),
                country=request.get("country", "us"),
                language=request.get("language", "en"),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "search_multi":
            search_type = request.get("search_type", "web")
            provider = request.get("provider", "duckduckgo")

            results = await self.search_agent.search_multiple(
                queries=request.get("queries", []),
                search_type=SearchType(search_type),
                provider=SearchProvider(provider),
                max_concurrent=request.get("max_concurrent", 3),
                num_results=request.get("num_results", 10),
                safe_search=request.get("safe_search", True),
            )

            return {"success": True, "results": results}

        elif operation == "fetch":
            return await self.search_agent.get_page_content(
                url=request.get("url"), timeout=request.get("timeout", 10)
            )

        elif operation == "search_fetch":
            search_type = request.get("search_type", "web")
            provider = request.get("provider", "duckduckgo")

            return await self.search_agent.search_and_fetch(
                query=request.get("query"),
                num_results=request.get("num_results", 5),
                fetch_content=request.get("fetch_content", True),
                search_type=SearchType(search_type),
                provider=SearchProvider(provider),
                safe_search=request.get("safe_search", True),
            )

        elif operation == "history":
            return {
                "success": True,
                "history": self.search_agent.get_search_history(
                    limit=request.get("limit")
                ),
            }

        elif operation == "stats":
            return self.search_agent.get_stats()

        elif operation == "clear_cache":
            self.search_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        elif operation == "clear_history":
            self.search_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "SearchAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.search_agent.get_stats(),
            "supported_providers": [p.value for p in SearchProvider],
            "supported_types": [t.value for t in SearchType],
        }

    async def close(self):
        """Clean up resources"""
        await self.search_agent.close()


# Example usage and testing
async def test_search_agent():
    """Test the search agent functionality"""

    # Initialize agent
    agent = SearchAgent()

    print("=== Search Agent Test ===\n")

    # Test basic web search
    print("1. Basic Web Search")
    result = await agent.search(
        query="artificial intelligence news", search_type=SearchType.WEB, num_results=5
    )

    if result["success"]:
        print(
            f"   Found {len(result['results'])} results in {result['execution_time']:.2f}s"
        )
        for i, res in enumerate(result["results"][:3], 1):
            print(f"   {i}. {res['title'][:60]}")
            print(f"      {res['snippet'][:100]}...")
            print()
    else:
        print(f"   Error: {result.get('error')}")

    # Test news search
    print("2. News Search")
    result = await agent.search(
        query="technology breakthroughs 2026",
        search_type=SearchType.NEWS,
        num_results=3,
    )

    if result["success"]:
        print(f"   Found {len(result['results'])} news articles")
        for res in result["results"][:2]:
            print(f"   • {res['title'][:50]}")
    else:
        print(f"   Error: {result.get('error')}")

    # Test fetching page content
    print("\n3. Fetch Page Content")
    content = await agent.get_page_content("https://example.com")
    if content["success"]:
        print(f"   Title: {content['title']}")
        print(f"   Content length: {content['content_length']} chars")
        print(f"   Preview: {content['content'][:100]}...")
    else:
        print(f"   Error: {content.get('error')}")

    # Test multiple searches
    print("\n4. Multiple Concurrent Searches")
    queries = ["python programming", "machine learning", "web development"]
    results = await agent.search_multiple(queries, num_results=3, max_concurrent=3)

    for i, res in enumerate(results):
        if res["success"]:
            print(f"   Query '{queries[i]}': {len(res['results'])} results")
        else:
            print(f"   Query '{queries[i]}': Failed - {res.get('error')}")

    # Test search and fetch
    print("\n5. Search and Fetch")
    result = await agent.search_and_fetch(
        query="latest tech news", num_results=2, fetch_content=True
    )

    if result["success"]:
        for res in result["results"]:
            print(f"   Title: {res['title'][:50]}")
            if "page_content" in res and res["page_content"]["success"]:
                print(f"   Content preview: {res['page_content']['content'][:80]}...")
            print()

    # Get statistics
    print("\n6. Agent Statistics")
    stats = agent.get_stats()
    print(f"   Total searches: {stats['total_searches']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"   Cache size: {stats['cache_size']} entries")

    # Close agent
    await agent.close()

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_search_agent())
