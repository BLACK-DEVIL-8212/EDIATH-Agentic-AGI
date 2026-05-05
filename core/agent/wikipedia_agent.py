"""
EDIATH Wikipedia Agent - Production Level
Fetches summaries from Wikipedia safely with caching, rate limiting, and error recovery
"""

import asyncio
import wikipedia
import time
import hashlib
import re
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from core.utils.logger import logger

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class WikipediaCache:
    """Thread-safe cache for Wikipedia results"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(60, ttl_seconds)
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()

    def _make_key(self, query: str) -> str:
        """Generate cache key from query"""
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if valid"""
        async with self._lock:
            key = self._make_key(query)

            if key not in self.cache:
                self.misses += 1
                return None

            result, timestamp = self.cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            return result

    async def set(self, query: str, result: Dict[str, Any]):
        """Store result in cache"""
        async with self._lock:
            key = self._make_key(query)

            # Remove if exists
            if key in self.cache:
                del self.cache[key]

            # Evict oldest if needed
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = (result, time.time())

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(self.hits / max(1, self.hits + self.misses), 3),
                "ttl_seconds": self.ttl_seconds,
            }

    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self.cache.clear()
            logger.info("Wikipedia cache cleared")


class RateLimiter:
    """Simple rate limiter for Wikipedia API"""

    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / max(0.1, requests_per_second)
        self._last_call = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if needed to respect rate limit"""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)
            self._last_call = time.time()


class WikipediaAgent:
    """Production-ready Wikipedia agent with caching, rate limiting, and error handling"""

    def __init__(
        self,
        language: str = "en",
        cache_size: int = 100,
        cache_ttl_seconds: int = 3600,
        max_retries: int = 2,
        timeout_seconds: float = 10.0,
        requests_per_second: float = 1.0,
    ):
        """
        Initialize Wikipedia Agent

        Args:
            language: Wikipedia language code (default: "en")
            cache_size: Maximum cache entries
            cache_ttl_seconds: Cache TTL in seconds (default: 1 hour)
            max_retries: Maximum retry attempts
            timeout_seconds: Request timeout in seconds
            requests_per_second: Rate limit for API calls
        """
        self.language = language
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.name = "wikipedia_agent"

        # Set Wikipedia language
        try:
            wikipedia.set_lang(language)
        except Exception as e:
            logger.warning(f"Failed to set Wikipedia language: {e}")

        # Initialize cache and rate limiter
        self.cache = WikipediaCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self.rate_limiter = RateLimiter(requests_per_second=requests_per_second)

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "total_tokens": 0,
        }
        self._stats_lock = asyncio.Lock()

        logger.info(
            f"WikipediaAgent initialized (lang={language}, cache_size={cache_size}, ttl={cache_ttl_seconds}s)"
        )

    # ------------------------
    # MAIN ENTRY
    # ------------------------
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search Wikipedia and return summary

        Args:
            query: Search query
            **kwargs: Additional options (sentences, auto_suggest, redirect)

        Returns:
            Dict with success, output, source, and metadata
        """
        start_time = time.time()

        try:
            if not query or not isinstance(query, str):
                return self._fail("Invalid query: must be a non-empty string")

            query = query.strip()
            if len(query) < 2:
                return self._fail("Query too short (minimum 2 characters)")

            if len(query) > 200:
                query = query[:200]
                logger.debug("Query truncated to 200 chars")

            # Check cache first
            cached = await self.cache.get(query)
            if cached:
                async with self._stats_lock:
                    self._stats["cached_responses"] += 1
                    self._stats["total_requests"] += 1

                cached["cached"] = True
                cached["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                logger.debug(f"Wikipedia cache hit: {query[:50]}")
                return cached

            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Execute search with retries
            result = await self._search_with_retries(query, **kwargs)

            if not result:
                async with self._stats_lock:
                    self._stats["failed_requests"] += 1
                    self._stats["total_requests"] += 1
                return self._fail(f"No results found for: {query[:50]}")

            # Prepare response
            response = {
                "success": True,
                "output": result["summary"],
                "source": "wikipedia",
                "title": result.get("title", query),
                "url": result.get("url", ""),
                "cached": False,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

            # Add optional metadata
            if "sentences" in kwargs:
                response["sentences_used"] = kwargs.get("sentences", 3)

            # Cache the result
            await self.cache.set(query, response)

            # Update stats
            async with self._stats_lock:
                self._stats["successful_requests"] += 1
                self._stats["total_requests"] += 1
                self._stats["total_tokens"] += len(result.get("summary", ""))

            logger.info(
                f"Wikipedia search success: {query[:50]}... ({response['latency_ms']}ms)"
            )
            return response

        except asyncio.TimeoutError:
            logger.error(f"Wikipedia timeout for: {query[:50]}")
            return self._fail(
                "Request timeout - Wikipedia may be slow, please try again"
            )

        except Exception as e:
            logger.error(f"Wikipedia agent error: {e}")
            return self._fail(f"Wikipedia error: {str(e)[:100]}")

    # ------------------------
    # SEARCH WITH RETRIES
    # ------------------------
    async def _search_with_retries(
        self, query: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Execute Wikipedia search with retry logic"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._search_sync, query, **kwargs),
                    timeout=self.timeout_seconds,
                )

                if result:
                    return result

            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"Wikipedia attempt {attempt + 1} timeout: {query[:50]}")

            except wikipedia.exceptions.WikipediaException as e:
                last_error = str(e)
                logger.warning(f"Wikipedia attempt {attempt + 1} failed: {e}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Wikipedia attempt {attempt + 1} error: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 0.5 * (2**attempt)
                await asyncio.sleep(wait_time)

        logger.debug(
            f"All Wikipedia attempts failed for: {query[:50]}, last_error: {last_error}"
        )
        return None

    # ------------------------
    # SYNC SEARCH LOGIC
    # ------------------------
    def _search_sync(
        self,
        query: str,
        sentences: int = 3,
        auto_suggest: bool = True,
        redirect: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous Wikipedia search (runs in thread pool)

        Args:
            query: Search query
            sentences: Number of sentences for summary (1-10)
            auto_suggest: Auto-suggest corrections
            redirect: Follow redirects

        Returns:
            Dict with summary, title, and URL, or None if not found
        """
        try:
            # Validate parameters
            sentences = max(1, min(10, sentences))

            # Try to get page directly
            try:
                page = wikipedia.page(
                    query, auto_suggest=auto_suggest, redirect=redirect
                )
                summary = wikipedia.summary(
                    query, sentences=sentences, auto_suggest=auto_suggest
                )

                return {
                    "summary": self._clean_summary(summary),
                    "title": page.title,
                    "url": page.url,
                }

            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation - try first option
                if e.options:
                    first_option = e.options[0]
                    logger.debug(
                        f"Disambiguation for '{query}', trying '{first_option}'"
                    )

                    try:
                        page = wikipedia.page(first_option, auto_suggest=False)
                        summary = wikipedia.summary(first_option, sentences=sentences)

                        return {
                            "summary": self._clean_summary(summary),
                            "title": page.title,
                            "url": page.url,
                            "disambiguation": True,
                            "original_query": query,
                        }
                    except Exception:
                        pass

                raise

            except wikipedia.exceptions.PageError:
                # Try search as fallback
                search_results = wikipedia.search(query, results=3)

                if search_results:
                    best_match = search_results[0]
                    logger.debug(f"Page not found, using search result: '{best_match}'")

                    page = wikipedia.page(best_match, auto_suggest=False)
                    summary = wikipedia.summary(best_match, sentences=sentences)

                    return {
                        "summary": self._clean_summary(summary),
                        "title": page.title,
                        "url": page.url,
                        "search_fallback": True,
                        "original_query": query,
                    }

                raise

        except wikipedia.exceptions.RedirectError as e:
            # Handle redirects
            logger.debug(f"Redirect for '{query}': {e}")
            try:
                page = wikipedia.page(str(e).split("'")[1] if "'" in str(e) else query)
                summary = wikipedia.summary(page.title, sentences=sentences)

                return {
                    "summary": self._clean_summary(summary),
                    "title": page.title,
                    "url": page.url,
                    "redirected": True,
                }
            except Exception:
                return None

        except Exception as e:
            logger.debug(f"Wikipedia search error: {e}")
            return None

    # ------------------------
    # TEXT CLEANING
    # ------------------------
    def _clean_summary(self, text: str) -> str:
        """Clean and normalize Wikipedia summary"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove citation markers like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Remove edit markers
        text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)

        # Ensure proper punctuation at end
        if text and text[-1] not in ".!?":
            text += "."

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text.strip()

    # ------------------------
    # BATCH SEARCH
    # ------------------------
    async def batch_search(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Search multiple queries concurrently with rate limiting

        Args:
            queries: List of search queries
            **kwargs: Additional options passed to run()

        Returns:
            List of results in same order as queries
        """
        if not queries:
            return []

        # Deduplicate queries
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        # Run searches with semaphore to control concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

        async def search_one(query):
            async with semaphore:
                return await self.run(query, **kwargs)

        tasks = [search_one(q) for q in unique_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(self._fail(f"Error: {str(result)[:100]}"))
            else:
                final_results.append(result)

        return final_results

    # ------------------------
    # GET PAGE CONTENT (Full)
    # ------------------------
    async def get_page_content(
        self, query: str, sections: bool = False
    ) -> Dict[str, Any]:
        """
        Get full page content (not just summary)

        Args:
            query: Page title or search query
            sections: Include section breakdown

        Returns:
            Dict with full content and metadata
        """
        try:
            await self.rate_limiter.acquire()

            def get_content():
                try:
                    page = wikipedia.page(query, auto_suggest=True)
                    content = page.content

                    result = {
                        "title": page.title,
                        "url": page.url,
                        "content": content[:10000],  # Limit size
                        "length": len(content),
                    }

                    if sections:
                        # Simple section extraction
                        lines = content.split("\n")
                        result["sections"] = [
                            l.strip()
                            for l in lines[:20]
                            if l.strip() and len(l.strip()) < 100
                        ]

                    return result
                except Exception as e:
                    logger.debug(f"Get content failed: {e}")
                    return None

            result = await asyncio.wait_for(
                asyncio.to_thread(get_content), timeout=self.timeout_seconds * 2
            )

            if result:
                return {"success": True, "source": "wikipedia", **result}
            else:
                return self._fail(f"Could not retrieve page: {query}")

        except asyncio.TimeoutError:
            return self._fail("Request timeout")
        except Exception as e:
            return self._fail(f"Error: {str(e)[:100]}")

    # ------------------------
    # GET STATISTICS
    # ------------------------
    async def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        async with self._stats_lock:
            cache_stats = await self.cache.get_stats()

            total = self._stats["total_requests"]
            success_rate = round(
                self._stats["successful_requests"] / max(1, total) * 100, 1
            )

            return {
                "agent": "wikipedia",
                "language": self.language,
                "cache": cache_stats,
                "requests": {
                    "total": total,
                    "successful": self._stats["successful_requests"],
                    "failed": self._stats["failed_requests"],
                    "cached": self._stats["cached_responses"],
                    "success_rate_percent": success_rate,
                },
                "total_tokens": self._stats["total_tokens"],
                "config": {
                    "max_retries": self.max_retries,
                    "timeout_seconds": self.timeout_seconds,
                },
            }

    # ------------------------
    # CLEAR CACHE
    # ------------------------
    async def clear_cache(self):
        """Clear the result cache"""
        await self.cache.clear()
        logger.info("Wikipedia cache cleared")

    # ------------------------
    # SET LANGUAGE
    # ------------------------
    async def set_language(self, language: str):
        """Change Wikipedia language"""
        try:
            wikipedia.set_lang(language)
            self.language = language
            await self.clear_cache()
            logger.info(f"Wikipedia language changed to: {language}")
            return True
        except Exception as e:
            logger.error(f"Failed to set language: {e}")
            return False

    # ------------------------
    # FAIL RESPONSE
    # ------------------------
    def _fail(self, msg: str) -> Dict[str, Any]:
        """Generate failure response"""
        return {"success": False, "output": msg, "source": "wikipedia", "error": True}


# ------------------------
# CONVENIENCE FUNCTIONS
# ------------------------

_global_wikipedia_agent: Optional[WikipediaAgent] = None
_agent_lock = asyncio.Lock()


async def get_wikipedia_agent(
    language: str = "en", cache_size: int = 100
) -> WikipediaAgent:
    """Get or create global Wikipedia agent instance"""
    global _global_wikipedia_agent

    async with _agent_lock:
        if _global_wikipedia_agent is None:
            _global_wikipedia_agent = WikipediaAgent(
                language=language, cache_size=cache_size
            )
        return _global_wikipedia_agent


async def search_wikipedia(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to search Wikipedia"""
    agent = await get_wikipedia_agent()
    return await agent.run(query, **kwargs)


__all__ = [
    "WikipediaAgent",
    "WikipediaCache",
    "get_wikipedia_agent",
    "search_wikipedia",
]
