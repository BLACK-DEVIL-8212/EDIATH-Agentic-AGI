"""
EDIATH Research Agent - Production Level
Advanced web research with multiple backends, caching, rate limiting, and intelligent scraping
"""

import asyncio
import aiohttp
import hashlib
import re
import time
from typing import Dict, Any, Optional, List
from collections import OrderedDict
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup

from core.utils.logger import logger

# Try to import optional dependencies
try:
    from googlesearch import search as google_search

    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    logger.warning("googlesearch-python not installed, Google search disabled")

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False


class ResearchCache:
    """Thread-safe cache for research results"""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 7200):
        self.cache = OrderedDict()
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(300, ttl_seconds)
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()

    def _make_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters"""
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)

        # Include key parameters in cache key
        params = []
        if kwargs.get("num_results"):
            params.append(f"num={kwargs['num_results']}")
        if kwargs.get("language"):
            params.append(f"lang={kwargs['language']}")
        if kwargs.get("timeout"):
            params.append(f"timeout={kwargs['timeout']}")

        key_str = normalized + "|" + "|".join(params)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    async def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result if valid"""
        async with self._lock:
            key = self._make_key(query, **kwargs)

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

    async def set(self, query: str, result: Dict[str, Any], **kwargs):
        """Store result in cache"""
        async with self._lock:
            key = self._make_key(query, **kwargs)

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
            logger.info("Research cache cleared")


class RateLimiter:
    """Rate limiter for API calls"""

    def __init__(self, requests_per_second: float = 1.0):
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


class ContentExtractor:
    """Extract and clean content from HTML"""

    @staticmethod
    def extract_text(html: str, max_length: int = 5000) -> str:
        """Extract main text content from HTML"""
        try:
            # Try trafilatura first for best extraction
            if TRAFILATURA_AVAILABLE:
                text = trafilatura.extract(
                    html, include_comments=False, include_tables=True
                )
                if text:
                    return text[:max_length]

            # Fallback to BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                element.decompose()

            # Get text
            text = soup.get_text(separator=" ", strip=True)

            # Clean up
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n\s*\n", "\n\n", text)

            return text[:max_length]

        except Exception as e:
            logger.debug(f"Content extraction failed: {e}")
            return ""

    @staticmethod
    def extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from HTML"""
        metadata = {}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Meta description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag["content"][:500]

        # Meta keywords
        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            metadata["keywords"] = keywords_tag["content"][:300]

        return metadata

    @staticmethod
    def extract_links(
        soup: BeautifulSoup, base_url: str, max_links: int = 20
    ) -> List[str]:
        """Extract relevant links from page"""
        links = []
        seen = set()

        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Skip empty or javascript links
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Handle relative URLs
            if href.startswith("/"):
                parsed = urlparse(base_url)
                href = f"{parsed.scheme}://{parsed.netloc}{href}"

            # Filter out non-http links
            if not href.startswith(("http://", "https://")):
                continue

            # Remove fragments
            if "#" in href:
                href = href.split("#")[0]

            if href not in seen and len(links) < max_links:
                seen.add(href)
                links.append(href)

        return links


class ResearchAgent:
    """
    Production-ready research agent with multiple backends, caching, and intelligent scraping
    """

    def __init__(
        self,
        cache_size: int = 200,
        cache_ttl_seconds: int = 7200,
        max_retries: int = 2,
        timeout_seconds: float = 15.0,
        requests_per_second: float = 1.0,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    ):
        """
        Initialize Research Agent

        Args:
            cache_size: Maximum cache entries
            cache_ttl_seconds: Cache TTL in seconds (default: 2 hours)
            max_retries: Maximum retry attempts
            timeout_seconds: Request timeout in seconds
            requests_per_second: Rate limit for API calls
            user_agent: User-Agent string for requests
        """
        self.max_retries = max(0, max_retries)
        self.timeout_seconds = max(1.0, timeout_seconds)
        self.user_agent = user_agent
        self.name = "research_agent"

        # Initialize cache and rate limiter
        self.cache = ResearchCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self.rate_limiter = RateLimiter(requests_per_second=requests_per_second)

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Content extractor
        self.extractor = ContentExtractor()

        # Statistics
        self._stats = {
            "total_researches": 0,
            "successful_researches": 0,
            "failed_researches": 0,
            "cached_responses": 0,
            "pages_crawled": 0,
            "total_tokens": 0,
        }
        self._stats_lock = asyncio.Lock()

        logger.info(
            f"ResearchAgent initialized (cache_size={cache_size}, ttl={cache_ttl_seconds}s)"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=5,
                        ttl_dns_cache=300,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": self.user_agent},
                    )
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------
    # MAIN ENTRY
    # ------------------------
    async def run(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en",
        extract_content: bool = True,
        max_content_length: int = 3000,
        include_metadata: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Research a topic by searching and extracting information

        Args:
            query: Research query
            num_results: Number of search results to process (1-10)
            language: Language code for search
            extract_content: Extract full content from pages
            max_content_length: Maximum content length per page
            include_metadata: Include page metadata in results

        Returns:
            Dict with research results and metadata
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

            # Validate num_results
            num_results = max(1, min(10, num_results))

            # Check cache
            cache_key_kwargs = {"num_results": num_results, "language": language}
            cached = await self.cache.get(query, **cache_key_kwargs)
            if cached:
                async with self._stats_lock:
                    self._stats["cached_responses"] += 1
                    self._stats["total_researches"] += 1

                cached["cached"] = True
                cached["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                logger.debug(f"Research cache hit: {query[:50]}")
                return cached

            # Execute research
            result = await self._research_with_retries(
                query=query,
                num_results=num_results,
                language=language,
                extract_content=extract_content,
                max_content_length=max_content_length,
                include_metadata=include_metadata,
                **kwargs,
            )

            if not result or not result.get("success"):
                async with self._stats_lock:
                    self._stats["failed_researches"] += 1
                    self._stats["total_researches"] += 1
                return self._fail(f"Research failed for: {query[:50]}")

            # Prepare response
            response = {
                "success": True,
                "output": result.get("summary", ""),
                "source": "research",
                "query": query,
                "results": result.get("results", []),
                "num_results": len(result.get("results", [])),
                "cached": False,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

            # Add compiled summary
            if result.get("compiled_summary"):
                response["compiled_summary"] = result["compiled_summary"]

            # Cache the result
            await self.cache.set(query, response, **cache_key_kwargs)

            # Update stats
            async with self._stats_lock:
                self._stats["successful_researches"] += 1
                self._stats["total_researches"] += 1
                self._stats["total_tokens"] += len(result.get("summary", ""))

            logger.info(
                f"Research complete: {query[:50]}... ({response['latency_ms']}ms, {response['num_results']} results)"
            )
            return response

        except asyncio.TimeoutError:
            logger.error(f"Research timeout for: {query[:50]}")
            return self._fail("Research timeout - please try again")

        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return self._fail(f"Research error: {str(e)[:100]}")

    # ------------------------
    # RESEARCH WITH RETRIES
    # ------------------------
    async def _research_with_retries(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en",
        extract_content: bool = True,
        max_content_length: int = 3000,
        include_metadata: bool = True,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Execute research with retry logic"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._research_sync(
                        query=query,
                        num_results=num_results,
                        language=language,
                        extract_content=extract_content,
                        max_content_length=max_content_length,
                        include_metadata=include_metadata,
                    ),
                    timeout=self.timeout_seconds * 2,
                )

                if result:
                    return result

            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"Research attempt {attempt + 1} timeout: {query[:50]}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Research attempt {attempt + 1} failed: {e}")

            if attempt < self.max_retries:
                wait_time = 0.5 * (2**attempt)
                await asyncio.sleep(wait_time)

        logger.debug(f"All research attempts failed for: {query[:50]}")
        return None

    # ------------------------
    # SYNC RESEARCH LOGIC
    # ------------------------
    def _research_sync(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en",
        extract_content: bool = True,
        max_content_length: int = 3000,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Production-level synchronous research pipeline
        - Safe
        - Fast
        - Thread-friendly
        - No async misuse
        """

        start_time = time.time()

        try:
            # ------------------------
            # VALIDATION
            # ------------------------
            if not query or not isinstance(query, str):
                return {"success": False, "error": "Invalid query"}

            query = query.strip()
            if len(query) < 2:
                return {"success": False, "error": "Query too short"}

            num_results = max(1, min(num_results, 10))

            # ------------------------
            # STEP 1: SEARCH
            # ------------------------
            try:
                urls = self._get_search_urls(query, num_results, language)
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return {"success": False, "error": "Search failed"}

            if not urls:
                return {"success": False, "error": "No search results found"}

            # ------------------------
            # STEP 2: FETCH (PARALLEL)
            # ------------------------
            results = []
            all_content = []

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def safe_fetch(url):
                try:
                    return self._fetch_page(
                        url, extract_content, max_content_length, include_metadata
                    )
                except Exception as e:
                    logger.debug(f"Fetch failed: {url} | {e}")
                    return None

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(safe_fetch, url) for url in urls]

                for future in as_completed(futures):
                    page_data = future.result()

                    if not page_data:
                        continue

                    results.append(page_data)

                    content = page_data.get("content")
                    if content:
                        all_content.append(content)

            if not results:
                return {"success": False, "error": "No content extracted"}

            # ------------------------
            # STEP 3: STATS (SAFE)
            # ------------------------
            try:
                if hasattr(self, "_stats"):
                    self._stats["pages_crawled"] = self._stats.get(
                        "pages_crawled", 0
                    ) + len(results)
            except Exception:
                pass

            # ------------------------
            # STEP 4: SUMMARY
            # ------------------------
            try:
                summary = self._compile_summary(query, results, all_content)
            except Exception as e:
                logger.warning(f"Summary failed: {e}")
                summary = "Summary unavailable"

            try:
                compiled = self._create_compiled_summary(results, all_content)
            except Exception as e:
                logger.warning(f"Compiled summary failed: {e}")
                compiled = ""

            # ------------------------
            # STEP 5: CLEAN OUTPUT
            # ------------------------
            execution_time = round(time.time() - start_time, 2)

            return {
                "success": True,
                "query": query,
                "results": results[:num_results],
                "summary": summary,
                "compiled_summary": compiled,
                "total_results": len(results),
                "execution_time": execution_time,
            }

        except Exception as e:
            logger.error(f"Research sync error: {e}")
            return {"success": False, "error": str(e)}

    def _get_search_urls(
        self, query: str, num_results: int, language: str
    ) -> List[str]:
        """Get search result URLs using available search backends"""
        urls = []

        # Try Google search first
        if GOOGLE_SEARCH_AVAILABLE:
            try:
                search_results = list(
                    google_search(
                        query, num_results=num_results, lang=language, advanced=False
                    )
                )
                urls = search_results[:num_results]
                logger.debug(f"Google search returned {len(urls)} results")
            except Exception as e:
                logger.warning(f"Google search failed: {e}")

        # Fallback to simulated search or cache
        if not urls:
            urls = self._get_fallback_urls(query, num_results)

        # Filter and validate URLs
        valid_urls = []
        for url in urls:
            if self._is_valid_url(url):
                valid_urls.append(url)

        return valid_urls[:num_results]

    def _get_fallback_urls(self, query: str, num_results: int) -> List[str]:
        """Generate fallback URLs when search is unavailable"""
        encoded_query = quote_plus(query)
        base_urls = [
            f"https://en.wikipedia.org/wiki/{encoded_query.replace('+', '_')}",
            f"https://www.britannica.com/search?query={encoded_query}",
            f"https://www.bing.com/search?q={encoded_query}",
        ]
        return base_urls[:num_results]

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and domain"""
        if not url or not isinstance(url, str):
            return False

        # Check basic URL format
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False

            # Skip known problematic domains
            skip_domains = [
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "reddit.com",
                "youtube.com",
                "pinterest.com",
                "amazon.com",
                "ebay.com",
                "aliexpress.com",
            ]

            for skip in skip_domains:
                if skip in parsed.netloc.lower():
                    return False

            return True
        except Exception:
            return False

    def _fetch_page(
        self,
        url: str,
        extract_content: bool,
        max_content_length: int,
        include_metadata: bool,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and extract content from a single page"""
        try:
            # Make request
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )

            if response.status_code != 200:
                return None

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            result = {"url": url, "status_code": response.status_code}

            # Extract metadata
            if include_metadata:
                metadata = self.extractor.extract_metadata(soup)
                result.update(metadata)

            # Extract content
            if extract_content:
                content = self.extractor.extract_text(response.text, max_content_length)
                if content:
                    result["content"] = content

            # Extract links
            links = self.extractor.extract_links(soup, url, max_links=10)
            if links:
                result["related_links"] = links[:5]

            return result

        except Exception as e:
            logger.debug(f"Page fetch error for {url}: {e}")
            return None

    def _compile_summary(
        self, query: str, results: List[Dict], all_content: List[str]
    ) -> str:
        """Compile a concise summary from research results"""
        if not results:
            return f"No research results found for '{query}'."

        summary_parts = []

        # Add main findings
        summary_parts.append(f"Research results for '{query}':")

        for i, result in enumerate(results[:3], 1):
            title = result.get("title", "Untitled")
            description = result.get("description", "")

            if description:
                summary_parts.append(f"{i}. {title}: {description[:200]}")
            elif result.get("content"):
                content_preview = result["content"][:150]
                summary_parts.append(f"{i}. {title}: {content_preview}...")
            else:
                summary_parts.append(f"{i}. {title}")

        return "\n".join(summary_parts)

    def _create_compiled_summary(
        self, results: List[Dict], all_content: List[str]
    ) -> str:
        """Create a comprehensive compiled summary"""
        if not all_content:
            return ""

        # Combine all content
        combined = " ".join(all_content)

        # Extract key sentences (simple approach)
        sentences = re.split(r"[.!?]+", combined)

        # Score sentences by relevance (presence of query terms)
        query_terms = set()
        if results and results[0].get("title"):
            query_terms.update(results[0]["title"].lower().split()[:5])

        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 30:
                continue

            score = 0
            sentence_lower = sentence.lower()
            for term in query_terms:
                if term in sentence_lower:
                    score += 1

            # Prefer sentences with numbers, dates, or specific facts
            if re.search(r"\d+", sentence):
                score += 1

            scored_sentences.append((score, sentence.strip()))

        # Sort by score and take top 5
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in scored_sentences[:5] if len(s) > 20]

        if top_sentences:
            return "Key findings:\n• " + "\n• ".join(top_sentences[:3])

        return ""

    # ------------------------
    # ADVANCED RESEARCH
    # ------------------------
    async def deep_research(
        self, query: str, depth: int = 2, max_sources: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform deep research by following related links

        Args:
            query: Research query
            depth: How many levels of links to follow (1-3)
            max_sources: Maximum number of sources to include
            **kwargs: Additional arguments passed to run()

        Returns:
            Deep research results with multiple sources
        """
        start_time = time.time()

        try:
            # Initial research
            initial = await self.run(query, num_results=min(5, max_sources), **kwargs)

            if not initial.get("success"):
                return initial

            all_sources = initial.get("results", [])
            seen_urls = {s.get("url") for s in all_sources if s.get("url")}

            # Follow related links if depth > 1
            if depth > 1 and initial.get("results"):
                for result in initial["results"][:3]:
                    related_links = result.get("related_links", [])

                    for link in related_links[:2]:
                        if link not in seen_urls and len(all_sources) < max_sources:
                            # Fetch the related page
                            try:
                                await self.rate_limiter.acquire()

                                page_data = await asyncio.to_thread(
                                    self._fetch_page, link, True, 2000, True
                                )

                                if page_data and page_data.get("content"):
                                    page_data["source_depth"] = 2
                                    all_sources.append(page_data)
                                    seen_urls.add(link)
                            except Exception as e:
                                logger.debug(f"Deep research link failed: {e}")

            # Compile deep research summary
            all_content = [
                s.get("content", "") for s in all_sources if s.get("content")
            ]
            deep_summary = self._create_compiled_summary(all_sources, all_content)

            return {
                "success": True,
                "output": deep_summary or initial.get("output", ""),
                "source": "deep_research",
                "query": query,
                "sources": all_sources,
                "total_sources": len(all_sources),
                "depth_used": depth,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

        except Exception as e:
            logger.error(f"Deep research error: {e}")
            return self._fail(f"Deep research failed: {str(e)[:100]}")

    # ------------------------
    # FACT CHECKING
    # ------------------------
    async def fact_check(self, claim: str) -> Dict[str, Any]:
        """
        Fact-check a claim by researching multiple sources

        Args:
            claim: Statement to fact-check

        Returns:
            Fact-check results with confidence and sources
        """
        try:
            # Search for the claim
            results = await self.run(claim, num_results=5, extract_content=True)

            if not results.get("success"):
                return {
                    "success": False,
                    "claim": claim,
                    "verdict": "unknown",
                    "confidence": 0.0,
                    "message": "Could not find sufficient sources",
                }

            # Analyze sources for consistency
            sources = results.get("results", [])
            supporting_count = 0
            contradicting_count = 0

            # Simple analysis (can be enhanced with NLP)
            claim_lower = claim.lower()
            claim_words = set(claim_lower.split())

            for source in sources:
                content = source.get("content", "").lower()
                if not content:
                    continue

                # Check if source supports or contradicts
                if claim_lower in content:
                    supporting_count += 1
                else:
                    # Check for contradictory phrases
                    contradict_phrases = [
                        "not",
                        "false",
                        "incorrect",
                        "myth",
                        "debunked",
                    ]
                    if any(phrase in content for phrase in contradict_phrases):
                        contradicting_count += 1

            # Determine verdict
            total = supporting_count + contradicting_count
            if total == 0:
                verdict = "unknown"
                confidence = 0.0
            elif supporting_count > contradicting_count * 2:
                verdict = "likely_true"
                confidence = supporting_count / total
            elif contradicting_count > supporting_count * 2:
                verdict = "likely_false"
                confidence = contradicting_count / total
            elif supporting_count > 0 and contradicting_count > 0:
                verdict = "mixed"
                confidence = 0.5
            elif supporting_count > 0:
                verdict = "possibly_true"
                confidence = 0.6
            elif contradicting_count > 0:
                verdict = "possibly_false"
                confidence = 0.6
            else:
                verdict = "insufficient_evidence"
                confidence = 0.2

            return {
                "success": True,
                "claim": claim,
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "sources_analyzed": total,
                "supporting_sources": supporting_count,
                "contradicting_sources": contradicting_count,
                "summary": results.get("output", ""),
                "sources": sources[:3],
            }

        except Exception as e:
            logger.error(f"Fact check error: {e}")
            return {
                "success": False,
                "claim": claim,
                "verdict": "error",
                "confidence": 0.0,
                "error": str(e)[:100],
            }

    # ------------------------
    # GET STATISTICS
    # ------------------------
    async def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        async with self._stats_lock:
            cache_stats = await self.cache.get_stats()

            total = self._stats["total_researches"]
            success_rate = round(
                self._stats["successful_researches"] / max(1, total) * 100, 1
            )

            return {
                "agent": "research",
                "cache": cache_stats,
                "requests": {
                    "total": total,
                    "successful": self._stats["successful_researches"],
                    "failed": self._stats["failed_researches"],
                    "cached": self._stats["cached_responses"],
                    "success_rate_percent": success_rate,
                },
                "pages_crawled": self._stats["pages_crawled"],
                "total_tokens": self._stats["total_tokens"],
                "config": {
                    "max_retries": self.max_retries,
                    "timeout_seconds": self.timeout_seconds,
                },
                "backends": {
                    "google_search": GOOGLE_SEARCH_AVAILABLE,
                    "trafilatura": TRAFILATURA_AVAILABLE,
                    "requests": REQUESTS_AVAILABLE,
                },
            }

    # ------------------------
    # CLEAR CACHE
    # ------------------------
    async def clear_cache(self):
        """Clear the result cache"""
        await self.cache.clear()
        logger.info("Research cache cleared")

    # ------------------------
    # FAIL RESPONSE
    # ------------------------
    def _fail(self, msg: str) -> Dict[str, Any]:
        """Generate failure response"""
        return {"success": False, "output": msg, "source": "research", "error": True}


# ------------------------
# CONVENIENCE FUNCTIONS
# ------------------------

_global_research_agent: Optional[ResearchAgent] = None
_agent_lock = asyncio.Lock()


async def get_research_agent(
    cache_size: int = 200, requests_per_second: float = 1.0
) -> ResearchAgent:
    """Get or create global research agent instance"""
    global _global_research_agent

    async with _agent_lock:
        if _global_research_agent is None:
            _global_research_agent = ResearchAgent(
                cache_size=cache_size, requests_per_second=requests_per_second
            )
        return _global_research_agent


async def research(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to research a topic"""
    agent = await get_research_agent()
    return await agent.run(query, **kwargs)


async def fact_check(claim: str) -> Dict[str, Any]:
    """Convenience function to fact-check a claim"""
    agent = await get_research_agent()
    return await agent.fact_check(claim)


__all__ = [
    "ResearchAgent",
    "ResearchCache",
    "get_research_agent",
    "research",
    "fact_check",
]
