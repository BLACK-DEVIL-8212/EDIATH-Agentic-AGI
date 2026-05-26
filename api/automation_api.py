"""Automation API - RESTful API for automation operations."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime
from pathlib import Path
import json
import aiohttp
from bs4 import BeautifulSoup
import re

from core.automation import CitationManager, DataExtractor, WebResearcher
from core.automation.chrome_controller import ChromeController

from ..core.utils.logger import logger

# =========================
# REQUEST/RESPONSE MODELS
# =========================


class SearchRequest(BaseModel):
    """Request to search web."""

    query: str
    limit: int = Field(10, ge=1, le=100)
    engine: str = Field("google", description="Search engine: google, bing, duckduckgo")


class SearchResponse(BaseModel):
    """Response from web search."""

    results: List[Dict[str, Any]]
    query: str
    result_count: int
    engine: str
    timestamp: str


class NavigateRequest(BaseModel):
    """Request to navigate."""

    url: str
    headless: bool = True
    wait_time: int = Field(3, ge=1, le=30)


class NavigateResponse(BaseModel):
    """Response from navigation."""

    status: str
    url: str
    title: Optional[str] = None
    timestamp: str
    error: Optional[str] = None


class DataExtractionRequest(BaseModel):
    """Request to extract data."""

    source: str
    pattern: str
    extraction_type: str = Field("regex", description="regex, css, xpath")


class DataExtractionResponse(BaseModel):
    """Response from data extraction."""

    status: str
    data: List[Any]
    count: int
    extraction_type: str
    timestamp: str


class CitationRequest(BaseModel):
    """Request to add citation."""

    url: str
    title: str
    author: Optional[str] = None
    date: Optional[str] = None


class CitationResponse(BaseModel):
    """Response for citation operations."""

    status: str
    citation_id: Optional[str] = None
    message: str
    timestamp: str


class BatchSearchRequest(BaseModel):
    """Request for batch search operations."""

    queries: List[str]
    limit_per_query: int = Field(5, ge=1, le=20)


# =========================
# MAIN API CLASS
# =========================


class AutomationApi:
    """API for automation operations with web search, navigation, extraction, and citations."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize automation API.

        Args:
            data_dir: Directory for storing automation data (default: automation_data/)
        """
        self.data_dir = data_dir or Path("automation_data")
        self.data_dir.mkdir(exist_ok=True)

        self.researcher = WebResearcher()
        self.browser = ChromeController()
        self.extractor = DataExtractor()
        self.citations = CitationManager()

        self.request_count = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self._citations_file = self.data_dir / "citations.json"
        self._load_citations()

    # =========================
    # INITIALIZATION
    # =========================

    async def initialize(self, *args, **kwargs):
        """Initialize async resources."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        await self._initialize_browser()

    async def _initialize_browser(self):
        """Initialize browser controller."""
        try:
            if hasattr(self.browser, "initialize"):
                await self.browser.initialize()
            logger.info("Browser initialized successfully")
        except Exception as e:
            logger.warning(f"Browser initialization failed: {e}")

    def _load_citations(self):
        """Load citations from file."""
        try:
            if self._citations_file.exists():
                with open(self._citations_file, "r") as f:
                    data = json.load(f)
                    if hasattr(self.citations, "load"):
                        self.citations.load(data)
                    elif hasattr(self.citations, "citations"):
                        self.citations.citations = data
                logger.info(f"Loaded {len(data)} citations")
        except Exception as e:
            logger.warning(f"Failed to load citations: {e}")

    def _save_citations(self):
        """Save citations to file."""
        try:
            citations_data = self.get_citations()
            with open(self._citations_file, "w") as f:
                json.dump(citations_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save citations: {e}")

    # =========================
    # WEB SEARCH
    # =========================

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search the web using specified engine."""
        try:
            self.request_count += 1
            logger.info(f"Searching {request.engine} for: {request.query}")

            results = []

            # Try different search methods
            if hasattr(self.researcher, "search"):
                results = await self.researcher.search(
                    request.query, engine=request.engine
                )
            elif hasattr(self.researcher, "research"):
                results = await self.researcher.research(request.query)
            else:
                # Fallback to direct web search
                results = await self._direct_search(request.query, request.engine)

            results = results[: request.limit] if results else []

            logger.info(f"Web search completed: {len(results)} results")

            return SearchResponse(
                results=results,
                query=request.query,
                result_count=len(results),
                engine=request.engine,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResponse(
                results=[],
                query=request.query,
                result_count=0,
                engine=request.engine,
                timestamp=datetime.now().isoformat(),
            )

    async def _direct_search(self, query: str, engine: str) -> List[Dict[str, Any]]:
        """Direct web search without external dependencies."""
        results = []

        search_urls = {
            "google": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}",
            "duckduckgo": f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}",
        }

        url = search_urls.get(engine, search_urls["google"])

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Extract search results
                    for result in soup.find_all("div", class_=re.compile("g|result"))[
                        :10
                    ]:
                        title_elem = result.find("h3") or result.find("a")
                        link_elem = result.find("a", href=True)

                        if title_elem and link_elem:
                            results.append(
                                {
                                    "title": title_elem.get_text(strip=True),
                                    "url": link_elem["href"],
                                    "snippet": result.get_text(strip=True)[:200],
                                }
                            )

        except Exception as e:
            logger.error(f"Direct search error: {e}")

        return results

    async def batch_search(self, request: BatchSearchRequest) -> List[SearchResponse]:
        """Perform multiple searches in parallel."""
        tasks = []
        for query in request.queries:
            search_req = SearchRequest(query=query, limit=request.limit_per_query)
            tasks.append(self.search(search_req))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, SearchResponse):
                valid_results.append(r)
            else:
                logger.error(f"Batch search error: {r}")

        return valid_results

    # =========================
    # NAVIGATION
    # =========================

    async def navigate(self, request: NavigateRequest) -> NavigateResponse:
        """Navigate to URL and get page info."""
        try:
            self.request_count += 1
            logger.info(f"Navigating to: {request.url}")

            title = None

            # Try browser navigation
            if hasattr(self.browser, "navigate"):
                title = await self.browser.navigate(
                    request.url, wait_time=request.wait_time
                )
            elif hasattr(self.browser, "open"):
                await self.browser.open(request.url)
                if hasattr(self.browser, "get_title"):
                    title = await self.browser.get_title()
            else:
                # Fallback: HTTP GET
                async with self.session.get(request.url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        title = soup.title.string if soup.title else request.url

            logger.info(f"Navigation completed: {request.url}")

            return NavigateResponse(
                status="success",
                url=request.url,
                title=title,
                timestamp=datetime.now().isoformat(),
                error=None,
            )

        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return NavigateResponse(
                status="error",
                url=request.url,
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    async def get_page_content(self, url: str) -> Optional[str]:
        """Get HTML content of a page."""
        try:
            if hasattr(self.browser, "get_content"):
                return await self.browser.get_content()
            else:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
        except Exception as e:
            logger.error(f"Get page content error: {e}")
        return None

    # =========================
    # DATA EXTRACTION
    # =========================

    async def extract_data(
        self, request: DataExtractionRequest
    ) -> DataExtractionResponse:
        """Extract data from source using specified pattern."""
        try:
            self.request_count += 1
            logger.info(f"Extracting data from: {request.source[:100]}")

            data = []

            # Get content if source is URL
            content = request.source
            if request.source.startswith(("http://", "https://")):
                content = await self.get_page_content(request.source) or request.source

            # Extract based on type
            if request.extraction_type == "regex":
                pattern = re.compile(request.pattern, re.IGNORECASE)
                data = pattern.findall(content)

            elif request.extraction_type == "css":
                soup = BeautifulSoup(content, "html.parser")
                elements = soup.select(request.pattern)
                data = [elem.get_text(strip=True) for elem in elements]

            elif request.extraction_type == "xpath":
                # Simple XPath simulation (for basic cases)
                data = await self._extract_xpath(content, request.pattern)

            # Use extractor if available
            if hasattr(self.extractor, "extract") and not data:
                data = self.extractor.extract(request.source, request.pattern)

            logger.info(f"Data extraction completed: {len(data)} items")

            return DataExtractionResponse(
                status="success",
                data=data[:100],  # Limit results
                count=len(data),
                extraction_type=request.extraction_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return DataExtractionResponse(
                status="error",
                data=[],
                count=0,
                extraction_type=request.extraction_type,
                timestamp=datetime.now().isoformat(),
            )

    async def _extract_xpath(self, content: str, xpath: str) -> List[str]:
        """Simple XPath-like extraction."""
        # This is a simplified implementation
        soup = BeautifulSoup(content, "html.parser")

        # Handle common XPath patterns
        if xpath.startswith("//"):
            tag = xpath[2:].split("/")[0]
            elements = soup.find_all(tag)
            return [elem.get_text(strip=True) for elem in elements]

        return []

    # =========================
    # CITATIONS
    # =========================

    def add_citation(self, request: CitationRequest) -> CitationResponse:
        """Add a citation."""
        try:
            citation_id = f"cit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            citation = {
                "id": citation_id,
                "url": request.url,
                "title": request.title,
                "author": request.author,
                "date": request.date or datetime.now().isoformat(),
                "added_at": datetime.now().isoformat(),
            }

            if hasattr(self.citations, "add_citation"):
                self.citations.add_citation(request.url, request.title)
            elif hasattr(self.citations, "citations"):
                if not hasattr(self.citations, "citations"):
                    self.citations.citations = []
                self.citations.citations.append(citation)

            self._save_citations()
            logger.info(f"Citation added: {request.title}")

            return CitationResponse(
                status="success",
                citation_id=citation_id,
                message=f"Citation '{request.title}' added successfully",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Citation error: {e}")
            return CitationResponse(
                status="error", message=str(e), timestamp=datetime.now().isoformat()
            )

    def get_citations(self) -> List[Dict[str, str]]:
        """Get all citations."""
        try:
            if hasattr(self.citations, "get_citations"):
                return self.citations.get_citations()
            elif hasattr(self.citations, "citations"):
                return self.citations.citations
            return []
        except Exception as e:
            logger.error(f"Get citations error: {e}")
            return []

    def get_citation(self, citation_id: str) -> Optional[Dict[str, str]]:
        """Get a specific citation by ID."""
        citations = self.get_citations()
        for citation in citations:
            if citation.get("id") == citation_id:
                return citation
        return None

    def delete_citation(self, citation_id: str) -> CitationResponse:
        """Delete a citation."""
        try:
            if hasattr(self.citations, "delete_citation"):
                self.citations.delete_citation(citation_id)
            elif hasattr(self.citations, "citations"):
                self.citations.citations = [
                    c for c in self.citations.citations if c.get("id") != citation_id
                ]

            self._save_citations()

            return CitationResponse(
                status="success",
                message=f"Citation {citation_id} deleted",
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Delete citation error: {e}")
            return CitationResponse(
                status="error", message=str(e), timestamp=datetime.now().isoformat()
            )

    def format_citations(self, format_type: str = "apa") -> List[str]:
        """Format citations in specified style."""
        citations = self.get_citations()
        formatted = []

        for cit in citations:
            if format_type == "apa":
                author = cit.get("author", "Unknown")
                year = cit.get("date", "n.d.")[:4]
                title = cit.get("title", "Untitled")
                url = cit.get("url", "")
                formatted.append(f"{author} ({year}). {title}. Retrieved from {url}")

            elif format_type == "mla":
                author = cit.get("author", "Unknown")
                title = cit.get("title", "Untitled")
                url = cit.get("url", "")
                formatted.append(f'{author}. "{title}." Web. {url}')

            elif format_type == "simple":
                formatted.append(
                    f"- {cit.get('title', 'Untitled')}: {cit.get('url', '')}"
                )

        return formatted

    # =========================
    # STATISTICS & UTILITIES
    # =========================

    def get_stats(self) -> Dict[str, Any]:
        """Get API statistics."""
        try:
            citations = self.get_citations()

            return {
                "request_count": self.request_count,
                "citations_count": len(citations),
                "citations_file": str(self._citations_file),
                "data_dir": str(self.data_dir),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {
                "request_count": self.request_count,
                "citations_count": 0,
                "error": str(e),
            }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear cached data."""
        try:
            self.request_count = 0

            if hasattr(self.citations, "clear"):
                self.citations.clear()
            elif hasattr(self.citations, "citations"):
                self.citations.citations = []

            self._save_citations()

            return {
                "status": "success",
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Clear cache error: {e}")
            return {"status": "error", "message": str(e)}

    async def close(self):
        """Close API resources."""
        try:
            if self.session:
                await self.session.close()

            if hasattr(self.browser, "close"):
                await self.browser.close()

            logger.info("Automation API closed")
        except Exception as e:
            logger.error(f"Close error: {e}")

    # =========================
    # CONTEXT MANAGER
    # =========================

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# =========================
# FASTAPI ROUTER (OPTIONAL)
# =========================

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse

    router = APIRouter(prefix="/automation", tags=["automation"])
    api_instance = AutomationApi()

    @router.on_event("startup")
    async def startup_event():
        await api_instance.initialize()

    @router.on_event("shutdown")
    async def shutdown_event():
        await api_instance.close()

    @router.post("/search", response_model=SearchResponse)
    async def search_endpoint(request: SearchRequest):
        """Search the web."""
        return await api_instance.search(request)

    @router.post("/batch-search", response_model=List[SearchResponse])
    async def batch_search_endpoint(request: BatchSearchRequest):
        """Perform batch web search."""
        return await api_instance.batch_search(request)

    @router.post("/navigate", response_model=NavigateResponse)
    async def navigate_endpoint(request: NavigateRequest):
        """Navigate to a URL."""
        return await api_instance.navigate(request)

    @router.post("/extract", response_model=DataExtractionResponse)
    async def extract_endpoint(request: DataExtractionRequest):
        """Extract data from source."""
        return await api_instance.extract_data(request)

    @router.post("/citations")
    async def add_citation_endpoint(request: CitationRequest):
        """Add a citation."""
        return api_instance.add_citation(request)

    @router.get("/citations")
    async def get_citations_endpoint():
        """Get all citations."""
        return {"citations": api_instance.get_citations()}

    @router.get("/citations/{citation_id}")
    async def get_citation_endpoint(citation_id: str):
        """Get specific citation."""
        citation = api_instance.get_citation(citation_id)
        if not citation:
            raise HTTPException(status_code=404, detail="Citation not found")
        return citation

    @router.delete("/citations/{citation_id}")
    async def delete_citation_endpoint(citation_id: str):
        """Delete a citation."""
        return api_instance.delete_citation(citation_id)

    @router.get("/stats")
    async def stats_endpoint():
        """Get API statistics."""
        return api_instance.get_stats()

    @router.post("/clear-cache")
    async def clear_cache_endpoint():
        """Clear cache."""
        return api_instance.clear_cache()

except ImportError:
    logger.warning("FastAPI not available - router not created")
    router = None


__all__ = [
    "AutomationApi",
    "SearchRequest",
    "SearchResponse",
    "NavigateRequest",
    "NavigateResponse",
    "DataExtractionRequest",
    "DataExtractionResponse",
    "CitationRequest",
    "CitationResponse",
    "BatchSearchRequest",
    "router",
]
