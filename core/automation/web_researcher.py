"""
Advanced Web Researcher - Real AI Research Engine
"""

from typing import List
from datetime import datetime

from ..utils.logger import logger
from ..automation.chrome_controller import ChromeController
from .data_extractor import DataExtractor
from .citation_manager import CitationManager


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str, relevance: float = 0.5):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.relevance = relevance
        self.timestamp = datetime.now()

    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "relevance": self.relevance,
        }


class WebResearcher:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

        self.browser = ChromeController(headless=True)
        self.extractor = DataExtractor()
        self.citations = CitationManager()

        self.cache = {}
        self.history = []

    # ------------------------
    # SEARCH 🔥 (REAL)
    # ------------------------
    async def search(self, query: str) -> List[SearchResult]:
        if query in self.cache:
            return self.cache[query]

        logger.info(f"🔍 Searching: {query}")

        await self.browser.start()

        # Google search
        search_url = f"https://www.google.com/search?q={query}"
        await self.browser.navigate(search_url)

        html = await self.browser.get_content()

        # Extract links
        import re

        links = re.findall(r'href="(https?://[^"]+)"', html)

        results = []
        for i, link in enumerate(links[: self.max_results]):
            results.append(
                SearchResult(
                    title=f"Result {i+1}",
                    url=link,
                    snippet="Extracted from page",
                    relevance=0.8 - i * 0.1,
                )
            )

        self.cache[query] = results
        self.history.append(query)

        return results

    # ------------------------
    # SCRAPE 🔥
    # ------------------------
    async def scrape(self, url: str) -> str:
        await self.browser.navigate(url)
        content = await self.browser.get_content()

        # Save citation
        self.citations.extract_from_browser(url, content)

        return content

    # ------------------------
    # EXTRACT 🔥
    # ------------------------
    async def extract_data(self, html: str):
        self.extractor.register_pattern(
            "headings", "h1, h2", self.extractor.patterns.get("html", None)
        )

        return await self.extractor.extract(html, "headings")

    # ------------------------
    # FULL RESEARCH 🔥
    # ------------------------
    async def research(self, query: str):
        results = await self.search(query)

        collected = []

        for r in results:
            try:
                content = await self.scrape(r.url)

                extracted = await self.extract_data(content)

                collected.append({"url": r.url, "data": extracted})

            except Exception:
                logger.warning(f"Failed scraping {r.url}")

        return collected

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self):
        return {"queries": len(self.history), "cache": len(self.cache)}
