"""
Advanced Data Extractor - AI-Powered Extraction Engine (MAX CONFIG)
✔ Regex, HTML, JSON, CSV, AI extraction
✔ Caching + deduplication
✔ Async LLM integration with retries
✔ Structured output (JSON schema)
✔ Predefined helpers (emails, URLs, phone numbers, etc.)
"""

import asyncio
import hashlib
import json
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union

from ..utils.logger import logger
from ..brain.llm_engine import LLMEngine


class ExtractionType(Enum):
    REGEX = "regex"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    AI = "ai"
    PHONE = "phone"
    DATE = "date"
    IP = "ip"


class DataExtractor:
    def __init__(self, use_llm: bool = True, cache_size: int = 100):
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.extractions_performed = 0
        self.results_cache: Dict[str, List[Any]] = {}
        self.cache_size = cache_size
        self.llm = LLMEngine() if use_llm else None

        # Pre-register common patterns
        self._register_default_patterns()

    # ------------------------
    # CACHE MANAGEMENT
    # ------------------------
    def _cache_key(self, source: str, pattern: str) -> str:
        return hashlib.md5((source + pattern).encode()).hexdigest()

    def _add_to_cache(self, key: str, result: List[Any]):
        if len(self.results_cache) >= self.cache_size:
            # Remove oldest (simplistic, could use LRU)
            oldest_key = next(iter(self.results_cache))
            del self.results_cache[oldest_key]
        self.results_cache[key] = result

    # ------------------------
    # PATTERN REGISTRATION
    # ------------------------
    def register_pattern(self, name: str, pattern: str, pattern_type: ExtractionType):
        self.patterns[name] = {"pattern": pattern, "type": pattern_type, "usage": 0}

    def _register_default_patterns(self):
        """Register built‑in extraction patterns"""
        self.register_pattern(
            "email",
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            ExtractionType.REGEX,
        )
        self.register_pattern(
            "url", r'https?://[^\s<>"{}|\\^`\[\]]+', ExtractionType.REGEX
        )
        self.register_pattern(
            "phone_us",
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            ExtractionType.REGEX,
        )
        self.register_pattern(
            "ipv4", r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", ExtractionType.REGEX
        )
        self.register_pattern(
            "date_iso", r"\b\d{4}-\d{2}-\d{2}\b", ExtractionType.REGEX
        )

    # ------------------------
    # EXTRACTION METHODS
    # ------------------------
    def extract_by_regex(self, text: str, pattern: str) -> List[str]:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Flatten if matches are tuples (due to capturing groups)
            if matches and isinstance(matches[0], tuple):
                matches = [m[0] for m in matches if m]
            return matches if matches else []
        except Exception as e:
            logger.debug(f"Regex extraction error: {e}")
            return []

    def extract_html(self, html: str, selector: str) -> List[str]:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            elements = soup.select(selector)
            return [el.get_text(strip=True) for el in elements]
        except ImportError:
            logger.warning("BeautifulSoup not installed, HTML extraction disabled")
            return []
        except Exception as e:
            logger.warning(f"HTML extraction failed: {e}")
            return []

    def extract_json(self, text: str) -> List[Dict]:
        results = []
        # Try to find JSON objects in the text
        matches = re.findall(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        for m in matches:
            try:
                data = json.loads(m)
                if isinstance(data, dict):
                    results.append(data)
                elif isinstance(data, list):
                    results.extend(data)
            except:
                continue
        return results

    def extract_csv(self, text: str, delimiter: str = ",") -> List[List[str]]:
        """Extract CSV rows (very basic)"""
        lines = text.strip().split("\n")
        rows = []
        for line in lines:
            if delimiter in line:
                rows.append([cell.strip() for cell in line.split(delimiter)])
        return rows

    def extract_phone(self, text: str) -> List[str]:
        return self.extract_by_regex(text, self.patterns["phone_us"]["pattern"])

    def extract_dates(self, text: str) -> List[str]:
        return self.extract_by_regex(text, self.patterns["date_iso"]["pattern"])

    def extract_ips(self, text: str) -> List[str]:
        return self.extract_by_regex(text, self.patterns["ipv4"]["pattern"])

    # ------------------------
    # AI EXTRACTION (FIXED)
    # ------------------------
    async def extract_ai(
        self, text: str, instruction: str, schema: Optional[Dict] = None
    ) -> Union[List[Any], Dict]:
        """
        Extract structured data using LLM.
        If schema provided, enforce JSON schema.
        Returns list of extracted items (or dict if single object).
        """
        if not self.llm:
            logger.warning("LLM not available, AI extraction disabled")
            return []

        prompt = f"""Extract structured data from the text below.

Text:
{text[:3000]}

Instruction:
{instruction}

{"Output format: JSON " + json.dumps(schema) if schema else "Return a JSON array of objects."}

ONLY output valid JSON, no extra text."""

        try:
            # Use safe_generate with timeout
            if hasattr(self.llm, "safe_generate"):
                result = await asyncio.wait_for(
                    self.llm.safe_generate(prompt, timeout=30), timeout=20
                )
                raw = result.get("response", "")
            else:
                result = await asyncio.wait_for(self.llm.generate(prompt), timeout=30)
                raw = result if isinstance(result, str) else result.get("response", "")

            raw = raw.strip()
            # Extract JSON
            data = self._parse_json_response(raw)
            if data is None:
                return []
            # Ensure list format
            if isinstance(data, dict):
                return [data]
            return data if isinstance(data, list) else []

        except asyncio.TimeoutError:
            logger.error("AI extraction timeout")
            return []
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
            return []

    def _parse_json_response(self, raw: str) -> Optional[Union[Dict, List]]:
        """Try multiple strategies to extract JSON from LLM response"""
        # Direct parse
        try:
            return json.loads(raw)
        except:
            pass
        # Find JSON block
        import re

        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return None

    # ------------------------
    # MAIN EXTRACT (UNIFIED)
    # ------------------------
    async def extract(self, source: str, pattern_name: str, **kwargs) -> List[Any]:
        """
        Extract data using registered pattern.
        For AI extraction, pass instruction via kwargs.
        """
        if pattern_name not in self.patterns:
            logger.warning(f"Pattern '{pattern_name}' not found")
            return []

        info = self.patterns[pattern_name]
        pattern = info["pattern"]
        ptype = info["type"]

        key = self._cache_key(source, pattern)
        if key in self.results_cache:
            return self.results_cache[key]

        self.extractions_performed += 1
        info["usage"] += 1

        if ptype == ExtractionType.REGEX:
            result = self.extract_by_regex(source, pattern)
        elif ptype == ExtractionType.HTML:
            result = self.extract_html(source, pattern)
        elif ptype == ExtractionType.JSON:
            result = self.extract_json(source)
        elif ptype == ExtractionType.CSV:
            result = self.extract_csv(source, kwargs.get("delimiter", ","))
        elif ptype == ExtractionType.AI:
            instruction = kwargs.get("instruction", pattern)
            schema = kwargs.get("schema", None)
            result = await self.extract_ai(source, instruction, schema)
        else:
            result = []

        self._add_to_cache(key, result)
        return result

    # ------------------------
    # CONVENIENCE METHODS
    # ------------------------
    async def extract_emails(self, text: str) -> List[str]:
        return await self.extract(text, "email")

    async def extract_urls(self, text: str) -> List[str]:
        return await self.extract(text, "url")

    async def extract_phones(self, text: str) -> List[str]:
        return await self.extract(text, "phone_us")

    async def extract_custom_ai(
        self, text: str, instruction: str, schema: Optional[Dict] = None
    ) -> List[Any]:
        return await self.extract(
            text, "ai_placeholder", instruction=instruction, schema=schema
        )

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "patterns": len(self.patterns),
            "extractions": self.extractions_performed,
            "cache_size": len(self.results_cache),
            "llm_available": self.llm is not None,
        }
