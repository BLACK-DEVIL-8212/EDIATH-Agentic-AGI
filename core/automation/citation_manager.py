"""
Advanced Citation Manager - AI Knowledge Tracking (GGUF Edition)
✔ Vector search (FAISS with GGUF embeddings)
✔ Persistent storage (SQLite/JSON)
✔ Citation formatting (MLA, APA, Chicago)
✔ Reliability scoring (domain + content quality)
✔ Metadata extraction (authors, date, keywords)
✔ Memory integration (EDIATH)
✔ Async support
✔ GGUF model for embeddings
"""

import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3

import numpy as np

from ..utils.logger import logger

# GGUF Model for embeddings
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Optional advanced imports
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class Citation:
    """Individual citation with rich metadata and formatting"""

    def __init__(
        self,
        title: str,
        authors: List[str],
        url: str,
        source: str,
        content: Optional[str] = None,
        publication_date: Optional[str] = None,
        publisher: Optional[str] = None,
        doi: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ):
        self.id = self._generate_id(title, url)
        self.title = title
        self.authors = authors
        self.url = url
        self.source = source
        self.content = content
        self.publication_date = publication_date or datetime.now().strftime("%Y-%m-%d")
        self.publisher = publisher or source
        self.doi = doi
        self.keywords = keywords or []
        self.accessed_date = datetime.now().isoformat()
        self.created_at = datetime.now()
        self.reliability_score = self._calculate_reliability()
        self.embedding: Optional[np.ndarray] = None

    def _generate_id(self, title: str, url: str) -> str:
        return hashlib.sha256(f"{title}{url}".encode()).hexdigest()[:16]

    def _calculate_reliability(self) -> float:
        """Advanced reliability scoring based on domain, content, and metadata"""
        score = 0.6  # base

        # Domain-based scoring
        if self.url:
            domain = self.url.lower()
            if any(tld in domain for tld in [".gov", ".edu"]):
                score = 0.95
            elif "wikipedia" in domain:
                score = 0.70
            elif "scholar.google" in domain:
                score = 0.85
            elif "arxiv.org" in domain:
                score = 0.80
            elif "medium.com" in domain or "blog" in domain:
                score = 0.50
            elif "news" in domain:
                score = 0.65

        # Author presence boosts
        if self.authors and len(self.authors) > 0:
            score += 0.05

        # DOI presence boosts
        if self.doi:
            score += 0.10

        # Content length (proxy for depth)
        if self.content and len(self.content) > 1000:
            score += 0.05

        # Keyword richness
        if len(self.keywords) >= 3:
            score += 0.05

        return min(1.0, score)

    def format_citation(self, style: str = "apa") -> str:
        """Generate formatted citation in MLA, APA, or Chicago style"""
        authors_str = ", ".join(self.authors) if self.authors else "Unknown Author"
        year = self.publication_date[:4] if self.publication_date else "n.d."

        if style.lower() == "apa":
            return f"{authors_str} ({year}). {self.title}. {self.publisher}. Retrieved from {self.url}"
        elif style.lower() == "mla":
            return (
                f'{authors_str}. "{self.title}." {self.publisher}, {year}, {self.url}'
            )
        elif style.lower() == "chicago":
            return f'{authors_str}, "{self.title}," {self.publisher}, accessed {self.accessed_date[:10]}, {self.url}'
        else:
            return f"{authors_str}: {self.title} ({year}) - {self.url}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "source": self.source,
            "content": self.content[:500] if self.content else None,
            "publication_date": self.publication_date,
            "publisher": self.publisher,
            "doi": self.doi,
            "keywords": self.keywords,
            "reliability": self.reliability_score,
            "accessed_date": self.accessed_date,
            "created_at": self.created_at.isoformat(),
        }


class CitationManager:
    """
    Maximum configuration citation manager with GGUF vector search,
    persistent storage, and memory integration.
    """

    def __init__(
        self,
        storage_path: str = "data/citations.db",
        model_path: str = "./models/EDIATH-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
    ):
        """
        Initialize Citation Manager with GGUF model for embeddings

        Args:
            storage_path: Path to SQLite database
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads for inference
        """
        self.citations: Dict[str, Citation] = {}
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # GGUF Model configuration
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.use_gguf = LLAMA_AVAILABLE
        self.embedding_dimension = 384  # Will be updated from model
        self.gguf_model = None

        # Vector search setup with GGUF
        self._faiss_index = None
        self._citation_ids = []  # order matching FAISS index
        self._use_vector = FAISS_AVAILABLE

        if self._use_vector:
            self._init_gguf_model()
            self._init_faiss()
            if self.gguf_model:
                logger.info("✅ GGUF vector search enabled for citations")
            else:
                logger.warning("GGUF model not loaded, vector search disabled")
                self._use_vector = False
        else:
            logger.info("Vector search disabled (install faiss-cpu)")

        # Load existing citations
        self._load_from_disk()

    def _init_gguf_model(self):
        """Initialize GGUF model for embeddings"""
        if not self.use_gguf:
            return

        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.warning(f"GGUF model not found at {model_path}")
                self.use_gguf = False
                return

            logger.info(f"Loading GGUF model: {self.model_path}")
            self.gguf_model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                embedding=True,
            )

            # Get actual embedding dimension
            try:
                test_embedding = self.gguf_model.embed("test")
                self.embedding_dimension = len(test_embedding)
                logger.info(
                    f"GGUF model loaded, embedding dimension: {self.embedding_dimension}"
                )
            except Exception as e:
                logger.warning(f"Could not determine embedding dimension: {e}")

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            self.use_gguf = False

    def _init_faiss(self, dimension: int = None):
        """Initialize FAISS index (cosine similarity)"""
        dim = dimension or self.embedding_dimension
        self._faiss_index = faiss.IndexFlatIP(
            dim
        )  # Inner product (cosine after normalization)

    def _normalize_embedding(self, emb: np.ndarray) -> np.ndarray:
        """L2 normalize for cosine similarity"""
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using GGUF model"""
        if not self._use_vector or not self.gguf_model:
            return None

        try:
            # Truncate long text
            if len(text) > self.n_ctx * 4:
                text = text[: self.n_ctx * 4]

            embedding = self.gguf_model.embed(text)
            emb_array = np.array(embedding, dtype=np.float32)
            return self._normalize_embedding(emb_array)

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def _add_to_vector_index(self, citation: Citation):
        """Add citation embedding to FAISS index"""
        if not self._use_vector or not citation.content:
            return
        text_for_embedding = f"{citation.title} {citation.content[:1000]}"
        emb = self._get_embedding(text_for_embedding)
        if emb is not None:
            self._faiss_index.add(emb.reshape(1, -1))
            self._citation_ids.append(citation.id)

    def _load_from_disk(self):
        """Load citations from SQLite database"""
        if not self.storage_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    url TEXT,
                    source TEXT,
                    content TEXT,
                    publication_date TEXT,
                    publisher TEXT,
                    doi TEXT,
                    keywords TEXT,
                    reliability REAL,
                    accessed_date TEXT,
                    created_at TEXT
                )
            """)
            cursor.execute("SELECT * FROM citations")
            rows = cursor.fetchall()
            for row in rows:
                citation = Citation(
                    title=row[1],
                    authors=json.loads(row[2]) if row[2] else [],
                    url=row[3],
                    source=row[4],
                    content=row[5],
                    publication_date=row[6],
                    publisher=row[7],
                    doi=row[8],
                    keywords=json.loads(row[9]) if row[9] else [],
                )
                citation.id = row[0]
                citation.reliability_score = row[10]
                citation.accessed_date = row[11]
                citation.created_at = datetime.fromisoformat(row[12])
                self.citations[citation.id] = citation
                # Add to vector index after load (if content exists)
                if self._use_vector and citation.content:
                    self._add_to_vector_index(citation)

            conn.close()
            logger.info(f"Loaded {len(self.citations)} citations from storage")
        except Exception as e:
            logger.warning(f"Failed to load citations: {e}")

    def _save_to_disk(self):
        """Save citations to SQLite database"""
        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    url TEXT,
                    source TEXT,
                    content TEXT,
                    publication_date TEXT,
                    publisher TEXT,
                    doi TEXT,
                    keywords TEXT,
                    reliability REAL,
                    accessed_date TEXT,
                    created_at TEXT
                )
            """)
            for c in self.citations.values():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO citations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                    (
                        c.id,
                        c.title,
                        json.dumps(c.authors),
                        c.url,
                        c.source,
                        c.content,
                        c.publication_date,
                        c.publisher,
                        c.doi,
                        json.dumps(c.keywords),
                        c.reliability_score,
                        c.accessed_date,
                        c.created_at.isoformat(),
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save citations: {e}")

    # ------------------------
    # PUBLIC API
    # ------------------------
    def add_citation(
        self,
        title: str,
        authors: List[str],
        url: str,
        source: str,
        content: Optional[str] = None,
        publication_date: Optional[str] = None,
        publisher: Optional[str] = None,
        doi: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> str:
        """Add a new citation (deduplicated)"""
        citation = Citation(
            title=title,
            authors=authors,
            url=url,
            source=source,
            content=content,
            publication_date=publication_date,
            publisher=publisher,
            doi=doi,
            keywords=keywords,
        )
        if citation.id in self.citations:
            return citation.id  # already exists

        self.citations[citation.id] = citation
        self._add_to_vector_index(citation)
        self._save_to_disk()
        logger.info(f"📚 Citation added: {title}")
        return citation.id

    def extract_from_browser(self, url: str, html_content: str) -> str:
        """Auto-extract metadata from HTML content"""
        title = self._extract_title(html_content)
        keywords = self._extract_keywords(html_content)
        authors = self._extract_authors(html_content)
        return self.add_citation(
            title=title,
            authors=authors,
            url=url,
            source="web",
            content=html_content[:5000],
            keywords=keywords,
        )

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else "Untitled"

    def _extract_keywords(self, html: str) -> List[str]:
        match = re.search(
            r'<meta name=["\']keywords["\'] content=["\'](.*?)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            return [kw.strip() for kw in match.group(1).split(",")]
        return []

    def _extract_authors(self, html: str) -> List[str]:
        patterns = [
            r'<meta name=["\']author["\'] content=["\'](.*?)["\']',
            r'<a rel=["\']author["\']>(.*?)</a>',
        ]
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return [match.group(1).strip()]
        return []

    async def search(
        self, query: str, top_k: int = 5, use_vector: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search citations by keyword or vector similarity using GGUF embeddings.
        """
        if (
            use_vector
            and self._use_vector
            and self._faiss_index
            and self._faiss_index.ntotal > 0
        ):
            return await self._vector_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    async def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Semantic search using GGUF embeddings and FAISS"""
        query_emb = self._get_embedding(query)
        if query_emb is None:
            return self._keyword_search(query, top_k)

        scores, indices = self._faiss_index.search(
            query_emb.reshape(1, -1), min(top_k, self._faiss_index.ntotal)
        )
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self._citation_ids):
                cid = self._citation_ids[idx]
                citation = self.citations.get(cid)
                if citation:
                    item = citation.to_dict()
                    item["similarity"] = float(score)
                    item["embedding_model"] = "GGUF"
                    results.append(item)
        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        query_lower = query.lower()
        scored = []
        for citation in self.citations.values():
            score = 0
            if query_lower in citation.title.lower():
                score += 10
            if any(query_lower in kw.lower() for kw in citation.keywords):
                score += 5
            if citation.content and query_lower in citation.content.lower():
                score += 2
            if score > 0:
                scored.append((score, citation))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c.to_dict() for _, c in scored[:top_k]]

    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a citation by ID"""
        c = self.citations.get(citation_id)
        return c.to_dict() if c else None

    def get_top_sources(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return highest reliability citations"""
        sorted_cits = sorted(
            self.citations.values(), key=lambda c: c.reliability_score, reverse=True
        )
        return [c.to_dict() for c in sorted_cits[:limit]]

    def format_citation(self, citation_id: str, style: str = "apa") -> Optional[str]:
        """Get formatted citation string"""
        c = self.citations.get(citation_id)
        return c.format_citation(style) if c else None

    def delete_citation(self, citation_id: str) -> bool:
        """Remove a citation (also from vector index if needed)"""
        if citation_id in self.citations:
            del self.citations[citation_id]
            self._save_to_disk()
            self._rebuild_vector_index()
            return True
        return False

    def _rebuild_vector_index(self):
        """Rebuild FAISS index from current citations using GGUF embeddings"""
        if not self._use_vector:
            return
        self._init_faiss(self.embedding_dimension)
        self._citation_ids.clear()
        for c in self.citations.values():
            if c.content:
                self._add_to_vector_index(c)
        logger.info("Vector index rebuilt with GGUF embeddings")

    def export(self, filepath: str):
        """Export all citations to JSON"""
        data = [c.to_dict() for c in self.citations.values()]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_data(self, filepath: str):
        """Import citations from JSON"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.add_citation(
                title=item["title"],
                authors=item.get("authors", []),
                url=item.get("url", ""),
                source=item.get("source", "imported"),
                content=item.get("content"),
                publication_date=item.get("publication_date"),
                publisher=item.get("publisher"),
                doi=item.get("doi"),
                keywords=item.get("keywords", []),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the citation database"""
        total = len(self.citations)
        avg_reliability = sum(
            c.reliability_score for c in self.citations.values()
        ) / max(1, total)
        return {
            "total": total,
            "avg_reliability": round(avg_reliability, 3),
            "vector_enabled": self._use_vector,
            "embedding_model": "GGUF",
            "model_path": str(self.model_path) if self.use_gguf else None,
            "embedding_dimension": self.embedding_dimension,
            "vector_size": self._faiss_index.ntotal if self._faiss_index else 0,
            "storage_path": str(self.storage_path),
        }

    async def integrate_with_memory(self, memory_manager):
        """Push important citations into EDIATH's memory system"""
        top_cits = self.get_top_sources(limit=10)
        for cit in top_cits:
            await memory_manager.store(
                {
                    "type": "citation",
                    "title": cit["title"],
                    "reliability": cit["reliability"],
                    "summary": f"Source: {cit['title']} - {cit['url']}",
                    "timestamp": datetime.now().isoformat(),
                    "embedding_model": "GGUF",
                }
            )
        logger.info(f"Integrated {len(top_cits)} citations into memory")


# Integration wrapper for EDIATH
class CitationManagerWrapper:
    """Wrapper class to integrate CitationManager with EDIATH"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.citation_manager = CitationManager(
            storage_path=config.get("storage_path", "data/citations.db"),
            model_path=config.get("model_path", "./models/EDIATH-q4_k_m.gguf"),
            n_ctx=config.get("n_ctx", 2048),
            n_threads=config.get("n_threads", 4),
        )
        self.agent_type = "citation_manager"
        self.capabilities = [
            "add_citation",
            "search_citations",
            "format_citation",
            "get_top_sources",
            "export_citations",
            "integrate_with_memory",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a citation management request"""
        operation = request.get("operation")

        if operation == "add":
            citation_id = self.citation_manager.add_citation(
                title=request.get("title"),
                authors=request.get("authors", []),
                url=request.get("url"),
                source=request.get("source"),
                content=request.get("content"),
                publication_date=request.get("publication_date"),
                publisher=request.get("publisher"),
                doi=request.get("doi"),
                keywords=request.get("keywords", []),
            )
            return {
                "success": True,
                "citation_id": citation_id,
                "message": "Citation added successfully",
            }

        elif operation == "search":
            results = await self.citation_manager.search(
                query=request.get("query"),
                top_k=request.get("top_k", 5),
                use_vector=request.get("use_vector", True),
            )
            return {"success": True, "results": results, "total": len(results)}

        elif operation == "format":
            formatted = self.citation_manager.format_citation(
                citation_id=request.get("citation_id"),
                style=request.get("style", "apa"),
            )
            return (
                {"success": True, "formatted_citation": formatted}
                if formatted
                else {"success": False, "error": "Citation not found"}
            )

        elif operation == "top_sources":
            sources = self.citation_manager.get_top_sources(
                limit=request.get("limit", 5)
            )
            return {"success": True, "sources": sources}

        elif operation == "export":
            self.citation_manager.export(request.get("filepath"))
            return {
                "success": True,
                "message": f'Exported to {request.get("filepath")}',
            }

        elif operation == "stats":
            return self.citation_manager.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "CitationManager",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.citation_manager.get_stats(),
            "embedding_model": "GGUF",
        }


# Example usage
async def test_citation_manager():
    """Test the citation manager with GGUF"""

    config = {
        "model_path": "./models/EDIATH-q4_k_m.gguf",
        "storage_path": "data/test_citations.db",
        "n_ctx": 2048,
        "n_threads": 4,
    }

    manager = CitationManager(**config)
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=== Citation Manager Test with GGUF ===\n")
    stats = manager.get_stats()
    logger.info("Vector Enabled: %s", stats.get("vector_enabled"))
    logger.info("Embedding Model: %s", stats.get("embedding_model"))
    logger.info("Model Path: %s", stats.get("model_path", "N/A"))
    logger.info("Embedding Dimension: %s", stats.get("embedding_dimension"))

    # Add a test citation
    logger.info("\n1. Adding Citation...")
    citation_id = manager.add_citation(
        title="Artificial Intelligence in Modern Computing",
        authors=["John Smith", "Jane Doe"],
        url="https://example.com/ai-paper",
        source="Academic Journal",
        content="This paper discusses the latest advances in artificial intelligence and machine learning, with a focus on transformer architectures and large language models.",
        publication_date="2024-01-15",
        publisher="AI Research Press",
        keywords=["AI", "Machine Learning", "Transformers"],
    )
    logger.info("   Citation ID: %s", citation_id)

    # Search citations
    logger.info("\n2. Searching Citations...")
    results = await manager.search("artificial intelligence advances", top_k=3)
    for r in results:
        logger.info("   Score %.3f: %s", r.get("similarity", 0), r.get("title"))

    # Format citation
    logger.info("\n3. Formatting Citation...")
    formatted = manager.format_citation(citation_id, style="apa")
    logger.info("   APA: %s", formatted)

    formatted = manager.format_citation(citation_id, style="mla")
    logger.info("   MLA: %s", formatted)

    # Get stats
    logger.info("\n4. Statistics...")
    stats = manager.get_stats()
    logger.info("   Total citations: %s", stats.get("total"))
    logger.info("   Average reliability: %s", stats.get("avg_reliability"))
    logger.info("   Vector size: %s", stats.get("vector_size"))

    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_citation_manager())
