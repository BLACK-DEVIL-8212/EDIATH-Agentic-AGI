"""
<<<<<<< HEAD
Advanced Citation Manager - Ultimate Edition (GGUF + Full Features)
✔ Vector search (FAISS with GGUF embeddings)
✔ Persistent storage (SQLite/JSON)
✔ Citation formatting (MLA, APA, Chicago, Harvard, IEEE, Vancouver)
✔ Reliability scoring (domain + content quality + peer review)
✔ Metadata extraction (authors, date, keywords, references)
✔ Memory integration (EDIATH)
✔ Async support
✔ GGUF model for embeddings
✔ Full CRUD operations
✔ Batch processing
✔ Export/Import (JSON, CSV, BibTeX, RIS, EndNote)
✔ Duplicate detection
✔ Reference extraction
✔ Citation graph/network
✔ Auto-tagging
✔ Summarization
✔ Cross-referencing
✔ Zotero/Mendeley integration
✔ Full-text search
✔ Citation recommendations
=======
Advanced Citation Manager - AI Knowledge Tracking (GGUF Edition)
✔ Vector search (FAISS with GGUF embeddings)
✔ Persistent storage (SQLite/JSON)
✔ Citation formatting (MLA, APA, Chicago)
✔ Reliability scoring (domain + content quality)
✔ Metadata extraction (authors, date, keywords)
✔ Memory integration (EDIATH)
✔ Async support
✔ GGUF model for embeddings
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
"""

import asyncio
import hashlib
import json
import re
<<<<<<< HEAD
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import sqlite3
import pickle

import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

=======
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3

import numpy as np

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
from ..utils.logger import logger

# GGUF Model for embeddings
try:
    from llama_cpp import Llama
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Optional advanced imports
try:
    import faiss
<<<<<<< HEAD
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

<<<<<<< HEAD
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CitationStyle(Enum):
    """Supported citation styles"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    AMA = "ama"
    ACS = "acs"


class ReferenceType(Enum):
    """Types of references"""
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    WEBPAGE = "webpage"
    REPORT = "report"
    NEWSPAPER_ARTICLE = "newspaper_article"
    MAGAZINE_ARTICLE = "magazine_article"
    PATENT = "patent"
    VIDEO = "video"
    PODCAST = "podcast"
    SOCIAL_MEDIA = "social_media"
    DATASET = "dataset"
    SOFTWARE = "software"


class Citation:
    """Individual citation with rich metadata and formatting (Extended)"""
    
=======

class Citation:
    """Individual citation with rich metadata and formatting"""

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
        isbn: Optional[str] = None,
        issn: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        reference_type: ReferenceType = ReferenceType.WEBPAGE,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        edition: Optional[int] = None,
        series: Optional[str] = None,
        institution: Optional[str] = None,
        conference: Optional[str] = None,
        abstract: Optional[str] = None,
        language: str = "en",
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        rating: Optional[float] = None,
        read_status: bool = False,
        important: bool = False,
        references: Optional[List[str]] = None,
        citations: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ):
        self.id = self._generate_id(title, url, doi)
=======
        keywords: Optional[List[str]] = None,
    ):
        self.id = self._generate_id(title, url)
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        self.title = title
        self.authors = authors
        self.url = url
        self.source = source
        self.content = content
        self.publication_date = publication_date or datetime.now().strftime("%Y-%m-%d")
        self.publisher = publisher or source
        self.doi = doi
<<<<<<< HEAD
        self.isbn = isbn
        self.issn = issn
        self.pmid = pmid
        self.arxiv_id = arxiv_id
        self.keywords = keywords or []
        self.reference_type = reference_type
        self.volume = volume
        self.issue = issue
        self.pages = pages
        self.edition = edition
        self.series = series
        self.institution = institution
        self.conference = conference
        self.abstract = abstract
        self.language = language
        self.tags = tags or []
        self.notes = notes
        self.rating = rating
        self.read_status = read_status
        self.important = important
        self.references = references or []
        self.citations = citations or []
        self.categories = categories or []
        self.accessed_date = datetime.now().isoformat()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.reliability_score = self._calculate_reliability()
        self.embedding: Optional[np.ndarray] = None
        self.summary: Optional[str] = None
        
    def _generate_id(self, title: str, url: str, doi: Optional[str] = None) -> str:
        """Generate unique ID for citation"""
        unique_string = f"{title}{url}{doi or ''}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _calculate_reliability(self) -> float:
        """Advanced reliability scoring with multiple factors"""
        score = 0.6  # base
        
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
            elif "nature.com" in domain or "science.org" in domain:
                score = 0.95
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            elif "medium.com" in domain or "blog" in domain:
                score = 0.50
            elif "news" in domain:
                score = 0.65
<<<<<<< HEAD
        
        # Publication type scoring
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            score += 0.15
        elif self.reference_type == ReferenceType.BOOK:
            score += 0.10
        elif self.reference_type == ReferenceType.CONFERENCE_PAPER:
            score += 0.10
        elif self.reference_type == ReferenceType.THESIS:
            score += 0.05
        
        # Author presence boosts
        if self.authors and len(self.authors) > 0:
            score += 0.05
            if len(self.authors) >= 3:
                score += 0.03
        
        # DOI/ISBN/PMID presence
        if self.doi:
            score += 0.10
        if self.isbn:
            score += 0.05
        if self.pmid:
            score += 0.10
        
        # Content metrics
        if self.content and len(self.content) > 1000:
            score += 0.05
        if self.abstract:
            score += 0.05
        
        # Metadata completeness
        metadata_count = sum([
            1 for x in [self.volume, self.issue, self.pages, self.edition] if x
        ])
        score += metadata_count * 0.02
        
        # Keyword richness
        if len(self.keywords) >= 3:
            score += 0.05
        if len(self.tags) >= 3:
            score += 0.03
        
        # Rating
        if self.rating:
            score += (self.rating - 3) * 0.05  # rating > 3 increases score
        
        return min(1.0, score)
    
    def format_citation(self, style: Union[str, CitationStyle] = "apa") -> str:
        """Generate formatted citation in various styles"""
        if isinstance(style, str):
            style = CitationStyle(style.lower())
        
        authors_str = self._format_authors(style)
        year = self.publication_date[:4] if self.publication_date else "n.d."
        title_str = self._format_title(style)
        
        if style == CitationStyle.APA:
            return self._format_apa(authors_str, year, title_str)
        elif style == CitationStyle.MLA:
            return self._format_mla(authors_str, title_str, year)
        elif style == CitationStyle.CHICAGO:
            return self._format_chicago(authors_str, title_str, year)
        elif style == CitationStyle.HARVARD:
            return self._format_harvard(authors_str, year, title_str)
        elif style == CitationStyle.IEEE:
            return self._format_ieee(authors_str, title_str, year)
        elif style == CitationStyle.VANCOUVER:
            return self._format_vancouver(authors_str, title_str, year)
        elif style == CitationStyle.AMA:
            return self._format_ama(authors_str, title_str, year)
        elif style == CitationStyle.ACS:
            return self._format_acs(authors_str, title_str, year)
        else:
            return f"{authors_str}: {title_str} ({year}) - {self.url}"
    
    def _format_authors(self, style: CitationStyle) -> str:
        """Format authors based on citation style"""
        if not self.authors:
            return "Unknown Author"
        
        if style in [CitationStyle.APA, CitationStyle.HARVARD, CitationStyle.AMA]:
            if len(self.authors) == 1:
                return f"{self.authors[0]}"
            elif len(self.authors) == 2:
                return f"{self.authors[0]} & {self.authors[1]}"
            elif len(self.authors) <= 7:
                return f"{', '.join(self.authors[:-1])}, & {self.authors[-1]}"
            else:
                return f"{', '.join(self.authors[:6])}, ... {self.authors[-1]}"
        
        elif style == CitationStyle.MLA:
            if len(self.authors) == 1:
                return self.authors[0]
            else:
                return f"{self.authors[0]}, et al."
        
        elif style == CitationStyle.IEEE:
            return f"{', '.join(self.authors)}"
        
        elif style == CitationStyle.VANCOUVER:
            if len(self.authors) <= 6:
                return f"{', '.join(self.authors)}"
            else:
                return f"{', '.join(self.authors[:3])}, et al."
        
        else:
            return f"{', '.join(self.authors)}"
    
    def _format_title(self, style: CitationStyle) -> str:
        """Format title based on citation style"""
        if style == CitationStyle.APA:
            return self.title
        elif style == CitationStyle.MLA:
            return f'"{self.title}"'
        elif style == CitationStyle.CHICAGO:
            return f'"{self.title}"'
        else:
            return self.title
    
    def _format_apa(self, authors: str, year: str, title: str) -> str:
        """APA 7th edition format"""
        parts = [f"{authors} ({year}).", title]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            journal_part = f"{self.source}"
            if self.volume:
                journal_part += f", {self.volume}"
                if self.issue:
                    journal_part += f"({self.issue})"
            if self.pages:
                journal_part += f", {self.pages}"
            parts.append(journal_part)
        
        elif self.reference_type == ReferenceType.BOOK:
            parts.append(self.publisher)
            if self.doi:
                parts.append(f"https://doi.org/{self.doi}")
        
        else:
            parts.append(self.source)
        
        if self.doi and self.reference_type != ReferenceType.JOURNAL_ARTICLE:
            parts.append(f"https://doi.org/{self.doi}")
        elif self.url:
            parts.append(f"Retrieved from {self.url}")
        
        return " ".join(parts)
    
    def _format_mla(self, authors: str, title: str, year: str) -> str:
        """MLA 9th edition format"""
        parts = [f"{authors}."]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            parts.append(f'"{title}."')
            parts.append(self.source)
            if self.volume:
                parts.append(f"vol. {self.volume}")
            if self.issue:
                parts.append(f"no. {self.issue}")
            if self.pages:
                parts.append(f"pp. {self.pages}")
            parts.append(f"({year})")
        else:
            parts.append(f'"{title}."')
            parts.append(self.source)
            parts.append(year)
        
        if self.doi:
            parts.append(f"doi:{self.doi}")
        elif self.url:
            parts.append(self.url)
        
        return " ".join(parts)
    
    def _format_chicago(self, authors: str, title: str, year: str) -> str:
        """Chicago 17th edition format"""
        parts = [f"{authors}."]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            parts.append(f'"{title}."')
            parts.append(self.source)
            if self.volume:
                parts.append(f"{self.volume}")
            if self.issue:
                parts.append(f"no. {self.issue}")
            parts.append(f"({year}):")
            if self.pages:
                parts.append(self.pages)
        else:
            parts.append(f'"{title}."')
            parts.append(self.source)
            parts.append(year)
        
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        elif self.url:
            parts.append(self.url)
        
        return " ".join(parts)
    
    def _format_harvard(self, authors: str, year: str, title: str) -> str:
        """Harvard format"""
        parts = [f"{authors} ({year})"]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            parts.append(f"'{title}'")
            parts.append(self.source)
            if self.volume:
                parts.append(self.volume)
            if self.issue:
                parts.append(f"({self.issue})")
            if self.pages:
                parts.append(f"pp. {self.pages}")
        else:
            parts.append(title)
            parts.append(self.source)
        
        if self.url:
            parts.append(f"Available at: {self.url}")
            parts.append(f"(Accessed: {self.accessed_date[:10]})")
        
        return " ".join(parts)
    
    def _format_ieee(self, authors: str, title: str, year: str) -> str:
        """IEEE format"""
        parts = [f"{authors}, "]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            parts.append(f'"{title},"')
            parts.append(self.source)
            if self.volume:
                parts.append(f"vol. {self.volume}")
            if self.issue:
                parts.append(f"no. {self.issue}")
            if self.pages:
                parts.append(f"pp. {self.pages}")
            parts.append(f"{year}.")
        else:
            parts.append(f'"{title},"')
            parts.append(self.source)
            parts.append(f"{year}.")
        
        if self.doi:
            parts.append(f"doi: {self.doi}")
        
        return " ".join(parts)
    
    def _format_vancouver(self, authors: str, title: str, year: str) -> str:
        """Vancouver format"""
        parts = [f"{authors}."]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            parts.append(title)
            parts.append(self.source)
            if self.publication_date:
                parts.append(year)
            if self.volume:
                parts.append(self.volume)
            if self.issue:
                parts.append(f"({self.issue})")
            if self.pages:
                parts.append(f":{self.pages}")
        else:
            parts.append(title)
            parts.append(self.source)
            parts.append(year)
        
        if self.doi:
            parts.append(f"doi:{self.doi}")
        
        return " ".join(parts)
    
    def _format_ama(self, authors: str, title: str, year: str) -> str:
        """AMA format"""
        parts = [f"{authors}."]
        
        if self.reference_type == ReferenceType.JOURNAL_ARTICLE:
            parts.append(title)
            parts.append(self.source)
            if self.publication_date:
                parts.append(year)
            if self.volume:
                parts.append(self.volume)
            if self.issue:
                parts.append(f"({self.issue})")
            if self.pages:
                parts.append(f":{self.pages}")
        else:
            parts.append(title)
            parts.append(self.source)
            parts.append(year)
        
        if self.url:
            parts.append(self.url)
        
        return " ".join(parts)
    
    def _format_acs(self, authors: str, title: str, year: str) -> str:
        """ACS format"""
        parts = [f"{authors}."]
        parts.append(title)
        parts.append(self.source)
        
        if self.volume:
            parts.append(self.volume)
        if self.issue:
            parts.append(f"({self.issue})")
        if self.pages:
            parts.append(self.pages)
        
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        
        return " ".join(parts)
    
    def generate_bibtex(self) -> str:
        """Generate BibTeX entry"""
        citation_type = "article"
        if self.reference_type == ReferenceType.BOOK:
            citation_type = "book"
        elif self.reference_type == ReferenceType.CONFERENCE_PAPER:
            citation_type = "inproceedings"
        elif self.reference_type == ReferenceType.THESIS:
            citation_type = "phdthesis"
        
        bibtex = f"@{citation_type}{{{self.id},\n"
        
        fields = {
            "title": self.title,
            "author": " and ".join(self.authors),
            "year": self.publication_date[:4] if self.publication_date else "n.d.",
            "journal": self.source if self.reference_type == ReferenceType.JOURNAL_ARTICLE else None,
            "publisher": self.publisher,
            "url": self.url,
            "doi": self.doi,
            "isbn": self.isbn,
            "volume": self.volume,
            "number": self.issue,
            "pages": self.pages,
        }
        
        for key, value in fields.items():
            if value:
                bibtex += f"  {key} = {{{value}}},\n"
        
        bibtex += "}\n"
        return bibtex
    
    def generate_ris(self) -> str:
        """Generate RIS entry"""
        ris = []
        
        type_map = {
            ReferenceType.JOURNAL_ARTICLE: "JOUR",
            ReferenceType.BOOK: "BOOK",
            ReferenceType.CONFERENCE_PAPER: "CONF",
            ReferenceType.THESIS: "THES",
            ReferenceType.WEBPAGE: "WEB",
        }
        
        ris.append(f"TY  - {type_map.get(self.reference_type, 'GEN')}")
        ris.append(f"TI  - {self.title}")
        
        for author in self.authors:
            ris.append(f"AU  - {author}")
        
        ris.append(f"PY  - {self.publication_date[:4] if self.publication_date else ''}")
        ris.append(f"UR  - {self.url}")
        
        if self.doi:
            ris.append(f"DO  - {self.doi}")
        
        ris.append(f"PB  - {self.publisher}")
        ris.append(f"KW  - {', '.join(self.keywords)}")
        ris.append(f"AB  - {self.abstract or ''}")
        ris.append(f"N2  - {self.content[:500] if self.content else ''}")
        ris.append("ER  -")
        
        return "\n".join(ris)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "source": self.source,
<<<<<<< HEAD
            "content": self.content[:1000] if self.content else None,
            "publication_date": self.publication_date,
            "publisher": self.publisher,
            "doi": self.doi,
            "isbn": self.isbn,
            "issn": self.issn,
            "pmid": self.pmid,
            "arxiv_id": self.arxiv_id,
            "keywords": self.keywords,
            "reference_type": self.reference_type.value,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "edition": self.edition,
            "series": self.series,
            "institution": self.institution,
            "conference": self.conference,
            "abstract": self.abstract,
            "language": self.language,
            "tags": self.tags,
            "notes": self.notes,
            "rating": self.rating,
            "read_status": self.read_status,
            "important": self.important,
            "references": self.references,
            "citations": self.citations,
            "categories": self.categories,
            "reliability": self.reliability_score,
            "accessed_date": self.accessed_date,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "summary": self.summary,
        }


@dataclass
class SearchFilter:
    """Search filters for citation queries"""
    keywords: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_reliability: Optional[float] = None
    reference_types: Optional[List[ReferenceType]] = None
    important_only: bool = False
    read_status: Optional[bool] = None
    min_rating: Optional[float] = None
    language: Optional[str] = None


class CitationGraph:
    """Citation relationship graph using NetworkX"""
    
    def __init__(self):
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        
    def add_citation(self, citation_id: str, references: List[str]):
        """Add citation relationships"""
        if not self.graph:
            return
        
        for ref_id in references:
            self.graph.add_edge(citation_id, ref_id)
    
    def get_citation_path(self, source_id: str, target_id: str) -> List[str]:
        """Find citation path between two papers"""
        if not self.graph:
            return []
        
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return []
    
    def get_most_cited(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get most cited papers"""
        if not self.graph:
            return []
        
        citations = defaultdict(int)
        for _, target in self.graph.edges():
            citations[target] += 1
        
        return sorted(citations.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_citation_clusters(self) -> List[List[str]]:
        """Find citation clusters"""
        if not self.graph:
            return []
        
        return [list(cluster) for cluster in nx.strongly_connected_components(self.graph)]
    
    def export_graph(self, filepath: str, format: str = "graphml"):
        """Export citation graph"""
        if not self.graph:
            return
        
        if format == "graphml":
            nx.write_graphml(self.graph, filepath)
        elif format == "gexf":
            nx.write_gexf(self.graph, filepath)
        elif format == "json":
            data = nx.node_link_data(self.graph)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


class CitationManager:
    """
    Ultimate Citation Manager with maximum features
    """
    
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def __init__(
        self,
        storage_path: str = "data/citations.db",
        model_path: str = "./models/EDIATH-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
<<<<<<< HEAD
        use_vector: bool = True,
        use_tfidf: bool = True,
        auto_backup: bool = True,
        backup_interval_hours: int = 24,
    ):
        """
        Initialize Citation Manager with GGUF model for embeddings
        
=======
    ):
        """
        Initialize Citation Manager with GGUF model for embeddings

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        Args:
            storage_path: Path to SQLite database
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads for inference
<<<<<<< HEAD
            use_vector: Enable vector search
            use_tfidf: Enable TF-IDF search fallback
            auto_backup: Enable automatic backups
            backup_interval_hours: Backup interval in hours
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        """
        self.citations: Dict[str, Citation] = {}
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
        self.auto_backup = auto_backup
        self.backup_interval_hours = backup_interval_hours
        self.last_backup = datetime.now()
        
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        # GGUF Model configuration
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.use_gguf = LLAMA_AVAILABLE
        self.embedding_dimension = 384  # Will be updated from model
        self.gguf_model = None
<<<<<<< HEAD
        
        # Vector search setup
        self.use_vector = use_vector and FAISS_AVAILABLE
        self._faiss_index = None
        self._citation_ids = []  # order matching FAISS index
        
        # TF-IDF fallback
        self.use_tfidf = use_tfidf and SKLEARN_AVAILABLE
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Citation graph
        self.citation_graph = CitationGraph()
        
        # Statistics and tracking
        self.search_history: List[Dict[str, Any]] = []
        self.import_history: List[Dict[str, Any]] = []
        
        # Initialize components
        if self.use_vector:
=======

        # Vector search setup with GGUF
        self._faiss_index = None
        self._citation_ids = []  # order matching FAISS index
        self._use_vector = FAISS_AVAILABLE

        if self._use_vector:
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            self._init_gguf_model()
            self._init_faiss()
            if self.gguf_model:
                logger.info("✅ GGUF vector search enabled for citations")
            else:
                logger.warning("GGUF model not loaded, vector search disabled")
<<<<<<< HEAD
                self.use_vector = False
        
        if self.use_tfidf:
            self._init_tfidf()
        
        # Load existing citations
        self._load_from_disk()
        
        # Start backup task if auto_backup enabled
        if auto_backup:
            asyncio.create_task(self._auto_backup_loop())
    
=======
                self._use_vector = False
        else:
            logger.info("Vector search disabled (install faiss-cpu)")

        # Load existing citations
        self._load_from_disk()

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def _init_gguf_model(self):
        """Initialize GGUF model for embeddings"""
        if not self.use_gguf:
            return
<<<<<<< HEAD
        
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.warning(f"GGUF model not found at {model_path}")
                self.use_gguf = False
                return
<<<<<<< HEAD
            
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            logger.info(f"Loading GGUF model: {self.model_path}")
            self.gguf_model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                embedding=True,
            )
<<<<<<< HEAD
            
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            # Get actual embedding dimension
            try:
                test_embedding = self.gguf_model.embed("test")
                self.embedding_dimension = len(test_embedding)
<<<<<<< HEAD
                logger.info(f"GGUF model loaded, embedding dimension: {self.embedding_dimension}")
            except Exception as e:
                logger.warning(f"Could not determine embedding dimension: {e}")
                
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            self.use_gguf = False
    
    def _init_faiss(self, dimension: int = None):
        """Initialize FAISS index (cosine similarity)"""
        dim = dimension or self.embedding_dimension
        self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def _normalize_embedding(self, emb: np.ndarray) -> np.ndarray:
        """L2 normalize for cosine similarity"""
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
<<<<<<< HEAD
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using GGUF model"""
        if not self.use_vector or not self.gguf_model:
            return None
        
        try:
            # Truncate long text
            if len(text) > self.n_ctx * 4:
                text = text[:self.n_ctx * 4]
            
            embedding = self.gguf_model.embed(text)
            emb_array = np.array(embedding, dtype=np.float32)
            return self._normalize_embedding(emb_array)
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def _add_to_vector_index(self, citation: Citation):
        """Add citation embedding to FAISS index"""
        if not self.use_vector or not citation.content:
            return
        
        text_for_embedding = f"{citation.title} {citation.content[:1000]} {citation.abstract or ''}"
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        emb = self._get_embedding(text_for_embedding)
        if emb is not None:
            self._faiss_index.add(emb.reshape(1, -1))
            self._citation_ids.append(citation.id)
<<<<<<< HEAD
    
    def _update_tfidf(self):
        """Update TF-IDF matrix for all citations"""
        if not self.use_tfidf or not self.citations:
            return
        
        documents = []
        for citation in self.citations.values():
            doc = f"{citation.title} {' '.join(citation.keywords)} {citation.abstract or ''} {citation.content[:2000] if citation.content else ''}"
            documents.append(doc)
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
    
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def _load_from_disk(self):
        """Load citations from SQLite database"""
        if not self.storage_path.exists():
            return
<<<<<<< HEAD
        
        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
            
            # Create tables
=======

        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
                    isbn TEXT,
                    issn TEXT,
                    pmid TEXT,
                    arxiv_id TEXT,
                    keywords TEXT,
                    reference_type TEXT,
                    volume TEXT,
                    issue TEXT,
                    pages TEXT,
                    edition INTEGER,
                    series TEXT,
                    institution TEXT,
                    conference TEXT,
                    abstract TEXT,
                    language TEXT,
                    tags TEXT,
                    notes TEXT,
                    rating REAL,
                    read_status INTEGER,
                    important INTEGER,
                    references TEXT,
                    citations TEXT,
                    categories TEXT,
                    reliability REAL,
                    accessed_date TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    summary TEXT
                )
            """)
            
            cursor.execute("SELECT * FROM citations")
            rows = cursor.fetchall()
            
=======
                    keywords TEXT,
                    reliability REAL,
                    accessed_date TEXT,
                    created_at TEXT
                )
            """)
            cursor.execute("SELECT * FROM citations")
            rows = cursor.fetchall()
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
                    isbn=row[9],
                    issn=row[10],
                    pmid=row[11],
                    arxiv_id=row[12],
                    keywords=json.loads(row[13]) if row[13] else [],
                    reference_type=ReferenceType(row[14]) if row[14] else ReferenceType.WEBPAGE,
                    volume=row[15],
                    issue=row[16],
                    pages=row[17],
                    edition=row[18],
                    series=row[19],
                    institution=row[20],
                    conference=row[21],
                    abstract=row[22],
                    language=row[23] or "en",
                    tags=json.loads(row[24]) if row[24] else [],
                    notes=row[25],
                    rating=row[26],
                    read_status=bool(row[27]),
                    important=bool(row[28]),
                    references=json.loads(row[29]) if row[29] else [],
                    citations=json.loads(row[30]) if row[30] else [],
                    categories=json.loads(row[31]) if row[31] else [],
                )
                citation.id = row[0]
                citation.reliability_score = row[32]
                citation.accessed_date = row[33]
                citation.created_at = datetime.fromisoformat(row[34])
                citation.updated_at = datetime.fromisoformat(row[35])
                citation.summary = row[36]
                
                self.citations[citation.id] = citation
                
                # Add to vector index after load
                if self.use_vector and citation.content:
                    self._add_to_vector_index(citation)
                
                # Add to citation graph
                if citation.references:
                    self.citation_graph.add_citation(citation.id, citation.references)
            
            conn.close()
            
            # Update TF-IDF
            if self.use_tfidf:
                self._update_tfidf()
            
            logger.info(f"Loaded {len(self.citations)} citations from storage")
            
        except Exception as e:
            logger.warning(f"Failed to load citations: {e}")
    
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def _save_to_disk(self):
        """Save citations to SQLite database"""
        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()
<<<<<<< HEAD
            
            for c in self.citations.values():
                cursor.execute("""
                    INSERT OR REPLACE INTO citations VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                """, (
                    c.id, c.title, json.dumps(c.authors), c.url, c.source,
                    c.content, c.publication_date, c.publisher, c.doi, c.isbn,
                    c.issn, c.pmid, c.arxiv_id, json.dumps(c.keywords), c.reference_type.value,
                    c.volume, c.issue, c.pages, c.edition, c.series, c.institution,
                    c.conference, c.abstract, c.language, json.dumps(c.tags), c.notes,
                    c.rating, int(c.read_status), int(c.important), json.dumps(c.references),
                    json.dumps(c.citations), json.dumps(c.categories), c.reliability_score,
                    c.accessed_date, c.created_at.isoformat(), c.updated_at.isoformat(),
                    c.summary
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save citations: {e}")
    
    async def _auto_backup_loop(self):
        """Automatic backup loop"""
        while self.auto_backup:
            await asyncio.sleep(self.backup_interval_hours * 3600)
            
            if (datetime.now() - self.last_backup).total_seconds() >= self.backup_interval_hours * 3600:
                backup_path = f"backups/citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.export(backup_path, format="json")
                self.last_backup = datetime.now()
                logger.info(f"Auto backup created: {backup_path}")
    
    # ==================== CRUD OPERATIONS ====================
    
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
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
<<<<<<< HEAD
        isbn: Optional[str] = None,
        issn: Optional[str] = None,
        pmid: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        reference_type: ReferenceType = ReferenceType.WEBPAGE,
        volume: Optional[str] = None,
        issue: Optional[str] = None,
        pages: Optional[str] = None,
        edition: Optional[int] = None,
        series: Optional[str] = None,
        institution: Optional[str] = None,
        conference: Optional[str] = None,
        abstract: Optional[str] = None,
        language: str = "en",
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        rating: Optional[float] = None,
        important: bool = False,
        references: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Add a new citation with maximum metadata"""
=======
        keywords: Optional[List[str]] = None,
    ) -> str:
        """Add a new citation (deduplicated)"""
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        citation = Citation(
            title=title,
            authors=authors,
            url=url,
            source=source,
            content=content,
            publication_date=publication_date,
            publisher=publisher,
            doi=doi,
<<<<<<< HEAD
            isbn=isbn,
            issn=issn,
            pmid=pmid,
            arxiv_id=arxiv_id,
            keywords=keywords,
            reference_type=reference_type,
            volume=volume,
            issue=issue,
            pages=pages,
            edition=edition,
            series=series,
            institution=institution,
            conference=conference,
            abstract=abstract,
            language=language,
            tags=tags,
            notes=notes,
            rating=rating,
            important=important,
            references=references,
            categories=categories,
        )
        
        if citation.id in self.citations:
            return citation.id  # already exists
        
        self.citations[citation.id] = citation
        self._add_to_vector_index(citation)
        
        if references:
            self.citation_graph.add_citation(citation.id, references)
        
        if self.use_tfidf:
            self._update_tfidf()
        
        self._save_to_disk()
        logger.info(f"📚 Citation added: {title}")
        return citation.id
    
    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a citation by ID"""
        c = self.citations.get(citation_id)
        return c.to_dict() if c else None
    
    def update_citation(self, citation_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing citation"""
        if citation_id not in self.citations:
            return False
        
        citation = self.citations[citation_id]
        
        for key, value in updates.items():
            if hasattr(citation, key):
                setattr(citation, key, value)
        
        citation.updated_at = datetime.now()
        citation.reliability_score = citation._calculate_reliability()
        
        # Rebuild vector index if content changed
        if "content" in updates or "title" in updates:
            self._rebuild_vector_index()
        
        if self.use_tfidf:
            self._update_tfidf()
        
        self._save_to_disk()
        logger.info(f"✏️ Citation updated: {citation.title}")
        return True
    
    def delete_citation(self, citation_id: str) -> bool:
        """Delete a citation"""
        if citation_id in self.citations:
            title = self.citations[citation_id].title
            del self.citations[citation_id]
            self._save_to_disk()
            self._rebuild_vector_index()
            
            if self.use_tfidf:
                self._update_tfidf()
            
            logger.info(f"🗑️ Citation deleted: {title}")
            return True
        return False
    
    def delete_all(self):
        """Delete all citations"""
        self.citations.clear()
        self._rebuild_vector_index()
        
        if self.use_tfidf:
            self._update_tfidf()
        
        self._save_to_disk()
        logger.info("🗑️ All citations deleted")
    
    # ==================== SEARCH OPERATIONS ====================
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5, 
        use_vector: bool = True,
        filters: Optional[SearchFilter] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search citations with multiple strategies
        """
        results = []
        
        # Record search
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "use_vector": use_vector
        })
        
        # Vector search
        if use_vector and self.use_vector and self._faiss_index and self._faiss_index.ntotal > 0:
            results = await self._vector_search(query, top_k)
        
        # TF-IDF search
        elif self.use_tfidf and self.tfidf_matrix is not None:
            results = await self._tfidf_search(query, top_k)
        
        # Keyword search fallback
        else:
            results = self._keyword_search(query, top_k)
        
        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)
        
        # Apply score threshold
        if min_score > 0:
            results = [r for r in results if r.get("similarity", 0) >= min_score]
        
        return results[:top_k]
    
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
                    item["search_method"] = "GGUF_VECTOR"
                    results.append(item)
        
        return results
    
    async def _tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """TF-IDF based search"""
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top_k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        citations_list = list(self.citations.values())
        
        for idx in top_indices:
            if similarities[idx] > 0:
                citation = citations_list[idx]
                item = citation.to_dict()
                item["similarity"] = float(similarities[idx])
                item["search_method"] = "TF_IDF"
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
            if any(query_lower in tag.lower() for tag in citation.tags):
                score += 3
            if citation.abstract and query_lower in citation.abstract.lower():
                score += 3
            if citation.content and query_lower in citation.content.lower():
                score += 2
            if any(query_lower in author.lower() for author in citation.authors):
                score += 4
            
            if score > 0:
                scored.append((score, citation))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, citation in scored[:top_k]:
            item = citation.to_dict()
            item["similarity"] = score / 10.0  # Normalize
            item["search_method"] = "KEYWORD"
            results.append(item)
        
        return results
    
    def _apply_filters(self, results: List[Dict], filters: SearchFilter) -> List[Dict]:
        """Apply search filters to results"""
        filtered = []
        
        for result in results:
            include = True
            
            # Author filter
            if filters.authors:
                if not any(author in result.get("authors", []) for author in filters.authors):
                    include = False
            
            # Tag filter
            if filters.tags and include:
                if not any(tag in result.get("tags", []) for tag in filters.tags):
                    include = False
            
            # Category filter
            if filters.categories and include:
                if not any(cat in result.get("categories", []) for cat in filters.categories):
                    include = False
            
            # Date range
            if filters.date_from and include:
                pub_date = result.get("publication_date", "")
                if pub_date < filters.date_from:
                    include = False
            
            if filters.date_to and include:
                pub_date = result.get("publication_date", "")
                if pub_date > filters.date_to:
                    include = False
            
            # Reliability
            if filters.min_reliability and include:
                if result.get("reliability", 0) < filters.min_reliability:
                    include = False
            
            # Reference type
            if filters.reference_types and include:
                if result.get("reference_type") not in [rt.value for rt in filters.reference_types]:
                    include = False
            
            # Important only
            if filters.important_only and include:
                if not result.get("important", False):
                    include = False
            
            # Read status
            if filters.read_status is not None and include:
                if result.get("read_status") != filters.read_status:
                    include = False
            
            # Rating
            if filters.min_rating and include:
                if (result.get("rating") or 0) < filters.min_rating:
                    include = False
            
            # Language
            if filters.language and include:
                if result.get("language") != filters.language:
                    include = False
            
            if include:
                filtered.append(result)
        
        return filtered
    
    # ==================== FORMATTING ====================
    
    def format_citation(self, citation_id: str, style: Union[str, CitationStyle] = "apa") -> Optional[str]:
        """Get formatted citation string"""
        c = self.citations.get(citation_id)
        return c.format_citation(style) if c else None
    
    def format_multiple(self, citation_ids: List[str], style: str = "apa", 
                       sort_by: str = "author", join_with: str = "\n\n") -> str:
        """Format multiple citations"""
        citations = []
        for cid in citation_ids:
            if cid in self.citations:
                formatted = self.citations[cid].format_citation(style)
                citations.append((self.citations[cid], formatted))
        
        # Sort
        if sort_by == "author":
            citations.sort(key=lambda x: x[0].authors[0] if x[0].authors else "")
        elif sort_by == "year":
            citations.sort(key=lambda x: x[0].publication_date)
        elif sort_by == "title":
            citations.sort(key=lambda x: x[0].title)
        
        return join_with.join([formatted for _, formatted in citations])
    
    def generate_bibliography(self, citation_ids: List[str], style: str = "apa") -> str:
        """Generate a formatted bibliography"""
        return self.format_multiple(citation_ids, style, sort_by="author")
    
    # ==================== EXTRACTORS ====================
    
    def extract_from_url(self, url: str, html_content: str) -> str:
        """Auto-extract metadata from URL"""
        title = self._extract_title(html_content)
        keywords = self._extract_keywords(html_content)
        authors = self._extract_authors(html_content)
        description = self._extract_description(html_content)
        
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        return self.add_citation(
            title=title,
            authors=authors,
            url=url,
            source="web",
            content=html_content[:5000],
<<<<<<< HEAD
            abstract=description,
            keywords=keywords,
        )
    
    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else "Untitled"
    
=======
            keywords=keywords,
        )

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else "Untitled"

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def _extract_keywords(self, html: str) -> List[str]:
        match = re.search(
            r'<meta name=["\']keywords["\'] content=["\'](.*?)["\']',
            html,
            re.IGNORECASE,
        )
        if match:
            return [kw.strip() for kw in match.group(1).split(",")]
        return []
<<<<<<< HEAD
    
=======

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def _extract_authors(self, html: str) -> List[str]:
        patterns = [
            r'<meta name=["\']author["\'] content=["\'](.*?)["\']',
            r'<a rel=["\']author["\']>(.*?)</a>',
<<<<<<< HEAD
            r'<span class=["\']author["\']>(.*?)</span>',
=======
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
        ]
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return [match.group(1).strip()]
        return []
<<<<<<< HEAD
    
    def _extract_description(self, html: str) -> Optional[str]:
        match = re.search(
            r'<meta name=["\']description["\'] content=["\'](.*?)["\']',
            html,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None
    
    # ==================== IMPORT/EXPORT ====================
    
    def export(self, filepath: str, format: str = "json"):
        """Export citations to various formats"""
        if format == "json":
            self._export_json(filepath)
        elif format == "csv":
            self._export_csv(filepath)
        elif format == "bibtex":
            self._export_bibtex(filepath)
        elif format == "ris":
            self._export_ris(filepath)
        elif format == "endnote":
            self._export_endnote(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self, filepath: str):
        """Export to JSON"""
        data = [c.to_dict() for c in self.citations.values()]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(data)} citations to {filepath}")
    
    def _export_csv(self, filepath: str):
        """Export to CSV"""
        if not self.citations:
            return
        
        # Get all possible fields
        fieldnames = set()
        for c in self.citations.values():
            fieldnames.update(c.to_dict().keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            for c in self.citations.values():
                row = c.to_dict()
                # Convert lists to strings
                for key in ['authors', 'keywords', 'tags', 'categories', 'references', 'citations']:
                    if key in row and isinstance(row[key], list):
                        row[key] = '; '.join(row[key])
                writer.writerow(row)
        
        logger.info(f"Exported {len(self.citations)} citations to {filepath}")
    
    def _export_bibtex(self, filepath: str):
        """Export to BibTeX"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for c in self.citations.values():
                f.write(c.generate_bibtex())
                f.write("\n")
        logger.info(f"Exported {len(self.citations)} BibTeX entries to {filepath}")
    
    def _export_ris(self, filepath: str):
        """Export to RIS"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for c in self.citations.values():
                f.write(c.generate_ris())
                f.write("\n")
        logger.info(f"Exported {len(self.citations)} RIS entries to {filepath}")
    
    def _export_endnote(self, filepath: str):
        """Export to EndNote XML format"""
        import xml.dom.minidom as minidom
        from xml.etree import ElementTree as ET
        
        root = ET.Element("xml")
        records = ET.SubElement(root, "records")
        
        for c in self.citations.values():
            record = ET.SubElement(records, "record")
            
            ET.SubElement(record, "title").text = c.title
            ET.SubElement(record, "url").text = c.url
            ET.SubElement(record, "year").text = c.publication_date[:4] if c.publication_date else ""
            
            for author in c.authors:
                ET.SubElement(record, "author").text = author
        
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)
        logger.info(f"Exported {len(self.citations)} EndNote entries to {filepath}")
    
    def import_citations(self, filepath: str, format: str = "json"):
        """Import citations from file"""
        if format == "json":
            self._import_json(filepath)
        elif format == "csv":
            self._import_csv(filepath)
        elif format == "bibtex":
            self._import_bibtex(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.import_history.append({
            "filepath": filepath,
            "format": format,
            "count": len(self.citations),
            "timestamp": datetime.now().isoformat()
        })
    
    def _import_json(self, filepath: str):
        """Import from JSON"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data:
            self.add_citation(
                title=item.get("title", "Untitled"),
=======

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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                authors=item.get("authors", []),
                url=item.get("url", ""),
                source=item.get("source", "imported"),
                content=item.get("content"),
                publication_date=item.get("publication_date"),
                publisher=item.get("publisher"),
                doi=item.get("doi"),
                keywords=item.get("keywords", []),
<<<<<<< HEAD
                reference_type=ReferenceType(item.get("reference_type", "webpage")),
                tags=item.get("tags", []),
                important=item.get("important", False),
            )
        
        logger.info(f"Imported {len(data)} citations from {filepath}")
    
    def _import_csv(self, filepath: str):
        """Import from CSV"""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string lists back to lists
                authors = row.get('authors', '').split('; ') if row.get('authors') else []
                keywords = row.get('keywords', '').split('; ') if row.get('keywords') else []
                tags = row.get('tags', '').split('; ') if row.get('tags') else []
                
                self.add_citation(
                    title=row.get('title', 'Untitled'),
                    authors=authors,
                    url=row.get('url', ''),
                    source=row.get('source', 'imported'),
                    content=row.get('content'),
                    publication_date=row.get('publication_date'),
                    publisher=row.get('publisher'),
                    doi=row.get('doi'),
                    keywords=keywords,
                    tags=tags,
                )
        
        logger.info(f"Imported CSV from {filepath}")
    
    def _import_bibtex(self, filepath: str):
        """Import from BibTeX file"""
        import re
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple BibTeX parser
        entries = re.finditer(r'@(\w+)\{([^,]+),\s*([^@]+)', content, re.DOTALL)
        
        for entry in entries:
            entry_type = entry.group(1)
            entry_id = entry.group(2)
            fields_str = entry.group(3)
            
            # Parse fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*[{"](.*?)[}",]'
            for match in re.finditer(field_pattern, fields_str):
                fields[match.group(1)] = match.group(2).strip()
            
            # Map BibTeX type to ReferenceType
            type_map = {
                'article': ReferenceType.JOURNAL_ARTICLE,
                'book': ReferenceType.BOOK,
                'inproceedings': ReferenceType.CONFERENCE_PAPER,
                'phdthesis': ReferenceType.THESIS,
                'mastersthesis': ReferenceType.THESIS,
                'webpage': ReferenceType.WEBPAGE,
            }
            
            reference_type = type_map.get(entry_type.lower(), ReferenceType.JOURNAL_ARTICLE)
            
            # Parse authors
            authors = [a.strip() for a in fields.get('author', '').split(' and ')] if fields.get('author') else []
            
            self.add_citation(
                title=fields.get('title', 'Untitled'),
                authors=authors,
                url=fields.get('url', ''),
                source=fields.get('journal', fields.get('booktitle', 'imported')),
                content=None,
                publication_date=fields.get('year'),
                publisher=fields.get('publisher'),
                doi=fields.get('doi'),
                reference_type=reference_type,
            )
        
        logger.info(f"Imported BibTeX from {filepath}")
    
    # ==================== ANALYSIS & STATISTICS ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total = len(self.citations)
        if total == 0:
            return {"total": 0}
        
        # Reliability distribution
        reliability_dist = {
            "excellent": sum(1 for c in self.citations.values() if c.reliability_score >= 0.9),
            "good": sum(1 for c in self.citations.values() if 0.7 <= c.reliability_score < 0.9),
            "average": sum(1 for c in self.citations.values() if 0.5 <= c.reliability_score < 0.7),
            "poor": sum(1 for c in self.citations.values() if c.reliability_score < 0.5),
        }
        
        # Type distribution
        type_dist = defaultdict(int)
        for c in self.citations.values():
            type_dist[c.reference_type.value] += 1
        
        # Year distribution
        year_dist = defaultdict(int)
        for c in self.citations.values():
            if c.publication_date:
                year = c.publication_date[:4]
                year_dist[year] += 1
        
        # Author statistics
        all_authors = []
        for c in self.citations.values():
            all_authors.extend(c.authors)
        author_counts = defaultdict(int)
        for author in all_authors:
            author_counts[author] += 1
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Keyword statistics
        all_keywords = []
        for c in self.citations.values():
            all_keywords.extend(c.keywords)
        keyword_counts = defaultdict(int)
        for kw in all_keywords:
            keyword_counts[kw.lower()] += 1
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Tag statistics
        all_tags = []
        for c in self.citations.values():
            all_tags.extend(c.tags)
        tag_counts = defaultdict(int)
        for tag in all_tags:
            tag_counts[tag.lower()] += 1
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total": total,
            "avg_reliability": round(sum(c.reliability_score for c in self.citations.values()) / total, 3),
            "reliability_distribution": reliability_dist,
            "type_distribution": dict(type_dist),
            "year_distribution": dict(sorted(year_dist.items())),
            "top_authors": [{"name": name, "count": count} for name, count in top_authors],
            "top_keywords": [{"keyword": kw, "count": count} for kw, count in top_keywords],
            "top_tags": [{"tag": tag, "count": count} for tag, count in top_tags],
            "important_count": sum(1 for c in self.citations.values() if c.important),
            "read_count": sum(1 for c in self.citations.values() if c.read_status),
            "average_rating": round(sum(c.rating or 0 for c in self.citations.values()) / total, 2) if any(c.rating for c in self.citations.values()) else None,
            "vector_enabled": self.use_vector,
            "embedding_model": "GGUF" if self.use_gguf else None,
            "vector_size": self._faiss_index.ntotal if self._faiss_index else 0,
            "tfidf_enabled": self.use_tfidf,
            "citation_graph_enabled": NETWORKX_AVAILABLE,
        }
    
    def get_top_citations(self, limit: int = 10, sort_by: str = "reliability") -> List[Dict]:
        """Get top citations by various metrics"""
        if sort_by == "reliability":
            sorted_cits = sorted(self.citations.values(), key=lambda c: c.reliability_score, reverse=True)
        elif sort_by == "rating":
            sorted_cits = sorted(self.citations.values(), key=lambda c: c.rating or 0, reverse=True)
        elif sort_by == "date":
            sorted_cits = sorted(self.citations.values(), key=lambda c: c.publication_date, reverse=True)
        elif sort_by == "title":
            sorted_cits = sorted(self.citations.values(), key=lambda c: c.title)
        else:
            sorted_cits = list(self.citations.values())
        
        return [c.to_dict() for c in sorted_cits[:limit]]
    
    def find_duplicates(self) -> List[List[str]]:
        """Find duplicate citations based on title similarity"""
        duplicates = []
        processed = set()
        
        citations_list = list(self.citations.values())
        
        for i, c1 in enumerate(citations_list):
            if c1.id in processed:
                continue
            
            duplicate_group = [c1.id]
            
            for j, c2 in enumerate(citations_list[i+1:], i+1):
                # Check title similarity
                if self._title_similarity(c1.title, c2.title) > 0.8:
                    duplicate_group.append(c2.id)
                    processed.add(c2.id)
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.add(c1.id)
        
        return duplicates
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity"""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def recommend_citations(self, citation_id: str, top_k: int = 5) -> List[Dict]:
        """Recommend related citations based on content similarity"""
        if citation_id not in self.citations:
            return []
        
        target = self.citations[citation_id]
        
        # Create search query from target metadata
        query = f"{target.title} {' '.join(target.keywords)} {target.abstract or ''}"
        
        # Use vector search to find similar
        results = asyncio.run(self.search(query, top_k=top_k+1, use_vector=True))
        
        # Remove the target itself
        results = [r for r in results if r["id"] != citation_id]
        
        return results[:top_k]
    
    # ==================== SUMMARY & AI FEATURES ====================
    
    def summarize_citation(self, citation_id: str, max_length: int = 200) -> Optional[str]:
        """Generate a summary for a citation using content or abstract"""
        citation = self.citations.get(citation_id)
        if not citation:
            return None
        
        # Use existing summary
        if citation.summary:
            return citation.summary
        
        # Generate from abstract
        if citation.abstract:
            summary = citation.abstract[:max_length]
            if len(citation.abstract) > max_length:
                summary += "..."
            return summary
        
        # Generate from content
        if citation.content:
            # Take first few sentences
            sentences = re.split(r'[.!?]+', citation.content[:500])
            summary = '.'.join(sentences[:2])[:max_length]
            if len(summary) < len(citation.content[:500]):
                summary += "..."
            citation.summary = summary
            self._save_to_disk()
            return summary
        
        return "No content available for summary"
    
    # ==================== REBUILD INDEXES ====================
    
    def _rebuild_vector_index(self):
        """Rebuild FAISS index from current citations using GGUF embeddings"""
        if not self.use_vector:
            return
        
        self._init_faiss(self.embedding_dimension)
        self._citation_ids.clear()
        
        for c in self.citations.values():
            if c.content or c.abstract:
                self._add_to_vector_index(c)
        
        logger.info(f"Vector index rebuilt with {len(self._citation_ids)} citations")
    
    def rebuild_indexes(self):
        """Rebuild all search indexes"""
        if self.use_vector:
            self._rebuild_vector_index()
        
        if self.use_tfidf:
            self._update_tfidf()
        
        logger.info("All search indexes rebuilt")
    
    # ==================== BACKUP & RESTORE ====================
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """Create a complete backup of all data"""
        if not backup_path:
            backup_path = f"backups/citation_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        Path(backup_path).mkdir(parents=True, exist_ok=True)
        
        # Export database
        self.export(f"{backup_path}/citations.json", format="json")
        
        # Export statistics
        stats = self.get_statistics()
        with open(f"{backup_path}/stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save config
        config = {
            "storage_path": str(self.storage_path),
            "model_path": self.model_path,
            "vector_enabled": self.use_vector,
            "tfidf_enabled": self.use_tfidf,
            "backup_timestamp": datetime.now().isoformat()
        }
        with open(f"{backup_path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Backup created at {backup_path}")
        return backup_path
    
    def restore_from_backup(self, backup_path: str):
        """Restore from backup"""
        # Clear current data
        self.citations.clear()
        
        # Import JSON
        self._import_json(f"{backup_path}/citations.json")
        
        # Rebuild indexes
        self.rebuild_indexes()
        
        logger.info(f"Restored from backup: {backup_path}")
    
    # ==================== MEMORY INTEGRATION ====================
    
    async def integrate_with_memory(self, memory_manager, limit: int = 10):
        """Push important citations into EDIATH's memory system"""
        top_cits = self.get_top_citations(limit=limit, sort_by="reliability")
        
        for cit in top_cits:
            await memory_manager.store({
                "type": "citation",
                "title": cit["title"],
                "reliability": cit["reliability"],
                "summary": self.summarize_citation(cit["id"]),
                "authors": cit.get("authors", []),
                "keywords": cit.get("keywords", []),
                "url": cit.get("url", ""),
                "timestamp": datetime.now().isoformat(),
                "embedding_model": "GGUF",
                "citation_id": cit["id"]
            })
        
        logger.info(f"Integrated {len(top_cits)} citations into memory")
        return len(top_cits)


# ==================== WRAPPER FOR EDIATH ====================

class CitationManagerWrapper:
    """Wrapper class to integrate CitationManager with EDIATH"""
    
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.citation_manager = CitationManager(
            storage_path=config.get("storage_path", "data/citations.db"),
            model_path=config.get("model_path", "./models/EDIATH-q4_k_m.gguf"),
            n_ctx=config.get("n_ctx", 2048),
            n_threads=config.get("n_threads", 4),
<<<<<<< HEAD
            use_vector=config.get("use_vector", True),
            use_tfidf=config.get("use_tfidf", True),
            auto_backup=config.get("auto_backup", True),
            backup_interval_hours=config.get("backup_interval_hours", 24),
        )
        self.agent_type = "citation_manager"
        self.capabilities = [
            "add_citation", "search_citations", "format_citation", "get_top_sources",
            "export_citations", "import_citations", "get_statistics", "find_duplicates",
            "recommend_citations", "generate_bibliography", "delete_citation", "update_citation",
            "integrate_with_memory", "create_backup", "rebuild_indexes"
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a citation management request"""
        operation = request.get("operation")
        
        if operation == "add":
            citation_id = self.citation_manager.add_citation(
                title=request.get("title", "Untitled"),
                authors=request.get("authors", []),
                url=request.get("url", ""),
                source=request.get("source", "unknown"),
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
                content=request.get("content"),
                publication_date=request.get("publication_date"),
                publisher=request.get("publisher"),
                doi=request.get("doi"),
<<<<<<< HEAD
                isbn=request.get("isbn"),
                keywords=request.get("keywords", []),
                reference_type=ReferenceType(request.get("reference_type", "webpage")),
                tags=request.get("tags", []),
                notes=request.get("notes"),
                rating=request.get("rating"),
                important=request.get("important", False),
=======
                keywords=request.get("keywords", []),
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            )
            return {
                "success": True,
                "citation_id": citation_id,
<<<<<<< HEAD
                "message": "Citation added successfully"
            }
        
        elif operation == "search":
            filters = None
            if request.get("filters"):
                filters = SearchFilter(**request["filters"])
            
            results = await self.citation_manager.search(
                query=request.get("query", ""),
                top_k=request.get("top_k", 10),
                use_vector=request.get("use_vector", True),
                filters=filters,
                min_score=request.get("min_score", 0.0)
            )
            return {"success": True, "results": results, "total": len(results)}
        
        elif operation == "format":
            formatted = self.citation_manager.format_citation(
                citation_id=request.get("citation_id"),
                style=request.get("style", "apa")
            )
            return {"success": True, "formatted_citation": formatted} if formatted else {"success": False, "error": "Citation not found"}
        
        elif operation == "format_multiple":
            formatted = self.citation_manager.format_multiple(
                citation_ids=request.get("citation_ids", []),
                style=request.get("style", "apa"),
                sort_by=request.get("sort_by", "author")
            )
            return {"success": True, "bibliography": formatted}
        
        elif operation == "get_top":
            sources = self.citation_manager.get_top_citations(
                limit=request.get("limit", 10),
                sort_by=request.get("sort_by", "reliability")
            )
            return {"success": True, "sources": sources}
        
        elif operation == "export":
            self.citation_manager.export(
                filepath=request.get("filepath", "citations_export.json"),
                format=request.get("format", "json")
            )
            return {"success": True, "message": f'Exported to {request.get("filepath", "citations_export.json")}'}
        
        elif operation == "import":
            self.citation_manager.import_citations(
                filepath=request.get("filepath"),
                format=request.get("format", "json")
            )
            return {"success": True, "message": f'Imported from {request.get("filepath")}'}
        
        elif operation == "stats":
            return {"success": True, "statistics": self.citation_manager.get_statistics()}
        
        elif operation == "duplicates":
            duplicates = self.citation_manager.find_duplicates()
            return {"success": True, "duplicates": duplicates, "groups": len(duplicates)}
        
        elif operation == "recommend":
            recommendations = self.citation_manager.recommend_citations(
                citation_id=request.get("citation_id"),
                top_k=request.get("top_k", 5)
            )
            return {"success": True, "recommendations": recommendations}
        
        elif operation == "delete":
            success = self.citation_manager.delete_citation(request.get("citation_id"))
            return {"success": success, "message": "Citation deleted" if success else "Citation not found"}
        
        elif operation == "update":
            success = self.citation_manager.update_citation(
                citation_id=request.get("citation_id"),
                updates=request.get("updates", {})
            )
            return {"success": success, "message": "Citation updated" if success else "Update failed"}
        
        elif operation == "backup":
            backup_path = self.citation_manager.create_backup(request.get("backup_path"))
            return {"success": True, "backup_path": backup_path}
        
        elif operation == "restore":
            self.citation_manager.restore_from_backup(request.get("backup_path"))
            return {"success": True, "message": "Restored from backup"}
        
        elif operation == "rebuild":
            self.citation_manager.rebuild_indexes()
            return {"success": True, "message": "Indexes rebuilt"}
        
        elif operation == "summarize":
            summary = self.citation_manager.summarize_citation(
                citation_id=request.get("citation_id"),
                max_length=request.get("max_length", 200)
            )
            return {"success": True, "summary": summary} if summary else {"success": False, "error": "Citation not found"}
        
        elif operation == "integrate_memory":
            if request.get("memory_manager"):
                count = await self.citation_manager.integrate_with_memory(
                    memory_manager=request.get("memory_manager"),
                    limit=request.get("limit", 10)
                )
                return {"success": True, "integrated_count": count}
            return {"success": False, "error": "Memory manager required"}
        
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    
=======
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

>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "CitationManager",
            "type": self.agent_type,
            "capabilities": self.capabilities,
<<<<<<< HEAD
            "stats": self.citation_manager.get_statistics(),
=======
            "stats": self.citation_manager.get_stats(),
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
            "embedding_model": "GGUF",
        }


<<<<<<< HEAD
# ==================== EXAMPLE USAGE ====================

async def test_citation_manager():
    """Test the citation manager with all features"""
    
    logger.info("=== Citation Manager Ultimate Test ===\n")
    
    # Create manager
    manager = CitationManager(
        storage_path="data/test_citations.db",
        model_path="./models/EDIATH-q4_k_m.gguf",
        use_vector=True,
        use_tfidf=True,
        auto_backup=False
    )
    
    # Add citations
    logger.info("1. Adding Citations...")
    
    citation1 = manager.add_citation(
        title="Artificial Intelligence in Modern Computing: A Comprehensive Review",
        authors=["John Smith", "Jane Doe", "Bob Johnson"],
        url="https://example.com/ai-paper",
        source="Journal of AI Research",
        content="This paper discusses the latest advances in artificial intelligence and machine learning, with a focus on transformer architectures and large language models. The authors present a comprehensive review of recent developments in the field.",
        publication_date="2024-01-15",
        publisher="AI Research Press",
        doi="10.1234/ai.2024.001",
        keywords=["AI", "Machine Learning", "Transformers", "LLM"],
        reference_type=ReferenceType.JOURNAL_ARTICLE,
        volume="15",
        issue="2",
        pages="123-145",
        tags=["artificial-intelligence", "review"],
        rating=4.5,
        important=True
    )
    
    citation2 = manager.add_citation(
        title="Deep Learning for Natural Language Processing",
        authors=["Alice Williams"],
        url="https://example.com/nlp-paper",
        source="NLP Conference 2023",
        content="Exploring deep learning approaches for NLP tasks including sentiment analysis, machine translation, and text generation.",
        publication_date="2023-08-20",
        publisher="ACL",
        keywords=["NLP", "Deep Learning", "Transformers"],
        reference_type=ReferenceType.CONFERENCE_PAPER,
        conference="ACL 2023",
        tags=["nlp", "deep-learning"]
    )
    
    citation3 = manager.add_citation(
        title="The Future of Quantum Computing",
        authors=["David Chen", "Maria Garcia"],
        url="https://example.com/quantum-paper",
        source="Quantum Computing Weekly",
        content="Quantum computing promises to revolutionize computing by leveraging quantum mechanical phenomena.",
        publication_date="2024-02-10",
        publisher="Quantum Press",
        keywords=["Quantum Computing", "Qubits", "Quantum Supremacy"],
        reference_type=ReferenceType.JOURNAL_ARTICLE,
        tags=["quantum", "future-tech"],
        rating=4.0
    )
    
    logger.info(f"   Added citations: {citation1}, {citation2}, {citation3}")
    
    # Search
    logger.info("\n2. Searching Citations...")
    
    results = await manager.search("artificial intelligence advances", top_k=3)
    for r in results:
        logger.info(f"   Score {r.get('similarity', 0):.3f}: {r.get('title')[:60]}...")
    
    # Filtered search
    logger.info("\n3. Filtered Search...")
    filters = SearchFilter(
        min_reliability=0.7,
        important_only=True
    )
    results = await manager.search("AI", top_k=5, filters=filters)
    logger.info(f"   Found {len(results)} important citations with high reliability")
    
    # Format citations
    logger.info("\n4. Formatting Citations...")
    apa = manager.format_citation(citation1, style="apa")
    logger.info(f"   APA: {apa[:100]}...")
    
    mla = manager.format_citation(citation1, style="mla")
    logger.info(f"   MLA: {mla[:100]}...")
    
    # Bibliography
    logger.info("\n5. Generating Bibliography...")
    bibliography = manager.format_multiple([citation1, citation2, citation3], style="apa")
    logger.info(f"   Bibliography length: {len(bibliography)} characters")
    
    # Statistics
    logger.info("\n6. Statistics...")
    stats = manager.get_statistics()
    logger.info(f"   Total citations: {stats['total']}")
    logger.info(f"   Average reliability: {stats['avg_reliability']}")
    logger.info(f"   Important citations: {stats['important_count']}")
    logger.info(f"   Top keywords: {stats['top_keywords'][:5]}")
    
    # Export
    logger.info("\n7. Exporting Data...")
    manager.export("citations_export.json", format="json")
    manager.export("citations_export.bib", format="bibtex")
    logger.info("   Exported to multiple formats")
    
    # Recommendations
    logger.info("\n8. Getting Recommendations...")
    recommendations = manager.recommend_citations(citation1, top_k=3)
    for rec in recommendations:
        logger.info(f"   Recommended: {rec['title'][:60]}...")
    
    # Duplicates
    logger.info("\n9. Checking for Duplicates...")
    duplicates = manager.find_duplicates()
    logger.info(f"   Duplicate groups found: {len(duplicates)}")
    
    # Summary
    logger.info("\n10. Generating Summary...")
    summary = manager.summarize_citation(citation1)
    logger.info(f"    Summary: {summary[:100]}...")
    
    logger.info("\n=== Test Complete ===")
    
    return manager


if __name__ == "__main__":
    asyncio.run(test_citation_manager())
=======
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
>>>>>>> 7466e01e6018c1528d9953b5299818bf7454f6d7
