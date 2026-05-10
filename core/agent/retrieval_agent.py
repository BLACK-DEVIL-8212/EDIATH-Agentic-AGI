"""
Retrieval Agent for EDIATH using GGUF Model
Advanced RAG-style retrieval: hybrid search, reranking, multi-source retrieval, context optimization
Uses EDIATH GGUF model for embeddings and retrieval
"""

import asyncio
import hashlib
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import logging

# GGUF Model for embeddings
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Vector search fallbacks (without sentence-transformers)
try:
    import numpy as np

    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class RetrievalStrategy(Enum):
    """Retrieval strategies"""

    DENSE = "dense"  # Dense vector retrieval (GGUF embeddings)
    SPARSE = "sparse"  # Sparse (BM25) retrieval
    HYBRID = "hybrid"  # Hybrid dense + sparse
    MULTI_QUERY = "multi_query"  # Multiple query variations
    HYPOTHETICAL = "hypothetical"  # HyDE - Hypothetical Document Embeddings


class RetrievalSource(Enum):
    """Retrieval sources"""

    KNOWLEDGE_BASE = "knowledge_base"
    WEB = "web"
    DOCUMENTS = "documents"
    DATABASE = "database"
    MEMORY = "memory"
    API = "api"


@dataclass
class RetrievedChunk:
    """Retrieved chunk with metadata"""

    id: str
    content: str
    score: float
    source: RetrievalSource
    source_name: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0


@dataclass
class RetrievalResult:
    """Complete retrieval result"""

    query: str
    results: List[RetrievedChunk]
    total_results: int
    retrieval_time: float
    strategy_used: RetrievalStrategy
    expanded_queries: List[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    """Optimized context for LLM"""

    original_query: str
    retrieved_chunks: List[RetrievedChunk]
    formatted_context: str
    total_tokens: int
    sources_summary: str


class RetrievalAgent:
    """
    Advanced RAG-style retrieval agent using EDIATH GGUF model:
    - Dense vector retrieval (GGUF embeddings)
    - Sparse retrieval (BM25)
    - Hybrid search (dense + sparse fusion)
    - Multi-query expansion
    - HyDE (Hypothetical Document Embeddings)
    - Context optimization and compression
    - Relevance filtering
    - Source diversification
    - Recursive retrieval
    - Query rewriting
    - Metadata filtering
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Retrieval Agent with GGUF model

        Args:
            config: Configuration dictionary with:
                - model_path: Path to GGUF model file
                - n_ctx: Context window size (default: 2048)
                - n_threads: Number of threads (default: 4)
                - embedding_dimension: Embedding dimension (default: 384)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # GGUF Model configuration
        self.model_path = self.config.get("model_path", "./models/EDIATH-q4_k_m.gguf")
        self.n_ctx = self.config.get("n_ctx", 2048)
        self.n_threads = self.config.get("n_threads", 4)
        self.use_gguf = self.config.get("use_gguf", True)
        self.embedding_dimension = self.config.get("embedding_dimension", 384)

        # GGUF model
        self.gguf_model = None

        # Retrieval configuration
        self.default_top_k = self.config.get("default_top_k", 10)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)

        # Hybrid search weights
        self.dense_weight = self.config.get("dense_weight", 0.6)
        self.sparse_weight = self.config.get("sparse_weight", 0.4)

        # Query expansion
        self.query_expansion_enabled = self.config.get("query_expansion_enabled", True)
        self.num_query_expansions = self.config.get("num_query_expansions", 3)

        # Initialize components
        self.bm25_index = None
        self.corpus: List[str] = []
        self.corpus_metadata: List[Dict] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

        # Vector store reference (can be connected to KnowledgeBaseAgent)
        self.vector_store = None
        self.knowledge_base = None

        # Statistics
        self.stats = {
            "total_retrievals": 0,
            "average_retrieval_time": 0.0,
            "average_results_count": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "embedding_model": "GGUF",
        }

        # Cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)
        self.retrieval_cache: Dict[str, tuple] = {}

        # Initialize GGUF model
        self._init_gguf_model()

        self.logger.info(
            f"Retrieval Agent initialized with GGUF model (dimension: {self.embedding_dimension})"
        )

    def _init_gguf_model(self):
        """Initialize GGUF model for embeddings"""
        if not self.use_gguf:
            self.logger.info("GGUF model disabled, using fallback methods")
            return

        if not LLAMA_AVAILABLE:
            self.logger.warning(
                "llama-cpp-python not available. Install with: pip install llama-cpp-python"
            )
            self.use_gguf = False
            return

        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                self.logger.warning(f"GGUF model not found at {model_path}")
                self.use_gguf = False
                return

            self.logger.info(f"Loading GGUF model: {self.model_path}")
            self.gguf_model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                embedding=True,  # Enable embeddings
            )

            # Get actual embedding dimension
            try:
                test_embedding = self.gguf_model.embed("test")
                self.embedding_dimension = len(test_embedding)
                self.logger.info(
                    f"GGUF model loaded, embedding dimension: {self.embedding_dimension}"
                )
            except Exception as e:
                self.logger.warning(f"Could not determine embedding dimension: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load GGUF model: {str(e)}")
            self.use_gguf = False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using GGUF model"""
        if not self.use_gguf or not self.gguf_model:
            return None

        try:
            # Truncate long text
            if len(text) > self.n_ctx * 4:
                text = text[: self.n_ctx * 4]

            embedding = self.gguf_model.embed(text)
            embedding_array = np.array(embedding, dtype=np.float32)

            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

            return embedding_array

        except Exception as e:
            self.logger.debug(f"GGUF embedding failed: {e}")
            return None

    async def index_corpus(
        self, documents: List[Dict[str, Any]], force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index a corpus for retrieval using GGUF embeddings

        Args:
            documents: List of document dicts with 'content' and 'metadata'
            force_reindex: Force reindexing even if already indexed

        Returns:
            Dictionary with indexing result
        """
        if not documents:
            return {"success": False, "error": "Empty corpus"}

        try:
            # Extract content and metadata
            contents = [doc["content"] for doc in documents]
            metadata = [doc.get("metadata", {}) for doc in documents]

            # Build BM25 index (sparse retrieval)
            if BM25_AVAILABLE:
                tokenized_corpus = [self._tokenize(content) for content in contents]
                self.bm25_index = BM25Okapi(tokenized_corpus)
                self.logger.info(f"BM25 index built with {len(contents)} documents")

            # Build dense embeddings using GGUF
            if self.use_gguf and self.gguf_model:
                self.logger.info(
                    f"Generating GGUF embeddings for {len(contents)} documents"
                )
                embeddings = []

                for i, content in enumerate(contents):
                    embedding = self._get_embedding(content)
                    if embedding is not None:
                        embeddings.append(embedding)
                    else:
                        # Use zero vector as fallback
                        embeddings.append(
                            np.zeros(self.embedding_dimension, dtype=np.float32)
                        )

                    if (i + 1) % 50 == 0:
                        self.logger.info(
                            f"  Embedded {i + 1}/{len(contents)} documents"
                        )

                self.corpus_embeddings = np.array(embeddings, dtype=np.float32)

                # Build FAISS index for fast similarity search
                if FAISS_AVAILABLE:
                    self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
                    # Embeddings are already normalized
                    self.faiss_index.add(self.corpus_embeddings)
                    self.logger.info("FAISS index built with GGUF embeddings")

            # Store corpus
            self.corpus = contents
            self.corpus_metadata = metadata

            return {
                "success": True,
                "documents_indexed": len(documents),
                "embedding_dimension": self.embedding_dimension,
                "embedding_model": "GGUF",
                "bm25_available": BM25_AVAILABLE,
                "faiss_available": FAISS_AVAILABLE,
            }

        except Exception as e:
            self.logger.error(f"Indexing error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int = None,
        metadata_filter: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query using GGUF embeddings

        Args:
            query: Search query
            strategy: Retrieval strategy to use
            top_k: Number of results to retrieve
            metadata_filter: Filter by metadata
            use_cache: Use cached results

        Returns:
            RetrievalResult object
        """
        start_time = datetime.now()
        top_k = top_k or self.default_top_k

        # Check cache
        cache_key = self._get_cache_key(query, strategy, top_k)
        if use_cache and self.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached:
                self.stats["cache_hits"] += 1
                return cached

        # Ensure corpus is indexed
        if not self.corpus:
            self.logger.warning("No corpus indexed. Call index_corpus first.")
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                retrieval_time=0,
                strategy_used=strategy,
            )

        try:
            # Apply strategy
            if strategy == RetrievalStrategy.DENSE:
                results = await self._dense_retrieval(query, top_k * 2)

            elif strategy == RetrievalStrategy.SPARSE:
                results = await self._sparse_retrieval(query, top_k * 2)

            elif strategy == RetrievalStrategy.HYBRID:
                results = await self._hybrid_retrieval(query, top_k * 2)

            elif strategy == RetrievalStrategy.MULTI_QUERY:
                results = await self._multi_query_retrieval(query, top_k * 2)

            elif strategy == RetrievalStrategy.HYPOTHETICAL:
                results = await self._hypothetical_retrieval(query, top_k * 2)

            else:
                results = await self._dense_retrieval(query, top_k * 2)

            # Apply metadata filter
            if metadata_filter:
                results = self._apply_metadata_filter(results, metadata_filter)

            # Take top-k
            results = results[:top_k]

            # Calculate relevance scores
            for result in results:
                result.relevance_score = self._calculate_relevance(
                    query, result.content
                )

            # Create result object
            retrieval_time = (datetime.now() - start_time).total_seconds()
            result_obj = RetrievalResult(
                query=query,
                results=results,
                total_results=len(results),
                retrieval_time=retrieval_time,
                strategy_used=strategy,
            )

            # Update statistics
            self.stats["total_retrievals"] += 1
            self.stats["average_retrieval_time"] = (
                self.stats["average_retrieval_time"]
                * (self.stats["total_retrievals"] - 1)
                + retrieval_time
            ) / self.stats["total_retrievals"]
            self.stats["average_results_count"] = (
                self.stats["average_results_count"]
                * (self.stats["total_retrievals"] - 1)
                + len(results)
            ) / self.stats["total_retrievals"]

            # Cache result
            self._add_to_cache(cache_key, result_obj)

            return result_obj

        except Exception as e:
            self.logger.error(f"Retrieval error: {str(e)}")
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                retrieval_time=0,
                strategy_used=strategy,
            )

    async def _dense_retrieval(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Dense vector retrieval using GGUF embeddings"""
        if not self.use_gguf or not self.gguf_model or self.corpus_embeddings is None:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return []

        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        if FAISS_AVAILABLE and self.faiss_index:
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), top_k
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Manual similarity search
            similarities = np.dot(self.corpus_embeddings, query_embedding)
            indices = np.argsort(similarities)[-top_k:][::-1]
            scores = similarities[indices]

        # Build results
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(self.corpus) and score > 0:
                results.append(
                    RetrievedChunk(
                        id=f"dense_{idx}",
                        content=self.corpus[idx][:1000],
                        score=float(score),
                        source=RetrievalSource.KNOWLEDGE_BASE,
                        source_name="gguf_dense_retrieval",
                        metadata=(
                            self.corpus_metadata[idx]
                            if idx < len(self.corpus_metadata)
                            else {}
                        ),
                    )
                )

        return results

    async def _sparse_retrieval(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Sparse retrieval using BM25"""
        if not BM25_AVAILABLE or self.bm25_index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    RetrievedChunk(
                        id=f"sparse_{idx}",
                        content=self.corpus[idx][:1000],
                        score=float(scores[idx]),
                        source=RetrievalSource.KNOWLEDGE_BASE,
                        source_name="bm25_retrieval",
                        metadata=(
                            self.corpus_metadata[idx]
                            if idx < len(self.corpus_metadata)
                            else {}
                        ),
                    )
                )

        return results

    async def _hybrid_retrieval(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Hybrid retrieval combining GGUF dense and BM25 sparse"""
        # Get results from both methods
        dense_results = await self._dense_retrieval(query, top_k * 2)
        sparse_results = await self._sparse_retrieval(query, top_k * 2)

        # Combine and normalize scores
        all_results = {}

        for result in dense_results:
            all_results[result.id] = {
                "result": result,
                "dense_score": result.score,
                "sparse_score": 0,
            }

        for result in sparse_results:
            if result.id in all_results:
                all_results[result.id]["sparse_score"] = result.score
            else:
                all_results[result.id] = {
                    "result": result,
                    "dense_score": 0,
                    "sparse_score": result.score,
                }

        # Normalize scores
        dense_scores = [
            v["dense_score"] for v in all_results.values() if v["dense_score"] > 0
        ]
        sparse_scores = [
            v["sparse_score"] for v in all_results.values() if v["sparse_score"] > 0
        ]

        max_dense = max(dense_scores) if dense_scores else 1
        max_sparse = max(sparse_scores) if sparse_scores else 1

        # Compute hybrid scores
        hybrid_results = []
        for item in all_results.values():
            norm_dense = item["dense_score"] / max_dense if max_dense > 0 else 0
            norm_sparse = item["sparse_score"] / max_sparse if max_sparse > 0 else 0

            hybrid_score = (self.dense_weight * norm_dense) + (
                self.sparse_weight * norm_sparse
            )
            item["result"].score = hybrid_score
            hybrid_results.append(item["result"])

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        return hybrid_results[:top_k]

    async def _multi_query_retrieval(
        self, query: str, top_k: int
    ) -> List[RetrievedChunk]:
        """Multi-query retrieval with query expansion"""
        # Generate query variations
        variations = self._expand_query(query)

        # Retrieve for each variation
        all_results = {}

        for var_query in variations[: self.num_query_expansions]:
            results = await self._hybrid_retrieval(var_query, top_k)

            for result in results:
                if result.id not in all_results:
                    all_results[result.id] = result
                    all_results[result.id].score = 0
                all_results[result.id].score += result.score

        # Average scores
        for result in all_results.values():
            result.score /= len(variations)

        # Sort and return
        results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)

        return results[:top_k]

    async def _hypothetical_retrieval(
        self, query: str, top_k: int
    ) -> List[RetrievedChunk]:
        """HyDE - Hypothetical Document Embeddings using GGUF"""
        # Generate hypothetical answer/document using GGUF
        hypothetical = await self._generate_hypothetical_gguf(query)

        # Use hypothetical for retrieval
        return await self._hybrid_retrieval(hypothetical, top_k)

    async def _generate_hypothetical_gguf(self, query: str) -> str:
        """Generate hypothetical document using GGUF model"""
        if self.use_gguf and self.gguf_model:
            prompt = f"""Given the question: "{query}"
Write a hypothetical answer that would be found in a relevant document. Write a short paragraph:

Hypothetical document:"""

            try:
                response = await asyncio.to_thread(
                    self.gguf_model.create_completion,
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.5,
                    stop=["Question:", "---"],
                )
                hypothetical = response["choices"][0]["text"].strip()
                if hypothetical and len(hypothetical) > 20:
                    return hypothetical
            except Exception as e:
                self.logger.debug(f"GGUF hypothetical generation failed: {e}")

        # Fallback: use expanded query
        expanded = self._expand_query(query)
        return " ".join(expanded[:2]) if expanded else query

    async def retrieve_with_context(
        self, query: str, max_tokens: int = 2000, **kwargs
    ) -> RetrievalContext:
        """
        Retrieve and format context for LLM

        Args:
            query: User query
            max_tokens: Maximum tokens for context
            **kwargs: Additional retrieval parameters

        Returns:
            RetrievalContext object with formatted context
        """
        # Retrieve results
        result = await self.retrieve(query, **kwargs)

        if not result.results:
            return RetrievalContext(
                original_query=query,
                retrieved_chunks=[],
                formatted_context="No relevant information found.",
                total_tokens=0,
                sources_summary="No sources",
            )

        # Format context
        context_parts = []
        sources = []
        total_tokens = 0

        for i, chunk in enumerate(result.results):
            # Estimate tokens (rough: 1 token per 4 chars)
            chunk_tokens = len(chunk.content) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(
                f"[Source {i+1} - {chunk.source_name} (score: {chunk.score:.3f})]\n{chunk.content}"
            )
            sources.append(f"{i+1}. {chunk.source_name}")
            total_tokens += chunk_tokens

        formatted_context = "\n\n---\n\n".join(context_parts)

        return RetrievalContext(
            original_query=query,
            retrieved_chunks=result.results[: len(context_parts)],
            formatted_context=formatted_context,
            total_tokens=total_tokens,
            sources_summary="\n".join(sources),
        )

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with variations"""
        variations = [query]

        # Lowercase variation
        variations.append(query.lower())

        # Remove stopwords (simple)
        stopwords = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
        }
        words = query.split()
        filtered = [w for w in words if w.lower() not in stopwords]
        variations.append(" ".join(filtered))

        # Add key terms only
        if len(words) > 3:
            variations.append(" ".join(words[:3]))

        return list(set(variations))

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

    def _apply_metadata_filter(
        self, results: List[RetrievedChunk], metadata_filter: Dict
    ) -> List[RetrievedChunk]:
        """Apply metadata filter to results"""
        filtered = []
        for result in results:
            match = True
            for key, value in metadata_filter.items():
                if result.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(result)
        return filtered

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_terms = set(self._tokenize(query))
        content_terms = set(self._tokenize(content))

        if not query_terms:
            return 0.0

        intersection = len(query_terms & content_terms)
        union = len(query_terms | content_terms)

        return intersection / union if union > 0 else 0.0

    def _get_cache_key(
        self, query: str, strategy: RetrievalStrategy, top_k: int
    ) -> str:
        """Generate cache key"""
        key_data = f"{query}:{strategy.value}:{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[RetrievalResult]:
        """Get from cache"""
        if cache_key in self.retrieval_cache:
            timestamp, result = self.retrieval_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return result
            else:
                del self.retrieval_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, result: RetrievalResult):
        """Add to cache"""
        if len(self.retrieval_cache) > 1000:
            items = sorted(self.retrieval_cache.items(), key=lambda x: x[1][0])
            for key, _ in items[:100]:
                del self.retrieval_cache[key]

        self.retrieval_cache[cache_key] = (datetime.now(), result)
        self.stats["cache_misses"] += 1

    async def search_similar(self, text: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Search for similar text in indexed corpus"""
        result = await self._dense_retrieval(text, top_k)
        return result

    async def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document by ID"""
        for idx, doc in enumerate(self.corpus):
            if f"dense_{idx}" == doc_id or f"sparse_{idx}" == doc_id:
                return {
                    "content": doc,
                    "metadata": (
                        self.corpus_metadata[idx]
                        if idx < len(self.corpus_metadata)
                        else {}
                    ),
                    "index": idx,
                }
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "corpus_size": len(self.corpus),
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": "GGUF",
            "use_gguf": self.use_gguf,
            "model_path": str(self.model_path) if self.use_gguf else None,
            "cache_size": len(self.retrieval_cache),
            "bm25_available": BM25_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE,
        }

    def clear_cache(self):
        """Clear retrieval cache"""
        self.retrieval_cache.clear()
        self.logger.info("Retrieval cache cleared")

    def clear_corpus(self):
        """Clear indexed corpus"""
        self.corpus = []
        self.corpus_metadata = []
        self.corpus_embeddings = None
        self.bm25_index = None
        self.faiss_index = None
        self.logger.info("Corpus cleared")


# Integration wrapper for EDIATH
class RetrievalAgentWrapper:
    """
    Wrapper class to integrate RetrievalAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.retrieval_agent = RetrievalAgent(config)
        self.agent_type = "retrieval"
        self.capabilities = [
            "index_corpus",
            "retrieve",
            "retrieve_with_context",
            "search_similar",
            "hybrid_search",
            "gguf_embeddings",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a retrieval request"""
        operation = request.get("operation")

        if operation == "index":
            return await self.retrieval_agent.index_corpus(
                documents=request.get("documents", []),
                force_reindex=request.get("force_reindex", False),
            )

        elif operation == "retrieve":
            strategy = request.get("strategy", "hybrid")
            result = await self.retrieval_agent.retrieve(
                query=request.get("query"),
                strategy=RetrievalStrategy(strategy),
                top_k=request.get("top_k"),
                metadata_filter=request.get("metadata_filter"),
                use_cache=request.get("use_cache", True),
            )

            return {
                "success": True,
                "query": result.query,
                "results": [
                    {
                        "content": r.content,
                        "score": r.score,
                        "relevance_score": r.relevance_score,
                        "source": r.source.value,
                        "source_name": r.source_name,
                        "metadata": r.metadata,
                    }
                    for r in result.results
                ],
                "total_results": result.total_results,
                "retrieval_time": result.retrieval_time,
                "strategy_used": result.strategy_used.value,
                "embedding_model": "GGUF",
            }

        elif operation == "context":
            result = await self.retrieval_agent.retrieve_with_context(
                query=request.get("query"),
                max_tokens=request.get("max_tokens", 2000),
                strategy=RetrievalStrategy(request.get("strategy", "hybrid")),
                top_k=request.get("top_k"),
            )

            return {
                "success": True,
                "query": result.original_query,
                "context": result.formatted_context,
                "total_tokens": result.total_tokens,
                "sources_summary": result.sources_summary,
                "chunks_used": len(result.retrieved_chunks),
            }

        elif operation == "similar":
            results = await self.retrieval_agent.search_similar(
                text=request.get("text"), top_k=request.get("top_k", 5)
            )

            return {
                "success": True,
                "results": [
                    {"content": r.content, "score": r.score, "source": r.source.value}
                    for r in results
                ],
            }

        elif operation == "stats":
            return self.retrieval_agent.get_stats()

        elif operation == "clear_cache":
            self.retrieval_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        elif operation == "clear_corpus":
            self.retrieval_agent.clear_corpus()
            return {"success": True, "message": "Corpus cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "RetrievalAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.retrieval_agent.get_stats(),
            "strategies": [s.value for s in RetrievalStrategy],
            "embedding_model": "GGUF",
            "model_path": self.retrieval_agent.model_path,
        }


# Example usage
async def test_retrieval_agent():
    """Test the retrieval agent functionality with GGUF"""

    config = {
        "model_path": "./models/EDIATH-q4_k_m.gguf",
        "use_gguf": True,
        "n_ctx": 2048,
        "n_threads": 4,
    }

    agent = RetrievalAgent(config)
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=== Retrieval Agent Test with GGUF ===\n")
    stats = agent.get_stats()
    logger.info("Embedding Model: %s", stats.get("embedding_model"))
    logger.info("Model Path: %s", stats.get("model_path", "N/A"))
    logger.info("Embedding Dimension: %s", stats.get("embedding_dimension"))

    # Create test corpus
    documents = [
        {
            "content": "EDIATH is an advanced AI assistant that helps users with various tasks including code execution, file management, and web browsing.",
            "metadata": {"category": "overview", "topic": "general"},
        },
        {
            "content": "The code execution feature allows running Python, JavaScript, Bash, and other languages in a secure sandboxed environment.",
            "metadata": {"category": "features", "topic": "code"},
        },
        {
            "content": "File management capabilities include reading, writing, editing, searching, and organizing files with security boundaries.",
            "metadata": {"category": "features", "topic": "files"},
        },
        {
            "content": "Web browsing and automation can navigate pages, click elements, fill forms, and extract data from websites.",
            "metadata": {"category": "features", "topic": "web"},
        },
        {
            "content": "The knowledge base system stores documents locally and provides RAG-based question answering using vector search.",
            "metadata": {"category": "features", "topic": "knowledge"},
        },
    ]

    # Index corpus
    logger.info("\n1. Indexing with GGUF Embeddings...")
    result = await agent.index_corpus(documents)
    logger.info("   Indexed %s documents", result.get("documents_indexed"))
    logger.info("   Embedding dimension: %s", result.get("embedding_dimension"))
    logger.info("   Model: %s", result.get("embedding_model"))

    # Test retrieval
    logger.info("\n2. Testing Retrieval...")
    result = await agent.retrieve(
        "What programming languages can be executed?",
        strategy=RetrievalStrategy.HYBRID,
        top_k=3,
    )
    logger.info(
        "   Found %s results in %.3fs", result.total_results, result.retrieval_time
    )
    for r in result.results:
        logger.info("     Score %.3f: %s...", r.score, r.content[:80])

    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    from pathlib import Path

    asyncio.run(test_retrieval_agent())
