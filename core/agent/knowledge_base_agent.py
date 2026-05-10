"""
Knowledge Base Agent for EDIATH
Local document Q&A: document ingestion, vector search, semantic retrieval, RAG-based answers
Uses EDIATH GGUF model ONLY for embeddings and answer generation (no fallback)
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import logging
import re

# Document parsing
try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        UnstructuredWordDocumentLoader,
        CSVLoader,
        JSONLoader,
        UnstructuredMarkdownLoader,
    )

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# GGUF Model for embeddings (using llama-cpp-python) - ONLY embedding source
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Vector databases
try:
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class VectorStoreType(Enum):
    """Vector database types"""

    CHROMA = "chroma"
    QDRANT = "qdrant"
    IN_MEMORY = "in_memory"


class DocumentStatus(Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Document:
    """Document information"""

    id: str
    name: str
    path: Path
    type: str
    size: int
    status: DocumentStatus
    chunks: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Document chunk for embedding"""

    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result"""

    content: str
    score: float
    document_name: str
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass
class Answer:
    """Answer with sources"""

    question: str
    answer: str
    sources: List[SearchResult]
    confidence: float
    processing_time: float


class KnowledgeBaseAgent:
    """
    Advanced knowledge base agent using EDIATH GGUF model ONLY:
    - Document ingestion (PDF, DOCX, TXT, MD, CSV, JSON)
    - Text chunking and embedding (via GGUF)
    - Vector similarity search
    - Semantic search and retrieval
    - RAG-based question answering
    - Document management
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Knowledge Base Agent with GGUF model

        Args:
            config: Configuration dictionary with:
                - model_path: Path to GGUF model file (required)
                - n_ctx: Context window size (default: 2048)
                - n_threads: Number of threads (default: 4)
                - data_dir: Directory for storing knowledge base (default: './knowledge_base')
                - chunk_size: Size of text chunks (default: 500)
                - chunk_overlap: Overlap between chunks (default: 50)
                - default_top_k: Default number of search results (default: 5)
                - similarity_threshold: Minimum similarity score (default: 0.5)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Storage paths
        self.data_dir = Path(self.config.get("data_dir", "./knowledge_base"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir = self.data_dir / "documents"
        self.documents_dir.mkdir(exist_ok=True)
        self.chunks_dir = self.data_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)

        # GGUF Model configuration (required)
        self.model_path = self.config.get("model_path", "./models/EDIATH-q4_k_m.gguf")
        self.n_ctx = self.config.get("n_ctx", 2048)
        self.n_threads = self.config.get("n_threads", 4)

        # Vector store configuration
        self.vector_store_type = VectorStoreType(
            self.config.get("vector_store", "in_memory")
        )
        self.embedding_dimension = self.config.get("embedding_dimension", 384)

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)

        # Search configuration
        self.default_top_k = self.config.get("default_top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.5)

        # Initialize components
        self.gguf_model = None
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, DocumentChunk] = {}

        # Use GGUF for answer generation
        self.use_llm = self.config.get("use_llm", True)

        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "successful_queries": 0,
            "average_search_time": 0.0,
            "embedding_model": "GGUF",
        }

        # Query history
        self.query_history: List[Answer] = []
        self.max_history = self.config.get("max_history", 100)

        # Initialize GGUF model (required - no fallback)
        self._init_gguf_model()
        self._init_vector_store()
        self._load_metadata()

        if not self.gguf_model:
            raise RuntimeError(
                f"Failed to load GGUF model from {self.model_path}. Cannot proceed."
            )

        self.logger.info(
            f"Knowledge Base Agent initialized with GGUF model (dimension: {self.embedding_dimension})"
        )

    def _init_gguf_model(self):
        """Initialize GGUF model for embeddings and generation (required)"""
        if not LLAMA_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python not available. Install with: pip install llama-cpp-python"
            )

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found at {model_path}")

        try:
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
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using GGUF model (only source)

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        # Truncate long text
        if len(text) > self.n_ctx * 4:
            text = text[: self.n_ctx * 4]

        try:
            embedding = self.gguf_model.embed(text)
            return (
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            )
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}")
            raise

    async def _get_embedding_async(self, text: str) -> List[float]:
        """Async wrapper for getting embedding"""
        return await asyncio.to_thread(self._get_embedding, text)

    def _init_vector_store(self):
        """Initialize vector database with GGUF custom embedding"""
        if self.vector_store_type == VectorStoreType.CHROMA and CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.data_dir / "chroma_db")
                )

                # Custom embedding function for ChromaDB using GGUF
                class GGUFEmbeddingFunction(embedding_functions.EmbeddingFunction):
                    def __init__(self, agent):
                        self.agent = agent

                    def __call__(self, texts):
                        return [self.agent._get_embedding(text) for text in texts]

                embedding_fn = GGUFEmbeddingFunction(self)

                self.collection = self.chroma_client.get_or_create_collection(
                    name="knowledge_base", embedding_function=embedding_fn
                )
                self.logger.info("ChromaDB initialized with GGUF embeddings")
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
                self.vector_store_type = VectorStoreType.IN_MEMORY

        elif self.vector_store_type == VectorStoreType.QDRANT and QDRANT_AVAILABLE:
            try:
                self.qdrant_client = QdrantClient(path=str(self.data_dir / "qdrant_db"))
                self.qdrant_client.recreate_collection(
                    collection_name="knowledge_base",
                    vectors_config=VectorParams(
                        size=self.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                self.logger.info("Qdrant initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Qdrant: {str(e)}")
                self.vector_store_type = VectorStoreType.IN_MEMORY

        else:
            self.vector_store_type = VectorStoreType.IN_MEMORY
            self.in_memory_vectors: Dict[str, List[float]] = {}
            self.logger.info("Using in-memory vector store with GGUF embeddings")

    def _load_metadata(self):
        """Load document and chunk metadata from disk"""
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                    # Load documents
                    for doc_data in data.get("documents", []):
                        doc = Document(
                            id=doc_data["id"],
                            name=doc_data["name"],
                            path=Path(doc_data["path"]),
                            type=doc_data["type"],
                            size=doc_data["size"],
                            status=DocumentStatus(doc_data["status"]),
                            chunks=doc_data["chunks"],
                            created_at=datetime.fromisoformat(doc_data["created_at"]),
                            metadata=doc_data.get("metadata", {}),
                        )
                        self.documents[doc.id] = doc

                    # Load chunks (without embeddings for memory efficiency)
                    for chunk_data in data.get("chunks", []):
                        chunk = DocumentChunk(
                            id=chunk_data["id"],
                            document_id=chunk_data["document_id"],
                            content=chunk_data["content"],
                            metadata=chunk_data.get("metadata", {}),
                        )
                        self.chunks[chunk.id] = chunk

                    self.stats["total_documents"] = len(self.documents)
                    self.stats["total_chunks"] = len(self.chunks)

                    self.logger.info(
                        f"Loaded {len(self.documents)} documents and {len(self.chunks)} chunks"
                    )

            except Exception as e:
                self.logger.error(f"Failed to load metadata: {str(e)}")

    def _save_metadata(self):
        """Save document and chunk metadata to disk"""
        metadata = {
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "path": str(doc.path),
                    "type": doc.type,
                    "size": doc.size,
                    "status": doc.status.value,
                    "chunks": doc.chunks,
                    "created_at": doc.created_at.isoformat(),
                    "metadata": doc.metadata,
                }
                for doc in self.documents.values()
            ],
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks.values()
            ],
        }

        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    async def add_document(
        self, file_path: Union[str, Path], metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Add a document to the knowledge base"""
        file_path = Path(file_path)

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        # Check if already added
        for doc in self.documents.values():
            if doc.path == file_path:
                return {
                    "success": False,
                    "error": f"Document already in knowledge base: {file_path}",
                }

        # Determine document type
        ext = file_path.suffix.lower()
        doc_type = {
            ".txt": "text",
            ".pdf": "pdf",
            ".docx": "docx",
            ".md": "markdown",
            ".csv": "csv",
            ".json": "json",
        }.get(ext, "unknown")

        if doc_type == "unknown":
            return {"success": False, "error": f"Unsupported file type: {ext}"}

        # Create document record
        doc_id = self._generate_id()
        document = Document(
            id=doc_id,
            name=file_path.name,
            path=file_path,
            type=doc_type,
            size=file_path.stat().st_size,
            status=DocumentStatus.PENDING,
            metadata=metadata or {},
        )

        self.documents[doc_id] = document
        self._save_metadata()

        # Process document asynchronously
        asyncio.create_task(self._process_document(doc_id))

        return {
            "success": True,
            "document_id": doc_id,
            "name": file_path.name,
            "message": f"Document {file_path.name} added, processing started",
        }

    async def add_documents_batch(
        self, file_paths: List[Union[str, Path]]
    ) -> Dict[str, Any]:
        """Add multiple documents to the knowledge base"""
        results = []
        for file_path in file_paths:
            result = await self.add_document(file_path)
            results.append(result)

        successful = sum(1 for r in results if r["success"])

        return {
            "success": successful > 0,
            "total": len(file_paths),
            "successful": successful,
            "failed": len(file_paths) - successful,
            "results": results,
        }

    async def _process_document(self, doc_id: str):
        """Process document: parse, chunk, embed, index"""
        document = self.documents.get(doc_id)
        if not document:
            return

        try:
            document.status = DocumentStatus.PROCESSING
            self._save_metadata()

            # Parse document
            chunks = await self._parse_document(document)

            # Create embeddings and index
            for chunk in chunks:
                await self._index_chunk(chunk)
                self.chunks[chunk.id] = chunk

            document.chunks = len(chunks)
            document.status = DocumentStatus.COMPLETED

            self.stats["total_documents"] += 1
            self.stats["total_chunks"] += len(chunks)

            self.logger.info(
                f"Processed document {document.name}: {len(chunks)} chunks"
            )

        except Exception as e:
            self.logger.error(f"Failed to process document {document.name}: {str(e)}")
            document.status = DocumentStatus.FAILED

        finally:
            self._save_metadata()

    async def _parse_document(self, document: Document) -> List[DocumentChunk]:
        """Parse document and split into chunks"""
        # Read document content based on type
        if document.type == "text":
            with open(document.path, "r", encoding="utf-8") as f:
                content = f.read()

        elif document.type == "pdf":
            if LANGCHAIN_AVAILABLE:
                loader = PDFMinerLoader(str(document.path))
                pages = loader.load()
                content = "\n".join([page.page_content for page in pages])
            else:
                try:
                    import PyPDF2

                    content = ""
                    with open(document.path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            content += page.extract_text()
                except ImportError:
                    raise Exception("PDF parsing requires langchain or PyPDF2")

        elif document.type == "docx":
            if LANGCHAIN_AVAILABLE:
                loader = UnstructuredWordDocumentLoader(str(document.path))
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
            else:
                try:
                    import docx

                    doc = docx.Document(document.path)
                    content = "\n".join(
                        [paragraph.text for paragraph in doc.paragraphs]
                    )
                except ImportError:
                    raise Exception("DOCX parsing requires langchain or python-docx")

        elif document.type == "markdown":
            with open(document.path, "r", encoding="utf-8") as f:
                content = f.read()

        elif document.type == "csv":
            import csv

            content = ""
            with open(document.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content += json.dumps(row) + "\n"

        elif document.type == "json":
            with open(document.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                content = json.dumps(data, indent=2)

        else:
            raise ValueError(f"Unsupported document type: {document.type}")

        # Split into chunks
        chunks = await self._split_text(content, document)

        return chunks

    async def _split_text(self, text: str, document: Document) -> List[DocumentChunk]:
        """Split text into chunks"""
        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            split_texts = text_splitter.split_text(text)
        else:
            # Simple chunking
            split_texts = []
            words = text.split()
            chunk = []
            chunk_length = 0

            for word in words:
                chunk_length += len(word) + 1
                if chunk_length > self.chunk_size and chunk:
                    split_texts.append(" ".join(chunk))
                    chunk = []
                    chunk_length = 0
                chunk.append(word)

            if chunk:
                split_texts.append(" ".join(chunk))

        chunks = []
        for i, chunk_text in enumerate(split_texts):
            chunk_id = self._generate_id()
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_text.strip(),
                    metadata={
                        "document_name": document.name,
                        "chunk_index": i,
                        "total_chunks": len(split_texts),
                        **document.metadata,
                    },
                )
            )

        return chunks

    async def _index_chunk(self, chunk: DocumentChunk):
        """Index a chunk in the vector store using GGUF"""
        # Generate embedding using GGUF
        chunk.embedding = await self._get_embedding_async(chunk.content)

        # Store in vector database
        if self.vector_store_type == VectorStoreType.CHROMA and CHROMADB_AVAILABLE:
            self.collection.add(
                ids=[chunk.id],
                embeddings=[chunk.embedding],
                metadatas=[chunk.metadata],
                documents=[chunk.content],
            )

        elif self.vector_store_type == VectorStoreType.QDRANT and QDRANT_AVAILABLE:
            self.qdrant_client.upsert(
                collection_name="knowledge_base",
                points=[
                    {
                        "id": hash(chunk.id) % (2**31),
                        "vector": chunk.embedding,
                        "payload": {
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        },
                    }
                ],
            )

        else:  # IN_MEMORY
            self.in_memory_vectors[chunk.id] = chunk.embedding

    async def search(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[Dict] = None,
        min_score: float = None,
    ) -> Dict[str, Any]:
        """Search the knowledge base using GGUF embeddings"""
        start_time = datetime.now()
        top_k = top_k or self.default_top_k
        min_score = min_score or self.similarity_threshold

        try:
            # Generate query embedding using GGUF
            query_embedding = await self._get_embedding_async(query)

            # Search in vector store
            if self.vector_store_type == VectorStoreType.CHROMA and CHROMADB_AVAILABLE:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=metadata_filter,
                )

                search_results = []
                if results["ids"] and results["ids"][0]:
                    for i, chunk_id in enumerate(results["ids"][0]):
                        score = 1 - (
                            results["distances"][0][i] if results["distances"] else 0
                        )
                        if score >= min_score:
                            chunk = self.chunks.get(chunk_id)
                            if chunk:
                                search_results.append(
                                    SearchResult(
                                        content=chunk.content,
                                        score=score,
                                        document_name=chunk.metadata.get(
                                            "document_name", "Unknown"
                                        ),
                                        chunk_id=chunk_id,
                                        metadata=chunk.metadata,
                                    )
                                )

            elif self.vector_store_type == VectorStoreType.QDRANT and QDRANT_AVAILABLE:
                results = self.qdrant_client.search(
                    collection_name="knowledge_base",
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=min_score,
                )

                search_results = []
                for result in results:
                    chunk = self.chunks.get(result.payload.get("chunk_id", ""))
                    if chunk:
                        search_results.append(
                            SearchResult(
                                content=chunk.content,
                                score=result.score,
                                document_name=chunk.metadata.get(
                                    "document_name", "Unknown"
                                ),
                                chunk_id=result.payload.get("chunk_id", ""),
                                metadata=chunk.metadata,
                            )
                        )

            else:  # IN_MEMORY
                search_results = await self._in_memory_search(
                    query_embedding, top_k, min_score, metadata_filter
                )

            # Sort by score
            search_results.sort(key=lambda x: x.score, reverse=True)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_queries"] += 1
            self.stats["successful_queries"] += 1
            self.stats["average_search_time"] = (
                self.stats["average_search_time"] * (self.stats["total_queries"] - 1)
                + processing_time
            ) / self.stats["total_queries"]

            return {
                "success": True,
                "query": query,
                "results": search_results,
                "total_results": len(search_results),
                "processing_time": processing_time,
                "embedding_model": "GGUF",
            }

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return {"success": False, "error": str(e), "query": query}

    async def _in_memory_search(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        metadata_filter: Optional[Dict],
    ) -> List[SearchResult]:
        """Perform in-memory vector search with GGUF embeddings"""
        if not SKLEARN_AVAILABLE:
            # Fallback to simple similarity
            results = []
            for chunk_id, embedding in self.in_memory_vectors.items():
                similarity = sum(a * b for a, b in zip(query_embedding, embedding))
                if similarity >= min_score:
                    chunk = self.chunks.get(chunk_id)
                    if chunk:
                        if metadata_filter:
                            match = all(
                                chunk.metadata.get(k) == v
                                for k, v in metadata_filter.items()
                            )
                            if not match:
                                continue

                        results.append(
                            SearchResult(
                                content=chunk.content[:500],
                                score=similarity,
                                document_name=chunk.metadata.get(
                                    "document_name", "Unknown"
                                ),
                                chunk_id=chunk_id,
                                metadata=chunk.metadata,
                            )
                        )

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        else:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            chunk_ids = list(self.in_memory_vectors.keys())
            if not chunk_ids:
                return []

            embeddings = np.array([self.in_memory_vectors[cid] for cid in chunk_ids])
            query_vec = np.array(query_embedding).reshape(1, -1)

            similarities = cosine_similarity(query_vec, embeddings)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= min_score:
                    chunk_id = chunk_ids[idx]
                    chunk = self.chunks.get(chunk_id)
                    if chunk:
                        if metadata_filter:
                            match = all(
                                chunk.metadata.get(k) == v
                                for k, v in metadata_filter.items()
                            )
                            if not match:
                                continue

                        results.append(
                            SearchResult(
                                content=chunk.content[:500],
                                score=float(similarity),
                                document_name=chunk.metadata.get(
                                    "document_name", "Unknown"
                                ),
                                chunk_id=chunk_id,
                                metadata=chunk.metadata,
                            )
                        )

            return results

    async def ask(
        self, question: str, top_k: int = None, use_context: bool = True
    ) -> Dict[str, Any]:
        """Ask a question and get an answer using GGUF model for generation"""
        start_time = datetime.now()

        # Search for relevant chunks
        search_result = await self.search(question, top_k=top_k)

        if not search_result["success"]:
            return search_result

        if not search_result["results"]:
            return {
                "success": True,
                "question": question,
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "embedding_model": "GGUF",
            }

        # Generate answer using GGUF model
        answer = await self._generate_answer_with_gguf(
            question, search_result["results"]
        )

        # Create answer object
        answer_obj = Answer(
            question=question,
            answer=answer["answer"],
            sources=search_result["results"],
            confidence=answer.get("confidence", 0.7),
            processing_time=(datetime.now() - start_time).total_seconds(),
        )

        # Add to history
        self.query_history.append(answer_obj)
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history :]

        return {
            "success": True,
            "question": question,
            "answer": answer_obj.answer,
            "sources": [
                {
                    "content": s.content[:200],
                    "score": s.score,
                    "document_name": s.document_name,
                }
                for s in answer_obj.sources[:3]
            ],
            "confidence": answer_obj.confidence,
            "processing_time": answer_obj.processing_time,
            "embedding_model": "GGUF",
        }

    async def _generate_answer_with_gguf(
        self, question: str, sources: List[SearchResult]
    ) -> Dict:
        """Generate answer using GGUF model with RAG context"""
        # Combine context from top sources
        context = "\n\n".join(
            [f"Source {i+1}: {s.content}" for i, s in enumerate(sources[:3])]
        )

        # Truncate context if too long
        max_context_length = self.n_ctx - 500
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # Create prompt for the model
        prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer concisely and accurately based only on the context above. If the context doesn't contain the answer, say "I cannot find that information in the knowledge base."

Answer:"""

        try:
            # Generate answer using GGUF
            response = await asyncio.to_thread(
                self.gguf_model.create_completion,
                prompt=prompt,
                max_tokens=300,
                temperature=0.3,
                top_p=0.9,
                stop=["\n\n", "Context:", "Question:"],
            )

            answer_text = response["choices"][0]["text"].strip()
            confidence = 0.8 if len(answer_text) > 10 else 0.5

            return {"answer": answer_text, "confidence": confidence}

        except Exception as e:
            self.logger.error(f"GGUF generation error: {e}")
            # Fallback to extractive method
            return await self._generate_answer_extractive(question, sources)

    async def _generate_answer_extractive(
        self, question: str, sources: List[SearchResult]
    ) -> Dict:
        """Fallback extractive answer generation"""
        question_terms = set(question.lower().split())

        best_sentence = None
        best_score = 0

        for source in sources[:3]:
            sentences = re.split(r"[.!?]+", source.content)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_lower = sentence.lower()
                matches = sum(1 for term in question_terms if term in sentence_lower)
                score = matches / len(question_terms) if question_terms else 0

                if score > 0:
                    length_score = min(1.0, 100 / len(sentence))
                    total_score = score * 0.7 + length_score * 0.3

                    if total_score > best_score:
                        best_score = total_score
                        best_sentence = sentence

        if best_sentence and best_score > 0.3:
            return {"answer": best_sentence, "confidence": best_score}
        else:
            if sources:
                return {
                    "answer": f"Based on the documents: {sources[0].content[:300]}...",
                    "confidence": 0.5,
                }
            else:
                return {
                    "answer": "I couldn't find specific information to answer your question.",
                    "confidence": 0.0,
                }

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the knowledge base"""
        if document_id not in self.documents:
            return {"success": False, "error": f"Document {document_id} not found"}

        document = self.documents[document_id]

        # Remove chunks from vector store
        chunks_to_delete = [
            chunk for chunk in self.chunks.values() if chunk.document_id == document_id
        ]

        for chunk in chunks_to_delete:
            if self.vector_store_type == VectorStoreType.CHROMA and CHROMADB_AVAILABLE:
                self.collection.delete(ids=[chunk.id])
            elif self.vector_store_type == VectorStoreType.QDRANT and QDRANT_AVAILABLE:
                self.qdrant_client.delete(
                    collection_name="knowledge_base",
                    points_selector=[hash(chunk.id) % (2**31)],
                )
            else:
                if chunk.id in self.in_memory_vectors:
                    del self.in_memory_vectors[chunk.id]

            del self.chunks[chunk.id]

        del self.documents[document_id]

        self.stats["total_documents"] = len(self.documents)
        self.stats["total_chunks"] = len(self.chunks)

        self._save_metadata()
        self.logger.info(f"Deleted document: {document.name}")

        return {
            "success": True,
            "document_id": document_id,
            "name": document.name,
            "message": f"Document {document.name} deleted",
        }

    async def list_documents(self) -> Dict[str, Any]:
        """List all documents in the knowledge base"""
        documents = []
        for doc in self.documents.values():
            documents.append(
                {
                    "id": doc.id,
                    "name": doc.name,
                    "type": doc.type,
                    "size": doc.size,
                    "status": doc.status.value,
                    "chunks": doc.chunks,
                    "created_at": doc.created_at.isoformat(),
                    "metadata": doc.metadata,
                }
            )

        return {"success": True, "total": len(documents), "documents": documents}

    async def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get detailed information about a document"""
        if document_id not in self.documents:
            return {"success": False, "error": f"Document {document_id} not found"}

        doc = self.documents[document_id]
        doc_chunks = [
            chunk for chunk in self.chunks.values() if chunk.document_id == document_id
        ]

        return {
            "success": True,
            "document": {
                "id": doc.id,
                "name": doc.name,
                "path": str(doc.path),
                "type": doc.type,
                "size": doc.size,
                "status": doc.status.value,
                "chunks": doc.chunks,
                "created_at": doc.created_at.isoformat(),
                "metadata": doc.metadata,
            },
            "chunks": [
                {
                    "id": chunk.id,
                    "content_preview": chunk.content[:200],
                    "metadata": chunk.metadata,
                }
                for chunk in doc_chunks[:10]
            ],
        }

    async def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from the knowledge base"""
        if self.vector_store_type == VectorStoreType.CHROMA and CHROMADB_AVAILABLE:
            self.collection.delete(ids=self.collection.get()["ids"])
        elif self.vector_store_type == VectorStoreType.QDRANT and QDRANT_AVAILABLE:
            self.qdrant_client.delete_collection(collection_name="knowledge_base")
            self.qdrant_client.recreate_collection(
                collection_name="knowledge_base",
                vectors_config=VectorParams(
                    size=self.embedding_dimension, distance=Distance.COSINE
                ),
            )
        else:
            self.in_memory_vectors.clear()

        self.documents.clear()
        self.chunks.clear()

        self.stats["total_documents"] = 0
        self.stats["total_chunks"] = 0

        self._save_metadata()
        self.logger.info("Knowledge base cleared")

        return {"success": True, "message": "Knowledge base cleared successfully"}

    def get_query_history(self, limit: int = None) -> List[Dict]:
        """Get question-answer history"""
        history = self.query_history
        if limit:
            history = history[-limit:]

        return [
            {
                "question": h.question,
                "answer_preview": h.answer[:200],
                "confidence": h.confidence,
                "sources_count": len(h.sources),
                "processing_time": h.processing_time,
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "vector_store_type": self.vector_store_type.value,
            "embedding_model": "GGUF",
            "model_path": str(self.model_path),
            "embedding_dimension": self.embedding_dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "query_history_size": len(self.query_history),
            "use_llm": self.use_llm,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
        }

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid

        return str(uuid.uuid4())


# Integration wrapper for EDIATH
class KnowledgeBaseAgentWrapper:
    """Wrapper class to integrate KnowledgeBaseAgent with EDIATH's agent architecture"""

    def __init__(self, config: Optional[Dict] = None):
        self.kb_agent = KnowledgeBaseAgent(config)
        self.agent_type = "knowledge_base"
        self.capabilities = [
            "add_document",
            "search",
            "ask_question",
            "list_documents",
            "delete_document",
            "clear_knowledge_base",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge base request"""
        operation = request.get("operation")

        if operation == "add":
            return await self.kb_agent.add_document(
                file_path=request.get("file_path"), metadata=request.get("metadata")
            )

        elif operation == "add_batch":
            return await self.kb_agent.add_documents_batch(
                file_paths=request.get("file_paths", [])
            )

        elif operation == "search":
            return await self.kb_agent.search(
                query=request.get("query"),
                top_k=request.get("top_k"),
                metadata_filter=request.get("metadata_filter"),
                min_score=request.get("min_score"),
            )

        elif operation == "ask":
            return await self.kb_agent.ask(
                question=request.get("question"),
                top_k=request.get("top_k"),
                use_context=request.get("use_context", True),
            )

        elif operation == "list":
            return await self.kb_agent.list_documents()

        elif operation == "info":
            return await self.kb_agent.get_document_info(
                document_id=request.get("document_id")
            )

        elif operation == "delete":
            return await self.kb_agent.delete_document(
                document_id=request.get("document_id")
            )

        elif operation == "clear":
            return await self.kb_agent.clear_knowledge_base()

        elif operation == "history":
            return {
                "success": True,
                "history": self.kb_agent.get_query_history(limit=request.get("limit")),
            }

        elif operation == "stats":
            return self.kb_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "KnowledgeBaseAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.kb_agent.get_stats(),
            "vector_store": self.kb_agent.vector_store_type.value,
            "embedding_model": "GGUF",
        }


# Example usage
async def test_knowledge_base_agent():
    """Test the knowledge base agent functionality with GGUF model"""

    config = {
        "model_path": "./models/EDIATH-q4_k_m.gguf",
        "n_ctx": 2048,
        "n_threads": 4,
        "vector_store": "in_memory",
        "chunk_size": 500,
        "chunk_overlap": 50,
    }

    print("=== Knowledge Base Agent Test with GGUF Model ===\n")

    try:
        agent = KnowledgeBaseAgent(config)
        stats = agent.get_stats()
        print(f"Embedding Model: {stats['embedding_model']}")
        print(f"Model Path: {stats['model_path']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        print("✓ GGUF model loaded successfully")
        print("\n=== Test Complete ===")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")


if __name__ == "__main__":
    asyncio.run(test_knowledge_base_agent())
