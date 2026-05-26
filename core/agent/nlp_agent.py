"""
NLP Agent for EDIATH using GGUF Model
Advanced natural language processing: text analysis, sentiment, entities, summarization, translation, embeddings
Uses EDIATH GGUF model for embeddings and language understanding
"""

import asyncio
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
from collections import Counter
from pathlib import Path

# GGUF Model for embeddings and NLP
try:
    from llama_cpp import Llama

    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Core NLP libraries (fallback)
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Transformers as fallback for specific tasks
try:
    from transformers import pipeline, AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Language(Enum):
    """Supported languages"""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"


class SentimentType(Enum):
    """Sentiment classification"""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class EntityType(Enum):
    """Entity types for NER"""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    PERCENT = "percent"
    PRODUCT = "product"
    EVENT = "event"
    LANGUAGE = "language"


@dataclass
class SentimentResult:
    """Sentiment analysis result"""

    text: str
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    sentiment: SentimentType
    confidence: float


@dataclass
class Entity:
    """Named entity"""

    text: str
    type: EntityType
    start: int
    end: int
    confidence: float


@dataclass
class TokenInfo:
    """Token information"""

    text: str
    lemma: str
    pos: str
    tag: str
    is_stopword: bool
    is_punct: bool


@dataclass
class TextSummary:
    """Text summarization result"""

    original_length: int
    summary_length: int
    summary: str
    compression_ratio: float
    method: str


class NLPAgent:
    """
    Advanced natural language processing agent using EDIATH GGUF model
    - Sentiment analysis (via GGUF)
    - Named entity recognition (NER)
    - Text summarization (extractive & abstractive)
    - Keyword extraction
    - Language detection
    - Text classification
    - Tokenization and lemmatization
    - Text similarity
    - Embedding generation (GGUF)
    - Text cleaning and normalization
    - Readability scoring
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize NLP Agent with EDIATH GGUF model

        Args:
            config: Configuration dictionary with:
                - model_path: Path to GGUF model file
                - n_ctx: Context window size (default: 2048)
                - n_threads: Number of threads (default: 4)
                - use_gguf: Use GGUF model (default: True)
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

        # Fallback models (if GGUF unavailable)
        self.use_spacy = self.config.get("use_spacy", True) and SPACY_AVAILABLE
        self.use_transformers = (
            self.config.get("use_transformers", True) and TRANSFORMERS_AVAILABLE
        )
        self.device = self.config.get(
            "device", "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        )

        # SpaCy models (fallback)
        self.nlp_spacy = None
        self.spacy_model_name = self.config.get("spacy_model", "en_core_web_sm")

        # Transformers pipelines (fallback)
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.summarizer_pipeline = None

        # NLTK data
        self.stop_words = set()
        self.lemmatizer = None
        self.stemmer = None

        # Statistics
        self.stats = {
            "total_analyses": 0,
            "average_processing_time": 0.0,
            "by_task": {},
            "embedding_model": "GGUF" if self.use_gguf else "fallback",
        }

        # Initialize models
        self._init_gguf_model()
        self._init_fallback_models()

        self.logger.info(
            f"NLP Agent initialized (GGUF: {self.use_gguf and self.gguf_model is not None})"
        )

    def _init_gguf_model(self):
        """Initialize GGUF model for NLP tasks"""
        if not self.use_gguf:
            self.logger.info("GGUF model disabled, using fallback models")
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
                self.logger.warning(
                    f"GGUF model not found at {model_path}, using fallback"
                )
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
            self.logger.info("GGUF model loaded successfully")

            # Get embedding dimension
            try:
                test_embedding = self.gguf_model.embed("test")
                self.embedding_dimension = len(test_embedding)
                self.logger.info(
                    f"GGUF embedding dimension: {self.embedding_dimension}"
                )
            except:
                self.logger.warning(
                    f"Could not determine embedding dimension, using default: {self.embedding_dimension}"
                )

        except Exception as e:
            self.logger.error(f"Failed to load GGUF model: {str(e)}")
            self.use_gguf = False

    def _init_fallback_models(self):
        """Initialize fallback NLP models (for when GGUF is unavailable)"""
        # Initialize SpaCy (fallback)
        if self.use_spacy:
            try:
                self.nlp_spacy = spacy.load(self.spacy_model_name)
                self.logger.info(f"SpaCy model loaded: {self.spacy_model_name}")
            except OSError:
                self.logger.warning(
                    f"SpaCy model {self.spacy_model_name} not found. Run: python -m spacy download {self.spacy_model_name}"
                )
                self.use_spacy = False

        # Initialize NLTK
        if NLTK_AVAILABLE:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)

            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()

        # Initialize Transformers pipelines (fallback)
        if self.use_transformers and not self.use_gguf:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if self.device == "cuda" else -1,
                )
                self.logger.info("Sentiment analysis pipeline loaded (fallback)")
            except Exception as e:
                self.logger.warning(f"Failed to load sentiment pipeline: {e}")

            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    device=0 if self.device == "cuda" else -1,
                )
                self.logger.info("NER pipeline loaded (fallback)")
            except Exception as e:
                self.logger.warning(f"Failed to load NER pipeline: {e}")

            try:
                self.summarizer_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if self.device == "cuda" else -1,
                )
                self.logger.info("Summarization pipeline loaded (fallback)")
            except Exception as e:
                self.logger.warning(f"Failed to load summarization pipeline: {e}")

    def _update_stats(self, task: str, processing_time: float):
        """Update statistics"""
        self.stats["total_analyses"] += 1
        self.stats["average_processing_time"] = (
            self.stats["average_processing_time"] * (self.stats["total_analyses"] - 1)
            + processing_time
        ) / self.stats["total_analyses"]

        if task not in self.stats["by_task"]:
            self.stats["by_task"][task] = 0
        self.stats["by_task"][task] += 1

    def _get_gguf_embedding(self, text: str) -> List[float]:
        """Get embedding from GGUF model"""
        if not self.use_gguf or not self.gguf_model:
            return None

        try:
            # Truncate long text
            if len(text) > self.n_ctx * 4:
                text = text[: self.n_ctx * 4]

            embedding = self.gguf_model.embed(text)
            return (
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            )
        except Exception as e:
            self.logger.debug(f"GGUF embedding failed: {e}")
            return None

    async def _call_gguf(
        self, prompt: str, max_tokens: int = 200, temperature: float = 0.3
    ) -> str:
        """Call GGUF model for text generation"""
        if not self.use_gguf or not self.gguf_model:
            return None

        try:
            response = await asyncio.to_thread(
                self.gguf_model.create_completion,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["\n\n", "Context:", "Question:", "Answer:"],
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            self.logger.debug(f"GGUF generation failed: {e}")
            return None

    # ============
    # Text Preprocessing
    # ============

    async def clean_text(
        self,
        text: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True,
        remove_stopwords: bool = False,
    ) -> Dict[str, Any]:
        """Clean and normalize text"""
        start_time = datetime.now()

        original_length = len(text)
        cleaned = text

        if lowercase:
            cleaned = cleaned.lower()

        if remove_punctuation:
            cleaned = re.sub(r"[^\w\s]", "", cleaned)

        if remove_numbers:
            cleaned = re.sub(r"\d+", "", cleaned)

        if remove_extra_spaces:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if remove_stopwords and NLTK_AVAILABLE:
            words = cleaned.split()
            words = [w for w in words if w not in self.stop_words]
            cleaned = " ".join(words)

        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_stats("clean_text", processing_time)

        return {
            "success": True,
            "original_text": text[:200],
            "cleaned_text": cleaned[:500],
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "removed_chars": original_length - len(cleaned),
            "processing_time": processing_time,
        }

    # ============
    # Sentiment Analysis (GGUF-enhanced)
    # ============

    async def analyze_sentiment(
        self, text: str, use_transformers: bool = True
    ) -> Dict[str, Any]:
        """Analyze sentiment using GGUF or fallback models"""
        start_time = datetime.now()

        try:
            # Try GGUF first
            if self.use_gguf and self.gguf_model:
                prompt = f"""Analyze the sentiment of the following text. Respond with JSON only.

Text: {text[:500]}

Return JSON format:
{{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "polarity": -1.0-1.0}}

Sentiment:"""

                response = await self._call_gguf(
                    prompt, max_tokens=100, temperature=0.1
                )

                if response:
                    try:
                        # Try to parse JSON response
                        json_match = re.search(r"\{.*\}", response, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                            sentiment_str = result.get("sentiment", "neutral")
                            confidence = float(result.get("confidence", 0.7))
                            polarity = float(result.get("polarity", 0.0))

                            if sentiment_str == "positive":
                                if confidence > 0.9:
                                    sentiment = SentimentType.VERY_POSITIVE
                                else:
                                    sentiment = SentimentType.POSITIVE
                            elif sentiment_str == "negative":
                                if confidence > 0.9:
                                    sentiment = SentimentType.VERY_NEGATIVE
                                else:
                                    sentiment = SentimentType.NEGATIVE
                            else:
                                sentiment = SentimentType.NEUTRAL

                            processing_time = (
                                datetime.now() - start_time
                            ).total_seconds()
                            self._update_stats("sentiment_analysis", processing_time)

                            return {
                                "success": True,
                                "text": text[:200],
                                "polarity": round(polarity, 3),
                                "subjectivity": 0.5,
                                "sentiment": sentiment.value,
                                "confidence": round(confidence, 3),
                                "processing_time": processing_time,
                                "model": "GGUF",
                            }
                    except:
                        pass

            # Fallback to transformers or TextBlob
            if use_transformers and self.sentiment_pipeline:
                result = self.sentiment_pipeline(text[:512])[0]
                label = result["label"]
                confidence = result["score"]

                if label == "POSITIVE":
                    polarity = 0.5 + (confidence * 0.5)
                    if confidence > 0.95:
                        sentiment = SentimentType.VERY_POSITIVE
                    else:
                        sentiment = SentimentType.POSITIVE
                else:
                    polarity = -0.5 - (confidence * 0.5)
                    if confidence > 0.95:
                        sentiment = SentimentType.VERY_NEGATIVE
                    else:
                        sentiment = SentimentType.NEGATIVE

                subjectivity = 0.7

            elif TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                confidence = abs(polarity)

                if polarity >= 0.6:
                    sentiment = SentimentType.VERY_POSITIVE
                elif polarity >= 0.2:
                    sentiment = SentimentType.POSITIVE
                elif polarity >= -0.2:
                    sentiment = SentimentType.NEUTRAL
                elif polarity >= -0.6:
                    sentiment = SentimentType.NEGATIVE
                else:
                    sentiment = SentimentType.VERY_NEGATIVE

            else:
                raise Exception("No sentiment analysis backend available")

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("sentiment_analysis", processing_time)

            return {
                "success": True,
                "text": text[:200],
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "sentiment": sentiment.value,
                "confidence": round(confidence, 3),
                "processing_time": processing_time,
                "model": "fallback",
            }

        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Named Entity Recognition
    # ============

    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities from text"""
        start_time = datetime.now()

        entities = []

        try:
            # Try GGUF for entity extraction
            if self.use_gguf and self.gguf_model:
                prompt = f"""Extract named entities from the following text. Return JSON only.

Text: {text[:500]}

Return JSON format:
{{"entities": [{{"text": "entity", "type": "PERSON/ORG/LOCATION/DATE", "confidence": 0.0-1.0}}]}}

Entities:"""

                response = await self._call_gguf(
                    prompt, max_tokens=200, temperature=0.1
                )

                if response:
                    try:
                        json_match = re.search(r"\{.*\}", response, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                            for ent in result.get("entities", [])[:20]:
                                entity_type_map = {
                                    "PERSON": EntityType.PERSON,
                                    "ORG": EntityType.ORGANIZATION,
                                    "LOCATION": EntityType.LOCATION,
                                    "DATE": EntityType.DATE,
                                }
                                mapped_type = entity_type_map.get(
                                    ent.get("type", "PRODUCT"), EntityType.PRODUCT
                                )

                                entities.append(
                                    Entity(
                                        text=ent.get("text", ""),
                                        type=mapped_type,
                                        start=0,
                                        end=0,
                                        confidence=float(ent.get("confidence", 0.8)),
                                    )
                                )
                    except:
                        pass

            # Fallback to transformers or SpaCy
            if not entities and self.use_transformers and self.ner_pipeline:
                ner_results = self.ner_pipeline(text)

                current_entity = None
                for result in ner_results:
                    entity_text = result["word"]
                    entity_type = result["entity"].replace("B-", "").replace("I-", "")
                    confidence = result["score"]

                    entity_type_map = {
                        "PER": EntityType.PERSON,
                        "ORG": EntityType.ORGANIZATION,
                        "LOC": EntityType.LOCATION,
                        "MISC": EntityType.EVENT,
                    }

                    mapped_type = entity_type_map.get(entity_type, EntityType.PRODUCT)

                    entities.append(
                        Entity(
                            text=entity_text,
                            type=mapped_type,
                            start=result["start"],
                            end=result["end"],
                            confidence=confidence,
                        )
                    )

            elif not entities and self.use_spacy:
                doc = self.nlp_spacy(text)

                for ent in doc.ents:
                    entity_type_map = {
                        "PERSON": EntityType.PERSON,
                        "ORG": EntityType.ORGANIZATION,
                        "GPE": EntityType.LOCATION,
                        "LOC": EntityType.LOCATION,
                        "DATE": EntityType.DATE,
                        "TIME": EntityType.TIME,
                        "MONEY": EntityType.MONEY,
                        "PERCENT": EntityType.PERCENT,
                        "PRODUCT": EntityType.PRODUCT,
                        "EVENT": EntityType.EVENT,
                    }

                    mapped_type = entity_type_map.get(ent.label_, EntityType.PRODUCT)

                    entities.append(
                        Entity(
                            text=ent.text,
                            type=mapped_type,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.9,
                        )
                    )

            # Group and deduplicate
            unique_entities = {}
            for ent in entities:
                key = f"{ent.text}_{ent.type.value}"
                if (
                    key not in unique_entities
                    or ent.confidence > unique_entities[key].confidence
                ):
                    unique_entities[key] = ent

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("ner", processing_time)

            # Group by type
            entities_by_type = {}
            for ent in unique_entities.values():
                if ent.type.value not in entities_by_type:
                    entities_by_type[ent.type.value] = []
                entities_by_type[ent.type.value].append(ent.text)

            return {
                "success": True,
                "text": text[:200],
                "total_entities": len(unique_entities),
                "entities": [
                    {
                        "text": e.text,
                        "type": e.type.value,
                        "confidence": round(e.confidence, 3),
                    }
                    for e in list(unique_entities.values())[:50]
                ],
                "entities_by_type": entities_by_type,
                "processing_time": processing_time,
                "model": "GGUF" if self.use_gguf else "fallback",
            }

        except Exception as e:
            self.logger.error(f"NER error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Text Summarization (GGUF-enhanced)
    # ============

    async def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        method: str = "auto",
    ) -> Dict[str, Any]:
        """Summarize text using GGUF or fallback"""
        start_time = datetime.now()

        original_length = len(text)

        try:
            # Try GGUF for abstractive summarization
            if (
                method in ["auto", "abstractive"]
                and self.use_gguf
                and self.gguf_model
                and len(text) > 200
            ):
                prompt = f"""Summarize the following text concisely:

Text: {text[:1500]}

Summary (max {max_length} words):"""

                summary = await self._call_gguf(
                    prompt, max_tokens=max_length * 2, temperature=0.3
                )

                if summary and len(summary) > 20:
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self._update_stats("summarization", processing_time)

                    return {
                        "success": True,
                        "original_length": original_length,
                        "summary_length": len(summary),
                        "summary": summary[:500],
                        "compression_ratio": round(
                            len(summary) / original_length * 100, 1
                        ),
                        "method": "abstractive (GGUF)",
                        "processing_time": processing_time,
                    }

            # Fallback to transformer summarizer
            if self.summarizer_pipeline and len(text) > 200:
                max_input_length = 1024
                if len(text) > max_input_length:
                    text = text[:max_input_length]

                result = self.summarizer_pipeline(
                    text, max_length=max_length, min_length=min_length, do_sample=False
                )
                summary = result[0]["summary_text"]

            else:
                # Extractive summarization
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split(".")
                words = re.findall(r"\w+", text.lower())
                word_freq = Counter(words)

                sentence_scores = {}
                for sent in sentences:
                    sent_words = re.findall(r"\w+", sent.lower())
                    score = sum(word_freq.get(w, 0) for w in sent_words)
                    sentence_scores[sent] = score / max(len(sent_words), 1)

                sorted_sentences = sorted(
                    sentence_scores.items(), key=lambda x: x[1], reverse=True
                )

                summary_sentences = []
                current_length = 0
                for sent, score in sorted_sentences:
                    sent_len = len(sent)
                    if current_length + sent_len <= max_length * 4:
                        summary_sentences.append(sent)
                        current_length += sent_len
                    if len(summary_sentences) >= max_length / 20:
                        break

                summary_sentences.sort(
                    key=lambda x: sentences.index(x) if x in sentences else 999
                )
                summary = " ".join(summary_sentences)

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("summarization", processing_time)

            return {
                "success": True,
                "original_length": original_length,
                "summary_length": len(summary),
                "summary": summary[:500],
                "compression_ratio": round(len(summary) / original_length * 100, 1),
                "method": method,
                "processing_time": processing_time,
            }

        except Exception as e:
            self.logger.error(f"Summarization error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Keyword Extraction (GGUF-enhanced)
    # ============

    async def extract_keywords(
        self, text: str, top_k: int = 10, method: str = "auto"
    ) -> Dict[str, Any]:
        """Extract keywords using GGUF or frequency-based method"""
        start_time = datetime.now()

        try:
            # Try GGUF for intelligent keyword extraction
            if method in ["auto", "llm"] and self.use_gguf and self.gguf_model:
                prompt = f"""Extract the top {top_k} key phrases from the following text. Return JSON only.

Text: {text[:800]}

Return JSON format:
{{"keywords": [{{"word": "keyword", "relevance": 0.0-1.0}}]}}

Keywords:"""

                response = await self._call_gguf(
                    prompt, max_tokens=200, temperature=0.2
                )

                if response:
                    try:
                        json_match = re.search(r"\{.*\}", response, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                            keywords_data = result.get("keywords", [])[:top_k]

                            processing_time = (
                                datetime.now() - start_time
                            ).total_seconds()
                            self._update_stats("keyword_extraction", processing_time)

                            return {
                                "success": True,
                                "keywords": keywords_data,
                                "total_unique_words": len(keywords_data),
                                "processing_time": processing_time,
                                "method": "GGUF",
                            }
                    except:
                        pass

            # Fallback to frequency-based extraction
            clean_text = re.sub(r"[^\w\s]", "", text.lower())
            words = clean_text.split()

            if NLTK_AVAILABLE:
                words = [w for w in words if w not in self.stop_words and len(w) > 2]

            word_freq = Counter(words)
            keywords = word_freq.most_common(top_k)

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("keyword_extraction", processing_time)

            return {
                "success": True,
                "keywords": [
                    {"word": k, "score": v / max(word_freq.values())}
                    for k, v in keywords
                ],
                "total_unique_words": len(word_freq),
                "processing_time": processing_time,
                "method": "frequency",
            }

        except Exception as e:
            self.logger.error(f"Keyword extraction error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Tokenization and POS Tagging
    # ============

    async def tokenize(self, text: str, include_pos: bool = True) -> Dict[str, Any]:
        """Tokenize text and optionally get POS tags"""
        start_time = datetime.now()

        try:
            tokens = []

            if self.use_spacy:
                doc = self.nlp_spacy(text)

                for token in doc:
                    token_info = TokenInfo(
                        text=token.text,
                        lemma=token.lemma_,
                        pos=token.pos_,
                        tag=token.tag_,
                        is_stopword=token.is_stop,
                        is_punct=token.is_punct,
                    )
                    tokens.append(token_info)

            elif NLTK_AVAILABLE:
                words = word_tokenize(text)
                pos_tags = nltk.pos_tag(words) if include_pos else None

                for i, word in enumerate(words):
                    token_info = TokenInfo(
                        text=word,
                        lemma=(
                            self.lemmatizer.lemmatize(word) if self.lemmatizer else word
                        ),
                        pos=pos_tags[i][1] if pos_tags else "",
                        tag=pos_tags[i][1] if pos_tags else "",
                        is_stopword=word.lower() in self.stop_words,
                        is_punct=not re.match(r"^\w+$", word),
                    )
                    tokens.append(token_info)

            else:
                for word in text.split():
                    token_info = TokenInfo(
                        text=word,
                        lemma=word,
                        pos="",
                        tag="",
                        is_stopword=False,
                        is_punct=not re.match(r"^\w+$", word),
                    )
                    tokens.append(token_info)

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("tokenization", processing_time)

            return {
                "success": True,
                "total_tokens": len(tokens),
                "tokens": [
                    {
                        "text": t.text,
                        "lemma": t.lemma,
                        "pos": t.pos,
                        "is_stopword": t.is_stopword,
                        "is_punct": t.is_punct,
                    }
                    for t in tokens[:100]
                ],
                "processing_time": processing_time,
            }

        except Exception as e:
            self.logger.error(f"Tokenization error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Language Detection
    # ============

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text"""
        start_time = datetime.now()

        try:
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                detected_lang = blob.detect_language()

                lang_map = {
                    "en": Language.ENGLISH,
                    "es": Language.SPANISH,
                    "fr": Language.FRENCH,
                    "de": Language.GERMAN,
                    "it": Language.ITALIAN,
                    "pt": Language.PORTUGUESE,
                    "nl": Language.DUTCH,
                    "ru": Language.RUSSIAN,
                    "zh": Language.CHINESE,
                    "ja": Language.JAPANESE,
                    "ko": Language.KOREAN,
                    "ar": Language.ARABIC,
                }

                language = lang_map.get(detected_lang, Language.ENGLISH)
                confidence = 0.8
            else:
                language = Language.ENGLISH
                confidence = 0.5

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("language_detection", processing_time)

            return {
                "success": True,
                "language": language.value,
                "language_name": language.name,
                "confidence": round(confidence, 3),
                "processing_time": processing_time,
            }

        except Exception as e:
            self.logger.error(f"Language detection error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Text Similarity (GGUF-enhanced)
    # ============

    async def text_similarity(
        self, text1: str, text2: str, method: str = "cosine"
    ) -> Dict[str, Any]:
        """Calculate similarity between two texts using embeddings"""
        start_time = datetime.now()

        try:
            # Try GGUF embeddings for similarity
            if method in ["cosine", "auto"] and self.use_gguf and self.gguf_model:
                embedding1 = self._get_gguf_embedding(text1)
                embedding2 = self._get_gguf_embedding(text2)

                if embedding1 and embedding2:
                    # Cosine similarity
                    import math

                    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                    norm1 = math.sqrt(sum(a * a for a in embedding1))
                    norm2 = math.sqrt(sum(b * b for b in embedding2))

                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)

                        processing_time = (datetime.now() - start_time).total_seconds()
                        self._update_stats("similarity", processing_time)

                        return {
                            "success": True,
                            "similarity": round(similarity, 4),
                            "method": "cosine (GGUF)",
                            "text1_preview": text1[:100],
                            "text2_preview": text2[:100],
                            "processing_time": processing_time,
                        }

            # Fallback to traditional methods
            words1 = set(re.findall(r"\w+", text1.lower()))
            words2 = set(re.findall(r"\w+", text2.lower()))

            if not words1 or not words2:
                similarity = 0.0
            else:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("similarity", processing_time)

            return {
                "success": True,
                "similarity": round(similarity, 4),
                "method": "jaccard",
                "text1_preview": text1[:100],
                "text2_preview": text2[:100],
                "processing_time": processing_time,
            }

        except Exception as e:
            self.logger.error(f"Similarity error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # Readability Analysis
    # ============

    async def readability_score(self, text: str) -> Dict[str, Any]:
        """Calculate readability scores for text"""
        start_time = datetime.now()

        try:
            sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in sentences if s.strip()]
            words = re.findall(r"\w+", text)
            syllables = sum(self._count_syllables(w) for w in words)

            num_sentences = len(sentences)
            num_words = len(words)
            num_syllables = syllables

            if num_sentences == 0 or num_words == 0:
                return {
                    "success": True,
                    "error": "Text too short for readability analysis",
                }

            flesch = (
                206.835
                - 1.015 * (num_words / num_sentences)
                - 84.6 * (num_syllables / num_words)
            )
            flesch = max(0, min(100, flesch))

            grade_level = (
                0.39 * (num_words / num_sentences)
                + 11.8 * (num_syllables / num_words)
                - 15.59
            )
            grade_level = max(0, min(18, grade_level))

            if flesch >= 90:
                reading_level = "Very Easy (5th grade)"
            elif flesch >= 80:
                reading_level = "Easy (6th grade)"
            elif flesch >= 70:
                reading_level = "Fairly Easy (7th grade)"
            elif flesch >= 60:
                reading_level = "Standard (8th-9th grade)"
            elif flesch >= 50:
                reading_level = "Fairly Difficult (10th-12th grade)"
            elif flesch >= 30:
                reading_level = "Difficult (College)"
            else:
                reading_level = "Very Difficult (College Graduate)"

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("readability", processing_time)

            return {
                "success": True,
                "flesch_reading_ease": round(flesch, 1),
                "flesch_kincaid_grade": round(grade_level, 1),
                "reading_level": reading_level,
                "metrics": {
                    "sentences": num_sentences,
                    "words": num_words,
                    "syllables": num_syllables,
                    "avg_words_per_sentence": round(num_words / num_sentences, 1),
                    "avg_syllables_per_word": round(num_syllables / num_words, 2),
                },
                "processing_time": processing_time,
            }

        except Exception as e:
            self.logger.error(f"Readability error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"

        if word and word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count

    # ============
    # Embeddings (GGUF)
    # ============

    async def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generate text embeddings using GGUF model"""
        start_time = datetime.now()

        try:
            if not self.use_gguf or not self.gguf_model:
                return {
                    "success": False,
                    "error": "GGUF model not available for embeddings",
                }

            embedding = self._get_gguf_embedding(text)

            if not embedding:
                return {"success": False, "error": "Failed to generate embedding"}

            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats("embeddings", processing_time)

            return {
                "success": True,
                "embedding_dimension": len(embedding),
                "embedding_preview": embedding[:10],
                "processing_time": processing_time,
                "model": "GGUF",
            }

        except Exception as e:
            self.logger.error(f"Embedding error: {str(e)}")
            return {"success": False, "error": str(e), "text": text[:100]}

    # ============
    # Batch Processing
    # ============

    async def batch_analyze(
        self, texts: List[str], task: str = "sentiment"
    ) -> Dict[str, Any]:
        """Analyze multiple texts in batch"""
        start_time = datetime.now()

        results = []

        for text in texts:
            if task == "sentiment":
                result = await self.analyze_sentiment(text)
            elif task == "entities":
                result = await self.extract_entities(text)
            elif task == "keywords":
                result = await self.extract_keywords(text)
            elif task == "clean":
                result = await self.clean_text(text)
            elif task == "embeddings":
                result = await self.generate_embeddings(text)
            else:
                result = {"success": False, "error": f"Unknown task: {task}"}

            results.append(result)

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "total_texts": len(texts),
            "successful": sum(1 for r in results if r.get("success", False)),
            "results": results,
            "processing_time": processing_time,
        }

    # ============
    # Utility Methods
    # ============

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "gguf_available": self.use_gguf and self.gguf_model is not None,
            "spacy_available": SPACY_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "device": self.device,
            "model_path": str(self.model_path) if self.use_gguf else None,
            "n_ctx": self.n_ctx,
        }


# Integration wrapper for EDIATH
class NLPAgentWrapper:
    """
    Wrapper class to integrate NLPAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.nlp_agent = NLPAgent(config)
        self.agent_type = "nlp"
        self.capabilities = [
            "sentiment_analysis",
            "entity_recognition",
            "text_summarization",
            "keyword_extraction",
            "text_similarity",
            "readability_analysis",
            "language_detection",
            "text_cleaning",
            "embeddings",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an NLP request"""
        operation = request.get("operation")

        if operation == "sentiment":
            return await self.nlp_agent.analyze_sentiment(
                text=request.get("text"),
                use_transformers=request.get("use_transformers", True),
            )

        elif operation == "entities":
            return await self.nlp_agent.extract_entities(text=request.get("text"))

        elif operation == "summarize":
            return await self.nlp_agent.summarize(
                text=request.get("text"),
                max_length=request.get("max_length", 150),
                min_length=request.get("min_length", 30),
                method=request.get("method", "auto"),
            )

        elif operation == "keywords":
            return await self.nlp_agent.extract_keywords(
                text=request.get("text"),
                top_k=request.get("top_k", 10),
                method=request.get("method", "auto"),
            )

        elif operation == "similarity":
            return await self.nlp_agent.text_similarity(
                text1=request.get("text1"),
                text2=request.get("text2"),
                method=request.get("method", "cosine"),
            )

        elif operation == "readability":
            return await self.nlp_agent.readability_score(text=request.get("text"))

        elif operation == "language":
            return await self.nlp_agent.detect_language(text=request.get("text"))

        elif operation == "clean":
            return await self.nlp_agent.clean_text(
                text=request.get("text"),
                lowercase=request.get("lowercase", True),
                remove_punctuation=request.get("remove_punctuation", True),
                remove_numbers=request.get("remove_numbers", False),
                remove_stopwords=request.get("remove_stopwords", False),
            )

        elif operation == "tokenize":
            return await self.nlp_agent.tokenize(
                text=request.get("text"), include_pos=request.get("include_pos", True)
            )

        elif operation == "embeddings":
            return await self.nlp_agent.generate_embeddings(text=request.get("text"))

        elif operation == "batch":
            return await self.nlp_agent.batch_analyze(
                texts=request.get("texts", []), task=request.get("task", "sentiment")
            )

        elif operation == "stats":
            return self.nlp_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "NLPAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.nlp_agent.get_stats(),
            "languages": [l.value for l in Language],
            "sentiment_types": [s.value for s in SentimentType],
        }


# Example usage
async def test_nlp_agent():
    """Test the NLP agent functionality with GGUF"""

    config = {
        "model_path": "./models/EDIATH-q4_k_m.gguf",
        "use_gguf": True,
        "n_ctx": 2048,
        "n_threads": 4,
    }

    agent = NLPAgent(config)

    import logging

    logger = logging.getLogger(__name__)

    logger.info("=== NLP Agent Test with GGUF Model ===\n")
    stats = agent.get_stats()
    logger.info("GGUF Available: %s", stats.get("gguf_available"))
    logger.info("Model Path: %s", stats.get("model_path", "N/A"))

    test_text = """
    EDIATH is an amazing AI assistant that helps users with various tasks. 
    It can execute code, manage files, search the web, and much more. 
    The system uses advanced machine learning models to understand natural language.
    """

    # Test sentiment analysis
    logger.info("\n1. Sentiment Analysis...")
    result = await agent.analyze_sentiment(test_text)
    if result["success"]:
        logger.info("   Model: %s", result.get("model", "unknown"))
        logger.info("   Sentiment: %s", result["sentiment"])

    # Test embeddings
    logger.info("\n2. Embeddings Generation...")
    result = await agent.generate_embeddings(test_text)
    if result["success"]:
        logger.info("   Model: %s", result.get("model", "unknown"))
        logger.info("   Dimension: %s", result["embedding_dimension"])

    # Rest of test continues...


if __name__ == "__main__":
    from pathlib import Path

    asyncio.run(test_nlp_agent())
