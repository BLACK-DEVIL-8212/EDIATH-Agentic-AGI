"""LLM API - RESTful API for LLM operations with MongoDB integration."""

from typing import Optional, Dict, Any, List, AsyncIterator, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import asyncio
import uuid
import time
import hashlib
import json
from enum import Enum
from pathlib import Path

from ..core.brain import LLMEngine
from ..core.utils.logger import logger
from ..core.memory.mongo_client import mongo_client, get_metrics_collection

# =========================
# ENUMS AND CONSTANTS
# =========================


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    OLLAMA = "ollama"
    GROQ = "groq"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"


class FinishReason(str, Enum):
    """Reason for generation completion."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Role(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


# =========================
# REQUEST/RESPONSE MODELS
# =========================


class Message(BaseModel):
    """Chat message."""

    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    class Config:
        use_enum_values = True


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(
        ..., description="Input prompt", min_length=1, max_length=100000
    )
    max_tokens: int = Field(
        2000, description="Maximum tokens to generate", ge=1, le=32000
    )
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    frequency_penalty: float = Field(
        0.0, description="Frequency penalty", ge=-2.0, le=2.0
    )
    presence_penalty: float = Field(
        0.0, description="Presence penalty", ge=-2.0, le=2.0
    )
    system_prompt: Optional[str] = Field(
        None, description="System prompt", max_length=10000
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    stream: bool = Field(False, description="Enable streaming response")
    cache_enabled: bool = Field(True, description="Enable response caching")
    model: Optional[str] = Field(None, description="Model to use (overrides default)")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    timeout_seconds: int = Field(
        60, description="Request timeout in seconds", ge=5, le=300
    )

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        if len(v) > 100000:
            raise ValueError("Prompt too long (max 100000 characters)")
        return v.strip()

    @validator("temperature")
    def validate_temperature(cls, v):
        if v == 0:
            return 0.01  # Avoid zero temperature issues
        return v

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v < 1:
            return 1
        return v


class ChatRequest(BaseModel):
    """Request model for chat completion."""

    messages: List[Message] = Field(..., description="Chat messages", min_items=1)
    max_tokens: int = Field(2000, ge=1, le=32000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    stream: bool = False
    cache_enabled: bool = True
    model: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        return v


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    response_id: str = Field(..., description="Unique response identifier")
    response: str = Field(..., description="Generated text")
    tokens_used: int = Field(..., description="Number of tokens used")
    prompt_tokens: int = Field(0, description="Number of tokens in prompt")
    completion_tokens: int = Field(0, description="Number of tokens in completion")
    generation_time: float = Field(..., description="Generation time in seconds")
    timestamp: str = Field(..., description="Response timestamp")
    cached: bool = Field(False, description="Whether response came from cache")
    model: str = Field("default", description="Model used for generation")
    finish_reason: Optional[str] = Field(None, description="Reason generation finished")
    safety_ratings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Safety ratings"
    )
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")
    cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")


class ChatResponse(BaseModel):
    """Response model for chat completion."""

    response_id: str
    message: Message
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    generation_time: float
    timestamp: str
    cached: bool
    model: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cost_usd: Optional[float] = None


class BatchGenerateRequest(BaseModel):
    """Request model for batch generation."""

    requests: List[GenerateRequest] = Field(
        ..., description="List of generation requests", max_items=100
    )
    parallel: bool = Field(True, description="Process requests in parallel")
    max_concurrent: int = Field(10, description="Max concurrent requests", ge=1, le=50)


class BatchGenerateResponse(BaseModel):
    """Response model for batch generation."""

    responses: List[GenerateResponse]
    total_time: float
    successful: int
    failed: int
    total_tokens: int
    total_cost: float
    batch_id: str


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    provider: ModelProvider
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool = False
    supports_tools: bool = False
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    context_length: int = 4096
    description: Optional[str] = None
    available: bool = True


class TokenCountRequest(BaseModel):
    """Request for token counting."""

    text: str
    model: Optional[str] = None


class TokenCountResponse(BaseModel):
    """Response for token counting."""

    tokens: int
    characters: int
    words: int
    model: str
    encoding: str


class ConversationTurn(BaseModel):
    """Single turn in a conversation."""

    turn_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    tokens_used: int
    model_used: str


class Conversation(BaseModel):
    """Full conversation history."""

    conversation_id: str
    turns: List[ConversationTurn]
    created_at: datetime
    updated_at: datetime
    total_tokens: int
    user_id: Optional[str] = None


# =========================
# RATE LIMITER
# =========================


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: List[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a rate limit token."""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [t for t in self.requests if now - t < 60]

            if len(self.requests) >= self.requests_per_minute:
                # Wait until oldest request expires
                wait_time = 60 - (now - self.requests[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                return await self.acquire()

            self.requests.append(now)
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        active_requests = [t for t in self.requests if now - t < 60]
        return {
            "active_requests_last_minute": len(active_requests),
            "limit_per_minute": self.requests_per_minute,
            "utilization": (
                len(active_requests) / self.requests_per_minute
                if self.requests_per_minute > 0
                else 0
            ),
        }


# =========================
# TOKEN COUNTER
# =========================


class TokenCounter:
    """Token counter for various models."""

    # Rough estimation (will be replaced with proper tokenizer)
    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-4") -> int:
        """Estimate token count for a text."""
        if not text:
            return 0

        # Simple heuristic
        words = len(text.split())
        chars = len(text)

        # Average token length ~4 characters for English
        estimated = chars // 4

        # Adjust for model
        model_lower = model.lower()
        if "gpt-4" in model_lower or "claude" in model_lower:
            return max(estimated, int(words / 0.75))
        elif "gpt-3.5" in model_lower:
            return max(estimated, int(words / 0.75))
        elif "llama" in model_lower:
            return max(estimated, int(words / 0.8))

        return estimated

    @staticmethod
    def count_messages_tokens(messages: List[Message], model: str = "gpt-4") -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            total += TokenCounter.estimate_tokens(msg.content, model)
            total += 4  # Overhead per message
        total += 2  # Base overhead
        return total

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for token usage."""
        # Pricing per 1K tokens (approximate)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "llama-3-70b": {"input": 0.0009, "output": 0.0009},
            "llama-3-8b": {"input": 0.0002, "output": 0.0002},
            "default": {"input": 0.001, "output": 0.002},
        }

        model_lower = model.lower()
        rate = pricing.get("default")

        for key in pricing:
            if key in model_lower:
                rate = pricing[key]
                break

        input_cost = (input_tokens / 1000) * rate["input"]
        output_cost = (output_tokens / 1000) * rate["output"]

        return input_cost + output_cost


# =========================
# MAIN API CLASS
# =========================


class LLMApi:
    """Production-ready API for LLM operations with MongoDB integration."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize LLM API.

        Args:
            data_dir: Directory for storing LLM data
        """
        self.data_dir = data_dir or Path("llm_data")
        self.data_dir.mkdir(exist_ok=True)

        self.llm_engine = LLMEngine()
        self.request_count = 0
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        self._max_cache_size = 1000
        self._rate_limiter = RateLimiter()
        self._token_counter = TokenCounter()

        # Request tracking
        self._active_requests: Dict[str, datetime] = {}
        self._request_history: List[Dict[str, Any]] = []
        self._max_history_size = 10000

        # Conversation storage
        self._conversations: Dict[str, Conversation] = {}

        # Metrics collection
        self.metrics_collection = get_metrics_collection()

        # Cache file
        self._cache_file = self.data_dir / "cache.json"

        # Load configuration
        self._load_configuration()

        # Load persisted cache
        self._load_cache()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None

        logger.info("✅ LLM API initialized with MongoDB integration")

    def _load_configuration(self):
        """Load configuration from environment or database."""
        # Default configuration
        self.config = {
            "default_model": "gpt-4",
            "default_provider": "openai",
            "max_concurrent_requests": 10,
            "rate_limit_per_minute": 60,
            "enable_cache": True,
            "cache_ttl": 3600,
            "enable_metrics": True,
            "enable_audit_log": True,
            "enable_rate_limiting": True,
            "max_prompt_length": 100000,
            "default_temperature": 0.7,
            "default_max_tokens": 2000,
            "timeout_seconds": 60,
            "retry_attempts": 3,
            "retry_delay_seconds": 1,
            "max_cache_size": 1000,
            "enable_cost_tracking": True,
            "enable_conversation_history": True,
        }

        # Try to load from MongoDB
        if mongo_client.enabled:
            try:
                config_collection = mongo_client.get_collection("config")
                if config_collection is not None:
                    db_config = config_collection.find_one({"type": "llm_api"})
                    if db_config:
                        self.config.update(db_config.get("config", {}))
                        logger.info("Loaded LLM API configuration from database")
            except Exception as e:
                logger.warning(f"Failed to load config from MongoDB: {e}")

        # Update rate limiter
        if self.config["enable_rate_limiting"]:
            self._rate_limiter = RateLimiter(self.config["rate_limit_per_minute"])

        # Update cache settings
        self._cache_ttl = self.config["cache_ttl"]
        self._max_cache_size = self.config["max_cache_size"]

    def _load_cache(self):
        """Load cache from disk."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, "r") as f:
                    cache_data = json.load(f)
                    # Only load non-expired entries
                    now = datetime.now()
                    for key, value in cache_data.items():
                        cache_time = datetime.fromisoformat(
                            value.get("timestamp", "2000-01-01")
                        )
                        if (now - cache_time).seconds < self._cache_ttl:
                            self._response_cache[key] = value
                    logger.info(
                        f"Loaded {len(self._response_cache)} cached responses from disk"
                    )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._response_cache, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _get_cache_key(self, request: Union[GenerateRequest, ChatRequest]) -> str:
        """Generate cache key for request."""
        if isinstance(request, ChatRequest):
            cache_data = {
                "messages": [msg.dict() for msg in request.messages],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "model": request.model,
            }
        else:
            cache_data = {
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "model": request.model,
            }
        return hashlib.sha256(
            json.dumps(cache_data, sort_keys=True).encode()
        ).hexdigest()

    def _check_cache(
        self, cache_key: str
    ) -> Optional[Union[GenerateResponse, ChatResponse]]:
        """Check cache for existing response."""
        if not self.config["enable_cache"]:
            return None

        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            cache_time = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))

            # Check if cache is still valid
            if (datetime.now() - cache_time).seconds < self._cache_ttl:
                logger.info(f"Cache hit for key: {cache_key[:8]}")
                response_data = cached.get("response")

                if response_data:
                    if "message" in response_data:
                        return ChatResponse(**response_data)
                    else:
                        return GenerateResponse(**response_data)
            else:
                # Remove expired cache entry
                del self._response_cache[cache_key]

        return None

    def _save_to_cache(
        self, cache_key: str, response: Union[GenerateResponse, ChatResponse]
    ):
        """Save response to cache."""
        if not self.config["enable_cache"]:
            return

        # Enforce cache size limit
        if len(self._response_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self._response_cache.keys(),
                key=lambda k: self._response_cache[k].get("timestamp", "2000-01-01"),
            )
            del self._response_cache[oldest_key]

        self._response_cache[cache_key] = {
            "response": response.dict(),
            "timestamp": datetime.now().isoformat(),
        }

        # Periodically save cache
        if len(self._response_cache) % 100 == 0:
            self._save_cache()

    async def _log_metrics(
        self,
        request: Union[GenerateRequest, ChatRequest],
        response: Union[GenerateResponse, ChatResponse],
        start_time: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log metrics to MongoDB."""
        if not self.config["enable_metrics"]:
            return

        if not mongo_client.enabled or self.metrics_collection is None:
            return

        try:
            metric = {
                "timestamp": datetime.now(),
                "operation": "chat" if isinstance(request, ChatRequest) else "generate",
                "request_id": response.response_id,
                "user_id": getattr(request, "user_id", None),
                "session_id": getattr(request, "session_id", None),
                "prompt_length": (
                    len(getattr(request, "prompt", ""))
                    if hasattr(request, "prompt")
                    else sum(len(m.content) for m in request.messages)
                ),
                "response_length": len(
                    response.response
                    if hasattr(response, "response")
                    else response.message.content
                ),
                "tokens_used": response.tokens_used,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "generation_time": response.generation_time,
                "total_time_ms": (time.time() - start_time) * 1000,
                "temperature": getattr(request, "temperature", 0.7),
                "top_p": getattr(request, "top_p", 0.9),
                "max_tokens": getattr(request, "max_tokens", 2000),
                "cached": response.cached,
                "success": success,
                "error": error,
                "model": response.model,
                "finish_reason": getattr(response, "finish_reason", None),
                "cost_usd": getattr(response, "cost_usd", None),
            }

            # Add metadata if present
            if hasattr(request, "metadata") and request.metadata:
                metric["metadata"] = request.metadata

            # Insert metric asynchronously (fire and forget)
            from core.utils.task_registry import create_task

            create_task(
                self.metrics_collection.insert_one(metric), name="llmapi.log_metric"
            )

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    async def _log_audit(
        self,
        request: Union[GenerateRequest, ChatRequest],
        response: Union[GenerateResponse, ChatResponse],
        success: bool,
    ):
        """Log audit trail to MongoDB."""
        if not self.config["enable_audit_log"]:
            return

        audit_collection = mongo_client.get_collection("audit_log")
        if not mongo_client.enabled or audit_collection is None:
            return

        try:
            audit_entry = {
                "timestamp": datetime.now(),
                "action": "chat" if isinstance(request, ChatRequest) else "generate",
                "request_id": response.response_id,
                "user_id": getattr(request, "user_id", None),
                "session_id": getattr(request, "session_id", None),
                "success": success,
                "tokens_used": response.tokens_used if success else 0,
                "model": response.model,
                "metadata": getattr(request, "metadata", {}),
            }

            # Add prompt/response previews (truncated for privacy)
            if isinstance(request, ChatRequest):
                audit_entry["messages_preview"] = [
                    {"role": m.role.value, "content": m.content[:200]}
                    for m in request.messages[:5]
                ]
            else:
                audit_entry["prompt_preview"] = request.prompt[:500]

            if success:
                if hasattr(response, "response"):
                    audit_entry["response_preview"] = response.response[:500]
                else:
                    audit_entry["response_preview"] = response.message.content[:500]

            # Insert audit log asynchronously
            from core.utils.task_registry import create_task

            create_task(
                audit_collection.insert_one(audit_entry), name="llmapi.log_audit"
            )

        except Exception as e:
            logger.error(f"Failed to log audit: {e}")

    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_error = None

        for attempt in range(self.config["retry_attempts"]):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config["retry_attempts"] - 1:
                    wait_time = self.config["retry_delay_seconds"] * (2**attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.config['retry_attempts']} after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

        raise last_error

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> Optional[float]:
        """Calculate cost for token usage."""
        if not self.config["enable_cost_tracking"]:
            return None
        return self._token_counter.calculate_cost(model, input_tokens, output_tokens)

    def _trim_history(self):
        """Trim request history to max size."""
        if len(self._request_history) > self._max_history_size:
            self._request_history = self._request_history[-self._max_history_size :]

    # =========================
    # TEXT GENERATION
    # =========================

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text from prompt with caching, rate limiting, and metrics."""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            self.request_count += 1

            # Apply rate limiting
            if self.config["enable_rate_limiting"]:
                await self._rate_limiter.acquire()

            # Validate request
            if not request.prompt or not request.prompt.strip():
                raise ValueError("Prompt cannot be empty")

            if len(request.prompt) > self.config["max_prompt_length"]:
                raise ValueError(
                    f"Prompt too long (max {self.config['max_prompt_length']} characters)"
                )

            # Check cache
            cache_key = self._get_cache_key(request)
            cached_response = (
                self._check_cache(cache_key) if request.cache_enabled else None
            )

            if cached_response and isinstance(cached_response, GenerateResponse):
                cached_response.response_id = request_id
                cached_response.cached = True

                # Log metrics for cached response
                await self._log_metrics(request, cached_response, start_time, True)
                await self._log_audit(request, cached_response, True)

                logger.info(f"LLM generation (cached) completed: {self.request_count}")
                return cached_response

            # Track active request
            self._active_requests[request_id] = datetime.now()

            # Generate new response with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_with_retry(
                        self.llm_engine.generate,
                        prompt=request.prompt,
                        system_prompt=request.system_prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        model=request.model,
                        stop_sequences=request.stop_sequences,
                    ),
                    timeout=request.timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Generation timed out after {request.timeout_seconds} seconds"
                )

            # Safely extract response data
            response_text = result.get("response", "")
            tokens_used = result.get(
                "tokens_used",
                self._token_counter.estimate_tokens(response_text, request.model),
            )
            generation_time = result.get("generation_time", 0.0)
            timestamp = result.get("timestamp", datetime.now().isoformat())

            # Calculate token breakdown
            prompt_tokens = result.get(
                "prompt_tokens",
                self._token_counter.estimate_tokens(request.prompt, request.model),
            )
            if request.system_prompt:
                prompt_tokens += self._token_counter.estimate_tokens(
                    request.system_prompt, request.model
                )
            completion_tokens = (
                tokens_used - prompt_tokens
                if tokens_used > prompt_tokens
                else tokens_used
            )

            # Calculate cost
            model_used = result.get(
                "model", request.model or self.config["default_model"]
            )
            cost = self._calculate_cost(model_used, prompt_tokens, completion_tokens)

            response = GenerateResponse(
                response_id=request_id,
                response=response_text,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_time=generation_time,
                timestamp=timestamp,
                cached=False,
                model=model_used,
                finish_reason=result.get("finish_reason", FinishReason.STOP.value),
                safety_ratings=result.get("safety_ratings"),
                logprobs=result.get("logprobs"),
                cost_usd=cost,
            )

            # Save to cache
            if request.cache_enabled:
                self._save_to_cache(cache_key, response)

            # Log metrics and audit
            await self._log_metrics(request, response, start_time, True)
            await self._log_audit(request, response, True)

            # Update request history
            self._request_history.append(
                {
                    "request_id": request_id,
                    "timestamp": datetime.now(),
                    "tokens": tokens_used,
                    "cost": cost,
                    "success": True,
                }
            )
            self._trim_history()

            logger.info(
                f"LLM generation completed: {self.request_count} (tokens: {tokens_used}, cost: ${cost:.6f})"
            )
            return response

        except Exception as e:
            logger.error(f"LLM API error: {e}")

            # Create error response
            error_response = GenerateResponse(
                response_id=request_id,
                response=f"[ERROR] LLM generation failed: {str(e)}",
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                generation_time=0.0,
                timestamp=datetime.now().isoformat(),
                cached=False,
                finish_reason=FinishReason.ERROR.value,
            )

            # Log error metrics
            await self._log_metrics(request, error_response, start_time, False, str(e))

            # Update request history
            self._request_history.append(
                {
                    "request_id": request_id,
                    "timestamp": datetime.now(),
                    "tokens": 0,
                    "success": False,
                    "error": str(e),
                }
            )
            self._trim_history()

            return error_response
        finally:
            # Remove from active requests
            self._active_requests.pop(request_id, None)

    # =========================
    # CHAT COMPLETION
    # =========================

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion."""
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            self.request_count += 1

            # Apply rate limiting
            if self.config["enable_rate_limiting"]:
                await self._rate_limiter.acquire()

            # Check cache
            cache_key = self._get_cache_key(request)
            cached_response = (
                self._check_cache(cache_key) if request.cache_enabled else None
            )

            if cached_response and isinstance(cached_response, ChatResponse):
                cached_response.response_id = request_id
                cached_response.cached = True

                await self._log_metrics(request, cached_response, start_time, True)
                await self._log_audit(request, cached_response, True)

                logger.info("Chat completion (cached) completed")
                return cached_response

            # Track active request
            self._active_requests[request_id] = datetime.now()

            # Prepare messages
            messages_dict = [msg.dict(exclude_none=True) for msg in request.messages]

            # Generate response with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_with_retry(
                        self.llm_engine.chat,
                        messages=messages_dict,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        model=request.model,
                        stop_sequences=request.stop_sequences,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                    ),
                    timeout=self.config["timeout_seconds"],
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Chat completion timed out after {self.config['timeout_seconds']} seconds"
                )

            # Extract response
            response_content = result.get("response", "")
            response_message = Message(
                role=Role.ASSISTANT,
                content=response_content,
                tool_calls=result.get("tool_calls"),
            )

            tokens_used = result.get(
                "tokens_used",
                self._token_counter.estimate_tokens(response_content, request.model),
            )
            prompt_tokens = result.get(
                "prompt_tokens",
                self._token_counter.count_messages_tokens(
                    request.messages, request.model
                ),
            )
            completion_tokens = (
                tokens_used - prompt_tokens
                if tokens_used > prompt_tokens
                else tokens_used
            )

            model_used = result.get(
                "model", request.model or self.config["default_model"]
            )
            cost = self._calculate_cost(model_used, prompt_tokens, completion_tokens)

            response = ChatResponse(
                response_id=request_id,
                message=response_message,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_time=result.get("generation_time", 0.0),
                timestamp=result.get("timestamp", datetime.now().isoformat()),
                cached=False,
                model=model_used,
                finish_reason=result.get("finish_reason", FinishReason.STOP.value),
                tool_calls=result.get("tool_calls"),
                cost_usd=cost,
            )

            # Save to cache
            if request.cache_enabled:
                self._save_to_cache(cache_key, response)

            # Save conversation if enabled
            if self.config["enable_conversation_history"] and request.session_id:
                await self._save_conversation_turn(request, response)

            # Log metrics
            await self._log_metrics(request, response, start_time, True)
            await self._log_audit(request, response, True)

            logger.info(
                f"Chat completion completed: {tokens_used} tokens, cost: ${cost:.6f}"
            )
            return response

        except Exception as e:
            logger.error(f"Chat API error: {e}")

            error_response = ChatResponse(
                response_id=request_id,
                message=Message(role=Role.ASSISTANT, content=f"[ERROR] {str(e)}"),
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                generation_time=0.0,
                timestamp=datetime.now().isoformat(),
                cached=False,
                finish_reason=FinishReason.ERROR.value,
            )

            await self._log_metrics(request, error_response, start_time, False, str(e))

            return error_response
        finally:
            self._active_requests.pop(request_id, None)

    async def _save_conversation_turn(
        self, request: ChatRequest, response: ChatResponse
    ):
        """Save a conversation turn to history."""
        try:
            # Get last user message
            user_messages = [m for m in request.messages if m.role == Role.USER]
            if not user_messages:
                return

            last_user_msg = user_messages[-1]

            turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                user_message=last_user_msg.content,
                assistant_response=response.message.content,
                timestamp=datetime.now(),
                tokens_used=response.tokens_used,
                model_used=response.model,
            )

            conv_id = request.session_id or f"conv_{request.user_id or 'anonymous'}"

            if conv_id in self._conversations:
                self._conversations[conv_id].turns.append(turn)
                self._conversations[conv_id].updated_at = datetime.now()
                self._conversations[conv_id].total_tokens += response.tokens_used
            else:
                self._conversations[conv_id] = Conversation(
                    conversation_id=conv_id,
                    turns=[turn],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    total_tokens=response.tokens_used,
                    user_id=request.user_id,
                )

            # Limit conversation turns
            if len(self._conversations[conv_id].turns) > 100:
                self._conversations[conv_id].turns = self._conversations[conv_id].turns[
                    -100:
                ]

        except Exception as e:
            logger.error(f"Failed to save conversation turn: {e}")

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation history."""
        return self._conversations.get(conversation_id)

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    # =========================
    # STREAMING
    # =========================

    async def stream_generate(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream text generation."""
        start_time = time.time()

        try:
            # Apply rate limiting
            if self.config["enable_rate_limiting"]:
                await self._rate_limiter.acquire()

            # Check cache first
            if request.cache_enabled:
                cache_key = self._get_cache_key(request)
                cached_response = self._check_cache(cache_key)
                if cached_response and isinstance(cached_response, GenerateResponse):
                    # Stream cached response
                    yield cached_response.response
                    return

            # Stream from engine
            async for chunk in self.llm_engine.generate_stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                model=request.model,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"[ERROR] {str(e)}"

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Stream chat completion."""
        try:
            # Apply rate limiting
            if self.config["enable_rate_limiting"]:
                await self._rate_limiter.acquire()

            # Check cache
            if request.cache_enabled:
                cache_key = self._get_cache_key(request)
                cached_response = self._check_cache(cache_key)
                if cached_response and isinstance(cached_response, ChatResponse):
                    yield cached_response.message.content
                    return

            # Stream from engine
            messages_dict = [msg.dict(exclude_none=True) for msg in request.messages]

            async for chunk in self.llm_engine.chat_stream(
                messages=messages_dict,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                model=request.model,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Chat streaming error: {e}")
            yield f"[ERROR] {str(e)}"

    # =========================
    # BATCH GENERATION
    # =========================

    async def batch_generate(
        self, request: BatchGenerateRequest
    ) -> BatchGenerateResponse:
        """Generate multiple responses in batch."""
        start_time = time.time()
        batch_id = str(uuid.uuid4())

        logger.info(
            f"Processing batch {batch_id} with {len(request.requests)} requests"
        )

        if request.parallel:
            # Process with concurrency limit
            semaphore = asyncio.Semaphore(request.max_concurrent)

            async def process_with_limit(req):
                async with semaphore:
                    return await self.generate(req)

            tasks = [process_with_limit(req) for req in request.requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            processed_responses = []
            successful = 0
            failed = 0
            total_tokens = 0
            total_cost = 0.0

            for resp in responses:
                if isinstance(resp, Exception):
                    failed += 1
                    error_response = GenerateResponse(
                        response_id=str(uuid.uuid4()),
                        response=f"[ERROR] {str(resp)}",
                        tokens_used=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        generation_time=0.0,
                        timestamp=datetime.now().isoformat(),
                        finish_reason=FinishReason.ERROR.value,
                    )
                    processed_responses.append(error_response)
                else:
                    successful += 1
                    total_tokens += resp.tokens_used
                    total_cost += resp.cost_usd or 0
                    processed_responses.append(resp)
        else:
            # Process sequentially
            processed_responses = []
            successful = 0
            failed = 0
            total_tokens = 0
            total_cost = 0.0

            for req in request.requests:
                resp = await self.generate(req)
                processed_responses.append(resp)
                if not resp.response.startswith("[ERROR]"):
                    successful += 1
                    total_tokens += resp.tokens_used
                    total_cost += resp.cost_usd or 0
                else:
                    failed += 1

        total_time = time.time() - start_time

        logger.info(
            f"Batch {batch_id} completed: {successful} successful, {failed} failed, {total_tokens} tokens, ${total_cost:.6f} in {total_time:.2f}s"
        )

        return BatchGenerateResponse(
            responses=processed_responses,
            total_time=total_time,
            successful=successful,
            failed=failed,
            total_tokens=total_tokens,
            total_cost=total_cost,
            batch_id=batch_id,
        )

    # =========================
    # TOKEN COUNTING
    # =========================

    async def count_tokens(self, request: TokenCountRequest) -> TokenCountResponse:
        """Count tokens in text."""
        tokens = self._token_counter.estimate_tokens(request.text, request.model)
        chars = len(request.text)
        words = len(request.text.split())

        return TokenCountResponse(
            tokens=tokens,
            characters=chars,
            words=words,
            model=request.model or self.config["default_model"],
            encoding="cl100k_base",  # Default encoding
        )

    # =========================
    # MODEL MANAGEMENT
    # =========================

    async def get_models(self) -> List[ModelInfo]:
        """Get available models."""
        try:
            if hasattr(self.llm_engine, "get_available_models"):
                models = await self.llm_engine.get_available_models()
                return [ModelInfo(**model) for model in models]
            else:
                # Return comprehensive default models
                return [
                    ModelInfo(
                        name="gpt-4",
                        provider=ModelProvider.OPENAI,
                        max_tokens=8192,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_tools=True,
                        context_length=8192,
                        cost_per_1k_input_tokens=0.03,
                        cost_per_1k_output_tokens=0.06,
                        description="Most capable GPT-4 model",
                    ),
                    ModelInfo(
                        name="gpt-4-turbo",
                        provider=ModelProvider.OPENAI,
                        max_tokens=128000,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_vision=True,
                        supports_tools=True,
                        context_length=128000,
                        cost_per_1k_input_tokens=0.01,
                        cost_per_1k_output_tokens=0.03,
                        description="GPT-4 Turbo with vision",
                    ),
                    ModelInfo(
                        name="gpt-3.5-turbo",
                        provider=ModelProvider.OPENAI,
                        max_tokens=16384,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_tools=True,
                        context_length=16384,
                        cost_per_1k_input_tokens=0.0005,
                        cost_per_1k_output_tokens=0.0015,
                        description="Fast and cost-effective GPT-3.5 Turbo",
                    ),
                    ModelInfo(
                        name="claude-3-opus",
                        provider=ModelProvider.ANTHROPIC,
                        max_tokens=200000,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_vision=True,
                        supports_tools=True,
                        context_length=200000,
                        cost_per_1k_input_tokens=0.015,
                        cost_per_1k_output_tokens=0.075,
                        description="Most powerful Claude model",
                    ),
                    ModelInfo(
                        name="claude-3-sonnet",
                        provider=ModelProvider.ANTHROPIC,
                        max_tokens=200000,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_vision=True,
                        supports_tools=True,
                        context_length=200000,
                        cost_per_1k_input_tokens=0.003,
                        cost_per_1k_output_tokens=0.015,
                        description="Balanced Claude model",
                    ),
                    ModelInfo(
                        name="claude-3-haiku",
                        provider=ModelProvider.ANTHROPIC,
                        max_tokens=200000,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_vision=True,
                        supports_tools=True,
                        context_length=200000,
                        cost_per_1k_input_tokens=0.00025,
                        cost_per_1k_output_tokens=0.00125,
                        description="Fastest Claude model",
                    ),
                    ModelInfo(
                        name="llama-3-70b",
                        provider=ModelProvider.LOCAL,
                        max_tokens=8192,
                        supports_streaming=True,
                        supports_function_calling=False,
                        supports_tools=False,
                        context_length=8192,
                        cost_per_1k_input_tokens=0.0009,
                        cost_per_1k_output_tokens=0.0009,
                        description="Meta Llama 3 70B",
                    ),
                    ModelInfo(
                        name="llama-3-8b",
                        provider=ModelProvider.LOCAL,
                        max_tokens=8192,
                        supports_streaming=True,
                        supports_function_calling=False,
                        supports_tools=False,
                        context_length=8192,
                        cost_per_1k_input_tokens=0.0002,
                        cost_per_1k_output_tokens=0.0002,
                        description="Meta Llama 3 8B",
                    ),
                    ModelInfo(
                        name="deepseek-chat",
                        provider=ModelProvider.DEEPSEEK,
                        max_tokens=8192,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_tools=True,
                        context_length=8192,
                        cost_per_1k_input_tokens=0.00014,
                        cost_per_1k_output_tokens=0.00028,
                        description="DeepSeek Chat model",
                    ),
                    ModelInfo(
                        name="mistral-large",
                        provider=ModelProvider.MISTRAL,
                        max_tokens=32768,
                        supports_streaming=True,
                        supports_function_calling=True,
                        supports_tools=True,
                        context_length=32768,
                        cost_per_1k_input_tokens=0.008,
                        cost_per_1k_output_tokens=0.024,
                        description="Mistral Large model",
                    ),
                ]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []

    async def switch_model(self, model_name: str) -> bool:
        """Switch the active model."""
        try:
            if hasattr(self.llm_engine, "set_model"):
                await self.llm_engine.set_model(model_name)
                self.config["default_model"] = model_name
                logger.info(f"Switched to model: {model_name}")
                return True
            else:
                logger.warning("Model switching not supported")
                return False
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    # =========================
    # STATISTICS
    # =========================

    async def get_stats(self) -> Dict[str, Any]:
        """Get API statistics from MongoDB."""
        try:
            engine_stats = {}
            if hasattr(self.llm_engine, "get_metrics"):
                engine_stats = self.llm_engine.get_metrics()

            # Calculate success rate from history
            total_requests = len(self._request_history)
            successful_requests = sum(
                1 for r in self._request_history if r.get("success", False)
            )
            success_rate = (
                (successful_requests / total_requests) if total_requests > 0 else 0
            )

            total_cost = sum(r.get("cost", 0) for r in self._request_history)
            total_tokens = sum(r.get("tokens", 0) for r in self._request_history)

            # Get metrics from MongoDB
            metrics_data = {}
            if mongo_client.enabled and self.metrics_collection is not None:
                # Get total requests
                total_requests_db = await self.metrics_collection.count_documents({})

                # Get average tokens per request
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "avg_tokens": {"$avg": "$tokens_used"},
                            "avg_time": {"$avg": "$total_time_ms"},
                            "total_tokens": {"$sum": "$tokens_used"},
                            "success_rate": {"$avg": {"$cond": ["$success", 1, 0]}},
                        }
                    }
                ]

                agg_result = list(
                    await self.metrics_collection.aggregate(pipeline).to_list(length=1)
                )
                if agg_result:
                    metrics_data = agg_result[0]
                    metrics_data.pop("_id", None)

                # Get recent requests (last hour)
                one_hour_ago = datetime.now() - timedelta(hours=1)
                recent_requests = await self.metrics_collection.count_documents(
                    {"timestamp": {"$gte": one_hour_ago}}
                )

                # Get requests by model
                model_pipeline = [
                    {
                        "$group": {
                            "_id": "$model",
                            "count": {"$sum": 1},
                            "total_tokens": {"$sum": "$tokens_used"},
                            "total_cost": {"$sum": "$cost_usd"},
                        }
                    },
                    {"$sort": {"count": -1}},
                ]
                model_stats = list(
                    await self.metrics_collection.aggregate(model_pipeline).to_list(
                        length=10
                    )
                )

                metrics_data["recent_requests"] = recent_requests
                metrics_data["total_requests"] = total_requests_db
                metrics_data["model_breakdown"] = model_stats

            return {
                "request_count": self.request_count,
                "cache_size": len(self._response_cache),
                "cache_enabled": self.config["enable_cache"],
                "active_requests": len(self._active_requests),
                "success_rate": success_rate,
                "total_requests_history": total_requests,
                "total_tokens_used": total_tokens,
                "total_cost_usd": total_cost,
                "conversations_count": len(self._conversations),
                "rate_limiter": self._rate_limiter.get_stats(),
                "engine_stats": engine_stats,
                "metrics": metrics_data,
                "active_model": getattr(
                    self.llm_engine, "model_name", self.config["default_model"]
                ),
                "rate_limit": self.config["rate_limit_per_minute"],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {
                "request_count": self.request_count,
                "cache_size": len(self._response_cache),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # =========================
    # CACHE MANAGEMENT
    # =========================

    def clear_cache(self) -> Dict[str, Any]:
        """Clear response cache."""
        cache_size = len(self._response_cache)
        self._response_cache.clear()
        self._save_cache()
        logger.info(f"Cleared {cache_size} cached responses")
        return {"cleared": cache_size, "timestamp": datetime.now().isoformat()}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._response_cache),
            "max_size": self._max_cache_size,
            "ttl_seconds": self._cache_ttl,
            "enabled": self.config["enable_cache"],
            "utilization": (
                len(self._response_cache) / self._max_cache_size
                if self._max_cache_size > 0
                else 0
            ),
            "cache_file": str(self._cache_file),
        }

    # =========================
    # HEALTH CHECK
    # =========================

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "components": {},
        }

        # Check LLM engine
        try:
            if hasattr(self.llm_engine, "health_check"):
                llm_status = await self.llm_engine.health_check()
                status["components"]["llm_engine"] = llm_status
            else:
                # Simple test generation
                test_result = await self.llm_engine.generate(
                    prompt="Test", max_tokens=5
                )
                status["components"]["llm_engine"] = {
                    "status": "healthy",
                    "response_time": test_result.get("generation_time", 0),
                }
        except Exception as e:
            status["components"]["llm_engine"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            status["status"] = "degraded"

        # Check MongoDB
        if mongo_client.enabled:
            if mongo_client.status.value == "connected":
                status["components"]["mongodb"] = {
                    "status": "healthy",
                    "database": mongo_client.database_name,
                }
            else:
                status["components"]["mongodb"] = {
                    "status": "unhealthy",
                    "error": f"Status: {mongo_client.status.value}",
                }
                status["status"] = "degraded"

        # Check cache
        status["components"]["cache"] = {
            "status": "healthy",
            "size": len(self._response_cache),
            "enabled": self.config["enable_cache"],
        }

        # Overall status
        if any(
            comp.get("status") == "unhealthy" for comp in status["components"].values()
        ):
            status["status"] = "unhealthy"

        return status

    # =========================
    # SHUTDOWN
    # =========================

    async def shutdown(self):
        """Gracefully shutdown the API."""
        logger.info("Shutting down LLM API...")

        # Save cache
        self._save_cache()

        # Save any pending metrics
        if mongo_client.enabled:
            logger.info("Saving final metrics to MongoDB...")

        # Clear cache
        self.clear_cache()

        logger.info("LLM API shutdown complete")


# =========================
# GLOBAL INSTANCE
# =========================

llm_api = LLMApi()


__all__ = [
    "LLMApi",
    "GenerateRequest",
    "GenerateResponse",
    "ChatRequest",
    "ChatResponse",
    "BatchGenerateRequest",
    "BatchGenerateResponse",
    "ModelInfo",
    "ModelProvider",
    "Message",
    "Role",
    "TokenCountRequest",
    "TokenCountResponse",
    "Conversation",
    "ConversationTurn",
    "llm_api",
]
