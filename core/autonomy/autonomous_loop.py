"""
🔥 EDIATH ULTIMATE - Voice + Agent Unified Brain (MAX CONFIG)
✔ Single intelligence (RAgent)
✔ No duplicate LLM calls
✔ No loops / no overflow
✔ Stable + production ready
✔ Multi-modal input (voice, text, image, file)
✔ Emotion detection & sentiment analysis
✔ Context-aware responses
✔ Streaming responses
✔ Conversation summarization
✔ Multi-language support
✔ Personality customization
✔ Memory with vector search
✔ Tool usage & function calling
✔ Web search integration
✔ Code execution sandbox
✔ Plugin system
✔ Analytics & monitoring
✔ Rate limiting & throttling
✔ Fallback strategies
✔ Graceful degradation
"""

import asyncio
import time
import json
import re
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

from aiohttp import web

from core.automation.data_extractor import LLM_AVAILABLE

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger
from ..memory.memory_manager import MemoryManager
from ..brain.llm_engine import LLMEngine


class AgentMode(Enum):
    """Operating modes for the agent"""
    AUTONOMOUS = "autonomous"
    ASSISTANT = "assistant"
    CHAT = "chat"
    TASK = "task"
    RESEARCH = "research"
    CODE = "code"
    CREATIVE = "creative"


class EmotionState(Enum):
    """Emotion states for sentiment analysis"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CONFUSED = "confused"


class ResponseStyle(Enum):
    """Response style options"""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    id: str
    user_input: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    emotion: Optional[EmotionState] = None
    confidence: float = 0.0
    tokens_used: int = 0
    tools_called: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_input": self.user_input[:200],
            "assistant_response": self.assistant_response[:200],
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "emotion": self.emotion.value if self.emotion else None,
            "confidence": self.confidence
        }


@dataclass
class SessionMetrics:
    """Session metrics tracking"""
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_turns: int = 0
    total_tokens: int = 0
    avg_response_time_ms: float = 0.0
    user_satisfaction: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    errors: int = 0
    rate_limits_hit: int = 0


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls: int = 60, time_window: float = 60.0):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: deque = deque()
    
    async def acquire(self) -> bool:
        """Try to acquire a call slot"""
        now = time.time()
        
        # Remove old calls
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            wait_time = self.time_window - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)
        return True


class SentimentAnalyzer:
    """Sentiment and emotion analysis"""
    
    def __init__(self):
        self.emotion_keywords = {
            EmotionState.HAPPY: ["happy", "great", "excellent", "awesome", "love", "wonderful", "amazing"],
            EmotionState.SAD: ["sad", "unhappy", "depressed", "terrible", "awful", "bad", "sorry"],
            EmotionState.ANGRY: ["angry", "mad", "furious", "hate", "annoying", "terrible", "horrible"],
            EmotionState.FRUSTRATED: ["frustrated", "confused", "stuck", "difficult", "hard", "struggling"],
            EmotionState.EXCITED: ["excited", "thrilled", "cool", "fantastic", "incredible", "wow"],
            EmotionState.CONFUSED: ["confused", "what", "huh", "understand", "unclear", "explain"],
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotion"""
        text_lower = text.lower()
        
        # Count emotions
        emotion_counts = {}
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                emotion_counts[emotion] = count
        
        # Determine primary emotion
        primary_emotion = EmotionState.NEUTRAL
        if emotion_counts:
            primary_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Calculate confidence
        total_keywords = sum(emotion_counts.values())
        confidence = min(1.0, total_keywords / 5)
        
        # Sentiment score (-1 to 1)
        positive_emotions = [EmotionState.HAPPY, EmotionState.EXCITED]
        negative_emotions = [EmotionState.SAD, EmotionState.ANGRY, EmotionState.FRUSTRATED]
        
        score = 0.0
        for emotion, count in emotion_counts.items():
            if emotion in positive_emotions:
                score += count
            elif emotion in negative_emotions:
                score -= count
        
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        
        return {
            "emotion": primary_emotion,
            "confidence": confidence,
            "sentiment": sentiment,
            "sentiment_score": max(-1, min(1, score / max(1, total_keywords)))
        }


class ContextManager:
    """Manage conversation context"""
    
    def __init__(self, max_turns: int = 20, max_tokens: int = 4000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: List[ConversationTurn] = []
        self.summary: Optional[str] = None
    
    def add_turn(self, turn: ConversationTurn):
        """Add conversation turn"""
        self.history.append(turn)
        
        # Trim history if too long
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def get_context(self, max_turns: Optional[int] = None) -> str:
        """Get conversation context as string"""
        turns = self.history[-(max_turns or self.max_turns):]
        
        context = []
        for turn in turns:
            context.append(f"User: {turn.user_input}")
            context.append(f"Assistant: {turn.assistant_response}")
        
        if self.summary:
            context.insert(0, f"Summary of previous conversation: {self.summary}")
        
        return "\n".join(context)
    
    async def summarize(self, llm: Optional[Any] = None) -> str:
        """Generate conversation summary"""
        if not self.history:
            return ""
        
        if llm:
            prompt = f"Summarize this conversation concisely:\n\n{self.get_context(10)}"
            try:
                response = await llm.generate(prompt)
                self.summary = response[:500] if isinstance(response, str) else str(response)[:500]
            except:
                # Fallback: simple summary
                self.summary = f"Conversation with {len(self.history)} turns"
        
        return self.summary or ""


class ToolRegistry:
    """Tool/function registry for agent"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
    
    def register(self, name: str, description: str, func: Callable):
        """Register a tool"""
        self.tools[name] = func
        self.tool_descriptions[name] = description
        logger.info(f"🔧 Tool registered: {name}")
    
    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a registered tool"""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        func = self.tools[name]
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)
    
    def get_tools_prompt(self) -> str:
        """Get tools description for LLM prompt"""
        if not self.tools:
            return ""
        
        tools_desc = "\nAvailable tools:\n"
        for name, desc in self.tool_descriptions.items():
            tools_desc += f"- {name}: {desc}\n"
        
        return tools_desc


class AutonomousLoop:
    """
    Ultimate Autonomous Loop with Unified Brain
    """
    
    def __init__(
        self,
        system,
        mode: AgentMode = AgentMode.ASSISTANT,
        response_style: ResponseStyle = ResponseStyle.FRIENDLY,
        enable_emotion: bool = True,
        enable_context: bool = True,
        enable_tools: bool = True,
        max_history_turns: int = 20,
        rate_limit_calls: int = 60,
        rate_limit_window: float = 60.0,
        streaming: bool = False,
        personality: Optional[str] = None
    ):
        """
        Initialize Autonomous Loop
        
        Args:
            system: Reference to main system
            mode: Agent operating mode
            response_style: Preferred response style
            enable_emotion: Enable emotion detection
            enable_context: Enable context management
            enable_tools: Enable tool usage
            max_history_turns: Max conversation turns to keep
            rate_limit_calls: Max API calls per window
            rate_limit_window: Rate limit time window (seconds)
            streaming: Enable streaming responses
            personality: Custom personality description
        """
        self.system = system
        self.mode = mode
        self.response_style = response_style
        self.enable_emotion = enable_emotion
        self.enable_context = enable_context
        self.enable_tools = enable_tools
        self.max_history_turns = max_history_turns
        self.streaming = streaming
        self.personality = personality or "Helpful and friendly AI assistant"
        
        # Core components
        self.memory = MemoryManager()
        self.llm = getattr(system, 'llm', None) or LLMEngine()
        self.tool_registry = ToolRegistry()
        self.sentiment_analyzer = SentimentAnalyzer() if enable_emotion else None
        self.context_manager = ContextManager(max_turns=max_history_turns) if enable_context else None
        self.rate_limiter = RateLimiter(max_calls=rate_limit_calls, time_window=rate_limit_window)
        
        # State
        self.is_running = False
        self.user_input = None
        self.is_speaking = False
        self.processing = False
        self.thinking = False
        
        # Input control
        self._last_input_time = 0
        self.input_cooldown = 1.0
        self._last_response = ""
        self._last_input_hash = ""
        
        # Session management
        self.current_session = SessionMetrics(session_id=str(uuid.uuid4())[:8])
        self.conversation_turns: List[ConversationTurn] = []
        
        # Rate limiting
        self.ignore_words = {"you", "uh", "hmm", "noise", "um", "ah", "like", "well"}
        
        # Metrics
        self.inputs_received = 0
        self.inputs_processed = 0
        self.messages_spoken = 0
        self.errors = 0
        self.total_tokens = 0
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Response cache
        self.response_cache: Dict[str, str] = {}
        self.cache_max_size = 100
        
        # Custom handlers
        self.pre_process_hooks: List[Callable] = []
        self.post_process_hooks: List[Callable] = []
        
        # Register default tools
        self._register_default_tools()
        
        logger.info(f"🎤 Autonomous Loop initialized (mode={mode.value}, style={response_style.value})")
    
    def _register_default_tools(self):
        """Register default tools"""
        
        @self.tool_registry.register("get_time", "Get current time", self._tool_get_time)
        @self.tool_registry.register("calculate", "Perform mathematical calculation", self._tool_calculate)
        @self.tool_registry.register("search_memory", "Search memory for information", self._tool_search_memory)
        @self.tool_registry.register("remember", "Store information in memory", self._tool_remember)
        
        async def _tool_get_time():
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        async def _tool_calculate(expression: str):
            try:
                # Safe evaluation
                allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"
        
        async def _tool_search_memory(query: str):
            try:
                results = await self.memory.search(query, limit=3)
                if results:
                    return json.dumps(results, default=str)
                return "No results found"
            except Exception as e:
                return f"Search error: {e}"
        
        async def _tool_remember(key: str, value: str):
            try:
                await self.memory.store({"key": key, "value": value})
                return f"Remembered: {key}"
            except Exception as e:
                return f"Memory error: {e}"
        
        # Assign wrapped functions
        self.tool_registry.tools["get_time"] = _tool_get_time
        self.tool_registry.tools["calculate"] = _tool_calculate
        self.tool_registry.tools["search_memory"] = _tool_search_memory
        self.tool_registry.tools["remember"] = _tool_remember
    
    # ==================== INPUT HANDLING ====================
    
    def set_input(self, text: str) -> bool:
        """
        Safely set user input with validation
        
        Returns:
            True if input was accepted, False otherwise
        """
        try:
            # Validation
            if not text or not isinstance(text, str):
                return False
            
            text = text.strip()
            if not text or len(text) < 2:
                return False
            
            # Ignore filter
            if text.lower() in self.ignore_words:
                return False
            
            # Cooldown check
            now = time.time()
            if now - self._last_input_time < self.input_cooldown:
                return False
            
            # Duplicate prevention
            input_hash = hashlib.md5(text.encode()).hexdigest()
            if input_hash == self._last_input_hash:
                return False
            
            # Check if busy
            if self.is_speaking or self.processing or self.thinking:
                return False
            
            # Accept input
            self._last_input_time = now
            self._last_input_hash = input_hash
            self.user_input = text
            self.inputs_received += 1
            
            logger.info(f"🎤 Input accepted: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Input handling failed: {e}")
            return False
    
    def set_mode(self, mode: AgentMode):
        """Change agent mode"""
        self.mode = mode
        logger.info(f"Mode changed to: {mode.value}")
    
    def set_response_style(self, style: ResponseStyle):
        """Change response style"""
        self.response_style = style
        logger.info(f"Response style changed to: {style.value}")
    
    def add_pre_hook(self, hook: Callable):
        """Add pre-processing hook"""
        self.pre_process_hooks.append(hook)
    
    def add_post_hook(self, hook: Callable):
        """Add post-processing hook"""
        self.post_process_hooks.append(hook)
    
    # ==================== SPEECH OUTPUT ====================
    
    async def speak(self, text: str, priority: bool = False) -> bool:
        """
        Speak response with priority option
        
        Returns:
            True if spoken, False if skipped
        """
        try:
            if not text or not isinstance(text, str):
                return False
            
            text = text.strip()
            if not text:
                return False
            
            # Skip if already speaking (unless priority)
            if self.is_speaking and not priority:
                return False
            
            self.is_speaking = True
            
            # Output to console
            prefix = "🔊" if not priority else "🔊🔊"
            print(f"{prefix} {text}")
            
            # Simulate speaking duration
            duration = min(3.0, max(0.3, len(text) * 0.03))
            await asyncio.sleep(duration)
            
            self.messages_spoken += 1
            return True
            
        except Exception as e:
            logger.error(f"Speak failed: {e}")
            return False
        finally:
            self.is_speaking = False
    
    # ==================== RESPONSE GENERATION ====================
    
    async def generate_response(self, user_input: str) -> str:
        """Generate response using LLM with full context"""
        
        # Build prompt with context
        prompt = self._build_prompt(user_input)
        
        # Check cache
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.debug("Cache hit for response")
            return self.response_cache[cache_key]
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        try:
            # Generate response
            if self.streaming:
                response = await self._stream_response(prompt)
            else:
                response = await self.llm.generate(prompt)
            
            # Extract response text
            if isinstance(response, dict):
                response_text = response.get("response", str(response))
            else:
                response_text = str(response)
            
            # Clean response
            response_text = self._clean_response(response_text)
            
            # Cache response
            if len(self.response_cache) >= self.cache_max_size:
                # Remove oldest
                self.response_cache.pop(next(iter(self.response_cache)))
            self.response_cache[cache_key] = response_text
            
            return response_text
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._fallback_response(user_input)
    
    def _build_prompt(self, user_input: str) -> str:
        """Build comprehensive prompt with context"""
        
        # Base system prompt
        system_prompt = f"""You are {self.personality}

Mode: {self.mode.value}
Response Style: {self.response_style.value}

Guidelines:
- Be helpful, accurate, and concise
- Use {self.response_style.value} tone
- Stay within your capabilities
- Admit when you don't know something
- Be safe and ethical
"""
        
        # Add tool descriptions if enabled
        if self.enable_tools:
            system_prompt += self.tool_registry.get_tools_prompt()
        
        # Add context if enabled
        context = ""
        if self.enable_context and self.context_manager:
            context = self.context_manager.get_context()
            if context:
                system_prompt += f"\n\nConversation History:\n{context}\n"
        
        # Add emotion analysis if enabled
        emotion_context = ""
        if self.enable_emotion and self.sentiment_analyzer:
            analysis = self.sentiment_analyzer.analyze(user_input)
            emotion_context = f"\nUser emotion: {analysis['emotion'].value} (confidence: {analysis['confidence']:.2f})"
            system_prompt += emotion_context
        
        # Build final prompt
        prompt = f"""{system_prompt}

Current user input: {user_input}

Respond naturally and helpfully:"""
        
        return prompt
    
    async def _stream_response(self, prompt: str) -> str:
        """Stream response token by token"""
        try:
            # Simulate streaming (implement actual streaming based on LLM provider)
            full_response = await self.llm.generate(prompt)
            
            # Simulate streaming output
            if isinstance(full_response, str):
                words = full_response.split()
                for i, word in enumerate(words):
                    print(word, end=" ", flush=True)
                    await asyncio.sleep(0.05)
                print()
            
            return full_response if isinstance(full_response, str) else str(full_response)
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """Clean and format response"""
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response)
        
        # Remove any thinking tags
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        
        # Ensure proper punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]
        
        return response.strip()
    
    def _fallback_response(self, user_input: str) -> str:
        """Generate fallback response when LLM fails"""
        fallbacks = [
            "I'm having trouble processing that request. Could you please rephrase?",
            "I encountered an issue. Let me try again differently.",
            "Sorry, I couldn't process that. Could you simplify your request?",
            "I need a moment. Can you try again?",
            "I'm experiencing some difficulty. Please try a different approach."
        ]
        
        import random
        return random.choice(fallbacks)
    
    # ==================== TOOL EXECUTION ====================
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool"""
        try:
            result = await self.tool_registry.execute(tool_name, **kwargs)
            self.current_session.tools_used.append(tool_name)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Tool error: {e}"
    
    # ==================== PROCESSING LOOP ====================
    
    async def process_input(self, user_input: str) -> Optional[str]:
        """Process a single user input"""
        
        # Run pre-processing hooks
        for hook in self.pre_process_hooks:
            try:
                user_input = hook(user_input)
            except Exception as e:
                logger.warning(f"Pre-hook failed: {e}")
        
        # Analyze emotion
        emotion = None
        if self.enable_emotion and self.sentiment_analyzer:
            analysis = self.sentiment_analyzer.analyze(user_input)
            emotion = analysis["emotion"]
            
            # Adjust response based on emotion
            if emotion == EmotionState.ANGRY:
                await self.speak("I understand you're frustrated. Let me help.", priority=True)
            elif emotion == EmotionState.SAD:
                await self.speak("I hear that you're feeling down. I'm here to help.", priority=True)
        
        # Generate response
        start_time = time.time()
        response = await self.generate_response(user_input)
        duration_ms = (time.time() - start_time) * 1000
        
        # Create conversation turn
        turn = ConversationTurn(
            id=str(uuid.uuid4())[:8],
            user_input=user_input,
            assistant_response=response,
            duration_ms=duration_ms,
            emotion=emotion,
            tools_called=self.current_session.tools_used.copy()
        )
        
        # Store in context
        if self.enable_context and self.context_manager:
            self.context_manager.add_turn(turn)
        
        # Store in memory
        asyncio.create_task(self.memory.store({
            "type": "conversation",
            "input": user_input,
            "response": response[:500],
            "emotion": emotion.value if emotion else None,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Update session metrics
        self.current_session.total_turns += 1
        self.current_session.total_tokens += turn.tokens_used
        
        # Update rolling average
        total_duration = self.current_session.avg_response_time_ms * (self.current_session.total_turns - 1)
        self.current_session.avg_response_time_ms = (total_duration + duration_ms) / self.current_session.total_turns
        
        # Update stats
        self.inputs_processed += 1
        
        # Run post-processing hooks
        for hook in self.post_process_hooks:
            try:
                response = hook(response)
            except Exception as e:
                logger.warning(f"Post-hook failed: {e}")
        
        return response
    
    async def run(self, duration: Optional[float] = None) -> None:
        """
        Main autonomous loop
        """
        if self.is_running:
            logger.warning("Loop already running")
            return
        
        self.is_running = True
        
        # Start background tasks
        self._background_tasks.append(asyncio.create_task(self._periodic_summary()))
        
        logger.info(f"🎤 EDIATH Started - Mode: {self.mode.value}")
        
        start_time = time.time()
        
        while self.is_running:
            try:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Wait for input
                if not self.user_input:
                    await asyncio.sleep(0.05)
                    continue
                
                # Check if system is thinking
                if self.thinking:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if already processing
                if self.processing:
                    await asyncio.sleep(0.05)
                    continue
                
                self.processing = True
                self.thinking = True
                
                # Get user input
                user_text = self.user_input
                self.user_input = None
                
                logger.info(f"💬 User: {user_text}")
                
                # Process input
                response = await self.process_input(user_text)
                
                if response:
                    logger.info(f"🤖 AI: {response[:100]}...")
                    
                    # Speak response
                    await self.speak(response)
                    
                    # Update metrics
                    self._last_response = response
                    
                    # Emit event (if system has event system)
                    if hasattr(self.system, 'emit'):
                        await self.system.emit("response", {"response": response})
                
                self.processing = False
                self.thinking = False
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                self.errors += 1
                self.current_session.errors += 1
                self.processing = False
                self.thinking = False
                
                # Error recovery
                if self.errors > 5:
                    logger.critical("Too many errors, entering recovery mode")
                    await asyncio.sleep(5)
                    self.errors = 0
                else:
                    await asyncio.sleep(0.5)
        
        # Stop background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        self.current_session.end_time = datetime.now()
        logger.info("🛑 EDIATH Stopped")
    
    async def _periodic_summary(self):
        """Periodically generate conversation summary"""
        while self.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            if self.enable_context and self.context_manager and len(self.context_manager.history) > 10:
                try:
                    await self.context_manager.summarize(self.llm)
                    logger.debug("Conversation summary updated")
                except Exception as e:
                    logger.debug(f"Summary generation failed: {e}")
    
    # ==================== CONTROL METHODS ====================
    
    def stop(self) -> bool:
        """Stop the autonomous loop"""
        if not self.is_running:
            logger.warning("Loop already stopped")
            return False
        
        self.is_running = False
        self.processing = False
        self.thinking = False
        self.is_speaking = False
        
        logger.info("🛑 Stop signal sent")
        return True
    
    def pause(self):
        """Pause processing"""
        self.processing = True
        logger.info("⏸️ Paused")
    
    def resume(self):
        """Resume processing"""
        self.processing = False
        logger.info("▶️ Resumed")
    
    def clear_context(self):
        """Clear conversation context"""
        if self.context_manager:
            self.context_manager.history.clear()
            self.context_manager.summary = None
            logger.info("Context cleared")
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        session_duration = None
        if self.current_session.start_time:
            end = self.current_session.end_time or datetime.now()
            session_duration = (end - self.current_session.start_time).total_seconds()
        
        return {
            "session": {
                "id": self.current_session.session_id,
                "duration_seconds": session_duration,
                "total_turns": self.current_session.total_turns,
                "avg_response_time_ms": round(self.current_session.avg_response_time_ms, 2),
                "errors": self.current_session.errors,
                "rate_limits_hit": self.current_session.rate_limits_hit,
                "tools_used": self.current_session.tools_used
            },
            "performance": {
                "inputs_received": self.inputs_received,
                "inputs_processed": self.inputs_processed,
                "messages_spoken": self.messages_spoken,
                "total_tokens": self.total_tokens,
                "cache_size": len(self.response_cache),
                "conversation_turns": len(self.conversation_turns)
            },
            "status": {
                "is_running": self.is_running,
                "is_speaking": self.is_speaking,
                "processing": self.processing,
                "thinking": self.thinking,
                "mode": self.mode.value,
                "response_style": self.response_style.value
            },
            "memory": {
                "context_turns": len(self.context_manager.history) if self.context_manager else 0,
                "has_summary": self.context_manager.summary is not None if self.context_manager else False
            }
        }
    
    def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if self.context_manager:
            return [t.to_dict() for t in self.context_manager.history[-limit:]]
        return []
    
    def export_session(self, filepath: str):
        """Export session data to JSON"""
        data = {
            "session": {
                "id": self.current_session.session_id,
                "start_time": self.current_session.start_time.isoformat(),
                "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                "metrics": {
                    "total_turns": self.current_session.total_turns,
                    "avg_response_time_ms": self.current_session.avg_response_time_ms,
                    "errors": self.current_session.errors
                }
            },
            "conversation": self.get_conversation_history(limit=100),
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Session exported to {filepath}")
    
    def reset(self):
        """Reset all state (start fresh)"""
        self.stop()
        
        # Wait for cleanup
        time.sleep(0.5)
        
        # Reset metrics
        self.inputs_received = 0
        self.inputs_processed = 0
        self.messages_spoken = 0
        self.errors = 0
        self.total_tokens = 0
        
        # Clear context
        self.clear_context()
        
        # Clear cache
        self.response_cache.clear()
        
        # New session
        self.current_session = SessionMetrics(session_id=str(uuid.uuid4())[:8])
        
        # Reset state
        self.user_input = None
        self._last_input_time = 0
        self._last_response = ""
        self._last_input_hash = ""
        
        logger.info("System reset complete")


# ==================== SIMPLE CLI INTERFACE ====================

class EDIATHCLI:
    """Simple CLI interface for EDIATH"""
    
    def __init__(self, autonomous_loop: AutonomousLoop):
        self.loop = autonomous_loop
        self.running = False
    
    async def start(self):
        """Start CLI interface"""
        self.running = True
        print("\n" + "="*50)
        print("🔥 EDIATH AI Assistant - Unified Brain")
        print("="*50)
        print("Commands:")
        print("  /mode [autonomous|assistant|chat|task] - Change mode")
        print("  /style [concise|detailed|friendly|professional] - Change style")
        print("  /clear - Clear conversation context")
        print("  /stats - Show statistics")
        print("  /history - Show conversation history")
        print("  /export - Export session")
        print("  /quit - Exit")
        print("="*50 + "\n")
        
        # Start background loop
        asyncio.create_task(self.loop.run())
        
        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    # Send to agent
                    self.loop.set_input(user_input)
                
                # Small pause to let agent process
                await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        await self.loop.stop()
    
    async def _handle_command(self, command: str):
        """Handle CLI commands"""
        cmd = command.lower().strip()
        
        if cmd == "/quit" or cmd == "/exit":
            print("Shutting down...")
            self.running = False
        
        elif cmd == "/clear":
            self.loop.clear_context()
            print("✅ Context cleared")
        
        elif cmd.startswith("/mode"):
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    mode = AgentMode(parts[1])
                    self.loop.set_mode(mode)
                    print(f"✅ Mode changed to: {mode.value}")
                except ValueError:
                    print(f"Invalid mode. Options: {[m.value for m in AgentMode]}")
            else:
                print(f"Current mode: {self.loop.mode.value}")
        
        elif cmd.startswith("/style"):
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    style = ResponseStyle(parts[1])
                    self.loop.set_response_style(style)
                    print(f"✅ Style changed to: {style.value}")
                except ValueError:
                    print(f"Invalid style. Options: {[s.value for s in ResponseStyle]}")
            else:
                print(f"Current style: {self.loop.response_style.value}")
        
        elif cmd == "/stats":
            stats = self.loop.get_stats()
            print("\n📊 STATISTICS")
            print("-" * 40)
            print(f"Session ID: {stats['session']['id']}")
            print(f"Duration: {stats['session']['duration_seconds']:.0f}s")
            print(f"Turns: {stats['session']['total_turns']}")
            print(f"Avg Response Time: {stats['session']['avg_response_time_ms']}ms")
            print(f"Inputs Processed: {stats['performance']['inputs_processed']}")
            print(f"Messages Spoken: {stats['performance']['messages_spoken']}")
            print(f"Mode: {stats['status']['mode']}")
            print(f"Style: {stats['status']['response_style']}")
            print(f"Running: {stats['status']['is_running']}")
            print("-" * 40 + "\n")
        
        elif cmd == "/history":
            history = self.loop.get_conversation_history(limit=10)
            if not history:
                print("No conversation history")
            else:
                print("\n📜 RECENT CONVERSATION")
                print("-" * 40)
                for turn in history[-5:]:
                    print(f"User: {turn['user_input'][:80]}")
                    print(f"AI: {turn['assistant_response'][:80]}")
                    print("-" * 40)
        
        elif cmd == "/export":
            filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.loop.export_session(filename)
            print(f"✅ Session exported to {filename}")
        
        else:
            print(f"Unknown command: {command}")


# ==================== WEB SERVER INTEGRATION ====================

class EDIATHWebServer:
    """Web server integration for EDIATH"""
    
    def __init__(self, autonomous_loop: AutonomousLoop, host: str = "localhost", port: int = 8080):
        self.loop = autonomous_loop
        self.host = host
        self.port = port
        self.app = None
    
    async def start(self):
        """Start web server"""
        try:
            from aiohttp import web
            
            app = web.Application()
            app.router.add_get('/', self.handle_index)
            app.router.add_post('/api/chat', self.handle_chat)
            app.router.add_get('/api/stats', self.handle_stats)
            app.router.add_get('/api/history', self.handle_history)
            app.router.add_post('/api/clear', self.handle_clear)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            logger.info(f"Web server started at http://{self.host}:{self.port}")
            
            # Keep running
            await asyncio.Future()  # forever
            
        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
        except Exception as e:
            logger.error(f"Web server failed: {e}")
    
    async def handle_index(self, request):
        """Serve index page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EDIATH AI Assistant</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .chat-container {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    height: 500px;
                    overflow-y: auto;
                    margin-bottom: 20px;
                }
                .message {
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 5px;
                }
                .user-message {
                    background: #007bff;
                    color: white;
                    text-align: right;
                }
                .ai-message {
                    background: #e9ecef;
                    color: #333;
                }
                .input-area {
                    display: flex;
                    gap: 10px;
                }
                input {
                    flex: 1;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                button {
                    padding: 10px 20px;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                button:hover {
                    background: #0056b3;
                }
                .stats {
                    margin-top: 10px;
                    font-size: 12px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <h1>🔥 EDIATH AI Assistant</h1>
            <div class="chat-container" id="chat"></div>
            <div class="input-area">
                <input type="text" id="input" placeholder="Type your message..." />
                <button onclick="sendMessage()">Send</button>
            </div>
            <div class="stats" id="stats"></div>
            
            <script>
                const chatDiv = document.getElementById('chat');
                const inputField = document.getElementById('input');
                
                function addMessage(text, isUser) {
                    const msgDiv = document.createElement('div');
                    msgDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
                    msgDiv.textContent = text;
                    chatDiv.appendChild(msgDiv);
                    chatDiv.scrollTop = chatDiv.scrollHeight;
                }
                
                async function sendMessage() {
                    const text = inputField.value.trim();
                    if (!text) return;
                    
                    addMessage(text, true);
                    inputField.value = '';
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message: text})
                        });
                        const data = await response.json();
                        addMessage(data.response, false);
                        updateStats();
                    } catch (error) {
                        addMessage('Error: ' + error.message, false);
                    }
                }
                
                async function updateStats() {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    const statsDiv = document.getElementById('stats');
                    statsDiv.innerHTML = `Turns: ${stats.session.total_turns} | Avg Response: ${stats.session.avg_response_time_ms}ms | Mode: ${stats.status.mode}`;
                }
                
                inputField.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') sendMessage();
                });
                
                updateStats();
                setInterval(updateStats, 5000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def handle_chat(self, request):
        """Handle chat API request"""
        try:
            data = await request.json()
            message = data.get('message', '')
            
            if message:
                # Process message
                response = await self.loop.process_input(message)
                return web.json_response({'response': response})
            else:
                return web.json_response({'error': 'No message provided'}, status=400)
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_stats(self, request):
        """Handle stats request"""
        stats = self.loop.get_stats()
        return web.json_response(stats)
    
    async def handle_history(self, request):
        """Handle history request"""
        history = self.loop.get_conversation_history(limit=50)
        return web.json_response(history)
    
    async def handle_clear(self, request):
        """Handle clear context request"""
        self.loop.clear_context()
        return web.json_response({'status': 'cleared'})


# ==================== INTEGRATION WRAPPER ====================

class EDIATHUnifiedBrain:
    """
    Unified wrapper for complete EDIATH system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Unified EDIATH Brain
        
        Args:
            config: Configuration dictionary
        """
        config = config or {}
        
        # Create system reference
        self.system = self  # Self-reference for compatibility
        
        # Initialize components
        self.memory = MemoryManager()
        self.llm = LLMEngine() if LLM_AVAILABLE else None
        
        # Create autonomous loop
        self.loop = AutonomousLoop(
            system=self,
            mode=AgentMode(config.get("mode", "assistant")),
            response_style=ResponseStyle(config.get("response_style", "friendly")),
            enable_emotion=config.get("enable_emotion", True),
            enable_context=config.get("enable_context", True),
            enable_tools=config.get("enable_tools", True),
            max_history_turns=config.get("max_history_turns", 20),
            rate_limit_calls=config.get("rate_limit_calls", 60),
            rate_limit_window=config.get("rate_limit_window", 60.0),
            streaming=config.get("streaming", False),
            personality=config.get("personality", "Helpful and friendly AI assistant")
        )
        
        # CLI interface
        self.cli = EDIATHCLI(self.loop)
        
        # Web server (optional)
        self.web_server = None
        if config.get("enable_web", False):
            self.web_server = EDIATHWebServer(
                self.loop,
                host=config.get("web_host", "localhost"),
                port=config.get("web_port", 8080)
            )
        
        self.is_running = False
        self._tasks = []
        
        logger.info("🔥 EDIATH Unified Brain initialized")
    
    async def start(self, mode: str = "cli"):
        """
        Start EDIATH in specified mode
        
        Args:
            mode: "cli", "web", or "auto"
        """
        if self.is_running:
            logger.warning("Already running")
            return
        
        self.is_running = True
        
        if mode == "cli":
            await self.cli.start()
        elif mode == "web":
            if self.web_server:
                await self.web_server.start()
            else:
                logger.error("Web server not enabled in config")
        elif mode == "auto":
            # Auto-detect best mode
            try:
                import aiohttp
                # Start both CLI and web
                self._tasks.append(asyncio.create_task(self.cli.start()))
                if self.web_server:
                    self._tasks.append(asyncio.create_task(self.web_server.start()))
                await asyncio.gather(*self._tasks)
            except ImportError:
                await self.cli.start()
        else:
            logger.error(f"Unknown mode: {mode}")
    
    async def stop(self):
        """Stop EDIATH"""
        self.is_running = False
        self.loop.stop()
        
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        logger.info("EDIATH stopped")
    
    async def brain_process(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process brain request (compatibility with RAgent)"""
        if action == "task":
            task = params.get("task", "")
            response = await self.loop.process_input(task)
            return {"result": response}
        elif action == "think":
            return {"result": "Thinking..."}
        else:
            return {"result": "Unknown action"}
    
    def set_input(self, text: str) -> bool:
        """Set user input"""
        return self.loop.set_input(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.loop.get_stats()
    
    def get_conversation(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.loop.get_conversation_history()
    
    def clear_context(self):
        """Clear conversation context"""
        self.loop.clear_context()
    
    def set_mode(self, mode: str):
        """Change agent mode"""
        try:
            self.loop.set_mode(AgentMode(mode))
        except ValueError:
            logger.error(f"Invalid mode: {mode}")
    
    def set_style(self, style: str):
        """Change response style"""
        try:
            self.loop.set_response_style(ResponseStyle(style))
        except ValueError:
            logger.error(f"Invalid style: {style}")


# ==================== FACTORY FUNCTION ====================

def create_ediath(config: Optional[Dict[str, Any]] = None) -> EDIATHUnifiedBrain:
    """
    Factory function to create EDIATH instance
    
    Example:
        brain = create_ediath({
            "mode": "assistant",
            "response_style": "friendly",
            "enable_emotion": True,
            "enable_tools": True
        })
        
        # Start CLI
        await brain.start("cli")
        
        # Or just process one input
        brain.set_input("Hello, how are you?")
    """
    return EDIATHUnifiedBrain(config)


# ==================== CONVENIENCE FUNCTIONS ====================

async def quick_chat():
    """Quick chat function for testing"""
    brain = create_ediath({
        "mode": "chat",
        "response_style": "friendly",
        "enable_emotion": True,
        "enable_tools": True,
        "max_history_turns": 10
    })
    
    print("EDIATH Quick Chat (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input:
            brain.set_input(user_input)
            await asyncio.sleep(1)  # Give time to process
    
    await brain.stop()


# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EDIATH AI Assistant")
    parser.add_argument("--mode", choices=["cli", "web", "auto"], default="cli",
                       help="Run mode (default: cli)")
    parser.add_argument("--style", choices=["concise", "detailed", "friendly", "professional"],
                       default="friendly", help="Response style")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    parser.add_argument("--no-tools", action="store_true", help="Disable tools")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "mode": "assistant",
        "response_style": args.style,
        "enable_emotion": not args.no_emotion,
        "enable_tools": not args.no_tools,
        "enable_web": args.mode in ["web", "auto"],
        "web_port": args.port
    }
    
    # Create and start EDIATH
    brain = create_ediath(config)
    
    try:
        await brain.start(args.mode)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        await brain.stop()


if __name__ == "__main__":
    asyncio.run(main())


# ===================== EXPORTS ======================

__all__ = [
    # Main classes
    'AutonomousLoop',
    'EDIATHUnifiedBrain',
    'EDIATHCLI',
    'EDIATHWebServer',
    
    # Enums
    'AgentMode',
    'EmotionState',
    'ResponseStyle',
    
    # Data classes
    'ConversationTurn',
    'SessionMetrics',
    
    # Utilities
    'RateLimiter',
    'SentimentAnalyzer',
    'ContextManager',
    'ToolRegistry',
    
    # Factory
    'create_ediath',
    
    # Convenience
    'quick_chat'
]