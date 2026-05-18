"""
Reasoning Engine - handles complex reasoning and problem-solving.
(UPGRADED - PRODUCTION READY, WINDOWS COMPATIBLE)
"""

import json
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, datetimek 

from ..utils.logger import logger

# Safe imports with fallbacks
try:
    from ..brain.llm_engine import LLMEngine
except ImportError:
    LLMEngine = None
    logger.warning("LLMEngine not available for reasoning")

try:
    from ..memory.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None
    logger.warning("MemoryManager not available for reasoning")


class ReasoningStrategy(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    MULTI_STEP = "multi_step"
    ANALOGY = "analogy"
    ABDUCTION = "abduction"


class ReasoningStep:
    def __init__(self, index: int, content: str, confidence: float = 0.7):
        self.index = index
        self.content = content
        self.confidence = max(0, min(1, confidence))
        self.timestamp = datetime.now()
        self.annotations: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "annotations": self.annotations,
        }


class ReasoningChain:
    def __init__(self, problem: str):
        self.problem = problem
        self.steps: List[ReasoningStep] = []
        self.strategy: Optional[ReasoningStrategy] = None
        self.conclusion: Optional[str] = None
        self.confidence: float = 0.0
        self.created_at = datetime.now()

    def add_step(self, content: str, confidence: float = 0.7) -> ReasoningStep:
        step = ReasoningStep(len(self.steps), content, confidence)
        self.steps.append(step)
        return step

    def set_conclusion(self, conclusion: str, confidence: float = 0.8) -> None:
        self.conclusion = conclusion
        self.confidence = max(0, min(1, confidence))

    def get_reasoning_text(self) -> str:
        text = f"Problem: {self.problem}\n\n"
        for step in self.steps:
            text += f"Step {step.index + 1}: {step.content}\n"
        if self.conclusion:
            text += f"\nConclusion: {self.conclusion}\n"
            text += f"Confidence: {self.confidence:.2%}\n"
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "strategy": self.strategy.value if self.strategy else None,
            "steps": [step.to_dict() for step in self.steps],
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


class ReasoningEngine:
    def __init__(self):
        self.chains: Dict[str, ReasoningChain] = {}
        self.reasoning_count = 0

        # Rate limiting
        self._last_reasoning_time = 0

        # The main system injects the shared LLM later. Do not load the model
        # in this constructor.
        self.llm = None

        try:
            if MemoryManager:
                self.memory = MemoryManager()
            else:
                self.memory = None
                logger.warning("MemoryManager not available")
        except Exception as e:
            logger.warning(f"Memory init failed: {e}")
            self.memory = None

    def set_llm_engine(self, llm_engine: Any) -> None:
        if llm_engine is not None:
            self.llm = llm_engine

    def set_llm(self, llm_engine: Any) -> None:
        self.set_llm_engine(llm_engine)

    def create_chain(
        self,
        problem: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
    ) -> ReasoningChain:
        chain = ReasoningChain(problem)
        chain.strategy = strategy

        chain_id = f"chain_{self.reasoning_count}"
        self.chains[chain_id] = chain
        self.reasoning_count += 1

        logger.debug(f"Created reasoning chain: {chain_id} ({strategy.value})")
        return chain

    # ------------------------
    # AI REASONING CORE
    # ------------------------
    async def _generate_reasoning(
        self, problem: str, strategy: ReasoningStrategy
    ) -> Dict[str, Any]:
        """Generate reasoning using LLM with timeout and retry"""

        if not self.llm:
            return {
                "steps": ["LLM not available, using fallback reasoning"],
                "conclusion": "Unable to perform AI reasoning",
                "confidence": 0.3,
            }

        prompt = f"""
You are EDIATH, an intelligent reasoning engine.

Your role is to solve problems clearly, logically, and efficiently.

Problem:
{problem}

Use {strategy.value} reasoning.

Instructions:
- Think step-by-step.
- Keep each step short and precise.
- Focus on clarity and correctness.
- Do not include unnecessary explanation.

Output Rules:
- Return ONLY valid JSON.
- Do NOT include any text outside the JSON.
- Do NOT use markdown.

JSON Format:
{{
  "steps": ["step 1", "step 2", "step 3"],
  "conclusion": "final answer",
  "confidence": 0.0
}}

Constraints:
- "steps": list of short reasoning steps
- "conclusion": one clear sentence
- "confidence": number between 0 and 1

IMPORTANT:
Return ONLY the JSON object. No extra text.
"""

        try:
            # Use timeout for LLM call
            if hasattr(self.llm, "safe_generate"):
                result = await asyncio.wait_for(
                    self.llm.safe_generate(prompt, timeout=10.0), timeout=12.0
                )

                if isinstance(result, dict):
                    raw = result.get("response", "")
                else:
                    raw = str(result)
            elif hasattr(self.llm, "generate"):
                result = await asyncio.wait_for(self.llm.generate(prompt), timeout=10.0)

                if isinstance(result, dict):
                    raw = result.get("response", "")
                else:
                    raw = str(result)
            else:
                logger.error("LLM has no generate method")
                return {
                    "steps": ["LLM method not available"],
                    "conclusion": "LLM interface error",
                    "confidence": 0.3,
                }

            # Ensure string
            if not isinstance(raw, str):
                raw = str(raw)

            raw = raw.strip()

            # Try to extract JSON
            data = self._extract_json(raw)

            if data and isinstance(data, dict):
                # Validate required fields
                if "steps" in data and "conclusion" in data:
                    return data
                else:
                    logger.warning(f"Missing required fields in JSON: {data}")
                    return {
                        "steps": ["Missing steps or conclusion in AI response"],
                        "conclusion": str(
                            data.get("conclusion", "Incomplete reasoning")
                        ),
                        "confidence": float(data.get("confidence", 0.4)),
                    }

            # Fallback if JSON extraction failed
            logger.warning(f"Failed to extract JSON from: {raw[:200]}")
            return {
                "steps": ["AI response parsing failed"],
                "conclusion": "Unable to parse reasoning from AI",
                "confidence": 0.3,
            }

        except asyncio.TimeoutError:
            logger.error(f"Reasoning timeout for problem: {problem[:100]}")
            return {
                "steps": ["Reasoning timeout occurred"],
                "conclusion": "Reasoning took too long, using fallback",
                "confidence": 0.4,
            }
        except Exception as e:
            logger.error(f"AI reasoning failed: {e}")
            return {
                "steps": ["Fallback reasoning applied due to error"],
                "conclusion": f"Unable to fully reason: {str(e)[:100]}",
                "confidence": 0.4,
            }

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text with multiple strategies"""

        # Strategy 1: Direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except:
            pass

        # Strategy 2: Find JSON objects with regex
        try:
            # Find all JSON-like objects
            matches = re.findall(r"\{[^{}]*\}", text)
            for match in reversed(matches):  # Try from the end (most likely valid)
                try:
                    data = json.loads(match)
                    if isinstance(data, dict) and "steps" in data:
                        return data
                except:
                    continue
        except:
            pass

        # Strategy 3: Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed = re.sub(r",\s*}", "}", text)
            fixed = re.sub(r",\s*]", "]", fixed)
            # Add quotes to unquoted keys
            fixed = re.sub(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', fixed)

            data = json.loads(fixed)
            if isinstance(data, dict):
                return data
        except:
            pass

        return None

    # ------------------------
    # MAIN REASONING METHOD
    # ------------------------
    async def reason(
        self,
        problem: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_steps: int = 10,
        timeout: float = 15.0,
    ) -> ReasoningChain:
        """Perform reasoning on a problem"""

        # Rate limiting
        import time

        now = time.time()
        if now - getattr(self, "_last_reasoning_time", 0) < 1.0:
            await asyncio.sleep(0.5)
        self._last_reasoning_time = time.time()

        chain = self.create_chain(problem, strategy)
        logger.info(f"Reasoning on problem: {problem[:100]}...")

        try:
            # AI reasoning with timeout
            result = await asyncio.wait_for(
                self._generate_reasoning(problem, strategy), timeout=timeout
            )

            steps = result.get("steps", [])[:max_steps]

            if not steps:
                # Add a default step if none provided
                steps = [f"Analyzing problem: {problem[:100]}"]

            for step_text in steps:
                if step_text and isinstance(step_text, str):
                    chain.add_step(step_text, confidence=0.7)

            conclusion = result.get("conclusion", "Analysis complete")
            confidence = float(result.get("confidence", 0.5))

            chain.set_conclusion(conclusion, confidence)

            # Store in memory if available
            if self.memory and hasattr(self.memory, "store"):
                try:
                    await self.memory.store(
                        {
                            "type": "reasoning",
                            "problem": problem,
                            "strategy": strategy.value,
                            "reasoning": chain.to_dict(),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to store reasoning in memory: {e}")

        except asyncio.TimeoutError:
            logger.error(f"Reasoning timeout for problem: {problem[:100]}")
            chain.add_step("Reasoning process timed out", confidence=0.3)
            chain.set_conclusion("Reasoning incomplete due to timeout", 0.3)

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            chain.add_step(f"Reasoning error: {str(e)[:100]}", confidence=0.3)
            chain.set_conclusion("Reasoning encountered an error", 0.3)

        return chain

    # ------------------------
    # SYNC WRAPPER
    # ------------------------
    def reason_sync(
        self,
        problem: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_steps: int = 10,
    ) -> ReasoningChain:
        """Synchronous wrapper for reason method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.reason(problem, strategy, max_steps))
        finally:
            loop.close()

    # ------------------------
    # VERIFICATION
    # ------------------------
    def verify_reasoning(self, chain: ReasoningChain) -> Tuple[bool, float]:
        if not chain.steps:
            return False, 0.0

        if not chain.conclusion:
            return False, 0.0

        avg_step_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)
        overall_confidence = (avg_step_confidence + chain.confidence) / 2

        is_valid = overall_confidence >= 0.6
        return is_valid, overall_confidence

    # ------------------------
    # UTILITY METHODS
    # ------------------------
    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        return self.chains.get(chain_id)

    def get_chain_by_index(self, index: int) -> Optional[ReasoningChain]:
        chain_id = f"chain_{index}"
        return self.chains.get(chain_id)

    def clear_chains(self) -> None:
        self.chains.clear()
        logger.info("All reasoning chains cleared")

    def get_stats(self) -> Dict[str, Any]:
        total_chains = len(self.chains)
        avg_steps = 0

        if total_chains > 0:
            avg_steps = sum(len(c.steps) for c in self.chains.values()) / total_chains

        return {
            "total_chains": total_chains,
            "total_reasonings": self.reasoning_count,
            "avg_steps_per_chain": round(avg_steps, 2),
            "llm_available": self.llm is not None,
            "memory_available": self.memory is not None,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of all reasoning chains"""
        if not self.chains:
            return "No reasoning chains available."

        summary = f"Reasoning Engine Summary\n{'='*40}\n"
        summary += f"Total Chains: {len(self.chains)}\n"
        summary += f"Total Reasonings: {self.reasoning_count}\n"
        summary += f"LLM Available: {self.llm is not None}\n"
        summary += f"Memory Available: {self.memory is not None}\n\n"

        for chain_id, chain in list(self.chains.items())[:5]:  # Show last 5
            summary += f"Chain: {chain_id}\n"
            summary += f"  Problem: {chain.problem[:80]}...\n"
            summary += f"  Steps: {len(chain.steps)}\n"
            summary += f"  Confidence: {chain.confidence:.2%}\n"
            summary += f"  Conclusion: {chain.conclusion[:80] if chain.conclusion else 'None'}...\n\n"

        return summary


__all__ = ["ReasoningEngine", "ReasoningChain", "ReasoningStrategy", "ReasoningStep"]
