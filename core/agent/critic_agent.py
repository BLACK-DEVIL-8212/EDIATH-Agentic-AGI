"""
🔥 FINAL PRODUCTION Critic Agent for EDIATH
✔ Multi-dimensional output evaluation
✔ Automated quality scoring (0-100)
✔ Error detection and classification
✔ Grammar and style checking (LanguageTool integration)
✔ Readability analysis (Flesch-Kincaid, Flesch Reading Ease)
✔ Safety and compliance validation
✔ Reinforcement learning feedback generation
✔ Batch evaluation and comparative analysis
✔ Historical tracking with persistence
✔ Confidence scoring
✔ Production ready
"""

import asyncio
import re
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, time, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import logging
import hashlib
import traceback

# NLP for evaluation
try:
    from textstat import textstat

    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import language_tool_python

    LANGTOOL_AVAILABLE = True
except ImportError:
    LANGTOOL_AVAILABLE = False


# =========================
# ENUMS AND CONSTANTS
# =========================


class CritiqueDimension(Enum):
    """Dimensions to critique"""

    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    GRAMMAR = "grammar"
    STYLE = "style"
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
    CONSISTENCY = "consistency"
    FORMATTING = "formatting"


class SeverityLevel(Enum):
    """Severity of critique"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EvaluationStatus(Enum):
    """Evaluation status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =========================
# DATACLASSES
# =========================


@dataclass
class Critique:
    """Individual critique item"""

    id: str
    dimension: CritiqueDimension
    issue: str
    suggestion: str
    severity: SeverityLevel
    confidence: float
    evidence: Optional[str] = None
    score: float = 0.0  # 0-10 scale
    location: Optional[str] = None  # Location in text (e.g., line number)
    category: str = "general"


@dataclass
class EvaluationResult:
    """Complete evaluation result"""

    id: str
    overall_score: float  # 0-100
    passed: bool
    critiques: List[Critique]
    summary: str
    dimensions_scores: Dict[str, float]
    execution_time: float
    timestamp: datetime
    improvement_plan: List[str]
    status: EvaluationStatus = EvaluationStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================
# MAIN AGENT
# =========================


class CriticAgent:
    """
    Advanced evaluation agent capable of:
    - Multi-dimensional output assessment
    - Automated quality scoring
    - Error detection and classification
    - Improvement suggestions
    - Grammar and style checking
    - Safety and compliance validation
    - Performance evaluation
    - Reinforcement learning feedback generation
    - Batch evaluation
    - Comparative analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Critic Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Evaluation configuration
        self.pass_threshold = self.config.get("pass_threshold", 70)
        self.enable_grammar_check = self.config.get("enable_grammar_check", True)
        self.enable_readability_check = self.config.get(
            "enable_readability_check", True
        )
        self.enable_safety_check = self.config.get("enable_safety_check", True)

        # Dimension weights (must sum to 1.0)
        self.dimension_weights = self.config.get(
            "dimension_weights",
            {
                "accuracy": 0.20,
                "completeness": 0.15,
                "clarity": 0.15,
                "relevance": 0.15,
                "coherence": 0.10,
                "grammar": 0.10,
                "safety": 0.10,
                "style": 0.05,
            },
        )

        # Validate weights sum to ~1.0
        weight_sum = sum(self.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            self.logger.warning(f"Dimension weights sum to {weight_sum}, normalizing")
            # Normalize weights
            for k in self.dimension_weights:
                self.dimension_weights[k] /= weight_sum

        # Initialize grammar tool
        self.grammar_tool = None
        if LANGTOOL_AVAILABLE and self.enable_grammar_check:
            try:
                self.grammar_tool = language_tool_python.LanguageTool("en-US")
                self.logger.info("Grammar tool initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize grammar tool: {e}")

        # Safety patterns
        self.safety_patterns = {
            "harmful_content": {
                "patterns": [
                    r"\bharm\b",
                    r"\bkill\b",
                    r"\bdestroy\b",
                    r"\battack\b",
                    r"\bviolence\b",
                    r"\billegal\b",
                    r"\bunauthorized\b",
                    r"\bexploit\b",
                    r"\bmalicious\b",
                    r"\bthreat\b",
                    r"\bdanger\b",
                    r"\babuse\b",
                    r"\bharass\b",
                ],
                "severity": SeverityLevel.CRITICAL,
                "weight": 0.3,
            },
            "sensitive_info": {
                "patterns": [
                    r"\bpassword\b",
                    r"\bcredential\b",
                    r"\bsecret\b",
                    r"api[_\s]*key\b",
                    r"\btoken\b",
                    r"\bauth\b",
                    r"\bprivate\b",
                    r"\bconfidential\b",
                    r"\bssn\b",
                    r"\bcredit[_\s]*card\b",
                ],
                "severity": SeverityLevel.CRITICAL,
                "weight": 0.3,
            },
            "misinformation": {
                "patterns": [
                    r"\bfake\b",
                    r"\bfalse\b",
                    r"\bmisleading\b",
                    r"\bincorrect\b",
                    r"\bnot true\b",
                    r"\bunverified\b",
                    r"\brumor\b",
                    r"\ballegedly\b",
                ],
                "severity": SeverityLevel.HIGH,
                "weight": 0.2,
            },
            "inappropriate": {
                "patterns": [
                    r"\boffensive\b",
                    r"\bprofanity\b",
                    r"\bhate\b",
                    r"\bdiscrimination\b",
                    r"\bsexist\b",
                    r"\bracist\b",
                    r"\bharassment\b",
                ],
                "severity": SeverityLevel.CRITICAL,
                "weight": 0.2,
            },
        }

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_score": 0.0,
            "pass_rate": 0.0,
            "by_dimension": {},
            "by_severity": {},
            "total_execution_time": 0.0,
            "start_time": datetime.now().isoformat(),
        }

        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        self.max_history = self.config.get("max_history", 1000)
        self.history_file = Path(
            self.config.get("history_file", "./data/evaluation_history.json")
        )
        self._load_history()

        # Cache for evaluation results
        self._cache: Dict[str, EvaluationResult] = {}
        self._cache_max_size = self.config.get("cache_max_size", 100)
        self._cache_ttl = self.config.get("cache_ttl_seconds", 3600)

        # Circuit breaker
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime] = None
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.circuit_breaker_timeout = self.config.get("circuit_breaker_timeout", 60)
        self._consecutive_failures = 0

        # Rate limiting
        self._request_timestamps: deque = deque(maxlen=100)
        self.max_requests_per_minute = self.config.get("max_requests_per_minute", 60)
        self._last_request_time = 0
        self.min_request_interval = self.config.get("min_request_interval", 0.1)

        self.logger.info("Critic Agent initialized")

    # =========================
    # CIRCUIT BREAKER
    # =========================

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open"""
        if not self._circuit_open:
            return True

        if self._circuit_open_until and datetime.now() >= self._circuit_open_until:
            self._circuit_open = False
            self._consecutive_failures = 0
            self.logger.info("Circuit breaker closed")
            return True

        return False

    async def _record_failure(self):
        """Record a failure for circuit breaker"""
        self._consecutive_failures += 1
        self.stats["failed_evaluations"] += 1

        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_open = True
            self._circuit_open_until = datetime.now() + timedelta(
                seconds=self.circuit_breaker_timeout
            )
            self.logger.error(
                f"Circuit breaker opened after {self._consecutive_failures} failures"
            )

    async def _record_success(self):
        """Record a successful evaluation"""
        self._consecutive_failures = 0
        self.stats["successful_evaluations"] += 1

    # =========================
    # RATE LIMITING
    # =========================

    async def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()

        if now - self._last_request_time < self.min_request_interval:
            await asyncio.sleep(
                self.min_request_interval - (now - self._last_request_time)
            )

        self._request_timestamps.append(now)
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            oldest = self._request_timestamps[0]
            if now - oldest < 60:
                wait_time = 60 - (now - oldest)
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self._last_request_time = time.time()
        return True

    # =========================
    # PERSISTENCE
    # =========================

    def _load_history(self):
        """Load evaluation history from file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[-self.max_history :]:
                    result = self._dict_to_result(item)
                    self.evaluation_history.append(result)
            self.logger.info(f"Loaded {len(self.evaluation_history)} history entries")
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save evaluation history to file"""
        try:
            data = []
            for result in self.evaluation_history[-self.max_history :]:
                data.append(self._result_to_dict(result))

            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save history: {e}")

    def _result_to_dict(self, result: EvaluationResult) -> Dict:
        """Convert EvaluationResult to dictionary"""
        return {
            "id": result.id,
            "overall_score": result.overall_score,
            "passed": result.passed,
            "critiques": [
                {
                    "id": c.id,
                    "dimension": c.dimension.value,
                    "issue": c.issue,
                    "suggestion": c.suggestion,
                    "severity": c.severity.value,
                    "confidence": c.confidence,
                    "score": c.score,
                    "category": c.category,
                }
                for c in result.critiques
            ],
            "summary": result.summary,
            "dimensions_scores": result.dimensions_scores,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp.isoformat(),
            "improvement_plan": result.improvement_plan,
            "status": result.status.value,
            "error": result.error,
        }

    def _dict_to_result(self, data: Dict) -> EvaluationResult:
        """Convert dictionary to EvaluationResult"""
        critiques = []
        for c in data.get("critiques", []):
            critiques.append(
                Critique(
                    id=c["id"],
                    dimension=CritiqueDimension(c["dimension"]),
                    issue=c["issue"],
                    suggestion=c["suggestion"],
                    severity=SeverityLevel(c["severity"]),
                    confidence=c["confidence"],
                    score=c.get("score", 0.0),
                    category=c.get("category", "general"),
                )
            )

        return EvaluationResult(
            id=data["id"],
            overall_score=data["overall_score"],
            passed=data["passed"],
            critiques=critiques,
            summary=data["summary"],
            dimensions_scores=data["dimensions_scores"],
            execution_time=data["execution_time"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            improvement_plan=data["improvement_plan"],
            status=EvaluationStatus(data.get("status", "completed")),
            error=data.get("error"),
        )

    def _get_cache_key(
        self, output: str, dimensions: List[str], reference: Optional[str]
    ) -> str:
        """Generate cache key for evaluation"""
        cache_data = {
            "output_hash": hashlib.md5(output.encode()).hexdigest(),
            "dimensions": sorted(dimensions),
            "reference_hash": (
                hashlib.md5(reference.encode()).hexdigest() if reference else None
            ),
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

    # =========================
    # EVALUATION METHODS
    # =========================

    async def evaluate(
        self,
        output: str,
        context: Optional[Dict] = None,
        dimensions: Optional[List[CritiqueDimension]] = None,
        reference: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate an output

        Args:
            output: Output text to evaluate
            context: Additional context (expected output, constraints)
            dimensions: Dimensions to evaluate
            reference: Reference/expected output for comparison
            use_cache: Use cached results

        Returns:
            Dictionary with evaluation results
        """
        evaluation_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Check circuit breaker
        if not await self._check_circuit_breaker():
            return {
                "success": False,
                "error": "Circuit breaker open - system temporarily unavailable",
                "evaluation_id": evaluation_id,
            }

        # Check rate limit
        if not await self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded - please try again later",
                "evaluation_id": evaluation_id,
            }

        if dimensions is None:
            dimensions = list(CritiqueDimension)

        # Check cache
        cache_key = None
        if use_cache:
            dim_names = [d.value for d in dimensions]
            cache_key = self._get_cache_key(output, dim_names, reference)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cache_age = (datetime.now() - cached.timestamp).total_seconds()
                if cache_age < self._cache_ttl:
                    self.logger.debug("Cache hit for evaluation")
                    return self._format_result(cached)

        try:
            # Evaluate each dimension
            critiques = []
            dimension_scores = {}

            for dimension in dimensions:
                if dimension == CritiqueDimension.ACCURACY:
                    score, crit = await self._evaluate_accuracy(
                        output, context, reference
                    )
                elif dimension == CritiqueDimension.COMPLETENESS:
                    score, crit = await self._evaluate_completeness(output, context)
                elif dimension == CritiqueDimension.CLARITY:
                    score, crit = await self._evaluate_clarity(output)
                elif dimension == CritiqueDimension.RELEVANCE:
                    score, crit = await self._evaluate_relevance(output, context)
                elif dimension == CritiqueDimension.COHERENCE:
                    score, crit = await self._evaluate_coherence(output)
                elif dimension == CritiqueDimension.GRAMMAR:
                    score, crit = await self._evaluate_grammar(output)
                elif dimension == CritiqueDimension.STYLE:
                    score, crit = await self._evaluate_style(output)
                elif dimension == CritiqueDimension.SAFETY and self.enable_safety_check:
                    score, crit = await self._evaluate_safety(output)
                elif dimension == CritiqueDimension.EFFICIENCY:
                    score, crit = await self._evaluate_efficiency(output, context)
                elif dimension == CritiqueDimension.CREATIVITY:
                    score, crit = await self._evaluate_creativity(output)
                elif dimension == CritiqueDimension.CONSISTENCY:
                    score, crit = await self._evaluate_consistency(output, reference)
                elif dimension == CritiqueDimension.FORMATTING:
                    score, crit = await self._evaluate_formatting(output)
                else:
                    score, crit = 5.0, []

                dimension_scores[dimension.value] = score
                if crit:
                    critiques.extend(crit)

            # Deduplicate critiques
            critiques = self._deduplicate_critiques(critiques)

            # Calculate weighted overall score
            overall_score = self._calculate_weighted_score(dimension_scores)

            # Generate improvement plan
            improvement_plan = self._generate_improvement_plan(critiques)

            # Generate summary
            summary = self._generate_summary(overall_score, critiques)

            # Determine if passed
            passed = overall_score >= self.pass_threshold

            # Create evaluation result
            result = EvaluationResult(
                id=evaluation_id,
                overall_score=overall_score,
                passed=passed,
                critiques=critiques,
                summary=summary,
                dimensions_scores=dimension_scores,
                execution_time=(time.time() - start_time),
                timestamp=datetime.now(),
                improvement_plan=improvement_plan,
            )

            # Update statistics
            self._update_stats(result)

            # Add to history
            self._add_to_history(result)
            await self._record_success()

            # Update cache
            if use_cache and cache_key and len(self._cache) < self._cache_max_size:
                self._cache[cache_key] = result

            return self._format_result(result)

        except Exception as e:
            self.logger.error(f"Evaluation error: {e}\n{traceback.format_exc()}")
            await self._record_failure()
            return {
                "success": False,
                "error": str(e),
                "evaluation_id": evaluation_id,
                "output": output[:200],
            }

    def _deduplicate_critiques(self, critiques: List[Critique]) -> List[Critique]:
        """Remove duplicate critiques"""
        seen = set()
        unique = []
        for c in critiques:
            key = f"{c.dimension.value}_{c.issue[:50]}"
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    # =========================
    # DIMENSION EVALUATION METHODS
    # =========================

    async def _evaluate_accuracy(
        self, output: str, context: Optional[Dict], reference: Optional[str]
    ) -> Tuple[float, List[Critique]]:
        """Evaluate factual accuracy"""
        critiques = []
        score = 8.0

        if reference:
            similarity = self._calculate_similarity(output, reference)
            if similarity < 0.5:
                score -= 4
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.ACCURACY,
                        issue="Output significantly differs from expected reference",
                        suggestion="Review the expected output and ensure factual alignment",
                        severity=SeverityLevel.HIGH,
                        confidence=0.85,
                        score=similarity * 10,
                    )
                )
            elif similarity < 0.8:
                score -= 2
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.ACCURACY,
                        issue="Output partially matches expected reference",
                        suggestion="Verify key facts and figures against reference",
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.75,
                        score=similarity * 10,
                    )
                )
        else:
            # Check for uncertainty markers
            uncertainty_markers = [
                "maybe",
                "perhaps",
                "possibly",
                "might",
                "could be",
                "not sure",
                "uncertain",
                "unclear",
                "approximately",
            ]

            uncertainty_count = sum(
                1 for m in uncertainty_markers if m in output.lower()
            )
            if uncertainty_count > 3:
                score -= 1.5
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.ACCURACY,
                        issue=f"Contains {uncertainty_count} uncertain statements",
                        suggestion="Provide more definitive and confident answers when possible",
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.7,
                        score=score,
                    )
                )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_completeness(
        self, output: str, context: Optional[Dict]
    ) -> Tuple[float, List[Critique]]:
        """Evaluate completeness of response"""
        critiques = []
        score = 8.0

        word_count = len(output.split())

        if word_count < 20:
            score -= 3
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.COMPLETENESS,
                    issue="Response is too short",
                    suggestion="Provide more detailed and comprehensive information",
                    severity=SeverityLevel.HIGH,
                    confidence=0.9,
                    score=score,
                )
            )
        elif word_count < 50:
            score -= 1
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.COMPLETENESS,
                    issue="Response could be more detailed",
                    suggestion="Expand on key points with additional context",
                    severity=SeverityLevel.LOW,
                    confidence=0.7,
                    score=score,
                )
            )

        # Check for missing sections
        if context and "expected_sections" in context:
            expected_sections = context["expected_sections"]
            for section in expected_sections:
                if section.lower() not in output.lower():
                    score -= 1.5
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.COMPLETENESS,
                            issue=f"Missing expected section: {section}",
                            suggestion=f"Add a section about {section} to improve completeness",
                            severity=SeverityLevel.MEDIUM,
                            confidence=0.8,
                            score=score,
                        )
                    )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_clarity(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate clarity and readability"""
        critiques = []
        score = 8.0

        if TEXTSTAT_AVAILABLE and self.enable_readability_check:
            try:
                flesch_score = textstat.flesch_reading_ease(output)
                grade_level = textstat.flesch_kincaid_grade(output)

                if flesch_score < 30:
                    score -= 3
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.CLARITY,
                            issue=f"Very difficult to read (Flesch score: {flesch_score:.0f})",
                            suggestion="Use shorter sentences and simpler words",
                            severity=SeverityLevel.HIGH,
                            confidence=0.85,
                            score=flesch_score / 10,
                        )
                    )
                elif flesch_score < 50:
                    score -= 1.5
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.CLARITY,
                            issue=f"Difficult to read (Flesch score: {flesch_score:.0f})",
                            suggestion="Simplify complex sentences",
                            severity=SeverityLevel.MEDIUM,
                            confidence=0.7,
                            score=flesch_score / 10,
                        )
                    )

                if grade_level > 12:
                    score -= 1
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.CLARITY,
                            issue=f"Reading level too high (Grade {grade_level:.1f})",
                            suggestion="Aim for 8th-10th grade reading level",
                            severity=SeverityLevel.MEDIUM,
                            confidence=0.7,
                            score=max(0, 10 - grade_level),
                        )
                    )
            except Exception as e:
                self.logger.debug(f"Readability check failed: {e}")

        # Check sentence length
        sentences = re.split(r"[.!?]+", output)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            score -= min(2, len(long_sentences) * 0.5)
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.CLARITY,
                    issue=f"Contains {len(long_sentences)} very long sentences",
                    suggestion="Break long sentences into shorter ones",
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.8,
                    score=score,
                )
            )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_relevance(
        self, output: str, context: Optional[Dict]
    ) -> Tuple[float, List[Critique]]:
        """Evaluate relevance to the query/context"""
        critiques = []
        score = 8.0

        if context and "query" in context:
            query = context["query"]
            query_words = set(query.lower().split())
            output_words = set(output.lower().split())

            overlap = len(query_words & output_words)
            total = len(query_words)

            relevance_ratio = overlap / total if total > 0 else 0

            if relevance_ratio < 0.3:
                score -= 4
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.RELEVANCE,
                        issue=f"Low relevance to query (only {overlap}/{total} key terms match)",
                        suggestion="Focus response on directly addressing the query",
                        severity=SeverityLevel.HIGH,
                        confidence=0.85,
                        score=relevance_ratio * 10,
                    )
                )
            elif relevance_ratio < 0.6:
                score -= 1.5
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.RELEVANCE,
                        issue="Partially relevant to query",
                        suggestion="Ensure all parts of the response address the main question",
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.7,
                        score=relevance_ratio * 10,
                    )
                )

        # Check for off-topic content
        if context and "topic" in context:
            topic = context["topic"].lower()
            if topic and topic not in output.lower():
                score -= 2
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.RELEVANCE,
                        issue="Response may be off-topic",
                        suggestion=f"Ensure response stays focused on the topic: {topic}",
                        severity=SeverityLevel.HIGH,
                        confidence=0.6,
                        score=score,
                    )
                )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_coherence(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate logical flow and coherence"""
        critiques = []
        score = 8.0

        # Check for transition words
        transitions = [
            "however",
            "therefore",
            "consequently",
            "furthermore",
            "moreover",
            "nevertheless",
            "accordingly",
            "thus",
            "first",
            "second",
            "third",
            "finally",
            "additionally",
            "in addition",
            "on the other hand",
            "as a result",
        ]

        sentences = [s for s in re.split(r"[.!?]+", output) if s.strip()]
        transition_count = sum(1 for t in transitions if t in output.lower())

        if len(sentences) > 3 and transition_count == 0:
            score -= 2
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.COHERENCE,
                    issue="Lacks transition words between ideas",
                    suggestion="Use transition words to improve logical flow",
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.75,
                    score=score,
                )
            )
        elif len(sentences) > 5 and transition_count < 2:
            score -= 1
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.COHERENCE,
                    issue="Few transition words",
                    suggestion="Add more transition words to connect ideas",
                    severity=SeverityLevel.LOW,
                    confidence=0.65,
                    score=score,
                )
            )

        # Check for logical contradictions
        contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("always", "never"),
            ("increase", "decrease"),
            ("high", "low"),
            ("good", "bad"),
            ("positive", "negative"),
            ("success", "failure"),
        ]

        for a, b in contradiction_pairs:
            if a in output.lower() and b in output.lower():
                # Check if they appear in the same context
                sentences_with_a = [s for s in sentences if a in s.lower()]
                sentences_with_b = [s for s in sentences if b in s.lower()]
                if sentences_with_a and sentences_with_b:
                    score -= 1.5
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.COHERENCE,
                            issue=f"Potential contradiction: contains both '{a}' and '{b}'",
                            suggestion="Review for logical consistency",
                            severity=SeverityLevel.MEDIUM,
                            confidence=0.6,
                            score=score,
                        )
                    )
                    break

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_grammar(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate grammar and spelling"""
        critiques = []
        score = 9.0

        if self.grammar_tool and self.enable_grammar_check:
            try:
                matches = self.grammar_tool.check(output)

                # Categorize by severity
                error_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for match in matches:
                    if match.category == "GRAMMAR":
                        error_counts["HIGH"] += 1
                        score -= 0.5
                    elif match.category == "TYPOS":
                        error_counts["MEDIUM"] += 1
                        score -= 0.3
                    elif match.category == "PUNCTUATION":
                        error_counts["LOW"] += 1
                        score -= 0.1
                    else:
                        error_counts["LOW"] += 1
                        score -= 0.1

                if matches:
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.GRAMMAR,
                            issue=f"Found {len(matches)} grammar/spelling issues",
                            suggestion="Review grammar suggestions: "
                            + ", ".join([m.message[:60] for m in matches[:3]]),
                            severity=SeverityLevel.MEDIUM,
                            confidence=0.9,
                            score=score,
                        )
                    )
            except Exception as e:
                self.logger.warning(f"Grammar check failed: {e}")

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_style(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate writing style"""
        critiques = []
        score = 8.0

        # Check for passive voice
        passive_patterns = [
            r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b",
            r"\b(?:is|are|was|were)\s+\w+en\b",
            r"\b(?:has|have|had)\s+been\s+\w+ed\b",
        ]

        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, output.lower()))

        if passive_count > 4:
            score -= 2
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.STYLE,
                    issue=f"Overuse of passive voice ({passive_count} instances)",
                    suggestion="Use active voice for clearer, more direct writing",
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.75,
                    score=score,
                )
            )
        elif passive_count > 2:
            score -= 1
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.STYLE,
                    issue=f"Consider reducing passive voice ({passive_count} instances)",
                    suggestion="Active voice often improves clarity",
                    severity=SeverityLevel.LOW,
                    confidence=0.65,
                    score=score,
                )
            )

        # Check for repetitive language
        words = output.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        repetitive = [(w, c) for w, c in word_freq.items() if c > 5]
        if repetitive:
            score -= 1
            top_repetitive = repetitive[:3]
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.STYLE,
                    issue=f"Repetitive language: {', '.join([w for w, _ in top_repetitive])}",
                    suggestion="Use synonyms to avoid repetition",
                    severity=SeverityLevel.LOW,
                    confidence=0.7,
                    score=score,
                )
            )

        # Check for filler words
        filler_words = [
            "very",
            "really",
            "quite",
            "somewhat",
            "basically",
            "actually",
            "literally",
        ]
        filler_count = sum(1 for f in filler_words if f in output.lower())
        if filler_count > 3:
            score -= 0.5
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.STYLE,
                    issue=f"Contains {filler_count} filler words",
                    suggestion="Remove unnecessary filler words for more concise writing",
                    severity=SeverityLevel.INFO,
                    confidence=0.6,
                    score=score,
                )
            )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_safety(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate safety and appropriateness"""
        critiques = []
        score = 10.0
        output_lower = output.lower()

        for category, category_info in self.safety_patterns.items():
            for pattern in category_info["patterns"]:
                matches = re.findall(pattern, output_lower)
                if matches:
                    score -= 3 * category_info["weight"]
                    severity = category_info["severity"]
                    critiques.append(
                        Critique(
                            id=str(uuid.uuid4())[:8],
                            dimension=CritiqueDimension.SAFETY,
                            issue=f"Potential {category} detected: '{matches[0]}'",
                            suggestion="Remove or rephrase potentially harmful content",
                            severity=severity,
                            confidence=0.85,
                            score=score,
                            category=category,
                        )
                    )
                    break  # Only add one per category

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_efficiency(
        self, output: str, context: Optional[Dict]
    ) -> Tuple[float, List[Critique]]:
        """Evaluate efficiency and conciseness"""
        critiques = []
        score = 8.0

        word_count = len(output.split())

        # Check for verbosity
        if context and "expected_length" in context:
            expected = context["expected_length"]
            ratio = word_count / expected if expected > 0 else 1.0

            if ratio > 2.0:
                score -= 2
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.EFFICIENCY,
                        issue=f"Response is {word_count} words, expected ~{expected}",
                        suggestion="Be more concise and focus on key points",
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.8,
                        score=score,
                    )
                )
            elif ratio > 1.5:
                score -= 0.5
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.EFFICIENCY,
                        issue=f"Response slightly longer than expected ({word_count} vs {expected} words)",
                        suggestion="Consider tightening the response for better efficiency",
                        severity=SeverityLevel.LOW,
                        confidence=0.6,
                        score=score,
                    )
                )

        # Check for redundant phrases
        redundant_phrases = [
            "in order to",
            "due to the fact that",
            "at this point in time",
            "in the event that",
            "for the purpose of",
            "with the exception of",
            "as a matter of fact",
            "in spite of the fact that",
        ]

        redundancy_count = sum(1 for p in redundant_phrases if p in output.lower())
        if redundancy_count > 0:
            score -= min(1, redundancy_count * 0.5)
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.EFFICIENCY,
                    issue=f"Contains {redundancy_count} redundant phrases",
                    suggestion="Use simpler, more direct language (e.g., 'to' instead of 'in order to')",
                    severity=SeverityLevel.LOW,
                    confidence=0.7,
                    score=score,
                )
            )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_creativity(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate creativity and originality"""
        critiques = []
        score = 7.0

        # Check for unique vocabulary
        words = set(output.lower().split())
        unique_ratio = len(words) / max(1, len(output.split()))

        if unique_ratio > 0.7:
            score += 1.5
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.CREATIVITY,
                    issue="High vocabulary diversity detected",
                    suggestion="Continue using varied vocabulary",
                    severity=SeverityLevel.INFO,
                    confidence=0.7,
                    score=score,
                )
            )
        elif unique_ratio < 0.3 and len(output.split()) > 50:
            score -= 0.5
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.CREATIVITY,
                    issue="Limited vocabulary diversity",
                    suggestion="Use more varied word choices to enhance engagement",
                    severity=SeverityLevel.LOW,
                    confidence=0.6,
                    score=score,
                )
            )

        # Check for creative expressions
        creative_markers = [
            "imagine",
            "picture",
            "consider",
            "what if",
            "suppose",
            "alternatively",
            "interestingly",
            "notably",
            "surprisingly",
            "creatively",
            "innovatively",
            "unique",
            "novel",
        ]

        creative_count = sum(1 for m in creative_markers if m in output.lower())
        if creative_count > 3:
            score += 0.5
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.CREATIVITY,
                    issue="Good use of creative expressions",
                    suggestion="Continue using engaging language",
                    severity=SeverityLevel.INFO,
                    confidence=0.6,
                    score=score,
                )
            )

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_consistency(
        self, output: str, reference: Optional[str]
    ) -> Tuple[float, List[Critique]]:
        """Evaluate internal consistency"""
        critiques = []
        score = 8.0

        if reference:
            # Check for contradictions between output and reference
            output_lower = output.lower()
            ref_lower = reference.lower()

            # Simple inconsistency detection
            if ("yes" in output_lower and "no" in ref_lower) or (
                "true" in output_lower and "false" in ref_lower
            ):
                score -= 2
                critiques.append(
                    Critique(
                        id=str(uuid.uuid4())[:8],
                        dimension=CritiqueDimension.CONSISTENCY,
                        issue="Output contradicts reference information",
                        suggestion="Ensure consistency with provided reference",
                        severity=SeverityLevel.HIGH,
                        confidence=0.8,
                        score=score,
                    )
                )

        # Check for internal contradictions
        sentences = re.split(r"[.!?]+", output)
        for i, s1 in enumerate(sentences):
            for s2 in sentences[i + 1 :]:
                if len(s1) > 20 and len(s2) > 20:
                    # Simple contradiction detection
                    if ("not" in s1 and "not" not in s2) or (
                        "never" in s1 and "always" in s2
                    ):
                        score -= 0.5
                        break

        return max(0.0, min(10.0, score)), critiques

    async def _evaluate_formatting(self, output: str) -> Tuple[float, List[Critique]]:
        """Evaluate text formatting"""
        critiques = []
        score = 9.0

        # Check for proper capitalization
        sentences = re.split(r"[.!?]+", output)
        uncapitalized = [
            s for s in sentences if s.strip() and not s.strip()[0].isupper()
        ]
        if uncapitalized:
            score -= min(1, len(uncapitalized) * 0.3)
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.FORMATTING,
                    issue=f"{len(uncapitalized)} sentence(s) start without capitalization",
                    suggestion="Capitalize the first letter of each sentence",
                    severity=SeverityLevel.LOW,
                    confidence=0.8,
                    score=score,
                )
            )

        # Check for excessive punctuation
        if "!!!" in output or "???" in output:
            score -= 0.5
            critiques.append(
                Critique(
                    id=str(uuid.uuid4())[:8],
                    dimension=CritiqueDimension.FORMATTING,
                    issue="Excessive punctuation detected",
                    suggestion="Use single punctuation marks for professional writing",
                    severity=SeverityLevel.LOW,
                    confidence=0.9,
                    score=score,
                )
            )

        return max(0.0, min(10.0, score)), critiques

    # =========================
    # UTILITY METHODS
    # =========================

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_weighted_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score (0-100)"""
        total = 0.0
        weight_sum = 0.0

        for dimension, score in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 0.1)
            total += score * weight * 10  # Scale to 0-100
            weight_sum += weight

        if weight_sum > 0:
            return total / weight_sum
        return 0.0

    def _generate_improvement_plan(self, critiques: List[Critique]) -> List[str]:
        """Generate improvement plan from critiques"""
        plan = []

        # Group by severity
        critical = [c for c in critiques if c.severity == SeverityLevel.CRITICAL]
        high = [c for c in critiques if c.severity == SeverityLevel.HIGH]
        medium = [c for c in critiques if c.severity == SeverityLevel.MEDIUM]

        # Prioritize critical issues
        for c in critical[:3]:
            plan.append(f"[CRITICAL] {c.suggestion}")

        for c in high[:3]:
            plan.append(f"[HIGH] {c.suggestion}")

        for c in medium[:3]:
            plan.append(f"[MEDIUM] {c.suggestion}")

        if not plan:
            plan.append(
                "No major improvements needed - the output meets quality standards"
            )

        return plan

    def _generate_summary(self, overall_score: float, critiques: List[Critique]) -> str:
        """Generate evaluation summary"""
        if overall_score >= 85:
            return "Excellent quality. The output meets high standards with minor improvements possible."
        elif overall_score >= 70:
            return f"Good quality (Score: {overall_score:.1f}). Several areas need improvement for better results."
        elif overall_score >= 50:
            return f"Adequate quality (Score: {overall_score:.1f}). Significant improvements recommended to meet standards."
        else:
            return f"Poor quality (Score: {overall_score:.1f}). Major revisions needed to address critical issues."

    def _update_stats(self, result: EvaluationResult):
        """Update evaluation statistics"""
        self.stats["total_evaluations"] += 1

        # Update average score
        n = self.stats["total_evaluations"]
        self.stats["average_score"] = (
            self.stats["average_score"] * (n - 1) + result.overall_score
        ) / n

        # Update pass rate
        passes = sum(1 for r in self.evaluation_history if r.passed) + (
            1 if result.passed else 0
        )
        self.stats["pass_rate"] = (passes / n) * 100

        # Update dimension stats
        for dim, score in result.dimensions_scores.items():
            if dim not in self.stats["by_dimension"]:
                self.stats["by_dimension"][dim] = {"total": 0, "sum": 0}
            self.stats["by_dimension"][dim]["total"] += 1
            self.stats["by_dimension"][dim]["sum"] += score

        # Update severity stats
        for critique in result.critiques:
            severity = critique.severity.value
            self.stats["by_severity"][severity] = (
                self.stats["by_severity"].get(severity, 0) + 1
            )

        # Update execution time
        self.stats["total_execution_time"] += result.execution_time

    def _add_to_history(self, result: EvaluationResult):
        """Add to evaluation history"""
        self.evaluation_history.append(result)
        if len(self.evaluation_history) > self.max_history:
            self.evaluation_history = self.evaluation_history[-self.max_history :]
        self._save_history()

    def _format_result(self, result: EvaluationResult) -> Dict[str, Any]:
        """Format evaluation result for output"""
        return {
            "success": True,
            "evaluation_id": result.id,
            "overall_score": round(result.overall_score, 1),
            "passed": result.passed,
            "summary": result.summary,
            "dimensions_scores": {
                k: round(v, 1) for k, v in result.dimensions_scores.items()
            },
            "critiques": [
                {
                    "id": c.id,
                    "dimension": c.dimension.value,
                    "issue": c.issue,
                    "suggestion": c.suggestion,
                    "severity": c.severity.value,
                    "confidence": round(c.confidence, 2),
                    "score": round(c.score, 1),
                    "category": c.category,
                }
                for c in result.critiques
            ],
            "improvement_plan": result.improvement_plan,
            "execution_time": round(result.execution_time, 3),
            "timestamp": result.timestamp.isoformat(),
        }

    # =========================
    # BATCH EVALUATION
    # =========================

    async def evaluate_batch(
        self, outputs: List[str], contexts: Optional[List[Dict]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate multiple outputs in batch

        Args:
            outputs: List of outputs to evaluate
            contexts: List of contexts for each output
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary with batch results
        """
        if contexts is None:
            contexts = [{}] * len(outputs)

        tasks = []
        for output, context in zip(outputs, contexts):
            task = self.evaluate(output, context, **kwargs)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluations = []
        successful = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                evaluations.append({"index": i, "success": False, "error": str(result)})
            else:
                successful += 1
                evaluations.append(result)

        avg_score = sum(
            r.get("overall_score", 0) for r in evaluations if r.get("success")
        ) / max(1, successful)
        pass_count = sum(1 for r in evaluations if r.get("passed", False))

        return {
            "success": True,
            "batch_id": str(uuid.uuid4())[:8],
            "total": len(outputs),
            "successful": successful,
            "failed": len(outputs) - successful,
            "evaluations": evaluations,
            "average_score": round(avg_score, 1),
            "pass_rate": round((pass_count / len(outputs)) * 100, 1) if outputs else 0,
        }

    # =========================
    # COMPARATIVE EVALUATION
    # =========================

    async def compare_outputs(
        self, output1: str, output2: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Compare two outputs and determine which is better

        Args:
            output1: First output
            output2: Second output
            context: Evaluation context

        Returns:
            Dictionary with comparison results
        """
        eval1 = await self.evaluate(output1, context)
        eval2 = await self.evaluate(output2, context)

        score1 = eval1.get("overall_score", 0)
        score2 = eval2.get("overall_score", 0)

        if score1 > score2:
            winner = "output1"
            winner_score = score1
            loser_score = score2
            difference = score1 - score2
        elif score2 > score1:
            winner = "output2"
            winner_score = score2
            loser_score = score1
            difference = score2 - score1
        else:
            winner = "tie"
            winner_score = score1
            loser_score = score2
            difference = 0

        # Generate comparison insights
        insights = []
        dims1 = eval1.get("dimensions_scores", {})
        dims2 = eval2.get("dimensions_scores", {})

        for dim in set(dims1.keys()) | set(dims2.keys()):
            diff = dims1.get(dim, 0) - dims2.get(dim, 0)
            if abs(diff) > 1.0:
                better = "output1" if diff > 0 else "output2"
                insights.append(f"{dim}: {better} is better by {abs(diff):.1f} points")

        return {
            "success": True,
            "comparison_id": str(uuid.uuid4())[:8],
            "output1_score": round(score1, 1),
            "output2_score": round(score2, 1),
            "winner": winner,
            "difference": round(difference, 1),
            "winner_score": round(winner_score, 1),
            "loser_score": round(loser_score, 1),
            "insights": insights[:5],
            "output1_critiques": eval1.get("critiques", [])[:3],
            "output2_critiques": eval2.get("critiques", [])[:3],
            "recommendation": (
                f"Output {winner} is better by {difference:.1f} points"
                if winner != "tie"
                else "Both outputs are equally good"
            ),
        }

    # =========================
    # RL FEEDBACK GENERATION
    # =========================

    def generate_rl_feedback(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        """
        Generate reinforcement learning feedback from evaluation

        Args:
            evaluation: Evaluation result

        Returns:
            Dictionary with RL feedback
        """
        # Calculate reward (0-1 scale)
        reward = evaluation.overall_score / 100.0

        # Generate advantage scores per dimension (-1 to 1)
        advantages = {}
        for dimension, score in evaluation.dimensions_scores.items():
            # Convert 0-10 scale to -1 to 1 advantage
            advantages[dimension] = (score - 5) / 5.0

        # Identify key improvement areas
        improvement_areas = []
        for critique in evaluation.critiques:
            if critique.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                improvement_areas.append(
                    {
                        "dimension": critique.dimension.value,
                        "issue": critique.issue,
                        "suggestion": critique.suggestion,
                        "severity": critique.severity.value,
                        "confidence": critique.confidence,
                    }
                )

        return {
            "reward": reward,
            "advantages": advantages,
            "improvement_areas": improvement_areas,
            "passed": evaluation.passed,
            "overall_score": evaluation.overall_score,
            "timestamp": datetime.now().isoformat(),
            "evaluation_id": evaluation.id,
        }

    # =========================
    # HISTORY AND STATISTICS
    # =========================

    def get_history(
        self,
        limit: int = None,
        passed_only: bool = False,
        min_score: Optional[float] = None,
    ) -> List[Dict]:
        """Get evaluation history"""
        history = self.evaluation_history

        if passed_only:
            history = [h for h in history if h.passed]

        if min_score is not None:
            history = [h for h in history if h.overall_score >= min_score]

        if limit:
            history = history[-limit:]

        return [
            {
                "id": h.id,
                "overall_score": round(h.overall_score, 1),
                "passed": h.passed,
                "summary": h.summary,
                "critiques_count": len(h.critiques),
                "execution_time": round(h.execution_time, 3),
                "timestamp": h.timestamp.isoformat(),
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "history_size": len(self.evaluation_history),
            "dimension_weights": self.dimension_weights,
            "pass_threshold": self.pass_threshold,
            "grammar_check_enabled": self.enable_grammar_check
            and self.grammar_tool is not None,
            "readability_check_enabled": self.enable_readability_check
            and TEXTSTAT_AVAILABLE,
            "safety_check_enabled": self.enable_safety_check,
            "cache_size": len(self._cache),
            "circuit_breaker_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
        }

    def clear_history(self) -> Dict[str, Any]:
        """Clear evaluation history"""
        count = len(self.evaluation_history)
        self.evaluation_history.clear()
        self._save_history()
        self.logger.info(f"Evaluation history cleared ({count} entries)")
        return {
            "success": True,
            "cleared": count,
            "message": f"Cleared {count} history entries",
        }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear evaluation cache"""
        cache_size = len(self._cache)
        self._cache.clear()
        return {
            "success": True,
            "cleared": cache_size,
            "message": f"Cleared {cache_size} cache entries",
        }


# =========================
# INTEGRATION WRAPPER
# =========================


class CriticAgentWrapper:
    """
    Wrapper class to integrate CriticAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.critic_agent = CriticAgent(config)
        self.agent_type = "critic"
        self.capabilities = [
            "evaluate_output",
            "compare_outputs",
            "batch_evaluation",
            "generate_feedback",
            "quality_assessment",
            "grammar_check",
            "readability_analysis",
            "safety_check",
        ]
        self._initialized = True

    async def initialize(self, *args, **kwargs) -> bool:
        """Initialize the wrapper"""
        return True

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a critique request

        Request format:
        {
            'operation': 'evaluate|compare|batch|history|stats|clear_history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        try:
            if operation == "evaluate":
                dimensions = request.get("dimensions")
                if dimensions:
                    dimensions = [CritiqueDimension(d) for d in dimensions]

                return await self.critic_agent.evaluate(
                    output=request.get("output", ""),
                    context=request.get("context"),
                    dimensions=dimensions,
                    reference=request.get("reference"),
                    use_cache=request.get("use_cache", True),
                )

            elif operation == "compare":
                return await self.critic_agent.compare_outputs(
                    output1=request.get("output1", ""),
                    output2=request.get("output2", ""),
                    context=request.get("context"),
                )

            elif operation == "batch":
                return await self.critic_agent.evaluate_batch(
                    outputs=request.get("outputs", []),
                    contexts=request.get("contexts"),
                    dimensions=request.get("dimensions"),
                )

            elif operation == "history":
                return {
                    "success": True,
                    "history": self.critic_agent.get_history(
                        limit=request.get("limit"),
                        passed_only=request.get("passed_only", False),
                        min_score=request.get("min_score"),
                    ),
                }

            elif operation == "stats":
                return self.critic_agent.get_stats()

            elif operation == "clear_history":
                return self.critic_agent.clear_history()

            elif operation == "clear_cache":
                return self.critic_agent.clear_cache()

            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}

        except Exception as e:
            self.critic_agent.logger.error(f"Request error: {e}")
            return {"success": False, "error": str(e), "operation": operation}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "CriticAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.critic_agent.get_stats(),
            "dimensions": [d.value for d in CritiqueDimension],
            "severity_levels": [s.value for s in SeverityLevel],
            "version": "2.0.0",
        }

    async def close(self):
        """Clean up resources"""
        if self.critic_agent.grammar_tool:
            try:
                self.critic_agent.grammar_tool.close()
            except:
                pass
        self._initialized = False


# =========================
# TESTING
# =========================


async def test_critic_agent():
    """Test the critic agent functionality"""

    # Initialize agent
    agent = CriticAgent()

    print("=== Critic Agent Test ===\n")

    # Test good output
    print("1. Evaluating Good Output...")
    good_output = """
    The EDIATH AI assistant is a powerful tool for automating tasks. 
    It can execute code, manage files, search the web, and perform many other functions. 
    The system uses advanced machine learning models to understand natural language and generate appropriate responses.
    """

    result = await agent.evaluate(good_output)
    if result["success"]:
        print(f"   Score: {result['overall_score']:.1f}")
        print(f"   Passed: {result['passed']}")
        print(f"   Summary: {result['summary']}")
        print(f"   Dimensions: {result['dimensions_scores']}")
        print(f"   Critiques: {len(result['critiques'])}")

    # Test poor output
    print("\n2. Evaluating Poor Output...")
    poor_output = """
    maybe the thing does stuff. it might work. i think it's good.
    not sure. could be better. maybe later. the system might be helpful.
    """

    result = await agent.evaluate(poor_output)
    if result["success"]:
        print(f"   Score: {result['overall_score']:.1f}")
        print(f"   Passed: {result['passed']}")
        print(f"   Critiques found: {len(result['critiques'])}")
        for crit in result["critiques"][:3]:
            print(
                f"     - [{crit['severity']}] {crit['dimension']}: {crit['issue'][:50]}..."
            )
        print(f"   Improvement plan: {result['improvement_plan'][0]}")

    # Test with reference
    print("\n3. Evaluating with Reference...")
    reference = "EDIATH can execute Python code safely in a sandboxed environment."
    result = await agent.evaluate(
        "EDIATH runs code securely in an isolated environment.",
        context={"query": "What can EDIATH do?"},
        reference=reference,
    )
    if result["success"]:
        print(f"   Score: {result['overall_score']:.1f}")
        print(
            f"   Accuracy score: {result['dimensions_scores'].get('accuracy', 0):.1f}"
        )

    # Test safety detection
    print("\n4. Safety Detection...")
    unsafe_output = (
        "This is a test with harmful content like violence and illegal activities."
    )
    result = await agent.evaluate(unsafe_output)
    if result["success"]:
        print(f"   Score: {result['overall_score']:.1f}")
        safety_critiques = [
            c for c in result["critiques"] if c["dimension"] == "safety"
        ]
        print(f"   Safety issues detected: {len(safety_critiques)}")
        for crit in safety_critiques:
            print(f"     - {crit['issue'][:60]}...")

    # Test comparison
    print("\n5. Comparing Two Outputs...")
    output1 = "EDIATH is an AI assistant that helps with tasks and automation."
    output2 = "The system provides comprehensive automation capabilities including code execution, file management, web searching, and voice interaction."

    result = await agent.compare_outputs(output1, output2)
    if result["success"]:
        print(f"   Output1 score: {result['output1_score']:.1f}")
        print(f"   Output2 score: {result['output2_score']:.1f}")
        print(f"   Winner: {result['winner']}")
        print(f"   Recommendation: {result['recommendation']}")
        if result.get("insights"):
            print(f"   Insights: {result['insights'][0]}")

    # Test batch evaluation
    print("\n6. Batch Evaluation...")
    outputs = [
        "Short response",
        "A more detailed and comprehensive response that provides complete information about the topic being discussed.",
        "This response contains grammatical errors and might be hard to understand for many readers.",
    ]

    result = await agent.evaluate_batch(outputs)
    if result["success"]:
        print(f"   Total: {result['total']}")
        print(f"   Successful: {result['successful']}")
        print(f"   Average score: {result['average_score']:.1f}")
        print(f"   Pass rate: {result['pass_rate']:.1f}%")

    # Get statistics
    print("\n7. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total evaluations: {stats['total_evaluations']}")
    print(f"   Successful: {stats['successful_evaluations']}")
    print(f"   Failed: {stats['failed_evaluations']}")
    print(f"   Average score: {stats['average_score']:.1f}")
    print(f"   Pass rate: {stats['pass_rate']:.1f}%")
    print(f"   History size: {stats['history_size']}")
    if stats.get("by_severity"):
        print(f"   By severity: {stats['by_severity']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_critic_agent())
