"""Knowledge Refiner - refines knowledge based on feedback and experience with advanced learning, versioning, and optimization."""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import copy
import statistics

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger


class RefinementType(Enum):
    """Types of knowledge refinement."""

    CONFIDENCE = "confidence"
    CONTENT = "content"
    STRUCTURE = "structure"
    WEIGHT = "weight"
    RELATIONSHIP = "relationship"
    CONTEXT = "context"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"


class RefinementStrategy(Enum):
    """Strategies for applying refinements."""

    INCREMENTAL = "incremental"  # Small gradual changes
    AGRESSIVE = "aggressive"  # Large changes based on strong feedback
    CONSERVATIVE = "conservative"  # Careful minimal changes
    ADAPTIVE = "adaptive"  # Adjust rate based on feedback strength
    EXPONENTIAL = "exponential"  # Accelerate based on confidence


class KnowledgeType(Enum):
    """Types of knowledge that can be refined."""

    FACT = "fact"
    RULE = "rule"
    PATTERN = "pattern"
    PROCEDURE = "procedure"
    RELATIONSHIP = "relationship"
    CONTEXT = "context"
    PREFERENCE = "preference"
    SKILL = "skill"
    STRATEGY = "strategy"


@dataclass
class RefinementMetadata:
    """Metadata for refinement operations."""

    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    reason: Optional[str] = None
    confidence_delta: float = 0.0
    performance_impact: float = 0.0
    version_before: str = ""
    version_after: str = ""
    tags: List[str] = field(default_factory=list)
    rollback_possible: bool = True


@dataclass
class KnowledgeVersion:
    """Version of knowledge at a point in time."""

    version: str
    content: Any
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of a refinement operation."""

    success: bool
    original_knowledge: Dict[str, Any]
    refined_knowledge: Dict[str, Any]
    changes: Dict[str, Any]
    confidence_change: float
    refinement_type: RefinementType
    processing_time_ms: float
    error: Optional[str] = None


class KnowledgeValidator:
    """Validates knowledge before and after refinement."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_rules: List[Callable] = []

    def add_rule(self, rule: Callable[[Dict[str, Any]], bool]):
        """Add validation rule."""
        self.validation_rules.append(rule)

    def validate(self, knowledge: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate knowledge against all rules."""
        errors = []

        # Basic structure validation
        if not isinstance(knowledge, dict):
            errors.append("Knowledge must be a dictionary")
            return False, errors

        # Check required fields
        if "content" not in knowledge and "id" not in knowledge:
            errors.append("Knowledge missing required fields (content or id)")

        # Check confidence range
        if "confidence" in knowledge:
            confidence = knowledge["confidence"]
            if (
                not isinstance(confidence, (int, float))
                or confidence < 0
                or confidence > 1
            ):
                errors.append(f"Confidence must be between 0 and 1, got {confidence}")

        # Apply custom rules
        for rule in self.validation_rules:
            try:
                if not rule(knowledge):
                    errors.append(f"Rule validation failed: {rule.__name__}")
            except Exception as e:
                errors.append(f"Rule validation error: {e}")

        if self.strict_mode and errors:
            return False, errors

        return len(errors) == 0, errors

    def validate_change(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that changes are acceptable."""
        errors = []

        # Check for catastrophic changes
        if "confidence" in before and "confidence" in after:
            confidence_change = abs(after["confidence"] - before["confidence"])
            if confidence_change > 0.5:
                errors.append(f"Confidence change too large: {confidence_change}")

        # Check content not empty
        if "content" in after and after["content"] is None:
            errors.append("Content cannot be None")

        if (
            "content" in after
            and isinstance(after["content"], str)
            and len(str(after["content"])) == 0
        ):
            errors.append("Content cannot be empty")

        return len(errors) == 0, errors


class KnowledgeMerger:
    """Merge knowledge from multiple sources."""

    def __init__(self, conflict_resolution: str = "confidence"):
        """
        Args:
            conflict_resolution: Strategy for resolving conflicts
                - "confidence": Use higher confidence
                - "recency": Use more recent
                - "average": Average values
                - "voting": Majority vote
        """
        self.conflict_resolution = conflict_resolution

    def merge(self, knowledge_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple knowledge items into one."""
        if not knowledge_list:
            return {}

        if len(knowledge_list) == 1:
            return knowledge_list[0].copy()

        # Group by ID
        by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for knowledge in knowledge_list:
            kid = knowledge.get("id", "unknown")
            by_id[kid].append(knowledge)

        merged = {}
        for kid, items in by_id.items():
            merged[kid] = self._merge_items(items)

        return merged if len(merged) > 1 else merged.get("unknown", {})

    def _merge_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple items with the same ID."""
        if not items:
            return {}

        # Start with a copy of the first item
        merged = items[0].copy()

        for item in items[1:]:
            # Merge content
            if "content" in item:
                if self.conflict_resolution == "recency":
                    # Keep most recent
                    if item.get("timestamp", 0) > merged.get("timestamp", 0):
                        merged["content"] = item["content"]
                elif self.conflict_resolution == "voting":
                    # Would need more sophisticated voting
                    pass
                else:
                    # Default to confidence-based or just take first
                    if item.get("confidence", 0) > merged.get("confidence", 0):
                        merged["content"] = item["content"]

            # Merge confidence (average)
            if "confidence" in item:
                old_conf = merged.get("confidence", 0.5)
                merged["confidence"] = (old_conf + item["confidence"]) / 2

            # Merge metadata
            if "metadata" in item:
                if "metadata" not in merged:
                    merged["metadata"] = {}
                merged["metadata"].update(item["metadata"])

        return merged


class KnowledgeRefiner:
    """Advanced knowledge refiner with versioning, rollback, and optimization."""

    def __init__(
        self,
        max_history: int = 100,
        enable_versioning: bool = True,
        enable_rollback: bool = True,
        auto_save: bool = True,
        persistence_path: str = "data/knowledge",
        min_confidence_threshold: float = 0.3,
        max_confidence_threshold: float = 0.95,
        decay_rate: float = 0.01,
        enable_ml: bool = True,
    ):
        """Initialize advanced knowledge refiner."""
        self.refinement_count = 0
        self.confidence_updates: Dict[str, List[float]] = defaultdict(list)

        # Versioning
        self.enable_versioning = enable_versioning
        self.enable_rollback = enable_rollback
        self.max_history = max_history
        self.knowledge_versions: Dict[str, List[KnowledgeVersion]] = defaultdict(list)

        # Tracking
        self.refinement_history: List[RefinementResult] = []
        self.failed_refinements: List[RefinementResult] = []

        # Metrics
        self.performance_metrics: Dict[str, Any] = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "failed_refinements": 0,
            "rolled_back": 0,
            "average_confidence_gain": 0.0,
            "refinement_latency_ms": [],
            "by_type": defaultdict(int),
        }

        # Learning rates by type
        self.learning_rates: Dict[RefinementType, float] = {
            RefinementType.CONFIDENCE: 0.1,
            RefinementType.CONTENT: 0.15,
            RefinementType.STRUCTURE: 0.05,
            RefinementType.WEIGHT: 0.1,
            RefinementType.RELATIONSHIP: 0.08,
            RefinementType.CONTEXT: 0.12,
            RefinementType.TEMPORAL: 0.1,
            RefinementType.SEMANTIC: 0.07,
        }

        # Strategy settings
        self.strategies: Dict[RefinementStrategy, Dict[str, float]] = {
            RefinementStrategy.INCREMENTAL: {"rate_multiplier": 0.5, "threshold": 0.7},
            RefinementStrategy.AGRESSIVE: {"rate_multiplier": 2.0, "threshold": 0.9},
            RefinementStrategy.CONSERVATIVE: {"rate_multiplier": 0.3, "threshold": 0.6},
            RefinementStrategy.ADAPTIVE: {"rate_multiplier": 1.0, "threshold": 0.5},
            RefinementStrategy.EXPONENTIAL: {"rate_multiplier": 1.5, "threshold": 0.8},
        }

        # Persistence
        self.auto_save = auto_save
        self.persistence_path = Path(persistence_path)
        if auto_save:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Validators and mergers
        self.validator = KnowledgeValidator(strict_mode=False)
        self.merger = KnowledgeMerger()

        # Callbacks
        self._callbacks: List[Callable[[RefinementResult], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # ML components
        self.enable_ml = enable_ml and NUMPY_AVAILABLE
        self._optimization_cache: Dict[str, Any] = {}

        # Confidence decay
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence_threshold
        self.max_confidence = max_confidence_threshold

        # Async task
        self._save_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"✅ Advanced KnowledgeRefiner initialized (versioning={enable_versioning}, ml={self.enable_ml})"
        )

    # #==================== Core Refinement #====================

    def refine(
        self,
        knowledge: Dict[str, Any],
        feedback: Dict[str, Any],
        learning_rate: Optional[float] = None,
        refinement_type: Optional[RefinementType] = None,
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Refine knowledge based on feedback with advanced options.

        Args:
            knowledge: Knowledge to refine
            feedback: Feedback to apply
            learning_rate: Override default learning rate
            refinement_type: Type of refinement to apply
            strategy: Refinement strategy to use
            validate: Whether to validate before/after

        Returns:
            Refined knowledge
        """
        import time

        start_time = time.time()

        # Validate before refinement
        if validate:
            is_valid, errors = self.validator.validate(knowledge)
            if not is_valid:
                logger.warning(f"Knowledge invalid before refinement: {errors}")
                if self.validator.strict_mode:
                    return knowledge

        # Create version before refinement
        before_version = self._create_version(knowledge)

        # Determine refinement type
        if not refinement_type:
            refinement_type = self._determine_refinement_type(knowledge, feedback)

        # Calculate effective learning rate
        effective_rate = self._calculate_learning_rate(
            learning_rate, refinement_type, strategy, feedback
        )

        # Apply refinement
        try:
            refined = self._apply_refinement(
                knowledge, feedback, effective_rate, refinement_type
            )

            # Validate after refinement
            if validate:
                is_valid, errors = self.validator.validate(refined)
                if not is_valid:
                    logger.warning(f"Knowledge invalid after refinement: {errors}")
                    if self.validator.strict_mode:
                        return knowledge

            # Validate change
            if validate:
                is_valid, errors = self.validator.validate_change(knowledge, refined)
                if not is_valid:
                    logger.warning(f"Invalid change: {errors}")
                    if self.enable_rollback:
                        return knowledge

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            result = self._create_failed_result(
                knowledge, knowledge, e, refinement_type, start_time
            )
            self.failed_refinements.append(result)
            self._trigger_callbacks(result)
            return knowledge

        # Track changes
        changes = self._compute_changes(knowledge, refined)
        confidence_change = refined.get("confidence", 0.5) - knowledge.get(
            "confidence", 0.5
        )

        # Store version
        if self.enable_versioning:
            after_version = self._create_version(refined)
            self._add_version(knowledge.get("id", "unknown"), before_version)
            self._add_version(knowledge.get("id", "unknown"), after_version)

        # Update tracking
        self._update_tracking(knowledge, refined, refinement_type, confidence_change)

        # Create result
        processing_time = (time.time() - start_time) * 1000
        result = RefinementResult(
            success=True,
            original_knowledge=knowledge,
            refined_knowledge=refined,
            changes=changes,
            confidence_change=confidence_change,
            refinement_type=refinement_type,
            processing_time_ms=processing_time,
        )

        self.refinement_history.append(result)
        self.performance_metrics["refinement_latency_ms"].append(processing_time)

        # Trim history
        if len(self.refinement_history) > self.max_history:
            self.refinement_history = self.refinement_history[-self.max_history :]

        # Trigger callbacks
        self._trigger_callbacks(result)

        # Auto-save
        if self.auto_save:
            asyncio.create_task(self.save_knowledge())

        logger.debug(
            f"🔧 Refined {knowledge.get('id', 'unknown')}: confidence {knowledge.get('confidence', 0.5):.2f} -> {refined.get('confidence', 0.5):.2f}"
        )

        return refined

    def _determine_refinement_type(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any]
    ) -> RefinementType:
        """Determine appropriate refinement type based on feedback."""

        # Check for specific refinement indicators
        if "correction" in feedback:
            return RefinementType.CONTENT
        elif "relationship" in feedback:
            return RefinementType.RELATIONSHIP
        elif "context" in feedback:
            return RefinementType.CONTEXT
        elif "weight" in feedback:
            return RefinementType.WEIGHT
        elif "structure" in feedback:
            return RefinementType.STRUCTURE
        elif "temporal" in feedback:
            return RefinementType.TEMPORAL
        elif "semantic" in feedback:
            return RefinementType.SEMANTIC
        elif "confidence" in feedback:
            return RefinementType.CONFIDENCE

        # Default based on knowledge type
        knowledge_type = knowledge.get("type", "fact")
        if knowledge_type == "rule":
            return RefinementType.STRUCTURE
        elif knowledge_type == "procedure":
            return RefinementType.CONTENT
        elif knowledge_type == "relationship":
            return RefinementType.RELATIONSHIP

        return RefinementType.CONFIDENCE

    def _calculate_learning_rate(
        self,
        provided_rate: Optional[float],
        refinement_type: RefinementType,
        strategy: RefinementStrategy,
        feedback: Dict[str, Any],
    ) -> float:
        """Calculate effective learning rate based on multiple factors."""

        # Base rate from type
        base_rate = provided_rate or self.learning_rates.get(refinement_type, 0.1)

        # Apply strategy multiplier
        strategy_config = self.strategies.get(strategy, {"rate_multiplier": 1.0})
        rate_multiplier = strategy_config["rate_multiplier"]

        # Adjust based on feedback strength
        feedback_strength = feedback.get("strength", 1.0)

        # Adjust based on confidence difference
        if "confidence" in feedback:
            current_conf = feedback.get("current_confidence", 0.5)
            feedback_conf = feedback["confidence"]
            confidence_gap = abs(feedback_conf - current_conf)
            feedback_strength *= 1 + confidence_gap

        # Calculate final rate
        effective_rate = base_rate * rate_multiplier * min(2.0, feedback_strength)

        # Clamp to reasonable range
        return max(0.01, min(0.5, effective_rate))

    def _apply_refinement(
        self,
        knowledge: Dict[str, Any],
        feedback: Dict[str, Any],
        learning_rate: float,
        refinement_type: RefinementType,
    ) -> Dict[str, Any]:
        """Apply specific refinement based on type."""

        refined = knowledge.copy()

        if refinement_type == RefinementType.CONFIDENCE:
            refined = self._refine_confidence(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.CONTENT:
            refined = self._refine_content(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.STRUCTURE:
            refined = self._refine_structure(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.WEIGHT:
            refined = self._refine_weights(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.RELATIONSHIP:
            refined = self._refine_relationships(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.CONTEXT:
            refined = self._refine_context(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.TEMPORAL:
            refined = self._refine_temporal(refined, feedback, learning_rate)

        elif refinement_type == RefinementType.SEMANTIC:
            refined = self._refine_semantic(refined, feedback, learning_rate)

        # Apply confidence decay if needed
        if "confidence" in refined and self.decay_rate > 0:
            refined["confidence"] = self._apply_confidence_decay(refined["confidence"])

        return refined

    def _refine_confidence(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine confidence scores."""
        refined = knowledge.copy()

        current_confidence = refined.get("confidence", 0.5)
        feedback_confidence = feedback.get("confidence", current_confidence)

        # Exponential moving average
        refined["confidence"] = (
            current_confidence * (1 - learning_rate)
            + feedback_confidence * learning_rate
        )

        # Clamp to thresholds
        refined["confidence"] = max(
            self.min_confidence, min(self.max_confidence, refined["confidence"])
        )

        # Add confidence metadata
        if "metadata" not in refined:
            refined["metadata"] = {}
        refined["metadata"]["confidence_updated"] = datetime.now().isoformat()
        refined["metadata"]["confidence_delta"] = (
            refined["confidence"] - current_confidence
        )

        return refined

    def _refine_content(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine knowledge content."""
        refined = knowledge.copy()

        if "correction" in feedback:
            # Apply correction
            current_content = refined.get("content", "")
            correction = feedback["correction"]

            # Blend content based on confidence
            current_conf = refined.get("confidence", 0.5)
            feedback_conf = feedback.get("confidence", 0.5)

            if current_conf < feedback_conf:
                refined["content"] = correction
            elif learning_rate > 0.3:
                # Partial update
                refined["content"] = self._merge_content(
                    current_content, correction, learning_rate
                )

        # Update version tracking
        refined["version"] = refined.get("version", 0) + 1
        refined["last_modified"] = datetime.now().isoformat()

        return refined

    def _refine_structure(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine knowledge structure."""
        refined = knowledge.copy()

        if "structure" in feedback:
            new_structure = feedback["structure"]

            # Apply structural changes based on confidence
            if feedback.get("confidence", 0.5) > 0.7:
                refined["structure"] = new_structure
                refined["structure_version"] = refined.get("structure_version", 0) + 1

        return refined

    def _refine_weights(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine knowledge weights."""
        refined = knowledge.copy()

        if "weights" in knowledge and "weight_adjustments" in feedback:
            weights = knowledge["weights"].copy()
            adjustments = feedback["weight_adjustments"]

            for key, adjustment in adjustments.items():
                if key in weights:
                    weights[key] = (
                        weights[key] * (1 - learning_rate) + adjustment * learning_rate
                    )
                    weights[key] = max(0, min(1, weights[key]))

            refined["weights"] = weights

        return refined

    def _refine_relationships(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine knowledge relationships."""
        refined = knowledge.copy()

        if "relationships" in feedback:
            relationships = refined.get("relationships", {})

            for rel_type, rel_value in feedback["relationships"].items():
                if rel_type in relationships:
                    # Update existing relationship
                    old_strength = relationships[rel_type].get("strength", 0.5)
                    new_strength = rel_value.get("strength", old_strength)
                    relationships[rel_type]["strength"] = (
                        old_strength * (1 - learning_rate)
                        + new_strength * learning_rate
                    )
                else:
                    # Add new relationship
                    relationships[rel_type] = rel_value

            refined["relationships"] = relationships

        return refined

    def _refine_context(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine knowledge context."""
        refined = knowledge.copy()

        if "context" in feedback:
            context = refined.get("context", {})
            new_context = feedback["context"]

            # Merge context with learning rate
            for key, value in new_context.items():
                if key in context:
                    # Blend existing and new context
                    if isinstance(value, (int, float)):
                        context[key] = (
                            context[key] * (1 - learning_rate) + value * learning_rate
                        )
                    else:
                        context[key] = value
                else:
                    context[key] = value

            refined["context"] = context

        return refined

    def _refine_temporal(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine temporal aspects of knowledge."""
        refined = knowledge.copy()

        # Update timestamps
        refined["last_refined"] = datetime.now().isoformat()

        # Apply temporal decay if feedback suggests
        if feedback.get("temporal_decay", False):
            if "confidence" in refined:
                refined["confidence"] *= 1 - self.decay_rate
                refined["confidence"] = max(self.min_confidence, refined["confidence"])

        # Update temporal patterns
        if "temporal_patterns" in feedback:
            patterns = refined.get("temporal_patterns", {})
            patterns.update(feedback["temporal_patterns"])
            refined["temporal_patterns"] = patterns

        return refined

    def _refine_semantic(
        self, knowledge: Dict[str, Any], feedback: Dict[str, Any], learning_rate: float
    ) -> Dict[str, Any]:
        """Refine semantic understanding."""
        refined = knowledge.copy()

        if "semantic" in feedback:
            semantic = refined.get("semantic", {})

            # Update semantic embeddings or meanings
            for concept, meaning in feedback["semantic"].items():
                if concept in semantic:
                    # Blend meanings
                    if isinstance(meaning, (int, float)):
                        semantic[concept] = (
                            semantic[concept] * (1 - learning_rate)
                            + meaning * learning_rate
                        )
                    else:
                        semantic[concept] = meaning
                else:
                    semantic[concept] = meaning

            refined["semantic"] = semantic

        return refined

    def _merge_content(self, current: Any, new: Any, learning_rate: float) -> Any:
        """Merge current and new content."""
        if isinstance(current, str) and isinstance(new, str):
            # For strings, use weighted combination or take the new one
            if learning_rate > 0.5:
                return new
            else:
                # Simple concatenation for high learning rate
                return f"{current} (updated: {new})"

        elif isinstance(current, dict) and isinstance(new, dict):
            # Recursively merge dictionaries
            merged = current.copy()
            for key, value in new.items():
                if key in merged:
                    merged[key] = self._merge_content(merged[key], value, learning_rate)
                else:
                    merged[key] = value
            return merged

        elif isinstance(current, list) and isinstance(new, list):
            # Merge lists
            if learning_rate > 0.5:
                return new
            else:
                return current + [x for x in new if x not in current]

        else:
            # Default to new content
            return new

    def _apply_confidence_decay(self, confidence: float) -> float:
        """Apply time-based confidence decay."""
        decayed = confidence * (1 - self.decay_rate)
        return max(self.min_confidence, decayed)

    def _compute_changes(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute detailed changes between two knowledge states."""
        changes = {}

        # Check confidence change
        if before.get("confidence") != after.get("confidence"):
            changes["confidence"] = {
                "before": before.get("confidence"),
                "after": after.get("confidence"),
                "delta": after.get("confidence", 0) - before.get("confidence", 0),
            }

        # Check content change
        if before.get("content") != after.get("content"):
            changes["content"] = {
                "before": str(before.get("content"))[:100],
                "after": str(after.get("content"))[:100],
            }

        # Check structural changes
        if before.get("structure") != after.get("structure"):
            changes["structure"] = {"changed": True}

        # Check version
        if before.get("version") != after.get("version"):
            changes["version"] = {
                "before": before.get("version"),
                "after": after.get("version"),
            }

        return changes

    def _create_version(self, knowledge: Dict[str, Any]) -> KnowledgeVersion:
        """Create a version snapshot of knowledge."""
        version_num = knowledge.get("version", 0) + 1

        return KnowledgeVersion(
            version=f"v{version_num}",
            content=copy.deepcopy(knowledge.get("content")),
            confidence=knowledge.get("confidence", 0.5),
            timestamp=datetime.now(),
            metadata={
                "type": knowledge.get("type", "unknown"),
                "source": knowledge.get("source", "unknown"),
            },
        )

    def _add_version(self, knowledge_id: str, version: KnowledgeVersion):
        """Add version to history."""
        versions = self.knowledge_versions[knowledge_id]
        versions.append(version)

        # Maintain max history
        if len(versions) > self.max_history:
            versions.pop(0)

    def _update_tracking(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        refinement_type: RefinementType,
        confidence_change: float,
    ):
        """Update tracking metrics."""
        self.refinement_count += 1
        self.performance_metrics["total_refinements"] += 1
        self.performance_metrics["successful_refinements"] += 1
        self.performance_metrics["by_type"][refinement_type.value] += 1

        # Update confidence history
        knowledge_id = before.get("id", "unknown")
        self.confidence_updates[knowledge_id].append(after.get("confidence", 0.5))

        # Update average confidence gain
        total_gain = self.performance_metrics["average_confidence_gain"] * (
            self.performance_metrics["successful_refinements"] - 1
        )
        total_gain += confidence_change
        self.performance_metrics["average_confidence_gain"] = (
            total_gain / self.performance_metrics["successful_refinements"]
        )

    def _create_failed_result(
        self,
        original: Dict[str, Any],
        refined: Dict[str, Any],
        error: Exception,
        refinement_type: RefinementType,
        start_time: float,
    ) -> RefinementResult:
        """Create failed refinement result."""
        processing_time = (time.time() - start_time) * 1000
        self.performance_metrics["failed_refinements"] += 1

        return RefinementResult(
            success=False,
            original_knowledge=original,
            refined_knowledge=refined,
            changes={},
            confidence_change=0,
            refinement_type=refinement_type,
            processing_time_ms=processing_time,
            error=str(error),
        )

    # #==================== Batch Operations #====================

    def batch_refine(
        self,
        knowledge_list: List[Dict[str, Any]],
        feedback_list: List[Dict[str, Any]],
        learning_rate: float = 0.1,
        strategy: RefinementStrategy = RefinementStrategy.ADAPTIVE,
    ) -> List[Dict[str, Any]]:
        """Refine multiple knowledge items."""
        results = []

        for knowledge, feedback in zip(knowledge_list, feedback_list):
            result = self.refine(knowledge, feedback, learning_rate, strategy=strategy)
            results.append(result)

        return results

    def batch_refine_async(
        self,
        knowledge_list: List[Dict[str, Any]],
        feedback_list: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """Asynchronously refine multiple knowledge items."""
        import concurrent.futures

        results = [None] * len(knowledge_list)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as executor:
            futures = {}
            for i, (knowledge, feedback) in enumerate(
                zip(knowledge_list, feedback_list)
            ):
                future = executor.submit(self.refine, knowledge, feedback)
                futures[future] = i

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch refine failed for item {idx}: {e}")
                    results[idx] = knowledge_list[idx]

        return results

    # #==================== Version Management #====================

    def get_version(
        self, knowledge_id: str, version: Optional[str] = None
    ) -> Optional[KnowledgeVersion]:
        """Get specific version of knowledge."""
        versions = self.knowledge_versions.get(knowledge_id, [])
        if not versions:
            return None

        if version:
            for v in versions:
                if v.version == version:
                    return v
            return None

        return versions[-1]  # Latest version

    def get_version_history(self, knowledge_id: str) -> List[KnowledgeVersion]:
        """Get full version history."""
        return self.knowledge_versions.get(knowledge_id, [])

    def rollback(
        self, knowledge_id: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Rollback knowledge to previous version."""
        if not self.enable_rollback:
            logger.warning("Rollback not enabled")
            return None

        versions = self.knowledge_versions.get(knowledge_id, [])
        if len(versions) < 2:
            logger.warning(f"Not enough versions to rollback {knowledge_id}")
            return None

        # Get target version
        if version:
            target_version = self.get_version(knowledge_id, version)
        else:
            target_version = versions[-2]  # Previous version

        if not target_version:
            return None

        # Create knowledge from version
        rolled_back = {
            "id": knowledge_id,
            "content": target_version.content,
            "confidence": target_version.confidence,
            "version": target_version.version,
            "rolled_back_at": datetime.now().isoformat(),
        }

        self.performance_metrics["rolled_back"] += 1
        logger.info(f"🔄 Rolled back {knowledge_id} to {target_version.version}")

        return rolled_back

    # #==================== Queries and Analytics #====================

    def get_refinement_history(self, key: str) -> Optional[List[float]]:
        """Get refinement history for knowledge."""
        return self.confidence_updates.get(key)

    def get_confidence_trend(self, knowledge_id: str) -> Dict[str, Any]:
        """Analyze confidence trend over time."""
        history = self.confidence_updates.get(knowledge_id, [])

        if len(history) < 2:
            return {
                "trend": "insufficient_data",
                "current": history[-1] if history else 0.5,
            }

        # Calculate trend
        first = history[0]
        last = history[-1]

        if last > first:
            trend = "improving"
        elif last < first:
            trend = "declining"
        else:
            trend = "stable"

        # Calculate rate of change
        if NUMPY_AVAILABLE and len(history) >= 3:
            x = np.arange(len(history))
            y = np.array(history)
            slope = np.polyfit(x, y, 1)[0]
            rate = slope * 100  # Percent change per refinement
        else:
            rate = ((last - first) / len(history)) * 100

        return {
            "trend": trend,
            "current": last,
            "initial": first,
            "change": last - first,
            "rate_per_refinement": rate,
            "history_length": len(history),
        }

    def get_knowledge_health(self, knowledge_id: str) -> Dict[str, Any]:
        """Get overall health metrics for knowledge."""
        history = self.confidence_updates.get(knowledge_id, [])
        versions = self.knowledge_versions.get(knowledge_id, [])

        current_confidence = history[-1] if history else 0.5

        # Calculate stability (variance in recent confidence)
        recent = history[-10:] if len(history) >= 10 else history
        stability = 1 - (statistics.stdev(recent) if len(recent) > 1 else 0)

        # Calculate refinement frequency
        refinement_frequency = len(history) / (max(1, len(versions)))

        # Health score (0-1)
        health_score = (
            current_confidence * 0.5
            + stability * 0.3
            + (1 - min(1, refinement_frequency)) * 0.2
        )

        return {
            "knowledge_id": knowledge_id,
            "health_score": health_score,
            "current_confidence": current_confidence,
            "stability": stability,
            "refinement_count": len(history),
            "version_count": len(versions),
            "refinement_frequency": refinement_frequency,
            "needs_review": health_score < 0.6,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive refiner statistics."""
        all_confidences = []
        for conf_list in self.confidence_updates.values():
            all_confidences.extend(conf_list)

        # Calculate average latency
        avg_latency = 0
        if self.performance_metrics["refinement_latency_ms"]:
            avg_latency = statistics.mean(
                self.performance_metrics["refinement_latency_ms"]
            )

        # Get knowledge health for all items
        knowledge_health = []
        for kid in self.confidence_updates.keys():
            knowledge_health.append(self.get_knowledge_health(kid))

        avg_health = (
            statistics.mean([h["health_score"] for h in knowledge_health])
            if knowledge_health
            else 0.5
        )

        return {
            "refinements": self.refinement_count,
            "success_rate": self.performance_metrics["successful_refinements"]
            / max(1, self.performance_metrics["total_refinements"]),
            "avg_confidence": (
                statistics.mean(all_confidences) if all_confidences else 0
            ),
            "min_confidence": min(all_confidences) if all_confidences else 0,
            "max_confidence": max(all_confidences) if all_confidences else 0,
            "unique_knowledge_items": len(self.confidence_updates),
            "total_versions": sum(len(v) for v in self.knowledge_versions.values()),
            "rolled_back": self.performance_metrics["rolled_back"],
            "avg_latency_ms": avg_latency,
            "avg_health_score": avg_health,
            "by_type": dict(self.performance_metrics["by_type"]),
            "average_confidence_gain": self.performance_metrics[
                "average_confidence_gain"
            ],
            "health_distribution": {
                "healthy": sum(1 for h in knowledge_health if h["health_score"] >= 0.7),
                "moderate": sum(
                    1 for h in knowledge_health if 0.4 <= h["health_score"] < 0.7
                ),
                "needs_review": sum(
                    1 for h in knowledge_health if h["health_score"] < 0.4
                ),
            },
        }

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for optimizing knowledge refinement."""
        suggestions = []
        stats = self.get_stats()

        # Check success rate
        if stats["success_rate"] < 0.8:
            suggestions.append(
                {
                    "area": "success_rate",
                    "issue": f"Low refinement success rate: {stats['success_rate']:.1%}",
                    "suggestion": "Review feedback quality or reduce learning rate",
                    "priority": "high",
                }
            )

        # Check health scores
        if stats["avg_health_score"] < 0.6:
            suggestions.append(
                {
                    "area": "knowledge_health",
                    "issue": f"Low average health score: {stats['avg_health_score']:.2f}",
                    "suggestion": "Increase refinement frequency or improve feedback quality",
                    "priority": "medium",
                }
            )

        # Check confidence gain
        if stats["average_confidence_gain"] < 0.01:
            suggestions.append(
                {
                    "area": "learning",
                    "issue": "Minimal confidence improvement from refinements",
                    "suggestion": "Increase learning rate or use more aggressive strategy",
                    "priority": "medium",
                }
            )

        # Check version count
        if stats["total_versions"] > 1000:
            suggestions.append(
                {
                    "area": "storage",
                    "issue": f"Large version history: {stats['total_versions']} versions",
                    "suggestion": "Consider pruning old versions or reducing max_history",
                    "priority": "low",
                }
            )

        return suggestions

    # #==================== Persistence #====================

    async def save_knowledge(self, filename: Optional[str] = None) -> bool:
        """Save knowledge state to disk."""
        if not filename:
            filename = (
                f"knowledge_refiner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        filepath = self.persistence_path / filename

        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "refinement_count": self.refinement_count,
                "confidence_updates": dict(self.confidence_updates),
                "performance_metrics": self.performance_metrics,
                "knowledge_versions": {},
                "stats": self.get_stats(),
            }

            # Convert versions to serializable format
            for kid, versions in self.knowledge_versions.items():
                data["knowledge_versions"][kid] = [
                    {
                        "version": v.version,
                        "confidence": v.confidence,
                        "timestamp": v.timestamp.isoformat(),
                        "metadata": v.metadata,
                    }
                    for v in versions
                ]

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"💾 Saved knowledge refiner state to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
            return False

    async def load_knowledge(self, filepath: Optional[str] = None) -> bool:
        """Load knowledge state from disk."""
        if not filepath:
            files = sorted(
                self.persistence_path.glob("knowledge_refiner_*.json"), reverse=True
            )
            if not files:
                return False
            filepath = files[0]
        else:
            filepath = Path(filepath)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.refinement_count = data.get("refinement_count", 0)

            # Restore confidence updates
            self.confidence_updates.clear()
            for kid, updates in data.get("confidence_updates", {}).items():
                self.confidence_updates[kid] = updates

            # Restore performance metrics
            self.performance_metrics.update(data.get("performance_metrics", {}))

            # Restore versions
            self.knowledge_versions.clear()
            for kid, versions in data.get("knowledge_versions", {}).items():
                self.knowledge_versions[kid] = [
                    KnowledgeVersion(
                        version=v["version"],
                        content=None,  # Content not stored to save space
                        confidence=v["confidence"],
                        timestamp=datetime.fromisoformat(v["timestamp"]),
                        metadata=v["metadata"],
                    )
                    for v in versions
                ]

            logger.info(f"📂 Loaded knowledge refiner state from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")
            return False

    # #==================== Callbacks #====================

    def add_callback(self, callback: Callable[[RefinementResult], None]):
        """Add callback for refinement results."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[RefinementResult], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback."""
        self._error_callbacks.append(callback)

    def _trigger_callbacks(self, result: RefinementResult):
        """Trigger all callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                for err_cb in self._error_callbacks:
                    try:
                        err_cb(e)
                    except Exception:
                        pass

    # #==================== Cleanup #====================

    def prune_old_versions(self, keep_count: int = 10):
        """Prune old versions to save memory."""
        for kid in self.knowledge_versions:
            if len(self.knowledge_versions[kid]) > keep_count:
                self.knowledge_versions[kid] = self.knowledge_versions[kid][
                    -keep_count:
                ]

        logger.info(f"✂️ Pruned versions to {keep_count} per knowledge item")

    def reset(self):
        """Reset all refinement data."""
        self.refinement_count = 0
        self.confidence_updates.clear()
        self.knowledge_versions.clear()
        self.refinement_history.clear()
        self.failed_refinements.clear()
        self.performance_metrics = {
            "total_refinements": 0,
            "successful_refinements": 0,
            "failed_refinements": 0,
            "rolled_back": 0,
            "average_confidence_gain": 0.0,
            "refinement_latency_ms": [],
            "by_type": defaultdict(int),
        }
        self._optimization_cache.clear()

        logger.info("🔄 Knowledge refiner reset")

    # #==================== Lifecycle #====================

    async def start(self):
        """Start background tasks."""
        self._running = True
        logger.info("🚀 Knowledge refiner started")

    async def stop(self):
        """Stop background tasks."""
        self._running = False

        if self.auto_save:
            await self.save_knowledge()

        logger.info("🛑 Knowledge refiner stopped")


# #==================== Convenience Functions #====================

__all__ = [
    "KnowledgeRefiner",
    "RefinementType",
    "RefinementStrategy",
    "KnowledgeType",
    "RefinementMetadata",
    "KnowledgeVersion",
    "RefinementResult",
    "KnowledgeValidator",
    "KnowledgeMerger",
]
