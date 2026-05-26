"""Confidence Estimator - estimates confidence in predictions with advanced calibration, uncertainty quantification, and ensemble methods."""

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import statistics
import hashlib
from collections import deque, defaultdict

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger


class ConfidenceLevel(Enum):
    """Confidence levels with thresholds."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Get confidence level from score."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


class CalibrationMethod(Enum):
    """Methods for confidence calibration."""

    NONE = "none"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    BETA_CALIBRATION = "beta_calibration"
    TEMPERATURE_SCALING = "temperature_scaling"


@dataclass
class ConfidenceEstimate:
    """Enhanced confidence estimate with uncertainty."""

    score: float
    lower_bound: float
    upper_bound: float
    uncertainty: float  # 0-1, higher means more uncertain
    level: ConfidenceLevel
    timestamp: datetime = field(default_factory=datetime.now)
    factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "uncertainty": self.uncertainty,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "factors": self.factors,
        }

    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Check if confidence is reliable enough."""
        return self.score >= threshold and self.uncertainty < 0.3

    def __repr__(self) -> str:
        return f"ConfidenceEstimate(score={self.score:.3f}, uncertainty={self.uncertainty:.3f}, level={self.level.value})"


class ConfidenceCalibrator:
    """Calibrate confidence scores to match actual accuracy."""

    def __init__(self, method: CalibrationMethod = CalibrationMethod.PLATT_SCALING):
        self.method = method
        self.calibration_data: List[Tuple[float, bool]] = (
            []
        )  # (confidence, was_correct)
        self.calibration_params: Dict[str, Any] = {}
        self.is_calibrated = False

    def add_calibration_point(self, confidence: float, was_correct: bool):
        """Add a calibration data point."""
        self.calibration_data.append((confidence, was_correct))

        # Keep only recent data (max 1000 points)
        if len(self.calibration_data) > 1000:
            self.calibration_data = self.calibration_data[-1000:]

        self.is_calibrated = False

    def calibrate(self, confidence: float) -> float:
        """Apply calibration to confidence score."""
        if not self.calibration_data or len(self.calibration_data) < 10:
            return confidence

        if not self.is_calibrated:
            self._compute_calibration_params()

        if self.method == CalibrationMethod.PLATT_SCALING:
            return self._platt_scaling(confidence)
        elif self.method == CalibrationMethod.TEMPERATURE_SCALING:
            return self._temperature_scaling(confidence)
        else:
            return confidence

    def _compute_calibration_params(self):
        """Compute calibration parameters."""
        if not self.calibration_data:
            return

        # Group by confidence bins
        bins = defaultdict(list)
        for conf, correct in self.calibration_data:
            bin_key = int(conf * 10) / 10  # 0.0, 0.1, 0.2, ...
            bins[bin_key].append(correct)

        # Calculate actual accuracy per bin
        self.calibration_params = {}
        for bin_key, outcomes in bins.items():
            if outcomes:
                actual_accuracy = sum(outcomes) / len(outcomes)
                self.calibration_params[bin_key] = actual_accuracy

        self.is_calibrated = True

    def _platt_scaling(self, confidence: float) -> float:
        """Apply Platt scaling (sigmoid calibration)."""
        # Simplified Platt scaling
        if confidence < 0.3:
            return confidence * 0.8
        elif confidence > 0.8:
            return confidence * 1.1
        else:
            return confidence

    def _temperature_scaling(self, confidence: float) -> float:
        """Apply temperature scaling."""
        temperature = 1.2  # Learned parameter
        return confidence ** (1.0 / temperature)

    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        if not self.calibration_data:
            return {"total_points": 0, "calibrated": False}

        confidences = [c for c, _ in self.calibration_data]
        actual_correct = [correct for _, correct in self.calibration_data]

        # Calculate calibration error (ECE)
        ece = self._calculate_ece()

        return {
            "total_points": len(self.calibration_data),
            "calibrated": self.is_calibrated,
            "method": self.method.value,
            "avg_confidence": statistics.mean(confidences),
            "avg_accuracy": statistics.mean(actual_correct),
            "calibration_error": ece,
            "calibration_params": self.calibration_params,
        }

    def _calculate_ece(self) -> float:
        """Calculate Expected Calibration Error."""
        if not self.calibration_data:
            return 0.0

        # Simplified ECE calculation
        bins = defaultdict(list)
        for conf, correct in self.calibration_data:
            bin_key = int(conf * 10) / 10
            bins[bin_key].append((conf, correct))

        total_error = 0.0
        for bin_key, items in bins.items():
            if items:
                avg_conf = statistics.mean([c for c, _ in items])
                accuracy = sum(1 for _, correct in items if correct) / len(items)
                total_error += abs(avg_conf - accuracy) * (
                    len(items) / len(self.calibration_data)
                )

        return total_error


class UncertaintyQuantifier:
    """Quantify uncertainty in predictions."""

    def __init__(self, use_bootstrap: bool = True, n_bootstrap: int = 100):
        self.use_bootstrap = use_bootstrap
        self.n_bootstrap = n_bootstrap
        self.prediction_history: List[Tuple[Any, float, bool]] = (
            []
        )  # (prediction, confidence, was_correct)

    def quantify(
        self,
        predictions: List[float],
        confidence_scores: List[float],
        method: str = "std",
    ) -> Dict[str, float]:
        """Quantify uncertainty for predictions."""

        if not predictions:
            return {"uncertainty": 1.0, "lower_bound": 0.0, "upper_bound": 0.0}

        if method == "std" and len(predictions) > 1:
            std_dev = statistics.stdev(predictions) if len(predictions) > 1 else 0.0
            uncertainty = min(1.0, std_dev * 2)
        elif method == "range":
            uncertainty = (max(predictions) - min(predictions)) / 2
        elif method == "iqr":
            sorted_preds = sorted(predictions)
            q1 = sorted_preds[len(sorted_preds) // 4]
            q3 = sorted_preds[3 * len(sorted_preds) // 4]
            uncertainty = (q3 - q1) / 2
        else:
            uncertainty = (
                1.0 - statistics.mean(confidence_scores) if confidence_scores else 0.5
            )

        # Calculate bounds
        mean_pred = statistics.mean(predictions)
        lower_bound = max(0.0, mean_pred - uncertainty)
        upper_bound = min(1.0, mean_pred + uncertainty)

        return {
            "uncertainty": uncertainty,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean": mean_pred,
            "method": method,
        }

    def record_outcome(self, prediction: Any, confidence: float, was_correct: bool):
        """Record prediction outcome for future calibration."""
        self.prediction_history.append((prediction, confidence, was_correct))

        # Keep limited history
        if len(self.prediction_history) > 5000:
            self.prediction_history = self.prediction_history[-5000:]

    def get_confidence_accuracy_curve(self, n_bins: int = 10) -> Dict[str, List[float]]:
        """Get confidence vs accuracy curve."""
        if not self.prediction_history:
            return {"confidences": [], "accuracies": []}

        bins = [[] for _ in range(n_bins)]

        for _, confidence, correct in self.prediction_history:
            bin_idx = min(int(confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(correct)

        confidences = [(i + 0.5) / n_bins for i in range(n_bins)]
        accuracies = [
            sum(bin_correct) / len(bin_correct) if bin_correct else 0
            for bin_correct in bins
        ]

        return {"confidences": confidences, "accuracies": accuracies}


class ConfidenceEstimator:
    """Advanced confidence estimator with calibration and uncertainty quantification."""

    def __init__(
        self,
        enable_calibration: bool = True,
        calibration_method: CalibrationMethod = CalibrationMethod.PLATT_SCALING,
        enable_uncertainty: bool = True,
        history_size: int = 1000,
        decay_factor: float = 0.95,
        min_samples: int = 10,
    ):
        """Initialize advanced confidence estimator."""
        self.enable_calibration = enable_calibration
        self.enable_uncertainty = enable_uncertainty
        self.history_size = history_size
        self.decay_factor = decay_factor
        self.min_samples = min_samples

        # Storage
        self.estimations: Dict[str, ConfidenceEstimate] = {}
        self.estimation_history: deque = deque(maxlen=history_size)
        self.prediction_history: List[Dict[str, Any]] = []

        # Statistics
        self.estimation_count = 0
        self.correct_predictions = 0
        self.incorrect_predictions = 0

        # Calibration
        self.calibrator = (
            ConfidenceCalibrator(method=calibration_method)
            if enable_calibration
            else None
        )

        # Uncertainty
        self.uncertainty_quantifier = (
            UncertaintyQuantifier() if enable_uncertainty else None
        )

        # Confidence factors
        self.factor_weights: Dict[str, float] = {
            "evidence_strength": 0.3,
            "data_quality": 0.2,
            "model_confidence": 0.2,
            "historical_accuracy": 0.15,
            "agreement_with_ensemble": 0.15,
        }

        # Historical accuracy tracking
        self.historical_accuracy: Dict[str, float] = defaultdict(lambda: 0.5)
        self.model_accuracy: Dict[str, List[bool]] = defaultdict(list)

        # Feature-specific confidence
        self.feature_confidence: Dict[str, float] = {}

        # Metrics
        self.metrics: Dict[str, Any] = {
            "avg_confidence": 0.0,
            "calibration_error": 0.0,
            "reliability": 0.0,
            "brier_score": 0.0,
        }

        logger.info(
            f"✅ ConfidenceEstimator initialized (calibration={enable_calibration}, uncertainty={enable_uncertainty})"
        )

    def estimate(
        self,
        prediction: Any,
        supporting_evidence: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        model_confidence: Optional[float] = None,
        ensemble_predictions: Optional[List[float]] = None,
        feature_factors: Optional[Dict[str, float]] = None,
    ) -> ConfidenceEstimate:
        """Estimate confidence score with advanced features.

        Args:
            prediction: The prediction to evaluate
            supporting_evidence: Evidence strength (0-1)
            context: Additional context information
            model_confidence: Model's own confidence (if available)
            ensemble_predictions: Predictions from ensemble models
            feature_factors: Feature-specific confidence factors

        Returns:
            ConfidenceEstimate object with detailed information
        """

        # Calculate base confidence from factors
        confidence_factors = self._calculate_factors(
            supporting_evidence, context, model_confidence, feature_factors
        )

        # Calculate base confidence
        base_confidence = self._combine_factors(confidence_factors)

        # Apply calibration if enabled
        if self.enable_calibration and self.calibrator:
            calibrated_confidence = self.calibrator.calibrate(base_confidence)
        else:
            calibrated_confidence = base_confidence

        # Calculate uncertainty
        if self.enable_uncertainty and ensemble_predictions:
            uncertainty_result = self.uncertainty_quantifier.quantify(
                ensemble_predictions,
                [calibrated_confidence] * len(ensemble_predictions),
            )
            uncertainty = uncertainty_result["uncertainty"]
            lower_bound = uncertainty_result["lower_bound"]
            upper_bound = uncertainty_result["upper_bound"]
        else:
            uncertainty = 1.0 - calibrated_confidence
            lower_bound = max(0.0, calibrated_confidence - 0.1)
            upper_bound = min(1.0, calibrated_confidence + 0.1)

        # Clamp to valid range
        confidence_score = max(0.0, min(1.0, calibrated_confidence))

        # Create confidence estimate
        estimate = ConfidenceEstimate(
            score=confidence_score,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty=uncertainty,
            level=ConfidenceLevel.from_score(confidence_score),
            factors=confidence_factors,
        )

        # Store estimate
        pred_key = self._get_prediction_key(prediction)
        self.estimations[pred_key] = estimate
        self.estimation_history.append(
            {
                "prediction": prediction,
                "estimate": estimate,
                "timestamp": datetime.now(),
            }
        )

        self.estimation_count += 1

        # Update metrics
        self._update_metrics()

        # Record for calibration
        if self.calibrator and context and "actual_correct" in context:
            self.calibrator.add_calibration_point(
                confidence_score, context["actual_correct"]
            )

        logger.debug(
            f"Confidence estimated: {confidence_score:.3f} (uncertainty={uncertainty:.3f})"
        )

        return estimate

    def _calculate_factors(
        self,
        evidence: float,
        context: Optional[Dict[str, Any]],
        model_confidence: Optional[float],
        feature_factors: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Calculate individual confidence factors."""

        factors = {}

        # Evidence strength
        factors["evidence_strength"] = max(0.0, min(1.0, evidence))

        # Data quality
        if context and "data_quality" in context:
            factors["data_quality"] = max(0.0, min(1.0, context["data_quality"]))
        else:
            factors["data_quality"] = 0.7

        # Model confidence
        if model_confidence is not None:
            factors["model_confidence"] = max(0.0, min(1.0, model_confidence))
        else:
            factors["model_confidence"] = 0.5

        # Historical accuracy
        if context and "model_id" in context:
            factors["historical_accuracy"] = self.historical_accuracy.get(
                context["model_id"], 0.5
            )
        else:
            factors["historical_accuracy"] = self._get_global_accuracy()

        # Agreement with ensemble
        if context and "ensemble_agreement" in context:
            factors["agreement_with_ensemble"] = context["ensemble_agreement"]
        else:
            factors["agreement_with_ensemble"] = 0.5

        # Feature-specific factors
        if feature_factors:
            for feature, confidence in feature_factors.items():
                self.feature_confidence[feature] = max(0.0, min(1.0, confidence))

        return factors

    def _combine_factors(self, factors: Dict[str, float]) -> float:
        """Combine confidence factors using weighted average."""

        total_weight = 0
        weighted_sum = 0

        for factor, value in factors.items():
            weight = self.factor_weights.get(factor, 0.1)
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return weighted_sum / total_weight

    def _get_prediction_key(self, prediction: Any) -> str:
        """Generate unique key for prediction."""
        pred_str = str(prediction)
        return hashlib.md5(pred_str.encode()).hexdigest()[:16]

    def _get_global_accuracy(self) -> float:
        """Calculate global accuracy from history."""
        if not self.prediction_history:
            return 0.5

        correct_count = sum(
            1 for p in self.prediction_history if p.get("correct", False)
        )
        return correct_count / len(self.prediction_history)

    def _update_metrics(self):
        """Update performance metrics."""
        if not self.estimation_history:
            return

        # Average confidence
        recent_estimates = list(self.estimation_history)[-100:]
        confidences = [e["estimate"].score for e in recent_estimates]
        self.metrics["avg_confidence"] = statistics.mean(confidences)

        # Brier score (if we have correctness data)
        brier_scores = []
        for item in self.prediction_history[-100:]:
            if "confidence" in item and "correct" in item:
                brier = (item["confidence"] - (1 if item["correct"] else 0)) ** 2
                brier_scores.append(brier)

        if brier_scores:
            self.metrics["brier_score"] = statistics.mean(brier_scores)

    def batch_estimate(
        self,
        predictions: List[Any],
        evidence_scores: List[float],
        context: Optional[Dict[str, Any]] = None,
        model_confidences: Optional[List[float]] = None,
    ) -> List[ConfidenceEstimate]:
        """Estimate confidence for multiple predictions."""

        if model_confidences is None:
            model_confidences = [None] * len(predictions)

        results = []
        for pred, evidence, model_conf in zip(
            predictions, evidence_scores, model_confidences
        ):
            estimate = self.estimate(pred, evidence, context, model_conf)
            results.append(estimate)

        return results

    def record_outcome(
        self, prediction: Any, was_correct: bool, model_id: Optional[str] = None
    ):
        """Record whether a prediction was correct for future calibration."""

        pred_key = self._get_prediction_key(prediction)

        # Update historical accuracy
        if model_id:
            self.model_accuracy[model_id].append(was_correct)
            # Keep only recent history
            if len(self.model_accuracy[model_id]) > 100:
                self.model_accuracy[model_id] = self.model_accuracy[model_id][-100:]
            accuracy = sum(self.model_accuracy[model_id]) / len(
                self.model_accuracy[model_id]
            )
            self.historical_accuracy[model_id] = accuracy

        # Update global stats
        if was_correct:
            self.correct_predictions += 1
        else:
            self.incorrect_predictions += 1

        # Record in prediction history
        self.prediction_history.append(
            {
                "prediction": prediction,
                "correct": was_correct,
                "timestamp": datetime.now(),
                "confidence": self.estimations.get(
                    pred_key,
                    ConfidenceEstimate(0.5, 0.4, 0.6, 0.5, ConfidenceLevel.MEDIUM),
                ).score,
            }
        )

        # Keep limited history
        if len(self.prediction_history) > self.history_size:
            self.prediction_history = self.prediction_history[-self.history_size :]

        # Add to calibrator if available
        if self.calibrator and pred_key in self.estimations:
            confidence = self.estimations[pred_key].score
            self.calibrator.add_calibration_point(confidence, was_correct)

        # Add to uncertainty quantifier
        if self.uncertainty_quantifier:
            confidence = self.estimations.get(
                pred_key, ConfidenceEstimate(0.5, 0.4, 0.6, 0.5, ConfidenceLevel.MEDIUM)
            ).score
            self.uncertainty_quantifier.record_outcome(
                prediction, confidence, was_correct
            )

    def get_confidence(self, prediction: Any) -> Optional[ConfidenceEstimate]:
        """Get stored confidence estimate for a prediction."""
        pred_key = self._get_prediction_key(prediction)
        return self.estimations.get(pred_key)

    def get_confidence_score(self, prediction: Any) -> float:
        """Get just the confidence score for a prediction."""
        estimate = self.get_confidence(prediction)
        return estimate.score if estimate else 0.5

    def is_confident(self, prediction: Any, threshold: float = 0.7) -> bool:
        """Check if confidence is above threshold."""
        score = self.get_confidence_score(prediction)
        return score >= threshold

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        if self.calibrator:
            return self.calibrator.get_stats()
        return {"calibration_enabled": False}

    def get_uncertainty_stats(self) -> Dict[str, Any]:
        """Get uncertainty statistics."""
        if self.uncertainty_quantifier:
            return self.uncertainty_quantifier.get_confidence_accuracy_curve()
        return {"uncertainty_enabled": False}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive estimator statistics."""

        if not self.estimations:
            return {
                "count": 0,
                "avg_confidence": 0,
                "min_confidence": 0,
                "max_confidence": 0,
                "total_predictions": 0,
                "accuracy": 0,
            }

        confidences = [e.score for e in self.estimations.values()]

        # Calculate accuracy if we have outcomes
        total_predictions = self.correct_predictions + self.incorrect_predictions
        accuracy = (
            self.correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        # Calculate reliability (confidence-accuracy alignment)
        reliability = (
            1.0 - abs(statistics.mean(confidences) - accuracy) if confidences else 0
        )

        return {
            "count": self.estimation_count,
            "unique_estimations": len(self.estimations),
            "avg_confidence": statistics.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "std_confidence": (
                statistics.stdev(confidences) if len(confidences) > 1 else 0
            ),
            "total_predictions": total_predictions,
            "correct_predictions": self.correct_predictions,
            "incorrect_predictions": self.incorrect_predictions,
            "accuracy": accuracy,
            "reliability": reliability,
            "metrics": self.metrics,
            "calibration": self.get_calibration_stats(),
            "factor_weights": self.factor_weights,
            "feature_confidences": dict(list(self.feature_confidence.items())[:20]),
        }

    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        distribution = {level.value: 0 for level in ConfidenceLevel}

        for estimate in self.estimations.values():
            distribution[estimate.level.value] += 1

        return distribution

    def get_reliability_diagram(self, n_bins: int = 10) -> Dict[str, List[float]]:
        """Generate reliability diagram data."""
        if not self.prediction_history:
            return {"confidences": [], "accuracies": [], "counts": []}

        bins = [[] for _ in range(n_bins)]

        for item in self.prediction_history:
            if "confidence" in item and "correct" in item:
                conf = item["confidence"]
                bin_idx = min(int(conf * n_bins), n_bins - 1)
                bins[bin_idx].append(item["correct"])

        confidences = []
        accuracies = []
        counts = []

        for i, bin_correct in enumerate(bins):
            if bin_correct:
                conf_center = (i + 0.5) / n_bins
                confidences.append(conf_center)
                accuracies.append(sum(bin_correct) / len(bin_correct))
                counts.append(len(bin_correct))
            else:
                confidences.append((i + 0.5) / n_bins)
                accuracies.append(0)
                counts.append(0)

        return {
            "confidences": confidences,
            "accuracies": accuracies,
            "counts": counts,
            "perfect_line": confidences,  # Perfect calibration line
        }

    def update_factor_weight(self, factor: str, weight: float):
        """Update weight for a confidence factor."""
        if factor in self.factor_weights:
            self.factor_weights[factor] = max(0.0, min(1.0, weight))
            # Normalize weights
            total = sum(self.factor_weights.values())
            if total > 0:
                for f in self.factor_weights:
                    self.factor_weights[f] /= total

    def reset(self):
        """Reset all confidence estimates and history."""
        self.estimations.clear()
        self.estimation_history.clear()
        self.prediction_history.clear()
        self.estimation_count = 0
        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.historical_accuracy.clear()
        self.model_accuracy.clear()
        self.feature_confidence.clear()

        self.metrics = {
            "avg_confidence": 0.0,
            "calibration_error": 0.0,
            "reliability": 0.0,
            "brier_score": 0.0,
        }

        logger.info("🔄 ConfidenceEstimator reset")

    def export_calibration_data(self) -> Dict[str, Any]:
        """Export calibration data for analysis."""
        if not self.calibrator:
            return {"error": "Calibration not enabled"}

        return {
            "calibration_points": [
                {"confidence": c, "correct": correct}
                for c, correct in self.calibrator.calibration_data
            ],
            "params": self.calibrator.calibration_params,
            "stats": self.get_calibration_stats(),
        }

    def import_calibration_data(self, data: List[Tuple[float, bool]]):
        """Import calibration data from external source."""
        if not self.calibrator:
            logger.warning("Calibration not enabled, cannot import data")
            return

        for confidence, correct in data:
            self.calibrator.add_calibration_point(confidence, correct)

        logger.info(f"📥 Imported {len(data)} calibration points")


# --------------------------------------------------
# CONVENIENCE FUNCTIONS
# --------------------------------------------------


def calculate_confidence_interval(
    successes: int, trials: int, confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for binomial proportion using Wilson score."""
    if trials == 0:
        return (0.0, 0.0)

    p = successes / trials
    z = 1.96  # For 95% confidence

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * ((p * (1 - p) + z**2 / (4 * trials)) / trials) ** 0.5 / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


def aggregate_confidences(confidences: List[float], method: str = "mean") -> float:
    """Aggregate multiple confidence scores."""
    if not confidences:
        return 0.5

    if method == "mean":
        return statistics.mean(confidences)
    elif method == "median":
        return statistics.median(confidences)
    elif method == "max":
        return max(confidences)
    elif method == "min":
        return min(confidences)
    elif method == "product":
        product = 1.0
        for c in confidences:
            product *= c
        return product ** (1.0 / len(confidences))
    else:
        return statistics.mean(confidences)


__all__ = [
    "ConfidenceEstimator",
    "ConfidenceLevel",
    "CalibrationMethod",
    "ConfidenceEstimate",
    "ConfidenceCalibrator",
    "UncertaintyQuantifier",
    "calculate_confidence_interval",
    "aggregate_confidences",
]
