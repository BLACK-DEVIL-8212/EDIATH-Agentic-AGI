"""Online Learning - incremental learning from streaming data with adaptive algorithms, concept drift detection, and real-time optimization."""

import asyncio
import json
import math
import statistics
from typing import Dict, Any, Optional, List, Callable, Union, Deque
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import random
import time

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..utils.logger import logger


class LearningMode(Enum):
    """Learning modes for different scenarios."""

    STREAMING = "streaming"  # Continuous streaming data
    BATCH = "batch"  # Batch processing
    ADAPTIVE = "adaptive"  # Adaptive batch size
    REAL_TIME = "real_time"  # Real-time learning
    INCREMENTAL = "incremental"  # One sample at a time


class UpdateStrategy(Enum):
    """Strategies for model updates."""

    GRADIENT_DESCENT = "gradient_descent"
    STOCHASTIC = "stochastic"
    MINI_BATCH = "mini_batch"
    ONLINE_NEWTON = "online_newton"
    ADAPTIVE = "adaptive"


class ConceptDriftStrategy(Enum):
    """Strategies for handling concept drift."""

    DETECT_ONLY = "detect_only"
    RESET_MODEL = "reset_model"
    ADAPTIVE_LEARNING = "adaptive_learning"
    ENSEMBLE = "ensemble"
    WINDOWING = "windowing"


class SampleImportance(Enum):
    """Importance levels for samples."""

    NORMAL = 1.0
    IMPORTANT = 2.0
    CRITICAL = 3.0
    TEST = 0.5


@dataclass
class LearningSample:
    """Enhanced learning sample with metadata."""

    id: str
    features: Dict[str, Any]
    target: Any
    timestamp: datetime = field(default_factory=datetime.now)
    weight: float = 1.0
    importance: SampleImportance = SampleImportance.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    predicted: Optional[Any] = None
    error: float = 0.0


@dataclass
class ConceptDriftInfo:
    """Information about concept drift detection."""

    detected: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    severity: float = 0.0  # 0-1
    location: int = 0
    strategy_used: ConceptDriftStrategy = ConceptDriftStrategy.DETECT_ONLY
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineLearningMetrics:
    """Comprehensive learning metrics."""

    total_samples: int = 0
    total_batches: int = 0
    current_loss: float = 0.0
    current_accuracy: float = 0.0
    moving_avg_loss: float = 0.0
    moving_avg_accuracy: float = 0.0
    learning_rate: float = 0.01
    adaptive_lr: float = 0.01
    concept_drift_count: int = 0
    last_concept_drift: Optional[datetime] = None
    average_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    convergence_rate: float = 0.0
    error_stddev: float = 0.0
    recent_errors: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


class AdaptiveBatchSize:
    """Dynamically adjust batch size based on performance."""

    def __init__(
        self,
        initial_size: int = 32,
        min_size: int = 1,
        max_size: int = 256,
        adjustment_interval: int = 10,
    ):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.adjustment_interval = adjustment_interval
        self.performance_history: Deque[float] = deque(maxlen=20)
        self.adjustments_made = 0

    def update(self, performance: float) -> int:
        """Update batch size based on performance."""
        self.performance_history.append(performance)

        if len(self.performance_history) >= self.adjustment_interval:
            recent_avg = statistics.mean(self.performance_history)
            old_avg = (
                statistics.mean(list(self.performance_history)[:-5])
                if len(self.performance_history) > 5
                else recent_avg
            )

            # Adjust based on performance trend
            if recent_avg > old_avg * 1.1:  # Improving
                self.current_size = min(self.max_size, self.current_size + 4)
                self.adjustments_made += 1
            elif recent_avg < old_avg * 0.9:  # Declining
                self.current_size = max(self.min_size, self.current_size - 2)
                self.adjustments_made += 1

            # Clear history for next cycle
            self.performance_history.clear()

        return self.current_size

    def get_stats(self) -> Dict[str, Any]:
        """Get batch size statistics."""
        return {
            "current_size": self.current_size,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "adjustments": self.adjustments_made,
        }


class ConceptDriftDetector:
    """Detect concept drift in streaming data."""

    def __init__(
        self, window_size: int = 100, threshold: float = 0.05, min_samples: int = 30
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples

        self.error_history: Deque[float] = deque(maxlen=window_size)
        self.detections: List[ConceptDriftInfo] = []
        self.drift_count = 0

    def add_error(self, error: float) -> Optional[ConceptDriftInfo]:
        """Add prediction error and check for drift."""
        self.error_history.append(error)

        if len(self.error_history) < self.min_samples:
            return None

        # Calculate statistics
        recent = list(self.error_history)[-min(20, len(self.error_history)) :]
        older = list(self.error_history)[:-20] if len(self.error_history) > 20 else []

        if len(older) < self.min_samples:
            return None

        recent_mean = statistics.mean(recent)
        recent_std = statistics.stdev(recent) if len(recent) > 1 else 0
        older_mean = statistics.mean(older)

        # Detect drift using statistical test
        drift_detected = False
        severity = 0.0

        if abs(recent_mean - older_mean) > self.threshold:
            drift_detected = True
            severity = min(1.0, abs(recent_mean - older_mean) / self.threshold)

        # Also check for increasing variance (gradual drift)
        if len(older) > 10 and len(recent) > 10:
            older_var = statistics.variance(older) if len(older) > 1 else 0
            recent_var = statistics.variance(recent) if len(recent) > 1 else 0

            if recent_var > older_var * 2:
                drift_detected = True
                severity = max(severity, min(1.0, recent_var / older_var / 2))

        if drift_detected:
            drift_info = ConceptDriftInfo(
                detected=True,
                timestamp=datetime.now(),
                severity=severity,
                location=len(self.error_history),
                metrics={
                    "recent_mean": recent_mean,
                    "older_mean": older_mean,
                    "recent_std": recent_std,
                    "mean_shift": abs(recent_mean - older_mean),
                },
            )
            self.detections.append(drift_info)
            self.drift_count += 1
            return drift_info

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get drift detection statistics."""
        if not self.detections:
            return {
                "drift_count": 0,
                "last_drift": None,
                "average_severity": 0,
                "drift_rate": 0,
            }

        severities = [d.severity for d in self.detections]

        return {
            "drift_count": self.drift_count,
            "last_drift": (
                self.detections[-1].timestamp.isoformat() if self.detections else None
            ),
            "average_severity": statistics.mean(severities),
            "max_severity": max(severities),
            "drift_rate": (
                self.drift_count / (len(self.error_history) / self.window_size)
                if self.error_history
                else 0
            ),
        }


class AdaptiveLearningRate:
    """Adaptively adjust learning rate based on gradient information."""

    def __init__(
        self,
        initial_rate: float = 0.01,
        min_rate: float = 0.0001,
        max_rate: float = 0.1,
        decay: float = 0.99,
        momentum: float = 0.9,
    ):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.decay = decay
        self.momentum = momentum
        self.gradient_history: Deque[float] = deque(maxlen=10)
        self.updates_count = 0

    def update(self, gradient: float) -> float:
        """Update learning rate based on gradient."""
        self.gradient_history.append(abs(gradient))
        self.updates_count += 1

        if len(self.gradient_history) >= 5:
            # Calculate gradient statistics
            avg_gradient = statistics.mean(self.gradient_history)
            grad_variance = (
                statistics.variance(self.gradient_history)
                if len(self.gradient_history) > 1
                else 0
            )

            # AdaGrad-like adjustment
            if avg_gradient > 0.1:
                self.current_rate *= 0.95  # Decrease if gradients are large
            elif avg_gradient < 0.01:
                self.current_rate *= 1.05  # Increase if gradients are small

            # Apply decay
            self.current_rate *= self.decay

            # Clamp to bounds
            self.current_rate = max(
                self.min_rate, min(self.max_rate, self.current_rate)
            )

        return self.current_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get learning rate statistics."""
        return {
            "current_rate": self.current_rate,
            "updates": self.updates_count,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "avg_gradient": (
                statistics.mean(self.gradient_history) if self.gradient_history else 0
            ),
        }


class OnlineLearner:
    """Advanced online learner with adaptive algorithms and drift detection."""

    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mode: LearningMode = LearningMode.ADAPTIVE,
        update_strategy: UpdateStrategy = UpdateStrategy.ADAPTIVE,
        enable_drift_detection: bool = True,
        enable_adaptive_lr: bool = True,
        enable_importance_sampling: bool = True,
        memory_size: int = 1000,
        decay_rate: float = 0.99,
        persistence_path: str = "data/online_learning",
    ):
        """Initialize advanced online learner."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mode = mode
        self.update_strategy = update_strategy
        self.enable_drift_detection = enable_drift_detection
        self.enable_adaptive_lr = enable_adaptive_lr
        self.enable_importance_sampling = enable_importance_sampling
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.persistence_path = Path(persistence_path)

        # Core learning state
        self.samples_processed = 0
        self.current_batch: List[LearningSample] = []
        self.sample_memory: Deque[LearningSample] = deque(maxlen=memory_size)

        # Metrics
        self.metrics = OnlineLearningMetrics()
        self.error_buffer: Deque[float] = deque(maxlen=100)
        self.performance_history: List[Dict[str, Any]] = []

        # Adaptive components
        self.adaptive_batch = AdaptiveBatchSize(initial_size=batch_size)
        self.drift_detector = ConceptDriftDetector() if enable_drift_detection else None
        self.adaptive_lr = (
            AdaptiveLearningRate(initial_rate=learning_rate)
            if enable_adaptive_lr
            else None
        )

        # Model state (simulated)
        self.model_weights: Dict[str, float] = {}
        self.model_version = 0

        # Learning statistics
        self.learning_stats: Dict[str, Any] = {
            "loss_history": [],
            "accuracy_history": [],
            "learning_curve": [],
            "concept_drifts": [],
        }

        # Callbacks
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._drift_callbacks: List[Callable[[ConceptDriftInfo], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Persistence
        self.auto_save = True
        if persistence_path:
            self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Async task
        self._save_task: Optional[asyncio.Task] = None
        self._running = False

        # Performance tracking
        self._last_sample_time = datetime.now()
        self._sample_times: Deque[float] = deque(maxlen=100)

        logger.info(
            f"✅ Advanced OnlineLearner initialized (mode={mode.value}, batch_size={batch_size}, drift_detection={enable_drift_detection})"
        )

    # #==================== Core Learning #====================

    async def learn_from_sample(
        self,
        sample: Union[Dict[str, Any], LearningSample],
        target: Any = None,
        weight: float = 1.0,
        importance: SampleImportance = SampleImportance.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Learn from single sample with advanced options."""

        start_time = time.time()

        # Create learning sample
        if isinstance(sample, LearningSample):
            learning_sample = sample
        else:
            learning_sample = LearningSample(
                id=f"sample_{self.samples_processed + 1}",
                features=sample,
                target=target,
                weight=weight,
                importance=importance,
                metadata=metadata or {},
            )

        # Apply importance weighting
        if self.enable_importance_sampling:
            importance_weight = importance.value
            learning_sample.weight *= importance_weight

        # Store in memory
        self.current_batch.append(learning_sample)
        self.sample_memory.append(learning_sample)
        self.samples_processed += 1

        # Update performance metrics
        elapsed = (time.time() - start_time) * 1000
        self._sample_times.append(elapsed)

        # Check if batch is ready
        batch_ready = False
        batch_result = None

        if self.mode == LearningMode.REAL_TIME:
            batch_ready = True
        elif self.mode == LearningMode.INCREMENTAL:
            batch_ready = True
        elif len(self.current_batch) >= self._get_target_batch_size():
            batch_ready = True

        if batch_ready:
            batch_result = await self._process_batch()

        # Update metrics
        await self._update_metrics()

        # Check for concept drift
        if self.enable_drift_detection and self.metrics.current_loss > 0.5:
            drift_info = await self._check_concept_drift()
            if drift_info and drift_info.detected:
                await self._handle_concept_drift(drift_info)

        return batch_result

    def _get_target_batch_size(self) -> int:
        """Get current target batch size based on mode."""
        if self.mode == LearningMode.ADAPTIVE:
            return self.adaptive_batch.current_size
        elif self.mode == LearningMode.REAL_TIME:
            return 1
        else:
            return self.batch_size

    async def _process_batch(self) -> Dict[str, Any]:
        """Process accumulated batch with advanced updates."""
        start_time = time.time()

        batch = self.current_batch
        self.current_batch = []

        if not batch:
            return {}

        # Simulate forward pass
        predictions = []
        errors = []

        for sample in batch:
            # Simulate prediction (replace with actual model inference)
            prediction = self._predict(sample.features)
            error = self._calculate_error(prediction, sample.target)

            sample.predicted = prediction
            sample.error = error

            predictions.append(prediction)
            errors.append(error)

            # Update metrics
            self.error_buffer.append(error)
            self.metrics.recent_errors.append(error)

        # Calculate batch metrics
        batch_loss = statistics.mean(errors) if errors else 0
        batch_accuracy = 1 - batch_loss

        # Update learning rate
        effective_lr = self.learning_rate
        if self.enable_adaptive_lr and self.adaptive_lr:
            avg_gradient = batch_loss  # Simplified gradient
            effective_lr = self.adaptive_lr.update(avg_gradient)
            self.metrics.adaptive_lr = effective_lr
        else:
            effective_lr = self.learning_rate * (
                self.decay_rate ** (self.samples_processed / 1000)
            )

        # Update model (simulated)
        await self._update_model(batch, effective_lr)

        # Calculate batch metrics
        processing_time = (time.time() - start_time) * 1000

        # Update overall metrics
        self.metrics.total_batches += 1
        self.metrics.current_loss = batch_loss
        self.metrics.current_accuracy = batch_accuracy

        # Update moving averages
        alpha = 0.1  # Smoothing factor
        self.metrics.moving_avg_loss = (
            alpha * batch_loss + (1 - alpha) * self.metrics.moving_avg_loss
        )
        self.metrics.moving_avg_accuracy = (
            alpha * batch_accuracy + (1 - alpha) * self.metrics.moving_avg_accuracy
        )

        # Update average latency
        total_avg = (
            self.metrics.average_latency_ms * (self.metrics.total_batches - 1)
            + processing_time
        )
        self.metrics.average_latency_ms = (
            total_avg / self.metrics.total_batches
            if self.metrics.total_batches > 0
            else processing_time
        )

        # Calculate throughput
        if self._sample_times:
            avg_sample_time = statistics.mean(self._sample_times)
            self.metrics.throughput_per_second = (
                1000 / avg_sample_time if avg_sample_time > 0 else 0
            )

        # Update error standard deviation
        if len(self.error_buffer) > 1:
            self.metrics.error_stddev = statistics.stdev(self.error_buffer)

        # Calculate convergence rate
        if len(self.learning_stats["loss_history"]) > 10:
            recent_losses = self.learning_stats["loss_history"][-10:]
            if recent_losses[0] > 0:
                self.metrics.convergence_rate = (
                    recent_losses[0] - batch_loss
                ) / recent_losses[0]

        # Store history
        self.learning_stats["loss_history"].append(batch_loss)
        self.learning_stats["accuracy_history"].append(batch_accuracy)

        # Track performance
        self.performance_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(batch),
                "loss": batch_loss,
                "accuracy": batch_accuracy,
                "learning_rate": effective_lr,
                "processing_time_ms": processing_time,
            }
        )

        # Limit history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

        # Trigger callbacks
        result = {
            "batch_size": len(batch),
            "samples_processed": self.samples_processed,
            "loss": batch_loss,
            "accuracy": batch_accuracy,
            "learning_rate": effective_lr,
            "processing_time_ms": processing_time,
            "concept_drift_detected": False,
        }

        await self._trigger_callbacks(result)

        logger.debug(
            f"📊 Batch processed: size={len(batch)}, loss={batch_loss:.4f}, acc={batch_accuracy:.4f}, lr={effective_lr:.5f}"
        )

        return result

    def _predict(self, features: Dict[str, Any]) -> Any:
        """Make prediction using current model (simulated)."""
        # Simple linear combination for demonstration
        if not self.model_weights:
            # Initialize random weights
            for key in features.keys():
                self.model_weights[key] = random.uniform(-1, 1)

        # Linear prediction
        prediction = 0.0
        for key, value in features.items():
            if key in self.model_weights:
                prediction += (
                    self.model_weights[key] * float(value)
                    if isinstance(value, (int, float))
                    else 0
                )

        # Sigmoid activation for classification
        prediction = 1 / (1 + math.exp(-prediction))

        return prediction

    def _calculate_error(self, prediction: Any, target: Any) -> float:
        """Calculate prediction error."""
        if isinstance(target, (int, float)):
            # Regression
            error = abs(prediction - float(target))
        else:
            # Classification
            error = 0.0 if prediction == target else 1.0

        return min(1.0, error)

    async def _update_model(self, batch: List[LearningSample], learning_rate: float):
        """Update model weights (simulated)."""
        # Simulate weight update based on errors
        for sample in batch:
            if sample.error > 0:
                # Adjust weights based on features
                for key, value in sample.features.items():
                    if key in self.model_weights and isinstance(value, (int, float)):
                        adjustment = (
                            learning_rate * sample.error * sample.weight * value
                        )
                        self.model_weights[key] -= adjustment

        self.model_version += 1

    async def _update_metrics(self):
        """Update real-time metrics."""
        # Update learning rate in metrics
        if self.adaptive_lr:
            self.metrics.learning_rate = self.adaptive_lr.current_rate

    async def _check_concept_drift(self) -> Optional[ConceptDriftInfo]:
        """Check for concept drift in the data stream."""
        if not self.drift_detector or len(self.error_buffer) < 30:
            return None

        # Add recent errors to detector
        recent_errors = list(self.error_buffer)[-20:]
        drift_info = None

        for error in recent_errors:
            detected = self.drift_detector.add_error(error)
            if detected:
                drift_info = detected

        return drift_info

    async def _handle_concept_drift(self, drift_info: ConceptDriftInfo):
        """Handle detected concept drift."""
        logger.warning(
            f"⚠️ Concept drift detected! Severity: {drift_info.severity:.2f}"
        )

        self.metrics.concept_drift_count += 1
        self.metrics.last_concept_drift = drift_info.timestamp
        self.learning_stats["concept_drifts"].append(drift_info)

        # Apply strategy
        strategy = drift_info.strategy_used

        if strategy == ConceptDriftStrategy.RESET_MODEL:
            # Reset model weights
            self.model_weights.clear()
            self.model_version = 0
            logger.info("🔄 Model reset due to concept drift")

        elif strategy == ConceptDriftStrategy.ADAPTIVE_LEARNING:
            # Increase learning rate temporarily
            if self.adaptive_lr:
                self.adaptive_lr.current_rate = min(
                    self.adaptive_lr.max_rate, self.adaptive_lr.current_rate * 2
                )
            logger.info("📈 Increased learning rate to adapt to drift")

        elif strategy == ConceptDriftStrategy.WINDOWING:
            # Reduce memory window
            old_size = self.sample_memory.maxlen
            new_size = max(100, old_size // 2)
            self.sample_memory = deque(self.sample_memory, maxlen=new_size)
            logger.info(f"📉 Reduced memory window from {old_size} to {new_size}")

        # Trigger drift callbacks
        for callback in self._drift_callbacks:
            try:
                callback(drift_info)
            except Exception as e:
                logger.error(f"Drift callback error: {e}")

    # #==================== Batch Learning #====================

    async def learn_from_stream(
        self,
        samples: List[Union[Dict[str, Any], LearningSample]],
        targets: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
        importance: Optional[List[SampleImportance]] = None,
    ) -> List[Dict[str, Any]]:
        """Learn from data stream with batch processing."""

        if targets and len(samples) != len(targets):
            raise ValueError("samples and targets must have the same length")

        results = []

        for i, sample in enumerate(samples):
            target = targets[i] if targets else None
            weight = weights[i] if weights and i < len(weights) else 1.0
            imp = (
                importance[i]
                if importance and i < len(importance)
                else SampleImportance.NORMAL
            )

            result = await self.learn_from_sample(sample, target, weight, imp)
            if result:
                results.append(result)

        # Process remaining batch
        if self.current_batch:
            result = await self._process_batch()
            results.append(result)

        return results

    async def learn_continuously(
        self,
        sample_generator: Callable,
        max_samples: Optional[int] = None,
        stop_condition: Optional[Callable] = None,
    ):
        """Continuously learn from a sample generator."""

        samples_learned = 0

        while self._running:
            try:
                # Get next sample
                sample = (
                    await sample_generator()
                    if asyncio.iscoroutinefunction(sample_generator)
                    else sample_generator()
                )

                if sample is None:
                    break

                # Learn from sample
                result = await self.learn_from_sample(sample)

                samples_learned += 1

                # Check stop conditions
                if max_samples and samples_learned >= max_samples:
                    break

                if stop_condition and stop_condition():
                    break

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Continuous learning error: {e}")
                for callback in self._error_callbacks:
                    try:
                        callback(e)
                    except Exception:
                        pass
                await asyncio.sleep(0.1)

    # #==================== Model Management #====================

    def get_current_model(self) -> Dict[str, Any]:
        """Get current model state."""
        return {
            "version": self.model_version,
            "weights": self.model_weights.copy(),
            "samples_processed": self.samples_processed,
            "metrics": {
                "loss": self.metrics.current_loss,
                "accuracy": self.metrics.current_accuracy,
            },
        }

    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curve data."""
        return {
            "loss": self.learning_stats["loss_history"][-100:],
            "accuracy": self.learning_stats["accuracy_history"][-100:],
            "samples": list(range(len(self.learning_stats["loss_history"][-100:]))),
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics."""
        return {
            "samples_processed": self.samples_processed,
            "batch_size": self.batch_size,
            "adaptive_batch_size": (
                self.adaptive_batch.current_size
                if self.mode == LearningMode.ADAPTIVE
                else self.batch_size
            ),
            "batches_processed": self.metrics.total_batches,
            "metrics": {
                "current_loss": self.metrics.current_loss,
                "current_accuracy": self.metrics.current_accuracy,
                "moving_avg_loss": self.metrics.moving_avg_loss,
                "moving_avg_accuracy": self.metrics.moving_avg_accuracy,
                "learning_rate": self.metrics.learning_rate,
                "adaptive_lr": self.metrics.adaptive_lr,
                "error_stddev": self.metrics.error_stddev,
                "convergence_rate": self.metrics.convergence_rate,
            },
            "performance": {
                "average_latency_ms": self.metrics.average_latency_ms,
                "throughput_per_second": self.metrics.throughput_per_second,
                "samples_in_memory": len(self.sample_memory),
            },
            "drift_detection": (
                self.drift_detector.get_stats() if self.drift_detector else None
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive online learner statistics."""
        stats = {
            "samples_processed": self.samples_processed,
            "total_batches": self.metrics.total_batches,
            "current_loss": self.metrics.current_loss,
            "current_accuracy": self.metrics.current_accuracy,
            "moving_avg_loss": self.metrics.moving_avg_loss,
            "moving_avg_accuracy": self.metrics.moving_avg_accuracy,
            "learning_rate": self.metrics.learning_rate,
            "concept_drifts_detected": self.metrics.concept_drift_count,
            "model_version": self.model_version,
            "memory_usage": len(self.sample_memory),
            "batch_queue_size": len(self.current_batch),
            "mode": self.mode.value,
            "update_strategy": self.update_strategy.value,
            "enable_drift_detection": self.enable_drift_detection,
            "enable_adaptive_lr": self.enable_adaptive_lr,
            "enable_importance_sampling": self.enable_importance_sampling,
            "performance": {
                "avg_latency_ms": self.metrics.average_latency_ms,
                "throughput_per_second": self.metrics.throughput_per_second,
                "error_stddev": self.metrics.error_stddev,
                "convergence_rate": self.metrics.convergence_rate,
            },
        }

        # Add adaptive components stats
        if self.mode == LearningMode.ADAPTIVE:
            stats["adaptive_batch"] = self.adaptive_batch.get_stats()

        if self.enable_adaptive_lr and self.adaptive_lr:
            stats["adaptive_lr"] = self.adaptive_lr.get_stats()

        if self.drift_detector:
            stats["drift_detector"] = self.drift_detector.get_stats()

        return stats

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        if not self.performance_history:
            return {"message": "No performance data available"}

        recent_perf = self.performance_history[-100:]

        losses = [p["loss"] for p in recent_perf]
        accuracies = [p["accuracy"] for p in recent_perf]
        latencies = [p["processing_time_ms"] for p in recent_perf]

        return {
            "summary": {
                "total_batches": len(self.performance_history),
                "average_loss": statistics.mean(losses) if losses else 0,
                "best_loss": min(losses) if losses else 0,
                "worst_loss": max(losses) if losses else 0,
                "average_accuracy": statistics.mean(accuracies) if accuracies else 0,
                "best_accuracy": max(accuracies) if accuracies else 0,
                "average_latency_ms": statistics.mean(latencies) if latencies else 0,
            },
            "trend": {
                "loss_trend": self._calculate_trend(losses),
                "accuracy_trend": self._calculate_trend(accuracies),
                "improvement_rate": self.metrics.convergence_rate,
            },
            "recent_performance": recent_perf[-10:],
            "concept_drifts": len(self.learning_stats["concept_drifts"]),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"

        first_half = statistics.mean(values[: len(values) // 2])
        second_half = statistics.mean(values[len(values) // 2 :])

        if second_half < first_half * 0.95:
            return "improving"
        elif second_half > first_half * 1.05:
            return "declining"
        else:
            return "stable"

    # #==================== Persistence #====================

    async def save_state(self, filename: Optional[str] = None) -> bool:
        """Save learner state to disk."""
        if not filename:
            filename = f"online_learner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.persistence_path / filename

        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "samples_processed": self.samples_processed,
                "model_version": self.model_version,
                "model_weights": self.model_weights,
                "metrics": {
                    "total_batches": self.metrics.total_batches,
                    "current_loss": self.metrics.current_loss,
                    "current_accuracy": self.metrics.current_accuracy,
                    "moving_avg_loss": self.metrics.moving_avg_loss,
                    "moving_avg_accuracy": self.metrics.moving_avg_accuracy,
                    "concept_drift_count": self.metrics.concept_drift_count,
                    "average_latency_ms": self.metrics.average_latency_ms,
                    "throughput_per_second": self.metrics.throughput_per_second,
                },
                "learning_stats": {
                    "loss_history": self.learning_stats["loss_history"][-1000:],
                    "accuracy_history": self.learning_stats["accuracy_history"][-1000:],
                },
                "performance_history": self.performance_history[-1000:],
                "config": {
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "mode": self.mode.value,
                    "update_strategy": self.update_strategy.value,
                    "enable_drift_detection": self.enable_drift_detection,
                    "enable_adaptive_lr": self.enable_adaptive_lr,
                    "memory_size": self.memory_size,
                },
            }

            # Add adaptive component states
            if self.adaptive_batch:
                state["adaptive_batch"] = self.adaptive_batch.get_stats()

            if self.adaptive_lr:
                state["adaptive_lr"] = self.adaptive_lr.get_stats()

            with open(filepath, "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"💾 Saved online learner state to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    async def load_state(self, filepath: Optional[str] = None) -> bool:
        """Load learner state from disk."""
        if not filepath:
            files = sorted(
                self.persistence_path.glob("online_learner_*.json"), reverse=True
            )
            if not files:
                return False
            filepath = files[0]
        else:
            filepath = Path(filepath)

        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            self.samples_processed = state.get("samples_processed", 0)
            self.model_version = state.get("model_version", 0)
            self.model_weights = state.get("model_weights", {})

            metrics_state = state.get("metrics", {})
            self.metrics.total_batches = metrics_state.get("total_batches", 0)
            self.metrics.current_loss = metrics_state.get("current_loss", 0)
            self.metrics.current_accuracy = metrics_state.get("current_accuracy", 0)
            self.metrics.moving_avg_loss = metrics_state.get("moving_avg_loss", 0)
            self.metrics.moving_avg_accuracy = metrics_state.get(
                "moving_avg_accuracy", 0
            )
            self.metrics.concept_drift_count = metrics_state.get(
                "concept_drift_count", 0
            )
            self.metrics.average_latency_ms = metrics_state.get("average_latency_ms", 0)
            self.metrics.throughput_per_second = metrics_state.get(
                "throughput_per_second", 0
            )

            learning_stats = state.get("learning_stats", {})
            self.learning_stats["loss_history"] = learning_stats.get("loss_history", [])
            self.learning_stats["accuracy_history"] = learning_stats.get(
                "accuracy_history", []
            )

            self.performance_history = state.get("performance_history", [])

            logger.info(f"📂 Loaded online learner state from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    # #==================== Callbacks #====================

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for batch processing results."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def add_drift_callback(self, callback: Callable[[ConceptDriftInfo], None]):
        """Add callback for concept drift detection."""
        self._drift_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add error callback."""
        self._error_callbacks.append(callback)

    async def _trigger_callbacks(self, result: Dict[str, Any]):
        """Trigger all callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                for err_cb in self._error_callbacks:
                    try:
                        err_cb(e)
                    except Exception:
                        pass

    # #==================== Control #====================

    def reset(self):
        """Reset learner state."""
        self.samples_processed = 0
        self.current_batch.clear()
        self.sample_memory.clear()
        self.model_weights.clear()
        self.model_version = 0

        self.metrics = OnlineLearningMetrics()
        self.error_buffer.clear()
        self.performance_history.clear()

        self.learning_stats = {
            "loss_history": [],
            "accuracy_history": [],
            "learning_curve": [],
            "concept_drifts": [],
        }

        if self.adaptive_batch:
            self.adaptive_batch = AdaptiveBatchSize(initial_size=self.batch_size)

        if self.drift_detector:
            self.drift_detector = ConceptDriftDetector()

        if self.adaptive_lr:
            self.adaptive_lr = AdaptiveLearningRate(initial_rate=self.learning_rate)

        logger.info("🔄 Online learner reset")

    def set_mode(self, mode: LearningMode):
        """Change learning mode."""
        self.mode = mode
        logger.info(f"📊 Learning mode changed to {mode.value}")

    def set_learning_rate(self, learning_rate: float):
        """Set new learning rate."""
        self.learning_rate = max(0.0001, min(0.5, learning_rate))
        if self.adaptive_lr:
            self.adaptive_lr.current_rate = self.learning_rate
        logger.info(f"📈 Learning rate set to {self.learning_rate}")

    async def start(self):
        """Start background tasks."""
        self._running = True
        if self.auto_save:
            self._save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("🚀 Online learner started")

    async def stop(self):
        """Stop background tasks."""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        if self.auto_save:
            await self.save_state()

        logger.info("🛑 Online learner stopped")

    async def _auto_save_loop(self):
        """Auto-save loop."""
        while self._running:
            await asyncio.sleep(300)  # Save every 5 minutes
            if self.samples_processed > 0:
                await self.save_state()


# #==================== Convenience Functions #====================

__all__ = [
    "OnlineLearner",
    "LearningMode",
    "UpdateStrategy",
    "ConceptDriftStrategy",
    "SampleImportance",
    "LearningSample",
    "ConceptDriftInfo",
    "OnlineLearningMetrics",
    "AdaptiveBatchSize",
    "ConceptDriftDetector",
    "AdaptiveLearningRate",
]
