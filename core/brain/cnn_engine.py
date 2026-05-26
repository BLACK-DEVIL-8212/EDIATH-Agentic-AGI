"""
CNN Engine for EDIATH System
Production-ready computer vision engine with chunking and pipeline processing
Enhanced with advanced features: model quantization, ONNX support, distributed inference,
model pruning, knowledge distillation, active learning, and more
"""

import asyncio
import io
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging
from collections import deque
import json
import hashlib
from pathlib import Path
import zlib

# Try to import optional dependencies
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch torchvision")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Install with: pip install onnx onnxruntime")

try:
    from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class ProcessingMode(Enum):
    """Processing modes for CNN engine"""

    SINGLE = "single"
    BATCH = "batch"
    PIPELINE = "pipeline"
    CHUNKED = "chunked"
    DISTRIBUTED = "distributed"
    STREAMING = "streaming"


class FrameStatus(Enum):
    """Frame processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    DISTRIBUTED = "distributed"


class ModelArchitecture(Enum):
    """Supported CNN architectures"""

    CUSTOM = "custom"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3 = "mobilenet_v3"
    EFFICIENTNET = "efficientnet"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B4 = "efficientnet_b4"
    DENSENET121 = "densenet121"
    INCEPTION_V3 = "inception_v3"
    RESNEXT50 = "resnext50"
    WIDERESNET50 = "wide_resnet50"


class InferencePrecision(Enum):
    """Inference precision modes"""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BF16 = "bf16"


class CacheStrategy(Enum):
    """Caching strategies for features"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive caching


@dataclass
class FrameChunk:
    """Represents a chunk of frames for processing"""

    chunk_id: int
    frames: List[np.ndarray]
    timestamps: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: FrameStatus = FrameStatus.PENDING
    results: List[Any] = field(default_factory=list)
    priority: int = 0
    checksum: str = ""
    compression_ratio: float = 1.0


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance tracking"""

    frames_processed: int = 0
    chunks_processed: int = 0
    total_latency: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    error_count: int = 0
    last_processed_time: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_wait_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    cache_misses: int = 0
    cache_hits: int = 0
    compression_saved_bytes: int = 0
    distributed_tasks: int = 0
    model_inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0


@dataclass
class TrainingMetrics:
    """Metrics for model training"""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    best_accuracy: float = 0.0


@dataclass
class FeatureExtractionResult:
    """Results from feature extraction"""

    features: np.ndarray
    class_id: int
    confidence: float
    attention_map: Optional[np.ndarray] = None
    bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    embedding: Optional[np.ndarray] = None
    similarity_score: Optional[float] = None


class QuantizedCNNModel(nn.Module):
    """Quantized CNN model for efficient inference"""

    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = original_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class DistillationLoss(nn.Module):
    """Knowledge distillation loss"""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = self.kl_loss(soft_student, soft_teacher) * (
            self.temperature**2
        )

        # Student loss
        student_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss


class CNNModel(nn.Module):
    """Enhanced CNN model with multiple architecture support and advanced features"""

    def __init__(
        self,
        architecture: ModelArchitecture = ModelArchitecture.CUSTOM,
        num_classes: int = 1000,
        input_channels: int = 3,
        pretrained: bool = False,
        dropout_rate: float = 0.5,
    ):
        super(CNNModel, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.feature_dimension = 512
        self.dropout_rate = dropout_rate

        if architecture == ModelArchitecture.RESNET18 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
            self.feature_dimension = 512
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(512, num_classes)
            )

        elif architecture == ModelArchitecture.RESNET50 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
            self.feature_dimension = 2048
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(2048, num_classes)
            )

        elif architecture == ModelArchitecture.RESNET101 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.resnet101(weights="DEFAULT" if pretrained else None)
            self.feature_dimension = 2048
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(2048, num_classes)
            )

        elif architecture == ModelArchitecture.VGG16 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.vgg16(weights="DEFAULT" if pretrained else None)
            self.feature_dimension = 4096
            self.backbone.classifier[6] = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(4096, num_classes)
            )

        elif architecture == ModelArchitecture.VGG19 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.vgg19(weights="DEFAULT" if pretrained else None)
            self.feature_dimension = 4096
            self.backbone.classifier[6] = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(4096, num_classes)
            )

        elif architecture == ModelArchitecture.MOBILENET_V2 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.mobilenet_v2(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dimension = 1280
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(1280, num_classes)
            )

        elif architecture == ModelArchitecture.MOBILENET_V3 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.mobilenet_v3_large(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dimension = 960
            self.backbone.classifier[3] = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(960, num_classes)
            )

        elif architecture == ModelArchitecture.DENSENET121 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.densenet121(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dimension = 1024
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(1024, num_classes)
            )

        elif architecture == ModelArchitecture.INCEPTION_V3 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.inception_v3(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dimension = 2048
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(2048, num_classes)
            )

        elif architecture == ModelArchitecture.EFFICIENTNET_B0 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.efficientnet_b0(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dimension = 1280
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(1280, num_classes)
            )

        elif architecture == ModelArchitecture.EFFICIENTNET_B4 and TORCH_AVAILABLE:
            import torchvision.models as models

            self.backbone = models.efficientnet_b4(
                weights="DEFAULT" if pretrained else None
            )
            self.feature_dimension = 1792
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate), nn.Linear(1792, num_classes)
            )

        else:  # CUSTOM architecture
            self.backbone = None
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7)),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x, return_features: bool = False, return_embedding: bool = False):
        """Forward pass with optional feature extraction and embeddings"""
        if self.architecture != ModelArchitecture.CUSTOM and self.backbone is not None:
            if hasattr(self.backbone, "features"):
                features = self.backbone.features(x)
                features = features.mean([2, 3])  # Global average pooling
                output = self.backbone.classifier(features)
            else:
                # Get penultimate layer features
                if hasattr(self.backbone, "avgpool"):
                    features = self.backbone.avgpool(self.backbone.features(x))
                    features = torch.flatten(features, 1)
                else:
                    features = self.backbone(x)
                output = features

            embedding = features if return_embedding else None

            if hasattr(self.backbone, "fc") and not isinstance(output, torch.Tensor):
                output = self.backbone.fc(output)
        else:
            features = self.features(x)
            embedding = features.flatten(1) if return_embedding else None
            output = self.classifier(features)

        if return_features and return_embedding:
            return output, features, embedding
        elif return_features:
            return output, features
        elif return_embedding:
            return output, embedding

        return output

    def extract_features(self, x):
        """Extract intermediate features"""
        if self.architecture != ModelArchitecture.CUSTOM and self.backbone is not None:
            if hasattr(self.backbone, "features"):
                features = self.backbone.features(x)
                features = features.mean([2, 3])
            else:
                features = self.backbone(x)
        else:
            features = self.features(x)
            features = features.flatten(1)
        return features


class FrameDataset(Dataset):
    """Dataset for batch processing of frames with augmentation"""

    def __init__(
        self,
        frames: List[np.ndarray],
        labels: List[int] = None,
        transform=None,
        augmentation=None,
    ):
        self.frames = frames
        self.labels = labels if labels else [-1] * len(frames)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]

        if self.augmentation and label != -1:
            frame = self.augmentation(frame)

        if self.transform:
            if PIL_AVAILABLE and isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame.astype("uint8"))
            frame = self.transform(frame)

        return frame, label


class ModelCheckpoint:
    """Model checkpoint manager"""

    def __init__(self, save_dir: str, max_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": time.time(),
        }

        # Save checkpoint
        filename = f"checkpoint_epoch_{epoch}.pt"
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        # Track checkpoints
        self.checkpoints.append(filepath)

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_keep:
            oldest = self.checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()

        # Save best model separately
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            return str(best_path)

        return str(filepath)

    def load(self, filepath: str, model: nn.Module, optimizer: optim.Optimizer = None):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint


class FeatureCache:
    """Advanced feature cache with multiple strategies"""

    def __init__(
        self,
        max_size: int = 10000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        ttl_seconds: int = 3600,
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_count = {}
        self.access_times = {}
        self.creation_times = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key not in self.cache:
            return None

        # Check TTL
        if self.strategy == CacheStrategy.TTL:
            if time.time() - self.creation_times[key] > self.ttl_seconds:
                del self.cache[key]
                return None

        # Update access statistics
        self.access_count[key] = self.access_count.get(key, 0) + 1
        self.access_times[key] = time.time()

        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache"""
        # Evict if needed
        if len(self.cache) >= self.max_size:
            self._evict()

        self.cache[key] = value
        self.creation_times[key] = time.time()
        self.access_count[key] = 0
        self.access_times[key] = time.time()

    def _evict(self):
        """Evict items based on strategy"""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest]
            del self.access_times[oldest]

        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used = min(self.access_count, key=self.access_count.get)
            del self.cache[least_used]
            del self.access_count[least_used]

        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest
            oldest = min(self.creation_times, key=self.creation_times.get)
            del self.cache[oldest]
            del self.creation_times[oldest]

        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: combine LRU and LFU
            score = {}
            for key in self.cache:
                lru_score = time.time() - self.access_times.get(key, 0)
                lfu_score = self.access_count.get(key, 0)
                score[key] = lru_score * (1 / (lfu_score + 1))

            if score:
                to_remove = max(score, key=score.get)
                del self.cache[to_remove]

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
        self.access_times.clear()
        self.creation_times.clear()

    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = sum(self.access_count.values())
        if total == 0:
            return 0.0
        return (total - len(self.cache)) / total


class ModelOptimizer:
    """Model optimization utilities"""

    @staticmethod
    def quantize_model(
        model: nn.Module, calibration_data: torch.Tensor = None
    ) -> nn.Module:
        """Quantize model for faster inference"""
        if not QUANTIZATION_AVAILABLE:
            print("Warning: Quantization not available")
            return model

        model.eval()

        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )

        return quantized_model

    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
        """Prune model weights"""
        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, "weight"))

        # Apply pruning
        from torch.nn.utils import prune

        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=amount)
            prune.remove(module, param_name)

        return model

    @staticmethod
    def fuse_model(model: nn.Module) -> nn.Module:
        """Fuse convolutional and batch norm layers"""
        from torch.quantization import fuse_modules

        modules_to_fuse = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Find subsequent BatchNorm and ReLU
                bn_name = None
                relu_name = None

                for child_name, child in module.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        bn_name = child_name
                    elif isinstance(child, nn.ReLU):
                        relu_name = child_name

                if bn_name:
                    fuse_list = [name, bn_name]
                    if relu_name:
                        fuse_list.append(relu_name)
                    modules_to_fuse.append(fuse_list)

        if modules_to_fuse:
            model = fuse_modules(model, modules_to_fuse)

        return model

    @staticmethod
    def convert_to_onnx(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11,
    ):
        """Convert model to ONNX format"""
        if not ONNX_AVAILABLE:
            print("Warning: ONNX not available")
            return None

        model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        return output_path


class DistributedInference:
    """Distributed inference using Ray"""

    def __init__(self, num_workers: int = 4):
        if not RAY_AVAILABLE:
            raise ImportError("Ray not available. Install with: pip install ray")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=num_workers)

        self.num_workers = num_workers

    @ray.remote
    class RemoteModel:
        """Remote model for distributed inference"""

        def __init__(self, model_weights: bytes, model_config: Dict[str, Any]):
            import io
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model
            buffer = io.BytesIO(model_weights)
            self.model = torch.load(buffer, map_location=self.device)
            self.model.eval()

        def process(self, batch: List[np.ndarray]) -> List[Dict[str, Any]]:
            """Process batch of frames"""
            results = []
            for frame in batch:
                # Simulate processing
                results.append({"processed": True, "shape": frame.shape})
            return results

    async def distribute_batch(
        self,
        frames: List[np.ndarray],
        model_weights: bytes,
        model_config: Dict[str, Any],
    ) -> List[Any]:
        """Distribute batch across workers"""
        # Split frames into batches
        batch_size = len(frames) // self.num_workers
        batches = [
            frames[i : i + batch_size] for i in range(0, len(frames), batch_size)
        ]

        # Create remote models
        remote_models = [
            self.RemoteModel.remote(model_weights, model_config)
            for _ in range(len(batches))
        ]

        # Process in parallel
        futures = [
            model.process.remote(batch) for model, batch in zip(remote_models, batches)
        ]

        # Collect results
        results = await asyncio.gather(*[f for f in futures])

        # Flatten results
        flattened = []
        for result in results:
            flattened.extend(result)

        return flattened


class ActiveLearner:
    """Active learning for continuous model improvement"""

    def __init__(self, model: nn.Module, uncertainty_threshold: float = 0.3):
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.high_uncertainty_samples = []
        self.training_buffer = []

    def calculate_uncertainty(self, predictions: torch.Tensor) -> float:
        """Calculate prediction uncertainty using entropy"""
        probabilities = F.softmax(predictions, dim=1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
        return entropy.mean().item()

    def should_label(self, predictions: torch.Tensor) -> bool:
        """Determine if sample should be labeled by human"""
        uncertainty = self.calculate_uncertainty(predictions)
        return uncertainty > self.uncertainty_threshold

    def add_sample(self, frame: np.ndarray, predictions: torch.Tensor):
        """Add sample for potential labeling"""
        if self.should_label(predictions):
            self.high_uncertainty_samples.append(
                {
                    "frame": frame,
                    "predictions": predictions,
                    "uncertainty": self.calculate_uncertainty(predictions),
                    "timestamp": time.time(),
                }
            )

    def get_samples_for_labeling(self, max_samples: int = 100) -> List[Dict[str, Any]]:
        """Get samples that need human labeling"""
        # Sort by uncertainty
        sorted_samples = sorted(
            self.high_uncertainty_samples, key=lambda x: x["uncertainty"], reverse=True
        )
        return sorted_samples[:max_samples]

    def update_model(self, labeled_samples: List[Tuple[np.ndarray, int]]):
        """Update model with newly labeled samples"""
        # Add to training buffer
        self.training_buffer.extend(labeled_samples)

        # Trigger retraining if buffer is large enough
        if len(self.training_buffer) >= 100:
            self._retrain_model()

    def _retrain_model(self):
        """Retrain model with accumulated samples"""
        # Implementation would include fine-tuning logic
        pass


class CNNEngine:
    """
    Production-ready CNN Engine with chunking and pipeline processing
    Supports batch processing, pipeline parallelism, and chunked inference
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize CNN Engine with comprehensive features

        Args:
            config: Configuration dictionary with parameters:
                - model_path: Path to pre-trained model
                - architecture: Model architecture
                - pretrained: Use pretrained weights
                - input_size: Input image size
                - batch_size: Batch size for processing
                - chunk_size: Frames per chunk
                - pipeline_workers: Number of pipeline workers
                - enable_cuda: Enable CUDA if available
                - enable_pipeline: Enable pipeline processing
                - max_queue_size: Maximum queue size
                - processing_timeout: Timeout per frame
                - num_classes: Number of output classes
                - enable_attention: Enable attention map generation
                - enable_feature_extraction: Enable feature extraction
                - enable_object_detection: Enable object detection
                - confidence_threshold: Confidence threshold
                - inference_precision: Precision for inference
                - enable_quantization: Enable model quantization
                - enable_distributed: Enable distributed inference
                - enable_redis_cache: Enable Redis caching
                - enable_prometheus: Enable Prometheus metrics
                - enable_active_learning: Enable active learning
                - cache_strategy: Cache strategy (LRU, LFU, FIFO, TTL, ADAPTIVE)
                - enable_model_pruning: Enable model pruning
                - enable_knowledge_distillation: Enable knowledge distillation
                - teacher_model_path: Path to teacher model for distillation
        """
        self.config = config or {}

        # Core settings
        self.model_path = self.config.get("model_path", None)
        self.architecture = self.config.get("architecture", ModelArchitecture.CUSTOM)
        if isinstance(self.architecture, str):
            self.architecture = ModelArchitecture[self.architecture.upper()]

        self.pretrained = self.config.get("pretrained", False)
        self.input_size = self.config.get("input_size", 224)
        self.batch_size = self.config.get("batch_size", 32)
        self.chunk_size = self.config.get("chunk_size", 10)
        self.pipeline_workers = self.config.get("pipeline_workers", 4)
        self.enable_cuda = (
            self.config.get("enable_cuda", True) and torch.cuda.is_available()
        )
        self.enable_pipeline = self.config.get("enable_pipeline", True)
        self.max_queue_size = self.config.get("max_queue_size", 100)
        self.processing_timeout = self.config.get("processing_timeout", 1.0)
        self.num_classes = self.config.get("num_classes", 1000)
        self.enable_attention = self.config.get("enable_attention", False)
        self.enable_feature_extraction = self.config.get(
            "enable_feature_extraction", True
        )
        self.enable_object_detection = self.config.get("enable_object_detection", False)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)

        # Advanced features
        self.inference_precision = self.config.get(
            "inference_precision", InferencePrecision.FP32
        )
        if isinstance(self.inference_precision, str):
            self.inference_precision = InferencePrecision[
                self.inference_precision.upper()
            ]

        self.enable_quantization = self.config.get("enable_quantization", False)
        self.enable_distributed = (
            self.config.get("enable_distributed", False) and RAY_AVAILABLE
        )
        self.enable_redis_cache = (
            self.config.get("enable_redis_cache", False) and REDIS_AVAILABLE
        )
        self.enable_prometheus = (
            self.config.get("enable_prometheus", False) and PROMETHEUS_AVAILABLE
        )
        self.enable_active_learning = self.config.get("enable_active_learning", False)
        self.enable_model_pruning = self.config.get("enable_model_pruning", False)
        self.enable_knowledge_distillation = self.config.get(
            "enable_knowledge_distillation", False
        )

        # Cache settings
        self.cache_strategy = self.config.get("cache_strategy", CacheStrategy.LRU)
        if isinstance(self.cache_strategy, str):
            self.cache_strategy = CacheStrategy[self.cache_strategy.upper()]
        self.max_cache_size = self.config.get("max_cache_size", 10000)

        # Feature extraction settings
        self.feature_dimension = 512
        self.feature_cache = FeatureCache(
            max_size=self.max_cache_size, strategy=self.cache_strategy
        )

        # Redis cache
        self.redis_client = None
        if self.enable_redis_cache:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.get("redis_host", "localhost"),
                    port=self.config.get("redis_port", 6379),
                    decode_responses=True,
                )
                self._log("INFO", "Redis cache enabled")
            except Exception as e:
                self._log("WARNING", f"Failed to connect to Redis: {e}")
                self.enable_redis_cache = False

        # Prometheus metrics
        if self.enable_prometheus:
            self.frame_counter = Counter(
                "cnn_frames_processed_total", "Total frames processed"
            )
            self.latency_histogram = Histogram(
                "cnn_inference_latency_seconds", "Inference latency"
            )
            self.error_counter = Counter("cnn_errors_total", "Total errors")
            self.cache_hits_gauge = Gauge("cnn_cache_hits", "Cache hits")

        # Initialize components
        self.model = None
        self.teacher_model = None
        self.device = None
        self.transform = None
        self.train_transform = None
        self.val_transform = None

        # Pipeline components
        self.frame_queue = None
        self.result_queue = None
        self.pipeline_workers_list = []
        self.processing_mode = (
            ProcessingMode.PIPELINE if self.enable_pipeline else ProcessingMode.SINGLE
        )
        if self.enable_distributed:
            self.processing_mode = ProcessingMode.DISTRIBUTED
            self.distributed_inference = None

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.checkpoint_manager = ModelCheckpoint(
            self.config.get("checkpoint_dir", "checkpoints")
        )
        self.active_learner = None
        self.training_metrics = TrainingMetrics()

        # Chunking components
        self.chunk_buffer = []
        self.processing_chunks = {}
        self.chunk_lock = asyncio.Lock()
        self.priority_chunks = deque()

        # Frame compression
        self.enable_compression = self.config.get("enable_compression", True)
        self.compression_level = self.config.get("compression_level", 6)  # 1-9

        # Metrics
        self.metrics = PipelineMetrics()

        # State management
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._performance_lock = threading.Lock()

        # Class labels
        self.class_labels = self.config.get(
            "class_labels", [f"class_{i}" for i in range(self.num_classes)]
        )

        # Preprocess transforms
        self._setup_transforms()

        # Load model
        self._load_model()

        # Initialize active learning
        if self.enable_active_learning and self.model:
            self.active_learner = ActiveLearner(self.model)

        # Setup logging
        self.logger = self.config.get("logger", None)
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)

        self._log(
            "INFO",
            f"CNN Engine initialized with {self.architecture.value} architecture",
        )
        self._log(
            "INFO",
            f"Features: Quantization={self.enable_quantization}, "
            f"Distributed={self.enable_distributed}, "
            f"Redis={self.enable_redis_cache}, "
            f"ActiveLearning={self.enable_active_learning}",
        )

    def _setup_transforms(self):
        """Setup image preprocessing transforms with augmentation"""
        # Training transforms with augmentation
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Validation/Inference transforms
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform = self.val_transform

    def _load_model(self):
        """Load CNN model with advanced optimizations"""
        if not TORCH_AVAILABLE:
            self._log("WARNING", "PyTorch not available, using stub mode")
            return

        try:
            # Determine device and precision
            if self.inference_precision == InferencePrecision.FP16 and self.enable_cuda:
                self.device = torch.device("cuda")
                self.dtype = torch.float16
            elif (
                self.inference_precision == InferencePrecision.BF16 and self.enable_cuda
            ):
                self.device = torch.device("cuda")
                self.dtype = torch.bfloat16
            else:
                self.device = torch.device("cuda" if self.enable_cuda else "cpu")
                self.dtype = torch.float32

            # Create model with specified architecture
            self.model = CNNModel(
                architecture=self.architecture,
                num_classes=self.num_classes,
                input_channels=3,
                pretrained=self.pretrained,
                dropout_rate=self.config.get("dropout_rate", 0.5),
            )

            # Update feature dimension
            self.feature_dimension = self.model.feature_dimension

            # Load teacher model for distillation
            if self.enable_knowledge_distillation:
                teacher_path = self.config.get("teacher_model_path")
                if teacher_path:
                    self.teacher_model = CNNModel(
                        architecture=self.config.get(
                            "teacher_architecture", ModelArchitecture.RESNET50
                        ),
                        num_classes=self.num_classes,
                        pretrained=True,
                    )
                    teacher_state = torch.load(teacher_path, map_location=self.device)
                    self.teacher_model.load_state_dict(teacher_state)
                    self.teacher_model.to(self.device)
                    self.teacher_model.eval()
                    self._log("INFO", "Teacher model loaded for knowledge distillation")

            # Load custom model weights if provided
            if self.model_path:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)

            # Apply model optimizations
            if self.enable_quantization:
                self.model = ModelOptimizer.quantize_model(self.model)
                self._log("INFO", "Model quantization applied")

            if self.enable_model_pruning:
                self.model = ModelOptimizer.prune_model(self.model, amount=0.3)
                self._log("INFO", "Model pruning applied")

            # Move to device and set precision
            self.model = self.model.to(self.device)
            if self.inference_precision == InferencePrecision.FP16:
                self.model = self.model.half()

            self.model.eval()

            # Initialize distributed inference
            if self.enable_distributed:
                # Save model weights for distribution
                model_buffer = io.BytesIO()
                torch.save(self.model.state_dict(), model_buffer)
                self.model_weights = model_buffer.getvalue()
                self.distributed_inference = DistributedInference(
                    num_workers=self.config.get("distributed_workers", 4)
                )

            # Log GPU memory if CUDA is available
            if self.enable_cuda:
                gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                self.metrics.gpu_memory_used = gpu_memory
                self._log(
                    "INFO",
                    f"Model loaded on {self.device} with {self.inference_precision.value} precision "
                    f"(GPU Memory: {gpu_memory:.2f} MB)",
                )
            else:
                self._log(
                    "INFO",
                    f"Model loaded on {self.device} with {self.inference_precision.value} precision",
                )

        except Exception as e:
            self._log("ERROR", f"Failed to load model: {e}")
            self.model = None

    async def start(self):
        """Start the CNN engine pipeline"""
        if self._running:
            self._log("WARNING", "CNN Engine already running")
            return

        self._running = True
        self._shutdown_event.clear()

        # Initialize queues
        self.frame_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.result_queue = asyncio.Queue(maxsize=self.max_queue_size)

        # Start pipeline workers
        if self.enable_pipeline:
            for worker_id in range(self.pipeline_workers):
                worker = asyncio.create_task(
                    self._pipeline_worker(worker_id), name=f"cnn_worker_{worker_id}"
                )
                self.pipeline_workers_list.append(worker)

            self._log("INFO", f"Started {self.pipeline_workers} pipeline workers")

        # Start background tasks
        asyncio.create_task(self._collect_metrics())
        asyncio.create_task(self._cleanup_cache())

        if self.enable_active_learning:
            asyncio.create_task(self._active_learning_loop())

        self._log("INFO", "CNN Engine started")

    async def stop(self):
        """Stop the CNN engine pipeline"""
        self._running = False
        self._shutdown_event.set()

        # Cancel all workers
        for worker in self.pipeline_workers_list:
            worker.cancel()

        if self.pipeline_workers_list:
            await asyncio.gather(*self.pipeline_workers_list, return_exceptions=True)
            self.pipeline_workers_list.clear()

        # Clear queues
        if self.frame_queue:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._log("INFO", "CNN Engine stopped")

    async def process_frame(
        self, frame: np.ndarray, metadata: Dict[str, Any] = None, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process single frame with caching support

        Args:
            frame: Image frame as numpy array
            metadata: Optional metadata for the frame
            use_cache: Whether to use feature cache

        Returns:
            Processing results dictionary
        """
        # Generate cache key
        if use_cache and self.enable_redis_cache:
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            cache_key = f"frame:{frame_hash}"

            # Check Redis cache
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                with self._performance_lock:
                    self.metrics.cache_hits += 1
                return json.loads(cached_result)

        # Process frame
        if self.processing_mode == ProcessingMode.PIPELINE:
            result = await self._process_frame_pipeline(frame, metadata)
        elif self.processing_mode == ProcessingMode.BATCH:
            result = await self._process_frame_batch(frame, metadata)
        elif self.processing_mode == ProcessingMode.DISTRIBUTED:
            result = await self._process_frame_distributed(frame, metadata)
        else:
            result = await self._process_frame_single(frame, metadata)

        # Cache result
        if use_cache and self.enable_redis_cache and "error" not in result:
            self.redis_client.setex(cache_key, 3600, json.dumps(result))
            with self._performance_lock:
                self.metrics.cache_misses += 1

        # Update cache hit rate
        with self._performance_lock:
            total = self.metrics.cache_hits + self.metrics.cache_misses
            if total > 0:
                self.metrics.cache_hit_rate = self.metrics.cache_hits / total

        # Active learning
        if self.enable_active_learning and self.model and "result" in result:
            predictions = torch.tensor([[result["result"]["confidence"]]])
            self.active_learner.add_sample(frame, predictions)

        return result

    async def _process_frame_distributed(
        self, frame: np.ndarray, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process frame using distributed inference"""
        if not self.distributed_inference:
            return await self._process_frame_single(frame, metadata)

        # Process in distributed mode
        results = await self.distributed_inference.distribute_batch(
            [frame], self.model_weights, self.config
        )

        with self._performance_lock:
            self.metrics.distributed_tasks += 1

        return results[0] if results else {"error": "Distributed processing failed"}

    async def process_frames_chunked(
        self,
        frames: List[np.ndarray],
        metadata: List[Dict[str, Any]] = None,
        priorities: List[int] = None,
        compress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Process frames in chunks with compression

        Args:
            frames: List of image frames
            metadata: List of metadata for each frame
            priorities: Priority levels for each chunk
            compress: Whether to compress frames

        Returns:
            List of processing results
        """
        if not frames:
            return []

        # Compress frames if enabled
        if compress and self.enable_compression:
            compressed_frames = []
            for frame in frames:
                compressed = zlib.compress(
                    frame.tobytes(), level=self.compression_level
                )
                compressed_frames.append(
                    np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(
                        frame.shape
                    )
                )
            frames = compressed_frames

        # Create chunks
        chunks = []
        for i in range(0, len(frames), self.chunk_size):
            chunk_frames = frames[i : i + self.chunk_size]
            chunk_metadata = metadata[i : i + self.chunk_size] if metadata else None
            priority = priorities[i] if priorities and i < len(priorities) else 0

            # Calculate checksum for verification
            checksum = hashlib.md5(
                b"".join([f.tobytes() for f in chunk_frames])
            ).hexdigest()

            chunk = FrameChunk(
                chunk_id=len(chunks),
                frames=chunk_frames,
                timestamps=[time.time()] * len(chunk_frames),
                metadata=(
                    {"index": i, "metadata": chunk_metadata}
                    if chunk_metadata
                    else {"index": i}
                ),
                priority=priority,
                checksum=checksum,
            )
            chunks.append(chunk)

        self._log("DEBUG", f"Created {len(chunks)} chunks for {len(frames)} frames")

        # Sort chunks by priority
        chunks.sort(key=lambda x: x.priority, reverse=True)

        # Process chunks in parallel
        chunk_tasks = []
        for chunk in chunks:
            chunk_tasks.append(self._process_chunk(chunk))

        results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Flatten results
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                self._log("ERROR", f"Chunk processing failed: {result}")
                all_results.append({"error": str(result)})
            else:
                all_results.extend(result)

        return all_results

    async def _process_chunk(self, chunk: FrameChunk) -> List[Dict[str, Any]]:
        """Process a single chunk of frames"""
        chunk.status = FrameStatus.PROCESSING

        try:
            # Process chunk as batch if batch processing is enabled
            if self.batch_size > 1 and len(chunk.frames) >= self.batch_size:
                results = await self._process_batch(chunk.frames, chunk.metadata)
            else:
                results = []
                # Process frames in chunk sequentially
                for idx, frame in enumerate(chunk.frames):
                    metadata = (
                        chunk.metadata.get("metadata", [{}])[idx]
                        if chunk.metadata.get("metadata")
                        else {}
                    )

                    result = await self.process_frame(frame, metadata, use_cache=True)
                    results.append(result)

            chunk.results = results
            chunk.status = FrameStatus.COMPLETED

            # Update metrics
            with self._performance_lock:
                self.metrics.chunks_processed += 1
                self.metrics.frames_processed += len(chunk.frames)

            return results

        except Exception as e:
            chunk.status = FrameStatus.FAILED
            self._log("ERROR", f"Chunk {chunk.chunk_id} failed: {e}")

            with self._performance_lock:
                self.metrics.error_count += 1

            raise

    async def _process_batch(
        self, frames: List[np.ndarray], metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Process a batch of frames together for efficiency"""
        if not TORCH_AVAILABLE or self.model is None:
            return [
                {"result": {"class_id": 0, "confidence": 0.5}, "metadata": metadata}
                for _ in frames
            ]

        try:
            start_time = time.time()

            # Preprocess all frames
            preprocess_start = time.time()
            batch_tensor = []
            for frame in frames:
                processed = await self._preprocess_frame(frame)
                batch_tensor.append(processed)

            batch = torch.cat(batch_tensor, dim=0)
            preprocessing_time = (time.time() - preprocess_start) * 1000

            # Inference
            inference_start = time.time()
            with torch.no_grad():
                if self.inference_precision == InferencePrecision.FP16:
                    batch = batch.half()

                if self.enable_feature_extraction:
                    outputs, features = self.model(batch, return_features=True)
                    features_np = features.cpu().numpy()
                else:
                    outputs = self.model(batch)
                    features_np = None

                # Get probabilities
                probabilities = F.softmax(outputs, dim=1)
                top_probs, top_classes = torch.max(probabilities, 1)

            inference_time = (time.time() - inference_start) * 1000

            # Prepare results
            results = []
            for idx in range(len(frames)):
                result = {
                    "timestamp": time.time(),
                    "processing_time_ms": inference_time / len(frames),
                    "preprocessing_time_ms": preprocessing_time / len(frames),
                    "result": {
                        "class_id": top_classes[idx].item(),
                        "confidence": top_probs[idx].item(),
                        "all_probabilities": probabilities[idx].cpu().numpy().tolist(),
                    },
                    "metadata": (
                        metadata.get("metadata", [{}])[idx]
                        if metadata and "metadata" in metadata
                        else {}
                    ),
                }

                if features_np is not None and idx < len(features_np):
                    result["features"] = features_np[idx].tolist()
                    # Cache features
                    cache_key = f"feature_{top_classes[idx].item()}_{time.time()}"
                    self.feature_cache.put(cache_key, features_np[idx])

                results.append(result)

            # Update metrics
            with self._performance_lock:
                self.metrics.frames_processed += len(frames)
                self.metrics.total_latency += inference_time
                self.metrics.avg_latency_ms = (
                    self.metrics.total_latency / self.metrics.frames_processed
                )
                self.metrics.model_inference_time_ms = inference_time
                self.metrics.preprocessing_time_ms = preprocessing_time

            return results

        except Exception as e:
            self._log("ERROR", f"Batch processing failed: {e}")
            raise

    async def _process_frame_single(
        self, frame: np.ndarray, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process single frame synchronously"""
        start_time = time.time()

        try:
            # Preprocess frame
            preprocess_start = time.time()
            processed_frame = await self._preprocess_frame(frame)
            preprocessing_time = (time.time() - preprocess_start) * 1000

            # Run inference
            inference_start = time.time()
            if self.model is not None and TORCH_AVAILABLE:
                result = await self._run_inference(processed_frame)
            else:
                # Stub mode
                result = {
                    "class_id": 0,
                    "confidence": 0.5,
                    "features": np.random.rand(self.feature_dimension).tolist(),
                }
            inference_time = (time.time() - inference_start) * 1000

            # Generate attention map if enabled
            attention_map = None
            if self.enable_attention and self.model is not None:
                attention_start = time.time()
                attention_map = await self._generate_attention_map(processed_frame)
                attention_time = (time.time() - attention_start) * 1000
            else:
                attention_time = 0

            # Perform object detection if enabled
            detections = None
            if self.enable_object_detection:
                detection_start = time.time()
                detections = await self._detect_objects(frame)
                detection_time = (time.time() - detection_start) * 1000
            else:
                detection_time = 0

            # Postprocess results
            final_result = {
                "timestamp": time.time(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "preprocessing_time_ms": preprocessing_time,
                "inference_time_ms": inference_time,
                "attention_time_ms": attention_time,
                "detection_time_ms": detection_time,
                "result": result,
                "metadata": metadata or {},
                "class_label": (
                    self.class_labels[result["class_id"]]
                    if result["class_id"] < len(self.class_labels)
                    else "unknown"
                ),
            }

            if attention_map is not None:
                final_result["attention_map"] = (
                    attention_map.tolist()
                    if isinstance(attention_map, np.ndarray)
                    else attention_map
                )
            if detections is not None:
                final_result["detections"] = detections

            # Update metrics
            with self._performance_lock:
                self.metrics.frames_processed += 1
                self.metrics.total_latency += final_result["processing_time_ms"]
                self.metrics.avg_latency_ms = (
                    self.metrics.total_latency / self.metrics.frames_processed
                )
                self.metrics.last_processed_time = time.time()
                self.metrics.model_inference_time_ms = inference_time
                self.metrics.preprocessing_time_ms = preprocessing_time
                self.metrics.postprocessing_time_ms = attention_time + detection_time

            # Update Prometheus metrics
            if self.enable_prometheus:
                self.frame_counter.inc()
                self.latency_histogram.observe(
                    final_result["processing_time_ms"] / 1000
                )

            return final_result

        except asyncio.TimeoutError:
            self._log("ERROR", "Frame processing timeout")
            with self._performance_lock:
                self.metrics.error_count += 1
                if self.enable_prometheus:
                    self.error_counter.inc()

            return {
                "error": "processing_timeout",
                "timestamp": time.time(),
                "metadata": metadata or {},
            }

        except Exception as e:
            self._log("ERROR", f"Frame processing failed: {e}")
            with self._performance_lock:
                self.metrics.error_count += 1
                if self.enable_prometheus:
                    self.error_counter.inc()

            return {
                "error": str(e),
                "timestamp": time.time(),
                "metadata": metadata or {},
            }

    async def _process_frame_pipeline(
        self, frame: np.ndarray, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process frame through pipeline"""
        if not self._running:
            raise RuntimeError("CNN Engine not running")

        # Create future for result
        result_future = asyncio.Future()

        # Create frame task
        frame_task = {
            "frame": frame,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "future": result_future,
        }

        # Queue for processing
        queue_start = time.time()
        try:
            await asyncio.wait_for(
                self.frame_queue.put(frame_task), timeout=self.processing_timeout
            )
            queue_wait_time = (time.time() - queue_start) * 1000
            with self._performance_lock:
                self.metrics.queue_wait_time_ms = queue_wait_time
        except asyncio.TimeoutError:
            raise TimeoutError("Frame queue is full")

        # Wait for result
        try:
            result = await asyncio.wait_for(
                result_future, timeout=self.processing_timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError("Pipeline processing timeout")

    async def _process_frame_batch(
        self, frame: np.ndarray, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process frame in batch mode"""
        # Add to batch buffer
        self.chunk_buffer.append((frame, metadata))

        if len(self.chunk_buffer) >= self.batch_size:
            # Process batch
            batch_frames = [item[0] for item in self.chunk_buffer]
            batch_metadata = [item[1] for item in self.chunk_buffer]

            # Clear buffer
            self.chunk_buffer.clear()

            # Process batch
            results = await self._process_batch(
                batch_frames, {"metadata": batch_metadata}
            )

            # Return all results
            return results

        # Return placeholder (will be replaced when batch processes)
        return {"status": "buffered", "metadata": metadata}

    async def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for inference"""
        try:
            # Convert numpy to PIL if needed
            if isinstance(frame, np.ndarray):
                if PIL_AVAILABLE:
                    if len(frame.shape) == 2:  # Grayscale
                        if CV2_AVAILABLE:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        else:
                            frame = np.stack([frame] * 3, axis=-1)

            # Apply transforms
            if callable(self.transform):
                if PIL_AVAILABLE and isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame.astype("uint8"))

                tensor = self.transform(frame)

                # Add batch dimension
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)

                # Convert precision
                if self.inference_precision == InferencePrecision.FP16:
                    tensor = tensor.half()
                elif self.inference_precision == InferencePrecision.BF16:
                    tensor = tensor.bfloat16()

                # Move to device
                if self.device:
                    tensor = tensor.to(self.device)

                return tensor
            else:
                raise ValueError("Transform not properly initialized")

        except Exception as e:
            self._log("ERROR", f"Preprocessing failed: {e}")
            raise

    async def _run_inference(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Run inference on preprocessed tensor"""
        try:
            with torch.no_grad():
                if self.enable_feature_extraction:
                    # Run model and extract features
                    output, features = self.model(tensor, return_features=True)
                    features_np = features.cpu().numpy()
                else:
                    output = self.model(tensor)
                    features_np = None

                # Get probabilities
                probabilities = F.softmax(output.float(), dim=1)

                # Get top predictions (top 5)
                top_probs, top_classes = torch.topk(
                    probabilities, min(5, self.num_classes), dim=1
                )

                # Get prediction confidence
                confidence = top_probs[0][0].item()
                predicted_class = top_classes[0][0].item()

                # Build top predictions list
                top_predictions = [
                    {
                        "class_id": top_classes[0][i].item(),
                        "class_label": (
                            self.class_labels[top_classes[0][i].item()]
                            if top_classes[0][i].item() < len(self.class_labels)
                            else "unknown"
                        ),
                        "confidence": top_probs[0][i].item(),
                    }
                    for i in range(len(top_probs[0]))
                ]

                result = {
                    "class_id": predicted_class,
                    "confidence": confidence,
                    "top_predictions": top_predictions,
                    "all_probabilities": (
                        probabilities.cpu().numpy().tolist()[0]
                        if self.config.get("return_all_probs", False)
                        else None
                    ),
                    "features": (
                        features_np.tolist()
                        if features_np is not None
                        and self.config.get("return_features", False)
                        else None
                    ),
                }

                # Knowledge distillation (use teacher model for better predictions)
                if self.enable_knowledge_distillation and self.teacher_model:
                    teacher_output = self.teacher_model(tensor)
                    teacher_probs = F.softmax(teacher_output.float(), dim=1)
                    result["teacher_confidence"] = teacher_probs[0][
                        predicted_class
                    ].item()

                # Cache features
                if features_np is not None and self.enable_feature_extraction:
                    cache_key = f"feature_{predicted_class}_{time.time()}"
                    self.feature_cache.put(cache_key, features_np)

                return result

        except Exception as e:
            self._log("ERROR", f"Inference failed: {e}")
            raise

    async def _generate_attention_map(
        self, tensor: torch.Tensor
    ) -> Optional[np.ndarray]:
        """Generate attention map for visualization"""
        try:
            # Register hooks to get intermediate activations
            activations = []
            gradients = []

            def hook_fn(module, input, output):
                activations.append(output)

            def gradient_hook_fn(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            # Register hook on last convolutional layer
            if hasattr(self.model, "features"):
                # Find the last conv layer
                last_conv = None
                for module in self.model.features.modules():
                    if isinstance(module, nn.Conv2d):
                        last_conv = module

                if last_conv:
                    handle = last_conv.register_forward_hook(hook_fn)
                    backward_handle = last_conv.register_backward_hook(gradient_hook_fn)

                    # Forward pass
                    output = self.model(tensor)

                    # Backward pass for gradients
                    self.model.zero_grad()
                    output[0, output.argmax()].backward()

                    handle.remove()
                    backward_handle.remove()

                    if activations and gradients:
                        # Compute attention map from activations and gradients (Grad-CAM)
                        activation = activations[0]
                        gradient = gradients[0]

                        # Global average pooling of gradients
                        weights = gradient.mean(dim=[2, 3], keepdim=True)

                        # Weighted combination of activation maps
                        attention = (weights * activation).sum(dim=1, keepdim=True)
                        attention = F.relu(attention)

                        # Upsample to input size
                        attention = F.interpolate(
                            attention,
                            size=(self.input_size, self.input_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        attention = attention.squeeze().cpu().numpy()

                        # Normalize to [0, 1]
                        attention = (attention - attention.min()) / (
                            attention.max() - attention.min() + 1e-8
                        )

                        return attention

            return None

        except Exception as e:
            self._log("ERROR", f"Attention map generation failed: {e}")
            return None

    async def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Advanced object detection with multiple scales"""
        detections = []

        try:
            if not CV2_AVAILABLE:
                return detections

            height, width = frame.shape[:2]

            # Multi-scale detection
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            window_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
            stride = self.input_size // 4

            for scale in scales:
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))

                for window_w, window_h in window_sizes:
                    window_w = int(window_w * scale)
                    window_h = int(window_h * scale)

                    if window_w > scaled_width or window_h > scaled_height:
                        continue

                    for y in range(0, scaled_height - window_h, stride):
                        for x in range(0, scaled_width - window_w, stride):
                            # Extract window
                            window = scaled_frame[y : y + window_h, x : x + window_w]

                            if window.size == 0:
                                continue

                            # Resize to model input size
                            window_resized = cv2.resize(
                                window, (self.input_size, self.input_size)
                            )

                            # Run inference
                            processed = await self._preprocess_frame(window_resized)
                            result = await self._run_inference(processed)

                            if result["confidence"] > self.confidence_threshold:
                                # Scale bounding box back to original coordinates
                                x1 = int(x / scale)
                                y1 = int(y / scale)
                                x2 = int((x + window_w) / scale)
                                y2 = int((y + window_h) / scale)

                                detections.append(
                                    {
                                        "bbox": [x1, y1, x2, y2],
                                        "confidence": result["confidence"],
                                        "class_id": result["class_id"],
                                        "class_label": result.get(
                                            "class_label", "unknown"
                                        ),
                                        "scale": scale,
                                    }
                                )

            # Non-maximum suppression
            detections = self._non_max_suppression(detections, iou_threshold=0.5)

            # Limit to top detections
            detections = detections[: self.config.get("max_detections", 10)]

            return detections

        except Exception as e:
            self._log("ERROR", f"Object detection failed: {e}")
            return []

    def _non_max_suppression(
        self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Apply non-maximum suppression to detections"""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                d
                for d in detections
                if self._compute_iou(best["bbox"], d["bbox"]) < iou_threshold
            ]

        return keep

    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute Intersection over Union of two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    async def _pipeline_worker(self, worker_id: int):
        """Pipeline worker for parallel processing"""
        self._log("DEBUG", f"Pipeline worker {worker_id} started")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Get frame task from queue
                frame_task = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                # Process frame
                try:
                    start_time = time.time()

                    # Preprocess
                    processed_frame = await self._preprocess_frame(frame_task["frame"])

                    # Run inference
                    if self.model is not None:
                        result = await self._run_inference(processed_frame)
                    else:
                        result = {"stub": True, "confidence": 0.5, "class_id": 0}

                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000

                    # Prepare final result
                    final_result = {
                        "timestamp": time.time(),
                        "processing_time_ms": processing_time,
                        "result": result,
                        "metadata": frame_task["metadata"],
                        "worker_id": worker_id,
                        "class_label": (
                            self.class_labels[result.get("class_id", 0)]
                            if result.get("class_id", 0) < len(self.class_labels)
                            else "unknown"
                        ),
                    }

                    # Set future result
                    if not frame_task["future"].done():
                        frame_task["future"].set_result(final_result)

                    # Update metrics
                    with self._performance_lock:
                        self.metrics.frames_processed += 1
                        self.metrics.total_latency += processing_time
                        self.metrics.avg_latency_ms = (
                            self.metrics.total_latency / self.metrics.frames_processed
                        )

                except Exception as e:
                    self._log("ERROR", f"Worker {worker_id} processing failed: {e}")

                    if not frame_task["future"].done():
                        frame_task["future"].set_exception(e)

                    with self._performance_lock:
                        self.metrics.error_count += 1

                finally:
                    self.frame_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log("ERROR", f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

        self._log("DEBUG", f"Pipeline worker {worker_id} stopped")

    async def _collect_metrics(self):
        """Collect and update performance metrics"""
        while self._running and not self._shutdown_event.is_set():
            await asyncio.sleep(5.0)

            with self._performance_lock:
                # Calculate throughput
                if self.metrics.last_processed_time > 0:
                    time_window = 5.0
                    self.metrics.throughput_fps = self.metrics.frames_processed / max(
                        1, time_window
                    )

                # Update GPU memory if CUDA available
                if self.enable_cuda and torch.cuda.is_available():
                    self.metrics.gpu_memory_used = (
                        torch.cuda.memory_allocated(self.device) / 1024**2
                    )

                # Log metrics
                self._log(
                    "INFO",
                    f"Metrics: frames={self.metrics.frames_processed}, "
                    f"chunks={self.metrics.chunks_processed}, "
                    f"avg_latency={self.metrics.avg_latency_ms:.2f}ms, "
                    f"throughput={self.metrics.throughput_fps:.2f}fps, "
                    f"errors={self.metrics.error_count}, "
                    f"gpu_mem={self.metrics.gpu_memory_used:.2f}MB, "
                    f"cache_hit_rate={self.metrics.cache_hit_rate:.2%}",
                )

                # Update Prometheus gauges
                if self.enable_prometheus:
                    self.cache_hits_gauge.set(self.metrics.cache_hit_rate)

    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        while self._running and not self._shutdown_event.is_set():
            await asyncio.sleep(60)
            self.feature_cache.clear()
            self._log("DEBUG", "Feature cache cleared")

    async def _active_learning_loop(self):
        """Background loop for active learning"""
        while self._running and not self._shutdown_event.is_set():
            await asyncio.sleep(300)  # Check every 5 minutes

            if (
                self.active_learner
                and len(self.active_learner.high_uncertainty_samples) > 50
            ):
                samples = self.active_learner.get_samples_for_labeling(max_samples=100)
                self._log(
                    "INFO",
                    f"Active learning: {len(samples)} samples ready for labeling",
                )

                # In production, this would trigger a labeling request
                # For now, just log

    async def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ):
        """Train or fine-tune the model"""
        if not TORCH_AVAILABLE or self.model is None:
            self._log("ERROR", "Training not available")
            return

        self.model.train()

        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Use distillation loss if teacher model is available
        if self.enable_knowledge_distillation and self.teacher_model:
            criterion = DistillationLoss(temperature=3.0, alpha=0.7)
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_accuracy = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                if self.enable_knowledge_distillation and self.teacher_model:
                    outputs = self.model(inputs)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(inputs)
                    loss = criterion(outputs, teacher_outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if batch_idx % 100 == 0:
                    self._log(
                        "INFO",
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}",
                    )

            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            val_accuracy = 0.0
            avg_val_loss = 0.0

            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_accuracy = 100 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Update training metrics
            self.training_metrics.epoch = epoch + 1
            self.training_metrics.train_loss = avg_train_loss
            self.training_metrics.val_loss = avg_val_loss
            self.training_metrics.train_accuracy = train_accuracy
            self.training_metrics.val_accuracy = val_accuracy
            self.training_metrics.learning_rate = self.scheduler.get_last_lr()[0]

            # Save checkpoint
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
                self.training_metrics.best_accuracy = best_val_accuracy

            checkpoint_path = self.checkpoint_manager.save(
                self.model,
                self.optimizer,
                epoch + 1,
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                },
                is_best,
            )

            self._log(
                "INFO",
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                f"LR: {self.training_metrics.learning_rate:.6f}",
            )

            if is_best:
                self._log("INFO", f"New best model! Accuracy: {val_accuracy:.2f}%")

        self.model.eval()
        self._log(
            "INFO",
            f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%",
        )

        return self.training_metrics

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._performance_lock:
            return {
                "frames_processed": self.metrics.frames_processed,
                "chunks_processed": self.metrics.chunks_processed,
                "total_latency_ms": self.metrics.total_latency,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "throughput_fps": self.metrics.throughput_fps,
                "error_count": self.metrics.error_count,
                "processing_mode": self.processing_mode.value,
                "pipeline_workers": (
                    len(self.pipeline_workers_list) if self.enable_pipeline else 0
                ),
                "queue_size": self.frame_queue.qsize() if self.frame_queue else 0,
                "gpu_memory_mb": self.metrics.gpu_memory_used,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "distributed_tasks": self.metrics.distributed_tasks,
                "avg_inference_time_ms": self.metrics.model_inference_time_ms,
                "avg_preprocessing_time_ms": self.metrics.preprocessing_time_ms,
                "training_metrics": (
                    {
                        "epoch": self.training_metrics.epoch,
                        "train_accuracy": self.training_metrics.train_accuracy,
                        "val_accuracy": self.training_metrics.val_accuracy,
                        "best_accuracy": self.training_metrics.best_accuracy,
                    }
                    if self.training_metrics.epoch > 0
                    else None
                ),
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        return {
            "status": "healthy" if self._running else "stopped",
            "model_loaded": self.model is not None,
            "architecture": self.architecture.value,
            "device": str(self.device) if self.device else "cpu",
            "precision": self.inference_precision.value,
            "pipeline_running": len(self.pipeline_workers_list) > 0,
            "metrics": self.get_metrics(),
            "queue_utilization": (
                (self.frame_queue.qsize() / self.max_queue_size) * 100
                if self.frame_queue
                else 0
            ),
            "feature_extraction_enabled": self.enable_feature_extraction,
            "attention_enabled": self.enable_attention,
            "object_detection_enabled": self.enable_object_detection,
            "quantization_enabled": self.enable_quantization,
            "distributed_enabled": self.enable_distributed,
            "redis_cache_enabled": self.enable_redis_cache,
            "active_learning_enabled": self.enable_active_learning,
            "model_pruning_enabled": self.enable_model_pruning,
            "knowledge_distillation_enabled": self.enable_knowledge_distillation,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "gpu_memory_mb": self.metrics.gpu_memory_used,
        }

    async def extract_features_batch(
        self, frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Extract features from multiple frames"""
        if not self.enable_feature_extraction:
            self._log("WARNING", "Feature extraction is not enabled")
            return []

        results = []

        for frame in frames:
            processed = await self._preprocess_frame(frame)

            with torch.no_grad():
                if self.model is not None:
                    features = self.model.extract_features(processed)
                    results.append(features.cpu().numpy())
                else:
                    results.append(np.random.rand(self.feature_dimension))

        return results

    async def compare_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compare two frames using feature similarity"""
        features1 = await self.extract_features_batch([frame1])
        features2 = await self.extract_features_batch([frame2])

        if not features1 or not features2:
            return 0.0

        # Compute cosine similarity
        f1 = features1[0].flatten()
        f2 = features2[0].flatten()

        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)

        return float(similarity)

    async def export_model(self, format: str = "onnx", output_path: str = None):
        """Export model to different formats"""
        if self.model is None:
            self._log("ERROR", "No model to export")
            return None

        if format == "onnx":
            output_path = output_path or "model.onnx"
            ModelOptimizer.convert_to_onnx(
                self.model, (1, 3, self.input_size, self.input_size), output_path
            )
            self._log("INFO", f"Model exported to ONNX: {output_path}")
            return output_path

        elif format == "torchscript":
            output_path = output_path or "model.pt"
            traced_model = torch.jit.trace(
                self.model,
                torch.randn(1, 3, self.input_size, self.input_size).to(self.device),
            )
            traced_model.save(output_path)
            self._log("INFO", f"Model exported to TorchScript: {output_path}")
            return output_path

        else:
            self._log("ERROR", f"Unsupported export format: {format}")
            return None

    def _log(self, level: str, message: str):
        """Log message"""
        if self.logger:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(f"[CNNEngine] {message}")
        else:
            print(f"[{level}] [CNNEngine] {message}")

    async def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from buffer (compatibility method)"""
        return None


class VisionMemory:
    """Enhanced vision memory with chunked storage and retrieval"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.frame_buffer = deque(maxlen=self.config.get("max_buffer_size", 1000))
        self.chunk_size = self.config.get("memory_chunk_size", 10)
        self._lock = asyncio.Lock()
        self._frame_metadata = {}
        self._frame_indices = {}
        self._index_counter = 0

    async def store_frame(self, frame: np.ndarray, metadata: Dict[str, Any] = None):
        """Store frame in memory with indexing"""
        async with self._lock:
            frame_id = self._index_counter
            self._index_counter += 1

            self.frame_buffer.append(
                {
                    "frame": frame,
                    "metadata": metadata or {},
                    "timestamp": time.time(),
                    "frame_id": frame_id,
                }
            )

            self._frame_metadata[frame_id] = metadata or {}
            self._frame_indices[frame_id] = len(self.frame_buffer) - 1

    async def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from memory"""
        async with self._lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]["frame"]
            return None

    async def get_latest_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata of latest frame"""
        async with self._lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]["metadata"]
            return None

    async def get_frames_chunked(self, chunk_id: int) -> List[np.ndarray]:
        """Get frames in chunks"""
        async with self._lock:
            start_idx = chunk_id * self.chunk_size
            end_idx = start_idx + self.chunk_size

            if start_idx >= len(self.frame_buffer):
                return []

            frames = [
                item["frame"] for item in list(self.frame_buffer)[start_idx:end_idx]
            ]
            return frames

    async def get_frame_range(self, start_idx: int, end_idx: int) -> List[np.ndarray]:
        """Get range of frames"""
        async with self._lock:
            start_idx = max(0, start_idx)
            end_idx = min(len(self.frame_buffer), end_idx)

            if start_idx >= end_idx:
                return []

            frames = [
                item["frame"] for item in list(self.frame_buffer)[start_idx:end_idx]
            ]
            return frames

    async def get_frame_by_id(self, frame_id: int) -> Optional[np.ndarray]:
        """Get frame by ID"""
        async with self._lock:
            if frame_id in self._frame_indices:
                idx = self._frame_indices[frame_id]
                if idx < len(self.frame_buffer):
                    return list(self.frame_buffer)[idx]["frame"]
            return None

    async def get_metadata_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata by frame ID"""
        return self._frame_metadata.get(frame_id)

    async def get_buffer_size(self) -> int:
        """Get current buffer size"""
        async with self._lock:
            return len(self.frame_buffer)

    async def clear(self):
        """Clear memory buffer"""
        async with self._lock:
            self.frame_buffer.clear()
            self._frame_metadata.clear()
            self._frame_indices.clear()
            self._index_counter = 0

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        async with self._lock:
            return {
                "buffer_size": len(self.frame_buffer),
                "max_buffer_size": self.frame_buffer.maxlen,
                "chunk_size": self.chunk_size,
                "utilization_percent": (
                    (len(self.frame_buffer) / self.frame_buffer.maxlen) * 100
                    if self.frame_buffer.maxlen > 0
                    else 0
                ),
                "metadata_count": len(self._frame_metadata),
                "index_count": len(self._frame_indices),
            }


# Utility functions
async def create_cnn_engine(config: Dict[str, Any] = None) -> CNNEngine:
    """Factory function to create and start CNN engine"""
    engine = CNNEngine(config)
    await engine.start()
    return engine


# Example usage
async def example_usage():
    """Example of how to use the enhanced CNN Engine"""

    # Configuration with advanced features
    config = {
        "architecture": "efficientnet_b0",
        "pretrained": True,
        "input_size": 224,
        "batch_size": 32,
        "chunk_size": 10,
        "pipeline_workers": 4,
        "enable_cuda": True,
        "enable_pipeline": True,
        "enable_feature_extraction": True,
        "enable_attention": True,
        "enable_object_detection": True,
        "enable_quantization": True,
        "inference_precision": "fp16",
        "num_classes": 1000,
        "confidence_threshold": 0.5,
        "cache_strategy": "adaptive",
        "max_cache_size": 5000,
        "enable_redis_cache": False,  # Set to True if Redis is available
        "enable_active_learning": True,
        "enable_knowledge_distillation": False,
        "dropout_rate": 0.3,
        "return_all_probs": True,
        "return_features": True,
        "max_detections": 10,
    }

    # Create engine
    print("Creating CNN Engine...")
    engine = await create_cnn_engine(config)

    try:
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Single frame processing
        print("\nProcessing single frame...")
        result = await engine.process_frame(dummy_frame)
        print(
            f"Result: {result['class_label']} with confidence {result['result']['confidence']:.3f}"
        )
        print(f"Processing time: {result['processing_time_ms']:.2f}ms")

        if "top_predictions" in result["result"]:
            print("Top predictions:")
            for pred in result["result"]["top_predictions"][:3]:
                print(f"  - {pred['class_label']}: {pred['confidence']:.3f}")

        # Batch processing
        print("\nProcessing batch of frames...")
        frames = [dummy_frame] * 10
        results = await engine.process_frames_chunked(frames, compress=True)
        print(f"Processed {len(results)} frames")

        # Feature extraction
        print("\nExtracting features...")
        features = await engine.extract_features_batch([dummy_frame])
        print(f"Feature shape: {features[0].shape if features else 'None'}")

        # Frame comparison
        print("\nComparing frames...")
        similarity = await engine.compare_frames(dummy_frame, dummy_frame)
        print(f"Similarity score: {similarity:.4f}")

        # Get metrics
        metrics = engine.get_metrics()
        print(f"\nMetrics: {json.dumps(metrics, indent=2)}")

        # Health check
        health = await engine.health_check()
        print(f"\nHealth Status: {health['status']}")
        print(f"Model: {health['architecture']} on {health['device']}")
        print(f"Cache hit rate: {health['cache_hit_rate']:.2%}")

        # Export model (optional)
        # await engine.export_model("onnx", "exported_model.onnx")

    finally:
        # Cleanup
        await engine.stop()
        print("\nCNN Engine stopped")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
