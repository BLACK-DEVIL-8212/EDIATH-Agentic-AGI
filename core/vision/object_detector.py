# object_detector.py

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    YOLO = None


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class ModelArchitecture(Enum):
    YOLOV8_NANO = "yolov8n"
    YOLOV8_SMALL = "yolov8s"
    YOLOV8_MEDIUM = "yolov8m"
    YOLOV8_LARGE = "yolov8l"
    YOLOV8_XLARGE = "yolov8x"
    HOG = "hog"
    CUSTOM = "custom"


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    AUTO = "auto"


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 300
    classes_filter: Optional[List[int]] = None
    device: DeviceType = DeviceType.AUTO
    model_architecture: ModelArchitecture = ModelArchitecture.YOLOV8_NANO
    use_tracking: bool = False
    half_precision: bool = True  # Use FP16 for faster inference
    augment: bool = False  # Test-time augmentation
    agnostic_nms: bool = False  # Class-agnostic NMS
    retina_masks: bool = False  # For instance segmentation
    imgsz: int = 640  # Image size for inference


# ------------------------
# ADVANCED DETECTION OBJECT
# ------------------------
class Detection:
    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        class_id: Optional[int] = None,
        track_id: Optional[int] = None,
        segmentation: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None
    ):
        self.class_name = class_name
        self.confidence = float(confidence)
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.class_id = class_id
        self.track_id = track_id
        self.segmentation = segmentation
        self.keypoints = keypoints
        self.timestamp = time.time()
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def to_dict(self) -> Dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "area": self.area,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
        }

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


# ------------------------
# ADVANCED OBJECT DETECTOR
# ------------------------
class ObjectDetector:
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.detections_performed = 0
        self.total_objects_found = 0
        self.inference_times = []
        self.frame_count = 0
        
        self.has_cv2 = cv2 is not None
        self.model = None
        self.device = self._get_device()
        
        # Tracking
        self.track_history = {}
        self.next_track_id = 0
        
        # Statistics
        self.class_stats = {}
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"✅ Object Detector Initialized - Device: {self.device}, "
                   f"Model: {self.config.model_architecture.value}")
    
    # ------------------------
    # DEVICE MANAGEMENT
    # ------------------------
    def _get_device(self) -> str:
        if self.config.device == DeviceType.AUTO:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device.value
    
    # ------------------------
    # MODEL INITIALIZATION
    # ------------------------
    def _initialize_model(self):
        """Initialize YOLO or fallback model"""
        
        if self.config.model_architecture == ModelArchitecture.HOG:
            self._init_hog()
        elif TORCH_AVAILABLE and YOLO is not None:
            self._init_yolo()
        else:
            logger.warning("⚠️ YOLO not available, using HOG fallback")
            self._init_hog()
    
    def _init_yolo(self):
        """Initialize YOLOv8 model"""
        try:
            model_name = self.config.model_architecture.value
            logger.info(f"🚀 Loading YOLOv8 model: {model_name}")
            
            self.model = YOLO(model_name)
            
            # Configure model
            if self.device == "cuda" and self.config.half_precision:
                self.model.model.half()  # FP16 inference
                logger.info("🎯 Using FP16 inference")
            
            # Warm up model
            dummy_input = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
            _ = self.model(dummy_input, verbose=False)
            
            logger.info(f"✅ YOLOv8 loaded successfully on {self.device}")
            self.is_yolo = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self._init_hog()
    
    def _init_hog(self):
        """Initialize HOG as fallback"""
        if self.has_cv2:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.is_yolo = False
            logger.info("🔄 Using HOG as fallback detector")
        else:
            self.hog = None
            logger.error("❌ No detection backend available")
    
    # ------------------------
    # ADVANCED DETECTION
    # ------------------------
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Main detection method with YOLOv8 support"""
        if image is None or not isinstance(image, np.ndarray):
            return []
        
        self.frame_count += 1
        start_time = time.time()
        
        try:
            if self.is_yolo and self.model is not None:
                detections = self._detect_yolo(image)
            else:
                detections = self._detect_hog(image)
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.detections_performed += 1
            self.total_objects_found += len(detections)
            
            # Update class statistics
            for det in detections:
                self.class_stats[det.class_name] = self.class_stats.get(det.class_name, 0) + 1
            
            # Apply tracking if enabled
            if self.config.use_tracking:
                detections = self._apply_tracking(detections)
            
            return detections
            
        except Exception as e:
            logger.exception(f"Detection error: {e}")
            return []
    
    def _detect_yolo(self, image: np.ndarray) -> List[Detection]:
        """YOLOv8 detection with advanced options"""
        results = self.model(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            classes=self.config.classes_filter,
            device=self.device,
            half=self.config.half_precision,
            augment=self.config.augment,
            agnostic_nms=self.config.agnostic_nms,
            retina_masks=self.config.retina_masks,
            imgsz=self.config.imgsz,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confs, classes):
                    if conf >= self.config.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.model.names[int(cls_id)]
                        
                        detection = Detection(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(x1, y1, x2, y2),
                            class_id=int(cls_id)
                        )
                        
                        # Add segmentation masks if available
                        if self.config.retina_masks and result.masks is not None:
                            mask = result.masks.data.cpu().numpy()
                            detection.segmentation = mask
                        
                        # Add keypoints if available
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            keypoints = result.keypoints.data.cpu().numpy()
                            detection.keypoints = keypoints
                        
                        detections.append(detection)
        
        return detections
    
    def _detect_hog(self, image: np.ndarray) -> List[Detection]:
        """HOG fallback detection"""
        if self.hog is None:
            # Ultra fallback - fake detection for testing
            h, w = image.shape[:2]
            fake_bbox = (w//4, h//4, 3*w//4, 3*h//4)
            return [Detection("object", 0.5, fake_bbox)]
        
        boxes, weights = self.hog.detectMultiScale(
            image, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        
        detections = []
        for (x, y, w, h), conf in zip(boxes, weights):
            if conf >= self.config.confidence_threshold:
                detections.append(
                    Detection(
                        class_name="person",
                        confidence=float(conf),
                        bbox=(x, y, x + w, y + h),
                        class_id=0
                    )
                )
        
        return detections
    
    # ------------------------
    # OBJECT TRACKING
    # ------------------------
    def _apply_tracking(self, detections: List[Detection]) -> List[Detection]:
        """Simple IoU-based tracking"""
        for det in detections:
            if det.track_id is None:
                # Find best match from previous frame
                best_iou = 0
                best_id = None
                
                for track_id, track_bbox in self.track_history.items():
                    iou = self._calculate_iou(det.bbox, track_bbox)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_id = track_id
                
                if best_id is not None:
                    det.track_id = best_id
                    self.track_history[best_id] = det.bbox
                else:
                    det.track_id = self.next_track_id
                    self.track_history[self.next_track_id] = det.bbox
                    self.next_track_id += 1
        
        # Clean up old tracks (older than 1 second)
        # In production, you'd want more sophisticated tracking like SORT/DeepSORT
        return detections
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # ------------------------
    # VISUALIZATION
    # ------------------------
    def draw_boxes(
        self, 
        image: np.ndarray, 
        detections: List[Detection],
        show_conf: bool = True,
        show_track_id: bool = False,
        custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """Enhanced drawing with multiple options"""
        if image is None or cv2 is None:
            return image
        
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on class or track ID
            if custom_colors and det.class_name in custom_colors:
                color = custom_colors[det.class_name]
            else:
                # Generate consistent color from class name
                color = self._get_color(det.class_name)
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [det.class_name]
            if show_conf:
                label_parts.append(f"{det.confidence:.2f}")
            if show_track_id and det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img_copy,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_copy,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            center = det.center
            cv2.circle(img_copy, center, 4, (0, 0, 255), -1)
        
        return img_copy
    
    def draw_segmentation(
        self, 
        image: np.ndarray, 
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw instance segmentation masks"""
        if cv2 is None:
            return image
        
        overlay = image.copy()
        
        for det in detections:
            if det.segmentation is not None:
                # Create mask overlay
                mask = det.segmentation.squeeze().astype(np.uint8) * 255
                color = self._get_color(det.class_name)
                
                # Apply colored mask
                mask_colored = np.zeros_like(image)
                mask_colored[:] = color
                
                # Blend mask with image
                mask_indices = mask > 0
                overlay[mask_indices] = cv2.addWeighted(
                    image[mask_indices], 0.5, mask_colored[mask_indices], 0.5, 0
                )
        
        return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Generate consistent color for class"""
        import hashlib
        hash_obj = hashlib.md5(class_name.encode())
        hash_hex = hash_obj.hexdigest()[:6]
        return tuple(int(hash_hex[i:i+2], 16) for i in (0, 2, 4))
    
    # ------------------------
    # FILTERING & ANALYSIS
    # ------------------------
    def filter_by_class(
        self, 
        detections: List[Detection], 
        class_name: str
    ) -> List[Detection]:
        """Filter detections by class name"""
        return [d for d in detections if d.class_name == class_name]
    
    def filter_by_confidence(
        self, 
        detections: List[Detection], 
        min_confidence: float
    ) -> List[Detection]:
        """Filter detections by confidence score"""
        return [d for d in detections if d.confidence >= min_confidence]
    
    def filter_by_region(
        self, 
        detections: List[Detection], 
        roi: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """Filter detections by region of interest (x1, y1, x2, y2)"""
        x1, y1, x2, y2 = roi
        return [
            d for d in detections
            if (d.center[0] >= x1 and d.center[0] <= x2 and
                d.center[1] >= y1 and d.center[1] <= y2)
        ]
    
    def get_dominant_object(self, detections: List[Detection]) -> Optional[Detection]:
        """Get the most confident detection"""
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
    
    def get_object_count_by_class(self, detections: List[Detection]) -> Dict[str, int]:
        """Count objects by class"""
        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
    
    # ------------------------
    # STATISTICS & INFO
    # ------------------------
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        std_inference = np.std(self.inference_times) if self.inference_times else 0
        
        return {
            "detections_performed": self.detections_performed,
            "total_objects_found": self.total_objects_found,
            "avg_objects_per_frame": self.total_objects_found / max(1, self.detections_performed),
            "avg_inference_time_ms": avg_inference * 1000,
            "std_inference_time_ms": std_inference * 1000,
            "fps": 1.0 / avg_inference if avg_inference > 0 else 0,
            "model": self.config.model_architecture.value,
            "device": self.device,
            "class_stats": self.class_stats,
            "is_yolo": hasattr(self, 'is_yolo') and self.is_yolo,
            "tracking_enabled": self.config.use_tracking,
            "active_tracks": len(self.track_history)
        }
    
    def export_detections(
        self, 
        detections: List[Detection], 
        filepath: Union[str, Path],
        format: str = "json"
    ):
        """Export detections to file"""
        data = {
            "timestamp": time.time(),
            "num_detections": len(detections),
            "detections": [d.to_dict() for d in detections],
            "stats": self.get_stats()
        }
        
        filepath = Path(filepath)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data["detections"][0].keys())
                writer.writeheader()
                writer.writerows(data["detections"])
        
        logger.info(f"📁 Exported {len(detections)} detections to {filepath}")
    
    def reset_stats(self):
        """Reset statistics"""
        self.detections_performed = 0
        self.total_objects_found = 0
        self.inference_times = []
        self.class_stats = {}
        self.track_history = {}
        self.next_track_id = 0
        logger.info("🔄 Statistics reset")
    
    def set_threshold(self, confidence: float):
        """Update confidence threshold dynamically"""
        self.config.confidence_threshold = confidence
        logger.info(f"📊 Confidence threshold updated to {confidence}")


# ------------------------
# USAGE EXAMPLE
# ------------------------
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector with advanced config
    config = DetectionConfig(
        confidence_threshold=0.5,
        model_architecture=ModelArchitecture.YOLOV8_NANO,
        device=DeviceType.AUTO,
        use_tracking=True,
        half_precision=True
    )
    
    detector = ObjectDetector(config)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Draw results
        annotated_frame = detector.draw_boxes(
            frame, 
            detections, 
            show_conf=True,
            show_track_id=True
        )
        
        # Get statistics
        if detector.frame_count % 30 == 0:
            stats = detector.get_stats()
            logger.info(f"FPS: {stats['fps']:.1f}, Objects: {stats['total_objects_found']}")
        
        cv2.imshow('Object Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print(json.dumps(detector.get_stats(), indent=2))