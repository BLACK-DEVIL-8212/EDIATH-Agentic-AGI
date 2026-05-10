"""Vision modules for visual perception and AI processing."""

from typing import Optional, Dict, Any, List
from .webcam import Webcam, FrameBuffer
from .object_detector import ObjectDetector, Detection
from .captioner import Captioner, ImageCaption
from .perception_loop import PerceptionLoop, PerceptionStep
from .vision_engine import VisionEngine
from .activity_recognition import ActivityRecognitionEngine
from .face_recognition import FaceRecognitionEngine
from .shared_memory import SharedVisionMemory

__all__ = [
    # Core classes
    "Webcam",
    "FrameBuffer",
    "ObjectDetector",
    "Detection",
    "ObjectClass",
    "Captioner",
    "ImageCaption",
    "PerceptionLoop",
    "PerceptionStep",
    "VisionEngine",
    "ActivityRecognitionEngine",
    "FaceRecognitionEngine",
    "SharedVisionMemory",
]
