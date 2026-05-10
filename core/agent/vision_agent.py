import asyncio
import base64
import hashlib
import io
import json
import logging
import time as _time  # FIX #1 – stdlib time module (not datetime.time)
import numpy as np

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

# ── Project imports ────────────────────────────────────────────────────────
from core.vision.shared_memory import SharedVisionMemory as VisionMemory
from core.vision.vision_engine import VisionEngine

# ── Optional heavy dependencies ────────────────────────────────────────────
try:
    import torch
    import torchvision  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
        DetrImageProcessor,
        DetrForObjectDetection,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        CLIPProcessor,
        CLIPModel,
        pipeline as transformers_pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None  # type: ignore
    EASYOCR_AVAILABLE = False

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    face_recognition = None  # type: ignore
    FACE_RECOGNITION_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    YOLO_AVAILABLE = False

# FIX #11 / Enhancement B – optional sklearn guard
try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None  # type: ignore
    SKLEARN_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


class VisionTask(Enum):
    CAPTIONING = "captioning"
    OBJECT_DETECTION = "object_detection"
    OCR = "ocr"
    FACE_RECOGNITION = "face_recognition"
    IMAGE_CLASSIFICATION = "classification"
    SCENE_UNDERSTANDING = "scene_understanding"
    COLOR_ANALYSIS = "color_analysis"
    TEXT_DETECTION = "text_detection"
    LANDMARK_DETECTION = "landmark_detection"
    QWEN_VL = "qwen_vl"


class DetectionModel(Enum):
    DETR = "detr"
    YOLO = "yolo"
    SSD = "ssd"
    FASTER_RCNN = "faster_rcnn"


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    color: Optional[str] = None


@dataclass
class DetectedFace:
    location: Tuple[int, int, int, int]  # top, right, bottom, left
    encoding: Optional[List[float]] = None
    name: Optional[str] = None
    confidence: float = 0.0
    emotions: Optional[Dict[str, float]] = None


@dataclass
class OCRText:
    text: str
    confidence: float
    bbox: List[Tuple[int, int]]
    language: str = "en"


@dataclass
class ImageAnalysis:
    timestamp: datetime
    image_path: Optional[str]
    image_size: Tuple[int, int]
    captions: List[str] = field(default_factory=list)
    objects: List[DetectedObject] = field(default_factory=list)
    faces: List[DetectedFace] = field(default_factory=list)
    text: List[OCRText] = field(default_factory=list)
    colors: Dict[str, float] = field(default_factory=dict)
    scene_type: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# VisionAgent
# ══════════════════════════════════════════════════════════════════════════════


class VisionAgent:
    """
    Advanced vision-understanding agent.

    Quickstart::

        agent = VisionAgent()
        await agent.initialize()            # loads known faces, optional warm-up
        result = await agent.generate_caption("photo.jpg")
    """

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Device
        self.device = self.config.get(
            "device",
            "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        )
        self.use_gpu = self.device == "cuda"

        # Model instances
        self.caption_processor = None
        self.caption_model = None
        self.detection_processor = None
        self.detection_model = None
        self.ocr_reader = None
        self.clip_processor = None
        self.clip_model = None
        self.yolo_model = None

        # FIX #2 – all flags start False; True only after confirmed load
        self.models_loaded: Dict[str, bool] = {
            "captioning": False,
            "detection": False,
            "ocr": False,
            "clip": False,
            "face_recognition": FACE_RECOGNITION_AVAILABLE,  # no loading needed
            "yolo": False,
        }
        # FIX #3 – track which loads have been attempted to avoid hammering
        self._model_load_attempted: set = set()

        # Config
        self.default_model = self.config.get("default_model", "blip")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.max_objects = self.config.get("max_objects", 10)

        # Cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_size = self.config.get("cache_size", 100)
        self.cache: Dict[str, Dict] = {}

        # Statistics
        self.stats: Dict[str, Any] = {
            "total_processed": 0,
            "total_objects_detected": 0,
            "total_faces_detected": 0,
            "total_text_extracted": 0,
            "average_processing_time": 0.0,
            "errors": 0,
        }

        # Known faces
        self.known_faces_dir = Path(self.config.get("known_faces_dir", "known_faces"))
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []

        # Vision Engine (connected later)
        self.vision_engine: Optional[VisionEngine] = None

        # Shared memory (lazy)
        self._vision_memory: Optional[VisionMemory] = None

        self.logger.info(
            f"VisionAgent initialised | device={self.device} | GPU={self.use_gpu}"
        )

    # ── Repr (Enhancement J) ───────────────────────────────────────────────
    def __repr__(self) -> str:
        loaded = [k for k, v in self.models_loaded.items() if v]
        return f"<VisionAgent device={self.device} loaded={loaded}>"

    # ── Startup ────────────────────────────────────────────────────────────

    async def initialize(
        self, preload_models: Optional[List[str]] = None, *args, **kwargs
    ) -> None:
        """
        Enhancement A – proper async startup.

        Loads known faces from disk and optionally pre-warms the specified
        models.  Call this after construction before processing any images.
        """
        await self.load_known_faces()
        if preload_models:
            await self.load_models(preload_models)
        self.logger.info("VisionAgent ready")

    # ── Vision Engine connection ───────────────────────────────────────────

    async def connect_vision_engine(self, vision_engine: VisionEngine) -> None:
        self.vision_engine = vision_engine
        self.logger.info("VisionAgent connected to VisionEngine")

    async def start_live_vision(self, interval: float = 0.05) -> None:
        if not self.vision_engine:
            self.logger.error("VisionEngine not connected.")
            return

        self.logger.info("🚀 Live Vision started")
        while True:
            try:
                result = await self.process_live_frame()
                if result.get("status") == "processed":
                    self.logger.debug(
                        f"[VISION] objects={result['detections']} "
                        f"faces={result['faces']} fps={result['fps']}"
                    )
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                self.logger.info("Live vision loop cancelled")
                break
            except Exception as e:
                # FIX #17 – log + separate recovery sleep, don't swallow stack
                self.logger.error(f"Vision loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    # ── Model loading ──────────────────────────────────────────────────────

    async def load_models(self, models: Optional[List[str]] = None) -> None:
        if models is None:
            models = ["captioning", "detection", "ocr"]

        for name in models:
            # FIX #3 – skip if already loaded or already attempted (missing lib)
            if self.models_loaded.get(name):
                continue
            if name in self._model_load_attempted:
                continue
            self._model_load_attempted.add(name)

            if name == "captioning":
                await self._load_captioning_model()
            elif name == "detection":
                await self._load_detection_model()
            elif name == "ocr":
                await self._load_ocr_model()
            elif name == "clip":
                await self._load_clip_model()
            elif name == "yolo":
                await self._load_yolo_model()

    async def _load_captioning_model(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("transformers not installed — captioning unavailable")
            return
        try:
            self.logger.info("Loading BLIP captioning model…")
            self.caption_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            if self.use_gpu:
                self.caption_model = self.caption_model.to(self.device)
            self.models_loaded["captioning"] = True
            self.logger.info("✓ Captioning model loaded")
        except Exception as e:
            self.logger.error(f"Captioning model load failed: {e}")

    async def _load_detection_model(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("transformers not installed — DETR unavailable")
            return
        try:
            self.logger.info("Loading DETR detection model…")
            self.detection_processor = DetrImageProcessor.from_pretrained(
                "facebook/detr-resnet-50"
            )
            self.detection_model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50"
            )
            if self.use_gpu:
                self.detection_model = self.detection_model.to(self.device)
            self.models_loaded["detection"] = True
            self.logger.info("✓ DETR detection model loaded")
        except Exception as e:
            self.logger.error(f"DETR model load failed: {e}")

    async def _load_ocr_model(self) -> None:
        if not EASYOCR_AVAILABLE:
            self.logger.warning("easyocr not installed — OCR unavailable")
            return
        try:
            self.logger.info("Loading EasyOCR model…")
            languages = self.config.get(
                "ocr_languages", ["en", "fr", "de", "es", "zh", "ja"]
            )
            self.ocr_reader = easyocr.Reader(
                languages,
                gpu=self.use_gpu,
                model_storage_directory="./models/easyocr",
            )
            self.models_loaded["ocr"] = True
            self.logger.info(f"✓ OCR model loaded (languages: {languages})")
        except Exception as e:
            self.logger.error(f"OCR model load failed: {e}")

    async def _load_clip_model(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            self.logger.info("Loading CLIP model…")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            if self.use_gpu:
                self.clip_model = self.clip_model.to(self.device)
            self.models_loaded["clip"] = True
            self.logger.info("✓ CLIP model loaded")
        except Exception as e:
            self.logger.error(f"CLIP model load failed: {e}")

    async def _load_yolo_model(self) -> None:
        if not YOLO_AVAILABLE:
            self.logger.warning("ultralytics not installed — YOLO unavailable")
            return
        try:
            model_name = self.config.get("yolo_model", "yolov8n.pt")
            self.logger.info(f"Loading YOLO model ({model_name})…")
            self.yolo_model = YOLO(model_name)
            self.models_loaded["yolo"] = True
            self.logger.info("✓ YOLO model loaded")
        except Exception as e:
            self.logger.error(f"YOLO model load failed: {e}")

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _get_cache_key(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        task: str,
    ) -> str:
        # FIX #22 – fast path for file-backed images
        if isinstance(image_source, (str, Path)):
            src = str(image_source)
            try:
                mtime = Path(src).stat().st_mtime
                return f"{src}:{mtime}:{task}"
            except OSError:
                return f"{src}:{task}"

        if isinstance(image_source, Image.Image):
            # Use filename if available (fast path)
            fn = getattr(image_source, "filename", None)
            if fn:
                try:
                    mtime = Path(fn).stat().st_mtime
                    return f"{fn}:{mtime}:{task}"
                except OSError:
                    pass
            buf = io.BytesIO()
            image_source.save(buf, format="PNG")
            h = hashlib.md5(buf.getvalue()).hexdigest()
            return f"pil:{h}:{task}"

        if isinstance(image_source, np.ndarray):
            h = hashlib.md5(image_source.tobytes()).hexdigest()
            return f"ndarray:{h}:{task}"

        if isinstance(image_source, bytes):
            h = hashlib.md5(image_source).hexdigest()
            return f"bytes:{h}:{task}"

        return f"unknown:{id(image_source)}:{task}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if not self.cache_enabled:
            return None
        entry = self.cache.get(key)
        if entry is None:
            return None
        # FIX #12 – use total_seconds(), not .seconds
        if (datetime.now() - entry["timestamp"]).total_seconds() >= 3600:
            del self.cache[key]
            return None
        return entry["result"]

    def _add_to_cache(self, key: str, result: Any) -> None:
        if not self.cache_enabled:
            return
        if len(self.cache) >= self.cache_size:
            oldest = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest]
        self.cache[key] = {"result": result, "timestamp": datetime.now()}

    # ── Image loading ──────────────────────────────────────────────────────

    def _load_image(
        self,
        source: Union[str, Path, "Image.Image", bytes, np.ndarray],
    ) -> "Image.Image":
        """Enhancement H – also accepts np.ndarray."""
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(source)}")

    # ── Rolling average helper (Enhancement F / FIX #21) ──────────────────

    @staticmethod
    def _rolling_average(current_avg: float, new_value: float, n: int) -> float:
        """Correct incremental mean: call AFTER incrementing n."""
        if n <= 1:
            return new_value
        return (current_avg * (n - 1) + new_value) / n

    # ── Core vision methods ────────────────────────────────────────────────

    async def generate_caption(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        max_length: int = 50,
        num_captions: int = 1,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Generate image caption(s) using BLIP."""
        if not self.models_loaded["captioning"]:
            await self.load_models(["captioning"])
            if not self.models_loaded["captioning"]:
                return {"success": False, "error": "Captioning model not available"}

        cache_key = self._get_cache_key(
            image_source, f"caption_{max_length}_{num_captions}"
        )
        if use_cache:
            hit = self._get_from_cache(cache_key)
            if hit:
                return hit

        start = _time.monotonic()
        try:
            image = self._load_image(image_source)

            # FIX #4 – max_length does NOT go to the processor
            inputs = self.caption_processor(image, return_tensors="pt")
            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            captions: List[str] = []
            for i in range(max(1, num_captions)):
                # FIX #5 – vary sampling for multiple captions
                gen_kwargs: Dict[str, Any] = dict(
                    max_length=max_length,
                    num_beams=4 if num_captions == 1 else 1,
                )
                if num_captions > 1 and i > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = 0.8 + i * 0.1

                with torch.no_grad():
                    output = self.caption_model.generate(**inputs, **gen_kwargs)

                text = self.caption_processor.decode(
                    output[0], skip_special_tokens=True
                )
                captions.append(text)

            elapsed = _time.monotonic() - start

            # FIX #21 – snapshot count before increment
            self.stats["total_processed"] += 1
            self.stats["average_processing_time"] = self._rolling_average(
                self.stats["average_processing_time"],
                elapsed,
                self.stats["total_processed"],
            )

            result = {
                "success": True,
                "captions": captions,
                "primary_caption": captions[0] if captions else None,
                "num_captions": len(captions),
                "processing_time": round(elapsed, 4),
            }
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Caption generation error: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}

    async def detect_objects(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        model: DetectionModel = DetectionModel.DETR,
        confidence_threshold: Optional[float] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Detect objects using DETR or YOLOv8."""
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        if model == DetectionModel.DETR:
            if not self.models_loaded["detection"]:
                await self.load_models(["detection"])
                if not self.models_loaded["detection"]:
                    return {"success": False, "error": "DETR model not available"}
        elif model == DetectionModel.YOLO:
            if not self.models_loaded["yolo"]:
                await self.load_models(["yolo"])
                if not self.models_loaded["yolo"]:
                    return {"success": False, "error": "YOLO model not available"}

        cache_key = self._get_cache_key(
            image_source, f"detection_{model.value}_{threshold}"
        )
        if use_cache:
            hit = self._get_from_cache(cache_key)
            if hit:
                return hit

        start = _time.monotonic()
        try:
            image = self._load_image(image_source)
            objects: List[DetectedObject] = []

            if model == DetectionModel.DETR:
                inputs = self.detection_processor(images=image, return_tensors="pt")
                if self.use_gpu:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.detection_model(**inputs)

                # FIX #6 – create tensor on the correct device
                target_sizes = torch.tensor(
                    [image.size[::-1]],  # (height, width)
                    device=self.detection_model.device,
                )
                results = self.detection_processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=threshold
                )[0]

                for score, label, box in zip(
                    results["scores"], results["labels"], results["boxes"]
                ):
                    name = self.detection_model.config.id2label[label.item()]
                    objects.append(
                        DetectedObject(
                            label=name,
                            confidence=round(score.item(), 4),
                            bbox=tuple(map(int, box.tolist())),
                        )
                    )

            elif model == DetectionModel.YOLO:
                yolo_results = self.yolo_model(image)
                for res in yolo_results:
                    boxes = res.boxes
                    # FIX #7 – proper empty-boxes guard
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        conf = box.conf[0].item()
                        if conf < threshold:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0].item())
                        objects.append(
                            DetectedObject(
                                label=self.yolo_model.names[cls],
                                confidence=round(conf, 4),
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                            )
                        )

            elapsed = _time.monotonic() - start

            # FIX #21
            self.stats["total_processed"] += 1
            self.stats["total_objects_detected"] += len(objects)
            self.stats["average_processing_time"] = self._rolling_average(
                self.stats["average_processing_time"],
                elapsed,
                self.stats["total_processed"],
            )

            result = {
                "success": True,
                "objects": [
                    {"label": o.label, "confidence": o.confidence, "bbox": o.bbox}
                    for o in objects
                ],
                "total_objects": len(objects),
                "processing_time": round(elapsed, 4),
                "model": model.value,
            }
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Object detection error: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}

    async def extract_text(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        languages: Optional[List[str]] = None,
        paragraph_mode: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Extract text using EasyOCR."""
        if not self.models_loaded["ocr"]:
            await self.load_models(["ocr"])
            if not self.models_loaded["ocr"]:
                return {
                    "success": False,
                    "error": "OCR model not available. Install: pip install easyocr",
                }

        cache_key = self._get_cache_key(image_source, f"ocr_{paragraph_mode}")
        if use_cache:
            hit = self._get_from_cache(cache_key)
            if hit:
                return hit

        start = _time.monotonic()
        try:
            image = self._load_image(image_source)
            # FIX #8 – numpy already imported at module level
            image_np = np.array(image)

            detections = self.ocr_reader.readtext(image_np, paragraph=paragraph_mode)

            text_blocks = []
            full_lines: List[str] = []
            for bbox, text, confidence in detections:
                text_blocks.append(
                    {"text": text, "confidence": confidence, "bbox": bbox}
                )
                full_lines.append(text)

            elapsed = _time.monotonic() - start

            # FIX #21
            self.stats["total_processed"] += 1
            self.stats["total_text_extracted"] += len(text_blocks)
            self.stats["average_processing_time"] = self._rolling_average(
                self.stats["average_processing_time"],
                elapsed,
                self.stats["total_processed"],
            )

            result = {
                "success": True,
                "text_blocks": text_blocks,
                "full_text": "\n".join(full_lines),
                "total_blocks": len(text_blocks),
                "processing_time": round(elapsed, 4),
            }
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Text extraction error: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}

    async def detect_faces(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        recognize_faces: bool = False,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Detect (and optionally recognise) faces."""
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "success": False,
                "error": "face-recognition not installed. Install: pip install face-recognition",
            }

        start = _time.monotonic()
        try:
            image = self._load_image(image_source)
            # FIX #9 – numpy already imported at module level
            image_np = np.array(image)

            face_locations = face_recognition.face_locations(image_np)
            face_encodings = face_recognition.face_encodings(image_np, face_locations)

            faces: List[DetectedFace] = []
            for location, encoding in zip(face_locations, face_encodings):
                face = DetectedFace(
                    location=location,
                    encoding=encoding.tolist() if recognize_faces else None,
                    confidence=1.0,
                )
                if recognize_faces and self.known_face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, encoding
                    )
                    if True in matches:
                        idx = matches.index(True)
                        face.name = self.known_face_names[idx]
                        face.confidence = 0.8
                faces.append(face)

            elapsed = _time.monotonic() - start

            # FIX #21
            self.stats["total_processed"] += 1
            self.stats["total_faces_detected"] += len(faces)
            self.stats["average_processing_time"] = self._rolling_average(
                self.stats["average_processing_time"],
                elapsed,
                self.stats["total_processed"],
            )

            return {
                "success": True,
                "faces": [
                    {
                        "location": {
                            "top": f.location[0],
                            "right": f.location[1],
                            "bottom": f.location[2],
                            "left": f.location[3],
                        },
                        "name": f.name,
                        "confidence": f.confidence,
                    }
                    for f in faces
                ],
                "total_faces": len(faces),
                "processing_time": round(elapsed, 4),
            }

        except Exception as e:
            self.logger.error(f"Face detection error: {e}", exc_info=True)
            self.stats["errors"] += 1
            return {"success": False, "error": str(e)}

    async def analyze_colors(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        num_colors: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Extract dominant colours using k-means clustering."""
        # Enhancement B – guard for missing sklearn
        if not SKLEARN_AVAILABLE:
            return {
                "success": False,
                "error": "scikit-learn not installed. Install: pip install scikit-learn",
            }

        cache_key = self._get_cache_key(image_source, f"colors_{num_colors}")
        if use_cache:
            hit = self._get_from_cache(cache_key)
            if hit:
                return hit

        start = _time.monotonic()
        try:
            image = self._load_image(image_source)
            image.thumbnail((200, 200))

            # FIX #11 – KMeans already imported at module level
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3)

            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)

            centers = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(kmeans.labels_)
            percentages = (counts / len(kmeans.labels_) * 100).tolist()

            color_analysis = []
            for i, center in enumerate(centers):
                rgb = tuple(center.tolist())
                hex_code = "#{:02x}{:02x}{:02x}".format(*rgb)
                color_analysis.append(
                    {
                        "rgb": rgb,
                        "hex": hex_code,
                        "percentage": round(percentages[i], 2),
                        "name": self._get_color_name(rgb),
                    }
                )

            # Sort by percentage descending
            color_analysis.sort(key=lambda c: c["percentage"], reverse=True)

            elapsed = _time.monotonic() - start
            result = {
                "success": True,
                "dominant_colors": color_analysis,
                "total_colors": len(color_analysis),
                "processing_time": round(elapsed, 4),
            }
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Color analysis error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Enhancement I – extended colour naming with grey detection."""
        r, g, b = rgb
        brightness = (r + g + b) / 3

        if r > 200 and g > 200 and b > 200:
            return "white"
        if r < 50 and g < 50 and b < 50:
            return "black"

        # Grey / Gray detection
        spread = max(r, g, b) - min(r, g, b)
        if spread < 30:
            if brightness > 180:
                return "light grey"
            if brightness > 100:
                return "grey"
            return "dark grey"

        if r > 200 and g < 100 and b < 100:
            return "red"
        if r < 100 and g > 200 and b < 100:
            return "green"
        if r < 100 and g < 100 and b > 200:
            return "blue"
        if r > 200 and g > 200 and b < 100:
            return "yellow"
        if r > 200 and g < 100 and b > 200:
            return "magenta"
        if r < 100 and g > 200 and b > 200:
            return "cyan"
        if r > 200 and g > 100 and b < 100:
            return "orange"
        if r > 200 and g < 100 and b > 100:
            return "pink"
        if r > 100 and g > 100 and b < 100:
            return "brown"
        if r > 100 and g < 100 and b > 100:
            return "purple"
        if r > 100 and g > 200 and b > 100:
            return "light green"

        return "rgb({},{},{})".format(r, g, b)

    async def classify_image(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        candidate_labels: List[str],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Zero-shot image classification using CLIP."""
        if not self.models_loaded["clip"]:
            await self.load_models(["clip"])
            if not self.models_loaded["clip"]:
                return {"success": False, "error": "CLIP model not available"}

        cache_key = self._get_cache_key(
            image_source, f"classify_{'_'.join(candidate_labels)}"
        )
        if use_cache:
            hit = self._get_from_cache(cache_key)
            if hit:
                return hit

        start = _time.monotonic()
        try:
            image = self._load_image(image_source)
            inputs = self.clip_processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]

            predictions = sorted(
                [
                    {"label": lbl, "confidence": round(p.item(), 4)}
                    for lbl, p in zip(candidate_labels, probs)
                ],
                key=lambda x: x["confidence"],
                reverse=True,
            )

            elapsed = _time.monotonic() - start
            result = {
                "success": True,
                "predictions": predictions,
                "top_label": predictions[0]["label"],
                "top_confidence": predictions[0]["confidence"],
                "processing_time": round(elapsed, 4),
            }
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            self.logger.error(f"Classification error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def analyze_image(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        tasks: Optional[List[VisionTask]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Run multiple vision tasks on a single image."""
        if tasks is None:
            tasks = [
                VisionTask.CAPTIONING,
                VisionTask.OBJECT_DETECTION,
                VisionTask.OCR,
                VisionTask.COLOR_ANALYSIS,
            ]

        start = _time.monotonic()
        cache_key = self._get_cache_key(
            image_source, f"analysis_{'_'.join(t.value for t in tasks)}"
        )
        if use_cache:
            hit = self._get_from_cache(cache_key)
            if hit:
                return hit

        results: Dict[str, Any] = {}

        task_dispatch = {
            VisionTask.CAPTIONING: (
                "captioning",
                lambda: self.generate_caption(image_source, use_cache=use_cache),
            ),
            VisionTask.OBJECT_DETECTION: (
                "object_detection",
                lambda: self.detect_objects(image_source, use_cache=use_cache),
            ),
            VisionTask.OCR: (
                "ocr",
                lambda: self.extract_text(image_source, use_cache=use_cache),
            ),
            VisionTask.COLOR_ANALYSIS: (
                "color_analysis",
                lambda: self.analyze_colors(image_source, use_cache=use_cache),
            ),
            VisionTask.FACE_RECOGNITION: (
                "face_recognition",
                lambda: self.detect_faces(image_source, use_cache=use_cache),
            ),
        }

        for task in tasks:
            if task in task_dispatch:
                key, coro_fn = task_dispatch[task]
                results[key] = await coro_fn()

        elapsed = _time.monotonic() - start

        # FIX #13 – success if at least one task succeeded
        any_success = any(r.get("success", False) for r in results.values())

        analysis = {
            "success": any_success,
            "results": results,
            "total_tasks": len(tasks),
            "processing_time": round(elapsed, 4),
            "timestamp": datetime.now().isoformat(),
        }
        self._add_to_cache(cache_key, analysis)
        return analysis

    # ── Face registration ──────────────────────────────────────────────────

    async def register_known_face(
        self,
        image_source: Union[str, Path, "Image.Image", bytes, np.ndarray],
        name: str,
    ) -> Dict[str, Any]:
        if not FACE_RECOGNITION_AVAILABLE:
            return {"success": False, "error": "face-recognition not available"}

        try:
            image = self._load_image(image_source)
            # FIX #10
            image_np = np.array(image)

            locations = face_recognition.face_locations(image_np)
            if not locations:
                return {"success": False, "error": "No face found in image"}

            # FIX #19 – reject ambiguous images with multiple faces
            if len(locations) > 1:
                return {
                    "success": False,
                    "error": f"Found {len(locations)} faces. Please provide an image with exactly one face.",
                }

            encoding = face_recognition.face_encodings(image_np, locations)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

            data = {
                "encodings": [e.tolist() for e in self.known_face_encodings],
                "names": self.known_face_names,
            }
            with open(self.known_faces_dir / "known_faces.json", "w") as f:
                json.dump(data, f)

            return {
                "success": True,
                "name": name,
                "message": f"Face registered for {name}",
            }

        except Exception as e:
            self.logger.error(f"Face registration error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def load_known_faces(self) -> None:
        """FIX #18 – load with correct dtype for face_recognition."""
        path = self.known_faces_dir / "known_faces.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            # FIX #18 – face_recognition internally uses float64
            self.known_face_encodings = [
                np.array(enc, dtype=np.float64) for enc in data["encodings"]
            ]
            self.known_face_names = data["names"]
            self.logger.info(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {e}")

    # ── Live frame processing ──────────────────────────────────────────────

    async def process_live_frame(self) -> Dict[str, Any]:
        if not self.vision_engine:
            return {"status": "no_engine"}

        try:
            frame = getattr(self.vision_engine, "current_frame", None)
            if frame is None:
                return {"status": "no_frame", "fps": 0}
            if not isinstance(frame, np.ndarray):
                return {"status": "invalid_frame", "fps": 0}

            detections = getattr(self.vision_engine, "current_detections", []) or []
            faces = getattr(self.vision_engine, "current_faces", []) or []
            caption = getattr(self.vision_engine, "current_caption", None)
            fps = getattr(self.vision_engine, "fps", 0)

            # FIX #1 / #24 – use stdlib time module
            timestamp = _time.time()

            frame_data = {
                "detections": detections,
                "faces": faces,
                "caption": caption,
                "timestamp": timestamp,
                "fps": fps,
            }

            try:
                if self._vision_memory is None:
                    self._vision_memory = VisionMemory()
                await self._vision_memory.publish_frame(frame_data)
            except Exception as e:
                self.logger.debug(f"Memory publish error: {e}")

            return {
                "status": "processed",
                "detections": len(detections),
                "faces": len(faces),
                "fps": fps,
            }

        except Exception as e:
            self.logger.error(f"Live frame fatal error: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "fps": 0}

    async def process_video_frame(
        self,
        frame: np.ndarray,
        tasks: Optional[List[VisionTask]] = None,
    ) -> Dict[str, Any]:
        """Process a raw video frame (numpy array)."""
        return await self.analyze_image(frame, tasks=tasks, use_cache=False)

    # ── Annotation (Enhancement C / D) ────────────────────────────────────

    def annotate_image(
        self,
        image: "Image.Image",
        objects: Optional[List[Dict]] = None,
        faces: Optional[List[Dict]] = None,
        caption: Optional[str] = None,
    ) -> "Image.Image":
        """
        Enhancement C – draw bounding boxes, labels, and an optional caption
        onto a PIL Image. Returns a new annotated copy.
        """
        annotated = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font_small = ImageFont.truetype("arial.ttf", 14)
            font_large = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font_small = ImageFont.load_default()
            font_large = font_small

        # Draw detected objects
        if objects:
            for obj in objects:
                x1, y1, x2, y2 = obj.get("bbox", (0, 0, 0, 0))
                label = f"{obj.get('label', '?')} {obj.get('confidence', 0):.0%}"
                draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80, 220), width=2)
                draw.rectangle(
                    [x1, y1 - 18, x1 + len(label) * 8, y1], fill=(255, 80, 80, 180)
                )
                draw.text(
                    (x1 + 2, y1 - 16), label, fill=(255, 255, 255, 255), font=font_small
                )

        # Draw detected faces
        if faces:
            for face in faces:
                loc = face.get("location", {})
                top, right, bottom, left = (
                    loc.get("top", 0),
                    loc.get("right", 0),
                    loc.get("bottom", 0),
                    loc.get("left", 0),
                )
                name = face.get("name", "Unknown")
                draw.rectangle(
                    [left, top, right, bottom], outline=(80, 200, 80, 220), width=2
                )
                draw.text(
                    (left, top - 18), name, fill=(80, 255, 80, 255), font=font_small
                )

        # Overlay caption at bottom
        if caption:
            w, h = annotated.size
            draw.rectangle([0, h - 30, w, h], fill=(0, 0, 0, 160))
            draw.text((8, h - 24), caption, fill=(255, 255, 255, 255), font=font_large)

        result = Image.alpha_composite(annotated, overlay).convert("RGB")
        return result

    def save_annotated_image(
        self,
        image: "Image.Image",
        output_path: Union[str, Path],
        objects: Optional[List[Dict]] = None,
        faces: Optional[List[Dict]] = None,
        caption: Optional[str] = None,
    ) -> str:
        """Enhancement D – annotate and save to disk, return the path."""
        annotated = self.annotate_image(
            image, objects=objects, faces=faces, caption=caption
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path)
        return str(output_path)

    # ── Batch processing (Enhancement E) ──────────────────────────────────

    async def batch_process(
        self,
        image_sources: List[Union[str, Path, "Image.Image", bytes, np.ndarray]],
        tasks: Optional[List[VisionTask]] = None,
        concurrency: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Enhancement E – process a list of images concurrently.
        `concurrency` caps the number of simultaneous analyses.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _process_one(src):
            async with semaphore:
                return await self.analyze_image(src, tasks=tasks)

        return await asyncio.gather(*(_process_one(src) for src in image_sources))

    # ── Statistics & cache ─────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Enhancement G – return a copy so callers cannot mutate internals."""
        stats = dict(self.stats)
        stats["models_loaded"] = dict(self.models_loaded)
        stats["device"] = self.device
        stats["cache_size"] = len(self.cache)
        stats["known_faces"] = len(self.known_face_names)

        if self.vision_engine:
            stats["vision_engine"] = {
                "active": getattr(self.vision_engine, "is_active", False),
                "fps": getattr(self.vision_engine, "fps", 0),
                "frames": getattr(self.vision_engine, "frames_processed", 0),
            }

        return stats

    async def clear_cache(self) -> None:
        self.cache.clear()
        self.logger.info("Vision cache cleared")


# ══════════════════════════════════════════════════════════════════════════════
# VisionAgentWrapper
# ══════════════════════════════════════════════════════════════════════════════


class VisionAgentWrapper:
    """Integrates VisionAgent with EDIATH's agent architecture."""

    def __init__(self, config: Optional[Dict] = None):
        self.vision_agent = VisionAgent(config)
        self.agent_type = "vision_understanding"
        self.capabilities = [
            "image_captioning",
            "object_detection",
            "ocr_text_extraction",
            "face_detection",
            "color_analysis",
            "image_classification",
            "scene_understanding",
            "batch_processing",
        ]

    async def initialize(
        self, preload_models: Optional[List[str]] = None, *args, **kwargs
    ) -> None:
        await self.vision_agent.initialize(preload_models)

    async def connect_to_vision_engine(self, vision_engine: VisionEngine) -> None:
        await self.vision_agent.connect_vision_engine(vision_engine)

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a vision request dict.

        Required keys:
            operation – one of: caption, detect, ocr, faces, colors, classify,
                        analyze, register_face, start_live, stats, clear_cache
            image     – file path, base64 data-URI, http(s) URL, or raw bytes
        """
        operation = request.get("operation")
        image_raw = request.get("image")

        # FIX #14 / #15 – always set image_source before dispatch
        image_source: Any = None

        if isinstance(image_raw, str) and image_raw.startswith("data:image"):
            try:
                b64 = image_raw.split(",", 1)[1]
                image_source = base64.b64decode(b64)
            except Exception as e:
                return {"success": False, "error": f"Base64 decode failed: {e}"}

        elif isinstance(image_raw, str) and image_raw.startswith(
            ("http://", "https://")
        ):
            # FIX #16 – guard aiohttp import
            try:
                import aiohttp
            except ImportError:
                return {
                    "success": False,
                    "error": "aiohttp not installed. Install: pip install aiohttp",
                }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_raw) as resp:
                        image_source = await resp.read()
            except Exception as e:
                return {"success": False, "error": f"URL download failed: {e}"}

        else:
            image_source = image_raw  # path, PIL Image, bytes, ndarray, or None

        # ── Dispatch ───────────────────────────────────────────────────────
        if operation == "caption":
            return await self.vision_agent.generate_caption(
                image_source,
                max_length=request.get("max_length", 50),
                num_captions=request.get("num_captions", 1),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "detect":
            model_name = request.get("model", "detr")
            try:
                model = DetectionModel(model_name)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Unknown detection model: {model_name}",
                }
            return await self.vision_agent.detect_objects(
                image_source,
                model=model,
                confidence_threshold=request.get("confidence_threshold"),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "ocr":
            return await self.vision_agent.extract_text(
                image_source,
                languages=request.get("languages"),
                paragraph_mode=request.get("paragraph_mode", False),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "faces":
            return await self.vision_agent.detect_faces(
                image_source,
                recognize_faces=request.get("recognize_faces", False),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "colors":
            return await self.vision_agent.analyze_colors(
                image_source,
                num_colors=request.get("num_colors", 5),
                use_cache=request.get("use_cache", True),
            )

        elif operation == "classify":
            labels = request.get("labels", [])
            if not labels:
                return {
                    "success": False,
                    "error": "'labels' list is required for classify",
                }
            return await self.vision_agent.classify_image(
                image_source,
                candidate_labels=labels,
                use_cache=request.get("use_cache", True),
            )

        elif operation == "analyze":
            task_names = request.get(
                "tasks", ["captioning", "object_detection", "ocr", "color_analysis"]
            )
            try:
                vtasks = [VisionTask(t) for t in task_names]
            except ValueError as e:
                return {"success": False, "error": f"Unknown task: {e}"}
            return await self.vision_agent.analyze_image(
                image_source,
                tasks=vtasks,
                use_cache=request.get("use_cache", True),
            )

        elif operation == "register_face":
            name = request.get("name")
            if not name:
                return {
                    "success": False,
                    "error": "'name' is required for register_face",
                }
            return await self.vision_agent.register_known_face(image_source, name)

        elif operation == "start_live":
            if not self.vision_agent.vision_engine:
                return {"success": False, "error": "Vision engine not connected"}
            # FIX #25 – guard against no-running-loop context
            try:
                asyncio.ensure_future(self.vision_agent.start_live_vision())
                return {"success": True, "message": "Live vision started"}
            except RuntimeError as e:
                return {"success": False, "error": f"Could not start live vision: {e}"}

        elif operation == "stats":
            return self.vision_agent.get_stats()

        elif operation == "clear_cache":
            await self.vision_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation!r}"}

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "VisionAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.vision_agent.get_stats(),
            "device": self.vision_agent.device,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Test / demo
# ══════════════════════════════════════════════════════════════════════════════


async def test_vision_agent():
    print("=" * 55)
    print("  Vision Agent — self-test")
    print("=" * 55)

    agent = VisionAgent()
    await agent.initialize(preload_models=["captioning", "detection", "ocr"])
    print(f"\nAgent: {agent!r}\n")

    # Create a synthetic test image
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 300], fill="red", outline="black", width=3)
    draw.rectangle([400, 100, 600, 300], fill="blue", outline="black", width=3)
    draw.ellipse([200, 350, 400, 550], fill="green", outline="black", width=3)
    draw.text((300, 20), "EDIATH Vision Test", fill="black")
    draw.text((150, 240), "RED SQUARE", fill="white")
    draw.text((450, 240), "BLUE SQUARE", fill="white")
    draw.text((270, 440), "GREEN CIRCLE", fill="white")

    test_path = Path("test_vision_image.jpg")
    img.save(test_path)
    print(f"Test image saved: {test_path}")

    # 1. Caption
    print("\n[1] Image Captioning…")
    r = await agent.generate_caption(str(test_path))
    if r["success"]:
        print(f"    Caption: {r['primary_caption']}")
        print(f"    Time:    {r['processing_time']:.3f}s")

    # 2. Object detection
    print("\n[2] Object Detection (DETR)…")
    r = await agent.detect_objects(str(test_path))
    if r["success"]:
        print(f"    Objects: {r['total_objects']}")
        for o in r["objects"][:5]:
            print(f"      {o['label']} ({o['confidence']:.2%})")

    # 3. OCR
    print("\n[3] Text Extraction (OCR)…")
    r = await agent.extract_text(str(test_path))
    if r["success"]:
        print(f"    Blocks: {r['total_blocks']}")
        print(f"    Text:   {r['full_text']}")

    # 4. Colour analysis
    print("\n[4] Colour Analysis…")
    r = await agent.analyze_colors(str(test_path), num_colors=3)
    if r["success"]:
        for c in r["dominant_colors"]:
            print(f"    {c['name']:15s} {c['hex']}  {c['percentage']:.1f}%")

    # 5. Full analysis
    print("\n[5] Full Analysis…")
    r = await agent.analyze_image(str(test_path))
    print(f"    Success:  {r['success']}")
    print(f"    Tasks:    {r['total_tasks']}")
    print(f"    Time:     {r['processing_time']:.3f}s")

    # 6. Annotation
    print("\n[6] Annotated Image…")
    objects_sample = r["results"].get("object_detection", {}).get("objects", [])
    caption_sample = r["results"].get("captioning", {}).get("primary_caption") or ""
    annotated_path = agent.save_annotated_image(
        img,
        "annotated_output.jpg",
        objects=objects_sample,
        caption=caption_sample,
    )
    print(f"    Saved: {annotated_path}")

    # 7. Batch
    print("\n[7] Batch Processing (2 images)…")
    batch = await agent.batch_process(
        [str(test_path), img],
        tasks=[VisionTask.CAPTIONING, VisionTask.COLOR_ANALYSIS],
    )
    for i, br in enumerate(batch):
        print(f"    [{i}] success={br['success']} time={br['processing_time']:.3f}s")

    # 8. Stats
    print("\n[8] Statistics…")
    s = agent.get_stats()
    print(f"    Processed: {s['total_processed']}")
    print(f"    Errors:    {s['errors']}")
    print(f"    Avg time:  {s['average_processing_time']:.4f}s")
    print(f"    Cache:     {s['cache_size']} entries")

    print("\n✅ Test complete")


if __name__ == "__main__":
    asyncio.run(test_vision_agent())
