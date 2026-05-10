# captioner.py

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)
import numpy as np
import hashlib
import time

try:
    import cv2
except ImportError:
    cv2 = None


# ------------------------
# CAPTION OBJECT
# ------------------------
class ImageCaption:
    def __init__(self, text: str, confidence: float = 0.8):
        self.text = text
        self.confidence = float(confidence)
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


# ------------------------
# CAPTIONER
# ------------------------
class Captioner:
    def __init__(self, model_name: str = "rule-based", max_cache: int = 100):
        self.model_name = model_name
        self.captions_generated = 0
        self.caption_cache: Dict[str, ImageCaption] = {}
        self.max_cache = max_cache

        self.has_numpy = np is not None
        self.has_cv2 = cv2 is not None

        self.model = None  # future AI model

        logger.info("✅ Captioner Initialized (Improved)")

    # ------------------------
    # HASH IMAGE
    # ------------------------
    def _hash_image(self, image: np.ndarray) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()

    # ------------------------
    # ADVANCED ANALYSIS (IMPROVED)
    # ------------------------
    def _analyze_image(self, image: np.ndarray) -> str:
        if image is None or not isinstance(image, np.ndarray):
            return "invalid image"

        h, w = image.shape[:2]

        # Handle grayscale safely
        if len(image.shape) == 2:
            brightness = np.mean(image)
            variance = np.var(image)
            color_desc = "grayscale"
        else:
            brightness = np.mean(image)
            variance = np.var(image)

            # 🔥 dominant color
            avg_color = np.mean(image.reshape(-1, 3), axis=0)
            b, g, r = avg_color

            if r > g and r > b:
                color_desc = "reddish"
            elif g > r and g > b:
                color_desc = "greenish"
            elif b > r and b > g:
                color_desc = "bluish"
            else:
                color_desc = "neutral color"

        # lighting
        if brightness < 60:
            lighting = "dark"
        elif brightness > 180:
            lighting = "bright"
        else:
            lighting = "normal lighting"

        # detail
        if variance < 300:
            detail = "simple scene"
        else:
            detail = "detailed scene"

        # orientation
        orientation = "landscape" if w > h else "portrait"

        return f"{lighting}, {color_desc}, {detail}, {orientation}"

    # ------------------------
    # CACHE CONTROL (FIXED)
    # ------------------------
    def _update_cache(self, key: str, caption: ImageCaption):
        if len(self.caption_cache) >= self.max_cache:
            # remove oldest item
            oldest = min(self.caption_cache.items(), key=lambda x: x[1].timestamp)[0]
            del self.caption_cache[oldest]

        self.caption_cache[key] = caption

    # ------------------------
    # MAIN CAPTION
    # ------------------------
    def caption(self, image: np.ndarray) -> ImageCaption:
        if image is None or not isinstance(image, np.ndarray):
            return ImageCaption("Invalid image input", 0.0)

        image_hash = self._hash_image(image)

        # cache
        if image_hash in self.caption_cache:
            return self.caption_cache[image_hash]

        # AI model (future)
        if self.model:
            caption_text = self._run_model(image)
            confidence = 0.9
        else:
            analysis = self._analyze_image(image)
            caption_text = f"A scene with {analysis}"
            confidence = 0.75

        caption = ImageCaption(caption_text, confidence)

        self._update_cache(image_hash, caption)
        self.captions_generated += 1

        return caption

    # ------------------------
    # MODEL HOOK
    # ------------------------
    def _run_model(self, image: np.ndarray) -> str:
        return "AI generated caption"

    # ------------------------
    # BATCH
    # ------------------------
    def caption_batch(self, images: List[np.ndarray]) -> List[ImageCaption]:
        return [self.caption(img) for img in images if img is not None]

    # ------------------------
    # WITH OBJECTS
    # ------------------------
    def caption_with_objects(
        self, image: np.ndarray, objects: List[str]
    ) -> ImageCaption:

        base = self.caption(image)

        if not objects:
            return base

        objects_str = ", ".join(objects)
        text = f"{base.text}. Detected: {objects_str}"

        confidence = min(0.95, base.confidence + len(objects) * 0.03)

        return ImageCaption(text, confidence)

    # ------------------------
    # CONFIDENCE FILTER
    # ------------------------
    def caption_with_confidence(
        self, image: np.ndarray, min_confidence: float = 0.5
    ) -> Optional[ImageCaption]:

        cap = self.caption(image)
        return cap if cap.confidence >= min_confidence else None

    # ------------------------
    # CACHE CONTROL
    # ------------------------
    def clear_cache(self):
        self.caption_cache.clear()

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "captions_generated": self.captions_generated,
            "cache_size": len(self.caption_cache),
            "model": self.model_name,
        }
