# captioner.py

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from collections import deque
import json
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        Blip2Processor, Blip2ForConditionalGeneration,
        VisionEncoderDecoderModel, ViTImageProcessor,
        AutoTokenizer, AutoProcessor, AutoModelForCausalLM
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class CaptionModelType(Enum):
    RULE_BASED = "rule_based"
    BLIP = "blip"
    BLIP2 = "blip2"
    OFA = "ofa"
    VIT_GPT2 = "vit_gpt2"
    CUSTOM = "custom"


class CaptionStyle(Enum):
    DESCRIPTIVE = "descriptive"
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    QUESTION_BASED = "question_based"


class AttentionType(Enum):
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    BOTH = "both"


@dataclass
class CaptionConfig:
    model_type: CaptionModelType = CaptionModelType.BLIP
    style: CaptionStyle = CaptionStyle.DESCRIPTIVE
    max_length: int = 50
    min_length: int = 10
    num_beams: int = 3
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    device: str = "auto"
    use_cache: bool = True
    max_cache_size: int = 1000
    confidence_threshold: float = 0.5
    enable_attention: bool = False
    enable_beam_search: bool = True
    diversity_penalty: float = 0.5
    length_penalty: float = 1.0
    early_stopping: bool = True
    use_gpu: bool = True
    batch_size: int = 8
    
    def __post_init__(self):
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# ------------------------
# ADVANCED CAPTION OBJECT
# ------------------------
class ImageCaption:
    def __init__(
        self,
        text: str,
        confidence: float = 0.8,
        attention_weights: Optional[np.ndarray] = None,
        alternative_captions: Optional[List[str]] = None,
        objects_detected: Optional[List[str]] = None,
        scene_attributes: Optional[Dict] = None
    ):
        self.text = text
        self.confidence = float(confidence)
        self.timestamp = time.time()
        self.attention_weights = attention_weights
        self.alternative_captions = alternative_captions or []
        self.objects_detected = objects_detected or []
        self.scene_attributes = scene_attributes or {}
        self.embedding = None  # For semantic search
        self.word_importance = {}  # Per-word importance scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "alternative_captions": self.alternative_captions,
            "objects_detected": self.objects_detected,
            "scene_attributes": self.scene_attributes,
            "num_words": len(self.text.split()),
            "char_length": len(self.text)
        }

    def __str__(self) -> str:
        return f"Caption: '{self.text}' (conf: {self.confidence:.2f})"


# ------------------------
# ATTENTION VISUALIZATION
# ------------------------
class AttentionVisualizer:
    def __init__(self):
        self.has_cv2 = cv2 is not None
    
    def visualize_attention(
        self,
        image: np.ndarray,
        attention_weights: np.ndarray,
        words: List[str],
        alpha: float = 0.5
    ) -> np.ndarray:
        """Visualize attention weights on image"""
        if not self.has_cv2 or image is None:
            return image
        
        if attention_weights.ndim == 3:
            # Average over attention heads
            attention_weights = attention_weights.mean(axis=0)
        
        # Resize attention to image size
        h, w = image.shape[:2]
        attention_map = cv2.resize(attention_weights, (w, h))
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-6)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def create_word_attention_grid(
        self,
        image: np.ndarray,
        attention_weights: Dict[str, np.ndarray],
        top_k: int = 5
    ) -> np.ndarray:
        """Create grid of attention maps for top words"""
        if not self.has_cv2:
            return image
        
        # Sort words by attention strength
        sorted_words = sorted(attention_weights.items(), key=lambda x: x[1].max(), reverse=True)[:top_k]
        
        if not sorted_words:
            return image
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(len(sorted_words))))
        cell_h, cell_w = image.shape[0] // grid_size, image.shape[1] // grid_size
        
        grid_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        for idx, (word, attn) in enumerate(sorted_words):
            row = idx // grid_size
            col = idx % grid_size
            
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            # Visualize attention for this word
            attn_map = cv2.resize(attn, (cell_w, cell_h))
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
            
            grid_image[y1:y2, x1:x2] = heatmap
            
            # Add word label
            cv2.putText(
                grid_image,
                word,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Blend with original
        return cv2.addWeighted(image, 0.5, grid_image, 0.5, 0)


# ------------------------
# DEEP LEARNING CAPTIONER
# ------------------------
class DeepCaptioner:
    def __init__(self, config: CaptionConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = config.device
        
        self._load_model()
    
    def _load_model(self):
        """Load deep learning model for captioning"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers library not available, using rule-based fallback")
            return
        
        try:
            if self.config.model_type == CaptionModelType.BLIP:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                logger.info("✅ BLIP model loaded")
                
            elif self.config.model_type == CaptionModelType.BLIP2:
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                logger.info("✅ BLIP-2 model loaded")
                
            elif self.config.model_type == CaptionModelType.VIT_GPT2:
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    "nlpconnect/vit-gpt2-image-captioning"
                ).to(self.device)
                self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                logger.info("✅ ViT-GPT2 model loaded")
            
            else:
                logger.warning(f"⚠️ Model type {self.config.model_type} not implemented")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def generate_caption(self, image: np.ndarray) -> Tuple[str, float, Optional[np.ndarray]]:
        """Generate caption using deep learning model"""
        if self.model is None or self.processor is None:
            return "AI model not available", 0.5, None
        
        try:
            # Preprocess image
            if self.config.model_type == CaptionModelType.BLIP:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=self.config.max_length,
                        min_length=self.config.min_length,
                        num_beams=self.config.num_beams if self.config.enable_beam_search else 1,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        repetition_penalty=self.config.repetition_penalty,
                        length_penalty=self.config.length_penalty,
                        early_stopping=self.config.early_stopping
                    )
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
                # Calculate confidence (based on logits)
                confidence = self._calculate_confidence(inputs, out)
                
                # Extract attention if needed
                attention = None
                if self.config.enable_attention and hasattr(out, 'attentions'):
                    attention = out.attentions[-1].cpu().numpy()
                
                return caption, confidence, attention
                
            elif self.config.model_type == CaptionModelType.VIT_GPT2:
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        pixel_values,
                        max_length=self.config.max_length,
                        num_beams=self.config.num_beams,
                        temperature=self.config.temperature
                    )
                
                caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return caption, 0.85, None
            
            else:
                return "Caption generation not implemented", 0.5, None
                
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return "Error generating caption", 0.3, None
    
    def _calculate_confidence(self, inputs, outputs) -> float:
        """Calculate confidence score for generated caption"""
        try:
            with torch.no_grad():
                outputs_model = self.model(**inputs, labels=outputs)
                log_likelihood = -outputs_model.loss.item()
                confidence = 1.0 / (1.0 + np.exp(-log_likelihood))
                return float(np.clip(confidence, 0.5, 0.95))
        except:
            return 0.75


# ------------------------
# RULE-BASED CAPTIONER (ADVANCED)
# ------------------------
class RuleBasedCaptioner:
    def __init__(self):
        self.scene_templates = {
            "dark": ["dimly lit", "moody", "atmospheric"],
            "bright": ["well-lit", "sunny", "vibrant"],
            "landscape": ["panoramic view", "wide shot", "scenic"],
            "portrait": ["close-up", "intimate shot", "framed"],
            "grayscale": ["monochromatic", "black and white", "classic"],
            "detailed": ["intricate", "complex", "rich in detail"],
            "simple": ["minimalist", "clean", "uncluttered"]
        }
        
        self.color_phrases = {
            "reddish": ["warm tones", "fiery hues", "crimson accents"],
            "greenish": ["earthy tones", "verdant shades", "natural hues"],
            "bluish": ["cool tones", "calm blues", "serene shades"],
            "neutral": ["balanced tones", "subtle colors", "muted palette"]
        }
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced image analysis"""
        if image is None or not isinstance(image, np.ndarray):
            return {"valid": False}
        
        h, w = image.shape[:2]
        
        # Handle grayscale
        if len(image.shape) == 2:
            brightness = np.mean(image)
            variance = np.var(image)
            color_desc = "grayscale"
            color_palette = ["gray"]
        else:
            brightness = np.mean(image)
            variance = np.var(image)
            
            # Dominant color analysis
            avg_color = np.mean(image.reshape(-1, 3), axis=0)
            b, g, r = avg_color
            
            # Color detection with thresholds
            if r > g * 1.2 and r > b * 1.2:
                color_desc = "reddish"
                color_palette = ["red", "crimson", "burgundy"]
            elif g > r * 1.2 and g > b * 1.2:
                color_desc = "greenish"
                color_palette = ["green", "emerald", "olive"]
            elif b > r * 1.2 and b > g * 1.2:
                color_desc = "bluish"
                color_palette = ["blue", "azure", "navy"]
            else:
                color_desc = "neutral color"
                color_palette = ["balanced", "neutral", "muted"]
        
        # Lighting classification
        if brightness < 60:
            lighting = "dark"
        elif brightness > 180:
            lighting = "bright"
        else:
            lighting = "normal lighting"
        
        # Detail level
        if variance < 300:
            detail = "simple scene"
        else:
            detail = "detailed scene"
        
        # Orientation
        orientation = "landscape" if w > h else "portrait"
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 1
        
        # Edge density (texture)
        if cv2 is not None:
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            edge_density = 0.1
        
        # Generate descriptive phrases
        style_phrases = []
        if lighting in self.scene_templates:
            style_phrases.extend(self.scene_templates[lighting][:2])
        if orientation in self.scene_templates:
            style_phrases.extend(self.scene_templates[orientation][:1])
        if detail in self.scene_templates:
            style_phrases.extend(self.scene_templates[detail][:1])
        
        return {
            "valid": True,
            "lighting": lighting,
            "color_desc": color_desc,
            "color_palette": color_palette,
            "detail": detail,
            "orientation": orientation,
            "aspect_ratio": aspect_ratio,
            "edge_density": edge_density,
            "brightness": float(brightness),
            "variance": float(variance),
            "style_phrases": style_phrases,
            "color_phrases": self.color_phrases.get(color_desc, ["colorful"])
        }
    
    def generate_caption(self, analysis: Dict[str, Any], style: CaptionStyle) -> str:
        """Generate caption based on analysis and style"""
        if not analysis.get("valid", False):
            return "Unable to analyze image"
        
        # Base template
        templates = {
            CaptionStyle.DESCRIPTIVE: "A {lighting}, {color_desc} {orientation} with {detail}.",
            CaptionStyle.CONCISE: "{lighting} {color_desc} {orientation}.",
            CaptionStyle.DETAILED: "This {orientation} image features {lighting} {color_desc} tones, presenting a {detail}. {style_phrases}",
            CaptionStyle.CREATIVE: "{style_phrases} dance across this {orientation} canvas, where {color_phrases} create a {lighting} atmosphere.",
            CaptionStyle.TECHNICAL: "Image analysis: {orientation} orientation, {lighting} conditions, {color_desc} color distribution, {detail} texture complexity.",
            CaptionStyle.QUESTION_BASED: "What story does this {lighting}, {color_desc} {orientation} scene tell? The {detail} invites closer inspection."
        }
        
        template = templates.get(style, templates[CaptionStyle.DESCRIPTIVE])
        
        # Fill template
        caption = template.format(
            lighting=analysis['lighting'],
            color_desc=analysis['color_desc'],
            orientation=analysis['orientation'],
            detail=analysis['detail'],
            style_phrases=" and ".join(analysis.get('style_phrases', [''])[:2]),
            color_phrases=analysis.get('color_phrases', [''])[0]
        )
        
        # Clean up
        caption = caption.replace("  ", " ").strip()
        
        return caption


# ------------------------
# MAIN CAPTIONER CLASS
# ------------------------
class Captioner:
    def __init__(self, config: Optional[CaptionConfig] = None):
        self.config = config or CaptionConfig()
        
        # Components
        self.rule_based = RuleBasedCaptioner()
        self.deep_captioner = DeepCaptioner(self.config) if TRANSFORMERS_AVAILABLE else None
        self.attention_viz = AttentionVisualizer() if self.config.enable_attention else None
        
        # Cache
        self.caption_cache: Dict[str, ImageCaption] = {}
        self.max_cache = self.config.max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.captions_generated = 0
        self.generation_times = deque(maxlen=100)
        
        # Semantic similarity (for caption ranking)
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Semantic similarity model loaded")
            except:
                pass
        
        logger.info(f"✅ Advanced Captioner Initialized - Model: {self.config.model_type.value}, Style: {self.config.style.value}")
    
    # ------------------------
    # IMAGE HASHING
    # ------------------------
    def _hash_image(self, image: np.ndarray) -> str:
        """Generate perceptual hash of image"""
        # Resize to consistent size for hashing
        if image is None:
            return ""
        
        # For large images, resize to reduce hash computation
        if image.size > 1024 * 1024:
            small = cv2.resize(image, (256, 256)) if cv2 else image
        else:
            small = image
        
        return hashlib.md5(small.tobytes()).hexdigest()
    
    # ------------------------
    # CAPTION GENERATION
    # ------------------------
    def caption(
        self,
        image: np.ndarray,
        style: Optional[CaptionStyle] = None,
        return_alternatives: bool = False
    ) -> ImageCaption:
        """Generate caption for image with advanced features"""
        if image is None or not isinstance(image, np.ndarray):
            return ImageCaption("Invalid image input", 0.0)
        
        start_time = time.time()
        
        # Check cache
        image_hash = self._hash_image(image)
        if self.config.use_cache and image_hash in self.caption_cache:
            self.cache_hits += 1
            return self.caption_cache[image_hash]
        
        self.cache_misses += 1
        
        # Use appropriate captioning method
        use_style = style or self.config.style
        
        if self.deep_captioner and self.deep_captioner.model is not None:
            # Deep learning caption
            text, confidence, attention = self.deep_captioner.generate_caption(image)
            
            # Generate alternatives if requested
            alternatives = []
            if return_alternatives:
                alternatives = self._generate_alternatives(image, text, 3)
            
            caption = ImageCaption(
                text=text,
                confidence=confidence,
                attention_weights=attention,
                alternative_captions=alternatives
            )
            
        else:
            # Rule-based caption
            analysis = self.rule_based.analyze_image(image)
            text = self.rule_based.generate_caption(analysis, use_style)
            
            # Calculate confidence based on analysis quality
            confidence = self._calculate_rule_confidence(analysis)
            
            caption = ImageCaption(
                text=text,
                confidence=confidence,
                scene_attributes=analysis
            )
        
        # Add semantic embedding
        if self.semantic_model:
            caption.embedding = self.semantic_model.encode(caption.text)
        
        # Update cache
        if self.config.use_cache:
            self._update_cache(image_hash, caption)
        
        # Update statistics
        self.captions_generated += 1
        self.generation_times.append(time.time() - start_time)
        
        return caption
    
    def _generate_alternatives(self, image: np.ndarray, base_caption: str, num: int) -> List[str]:
        """Generate alternative captions"""
        alternatives = []
        
        # Generate variations using rule-based
        if self.rule_based:
            analysis = self.rule_based.analyze_image(image)
            
            for style in [CaptionStyle.CONCISE, CaptionStyle.DETAILED, CaptionStyle.CREATIVE]:
                if len(alternatives) < num:
                    alt = self.rule_based.generate_caption(analysis, style)
                    if alt != base_caption:
                        alternatives.append(alt)
        
        # Add paraphrased versions
        if len(alternatives) < num:
            paraphrases = self._paraphrase_caption(base_caption, num - len(alternatives))
            alternatives.extend(paraphrases)
        
        return alternatives[:num]
    
    def _paraphrase_caption(self, caption: str, num: int) -> List[str]:
        """Simple paraphrasing (could be enhanced with a paraphraser model)"""
        # Simple word replacements
        replacements = {
            "bright": ["well-lit", "sunny", "luminous"],
            "dark": ["dim", "shadowy", "gloomy"],
            "beautiful": ["stunning", "gorgeous", "lovely"],
            "scene": ["view", "setting", "landscape"],
            "image": ["picture", "shot", "frame"]
        }
        
        paraphrases = set()
        
        for word, replacements_list in replacements.items():
            if word in caption:
                for rep in replacements_list[:num]:
                    paraphrases.add(caption.replace(word, rep))
        
        return list(paraphrases)[:num]
    
    def _calculate_rule_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score for rule-based caption"""
        if not analysis.get("valid", False):
            return 0.3
        
        confidence = 0.7
        confidence += min(0.1, analysis.get("edge_density", 0) * 0.5)
        confidence += min(0.1, analysis.get("brightness", 0) / 2550)
        
        return min(0.95, confidence)
    
    # ------------------------
    # CACHE MANAGEMENT
    # ------------------------
    def _update_cache(self, key: str, caption: ImageCaption):
        """Update cache with LRU eviction"""
        if len(self.caption_cache) >= self.max_cache:
            # Remove oldest entry
            oldest = min(self.caption_cache.items(), key=lambda x: x[1].timestamp)[0]
            del self.caption_cache[oldest]
        
        self.caption_cache[key] = caption
    
    def clear_cache(self):
        """Clear caption cache"""
        self.caption_cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.caption_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
    # ------------------------
    # BATCH PROCESSING
    # ------------------------
    def caption_batch(
        self,
        images: List[np.ndarray],
        show_progress: bool = True
    ) -> List[ImageCaption]:
        """Generate captions for batch of images"""
        captions = []
        
        for idx, img in enumerate(images):
            if img is not None:
                cap = self.caption(img)
                captions.append(cap)
                
                if show_progress and (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(images)} images")
        
        return captions
    
    async def caption_batch_async(
        self,
        images: List[np.ndarray],
        max_concurrent: int = 4
    ) -> List[ImageCaption]:
        """Asynchronously process batch of images"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(img):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self.caption, img
                )
        
        tasks = [process_one(img) for img in images if img is not None]
        return await asyncio.gather(*tasks)
    
    # ------------------------
    # ENHANCED CAPTIONS WITH OBJECTS
    # ------------------------
    def caption_with_objects(
        self,
        image: np.ndarray,
        objects: List[str],
        object_confidences: Optional[List[float]] = None
    ) -> ImageCaption:
        """Generate caption incorporating detected objects"""
        base_caption = self.caption(image)
        
        if not objects:
            return base_caption
        
        # Format objects list
        if len(objects) == 1:
            objects_str = f"a {objects[0]}"
        elif len(objects) <= 3:
            objects_str = ", ".join(objects[:-1]) + f" and a {objects[-1]}"
        else:
            objects_str = ", ".join(objects[:3]) + f" and {len(objects) - 3} more objects"
        
        # Integrate objects into caption
        templates = [
            f"{base_caption.text} Contains {objects_str}.",
            f"In this scene, {objects_str} can be seen. {base_caption.text}",
            f"Featuring {objects_str}, {base_caption.text.lower()}"
        ]
        
        # Choose best template based on caption length
        template = templates[0] if len(base_caption.text) < 50 else templates[1]
        enhanced_text = template
        
        # Adjust confidence
        confidence = min(0.95, base_caption.confidence + len(objects) * 0.02)
        
        return ImageCaption(
            text=enhanced_text,
            confidence=confidence,
            objects_detected=objects,
            scene_attributes=base_caption.scene_attributes
        )
    
    # ------------------------
    # VISUALIZATION
    # ------------------------
    def visualize_attention(
        self,
        image: np.ndarray,
        caption: ImageCaption,
        words: Optional[List[str]] = None
    ) -> np.ndarray:
        """Visualize attention weights on image"""
        if not self.attention_viz or caption.attention_weights is None:
            return image
        
        if words is None:
            words = caption.text.split()[:10]  # Top 10 words
        
        return self.attention_viz.create_word_attention_grid(
            image,
            {word: caption.attention_weights[i] for i, word in enumerate(words) if i < len(caption.attention_weights)},
            top_k=min(5, len(words))
        )
    
    def draw_caption(
        self,
        image: np.ndarray,
        caption: ImageCaption,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 0.7,
        thickness: int = 2,
        background: bool = True
    ) -> np.ndarray:
        """Draw caption on image with styling"""
        if cv2 is None or image is None:
            return image
        
        img_copy = image.copy()
        text = caption.text
        
        # Wrap text if too long
        max_width = image.shape[1] - position[0] - 10
        chars_per_line = int(max_width / (font_scale * 10))
        
        if len(text) > chars_per_line:
            # Simple text wrapping
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                if len(' '.join(current_line + [word])) <= chars_per_line:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
        else:
            lines = [text]
        
        # Draw background and text
        y_offset = position[1]
        for line in lines:
            if background:
                (text_w, text_h), baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(
                    img_copy,
                    (position[0] - 5, y_offset - text_h - 5),
                    (position[0] + text_w + 5, y_offset + 5),
                    (0, 0, 0),
                    -1
                )
            
            cv2.putText(
                img_copy,
                line,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness
            )
            
            y_offset += int(font_scale * 30)
        
        return img_copy
    
    # ------------------------
    # CAPTION RANKING & SIMILARITY
    # ------------------------
    def rank_captions(
        self,
        image: np.ndarray,
        candidate_captions: List[str]
    ) -> List[Tuple[str, float]]:
        """Rank multiple candidate captions for an image"""
        if not self.semantic_model:
            return [(cap, 0.5) for cap in candidate_captions]
        
        # Generate reference caption
        reference = self.caption(image)
        
        # Encode all captions
        all_captions = [reference.text] + candidate_captions
        embeddings = self.semantic_model.encode(all_captions)
        
        # Calculate similarities
        similarities = []
        for i in range(1, len(embeddings)):
            sim = 1 - cosine(embeddings[0], embeddings[i])
            similarities.append((candidate_captions[i-1], float(sim)))
        
        # Sort by similarity
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def find_similar_images(
        self,
        query_caption: str,
        captions_db: List[ImageCaption],
        top_k: int = 5
    ) -> List[Tuple[ImageCaption, float]]:
        """Find images with similar captions using semantic search"""
        if not self.semantic_model:
            return [(cap, 0.5) for cap in captions_db[:top_k]]
        
        query_embedding = self.semantic_model.encode([query_caption])[0]
        
        similarities = []
        for cap in captions_db:
            if cap.embedding is not None:
                sim = 1 - cosine(query_embedding, cap.embedding)
                similarities.append((cap, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # ------------------------
    # STATISTICS & EXPORT
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        avg_time = np.mean(self.generation_times) if self.generation_times else 0
        
        return {
            "captions_generated": self.captions_generated,
            "cache_stats": self.get_cache_stats(),
            "avg_generation_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "model": self.config.model_type.value,
            "style": self.config.style.value,
            "device": self.config.device,
            "deep_learning_enabled": self.deep_captioner is not None and self.deep_captioner.model is not None,
            "semantic_search_enabled": self.semantic_model is not None,
            "config": {
                "max_length": self.config.max_length,
                "num_beams": self.config.num_beams,
                "temperature": self.config.temperature
            }
        }
    
    def export_caption_history(self, filepath: str, captions: List[ImageCaption]):
        """Export captions to JSON file"""
        data = {
            "timestamp": time.time(),
            "total_captions": len(captions),
            "captions": [c.to_dict() for c in captions],
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📁 Exported {len(captions)} captions to {filepath}")
    
    def reset_stats(self):
        """Reset statistics"""
        self.captions_generated = 0
        self.generation_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Statistics reset")


# ------------------------
# USAGE EXAMPLE
# ------------------------
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure captioner
    config = CaptionConfig(
        model_type=CaptionModelType.BLIP,  # Use BLIP for deep learning
        style=CaptionStyle.DESCRIPTIVE,
        max_length=50,
        num_beams=3,
        temperature=0.7,
        use_gpu=True
    )
    
    # Initialize captioner
    captioner = Captioner(config)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Generate caption
        start = time.time()
        caption = captioner.caption(frame, return_alternatives=True)
        inference_time = (time.time() - start) * 1000
        
        # Draw on frame
        annotated = captioner.draw_caption(frame, caption)
        
        # Show stats
        cv2.putText(
            annotated,
            f"Time: {inference_time:.1f}ms | Conf: {caption.confidence:.2f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        cv2.imshow('AI Captioning', annotated)
        
        # Print caption
        print(f"📝 {caption}")
        if caption.alternative_captions:
            print(f"   Alternatives: {caption.alternative_captions[:2]}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    print("\n" + "="*50)
    print("FINAL STATISTICS:")
    print(json.dumps(captioner.get_stats(), indent=2))