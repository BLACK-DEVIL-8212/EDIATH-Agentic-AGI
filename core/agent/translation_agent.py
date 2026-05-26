"""
Translation Agent for EDIATH
Advanced language translation: text translation, language detection, batch processing, context-aware translation
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import logging

# Translation libraries
try:
    from googletrans import Translator as GoogleTranslator

    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    from deep_translator import MicrosoftTranslator, PonsTranslator, LingueeTranslator

    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False

try:
    import translators as ts

    TRANSLATORS_AVAILABLE = True
except ImportError:
    TRANSLATORS_AVAILABLE = False

try:
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    from transformers import MarianMTModel, MarianTokenizer
    from transformers import NllbTokenizer, NllbForConditionalGeneration

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from langdetect import detect, detect_langs

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class TranslationProvider(Enum):
    """Translation service providers"""

    GOOGLE = "google"
    MICROSOFT = "microsoft"
    DEEPL = "deepl"
    PONS = "pons"
    LINGUEE = "linguee"
    M2M100 = "m2m100"
    MARIAN = "marian"
    NLLB = "nllb"
    AUTO = "auto"


class TranslationQuality(Enum):
    """Translation quality levels"""

    FAST = "fast"  # Speed optimized
    BALANCED = "balanced"  # Balanced speed/quality
    HIGH = "high"  # Quality optimized


@dataclass
class Language:
    """Language information"""

    code: str
    name: str
    native_name: str
    direction: str = "ltr"


@dataclass
class TranslationResult:
    """Translation result container"""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    provider: str
    processing_time: float
    alternatives: List[str] = field(default_factory=list)


class TranslationAgent:
    """
    Advanced translation agent capable of:
    - Text translation between 100+ languages
    - Automatic language detection
    - Batch translation
    - Document translation (text files)
    - Context-aware translation
    - Transliteration
    - Dictionary lookup
    - Translation memory
    - Quality estimation
    - Multi-provider support with fallback
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Translation Agent

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # API keys
        self.google_api_key = self.config.get("google_api_key", "")
        self.microsoft_key = self.config.get("microsoft_key", "")
        self.deepl_key = self.config.get("deepl_key", "")

        # Translation providers
        self.default_provider = TranslationProvider(
            self.config.get("default_provider", "auto")
        )
        self.default_quality = TranslationQuality(
            self.config.get("default_quality", "balanced")
        )
        self.fallback_providers = self.config.get(
            "fallback_providers", ["google", "microsoft"]
        )

        # Cache settings
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour
        self.cache: Dict[str, Tuple[datetime, TranslationResult]] = {}

        # Model instances
        self.google_translator = None
        self.m2m100_model = None
        self.m2m100_tokenizer = None
        self.marian_models = {}
        self.nllb_model = None

        # Language data
        self.supported_languages = self._load_language_data()
        self.common_pairs = self.config.get(
            "common_pairs",
            [
                ("en", "es"),
                ("en", "fr"),
                ("en", "de"),
                ("en", "zh-cn"),
                ("es", "en"),
                ("fr", "en"),
                ("de", "en"),
                ("zh-cn", "en"),
            ],
        )

        # Translation memory
        self.translation_memory: Dict[str, str] = {}
        self.tm_file = Path(self.config.get("tm_file", "translation_memory.json"))
        self._load_translation_memory()

        # Statistics
        self.stats = {
            "total_translations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_confidence": 0.0,
            "total_processing_time": 0.0,
            "errors": 0,
            "by_provider": {},
        }

        # History
        self.translation_history: List[TranslationResult] = []
        self.max_history = self.config.get("max_history", 500)

        self.logger.info(
            f"Translation Agent initialized. Provider: {self.default_provider.value}"
        )

        # Initialize available providers
        self._init_providers()

    def _init_providers(self):
        """Initialize translation providers"""
        # Initialize Google Translator
        if GOOGLETRANS_AVAILABLE:
            try:
                self.google_translator = GoogleTranslator()
                self.logger.info("Google Translator initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google Translator: {e}")

        # Initialize M2M100 model if configured
        if self.config.get("load_m2m100", False) and TRANSFORMERS_AVAILABLE:
            self._load_m2m100()

        # Initialize NLLB model if configured
        if self.config.get("load_nllb", False) and TRANSFORMERS_AVAILABLE:
            self._load_nllb()

    def _load_m2m100(self):
        """Load M2M100 model for high-quality translation"""
        try:
            model_name = self.config.get("m2m100_model", "facebook/m2m100_418M")
            self.logger.info(f"Loading M2M100 model: {model_name}")
            self.m2m100_tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.m2m100_model = M2M100ForConditionalGeneration.from_pretrained(
                model_name
            )

            if self.config.get("use_gpu", False):
                self.m2m100_model = self.m2m100_model.to("cuda")

            self.logger.info("M2M100 model loaded")
        except Exception as e:
            self.logger.warning(f"Failed to load M2M100: {e}")

    def _load_nllb(self):
        """Load NLLB model for translation"""
        try:
            model_name = self.config.get(
                "nllb_model", "facebook/nllb-200-distilled-600M"
            )
            self.logger.info(f"Loading NLLB model: {model_name}")
            self.nllb_tokenizer = NllbTokenizer.from_pretrained(model_name)
            self.nllb_model = NllbForConditionalGeneration.from_pretrained(model_name)

            if self.config.get("use_gpu", False):
                self.nllb_model = self.nllb_model.to("cuda")

            self.logger.info("NLLB model loaded")
        except Exception as e:
            self.logger.warning(f"Failed to load NLLB: {e}")

    def _load_language_data(self) -> Dict[str, Language]:
        """Load supported languages data"""
        languages = {
            "en": Language("en", "English", "English"),
            "es": Language("es", "Spanish", "Español"),
            "fr": Language("fr", "French", "Français"),
            "de": Language("de", "German", "Deutsch"),
            "it": Language("it", "Italian", "Italiano"),
            "pt": Language("pt", "Portuguese", "Português"),
            "ru": Language("ru", "Russian", "Русский"),
            "zh-cn": Language("zh-cn", "Chinese (Simplified)", "中文"),
            "zh-tw": Language("zh-tw", "Chinese (Traditional)", "中文"),
            "ja": Language("ja", "Japanese", "日本語"),
            "ko": Language("ko", "Korean", "한국어"),
            "ar": Language("ar", "Arabic", "العربية"),
            "hi": Language("hi", "Hindi", "हिन्दी"),
            "tr": Language("tr", "Turkish", "Türkçe"),
            "nl": Language("nl", "Dutch", "Nederlands"),
            "pl": Language("pl", "Polish", "Polski"),
            "sv": Language("sv", "Swedish", "Svenska"),
            "da": Language("da", "Danish", "Dansk"),
            "fi": Language("fi", "Finnish", "Suomi"),
            "no": Language("no", "Norwegian", "Norsk"),
            "he": Language("he", "Hebrew", "עברית"),
            "th": Language("th", "Thai", "ไทย"),
            "vi": Language("vi", "Vietnamese", "Tiếng Việt"),
            "id": Language("id", "Indonesian", "Bahasa Indonesia"),
            "ms": Language("ms", "Malay", "Bahasa Melayu"),
        }

        # Add more languages from config
        extra_langs = self.config.get("extra_languages", {})
        for code, data in extra_langs.items():
            languages[code] = Language(code, data["name"], data["native_name"])

        return languages

    def _load_translation_memory(self):
        """Load translation memory from file"""
        if self.tm_file.exists():
            try:
                with open(self.tm_file, "r", encoding="utf-8") as f:
                    self.translation_memory = json.load(f)
                self.logger.info(
                    f"Loaded {len(self.translation_memory)} entries from translation memory"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load translation memory: {e}")

    def _save_translation_memory(self):
        """Save translation memory to file"""
        try:
            with open(self.tm_file, "w", encoding="utf-8") as f:
                json.dump(self.translation_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save translation memory: {e}")

    def _get_cache_key(
        self, text: str, source_lang: str, target_lang: str, provider: str
    ) -> str:
        """Generate cache key for translation"""
        content = f"{text}:{source_lang}:{target_lang}:{provider}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[TranslationResult]:
        """Get translation from cache"""
        if not self.cache_enabled:
            return None

        if cache_key in self.cache:
            timestamp, result = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return result
            else:
                del self.cache[cache_key]

        self.stats["cache_misses"] += 1
        return None

    def _add_to_cache(self, cache_key: str, result: TranslationResult):
        """Add translation to cache"""
        if self.cache_enabled:
            # Manage cache size
            if len(self.cache) > 1000:
                # Remove oldest 10%
                items = sorted(self.cache.items(), key=lambda x: x[1][0])
                for key, _ in items[:100]:
                    del self.cache[key]

            self.cache[cache_key] = (datetime.now(), result)

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text

        Args:
            text: Text to detect language

        Returns:
            Dictionary with detected language(s)
        """
        try:
            if LANGDETECT_AVAILABLE:
                # Get language detection with confidence
                detections = detect_langs(text)

                results = []
                for detection in detections:
                    lang_code = detection.lang
                    lang_info = self.supported_languages.get(lang_code)

                    results.append(
                        {
                            "language_code": lang_code,
                            "language_name": lang_info.name if lang_info else lang_code,
                            "confidence": detection.prob,
                        }
                    )

                primary = results[0] if results else None

                return {
                    "success": True,
                    "detected_languages": results,
                    "primary_language": primary,
                    "text_preview": text[:100],
                }
            else:
                # Fallback to Google translate detection
                if self.google_translator:
                    detection = self.google_translator.detect(text)
                    return {
                        "success": True,
                        "primary_language": {
                            "language_code": detection.lang,
                            "language_name": self.supported_languages.get(
                                detection.lang,
                                Language(
                                    detection.lang, detection.lang, detection.lang
                                ),
                            ).name,
                            "confidence": detection.confidence,
                        },
                    }
                else:
                    return {
                        "success": False,
                        "error": "No language detection available",
                    }

        except Exception as e:
            self.logger.error(f"Language detection error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        provider: Optional[TranslationProvider] = None,
        quality: Optional[TranslationQuality] = None,
        use_cache: bool = True,
        use_tm: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Translate text to target language

        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Source language code (auto-detect if None)
            provider: Translation provider to use
            quality: Translation quality level
            use_cache: Use cached translation
            use_tm: Use translation memory
            **kwargs: Provider-specific arguments

        Returns:
            Dictionary with translation result
        """
        start_time = datetime.now()

        # Determine provider
        if provider is None:
            provider = self.default_provider
            if provider == TranslationProvider.AUTO:
                provider = self._select_best_provider(text, target_lang)

        # Determine quality
        if quality is None:
            quality = self.default_quality

        # Detect source language if not provided
        if source_lang is None:
            detection = await self.detect_language(text)
            if detection["success"]:
                source_lang = detection["primary_language"]["language_code"]
            else:
                source_lang = "en"  # Default fallback

        # Check translation memory
        if use_tm:
            tm_key = f"{source_lang}:{target_lang}:{text}"
            if tm_key in self.translation_memory:
                self.logger.debug("Using translation memory")
                result = TranslationResult(
                    source_text=text,
                    translated_text=self.translation_memory[tm_key],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    confidence=0.9,
                    provider="memory",
                    processing_time=0.0,
                )

                return self._format_result(result, start_time)

        # Check cache
        cache_key = self._get_cache_key(text, source_lang, target_lang, provider.value)
        if use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return self._format_result(cached_result, start_time)

        # Perform translation
        try:
            translated_text, confidence = await self._translate_with_provider(
                text, source_lang, target_lang, provider, quality, **kwargs
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = TranslationResult(
                source_text=text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=confidence,
                provider=provider.value,
                processing_time=processing_time,
            )

            # Update statistics
            self.stats["total_translations"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_confidence"] = (
                self.stats["average_confidence"]
                * (self.stats["total_translations"] - 1)
                + confidence
            ) / self.stats["total_translations"]

            # Update provider stats
            if provider.value not in self.stats["by_provider"]:
                self.stats["by_provider"][provider.value] = 0
            self.stats["by_provider"][provider.value] += 1

            # Add to cache
            self._add_to_cache(cache_key, result)

            # Add to history
            self._add_to_history(result)

            return self._format_result(result, start_time)

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            self.stats["errors"] += 1

            # Try fallback providers
            if provider != TranslationProvider.AUTO:
                for fallback in self.fallback_providers:
                    if fallback != provider.value:
                        self.logger.info(f"Trying fallback provider: {fallback}")
                        try:
                            return await self.translate(
                                text,
                                target_lang,
                                source_lang,
                                TranslationProvider(fallback),
                                quality,
                                use_cache,
                                use_tm,
                                **kwargs,
                            )
                        except:
                            continue

            return {
                "success": False,
                "error": str(e),
                "source_text": text,
                "target_lang": target_lang,
                "source_lang": source_lang,
            }

    async def _translate_with_provider(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: TranslationProvider,
        quality: TranslationQuality,
        **kwargs,
    ) -> Tuple[str, float]:
        """Translate using specific provider"""

        if provider == TranslationProvider.GOOGLE:
            return await self._translate_google(
                text, source_lang, target_lang, **kwargs
            )

        elif provider == TranslationProvider.MICROSOFT:
            return await self._translate_microsoft(
                text, source_lang, target_lang, **kwargs
            )

        elif provider == TranslationProvider.DEEPL:
            return await self._translate_deepl(text, source_lang, target_lang, **kwargs)

        elif provider == TranslationProvider.M2M100:
            return await self._translate_m2m100(
                text, source_lang, target_lang, **kwargs
            )

        elif provider == TranslationProvider.MARIAN:
            return await self._translate_marian(
                text, source_lang, target_lang, **kwargs
            )

        elif provider == TranslationProvider.NLLB:
            return await self._translate_nllb(text, source_lang, target_lang, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _translate_google(
        self, text: str, source_lang: str, target_lang: str, **kwargs
    ) -> Tuple[str, float]:
        """Translate using Google Translate"""
        if not self.google_translator:
            if GOOGLETRANS_AVAILABLE:
                self.google_translator = GoogleTranslator()
            else:
                raise ImportError("Google Translate not available")

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        # Translate
        result = await loop.run_in_executor(
            None,
            lambda: self.google_translator.translate(
                text, src=source_lang, dest=target_lang
            ),
        )

        # Extract confidence (Google doesn't provide, so estimate)
        confidence = 0.85 if len(text) > 10 else 0.75

        return result.text, confidence

    async def _translate_microsoft(
        self, text: str, source_lang: str, target_lang: str, **kwargs
    ) -> Tuple[str, float]:
        """Translate using Microsoft Translator"""
        if not DEEP_TRANSLATOR_AVAILABLE:
            raise ImportError("Deep Translator not available")

        translator = MicrosoftTranslator(
            api_key=self.microsoft_key, source=source_lang, target=target_lang
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, translator.translate, text)

        return result, 0.85

    async def _translate_deepl(
        self, text: str, source_lang: str, target_lang: str, **kwargs
    ) -> Tuple[str, float]:
        """Translate using DeepL"""
        if not self.deepl_key:
            raise ValueError("DeepL API key required")

        try:
            import deepl

            translator = deepl.Translator(self.deepl_key)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: translator.translate_text(
                    text,
                    source_lang=source_lang.upper(),
                    target_lang=target_lang.upper(),
                ),
            )

            return result.text, 0.9

        except ImportError:
            raise ImportError("DeepL library not available")

    async def _translate_m2m100(
        self, text: str, source_lang: str, target_lang: str, **kwargs
    ) -> Tuple[str, float]:
        """Translate using M2M100 model"""
        if not self.m2m100_model:
            self._load_m2m100()
            if not self.m2m100_model:
                raise RuntimeError("M2M100 model not available")

        # Map language codes
        src_lang_map = {
            "zh-cn": "zh",
            "zh-tw": "zh",
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
        }

        src = src_lang_map.get(source_lang, source_lang)
        tgt = src_lang_map.get(target_lang, target_lang)

        # Tokenize
        self.m2m100_tokenizer.src_lang = src
        encoded = self.m2m100_tokenizer(text, return_tensors="pt")

        if self.config.get("use_gpu", False):
            encoded = {k: v.to("cuda") for k, v in encoded.items()}

        # Generate translation
        generated_tokens = self.m2m100_model.generate(
            **encoded, forced_bos_token_id=self.m2m100_tokenizer.get_lang_id(tgt)
        )

        translated = self.m2m100_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        return translated, 0.88

    async def _translate_marian(
        self, text: str, source_lang: str, target_lang: str, **kwargs
    ) -> Tuple[str, float]:
        """Translate using MarianMT model"""
        model_key = f"{source_lang}-{target_lang}"

        if model_key not in self.marian_models:
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

            try:
                self.marian_models[model_key] = {
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "model": MarianMTModel.from_pretrained(model_name),
                }

                if self.config.get("use_gpu", False):
                    self.marian_models[model_key]["model"] = self.marian_models[
                        model_key
                    ]["model"].to("cuda")

            except Exception as e:
                raise Exception(f"Failed to load Marian model for {model_key}: {e}")

        model_data = self.marian_models[model_key]

        # Tokenize and translate
        encoded = model_data["tokenizer"](text, return_tensors="pt", padding=True)

        if self.config.get("use_gpu", False):
            encoded = {k: v.to("cuda") for k, v in encoded.items()}

        generated = model_data["model"].generate(**encoded)
        translated = model_data["tokenizer"].batch_decode(
            generated, skip_special_tokens=True
        )[0]

        return translated, 0.87

    async def _translate_nllb(
        self, text: str, source_lang: str, target_lang: str, **kwargs
    ) -> Tuple[str, float]:
        """Translate using NLLB model"""
        if not self.nllb_model:
            self._load_nllb()
            if not self.nllb_model:
                raise RuntimeError("NLLB model not available")

        # Convert language codes to NLLB format
        nllb_lang_map = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "zh-cn": "zho_Hans",
            "zh-tw": "zho_Hant",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ru": "rus_Cyrl",
            "ar": "arb_Arab",
        }

        src = nllb_lang_map.get(source_lang, source_lang)
        tgt = nllb_lang_map.get(target_lang, target_lang)

        # Tokenize
        inputs = self.nllb_tokenizer(text, return_tensors="pt")

        if self.config.get("use_gpu", False):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate translation
        generated_tokens = self.nllb_model.generate(
            **inputs, forced_bos_token_id=self.nllb_tokenizer.lang_code_to_id[tgt]
        )

        translated = self.nllb_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        return translated, 0.89

    def _select_best_provider(self, text: str, target_lang: str) -> TranslationProvider:
        """Select best provider based on text and target language"""
        # Simple heuristic: use Google for short texts, M2M100 for long texts
        if len(text) < 100:
            return TranslationProvider.GOOGLE
        elif self.m2m100_model and len(text) > 500:
            return TranslationProvider.M2M100
        else:
            return TranslationProvider.GOOGLE

    async def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
        provider: Optional[TranslationProvider] = None,
        max_concurrent: int = 5,
    ) -> Dict[str, Any]:
        """
        Translate multiple texts in batch

        Args:
            texts: List of texts to translate
            target_lang: Target language
            source_lang: Source language (auto-detect if None)
            provider: Translation provider
            max_concurrent: Maximum concurrent translations

        Returns:
            Dictionary with batch results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def translate_with_semaphore(text):
            async with semaphore:
                return await self.translate(text, target_lang, source_lang, provider)

        tasks = [translate_with_semaphore(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = []
        failed = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append(
                    {"index": i, "text": texts[i][:100], "error": str(result)}
                )
            elif result.get("success", False):
                successful.append(result)
            else:
                failed.append(
                    {
                        "index": i,
                        "text": texts[i][:100],
                        "error": result.get("error", "Unknown error"),
                    }
                )

        return {
            "success": len(successful) > 0,
            "total": len(texts),
            "successful": len(successful),
            "failed": len(failed),
            "results": successful,
            "errors": failed,
        }

    async def translate_file(
        self,
        file_path: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        output_file: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Translate text file content

        Args:
            file_path: Path to text file
            target_lang: Target language
            source_lang: Source language
            output_file: Output file path (optional)
            **kwargs: Additional translation arguments

        Returns:
            Dictionary with translation result
        """
        try:
            # Read file
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split into paragraphs
            paragraphs = content.split("\n\n")

            # Translate each paragraph
            translated_paragraphs = []
            for para in paragraphs:
                if para.strip():
                    result = await self.translate(
                        para, target_lang, source_lang, **kwargs
                    )
                    if result.get("success"):
                        translated_paragraphs.append(result["translated_text"])
                    else:
                        translated_paragraphs.append(para)  # Keep original on error
                else:
                    translated_paragraphs.append("")

            translated_content = "\n\n".join(translated_paragraphs)

            # Save to output file
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translated_content)

            return {
                "success": True,
                "translated_content": translated_content,
                "output_file": str(output_file) if output_file else None,
                "paragraphs": len(paragraphs),
                "source_lang": source_lang,
                "target_lang": target_lang,
            }

        except Exception as e:
            self.logger.error(f"File translation error: {str(e)}")
            return {"success": False, "error": str(e), "file_path": file_path}

    async def get_alternatives(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None,
        num_alternatives: int = 3,
    ) -> Dict[str, Any]:
        """
        Get alternative translations

        Args:
            text: Text to translate
            target_lang: Target language
            source_lang: Source language
            num_alternatives: Number of alternatives

        Returns:
            Dictionary with alternative translations
        """
        alternatives = []

        # Use multiple providers to get alternatives
        providers = [TranslationProvider.GOOGLE, TranslationProvider.MICROSOFT]

        for provider in providers:
            try:
                result = await self.translate(
                    text, target_lang, source_lang, provider, use_cache=False
                )
                if result.get("success"):
                    alternatives.append(
                        {
                            "provider": provider.value,
                            "text": result["translated_text"],
                            "confidence": result["confidence"],
                        }
                    )
            except:
                continue

        return {
            "success": len(alternatives) > 0,
            "original_text": text,
            "target_lang": target_lang,
            "alternatives": alternatives[:num_alternatives],
        }

    async def transliterate(
        self, text: str, source_lang: str, target_script: str = "latn"
    ) -> Dict[str, Any]:
        """
        Transliterate text between scripts

        Args:
            text: Text to transliterate
            source_lang: Source language
            target_script: Target script (e.g., 'latn' for Latin)

        Returns:
            Dictionary with transliteration result
        """
        try:
            # Use Google Translate for transliteration
            if self.google_translator:
                # Transliterate by translating to same language with Latin script
                result = await self.translate(
                    text, source_lang, source_lang, TranslationProvider.GOOGLE
                )

                if result.get("success"):
                    return {
                        "success": True,
                        "original": text,
                        "transliterated": result["translated_text"],
                        "source_lang": source_lang,
                        "target_script": target_script,
                    }

            raise Exception("Transliteration failed")

        except Exception as e:
            self.logger.error(f"Transliteration error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def add_to_memory(
        self, source_text: str, target_text: str, source_lang: str, target_lang: str
    ):
        """
        Add translation pair to memory

        Args:
            source_text: Source text
            target_text: Translated text
            source_lang: Source language
            target_lang: Target language
        """
        tm_key = f"{source_lang}:{target_lang}:{source_text}"
        self.translation_memory[tm_key] = target_text
        self._save_translation_memory()

        return {"success": True, "message": "Added to translation memory"}

    def _format_result(
        self, result: TranslationResult, start_time: datetime
    ) -> Dict[str, Any]:
        """Format translation result for output"""
        return {
            "success": True,
            "source_text": result.source_text,
            "translated_text": result.translated_text,
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "confidence": result.confidence,
            "provider": result.provider,
            "processing_time": result.processing_time,
            "total_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
        }

    def _add_to_history(self, result: TranslationResult):
        """Add translation to history"""
        self.translation_history.append(result)
        if len(self.translation_history) > self.max_history:
            self.translation_history = self.translation_history[-self.max_history :]

    def get_history(self, limit: int = None) -> List[Dict]:
        """Get translation history"""
        history = self.translation_history
        if limit:
            history = history[-limit:]

        return [
            {
                "source_text": h.source_text[:100],
                "translated_text": h.translated_text[:100],
                "source_lang": h.source_lang,
                "target_lang": h.target_lang,
                "confidence": h.confidence,
                "provider": h.provider,
                "timestamp": h.processing_time,
            }
            for h in history
        ]

    def get_supported_languages(self) -> List[Dict]:
        """Get list of supported languages"""
        return [
            {"code": lang.code, "name": lang.name, "native_name": lang.native_name}
            for lang in self.supported_languages.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        avg_time = (
            (self.stats["total_processing_time"] / self.stats["total_translations"])
            if self.stats["total_translations"] > 0
            else 0
        )

        return {
            **self.stats,
            "average_processing_time": avg_time,
            "history_size": len(self.translation_history),
            "cache_size": len(self.cache),
            "memory_size": len(self.translation_memory),
            "default_provider": self.default_provider.value,
            "supported_languages": len(self.supported_languages),
        }

    def clear_cache(self):
        """Clear translation cache"""
        self.cache.clear()
        self.logger.info("Translation cache cleared")

    def clear_history(self):
        """Clear translation history"""
        self.translation_history.clear()
        self.logger.info("Translation history cleared")


# Integration wrapper for EDIATH
class TranslationAgentWrapper:
    """
    Wrapper class to integrate TranslationAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.translation_agent = TranslationAgent(config)
        self.agent_type = "translation"
        self.capabilities = [
            "text_translation",
            "language_detection",
            "batch_translation",
            "file_translation",
            "transliteration",
            "translation_memory",
            "alternative_translations",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a translation request

        Request format:
        {
            'operation': 'translate|detect|batch|file|alternatives|transliterate|memory',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "translate":
            provider = request.get("provider")
            if provider:
                provider = TranslationProvider(provider)

            quality = request.get("quality")
            if quality:
                quality = TranslationQuality(quality)

            return await self.translation_agent.translate(
                text=request.get("text"),
                target_lang=request.get("target_lang"),
                source_lang=request.get("source_lang"),
                provider=provider,
                quality=quality,
                use_cache=request.get("use_cache", True),
                use_tm=request.get("use_tm", True),
            )

        elif operation == "detect":
            return await self.translation_agent.detect_language(
                text=request.get("text")
            )

        elif operation == "batch":
            provider = request.get("provider")
            if provider:
                provider = TranslationProvider(provider)

            return await self.translation_agent.translate_batch(
                texts=request.get("texts", []),
                target_lang=request.get("target_lang"),
                source_lang=request.get("source_lang"),
                provider=provider,
                max_concurrent=request.get("max_concurrent", 5),
            )

        elif operation == "file":
            provider = request.get("provider")
            if provider:
                provider = TranslationProvider(provider)

            return await self.translation_agent.translate_file(
                file_path=request.get("file_path"),
                target_lang=request.get("target_lang"),
                source_lang=request.get("source_lang"),
                output_file=request.get("output_file"),
                provider=provider,
            )

        elif operation == "alternatives":
            return await self.translation_agent.get_alternatives(
                text=request.get("text"),
                target_lang=request.get("target_lang"),
                source_lang=request.get("source_lang"),
                num_alternatives=request.get("num_alternatives", 3),
            )

        elif operation == "transliterate":
            return await self.translation_agent.transliterate(
                text=request.get("text"),
                source_lang=request.get("source_lang"),
                target_script=request.get("target_script", "latn"),
            )

        elif operation == "memory":
            action = request.get("action", "add")
            if action == "add":
                return await self.translation_agent.add_to_memory(
                    source_text=request.get("source_text"),
                    target_text=request.get("target_text"),
                    source_lang=request.get("source_lang"),
                    target_lang=request.get("target_lang"),
                )
            else:
                return {"success": False, "error": f"Unknown memory action: {action}"}

        elif operation == "languages":
            return {
                "success": True,
                "languages": self.translation_agent.get_supported_languages(),
            }

        elif operation == "history":
            return {
                "success": True,
                "history": self.translation_agent.get_history(
                    limit=request.get("limit")
                ),
            }

        elif operation == "stats":
            return self.translation_agent.get_stats()

        elif operation == "clear_cache":
            self.translation_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        elif operation == "clear_history":
            self.translation_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "TranslationAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.translation_agent.get_stats(),
            "default_provider": self.translation_agent.default_provider.value,
            "supported_languages": len(self.translation_agent.supported_languages),
        }


# Example usage and testing
async def test_translation_agent():
    """Test the translation agent functionality"""

    # Initialize agent
    agent = TranslationAgent()

    print("=== Translation Agent Test ===\n")

    # Test language detection
    print("1. Language Detection...")
    result = await agent.detect_language("Hello, how are you today?")
    if result["success"]:
        print(f"   Detected: {result['primary_language']['language_name']}")
        print(f"   Confidence: {result['primary_language']['confidence']:.2f}")

    # Test translation
    print("\n2. Text Translation (English to Spanish)...")
    result = await agent.translate(
        text="Hello, how are you?", target_lang="es", source_lang="en"
    )
    if result["success"]:
        print(f"   Source: {result['source_text']}")
        print(f"   Translation: {result['translated_text']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Provider: {result['provider']}")

    # Test batch translation
    print("\n3. Batch Translation...")
    texts = ["Good morning!", "What is your name?", "Nice to meet you."]
    result = await agent.translate_batch(texts, target_lang="fr")
    if result["success"]:
        print(f"   Successful: {result['successful']}/{result['total']}")
        for r in result["results"][:2]:
            print(f"     {r['source_text']} -> {r['translated_text']}")

    # Test alternatives
    print("\n4. Alternative Translations...")
    result = await agent.get_alternatives(
        text="Hello world", target_lang="es", num_alternatives=2
    )
    if result["success"]:
        print(f"   Alternatives for '{result['original_text']}':")
        for alt in result["alternatives"]:
            print(f"     - {alt['text']} ({alt['provider']})")

    # Get supported languages
    print("\n5. Supported Languages...")
    languages = agent.get_supported_languages()
    print(f"   Total supported: {len(languages)} languages")
    print(f"   Examples: {', '.join([l['name'] for l in languages[:10]])}")

    # Get statistics
    print("\n6. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total translations: {stats['total_translations']}")
    print(f"   Average confidence: {stats['average_confidence']:.2f}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Memory size: {stats['memory_size']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_translation_agent())
