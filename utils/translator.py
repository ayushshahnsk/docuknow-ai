"""
Translation Utilities for DocuKnow AI - MULTILINGUAL EDITION

Purpose:
- Translate text between multiple languages
- Detect language of text
- Cache translations for performance
- Handle translation errors gracefully
- Support for 100+ languages via Google Translate

Features:
1. Google Translate API integration (free)
2. Language detection with confidence scores
3. Translation caching for performance
4. Batch translation support
5. Fallback mechanisms
6. Rate limiting and error handling
"""

import logging
import time
import hashlib
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache
from collections import OrderedDict
import threading

# Try to import Google Translate
try:
    from googletrans import Translator, LANGUAGES
    from googletrans.models import Translated
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False
    print("Google Translate not available. Install: pip install googletrans==3.1.0a0")
    Translator = None
    LANGUAGES = {}

logger = logging.getLogger(__name__)

# Supported languages for DocuKnow AI (subset of all available)
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'fr': 'French',
    'bn': 'Bengali',
    'de': 'German',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'es': 'Spanish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'ar': 'Arabic',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'it': 'Italian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'fa': 'Persian',
    'he': 'Hebrew',
    'el': 'Greek',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
}

# Language code normalization
LANGUAGE_CODE_MAP = {
    'zh': 'zh-cn',
    'zh-hans': 'zh-cn',
    'zh-hant': 'zh-tw',
    'zh-cn': 'zh-cn',
    'zh-tw': 'zh-tw',
    'pt-br': 'pt',
    'pt-pt': 'pt',
    'es-es': 'es',
    'es-mx': 'es',
    'en-us': 'en',
    'en-gb': 'en',
    'fr-fr': 'fr',
    'fr-ca': 'fr',
    'de-de': 'de',
    'de-at': 'de',
    'de-ch': 'de',
}

class TranslationCache:
    """
    LRU Cache for translations to improve performance.
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[str]:
        """
        Get cached translation.
        
        Args:
            key: Cache key (hash of source text + target language)
            
        Returns:
            Cached translation or None
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            self.misses += 1
            return None
    
    def set(self, key: str, value: str):
        """
        Cache a translation.
        
        Args:
            key: Cache key
            value: Translated text
        """
        with self.lock:
            if key in self.cache:
                # Remove existing entry
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.2%}",
                'keys': list(self.cache.keys())[:10]  # First 10 keys
            }

class RateLimiter:
    """
    Rate limiter for translation API calls.
    """
    
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.lock = threading.RLock()
    
    def wait_if_needed(self):
        """
        Wait if rate limit would be exceeded.
        """
        with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.calls_per_minute:
                # Calculate wait time
                oldest_call = self.call_times[0]
                wait_time = 60 - (now - oldest_call)
                
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    
                    # Update call times after waiting
                    now = time.time()
                    self.call_times = [t for t in self.call_times if now - t < 60]
            
            self.call_times.append(now)
    
    def get_wait_time(self) -> float:
        """
        Calculate required wait time.
        
        Returns:
            Seconds to wait before next call
        """
        with self.lock:
            now = time.time()
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) < self.calls_per_minute:
                return 0.0
            
            oldest_call = self.call_times[0]
            return max(0.0, 60 - (now - oldest_call))

class DocuKnowTranslator:
    """
    Main translator class for DocuKnow AI.
    """
    
    def __init__(
        self, 
        cache_size: int = 1000,
        rate_limit_per_minute: int = 30,
        timeout: int = 10,
        retries: int = 3
    ):
        """
        Initialize translator.
        
        Args:
            cache_size: Maximum number of translations to cache
            rate_limit_per_minute: API call rate limit
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        if not GOOGLE_TRANSLATE_AVAILABLE:
            raise ImportError(
                "Google Translate is not available. "
                "Please install it using: pip install googletrans==3.1.0a0"
            )
        
        self.translator = Translator(timeout=timeout)
        self.cache = TranslationCache(max_size=cache_size)
        self.rate_limiter = RateLimiter(calls_per_minute=rate_limit_per_minute)
        self.timeout = timeout
        self.retries = retries
        self._supported_languages = self._get_supported_languages()
        
        logger.info(f"Translator initialized with {len(self._supported_languages)} supported languages")
    
    def _get_supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages from Google Translate.
        
        Returns:
            Dictionary of language_code -> language_name
        """
        if not GOOGLE_TRANSLATE_AVAILABLE:
            return SUPPORTED_LANGUAGES
        
        try:
            # Google Translate supports 100+ languages
            all_langs = LANGUAGES
            
            # Filter to our supported subset for better performance
            supported = {}
            for code, name in all_langs.items():
                # Map to our preferred codes
                norm_code = self.normalize_language_code(code)
                if norm_code in SUPPORTED_LANGUAGES:
                    supported[norm_code] = SUPPORTED_LANGUAGES[norm_code]
            
            return supported
        except:
            return SUPPORTED_LANGUAGES
    
    @staticmethod
    def normalize_language_code(lang_code: str) -> str:
        """
        Normalize language code to standard format.
        
        Args:
            lang_code: Input language code
            
        Returns:
            Normalized language code
        """
        if not lang_code:
            return 'en'
        
        lang_code = lang_code.lower().strip()
        
        # Apply code mapping
        if lang_code in LANGUAGE_CODE_MAP:
            return LANGUAGE_CODE_MAP[lang_code]
        
        # Common mappings
        if lang_code.startswith('zh'):
            if 'tw' in lang_code or 'hant' in lang_code:
                return 'zh-tw'
            return 'zh-cn'
        
        # Return first two characters (standard language code)
        return lang_code[:2] if len(lang_code) >= 2 else lang_code
    
    def _generate_cache_key(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """
        Generate cache key for translation.
        
        Args:
            text: Source text
            target_lang: Target language
            source_lang: Source language (optional)
            
        Returns:
            Cache key string
        """
        key_str = f"{text}|{target_lang}|{source_lang or 'auto'}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text with confidence.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detection results
        """
        if not text or len(text.strip()) < 3:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'text': text,
                'method': 'too_short'
            }
        
        try:
            self.rate_limiter.wait_if_needed()
            
            detected = self.translator.detect(text)
            
            result = {
                'language': self.normalize_language_code(detected.lang),
                'confidence': detected.confidence,
                'text': text[:100],  # Sample
                'method': 'googletrans',
                'raw_language': detected.lang,
                'raw_confidence': detected.confidence
            }
            
            logger.debug(f"Language detected: {result['language']} (confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            
            # Fallback: check for non-Latin characters
            if any(ord(c) > 127 for c in text):
                # Contains non-ASCII characters
                return {
                    'language': 'unknown',
                    'confidence': 0.1,
                    'text': text,
                    'method': 'fallback_non_latin',
                    'error': str(e)
                }
            
            # Default to English
            return {
                'language': 'en',
                'confidence': 0.3,
                'text': text,
                'method': 'fallback_english',
                'error': str(e)
            }
    
    def translate_text(
        self, 
        text: str, 
        target_lang: str, 
        source_lang: Optional[str] = None,
        use_cache: bool = True,
        retry_on_failure: bool = True
    ) -> Dict[str, Any]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            source_lang: Source language code (auto-detected if None)
            use_cache: Whether to use translation cache
            retry_on_failure: Whether to retry on failure
            
        Returns:
            Dictionary with translation results
        """
        start_time = time.time()
        
        # Validate inputs
        if not text or len(text.strip()) == 0:
            return {
                'original': text,
                'translated': text,
                'source_language': source_lang or 'unknown',
                'target_language': target_lang,
                'success': True,
                'cached': False,
                'error': None,
                'translation_time': 0.0,
                'method': 'noop_empty'
            }
        
        # Normalize language codes
        target_lang = self.normalize_language_code(target_lang)
        if source_lang:
            source_lang = self.normalize_language_code(source_lang)
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(text, target_lang, source_lang)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return {
                    'original': text,
                    'translated': cached,
                    'source_language': source_lang or 'auto',
                    'target_language': target_lang,
                    'success': True,
                    'cached': True,
                    'error': None,
                    'translation_time': time.time() - start_time,
                    'method': 'cache'
                }
        
        # Check if translation is needed
        if source_lang and source_lang == target_lang:
            # Same language, no translation needed
            result = {
                'original': text,
                'translated': text,
                'source_language': source_lang,
                'target_language': target_lang,
                'success': True,
                'cached': False,
                'error': None,
                'translation_time': time.time() - start_time,
                'method': 'noop_same_language'
            }
            
            # Cache the result (even though it's the same)
            if cache_key:
                self.cache.set(cache_key, text)
            
            return result
        
        # Perform translation
        last_error = None
        
        for attempt in range(self.retries if retry_on_failure else 1):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                logger.debug(f"Translating {len(text)} chars from {source_lang or 'auto'} to {target_lang} (attempt {attempt + 1})")
                
                # Call Google Translate
                translated = self.translator.translate(
                    text, 
                    dest=target_lang,
                    src=source_lang
                )
                
                # Extract result
                translated_text = translated.text
                detected_src_lang = self.normalize_language_code(translated.src)
                
                # Build result
                result = {
                    'original': text,
                    'translated': translated_text,
                    'source_language': detected_src_lang,
                    'target_language': target_lang,
                    'success': True,
                    'cached': False,
                    'error': None,
                    'translation_time': time.time() - start_time,
                    'method': 'googletrans',
                    'attempts': attempt + 1,
                    'extra_pronunciation': getattr(translated, 'pronunciation', None),
                    'extra_extra_data': getattr(translated, 'extra_data', None)
                }
                
                # Cache the result
                if use_cache and cache_key:
                    self.cache.set(cache_key, translated_text)
                
                logger.debug(f"Translation successful ({len(text)} chars -> {len(translated_text)} chars)")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                
                # Wait before retry
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
        
        # All attempts failed
        error_result = {
            'original': text,
            'translated': text,  # Return original as fallback
            'source_language': source_lang or 'unknown',
            'target_language': target_lang,
            'success': False,
            'cached': False,
            'error': str(last_error),
            'translation_time': time.time() - start_time,
            'method': 'failed',
            'attempts': self.retries
        }
        
        logger.error(f"Translation failed after {self.retries} attempts: {last_error}")
        return error_result
    
    def batch_translate(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            target_lang: Target language code
            source_lang: Source language code
            use_cache: Whether to use cache
            
        Returns:
            List of translation results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.translate_text(
                    text=text,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    use_cache=use_cache,
                    retry_on_failure=True
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to translate text {i}: {e}")
                results.append({
                    'original': text,
                    'translated': text,
                    'source_language': source_lang or 'unknown',
                    'target_language': target_lang,
                    'success': False,
                    'cached': False,
                    'error': str(e),
                    'translation_time': 0.0,
                    'method': 'batch_error'
                })
            
            # Small delay between translations
            if i < len(texts) - 1:
                time.sleep(0.1)
        
        return results
    
    def translate_with_context(
        self,
        text: str,
        target_lang: str,
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Translate text with additional context for better accuracy.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            context: Additional context about the text
            use_cache: Whether to use cache
            
        Returns:
            Translation result with context
        """
        # If context is provided, prepend it to the text
        if context and len(context) > 0:
            # Format: [Context: ...] Actual text
            contextual_text = f"[Context: {context}]\n{text}"
            
            result = self.translate_text(
                text=contextual_text,
                target_lang=target_lang,
                source_lang=None,  # Auto-detect
                use_cache=False,  # Don't cache contextual translations
                retry_on_failure=True
            )
            
            # Remove context from translated text if present
            translated = result['translated']
            if translated.startswith('[Context:'):
                # Try to extract actual translation
                lines = translated.split('\n')
                if len(lines) > 1:
                    result['translated'] = '\n'.join(lines[1:]).strip()
                else:
                    # Fallback: remove context marker
                    result['translated'] = translated.replace('[Context:', '').replace(']', '').strip()
            
            result['method'] = 'contextual'
            result['context_used'] = True
            return result
        
        # Regular translation without context
        return self.translate_text(
            text=text,
            target_lang=target_lang,
            source_lang=None,
            use_cache=use_cache,
            retry_on_failure=True
        )
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get all supported languages.
        
        Returns:
            Dictionary of language_code -> language_name
        """
        return self._supported_languages.copy()
    
    def is_language_supported(self, lang_code: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            lang_code: Language code
            
        Returns:
            True if supported, False otherwise
        """
        norm_code = self.normalize_language_code(lang_code)
        return norm_code in self._supported_languages
    
    def get_language_name(self, lang_code: str) -> str:
        """
        Get human-readable language name.
        
        Args:
            lang_code: Language code
            
        Returns:
            Language name or code if unknown
        """
        norm_code = self.normalize_language_code(lang_code)
        return self._supported_languages.get(norm_code, lang_code)
    
    def clear_cache(self):
        """Clear translation cache."""
        self.cache.clear()
        logger.info("Translation cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get translator statistics.
        
        Returns:
            Dictionary with statistics
        """
        cache_stats = self.cache.stats()
        
        wait_time = self.rate_limiter.get_wait_time()
        
        return {
            'cache': cache_stats,
            'rate_limiter': {
                'calls_per_minute': self.rate_limiter.calls_per_minute,
                'current_calls': len(self.rate_limiter.call_times),
                'wait_time_seconds': wait_time
            },
            'supported_languages_count': len(self._supported_languages),
            'timeout': self.timeout,
            'retries': self.retries
        }

# Global translator instance
_global_translator = None

def get_translator() -> DocuKnowTranslator:
    """
    Get or create global translator instance.
    
    Returns:
        DocuKnowTranslator instance
    """
    global _global_translator
    if _global_translator is None:
        _global_translator = DocuKnowTranslator()
    return _global_translator

# Convenience functions
def translate_text(
    text: str, 
    target_lang: str, 
    source_lang: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to translate text.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        source_lang: Source language code
        
    Returns:
        Translated text or None if failed
    """
    try:
        translator = get_translator()
        result = translator.translate_text(text, target_lang, source_lang)
        
        if result['success']:
            return result['translated']
        else:
            logger.error(f"Translation failed: {result['error']}")
            return None
            
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return None

def detect_language(text: str) -> Dict[str, Any]:
    """
    Convenience function to detect language.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language detection result
    """
    try:
        translator = get_translator()
        return translator.detect_language(text)
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'error': str(e)
        }

def is_translation_available() -> bool:
    """
    Check if translation services are available.
    
    Returns:
        True if translation is available
    """
    return GOOGLE_TRANSLATE_AVAILABLE

# Test function
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing DocuKnow Translator")
    print("=" * 60)
    
    if not GOOGLE_TRANSLATE_AVAILABLE:
        print("ERROR: Google Translate not available. Please install:")
        print("pip install googletrans==3.1.0a0")
        exit(1)
    
    try:
        # Create translator
        translator = DocuKnowTranslator(cache_size=100)
        
        # Test language detection
        print("\n1. Language Detection Test:")
        print("-" * 40)
        
        test_texts = [
            "Hello, how are you?",
            "नमस्ते, कैसे हैं आप?",
            "Bonjour, comment allez-vous?",
            "你好，你今天好吗？",
            "Hola, ¿cómo estás?",
        ]
        
        for text in test_texts:
            detection = translator.detect_language(text)
            lang_name = translator.get_language_name(detection['language'])
            print(f"Text: {text[:30]}...")
            print(f"  Detected: {detection['language']} ({lang_name})")
            print(f"  Confidence: {detection['confidence']:.2f}")
            print()
        
        # Test translation
        print("\n2. Translation Test:")
        print("-" * 40)
        
        test_translations = [
            ("Hello world", "en", "hi"),
            ("Artificial Intelligence is amazing", "en", "fr"),
            ("मैं एक सॉफ्टवेयर इंजीनियर हूं", "hi", "en"),
            ("This is a test sentence", "en", "es"),
        ]
        
        for text, src, tgt in test_translations:
            print(f"\nTranslating: '{text}'")
            print(f"From: {src} -> To: {tgt}")
            
            result = translator.translate_text(text, tgt, src)
            
            if result['success']:
                print(f"Result: {result['translated']}")
                print(f"Time: {result['translation_time']:.2f}s, Cached: {result['cached']}")
            else:
                print(f"Failed: {result['error']}")
            
            time.sleep(0.5)  # Rate limiting
        
        # Test batch translation
        print("\n3. Batch Translation Test:")
        print("-" * 40)
        
        batch_texts = [
            "Good morning",
            "How are you?",
            "Thank you very much",
        ]
        
        batch_results = translator.batch_translate(batch_texts, "hi")
        
        for i, result in enumerate(batch_results):
            print(f"{i+1}. {batch_texts[i]} -> {result['translated']}")
        
        # Test cache
        print("\n4. Cache Test:")
        print("-" * 40)
        
        test_text = "Cache test translation"
        
        # First translation (should miss cache)
        result1 = translator.translate_text(test_text, "fr", use_cache=True)
        print(f"First translation: cached={result1['cached']}, time={result1['translation_time']:.3f}s")
        
        # Second translation (should hit cache)
        result2 = translator.translate_text(test_text, "fr", use_cache=True)
        print(f"Second translation: cached={result2['cached']}, time={result2['translation_time']:.3f}s")
        
        # Show stats
        print("\n5. Translator Statistics:")
        print("-" * 40)
        
        stats = translator.get_stats()
        print(f"Cache stats: {stats['cache']}")
        print(f"Supported languages: {len(translator.get_supported_languages())}")
        
        # List some supported languages
        print("\n6. Some Supported Languages:")
        print("-" * 40)
        
        languages = translator.get_supported_languages()
        for code, name in list(languages.items())[:15]:  # First 15
            print(f"{code}: {name}")
        
        print(f"\nTotal supported languages: {len(languages)}")
        
        # Test convenience functions
        print("\n7. Convenience Functions Test:")
        print("-" * 40)
        
        # Translate using convenience function
        translated = translate_text("Hello from convenience function", "es")
        print(f"Convenience translate: {translated}")
        
        # Detect language
        detection = detect_language("こんにちは世界")
        print(f"Language detection: {detection['language']} (confidence: {detection['confidence']:.2f})")
        
        # Check availability
        print(f"\nTranslation available: {is_translation_available()}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()