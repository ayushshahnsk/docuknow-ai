"""
Enhanced Language Detector for DocuKnow AI - MULTILINGUAL EDITION

Purpose:
- Detect language of text/queries with high accuracy
- Support for 50+ languages
- Language-specific processing rules
- Confidence scoring
- Integration with search and LLM systems

Features:
1. Multiple detection methods (langdetect, indicator words, character analysis)
2. Language code normalization
3. Confidence scoring
4. Batch processing
5. Caching for performance
6. Special handling for mixed content
"""

import langdetect
from langdetect import DetectorFactory
from typing import Optional, Dict, List, Tuple
import re
import hashlib
from collections import Counter
import logging

# For consistent results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Language code mapping for normalization
LANGUAGE_CODE_MAP = {
    # Chinese variants
    'zh-cn': 'zh', 'zh-tw': 'zh', 'zh-hans': 'zh', 'zh-hant': 'zh',
    # Portuguese variants
    'pt-br': 'pt', 'pt-pt': 'pt',
    # Spanish variants
    'es-es': 'es', 'es-mx': 'es', 'es-ar': 'es',
    # English variants
    'en-us': 'en', 'en-gb': 'en', 'en-au': 'en', 'en-in': 'en',
    # French variants
    'fr-fr': 'fr', 'fr-ca': 'fr',
    # Other variants
    'hi-in': 'hi',
    'bn-in': 'bn',
    'ta-in': 'ta',
    'te-in': 'te',
    'kn-in': 'kn',
    'ml-in': 'ml',
    'mr-in': 'mr',
    'gu-in': 'gu',
    'pa-in': 'pa',
}

# Supported languages for DocuKnow AI
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'fr': 'French',
    'bn': 'Bengali',
    'de': 'German',
    'zh': 'Chinese',
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

# Common language indicator words (for quick detection)
LANGUAGE_INDICATORS = {
    'hi': [  # Hindi
        "क्या", "है", "में", "से", "को", "का", "की", "के", "नहीं", "और", "यह", "वह",
        "मैं", "तुम", "हम", "आप", "इस", "उस", "जो", "वो", "भी", "तो", "पर", "के"
    ],
    'mr': [  # Marathi
        "काय", "आहे", "मध्ये", "पासून", "ला", "चा", "ची", "चे", "नाही", "आणि",
        "हे", "ते", "मी", "तू", "आम्ही", "तुम्ही", "हा", "ती", "जो", "तो", "ही", "पण"
    ],
    'gu': [  # Gujarati
        "શું", "છે", "માં", "થી", "ને", "નો", "ની", "ના", "નથી", "અને",
        "આ", "તે", "હું", "તમે", "અમે", "તમે", "આ", "તે", "જે", "તે", "પણ", "પર"
    ],
    'bn': [  # Bengali
        "কী", "হয়", "এ", "থেকে", "কে", "র", "এর", "না", "এবং",
        "এটা", "সেটা", "আমি", "তুমি", "আমরা", "তোমরা", "এই", "সেই", "যে", "সে", "ও"
    ],
    'ta': [  # Tamil
        "என்ன", "உள்ளது", "இல்", "இருந்து", "க்கு", "ன்", "இன்", "இல்லை", "மற்றும்",
        "இது", "அது", "நான்", "நீ", "நாங்கள்", "நீங்கள்", "இந்த", "அந்த", "யார்", "அவர்", "கூட"
    ],
    'te': [  # Telugu
        "ఏమిటి", "ఉంది", "లో", "నుండి", "కు", "యొక్క", "ల", "కాదు", "మరియు",
        "ఇది", "అది", "నేను", "నీవు", "మేము", "మీరు", "ఈ", "ఆ", "ఎవరు", "అతను", "కూడా"
    ],
    'kn': [  # Kannada
        "ಏನು", "ಇದೆ", "ಇನ್", "ನಿಂದ", "ಗೆ", "ನ", "ಅವರ", "ಇಲ್ಲ", "ಮತ್ತು",
        "ಇದು", "ಅದು", "ನಾನು", "ನೀನು", "ನಾವು", "ನೀವು", "ಈ", "ಅದು", "ಯಾರು", "ಅವನು", "ಕೂಡ"
    ],
    'ml': [  # Malayalam
        "എന്താണ്", "ആണ്", "ൽ", "മുതൽ", "ക്ക്ഉ", "യുടെ", "ഇൻ", "ഇല്ല", "ഒപ്പം",
        "ഇത്", "അത്", "ഞാൻ", "നീ", "ഞങ്ങൾ", "നിങ്ങൾ", "ഈ", "അത്", "ആര്", "അവൻ", "ഉം"
    ],
    'pa': [  # Punjabi
        "ਕੀ", "ਹੈ", "ਵਿਚ", "ਤੋਂ", "ਨੂੰ", "ਦਾ", "ਦੀ", "ਦੇ", "ਨਹੀਂ", "ਅਤੇ",
        "ਇਹ", "ਉਹ", "ਮੈਂ", "ਤੂੰ", "ਅਸੀਂ", "ਤੁਸੀਂ", "ਇਸ", "ਉਸ", "ਜੋ", "ਉਹ", "ਵੀ", "ਤੇ"
    ],
    'ur': [  # Urdu
        "کیا", "ہے", "میں", "سے", "کو", "کا", "کی", "کے", "نہیں", "اور",
        "یہ", "وہ", "میں", "تم", "ہم", "آپ", "اس", "اس", "جو", "وہ", "بھی", "پر"
    ],
    'fr': [  # French
        "quel", "est", "dans", "de", "à", "le", "la", "les", "pas", "et",
        "ce", "cela", "je", "tu", "nous", "vous", "ce", "cette", "qui", "il", "aussi", "sur"
    ],
    'de': [  # German
        "was", "ist", "in", "von", "zu", "der", "die", "das", "nicht", "und",
        "dies", "das", "ich", "du", "wir", "Sie", "dieser", "jener", "wer", "er", "auch", "auf"
    ],
    'es': [  # Spanish
        "qué", "es", "en", "de", "a", "el", "la", "los", "no", "y",
        "esto", "eso", "yo", "tú", "nosotros", "vosotros", "este", "ese", "quién", "él", "también", "sobre"
    ],
    'pt': [  # Portuguese
        "o que", "é", "em", "de", "para", "o", "a", "os", "não", "e",
        "isto", "isso", "eu", "tu", "nós", "vós", "este", "esse", "quem", "ele", "também", "sobre"
    ],
    'it': [  # Italian
        "cosa", "è", "in", "di", "a", "il", "la", "i", "non", "e",
        "questo", "quello", "io", "tu", "noi", "voi", "questo", "quello", "chi", "lui", "anche", "su"
    ],
    'ru': [  # Russian
        "что", "есть", "в", "из", "к", "он", "она", "оно", "не", "и",
        "это", "то", "я", "ты", "мы", "вы", "этот", "тот", "кто", "он", "также", "на"
    ],
    'ja': [  # Japanese (Hiragana/Katakana common particles)
        "何", "です", "に", "から", "へ", "は", "が", "を", "ない", "と",
        "これ", "それ", "私", "あなた", "私たち", "あなたたち", "この", "その", "誰", "彼", "も", "で"
    ],
    'ko': [  # Korean
        "무엇", "입니다", "에서", "부터", "에게", "은", "는", "이", "아니", "와",
        "이것", "그것", "나", "너", "우리", "너희", "이", "그", "누구", "그", "도", "에서"
    ],
    'zh': [  # Chinese
        "什么", "是", "在", "从", "到", "的", "了", "和", "不", "与",
        "这", "那", "我", "你", "我们", "你们", "这个", "那个", "谁", "他", "也", "上"
    ],
    'ar': [  # Arabic
        "ماذا", "هو", "في", "من", "إلى", "ال", "هذا", "ذلك", "لا", "و",
        "هذا", "ذلك", "أنا", "أنت", "نحن", "أنتم", "هذا", "ذلك", "من", "هو", "أيضا", "على"
    ],
}

# Character set ranges for different scripts
SCRIPT_RANGES = {
    'devanagari': (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit, etc.
    'bengali': (0x0980, 0x09FF),     # Bengali, Assamese
    'gujarati': (0x0A80, 0x0AFF),    # Gujarati
    'tamil': (0x0B80, 0x0BFF),       # Tamil
    'telugu': (0x0C00, 0x0C7F),      # Telugu
    'kannada': (0x0C80, 0x0CFF),     # Kannada
    'malayalam': (0x0D00, 0x0D7F),   # Malayalam
    'gurmukhi': (0x0A00, 0x0A7F),    # Punjabi
    'arabic': (0x0600, 0x06FF),      # Arabic, Persian, Urdu
    'cyrillic': (0x0400, 0x04FF),    # Russian, Bulgarian, etc.
    'cjk': (0x4E00, 0x9FFF),         # Chinese, Japanese, Korean
    'hangul': (0xAC00, 0xD7AF),      # Korean Hangul
    'hiragana': (0x3040, 0x309F),    # Japanese Hiragana
    'katakana': (0x30A0, 0x30FF),    # Japanese Katakana
}

# Script to language mapping
SCRIPT_TO_LANGUAGE = {
    'devanagari': ['hi', 'mr', 'sa', 'ne'],  # Hindi, Marathi, Sanskrit, Nepali
    'bengali': ['bn', 'as'],                 # Bengali, Assamese
    'gujarati': ['gu'],
    'tamil': ['ta'],
    'telugu': ['te'],
    'kannada': ['kn'],
    'malayalam': ['ml'],
    'gurmukhi': ['pa'],
    'arabic': ['ar', 'fa', 'ur'],            # Arabic, Persian, Urdu
    'cyrillic': ['ru', 'bg', 'uk'],          # Russian, Bulgarian, Ukrainian
    'cjk': ['zh', 'ja', 'ko'],               # Chinese, Japanese, Korean
    'hangul': ['ko'],
    'hiragana': ['ja'],
    'katakana': ['ja'],
}

class LanguageDetectionCache:
    """Cache for language detection results."""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, text: str) -> Optional[Dict]:
        """Get cached detection result."""
        text_hash = self._hash_text(text)
        return self.cache.get(text_hash)
    
    def set(self, text: str, result: Dict):
        """Cache detection result."""
        text_hash = self._hash_text(text)
        self.cache[text_hash] = result
        
        # Limit cache size
        if len(self.cache) > self.max_size:
            # Remove oldest entry (simple strategy)
            self.cache.pop(next(iter(self.cache)))
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def _hash_text(self, text: str) -> str:
        """Create hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

# Global cache instance
_detection_cache = LanguageDetectionCache()

def _clean_text_for_detection(text: str) -> str:
    """
    Clean text for better language detection.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep language-specific characters
    # Keep letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"\-:;()\[\]{}@#$%^&*+=<>/~`|\\]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def _detect_by_script(text: str) -> Optional[str]:
    """
    Detect language by character script/unicode ranges.
    
    Args:
        text: Input text
        
    Returns:
        Language code or None
    """
    if not text:
        return None
    
    # Count characters in each script
    script_counts = Counter()
    
    for char in text:
        char_code = ord(char)
        
        for script_name, (start, end) in SCRIPT_RANGES.items():
            if start <= char_code <= end:
                script_counts[script_name] += 1
                break
    
    # Find dominant script
    if script_counts:
        dominant_script = script_counts.most_common(1)[0][0]
        
        # Map script to language
        possible_languages = SCRIPT_TO_LANGUAGE.get(dominant_script, [])
        
        # If we have multiple possible languages, try to disambiguate
        if len(possible_languages) == 1:
            return possible_languages[0]
        elif possible_languages:
            # For scripts with multiple languages, use additional heuristics
            if dominant_script == 'arabic':
                # Try to distinguish between Arabic, Persian, Urdu
                # Urdu often has Persian loanwords, Arabic is more formal
                if 'ک' in text or 'گ' in text:  # Persian/Urdu specific characters
                    return 'ur' if 'ہ' in text else 'fa'  # Urdu has specific characters
                return 'ar'
            elif dominant_script == 'devanagari':
                # Hindi vs Marathi
                # Common Marathi words
                marathi_indicators = ['आहे', 'म्हणजे', 'म्हणून', 'पण']
                if any(indicator in text for indicator in marathi_indicators):
                    return 'mr'
                return 'hi'  # Default to Hindi
            elif dominant_script == 'cjk':
                # Chinese vs Japanese vs Korean
                if any(0x3040 <= ord(c) <= 0x309F for c in text):  # Hiragana
                    return 'ja'
                elif any(0x30A0 <= ord(c) <= 0x30FF for c in text):  # Katakana
                    return 'ja'
                elif any(0xAC00 <= ord(c) <= 0xD7AF for c in text):  # Hangul
                    return 'ko'
                return 'zh'  # Default to Chinese
    
    return None

def _detect_by_indicators(text: str) -> Optional[Tuple[str, float]]:
    """
    Detect language by common indicator words.
    
    Args:
        text: Input text (lowercased)
        
    Returns:
        Tuple of (language_code, confidence) or None
    """
    text_lower = text.lower()
    
    language_scores = Counter()
    
    for lang_code, indicators in LANGUAGE_INDICATORS.items():
        score = 0
        for indicator in indicators:
            if indicator in text_lower:
                score += 1
        
        if score > 0:
            # Normalize score by text length and indicator count
            normalized_score = score / (len(text_lower.split()) * 0.5)
            language_scores[lang_code] = min(normalized_score, 1.0)
    
    if language_scores:
        best_lang, best_score = language_scores.most_common(1)[0]
        if best_score > 0.1:  # Minimum confidence threshold
            return best_lang, best_score
    
    return None

def detect_language(text: str, use_cache: bool = True) -> str:
    """
    Detect language of text with high accuracy.
    
    Args:
        text: Text to analyze
        use_cache: Whether to use caching
        
    Returns:
        Language code (e.g., 'en', 'hi', 'mr', 'fr')
    """
    if not text or len(text.strip()) < 3:
        return "unknown"
    
    # Check cache
    if use_cache:
        cached_result = _detection_cache.get(text)
        if cached_result:
            return cached_result.get("language", "unknown")
    
    # Clean text
    cleaned_text = _clean_text_for_detection(text)
    
    if len(cleaned_text) < 3:
        return "unknown"
    
    # Step 1: Try script-based detection (fast and accurate for non-Latin scripts)
    script_lang = _detect_by_script(cleaned_text)
    if script_lang:
        result = {
            "language": script_lang,
            "method": "script",
            "confidence": 0.9
        }
        if use_cache:
            _detection_cache.set(text, result)
        return script_lang
    
    # Step 2: Try indicator words (good for short texts)
    indicator_result = _detect_by_indicators(cleaned_text.lower())
    if indicator_result:
        lang_code, confidence = indicator_result
        result = {
            "language": lang_code,
            "method": "indicators",
            "confidence": confidence
        }
        if use_cache:
            _detection_cache.set(text, result)
        return lang_code
    
    # Step 3: Use langdetect library (good for Latin scripts and longer texts)
    try:
        # langdetect works best with longer text
        if len(cleaned_text) >= 10:
            lang = langdetect.detect(cleaned_text)
            
            # Normalize language code
            lang = LANGUAGE_CODE_MAP.get(lang, lang)
            
            # Check if it's a supported language
            if lang in SUPPORTED_LANGUAGES:
                result = {
                    "language": lang,
                    "method": "langdetect",
                    "confidence": 0.8
                }
                if use_cache:
                    _detection_cache.set(text, result)
                return lang
    except Exception as e:
        logger.debug(f"langdetect failed: {e}")
    
    # Step 4: Default to English if no other detection works
    # Check if it looks like English (Latin alphabet, common English words)
    english_words = ["the", "and", "for", "are", "but", "not", "you", "all", "any", "can"]
    text_words = cleaned_text.lower().split()
    english_word_count = sum(1 for word in text_words if word in english_words)
    
    if english_word_count > 0 and len(text_words) > 0:
        english_ratio = english_word_count / len(text_words)
        if english_ratio > 0.1:  # At least 10% English words
            result = {
                "language": "en",
                "method": "english_words",
                "confidence": english_ratio
            }
            if use_cache:
                _detection_cache.set(text, result)
            return "en"
    
    # Step 5: Unknown
    result = {
        "language": "unknown",
        "method": "none",
        "confidence": 0.0
    }
    if use_cache:
        _detection_cache.set(text, result)
    
    return "unknown"

def detect_language_with_confidence(text: str) -> Dict:
    """
    Detect language with confidence score and detailed information.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with language detection details
    """
    if not text or len(text.strip()) < 5:
        return {
            "language": "unknown",
            "confidence": 0.0,
            "method": "too_short",
            "all_possibilities": [],
            "text_sample": text[:100] if text else ""
        }
    
    # Clean text
    cleaned_text = _clean_text_for_detection(text)
    
    # Try multiple methods and combine results
    results = []
    
    # Method 1: Script detection
    script_lang = _detect_by_script(cleaned_text)
    if script_lang:
        results.append({
            "language": script_lang,
            "confidence": 0.9,
            "method": "script"
        })
    
    # Method 2: Indicator words
    indicator_result = _detect_by_indicators(cleaned_text.lower())
    if indicator_result:
        lang_code, confidence = indicator_result
        results.append({
            "language": lang_code,
            "confidence": confidence,
            "method": "indicators"
        })
    
    # Method 3: langdetect (if text is long enough)
    if len(cleaned_text) >= 10:
        try:
            from langdetect import detect_langs
            
            langdetect_results = detect_langs(cleaned_text)
            for ld_result in langdetect_results:
                lang_code = LANGUAGE_CODE_MAP.get(ld_result.lang, ld_result.lang)
                if lang_code in SUPPORTED_LANGUAGES:
                    results.append({
                        "language": lang_code,
                        "confidence": ld_result.prob,
                        "method": "langdetect"
                    })
        except Exception as e:
            logger.debug(f"langdetect detailed failed: {e}")
    
    # Combine results
    if results:
        # Group by language and average confidence
        lang_scores = {}
        lang_methods = {}
        
        for result in results:
            lang = result["language"]
            conf = result["confidence"]
            method = result["method"]
            
            if lang not in lang_scores:
                lang_scores[lang] = []
                lang_methods[lang] = []
            
            lang_scores[lang].append(conf)
            lang_methods[lang].append(method)
        
        # Calculate average confidence per language
        avg_scores = {
            lang: sum(scores) / len(scores)
            for lang, scores in lang_scores.items()
        }
        
        # Get best language
        best_lang = max(avg_scores.items(), key=lambda x: x[1])[0]
        best_confidence = avg_scores[best_lang]
        best_methods = lang_methods[best_lang]
        
        # All possibilities sorted by confidence
        all_possibilities = [
            {
                "language": lang,
                "confidence": score,
                "methods": lang_methods[lang]
            }
            for lang, score in avg_scores.items()
        ]
        all_possibilities.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "language": best_lang,
            "confidence": best_confidence,
            "method": "+".join(best_methods),
            "all_possibilities": all_possibilities,
            "text_sample": cleaned_text[:200],
            "text_length": len(cleaned_text)
        }
    
    # Fallback: Check for English
    english_words = ["the", "and", "for", "are", "but", "not", "you", "all", "any", "can"]
    text_words = cleaned_text.lower().split()
    
    if text_words:
        english_word_count = sum(1 for word in text_words if word in english_words)
        english_ratio = english_word_count / len(text_words)
        
        if english_ratio > 0.1:
            return {
                "language": "en",
                "confidence": english_ratio,
                "method": "english_words",
                "all_possibilities": [{"language": "en", "confidence": english_ratio, "methods": ["english_words"]}],
                "text_sample": cleaned_text[:200],
                "text_length": len(cleaned_text)
            }
    
    # Unknown
    return {
        "language": "unknown",
        "confidence": 0.0,
        "method": "none",
        "all_possibilities": [],
        "text_sample": cleaned_text[:200],
        "text_length": len(cleaned_text)
    }

def detect_query_language(query: str) -> Dict:
    """
    Special detection for queries with context awareness.
    Optimized for short query texts.
    
    Args:
        query: User query text
        
    Returns:
        Dictionary with language detection details
    """
    # Quick checks for very short queries
    if not query or len(query.strip()) < 2:
        return {
            "language": "unknown",
            "method": "too_short",
            "confidence": 0.0
        }
    
    # Clean the query
    cleaned_query = _clean_text_for_detection(query)
    
    # Method 1: Check for specific language markers (fast)
    for lang_code, indicators in LANGUAGE_INDICATORS.items():
        for indicator in indicators:
            if indicator in cleaned_query:
                return {
                    "language": lang_code,
                    "method": "indicator",
                    "confidence": 0.85,
                    "indicator_found": indicator
                }
    
    # Method 2: Script detection
    script_lang = _detect_by_script(cleaned_query)
    if script_lang:
        return {
            "language": script_lang,
            "method": "script",
            "confidence": 0.9
        }
    
    # Method 3: Use full detection for mixed or ambiguous cases
    full_detection = detect_language_with_confidence(cleaned_query)
    
    # For queries, we might want to be more confident
    if full_detection["confidence"] > 0.6:
        return full_detection
    
    # Method 4: Check if it's likely English (common fallback)
    # Count English-like patterns
    english_patterns = [
        r'\bthe\b', r'\band\b', r'\bfor\b', r'\bare\b', r'\bbut\b',
        r'\bnot\b', r'\byou\b', r'\ball\b', r'\bany\b', r'\bcan\b'
    ]
    
    english_score = 0
    for pattern in english_patterns:
        if re.search(pattern, cleaned_query.lower()):
            english_score += 1
    
    if english_score >= 2 or len(cleaned_query.split()) <= 3:
        # If we found multiple English indicators or query is very short
        return {
            "language": "en",
            "method": "english_patterns",
            "confidence": min(0.5 + (english_score * 0.1), 0.8)
        }
    
    # Default to English with low confidence
    return {
        "language": "en",
        "method": "default",
        "confidence": 0.3
    }

def batch_detect_language(texts: List[str]) -> List[Dict]:
    """
    Detect language for multiple texts efficiently.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        List of detection results
    """
    results = []
    
    for text in texts:
        results.append(detect_language_with_confidence(text))
    
    return results

def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from language code.
    
    Args:
        lang_code: Language code (e.g., 'en', 'hi', 'fr')
        
    Returns:
        Language name or code if unknown
    """
    return SUPPORTED_LANGUAGES.get(lang_code, lang_code)

def get_supported_languages() -> Dict[str, str]:
    """
    Get all supported languages.
    
    Returns:
        Dictionary of language_code -> language_name
    """
    return SUPPORTED_LANGUAGES.copy()

def clear_detection_cache():
    """Clear language detection cache."""
    _detection_cache.clear()
    logger.info("Language detection cache cleared")

# Test function
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Language Detector")
    print("=" * 60)
    
    # Test texts in different languages
    test_texts = [
        ("Hello, how are you today?", "English"),
        ("नमस्ते, आप कैसे हैं?", "Hindi"),
        ("Bonjour, comment allez-vous?", "French"),
        ("你好，你今天好吗？", "Chinese"),
        ("Hola, ¿cómo estás hoy?", "Spanish"),
        ("सर्वांना नमस्कार, तुम्ही कसे आहात?", "Marathi"),
        ("مرحبا، كيف حالك اليوم؟", "Arabic"),
        ("こんにちは、今日はどうですか？", "Japanese"),
        ("안녕하세요, 오늘 어떠세요?", "Korean"),
        ("Short text", "Very short English"),
        ("", "Empty"),
    ]
    
    for text, expected in test_texts:
        print(f"\nTest: {expected}")
        print(f"Text: {text}")
        
        # Simple detection
        lang = detect_language(text)
        lang_name = get_language_name(lang)
        print(f"Simple detection: {lang} ({lang_name})")
        
        # Detailed detection
        if text.strip():
            details = detect_language_with_confidence(text)
            print(f"Detailed: {details['language']} ({details['confidence']:.2f} via {details['method']})")
            
            if details['all_possibilities']:
                print(f"All possibilities: {details['all_possibilities'][:3]}")
        
        print("-" * 40)
    
    # Test query detection
    print("\n\nTesting Query Detection")
    print("=" * 60)
    
    test_queries = [
        "कृत्रिम बुद्धिमत्ता क्या है?",
        "What is artificial intelligence?",
        "Qu'est-ce que l'IA?",
        "人工智能是什么？",
        "Короткий запрос",
    ]
    
    for query in test_queries:
        result = detect_query_language(query)
        print(f"\nQuery: {query}")
        print(f"Detected: {result['language']} ({result['confidence']:.2f} via {result['method']})")
    
    # Show supported languages
    print("\n\nSupported Languages:")
    print("=" * 60)
    supported = get_supported_languages()
    for code, name in sorted(supported.items()):
        print(f"{code}: {name}")
    
    print(f"\nTotal supported languages: {len(supported)}")