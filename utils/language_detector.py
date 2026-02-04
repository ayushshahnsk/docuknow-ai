# utils/language_detector.py
import langdetect
from langdetect import DetectorFactory
from typing import Optional

# For consistent results
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detect language of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code (e.g., 'en', 'hi', 'es')
    """
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

def detect_language_with_confidence(text: str) -> dict:
    """
    Detect language with confidence score.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with language and confidence
    """
    if not text or len(text.strip()) < 20:
        return {"language": "unknown", "confidence": 0.0}
    
    try:
        from langdetect import detect_langs
        languages = detect_langs(text)
        if languages:
            best = languages[0]
            return {
                "language": best.lang,
                "confidence": best.prob,
                "all_languages": [{"lang": l.lang, "prob": l.prob} for l in languages]
            }
    except:
        pass
    
    return {"language": "unknown", "confidence": 0.0}