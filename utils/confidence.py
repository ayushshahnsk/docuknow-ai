from typing import List, Dict

# ======================================================
# NEW: LANGUAGE DETECTION IMPORT
# ======================================================
try:
    from utils.language_detector import detect_language
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    print("Language detector not available. Install langdetect: pip install langdetect")

# ======================================================
# NEW: CONFIGURABLE THRESHOLDS FOR DIFFERENT LANGUAGES
# ======================================================
# Default thresholds (can be overridden from app.py)
CONFIDENCE_THRESHOLDS = {
    "default": {
        "high": 0.75,
        "medium": 0.55,
        "low": 0.0
    },
    "hi": {  # Hindi
        "high": 0.65,  # Lower thresholds for Hindi
        "medium": 0.45,
        "low": 0.0
    },
    "en": {  # English
        "high": 0.75,
        "medium": 0.55,
        "low": 0.0
    },
    "es": {  # Spanish
        "high": 0.70,
        "medium": 0.50,
        "low": 0.0
    },
    "fr": {  # French
        "high": 0.70,
        "medium": 0.50,
        "low": 0.0
    }
}

# ======================================================
# MAIN CONFIDENCE CALCULATION FUNCTION (UPDATED)
# ======================================================
def calculate_confidence(contexts: List[Dict]) -> Dict:
    """
    Calculate confidence based on similarity scores with language awareness.
    
    Args:
        contexts: List of retrieved contexts with scores
        
    Returns:
        Dictionary with level, score, and message
    """
    if not contexts:
        return {
            "level": "Low",
            "score": 0.0,
            "message": "Not enough relevant information found.",
            "contexts_count": 0
        }
    
    # Calculate average score
    avg_score = sum(c["score"] for c in contexts) / len(contexts)
    
    # ======================================================
    # NEW: DETECT LANGUAGE OF CONTENT FOR BETTER THRESHOLDS
    # ======================================================
    detected_language = "default"
    if LANGUAGE_DETECTION_AVAILABLE and contexts:
        # Sample text from first context for language detection
        sample_text = contexts[0].get("text", "")
        if sample_text:
            try:
                detected_language = detect_language(sample_text)
            except:
                detected_language = "default"
    
    # Get thresholds for detected language
    thresholds = CONFIDENCE_THRESHOLDS.get(detected_language, CONFIDENCE_THRESHOLDS["default"])
    
    # ======================================================
    # NEW: LANGUAGE-AWARE CONFIDENCE LEVELS
    # ======================================================
    if avg_score >= thresholds["high"]:
        level = "High"
        message = "Answer is highly confident based on your documents."
        if detected_language != "en" and detected_language != "default":
            message += f" (Detected {detected_language.upper()} content)"
            
    elif avg_score >= thresholds["medium"]:
        level = "Medium"
        message = "Answer is moderately confident based on your documents."
        if detected_language != "en" and detected_language != "default":
            message += f" (Detected {detected_language.upper()} content)"
            
    else:
        level = "Low"
        message = "Answer may be incomplete or uncertain."
        if detected_language != "en" and detected_language != "default":
            message += f" (Note: {detected_language.upper()} content may have lower similarity scores)"
    
    return {
        "level": level,
        "score": round(avg_score, 3),  # More precision
        "message": message,
        "contexts_count": len(contexts),
        "detected_language": detected_language,  # NEW: Return detected language
        "thresholds_used": thresholds  # NEW: Return thresholds used
    }

# ======================================================
# NEW: ENHANCED CONFIDENCE WITH SCORE DISTRIBUTION
# ======================================================
def calculate_confidence_detailed(contexts: List[Dict]) -> Dict:
    """
    Enhanced confidence calculation with score distribution analysis.
    Useful for debugging source selection issues.
    """
    if not contexts:
        return {
            "level": "Low",
            "score": 0.0,
            "message": "No contexts found.",
            "score_distribution": [],
            "score_range": (0.0, 0.0),
            "median_score": 0.0
        }
    
    # Get basic confidence
    basic_conf = calculate_confidence(contexts)
    
    # Calculate score distribution
    scores = [c["score"] for c in contexts]
    scores.sort()
    
    # Additional metrics
    score_range = (min(scores), max(scores))
    median_score = scores[len(scores) // 2] if scores else 0.0
    
    # Score categories
    high_scores = [s for s in scores if s >= 0.7]
    medium_scores = [s for s in scores if 0.4 <= s < 0.7]
    low_scores = [s for s in scores if s < 0.4]
    
    basic_conf.update({
        "score_distribution": {
            "high_count": len(high_scores),
            "medium_count": len(medium_scores),
            "low_count": len(low_scores),
            "high_percentage": len(high_scores) / len(scores) * 100,
            "medium_percentage": len(medium_scores) / len(scores) * 100,
            "low_percentage": len(low_scores) / len(scores) * 100
        },
        "score_range": score_range,
        "median_score": median_score,
        "all_scores": scores  # For debugging
    })
    
    return basic_conf

# ======================================================
# NEW: LANGUAGE-SPECIFIC CONFIDENCE ADJUSTMENT
# ======================================================
def adjust_confidence_for_language(confidence_score: float, language: str) -> Dict:
    """
    Adjust confidence interpretation based on language.
    Some languages naturally have lower similarity scores with multilingual embeddings.
    """
    adjustment_factors = {
        "hi": 1.15,  # Hindi: Boost confidence by 15%
        "ja": 1.10,  # Japanese: Boost by 10%
        "ko": 1.10,  # Korean: Boost by 10%
        "zh": 1.10,  # Chinese: Boost by 10%
        "ar": 1.15,  # Arabic: Boost by 15%
        "default": 1.0  # No adjustment
    }
    
    adjustment = adjustment_factors.get(language, adjustment_factors["default"])
    adjusted_score = confidence_score * adjustment
    
    # Cap at 1.0
    adjusted_score = min(adjusted_score, 1.0)
    
    # Determine level with adjusted score
    thresholds = CONFIDENCE_THRESHOLDS.get(language, CONFIDENCE_THRESHOLDS["default"])
    
    if adjusted_score >= thresholds["high"]:
        level = "High"
    elif adjusted_score >= thresholds["medium"]:
        level = "Medium"
    else:
        level = "Low"
    
    return {
        "original_score": confidence_score,
        "adjusted_score": round(adjusted_score, 3),
        "level": level,
        "adjustment_factor": adjustment,
        "language": language
    }

# ======================================================
# NEW: THRESHOLD MANAGEMENT FUNCTIONS
# ======================================================
def update_thresholds(language: str, high: float = None, medium: float = None):
    """
    Update confidence thresholds for a specific language.
    """
    if language not in CONFIDENCE_THRESHOLDS:
        CONFIDENCE_THRESHOLDS[language] = CONFIDENCE_THRESHOLDS["default"].copy()
    
    if high is not None:
        CONFIDENCE_THRESHOLDS[language]["high"] = high
    if medium is not None:
        CONFIDENCE_THRESHOLDS[language]["medium"] = medium
    
    print(f"Updated thresholds for {language}: High={CONFIDENCE_THRESHOLDS[language]['high']}, Medium={CONFIDENCE_THRESHOLDS[language]['medium']}")

def get_all_thresholds() -> Dict:
    """
    Get all current threshold settings.
    """
    return CONFIDENCE_THRESHOLDS.copy()

def reset_thresholds_to_default():
    """
    Reset all thresholds to default values.
    """
    global CONFIDENCE_THRESHOLDS
    CONFIDENCE_THRESHOLDS = {
        "default": {
            "high": 0.75,
            "medium": 0.55,
            "low": 0.0
        },
        "hi": {
            "high": 0.65,
            "medium": 0.45,
            "low": 0.0
        },
        "en": {
            "high": 0.75,
            "medium": 0.55,
            "low": 0.0
        },
        "es": {
            "high": 0.70,
            "medium": 0.50,
            "low": 0.0
        },
        "fr": {
            "high": 0.70,
            "medium": 0.50,
            "low": 0.0
        }
    }

# ======================================================
# NEW: DIAGNOSTIC FUNCTION FOR DEBUGGING
# ======================================================
def diagnose_confidence_issue(contexts: List[Dict], query: str = "") -> Dict:
    """
    Diagnostic function to understand why confidence might be low.
    Useful for debugging "PDF has answer but shows internet answer" issue.
    """
    if not contexts:
        return {
            "issue": "NO_CONTEXTS",
            "suggestion": "No documents were retrieved. Check if documents are properly indexed.",
            "severity": "HIGH"
        }
    
    basic_conf = calculate_confidence(contexts)
    detailed_conf = calculate_confidence_detailed(contexts)
    
    issues = []
    suggestions = []
    
    # Check 1: Very low scores
    if basic_conf["score"] < 0.2:
        issues.append("VERY_LOW_SCORES")
        suggestions.append("Similarity scores are very low. This could be due to:")
        suggestions.append("- Language mismatch between query and documents")
        suggestions.append("- Poor quality embeddings")
        suggestions.append("- Unrelated documents")
    
    # Check 2: Inconsistent scores
    score_range = detailed_conf["score_range"]
    if score_range[1] - score_range[0] > 0.5:
        issues.append("INCONSISTENT_SCORES")
        suggestions.append("Scores vary widely across contexts.")
        suggestions.append("Some contexts are relevant, others are not.")
    
    # Check 3: Mostly low scores
    if detailed_conf["score_distribution"]["low_percentage"] > 70:
        issues.append("MOSTLY_LOW_SCORES")
        suggestions.append("Most retrieved contexts have low similarity.")
        suggestions.append("Consider re-indexing with better embeddings.")
    
    # Check 4: Language mismatch
    if basic_conf["detected_language"] not in ["en", "default"]:
        issues.append("NON_ENGLISH_CONTENT")
        suggestions.append(f"Detected {basic_conf['detected_language'].upper()} content.")
        suggestions.append("Using lower confidence thresholds for this language.")
    
    # Check 5: Few contexts
    if len(contexts) < 2:
        issues.append("FEW_CONTEXTS")
        suggestions.append("Only found 1 relevant context.")
        suggestions.append("Consider increasing TOP_K in retrieval settings.")
    
    return {
        "basic_confidence": basic_conf,
        "detailed_analysis": detailed_conf,
        "issues_detected": issues,
        "suggestions": suggestions,
        "query": query,
        "context_count": len(contexts)
    }