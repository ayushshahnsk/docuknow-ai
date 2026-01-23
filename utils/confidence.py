from typing import List, Dict


def calculate_confidence(contexts: List[Dict]) -> Dict:
    """
    Calculate confidence based on similarity scores.
    """
    if not contexts:
        return {
            "level": "Low",
            "score": 0.0,
            "message": "Not enough relevant information found.",
        }

    avg_score = sum(c["score"] for c in contexts) / len(contexts)

    if avg_score >= 0.75:
        level = "High"
        message = "Answer is highly confident based on your documents."
    elif avg_score >= 0.55:
        level = "Medium"
        message = "Answer is moderately confident based on your documents."
    else:
        level = "Low"
        message = "Answer may be incomplete or uncertain."

    return {"level": level, "score": round(avg_score, 2), "message": message}
