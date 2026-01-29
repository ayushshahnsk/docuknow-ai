"""
web_search.py

Simple web search fallback using DuckDuckGo Instant Answer API.
Used ONLY when PDF does not contain the answer.
"""

import requests


def web_search(query: str) -> str:
    """
    Fetches real-time information from the web.
    """

    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        # 1️⃣ Best direct answer
        if data.get("AbstractText"):
            return data["AbstractText"]

        # 2️⃣ Definition
        if data.get("Definition"):
            return data["Definition"]

        # 3️⃣ Fallback to related topics
        related = data.get("RelatedTopics", [])
        if related:
            first = related[0]
            if isinstance(first, dict) and first.get("Text"):
                return first["Text"]

        return "No reliable information found on the internet."

    except Exception:
        return "Failed to fetch real-time information."
