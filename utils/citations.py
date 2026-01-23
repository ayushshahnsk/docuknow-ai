from typing import List, Dict


def format_citations(contexts: List[Dict]) -> List[str]:
    """
    Create clean citation strings from retrieved contexts.
    """
    seen = set()
    citations = []

    for ctx in contexts:
        source = ctx.get("source", "Unknown")
        page = ctx.get("page", "N/A")

        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append(f"{source} — Page {page}")

    return citations
