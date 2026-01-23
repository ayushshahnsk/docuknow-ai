"""
Caching utilities for DocuKnow AI.

Purpose:
- Avoid recomputing embeddings
- Speed up repeated document processing
- Improve overall response time

This cache is in-memory and safe for Streamlit apps.
"""

import hashlib
from functools import lru_cache
from typing import List


def _hash_texts(texts: List[str]) -> str:
    """
    Create a stable hash for a list of texts.
    Used as cache key.
    """
    joined = "||".join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def cached_embeddings(texts_hash: str, texts_tuple: tuple):
    """
    Cache embeddings using text hash.

    texts_tuple is required because lru_cache
    only accepts hashable arguments.
    """
    from core.embeddings import embed_texts

    return embed_texts(list(texts_tuple))


def get_embeddings_with_cache(texts: List[str]):
    """
    Public function to get embeddings with caching.
    """
    texts_hash = _hash_texts(texts)
    return cached_embeddings(texts_hash, tuple(texts))
