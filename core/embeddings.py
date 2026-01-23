from sentence_transformers import SentenceTransformer
from functools import lru_cache


@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load embedding model once (cached for speed).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert list of texts into embeddings.
    """
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings
