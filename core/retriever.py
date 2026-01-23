from typing import List, Dict
import numpy as np

from core.embeddings import embed_texts
from core.vectorstore import load_faiss_index


def retrieve_context(query: str, index_name: str, top_k: int = 4) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query.
    """
    # 1️⃣ Load vector index + metadata
    index, metadatas = load_faiss_index(index_name)

    # 2️⃣ Embed query
    query_embedding = embed_texts([query])
    query_vector = np.array(query_embedding).astype("float32")

    # 3️⃣ Search FAISS
    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = metadatas[idx]

        results.append(
            {
                "text": chunk["text"],
                "page": chunk.get("page"),
                "source": chunk.get("source"),
                "score": float(score),
            }
        )

    return results
