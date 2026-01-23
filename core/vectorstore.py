import faiss
import pickle
from pathlib import Path
from typing import List, Dict

VECTOR_DB_PATH = Path("data/vector_db")


def create_faiss_index(
    embeddings: List[list[float]], metadatas: List[Dict], index_name: str
):
    """
    Create and save FAISS index with metadata.
    """
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(VECTOR_DB_PATH / f"{index_name}.index"))

    with open(VECTOR_DB_PATH / f"{index_name}.meta", "wb") as f:
        pickle.dump(metadatas, f)


def load_faiss_index(index_name: str):
    """
    Load FAISS index and metadata.
    """
    index = faiss.read_index(str(VECTOR_DB_PATH / f"{index_name}.index"))

    with open(VECTOR_DB_PATH / f"{index_name}.meta", "rb") as f:
        metadatas = pickle.load(f)

    return index, metadatas
