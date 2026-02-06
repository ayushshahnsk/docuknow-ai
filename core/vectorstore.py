"""
Vector Store for DocuKnow AI - Updated for BGE-M3

Purpose:
- Create and manage FAISS vector indices
- Handle 1024-dimensional embeddings from BGE-M3
- Store metadata with embeddings
- Perform similarity search
"""

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import os

# Import embedding dimension from settings
try:
    from config.settings import EMBEDDING_DIMENSION, VECTOR_DB_DIR
except ImportError:
    # Default values for BGE-M3
    EMBEDDING_DIMENSION = 1024  # BGE-M3 has 1024 dimensions
    VECTOR_DB_DIR = Path("data/vector_db")

logger = logging.getLogger(__name__)

# Convert string path to Path object if needed
if isinstance(VECTOR_DB_DIR, str):
    VECTOR_DB_DIR = Path(VECTOR_DB_DIR)

def create_faiss_index(
    embeddings: List[List[float]], 
    metadatas: List[Dict], 
    index_name: str
) -> Tuple[bool, str]:
    """
    Create and save FAISS index with metadata - Updated for BGE-M3.
    
    Args:
        embeddings: List of embedding vectors (1024 dim for BGE-M3)
        metadatas: List of metadata dictionaries for each embedding
        index_name: Name to save the index as
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Ensure vector DB directory exists
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        if not embeddings:
            return False, "No embeddings provided"
        
        if len(embeddings) != len(metadatas):
            return False, f"Mismatch: {len(embeddings)} embeddings vs {len(metadatas)} metadatas"
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Verify embedding dimension
        actual_dim = embeddings_array.shape[1]
        if actual_dim != EMBEDDING_DIMENSION:
            logger.warning(f"Embedding dimension mismatch: Expected {EMBEDDING_DIMENSION}, got {actual_dim}")
        
        # Create FAISS index
        # BGE-M3 uses normalized embeddings, so we use IndexFlatIP (Inner Product)
        # For normalized vectors, dot product = cosine similarity
        dimension = actual_dim
        index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        # Save the index
        index_path = VECTOR_DB_DIR / f"{index_name}.index"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = VECTOR_DB_DIR / f"{index_name}.meta"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadatas, f)
        
        # Save dimension info
        info_path = VECTOR_DB_DIR / f"{index_name}.info"
        with open(info_path, "w") as f:
            f.write(f"dimension:{dimension}\n")
            f.write(f"count:{len(embeddings)}\n")
            f.write(f"model:BGE-M3\n")
        
        logger.info(f"Created FAISS index: {index_name} with {len(embeddings)} vectors (dim={dimension})")
        
        return True, f"Index created with {len(embeddings)} vectors"
        
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {e}")
        return False, f"Error creating index: {str(e)}"

def load_faiss_index(index_name: str) -> Tuple[Optional[faiss.Index], List[Dict]]:
    """
    Load FAISS index and metadata - Updated for BGE-M3 compatibility.
    
    Args:
        index_name: Name of the index to load
        
    Returns:
        Tuple of (FAISS index, metadata list)
        Returns (None, []) if loading fails
    """
    try:
        index_path = VECTOR_DB_DIR / f"{index_name}.index"
        metadata_path = VECTOR_DB_DIR / f"{index_name}.meta"
        
        if not index_path.exists():
            logger.error(f"Index file not found: {index_path}")
            return None, []
        
        if not metadata_path.exists():
            logger.error(f"Metadata file not found: {metadata_path}")
            return None, []
        
        # Load the index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, "rb") as f:
            metadatas = pickle.load(f)
        
        # Verify index dimension
        index_dim = index.d
        logger.info(f"Loaded FAISS index: {index_name} (dim={index_dim}, vectors={index.ntotal})")
        
        return index, metadatas
        
    except Exception as e:
        logger.error(f"Failed to load FAISS index {index_name}: {e}")
        return None, []

def search_index(
    query_vector: np.ndarray, 
    index_name: str, 
    top_k: int = 4
) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[List[Dict]]]:
    """
    Search FAISS index for similar vectors - Updated for BGE-M3.
    
    Args:
        query_vector: Query embedding vector (1024 dim)
        index_name: Name of index to search
        top_k: Number of results to return
        
    Returns:
        Tuple of (scores, indices, metadatas) or (None, None, None) if failed
    """
    try:
        # Load index and metadata
        index, metadatas = load_faiss_index(index_name)
        
        if index is None:
            return None, None, None
        
        # Ensure query_vector is 2D (1, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Ensure correct dtype
        query_vector = query_vector.astype(np.float32)
        
        # Verify dimension matches
        if query_vector.shape[1] != index.d:
            logger.warning(f"Dimension mismatch: Query {query_vector.shape[1]} vs Index {index.d}")
            # Try to reshape if possible
            if query_vector.shape[1] > index.d:
                query_vector = query_vector[:, :index.d]
            elif query_vector.shape[1] < index.d:
                # Pad with zeros (not ideal but works)
                padded = np.zeros((1, index.d), dtype=np.float32)
                padded[:, :query_vector.shape[1]] = query_vector
                query_vector = padded
        
        # Search the index
        # For IndexFlatIP (inner product), higher scores are better
        scores, indices = index.search(query_vector, top_k)
        
        # Convert to lists
        scores_list = scores[0].tolist()
        indices_list = indices[0].tolist()
        
        # Get corresponding metadata
        results_metadata = []
        for idx in indices_list:
            if idx >= 0 and idx < len(metadatas):  # Valid index
                results_metadata.append(metadatas[idx])
            else:
                results_metadata.append({})
        
        logger.debug(f"Search found {len([i for i in indices_list if i >= 0])} results")
        
        return scores_list, indices_list, results_metadata
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None, None, None

def get_index_info(index_name: str) -> Dict:
    """
    Get information about a FAISS index.
    
    Args:
        index_name: Name of the index
        
    Returns:
        Dictionary with index information
    """
    try:
        index_path = VECTOR_DB_DIR / f"{index_name}.index"
        metadata_path = VECTOR_DB_DIR / f"{index_name}.meta"
        info_path = VECTOR_DB_DIR / f"{index_name}.info"
        
        info = {
            "exists": False,
            "name": index_name,
            "vector_count": 0,
            "dimension": 0,
            "metadata_count": 0,
        }
        
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            info.update({
                "exists": True,
                "vector_count": index.ntotal,
                "dimension": index.d,
                "index_type": "FlatIP" if isinstance(index, faiss.IndexFlatIP) else "Unknown",
            })
        
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadatas = pickle.load(f)
            info["metadata_count"] = len(metadatas)
        
        if info_path.exists():
            with open(info_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        info[key] = value
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get index info: {e}")
        return {"exists": False, "error": str(e)}

def list_all_indices() -> List[Dict]:
    """
    List all available FAISS indices.
    
    Returns:
        List of index information dictionaries
    """
    indices = []
    
    if not VECTOR_DB_DIR.exists():
        return indices
    
    for file in VECTOR_DB_DIR.glob("*.index"):
        index_name = file.stem
        info = get_index_info(index_name)
        indices.append(info)
    
    return indices

def delete_index(index_name: str) -> bool:
    """
    Delete a FAISS index and its metadata.
    
    Args:
        index_name: Name of index to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        files = [
            VECTOR_DB_DIR / f"{index_name}.index",
            VECTOR_DB_DIR / f"{index_name}.meta",
            VECTOR_DB_DIR / f"{index_name}.info",
        ]
        
        deleted_count = 0
        for file in files:
            if file.exists():
                file.unlink()
                deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} files for index: {index_name}")
        return deleted_count > 0
        
    except Exception as e:
        logger.error(f"Failed to delete index {index_name}: {e}")
        return False

def clear_all_indices() -> bool:
    """
    Delete all FAISS indices.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not VECTOR_DB_DIR.exists():
            return True
        
        deleted_count = 0
        for file in VECTOR_DB_DIR.glob("*"):
            if file.is_file():
                file.unlink()
                deleted_count += 1
        
        logger.info(f"Cleared all indices: deleted {deleted_count} files")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear indices: {e}")
        return False

def validate_embeddings_for_index(embeddings: List[List[float]]) -> Tuple[bool, str]:
    """
    Validate embeddings before indexing.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if not embeddings:
        return False, "No embeddings provided"
    
    # Check all embeddings have same dimension
    dimensions = [len(emb) for emb in embeddings]
    if len(set(dimensions)) > 1:
        return False, f"Inconsistent embedding dimensions: {set(dimensions)}"
    
    # Check dimension matches expected
    actual_dim = dimensions[0]
    if actual_dim != EMBEDDING_DIMENSION:
        logger.warning(f"Embedding dimension {actual_dim} doesn't match expected {EMBEDDING_DIMENSION}")
        # Don't fail, just warn
    
    # Check for NaN or infinite values
    for i, emb in enumerate(embeddings):
        emb_array = np.array(emb, dtype=np.float32)
        if np.any(np.isnan(emb_array)):
            return False, f"NaN values in embedding {i}"
        if np.any(np.isinf(emb_array)):
            return False, f"Infinite values in embedding {i}"
    
    return True, f"Validated {len(embeddings)} embeddings (dim={actual_dim})"

# Test function
if __name__ == "__main__":
    # Configure logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    print("Testing BGE-M3 Vector Store")
    print("=" * 60)
    
    try:
        # Create test embeddings (1024-dimensional like BGE-M3)
        print("\n1. Creating test embeddings...")
        test_embeddings = []
        test_metadatas = []
        
        for i in range(5):
            # Create random 1024-dim embedding (normalized)
            emb = np.random.randn(1024).astype(np.float32)
            emb = emb / np.linalg.norm(emb)  # Normalize
            test_embeddings.append(emb.tolist())
            
            test_metadatas.append({
                "text": f"Test text {i}",
                "page": i + 1,
                "source": "test.pdf",
                "language": "hi" if i % 2 == 0 else "en"
            })
        
        # Create index
        print("\n2. Creating FAISS index...")
        success, message = create_faiss_index(
            test_embeddings, 
            test_metadatas, 
            "test_index_bge"
        )
        print(f"Create result: {success} - {message}")
        
        # Get index info
        print("\n3. Getting index info...")
        info = get_index_info("test_index_bge")
        print(f"Index info: {info}")
        
        # Search test
        print("\n4. Testing search...")
        query_emb = test_embeddings[0]  # Use first embedding as query
        scores, indices, metadatas = search_index(
            np.array(query_emb), 
            "test_index_bge", 
            top_k=3
        )
        
        if scores and indices:
            print(f"Search successful!")
            print(f"Scores: {scores}")
            print(f"Indices: {indices}")
            for i, (score, idx, meta) in enumerate(zip(scores, indices, metadatas)):
                print(f"Result {i+1}: Score={score:.4f}, Index={idx}, Meta={meta.get('text', 'N/A')}")
        
        # List all indices
        print("\n5. Listing all indices...")
        indices_list = list_all_indices()
        print(f"Found {len(indices_list)} indices:")
        for idx_info in indices_list:
            print(f"  - {idx_info['name']}: {idx_info['vector_count']} vectors")
        
        # Cleanup
        print("\n6. Cleaning up...")
        delete_index("test_index_bge")
        print("Test index deleted")
        
        print("\n✅ BGE-M3 vector store test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()