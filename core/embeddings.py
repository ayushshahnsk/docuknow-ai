"""
Embedding utilities for DocuKnow AI - Updated for BGE-M3

Purpose:
- Load and cache BGE-M3 embedding model
- Convert text to embeddings for vector search
- Handle multilingual text (especially Hindi/Indian languages)
- Optimized for BGE-M3 specific requirements
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging
from typing import List

# Import settings
try:
    from config.settings import (
        EMBEDDING_MODEL_NAME,
        EMBEDDING_BATCH_SIZE,
        BGE_M3_NORMALIZE_EMBEDDINGS,
        BGE_M3_ENCODE_OPTIONS
    )
except ImportError:
    # Fallback defaults
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE = 8
    BGE_M3_NORMALIZE_EMBEDDINGS = True
    BGE_M3_ENCODE_OPTIONS = {
        "normalize_embeddings": True,
        "batch_size": 8,
        "show_progress_bar": False
    }

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load BGE-M3 embedding model once (cached for speed).
    
    BGE-M3 is a multilingual embedding model that:
    - Supports 100+ languages including Hindi
    - Has 1024 dimensions (better than MiniLM's 384)
    - Performs well on Indian languages
    - Requires normalize_embeddings=True for best results
    """
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        
        # Load BGE-M3 model with specific settings
        model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            trust_remote_code=True  # BGE-M3 may need this
        )
        
        # Verify model loaded correctly
        model_dimension = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: {EMBEDDING_MODEL_NAME}")
        logger.info(f"Model dimension: {model_dimension}")
        logger.info(f"Model max sequence length: {model.max_seq_length}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}")
        logger.info("Falling back to smaller multilingual model...")
        
        # Fallback to a smaller model if BGE-M3 fails
        try:
            fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            logger.info(f"Trying fallback: {fallback_model}")
            model = SentenceTransformer(fallback_model)
            logger.info(f"Fallback model loaded: {fallback_model}")
            return model
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {fallback_error}")
            raise RuntimeError(f"Could not load any embedding model. Error: {e}")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert list of texts into embeddings using BGE-M3.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (each vector is list of 1024 floats for BGE-M3)
    
    BGE-M3 Specific Notes:
    1. Always use normalize_embeddings=True
    2. Works best with batch processing
    3. Handles multilingual text natively
    4. Returns 1024-dimensional vectors
    """
    if not texts:
        return []
    
    try:
        # Load model
        model = load_embedding_model()
        
        # Log embedding request
        logger.debug(f"Embedding {len(texts)} texts")
        logger.debug(f"Sample text: {texts[0][:100]}..." if texts else "No texts")
        
        # Use BGE-M3 specific encode options
        encode_kwargs = {
            "normalize_embeddings": BGE_M3_NORMALIZE_EMBEDDINGS,
            "batch_size": min(EMBEDDING_BATCH_SIZE, len(texts)),
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        
        # Update with any additional BGE-M3 specific options
        encode_kwargs.update(BGE_M3_ENCODE_OPTIONS)
        
        # Encode texts
        embeddings = model.encode(texts, **encode_kwargs)
        
        # Log completion
        logger.debug(f"Generated {len(embeddings)} embeddings")
        if embeddings.shape[0] > 0:
            logger.debug(f"Embedding shape: {embeddings.shape}")
            logger.debug(f"Sample embedding norm: {embeddings[0].dot(embeddings[0]):.4f}")
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist()
        
        return embeddings_list
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Return empty embeddings to avoid crashing
        return [[] for _ in range(len(texts))]

def embed_single_text(text: str) -> List[float]:
    """
    Convenience function to embed a single text.
    
    Args:
        text: Single text string to embed
        
    Returns:
        Embedding vector as list of floats
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return [0.0] * 1024  # Return zero vector of correct dimension
    
    embeddings = embed_texts([text])
    return embeddings[0] if embeddings else []

def get_embedding_dimension() -> int:
    """
    Get the dimension of embeddings.
    
    Returns:
        Embedding dimension (1024 for BGE-M3)
    """
    try:
        model = load_embedding_model()
        return model.get_sentence_embedding_dimension()
    except:
        # Default BGE-M3 dimension
        return 1024

def get_model_info() -> dict:
    """
    Get information about the embedding model.
    
    Returns:
        Dictionary with model information
    """
    try:
        model = load_embedding_model()
        return {
            "name": EMBEDDING_MODEL_NAME,
            "dimension": model.get_sentence_embedding_dimension(),
            "max_sequence_length": getattr(model, 'max_seq_length', 512),
            "normalized": BGE_M3_NORMALIZE_EMBEDDINGS,
            "batch_size": EMBEDDING_BATCH_SIZE,
        }
    except Exception as e:
        return {
            "name": EMBEDDING_MODEL_NAME,
            "error": str(e),
            "dimension": 1024,  # BGE-M3 default
        }

# Test function
if __name__ == "__main__":
    # Configure logging
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    print("Testing BGE-M3 Embeddings")
    print("=" * 60)
    
    try:
        # Test model loading
        print("\n1. Loading model...")
        model_info = get_model_info()
        print(f"Model: {model_info['name']}")
        print(f"Dimension: {model_info.get('dimension', 'unknown')}")
        
        # Test embeddings
        print("\n2. Testing embeddings...")
        
        # Test texts in different languages
        test_texts = [
            "Hello, this is a test in English.",
            "नमस्ते, यह हिंदी में एक परीक्षण है।",  # Hindi
            "Bonjour, ceci est un test en français.",  # French
            "Hola, esto es una prueba en español.",  # Spanish
        ]
        
        print(f"Embedding {len(test_texts)} texts...")
        embeddings = embed_texts(test_texts)
        
        print(f"Generated {len(embeddings)} embeddings")
        if embeddings:
            print(f"First embedding length: {len(embeddings[0])}")
            print(f"First embedding sample: {embeddings[0][:5]}...")
            
            # Test similarity (BGE-M3 uses normalized embeddings, so dot product = cosine similarity)
            if len(embeddings) >= 2:
                import numpy as np
                emb1 = np.array(embeddings[0])
                emb2 = np.array(embeddings[1])
                similarity = emb1.dot(emb2)  # Since embeddings are normalized
                print(f"Similarity between text 1 and 2: {similarity:.4f}")
        
        # Test single text embedding
        print("\n3. Testing single text embedding...")
        single_embedding = embed_single_text("एकल हिंदी पाठ")  # Single Hindi text
        print(f"Single embedding length: {len(single_embedding)}")
        
        print("\n✅ BGE-M3 embedding test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()