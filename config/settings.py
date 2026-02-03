"""
Central configuration for DocuKnow AI.

All tunable parameters live here:
- Chunking
- Retrieval
- Embeddings
- LLM
- Confidence thresholds

This avoids magic numbers spread across the codebase
and makes experimentation easy.
"""

# -----------------------------
# Chunking Settings
# -----------------------------
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlapping characters between chunks

# -----------------------------
# Retrieval Settings
# -----------------------------
TOP_K = 4  # number of chunks retrieved per query

# -----------------------------
# Embedding Model
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# Vector Database
# -----------------------------
VECTOR_DB_DIR = "data/vector_db"

# -----------------------------
# LLM (Ollama)
# -----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL_NAME = "gemma3:4b"
LLM_TIMEOUT = 60  # seconds

# -----------------------------
# Confidence Thresholds
# -----------------------------
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.55

# -----------------------------
# UI / Performance
# -----------------------------
ENABLE_CACHING = True
MAX_CHAT_HISTORY = 10  # prevent prompt from growing too large

# -----------------------------
# OCR Settings
# -----------------------------
ENABLE_OCR = True  # Master switch for OCR
OCR_LANGUAGES = ['en']  # Languages for OCR
OCR_USE_GPU = False  # Use GPU for OCR if available
OCR_TEXT_THRESHOLD = 0.1  # Minimum text ratio to avoid OCR (10%)

# PDF Detection
PDF_MIN_TEXT_LENGTH = 50  # Minimum characters to consider as text page