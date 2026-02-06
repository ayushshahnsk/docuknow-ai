"""
Central configuration for DocuKnow AI - MULTILINGUAL EDITION

All tunable parameters live here:
- Chunking
- Retrieval
- Embeddings
- LLM
- Confidence thresholds
- Multilingual settings

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
# Embedding Model (MULTILINGUAL) - UPDATED TO BGE-M3
# -----------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# BGE-M3 Advantages:
# 1. Excellent Hindi/Indian language support
# 2. 1024 dimensions (better than 384/768)
# 3. Supports dense, sparse, and multi-vector retrieval
# 4. Trained on diverse multilingual data
# Alternatives if BGE-M3 too large:
# - "BAAI/bge-small-en-v1.5" (English only)
# - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIMENSION = 1024  # BGE-M3 has 1024 dimensions

# -----------------------------
# Vector Database
# -----------------------------
VECTOR_DB_DIR = "data/vector_db"

# -----------------------------
# LLM (Ollama) - MULTILINGUAL OPTIONS
# -----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL_NAME = "qwen2.5:7b"  # Good multilingual support
# Alternatives: 
# - "llama3.2:3b" (lightweight, decent multilingual)
# - "mistral:7b" (good for European languages)
# - "gemma2:2b" (lightweight but less multilingual)
# - "nomic-embed-text" (for embeddings)
LLM_TIMEOUT = 60  # seconds

# -----------------------------
# Confidence Thresholds (OPTIMIZED FOR BGE-M3 + INDIAN LANGUAGES)
# -----------------------------
# BGE-M3 similarity scores range differently
CONFIDENCE_HIGH = 0.65      # Lowered for BGE-M3
CONFIDENCE_MEDIUM = 0.45    # Lowered for BGE-M3
CONFIDENCE_LOW = 0.25       # Added for clarity

# -----------------------------
# UI / Performance
# -----------------------------
ENABLE_CACHING = True
MAX_CHAT_HISTORY = 10  # prevent prompt from growing too large

# -----------------------------
# OCR Settings
# -----------------------------
ENABLE_OCR = True  # Master switch for OCR
OCR_LANGUAGES = ['en', 'hi']  # Added Hindi for OCR
OCR_USE_GPU = False  # Use GPU for OCR if available
OCR_TEXT_THRESHOLD = 0.1  # Minimum text ratio to avoid OCR (10%)

# PDF Detection
PDF_MIN_TEXT_LENGTH = 50  # Minimum characters to consider as text page

# -----------------------------
# Search Configuration (MULTILINGUAL)
# -----------------------------
SEARCH_PROVIDER = "hybrid"  # "duckduckgo", "wikipedia", "hybrid", "tavily"
SEARCH_TIMEOUT = 10
MAX_SEARCH_RESULTS = 3

# -----------------------------
# Language Configuration (MULTILINGUAL SUPPORT)
# -----------------------------
SUPPORTED_LANGUAGES = [
    "en",  # English
    "hi",  # Hindi
    "mr",  # Marathi
    "gu",  # Gujarati
    "fr",  # French
    "bn",  # Bengali
    "de",  # German
    "zh",  # Chinese
    "es",  # Spanish
    "ta",  # Tamil
    "te",  # Telugu
    "kn",  # Kannada
    "ml",  # Malayalam
    "pa",  # Punjabi
    "ur",  # Urdu
    "ar",  # Arabic
    "ja",  # Japanese
    "ko",  # Korean
    "ru",  # Russian
    "pt",  # Portuguese
    "it",  # Italian
]

# Language-specific settings
LANGUAGE_MODELS = {
    "en": "qwen2.5:7b",      # English
    "hi": "qwen2.5:7b",      # Hindi
    "mr": "qwen2.5:7b",      # Marathi
    "gu": "qwen2.5:7b",      # Gujarati
    "fr": "mistral:7b",      # French
    "bn": "qwen2.5:7b",      # Bengali
    "de": "mistral:7b",      # German
    "zh": "qwen2.5:7b",      # Chinese
    "es": "mistral:7b",      # Spanish
    "ta": "qwen2.5:7b",      # Tamil
    "te": "qwen2.5:7b",      # Telugu
    "kn": "qwen2.5:7b",      # Kannada
    "ml": "qwen2.5:7b",      # Malayalam
    "pa": "qwen2.5:7b",      # Punjabi
    "ur": "qwen2.5:7b",      # Urdu
    "ar": "qwen2.5:7b",      # Arabic
    "ja": "qwen2.5:7b",      # Japanese
    "ko": "qwen2.5:7b",      # Korean
    "ru": "llama3.2:3b",     # Russian
    "pt": "mistral:7b",      # Portuguese
    "it": "mistral:7b",      # Italian
    "default": "qwen2.5:7b"  # Fallback
}

# Language names for display
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "gu": "Gujarati",
    "fr": "French",
    "bn": "Bengali",
    "de": "German",
    "zh": "Chinese",
    "es": "Spanish",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ar": "Arabic",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# -----------------------------
# Translation Settings
# -----------------------------
ENABLE_TRANSLATION = True
TRANSLATION_CACHE_SIZE = 100
DEFAULT_TRANSLATION_SOURCE = "google"  # "google", "microsoft", "deepl"

# -----------------------------
# TTS Settings (Multilingual)
# -----------------------------
TTS_LANGUAGES = {
    "en": "en",      # English
    "hi": "hi",      # Hindi
    "mr": "mr",      # Marathi (may not be available in all TTS engines)
    "fr": "fr",      # French
    "de": "de",      # German
    "es": "es",      # Spanish
    "zh": "zh",      # Chinese
    "ja": "ja",      # Japanese
    "ko": "ko",      # Korean
    "ar": "ar",      # Arabic
    "default": "en"  # Fallback
}

# -----------------------------
# Performance Optimization
# -----------------------------
EMBEDDING_BATCH_SIZE = 8  # Reduced for BGE-M3 (larger model)
MAX_DOCUMENT_SIZE_MB = 50  # Maximum PDF size to process
ENABLE_PARALLEL_PROCESSING = True

# -----------------------------
# Debug/Development Settings
# -----------------------------
DEBUG_MODE = False
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SAVE_RAW_CONTEXTS = False  # For debugging retrieval issues

# -----------------------------
# Export/Import Settings
# -----------------------------
EXPORT_FORMATS = ["json", "csv", "txt"]
MAX_EXPORT_RECORDS = 1000

# -----------------------------
# Security Settings
# -----------------------------
MAX_FILE_UPLOADS_PER_SESSION = 10
ALLOWED_FILE_TYPES = [".pdf", ".txt", ".docx", ".pptx"]
MAX_QUERY_LENGTH = 1000

# -----------------------------
# BGE-M3 Specific Settings
# -----------------------------
# BGE-M3 works best with these settings
BGE_M3_NORMALIZE_EMBEDDINGS = True  # Must be True
BGE_M3_ENCODE_OPTIONS = {
    "normalize_embeddings": True,
    "batch_size": 8,
    "show_progress_bar": False
}