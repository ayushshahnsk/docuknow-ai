"""
Text-to-Speech Processor for DocuKnow AI

Purpose:
- Convert AI answer text to speech audio
- Generate audio files for playback
- Handle audio caching and cleanup
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import logging
from gtts import gTTS
import base64

# Initialize logger
logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self, cache_dir: str = "data/tts_cache"):
        """
        Initialize TTS processor with cache directory.
        
        Args:
            cache_dir: Directory to cache generated audio files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store audio data in memory for quick access
        self.audio_cache = {}  # Format: {text_hash: audio_data}
        
    def text_to_audio(self, text: str, language: str = 'en') -> Optional[str]:
        """
        Convert text to audio and return base64 encoded audio data.
        
        Args:
            text: Text to convert to speech
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Base64 encoded audio data as string, or None if failed
        """
        if not text or len(text.strip()) < 5:
            logger.warning("Text too short for TTS")
            return None
        
        try:
            # Clean text (remove citations markers, etc.)
            clean_text = self._clean_text_for_tts(text)
            
            # Create hash for caching
            import hashlib
            text_hash = hashlib.md5(f"{clean_text}_{language}".encode()).hexdigest()
            
            # Check cache first
            if text_hash in self.audio_cache:
                logger.debug(f"Using cached audio for text hash: {text_hash}")
                return self.audio_cache[text_hash]
            
            # Generate TTS audio
            logger.info(f"Generating TTS audio for {len(clean_text)} characters")
            
            # Create gTTS object
            tts = gTTS(text=clean_text, lang=language, slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                tts.save(temp_path)
            
            # Read audio file and encode to base64
            with open(temp_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Cache the audio
            self.audio_cache[text_hash] = audio_base64
            
            # Limit cache size (keep last 50 audio files)
            if len(self.audio_cache) > 50:
                oldest_key = next(iter(self.audio_cache))
                del self.audio_cache[oldest_key]
            
            return audio_base64
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean text for better TTS results.
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text
        """
        # Remove markdown formatting
        import re
        
        # Remove citations markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit length (gTTS has ~5000 character limit)
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text.strip()
    
    def clear_cache(self):
        """Clear audio cache."""
        self.audio_cache.clear()
        logger.info("TTS cache cleared")