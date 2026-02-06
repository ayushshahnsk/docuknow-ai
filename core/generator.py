"""
LLM Generator for DocuKnow AI - MULTILINGUAL EDITION

This module:
- Builds strict RAG prompts with multilingual support
- Injects conversational memory
- Calls appropriate LLM models based on language
- Prevents hallucinations
- Handles language-specific formatting
- Manages model selection intelligently

Key Features:
1. Language-aware model selection
2. Multilingual prompt templates
3. Conversation memory with language context
4. Fallback mechanisms for model failures
5. Token tracking and optimization
"""

import requests
import logging
from typing import List, Dict, Optional, Tuple
import json
import time
from functools import lru_cache

from config.prompts import build_rag_prompt
from config.settings import (
    OLLAMA_BASE_URL, 
    LLM_MODEL_NAME, 
    LLM_TIMEOUT,
    LANGUAGE_MODELS,
    SUPPORTED_LANGUAGES
)
from memory.conversation import ConversationMemory
from utils.language_detector import detect_query_language, get_language_name

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages LLM models for different languages.
    Handles model selection, fallbacks, and health checks.
    """
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.available_models = {}
        self.model_health = {}
        self._last_health_check = 0
        
    def check_model_health(self, model_name: str) -> bool:
        """
        Check if a model is available and healthy.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        # Cache health checks for 5 minutes
        current_time = time.time()
        if (model_name in self.model_health and 
            current_time - self.model_health[model_name]["timestamp"] < 300):
            return self.model_health[model_name]["healthy"]
        
        try:
            # Try to pull model info
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            healthy = response.status_code == 200
            self.model_health[model_name] = {
                "healthy": healthy,
                "timestamp": current_time
            }
            
            logger.debug(f"Model health check: {model_name} = {healthy}")
            return healthy
            
        except Exception as e:
            logger.warning(f"Health check failed for {model_name}: {e}")
            self.model_health[model_name] = {
                "healthy": False,
                "timestamp": current_time,
                "error": str(e)
            }
            return False
    
    def get_available_models(self) -> Dict[str, bool]:
        """
        Get list of available models.
        
        Returns:
            Dictionary of model_name -> availability
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.available_models = {
                    model["name"]: True 
                    for model in data.get("models", [])
                }
                return self.available_models
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
        
        return {}
    
    def select_model_for_language(self, lang: str, query: str = "") -> str:
        """
        Select the best model for a given language and query.
        
        Args:
            lang: Language code
            query: Optional query for context
            
        Returns:
            Selected model name
        """
        # Default model
        default_model = LLM_MODEL_NAME
        
        # Check language-specific model
        language_model = LANGUAGE_MODELS.get(lang, LANGUAGE_MODELS["default"])
        
        # Special cases based on query content
        if query:
            query_lower = query.lower()
            
            # For code or technical queries, prefer certain models
            if any(word in query_lower for word in ["code", "program", "function", "algorithm", "technical"]):
                if self.check_model_health("codellama:7b"):
                    return "codellama:7b"
            
            # For creative writing
            if any(word in query_lower for word in ["story", "poem", "creative", "write a"]):
                if self.check_model_health("mistral:7b"):
                    return "mistral:7b"
        
        # Check if language-specific model is available
        if self.check_model_health(language_model):
            return language_model
        
        # Fallback to default model
        if self.check_model_health(default_model):
            return default_model
        
        # Last resort: try any available model
        available = self.get_available_models()
        if available:
            return next(iter(available))
        
        # No models available
        raise RuntimeError("No LLM models available. Please ensure Ollama is running and models are pulled.")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
        
        return None

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

@lru_cache(maxsize=100)
def _call_llm_cached(
    model: str, 
    prompt: str, 
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> Optional[str]:
    """
    Cached LLM call for identical prompts.
    
    Args:
        model: Model name
        prompt: Prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text or None
    """
    return _call_llm_uncached(model, prompt, temperature, max_tokens)

def _call_llm_uncached(
    model: str, 
    prompt: str, 
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> Optional[str]:
    """
    Uncached LLM call.
    
    Args:
        model: Model name
        prompt: Prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text or None
    """
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": max_tokens,
                "repeat_penalty": 1.1,
                "stop": ["\n\n", "User:", "Assistant:", "###", "Question:"]
            }
        }
        
        logger.debug(f"Calling LLM: {model}, prompt length: {len(prompt)}")
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=LLM_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract response
            answer = data.get("response", "").strip()
            
            # Log token usage if available
            if "eval_count" in data:
                logger.debug(f"LLM generated {data.get('eval_count')} tokens")
            
            return answer
        
        else:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"LLM request timeout for model: {model}")
        return None
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to Ollama at {OLLAMA_BASE_URL}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error in LLM call: {e}")
        return None

def call_llm(
    model: str, 
    prompt: str, 
    temperature: float = 0.3,
    max_tokens: int = 2000,
    use_cache: bool = True
) -> Optional[str]:
    """
    Call LLM with optional caching.
    
    Args:
        model: Model name
        prompt: Prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        use_cache: Whether to use response caching
        
    Returns:
        Generated text or None
    """
    if use_cache and len(prompt) < 1000:  # Only cache shorter prompts
        return _call_llm_cached(model, prompt, temperature, max_tokens)
    else:
        return _call_llm_uncached(model, prompt, temperature, max_tokens)

def format_conversation_history(
    memory: Optional[ConversationMemory], 
    lang: str = "en"
) -> str:
    """
    Format conversation history for prompt with language context.
    
    Args:
        memory: Conversation memory object
        lang: Language code for formatting
        
    Returns:
        Formatted conversation history string
    """
    if not memory or not memory.history:
        return ""
    
    # Language-specific formatting
    lang_formats = {
        "en": {
            "user_prefix": "User",
            "assistant_prefix": "Assistant"
        },
        "hi": {
            "user_prefix": "उपयोगकर्ता",
            "assistant_prefix": "सहायक"
        },
        "fr": {
            "user_prefix": "Utilisateur",
            "assistant_prefix": "Assistant"
        },
        "es": {
            "user_prefix": "Usuario",
            "assistant_prefix": "Asistente"
        },
        "de": {
            "user_prefix": "Benutzer",
            "assistant_prefix": "Assistent"
        },
        "zh": {
            "user_prefix": "用户",
            "assistant_prefix": "助手"
        },
        "ja": {
            "user_prefix": "ユーザー",
            "assistant_prefix": "アシスタント"
        },
    }
    
    format_info = lang_formats.get(lang, lang_formats["en"])
    
    formatted = []
    for turn in memory.history:
        role_prefix = format_info["user_prefix"] if turn["role"] == "user" else format_info["assistant_prefix"]
        formatted.append(f"{role_prefix}: {turn['content']}")
    
    return "\n".join(formatted)

def build_final_prompt(
    query: str, 
    contexts: List[Dict], 
    memory: Optional[ConversationMemory] = None,
    lang: str = "en"
) -> str:
    """
    Build final prompt for LLM with all components.
    
    Args:
        query: User query
        contexts: Retrieved contexts
        memory: Conversation memory
        lang: Language code
        
    Returns:
        Complete prompt string
    """
    # Build base RAG prompt
    base_prompt = build_rag_prompt(query, contexts, lang=lang)
    
    # Add conversation history if available
    if memory and memory.history:
        history_block = format_conversation_history(memory, lang)
        
        # Language-specific history instructions
        history_instructions = {
            "en": "Consider the following conversation history when answering:",
            "hi": "उत्तर देते समय निम्नलिखित वार्तालाप इतिहास पर विचार करें:",
            "fr": "Considérez l'historique de conversation suivant lorsque vous répondez:",
            "es": "Considere el siguiente historial de conversación al responder:",
            "de": "Berücksichtigen Sie beim Antworten den folgenden Gesprächsverlauf:",
            "zh": "回答时请考虑以下对话历史：",
            "ja": "回答する際は、次の会話履歴を考慮してください：",
        }
        
        instruction = history_instructions.get(lang, history_instructions["en"])
        
        final_prompt = f"{instruction}\n\n{history_block}\n\n{base_prompt}"
    else:
        final_prompt = base_prompt
    
    # Add language instruction at the end for reinforcement
    lang_reminder = {
        "en": "Remember to respond in English.",
        "hi": "हिंदी में उत्तर देना याद रखें।",
        "fr": "N'oubliez pas de répondre en français.",
        "es": "Recuerde responder en español.",
        "de": "Denken Sie daran, auf Deutsch zu antworten.",
        "zh": "记得用中文回答。",
        "ja": "日本語で回答することを忘れないでください。",
    }
    
    reminder = lang_reminder.get(lang, lang_reminder["en"])
    final_prompt = f"{final_prompt}\n\n{reminder}"
    
    return final_prompt.strip()

def generate_answer(
    query: str, 
    contexts: List[Dict], 
    memory: Optional[ConversationMemory] = None,
    lang: str = "en",
    temperature: float = 0.3,
    max_retries: int = 2
) -> str:
    """
    Generate an answer using appropriate LLM with RAG + optional conversation memory.
    
    Args:
        query: User query
        contexts: Retrieved contexts from documents
        memory: Conversation memory for context
        lang: Language code for response
        temperature: Sampling temperature (0.1-1.0)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Generated answer as string
    """
    # Validate inputs
    if not query or not isinstance(query, str):
        return "Error: Invalid query provided."
    
    # Detect language if not provided
    if lang == "auto" or not lang:
        lang_info = detect_query_language(query)
        lang = lang_info.get("language", "en")
    
    # Ensure lang is supported
    if lang not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {lang}, defaulting to English")
        lang = "en"
    
    # Get model manager
    model_manager = get_model_manager()
    
    # Select appropriate model
    try:
        model_name = model_manager.select_model_for_language(lang, query)
        logger.info(f"Selected model '{model_name}' for language '{lang}' ({get_language_name(lang)})")
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        return f"Error: Unable to select language model. Please ensure Ollama is running with appropriate models."
    
    # Build final prompt
    final_prompt = build_final_prompt(query, contexts, memory, lang)
    
    # Generate answer with retries
    answer = None
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Adjust temperature based on attempt
            current_temp = temperature * (1.2 if attempt > 0 else 1.0)  # Slightly increase on retry
            
            # Call LLM
            answer = call_llm(
                model=model_name,
                prompt=final_prompt,
                temperature=current_temp,
                max_tokens=2000,
                use_cache=True
            )
            
            if answer:
                # Clean up the answer
                answer = answer.strip()
                
                # Remove any prompt leakage
                if "FINAL ANSWER:" in answer:
                    answer = answer.split("FINAL ANSWER:")[-1].strip()
                
                # Remove language reminder if it appears
                lang_reminders = [
                    "Remember to respond in English.",
                    "हिंदी में उत्तर देना याद रखें।",
                    "N'oubliez pas de répondre en français.",
                    "Recuerde responder en español.",
                    "Denken Sie daran, auf Deutsch zu antworten.",
                    "记得用中文回答。",
                    "日本語で回答することを忘れないでください。",
                ]
                
                for reminder in lang_reminders:
                    if reminder in answer:
                        answer = answer.replace(reminder, "").strip()
                
                # Update conversation memory
                if memory:
                    memory.add_turn(query, answer)
                
                logger.info(f"Successfully generated answer ({len(answer)} chars) in {lang}")
                return answer
                
            else:
                last_error = "LLM returned empty response"
                
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            # Try a different model on retry
            if attempt < max_retries:
                # Try fallback models
                fallback_models = ["llama3.2:3b", "mistral:7b", "gemma2:2b", LLM_MODEL_NAME]
                
                for fallback in fallback_models:
                    if fallback != model_name and model_manager.check_model_health(fallback):
                        logger.info(f"Trying fallback model: {fallback}")
                        model_name = fallback
                        break
                
                # Wait before retry
                time.sleep(1 * (attempt + 1))
    
    # All attempts failed
    error_msg = f"Error: Failed to generate answer after {max_retries + 1} attempts."
    if last_error:
        error_msg += f" Last error: {last_error}"
    
    logger.error(error_msg)
    return error_msg

def generate_answer_with_fallback(
    query: str, 
    contexts: List[Dict], 
    memory: Optional[ConversationMemory] = None,
    lang: str = "en"
) -> Tuple[str, Dict]:
    """
    Generate answer with detailed information about the generation process.
    
    Args:
        query: User query
        contexts: Retrieved contexts
        memory: Conversation memory
        lang: Language code
        
    Returns:
        Tuple of (answer_string, metadata_dict)
    """
    start_time = time.time()
    
    # Get model info before generation
    model_manager = get_model_manager()
    selected_model = model_manager.select_model_for_language(lang, query)
    model_info = model_manager.get_model_info(selected_model) or {}
    
    # Generate answer
    answer = generate_answer(query, contexts, memory, lang)
    
    # Calculate metrics
    generation_time = time.time() - start_time
    
    # Build metadata
    metadata = {
        "model_used": selected_model,
        "language": lang,
        "language_name": get_language_name(lang),
        "generation_time_seconds": round(generation_time, 2),
        "context_count": len(contexts) if contexts else 0,
        "model_details": {
            "name": model_info.get("name", selected_model),
            "size": model_info.get("size", "unknown"),
            "modified_at": model_info.get("modified_at", "unknown"),
        } if model_info else {},
        "has_memory": memory is not None and len(memory.history) > 0,
        "answer_length": len(answer),
        "timestamp": time.time()
    }
    
    return answer, metadata

def batch_generate_answers(
    queries: List[str],
    contexts_list: List[List[Dict]],
    langs: Optional[List[str]] = None,
    memory: Optional[ConversationMemory] = None
) -> List[str]:
    """
    Generate answers for multiple queries efficiently.
    
    Args:
        queries: List of queries
        contexts_list: List of context lists for each query
        langs: Optional list of language codes
        memory: Conversation memory
        
    Returns:
        List of answers
    """
    if langs is None:
        langs = ["auto"] * len(queries)
    
    if len(queries) != len(contexts_list) or len(queries) != len(langs):
        raise ValueError("All input lists must have the same length")
    
    answers = []
    
    for i, (query, contexts, lang) in enumerate(zip(queries, contexts_list, langs)):
        try:
            answer = generate_answer(query, contexts, memory, lang)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Failed to generate answer for query {i}: {e}")
            answers.append(f"Error generating answer: {str(e)}")
        
        # Small delay to avoid overwhelming the LLM
        if i < len(queries) - 1:
            time.sleep(0.5)
    
    return answers

def get_generation_stats() -> Dict:
    """
    Get statistics about the generation system.
    
    Returns:
        Dictionary with statistics
    """
    model_manager = get_model_manager()
    
    stats = {
        "model_manager": {
            "available_models": list(model_manager.available_models.keys()),
            "model_health": model_manager.model_health,
        },
        "default_settings": {
            "base_url": OLLAMA_BASE_URL,
            "default_model": LLM_MODEL_NAME,
            "timeout": LLM_TIMEOUT,
            "supported_languages": len(SUPPORTED_LANGUAGES),
        },
        "cache_info": {
            "cache_size": _call_llm_cached.cache_info().currsize,
            "cache_hits": _call_llm_cached.cache_info().hits,
            "cache_misses": _call_llm_cached.cache_info().misses,
        }
    }
    
    return stats

# Test function
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing LLM Generator - Multilingual Edition")
    print("=" * 60)
    
    # Test contexts
    test_contexts = [
        {
            "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
            "page": 1,
            "source": "AI Basics.pdf"
        }
    ]
    
    # Test queries in different languages
    test_queries = [
        ("What is Artificial Intelligence?", "en"),
        ("कृत्रिम बुद्धिमत्ता क्या है?", "hi"),
        ("Qu'est-ce que l'intelligence artificielle?", "fr"),
        ("¿Qué es la inteligencia artificial?", "es"),
    ]
    
    for query, lang in test_queries:
        print(f"\nQuery: {query}")
        print(f"Language: {lang}")
        print("-" * 40)
        
        try:
            answer = generate_answer(query, test_contexts, lang=lang)
            print(f"Answer: {answer[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    # Test with memory
    print("\n\nTesting with Conversation Memory")
    print("=" * 60)
    
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("What is AI?", "AI stands for Artificial Intelligence.")
    
    query = "Can you explain more?"
    answer = generate_answer(query, test_contexts, memory=memory, lang="en")
    
    print(f"Query: {query}")
    print(f"Answer with memory: {answer[:200]}...")
    
    # Show stats
    print("\n\nGeneration Statistics")
    print("=" * 60)
    stats = get_generation_stats()
    print(f"Available models: {stats['model_manager']['available_models']}")
    print(f"Cache info: {stats['cache_info']}")