"""
LLM Generator for DocuKnow AI.

This module:
- Builds strict RAG prompts
- Injects conversational memory
- Calls Gemma via Ollama
- Prevents hallucinations
"""

import requests
from typing import List, Dict, Optional

from config.prompts import build_rag_prompt
from config.settings import OLLAMA_BASE_URL, LLM_MODEL_NAME, LLM_TIMEOUT
from memory.conversation import ConversationMemory


def generate_answer(
    query: str, contexts: List[Dict], memory: Optional[ConversationMemory] = None
) -> str:
    """
    Generate an answer using Gemma with RAG + optional conversation memory.
    """

    # Format conversation memory
    memory_block = ""
    if memory:
        history = memory.get_formatted_history()
        if history:
            memory_block = f"""
PREVIOUS CONVERSATION:
{history}
"""

    # Build final prompt
    base_prompt = build_rag_prompt(query, contexts)
    final_prompt = f"{memory_block}\n\n{base_prompt}".strip()

    # Call Ollama
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL_NAME, "prompt": final_prompt, "stream": False},
            timeout=LLM_TIMEOUT,
        )
    except requests.exceptions.RequestException:
        return "Error: Unable to connect to the language model."

    if response.status_code != 200:
        return "Error: Language model failed to generate a response."

    data = response.json()
    answer = data.get("response", "").strip()

    # Update memory
    if memory:
        memory.add_turn(query, answer)

    return answer
