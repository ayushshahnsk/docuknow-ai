"""
Conversational memory management for DocuMind AI.

This module:
- Stores recent Q&A turns
- Formats memory for LLM prompts
- Prevents prompt overflow
- Enables follow-up questions like:
  "Explain again", "Give example", "Compare with previous answer"
"""

from typing import List, Dict
from collections import deque


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        """
        max_turns: number of recent Q&A pairs to keep
        """
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str):
        """
        Add a new conversation turn.
        """
        self.history.append({"question": question, "answer": answer})

    def get_formatted_history(self) -> str:
        """
        Format conversation history for LLM prompt.
        """
        if not self.history:
            return ""

        formatted = []
        for turn in self.history:
            formatted.append(f"User: {turn['question']}")
            formatted.append(f"Assistant: {turn['answer']}")

        return "\n".join(formatted)

    def clear(self):
        """
        Clear all conversation history.
        """
        self.history.clear()
