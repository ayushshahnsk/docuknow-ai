"""
Token usage tracking for DocuMind AI.

This module:
- Estimates input & output token usage
- Tracks per-message and per-chat tokens
- Is model-agnostic (works with Gemma, LLaMA, etc.)
- Designed for transparency & analytics

NOTE:
Gemma (via Ollama) does not return token counts,
so we use an accepted approximation:
1 token ≈ 4 characters
"""

from typing import Dict


class TokenTracker:
    def __init__(self):
        """
        Initialize token counters for a single chat session.
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # -----------------------------
    # Internal Utility
    # -----------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count from text length.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    # -----------------------------
    # Public API
    # -----------------------------
    def track_input(
        self,
        prompt: str,
        context: str = "",
        memory: str = ""
    ) -> int:
        """
        Track input tokens (prompt + context + memory).
        Returns tokens used for this request.
        """
        combined_text = f"{memory}\n{context}\n{prompt}"
        tokens = self._estimate_tokens(combined_text)
        self.total_input_tokens += tokens
        return tokens

    def track_output(self, response: str) -> int:
        """
        Track output tokens from LLM response.
        """
        tokens = self._estimate_tokens(response)
        self.total_output_tokens += tokens
        return tokens

    def get_totals(self) -> Dict[str, int]:
        """
        Get cumulative token usage for this chat.
        """
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }

    def reset(self):
        """
        Reset token counters (used when chat is cleared).
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
