"""
Chat Session Manager for DocuMind AI.

This module enables:
- ChatGPT-style multi-chat sessions
- Each chat tied to its own documents
- Independent vector DB, memory & token tracking per chat
- Safe switching between chats

Design principle:
Each chat is fully isolated.
"""

import uuid
from typing import Dict, Optional

from memory.conversation import ConversationMemory
from analytics.token_tracker import TokenTracker


class ChatSession:
    """
    Represents a single chat session.
    """

    def __init__(self, name: str):
        self.chat_id: str = str(uuid.uuid4())
        self.chat_name: str = name

        # RAG-related
        self.index_name: Optional[str] = None  # FAISS index id

        # Conversation
        self.memory = ConversationMemory()
        self.chat_history = []  # [{"role": "user"/"assistant", "content": str}]

        # Analytics
        self.token_tracker = TokenTracker()

    def reset(self):
        """
        Clear chat memory and analytics (documents remain).
        """
        self.memory.clear()
        self.chat_history.clear()
        self.token_tracker.reset()


class ChatManager:
    """
    Manages multiple ChatSession objects.
    """

    def __init__(self):
        self.chats: Dict[str, ChatSession] = {}
        self.active_chat_id: Optional[str] = None

    # -----------------------------
    # Chat Lifecycle
    # -----------------------------
    def create_chat(self, name: str) -> ChatSession:
        """
        Create a new chat session.
        """
        chat = ChatSession(name=name)
        self.chats[chat.chat_id] = chat
        self.active_chat_id = chat.chat_id
        return chat

    def delete_chat(self, chat_id: str):
        """
        Delete an existing chat.
        """
        if chat_id in self.chats:
            del self.chats[chat_id]

            if self.active_chat_id == chat_id:
                self.active_chat_id = (
                    next(iter(self.chats), None)
                )

    def switch_chat(self, chat_id: str):
        """
        Switch active chat.
        """
        if chat_id in self.chats:
            self.active_chat_id = chat_id

    # -----------------------------
    # Access Helpers
    # -----------------------------
    def get_active_chat(self) -> Optional[ChatSession]:
        """
        Get the currently active chat session.
        """
        if not self.active_chat_id:
            return None
        return self.chats.get(self.active_chat_id)

    def list_chats(self):
        """
        List all chats (id + name).
        """
        return [
            {"chat_id": cid, "chat_name": chat.chat_name}
            for cid, chat in self.chats.items()
        ]

    # -----------------------------
    # Message Handling
    # -----------------------------
    def add_user_message(self, message: str):
        """
        Add user message to active chat.
        """
        chat = self.get_active_chat()
        if chat:
            chat.chat_history.append({
                "role": "user",
                "content": message
            })

    def add_assistant_message(self, message: str):
        """
        Add assistant message to active chat.
        """
        chat = self.get_active_chat()
        if chat:
            chat.chat_history.append({
                "role": "assistant",
                "content": message
            })

    # -----------------------------
    # Analytics
    # -----------------------------
    def get_token_stats(self):
        """
        Get token usage stats for active chat.
        """
        chat = self.get_active_chat()
        if not chat:
            return None
        return chat.token_tracker.get_totals()
