"""
chat_manager.py

Manages ChatGPT-style chat sessions for DocuKnow AI.

Responsibilities:
- Create / rename / delete chats
- Switch active chat
- Maintain per-chat state
- Persist chats via chat_store.py
- Store per-chat PDF names
- Ensure isolation between chats
"""

import uuid
from typing import Dict, Optional, List

from chat.chat_store import save_chat, load_all_chats, delete_chat


class ChatSession:
    """
    Represents a single chat session.
    """

    def __init__(
        self,
        chat_id: str,
        chat_name: str,
        index_name: Optional[str] = None,
        chat_history: Optional[list] = None,
        pdf_names: Optional[List[str]] = None,
    ):
        self.chat_id = chat_id
        self.chat_name = chat_name
        self.index_name = index_name
        self.chat_history = chat_history or []

        # 🔥 NEW: PDFs attached to this chat only
        self.pdf_names = pdf_names or []

    def to_dict(self) -> dict:
        """
        Serialize chat for storage.
        """
        return {
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "index_name": self.index_name,
            "chat_history": self.chat_history,
            "pdf_names": self.pdf_names,
        }


class ChatManager:
    """
    Handles multiple chat sessions.
    """

    def __init__(self):
        self.chats: Dict[str, ChatSession] = {}
        self.active_chat_id: Optional[str] = None

        self._load_existing_chats()

    # --------------------------------
    # Load / Save
    # --------------------------------
    def _load_existing_chats(self):
        """
        Load chats from disk at startup.
        """
        saved_chats = load_all_chats()

        for data in saved_chats:
            chat = ChatSession(
                chat_id=data["chat_id"],
                chat_name=data["chat_name"],
                index_name=data.get("index_name"),
                chat_history=data.get("chat_history", []),
                pdf_names=data.get("pdf_names", []),
            )
            self.chats[chat.chat_id] = chat

        # Restore last chat if any
        if self.chats:
            self.active_chat_id = next(iter(self.chats))

    def _persist(self, chat: ChatSession):
        """
        Save chat to disk.
        """
        save_chat(chat.to_dict())

    # --------------------------------
    # Chat lifecycle
    # --------------------------------
    def create_chat(self, chat_name: str = "New Chat") -> ChatSession:
        chat_id = str(uuid.uuid4())
        chat = ChatSession(chat_id=chat_id, chat_name=chat_name)

        self.chats[chat_id] = chat
        self.active_chat_id = chat_id

        self._persist(chat)
        return chat

    def delete_chat(self, chat_id: str):
        if chat_id in self.chats:
            delete_chat(chat_id)
            del self.chats[chat_id]

            if self.active_chat_id == chat_id:
                self.active_chat_id = next(iter(self.chats), None)

    def rename_chat(self, chat_id: str, new_name: str):
        if chat_id in self.chats:
            self.chats[chat_id].chat_name = new_name
            self._persist(self.chats[chat_id])

    def switch_chat(self, chat_id: str):
        if chat_id in self.chats:
            self.active_chat_id = chat_id

    # --------------------------------
    # Access helpers
    # --------------------------------
    def get_active_chat(self) -> Optional[ChatSession]:
        if not self.active_chat_id:
            return None
        return self.chats.get(self.active_chat_id)

    def list_chats(self):
        """
        Used by sidebar UI.
        """
        return [
            {"chat_id": chat.chat_id, "chat_name": chat.chat_name}
            for chat in self.chats.values()
        ]

    # --------------------------------
    # Message handling
    # --------------------------------
    def add_user_message(self, message: str):
        chat = self.get_active_chat()
        if chat:
            chat.chat_history.append({"role": "user", "content": message})
            self._persist(chat)

    def add_assistant_message(self, message: str):
        chat = self.get_active_chat()
        if chat:
            chat.chat_history.append({"role": "assistant", "content": message})
            self._persist(chat)

    # --------------------------------
    # PDF handling (PER CHAT)
    # --------------------------------
    def set_pdfs_for_chat(self, chat_id: str, pdf_names: List[str]):
        """
        Attach PDF names to a chat.
        """
        if chat_id in self.chats:
            self.chats[chat_id].pdf_names = pdf_names
            self._persist(self.chats[chat_id])

    def clear_chat_messages(self, chat_id: str):
        """
        Clear messages but keep PDFs & index.
        """
        if chat_id in self.chats:
            self.chats[chat_id].chat_history.clear()
            self._persist(self.chats[chat_id])

    def set_index_for_chat(self, chat_id: str, index_name: str):
        """
        Attach FAISS index to chat.
        """
        if chat_id in self.chats:
            self.chats[chat_id].index_name = index_name
            self._persist(self.chats[chat_id])
