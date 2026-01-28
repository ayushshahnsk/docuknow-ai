"""
chat_store.py

Responsible for:
- Saving chat sessions to local disk
- Loading chat sessions on app restart
- Deleting chat data safely

This enables ChatGPT-style persistent chats.
"""

import json
from pathlib import Path
from typing import Dict, List

# Base directory for storing chats
CHAT_STORE_DIR = Path("data/chats")
CHAT_STORE_DIR.mkdir(parents=True, exist_ok=True)


def save_chat(chat_data: Dict):
    """
    Save a single chat to disk as JSON.

    chat_data must include:
    - chat_id
    - chat_name
    - index_name
    - chat_history
    """
    chat_id = chat_data["chat_id"]
    file_path = CHAT_STORE_DIR / f"{chat_id}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)


def load_all_chats() -> List[Dict]:
    """
    Load all saved chats from disk.
    Returns a list of chat dictionaries.
    """
    chats = []

    for file in CHAT_STORE_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                chats.append(json.load(f))
        except Exception:
            # Skip corrupted files safely
            continue

    return chats


def delete_chat(chat_id: str):
    """
    Delete a chat file from disk.
    """
    file_path = CHAT_STORE_DIR / f"{chat_id}.json"
    if file_path.exists():
        file_path.unlink()


def clear_all_chats():
    """
    Utility function (optional):
    Deletes all stored chats.
    """
    for file in CHAT_STORE_DIR.glob("*.json"):
        file.unlink()
