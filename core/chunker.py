from typing import List, Dict
import re


def smart_chunk(
    pages: List[Dict], chunk_size: int = 500, overlap: int = 100
) -> List[Dict]:
    """
    Convert pages into smart chunks with metadata.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]

        # Split by paragraphs
        paragraphs = re.split(r"\n\s*\n", text)

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += " " + para
            else:
                chunks.append({"text": current_chunk.strip(), "page": page_number})

                # overlap
                current_chunk = current_chunk[-overlap:] + " " + para

        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip(), "page": page_number})

    return chunks
