from typing import List, Dict
import fitz  # PyMuPDF


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load a PDF and return a list of pages with text + metadata.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text().strip()

        if text:
            pages.append({"text": text, "page": page_number + 1})  # human-readable

    doc.close()
    return pages
