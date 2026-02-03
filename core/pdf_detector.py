"""
PDF Type Detector for DocuKnow AI

Purpose:
- Detect if PDF is text-based or image-based
- Determine if OCR processing is needed
"""

import fitz  # PyMuPDF
from typing import Tuple

def detect_pdf_type(file_path: str, text_threshold: float = 0.1) -> Tuple[str, float]:
    """
    Detect if PDF is text-based or image-based.
    
    Args:
        file_path: Path to PDF file
        text_threshold: Minimum text ratio to consider as text-based (0.1 = 10%)
        
    Returns:
        Tuple of (pdf_type, text_ratio)
        - pdf_type: "text" or "image"
        - text_ratio: Ratio of text pages to total pages
    """
    
    doc = fitz.open(file_path)
    text_pages = 0
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # If page has more than 50 characters, consider it as text page
        if len(text) > 50:
            text_pages += 1
    
    doc.close()
    
    text_ratio = text_pages / total_pages if total_pages > 0 else 0
    
    # Determine PDF type
    if text_ratio >= text_threshold:
        pdf_type = "text"
    else:
        pdf_type = "image"
    
    return pdf_type, text_ratio