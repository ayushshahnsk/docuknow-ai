"""
Updated PDF Loader for DocuKnow AI with OCR support

Purpose:
- Load text-based PDFs normally
- Detect and process image-based PDFs with OCR
- Unified interface for all PDF types
"""

import fitz  # PyMuPDF
from typing import List, Dict
import logging
from pathlib import Path

# Import our new modules with graceful fallback
try:
    from core.pdf_detector import detect_pdf_type
    from core.ocr_processor import OCRProcessor, EASYOCR_AVAILABLE
    OCR_SUPPORT_AVAILABLE = True
except ImportError as e:
    OCR_SUPPORT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"OCR modules not available: {e}")

# Initialize logger
logger = logging.getLogger(__name__)

def load_pdf(file_path: str, enable_ocr: bool = True) -> List[Dict]:
    """
    Load a PDF with automatic OCR detection and processing.
    
    Args:
        file_path: Path to PDF file
        enable_ocr: Whether to enable OCR processing for image PDFs
        
    Returns:
        List of pages with text and metadata
    """
    file_path = str(file_path)  # Ensure string path
    
    # If OCR is not available, always use text extraction
    if not OCR_SUPPORT_AVAILABLE or not enable_ocr:
        return _extract_text_pdf(file_path)
    
    try:
        # Step 1: Detect PDF type
        pdf_type, text_ratio = detect_pdf_type(file_path)
        logger.info(f"PDF Type: {pdf_type} (Text ratio: {text_ratio:.2%})")
        
        # Step 2: Process based on type
        if pdf_type == "text":
            # Text-based PDF: Use standard extraction
            return _extract_text_pdf(file_path)
        else:
            # Image-based PDF: Use OCR
            return _extract_ocr_pdf(file_path)
            
    except Exception as e:
        logger.error(f"Error in PDF loading with OCR: {e}")
        # Fallback to standard extraction
        return _extract_text_pdf(file_path)

def _extract_text_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from text-based PDF using PyMuPDF.
    """
    doc = fitz.open(file_path)
    pages = []
    
    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text().strip()
        
        if text:
            pages.append({
                "text": text,
                "page": page_number + 1,  # Human-readable page number
                "processed_with": "text_extraction"
            })
    
    doc.close()
    logger.info(f"Text extraction: Extracted {len(pages)} pages")
    
    return pages

def _extract_ocr_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from image-based PDF using OCR.
    """
    try:
        # Initialize OCR processor
        ocr_processor = OCRProcessor(languages=['en'], gpu=False)
        
        # Process PDF with OCR
        pages = ocr_processor.process_pdf(file_path)
        
        if not pages:
            logger.warning(f"No text extracted from {file_path} via OCR")
            # Fallback to text extraction anyway
            return _extract_text_pdf(file_path)
        
        return pages
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        # Fallback to standard extraction
        logger.info("Falling back to standard text extraction")
        return _extract_text_pdf(file_path)