"""
OCR Processor for DocuKnow AI

Purpose:
- Extract text from image-based PDFs using OCR
- Handle scanned documents and image PDFs
"""

import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
from typing import List, Dict
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Try to import EasyOCR with graceful fallback
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not installed. OCR functionality will be limited.")

class OCRProcessor:
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize OCR processor.
        
        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            gpu: Use GPU acceleration if available
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR is not installed. Please install it using: "
                "pip install easyocr pillow opencv-python"
            )
        
        self.languages = languages
        self.gpu = gpu
        self.reader = None
        
    def _initialize_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self.reader is None:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
    
    def extract_text_from_page(self, page: fitz.Page) -> str:
        """
        Extract text from a single PDF page using OCR.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text as string
        """
        try:
            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution for better OCR
            img_data = pix.tobytes("ppm")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to numpy array for EasyOCR
            img_array = np.array(img)
            
            # Initialize reader if needed
            self._initialize_reader()
            
            # Perform OCR
            results = self.reader.readtext(
                img_array,
                paragraph=True,  # Group text into paragraphs
                detail=0  # Return only text, not coordinates
            )
            
            # Combine all text results
            text = "\n".join(results)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR failed for page: {e}")
            return ""
    
    def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Process entire PDF with OCR and return pages with text.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page dictionaries with text and metadata
        """
        doc = fitz.open(file_path)
        pages = []
        
        logger.info(f"Starting OCR processing for {file_path}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text using OCR
            text = self.extract_text_from_page(page)
            
            if text.strip():
                pages.append({
                    "text": text.strip(),
                    "page": page_num + 1,  # Human-readable page number
                    "processed_with": "ocr"
                })
            
            # Log progress for every 10 pages
            if (page_num + 1) % 10 == 0:
                logger.info(f"Processed {page_num + 1}/{len(doc)} pages")
        
        doc.close()
        logger.info(f"OCR completed. Extracted {len(pages)} pages with text.")
        
        return pages