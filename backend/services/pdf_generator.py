# backend/services/pdf_generator.py

from fpdf import FPDF
import os
import uuid
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the static directory exists
os.makedirs("static", exist_ok=True)

class PDFGenerator(FPDF):
    """
    Custom PDF class with Unicode support
    """
    def header(self):
        # Use DejaVu font for Unicode support
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.set_font('DejaVu', '', 12)
        self.cell(0, 10, 'Your AI Study Notes', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text_for_pdf(text: str) -> str:
    """
    Clean text to make it PDF-friendly while preserving Unicode characters.
    """
    # Clean up markdown formatting
    text = text.replace('**', '')  # Remove bold markers
    text = text.replace('##', '')  # Remove header markers
    text = text.replace('###', '') # Remove subheader markers
    
    # Replace markdown bullets with simple dashes
    text = re.sub(r'^\s*\*\s+', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*-\s+', '- ', text, flags=re.MULTILINE)
    
    # Remove multiple consecutive blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text

def create_pdf_from_text(summary_text: str, user_phone: str) -> str:
    """
    Generates a PDF from summary text with Unicode support.
    Falls back to ASCII-only if Unicode font is not available.
    
    Returns:
        str: The path to the generated PDF file (e.g., "static/filename.pdf")
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Try to use Unicode font, fall back to ASCII if not available
        use_unicode = False
        try:
            # Check if DejaVu font is available
            font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'DejaVuSans.ttf')
            if os.path.exists(font_path):
                pdf.add_font('DejaVu', '', font_path, uni=True)
                pdf.set_font('DejaVu', '', 11)
                use_unicode = True
                cleaned_text = clean_text_for_pdf(summary_text)
                logger.info("Using Unicode font for PDF")
            else:
                raise FileNotFoundError("DejaVu font not found")
        except Exception as font_error:
            # Fall back to ASCII-only approach
            logger.warning(f"Unicode font not available, using ASCII fallback: {font_error}")
            pdf.set_font('Helvetica', '', 11)
            use_unicode = False
            
            # ASCII fallback: strip all non-ASCII characters
            cleaned_text = clean_text_for_pdf(summary_text)
            
            # Replace Unicode punctuation with ASCII
            unicode_replacements = {
                '–': '-', '—': '-', ''': "'", ''': "'", 
                '"': '"', '"': '"', '…': '...', '•': '-',
                '·': '-', '×': 'x', '÷': '/',
            }
            for unicode_char, ascii_char in unicode_replacements.items():
                cleaned_text = cleaned_text.replace(unicode_char, ascii_char)
            
            # Remove emojis and other Unicode
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF"
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            cleaned_text = emoji_pattern.sub('', cleaned_text)
            
            # Final safety: convert to ASCII
            cleaned_text = cleaned_text.encode('ascii', errors='ignore').decode('ascii')
        
        # Add the text to the PDF
        pdf.multi_cell(0, 8, cleaned_text)
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4().hex[:8])
        timestamp = datetime.now().strftime("%Y%m%d")
        clean_phone = re.sub(r'[^\d]', '', user_phone)
        filename = f"{clean_phone}_{timestamp}_{unique_id}.pdf"
        filepath = os.path.join("static", filename)
        
        # Save the PDF
        pdf.output(filepath)
        
        logger.info(f"Generated PDF: {filepath} (Unicode: {use_unicode})")
        
        # Return the web-accessible path
        return f"static/{filename}"

    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        return None

# Global instance
pdf_generator = create_pdf_from_text