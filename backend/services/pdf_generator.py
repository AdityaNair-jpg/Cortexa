# backend/services/pdf_generator.py

from fpdf import FPDF
import os
import uuid
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the static directory exists
os.makedirs("static", exist_ok=True)

class PDFGenerator(FPDF):
    """
    Custom PDF class to create headers and footers.
    """
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Your AI Study Notes', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_from_text(summary_text: str, user_phone: str) -> str:
    """
    Generates a PDF from summary text and saves it to the static folder.
    
    Returns:
        str: The path to the generated PDF file (e.g., "static/filename.pdf")
    """
    try:
        pdf = PDFGenerator()
        pdf.add_page()
        pdf.set_font('Helvetica', '', 12)
        
        # We need to clean up the markdown-like text from Gemini for the PDF
        # This is a simple replacement; a more complex parser could be used.
        cleaned_text = summary_text.replace('**', '').replace('##', '').replace('*', '- ')
        
        # Add the text to the PDF
        pdf.multi_cell(0, 10, cleaned_text)
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4().hex[:8])
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{user_phone.replace('+', '')}_{timestamp}_{unique_id}.pdf"
        filepath = os.path.join("static", filename)
        
        # Save the PDF
        pdf.output(filepath)
        
        logger.info(f"Generated PDF: {filepath}")
        
        # Return the web-accessible path
        return filepath

    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None

# Global instance
pdf_generator = create_pdf_from_text