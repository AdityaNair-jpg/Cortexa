# backend/services/pdf_generator.py

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os
import uuid
import logging
from datetime import datetime
import re
from typing import List, Dict, Optional
from PIL import Image as PILImage
import requests
from io import BytesIO
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import networkx as nx
from core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the static directory exists
os.makedirs("static", exist_ok=True)
os.makedirs("static/temp", exist_ok=True)

class PDFGenerator:
    """
    Enhanced PDF generator with support for multiple images, mind maps, and visual elements
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#283593'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Subheading style
        self.subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#3949ab'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        )
        
        # Bullet point style
        self.bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=6,
            bulletIndent=10,
            fontName='Helvetica'
        )
        
        # Key point style (highlighted)
        self.keypoint_style = ParagraphStyle(
            'CustomKeyPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#d32f2f'),
            spaceAfter=8,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#ffebee'),
            borderPadding=8
        )
    
    def create_pdf_from_content(
        self, 
        content: str, 
        user_phone: str,
        image_urls: Optional[List[str]] = None,
        extracted_texts: Optional[List[str]] = None
    ) -> str:
        """
        Generate a visual, concise PDF with mind maps, figures, and short notes
        
        Args:
            content: AI-generated study content
            user_phone: User's phone number for filename
            image_urls: List of image URLs to include
            extracted_texts: List of extracted texts from images
            
        Returns:
            str: Path to generated PDF file
        """
        try:
            # Generate unique filename
            unique_id = str(uuid.uuid4().hex[:8])
            timestamp = datetime.now().strftime("%Y%m%d")
            clean_phone = re.sub(r'[^\d]', '', user_phone)
            filename = f"{clean_phone}_{timestamp}_{unique_id}.pdf"
            filepath = os.path.join("static", filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                filepath,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Build PDF content
            story = []
            
            # Add title
            story.append(Paragraph("📚 Study Notes", self.title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Parse and add structured content
            self._add_structured_content(story, content)
            
            # Add images if provided
            if image_urls:
                story.append(PageBreak())
                story.append(Paragraph("📷 Reference Images", self.heading_style))
                story.append(Spacer(1, 0.1*inch))
                self._add_images(story, image_urls)
            
            # Generate and add mind map
            story.append(PageBreak())
            story.append(Paragraph("🗺️ Concept Mind Map", self.heading_style))
            story.append(Spacer(1, 0.1*inch))
            mind_map_path = self._create_mind_map(content)
            if mind_map_path:
                story.append(Image(mind_map_path, width=7*inch, height=9*inch))
            
            # Build PDF
            doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
            
            logger.info(f"Generated enhanced PDF: {filepath}")
            return f"static/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}", exc_info=True)
            return None
    
    def _add_structured_content(self, story: List, content: str):
        """Parse AI content and add it to PDF in a structured, concise format"""
        # Split content into sections
        sections = self._parse_content_sections(content)
        
        for section in sections:
            section_type = section.get('type', 'text')
            section_content = section.get('content', '')
            
            if section_type == 'title':
                story.append(Paragraph(section_content, self.title_style))
                story.append(Spacer(1, 0.1*inch))
            
            elif section_type == 'heading':
                story.append(Paragraph(section_content, self.heading_style))
                story.append(Spacer(1, 0.05*inch))
            
            elif section_type == 'subheading':
                story.append(Paragraph(section_content, self.subheading_style))
                story.append(Spacer(1, 0.03*inch))
            
            elif section_type == 'key_point':
                # Extract key points and make them concise
                key_points = self._extract_key_points(section_content)
                for point in key_points:
                    story.append(Paragraph(f"🔑 {point}", self.keypoint_style))
                    story.append(Spacer(1, 0.05*inch))
            
            elif section_type == 'bullet_list':
                bullets = self._extract_bullets(section_content)
                for bullet in bullets:
                    # Make bullets concise (max 2 lines)
                    concise_bullet = self._make_concise(bullet, max_length=100)
                    story.append(Paragraph(f"• {concise_bullet}", self.bullet_style))
            
            elif section_type == 'text':
                # Make text concise
                concise_text = self._make_concise(section_content, max_length=150)
                story.append(Paragraph(concise_text, self.styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            # Add visual separator for sections
            if section_type in ['heading', 'subheading']:
                story.append(Spacer(1, 0.05*inch))
    
    def _parse_content_sections(self, content: str) -> List[Dict]:
        """Parse markdown-like content into structured sections"""
        sections = []
        lines = content.split('\n')
        
        current_section = {'type': 'text', 'content': ''}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_section['content']:
                    sections.append(current_section)
                    current_section = {'type': 'text', 'content': ''}
                continue
            
            # Detect section types
            if line.startswith('# '):
                if current_section['content']:
                    sections.append(current_section)
                sections.append({'type': 'title', 'content': line[2:].strip()})
                current_section = {'type': 'text', 'content': ''}
            
            elif line.startswith('## '):
                if current_section['content']:
                    sections.append(current_section)
                sections.append({'type': 'heading', 'content': line[3:].strip()})
                current_section = {'type': 'text', 'content': ''}
            
            elif line.startswith('### '):
                if current_section['content']:
                    sections.append(current_section)
                sections.append({'type': 'subheading', 'content': line[4:].strip()})
                current_section = {'type': 'text', 'content': ''}
            
            elif '🔑' in line or 'Key' in line and 'Concept' in line:
                if current_section['content']:
                    sections.append(current_section)
                current_section = {'type': 'key_point', 'content': line}
            
            elif line.startswith('- ') or line.startswith('* '):
                if current_section['type'] != 'bullet_list':
                    if current_section['content']:
                        sections.append(current_section)
                    current_section = {'type': 'bullet_list', 'content': ''}
                current_section['content'] += line + '\n'
            
            else:
                current_section['content'] += line + '\n'
        
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text, making them concise"""
        # Split by common separators
        points = re.split(r'[•\-\*]|\d+\.', text)
        key_points = []
        
        for point in points:
            point = point.strip()
            if len(point) > 10:  # Only include substantial points
                concise = self._make_concise(point, max_length=80)
                key_points.append(concise)
        
        return key_points[:5]  # Limit to 5 key points
    
    def _extract_bullets(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        bullets = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                bullets.append(line[2:].strip())
        return bullets
    
    def _make_concise(self, text: str, max_length: int = 100) -> str:
        """Make text more concise for quick memorization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If too long, truncate intelligently
        if len(text) > max_length:
            # Try to cut at sentence boundary
            sentences = re.split(r'[.!?]\s+', text)
            concise = ""
            for sentence in sentences:
                if len(concise + sentence) < max_length:
                    concise += sentence + ". "
                else:
                    break
            if concise:
                return concise.strip()
            # If no good sentence break, just truncate
            return text[:max_length-3] + "..."
        
        return text
    
    def _add_images(self, story: List, image_urls: List[str]):
        """Add multiple images to PDF"""
        for i, url in enumerate(image_urls, 1):
            try:
                # Download image
                if url.startswith('http'):
                    # For Twilio media URLs, use authentication
                    response = requests.get(
                        url,
                        auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                        timeout=10
                    )
                    response.raise_for_status()
                    img_data = BytesIO(response.content)
                else:
                    # Local file
                    with open(url, 'rb') as f:
                        img_data = BytesIO(f.read())
                
                # Open and process image
                pil_image = PILImage.open(img_data)
                
                # Resize if too large (max width 7 inches)
                max_width = 7 * 72  # 7 inches in points
                if pil_image.width > max_width:
                    ratio = max_width / pil_image.width
                    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
                
                # Save temporarily
                temp_path = os.path.join("static", "temp", f"img_{uuid.uuid4().hex[:8]}.png")
                pil_image.save(temp_path, 'PNG')
                
                # Add to PDF
                story.append(Paragraph(f"Image {i}", self.subheading_style))
                story.append(Image(temp_path, width=min(7*inch, pil_image.width*0.75), height=min(9*inch, pil_image.height*0.75)))
                story.append(Spacer(1, 0.2*inch))
                
            except Exception as e:
                logger.error(f"Error adding image {i}: {e}")
                story.append(Paragraph(f"Image {i}: Could not load", self.styles['Normal']))
    
    def _create_mind_map(self, content: str) -> Optional[str]:
        """Create a visual mind map from content"""
        try:
            # Extract key concepts
            concepts = self._extract_concepts(content)
            if len(concepts) < 2:
                return None
            
            # Create network graph
            G = nx.Graph()
            
            # Add central node
            central = concepts[0] if concepts else "Main Topic"
            G.add_node(central, size=2000, color='#1a237e')
            
            # Add other concepts
            for i, concept in enumerate(concepts[1:6]):  # Limit to 5 additional concepts
                G.add_node(concept, size=1000, color='#3949ab')
                G.add_edge(central, concept)
            
            # Create figure
            plt.figure(figsize=(10, 8), facecolor='white')
            ax = plt.gca()
            ax.set_facecolor('white')
            
            # Use spring layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            node_colors = [G.nodes[node].get('color', '#3949ab') for node in G.nodes()]
            node_sizes = [G.nodes[node].get('size', 1000) for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray', ax=ax)
            
            # Draw labels
            labels = {node: self._truncate_label(node, max_length=15) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax)
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save mind map
            mind_map_path = os.path.join("static", "temp", f"mindmap_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(mind_map_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return mind_map_path
            
        except Exception as e:
            logger.error(f"Error creating mind map: {e}")
            return None
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Look for headings, key terms, and important phrases
        concepts = []
        
        # Extract from headings
        headings = re.findall(r'#+\s+(.+)', content)
        concepts.extend([h.strip() for h in headings[:3]])
        
        # Extract capitalized phrases (likely concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        concepts.extend(capitalized[:5])
        
        # Remove duplicates and clean
        concepts = list(dict.fromkeys(concepts))  # Preserve order
        concepts = [c for c in concepts if len(c) > 3 and len(c) < 30]
        
        return concepts[:6]  # Return top 6 concepts
    
    def _truncate_label(self, label: str, max_length: int = 15) -> str:
        """Truncate label for mind map"""
        if len(label) <= max_length:
            return label
        return label[:max_length-3] + "..."
    
    def _add_header_footer(self, canvas_obj: canvas.Canvas, doc):
        """Add header and footer to each page"""
        canvas_obj.saveState()
        
        # Header
        canvas_obj.setFont('Helvetica', 9)
        canvas_obj.setFillColor(colors.HexColor('#666666'))
        canvas_obj.drawString(0.75*inch, 10.5*inch, "Study Notes - Cortexa")
        
        # Footer
        canvas_obj.drawString(0.75*inch, 0.5*inch, f"Page {canvas_obj.getPageNumber()}")
        
        canvas_obj.restoreState()


# Global instance
pdf_generator = PDFGenerator()

# Backward compatibility function
def create_pdf_from_text(summary_text: str, user_phone: str) -> str:
    """
    Legacy function for backward compatibility
    """
    return pdf_generator.create_pdf_from_content(summary_text, user_phone)
