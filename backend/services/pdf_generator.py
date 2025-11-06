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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import matplotlib.patheffects as path_effects
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
            
            # Add title (removed emoji to avoid font warnings)
            story.append(Paragraph("Study Notes", self.title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Parse and add structured content
            self._add_structured_content(story, content)
            
            # Add images if provided
            if image_urls:
                story.append(PageBreak())
                story.append(Paragraph("📷 Reference Images", self.heading_style))
                story.append(Spacer(1, 0.1*inch))
                self._add_images(story, image_urls)
            
            # Generate and add visual diagrams
            diagrams = self._create_visual_diagrams(content)
            
            if diagrams:
                story.append(PageBreak())
                story.append(Paragraph("📊 Visual Diagrams", self.heading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Add mind map
                if diagrams.get('mind_map'):
                    story.append(Paragraph("Concept Mind Map", self.subheading_style))
                    story.append(Spacer(1, 0.05*inch))
                    story.append(Image(diagrams['mind_map'], width=7*inch, height=5*inch))
                    story.append(Spacer(1, 0.2*inch))
                
                # Add flowchart
                if diagrams.get('flowchart'):
                    story.append(Paragraph("Process Flowchart", self.subheading_style))
                    story.append(Spacer(1, 0.05*inch))
                    story.append(Image(diagrams['flowchart'], width=7*inch, height=5*inch))
                    story.append(Spacer(1, 0.2*inch))
                
                # Add concept hierarchy
                if diagrams.get('hierarchy'):
                    story.append(Paragraph("Concept Hierarchy", self.subheading_style))
                    story.append(Spacer(1, 0.05*inch))
                    story.append(Image(diagrams['hierarchy'], width=7*inch, height=5*inch))
            
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
    
    def _create_visual_diagrams(self, content: str) -> Dict[str, Optional[str]]:
        """Create multiple visual diagrams from content"""
        diagrams = {}
        
        # Create mind map
        diagrams['mind_map'] = self._create_enhanced_mind_map(content)
        
        # Create flowchart
        diagrams['flowchart'] = self._create_flowchart(content)
        
        # Create concept hierarchy
        diagrams['hierarchy'] = self._create_concept_hierarchy(content)
        
        return {k: v for k, v in diagrams.items() if v is not None}
    
    def _create_enhanced_mind_map(self, content: str) -> Optional[str]:
        """Create an enhanced visual mind map with better styling"""
        try:
            # Extract key concepts and relationships
            concepts = self._extract_concepts(content)
            if len(concepts) < 2:
                return None
            
            # Create directed graph for better hierarchy
            G = nx.DiGraph()
            
            # Add central node
            central = concepts[0] if concepts else "Main Topic"
            G.add_node(central, level=0, color='#1a237e', size=3000)
            
            # Add primary concepts (level 1)
            primary_concepts = concepts[1:min(4, len(concepts))]
            for i, concept in enumerate(primary_concepts):
                G.add_node(concept, level=1, color='#3949ab', size=2000)
                G.add_edge(central, concept)
            
            # Add secondary concepts (level 2) if available
            if len(concepts) > 4:
                secondary_concepts = concepts[4:min(8, len(concepts))]
                for i, sec_concept in enumerate(secondary_concepts):
                    if i < len(primary_concepts):
                        parent = primary_concepts[i]
                        G.add_node(sec_concept, level=2, color='#5c6bc0', size=1500)
                        G.add_edge(parent, sec_concept)
            
            # Create figure with better styling
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
            ax.set_facecolor('#fafafa')
            
            # Use hierarchical layout
            pos = self._hierarchical_layout(G, central)
            
            # Draw edges with gradient
            for (u, v) in G.edges():
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=2, zorder=1)
            
            # Draw nodes with better styling
            for node in G.nodes():
                level = G.nodes[node].get('level', 0)
                color = G.nodes[node].get('color', '#3949ab')
                size = G.nodes[node].get('size', 1000)
                x, y = pos[node]
                
                # Draw node with shadow effect
                circle = Circle((x, y), radius=size/2000, color=color, 
                              alpha=0.9, zorder=2, edgecolor='white', linewidth=2)
                ax.add_patch(circle)
                
                # Add label with better formatting
                label = self._truncate_label(node, max_length=20)
                text = ax.text(x, y, label, ha='center', va='center', 
                             fontsize=9 if level == 0 else 8, fontweight='bold',
                             color='white', zorder=3)
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black', alpha=0.3)])
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.2, 1.2)
            ax.axis('off')
            plt.tight_layout()
            
            # Save mind map
            mind_map_path = os.path.join("static", "temp", f"mindmap_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(mind_map_path, dpi=200, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', pad_inches=0.1)
            plt.close()
            
            return mind_map_path
            
        except Exception as e:
            logger.error(f"Error creating enhanced mind map: {e}")
            return None
    
    def _hierarchical_layout(self, G: nx.DiGraph, root: str) -> Dict:
        """Create a hierarchical layout for the graph"""
        pos = {}
        
        # Position root at center top
        pos[root] = (0, 1)
        
        # Get children of root
        children = list(G.successors(root))
        if not children:
            return pos
        
        # Position level 1 nodes in a horizontal row below root
        n_children = len(children)
        if n_children == 1:
            pos[children[0]] = (0, 0.3)
        else:
            spacing = 1.6 / (n_children - 1) if n_children > 1 else 0
            start_x = -0.8
            for i, child in enumerate(children):
                pos[child] = (start_x + i * spacing, 0.3)
        
        # Position level 2 nodes below their parents
        for parent in children:
            grandchildren = list(G.successors(parent))
            for j, grandchild in enumerate(grandchildren):
                parent_x, parent_y = pos[parent]
                offset = (j - len(grandchildren)/2) * 0.3
                pos[grandchild] = (parent_x + offset, -0.3)
        
        return pos
    
    def _create_flowchart(self, content: str) -> Optional[str]:
        """Create a process flowchart from content"""
        try:
            # Extract process steps or sequential concepts
            steps = self._extract_process_steps(content)
            if len(steps) < 2:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
            ax.set_facecolor('#fafafa')
            
            # Create flowchart boxes
            box_width = 2.0
            box_height = 0.6
            spacing = 1.2
            
            # Draw process boxes
            boxes = []
            for i, step in enumerate(steps[:6]):  # Limit to 6 steps
                y_pos = 1.5 - i * spacing
                x_pos = 0
                
                # Create rounded rectangle
                box = FancyBboxPatch(
                    (x_pos - box_width/2, y_pos - box_height/2),
                    box_width, box_height,
                    boxstyle="round,pad=0.1",
                    facecolor='#3949ab',
                    edgecolor='#1a237e',
                    linewidth=2,
                    alpha=0.9
                )
                ax.add_patch(box)
                boxes.append((x_pos, y_pos, step))
                
                # Add arrow between boxes (except last)
                if i < len(steps) - 1:
                    arrow = FancyArrowPatch(
                        (x_pos, y_pos - box_height/2),
                        (x_pos, y_pos - spacing + box_height/2),
                        arrowstyle='->',
                        mutation_scale=20,
                        color='#666',
                        linewidth=2,
                        zorder=1
                    )
                    ax.add_patch(arrow)
                
                # Add text
                label = self._truncate_label(step, max_length=25)
                text = ax.text(x_pos, y_pos, label, ha='center', va='center',
                             fontsize=9, fontweight='bold', color='white',
                             wrap=True)
            
            # Add start and end indicators
            start_circle = Circle((0, 2.1), 0.15, color='#4caf50', zorder=2)
            end_circle = Circle((0, 1.5 - (len(steps)-1) * spacing - 0.5), 0.15, color='#f44336', zorder=2)
            ax.add_patch(start_circle)
            ax.add_patch(end_circle)
            
            ax.text(0, 2.1, 'START', ha='center', va='center', fontsize=8, 
                   fontweight='bold', color='white')
            ax.text(0, 1.5 - (len(steps)-1) * spacing - 0.5, 'END', ha='center', 
                   va='center', fontsize=8, fontweight='bold', color='white')
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(1.5 - (len(steps)-1) * spacing - 1, 2.5)
            ax.axis('off')
            plt.tight_layout()
            
            flowchart_path = os.path.join("static", "temp", f"flowchart_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(flowchart_path, dpi=200, bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.1)
            plt.close()
            
            return flowchart_path
            
        except Exception as e:
            logger.error(f"Error creating flowchart: {e}")
            return None
    
    def _create_concept_hierarchy(self, content: str) -> Optional[str]:
        """Create a hierarchical concept diagram"""
        try:
            concepts = self._extract_concepts(content)
            if len(concepts) < 3:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
            ax.set_facecolor('#fafafa')
            
            # Organize concepts hierarchically
            main_concept = concepts[0]
            sub_concepts = concepts[1:min(5, len(concepts))]
            
            # Draw main concept at top
            main_box = FancyBboxPatch(
                (-1, 1.5), 2, 0.5,
                boxstyle="round,pad=0.15",
                facecolor='#1a237e',
                edgecolor='#0d47a1',
                linewidth=3,
                alpha=0.95
            )
            ax.add_patch(main_box)
            ax.text(0, 1.75, self._truncate_label(main_concept, 20), 
                   ha='center', va='center', fontsize=11, fontweight='bold', 
                   color='white')
            
            # Draw sub-concepts in a row
            n_subs = len(sub_concepts)
            width_per_box = 3.0 / max(n_subs, 1)
            
            for i, sub_concept in enumerate(sub_concepts):
                x_pos = -1.5 + i * width_per_box + width_per_box/2
                y_pos = 0.5
                
                sub_box = FancyBboxPatch(
                    (x_pos - 0.4, y_pos - 0.25), 0.8, 0.5,
                    boxstyle="round,pad=0.1",
                    facecolor='#3949ab',
                    edgecolor='#283593',
                    linewidth=2,
                    alpha=0.9
                )
                ax.add_patch(sub_box)
                ax.text(x_pos, y_pos, self._truncate_label(sub_concept, 15),
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white')
                
                # Draw connecting line
                line = FancyArrowPatch(
                    (0, 1.5),
                    (x_pos, 0.75),
                    arrowstyle='->',
                    mutation_scale=15,
                    color='#666',
                    linewidth=1.5,
                    alpha=0.6,
                    connectionstyle="arc3,rad=0.2"
                )
                ax.add_patch(line)
            
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 2.2)
            ax.axis('off')
            plt.tight_layout()
            
            hierarchy_path = os.path.join("static", "temp", f"hierarchy_{uuid.uuid4().hex[:8]}.png")
            plt.savefig(hierarchy_path, dpi=200, bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.1)
            plt.close()
            
            return hierarchy_path
            
        except Exception as e:
            logger.error(f"Error creating concept hierarchy: {e}")
            return None
    
    def _extract_process_steps(self, content: str) -> List[str]:
        """Extract process steps or sequential information from content"""
        steps = []
        
        # Look for numbered lists
        numbered = re.findall(r'^\d+[\.\)]\s+(.+)$', content, re.MULTILINE)
        if numbered:
            steps.extend([s.strip() for s in numbered[:6]])
        
        # Look for step indicators
        step_patterns = [
            r'step\s+\d+[:\-]\s*(.+)',
            r'first[:\-]\s*(.+)',
            r'then[:\-]\s*(.+)',
            r'next[:\-]\s*(.+)',
            r'finally[:\-]\s*(.+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            steps.extend([m.strip() for m in matches[:3]])
        
        # If no steps found, use key concepts as steps
        if not steps:
            concepts = self._extract_concepts(content)
            steps = concepts[:6]
        
        # Clean and limit
        steps = [s for s in steps if len(s) > 5 and len(s) < 50]
        return steps[:6]
    
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
