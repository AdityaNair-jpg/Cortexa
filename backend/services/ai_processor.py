import google.generativeai as genai
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
import json
import time
import re
from typing import Dict, List, Optional
import logging
from services.pdf_generator import pdf_generator
from core.config import settings
from models.database import get_db, User, Conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini client
genai.configure(api_key=settings.gemini_api_key)

# Main Class
class AIProcessor:
    """
    AI Processing service with Gemini integration
    """
    
    def __init__(self):
        self.conversation_context = {}  # In-memory context storage

    def get_chat_response(self, text: str, user_phone: str) -> str:
        """
        Generate intelligent chat response using Gemini
        Includes retry logic for rate limits
        """
        max_retries = 2
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                user_context = self._get_user_context(user_phone)
                context_messages = self._build_conversation_context(user_phone)
                
                # The system prompt is now passed as the first message in the history
                system_prompt = self._create_system_prompt(user_context)
                full_history = [{"role": "user", "parts": [{"text": system_prompt}]}, {"role": "model", "parts": [{"text": "Understood. I am ready to assist."}]}] + context_messages

                model = genai.GenerativeModel(settings.gemini_model)
                chat = model.start_chat(history=full_history)
                
                response = chat.send_message(text)
                
                ai_response = response.text
                processing_time = time.time() - start_time
                
                self._store_conversation(
                    user_phone=user_phone,
                    message_type="text",
                    user_message=text,
                    ai_response=ai_response,
                    processing_time=processing_time
                )
                
                return ai_response
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if ("429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower()) and attempt < max_retries - 1:
                    retry_seconds = retry_delay * (2 ** attempt)
                    
                    # Try to extract retry delay from error message
                    retry_match = re.search(r'retry.*?(\d+)', error_str, re.IGNORECASE)
                    if retry_match:
                        retry_seconds = int(retry_match.group(1)) + 1
                    
                    logger.warning(f"Rate limit hit in chat, retrying in {retry_seconds} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_seconds)
                    continue
                else:
                    logger.error(f"Error generating chat response: {e}")
                    if "429" in error_str or "quota" in error_str.lower():
                        return "I'm currently experiencing high demand. Please wait a moment and try again. The free tier has rate limits, so I may need a few seconds between requests. ⏱️"
                    return "I'm having trouble processing your request right now. Please try again in a moment."
        
        return "I'm currently experiencing high demand. Please wait a moment and try again. ⏱️"

    def extract_text_from_image(self, media_url: str, user_phone: str, image_urls: Optional[List[str]] = None) -> Dict:
        """
        Download an image, extract text using Tesseract OCR, and get a study response.
        Supports multiple images.
        """
        try:
            # Download the image from the Twilio URL with authentication
            response = requests.get(
                media_url, 
                auth=(settings.twilio_account_sid, settings.twilio_auth_token)
            )
            response.raise_for_status() # Raise an exception for bad status codes
            
            # Open the image and perform OCR
            image = Image.open(BytesIO(response.content))
            extracted_text = pytesseract.image_to_string(image)
            
            # Collect all extracted texts if multiple images
            all_extracted_texts = [extracted_text] if extracted_text.strip() else []
            logger.info(f"Extracted {len(extracted_text)} characters from first image")
            
            # Process additional images if provided
            if image_urls:
                logger.info(f"Processing {len(image_urls)} additional image(s)")
                for idx, img_url in enumerate(image_urls, start=2):
                    try:
                        logger.info(f"Downloading image {idx}/{len(image_urls)+1}: {img_url[:50]}...")
                        img_response = requests.get(
                            img_url,
                            auth=(settings.twilio_account_sid, settings.twilio_auth_token),
                            timeout=30
                        )
                        img_response.raise_for_status()
                        img = Image.open(BytesIO(img_response.content))
                        img_text = pytesseract.image_to_string(img)
                        if img_text.strip():
                            all_extracted_texts.append(img_text)
                            logger.info(f"Extracted {len(img_text)} characters from image {idx}")
                        else:
                            logger.warning(f"No text found in image {idx}")
                    except Exception as e:
                        logger.error(f"Error processing additional image {idx} ({img_url[:50]}...): {e}")
            
            logger.info(f"Total images processed: {len(all_extracted_texts)}")
            
            # Combine all extracted texts with clear separators
            if all_extracted_texts:
                combined_text = "\n\n" + "="*50 + f"\nIMAGE 1\n" + "="*50 + "\n\n" + all_extracted_texts[0]
                for idx, text in enumerate(all_extracted_texts[1:], start=2):
                    combined_text += "\n\n" + "="*50 + f"\nIMAGE {idx}\n" + "="*50 + "\n\n" + text
            else:
                combined_text = ""
            
            logger.info(f"Combined text length: {len(combined_text)} characters from {len(all_extracted_texts)} image(s)")
            
            if not combined_text.strip():
                return {"message": "I couldn't find any text in the image(s) you sent. Please try taking clearer pictures! 📸"}

            # Store the extracted content
            self._store_conversation(
                user_phone=user_phone,
                message_type="image",
                media_url=media_url,
                extracted_content=combined_text
            )

            # Get a helpful study response based on the text (concise and visual)
            study_response = self._generate_concise_study_response(combined_text, user_phone)

            # Collect all image URLs for PDF
            all_image_urls = [media_url]
            if image_urls:
                all_image_urls.extend(image_urls)

            # Generate PDF with multiple images
            pdf_path = pdf_generator.create_pdf_from_content(
                study_response, 
                user_phone,
                image_urls=all_image_urls,
                extracted_texts=all_extracted_texts
            )
            
            if pdf_path:
                return {
                    "message": "I've processed your notes and generated a visual PDF summary with mind maps! 📝🗺️",
                    "pdf_path": pdf_path  # e.g., "static/filename.pdf"
                }
            else:
                return {"message": "I extracted the text but failed to generate a PDF. Please try again."}
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"message": "I had trouble reading that image. Please make sure it's a clear photo and try again."}

    def transcribe_audio(self, media_url: str, user_phone: str) -> Dict:
        """
        Download an audio file and transcribe it.
        (This is a placeholder - you'll need a speech-to-text library like Whisper or SpeechRecognition)
        """
        try:
            # This part is a placeholder. You would need to integrate a speech-to-text service.
            # For now, we'll simulate a transcription.
            transcribed_text = "This is a placeholder for the transcribed audio. Please replace with a real transcription service."

            # Store the conversation
            self._store_conversation(
                user_phone=user_phone,
                message_type="audio",
                media_url=media_url,
                extracted_content=transcribed_text
            )

            # Generate a helpful study response
            study_response = self._generate_study_response(transcribed_text, "audio_review", user_phone)
            
            return {"message": study_response}

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {"message": "I'm sorry, I had trouble processing that audio file. Please try recording again."}
    
    def _generate_study_response(self, content: str, analysis_type: str, user_phone: str) -> str:
        """
        Generate study-focused AI response for extracted content
        """
        try:
            user_context = self._get_user_context(user_phone)
            model = genai.GenerativeModel(settings.gemini_model)
            
            # --- REFINED PROMPT ---
            prompt = f"""
            **Role**: You are an expert AI Study Assistant.
            **Task**: Analyze the following content and provide a structured, helpful study guide.
            **User Profile**: The user is a {user_context.get('study_level', 'student')} studying {user_context.get('subjects', ['general topics'])}. Tailor your language and examples accordingly.

            **Content for Analysis**:
            ---
            {content}
            ---

            **Instructions**:
            Based on the content, generate the following sections. Use markdown for formatting (e.g., #, **, *, -).

            1.  **Key Concepts (🔑)**: Identify and explain the 3-5 most important concepts. For each concept, provide a concise definition and explain *why* it is important.
            2.  **Potential Pitfalls (🤔)**: What are some common misunderstandings or tricky points related to this material?
            3.  **Analogies & Examples (💡)**: Provide at least one simple analogy or real-world example to make the core ideas easier to understand.
            4.  **Practice Question (✍️)**: Create one open-ended question that would test a deep understanding of this material. Do not provide the answer.
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating study response: {e}")
            return "Here's the extracted content. I can help you study this material - just ask me questions about it!"
    
    def _generate_concise_study_response(self, content: str, user_phone: str) -> str:
        """
        Generate concise, visual study notes optimized for fast memorization
        Includes retry logic for rate limits
        """
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                user_context = self._get_user_context(user_phone)
                model = genai.GenerativeModel(settings.gemini_model)
                
                prompt = f"""
                **Role**: You are an expert AI Study Assistant specializing in creating concise, memorable study notes.
                **Task**: Transform the following content into SHORT, VISUAL study notes optimized for quick memorization.
                **User Profile**: {user_context.get('study_level', 'student')} studying {user_context.get('subjects', ['general topics'])}.

                **Content**:
                ---
                {content[:8000]}  # Increased limit for multiple images
                ---

                **CRITICAL INSTRUCTIONS**:
                1. **Be EXTREMELY CONCISE** - Each point should be 1-2 lines maximum. No long paragraphs.
                2. **Use Visual Markers** - Use emojis, symbols, and formatting to make it scannable.
                3. **Structure with Headings** - Use ## for main sections, ### for subsections.
                4. **Key Concepts First** - List 3-5 key concepts with VERY brief definitions (one line each).
                5. **Bullet Points Only** - Use bullet points, not paragraphs. Each bullet = one fact to remember.
                6. **Memory Aids** - Include mnemonics, acronyms, or memory tricks where helpful.
                7. **Visual Hierarchy** - Use formatting to create visual hierarchy (bold for important terms).

                **Format**:
                ## 🔑 Key Concepts
                - **Concept 1**: One-line definition
                - **Concept 2**: One-line definition
                - **Concept 3**: One-line definition

                ## 📝 Important Points
                - Point 1 (max 2 lines)
                - Point 2 (max 2 lines)
                - Point 3 (max 2 lines)

                ## 💡 Quick Facts
                - Fact 1
                - Fact 2
                - Fact 3

                ## 🧠 Memory Tips
                - Tip 1
                - Tip 2

                Keep the ENTIRE response under 500 words. Focus on what's essential to remember.
                """
                response = model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    # Extract retry delay from error if available
                    retry_seconds = retry_delay * (2 ** attempt)  # Exponential backoff
                    
                    # Try to extract retry delay from error message
                    retry_match = re.search(r'retry.*?(\d+)', error_str, re.IGNORECASE)
                    if retry_match:
                        retry_seconds = int(retry_match.group(1)) + 1
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_seconds} seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_seconds)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts. Using fallback content.")
                        # Return a better fallback that still creates a useful PDF
                        return self._create_fallback_study_notes(content)
                else:
                    # For other errors, log and return fallback
                    logger.error(f"Error generating concise study response (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return self._create_fallback_study_notes(content)
        
        # Final fallback
        return self._create_fallback_study_notes(content)
    
    def _create_fallback_study_notes(self, content: str) -> str:
        """
        Create basic study notes from content when AI is unavailable
        """
        # Extract key information from content using simple text processing
        lines = content.split('\n')
        key_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
        
        # Take first 20 meaningful lines
        key_content = '\n'.join(key_lines[:20])
        
        # Create a simple structured format
        fallback = f"""## 📝 Study Notes

## 🔑 Key Points
{self._extract_simple_key_points(content)}

## 📄 Content Summary
{key_content[:800]}

---
*Note: This is a basic summary. For enhanced AI-generated notes, please try again in a moment.*
"""
        return fallback
    
    def _extract_simple_key_points(self, content: str, max_points: int = 5) -> str:
        """Extract simple key points from content"""
        # Find sentences that might be important (contain keywords, are questions, etc.)
        sentences = re.split(r'[.!?]\s+', content)
        key_sentences = []
        
        important_keywords = ['important', 'key', 'main', 'primary', 'essential', 'critical', 'note', 'remember']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30 and len(sentence) < 150:
                # Check if sentence contains important keywords or starts with capital (likely a concept)
                if any(keyword in sentence.lower() for keyword in important_keywords) or sentence[0].isupper():
                    key_sentences.append(f"- {sentence}")
                    if len(key_sentences) >= max_points:
                        break
        
        if not key_sentences:
            # Fallback: just take first few sentences
            key_sentences = [f"- {s.strip()}" for s in sentences[:max_points] if len(s.strip()) > 20]
        
        return '\n'.join(key_sentences[:max_points])
    # --- All helper methods are now synchronous ---

    def _get_user_context(self, user_phone: str) -> Dict:
        """Get user context from database"""
        try:
            db = next(get_db())
            user = db.query(User).filter(User.phone_number == user_phone).first()
            
            if user:
                subjects = json.loads(user.subjects) if user.subjects else []
                return {
                    "study_level": user.study_level or "general",
                    "subjects": subjects,
                    "preferred_language": user.preferred_language or "en"
                }
            else:
                new_user = User(phone_number=user_phone)
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
            return {"study_level": "general", "subjects": [], "preferred_language": "en"}
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {"study_level": "general", "subjects": [], "preferred_language": "en"}

    def _build_conversation_context(self, user_phone: str, limit: int = 5) -> List[Dict]:
        """Build conversation context for AI"""
        try:
            db = next(get_db())
            recent_conversations = db.query(Conversation)\
                .filter(Conversation.user_phone == user_phone)\
                .order_by(Conversation.created_at.desc())\
                .limit(limit)\
                .all()
            
            context = []
            for conv in reversed(recent_conversations):
                if conv.user_message:
                    # For Gemini, the role for user(s) messages is 'user'
                    context.append({"role": "user", "parts": [{"text": conv.user_message}]})
                if conv.ai_response:
                    # For Gemini, the role for assistant(s) messages is 'model'
                    context.append({"role": "model", "parts": [{"text": conv.ai_response}]})
            
            return context
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return []
        
    def _create_system_prompt(self, user_context: Dict) -> str:
        """Create personalized system prompt to define the AI's persona and instructions."""
        # --- ENHANCED SYSTEM PROMPT - Now handles general questions too ---
        return f"""
        You are Cortexa, a friendly and highly effective AI assistant. While you specialize in study assistance, you can also answer general questions on any topic.

        **Your Persona**:
        - **Encouraging & Patient**: Always be supportive. Use emojis to convey a friendly tone (e.g., 🤔, 💡, ✅).
        - **Socratic Teacher**: When a user asks a question, guide them toward the answer instead of just stating it. Ask clarifying questions when helpful.
        - **Structured**: Present information clearly using markdown, bullet points, and bold text.
        - **Versatile**: You can answer questions about ANY topic - science, history, technology, general knowledge, etc.

        **User Profile**:
        - Study Level: {user_context.get('study_level', 'general')}
        - Subjects of Interest: {", ".join(user_context.get('subjects', ['various']))}

        **Core Capabilities**:
        - Answer questions on ANY topic (not just study-related)
        - Analyze text from images and audio
        - Create summaries, quizzes, and study guides
        - Provide explanations and examples for any subject
        - Help with general knowledge, problem-solving, and learning

        **Rules**:
        - Answer questions directly and helpfully, whether they're study-related or general
        - Provide clear explanations with examples when helpful
        - Keep responses concise and focused on the user's request
        - If you don't know something, say so. Do not make up information
        - For study questions, use the Socratic method when appropriate
        - For general questions, provide accurate, helpful information
        """

    def _store_conversation(self, **kwargs):
        """Store conversation in database"""
        try:
            db = next(get_db())
            conversation = Conversation(**kwargs)
            db.add(conversation)
            db.commit()
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")

# Global instance
ai_processor = AIProcessor()
