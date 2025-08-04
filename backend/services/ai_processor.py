import openai
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
import json
import asyncio
from typing import Optional, Dict, List
import logging
import time
from datetime import datetime, timedelta

from core.config import settings
from models.database import get_db, User, Conversation, StudySession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = settings.openai_api_key

class AIProcessor:
    """
    Enhanced AI Processing service with real integrations
    """
    
    def __init__(self):
        self.conversation_context = {}  # In-memory context storage
    
    async def extract_text_from_image(self, image_url: str, user_phone: str) -> Dict[str, str]:
        """
        Extract text from image using Tesseract OCR
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing image from URL: {image_url}")
            
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Process with PIL and Tesseract
            image = Image.open(BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text
            extracted_text = pytesseract.image_to_string(image, lang='eng')
            
            if not extracted_text.strip():
                return {
                    "success": False,
                    "message": "I couldn't find any readable text in this image. Please make sure the image is clear and contains visible text.",
                    "extracted_text": ""
                }
            
            # Generate AI summary and study suggestions
            ai_response = await self._generate_study_response(extracted_text, "image_analysis", user_phone)
            
            processing_time = time.time() - start_time
            
            # Store in database
            await self._store_conversation(
                user_phone=user_phone,
                message_type="image",
                extracted_content=extracted_text,
                ai_response=ai_response,
                processing_time=processing_time,
                media_url=image_url
            )
            
            return {
                "success": True,
                "message": f"📸 **Text Extracted from Image:**\n\n{extracted_text}\n\n📚 **Study Analysis:**\n\n{ai_response}",
                "extracted_text": extracted_text
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "success": False,
                "message": "I encountered an error processing your image. Please try again with a clearer image.",
                "extracted_text": ""
            }
    
    async def transcribe_audio(self, audio_url: str, user_phone: str) -> Dict[str, str]:
        """
        Transcribe audio using OpenAI Whisper
        """
        start_time = time.time()
        
        try:
            logger.info(f"Transcribing audio from URL: {audio_url}")
            
            # Download audio
            response = requests.get(audio_url, timeout=60)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_audio:
                temp_audio.write(response.content)
                temp_audio_path = temp_audio.name
            
            try:
                # Transcribe using OpenAI Whisper
                with open(temp_audio_path, 'rb') as audio_file:
                    transcript = openai.Audio.transcribe(
                        model=settings.whisper_model,
                        file=audio_file,
                        response_format="text"
                    )
                
                transcribed_text = transcript.strip()
                
                if not transcribed_text:
                    return {
                        "success": False,
                        "message": "I couldn't transcribe your audio. Please try recording again with clear speech.",
                        "transcription": ""
                    }
                
                # Generate AI response
                ai_response = await self._generate_study_response(transcribed_text, "audio_analysis", user_phone)
                
                processing_time = time.time() - start_time
                
                # Store in database
                await self._store_conversation(
                    user_phone=user_phone,
                    message_type="audio",
                    user_message=transcribed_text,
                    ai_response=ai_response,
                    processing_time=processing_time,
                    media_url=audio_url
                )
                
                return {
                    "success": True,
                    "message": f"🎵 **Audio Transcription:**\n\n{transcribed_text}\n\n📚 **Study Analysis:**\n\n{ai_response}",
                    "transcription": transcribed_text
                }
                
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "success": False,
                "message": "I had trouble processing your audio. Please try recording again.",
                "transcription": ""
            }
    
    async def get_chat_response(self, text: str, user_phone: str) -> str:
        """
        Generate intelligent chat response using OpenAI GPT
        """
        try:
            start_time = time.time()
            
            # Get user context
            user_context = await self._get_user_context(user_phone)
            
            # Build conversation history for context
            context_messages = await self._build_conversation_context(user_phone)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(user_context)
            
            # Generate response
            messages = [
                {"role": "system", "content": system_prompt},
                *context_messages,
                {"role": "user", "content": text}
            ]
            
            response = openai.ChatCompletion.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            processing_time = time.time() - start_time
            
            # Store conversation
            await self._store_conversation(
                user_phone=user_phone,
                message_type="text",
                user_message=text,
                ai_response=ai_response,
                processing_time=processing_time,
                tokens_used=response.usage.total_tokens
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I'm having trouble processing your request right now. Please try again in a moment."
    
    async def _generate_study_response(self, content: str, analysis_type: str, user_phone: str) -> str:
        """
        Generate study-focused AI response for extracted content
        """
        try:
            user_context = await self._get_user_context(user_phone)
            
            prompt = f"""
            As an AI study assistant, analyze this {analysis_type.replace('_', ' ')} content and provide helpful study insights:

            Content: {content}

            User Context: {user_context.get('study_level', 'general')} level, interested in {user_context.get('subjects', 'various subjects')}

            Please provide:
            1. Key concepts identified
            2. Study suggestions
            3. Potential quiz questions (if applicable)
            4. Related topics to explore

            Keep response concise but helpful for studying.
            """
            
            response = openai.ChatCompletion.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating study response: {e}")
            return "Here's the extracted content. I can help you study this material - just ask me questions about it!"
    
    def _create_system_prompt(self, user_context: Dict) -> str:
        """Create personalized system prompt"""
        return f"""
        You are an AI study assistant helping students learn effectively. 

        User Profile:
        - Study Level: {user_context.get('study_level', 'general')}
        - Subjects: {user_context.get('subjects', 'various')}
        - Language: {user_context.get('preferred_language', 'en')}

        Your capabilities:
        - Analyze images of notes and extract text
        - Transcribe audio recordings
        - Answer study questions
        - Create summaries and quizzes
        - Provide explanations and examples

        Guidelines:
        - Be encouraging and supportive
        - Break down complex topics
        - Provide practical study tips
        - Ask clarifying questions when needed
        - Use emojis sparingly but effectively

        Always aim to help the user learn and understand better.
        """
    
    async def _get_user_context(self, user_phone: str) -> Dict:
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
                # Create new user
                new_user = User(phone_number=user_phone)
                db.add(new_user)
                db.commit()
                
            return {"study_level": "general", "subjects": [], "preferred_language": "en"}
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {"study_level": "general", "subjects": [], "preferred_language": "en"}
    
    async def _build_conversation_context(self, user_phone: str, limit: int = 5) -> List[Dict]:
        """Build conversation context for AI"""
        try:
            db = next(get_db())
            recent_conversations = db.query(Conversation)\
                .filter(Conversation.user_phone == user_phone)\
                .order_by(Conversation.created_at.desc())\
                .limit(limit)\
                .all()
            
            context = []
            for conv in reversed(recent_conversations):  # Reverse to get chronological order
                if conv.user_message:
                    context.append({"role": "user", "content": conv.user_message})
                if conv.ai_response:
                    context.append({"role": "assistant", "content": conv.ai_response})
            
            return context
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return []
    
    async def _store_conversation(self, **kwargs):
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