import google.generativeai as genai
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
import json
import time
from typing import Dict, List
import logging

from core.config import settings
from models.database import get_db, User, Conversation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini client
genai.configure(api_key=settings.gemini_api_key)

class AIProcessor:
    """
    AI Processing service with Gemini integration
    """
    
    def __init__(self):
        self.conversation_context = {}  # In-memory context storage

    def get_chat_response(self, text: str, user_phone: str) -> str:
        """
        Generate intelligent chat response using Gemini
        """
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
            logger.error(f"Error generating chat response: {e}")
            return "I'm having trouble processing your request right now. Please try again in a moment."

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
        # --- NEW & IMPROVED SYSTEM PROMPT ---
        return f"""
        You are Cortexa, a friendly and highly effective AI study assistant. Your goal is to help students understand complex topics, not just give them answers.

        **Your Persona**:
        - **Encouraging & Patient**: Always be supportive. Use emojis to convey a friendly tone (e.g., 🤔, 💡, ✅).
        - **Socratic Teacher**: When a user asks a question, guide them toward the answer instead of just stating it. Ask clarifying questions.
        - **Structured**: Present information clearly using markdown, bullet points, and bold text.

        **User Profile**:
        - Study Level: {user_context.get('study_level', 'general')}
        - Subjects of Interest: {user_context.get('subjects', 'various')}

        **Core Capabilities**:
        - Analyze text from images and audio.
        - Create summaries, quizzes, and study guides.
        - Answer specific questions by providing explanations and examples.

        **Rules**:
        - NEVER just give the answer to a question without explanation.
        - Keep responses concise and focused on the user's request.
        - If you don't know something, say so. Do not make up information.
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