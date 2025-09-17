from typing import List, Dict, Optional
import json
import random
from datetime import datetime, timedelta
import google.generativeai as genai
from core.config import settings
from models.database import get_db, StudySession, Conversation
from core.config import settings

genai.configure(api_key=settings.gemini_api_key)

class StudyFeatures:
    """
    Advanced study features like quizzes, summaries, and progress tracking
    """
    
    def create_quiz(self, user_phone: str, content: str, num_questions: int = 5) -> Dict:
        """Generate quiz questions from study content"""
        try:
            model = genai.GenerativeModel(settings.gemini_model)
            
            prompt = f"""
            Create {num_questions} multiple choice quiz questions based on this content:
            
            {content}
            
            Format as JSON with this structure:
            {{
                "questions": [
                    {{
                        "question": "What is...?",
                        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                        "correct_answer": "A",
                        "explanation": "Brief explanation"
                    }}
                ]
            }}
            """
            
            response = model.generate_content_async(prompt)
            # Clean up the response to make sure it's valid JSON
            quiz_data = json.loads(response.text.replace("```json", "").replace("```", ""))
            
            # Store quiz session
            db = next(get_db())
            session = StudySession(
                user_phone=user_phone,
                session_type="quiz",
                content=json.dumps(quiz_data),
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            
            return {
                "success": True,
                "quiz": quiz_data,
                "session_id": session.id
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating quiz: {str(e)}"
            }
    
    async def create_summary(self, user_phone: str, content: str) -> str:
        """Create a comprehensive summary of study material"""
        try:
            model = genai.GenerativeModel(settings.gemini_model)
            
            prompt = f"""
            Create a comprehensive study summary of this content:
            
            {content}
            
            Include:
            - Key concepts and definitions
            - Important points to remember
            - Connections between ideas
            - Study tips for this material
            
            Format with clear headings and bullet points.
            """
            
            response = model.generate_content_async(prompt)
            
            summary = response.text
            
            # Store summary session
            db = next(get_db())
            session = StudySession(
                user_phone=user_phone,
                session_type="summary",
                content=summary,
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            
            return summary
            
        except Exception as e:
            return f"Error creating summary: {str(e)}"
    
    async def get_study_progress(self, user_phone: str, days: int = 7) -> Dict:
        """Get user's study progress over specified days"""
        try:
            db = next(get_db())
            start_date = datetime.utcnow() - timedelta(days=days)
            
            sessions = db.query(StudySession)\
                .filter(StudySession.user_phone == user_phone)\
                .filter(StudySession.created_at >= start_date)\
                .all()
            
            conversations = db.query(Conversation)\
                .filter(Conversation.user_phone == user_phone)\
                .filter(Conversation.created_at >= start_date)\
                .count()
            
            progress = {
                "total_sessions": len(sessions),
                "conversations": conversations,
                "session_types": {},
                "average_quiz_score": 0,
                "study_streak": self._calculate_study_streak(user_phone)
            }
            
            quiz_scores = []
            for session in sessions:
                session_type = session.session_type
                progress["session_types"][session_type] = progress["session_types"].get(session_type, 0) + 1
                
                if session_type == "quiz" and session.score:
                    quiz_scores.append(session.score)
            
            if quiz_scores:
                progress["average_quiz_score"] = sum(quiz_scores) / len(quiz_scores)
            
            return progress
            
        except Exception as e:
            return {"error": f"Error getting progress: {str(e)}"}
    
    async def _calculate_study_streak(self, user_phone: str) -> int:
        """Calculate consecutive days of study activity"""
        try:
            db = next(get_db())
            
            # Get distinct study days
            conversations = db.query(Conversation.created_at)\
                .filter(Conversation.user_phone == user_phone)\
                .order_by(Conversation.created_at.desc())\
                .all()
            
            if not conversations:
                return 0
            
            study_days = set()
            for conv in conversations:
                study_days.add(conv.created_at.date())
            
            # Calculate streak
            streak = 0
            current_date = datetime.utcnow().date()
            
            while current_date in study_days:
                streak += 1
                current_date -= timedelta(days=1)
            
            return streak
            
        except Exception as e:
            return 0

study_features = StudyFeatures()