# backend/models/database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import json

# Import the settings object
from core.config import settings

# Database setup
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    """User model for storing WhatsApp user information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Study preferences
    study_level = Column(String(50))  # high_school, college, graduate
    subjects = Column(Text)  # JSON array of subjects
    preferred_language = Column(String(10), default="en")

class Conversation(Base):
    """Conversation model for storing chat history"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_phone = Column(String(20), index=True, nullable=False)
    message_id = Column(String(100), unique=True)
    
    # Message content
    message_type = Column(String(20))  # text, image, audio
    user_message = Column(Text)
    ai_response = Column(Text)
    
    # Media information
    media_url = Column(String(500))
    media_type = Column(String(50))
    extracted_content = Column(Text)  # OCR text or transcription
    
    # Metadata
    processing_time = Column(Float)
    tokens_used = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class StudySession(Base):
    """Study session tracking"""
    __tablename__ = "study_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_phone = Column(String(20), index=True, nullable=False)
    session_type = Column(String(50))  # quiz, summary, q_and_a, note_review
    
    content = Column(Text)  # Session content/materials
    duration_minutes = Column(Integer)
    score = Column(Float)  # For quizzes
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()