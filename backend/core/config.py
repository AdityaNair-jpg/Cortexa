# backend/core/config.py

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List
import os

class Settings(BaseSettings):
    """
    Comprehensive application settings
    """
    # Application
    app_name: str = "WhatsApp AI Study Assistant"
    app_env: str = Field("development", env="APP_ENV")
    debug: bool = Field(False, env="DEBUG")
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # Twilio Configuration
    twilio_account_sid: str = Field(..., env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(..., env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(..., env="TWILIO_PHONE_NUMBER")
    
    # Gemini Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    # Using Flash model for better rate limits (15 req/min vs 2 req/min for Pro on free tier)
    gemini_model: str = "gemini-2.5-flash"
    
    # Database
    database_url: str = Field("sqlite:///./study_assistant.db", env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # File Storage
    max_file_size_mb: int = Field(10, env="MAX_FILE_SIZE_MB")
    upload_folder: str = Field("./uploads", env="UPLOAD_FOLDER")
    
    # Public URL for serving files (needed for Twilio media URLs)
    # Set this to your public URL (e.g., ngrok URL in development, or your domain in production)
    public_base_url: Optional[str] = Field(None, env="PUBLIC_BASE_URL")
    
    # AI Settings
    max_context_length: int = 4000
    conversation_timeout: int = 3600  # 1 hour
    
    # Rate Limiting
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 500
    
    @validator('upload_folder')
    def create_upload_folder(cls, v):
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()