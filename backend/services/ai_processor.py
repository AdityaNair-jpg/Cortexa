import requests
import pytesseract
from PIL import Image
from io import BytesIO
import tempfile
import os
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIProcessor:
    """
    AI Processing service for handling different types of media input
    """
    
    def __init__(self):
        """
        Initialize the AI processor with any required configurations
        """
        pass
    
    async def extract_text_from_image(self, image_url: str) -> str:
        """
        Extract text from an image using OCR (Tesseract)
        
        Args:
            image_url (str): URL of the image to process
            
        Returns:
            str: Extracted text from the image
        """
        try:
            logger.info(f"Processing image from URL: {image_url}")
            
            # Download the image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary (for better OCR results)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use Tesseract to extract text
            extracted_text = pytesseract.image_to_string(image, lang='eng')
            
            if not extracted_text.strip():
                return "I couldn't extract any readable text from this image. Please make sure the image is clear and contains visible text."
            
            logger.info("Text extraction successful")
            return f"I've extracted the following text from your image:\n\n{extracted_text.strip()}"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image: {e}")
            return "I couldn't download the image. Please try sending it again."
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return "I encountered an error while processing your image. Please try again with a different image."
    
    async def transcribe_audio(self, audio_url: str) -> str:
        """
        Transcribe audio to text using speech-to-text
        
        Args:
            audio_url (str): URL of the audio file to transcribe
            
        Returns:
            str: Transcribed text from the audio
        """
        try:
            logger.info(f"Processing audio from URL: {audio_url}")
            
            # Download the audio file
            response = requests.get(audio_url, timeout=60)
            response.raise_for_status()
            
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_audio:
                temp_audio.write(response.content)
                temp_audio_path = temp_audio.name
            
            try:
                # For now, return a mock response
                # In production, you would use OpenAI Whisper or another ASR service
                transcribed_text = "This is a mock transcription of your audio message. Please integrate with OpenAI Whisper or another ASR service for actual transcription."
                
                logger.info("Audio transcription successful")
                return f"I've transcribed your audio message:\n\n{transcribed_text}"
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading audio: {e}")
            return "I couldn't download the audio file. Please try sending it again."
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "I encountered an error while processing your audio. Please try again."
    
    async def get_chat_response(self, text: str) -> str:
        """
        Generate an AI chat response for text input
        
        Args:
            text (str): User's text message
            
        Returns:
            str: AI-generated response
        """
        try:
            logger.info(f"Processing text message: {text[:50]}...")
            
            # Analyze the message for common study-related keywords
            text_lower = text.lower()
            
            if any(keyword in text_lower for keyword in ['help', 'explain', 'what is', 'how to']):
                return f"I'd be happy to help you understand: '{text}'\n\nThis appears to be a question or request for explanation. I'm here to assist with your studies! (Note: This is a mock response - integrate with your preferred LLM for actual AI responses)"
            
            elif any(keyword in text_lower for keyword in ['quiz', 'test', 'practice']):
                return "Great! I can help you practice and create quizzes. Send me your study material and I'll help you prepare! (Note: This is a mock response - integrate with your preferred LLM for actual AI responses)"
            
            elif any(keyword in text_lower for keyword in ['summary', 'summarize', 'key points']):
                return "I can help summarize your study materials! Please send me the text or images of your notes, and I'll create a concise summary for you. (Note: This is a mock response - integrate with your preferred LLM for actual AI responses)"
            
            else:
                return f"I received your message: '{text}'\n\nI'm your AI study assistant! I can help you with:\n• Extracting text from images of notes\n• Transcribing audio recordings\n• Answering questions about your study materials\n• Creating summaries and quizzes\n\nHow can I assist you today? (Note: This is a mock response - integrate with your preferred LLM for actual AI responses)"
                
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I encountered an error while processing your message. Please try again."

# Create a global instance
ai_processor = AIProcessor()