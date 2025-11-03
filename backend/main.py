from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import logging
from typing import Optional
import uvicorn
import os

# Import our custom modules
from core.config import settings
from services.ai_processor import ai_processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="WhatsApp-based AI Study Assistant Backend",
    version="1.0.0"
)

# Create static directory if it doesn't exist
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# Mount the static directory to serve files
app.mount(f"/{static_dir}", StaticFiles(directory=static_dir), name=static_dir)

# Initialize Twilio client
twilio_client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "status": "ok",
        "message": "WhatsApp AI Study Assistant is running",
        "app_name": settings.app_name
    }


@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    To: str = Form(...),
    Body: Optional[str] = Form(None),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None),
    NumMedia: Optional[str] = Form("0")
):
    """
    WhatsApp webhook endpoint to handle incoming messages
    
    This endpoint receives all types of WhatsApp messages (text, images, audio)
    and processes them accordingly using AI services.
    """
    try:
        logger.info(f"Received WhatsApp message from {From} to {To}")
        logger.info(f"Message body: {Body}")
        logger.info(f"Number of media files: {NumMedia}")
        
        # Create TwiML response object
        response = MessagingResponse()
        
        # Check if there's media attached
        if MediaUrl0 and MediaContentType0:
            logger.info(f"Processing media: {MediaContentType0}")
            
            # Handle image files
            if MediaContentType0.startswith('image/'):
                logger.info("Processing image file")
                ai_response = ai_processor.extract_text_from_image(MediaUrl0, From)
                
                if ai_response.get("pdf_path"):
                    # Construct the full, public URL for the PDF
                    pdf_url = f"{str(request.base_url).rstrip('/')}/{ai_response['pdf_path']}"
                    
                    # Send both the message and PDF link
                    message = response.message()
                    message.body(ai_response['message'])
                    message.media(pdf_url)
                    
                    logger.info(f"Sending PDF URL: {pdf_url}")
                else:
                    # Send the error message if PDF generation failed
                    response.message(ai_response['message'])
            
            # Handle audio files
            elif MediaContentType0.startswith('audio/'):
                logger.info("Processing audio file")
                ai_response = ai_processor.transcribe_audio(MediaUrl0, From)
                response.message(ai_response['message'])
            
            # Handle other media types
            else:
                logger.warning(f"Unsupported media type: {MediaContentType0}")
                response.message(
                    f"I received a {MediaContentType0} file, but I can only process images and audio files at the moment. "
                    "Please send me an image of your notes or an audio recording, and I'll help you with your studies!"
                )
        
        # Handle text messages
        elif Body:
            logger.info("Processing text message")
            ai_response = ai_processor.get_chat_response(Body, From)
            response.message(ai_response)
        
        # Handle empty messages
        else:
            logger.warning("Received empty message")
            response.message(
                "Hi! I'm your AI study assistant. I can help you with:\n\n"
                "- Extract text from images of your notes\n"
                "- Transcribe your audio recordings\n"
                "- Answer questions about your study materials\n"
                "- Create summaries and practice quizzes\n\n"
                "Just send me a text message, image, or audio recording to get started!"
            )
        
        # Log the response
        logger.info("Sending TwiML response back to Twilio")
        logger.info(f"TwiML Response: {str(response)}")
        
        # Return TwiML response
        return Response(
            content=str(response),
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}", exc_info=True)
        
        # Send error message to user
        error_response = MessagingResponse()
        error_response.message(
            "I'm sorry, I encountered an error while processing your message. "
            "Please try again in a moment. If the problem persists, please contact support."
        )
        
        return Response(
            content=str(error_response),
            media_type="application/xml"
        )

@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint
    """
    try:
        # Test Twilio connection
        account = twilio_client.api.accounts(settings.twilio_account_sid).fetch()
        
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "twilio_connected": True,
            "twilio_account_status": account.status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "app_name": settings.app_name,
            "twilio_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )