from fastapi import FastAPI, Form, Request, HTTPException, BackgroundTasks
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

def send_pdf_via_twilio(to_phone: str, from_phone: str, pdf_path: str, base_url: str):
    """
    Send PDF file via Twilio WhatsApp API
    This runs as a background task after the webhook response
    """
    try:
        # Use public_base_url from settings if available, otherwise use request base_url
        if settings.public_base_url:
            base_url_str = settings.public_base_url.rstrip('/')
        else:
            base_url_str = str(base_url).rstrip('/')
        
        pdf_url = f"{base_url_str}/{pdf_path}"
        
        logger.info(f"Sending PDF via Twilio: {pdf_url} to {to_phone}")
        
        # Send message with media via Twilio API
        # Note: WhatsApp supports PDF files via media_url
        message = twilio_client.messages.create(
            from_=from_phone,
            to=to_phone,
            body="📄 Here's your study notes PDF!",
            media_url=[pdf_url]
        )
        
        logger.info(f"PDF sent successfully via Twilio. Message SID: {message.sid}")
        
    except Exception as e:
        logger.error(f"Error sending PDF via Twilio: {e}", exc_info=True)
        # Try to send a fallback message with the URL
        try:
            if settings.public_base_url:
                base_url_str = settings.public_base_url.rstrip('/')
            else:
                base_url_str = str(base_url).rstrip('/')
            pdf_url = f"{base_url_str}/{pdf_path}"
            
            twilio_client.messages.create(
                from_=from_phone,
                to=to_phone,
                body=f"📄 Your PDF is ready! Download it here: {pdf_url}"
            )
        except Exception as e2:
            logger.error(f"Error sending fallback message: {e2}")

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
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    To: str = Form(...),
    Body: Optional[str] = Form(None),
    MediaUrl0: Optional[str] = Form(None),
    MediaContentType0: Optional[str] = Form(None),
    MediaUrl1: Optional[str] = Form(None),
    MediaContentType1: Optional[str] = Form(None),
    MediaUrl2: Optional[str] = Form(None),
    MediaContentType2: Optional[str] = Form(None),
    MediaUrl3: Optional[str] = Form(None),
    MediaContentType3: Optional[str] = Form(None),
    NumMedia: Optional[str] = Form("0")
):
    """
    WhatsApp webhook endpoint to handle incoming messages
    
    This endpoint receives all types of WhatsApp messages (text, images, audio)
    and processes them accordingly using AI services. Now supports multiple images.
    """
    try:
        logger.info(f"Received WhatsApp message from {From} to {To}")
        logger.info(f"Message body: {Body}")
        logger.info(f"Number of media files: {NumMedia}")
        
        # Create TwiML response object
        response = MessagingResponse()
        
        # Check if there's media attached
        num_media = int(NumMedia) if NumMedia else 0
        
        if num_media > 0:
            # Collect all image URLs (Twilio supports up to 10 media items, we'll handle up to 4)
            image_urls = []
            media_types = []
            
            # Get form data to access all media fields dynamically
            form_data = await request.form()
            
            # Check all possible media slots (Twilio uses MediaUrl0, MediaUrl1, etc.)
            for i in range(min(num_media, 4)):  # Support up to 4 images
                media_url = form_data.get(f"MediaUrl{i}")
                media_type = form_data.get(f"MediaContentType{i}")
                
                if media_url and media_type:
                    media_types.append(media_type)
                    if media_type.startswith('image/'):
                        image_urls.append(media_url)
                        logger.info(f"Found image {i+1}: {media_url}")
            
            if image_urls:
                logger.info(f"Processing {len(image_urls)} image file(s)")
                # Use the first image URL as primary, pass others as additional
                primary_url = image_urls[0]
                additional_urls = image_urls[1:] if len(image_urls) > 1 else None
                
                ai_response = ai_processor.extract_text_from_image(
                    primary_url, 
                    From, 
                    image_urls=additional_urls
                )
                
                if ai_response.get("pdf_path"):
                    # Send the text message first
                    response.message(ai_response['message'])
                    
                    # Schedule PDF to be sent via Twilio API after webhook response
                    # This runs in the background after the response is sent
                    background_tasks.add_task(
                        send_pdf_via_twilio,
                        From, 
                        To,
                        ai_response['pdf_path'],
                        request.base_url
                    )

                else:
                    # Send the error message if PDF generation failed
                    response.message(ai_response['message'])
            
            # Handle audio files (single audio for now)
            elif MediaContentType0 and MediaContentType0.startswith('audio/'):
                logger.info("Processing audio file")
                ai_response = ai_processor.transcribe_audio(MediaUrl0, From)
                response.message(ai_response['message'])
            
            # Handle other media types
            else:
                logger.warning(f"Unsupported media type(s): {media_types}")
                response.message(
                    f"I received {num_media} media file(s), but I can only process images and audio files at the moment. "
                    "Please send me image(s) of your notes or an audio recording, and I'll help you with your studies!"
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
                "Hi! I'm Cortexa, your AI assistant! 🤖\n\n"
                "I can help you with:\n"
                "📸 Extract text from images (supports multiple images!)\n"
                "🎤 Transcribe audio recordings\n"
                "❓ Answer ANY questions (study-related or general)\n"
                "📝 Create visual study notes with mind maps\n"
                "🧠 Generate summaries and practice quizzes\n\n"
                "Just send me a text message, image(s), or audio recording to get started!"
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