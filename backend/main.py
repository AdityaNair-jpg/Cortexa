from fastapi import FastAPI, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import logging
from typing import Optional, Dict, List
import uvicorn
import os
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

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

# Image batching for multiple images sent together
# Stores pending images per user phone number
pending_images: Dict[str, Dict] = defaultdict(dict)
processing_tasks: Dict[str, bool] = {}  # Track if processing task is running
BATCH_WAIT_TIME = 3  # Wait 3 seconds for additional images

async def process_batched_images(user_key: str, from_phone: str, to_phone: str, base_url: str):
    """
    Wait for batch window, then process all collected images together
    """
    try:
        # Wait for the initial batch window
        await asyncio.sleep(BATCH_WAIT_TIME)
        
        # Keep checking until no new images arrive for BATCH_WAIT_TIME seconds
        max_wait_cycles = 5  # Maximum 5 cycles (15 seconds total)
        for cycle in range(max_wait_cycles):
            if user_key not in pending_images:
                logger.warning(f"Batch for {user_key} was already processed or cleared")
                if user_key in processing_tasks:
                    processing_tasks[user_key] = False
                return
            
            batch_data = pending_images[user_key]
            last_seen = batch_data.get('last_seen', batch_data.get('first_seen'))
            current_time = datetime.now()
            time_since_last = (current_time - last_seen).total_seconds()
            
            logger.info(f"Batch check cycle {cycle+1}/{max_wait_cycles} for {user_key}: {len(batch_data['images'])} images, {time_since_last:.1f}s since last image")
            
            # If no new images in the wait time, process the batch
            if time_since_last >= BATCH_WAIT_TIME:
                logger.info(f"No new images for {BATCH_WAIT_TIME}s, processing batch now")
                break
            
            # Wait a bit more for additional images
            remaining_wait = BATCH_WAIT_TIME - time_since_last
            if remaining_wait > 0:
                await asyncio.sleep(min(remaining_wait, 1))
        
        # Get final images and remove from pending
        if user_key not in pending_images:
            logger.warning(f"Batch for {user_key} was cleared before processing")
            if user_key in processing_tasks:
                processing_tasks[user_key] = False
            return
        
        final_images = pending_images[user_key]['images'].copy()
        final_from = pending_images[user_key].get('from', from_phone)
        final_to = pending_images[user_key].get('to', to_phone)
        del pending_images[user_key]
        if user_key in processing_tasks:
            processing_tasks[user_key] = False
        
        if not final_images:
            logger.warning(f"No images to process for batch {user_key}")
            return
        
        logger.info(f"Processing batched images for {user_key}: {len(final_images)} image(s)")
        
        # Process all images together
        primary_url = final_images[0]
        additional_urls = final_images[1:] if len(final_images) > 1 else None
        
        ai_response = ai_processor.extract_text_from_image(
            primary_url,
            final_from,
            image_urls=additional_urls
        )
        
        if ai_response.get("pdf_path"):
            # Send status message
            try:
                twilio_client.messages.create(
                    from_=final_to,
                    to=final_from,
                    body=f"✅ Processed {len(final_images)} image(s)! Generating your PDF..."
                )
            except:
                pass
            
            # Send PDF
            send_pdf_via_twilio(
                final_from,
                final_to,
                ai_response['pdf_path'],
                base_url
            )
        else:
            # Send error message
            try:
                twilio_client.messages.create(
                    from_=final_to,
                    to=final_from,
                    body=ai_response.get('message', 'Failed to process images. Please try again.')
                )
            except:
                pass
            
    except Exception as e:
        logger.error(f"Error processing batched images for {user_key}: {e}", exc_info=True)
        # Clean up on error
        if user_key in pending_images:
            del pending_images[user_key]
        if user_key in processing_tasks:
            processing_tasks[user_key] = False
        # Send error message to user
        try:
            twilio_client.messages.create(
                from_=to_phone,
                to=from_phone,
                body="Sorry, I encountered an error processing your images. Please try again."
            )
        except:
            pass

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
            max_images = 4  # Support up to 4 images
            for i in range(min(num_media, max_images)):
                media_url = form_data.get(f"MediaUrl{i}")
                media_type = form_data.get(f"MediaContentType{i}")
                
                if media_url and media_type:
                    media_types.append(media_type)
                    if media_type.startswith('image/'):
                        image_urls.append(media_url)
                        logger.info(f"Found image {i+1}/{num_media}: {media_url[:50]}...")
            
            if image_urls:
                logger.info(f"Processing {len(image_urls)} image file(s) out of {num_media} total media")
                
                # If this is a single image in a batch (NumMedia=1), batch them together
                if num_media == 1:
                    user_key = From
                    current_time = datetime.now()
                    
                    # Check if there's an existing batch for this user
                    if user_key in pending_images:
                        # Add to existing batch and reset timer
                        pending_images[user_key]['images'].extend(image_urls)
                        pending_images[user_key]['last_seen'] = current_time
                        logger.info(f"Added image to existing batch for {user_key}. Total: {len(pending_images[user_key]['images'])} images")
                        response.message(f"📸 Got it! Received {len(pending_images[user_key]['images'])} image(s), waiting a moment for more...")
                    else:
                        # Start new batch
                        pending_images[user_key] = {
                            'images': image_urls.copy(),
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'to': To,
                            'from': From
                        }
                        logger.info(f"Starting new image batch for {user_key}, waiting {BATCH_WAIT_TIME}s for more images")
                        
                        # Only start processing task if one isn't already running
                        if user_key not in processing_tasks or not processing_tasks[user_key]:
                            processing_tasks[user_key] = True
                            # Wait for additional images, then process
                            background_tasks.add_task(
                                process_batched_images,
                                user_key,
                                From,
                                To,
                                request.base_url
                            )
                        
                        # Send immediate acknowledgment
                        response.message("📸 Got it! Processing your image(s)...")
                else:
                    # Multiple images in one webhook - process immediately
                    logger.info(f"Received {num_media} images in single webhook, processing immediately")
                    
                    # Warn if more than max_images were sent
                    if num_media > max_images:
                        logger.warning(f"Received {num_media} media files, but only processing first {max_images} images")
                    
                    # Use the first image URL as primary, pass others as additional
                    primary_url = image_urls[0]
                    additional_urls = image_urls[1:] if len(image_urls) > 1 else None
                    
                    ai_response = ai_processor.extract_text_from_image(
                        primary_url, 
                        From, 
                        image_urls=additional_urls
                    )
                    
                    # Add note if some images were skipped
                    if num_media > max_images and ai_response.get("message"):
                        ai_response["message"] += f"\n\nNote: Processed first {len(image_urls)} images (max {max_images} supported)."
                    
                    if ai_response.get("pdf_path"):
                        # Send the text message first
                        response.message(ai_response['message'])
                        
                        # Schedule PDF to be sent via Twilio API after webhook response
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