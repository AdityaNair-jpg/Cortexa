import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your .env file to get the API key
load_dotenv()

try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env file.")
    else:
        genai.configure(api_key=api_key)
        
        print("Available models for your library version:")
        
        # This is how you call ListModels
        for model in genai.list_models():
            print(f"- {model.name}")

except Exception as e:
    print(f"An error occurred: {e}")