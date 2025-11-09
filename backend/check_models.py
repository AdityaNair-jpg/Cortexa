import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("GEMINI API DIAGNOSTIC")
print("=" * 70)

# Check library version 
import importlib.metadata
try:
    version = importlib.metadata.version('google-generativeai')
    print(f"\n✓ Library version: {version}")
except:
    print("\n✗ Could not determine library version")

# API Configuration
api_key = os.getenv('GEMINI_API_KEY')
print(f"✓ API key loaded: {api_key[:10]}..." if api_key else "✗ No API key found")

genai.configure(api_key=api_key)

# Try different approaches to list models
print("\n" + "=" * 70)
print("APPROACH 1: List all models")
print("=" * 70)
try:
    for m in genai.list_models():
        print(f"\nModel: {m.name}")
        print(f"  Supported methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"✗ Error: {e}")

# Try direct generation with different model names
print("\n" + "=" * 70)
print("APPROACH 2: Test different model name formats")
print("=" * 70)

test_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro", 
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "gemini-pro",
    "models/gemini-pro"
]

for model_name in test_models:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say hello")
        print(f"✓ {model_name:30s} - WORKS! Response: {response.text[:50]}")
        break  # Stop after first successful model
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"✗ {model_name:30s} - {error_msg}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("If no models worked, try:")
print("1. pip uninstall google-generativeai")
print("2. pip install google-generativeai>=0.3.0")
print("3. Check API key at: https://aistudio.google.com/apikey")
