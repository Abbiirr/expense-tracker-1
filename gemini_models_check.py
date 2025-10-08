import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment or `./.env`")

genai.configure(api_key=API_KEY)

def list_gemini_models():
    for m in genai.list_models():
        print("Name:", m.name)
        print("Description:", getattr(m, "description", None))
        print("Supported generation methods:", getattr(m, "supported_generation_methods", None))
        print("---")

if __name__ == "__main__":
    list_gemini_models()
