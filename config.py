"""
Steam Kullanıcı Analiz Sistemi - Yapılandırma
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============ API KEYS ============
STEAM_API_KEY = os.getenv('STEAM_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# ============ STEAM API ============
STEAM_API_BASE_URL = "https://api.steampowered.com"

# ============ LLM AYARLARI ============
LLM_PROVIDER = "groq"  # 'groq' veya 'ollama'
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq modeli
OLLAMA_URL = "http://localhost:11434"  # Ollama local server
OLLAMA_MODEL = "llama2"

# ============ FLASK ============
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = 5000
