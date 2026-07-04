from __future__ import annotations
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default model definitions
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Generation tuning variables
DEFAULT_TEMPERATURE = 0.2
MAX_RESPONSE_TOKENS = 1024

# Session-level prompt/result caching TTL (seconds)
CACHE_TTL = 300
