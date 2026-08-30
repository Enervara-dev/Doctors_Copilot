from __future__ import annotations

"""
llm.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import os

# API configuration keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default model definitions
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Generation tuning variables
DEFAULT_TEMPERATURE = 0.2

# Gemini 2.5 models spend "thinking" tokens out of the same budget as the reply, so a
# cap sized for the JSON alone truncates it mid-string: the response then fails schema
# validation and the pipeline silently serves canned fallback text instead. Leave
# headroom for both.
MAX_RESPONSE_TOKENS = int(os.getenv("CLINICAL_COPILOT_MAX_RESPONSE_TOKENS", "4096"))

# Optional cap on Gemini thinking tokens. Unset by default (the model decides);
# set to 0 to turn thinking off for these strictly-schema'd extraction tasks.
THINKING_BUDGET_TOKENS = (
    int(os.environ["CLINICAL_COPILOT_THINKING_BUDGET"])
    if os.getenv("CLINICAL_COPILOT_THINKING_BUDGET") not in (None, "")
    else None
)

# Session-level prompt/result caching TTL (seconds)
CACHE_TTL = 300
