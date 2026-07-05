import os
from app.core.config import settings

# Attempt to load settings from app settings (if they exist) or environment variables, falling back to sensible defaults.
ENABLE_SPEAKER_VERIFICATION = os.getenv(
    "CLINICAL_COPILOT_ENABLE_SPEAKER_VERIFICATION", 
    str(getattr(settings, "ENABLE_SPEAKER_VERIFICATION", True))
).lower() in ("true", "1", "yes")

SPEAKER_MATCH_THRESHOLD = float(os.getenv(
    "CLINICAL_COPILOT_SPEAKER_MATCH_THRESHOLD", 
    str(getattr(settings, "SPEAKER_MATCH_THRESHOLD", 0.85))
))

UNKNOWN_THRESHOLD = float(os.getenv(
    "CLINICAL_COPILOT_UNKNOWN_THRESHOLD", 
    str(getattr(settings, "UNKNOWN_THRESHOLD", 0.70))
))

MIN_SEGMENT_DURATION = float(os.getenv(
    "CLINICAL_COPILOT_MIN_SEGMENT_DURATION", 
    str(getattr(settings, "MIN_SEGMENT_DURATION", 0.8))
))

MAX_UNKNOWN_SPEAKERS = int(os.getenv(
    "CLINICAL_COPILOT_MAX_UNKNOWN_SPEAKERS", 
    str(getattr(settings, "MAX_UNKNOWN_SPEAKERS", 5))
))

MINIMUM_CONFIRMATIONS = int(os.getenv(
    "CLINICAL_COPILOT_MINIMUM_CONFIRMATIONS", 
    str(getattr(settings, "MINIMUM_CONFIRMATIONS", 3))
))

UNKNOWN_EXPIRY_SECONDS = float(os.getenv(
    "CLINICAL_COPILOT_UNKNOWN_EXPIRY_SECONDS", 
    str(getattr(settings, "UNKNOWN_EXPIRY_SECONDS", 600.0))
))

MIN_RMS_THRESHOLD = float(os.getenv(
    "CLINICAL_COPILOT_MIN_RMS_THRESHOLD", 
    str(getattr(settings, "MIN_RMS_THRESHOLD", 0.005))
))

CLIPPING_THRESHOLD = float(os.getenv(
    "CLINICAL_COPILOT_CLIPPING_THRESHOLD", 
    str(getattr(settings, "CLIPPING_THRESHOLD", 0.98))
))
