"""
FastAPI Server Core Entrypoint

Purpose:
Configures and boots the FastAPI web server, loading global middleware, API routes, WebSocket controllers, and background diarization queues.

Module/Component: API Core
"""

"""
FastAPI Server Core Entrypoint

Purpose:
Configures and boots the FastAPI web server, loading global middleware, API routes, WebSocket controllers, and background diarization queues.

Module/Component: API Core
"""

import os

# Hugging Face hub runs offline by default so a warm deployment never reaches the
# network on startup. Models must be in the local cache first: set
# HF_HUB_OFFLINE=0 for the initial download of the pyannote and ECAPA weights.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# `app.core.config.Settings` reads `.env` on its own (pydantic-settings'
# `env_file` mechanism reads the file directly, independent of os.environ).
# Nothing else in this process did that for plain `os.getenv()` calls, which
# is how they're read everywhere outside that one Settings class — including
# `app.services.copilot_bridge`. Without this, COPILOT_BRIDGE_ENABLED in .env
# was invisible to os.getenv() unless some unrelated route had already
# imported clinical_intelligence.config (which calls load_dotenv() as a side
# effect) before the bridge singleton was first constructed — an accident of
# import order, not a guarantee. Calling it here, first, removes that
# dependency on what happened to run before it.
from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from fastapi import FastAPI, Header, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.security import require_session_auth
from app.api.transcript_stream import router as transcript_router
from app.api.websocket import router as ws_router
from app.services.session import session_manager
from app.services.stt_manager import stt_manager
from app.utils.logging import logger
import time

app = FastAPI(
    title=settings.APP_NAME,
    description="Low-latency real-time PCM audio streaming ingestion layer.",
    version="1.0.0"
)

# Standard CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production domain restrictions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ws_router, prefix=settings.API_V1_STR)
# Read-only egress for the transcripts the STT workers publish. Audio goes in
# over the WebSocket above; without this nothing came back out.
app.include_router(transcript_router, prefix=settings.API_V1_STR)

# Startup logging
@app.on_event("startup")
async def startup_event():
    logger.info(
        "Application starting up",
        extra={
            "app_name": settings.APP_NAME,
            "sample_rate_hz": settings.AUDIO_SAMPLE_RATE,
            "min_frame_bytes": settings.min_frame_bytes,
            "max_frame_bytes": settings.max_frame_bytes,
            "buffer_chunk_bytes": settings.buffer_chunk_bytes,
            "diarization_mode": settings.DIARIZATION_MODE
        }
    )
    # Start background STT worker reconciliation orchestrator loop
    stt_manager.start_orchestrator()

    # Activate clinical state engine subscription
    from app.services.clinical import clinical_state_engine

    # Preload Pyannote diarization pipeline in production mode
    if settings.DIARIZATION_MODE == "production":
        from app.services.diarization_worker import diarization_worker_manager
        await diarization_worker_manager.preload_pipeline()

    # Preload and warm Speaker Verification backend if enabled
    from app.services.speaker_verification.config import ENABLE_SPEAKER_VERIFICATION
    if ENABLE_SPEAKER_VERIFICATION:
        from app.services.speaker_verification.speaker_verification import speaker_verification_service
        speaker_verification_service.load_backend()


# Shutdown logging
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")
    # Clean up and stop background STT workers
    await stt_manager.stop_orchestrator()

# Production Health Check Endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Returns app health, including count of active sessions for orchestration routing.
    """
    active_sessions = list(session_manager.list_active_sessions())
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions_count": len(active_sessions),
        "active_sessions": active_sessions
    }

# Dynamic Verification Endpoint for Step 2 Transcripts
@app.get("/v1/transcripts/{session_id}", status_code=status.HTTP_200_OK)
async def get_transcripts(
    session_id: str,
    include_partials: bool = False,
    authorization: Optional[str] = Header(default=None),
):
    """
    Retrieves buffered transcript events for a given session.
    If include_partials is False (default), only final (committed) events are returned.

    Requires the same bearer credential (shared token or JWT) a participant used
    to open the WebSocket for this session — a session id alone is no longer
    sufficient to read another session's transcript.
    """
    require_session_auth(session_id, authorization)
    from app.services.transcript_bus import transcript_bus
    events = transcript_bus.get_recent_events(session_id)
    if not include_partials:
        events = [e for e in events if e.is_final]
    return [event.dict() for event in events]

@app.get("/v1/clinical-state/{session_id}", status_code=status.HTTP_200_OK)
async def get_clinical_state(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Retrieves the accumulated incremental structured clinical state for a session.
    Requires a bearer credential valid for this session.
    """
    require_session_auth(session_id, authorization)
    from app.services.clinical import clinical_state_engine
    state = clinical_state_engine.get_state(session_id)
    return state.model_dump()

@app.get("/v1/clinical-state/{session_id}/provenance", status_code=status.HTTP_200_OK)
async def get_clinical_state_provenance(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Retrieves the internal fact provenance metadata dictionary for clinical state auditing.
    Requires a bearer credential valid for this session.
    """
    require_session_auth(session_id, authorization)
    from app.services.clinical import clinical_state_engine
    return clinical_state_engine.get_provenance(session_id)

@app.get("/v1/clinical-reasoning/{session_id}", status_code=status.HTTP_200_OK)
async def get_clinical_reasoning(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Retrieves the clinical reasoning results (candidate diagnoses, red flags,
    interactions, and recommendations) for a session.
    Requires a bearer credential valid for this session.
    """
    require_session_auth(session_id, authorization)
    from clinical_intelligence.integration import get_reasoning_result
    from clinical_intelligence.presentation.serializer import serialize_reasoning_result
    from clinical_intelligence.config.runtime import ENABLE_RED_FLAG_ALERTS

    result = get_reasoning_result(session_id)
    if not result:
        return {"message": "No reasoning results available yet."}
    return serialize_reasoning_result(result, show_red_flags=ENABLE_RED_FLAG_ALERTS)
