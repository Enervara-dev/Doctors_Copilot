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
os.environ["HF_HUB_OFFLINE"] = "1"

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
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
async def get_transcripts(session_id: str, include_partials: bool = False):
    """
    Retrieves buffered transcript events for a given session.
    If include_partials is False (default), only final (committed) events are returned.
    """
    from app.services.transcript_bus import transcript_bus
    events = transcript_bus.get_recent_events(session_id)
    if not include_partials:
        events = [e for e in events if e.is_final]
    return [event.dict() for event in events]

@app.get("/v1/clinical-state/{session_id}", status_code=status.HTTP_200_OK)
async def get_clinical_state(session_id: str):
    """
    Retrieves the accumulated incremental structured clinical state for a session.
    """
    from app.services.clinical import clinical_state_engine
    state = clinical_state_engine.get_state(session_id)
    return state.model_dump()

@app.get("/v1/clinical-state/{session_id}/provenance", status_code=status.HTTP_200_OK)
async def get_clinical_state_provenance(session_id: str):
    """
    Retrieves the internal fact provenance metadata dictionary for clinical state auditing.
    """
    from app.services.clinical import clinical_state_engine
    return clinical_state_engine.get_provenance(session_id)

@app.get("/v1/clinical-reasoning/{session_id}", status_code=status.HTTP_200_OK)
async def get_clinical_reasoning(session_id: str):
    """
    Retrieves the clinical reasoning results (candidate diagnoses, red flags,
    interactions, and recommendations) for a session.
    """
    from clinical_intelligence.integration import get_reasoning_result
    result = get_reasoning_result(session_id)
    if not result:
        return {"message": "No reasoning results available yet."}
    return result.model_dump()
