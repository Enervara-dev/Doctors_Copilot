"""
Token Validation & Security Utilities

Purpose:
Handles API authentication token validation, signature checks, and role-based access control (Doctor, Patient, Attender).

Module/Component: API Core
"""

"""
Token Validation & Security Utilities

Purpose:
Handles API authentication token validation, signature checks, and role-based access control (Doctor, Patient, Attender).

Module/Component: API Core
"""

from typing import Optional
from app.core.config import settings
from app.utils.logging import logger
from app.utils.jwt_utils import decode_jwt_token

def verify_session_token(
    token: str,
    expected_session_id: Optional[str] = None,
    expected_role: Optional[str] = None
) -> bool:
    """
    Validates the connection token. Supports both legacy AUTH_SECRET_TOKEN verification 
    and production-safe JWT claim verification (checking expiration and role/session matches).
    """
    if not token:
        logger.warning("Authentication failed: Missing token", extra={"event": "auth_failure", "reason": "missing_token"})
        return False
        
    # 1. Legacy auth check (for backward compatibility)
    if token == settings.AUTH_SECRET_TOKEN:
        return True
        
    # 2. JWT auth check
    payload = decode_jwt_token(token)
    if payload is not None:
        # If expected claims are provided, enforce matching logic
        if expected_session_id and payload.get("session_id") != expected_session_id:
            logger.warning(
                "JWT verification failed: Session ID mismatch",
                extra={"expected": expected_session_id, "received": payload.get("session_id")}
            )
            return False
            
        if expected_role and payload.get("role") != expected_role:
            logger.warning(
                "JWT verification failed: Role mismatch",
                extra={"expected": expected_role, "received": payload.get("role")}
            )
            return False
            
        return True
        
    logger.warning("Authentication failed: Invalid token or JWT claims", extra={"event": "auth_failure", "reason": "invalid_token"})
    return False

def verify_participant_role(role: str) -> bool:
    """
    Ensures that the client identifies as a valid consultation participant.
    """
    if role not in settings.ALLOWED_ROLES:
        logger.warning("Authorization failed: Invalid role", extra={"event": "auth_failure", "reason": "invalid_role", "role": role})
        return False
    return True


def require_session_auth(session_id: str, authorization: Optional[str]) -> None:
    """
    Same rule the WebSocket ingress has enforced since before this function
    existed, applied to the read-side HTTP endpoints that never had it.

    A GET on `/v1/transcripts/{id}`, `/v1/clinical-state/{id}`, its
    `/provenance` sibling, or `/v1/clinical-reasoning/{id}` used to require
    nothing but a guessable session id. This closes that: the caller must
    present the same shared token or JWT that a participant used to open the
    WebSocket for this session, checked against this exact session id via the
    same `verify_session_token` used there.

    Raises `HTTPException` rather than returning a bool because every caller
    is a FastAPI route dependency and wants the 401 shaped consistently.
    """
    from fastapi import HTTPException, status as http_status

    if not authorization:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Expected an Authorization: Bearer credential.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_session_token(token.strip(), expected_session_id=session_id):
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="This credential is not valid for this session.",
        )
