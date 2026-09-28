"""
Authentication for the Copilot transport.

No new scheme is introduced here. `app.core.security.verify_session_token` is
the contract this repository already has: a bearer credential that is either
the configured shared token or a JWT carrying `session_id` and `role` claims,
signed with the key in settings. The WebSocket ingress has authenticated that
way since before the Copilot existed, and a Doctor-issued token can be
validated by the same call once the platform mints one.

What this module adds is the seam: one dependency, applied to every route,
which binds the credential to the consultation being acted on. A token valid
for one consultation is not valid for another — which is the property a
per-consultation transport actually needs, and the reason the token is checked
against the path parameter rather than merely decoded.

No secret is defined here. Keys come from settings, which read the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Path, status

from app.core.security import verify_session_token


@dataclass(frozen=True)
class Caller:
    """Who is acting, and on which consultation."""

    consultation_id: str
    token: str


def _bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHENTICATED", "message": "A bearer token is required."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHENTICATED", "message": "Expected an Authorization: Bearer credential."},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token.strip()


def authorize(consultation_id: str, authorization: Optional[str]) -> Caller:
    """
    Validates a credential against one consultation.

    Kept as a plain function as well as a dependency so the session-creation
    route — where the consultation id arrives in the body rather than the path
    — can apply exactly the same rule instead of a looser one.
    """
    token = _bearer(authorization)
    if not verify_session_token(token, expected_session_id=consultation_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "FORBIDDEN",
                "message": "This credential is not valid for this consultation.",
            },
        )
    return Caller(consultation_id=consultation_id, token=token)


def require_caller(
    consultation_id: str = Path(..., description="The Doctor platform consultation id"),
    authorization: Optional[str] = Header(default=None),
) -> Caller:
    """FastAPI dependency for every route that names a consultation in its path."""
    return authorize(consultation_id, authorization)


CallerDep = Depends(require_caller)
