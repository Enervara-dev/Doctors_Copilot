"""
Configuration for the Copilot service.

Two rules shape the defaults.

**The clinical core needs nothing.** Every intelligence setting is optional and
the layer is off unless switched on. A deployment that sets no variable at all
starts, serves the full deterministic clinical API, and answers
`/sessions/{id}/intelligence` with a clean 404. That is the configuration every
existing test runs in, and it is the one to fall back to when anything about
the model layer is uncertain.

**No secret is defined here.** Keys are read from the environment by the
modules that already own them — `clinical_intelligence/config/llm.py` for the
model, `app/core/config.py` for the session token and JWT secret. This file
decides behaviour, not credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _number(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _csv(name: str) -> List[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_production_environment() -> bool:
    """
    Reads `ENVIRONMENT` directly rather than through `CopilotSettings`, so the
    one check that must never be bypassable (`check_auth_secret` in
    `bootstrap.py`) does not depend on the same settings object a test freely
    constructs with whatever it likes. `ENVIRONMENT` unset or anything other
    than exactly "production" is treated as non-production — the safer
    direction to default in test and local-dev contexts, where this function
    is not on the path that decides whether the service starts.
    """
    return os.getenv("ENVIRONMENT", "").strip().lower() == "production"


@dataclass(frozen=True)
class CopilotSettings:
    """Resolved once at startup and logged, so a deployment is self-describing."""

    #: Master switch. Off means no worker thread and no model client at all.
    intelligence_enabled: bool = False

    #: null | deterministic | langgraph.
    #: `deterministic` restates rule output and calls no model — useful for
    #: exercising the whole path in a staging environment with no key.
    intelligence_provider: str = "langgraph"

    #: gemini | openai. Only consulted when the provider is `langgraph`.
    model_provider: str = "gemini"
    model_name: Optional[str] = None
    model_temperature: float = 0.2

    #: Advisory. A synchronous provider cannot be interrupted, only measured,
    #: so this marks a snapshot as late rather than cancelling it.
    intelligence_budget_ms: float = 8000.0

    #: How many consultations keep a latest snapshot in memory.
    intelligence_capacity: int = 256

    #: Mirrors the runtime's own default; set both together or neither.
    session_idle_timeout_seconds: float = 30 * 60

    #: Permits startup with the placeholder JWT secret. Local development only:
    #: that secret is public, so a deployment using it authenticates nobody.
    #: `bootstrap.check_auth_secret` additionally refuses this override
    #: outright whenever `ENVIRONMENT=production` — see
    #: `is_production_environment` above. This field is what a non-production
    #: deployment sets; it is not itself the guarantee.
    allow_insecure_auth: bool = False

    #: Informational only — logged at boot so a deployment states what it
    #: believes it is. `is_production_environment()` re-reads the same
    #: variable independently for the one decision that must not be bypassable.
    environment: str = "development"

    #: Origins allowed to make browser-based (CORS) requests. Empty means no
    #: CORS headers are ever sent — the correct default for a service the
    #: Doctor platform calls server-to-server, never from a browser. Only set
    #: this if a browser client will call the API directly.
    cors_allowed_origins: List[str] = field(default_factory=list)

    #: `Host` header values this deployment accepts, enforced by
    #: TrustedHostMiddleware. Empty (the default) disables the check —
    #: correct for local development and for a first deployment before the
    #: real hostname is known, but a real deployment should set this to the
    #: ALB DNS name / custom domain to reject host-header spoofing.
    trusted_hosts: List[str] = field(default_factory=list)

    #: Path to a SQLite file for consultation state snapshots. Empty (the
    #: default) means no persistence: state is in-memory only and a restart
    #: loses every in-progress consultation. See `copilot_api/persistence.py`
    #: for what is and is not covered.
    state_db_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "CopilotSettings":
        return cls(
            intelligence_enabled=_flag("COPILOT_INTELLIGENCE_ENABLED", False),
            intelligence_provider=os.getenv(
                "COPILOT_INTELLIGENCE_PROVIDER", "langgraph"
            ).strip().lower(),
            model_provider=os.getenv("COPILOT_MODEL_PROVIDER", "gemini").strip().lower(),
            model_name=(os.getenv("COPILOT_MODEL_NAME") or None),
            model_temperature=_number("COPILOT_MODEL_TEMPERATURE", 0.2),
            intelligence_budget_ms=_number("COPILOT_INTELLIGENCE_BUDGET_MS", 8000.0),
            intelligence_capacity=int(_number("COPILOT_INTELLIGENCE_CAPACITY", 256)),
            session_idle_timeout_seconds=_number(
                "COPILOT_SESSION_IDLE_TIMEOUT_SECONDS", 30 * 60
            ),
            allow_insecure_auth=_flag("COPILOT_ALLOW_INSECURE_AUTH", False),
            environment=os.getenv("ENVIRONMENT", "development").strip().lower(),
            cors_allowed_origins=_csv("COPILOT_CORS_ALLOWED_ORIGINS"),
            trusted_hosts=_csv("COPILOT_TRUSTED_HOSTS"),
            state_db_path=(os.getenv("COPILOT_STATE_DB_PATH") or None),
        )

    def describe(self) -> dict:
        """What to log at boot. Contains no secret and never will."""
        return {
            "environment": self.environment,
            "intelligence_enabled": self.intelligence_enabled,
            "intelligence_provider": self.intelligence_provider,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "intelligence_budget_ms": self.intelligence_budget_ms,
            "session_idle_timeout_seconds": self.session_idle_timeout_seconds,
            "allow_insecure_auth": self.allow_insecure_auth,
            "cors_allowed_origins": self.cors_allowed_origins,
            "trusted_hosts": self.trusted_hosts,
            "state_persistence_enabled": bool(self.state_db_path),
        }
