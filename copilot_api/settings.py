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
from dataclasses import dataclass
from typing import Optional


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
    allow_insecure_auth: bool = False

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
        )

    def describe(self) -> dict:
        """What to log at boot. Contains no secret and never will."""
        return {
            "intelligence_enabled": self.intelligence_enabled,
            "intelligence_provider": self.intelligence_provider,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "intelligence_budget_ms": self.intelligence_budget_ms,
            "session_idle_timeout_seconds": self.session_idle_timeout_seconds,
            "allow_insecure_auth": self.allow_insecure_auth,
        }
