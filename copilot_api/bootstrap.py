"""
The composition root.

One place where configuration becomes objects: the runtime, the provider, the
gateway, the live service, the brokers and the app. Everything below this file
takes its collaborators as arguments, which is why the whole system is testable
without environment variables — and why this file is the only one that reads
them.

It also owns the worker lifecycle. The intelligence thread starts when the
application starts and is stopped on shutdown, through FastAPI's `lifespan`
rather than the deprecated `on_event` hooks used elsewhere in this repository.
A worker that outlives its process is how a container fails to exit.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Tuple

from fastapi import FastAPI

from copilot_api.main import ResultBroker, SnapshotBroker, create_app
from copilot_api.settings import CopilotSettings
from integration.copilot_runtime import CopilotRuntime
from integration.intelligence.contract import IntelligenceProvider
from integration.intelligence.gateway import IntelligenceGateway
from integration.intelligence.langgraph_provider import LangGraphIntelligenceProvider
from integration.intelligence.live import LiveIntelligenceService
from integration.intelligence.model_clients import build_model_client
from integration.intelligence.providers import (
    DeterministicIntelligenceProvider,
    NullIntelligenceProvider,
)

logger = logging.getLogger(__name__)


def build_provider(settings: CopilotSettings) -> IntelligenceProvider:
    """
    Selects a provider from configuration.

    An unknown name falls back to the null provider rather than raising. A
    typo in a deployment variable should degrade the optional layer, not stop
    a clinical service from booting.
    """
    choice = settings.intelligence_provider

    if choice == "deterministic":
        return DeterministicIntelligenceProvider()

    if choice == "langgraph":
        model = build_model_client(
            provider_name=settings.model_provider,
            model_name=settings.model_name,
            temperature=settings.model_temperature,
        )
        logger.info("Intelligence: langgraph over model client %r", model.name)
        return LangGraphIntelligenceProvider(model)

    if choice != "null":
        logger.warning(
            "Unknown COPILOT_INTELLIGENCE_PROVIDER %r; intelligence is disabled.", choice
        )
    return NullIntelligenceProvider()


def build_intelligence(
    settings: CopilotSettings, snapshots: SnapshotBroker
) -> Optional[LiveIntelligenceService]:
    """Returns None when the layer is switched off — the default."""
    if not settings.intelligence_enabled:
        logger.info("Intelligence layer disabled; serving deterministic output only.")
        return None

    gateway = IntelligenceGateway(
        build_provider(settings), budget_ms=settings.intelligence_budget_ms
    )
    return LiveIntelligenceService(
        gateway, snapshots, capacity=settings.intelligence_capacity
    )


def build_application(
    settings: Optional[CopilotSettings] = None,
) -> Tuple[FastAPI, Optional[LiveIntelligenceService]]:
    """
    Assembles the deployable application.

    Returns the service alongside the app so a caller that manages its own
    lifecycle — a test, an embedder — can do so. Under uvicorn the lifespan
    below handles it.
    """
    resolved = settings or CopilotSettings.from_env()
    logger.info("Copilot configuration: %s", resolved.describe())

    results = ResultBroker()
    snapshots = SnapshotBroker()
    runtime = CopilotRuntime(
        results, idle_timeout_seconds=resolved.session_idle_timeout_seconds
    )
    intelligence = build_intelligence(resolved, snapshots)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if intelligence is not None:
            intelligence.start()
            logger.info("Intelligence worker started.")
        try:
            yield
        finally:
            if intelligence is not None:
                # Give in-flight work a moment to land, then stop regardless.
                # A container that will not exit is worse than a lost snapshot.
                intelligence.wait_until_idle(timeout=5.0)
                intelligence.stop(timeout=5.0)
                logger.info("Intelligence worker stopped.")

    app = create_app(
        runtime,
        broker=results,
        intelligence=intelligence,
        snapshot_broker=snapshots,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    return app, intelligence


#: The value `app/core/config.py` ships when nothing is configured. It is in a
#: public repository, so a deployment still using it will accept a token minted
#: by anyone for any consultation.
PLACEHOLDER_JWT_SECRET = "production-jwt-secret-key-change-me"

#: The settings class carries `env_prefix = "CLINICAL_COPILOT_"`, so the
#: obvious `JWT_SECRET_KEY` is silently ignored. Naming the real variable here
#: is the difference between a one-line fix and an afternoon.
JWT_SECRET_ENV_VAR = "CLINICAL_COPILOT_JWT_SECRET_KEY"


def check_auth_secret(*, allow_insecure: bool = False) -> None:
    """
    Refuses to serve patient data behind a publicly known signing key.

    The placeholder is a working secret, not an empty one, so nothing fails
    without this check — the service comes up and authenticates happily
    against a key anybody can read. That is the failure mode worth making
    loud, and startup is the only place it can still be made loud cheaply.

    `COPILOT_ALLOW_INSECURE_AUTH=true` opts out for local development. It is
    deliberately awkward to set by accident.
    """
    from app.core.config import settings as app_settings

    if app_settings.JWT_SECRET_KEY != PLACEHOLDER_JWT_SECRET:
        return

    if allow_insecure:
        logger.warning(
            "JWT_SECRET_KEY is the shipped placeholder. Tokens are forgeable by "
            "anyone. Permitted only because COPILOT_ALLOW_INSECURE_AUTH is set."
        )
        return

    raise RuntimeError(
        "The JWT signing key is still the placeholder shipped in "
        "app/core/config.py, which means any caller can mint a valid token for "
        f"any consultation. Set {JWT_SECRET_ENV_VAR} in the environment — note "
        "the prefix; a bare JWT_SECRET_KEY is ignored — or set "
        "COPILOT_ALLOW_INSECURE_AUTH=true for local development only."
    )


def create_production_app() -> FastAPI:
    """
    ASGI entry point: `uvicorn copilot_api.bootstrap:create_production_app --factory`.

    The secret check lives here rather than in `build_application` on purpose:
    it is a property of deploying the service, and a test that assembles the
    app in-process should not have to care about the ambient environment.
    """
    settings = CopilotSettings.from_env()
    check_auth_secret(allow_insecure=settings.allow_insecure_auth)
    app, _ = build_application(settings)
    return app
