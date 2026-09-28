"""
Real model clients, built from what this repository already has.

`clinical_intelligence/llm/providers.py` already carries a Gemini and an OpenAI
channel with a circuit breaker, retry, structured-output support and a TLS
context that trusts the OS store — which matters behind the proxy this codebase
runs behind. Adding a second HTTP client for the graph would mean a second set
of those decisions to keep right.

So this is an adapter, not a client. Two impedance mismatches to bridge:

* those providers are **async** and return **text**; the `ModelClient` seam is
  sync and returns parsed JSON. The graph runs on a worker thread with no event
  loop, so `asyncio.run` is the correct bridge rather than a loop-reuse trick.
* a model asked for JSON sometimes wraps it in a fence or prepends a sentence.
  Recovering the object is the adapter's job; a node should never see raw text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, Optional

from integration.intelligence.model_client import ModelClient, ModelUnavailable

logger = logging.getLogger(__name__)

#: ```json { ... } ``` — the commonest way a model ruins otherwise valid output.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_object(text: str) -> Dict[str, Any]:
    """
    Recovers a JSON object from a model's reply.

    Three attempts, in order of how much benefit of the doubt each gives:
    the whole string, the contents of a fenced block, then the outermost
    braces. Anything else is a failure — guessing further would mean inventing
    structure the model did not produce.
    """
    candidates = [text.strip()]

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ModelUnavailable("The model did not return a JSON object.")


class PlatformModelClient:
    """
    Wraps one of the repository's existing LLM providers.

    Every failure becomes `ModelUnavailable`. The graph treats that as "this
    node produced nothing" and carries on, which is the behaviour a clinician
    needs: a model that is down must cost commentary, never the consultation.
    """

    def __init__(self, provider: Any, *, name: Optional[str] = None) -> None:
        self._provider = provider
        self.name = name or getattr(provider, "model_name", provider.__class__.__name__)

    def complete(
        self, *, prompt: str, schema: Dict[str, Any], purpose: str
    ) -> Dict[str, Any]:
        try:
            # A fresh loop per call. The worker thread has none of its own, and
            # borrowing one across threads is how event-loop bugs start.
            text = asyncio.run(self._provider.generate(prompt, schema))
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - every provider failure is the same to us
            logger.warning("Model call failed for %s: %s", purpose, exc)
            raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc

        if not text or not text.strip():
            raise ModelUnavailable(f"The model returned nothing for {purpose!r}.")

        return _parse_json_object(text)


def build_model_client(
    *,
    provider_name: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> ModelClient:
    """
    Builds a client from configuration, or declines.

    Returns `UnavailableModelClient` when no key is configured rather than
    raising: a deployment without a model key is a valid deployment, and it
    should start and serve clinical output exactly as it does with one.
    """
    from integration.intelligence.model_client import UnavailableModelClient

    try:
        from clinical_intelligence.config.llm import (
            DEFAULT_GEMINI_MODEL,
            DEFAULT_OPENAI_MODEL,
            GEMINI_API_KEY,
            MAX_RESPONSE_TOKENS,
            OPENAI_API_KEY,
        )
        from clinical_intelligence.llm.providers import GeminiProvider, OpenAIProvider
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM providers are unavailable: %s", exc)
        return UnavailableModelClient()

    tokens = max_tokens or MAX_RESPONSE_TOKENS
    selected = (provider_name or "gemini").strip().lower()

    if selected == "openai":
        if not OPENAI_API_KEY:
            logger.info("No OpenAI key configured; intelligence will be unavailable.")
            return UnavailableModelClient()
        # Annotated because the two branches bind different provider classes to
        # the same name; both satisfy the same duck-typed `generate` contract.
        provider: Any = OpenAIProvider(
            model_name=model_name or DEFAULT_OPENAI_MODEL,
            temperature=temperature,
            max_tokens=tokens,
        )
        return PlatformModelClient(provider, name=f"openai:{provider.model_name}")

    if selected == "gemini":
        if not GEMINI_API_KEY:
            logger.info("No Gemini key configured; intelligence will be unavailable.")
            return UnavailableModelClient()
        provider = GeminiProvider(
            model_name=model_name or DEFAULT_GEMINI_MODEL,
            temperature=temperature,
            max_tokens=tokens,
        )
        return PlatformModelClient(provider, name=f"gemini:{provider.model_name}")

    logger.warning("Unknown model provider %r; intelligence will be unavailable.", selected)
    return UnavailableModelClient()
