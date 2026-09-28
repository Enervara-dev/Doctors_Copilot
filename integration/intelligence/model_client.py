"""
The seam between the graph and a model.

The graph nodes never call a provider SDK directly. They call `ModelClient`,
which takes a prompt and a JSON schema and returns parsed JSON. That one
indirection is what makes the whole graph testable offline: a test injects a
scripted client and the graph runs to completion with no network, no key and
no nondeterminism.

It is also what keeps this package free of a model vendor. A Gemini or OpenAI
client implements `complete`; the repository already has both behind
`clinical_intelligence/llm/providers.py`, and wiring one in is a separate
decision from wiring in the graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class ModelUnavailable(Exception):
    """
    The model could not answer.

    Distinct from a bug: no key configured, a refused request, a timeout. The
    graph treats it as "this node produced nothing" and carries on, because a
    missing suggestion must never take down a consultation.
    """


@runtime_checkable
class ModelClient(Protocol):
    """Structured completion. Synchronous, like everything it sits beside."""

    name: str

    def complete(
        self,
        *,
        prompt: str,
        schema: Dict[str, Any],
        purpose: str,
    ) -> Dict[str, Any]:
        """
        Returns JSON matching `schema`.

        `purpose` names the graph node, so a client can route, cache or log per
        task without the node knowing how.

        Raises `ModelUnavailable` when there is no answer to give.
        """
        ...


class ScriptedModelClient:
    """
    A client that replays canned responses, keyed by node purpose.

    Lives in the package rather than the test file because it is how every
    consumer should test a graph — including whoever writes the next provider.
    An unscripted purpose raises `ModelUnavailable`, which exercises the
    degradation path rather than silently returning nothing.
    """

    name = "scripted"

    def __init__(
        self,
        responses: Optional[Dict[str, Any]] = None,
        *,
        fail_with: Optional[Dict[str, Exception]] = None,
    ) -> None:
        self._responses = responses or {}
        self._fail_with = fail_with or {}
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self, *, prompt: str, schema: Dict[str, Any], purpose: str
    ) -> Dict[str, Any]:
        self.calls.append({"purpose": purpose, "prompt": prompt, "schema": schema})

        if purpose in self._fail_with:
            raise self._fail_with[purpose]
        if purpose not in self._responses:
            raise ModelUnavailable(f"No scripted response for {purpose!r}.")

        response = self._responses[purpose]
        return response(prompt) if callable(response) else response

    def prompt_for(self, purpose: str) -> Optional[str]:
        """The prompt a node actually sent, so a test can assert on grounding."""
        for call in self.calls:
            if call["purpose"] == purpose:
                return call["prompt"]
        return None


class UnavailableModelClient:
    """Always declines. The default when nothing is configured."""

    name = "unavailable"

    def complete(
        self, *, prompt: str, schema: Dict[str, Any], purpose: str
    ) -> Dict[str, Any]:
        raise ModelUnavailable("No model client is configured.")
