"""
The LangGraph intelligence provider.

A real `langgraph.graph.StateGraph`, compiled once, with four nodes that each
produce one kind of proposal:

    START → synthesis → unresolved → explanations → investigation_notes → END

Sequential rather than parallel, because later nodes read what earlier ones
produced: the explanation node avoids repeating a point the synthesis already
made, and the notes node knows which investigations have been discussed. The
graph is small on purpose — this is the first one, and a shape that is easy to
reason about is worth more right now than concurrency.

What the graph is allowed to do is fixed by the boundary it plugs into, not by
this file. It receives a frozen `CopilotContext`, it returns a
`ProviderResponse`, and `IntelligenceGateway` validates every proposal against
that context before anything reaches a clinician. Nothing here can change the
clinical state, reorder a differential or alter an investigation priority —
there is no field on a proposal that could carry such a change.

Two properties the nodes are written to preserve:

* **Grounding is constructed, not requested.** Each node is handed the exact
  refs it may cite and the model is asked to choose among them. A ref the model
  invents is dropped before the proposal is built, so the validator's
  UNKNOWN_REFERENCE path should never be reached from here — and a test asserts
  it isn't.
* **A failing node costs one proposal, not the consultation.** Every node
  catches `ModelUnavailable` and anything else the client raises, records it,
  and returns the state unchanged.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from integration.intelligence.context import CopilotContext
from integration.intelligence.contract import (
    IntelligenceStatus,
    Proposal,
    ProposalKind,
    ProviderResponse,
)
from integration.intelligence.model_client import ModelClient, UnavailableModelClient

logger = logging.getLogger(__name__)

PROVIDER_NAME = "langgraph"
PROVIDER_VERSION = "1.0.0"

#: Caps per node. A clinician mid-consultation reads a handful of lines, not a
#: report, and an unbounded list is how a helpful panel becomes noise.
MAX_QUESTIONS = 4
MAX_EXPLANATIONS = 2
MAX_NOTES = 3


# ── graph state ───────────────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """
    What flows between nodes.

    `context` is the frozen snapshot and is never written to. `proposals`
    accumulates; `notes` records why a node produced nothing, which is what
    makes a quiet response diagnosable.
    """

    context: CopilotContext
    proposals: List[Proposal]
    notes: List[str]


# ── prompt construction ───────────────────────────────────────────────────


def _describe_observed(context: CopilotContext) -> str:
    observed = context.observed
    lines: List[str] = []
    who = [
        f"{observed.age} years old" if observed.age is not None else None,
        observed.gender,
    ]
    if any(who):
        lines.append("Patient: " + ", ".join(w for w in who if w))
    for symptom in observed.symptoms:
        state = "reports" if str(symptom.detail.get("present")) == "True" else "denies"
        severity = symptom.detail.get("severity")
        lines.append(
            f"- [{symptom.ref}] {state} {symptom.label}"
            + (f" ({severity})" if severity else "")
            + (f' — said: "{symptom.quote}"' if symptom.quote else "")
        )
    if observed.history:
        lines.append("History: " + ", ".join(observed.history))
    if observed.medications:
        lines.append("Medications: " + ", ".join(observed.medications))
    if observed.vitals:
        lines.append("Vitals: " + ", ".join(f"{k}={v}" for k, v in observed.vitals.items()))
    return "\n".join(lines) or "Nothing has been established yet."


def _describe_conclusions(context: CopilotContext) -> str:
    det = context.deterministic
    lines: List[str] = []
    for fact in det.red_flags:
        lines.append(
            f"- [{fact.ref}] RED FLAG {fact.label} ({fact.detail.get('severity')})"
        )
    for fact in det.differential:
        lines.append(
            f"- [{fact.ref}] candidate #{fact.detail.get('rank')} {fact.label} "
            f"(confidence {fact.detail.get('confidence')})"
        )
    for fact in det.investigations:
        lines.append(
            f"- [{fact.ref}] investigation {fact.label} ({fact.detail.get('priority')})"
        )
    for fact in det.missing_information:
        lines.append(f"- [{fact.ref}] unanswered: {fact.label}")
    return "\n".join(lines) or "The rule engine reached no conclusions."


_GUARDRAIL = (
    "You are assisting a clinician during a live consultation. A deterministic "
    "rule engine has already produced the conclusions below and remains "
    "authoritative: you must not contradict it, re-rank it, or claim a "
    "diagnosis. Cite only the bracketed references given to you. If you have "
    "nothing useful to add, return an empty list."
)


def _prompt(context: CopilotContext, task: str, allowed: List[str]) -> str:
    return "\n\n".join(
        [
            _GUARDRAIL,
            "OBSERVED (what the patient said):\n" + _describe_observed(context),
            "DETERMINISTIC CONCLUSIONS:\n" + _describe_conclusions(context),
            "REFERENCES YOU MAY CITE:\n" + (", ".join(allowed) or "(none)"),
            "TASK:\n" + task,
        ]
    )


def _stable_id(context: CopilotContext, *parts: str) -> str:
    """
    Content-addressed, so replaying a consultation reproduces proposal ids.

    A uuid here would break the runtime's replay determinism at this boundary.
    """
    digest = hashlib.sha256("|".join((context.fingerprint, *parts)).encode()).hexdigest()
    return f"lg_{digest[:12]}"


def _keep_known(refs: Any, allowed: set[str]) -> List[str]:
    """
    Drops any reference the model invented.

    The validator would reject the whole proposal for one bad ref; filtering
    here means a mostly-sound suggestion survives its own footnote error, and
    an entirely ungrounded one still fails because nothing is left to cite.
    """
    if not isinstance(refs, list):
        return []
    return [r for r in refs if isinstance(r, str) and r in allowed]


# ── schemas ───────────────────────────────────────────────────────────────

_ITEM_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "refs"],
            },
        }
    },
    "required": ["items"],
}

_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "refs"],
}


class LangGraphIntelligenceProvider:
    """
    Satisfies `IntelligenceProvider`; swappable for the null or deterministic one.

    The compiled graph is built once and reused: compilation is the expensive
    part and the graph has no per-consultation state — everything it needs
    arrives in `GraphState`.
    """

    name = PROVIDER_NAME
    version = PROVIDER_VERSION

    def __init__(self, model: Optional[ModelClient] = None) -> None:
        self._model: ModelClient = model or UnavailableModelClient()
        self._graph = self._build_graph()

    # ── nodes ─────────────────────────────────────────────────────────────

    def _synthesis(self, state: GraphState) -> GraphState:
        """One paragraph a clinician can read at a glance."""
        context = state["context"]
        allowed = set(context.references)
        if not allowed:
            return {"notes": state.get("notes", []) + ["synthesis: nothing to summarise"]}

        prompt = _prompt(
            context,
            "Write one short paragraph summarising where this consultation has got "
            "to. Describe only what is above. Do not state a diagnosis.",
            context.references,
        )
        try:
            answer = self._model.complete(
                prompt=prompt, schema=_SUMMARY_SCHEMA, purpose="synthesis"
            )
        except Exception as exc:  # noqa: BLE001 - a node failure is not a consultation failure
            return {"notes": state.get("notes", []) + [f"synthesis: {exc}"]}

        text = str(answer.get("summary", "")).strip()
        refs = _keep_known(answer.get("refs"), allowed)
        if not text or not refs:
            return {"notes": state.get("notes", []) + ["synthesis: no usable output"]}

        return {
            "proposals": state.get("proposals", [])
            + [
                Proposal(
                    proposal_id=_stable_id(context, "synthesis"),
                    kind=ProposalKind.SUMMARY,
                    content=text,
                    grounded_in=refs[:8],
                    metadata={"node": "synthesis"},
                )
            ]
        }

    def _unresolved(self, state: GraphState) -> GraphState:
        """Questions worth asking next, anchored to what is unanswered."""
        context = state["context"]
        anchors = context.deterministic.missing_information + context.observed.symptoms
        allowed = {f.ref for f in anchors}
        if not allowed:
            return {"notes": state.get("notes", []) + ["unresolved: nothing outstanding"]}

        prompt = _prompt(
            context,
            f"Suggest at most {MAX_QUESTIONS} questions the clinician has not yet "
            "asked that would change management. Each must cite the reference it "
            "follows from.",
            sorted(allowed),
        )
        try:
            answer = self._model.complete(
                prompt=prompt, schema=_ITEM_LIST_SCHEMA, purpose="unresolved"
            )
        except Exception as exc:  # noqa: BLE001
            return {"notes": state.get("notes", []) + [f"unresolved: {exc}"]}

        proposals = list(state.get("proposals", []))
        for item in (answer.get("items") or [])[:MAX_QUESTIONS]:
            text = str(item.get("text", "")).strip()
            refs = _keep_known(item.get("refs"), allowed)
            if not text or not refs:
                continue
            proposals.append(
                Proposal(
                    proposal_id=_stable_id(context, "question", text),
                    kind=ProposalKind.QUESTION,
                    content=text,
                    grounded_in=refs[:6],
                    metadata={"node": "unresolved"},
                )
            )
        return {"proposals": proposals}

    def _explanations(self, state: GraphState) -> GraphState:
        """
        Why the leading candidates fit the evidence.

        Explanations of existing candidates are SUMMARY, not
        DIAGNOSIS_CONSIDERATION: the candidate is already the rule engine's, and
        dressing an explanation as a consideration would double-count it. A
        possibility the rules did not raise is the only thing that becomes a
        consideration, and it is kept in its own channel.
        """
        context = state["context"]
        candidates = context.deterministic.differential[:MAX_EXPLANATIONS]
        if not candidates:
            return {"notes": state.get("notes", []) + ["explanations: no candidates"]}

        allowed = {f.ref for f in candidates} | {
            f.ref for f in context.observed.symptoms
        } | {f.ref for f in context.deterministic.red_flags}

        prompt = _prompt(
            context,
            "For each leading candidate, explain in one or two sentences which "
            "observed findings support it and which would argue against it. Cite "
            "the candidate and the findings. Do not assert that any candidate is "
            "correct.",
            sorted(allowed),
        )
        try:
            answer = self._model.complete(
                prompt=prompt, schema=_ITEM_LIST_SCHEMA, purpose="explanations"
            )
        except Exception as exc:  # noqa: BLE001
            return {"notes": state.get("notes", []) + [f"explanations: {exc}"]}

        proposals = list(state.get("proposals", []))
        for item in (answer.get("items") or [])[:MAX_EXPLANATIONS]:
            text = str(item.get("text", "")).strip()
            refs = _keep_known(item.get("refs"), allowed)
            if not text or not refs:
                continue
            about = next((r for r in refs if r.startswith("dx:")), None)
            proposals.append(
                Proposal(
                    proposal_id=_stable_id(context, "explanation", text),
                    kind=ProposalKind.SUMMARY,
                    content=text,
                    grounded_in=refs[:8],
                    about_ref=about,
                    metadata={"node": "explanations"},
                )
            )
        return {"proposals": proposals}

    def _investigation_notes(self, state: GraphState) -> GraphState:
        """
        Commentary on tests the rules already recommended.

        The node is only ever shown existing investigation refs, so it cannot
        propose a new test — the reconciliation rule enforced by construction as
        well as by the validator.
        """
        context = state["context"]
        investigations = context.deterministic.investigations[:MAX_NOTES]
        if not investigations:
            return {"notes": state.get("notes", []) + ["notes: nothing recommended"]}

        allowed = {f.ref for f in investigations} | {
            f.ref for f in context.observed.symptoms
        }
        prompt = _prompt(
            context,
            "For the investigations already recommended, add a short practical note "
            "— what it would rule in or out here, or what to watch for. You may not "
            "suggest additional tests or change any priority. Cite the investigation.",
            sorted(allowed),
        )
        try:
            answer = self._model.complete(
                prompt=prompt, schema=_ITEM_LIST_SCHEMA, purpose="investigation_notes"
            )
        except Exception as exc:  # noqa: BLE001
            return {"notes": state.get("notes", []) + [f"notes: {exc}"]}

        known_inv = {f.ref for f in context.deterministic.investigations}
        proposals = list(state.get("proposals", []))
        for item in (answer.get("items") or [])[:MAX_NOTES]:
            text = str(item.get("text", "")).strip()
            refs = _keep_known(item.get("refs"), allowed)
            about = next((r for r in refs if r in known_inv), None)
            if not text or about is None:
                # A note about a test that was not recommended has nothing to
                # attach to; dropping it here keeps it out of the reject log as
                # a validator failure it would deserve.
                continue
            proposals.append(
                Proposal(
                    proposal_id=_stable_id(context, "note", about, text),
                    kind=ProposalKind.INVESTIGATION_NOTE,
                    content=text,
                    grounded_in=refs[:6],
                    about_ref=about,
                    metadata={"node": "investigation_notes"},
                )
            )
        return {"proposals": proposals}

    # ── graph ─────────────────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("synthesis", self._synthesis)
        builder.add_node("unresolved", self._unresolved)
        builder.add_node("explanations", self._explanations)
        builder.add_node("investigation_notes", self._investigation_notes)

        builder.add_edge(START, "synthesis")
        builder.add_edge("synthesis", "unresolved")
        builder.add_edge("unresolved", "explanations")
        builder.add_edge("explanations", "investigation_notes")
        builder.add_edge("investigation_notes", END)
        return builder.compile()

    # ── provider interface ────────────────────────────────────────────────

    def propose(self, context: CopilotContext) -> ProviderResponse:
        """
        Runs the graph and returns whatever survived it.

        An empty consultation short-circuits: running four model calls to
        summarise nothing wastes a budget and risks a model inventing content to
        fill the silence.
        """
        if not context.references:
            return ProviderResponse(
                status=IntelligenceStatus.OK,
                provider=self.name,
                provider_version=self.version,
                message="Nothing has been established in this consultation yet.",
                telemetry={"model": getattr(self._model, "name", "unknown"), "nodes_run": 0},
            )

        final: Dict[str, Any] = self._graph.invoke(
            {"context": context, "proposals": [], "notes": []}
        )
        proposals: List[Proposal] = list(final.get("proposals", []))
        notes: List[str] = list(final.get("notes", []))

        # Every node declining is not a failure of the graph, it is the model
        # being unavailable. Reporting OK with nothing would hide that.
        status = (
            IntelligenceStatus.OK
            if proposals
            else IntelligenceStatus.UNAVAILABLE
        )
        return ProviderResponse(
            status=status,
            provider=self.name,
            provider_version=self.version,
            proposals=proposals,
            message="; ".join(notes) if notes and not proposals else None,
            telemetry={
                "model": getattr(self._model, "name", "unknown"),
                "proposal_count": len(proposals),
                "node_notes": notes,
            },
        )
