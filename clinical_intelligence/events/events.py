from __future__ import annotations

"""
events.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SessionStarted:
    session_id: str
    patient_id: str
    clinician_id: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionUpdated:
    session_id: str
    workflow_stage: str
    status: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionCompleted:
    session_id: str
    summary: Dict[str, Any]
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionArchived:
    session_id: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class QuestionAsked:
    session_id: str
    question: str
    purpose: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class QuestionAnswered:
    session_id: str
    question: str
    answer: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ClinicalStateUpdated:
    session_id: str
    changed_keys: Dict[str, bool]
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ReasoningStarted:
    session_id: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ReasoningCompleted:
    session_id: str
    candidate_diagnoses_count: int
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GraphCompleted:
    session_id: str
    query_count: int
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RetrievalCompleted:
    session_id: str
    retrieved_count: int
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class LLMCompleted:
    session_id: str
    latency_ms: float
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SOAPUpdated:
    session_id: str
    soap_draft: Dict[str, Any]
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class InvestigationSuggested:
    session_id: str
    investigation_name: str
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
