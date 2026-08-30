from __future__ import annotations

"""
evidence_models.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

class EvidenceContribution(BaseModel):
    trace_id: str
    source_id: Optional[str] = None
    timestamp: float = 0.0
    source: Literal["symptom", "vital", "examination", "demographics", "history", "risk_factor", "knowledge_graph", "guidelines", "exclusion"]
    finding: str
    polarity: Literal["positive", "negative"]
    confidence: float
    weight: float
    explanation: str

class EvidenceResult(BaseModel):
    provider: str
    score: float
    contributions: List[EvidenceContribution] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DiseaseProfile(BaseModel):
    id: str
    name: str
    category: str
    profile_version: str = "1.0.0"
    compiled_at: str
    source_hash: str
    cardinal_symptoms: List[Dict[str, Any]] = Field(default_factory=list)
    supportive_symptoms: List[Dict[str, Any]] = Field(default_factory=list)
    exclusionary_symptoms: List[str] = Field(default_factory=list)
    examination: List[str] = Field(default_factory=list)
    vitals: List[Dict[str, Any]] = Field(default_factory=list)
    risk_factor_modifiers: Dict[str, float] = Field(default_factory=dict)
    age_range: Optional[Dict[str, Any]] = None
    gender_modifier: Optional[Dict[str, Any]] = None
    guideline_tags: List[str] = Field(default_factory=list)

