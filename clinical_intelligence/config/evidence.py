from __future__ import annotations

"""
evidence.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

# Evidence-Based Clinical Reasoning Engine Constants and Thresholds
MIN_CANDIDATE_SCORE = 0.20
MIN_DIAGNOSIS_SCORE = 0.35
MAX_CANDIDATES = 15
MAX_DIFFERENTIALS = 10
EXCLUSION_PENALTY = 1.0

MAX_PROVIDER_CONTRIBUTIONS = {
    "symptom": 0.50,
    "vital": 0.15,
    "examination": 0.10,
    "history": 0.10,
    "risk_factor": 0.10,
    "knowledge_graph": 0.03,
    "guidelines": 0.02,
    "demographics": 0.05,
    "exclusion": -1.0
}
