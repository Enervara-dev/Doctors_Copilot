from __future__ import annotations

# Weights for multi-factor relevance ranking formula
WEIGHT_SEMANTIC_SIMILARITY = 0.40
WEIGHT_EVIDENCE_LEVEL = 0.20
WEIGHT_RECENCY = 0.15
WEIGHT_CONFIDENCE = 0.10
WEIGHT_SPECIALTY = 0.10
WEIGHT_SCORE_MATCH = 0.05

# Priority weights mapped to clinical evidence levels
EVIDENCE_LEVEL_WEIGHTS = {
    "Level A": 1.0,
    "Class I": 1.0,
    "Level B": 0.8,
    "Class IIa": 0.8,
    "Level C": 0.5,
    "Class IIb": 0.5,
    "Class III": 0.1,
}

# RAG guidelines diversification penalty weights
DIVERSITY_WEIGHTS = {
    "organization": 0.10,
    "specialty": 0.15,
    "publication_year": 0.05,
}
