from __future__ import annotations

"""
rag.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import os

# Paths to YAML documents registry
GUIDELINES_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "vector_store",
    "guidelines.yaml"
)

# Semantic retrieval matching thresholds
SIMILARITY_THRESHOLD = 0.60
MAX_GUIDELINES_TO_RETRIEVE = 5
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
