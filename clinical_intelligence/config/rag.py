from __future__ import annotations
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
