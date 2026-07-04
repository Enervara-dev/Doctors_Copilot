from __future__ import annotations
import os
import yaml
import math
import hashlib
import time
import logging
from typing import Any, Dict, List, Set, Tuple
import numpy as np

from clinical_intelligence.config.rag import GUIDELINES_YAML_PATH, DEFAULT_EMBEDDING_MODEL
from clinical_intelligence.config.llm import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class ClinicalVectorStore:
    """
    A serverless, lightweight guideline vector store database.
    Performs cosine similarity search using local TF-IDF representations.
    Exposes build metadata details for audit reproducibility.
    """

    def __init__(self, yaml_path: Optional[str] = None) -> None:
        self.yaml_path = yaml_path or GUIDELINES_YAML_PATH
        self.version = "1.0.0"
        self.build_timestamp = ""
        self.document_count = 0
        self.checksum = ""
        
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: List[str] = []
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: Optional[np.ndarray] = None
        
        self.load_and_build()

    def load_and_build(self) -> None:
        """Load YAML guideline databases, compute checksum, and index vocabulary TF-IDFs."""
        if not os.path.exists(self.yaml_path):
            logger.warning(f"Guidelines YAML file not found at {self.yaml_path}. Vector store is empty.")
            return

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                data = yaml.safe_load(content) or {}
                
            self.checksum = hashlib.md5(content.encode("utf-8")).hexdigest()
            self.documents = data.get("guidelines", [])
            self.document_count = len(self.documents)
            self.build_timestamp = datetime_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            logger.info(
                f"ClinicalVectorStore: loaded {self.document_count} documents. Checksum: {self.checksum}"
            )

            # Build TF-IDF vectors
            self._build_tfidf_index()
        except Exception as e:
            logger.error(f"Failed to load or build ClinicalVectorStore: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into clean, normalized word tokens."""
        words = []
        for w in text.lower().split():
            clean_w = "".join(c for c in w if c.isalnum())
            if clean_w:
                words.append(clean_w)
        return words

    def _build_tfidf_index(self) -> None:
        """Compute the vocabulary terms and document TF-IDF matrix."""
        if not self.documents:
            return

        # 1. Tokenize all docs and gather vocabulary
        tokenized_docs = [self._tokenize(doc.get("text_content", "")) for doc in self.documents]
        all_words: Set[str] = set()
        for doc in tokenized_docs:
            all_words.update(doc)
        
        self.vocabulary = sorted(list(all_words))
        word_to_idx = {w: idx for idx, w in enumerate(self.vocabulary)}

        # 2. Compute IDF: log(N / (1 + doc_freq))
        N = len(self.documents)
        self.idf = {}
        for word in self.vocabulary:
            doc_freq = sum(1 for doc in tokenized_docs if word in doc)
            self.idf[word] = math.log(N / (1.0 + doc_freq))

        # 3. Compute TF-IDF matrix
        vectors = np.zeros((N, len(self.vocabulary)))
        for doc_idx, doc in enumerate(tokenized_docs):
            if not doc:
                continue
            tf_dict: Dict[str, int] = {}
            for word in doc:
                tf_dict[word] = tf_dict.get(word, 0) + 1
            
            # Compute TF-IDF values
            max_tf = max(tf_dict.values())
            for word, freq in tf_dict.items():
                if word in word_to_idx:
                    tf_val = freq / max_tf
                    vectors[doc_idx, word_to_idx[word]] = tf_val * self.idf[word]
                    
        self.tfidf_vectors = vectors

    def _compute_query_vector(self, query_text: str) -> np.ndarray:
        """Project a text query onto the vocabulary TF-IDF vector space."""
        tokens = self._tokenize(query_text)
        vector = np.zeros(len(self.vocabulary))
        if not tokens or not self.vocabulary:
            return vector

        tf_dict: Dict[str, int] = {}
        for w in tokens:
            tf_dict[w] = tf_dict.get(w, 0) + 1

        word_to_idx = {w: idx for idx, w in enumerate(self.vocabulary)}
        max_tf = max(tf_dict.values())

        for word, freq in tf_dict.items():
            if word in word_to_idx:
                tf_val = freq / max_tf
                vector[word_to_idx[word]] = tf_val * self.idf[word]
        return vector

    def search(self, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute cosine similarity search against guidelines using local TF-IDF.
        Returns a list of tuples: (document, similarity_score)
        """
        if not self.documents or self.tfidf_vectors is None or len(self.vocabulary) == 0:
            return []

        query_vec = self._compute_query_vector(query_text)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            # Query terms completely outside document vocabulary
            return [(doc, 0.0) for doc in self.documents[:top_k]]

        results = []
        for idx, doc in enumerate(self.documents):
            doc_vec = self.tfidf_vectors[idx]
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
            
            results.append((doc, float(similarity)))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
