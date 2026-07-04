from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime

from clinical_intelligence.vector_store.store import ClinicalVectorStore
from clinical_intelligence.config.retrieval import (
    WEIGHT_SEMANTIC_SIMILARITY,
    WEIGHT_EVIDENCE_LEVEL,
    WEIGHT_RECENCY,
    WEIGHT_CONFIDENCE,
    WEIGHT_SPECIALTY,
    WEIGHT_SCORE_MATCH,
    EVIDENCE_LEVEL_WEIGHTS,
    DIVERSITY_WEIGHTS,
)
from clinical_intelligence.config.rag import SIMILARITY_THRESHOLD, MAX_GUIDELINES_TO_RETRIEVE

logger = logging.getLogger(__name__)


class ClinicalGuidelinesRetriever:
    """
    Retrieves and ranks clinical guidelines using a multi-factor scoring formula.
    Provides retrieval explainability metadata (score, reason, and matches).
    """

    def __init__(self, store: Optional[ClinicalVectorStore] = None) -> None:
        self.store = store or ClinicalVectorStore()

    def retrieve_explainable_guidelines(
        self,
        symptoms: List[str],
        candidates: List[Dict[str, Any]],
        risk_factors: List[str],
        medications: List[str],
        age: float,
        gender: str,
        red_flags: List[str],
        scores: Dict[str, float],
        stage: str = "History Taking",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves matching guidelines and scores them using the multi-factor relevance formula.
        Applies configured similarity thresholds and attaches explainability annotations.
        """
        k = top_k or MAX_GUIDELINES_TO_RETRIEVE
        if not self.store.documents:
            return []

        # 1. Build context search query text
        query_parts = []
        if symptoms:
            query_parts.append(f"symptoms: {', '.join(symptoms)}")
        if candidates:
            cand_names = [c.get("name", c.get("disease_id", "")) for c in candidates]
            query_parts.append(f"suspected diseases: {', '.join(cand_names)}")
        if risk_factors:
            query_parts.append(f"risk factors: {', '.join(risk_factors)}")
        if red_flags:
            query_parts.append(f"red flags: {', '.join(red_flags)}")
        if medications:
            query_parts.append(f"medications: {', '.join(medications)}")
        if scores:
            query_parts.append(f"clinical scores: {', '.join(scores.keys())}")
        query_parts.append(f"stage: {stage}")
        
        query_text = " ".join(query_parts)

        # 2. Get cosine similarity scores
        raw_results = self.store.search(query_text, top_k=len(self.store.documents))

        current_year = datetime.now().year
        ranked_results = []

        for doc, similarity in raw_results:
            # Factor A: Similarity (already calculated)
            
            # Factor B: Evidence Level Weight
            evidence_lvl = doc.get("evidence_level", "Level C")
            lvl_weight = EVIDENCE_LEVEL_WEIGHTS.get(evidence_lvl, 0.5)

            # Factor C: Recency (Guideline age, penalty over time)
            try:
                g_year = int(doc.get("version", current_year))
            except ValueError:
                g_year = current_year
            year_diff = max(0, current_year - g_year)
            recency_weight = max(0.0, 1.0 - (year_diff / 20.0))

            # Factor D: Confidence
            # Use max candidate confidence if doc specialty matches candidate diseases
            doc_specialty = doc.get("specialty", "").lower().strip()
            max_conf = 0.0
            specialty_match = 0.0
            
            for c in candidates:
                cand_category = c.get("category", "").lower().strip()
                if cand_category == doc_specialty or doc_specialty in cand_category:
                    specialty_match = 1.0
                    max_conf = max(max_conf, float(c.get("confidence", 0.0)))

            # Factor F: Clinical Score Match
            score_match = 0.0
            doc_title_lower = doc.get("title", "").lower()
            doc_text_lower = doc.get("text_content", "").lower()
            for score_name in scores.keys():
                score_clean = score_name.lower().split()[0]
                if score_clean in doc_title_lower or score_clean in doc_text_lower:
                    score_match = 1.0

            # 3. Calculate Final Multi-Factor Score
            final_score = (
                WEIGHT_SEMANTIC_SIMILARITY * similarity +
                WEIGHT_EVIDENCE_LEVEL * lvl_weight +
                WEIGHT_RECENCY * recency_weight +
                WEIGHT_CONFIDENCE * max_conf +
                WEIGHT_SPECIALTY * specialty_match +
                WEIGHT_SCORE_MATCH * score_match
            )

            # Detect matches
            matched_symptoms = [s for s in symptoms if s.lower() in doc_text_lower]
            matched_diagnoses = []
            for c in candidates:
                cand_name = c.get("name", c.get("disease_id", "")).lower()
                if cand_name and cand_name in doc_text_lower:
                    matched_diagnoses.append(c.get("name", c.get("disease_id")))
            matched_risk_factors = [rf for rf in risk_factors if rf.lower() in doc_text_lower]

            # Explanation reason
            reason_parts = [
                f"Similarity: {similarity:.2f}",
                f"Evidence: {lvl_weight:.2f} ({evidence_lvl})",
                f"Recency: {recency_weight:.2f} ({g_year})"
            ]
            if specialty_match > 0:
                reason_parts.append(f"Specialty Match: {doc_specialty}")
            if score_match > 0:
                reason_parts.append("Score Trigger")

            explainable_doc = {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "organization": doc.get("organization"),
                "version": doc.get("version"),
                "evidence_level": evidence_lvl,
                "specialty": doc_specialty,
                "text_content": doc.get("text_content"),
                "retrieval_score": round(final_score, 4),
                "normalized_score": round(final_score, 4),
                "similarity_score": round(similarity, 4),
                "confidence": max_conf,
                "retrieval_reason": ", ".join(reason_parts),
                "matching_entities": matched_diagnoses, # Keep matching_entities for backward-compatibility
                "matching_symptoms": matched_symptoms, # Keep matching_symptoms for backward-compatibility
                "matched_symptoms": matched_symptoms,
                "matched_diagnoses": matched_diagnoses,
                "matched_risk_factors": matched_risk_factors,
            }

            ranked_results.append(explainable_doc)

        # 4. Filter results based on quality threshold
        filtered_results = [
            res for res in ranked_results 
            if res["retrieval_score"] >= SIMILARITY_THRESHOLD
        ]

        # 5. Apply Greedy Categorical Diversification
        selected_results = []
        remaining_results = list(filtered_results)
        
        while len(selected_results) < k and remaining_results:
            scored_candidates = []
            for res in remaining_results:
                penalty = 0.0
                for sel in selected_results:
                    if res.get("organization") and res.get("organization") == sel.get("organization"):
                        penalty += DIVERSITY_WEIGHTS.get("organization", 0.0)
                    if res.get("specialty") and res.get("specialty") == sel.get("specialty"):
                        penalty += DIVERSITY_WEIGHTS.get("specialty", 0.0)
                    if res.get("version") and res.get("version") == sel.get("version"):
                        penalty += DIVERSITY_WEIGHTS.get("publication_year", 0.0)
                
                adjusted_score = res["retrieval_score"] - penalty
                scored_candidates.append((res, adjusted_score))
                
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_res = scored_candidates[0][0]
            selected_results.append(best_res)
            remaining_results.remove(best_res)

        return selected_results
