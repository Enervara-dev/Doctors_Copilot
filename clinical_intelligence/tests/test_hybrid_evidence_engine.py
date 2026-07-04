from __future__ import annotations

"""
Test suite for hybrid_evidence_engine

Purpose:
Defines automated unit and integration tests to verify code coverage and correctness.

Module/Component: Test Suites
"""

import unittest
from pathlib import Path
from clinical_intelligence.evidence.profile_compiler import ProfileCompiler
from clinical_intelligence.evidence.candidate_generator import CandidateRetriever, CandidateExpander
from clinical_intelligence.evidence.providers import EvidenceProviderRegistry, SymptomProvider, ExclusionProvider
from clinical_intelligence.evidence.normalizer import EvidenceNormalizer
from clinical_intelligence.evidence.hybrid_scorer import HybridEvidenceScorer
from clinical_intelligence.evidence.confidence_normalizer import ConfidenceNormalizer
from clinical_intelligence.evidence.question_planner import AdaptiveQuestionPlanner
from clinical_intelligence.evidence.explanation_builder import ExplanationBuilder
from clinical_intelligence.evidence.evidence_models import EvidenceContribution
from clinical_intelligence.evidence.evidence_context import EvidenceContext
from clinical_intelligence.rule_engine.loader import get_loader

class TestHybridEvidenceEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = get_loader()
        self.yaml_path = self.loader._rules_dir / "diseases.yaml"
        self.registry = ProfileCompiler.compile_file(self.yaml_path)

    def test_profile_compilation_and_indexing(self) -> None:
        self.assertIsNotNone(self.registry)
        # Check symptom index lookup
        chest_pain_candidates = self.registry.get_by_symptom("chest pain")
        self.assertIn("acs", chest_pain_candidates)
        self.assertIn("unstable_angina", chest_pain_candidates)

        # Check disease cache
        acs_profile = self.registry.get_profile("acs")
        self.assertIsNotNone(acs_profile)
        self.assertEqual(acs_profile.name, "Acute Coronary Syndrome")

    def test_candidate_retrieval_and_expansion(self) -> None:
        retriever = CandidateRetriever(self.registry)
        expander = CandidateExpander(self.registry)

        # Retrieve using symptom "chest pain"
        primary = retriever.retrieve(["chest pain"], [])
        self.assertIn("acs", primary)

        # Expand candidates using history of hypertension
        expanded = expander.expand(
            retrieved_candidates=primary,
            history=["hypertension"],
            risk_factors=[],
            vitals={},
            examination=[]
        )
        # Category of ACS is cardiovascular, other cardiovascular diseases should be considered
        self.assertIn("heart_failure", expanded)

    def test_registry_and_providers(self) -> None:
        registry = EvidenceProviderRegistry()
        registry.register("SymptomProvider", SymptomProvider)
        registry.register("ExclusionProvider", ExclusionProvider)

        providers = registry.get_providers()
        self.assertIn("SymptomProvider", providers)
        self.assertIn("ExclusionProvider", providers)

    def test_normalizer_conflict_resolution(self) -> None:
        normalizer = EvidenceNormalizer()

        # Create conflicting contributions: positive and negative for the same finding
        c1 = EvidenceContribution(
            trace_id="t1",
            source="symptom",
            finding="focal neurological deficit",
            polarity="positive",
            confidence=1.0,
            weight=0.25,
            explanation="Match"
        )
        c2 = EvidenceContribution(
            trace_id="t2",
            source="exclusion",
            finding="focal neurological deficit",
            polarity="negative",
            confidence=1.0,
            weight=1.0,
            explanation="Exclusion"
        )

        normalized = normalizer.normalize([c1, c2])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].polarity, "negative")

    def test_scorer_weight_capping(self) -> None:
        scorer = HybridEvidenceScorer()
        
        # Add symptom contributions exceeding symptom cap (0.50)
        c1 = EvidenceContribution(
            trace_id="1", source="symptom", finding="fever", polarity="positive", confidence=1.0, weight=0.35, explanation=""
        )
        c2 = EvidenceContribution(
            trace_id="2", source="symptom", finding="cough", polarity="positive", confidence=1.0, weight=0.35, explanation=""
        )

        score_res = scorer.score("viral_uri", [c1, c2])
        # Total contribution is 0.70, but capped at 0.50
        self.assertEqual(score_res.score_breakdown["symptoms"], 0.50)
        self.assertEqual(score_res.confidence, 0.50)

    def test_confidence_normalizer(self) -> None:
        normalizer = ConfidenceNormalizer()
        self.assertEqual(normalizer.normalize_confidence_level(0.85), "High")
        self.assertEqual(normalizer.normalize_confidence_level(0.55), "Medium")
        self.assertEqual(normalizer.normalize_confidence_level(0.20), "Low")

if __name__ == "__main__":
    unittest.main()
