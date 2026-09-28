"""
A red flag needs a presentation, not a risk profile.

`red_flags.yaml` documents age, history and risk factors as supporting
criteria — "any matching history item adds support". The detector used to
treat them as triggers, so a 62-year-old with hypertension and diabetes drew a
CRITICAL stroke alert with no neurological symptom anywhere in the
consultation, and a CRITICAL sepsis alert on the word "diabetes" alone.

These tests fix the boundary in both directions: a comorbidity must not raise
an alert on its own, and a genuine presentation must still raise one — with the
comorbidity recorded as the supporting evidence it always was.
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Optional

from clinical_intelligence.rule_engine.explainability import ExplainabilityTracer
from clinical_intelligence.rule_engine.loader import RuleLoader
from clinical_intelligence.rule_engine.red_flags import RedFlagDetector


class RedFlagCase(unittest.TestCase):
    """Drives the real detector against the real YAML."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = RuleLoader().get().red_flags

    def detect(
        self,
        *,
        symptoms: Optional[List[str]] = None,
        vitals: Optional[Dict[str, str]] = None,
        history: Optional[List[str]] = None,
        risk_factors: Optional[List[str]] = None,
        age: Optional[int] = None,
    ):
        detector = RedFlagDetector(self.rules)
        return detector.detect(
            present_symptoms={s.lower() for s in (symptoms or [])},
            negated_symptoms=set(),
            vital_signs_raw=vitals or {},
            medical_history=history or [],
            risk_factor_ids=risk_factors or [],
            age=age,
            tracer=ExplainabilityTracer(),
        )

    @staticmethod
    def ids(alerts) -> List[str]:
        return [a.flag_id for a in alerts]


class TestARiskProfileAloneRaisesNothing(RedFlagCase):
    def test_age_and_comorbidity_without_a_symptom_fire_nothing(self) -> None:
        """
        The exact defect.

        A 62-year-old with hypertension and diabetes, who has said nothing
        about how they feel, is not presenting with anything.
        """
        alerts = self.detect(
            history=["hypertension", "diabetes"],
            risk_factors=["hypertension", "diabetes"],
            age=62,
        )
        self.assertEqual(self.ids(alerts), [])

    def test_diabetes_alone_does_not_raise_sepsis(self) -> None:
        alerts = self.detect(history=["diabetes"], risk_factors=["diabetes"], age=62)
        self.assertNotIn("sepsis_red_flag", self.ids(alerts))

    def test_hypertension_alone_does_not_raise_stroke(self) -> None:
        alerts = self.detect(history=["hypertension"], risk_factors=["hypertension"], age=70)
        self.assertNotIn("stroke_red_flag", self.ids(alerts))

    def test_age_alone_raises_nothing(self) -> None:
        self.assertEqual(self.ids(self.detect(age=80)), [])


class TestCardiacPresentationDoesNotRaiseStroke(RedFlagCase):
    """The golden consultation's presentation, at the detector level."""

    def setUp(self) -> None:
        self.alerts = self.detect(
            symptoms=["chest pain", "shortness of breath", "nausea"],
            history=["hypertension", "diabetes"],
            risk_factors=["hypertension", "diabetes"],
            age=62,
        )

    def test_the_cardiac_flags_fire(self) -> None:
        fired = self.ids(self.alerts)
        self.assertIn("acs_red_flag", fired)
        self.assertIn("stemi_red_flag", fired)

    def test_the_stroke_flag_does_not_fire(self) -> None:
        self.assertNotIn("stroke_red_flag", self.ids(self.alerts))

    def test_the_sepsis_flag_does_not_fire(self) -> None:
        self.assertNotIn("sepsis_red_flag", self.ids(self.alerts))

    def test_comorbidity_is_still_recorded_as_supporting_evidence(self) -> None:
        """
        Demoted from trigger to evidence, not discarded.

        The clinician still needs to see that this patient's hypertension and
        diabetes bear on the cardiac alert.
        """
        acs = next(a for a in self.alerts if a.flag_id == "acs_red_flag")
        self.assertIn("chest pain", acs.supporting_evidence)
        self.assertIn("hypertension", acs.supporting_evidence)
        self.assertIn("diabetes", acs.supporting_evidence)
        self.assertTrue(any("age 62" in e for e in acs.supporting_evidence))


class TestGenuineStrokeStillFires(RedFlagCase):
    def test_facial_droop_and_speech_difficulty_raise_stroke(self) -> None:
        alerts = self.detect(
            symptoms=["sudden facial droop", "speech difficulty"],
            vitals={"bp": "168/100"},
            history=["hypertension"],
            age=55,
        )
        self.assertIn("stroke_red_flag", self.ids(alerts))

    def test_a_single_neurological_symptom_is_enough(self) -> None:
        alerts = self.detect(symptoms=["slurred speech"], age=55)
        self.assertIn("stroke_red_flag", self.ids(alerts))

    def test_stroke_fires_without_any_risk_history(self) -> None:
        """A first stroke in someone with no history must still be caught."""
        alerts = self.detect(symptoms=["arm weakness", "sudden vision loss"], age=44)
        self.assertIn("stroke_red_flag", self.ids(alerts))

    def test_the_stroke_action_is_preserved(self) -> None:
        alerts = self.detect(symptoms=["sudden facial droop"], age=60)
        stroke = next(a for a in alerts if a.flag_id == "stroke_red_flag")
        self.assertEqual(stroke.severity, "CRITICAL")
        self.assertIn("stroke code", stroke.recommended_action.lower())


class TestMixedPresentation(RedFlagCase):
    def test_cardiac_and_neurological_symptoms_raise_both(self) -> None:
        """
        Ambiguity is not something to resolve here.

        Chest pain with facial droop is a presentation that genuinely warrants
        both alerts; suppressing either would be the detector making a
        differential decision it has no basis for.
        """
        alerts = self.detect(
            symptoms=["chest pain", "sudden facial droop"],
            history=["hypertension"],
            age=62,
        )
        fired = self.ids(alerts)
        self.assertIn("acs_red_flag", fired)
        self.assertIn("stroke_red_flag", fired)

    def test_a_vital_sign_alone_still_raises_a_flag(self) -> None:
        """
        Vitals stay triggers.

        A blood pressure of 200/130 is an objective finding, not a risk
        profile, and must raise an alert even from a silent patient.
        """
        alerts = self.detect(vitals={"bp": "200/130"}, age=62)
        self.assertIn("hypertensive_crisis_red_flag", self.ids(alerts))

    def test_every_fired_flag_names_a_symptom_or_a_vital(self) -> None:
        """The invariant, stated once."""
        alerts = self.detect(
            symptoms=["chest pain", "fever"],
            vitals={"temperature": "39.1", "pulse": "112"},
            history=["hypertension", "diabetes"],
            risk_factors=["hypertension", "diabetes"],
            age=62,
        )
        self.assertTrue(alerts)
        demographic_only = {"hypertension", "diabetes"}
        for alert in alerts:
            clinical = [
                e
                for e in alert.supporting_evidence
                if e not in demographic_only and not e.startswith("age ")
            ]
            self.assertTrue(
                clinical,
                f"{alert.flag_id} fired on demographics alone: {alert.supporting_evidence}",
            )


class TestNoRuleBecameUnreachable(RedFlagCase):
    def test_every_rule_still_has_a_way_to_fire(self) -> None:
        """
        The change is safe only because no rule relied on history alone.

        If a future rule is added with neither any_of_symptoms nor
        vital_sign_criteria, it would be silently undetectable — this fails
        first and says so.
        """
        for rule in self.rules:
            self.assertTrue(
                rule.get("any_of_symptoms")
                or rule.get("required_symptoms")
                or rule.get("vital_sign_criteria"),
                f"{rule.get('id')} can never fire: no symptom or vital criteria",
            )


if __name__ == "__main__":
    unittest.main()
