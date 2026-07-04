"""
Disease Confidence Scorer.

Converts raw SymptomMatchResult objects into normalised confidence scores
[0, 1] by applying demographic, vital-sign, risk-factor, and negation
modifiers defined in the disease rules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from clinical_intelligence.rule_engine.explainability import ExplainabilityTracer
from clinical_intelligence.rule_engine.matcher import SymptomMatchResult
from clinical_intelligence.rule_engine.models import (
    CandidateDiagnosis,
    ClinicalGuideline,
    RuleMetadata,
    ClinicalScoreResult,
)

logger = logging.getLogger(__name__)


def _parse_vitals(vital_signs: Dict[str, Optional[str]]) -> Dict[str, Optional[float]]:
    """
    Convert the raw string vital-sign values from ClinicalState into floats.
    BP is split into systolic/diastolic.
    """
    parsed: Dict[str, Optional[float]] = {}

    bp_raw = vital_signs.get("bp")
    if bp_raw:
        match = re.search(r'(\d+)\s*/\s*(\d+)', str(bp_raw))
        if match:
            parsed["bp_systolic"] = float(match.group(1))
            parsed["bp_diastolic"] = float(match.group(2))

    pulse_raw = vital_signs.get("pulse") or vital_signs.get("heart_rate")
    if pulse_raw:
        match = re.search(r'(\d+)', str(pulse_raw))
        if match:
            parsed["pulse"] = float(match.group(1))
            parsed["heart_rate"] = float(match.group(1))

    temp_raw = vital_signs.get("temperature")
    if temp_raw:
        match = re.search(r'(\d+(?:\.\d+)?)', str(temp_raw))
        if match:
            parsed["temperature"] = float(match.group(1))

    spo2_raw = vital_signs.get("spo2")
    if spo2_raw:
        match = re.search(r'(\d+(?:\.\d+)?)', str(spo2_raw))
        if match:
            parsed["spo2"] = float(match.group(1))

    rr_raw = vital_signs.get("respiratory_rate") or vital_signs.get("respiration_rate")
    if rr_raw:
        match = re.search(r'(\d+)', str(rr_raw))
        if match:
            parsed["respiratory_rate"] = float(match.group(1))
            parsed["respiration_rate"] = float(match.group(1))

    return parsed


def _evaluate_vital_condition(
    condition: str,
    threshold: float,
    vitals: Dict[str, Optional[float]],
) -> bool:
    """Evaluate a single vital-sign criterion such as 'bp_systolic_gt'."""
    parts = condition.rsplit("_", 1)
    if len(parts) != 2:
        return False
    vital_key, operator = parts
    value = vitals.get(vital_key)
    if value is None:
        return False
    if operator == "gt":
        return value > threshold
    if operator == "lt":
        return value < threshold
    if operator == "gte":
        return value >= threshold
    if operator == "lte":
        return value <= threshold
    return False


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class DiseaseScorer:
    """
    Converts raw SymptomMatchResult objects to CandidateDiagnosis records
    with normalised confidence scores [0, 1].

    Scoring formula::

        base_score     = raw_score / max_possible_score
        vitals_mod     = sum of matching vital_sign_criteria modifiers
        demographics   = age_modifier + gender_modifier
        risk_mod       = sum of present risk_factor modifiers
        final_score    = clamp(base_score + vitals_mod + demographics
                               + risk_mod − negation_penalty, 0, 1)
    """

    def score(
        self,
        *,
        match_results: List[SymptomMatchResult],
        vital_signs: Dict[str, Optional[str]],
        age: Optional[int],
        gender: Optional[str],
        identified_risk_factor_ids: List[str],
        tracer: ExplainabilityTracer,
        clinical_scores: List[ClinicalScoreResult] = None,
    ) -> List[CandidateDiagnosis]:
        """
        Produce ranked CandidateDiagnosis records.

        Parameters
        ----------
        match_results:
            Output from SymptomMatcher.match().
        vital_signs:
            Raw vital-sign strings from ClinicalState.
        age:
            Patient age in years (or None if unknown).
        gender:
            Patient gender string (or None if unknown).
        identified_risk_factor_ids:
            IDs of risk factors confirmed present for this patient.
        tracer:
            Explainability tracer.
        clinical_scores:
            List of evaluated clinical scores to check agreement.
        """
        parsed_vitals = _parse_vitals(vital_signs)
        candidates: List[CandidateDiagnosis] = []

        for rank_idx, match in enumerate(match_results, start=1):
            disease = match.disease_definition
            min_threshold = float(disease.get("min_diagnosis_threshold", 0.25))

            # 1. Base score (symptom coverage ratio)
            base_score = match.raw_score / match.max_possible_score if match.max_possible_score > 0 else 0.0

            supporting_evidence: List[str] = []
            risk_factors_active: List[str] = []

            # 2. Vital-sign modifiers
            vitals_mod = 0.0
            for criterion in disease.get("vital_sign_criteria", []):
                condition = criterion.get("condition", "")
                threshold = float(criterion.get("threshold", 0))
                modifier = float(criterion.get("modifier", 0.0))
                if _evaluate_vital_condition(condition, threshold, parsed_vitals):
                    vitals_mod += modifier
                    evidence_str = f"Vital sign criterion met: {condition} threshold={threshold}"
                    supporting_evidence.append(evidence_str)
                    tracer.add_step(
                        component="Scorer",
                        description=f"Vital sign matched for {match.disease_name}",
                        input_facts=[evidence_str],
                        output=f"vitals_modifier += {modifier}",
                        confidence_delta=modifier,
                    )

            # 3. Age modifier
            age_mod = 0.0
            age_rule = disease.get("age_range")
            if age_rule and age is not None:
                min_age = age_rule.get("min", 0)
                max_age = age_rule.get("max", 200)
                modifier = float(age_rule.get("modifier", 0.0))
                if min_age <= age <= max_age:
                    age_mod = modifier
                    evidence_str = f"Age {age} within risk range [{min_age}–{max_age}]"
                    supporting_evidence.append(evidence_str)
                    tracer.add_step(
                        component="Scorer",
                        description=f"Age modifier applied for {match.disease_name}",
                        input_facts=[evidence_str],
                        output=f"age_modifier += {modifier}",
                        confidence_delta=modifier,
                    )

            # 4. Gender modifier
            gender_mod = 0.0
            gender_rule = disease.get("gender_modifier")
            if gender_rule and gender:
                if gender_rule.get("gender", "").lower() == gender.lower():
                    gender_mod = float(gender_rule.get("modifier", 0.0))
                    evidence_str = f"Gender ({gender}) matches disease risk profile"
                    supporting_evidence.append(evidence_str)
                    tracer.add_step(
                        component="Scorer",
                        description=f"Gender modifier applied for {match.disease_name}",
                        input_facts=[evidence_str],
                        output=f"gender_modifier += {gender_mod}",
                        confidence_delta=gender_mod,
                    )

            # 5. Risk factor modifiers
            risk_mod = 0.0
            rf_modifiers: Dict[str, float] = disease.get("risk_factor_modifiers", {})
            for rf_id in identified_risk_factor_ids:
                if rf_id in rf_modifiers:
                    mod_val = float(rf_modifiers[rf_id])
                    risk_mod += mod_val
                    risk_factors_active.append(rf_id)
                    supporting_evidence.append(f"Risk factor present: {rf_id}")
                    tracer.add_step(
                        component="Scorer",
                        description=f"Risk factor '{rf_id}' active for {match.disease_name}",
                        input_facts=[f"risk_factor: {rf_id}"],
                        output=f"risk_modifier += {mod_val}",
                        confidence_delta=mod_val,
                    )

            # 6. Compute final confidence
            raw_confidence = (
                base_score
                + vitals_mod
                + age_mod
                + gender_mod
                + risk_mod
                - match.negation_penalty
            )
            confidence = _clamp(raw_confidence)

            # Build matched and missing rule IDs / categories
            rules_fired = []
            rules_not_fired = []
            cardinal_syms = disease.get("cardinal_symptoms", [])
            supportive_syms = disease.get("supportive_symptoms", [])

            for s in cardinal_syms:
                s_name = s.get("name", "").lower().strip()
                if s_name in match.matched_cardinal:
                    rules_fired.append(f"{match.disease_id}_cardinal_{s_name.replace(' ', '_')}")
                else:
                    rules_not_fired.append(f"{match.disease_id}_cardinal_{s_name.replace(' ', '_')}")

            for s in supportive_syms:
                s_name = s.get("name", "").lower().strip()
                if s_name in match.matched_supportive:
                    rules_fired.append(f"{match.disease_id}_supportive_{s_name.replace(' ', '_')}")
                else:
                    rules_not_fired.append(f"{match.disease_id}_supportive_{s_name.replace(' ', '_')}")

            for rf_id in rf_modifiers.keys():
                if rf_id in risk_factors_active:
                    rules_fired.append(f"{match.disease_id}_risk_{rf_id}")
                else:
                    rules_not_fired.append(f"{match.disease_id}_risk_{rf_id}")

            for criterion in disease.get("vital_sign_criteria", []):
                cond = criterion.get("condition", "")
                threshold = float(criterion.get("threshold", 0))
                if _evaluate_vital_condition(cond, threshold, parsed_vitals):
                    rules_fired.append(f"{match.disease_id}_vital_{cond}")
                else:
                    rules_not_fired.append(f"{match.disease_id}_vital_{cond}")

            if age_mod > 0.0:
                rules_fired.append(f"{match.disease_id}_age_range")
            elif age_rule:
                rules_not_fired.append(f"{match.disease_id}_age_range")

            if gender_mod > 0.0:
                rules_fired.append(f"{match.disease_id}_gender_modifier")
            elif gender_rule:
                rules_not_fired.append(f"{match.disease_id}_gender_modifier")

            # Extract guideline details
            guideline_obj = None
            guideline_data = disease.get("guideline")
            if guideline_data and isinstance(guideline_data, dict):
                guideline_obj = ClinicalGuideline(
                    organization=guideline_data.get("organization", ""),
                    guideline=guideline_data.get("guideline", ""),
                    version=str(guideline_data.get("version", "")),
                    recommendation=guideline_data.get("recommendation"),
                    strength=guideline_data.get("strength"),
                    evidence_level=guideline_data.get("evidence_level")
                )

            rule_meta_obj = RuleMetadata(
                rule_id=disease.get("rule_id", match.disease_id),
                rule_version=disease.get("rule_version", "1.0.0"),
                created_date=disease.get("created_date", "2026-01-01"),
                updated_date=disease.get("updated_date", "2026-07-03")
            )

            # Confidence Calibration
            calib_bonus = 0.0
            calib_bonus += 0.02 * len(risk_factors_active)
            if guideline_obj and guideline_obj.strength == "Strong":
                calib_bonus += 0.02

            clinical_scores_used = []
            if clinical_scores:
                for cs in clinical_scores:
                    is_match = False
                    if cs.name in ("HEART Score", "TIMI") and match.disease_id in ("acs", "unstable_angina"):
                        is_match = True
                    elif cs.name == "Wells Score (PE)" and match.disease_id in ("pulmonary_embolism", "pe"):
                        is_match = True
                    elif cs.name in ("qSOFA", "NEWS2", "CURB-65") and match.disease_id in ("sepsis", "pneumonia"):
                        is_match = True
                    elif cs.name == "CHA2DS2-VASc" and match.disease_id in ("atrial_fibrillation", "ischemic_stroke", "stroke"):
                        is_match = True
                    
                    if is_match:
                        clinical_scores_used.append(cs.name)
                        if any(term in cs.interpretation.lower() for term in ["high", "medium", "likely", "probable", "moderate"]):
                            calib_bonus += 0.05

            calib_bonus -= 0.02 * len(match.missing_cardinal)
            calibrated_conf = _clamp(confidence + calib_bonus)

            tracer.add_step(
                component="Scorer",
                description=f"Final confidence computed for {match.disease_name}",
                input_facts=[
                    f"base={base_score:.3f}",
                    f"vitals_mod={vitals_mod:.3f}",
                    f"age_mod={age_mod:.3f}",
                    f"gender_mod={gender_mod:.3f}",
                    f"risk_mod={risk_mod:.3f}",
                    f"negation_penalty={match.negation_penalty:.3f}",
                    f"calib_bonus={calib_bonus:.3f}",
                ],
                output=f"confidence = {calibrated_conf:.3f} (clamped to [0,1])",
                confidence_delta=0.0,
            )

            # Filter below threshold
            if calibrated_conf < min_threshold:
                continue

            # Build matched rule IDs
            matched_rules = []
            if match.matched_cardinal:
                matched_rules.append(f"{match.disease_id}_cardinal_match")
            if match.matched_supportive:
                matched_rules.append(f"{match.disease_id}_supportive_match")
            if vitals_mod > 0:
                matched_rules.append(f"{match.disease_id}_vitals_match")
            if age_mod > 0:
                matched_rules.append(f"{match.disease_id}_age_match")
            if risk_factors_active:
                matched_rules.append(f"{match.disease_id}_risk_factor_match")

            # Keep track of original confidence along with candidate
            candidates.append((confidence, CandidateDiagnosis(
                disease_id=match.disease_id,
                name=match.disease_name,
                category=match.disease_category,
                confidence=calibrated_conf,  # Save unrounded calibrated
                rank=rank_idx,
                supporting_symptoms=match.matched_cardinal + match.matched_supportive,
                missing_symptoms=match.missing_cardinal,
                negated_symptoms=match.negated_cardinal,
                matched_rules=matched_rules,
                risk_factors=risk_factors_active,
                supporting_evidence=supporting_evidence,
                reasoning_trace=list(tracer._steps),  # full trace snapshot
                guideline=guideline_obj,
                rule_metadata=rule_meta_obj,
                rules_fired=rules_fired,
                rules_not_fired=rules_not_fired,
                clinical_scores_used=clinical_scores_used,
            )))

        # Sort strictly by the original confidence to preserve diagnostic ranking behavior, with deterministic tie-breaking on disease_id
        candidates.sort(key=lambda item: (-item[0], item[1].disease_id))

        # Enforce monotonicity and resolve ranks/percents
        final_candidates: List[CandidateDiagnosis] = []
        for i, (orig_conf, c) in enumerate(candidates):
            c.rank = i + 1
            calib_conf = c.confidence
            if i > 0:
                if calib_conf > final_candidates[i - 1].confidence:
                    calib_conf = final_candidates[i - 1].confidence
            
            c.confidence = round(calib_conf, 4)
            c.confidence_percent = round(calib_conf * 100.0, 2)
            final_candidates.append(c)

        # Generate deterministic explanations (why_suggested, why_competing_lower)
        for i, c in enumerate(final_candidates):
            sym_summary = f"{len(c.supporting_symptoms)} matching symptoms ({', '.join(c.supporting_symptoms[:3])})" if c.supporting_symptoms else "no supporting symptoms"
            rf_summary = f", risk factors ({', '.join(c.risk_factors)})" if c.risk_factors else ""
            c.why_suggested = f"Suggested because of {sym_summary}{rf_summary}. Calibrated confidence is {c.confidence_percent}%."

            if i == 0:
                c.why_competing_lower = "This is the primary candidate with the highest overall symptom alignment and risk profile."
            else:
                better_cand = final_candidates[i - 1]
                c.why_competing_lower = (
                    f"Scored lower than competing diagnosis '{better_cand.name}' (confidence {better_cand.confidence_percent}%) "
                    f"due to lower symptom weights or fewer matching risk factor modifiers."
                )

        return final_candidates
