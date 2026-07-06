import unittest
from datetime import datetime
from clinical_intelligence.rule_engine.models import (
    ClinicalReasoningResult,
    RedFlagAlert,
    ClinicalAlert,
    AlertType,
)
from clinical_intelligence.presentation.serializer import serialize_reasoning_result

class TestPresentationSerializer(unittest.TestCase):

    def test_serializer_hides_red_flags_but_preserves_reasoning(self):
        # 1. Arrange: construct a dummy ClinicalReasoningResult with red flags and alerts
        red_flag_alert = RedFlagAlert(
            flag_id="sepsis_red_flag",
            condition="Sepsis",
            severity="CRITICAL",
            supporting_evidence=["fever"],
            recommended_action="Sepsis-6 bundle",
            triggered_by=["fever"],
        )
        
        red_flag_clinical_alert = ClinicalAlert(
            alert_id="alert_sepsis_red_flag",
            alert_type=AlertType.RED_FLAG,
            severity="CRITICAL",
            title="⚠️ RED FLAG: Sepsis",
            message="Sepsis-6 bundle",
            supporting_evidence=["fever"],
        )
        
        info_clinical_alert = ClinicalAlert(
            alert_id="alert_info",
            alert_type=AlertType.RISK_FACTOR,
            severity="INFO",
            title="Risk Factor",
            message="Diabetes",
            supporting_evidence=["history"],
        )
        
        result = ClinicalReasoningResult(
            session_id="test-session-123",
            reasoning_timestamp=datetime.utcnow(),
            candidate_diagnoses=[],
            red_flags=[red_flag_alert],
            risk_factors=[],
            recommended_investigations=[],
            missing_information=[],
            contraindications=[],
            drug_interactions=[],
            clinical_alerts=[red_flag_clinical_alert, info_clinical_alert],
        )

        # 2. Act (show_red_flags=True)
        serialized_with_flags = serialize_reasoning_result(result, show_red_flags=True)
        self.assertEqual(len(serialized_with_flags["red_flags"]), 1)
        self.assertEqual(len(serialized_with_flags["clinical_alerts"]), 2)
        
        # 3. Act (show_red_flags=False)
        serialized_without_flags = serialize_reasoning_result(result, show_red_flags=False)
        
        # 4. Assert: verify presentation shows NO red flags
        self.assertEqual(len(serialized_without_flags["red_flags"]), 0)
        # Verify other types of alerts are preserved (like the info alert)
        self.assertEqual(len(serialized_without_flags["clinical_alerts"]), 1)
        self.assertEqual(serialized_without_flags["clinical_alerts"][0]["alert_id"], "alert_info")
        self.assertEqual(serialized_without_flags["clinical_alerts"][0]["alert_type"], AlertType.RISK_FACTOR.value)

        # Assert: verify original object remains completely untouched
        self.assertEqual(len(result.red_flags), 1)
        self.assertEqual(len(result.clinical_alerts), 2)


if __name__ == "__main__":
    unittest.main()
