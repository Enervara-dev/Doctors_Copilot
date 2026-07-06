import json
from datetime import datetime
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine
from clinical_intelligence.presentation.serializer import serialize_reasoning_result

def main():
    print("=== MANUAL PRESENTATION LAYER & LLM BOOTSTRAP VERIFICATION ===")

    # 1. Initialize reasoning engine
    engine = ClinicalReasoningEngine()

    # 2. Construct clinical state containing fever (which triggers sepsis_red_flag)
    state = {
        "session_id": "manual-verification-session-1",
        "patient_info": {
            "patient_name": "Jane Doe",
            "age": 45,
            "gender": "Female"
        },
        "chief_complaint": ["fever"],
        "symptoms": [
            {
                "entity_id": "symptom_1",
                "name": "fever",
                "severity": "high",
                "duration": "2 days",
                "status": "Active",
                "confidence": "High",
                "present": "True",
                "updated_at": str(datetime.utcnow().timestamp())
            }
        ],
        "vital_signs": {
            "bp": "120/80",
            "pulse": "85",
            "temperature": "38.9",
            "spo2": "98"
        },
        "medical_history": [],
        "current_medications": [],
        "allergies": []
    }

    # 3. Execute clinical reasoning with LLM enabled
    print("\nRunning ClinicalReasoningEngine.reason()...")
    result = engine.reason(state, run_llm=True)

    # 4. Print internal results (to verify correctness of reasoning engine)
    print(f"\n[INTERNAL ENGINE OUTPUT]")
    print(f" - Detected Red Flags internally: {len(result.red_flags)}")
    for rf in result.red_flags:
        print(f"   * {rf.flag_id}: {rf.condition}")

    # Verify that LLM tasks ran and were not bypassed
    print(f" - LLM Differential Explanations: {result.llm_differential_explanations}")
    print(f" - LLM Adaptive Questions: {len(result.llm_adaptive_questions)}")

    # 5. Serialize with show_red_flags=False
    print("\nSerializing with show_red_flags=False...")
    serialized = serialize_reasoning_result(result, show_red_flags=False)

    print(f"\n[SERIALIZED PRESENTATION OUTPUT]")
    print(f" - Red Flags list length: {len(serialized['red_flags'])}")
    print(f" - Clinical Alerts list length: {len(serialized['clinical_alerts'])}")
    for alert in serialized["clinical_alerts"]:
        print(f"   * Alert: {alert['alert_id']} ({alert['alert_type']}) - {alert['title']}")

    # 6. Verify assertions
    assert len(result.red_flags) > 0, "Red flags should be detected internally"
    assert len(serialized["red_flags"]) == 0, "Red flags should be filtered from presentation"
    assert not any(a["alert_type"] == "red_flag" for a in serialized["clinical_alerts"]), "Red flag alerts should be filtered"
    print("\nVerification successful! Both internal audit completeness and presentation filtering assertions passed.")

if __name__ == "__main__":
    main()
