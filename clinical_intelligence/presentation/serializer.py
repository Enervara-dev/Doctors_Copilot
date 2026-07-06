from clinical_intelligence.rule_engine.models import ClinicalReasoningResult, AlertType

def serialize_reasoning_result(result: ClinicalReasoningResult, *, show_red_flags: bool = True) -> dict:
    """
    Pure serialization function that converts a ClinicalReasoningResult to a dictionary.
    Optionally filters out red flag alerts from the output presentation for safety or UI preferences.
    """
    if show_red_flags:
        return result.model_dump()

    # Create a deep copy of the result model to perform type-safe filtering
    filtered = result.model_copy(deep=True)
    filtered.red_flags = []
    filtered.clinical_alerts = [
        alert for alert in filtered.clinical_alerts
        if alert.alert_type != AlertType.RED_FLAG
    ]
    return filtered.model_dump()
