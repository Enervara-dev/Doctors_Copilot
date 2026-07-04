import json
from clinical_intelligence.rule_engine.engine import ClinicalReasoningEngine

def main():
    engine = ClinicalReasoningEngine()
    
    # Define a clinical state payload
    state = {
        "session_id": "test-session-graph-001",
        "patient_info": {"age": 68, "gender": "male"},
        "chief_complaint": ["chest pain"],
        "symptoms": [
            {"name": "chest pain", "status": "Active", "confidence": "High"},
            {"name": "shortness of breath", "status": "Active", "confidence": "Medium"},
            {"name": "sweating", "status": "Active", "confidence": "High"}
        ],
        "vital_signs": {"respiratory_rate": "24", "spo2": "95", "temperature": "37.0", "bp_systolic": "90", "pulse": "110"},
        "medical_history": ["hypertension", "diabetes", "hyperlipidemia", "atrial_fibrillation"],
        "current_medications": ["aspirin"],
        "allergies": [],
        "family_history": [],
        "ecg_findings": "ST elevation",
        "lab_results": {"troponin": "highly elevated"}
    }
    
    print("=== CLINICAL REASONING PASS ===")
    res = engine.reason(state, debug=True)
    
    print("\n--- Output Schema Fields ---")
    print(f"Session ID: {res.session_id}")
    print(f"Engine Version: {res.engine_version}")
    print(f"Ruleset Version Summary: {res.rule_set_version}")
    print(f"Graph Metadata Loaded: {res.graph_metadata}")
    
    print("\n--- Candidate Diagnoses Enriched (Top 3) ---")
    for cand in res.candidate_diagnoses[:3]:
        print(f"\n* Candidate: {cand.name} (Category: {cand.category})")
        print(f"  Confidence: {cand.confidence} ({cand.confidence_percent}%)")
        print(f"  Why Suggested: {cand.why_suggested}")
        print(f"  Graph Evidence: {cand.graph_evidence}")
        if cand.graph_evidence and cand.graph_evidence.relationship_paths:
            print("  Paths:")
            for p in cand.graph_evidence.relationship_paths:
                node_names = [n.id for n in p.nodes]
                rels = [r.rel_type for r in p.relationships]
                path_str = ""
                for idx in range(len(rels)):
                    path_str += f"{node_names[idx]} -> [{rels[idx]}] -> "
                path_str += node_names[-1]
                print(f"    - {path_str}")

    print("\n--- Debug Diagnostics Telemetry ---")
    if res.debug_diagnostics:
        print(json.dumps(res.debug_diagnostics.model_dump(), indent=2))

if __name__ == "__main__":
    main()
