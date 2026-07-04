from __future__ import annotations

"""
workflow.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

from typing import Dict, List, Set

# List of all valid consultation stages in order
CLINICAL_STAGES = [
    "Greeting",
    "Chief Complaint",
    "HPI",
    "PMH",
    "Medication Review",
    "Allergy Review",
    "Examination",
    "Investigations",
    "Assessment",
    "Plan",
    "Follow-up",
    "Completed",
]

# State transition matrix defining valid next stages for each current stage
STAGED_TRANSITIONS = {
    "Greeting": {"Chief Complaint", "Completed"},
    "Chief Complaint": {"Greeting", "HPI", "Completed"},
    "HPI": {"Chief Complaint", "PMH", "Completed"},
    "PMH": {"HPI", "Medication Review", "Completed"},
    "Medication Review": {"PMH", "Allergy Review", "Completed"},
    "Allergy Review": {"Medication Review", "Examination", "Completed"},
    "Examination": {"Allergy Review", "Investigations", "Completed"},
    "Investigations": {"Examination", "Assessment", "Completed"},
    "Assessment": {"Investigations", "Plan", "Completed"},
    "Plan": {"Assessment", "Follow-up", "Completed"},
    "Follow-up": {"Plan", "Completed"},
    "Completed": {"Greeting", "Assessment", "Plan"},
}

# Timeout limits (in seconds) for each stage before expiration
STAGE_TIMEOUTS = {
    "Greeting": 60.0,
    "Chief Complaint": 300.0,
    "HPI": 600.0,
    "PMH": 300.0,
    "Medication Review": 300.0,
    "Allergy Review": 300.0,
    "Examination": 600.0,
    "Investigations": 600.0,
    "Assessment": 600.0,
    "Plan": 600.0,
    "Follow-up": 300.0,
    "Completed": 3600.0,
}

# Rollback permissions (whether stage is allowed to be rolled back to)
ROLLBACK_ALLOWED_STAGES = {
    "Greeting",
    "Chief Complaint",
    "HPI",
    "PMH",
    "Medication Review",
    "Allergy Review",
    "Examination",
    "Investigations",
    "Assessment",
    "Plan",
    "Follow-up",
}

# Estimated completion progress percentage mapping
STAGE_PROGRESS_ESTIMATES = {
    "Greeting": 5,
    "Chief Complaint": 15,
    "HPI": 40,
    "PMH": 50,
    "Medication Review": 60,
    "Allergy Review": 65,
    "Examination": 75,
    "Investigations": 80,
    "Assessment": 85,
    "Plan": 95,
    "Follow-up": 98,
    "Completed": 100,
}

# Required facts for each stage to validate progression readiness
STAGE_FACT_REQUIREMENTS = {
    "Greeting": [],
    "Chief Complaint": ["symptoms"],
    "HPI": ["symptoms"],
    "PMH": ["medical_history"],
    "Medication Review": [],
    "Allergy Review": [],
    "Examination": [],
    "Investigations": [],
    "Assessment": [],
    "Plan": [],
    "Follow-up": [],
    "Completed": [],
}
