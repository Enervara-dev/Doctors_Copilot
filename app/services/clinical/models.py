"""
models.py Component

Purpose:
Part of the App workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: App
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SymptomEntity(BaseModel):
    name: str                           # Canonical ID, e.g. "symptom_cough"
    display_name: str                   # Human readable name, e.g. "Cough"
    severity: Optional[str] = None
    duration: Optional[str] = None
    present: bool = True
    confidence: float = 1.0
    onset: Optional[str] = None         # e.g., "started yesterday"
    progression: Optional[str] = None   # e.g., "worsening"
    frequency: Optional[str] = None     # e.g., "intermittent"
    character: Optional[str] = None     # e.g., "sudden onset"
    source_text: Optional[str] = None   # Provenance: sentence string
    utterance_id: Optional[str] = None  # Provenance: transcript event ID
    speaker: Optional[str] = None       # "patient" or "doctor"

class MedicationEntity(BaseModel):
    name: str                           # generic name
    display_name: Optional[str] = None   # brand/reported name, e.g. "Tylenol"
    canonical_name: Optional[str] = None # generic name, e.g. "Paracetamol"
    drug_id: Optional[str] = None        # canonical ID, e.g. "drug_paracetamol"
    present: bool = True
    confidence: float = 1.0
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None
    instructions: Optional[str] = None
    prn: bool = False
    source_text: Optional[str] = None   # Provenance
    utterance_id: Optional[str] = None  # Provenance
    speaker: Optional[str] = None       # "patient" or "doctor"

class DiagnosisEntity(BaseModel):
    name: str
    present: bool = True
    confidence: float = 1.0

class ProcedureEntity(BaseModel):
    name: str
    present: bool = True
    confidence: float = 1.0

class RiskFactorEntity(BaseModel):
    name: str
    present: bool = True
    confidence: float = 1.0

class FamilyHistoryEntity(BaseModel):
    relationship: str
    condition: str
    present: bool = True
    confidence: float = 1.0

class ClinicalFact(BaseModel):
    entity_type: str
    value: str
    confidence: float = 1.0

class ClinicalExtractionResult(BaseModel):
    session_id: str
    speaker_id: str
    timestamp: float
    symptoms: List[SymptomEntity] = Field(default_factory=list)
    medications: List[MedicationEntity] = Field(default_factory=list)
    diagnoses: List[DiagnosisEntity] = Field(default_factory=list)
    procedures: List[ProcedureEntity] = Field(default_factory=list)
    risk_factors: List[RiskFactorEntity] = Field(default_factory=list)
    family_histories: List[FamilyHistoryEntity] = Field(default_factory=list)

class PatientInfo(BaseModel):
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

class ExaminationFinding(BaseModel):
    finding: str
    body_system: Optional[str] = None
    confidence: float = 1.0
    source: str = "examination"
    timestamp: Optional[float] = None
    source_text: Optional[str] = None
    utterance_id: Optional[str] = None

class VitalObservation(BaseModel):
    value: Optional[str] = None
    unit: Optional[str] = None
    qualitative: Optional[str] = None
    source: str = "examination"
    confidence: float = 1.0
    timestamp: Optional[float] = None
    source_text: Optional[str] = None
    utterance_id: Optional[str] = None

class VitalSigns(BaseModel):
    bp: Optional[VitalObservation] = None
    pulse: Optional[VitalObservation] = None
    temperature: Optional[VitalObservation] = None
    spo2: Optional[VitalObservation] = None
    weight: Optional[VitalObservation] = None
    height: Optional[VitalObservation] = None

class TreatmentPlan(BaseModel):
    medicines: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    investigations: List[str] = Field(default_factory=list)
    advice: List[str] = Field(default_factory=list)

class ClinicalState(BaseModel):
    session_id: str
    patient_info: PatientInfo = Field(default_factory=PatientInfo)
    chief_complaint: List[str] = Field(default_factory=list)
    symptoms: List[Dict[str, Optional[Any]]] = Field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    vital_signs: VitalSigns = Field(default_factory=VitalSigns)
    diagnosis_tentative: List[str] = Field(default_factory=list)
    treatment_plan: TreatmentPlan = Field(default_factory=TreatmentPlan)
    follow_up: List[str] = Field(default_factory=list)
    examination: List[ExaminationFinding] = Field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    version: int = 1
