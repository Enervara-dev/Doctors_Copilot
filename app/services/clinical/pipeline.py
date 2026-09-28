"""
Dialogue-to-Clinical Ingestion Pipeline

Purpose:
Coordinates normalisation, concept extraction, negation classification, and state updates from new transcript text.

Module/Component: Clinical Ingestion
"""

"""
Dialogue-to-Clinical Ingestion Pipeline

Purpose:
Coordinates normalisation, concept extraction, negation classification, and state updates from new transcript text.

Module/Component: Clinical Ingestion
"""

from typing import Dict, Any
from app.services.clinical.normalizer import ClinicalNormalizer
from app.services.clinical.extractor import ClinicalEntityExtractor
from app.services.clinical.models import ClinicalExtractionResult

import re
from typing import Any, Iterable, List

# The split the sub-extractors use when they record a sentence as provenance
# (symptom_extractor.py:130, medication_extractor.py:28). Matching it here is
# what lets a normalized fragment be paired back to the words that produced it.
_SENTENCE_SPLIT = re.compile(r"[.!?\n]")


def _sentences(text: str) -> List[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _original_fragments(original: str, normalized: str) -> Dict[str, str]:
    """
    Maps each normalized fragment back to the words actually spoken.

    Normalization is in-place regex substitution - it replaces phrases, never
    splits or joins sentences - so the two texts have the same sentence count
    and can be paired by position. When they do not, which would mean that
    assumption no longer holds, no sentence mapping is produced and the
    normalized text is left alone: showing a clinician the canonical form is
    poor, but guessing at an alignment would be worse.
    """
    mapping: Dict[str, str] = {}

    original_parts = _sentences(original)
    normalized_parts = _sentences(normalized)
    if len(original_parts) == len(normalized_parts):
        for normalized_part, original_part in zip(normalized_parts, original_parts):
            mapping.setdefault(normalized_part, original_part)

    # Extractors that record the whole utterance rather than one sentence
    # (vitals_extractor) need the same restoration.
    if normalized.strip():
        mapping.setdefault(normalized.strip(), original.strip())
    return mapping


def _restore_source_text(entities: Iterable[Any], mapping: Dict[str, str]) -> None:
    """
    Puts the spoken words back on the provenance field.

    Only `source_text` is rewritten. Everything the rule engine reads - the
    canonical `name`, the negation flag, the severity - is left exactly as the
    extractors produced it, so no diagnosis, score or rule can change because
    of this.
    """
    for entity in entities:
        recorded = getattr(entity, "source_text", None)
        if recorded and recorded in mapping:
            entity.source_text = mapping[recorded]


class ClinicalProcessingPipeline:
    def __init__(self):
        self.normalizer = ClinicalNormalizer()
        self.extractor = ClinicalEntityExtractor()

    def process(self, data: Dict[str, Any]) -> ClinicalExtractionResult:
        """
        Orchestrates the clinical processing pipeline:
        1. Normalizes the colloquial layman terms in the transcript.
        2. Extracts clinical entities (symptoms, medications, diagnoses, procedures,
           risk factors, and family histories).
           Passes the speaker_id to the extractor to ignore clinician questions.
        3. Constructs and returns a structured ClinicalExtractionResult Pydantic model.
        """
        session_id = data.get("session_id", "")
        speaker_id = data.get("speaker_id", "")
        transcript = data.get("transcript", "")
        timestamp = data.get("timestamp", 0.0)
        utterance_id = data.get("utterance_id")

        # 1. Normalize
        normalized_transcript = self.normalizer.normalize(transcript)

        # 2. Extract (passing speaker_id, session_id, timestamp, and utterance_id)
        extraction_results = self.extractor.extract(
            normalized_transcript, 
            speaker_id=speaker_id,
            session_id=session_id,
            timestamp=timestamp,
            utterance_id=utterance_id
        )

        # 2b. Extraction runs on normalized text, which is right for matching
        # and wrong for provenance: it leaves evidence reading "severe
        # symptom_chest_pain" instead of the words the patient used. The
        # canonical form is already carried by each entity's `name`, so the
        # source fragment is restored to the original transcript here - after
        # every match has been made, and without touching anything the rule
        # engine reads.
        fragments = _original_fragments(transcript, normalized_transcript)
        if fragments:
            for group in extraction_results.values():
                _restore_source_text(group, fragments)

        # 3. Construct and return result model
        return ClinicalExtractionResult(
            session_id=session_id,
            speaker_id=speaker_id,
            timestamp=timestamp,
            symptoms=extraction_results.get("symptoms", []),
            medications=extraction_results.get("medications", []),
            diagnoses=extraction_results.get("diagnoses", []),
            procedures=extraction_results.get("procedures", []),
            risk_factors=extraction_results.get("risk_factors", []),
            family_histories=extraction_results.get("family_histories", [])
        )
