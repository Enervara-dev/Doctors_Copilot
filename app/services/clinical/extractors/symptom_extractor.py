"""
symptom_extractor.py Component

Purpose:
Part of the App workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: App
"""

import re
from typing import List, Dict, Any, Optional
from app.services.clinical.models import SymptomEntity
from app.services.clinical.dictionary.loader import dictionary_manager
from app.services.clinical.extractors.negation import NegationDetector
from app.services.clinical.extractors.confidence import ConfidenceScorer
from app.services.clinical.extractors.context import context_manager

#: Words patients drop into the middle of a complaint without changing it:
#: "my face SUDDENLY drooped", "chest feels REALLY tight". A closed list, so
#: a multi-word alias tolerates them without the pattern becoming permissive.
_INTERPOSED = (
    "suddenly", "really", "very", "quite", "just", "now", "also", "still",
    "slightly", "completely", "totally", "absolutely", "definitely",
    "sometimes", "always", "constantly", "badly", "terribly",
)
_GAP = r"\s+(?:(?:" + "|".join(_INTERPOSED) + r")\s+)?"


def _alias_pattern(term: str) -> str:
    """
    A word-bounded pattern for one alias, tolerant of an interposed adverb.

    Single-word aliases are unchanged. A multi-word alias allows at most one
    filler between each pair of words, which is what lets "face suddenly
    drooped" match the alias "face drooped" without the pattern matching
    arbitrary text in between.
    """
    words = [re.escape(w) for w in term.split()]
    if len(words) == 1:
        return r"\b" + words[0] + r"\b"
    return r"\b" + _GAP.join(words) + r"\b"


class SymptomExtractor:
    def __init__(self):
        self.negation_detector = NegationDetector()
        self.symptoms_dict = dictionary_manager.get_symptoms()
        self.anatomy_dict = dictionary_manager.get_anatomy()
        
        # Expanded severity terms
        self.severity_terms = [
            "very severe", "excruciating", "moderate", "extreme", 
            "minimal", "severe", "slight", "mild"
        ]
        
        # General duration terms
        self.duration_terms = [
            "today", "yesterday", "two days", "three days", 
            "one week", "two weeks", "one month"
        ]

    def _clean_duration_value(self, val: str) -> str:
        return re.sub(r'^(?:for|since)\s+', '', val, flags=re.IGNORECASE).strip()

    def _detect_durations(self, clause: str) -> List[Dict[str, Any]]:
        durations = []
        # Pattern A: "for [X] years/days/weeks/months"
        pattern_for = r'\bfor\s+(?:\d+(?:-\d+)?|about\s+a|several|a\s+few|a|one|two|three|four|five|six|seven)?\s*(?:day|week|month|year)s?\b'
        # Pattern B: "since [last] Monday/week/etc."
        pattern_since = r'\bsince\s+(?:last\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year|college|childhood|yesterday|surgery)\b'
        # Pattern C: "on and off for [X]"
        pattern_on_off = r'\bon\s+and\s+off\s+for\s+\w+\s*(?:day|week|month|year)s?\b'
        
        for pattern in [pattern_for, pattern_since, pattern_on_off]:
            for match in re.finditer(pattern, clause, re.IGNORECASE):
                durations.append({
                    "value": self._clean_duration_value(match.group(0)),
                    "start": match.start()
                })
                
        # Also check for explicit single duration terms (like today, yesterday)
        for term in self.duration_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, clause, re.IGNORECASE):
                if not any(d["start"] <= match.start() < d["start"] + len(d["value"]) for d in durations):
                    durations.append({
                        "value": self._clean_duration_value(term),
                        "start": match.start()
                    })
                
        # Deduplicate overlapping matches (prefer longer match)
        durations.sort(key=lambda d: len(d["value"]), reverse=True)
        unique_durations = []
        for d in durations:
            if not any(u["start"] <= d["start"] < u["start"] + len(u["value"]) for u in unique_durations):
                unique_durations.append(d)
                
        return unique_durations

    def _detect_temporal_features(self, clause: str) -> dict:
        features = {"onset": None, "progression": None, "frequency": None, "character": None}
        c_lower = clause.lower()
        
        # Onset
        onset_patterns = [
            r"\b(started\s+(?:yesterday|today|last\s+night|this\s+morning|\d+\s+days?\s+ago|\d+\s+weeks?\s+ago))\b",
            r"\b(began\s+(?:yesterday|today|last\s+night|this\s+morning|\d+\s+days?\s+ago|\d+\s+weeks?\s+ago))\b",
            r"\b(since\s+(?:yesterday|today|last\s+night|this\s+morning|\d+\s+days?\s+ago|\d+\s+weeks?\s+ago))\b",
            r"\bonset\s+since\s+([^\.]+)\b"
        ]
        for pat in onset_patterns:
            m = re.search(pat, c_lower)
            if m:
                features["onset"] = m.group(1).strip()
                break
                
        # Progression
        prog_patterns = [
            r"\b(worsening|getting\s+worse|progressive|progressively\s+worse)\b",
            r"\b(improving|getting\s+better|resolving|resolved)\b"
        ]
        for pat in prog_patterns:
            m = re.search(pat, c_lower)
            if m:
                features["progression"] = m.group(1).strip()
                break
                
        # Frequency
        freq_patterns = [
            r"\b(intermittent|constant|episodic|on\s+and\s+off|comes\s+and\s+goes)\b"
        ]
        for pat in freq_patterns:
            m = re.search(pat, c_lower)
            if m:
                features["frequency"] = m.group(1).strip()
                break
                
        # Character
        char_patterns = [
            r"\b(sudden\s+onset|sudden|abrupt|gradual\s+onset|gradual)\b"
        ]
        for pat in char_patterns:
            m = re.search(pat, c_lower)
            if m:
                features["character"] = m.group(1).strip()
                break
                
        return features

    def extract_symptoms(self, text: str, speaker_id: Optional[str] = None, session_id: Optional[str] = None, section: str = "Chief Complaint", utterance_id: Optional[str] = None) -> List[SymptomEntity]:
        results = []
        if not text or not text.strip():
            return results

        processed_text = self.negation_detector.expand_contractions(text)
        sentences = [s.strip() for s in re.split(r'[.!?\n]', processed_text) if s.strip()]
        if not sentences:
            sentences = [processed_text]

        entity_counter = 0

        for sentence in sentences:
            # Clause isolation
            clauses = [c.strip() for c in re.split(r'[;]|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b', sentence, flags=re.IGNORECASE) if c.strip()]
            if not clauses:
                clauses = [sentence]

            for clause in clauses:
                detected_symptoms = []
                
                # Search symptoms
                for canonical, synonyms in self.symptoms_dict.items():
                    all_syns = [canonical] + synonyms
                    for term in all_syns:
                        pattern = _alias_pattern(term)
                        for match in re.finditer(pattern, clause, re.IGNORECASE):
                            span_start = match.start()
                            span_end = match.end()
                            # Longest match wins. "severe headache" and
                            # "headache" both match the same words, and
                            # dictionary order decides which is seen first, so
                            # rejecting only the shorter-arriving-second case
                            # left one utterance producing two symptoms. The
                            # more specific term is the one the patient said.
                            if any(
                                s["start"] <= span_start and span_end <= s["end"]
                                for s in detected_symptoms
                            ):
                                continue
                            detected_symptoms = [
                                s
                                for s in detected_symptoms
                                if not (span_start <= s["start"] and s["end"] <= span_end)
                            ]
                            detected_symptoms.append({
                                "name": canonical,
                                "matched_text": term,
                                "start": span_start,
                                "end": span_end
                            })

                if detected_symptoms:
                    # Detect severities
                    detected_severities = []
                    for term in self.severity_terms:
                        pattern = r'\b' + re.escape(term) + r'\b'
                        for match in re.finditer(pattern, clause, re.IGNORECASE):
                            span_start = match.start()
                            span_end = match.end()
                            if not any(s["start"] <= span_start and span_end <= s["end"] for s in detected_severities):
                                detected_severities.append({
                                    "value": term,
                                    "start": span_start,
                                    "end": span_end
                                })
                    
                    # Detect durations
                    detected_durations = self._detect_durations(clause)

                    # Map modifiers to closest symptoms in the clause
                    def find_closest_symptom(pos, symptoms):
                        if not symptoms:
                            return None
                        return min(symptoms, key=lambda s: abs(s["start"] - pos))

                    severity_mapping = {}
                    # Filter out severities modifying non-symptoms (e.g. "mild infection")
                    valid_severities = []
                    for sev in detected_severities:
                        sev_val = sev["value"].lower()
                        sev_index = clause.lower().find(sev_val)
                        if sev_index != -1:
                            following = clause.lower()[sev_index + len(sev_val):].strip()
                            if re.match(r'^(?:infection|illness|viral|virus|case|cold|symptoms)\b', following):
                                continue
                        valid_severities.append(sev)

                    if len(valid_severities) == 1:
                        # Distribute single modifier to all symptoms in listed clause
                        for sym in detected_symptoms:
                            key = (sym["name"], sym["start"])
                            severity_mapping[key] = valid_severities[0]
                    else:
                        for sev in valid_severities:
                            closest_sym = find_closest_symptom(sev["start"], detected_symptoms)
                            if closest_sym:
                                key = (closest_sym["name"], closest_sym["start"])
                                severity_mapping[key] = sev

                    duration_mapping = {}
                    if len(detected_durations) == 1:
                        # Distribute single modifier to all symptoms in listed clause
                        for sym in detected_symptoms:
                            key = (sym["name"], sym["start"])
                            duration_mapping[key] = detected_durations[0]
                    else:
                        for dur in detected_durations:
                            closest_sym = find_closest_symptom(dur["start"], detected_symptoms)
                            if closest_sym:
                                key = (closest_sym["name"], closest_sym["start"])
                                duration_mapping[key] = dur

                    for sym in detected_symptoms:
                        key = (sym["name"], sym["start"])
                        
                        sev_match = severity_mapping.get(key)
                        associated_severity = sev_match["value"] if sev_match else None

                        dur_match = duration_mapping.get(key)
                        associated_duration = dur_match["value"] if dur_match else None

                        # Negation detection
                        is_neg = self.negation_detector.is_negated(clause, sym["matched_text"], sym["start"], sym["end"])

                        # Expected speaker scoring logic: Patient reports symptoms
                        expected_speaker = (speaker_id == "patient")

                        # Ambiguity and confirmation features can be integrated via the context manager
                        exact_match = (sym["matched_text"].lower() == sym["name"].replace("symptom_", "").replace("_", " ").lower())
                        synonym_match = not exact_match

                        # Check for negation ambiguity (like hedging words in the clause)
                        low_hedging = {"maybe", "perhaps", "possibly", "suspect", "suspected", "might"}
                        negation_ambiguity = any(w in clause.lower() for w in low_hedging)

                        confidence_score = ConfidenceScorer.calculate_confidence(
                            exact_match=exact_match,
                            synonym_match=synonym_match,
                            expected_speaker=expected_speaker,
                            cross_turn_confirmed=False,  # Can be updated statefully later
                            duration_attached=bool(associated_duration),
                            severity_attached=bool(associated_severity),
                            negation_ambiguity=negation_ambiguity
                        )

                        temp_features = self._detect_temporal_features(clause)
                        display_name = sym["name"].replace("symptom_", "").replace("_", " ").capitalize()

                        results.append(SymptomEntity(
                            name=sym["name"],
                            display_name=display_name,
                            severity=associated_severity,
                            duration=associated_duration,
                            present=not is_neg,
                            confidence=confidence_score,
                            onset=temp_features["onset"],
                            progression=temp_features["progression"],
                            frequency=temp_features["frequency"],
                            character=temp_features["character"],
                            source_text=sentence,
                            utterance_id=utterance_id,
                            speaker=speaker_id
                        ))

        return results
