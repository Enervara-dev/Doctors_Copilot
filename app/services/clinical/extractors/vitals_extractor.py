"""
vitals_extractor.py Component

Purpose:
Part of the App workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: App
"""

import re
from typing import Dict, Optional, Any
from app.services.clinical.models import VitalObservation

class VitalsExtractor:
    def extract_vitals(self, text: str, utterance_id: Optional[str] = None) -> Dict[str, Optional[VitalObservation]]:
        results = {
            "bp": None,
            "pulse": None,
            "temperature": None,
            "spo2": None,
            "weight": None,
            "height": None
        }
        
        if not text:
            return results

        text_lower = text.lower()

        # BP
        bp_match = re.search(r'\b(?:bp|blood\s+pressure)\s*(?:is|of|was|be|at)?\s*(\d{2,3}\s*/\s*\d{2,3})\b', text_lower)
        if bp_match:
            results["bp"] = VitalObservation(
                value=bp_match.group(1).replace(" ", ""),
                unit="mmHg",
                qualitative=None,
                source="examination",
                confidence=0.98,
                source_text=text,
                utterance_id=utterance_id
            )
        else:
            if "hypertensive" in text_lower:
                results["bp"] = VitalObservation(value=None, unit=None, qualitative="hypertensive", source="examination", confidence=0.90, source_text=text, utterance_id=utterance_id)
            elif "hypotensive" in text_lower:
                results["bp"] = VitalObservation(value=None, unit=None, qualitative="hypotensive", source="examination", confidence=0.90, source_text=text, utterance_id=utterance_id)
            elif re.search(r'\b(?:bp|blood\s+pressure)\s+(?:is\s+)?normal\b', text_lower):
                results["bp"] = VitalObservation(value=None, unit=None, qualitative="normal", source="examination", confidence=0.90, source_text=text, utterance_id=utterance_id)

        # Pulse
        pulse_match = re.search(r'\b(?:pulse|heart\s+rate|hr)\s*(?:is|of|was|be|at)?\s*(\d{2,3})\b', text_lower)
        if pulse_match:
            results["pulse"] = VitalObservation(
                value=pulse_match.group(1),
                unit="bpm",
                qualitative=None,
                source="examination",
                confidence=0.98,
                source_text=text,
                utterance_id=utterance_id
            )
        else:
            if "tachycardic" in text_lower or "tachycardia" in text_lower:
                results["pulse"] = VitalObservation(value=None, unit=None, qualitative="tachycardic", source="examination", confidence=0.90, source_text=text, utterance_id=utterance_id)
            elif re.search(r'\b(?:pulse|heart\s+rate|hr)\s+(?:is\s+)?normal\b', text_lower):
                results["pulse"] = VitalObservation(value=None, unit=None, qualitative="normal", source="examination", confidence=0.90, source_text=text, utterance_id=utterance_id)

        # Temperature
        temp_match = re.search(r'\b(?:temp|temperature)\s*(?:is|of|was|be|at)?\s*(\d{2,3}(?:\.\d)?)\s*(?:f|c|degrees)?\b', text_lower)
        if temp_match:
            val = temp_match.group(1)
            qual = None
            try:
                val_f = float(val)
                if val_f >= 38.0 or val_f >= 100.4:
                    qual = "elevated"
            except ValueError:
                pass
            results["temperature"] = VitalObservation(
                value=val,
                unit="C" if "." in val and float(val) < 45.0 else "F",
                qualitative=qual,
                source="examination",
                confidence=0.98,
                source_text=text,
                utterance_id=utterance_id
            )
        else:
            # Check qualitative terms
            is_high = any(term in text_lower for term in ["slightly high", "high temperature", "feverish", "pyrexial"]) or (re.search(r'\btemp(?:erature)?\s+is\s+(?:slightly\s+)?high\b', text_lower) is not None)
            is_febrile = (re.search(r'\bfebrile\b', text_lower) is not None)
            if is_high or is_febrile:
                results["temperature"] = VitalObservation(
                    value=None,
                    unit=None,
                    qualitative="elevated",
                    source="examination",
                    confidence=0.90,
                    source_text=text,
                    utterance_id=utterance_id
                )
            elif "afebrile" in text_lower:
                results["temperature"] = VitalObservation(
                    value=None,
                    unit=None,
                    qualitative="normal",
                    source="examination",
                    confidence=0.90,
                    source_text=text,
                    utterance_id=utterance_id
                )
            elif re.search(r'\btemp(?:erature)?\s+(?:is\s+)?normal\b', text_lower):
                results["temperature"] = VitalObservation(
                    value=None,
                    unit=None,
                    qualitative="normal",
                    source="examination",
                    confidence=0.90,
                    source_text=text,
                    utterance_id=utterance_id
                )

        # SpO2
        spo2_match = re.search(r'\b(?:spo2|oxygen|saturation|o2\s+sat)\s*(?:is|of|was|be|at)?\s*(\d{2,3})\s*%?\b', text_lower)
        if spo2_match:
            results["spo2"] = VitalObservation(
                value=spo2_match.group(1),
                unit="%",
                qualitative=None,
                source="examination",
                confidence=0.98,
                source_text=text,
                utterance_id=utterance_id
            )

        # Weight
        weight_match = re.search(r'\b(?:weight|weighs?)\s*(?:is|of|was|be|at)?\s*(\d{2,3})\s*(?:kg|lbs|pounds)?\b', text_lower)
        if weight_match:
            results["weight"] = VitalObservation(
                value=weight_match.group(1),
                unit="kg" if "kg" in text_lower else "lbs",
                qualitative=None,
                source="examination",
                confidence=0.98,
                source_text=text,
                utterance_id=utterance_id
            )

        # Height
        height_match = re.search(r'\b(?:height|tall)\s*(?:is|of|was|be|at)?\s*(\d(?:\'\d{1,2}\"?)?)\b', text_lower)
        if height_match:
            results["height"] = VitalObservation(
                value=height_match.group(1),
                unit="ft/in",
                qualitative=None,
                source="examination",
                confidence=0.98,
                source_text=text,
                utterance_id=utterance_id
            )

        return results
