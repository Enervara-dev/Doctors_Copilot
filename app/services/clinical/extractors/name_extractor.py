"""
name_extractor.py Component

Purpose:
Part of the App workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: App
"""

import re
from typing import Optional

class NameExtractor:
    def __init__(self):
        # Case-sensitive name extraction patterns to enforce capitalized name parts
        # mr/mrs/ms/dr can be case-insensitive, but name must be capitalized
        self.intro_patterns = [
            r"\b[mM]y\s+name\s+is\s+((?:[mM]r\b\.?|[mM]rs\b\.?|[mI]s\b\.?|[mM]iss\b|[dD]r\b\.?)\s*[A-Z][a-zA-Z]+)\b",
            r"\b[mM]y\s+name\s+is\s+([A-Z][a-zA-Z]+)\b",
            r"\b[iI]\s+am\s+((?:[mM]r\b\.?|[mM]rs\b\.?|[mI]s\b\.?|[mM]iss\b|[dD]r\b\.?)\s*[A-Z][a-zA-Z]+)\b",
            r"\b[iI]\s+am\s+([A-Z][a-zA-Z]+)\b",
            r"\b[cCall]+\s+me\s+([A-Z][a-zA-Z]+)\b",
            r"\b[cCalled]+\s+([A-Z][a-zA-Z]+)\b"
        ]
        self.addressing_patterns = [
            r"\b(?:[hH]ello|[hH]i|[mM]orning|[gG]ood\s+morning|[aA]fternoon|[gG]ood\s+afternoon|[eE]vening|[gG]ood\s+evening)\s*,?\s*((?:[mM]r\b\.?|[mM]rs\b\.?|[mI]s\b\.?|[mM]iss\b|[dD]r\b\.?)\s*[A-Z][a-zA-Z]+)\b",
            r"\b(?:[hH]ello|[hH]i|[mM]orning|[gG]ood\s+morning|[aA]fternoon|[gG]ood\s+afternoon|[eE]vening|[gG]ood\s+evening)\s+([A-Z][a-zA-Z]+)\b"
        ]
        
        # Blocklist of common words that should never be extracted as names
        self.blocklist = {
            "not", "sick", "fine", "good", "bad", "tired", "here", "there", 
            "well", "better", "ill", "pain", "fever", "cough", "doctor", 
            "patient", "attender", "happy", "sad", "sore", "dry", "cold",
            "hot", "warm", "chills", "ache", "aches", "nausea", "vomiting"
        }

    def extract_name(self, text: str, speaker_id: Optional[str] = None) -> Optional[str]:
        """
        Extracts patient name from natural conversations.
        Enforces strict validation to prevent guessing from filler.
        """
        if not text:
            return None

        # 1. If speaker is a doctor, they might be addressing the patient.
        if speaker_id == "doctor":
            for pattern in self.addressing_patterns:
                for match in re.finditer(pattern, text):
                    candidate = match.group(1).strip()
                    if candidate.lower().startswith("dr"):
                        continue
                    name_part = candidate.split()[-1]
                    if name_part.lower() not in self.blocklist:
                        return candidate
            return None

        # 2. If speaker is patient (or unknown/default), check introductions first
        for pattern in self.intro_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                name_part = candidate.split()[-1]
                if name_part.lower() not in self.blocklist:
                    return candidate

        return None
