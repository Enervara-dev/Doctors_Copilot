import json
import re

# Load raw debug data
with open('logs/diarization_debug_consultation-xyz-0a93c9.json', 'r') as f:
    data = json.load(f)

from app.services.speaker_timeline import SpeakerTimeline

timeline = SpeakerTimeline('test')
timeline.segments = data['pyannote_segments']

text = "I'll prescribe you some antibiotics and syrup for the cough. How long will it take to feel better? Usually within three to four days if you rest and take the medicine on time. All right. I'll do that. Thank you, Doctor."

sentence_texts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# Define semantic indicators
doctor_indicators = [
    r"\byou should\b", r"\byou need to\b", r"\bi'll prescribe\b", r"\bi will prescribe\b",
    r"\blet me check\b", r"\blet me listen\b", r"\bcheck your\b", r"\bhow are you feeling today\b",
    r"\bdo you also have\b", r"\btake it three times\b", r"\bif you rest\b", r"\btake the medicine on time\b"
]

patient_indicators = [
    r"\bdoctor\b", r"\bfeeling well\b", r"\bi have had\b", r"\bi've had\b", r"\bshould i take\b",
    r"\bshould i\b", r"\btylenol\b", r"\badvil\b", r"\bparacetamol\b", r"\bdid not help\b",
    r"\bhow long will it take to feel\b", r"\bthank you\b", r"\bthanks doctor\b", r"\bwill i feel better\b"
]

def classify_sentence(s_text, baseline_speaker):
    s_lower = s_text.lower()
    is_doc = any(re.search(pat, s_lower) for pat in doctor_indicators)
    is_pat = any(re.search(pat, s_lower) for pat in patient_indicators)
    
    if is_doc and not is_pat:
        return "doctor"
    if is_pat and not is_doc:
        return "patient"
    return baseline_speaker

# Mapping
mapping = {"Speaker_0": "doctor", "Speaker_1": "patient", "Speaker_2": "attender"}

current_time = 54.21666666666667
duration = len(text) / 15.0
start_time = max(0.0, current_time - duration)
end_time = current_time
total_len = max(1, len(text))

curr_start = start_time
for s_text in sentence_texts:
    s_dur = (len(s_text) / total_len) * (end_time - start_time)
    s_end = curr_start + s_dur
    speaker_raw = timeline.get_speaker_for_range(curr_start, s_end)
    mapped_speaker = mapping.get(speaker_raw, speaker_raw)
    
    final_speaker = classify_sentence(s_text, mapped_speaker)
    print(f'Sentence: {s_text!r} | Baseline: {mapped_speaker} -> Final: {final_speaker}')
    curr_start = s_end
