# =====================================================================
# SPEAKER ALIGNMENT LAYER (app/services/speaker_alignment.py)
# =====================================================================
# Purpose: Maps transcripts/timestamps to speakers and retroactively
#          enriches cached events in-place.
# =====================================================================

import uuid
import time
from typing import List
from app.services.providers.models import TranscriptEvent
from app.services.speaker_timeline import speaker_timeline_manager
from app.utils.logging import logger
from app.core.config import settings

class SpeakerAlignmentService:
    SPEAKER_ROLE_MAPPING = {
        "Speaker_0": "doctor",
        "Speaker_1": "patient",
        "Speaker_2": "attender",
        "UNKNOWN": "UNKNOWN"
    }

    def __init__(self):
        self._committed_cache = {}
        self._in_realignment = False
        self._session_mappings = {}

    def classify_sentence(self, text: str, baseline_speaker: str) -> tuple:
        """
        Classifies the sentence semantically to prevent turn-taking role leakage.
        Returns:
            (final_speaker, speaker_confidence, speaker_method)
        """
        import re
        # Strong clinical indicators for doctor and patient.
        doctor_indicators = [
            r"\byou should\b", r"\byou need to\b", r"\bi'll prescribe\b", r"\bi will prescribe\b",
            r"\blet me check\b", r"\blet me listen\b", r"\bcheck your\b", r"\bhow are you feeling today\b",
            r"\bdo you also have\b", r"\btake it three times\b", r"\bif you rest\b", r"\btake the medicine on time\b",
            r"\blet's check your\b", r"\bi will check\b", r"\bi'll check\b", r"\blet me examine\b", r"\bcome back\b",
            r"\bhave you taken\b", r"\bmedicine yet\b", r"\blooks like a\b", r"\bmild infection\b",
            r"\bprescribe you\b", r"\bsyrup for the cough\b", r"\bdrink plenty of water\b",
            r"\bget enough sleep\b", r"\bcome back if\b", r"\bslightly high\b"
        ]

        patient_indicators = [
            r"\bdoctor\b", r"\bfeeling well\b", r"\bi have had\b", r"\bi've had\b", r"\bshould i take\b",
            r"\bshould i\b", r"\btylenol\b", r"\badvil\b", r"\bparacetamol\b", r"\bdid not help\b",
            r"\bdidn't help\b", r"\bhow long will it take to feel\b", r"\bthank you\b", r"\bthanks doctor\b",
            r"\bwill i feel better\b", r"\bmy name is\b", r"\bi am called\b", r"\bi am feeling\b", r"\bi'm feeling\b",
            r"\bmy throat\b", r"\bmy head\b", r"\bjust some tylenol\b", r"\bi'll do that\b",
            r"\bthanks again\b", r"\bthank you, doctor\b", r"\bthank you doctor\b",
            r"\bthroat feels dry\b", r"\bcoughing a little\b", r"\bhad a headache\b"
        ]
        
        s_lower = text.lower()
        is_doc = any(re.search(pat, s_lower) for pat in doctor_indicators)
        is_pat = any(re.search(pat, s_lower) for pat in patient_indicators)
        
        if is_doc and not is_pat:
            return "doctor", 0.95, "semantic_fallback"
        if is_pat and not is_doc:
            return "patient", 0.95, "semantic_fallback"
            
        return baseline_speaker, 0.70, "diarization"

    def align_and_segment(self, session_id: str, role: str, res: dict, provider) -> List[TranscriptEvent]:
        """
        Maps ElevenLabs Scribe transcript output words to the SpeakerTimeline.
        Groups contiguous speaker segments at the sentence level, assigns monotonic
        sequence numbers, and generates a list of TranscriptEvents.
        """
        import re
        from app.services.session import session_manager
        
        session = session_manager.get_session(session_id)
        timeline = speaker_timeline_manager.get_timeline(session_id)
        
        is_partial = (res.get("type") == "partial")
        
        # Cache raw committed responses for global re-alignment
        if not is_partial and res.get("type") == "committed" and not getattr(self, "_in_realignment", False):
            if session_id not in self._committed_cache:
                self._committed_cache[session_id] = []
            if res not in [x[1] for x in self._committed_cache[session_id]]:
                self._committed_cache[session_id].append((role, res, provider))
        text = res.get("text", "")
        confidence = res.get("confidence")
        words = res.get("words", [])
        
        # 1. Handle Partial Transcripts: return as a single event mapped to the active speaker in the last 3 seconds
        if is_partial:
            # Estimate current audio time based on total bytes processed
            if session and session.streams.get(role):
                current_time = session.streams.get(role).buffer.total_bytes_received / 32000.0
            else:
                from app.services.diarization_worker import diarization_worker_manager
                state = diarization_worker_manager._states.get(session_id)
                current_time = (state.total_bytes_received / 32000.0) if state else 0.0
                    
            start_time = max(0.0, current_time - 3.0)
            end_time = current_time
            speaker_raw = timeline.get_speaker_for_range(start_time, end_time)
            speaker = self.SPEAKER_ROLE_MAPPING.get(speaker_raw, speaker_raw)
            
            seq = session.get_and_increment_transcript_seq(role) if session else (provider.get_and_increment_event_seq() if provider else 0)
            
            event = TranscriptEvent(
                session_id=session_id,
                role=role,
                sequence_number=seq,
                timestamp=time.time(),
                transcript=text,
                is_partial=True,
                is_final=False,
                confidence=confidence,
                provider="scribe_v2",
                event_id=str(uuid.uuid4()),
                speaker_id=speaker,
                speaker_label=speaker,
                start_time=start_time,
                end_time=end_time,
                text=text,
                speaker_confidence=0.70,
                speaker_method="diarization"
            )
            return [event]

        # 2. Committed/Final transcripts: Segment by sentence and align to speaker timeline
        sentences_data = []  # List of tuples: (sentence_text, start_time, end_time, speaker, confidence, method)
        mapping = self._session_mappings.get(session_id, self.SPEAKER_ROLE_MAPPING)
        
        if words:
            # Group words into sentences based on punctuation (. ? !)
            current_sent_words = []
            for w in words:
                current_sent_words.append(w)
                w_text = w.get("text", "").strip()
                if w_text and w_text[-1] in ('.', '?', '!'):
                    # Finalize sentence
                    s_text = "".join(x.get("text", "") for x in current_sent_words).strip()
                    if s_text:
                        # Find the true boundaries of the sentence by looking at all valid word timestamps
                        valid_starts = [x.get("start") for x in current_sent_words if x.get("start") is not None and x.get("start") > 0.001]
                        valid_ends = [x.get("end") for x in current_sent_words if x.get("end") is not None and x.get("end") > 0.001]
                        s_start = min(valid_starts) if valid_starts else (current_sent_words[0].get("start", 0.0) if current_sent_words[0].get("start") is not None else 0.0)
                        s_end = max(valid_ends) if valid_ends else (current_sent_words[-1].get("end", 0.0) if current_sent_words[-1].get("end") is not None else 0.0)
                        
                        spk_raw, reason, details = timeline.get_speaker_for_range(s_start, s_end, return_details=True)
                        speaker = mapping.get(spk_raw, spk_raw)
                        
                        # Priority 1: Timestamp speaker
                        speaker_conf = 0.98
                        speaker_method = "timestamp"
                        
                        # Priority 4: Semantic Fallback override
                        semantic_spk, semantic_conf, semantic_method = self.classify_sentence(s_text, speaker)
                        if semantic_method == "semantic_fallback":
                            speaker = semantic_spk
                            speaker_conf = semantic_conf
                            speaker_method = semantic_method
                            
                        sentences_data.append((s_text, s_start, s_end, speaker, speaker_conf, speaker_method))
                    current_sent_words = []
            if current_sent_words:
                s_text = "".join(x.get("text", "") for x in current_sent_words).strip()
                if s_text:
                    # Find the true boundaries of the sentence by looking at all valid word timestamps
                    valid_starts = [x.get("start") for x in current_sent_words if x.get("start") is not None and x.get("start") > 0.001]
                    valid_ends = [x.get("end") for x in current_sent_words if x.get("end") is not None and x.get("end") > 0.001]
                    s_start = min(valid_starts) if valid_starts else (current_sent_words[0].get("start", 0.0) if current_sent_words[0].get("start") is not None else 0.0)
                    s_end = max(valid_ends) if valid_ends else (current_sent_words[-1].get("end", 0.0) if current_sent_words[-1].get("end") is not None else 0.0)
                    
                    spk_raw, reason, details = timeline.get_speaker_for_range(s_start, s_end, return_details=True)
                    speaker = mapping.get(spk_raw, spk_raw)
                    
                    # Priority 1: Timestamp speaker
                    speaker_conf = 0.98
                    speaker_method = "timestamp"
                    
                    # Priority 4: Semantic Fallback override
                    semantic_spk, semantic_conf, semantic_method = self.classify_sentence(s_text, speaker)
                    if semantic_method == "semantic_fallback":
                        speaker = semantic_spk
                        speaker_conf = semantic_conf
                        speaker_method = semantic_method
                        
                    sentences_data.append((s_text, s_start, s_end, speaker, speaker_conf, speaker_method))
        else:
            # Fallback when words is empty/not present: split by regex and estimate time range
            sentence_texts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
            if sentence_texts:
                # Estimate current audio time
                if session and session.streams.get(role):
                    current_time = session.streams.get(role).buffer.total_bytes_received / 32000.0
                else:
                    from app.services.diarization_worker import diarization_worker_manager
                    state = diarization_worker_manager._states.get(session_id)
                    current_time = (state.total_bytes_received / 32000.0) if state else 0.0
                
                # Estimate total duration based on 15 characters per second
                duration = len(text) / 15.0
                start_time = max(0.0, current_time - duration)
                end_time = current_time
                total_len = max(1, len(text))
                
                curr_start = start_time
                for s_text in sentence_texts:
                    s_dur = (len(s_text) / total_len) * (end_time - start_time)
                    s_end = curr_start + s_dur
                    spk_raw, reason, details = timeline.get_speaker_for_range(curr_start, s_end, return_details=True)
                    speaker = mapping.get(spk_raw, spk_raw)
                    
                    # Priority 2: Diarization Confidence based on overlap ratio
                    overlaps = details.get("overlaps", {})
                    total_overlap = sum(overlaps.values()) if overlaps else 0.0
                    chosen_overlap = overlaps.get(spk_raw, 0.0)
                    overlap_ratio = (chosen_overlap / total_overlap) if total_overlap > 0.001 else 0.0
                    
                    if overlap_ratio >= 0.80:
                        speaker_conf = overlap_ratio
                        speaker_method = "diarization"
                    else:
                        speaker_conf = 0.60
                        speaker_method = "overlap"
                        
                    # Priority 4: Semantic Fallback override
                    semantic_spk, semantic_conf, semantic_method = self.classify_sentence(s_text, speaker)
                    if semantic_method == "semantic_fallback":
                        speaker = semantic_spk
                        speaker_conf = semantic_conf
                        speaker_method = semantic_method
                        
                    sentences_data.append((s_text, curr_start, s_end, speaker, speaker_conf, speaker_method))
                    curr_start = s_end

        # Debug logger to inspect sentence alignments and timeline segments
        logger.info(
            "SpeakerAlignmentDebug: Aligning sentences for committed transcript",
            extra={
                "has_words": bool(words),
                "words_count": len(words) if words else 0,
                "sentences": [(s[0], round(s[1], 2), round(s[2], 2), s[3]) for s in sentences_data],
                "timeline": [(round(t["start"], 2), round(t["end"], 2), t["label"]) for t in timeline.segments]
            }
        )

        # Group contiguous sentences belonging to the same speaker
        groups = []
        if sentences_data:
            curr_speaker = sentences_data[0][3]
            curr_texts = [sentences_data[0][0]]
            curr_start = sentences_data[0][1]
            curr_end = sentences_data[0][2]
            curr_conf = sentences_data[0][4]
            curr_method = sentences_data[0][5]
            
            for s_text, s_start, s_end, speaker, conf, method in sentences_data[1:]:
                if speaker == curr_speaker:
                    curr_texts.append(s_text)
                    curr_end = s_end
                    if conf > curr_conf:
                        curr_conf = conf
                        curr_method = method
                else:
                    groups.append((curr_speaker, curr_texts, curr_start, curr_end, curr_conf, curr_method))
                    curr_speaker = speaker
                    curr_texts = [s_text]
                    curr_start = s_start
                    curr_end = s_end
                    curr_conf = conf
                    curr_method = method
            groups.append((curr_speaker, curr_texts, curr_start, curr_end, curr_conf, curr_method))

        # Create a TranscriptEvent for each grouped segment
        events = []
        for speaker, g_texts, g_start, g_end, g_conf, g_method in groups:
            group_text = " ".join(g_texts).strip()
            group_text = re.sub(r'\s+', ' ', group_text)
            if not group_text:
                continue
                
            seq = session.get_and_increment_transcript_seq(role) if session else (provider.get_and_increment_event_seq() if provider else 0)
            
            event = TranscriptEvent(
                session_id=session_id,
                role=role,
                sequence_number=seq,
                timestamp=time.time(),
                transcript=group_text,
                is_partial=False,
                is_final=True,
                confidence=confidence,
                provider="scribe_v2",
                event_id=str(uuid.uuid4()),
                speaker_id=speaker,
                speaker_label=speaker,
                start_time=g_start,
                end_time=g_end,
                text=group_text,
                speaker_confidence=g_conf,
                speaker_method=g_method
            )
            events.append(event)
            
        # If DIARIZATION_DEBUG is enabled, perform diagnostics
        if getattr(settings, "DIARIZATION_DEBUG", False):
            try:
                import os
                import json
                
                pyannote_segments = [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in timeline.segments]
                word_speaker_mapping = []
                for w in words:
                    text = w.get("text", "")
                    if not text:
                        continue
                    
                    is_word = (w.get("type", "word") == "word")
                    if is_word and w.get("start") is not None and w.get("end") is not None:
                        spk_raw, reason, details = timeline.get_speaker_for_range(w["start"], w["end"], return_details=True)
                        spk = mapping.get(spk_raw, spk_raw)
                        word_speaker_mapping.append({
                            "word": text.strip(),
                            "start": w["start"],
                            "end": w["end"],
                            "raw_speaker": spk_raw,
                            "mapped_speaker": spk,
                            "reason": reason,
                            "overlaps": details.get("overlaps", {}),
                            "proximity_scores": details.get("proximity_scores", {}),
                            "continuity_details": {
                                "latest_end_before_start": details.get("latest_end_before_start"),
                                "tied_speakers": details.get("tied_speakers")
                            }
                        })
                    else:
                        word_speaker_mapping.append({
                            "word": text,
                            "start": w.get("start"),
                            "end": w.get("end"),
                            "reason": "Spacing or no timestamps"
                        })
                
                final_events = [
                    {
                        "event_id": event.event_id,
                        "sequence_number": event.sequence_number,
                        "speaker_id": event.speaker_id,
                        "transcript": event.transcript,
                        "start_time": event.start_time,
                        "end_time": event.end_time
                    }
                    for event in events
                ]
                
                os.makedirs("logs", exist_ok=True)
                log_filepath = f"logs/diarization_debug_{session_id}.json"
                debug_data = {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "pyannote_segments": pyannote_segments,
                    "words": words,
                    "word_speaker_mapping": word_speaker_mapping,
                    "final_events": final_events
                }
                with open(log_filepath, "w", encoding="utf-8") as f:
                    json.dump(debug_data, f, indent=2, ensure_ascii=False)
                    
                pyannote_log = "\n".join(f"{s['start']:.2f} - {s['end']:.2f}   {s['label']}" for s in pyannote_segments)
                words_log = "\n".join(f"{w.get('text', '').strip():<10} {w.get('start', 0.0):.2f}-{w.get('end', 0.0):.2f}" for w in words if w.get('text', '').strip())
                
                word_mapping_entries = []
                for entry in word_speaker_mapping:
                    if "raw_speaker" in entry:
                        overlaps_str = ", ".join(f"{k}: {v:.2f}s" for k, v in entry["overlaps"].items()) if entry["overlaps"] else "None"
                        word_mapping_entries.append(
                            f"Word: {entry['word']}\n"
                            f"Time: {entry['start']:.2f}-{entry['end']:.2f}\n"
                            f"Candidate overlaps: {overlaps_str}\n"
                            f"Chosen: {entry['mapped_speaker']} (raw: {entry['raw_speaker']})\n"
                            f"Reason: {entry['reason']}"
                        )
                word_to_speaker_log = "\n\n".join(word_mapping_entries)
                final_events_log = "\n".join(f"{event.speaker_id} : {event.transcript}" for event in events)
                
                debug_message = f"""
========== PYANNOTE ==========
{pyannote_log}

========== WORDS ==========
{words_log}

========== WORD → SPEAKER ==========
{word_to_speaker_log}

========== FINAL EVENTS ==========
{final_events_log}
"""
                logger.info(debug_message)
            except Exception as e:
                logger.error(f"Error during diarization debugging: {e}", exc_info=True)
                
        return events

    def _detect_speaker_swap(self, session_id: str) -> bool:
        """
        Analyzes the text content in the committed cache to detect if the raw speaker labels
        Speaker_0 and Speaker_1 have been swapped (i.e. Speaker_1 behaves like the doctor and
        Speaker_0 behaves like the patient).
        Returns True if a swap is detected.
        """
        if session_id not in self._committed_cache:
            return False

        # Doctor indicators (phrases/patterns characteristic of a doctor asking questions/giving instructions)
        doctor_patterns = [
            r"\bwhat's the problem\b",
            r"\bwhat brings you\b",
            r"\bhow long\b",
            r"\bdo you have a fever\b",
            r"\bdo you have any allergies\b",
            r"\btake this medicine\b",
            r"\bi'll check your\b",
            r"\bcheck your temperature\b",
            r"\bany other symptoms\b",
            r"\bdo you have any other\b",
            r"\bmedicine after meals\b",
            r"\bcheck your throat\b",
            r"\bfeel this way\b",
            r"\bi'll prescribe\b",
            r"\bi will prescribe\b",
            r"\byou should take\b",
            r"\bhow are you feeling today\b",
            r"\bis there anything else\b",
            r"\bcome back\b",
            r"\bif you rest\b",
            r"\btake the medicine on time\b"
        ]

        # Patient indicators (phrases/patterns characteristic of a patient responding/naming symptoms)
        patient_patterns = [
            r"\bdoctor\b",
            r"\bheadache\b",
            r"\bsore throat\b",
            r"\bpainkiller\b",
            r"\bpainkillers\b",
            r"\bfever\b(?!.*\?)",  # fever not in a question
            r"\bsymptoms\b(?!.*\?)",
            r"\bdizziness\b",
            r"\bnausea\b",
            r"\bvomiting\b",
            r"\bthank you doctor\b",
            r"\bthanks doctor\b",
            r"\bgood morning doctor\b",
            r"\byes doctor\b",
            r"\bno doctor\b",
            r"\bfeeling well\b",
            r"\bi've had\b",
            r"\bi have had\b",
            r"\bi have a\b",
            r"\bi have been\b",
            r"\bsince yesterday\b",
            r"\btylenol\b",
            r"\badvil\b",
            r"\bparacetamol\b",
            r"\bdid not help\b",
            r"\bdidn't help\b",
            r"\bhow long will it take to feel\b",
            r"\bthank you\b",
            r"\bwill i feel better\b"
        ]

        scores = {"Speaker_0": 0, "Speaker_1": 0}
        timeline = speaker_timeline_manager.get_timeline(session_id)
        import re

        for role, res, provider in self._committed_cache[session_id]:
            text = res.get("text", "")
            words = res.get("words", [])
            
            # Temporary sentences data mapping sentences to raw speakers
            sentences = []
            if words:
                current_sent_words = []
                for w in words:
                    current_sent_words.append(w)
                    w_text = w.get("text", "").strip()
                    if w_text and w_text[-1] in ('.', '?', '!'):
                        s_text = "".join(x.get("text", "") for x in current_sent_words).strip()
                        if s_text:
                            valid_starts = [x.get("start") for x in current_sent_words if x.get("start") is not None and x.get("start") > 0.001]
                            valid_ends = [x.get("end") for x in current_sent_words if x.get("end") is not None and x.get("end") > 0.001]
                            s_start = min(valid_starts) if valid_starts else 0.0
                            s_end = max(valid_ends) if valid_ends else 0.0
                            speaker_raw = timeline.get_speaker_for_range(s_start, s_end)
                            sentences.append((s_text, speaker_raw))
                        current_sent_words = []
                if current_sent_words:
                    s_text = "".join(x.get("text", "") for x in current_sent_words).strip()
                    if s_text:
                        valid_starts = [x.get("start") for x in current_sent_words if x.get("start") is not None and x.get("start") > 0.001]
                        valid_ends = [x.get("end") for x in current_sent_words if x.get("end") is not None and x.get("end") > 0.001]
                        s_start = min(valid_starts) if valid_starts else 0.0
                        s_end = max(valid_ends) if valid_ends else 0.0
                        speaker_raw = timeline.get_speaker_for_range(s_start, s_end)
                        sentences.append((s_text, speaker_raw))
            else:
                sentence_texts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                for s_text in sentence_texts:
                    # Estimate speaker mapping
                    speaker_raw = timeline.get_speaker_for_range(0.0, 1000.0)
                    sentences.append((s_text, speaker_raw))

            for s_text, speaker_raw in sentences:
                if speaker_raw not in ["Speaker_0", "Speaker_1"]:
                    continue
                s_text_lower = s_text.lower()
                
                # Check doctor indicators
                for pattern in doctor_patterns:
                    if re.search(pattern, s_text_lower):
                        scores[speaker_raw] += 1
                        
                # Check patient indicators
                for pattern in patient_patterns:
                    if re.search(pattern, s_text_lower):
                        scores[speaker_raw] -= 1

        logger.info(
            "SpeakerAlignmentService: Semantic role matching scores",
            extra={"session_id": session_id, "scores": scores}
        )
        return scores["Speaker_1"] > scores["Speaker_0"]

    def realign_session(self, session_id: str) -> None:
        """
        Re-aligns and re-segments all cached committed STT responses for a session
        using the final correct SpeakerTimeline, and updates the TranscriptEventBus.
        """
        if session_id not in self._committed_cache:
            logger.info("SpeakerAlignmentService: No cached committed responses for session", extra={"session_id": session_id})
            return

        # Determine if Speaker_0 and Speaker_1 roles are swapped semantically
        if self._detect_speaker_swap(session_id):
            logger.info(
                "SpeakerAlignmentService: Swapping Speaker_0 and Speaker_1 roles semantically for session",
                extra={"session_id": session_id}
            )
            self._session_mappings[session_id] = {
                "Speaker_0": "patient",
                "Speaker_1": "doctor",
                "Speaker_2": "attender",
                "UNKNOWN": "UNKNOWN"
            }
        else:
            self._session_mappings[session_id] = self.SPEAKER_ROLE_MAPPING
            
        logger.info(
            "SpeakerAlignmentService: Starting global re-alignment for session",
            extra={"session_id": session_id, "cached_count": len(self._committed_cache[session_id])}
        )
        
        # 1. Clear all existing final events for this session in the TranscriptEventBus
        from app.services.transcript_bus import transcript_bus
        transcript_bus.clear_buffer(session_id)
        
        # Clear clinical state engine cache for clean realignment replay
        from app.services.clinical import clinical_state_engine
        clinical_state_engine.clear_state(session_id)
        
        # 2. Reset the session sequence counter so the sequence numbers are perfectly monotonic and start from 0
        from app.services.session import session_manager
        session = session_manager.get_session(session_id)
        if session:
            for r in ["doctor", "patient", "attender"]:
                session.transcript_seq_counters[r] = 0
                
        # 3. Temporarily disable caching in align_and_segment during the re-alignment pass
        self._in_realignment = True
        try:
            for role, res, provider in self._committed_cache[session_id]:
                events = self.align_and_segment(session_id, role, res, provider)
                for event in events:
                    transcript_bus.publish(event)
        finally:
            self._in_realignment = False

    def enrich_cached_events(self, session_id: str) -> None:
        """
        Scans the cached events in the TranscriptEventBus and retroactively
        resolves 'UNKNOWN' speaker labels based on the updated SpeakerTimeline.
        Updates the events in-place, preserving event_id and sequence_number.
        """
        from app.services.transcript_bus import transcript_bus
        
        timeline = speaker_timeline_manager.get_timeline(session_id)
        recent_events = transcript_bus.get_recent_events(session_id)
        
        for event in recent_events:
            # Enrich completed events that have an UNKNOWN speaker
            if event.is_final and event.speaker_id == "UNKNOWN":
                if event.start_time is not None and event.end_time is not None:
                    resolved_raw = timeline.get_speaker_for_range(event.start_time, event.end_time)
                    if resolved_raw != "UNKNOWN":
                        mapping = self._session_mappings.get(session_id, self.SPEAKER_ROLE_MAPPING)
                        resolved = mapping.get(resolved_raw, resolved_raw)
                        event.speaker_id = resolved
                        event.speaker_label = resolved
                        event.version += 1
                        event.updated_at = time.time()
                        logger.info(
                            "SpeakerAlignmentService: Retroactively enriched speaker label",
                            extra={
                                "session_id": session_id,
                                "event_id": event.event_id,
                                "seq": event.sequence_number,
                                "speaker": resolved,
                                "text": event.text
                            }
                        )

    def clear_session(self, session_id: str) -> None:
        """Clears cached data and dynamic role mappings for the given session to prevent memory leaks."""
        self._session_mappings.pop(session_id, None)
        self._committed_cache.pop(session_id, None)

# Singleton Instance
speaker_alignment_service = SpeakerAlignmentService()
