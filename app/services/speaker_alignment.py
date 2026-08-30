"""
Speaker Turn & Transcript Aligner

Purpose:
Performs word-level alignment, mapping diarized speaker segments to transcribed dialogue turns to determine who said what.

Module/Component: Diarization & Alignment
"""

"""
Speaker Turn & Transcript Aligner

Purpose:
Performs word-level alignment, mapping diarized speaker segments to transcribed dialogue turns to determine who said what.

Module/Component: Diarization & Alignment
"""

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

    # Maximum silence, in seconds, that may sit inside one speaker turn.
    TURN_GAP_SECONDS = 1.5

    def __init__(self):
        self._committed_cache = {}
        self._in_realignment = False
        self._session_mappings = {}

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
        sentences_data = []  # List of tuples: (sentence_text, start_time, end_time, speaker, confidence, method, spk_raw)
        if session_id not in self._session_mappings:
            self._session_mappings[session_id] = self.SPEAKER_ROLE_MAPPING.copy()
        mapping = self._session_mappings[session_id]
        
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
                        
                            
                        sentences_data.append((s_text, s_start, s_end, speaker, speaker_conf, speaker_method, spk_raw))
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
                    
                        
                    sentences_data.append((s_text, s_start, s_end, speaker, speaker_conf, speaker_method, spk_raw))
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
                        
                        
                    sentences_data.append((s_text, curr_start, s_end, speaker, speaker_conf, speaker_method, spk_raw))
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
            curr_spk_raw = sentences_data[0][6]
            
            for s_text, s_start, s_end, speaker, conf, method, spk_raw in sentences_data[1:]:
                # A silence longer than TURN_GAP_SECONDS is a turn boundary even when the
                # diarizer reports the same speaker on both sides. Without this, a single
                # commit spanning half a minute collapses into one event and a short reply
                # arrives carrying the preceding conversation.
                same_turn = (speaker == curr_speaker
                             and (s_start - curr_end) <= self.TURN_GAP_SECONDS)
                if same_turn:
                    curr_texts.append(s_text)
                    curr_end = s_end
                    if conf > curr_conf:
                        curr_conf = conf
                        curr_method = method
                else:
                    groups.append((curr_speaker, curr_texts, curr_start, curr_end, curr_conf, curr_method, curr_spk_raw))
                    curr_speaker = speaker
                    curr_texts = [s_text]
                    curr_start = s_start
                    curr_end = s_end
                    curr_conf = conf
                    curr_method = method
                    curr_spk_raw = spk_raw
            groups.append((curr_speaker, curr_texts, curr_start, curr_end, curr_conf, curr_method, curr_spk_raw))

        # Create a TranscriptEvent for each grouped segment
        events = []
        for speaker, g_texts, g_start, g_end, g_conf, g_method, g_spk_raw in groups:
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
            
            # Queue background speaker verification task
            if not is_partial and not getattr(self, "_in_realignment", False):
                from app.services.speaker_verification.speaker_verification import speaker_verification_service
                speaker_verification_service.enqueue_verification_task(
                    session_id=session_id,
                    start_time=g_start,
                    end_time=g_end,
                    proposed_role=speaker,
                    raw_speaker=g_spk_raw
                )

            
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

    # Aggregated, per-cluster role evidence. Applied once per session to a whole
    # speaker cluster rather than to individual sentences: a diarized cluster is a
    # person, so evidence pooled over everything they said is far more reliable than
    # keyword-matching one utterance. Scoring a sentence in isolation is what made the
    # same voice read as "doctor" in one turn and "patient" in the next.
    DOCTOR_PATTERNS = [
        r"\bwhat brings you\b",
        r"\bwhat's the problem\b",
        r"\bhow long have you\b",
        r"\bany other symptoms\b",
        r"\blet me (?:check|listen|examine)\b",
        r"\bi'?ll (?:check|prescribe|examine)\b",
        r"\bi will (?:check|prescribe|examine)\b",
        r"\bcheck your\b",
        r"\byou should (?:take|rest|avoid)\b",
        r"\byou need to\b",
        r"\btake (?:this|the) medicine\b",
        r"\bcome back if\b",
        r"\bhow are you feeling\b",
        r"\bis there anything else\b",
        r"\bhave you taken\b",
        r"\bprescribe you\b",
        r"\byour blood pressure is\b",
        r"\byour pulse is\b",
        r"\bdrink plenty of\b",
        r"\bdo you have (?:a |any )?(?:fever|cough|pain|allergies)\b",
    ]

    PATIENT_PATTERNS = [
        r"\bi have been having\b",
        r"\bi'?ve had\b",
        r"\bi have had\b",
        r"\bi have a\b",
        r"\bi feel\b",
        r"\bi'?m feeling\b",
        r"\bi am feeling\b",
        r"\bmy (?:throat|head|chest|stomach|back|father|mother)\b",
        r"\bsince yesterday\b",
        r"\bshould i take\b",
        r"\bwill i feel better\b",
        r"\bhow long will it take\b",
        r"\bthank you\b",
        r"\bmy name is\b",
        r"\bi was diagnosed\b",
        r"\bi take\b",
        r"\bdid ?n'?t help\b",
        r"\bdid not help\b",
        r"\bi smoke\b",
    ]

    def _cluster_sentences(self, session_id: str) -> dict:
        """Buckets every cached committed sentence by its raw diarization label."""
        import re
        timeline = speaker_timeline_manager.get_timeline(session_id)
        buckets: dict = {}

        for _role, res, _provider in self._committed_cache.get(session_id, []):
            words = res.get("words") or []
            spans = []
            if words:
                current = []
                for w in words:
                    current.append(w)
                    wt = (w.get("text") or "").strip()
                    if wt and wt[-1] in ('.', '?', '!'):
                        spans.append(current)
                        current = []
                if current:
                    spans.append(current)
            else:
                continue  # without timings a sentence cannot be tied to a cluster

            for sent_words in spans:
                s_text = "".join(x.get("text", "") for x in sent_words).strip()
                if not s_text:
                    continue
                starts = [x.get("start") for x in sent_words if x.get("start") is not None]
                ends = [x.get("end") for x in sent_words if x.get("end") is not None]
                if not starts or not ends:
                    continue
                raw = timeline.get_speaker_for_range(min(starts), max(ends))
                if isinstance(raw, tuple):
                    raw = raw[0]
                buckets.setdefault(raw, []).append(s_text)
        return buckets

    def assign_cluster_roles(self, session_id: str) -> dict:
        """
        Assigns one role per diarized speaker cluster for the whole session.

        Each cluster is scored once over all of its utterances, then roles are handed
        out best-first. The result is a stable raw-label -> role mapping: the same
        voice keeps the same role for every message in the session.
        """
        import re
        buckets = self._cluster_sentences(session_id)
        if not buckets:
            return self.SPEAKER_ROLE_MAPPING.copy()

        scores = {}
        for raw, sentences in buckets.items():
            if raw in ("UNKNOWN", None):
                continue
            doc = pat = 0
            for s in sentences:
                low = s.lower()
                doc += sum(1 for p in self.DOCTOR_PATTERNS if re.search(p, low))
                pat += sum(1 for p in self.PATIENT_PATTERNS if re.search(p, low))
            scores[raw] = {"doctor": doc, "patient": pat, "turns": len(sentences)}

        if not scores:
            return self.SPEAKER_ROLE_MAPPING.copy()

        mapping = {"UNKNOWN": "UNKNOWN"}
        remaining = dict(scores)

        # Doctor first: the strongest clinician signal wins the role outright.
        doc_pick = max(remaining, key=lambda r: (remaining[r]["doctor"] - remaining[r]["patient"],
                                                 remaining[r]["turns"]))
        mapping[doc_pick] = "doctor"
        remaining.pop(doc_pick)

        if remaining:
            pat_pick = max(remaining, key=lambda r: (remaining[r]["patient"] - remaining[r]["doctor"],
                                                     remaining[r]["turns"]))
            mapping[pat_pick] = "patient"
            remaining.pop(pat_pick)

        for raw in remaining:
            mapping[raw] = "attender"

        # Any cluster never seen in the cache keeps its positional default.
        for raw, role in self.SPEAKER_ROLE_MAPPING.items():
            mapping.setdefault(raw, role)

        logger.info(
            "SpeakerAlignmentService: Assigned roles per speaker cluster",
            extra={"session_id": session_id, "scores": scores, "mapping": mapping}
        )
        return mapping


    def realign_session(self, session_id: str) -> None:
        """
        Re-aligns and re-segments all cached committed STT responses for a session
        using the final correct SpeakerTimeline, and updates the TranscriptEventBus.
        """
        if session_id not in self._committed_cache:
            logger.info("SpeakerAlignmentService: No cached committed responses for session", extra={"session_id": session_id})
            return

        # Assign one role per speaker cluster from evidence pooled over the whole
        # session, rather than flipping a fixed Speaker_0/Speaker_1 pair.
        mapping = self.assign_cluster_roles(session_id)

        # Merge in verified speaker locks
        from app.services.speaker_verification.speaker_verification import speaker_verification_service
        locked_verifications = speaker_verification_service._locked_mappings.get(session_id, {})
        for raw, role in locked_verifications.items():
            mapping[raw] = role

        self._session_mappings[session_id] = mapping
            
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
            rebuilt = []
            for role, res, provider in self._committed_cache[session_id]:
                rebuilt.extend(self.align_and_segment(session_id, role, res, provider))
            for event in self._stitch_commit_boundaries(rebuilt):
                transcript_bus.publish(event)
        finally:
            self._in_realignment = False

    def stitch_against_published(self, session_id: str, events: List[TranscriptEvent]) -> List[TranscriptEvent]:
        """
        Applies commit-boundary stitching to freshly aligned events.

        Realignment replays the whole session at once, but commits that arrive after it
        are published straight through, so the join has to be attempted against the
        event already on the bus as well as within this batch.
        """
        if not events:
            return events

        from app.services.transcript_bus import transcript_bus
        published = [e for e in transcript_bus.get_recent_events(session_id) if e.is_final]
        if published:
            joined = self._stitch_commit_boundaries([published[-1]] + list(events))
            # The tail event was mutated in place if the first new event continued it.
            return joined[1:] if joined and joined[0] is published[-1] else joined
        return self._stitch_commit_boundaries(events)

    def _stitch_commit_boundaries(self, events: List[TranscriptEvent]) -> List[TranscriptEvent]:
        """
        Rejoins a word that the provider split across two commits.

        Scribe closes a commit on a byte boundary, not a word boundary, so a term can
        arrive as "...I take aml" followed by "odipine daily." Both halves are labelled
        with the same speaker, but the broken token defeats entity extraction and the
        medication is lost outright. Where two consecutive same-speaker events meet
        mid-word, splice them back into one.
        """
        if not events:
            return events

        merged = [events[0]]
        for event in events[1:]:
            prev = merged[-1]
            prev_text = (prev.transcript or "").rstrip()
            next_text = (event.transcript or "").lstrip()
            mid_word = (
                prev.speaker_id == event.speaker_id
                and prev_text and next_text
                and prev_text[-1] not in ".?!,;:"
                and next_text[0].islower()
                and (event.start_time or 0.0) - (prev.end_time or 0.0) <= self.TURN_GAP_SECONDS
            )
            if not mid_word:
                merged.append(event)
                continue

            prev.transcript = prev_text + next_text
            prev.text = prev.transcript
            prev.end_time = event.end_time
            prev.version += 1
            prev.updated_at = time.time()
            logger.info(
                "SpeakerAlignmentService: Rejoined a word split across a commit boundary",
                extra={
                    "session_id": prev.session_id,
                    "speaker": prev.speaker_id,
                    "joined_at": f"...{prev_text[-12:]}|{next_text[:12]}...",
                },
            )
        return merged

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
        from app.services.speaker_verification.speaker_verification import speaker_verification_service
        speaker_verification_service.clear_session(session_id)


# Singleton Instance
speaker_alignment_service = SpeakerAlignmentService()
