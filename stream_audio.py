"""
stream_audio.py Component

Purpose:
Part of the Stream Audio.Py workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Stream Audio.Py
"""

import asyncio
import websockets
import wave
import sys
import json
import urllib.request
import audioop

WS_URL = "ws://127.0.0.1:8005/v1/streaming/audio"
TOKEN = "production-secure-token-change-me"
SESSION_ID = "consultation-xyz-123"

async def stream_audio(file_path: str, role: str):
    print(f"Opening WAV file: {file_path}", flush=True)
    try:
        wf = wave.open(file_path, 'rb')
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr, flush=True)
        return
    except Exception as e:
        print(f"Error opening WAV file: {e}", file=sys.stderr, flush=True)
        return

    # Check WAV properties
    channels = wf.getnchannels()
    sample_rate = wf.getframerate()
    sample_width = wf.getsampwidth()
    
    print(f"Original WAV Format: Channels={channels}, Rate={sample_rate}Hz, Width={sample_width*8}-bit", flush=True)
    
    # Read raw audio frames
    raw_data = wf.readframes(wf.getnframes())
    wf.close()
    
    # 1. Convert to mono if it is stereo or multi-channel
    if channels > 1:
        print(f"Converting stereo/multi-channel ({channels} channels) to mono...", flush=True)
        raw_data = audioop.tomono(raw_data, sample_width, 0.5, 0.5)
        channels = 1
        
    # 2. Convert sample width to 16-bit PCM (2 bytes)
    if sample_width != 2:
        print(f"Converting sample width from {sample_width} bytes to 2 bytes (16-bit)...", flush=True)
        raw_data = audioop.lin2lin(raw_data, sample_width, 2)
        sample_width = 2
        
    # 3. Resample audio sample rate to 16000Hz
    if sample_rate != 16000:
        print(f"Resampling audio from {sample_rate}Hz to 16000Hz...", flush=True)
        state = None
        raw_data, state = audioop.ratecv(raw_data, sample_width, channels, sample_rate, 16000, state)
        sample_rate = 16000

    url = f"{WS_URL}?session_id={SESSION_ID}&role={role}&token={TOKEN}"
    print(f"Connecting to WebSocket: {url}", flush=True)
    
    try:
        async with websockets.connect(url) as ws:
            print("Connected successfully. Streaming live audio in real-time...", flush=True)
            
            # Send in 50ms chunks (1600 bytes at 16kHz mono 16-bit PCM)
            chunk_size = 1600
            total_bytes = len(raw_data)
            
            frames_sent = 0
            offset = 0
            while offset < total_bytes:
                data = raw_data[offset:offset+chunk_size]
                offset += chunk_size
                
                await ws.send(data)
                frames_sent += 1
                
                # Sleep 40ms to match 1.25x real-time recording speed
                await asyncio.sleep(0.04)
                
                if frames_sent % 20 == 0:
                    print(f"Sent {frames_sent} chunks...", flush=True)

            print(f"Finished streaming {frames_sent} audio chunks. Closing connection.", flush=True)
            
    except Exception as e:
        print(f"WebSocket error: {e}", file=sys.stderr, flush=True)

    # Retrieve transcripts (polling to accommodate slow CPU diarization)
    print("\nRetrieving transcripts from backend...", flush=True)
    max_attempts = 30
    for attempt in range(1, max_attempts + 1):
        try:
            url = f"http://127.0.0.1:8005/v1/transcripts/{SESSION_ID}"
            with urllib.request.urlopen(url) as response:
                transcripts = json.loads(response.read().decode())
                if transcripts:
                    print(f"Fetched {len(transcripts)} transcript events successfully:", flush=True)
                    for event in transcripts:
                        role = event['role'].upper()
                        speaker = event.get('speaker_id', 'UNKNOWN')
                        print(f" - [{role} / {speaker}] [final: {event['is_final']}]: '{event['transcript']}'", flush=True)
                    
                    # Fetch clinical state
                    print("\nRetrieving clinical state from Clinical State Engine...", flush=True)
                    try:
                        state_url = f"http://127.0.0.1:8005/v1/clinical-state/{SESSION_ID}"
                        with urllib.request.urlopen(state_url) as res_state:
                            state = json.loads(res_state.read().decode())
                            print("\n================ CLINICAL STATE ================ ", flush=True)
                            print(json.dumps(state, indent=4), flush=True)
                            print("================================================ ", flush=True)
                    except Exception as e:
                        print(f"Error fetching clinical state: {e}", file=sys.stderr, flush=True)
                        
                    # Fetch clinical reasoning
                    print("\nRetrieving clinical reasoning from Clinical Reasoning Engine...", flush=True)
                    try:
                        reasoning_url = f"http://127.0.0.1:8005/v1/clinical-reasoning/{SESSION_ID}"
                        with urllib.request.urlopen(reasoning_url) as res_reasoning:
                            res = json.loads(res_reasoning.read().decode())
                            if "message" in res:
                                print(res["message"], flush=True)
                            else:
                                print("\n================ CLINICAL REASONING ================ ", flush=True)
                                diags = res.get("candidate_diagnoses", [])
                                if diags:
                                    print("Suspected Candidate Diagnoses:", flush=True)
                                    for diag in diags:
                                        print(f" - {diag.get('name')} (Score: {diag.get('confidence', 0.0):.2f})", flush=True)
                                else:
                                    print("Suspected Candidate Diagnoses: None detected.", flush=True)
                                    
                                # LLM Differential Diagnosis Explanations
                                diff_exps = res.get("llm_differential_explanations", {})
                                if diff_exps:
                                    print("\nLLM Differential Diagnosis Explanations:", flush=True)
                                    for disease, explanation in diff_exps.items():
                                        print(f" * {disease}: {explanation}", flush=True)
                                    
                                red_flags = res.get("red_flag_alerts", [])
                                if red_flags:
                                    print("\n[!] WARNING - Red Flag Alerts:", flush=True)
                                    for rf in red_flags:
                                        print(f"  * {rf.get('red_flag_id')} [{rf.get('severity').upper()}]: {rf.get('action')}", flush=True)
                                        
                                contraindications = res.get("contraindications", [])
                                interactions = res.get("drug_interactions", [])
                                if contraindications or interactions:
                                    print("\n[!] Safety Concerns:", flush=True)
                                    for c in contraindications:
                                        print(f"  * Contraindication ({c.get('severity')}): Treatment '{c.get('drug_or_procedure')}' - {c.get('reason')} (Recommendation: {c.get('recommendation')})", flush=True)
                                    for i in interactions:
                                        print(f"  * Drug-Drug Interaction ({i.get('severity')}): {i.get('drug_a')} + {i.get('drug_b')} - {i.get('clinical_effect')} (Recommendation: {i.get('recommendation')})", flush=True)
                                        
                                invs = res.get("investigation_recommendations", [])
                                if invs:
                                    print("\nRecommended Investigations:", flush=True)
                                    for inv in invs:
                                        print(f" - {inv.get('name')} (Urgency: {inv.get('priority').upper()})", flush=True)
                                        
                                # Rule-Based Missing Information Questions
                                missing_info = res.get("missing_information", [])
                                if missing_info:
                                    print("\nRule-Based Missing Information Questions:", flush=True)
                                    for item in missing_info:
                                        print(f" ? {item.get('question')} (Relevant Conditions: {', '.join(item.get('relevant_conditions', []))})", flush=True)

                                # LLM-Generated Adaptive Questions
                                llm_questions = res.get("llm_adaptive_questions", [])
                                if llm_questions:
                                    print("\nLLM-Generated Adaptive Questions:", flush=True)
                                    for q in llm_questions:
                                        print(f" ? {q.get('question')} (Purpose: {q.get('purpose')} | Diagnosis: {q.get('related_diagnosis')} | Importance: {q.get('clinical_importance')})", flush=True)
                                print("==================================================== ", flush=True)
                    except Exception as e:
                        print(f"Error fetching clinical reasoning: {e}", file=sys.stderr, flush=True)
                    return
        except Exception as e:
            pass
        
        await asyncio.sleep(1.0)
        print(f"Waiting for transcripts to settle (attempt {attempt}/{max_attempts})...", flush=True)
        
    print("Timeout waiting for transcripts from backend. Diarization might still be processing on the server.", file=sys.stderr, flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stream_audio.py <path_to_wav_file> [role: doctor|patient|attender]", flush=True)
        sys.exit(1)
        
    wav_path = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else "doctor"
    
    if role not in ["doctor", "patient", "attender"]:
        print(f"Error: Invalid role '{role}'. Must be doctor, patient, or attender.", flush=True)
        sys.exit(1)
        
    asyncio.run(stream_audio(wav_path, role))
