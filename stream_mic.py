"""
stream_mic.py Component

Purpose:
Part of the Stream Mic.Py workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Stream Mic.Py
"""

import asyncio
import websockets
import sys
import json
import urllib.request
import time
import uuid

try:
    import pyaudio
except ImportError:
    print("Error: 'pyaudio' library is required to stream from your microphone.")
    print("Please install it by running: .\\.venv\\Scripts\\pip.exe install pyaudio")
    sys.exit(1)

WS_URL = "ws://127.0.0.1:8000/v1/streaming/audio"
TOKEN = "production-secure-token-change-me"
SESSION_ID = ""

# Audio parameters matching backend constraints
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 800  # 800 samples = 50ms of audio

async def stream_mic(role: str):
    p = pyaudio.PyAudio()
    
    # Open mic stream
    try:
        audio_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"Error opening microphone: {e}", file=sys.stderr)
        p.terminate()
        return

    url = f"{WS_URL}?session_id={SESSION_ID}&role={role}&token={TOKEN}"
    print(f"Connecting to WebSocket: {url}")
    print("Recording and streaming. Press CTRL+C to stop...")
    
    try:
        async with websockets.connect(url) as ws:
            print("Connected successfully! Start speaking into your microphone...")
            
            # Start streaming
            while True:
                try:
                    # Read raw PCM data from microphone (non-blocking read)
                    data = audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    await ws.send(data)
                except IOError:
                    # Input overflow, skip frame
                    pass
                # Short yield to allow event loop processing
                await asyncio.sleep(0.001)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nWebSocket error: {e}", file=sys.stderr)
    finally:
        # Clean up audio stream
        try:
            audio_stream.stop_stream()
            audio_stream.close()
        except Exception:
            pass
        p.terminate()

def fetch_transcripts():
    print("\nRetrieving transcripts from backend...")
    max_attempts = 30
    for attempt in range(1, max_attempts + 1):
        try:
            url = f"http://127.0.0.1:8000/v1/transcripts/{SESSION_ID}"
            with urllib.request.urlopen(url) as response:
                transcripts = json.loads(response.read().decode())
                if transcripts:
                    print(f"Fetched {len(transcripts)} transcript events successfully:")
                    for event in transcripts:
                        role = event['role'].upper()
                        speaker = event.get('speaker_id', 'UNKNOWN')
                        print(f" - [{role} / {speaker}] [final: {event['is_final']}]: '{event['transcript']}'")
                    return
        except Exception as e:
            pass
        
        time.sleep(1.0)
        print(f"Waiting for transcripts to settle (attempt {attempt}/{max_attempts})...")
        
    print("Timeout waiting for transcripts from backend. Diarization might still be processing on the server.", file=sys.stderr)

def fetch_clinical_reasoning():
    print("\nRetrieving clinical reasoning from Clinical Reasoning Engine...")
    try:
        url = f"http://127.0.0.1:8000/v1/clinical-reasoning/{SESSION_ID}"
        with urllib.request.urlopen(url) as response:
            res = json.loads(response.read().decode())
            if "message" in res:
                print(res["message"])
                return
            
            print("\n================ CLINICAL REASONING ================ ")
            
            # Suspected Candidate Diagnoses
            diags = res.get("candidate_diagnoses", [])
            if diags:
                print("Suspected Candidate Diagnoses:")
                for diag in diags:
                    score = diag.get("confidence", 0.0)
                    print(f" - {diag.get('name')} (Score: {score:.2f})")
            else:
                print("Suspected Candidate Diagnoses: None detected.")
                
            # LLM Differential Diagnosis Explanations
            diff_exps = res.get("llm_differential_explanations", {})
            if diff_exps:
                print("\nLLM Differential Diagnosis Explanations:")
                for disease, explanation in diff_exps.items():
                    print(f" * {disease}: {explanation}")
                
            # Red Flags
            red_flags = res.get("red_flag_alerts", [])
            if red_flags:
                print("\n[!] WARNING - Red Flag Alerts:")
                for rf in red_flags:
                    print(f"  * {rf.get('red_flag_id')} [{rf.get('severity').upper()}]: {rf.get('action')}")
            
            # Safety Checks (Contraindications & Drug Interactions)
            contraindications = res.get("contraindications", [])
            interactions = res.get("drug_interactions", [])
            if contraindications or interactions:
                print("\n[!] Safety Concerns:")
                for c in contraindications:
                    print(f"  * Contraindication ({c.get('severity')}): Treatment '{c.get('drug_or_procedure')}' - {c.get('reason')} (Recommendation: {c.get('recommendation')})")
                for i in interactions:
                    print(f"  * Drug-Drug Interaction ({i.get('severity')}): {i.get('drug_a')} + {i.get('drug_b')} - {i.get('clinical_effect')} (Recommendation: {i.get('recommendation')})")
            
            # Recommended Investigations
            invs = res.get("investigation_recommendations", [])
            if invs:
                print("\nRecommended Investigations:")
                for inv in invs:
                    print(f" - {inv.get('name')} (Urgency: {inv.get('priority').upper()})")
            
            # Rule-Based Missing Information Questions
            missing_info = res.get("missing_information", [])
            if missing_info:
                print("\nRule-Based Missing Information Questions:")
                for item in missing_info:
                    print(f" ? {item.get('question')} (Relevant Conditions: {', '.join(item.get('relevant_conditions', []))})")

            # LLM-Generated Adaptive Questions
            llm_questions = res.get("llm_adaptive_questions", [])
            if llm_questions:
                print("\nLLM-Generated Adaptive Questions:")
                for q in llm_questions:
                    print(f" ? {q.get('question')} (Purpose: {q.get('purpose')} | Diagnosis: {q.get('related_diagnosis')} | Importance: {q.get('clinical_importance')})")
            
            print("==================================================== ")
    except Exception as e:
        print(f"Error fetching clinical reasoning: {e}", file=sys.stderr)

if __name__ == "__main__":
    role = sys.argv[1] if len(sys.argv) > 1 else "doctor"
    if role not in ["doctor", "patient", "attender"]:
        print(f"Error: Invalid role '{role}'. Must be doctor, patient, or attender.")
        sys.exit(1)
        
        # Generates a random session ID for this live mic run
    SESSION_ID = f"consultation-xyz-{uuid.uuid4().hex[:6]}"
    try:
        asyncio.run(stream_mic(role))
    except KeyboardInterrupt:
        print("\nStopping microphone stream...")
    finally:
        fetch_transcripts()
        
        # Query and display the final clinical state from the engine
        print("\nRetrieving clinical state from Clinical State Engine...")
        try:
            url = f"http://127.0.0.1:8000/v1/clinical-state/{SESSION_ID}"
            with urllib.request.urlopen(url) as response:
                state = json.loads(response.read().decode())
                print("\n================ CLINICAL STATE ================ ")
                print(json.dumps(state, indent=4))
                print("================================================ ")
        except Exception as e:
            print(f"Error fetching clinical state: {e}", file=sys.stderr)
            
        fetch_clinical_reasoning()
