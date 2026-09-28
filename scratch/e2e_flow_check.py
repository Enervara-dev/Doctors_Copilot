"""End-to-end flow check: transcript bus -> clinical state -> reasoning -> serialization."""
import json, time, uuid
from app.services.providers.models import TranscriptEvent
from app.services.transcript_bus import transcript_bus
from app.services.clinical import clinical_state_engine
from clinical_intelligence.integration import get_reasoning_result
from clinical_intelligence.presentation.serializer import serialize_reasoning_result

SESSION = "e2e-check-001"

TURNS = [
    ("doctor",  "Good morning, what brings you in today?"),
    ("patient", "My name is Rahul Sharma. I am 58 years old."),
    ("patient", "I have been having severe chest pain since this morning, it radiates to my left arm."),
    ("patient", "I also feel short of breath and I am sweating a lot."),
    ("doctor",  "Do you have any fever?"),
    ("patient", "No, I don't have a fever."),
    ("patient", "I was diagnosed with high blood pressure three years ago, and I take amlodipine daily."),
    ("patient", "My father had a heart attack when he was sixty."),
    ("patient", "I smoke about ten cigarettes a day."),
    ("doctor",  "Your blood pressure is 160 over 95, pulse is 104, temperature 98.6."),
    ("doctor",  "I am allergic note: patient is allergic to penicillin."),
]

import asyncio

async def drive():
    t0 = time.time()
    for i, (role, text) in enumerate(TURNS, start=1):
        ev = TranscriptEvent(
            session_id=SESSION, sequence_number=i, role=role, speaker_id=role,
            transcript=text, text=text, is_final=True, is_partial=False,
            event_id=str(uuid.uuid4()), timestamp=t0 + i,
            start_time=float(i * 2), end_time=float(i * 2 + 2),
            provider="manual", confidence=0.95,
        )
        transcript_bus.publish(ev)
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.0)

asyncio.run(drive())

state = clinical_state_engine.get_state(SESSION)
print("=" * 70)
print("CLINICAL STATE")
print("=" * 70)
d = state.model_dump()
for k in ("patient_info", "symptoms", "medical_history", "current_medications",
          "allergies", "family_history", "risk_factors", "vital_signs", "diagnosis_tentative"):
    if k in d:
        print(f"  {k}: {json.dumps(d[k], default=str)[:400]}")

print()
print("=" * 70)
print("REASONING")
print("=" * 70)
t = time.time()
res = get_reasoning_result(SESSION)
elapsed = (time.time() - t) * 1000
if not res:
    print("  !! NO REASONING RESULT")
else:
    print(f"  latency: {elapsed:.1f} ms")
    ser = serialize_reasoning_result(res, show_red_flags=True)
    for k, v in ser.items():
        txt = json.dumps(v, default=str)
        print(f"  {k}: {txt[:300]}")
print()
print("FLOW CHECK COMPLETE")
