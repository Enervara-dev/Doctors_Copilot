# Enervera Copilot — consultation console

A dependency-free HTML/CSS/JS console for the clinical copilot. It streams the
microphone into the ingestion WebSocket and presents the live conversation, the
clinical state as it accumulates, and the reasoning engine's assessment.

Everything on screen comes from the backend. Nothing is simulated.

## Run

Start the backend from the repository root:

```powershell
# Production diarization + speaker verification (models load from the HF cache):
$env:CLINICAL_COPILOT_DIARIZATION_MODE = "production"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Serve this folder over HTTP — `getUserMedia` needs a secure context, so opening
`index.html` from the filesystem will not give you a microphone:

```powershell
python -m http.server 5500 --directory frontend
```

Open <http://127.0.0.1:5500>.

## Using it

Press **Start consultation**. Audio is captured, resampled to 16 kHz mono,
converted to 16-bit little-endian PCM and sent as 20 ms (640-byte) binary frames —
the contract `app/services/audio_buffer.py` validates. The level meter is driven by
the real RMS of the microphone signal, so it shows what the backend is actually
receiving.

The conversation, patient header, and vitals refresh every two seconds. Press
**Analyse** for the assessment, differential, questions to ask next, investigations,
safety checks and the consultation note. Settings (backend URL, session, role,
token, auto-analyse) live in the drawer behind the gear icon.

Each participant opens the page on their own device with the same session ID and
their own role.

## Deep links

A consultation is addressable, which is handy for demos and for reopening a session:

```
index.html?session=<id>&api=http://127.0.0.1:8000&role=doctor&theme=light
```

`session`, `api`, `role`, `token` and `theme` (`light` / `dark`) are all supported.
Without `theme`, the console follows the OS setting; the moon icon toggles and remembers.

## Design notes

- **Speaker colours** are the validated categorical slots — blue (doctor), aqua
  (patient), orange (attender) — which clear CVD and normal-vision separation in both
  light and dark mode. Colour never carries identity alone: every turn is labelled.
- **Vitals** use the reserved status ramp and always pair colour with an icon and a
  word ("Elevated", "Tachycardic"), never colour alone.
- **Differential confidence** is a single-hue sequential encoding: one blue fill on a
  lighter step of the same ramp, 4px rounded at the data end and square at the baseline.
- There is exactly **one hero figure** per view — the leading diagnosis's confidence.

## Testing

`node scratch/test_frontend.js <session-id> [port]` renders the page in jsdom against a
running backend and asserts the DOM matches the API payloads.
