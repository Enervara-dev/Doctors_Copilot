# Authentication and authorization

Source: `app/core/security.py`, `app/utils/jwt_utils.py`, `app/api/websocket.py`.

## Summary

| Surface | Authenticated | Authorized |
|---|---|---|
| `WS /v1/streaming/audio` | Yes | Role allow-list; JWT claims when a JWT is used |
| `GET /health` | **No** | — |
| `GET /v1/transcripts/{id}` | **No** | — |
| `GET /v1/clinical-state/{id}` | **No** | — |
| `GET /v1/clinical-state/{id}/provenance` | **No** | — |
| `GET /v1/clinical-reasoning/{id}` | **No** | — |

**Only the WebSocket authenticates.** Every HTTP endpoint is open: no dependency, header
check, cookie or middleware guards them. Knowing a `session_id` is enough to read that
consultation's full transcript, clinical state and reasoning output. Combined with
`allow_origins=["*"]` (`app/main.py:44`), any web page can read them from a browser.

Status: **Not implemented** (HTTP authentication).

## There are no users

Verified absent: user model, registration, login, logout, password handling, session
cookies, refresh tokens, roles-per-user, tenancy.

"Role" here means **which microphone in the room** — `doctor`, `patient`, `attender` — and
is self-declared as a query parameter. It is not an identity and confers no privileges
beyond how the audio is treated downstream.

Status: **Not implemented** (user accounts, identity).

## WebSocket authentication

`verify_session_token(token, expected_session_id, expected_role)` — `core/security.py:24`.

Two accepted paths, tried in order:

### 1. Static shared token

```python
if token == settings.AUTH_SECRET_TOKEN:
    return True
```

`core/security.py:38`. An **unconditional accept**: no session check, no role check, no
expiry. Anyone holding this string may connect as any role to any session.

Default value: `"production-secure-token-change-me"` (`app/core/config.py`). The deployed
`.env` does not override `CLINICAL_COPILOT_AUTH_SECRET_TOKEN`, so **the default is live**,
and it is the value pre-filled in the frontend settings drawer.

Status: **Implemented** — and it bypasses every check below.

### 2. JWT

`core/security.py:42-58` → `decode_jwt_token` (`utils/jwt_utils.py`).

- Algorithm `HS256`, secret `CLINICAL_COPILOT_JWT_SECRET_KEY`.
- Signature and `exp` enforced by PyJWT.
- Claims checked against the request: `session_id` must match, `role` must match.

`generate_jwt_token(session_id, role, expires_in_sec=3600)` mints one. Its docstring says
*"for testing and validation purposes"*, and **no endpoint issues tokens** — there is no
token endpoint, so JWTs must be minted out of band.

Status: **Implemented** (verification). **Not implemented** (issuance over the API).

## Role authorization

`verify_participant_role` (`core/security.py:62`) checks membership in `ALLOWED_ROLES` =
`["doctor", "patient", "attender"]`. Rejection closes the socket with `1008`.

This is the only authorization decision in the codebase.

## Where the role matters downstream

The declared role is not merely cosmetic:

- One STT worker and one audio buffer per `(session_id, role)`.
- Reconnecting with the same role **replaces** the stream and inherits its buffer
  (`session.py:49-67`) — so a second client claiming an in-use role takes it over.
- The role is stamped on every `TranscriptEvent` as `role`.

Note that clinical extraction is gated on `speaker_id` (from diarization), **not** on this
declared `role` — see [../flows/speaker-identification.md](../flows/speaker-identification.md).

## Failure handling

| Condition | Result |
|---|---|
| Missing token | Close `1008`, "Invalid authentication token" |
| Unknown token, not a valid JWT | Close `1008` |
| JWT expired | Logged as `jwt_error/expired`, close `1008` |
| JWT session mismatch | Logged with expected vs received, close `1008` |
| JWT role mismatch | Logged, close `1008` |
| Role not in allow-list | Close `1008`, "Invalid participant role" |

Failures are logged with structured `extra` fields (`event: auth_failure`) to stdout only.

## Token transport

The token travels as a **URL query parameter** on the WebSocket handshake. It will appear
in browser history, proxy logs and server access logs. The frontend places it there
verbatim (`frontend/app.js` `startRecording`).

There is no TLS enforcement in the application; `ws://` is used by default in development.

## Security gaps

| Gap | Status |
|---|---|
| HTTP endpoints unauthenticated | **Not implemented** |
| Static token bypasses all session/role checks | **Implemented as written** — by design in the code, weak by construction |
| Default secrets still in use (`production-secure-token-change-me`, `production-jwt-secret-key-change-me`) | Live defaults |
| No token issuance endpoint | **Not implemented** |
| No rate limiting or brute-force protection | **Not implemented** |
| No audit persistence — auth failures logged to stdout only | **Partially implemented** |
| CORS `*` with `allow_credentials=True` | **Implemented as written** |
| `.env` committed to git history (`27a2845`, on `origin/main`) containing a live ElevenLabs key | Known issue |

## Not determinable from the code

- Whether a reverse proxy, API gateway or network policy provides authentication in a
  deployed environment. No such configuration exists in this repository.
  **Unknown / not determinable from the current code.**
