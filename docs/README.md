# Documentation

Documentation of what this codebase **does today**, verified against the source.

Nothing here describes intended, planned, or future behaviour. Where behaviour could not
be established from the code, it is written as
`Unknown / Not determinable from the current code` rather than guessed.

## How to read this

Every claim carries a file reference (`path.py:line`) so it can be checked. Every feature
carries a status label:

| Label | Meaning |
|---|---|
| **Implemented** | Real, working code on the live execution path |
| **Partially implemented** | Works, but with a stated limitation or a path that does not execute |
| **Mocked** | Deliberately returns canned/synthetic data instead of doing the real work |
| **Placeholder** | Present in the UI or API surface but wired to nothing |
| **Not implemented** | Referenced or configured, but no code performs it |
| **Unknown** | Cannot be determined from the code alone |

## Contents

| Document | Covers |
|---|---|
| [architecture.md](architecture.md) | Components, processes, execution paths, what runs and what doesn't |
| [status.md](status.md) | **Implementation-status matrix for every subsystem** — start here |
| [configuration.md](configuration.md) | Environment variables, feature flags, which ones actually take effect |
| [flows/consultation-flow.md](flows/consultation-flow.md) | End-to-end: microphone → transcript → clinical state → reasoning |
| [flows/speaker-identification.md](flows/speaker-identification.md) | How "who said what" is decided |
| [flows/frontend-flow.md](flows/frontend-flow.md) | Screens, states, user actions, validation |
| [api/README.md](api/README.md) | Every endpoint, request/response shapes, the WebSocket contract |
| [data/models.md](data/models.md) | Schemas and types |
| [data/storage.md](data/storage.md) | Storage reality and session lifecycle |
| [auth/README.md](auth/README.md) | Authentication and authorization as implemented |
| [integrations/README.md](integrations/README.md) | External services |

## Three things to know before reading anything else

1. **There is no database.** All state lives in Python dictionaries in the API process and
   is lost on restart. The only file the application writes is a diarization debug dump.
   See [data/storage.md](data/storage.md).

2. **The dormant packages are gone.** An earlier audit found nine `interview/` modules,
   five `workflow/` modules, most of `evidence/`, plus `events/`, `session/`, `storage/`
   and `timeline/` imported but never executed. They were deleted in the dead-code
   cleanup — see [status.md](status.md#dormant-code) for what was removed and the two
   files that were kept after closer inspection.

3. **The LLM layer usually serves templated fallback text, not model output.** When the
   provider fails, `MockLLMProvider` returns canned strings that are presented in the same
   fields as real model output, with no marker distinguishing them.
   See [integrations/README.md](integrations/README.md#google-gemini).

## How the facts here were established

- **Route and endpoint inventory** — read from `app/main.py` and `app/api/websocket.py`.
- **Which code actually runs** — `sys.settrace` over a real reasoning call and a real
  ingestion sequence, recording every module in which a function was entered. Import alone
  does not count as execution; several packages import cleanly and never run.
- **Response shapes** — taken from the Pydantic models the endpoints return, and from live
  responses captured from a running server.
- **External service behaviour** — observed against the live services where credentials
  allowed, otherwise read from the client code and marked accordingly.

## Scope note

The `frontend/` directory is documented as it currently exists. This documentation task
made no changes to it.
