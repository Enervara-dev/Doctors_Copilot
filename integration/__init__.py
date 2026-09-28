"""
Integration boundary between the Doctor platform and this clinical core.

Nothing in here reasons about medicine. It exists to answer one question
precisely: what has to cross the wire, in what shape, for the existing Python
pipeline to behave the same way it does when driven by its own tests.

The boundary is deliberately one-directional and free of transport: no HTTP, no
broker, no persistence. `CopilotEvent` is the normalised envelope; the adapter
turns a Doctor utterance into one and then into the `TranscriptEvent` the
pipeline already consumes.
"""
