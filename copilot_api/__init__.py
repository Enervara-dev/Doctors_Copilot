"""
The transport boundary around the Copilot runtime.

Everything here is HTTP: authentication, serialization, status codes, streaming.
No clinical decision is made in this package, and no domain rule is restated —
`CopilotRuntime` remains the only source of behaviour, and it still knows
nothing about HTTP.

The separation is worth keeping deliberately: the runtime was proven end to end
before this existed, and every domain test still runs without importing a web
framework.
"""
