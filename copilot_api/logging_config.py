"""
Structured logging for the Copilot service's own modules.

Every module in `copilot_api/` and `integration/` logs through
`logging.getLogger(__name__)` — the standard, correct pattern. What was
missing is that nothing ever configured the *root* logger those module
loggers propagate to. `app.utils.logging.setup_logging()` builds a JSON
formatter, but attaches it only to a logger literally named
"clinical_copilot" — a different logger object from
"copilot_api.bootstrap", "integration.copilot_runtime", and every other
`__name__`-based logger in this service.

The practical effect, verified directly: with the root logger unconfigured
(Python's default), `logger.info(...)` calls throughout this service are
silently dropped — below the default root level of WARNING — and anything
that does reach WARNING or above falls through to `logging.lastResort`,
which prints one unstructured plain-text line to stderr. Session-lifecycle
logs, intelligence-worker start/stop, provider-selection warnings: none of it
reached anywhere a production log aggregator (CloudWatch Logs, etc.) could
parse.

This attaches the same JSON formatter to the root logger, once, so every
`__name__`-based logger in the process — this service's own, and any library
that logs the standard way — emits the same structured JSON `app.main`'s
"clinical_copilot" logger already does. Idempotent: calling it twice does not
duplicate handlers, the same guard `setup_logging()` uses.
"""

from __future__ import annotations

import logging
import sys

from app.utils.logging import JSONFormatter


def configure_structured_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    if any(isinstance(h.formatter, JSONFormatter) for h in root.handlers):
        return  # Already configured — idempotent, safe to call from anywhere.

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Uvicorn's own loggers have their own default handlers/formatters
    # attached at import time; without clearing them, every access/error log
    # line would be duplicated once in uvicorn's plain-text format and once
    # in ours after propagating to root. Clearing lets them propagate cleanly.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True
