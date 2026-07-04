from __future__ import annotations

"""
runner.py Component

Purpose:
Part of the Clinical Intelligence workspace layout. Handles logic, routines, or configurations in this layer.

Module/Component: Clinical Intelligence
"""

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def execute_llm_tasks_sync(
    task_manager: Any,
    session_id: str,
    state: Dict[str, Any],
    candidates: List[Any],
    red_flags: List[Any],
    risk_factors: List[Any],
    medications: List[str],
    missing_info: List[Any],
    guidelines: List[Any],
    graph_paths: List[Any],
    is_complete: bool = False,
) -> Dict[str, Any]:
    """
    Safely executes the async Task Manager coroutines from synchronous code.
    Isolates event loop creation, retrieval, and re-entry nesting.
    """
    coro = task_manager.run(
        session_id=session_id,
        state=state,
        candidates=candidates,
        red_flags=red_flags,
        risk_factors=risk_factors,
        medications=medications,
        missing_info=missing_info,
        guidelines=guidelines,
        graph_paths=graph_paths,
        is_complete=is_complete,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Inject nest_asyncio to support re-entrant calls in running loops (e.g. FastAPI)
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return loop.run_until_complete(coro)
