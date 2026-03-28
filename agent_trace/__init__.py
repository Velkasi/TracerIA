# ── agent_trace/__init__.py ───────────────────────────────────────────────────

from __future__ import annotations

from agent_trace.builder import TraceBuilder
from agent_trace.export import save_json, save_markdown, to_dict, to_json
from agent_trace.mermaid import events_to_mermaid, events_to_mermaid_block
from agent_trace.schema import (
    AgentSession,
    Contributor,
    FileRecord,
    GitInfo,
    Range,
    TraceEvent,
    TraceRecord,
    content_hash,
)

__all__ = [
    "get_builder",
    "reset_builder",
    "TraceBuilder",
    "TraceRecord",
    "TraceEvent",
    "FileRecord",
    "AgentSession",
    "Contributor",
    "GitInfo",
    "Range",
    "content_hash",
    "to_dict",
    "to_json",
    "save_json",
    "save_markdown",
    "events_to_mermaid",
    "events_to_mermaid_block",
]

# ── Process-level singleton ───────────────────────────────────────────────────

_active: TraceBuilder | None = None


def get_builder() -> TraceBuilder:
    """Return the current active TraceBuilder, creating one with defaults if needed."""
    global _active
    if _active is None:
        _active = TraceBuilder()
    return _active


def reset_builder(
    workspace_dir: str = "./workspace",
    task: str = "",
    session_id: str | None = None,
) -> TraceBuilder:
    """Replace the active builder with a fresh one. Call at the start of each run."""
    global _active
    _active = TraceBuilder(workspace_dir=workspace_dir, task=task, session_id=session_id)
    return _active
