# ── schema.py — Tracer data model ────────────────────────────────────────────

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Primitive types ───────────────────────────────────────────────────────────

ContributorType = Literal["human", "ai", "mixed", "unknown"]


# ── Record objects (serialisable via dataclasses.asdict) ─────────────────────

@dataclass
class Contributor:
    """Who produced a range of code."""
    type: ContributorType
    model_id: Optional[str] = None
    """The model identifier as passed by the calling system."""
    agent: Optional[str] = None
    """Name of the agent that triggered this contribution, e.g. 'dev'"""


@dataclass
class GitInfo:
    """Git repository state at the time the trace was recorded."""
    revision: str
    """Full commit SHA (40 chars) or short SHA if unavailable"""
    branch: Optional[str] = None


@dataclass
class Range:
    """A contiguous block of lines attributed to one contributor."""
    start_line: int
    """1-indexed, inclusive"""
    end_line: int
    """1-indexed, inclusive"""
    content_hash: Optional[str] = None
    """sha256:<16-char hex> — stable identifier even if lines move"""
    contributor: Optional[Contributor] = None
    """Per-range override (e.g. a patch applied by a different agent)"""


@dataclass
class AgentSession:
    """One agent's contribution to a file — groups its line ranges."""
    agent: str
    model: str
    ranges: list[Range] = field(default_factory=list)
    session_id: Optional[str] = None
    """Opaque identifier linking back to the run that produced this file"""


@dataclass
class FileRecord:
    """Attribution record for one file."""
    path: str
    """Relative path from workspace root, forward slashes"""
    sessions: list[AgentSession] = field(default_factory=list)


@dataclass
class TraceRecord:
    """Top-level trace document produced at the end of one agent run."""
    id: str
    """UUID v4 — unique per run"""
    timestamp: str
    """ISO 8601 UTC, e.g. '2026-03-28T14:30:00Z'"""
    task: str
    """The original user prompt that triggered this run"""
    files: list[FileRecord]
    git: Optional[GitInfo] = None
    metadata: Optional[dict] = None
    """Arbitrary key/value pairs — token summaries, step counts, etc."""


# ── Event types (internal, not persisted in TraceRecord) ─────────────────────

EventKind = Literal[
    "agent_start",
    "llm_call",
    "llm_response",
    "tool_call",
    "tool_result",
    "memory_op",
    "supervisor_route",
    "agent_done",
    "error",
]


@dataclass
class TraceEvent:
    """Lightweight event emitted in real time as agents execute."""
    ts: str          # HH:MM:SS UTC
    kind: EventKind
    agent: str
    payload: dict

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self.payload.get("model", "")

    @property
    def prompt_tokens(self) -> int:
        return self.payload.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.payload.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.payload.get("total_tokens", 0)


# ── Utility ───────────────────────────────────────────────────────────────────

def content_hash(text: str) -> str:
    """Return a stable sha256 prefix hash for a text block."""
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"sha256:{digest}"
