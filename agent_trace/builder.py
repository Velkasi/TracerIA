# ── builder.py — TraceBuilder ─────────────────────────────────────────────────
#
# Collects real-time events emitted by agents and produces a TraceRecord
# at the end of a run.

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from pathlib import Path
from typing import Literal, Optional

from agent_trace.schema import (
    AgentSession,
    Contributor,
    FileRecord,
    Range,
    TraceEvent,
    TraceRecord,
    content_hash,
)
from agent_trace.git_utils import file_ranges, get_git_info, to_relative

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S")


def _iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── TraceBuilder ──────────────────────────────────────────────────────────────

class TraceBuilder:
    """Accumulates events from agent execution and builds a TraceRecord.

    Typical lifecycle per run:
        builder = TraceBuilder(workspace_dir="./workspace", task="Build a REST API")

        # Called by tool_loop / agents:
        builder.agent_start("planner")
        builder.llm_call("planner", "my-model-name", prompt_tokens=80, ...)
        builder.llm_response("planner", has_tool_calls=False)
        builder.agent_done("planner", duration_ms=1200)

        builder.tool_call("dev", "write_file", {"path": "app.py", "content": "..."})
        builder.tool_result("dev", "write_file", "ok")

        # At the end of the run:
        record = builder.build_record()
    """

    def __init__(
        self,
        workspace_dir: str | Path = "./workspace",
        task: str = "",
        session_id: Optional[str] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.task = task
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.events: list[TraceEvent] = []
        self._run_id: str = str(uuid.uuid4())
        self._timestamp: str = _iso()
        self._started_at: str = _now()

        # { file_path: [(agent, model)] } — built as write_file calls arrive
        self._file_authors: dict[str, list[tuple[str, str]]] = {}
        self._current_model: dict[str, str] = {}   # agent → last model used

    # ── Event recorders ───────────────────────────────────────────────────────

    def agent_start(self, agent: str) -> None:
        self._emit("agent_start", agent, {})

    def agent_done(self, agent: str, duration_ms: int) -> None:
        self._emit("agent_done", agent, {"duration_ms": duration_ms})

    def supervisor_route(self, target: str) -> None:
        self._emit("supervisor_route", "supervisor", {"target": target})

    def llm_call(
        self,
        agent: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        iteration: int = 1,
    ) -> None:
        self._current_model[agent] = model
        self._emit("llm_call", agent, {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "iteration": iteration,
        })

    def llm_response(self, agent: str, has_tool_calls: bool = False) -> None:
        self._emit("llm_response", agent, {"has_tool_calls": has_tool_calls})

    def tool_call(self, agent: str, tool_name: str, args: dict) -> None:
        self._emit("tool_call", agent, {
            "tool": tool_name,
            "args_summary": _summarise(tool_name, args),
        })
        if tool_name == "write_file":
            path = args.get("path", "")
            if path:
                model = self._current_model.get(agent, "unknown")
                self._file_authors.setdefault(path, []).append((agent, model))

    def tool_result(self, agent: str, tool_name: str, result: str) -> None:
        short = result[:120].replace("\n", " ") + ("…" if len(result) > 120 else "")
        self._emit("tool_result", agent, {"tool": tool_name, "result": short})

    def memory_op(
        self,
        agent: str,
        operation: Literal["remember", "recall", "commit_to_identity", "inject"],
        layer: str,
        summary: str,
    ) -> None:
        self._emit("memory_op", agent, {
            "operation": operation,
            "layer": layer,
            "summary": summary[:120],
        })

    def error(self, agent: str, message: str) -> None:
        self._emit("error", agent, {"message": message[:200]})

    # ── TraceRecord construction ───────────────────────────────────────────────

    def build_record(self, files_written: Optional[list[str]] = None) -> TraceRecord:
        """Produce a TraceRecord from all accumulated events.

        Args:
            files_written: Additional file paths to include (e.g. from AgentState).
                           Merged with paths already captured via tool_call events.
        """
        all_paths = list(self._file_authors.keys())
        for p in files_written or []:
            if p not in all_paths:
                all_paths.append(p)

        git = get_git_info(self.workspace_dir)

        file_records: list[FileRecord] = []
        for raw_path in all_paths:
            authors = self._file_authors.get(raw_path, [])
            sessions: list[AgentSession] = []
            seen: set[tuple[str, str]] = set()

            for agent, model in authors:
                key = (agent, model)
                if key in seen:
                    continue
                seen.add(key)
                contributor = Contributor(type="ai", model_id=model, agent=agent)
                ranges = file_ranges(raw_path, self.workspace_dir, contributor)
                sessions.append(AgentSession(
                    agent=agent,
                    model=model,
                    ranges=ranges,
                    session_id=self.session_id,
                ))

            if not sessions:
                # File was in files_written but no write_file event was captured
                contributor = Contributor(type="ai")
                ranges = file_ranges(raw_path, self.workspace_dir, contributor)
                sessions.append(AgentSession(agent="unknown", model="unknown", ranges=ranges))

            rel = to_relative(raw_path, self.workspace_dir)
            file_records.append(FileRecord(path=rel, sessions=sessions))

        return TraceRecord(
            id=self._run_id,
            timestamp=self._timestamp,
            task=self.task,
            files=file_records,
            git=git,
            metadata={
                "version": _VERSION,
                "total_events": len(self.events),
                "token_summary": _token_summary(self.events),
            },
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, kind: str, agent: str, payload: dict) -> None:
        self.events.append(TraceEvent(ts=_now(), kind=kind, agent=agent, payload=payload))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarise(tool_name: str, args: dict) -> str:
    if tool_name == "write_file":
        content = args.get("content", "")
        lines = content.count("\n") + 1 if content else 0
        return f"path={args.get('path','?')}  ({lines} lines)"
    if tool_name == "read_file":
        return f"path={args.get('path','?')}"
    if tool_name == "run_shell":
        cmd = str(args.get("command", "?"))
        return (cmd[:80] + "…") if len(cmd) > 80 else cmd
    if tool_name in ("remember", "recall", "commit_to_identity"):
        v = str(args.get("content", args.get("query", "?")))
        return (v[:80] + "…") if len(v) > 80 else v
    if tool_name == "git_commit":
        return f"message={args.get('message','?')[:60]}"
    parts = [f"{k}={str(v)[:40]}" for k, v in list(args.items())[:3]]
    return "  ".join(parts) or "(no args)"


def _token_summary(events: list[TraceEvent]) -> dict:
    out: dict[str, dict] = {}
    for ev in events:
        if ev.kind == "llm_call":
            a = ev.agent
            if a not in out:
                out[a] = {"model": ev.model, "calls": 0, "prompt": 0, "completion": 0, "total": 0}
            out[a]["calls"]      += 1
            out[a]["prompt"]     += ev.prompt_tokens
            out[a]["completion"] += ev.completion_tokens
            out[a]["total"]      += ev.total_tokens
    return out
