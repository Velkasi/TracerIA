# ── export.py — Serialise TraceRecord and events to JSON / Markdown ──────────

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_trace.schema import TraceEvent, TraceRecord
from agent_trace.mermaid import events_to_mermaid_block


# ── JSON ──────────────────────────────────────────────────────────────────────

def to_dict(record: TraceRecord) -> dict:
    return _drop_none(dataclasses.asdict(record))


def to_json(record: TraceRecord, indent: int = 2) -> str:
    return json.dumps(to_dict(record), ensure_ascii=False, indent=indent)


def save_json(record: TraceRecord, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(record), encoding="utf-8")
    return p


# ── Markdown ──────────────────────────────────────────────────────────────────

def save_markdown(
    record: TraceRecord,
    events: list[TraceEvent],
    path: str | Path,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_build_md(record, events), encoding="utf-8")
    return p


def _build_md(record: TraceRecord, events: list[TraceEvent]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    s: list[str] = []

    s.append("# Trace Report\n")
    s.append(f"**Task:** {record.task or '(unknown)'}")
    s.append(f"**Run ID:** `{record.id}`")
    s.append(f"**Timestamp:** `{record.timestamp}`")
    s.append(f"**Generated:** {now}")
    if record.git:
        rev = record.git.revision
        branch = f" · branch `{record.git.branch}`" if record.git.branch else ""
        s.append(f"**Git:** `{rev[:12]}`{branch}")
    s.append(f"**Total events:** {len(events)}\n")

    s.append("## Token Usage\n")
    s.append(_token_table(events))

    s.append("\n## Memory Operations\n")
    s.append(_memory_table(events))

    s.append("\n## File Attribution\n")
    s.append(_attribution_table(record))

    s.append("\n## Sequence Diagram\n")
    s.append(events_to_mermaid_block(events))

    s.append("\n## Event Log\n")
    s.append("| # | Time | Agent | Kind | Detail |")
    s.append("|---|------|-------|------|--------|")
    for i, ev in enumerate(events, 1):
        s.append(f"| {i} | `{ev.ts}` | `{ev.agent}` | `{ev.kind}` | {_detail(ev)} |")

    s.append("\n## TraceRecord JSON\n")
    s.append("```json")
    s.append(to_json(record))
    s.append("```")

    return "\n".join(s)


# ── Table helpers ─────────────────────────────────────────────────────────────

def _token_table(events: list[TraceEvent]) -> str:
    usage: dict[str, dict] = {}
    for ev in events:
        if ev.kind == "llm_call":
            a = ev.agent
            if a not in usage:
                usage[a] = {"model": ev.model, "calls": 0, "prompt": 0, "completion": 0, "total": 0}
            usage[a]["calls"]      += 1
            usage[a]["prompt"]     += ev.prompt_tokens
            usage[a]["completion"] += ev.completion_tokens
            usage[a]["total"]      += ev.total_tokens
    if not usage:
        return "_No LLM calls recorded._"
    rows = [
        "| Agent | Model | Calls | Prompt | Completion | Total |",
        "|-------|-------|-------|--------|------------|-------|",
    ]
    for agent, u in usage.items():
        rows.append(
            f"| `{agent}` | `{u['model']}` | {u['calls']} "
            f"| {u['prompt']:,} | {u['completion']:,} | **{u['total']:,}** |"
        )
    rows.append(
        f"| **TOTAL** | | "
        f"| {sum(u['prompt'] for u in usage.values()):,} "
        f"| {sum(u['completion'] for u in usage.values()):,} "
        f"| **{sum(u['total'] for u in usage.values()):,}** |"
    )
    return "\n".join(rows)


def _memory_table(events: list[TraceEvent]) -> str:
    ops = [ev for ev in events if ev.kind == "memory_op"]
    if not ops:
        return "_No memory operations recorded._"
    rows = [
        "| Agent | Operation | Layer | Summary |",
        "|-------|-----------|-------|---------|",
    ]
    for ev in ops:
        p = ev.payload
        rows.append(
            f"| `{ev.agent}` | `{p.get('operation','?')}` "
            f"| `{p.get('layer','?')}` | {p.get('summary','')} |"
        )
    return "\n".join(rows)


def _attribution_table(record: TraceRecord) -> str:
    if not record.files:
        return "_No files attributed._"
    rows = [
        "| File | Agent | Model | Lines | Hash |",
        "|------|-------|-------|-------|------|",
    ]
    for f in record.files:
        for sess in f.sessions:
            for r in sess.ranges:
                ch = r.content_hash or "—"
                if len(ch) > 24:
                    ch = ch[:24] + "…"
                rows.append(
                    f"| `{f.path}` | `{sess.agent}` | `{sess.model}` "
                    f"| {r.start_line}–{r.end_line} | `{ch}` |"
                )
    return "\n".join(rows)


def _detail(ev: TraceEvent) -> str:
    p = ev.payload
    if ev.kind == "llm_call":
        return (
            f"model=`{p.get('model','?')}` iter={p.get('iteration',1)} "
            f"in={p.get('prompt_tokens',0):,} "
            f"out={p.get('completion_tokens',0):,} "
            f"tot={p.get('total_tokens',0):,}"
        )
    if ev.kind == "llm_response":
        return "+tools" if p.get("has_tool_calls") else "text only"
    if ev.kind == "tool_call":
        return f"`{p.get('tool','?')}` — {p.get('args_summary','')}"
    if ev.kind == "tool_result":
        return f"`{p.get('tool','?')}` → {p.get('result','')}"
    if ev.kind == "memory_op":
        return f"`{p.get('operation','?')}` [{p.get('layer','?')}] {p.get('summary','')}"
    if ev.kind == "supervisor_route":
        return f"→ `{p.get('target','?')}`"
    if ev.kind == "agent_done":
        return f"{p.get('duration_ms',0):,}ms"
    if ev.kind == "error":
        return p.get("message", "")
    return ""


# ── Internal ──────────────────────────────────────────────────────────────────

def _drop_none(obj):
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(i) for i in obj]
    return obj
