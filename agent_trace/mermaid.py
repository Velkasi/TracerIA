# ── mermaid.py — Sequence diagram renderer ───────────────────────────────────

from __future__ import annotations

from agent_trace.schema import TraceEvent

_LABELS: dict[str, str] = {
    "planner":    "Planner",
    "architect":  "Architect",
    "dev":        "Developer",
    "test":       "Tester",
    "debug":      "Debugger",
    "reviewer":   "Reviewer",
    "writeup":    "Writeup",
    "analyst":    "Analyst",
    "supervisor": "Supervisor",
    "user":       "User",
    "llm":        "LLM",
    "memory":     "Memory",
    "filesystem": "Filesystem",
}

_FS_TOOLS  = {"write_file", "read_file", "git_commit", "git_diff", "run_shell"}
_MEM_TOOLS = {"remember", "recall", "commit_to_identity"}


def events_to_mermaid(events: list[TraceEvent]) -> str:
    lines: list[str] = ["sequenceDiagram", "    autonumber", ""]
    for p in _participants(events):
        lines.append(f"    participant {p} as {_LABELS.get(p, p)}")
    lines.append("")
    for ev in events:
        lines.extend(_render(ev))
    return "\n".join(lines)


def events_to_mermaid_block(events: list[TraceEvent]) -> str:
    return "```mermaid\n" + events_to_mermaid(events) + "\n```"


# ── Sanitisation ──────────────────────────────────────────────────────────────

def _safe(text: str, max_len: int = 60) -> str:
    text = " ".join(text.split())
    for bad, good in [('"', "'"), (":", ";"), ("#", ""), (";", ","), ("%", "pct"), ("|", "I")]:
        text = text.replace(bad, good)
    text = "".join(ch for ch in text if ch >= " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def _note(text: str) -> str:
    return _safe(text, max_len=80)


# ── Participants ──────────────────────────────────────────────────────────────

def _participants(events: list[TraceEvent]) -> list[str]:
    order: list[str] = ["user"]
    seen: set[str] = {"user"}
    for ev in events:
        if ev.agent not in seen:
            order.append(ev.agent); seen.add(ev.agent)
        if ev.kind == "tool_call":
            tool = ev.payload.get("tool", "")
            if tool in _FS_TOOLS and "filesystem" not in seen:
                order.append("filesystem"); seen.add("filesystem")
            if tool in _MEM_TOOLS and "memory" not in seen:
                order.append("memory"); seen.add("memory")
        if ev.kind in ("llm_call", "llm_response") and "llm" not in seen:
            idx = order.index(ev.agent) + 1 if ev.agent in order else len(order)
            order.insert(idx, "llm"); seen.add("llm")
    return order


def _target(tool: str) -> str:
    return "memory" if tool in _MEM_TOOLS else "filesystem"


# ── Event rendering ───────────────────────────────────────────────────────────

def _render(ev: TraceEvent) -> list[str]:
    p = ev.payload
    out: list[str] = []

    if ev.kind == "agent_start":
        out.append(f"    Note over {ev.agent}: start")

    elif ev.kind == "agent_done":
        out.append(f"    Note over {ev.agent}: done {p.get('duration_ms', 0)}ms")

    elif ev.kind == "supervisor_route":
        tgt = p.get("target", "?")
        if tgt == "END":
            out.append(f"    Note over supervisor: END")
        else:
            out.append(f"    supervisor->>{tgt}: route to {_safe(tgt)}")

    elif ev.kind == "llm_call":
        model = _safe(p.get("model", "?"), max_len=40)
        it    = p.get("iteration", 1)
        inp   = p.get("prompt_tokens", 0)
        comp  = p.get("completion_tokens", 0)
        tot   = p.get("total_tokens", 0)
        out.append(f"    {ev.agent}->>llm: {model} iter={it}")
        if tot > 0:
            out.append(f"    Note right of llm: in={inp} out={comp} tot={tot}")

    elif ev.kind == "llm_response":
        suffix = " +tools" if p.get("has_tool_calls") else ""
        out.append(f"    llm-->>{ev.agent}: response{suffix}")

    elif ev.kind == "tool_call":
        tool    = p.get("tool", "?")
        summary = _safe(p.get("args_summary", ""), max_len=50)
        out.append(f"    {ev.agent}->>{_target(tool)}: {_safe(tool, 30)} {summary}")

    elif ev.kind == "tool_result":
        tool = p.get("tool", "?")
        out.append(f"    {_target(tool)}-->>{ev.agent}: ok {_note(p.get('result', ''))}")

    elif ev.kind == "memory_op":
        op      = p.get("operation", "?")
        layer   = _safe(p.get("layer", "?"), max_len=20)
        summary = _note(p.get("summary", ""))
        if op in ("remember", "commit_to_identity"):
            out.append(f"    {ev.agent}->>memory: {_safe(op)} [{layer}]")
            if summary:
                out.append(f"    Note right of memory: {summary}")
        else:
            out.append(f"    memory-->>{ev.agent}: {_safe(op)} [{layer}] {summary}")

    elif ev.kind == "error":
        out.append(f"    Note over {ev.agent}: ERROR {_note(p.get('message', ''))}")

    return out
