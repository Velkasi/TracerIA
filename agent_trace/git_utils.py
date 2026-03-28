# ── git_utils.py — Git repository helpers ────────────────────────────────────

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from agent_trace.schema import Contributor, GitInfo, Range, content_hash

logger = logging.getLogger(__name__)


def get_git_info(repo_dir: str | Path) -> Optional[GitInfo]:
    """Return the current HEAD SHA and branch, or None if not a git repo."""
    try:
        sha = _git(["rev-parse", "HEAD"], repo_dir)
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
        return GitInfo(revision=sha, branch=branch or None)
    except Exception as exc:
        logger.debug("git info unavailable: %s", exc)
    return None


def file_ranges(
    file_path: str | Path,
    workspace_dir: str | Path,
    contributor: Optional[Contributor] = None,
) -> list[Range]:
    """Build a single Range covering the entire file with a content hash.

    One Range per file is sufficient for whole-file attribution. Finer-grained
    ranges (e.g. per-function) can be added in the future via git blame.
    """
    abs_path = Path(file_path)
    if not abs_path.is_absolute():
        abs_path = Path(workspace_dir) / abs_path

    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
        line_count = len(text.splitlines()) or 1
        return [Range(
            start_line=1,
            end_line=line_count,
            content_hash=content_hash(text),
            contributor=contributor,
        )]
    except Exception as exc:
        logger.debug("Cannot build ranges for %s: %s", file_path, exc)
        return []


def to_relative(file_path: str | Path, workspace_dir: str | Path) -> str:
    """Return path relative to workspace_dir with forward slashes."""
    try:
        rel = Path(file_path).resolve().relative_to(Path(workspace_dir).resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(file_path).replace("\\", "/")


# ── Internal ──────────────────────────────────────────────────────────────────

def _git(args: list[str], cwd: str | Path) -> str:
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=False,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8", errors="replace").strip()
