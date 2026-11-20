"""Run provenance: git commit, working-tree state, environment."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import __version__

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo_dir: str | Path = _REPO_ROOT) -> dict[str, Any]:
    """Return {'commit': str|None, 'dirty': bool|None}. Never raises."""
    repo_dir = Path(repo_dir)
    commit = _git("rev-parse", "HEAD", cwd=repo_dir)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=repo_dir)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def run_environment() -> dict[str, Any]:
    return {
        "stereodic_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }
