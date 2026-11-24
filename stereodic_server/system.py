"""System information: compute backends, GPU, disks, git.

Backend probing imports CuPy, which would create a CUDA context in the API process,
so it runs once in a short-lived subprocess and is cached.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from stereodic.core.provenance import git_info, run_environment

from .settings import Settings

_PROBE = """
import json
from stereodic.dic.backends import available_backends, cuda_info, numba_info, resolve_backend
print(json.dumps({"numba": numba_info(), "cuda": cuda_info(), "available": available_backends(),
                  "auto": resolve_backend("auto")}))
"""

_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def backends(refresh: bool = False) -> dict[str, Any]:
    global _cache
    with _lock:
        if _cache is None or refresh:
            try:
                out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120)
                _cache = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {
                    "error": out.stderr.strip()[-500:]
                }
            except Exception as exc:  # noqa: BLE001
                _cache = {"error": str(exc)}
        return _cache


def gpus() -> list[dict[str, Any]]:
    """GPU utilisation from nvidia-smi (empty without NVIDIA driver)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, name, used, total, util, temp = parts
        rows.append({"index": int(idx), "name": name, "memory_used_mb": float(used), "memory_total_mb": float(total),
                     "utilization_pct": float(util), "temperature_c": float(temp)})
    return rows


def disk(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        u = shutil.disk_usage(p)
    except OSError:
        return None
    return {"path": str(path), "total_gb": u.total / 1e9, "free_gb": u.free / 1e9, "used_gb": u.used / 1e9}


def system_info(settings: Settings) -> dict[str, Any]:
    t0 = time.monotonic()
    info = {
        "environment": run_environment(),
        "git": git_info(),
        "backends": backends(),
        "gpus": gpus(),
        "roots": {k: str(v) for k, v in settings.roots().items()},
        "disks": {k: disk(v) for k, v in settings.roots().items()},
        "static_ui": settings.static_dir is not None,
    }
    info["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return info
