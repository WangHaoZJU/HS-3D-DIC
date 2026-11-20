"""Compute backend detection and selection.

numpy : FP64 reference implementation (correctness baseline)
numba : FP64 multi-core CPU (Mac and Linux)
cuda  : mixed-precision GPU via CuPy (Linux + NVIDIA), must pass consistency tests
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

BACKENDS = ("numpy", "numba", "cuda")


@lru_cache(maxsize=1)
def cuda_info() -> dict[str, Any]:
    """Probe CuPy / GPU availability. Never raises."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError, or CUDA libraries missing
        return {"available": False, "reason": f"cupy import failed: {exc}"}
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            return {"available": False, "reason": "no CUDA device"}
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        return {
            "available": True,
            "device_count": n,
            "device0": name,
            "memory_gb": round(props["totalGlobalMem"] / 1024**3, 1),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cupy": cp.__version__,
        }
    except Exception as exc:
        return {"available": False, "reason": f"CUDA runtime error: {exc}"}


@lru_cache(maxsize=1)
def numba_info() -> dict[str, Any]:
    try:
        import numba

        return {"available": True, "version": numba.__version__, "threads": numba.config.NUMBA_NUM_THREADS}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def available_backends() -> list[str]:
    out = ["numpy"]
    if numba_info()["available"]:
        out.append("numba")
    if cuda_info()["available"]:
        out.append("cuda")
    return out


def resolve_backend(requested: str) -> str:
    """Map 'auto' to the best available backend; error if an explicit one is unavailable."""
    avail = available_backends()
    if requested == "auto":
        for name in ("cuda", "numba", "numpy"):
            if name in avail:
                return name
    if requested not in BACKENDS:
        raise ValueError(f"unknown backend {requested!r}; expected auto or one of {BACKENDS}")
    if requested not in avail:
        reason = cuda_info().get("reason") if requested == "cuda" else numba_info().get("reason")
        raise RuntimeError(f"backend {requested!r} not available ({reason})")
    return requested
