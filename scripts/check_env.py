"""Print the runtime environment and available compute backends.

    uv run python scripts/check_env.py
"""

from __future__ import annotations

import json

import numpy as np

from stereodic.core.config import load_local_config
from stereodic.core.provenance import git_info, run_environment
from stereodic.dic.backends import available_backends, cuda_info, numba_info, resolve_backend


def main() -> None:
    report = {
        "environment": run_environment(),
        "git": git_info(),
        "numpy": np.__version__,
        "numba": numba_info(),
        "cuda": cuda_info(),
        "available_backends": available_backends(),
        "auto_backend": resolve_backend("auto"),
    }
    try:
        local = load_local_config()
        report["local_config"] = local.model_dump(mode="json")
    except Exception as exc:  # report, don't crash
        report["local_config"] = f"error: {exc}"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
