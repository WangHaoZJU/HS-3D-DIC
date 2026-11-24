"""Server settings: roots, workspace, job database, static files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from stereodic.core.config import REPO_ROOT, load_local_config


@dataclass
class Settings:
    data_root: Path | None
    results_root: Path | None
    workspace: Path  # <results_root>/experiments
    state_dir: Path  # job database and logs
    static_dir: Path | None = None
    gpu_concurrency: int = 1
    cancel_timeout_s: float = 60.0
    local_path: Path | None = None  # configs/local.yaml override (tests)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, local_path: str | Path | None = None) -> Settings:
        """Roots from configs/local.yaml (or STEREODIC_* env vars).

        ``STEREODIC_WORKSPACE`` overrides the experiment workspace, ``STEREODIC_SERVER_STATE``
        the job database directory, ``STEREODIC_WEBUI_DIST`` the built frontend.
        """
        local = load_local_config(local_path)
        data_root = Path(local.data_root).expanduser() if local.data_root else None
        results_root = Path(local.results_root).expanduser() if local.results_root else None
        ws_env = os.environ.get("STEREODIC_WORKSPACE")
        if ws_env:
            workspace = Path(ws_env).expanduser()
        elif results_root is not None:
            workspace = results_root / "experiments"
        else:
            raise RuntimeError(
                "results_root is not configured: copy configs/local.yaml.example to configs/local.yaml "
                "or set STEREODIC_RESULTS_ROOT / STEREODIC_WORKSPACE"
            )
        state_env = os.environ.get("STEREODIC_SERVER_STATE")
        state_dir = Path(state_env).expanduser() if state_env else workspace.parent / ".stereodic_server"
        dist_env = os.environ.get("STEREODIC_WEBUI_DIST")
        static = Path(dist_env).expanduser() if dist_env else REPO_ROOT / "webui" / "dist"
        return cls(
            data_root=data_root,
            results_root=results_root,
            workspace=workspace,
            state_dir=state_dir,
            static_dir=static if (static / "index.html").exists() else None,
            local_path=Path(local_path) if local_path else None,
        )

    def ensure_dirs(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.state_dir / "jobs.sqlite"

    def log_path(self, job_id: str) -> Path:
        return self.state_dir / "logs" / f"{job_id}.log"

    def roots(self) -> dict[str, Path]:
        out = {"workspace": self.workspace}
        if self.data_root is not None:
            out["data"] = self.data_root
        if self.results_root is not None:
            out["results"] = self.results_root
        return out
