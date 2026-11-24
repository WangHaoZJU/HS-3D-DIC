"""SQLite persistence for jobs (single table)."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATES = ("queued", "running", "cancelling", "succeeded", "failed", "cancelled")
ACTIVE = ("queued", "running", "cancelling")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    stage TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    force INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT '',
    done INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_utc TEXT NOT NULL,
    started_utc TEXT,
    finished_utc TEXT
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created_utc);
"""

_COLUMNS = ("experiment", "stage", "params", "force", "state", "phase", "done", "total", "message", "error",
            "started_utc", "finished_utc")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        d["force"] = bool(d["force"])
        d["fraction"] = (d["done"] / d["total"]) if d["total"] else None
        return d

    def create(self, experiment: str, stage: str, params: dict[str, Any], force: bool) -> dict[str, Any]:
        job_id = "j_" + uuid.uuid4().hex[:10]
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, experiment, stage, params, force, state, created_utc) VALUES (?,?,?,?,?,?,?)",
                (job_id, experiment, stage, json.dumps(params), int(force), "queued", now_utc()),
            )
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def list(self, limit: int = 100, experiment: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs", []
        where = []
        if experiment:
            where.append("experiment = ?")
            args.append(experiment)
        if active_only:
            where.append(f"state IN ({','.join('?' * len(ACTIVE))})")
            args.extend(ACTIVE)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_utc DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [self._row(r) for r in self._db.execute(sql, args).fetchall()]  # type: ignore[misc]

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        bad = set(fields) - set(_COLUMNS)
        if bad:
            raise ValueError(f"unknown job fields: {bad}")
        if "params" in fields:
            fields["params"] = json.dumps(fields["params"])
        if "force" in fields:
            fields["force"] = int(fields["force"])
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            with self._lock:
                self._db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id))
        return self.get(job_id)

    def queued(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM jobs WHERE state = 'queued' ORDER BY created_utc").fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def fail_interrupted(self) -> int:
        """Jobs left running by a previous server process are marked failed."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET state = 'failed', error = 'server restarted while the job was running', "
                "finished_utc = ? WHERE state IN ('running', 'cancelling')",
                (now_utc(),),
            )
            return cur.rowcount
