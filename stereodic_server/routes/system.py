"""System information and a sandboxed file browser over the configured roots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from .. import system

router = APIRouter(prefix="/api", tags=["system"])

IMAGE_EXT = {".tif", ".tiff", ".png", ".bmp"}
MAX_ENTRIES = 5000


@router.get("/system")
def system_info(request: Request, refresh: bool = False) -> dict[str, Any]:
    if refresh:
        system.backends(refresh=True)
    return system.system_info(request.app.state.settings)


def _resolve(request: Request, root: str, path: str) -> tuple[Path, Path]:
    roots = request.app.state.settings.roots()
    if root not in roots:
        raise HTTPException(400, f"unknown root {root!r}; available: {sorted(roots)}")
    base = roots[root].expanduser().resolve()
    target = (base / path.lstrip("/")).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(403, "path is outside the root")
    return base, target


@router.get("/fs")
def list_dir(request: Request, root: str = "data", path: str = "") -> dict[str, Any]:
    base, target = _resolve(request, root, path)
    if not target.exists():
        raise HTTPException(404, "not found")
    if not target.is_dir():
        raise HTTPException(400, "not a directory")
    entries = []
    n_images = 0
    truncated = False
    for p in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if p.name.startswith("."):
            continue
        is_dir = p.is_dir()
        ext = "" if is_dir else p.suffix.lower()
        if ext in IMAGE_EXT:
            n_images += 1
        if len(entries) >= MAX_ENTRIES:
            truncated = True
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": p.name,
                "kind": "dir" if is_dir else "file",
                "size": None if is_dir else st.st_size,
                "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
                "ext": ext,
            }
        )
    rel = "" if target == base else str(target.relative_to(base))
    parent = None if target == base else ("" if target.parent == base else str(target.parent.relative_to(base)))
    return {
        "root": root,
        "root_path": str(base),
        "path": rel,
        "abs_path": str(target),
        "parent": parent,
        "entries": entries,
        "n_images": n_images,
        "truncated": truncated,
    }
