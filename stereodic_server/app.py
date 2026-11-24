"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .jobs.manager import EventHub, JobManager
from .jobs.store import JobStore
from .routes import experiments, frames, jobs, system
from .settings import Settings
from .workspace import Workspace, WorkspaceError


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.jobs.start()
        try:
            yield
        finally:
            await app.state.jobs.stop()
            app.state.store.close()

    app = FastAPI(title="StereoDIC", version="0.1", lifespan=lifespan)
    app.state.settings = settings
    app.state.workspace = Workspace(settings)
    app.state.store = JobStore(settings.db_path)
    app.state.hub = EventHub()
    app.state.jobs = JobManager(settings, app.state.store, app.state.hub)

    @app.exception_handler(WorkspaceError)
    async def _workspace_error(request: Request, exc: WorkspaceError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    app.include_router(system.router)
    app.include_router(experiments.router)
    app.include_router(jobs.router)
    app.include_router(frames.router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    if settings.static_dir is not None:
        static = Path(settings.static_dir).resolve()

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(404)
            candidate = (static / full_path).resolve()
            if full_path and static in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static / "index.html")

    return app
