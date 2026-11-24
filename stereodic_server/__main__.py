"""Run the StereoDIC web server.

    uv run python -m stereodic_server                       # http://127.0.0.1:8000
    uv run python -m stereodic_server --port 8080 --reload  # development

Bind to 127.0.0.1 (default) and reach it through an SSH tunnel from other machines:
``ssh -N -L 8000:127.0.0.1:8000 <linux-host>``. There is no login.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(
        "stereodic_server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=["stereodic_server", "stereodic"] if args.reload else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()
