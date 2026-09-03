#!/usr/bin/env python3
"""
DynaMate2 — Production entry point (React UI)
────────────────────────────────────────────────────────────────────────────
Single-command successor to app.py (the Gradio UI, preserved on the
gradio-ui-legacy branch). Serves the FastAPI backend (backend/) together
with the built React frontend (frontend/dist/) on one port.

QUICK START
───────────
  cd frontend && npm install && npm run build && cd ..
  python server.py          # http://localhost:8888

  DYNAMATE_PORT=8899 python server.py   # use a different port, e.g. if 8888
                                         # is unavailable through an SSH/VS
                                         # Code tunnel on a remote machine

On a remote machine (HPC node, etc.) this only binds the port locally —
you still need to forward it to your own machine (an SSH -L tunnel, or
VS Code's Ports panel) before http://localhost:<port> will load in your
browser. No browser is auto-opened here since one on the remote host
wouldn't be the browser you're looking at.

For development (hot-reload on both sides), run the backend and frontend
dev server separately instead — see frontend/README.md.
"""

import os

import uvicorn

_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
_PORT = int(os.getenv("DYNAMATE_PORT", "8888"))


if __name__ == "__main__":
    if not os.path.isdir(_FRONTEND_DIST):
        raise SystemExit(
            "frontend/dist/ not found. Build the frontend first:\n"
            "  cd frontend && npm install && npm run build"
        )

    print(f"DynaMate2: listening on port {_PORT} (bound to all interfaces).", flush=True)
    print(f"  Open this in your browser: http://localhost:{_PORT}", flush=True)
    print(f"  (NOT http://0.0.0.0:{_PORT} — that's a bind address, not something "
          f"a browser can connect to)", flush=True)
    print("  On a remote/HPC host, forward the port to your own machine first "
          "(SSH -L, or VS Code's Ports panel) before that URL will load.", flush=True)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=_PORT)
