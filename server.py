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

For development (hot-reload on both sides), run the backend and frontend
dev server separately instead — see frontend/README.md.
"""

import os
import webbrowser

import uvicorn

_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")


if __name__ == "__main__":
    if not os.path.isdir(_FRONTEND_DIST):
        raise SystemExit(
            "frontend/dist/ not found. Build the frontend first:\n"
            "  cd frontend && npm install && npm run build"
        )

    webbrowser.open("http://localhost:8888")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8888)
