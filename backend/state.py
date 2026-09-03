"""
backend.state
─────────────
Filesystem paths and thread-index helpers. Ported from app.py so the FastAPI
backend and the legacy Gradio UI (gradio-ui-legacy branch) can share the same
ui_state/ directory format.
"""

import json
import os
from datetime import datetime

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Override with DYNAMATE_STATE_DIR to point at a scratch directory instead of
# the real ui_state/ — e.g. for a from-scratch tutorial walkthrough without
# touching live tools/agents/conversation history:
#   DYNAMATE_STATE_DIR=/tmp/dynamate_test_state python server.py
STATE_DIR   = os.getenv("DYNAMATE_STATE_DIR") or os.path.join(ROOT, "ui_state")
UPLOADS_DIR = os.path.join(STATE_DIR, "uploads")
THREADS_DB  = os.path.join(STATE_DIR, "threads.json")
TUTORIALS_DIR = os.path.join(ROOT, "tutorials")
MODEL_NAME  = os.getenv("DYNAMATE_MODEL", "gpt-4o-mini")


def ensure_dirs() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def new_thread_id() -> str:
    return "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def load_threads() -> list[dict]:
    if not os.path.exists(THREADS_DB):
        return []
    try:
        with open(THREADS_DB) as f:
            return json.load(f)
    except Exception:
        return []


def save_thread(thread_id: str, preview: str) -> None:
    data = load_threads()
    if any(t["id"] == thread_id for t in data):
        return
    data.append({
        "id": thread_id,
        "preview": preview,
        "created_at": datetime.now().isoformat(),
    })
    with open(THREADS_DB, "w") as f:
        json.dump(data, f, indent=2)
