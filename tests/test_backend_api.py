#!/usr/bin/env python3
"""
Test: FastAPI backend — non-streaming endpoints (Phase 1)
────────────────────────────────────────────────────────
Verifies /api/health, /api/status, and /api/threads against the real
ui_state/ (same state the Gradio UI reads/writes). No chat requests are
made here, so no LLM calls happen — this only exercises pool construction,
restore_state(), and response shapes.

Run:
    pytest tests/test_backend_api.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dotenv
dotenv.load_dotenv()

import pytest
from starlette.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_status_shape(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "agents" in body and "registry" in body
    agent_names = {a["name"] for a in body["agents"]}
    # shell_agent/compute_agent are always added as static system agents
    assert {"shell_agent", "compute_agent"}.issubset(agent_names)
    for agent in body["agents"]:
        assert isinstance(agent["base_tools"], list)
        assert isinstance(agent["extra_tools"], list)
    assert isinstance(body["registry"], list)


def test_create_and_list_threads(client):
    create_resp = client.post("/api/threads")
    assert create_resp.status_code == 200
    new_id = create_resp.json()["id"]
    assert new_id.startswith("session-")

    # A freshly created thread id is not persisted until a chat message is
    # sent on it (mirrors app.py's _save_thread, only called after a reply),
    # so just verify the listing endpoint returns well-shaped existing entries.
    list_resp = client.get("/api/threads")
    assert list_resp.status_code == 200
    threads = list_resp.json()
    assert isinstance(threads, list)
    for t in threads:
        assert {"id", "preview", "created_at"} <= t.keys()
