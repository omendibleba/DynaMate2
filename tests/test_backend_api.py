#!/usr/bin/env python3
"""
Test: FastAPI backend — non-streaming endpoints (Phase 1) + streaming chat (Phase 2)
─────────────────────────────────────────────────────────────────────────────────────
Verifies /api/health, /api/status, /api/threads (no LLM calls — pool
construction, restore_state(), response shapes only) and /api/chat/stream
(real LLM call, matching this repo's no-mocking test convention).

Run:
    pytest tests/test_backend_api.py -v
"""

import json
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


def test_chat_stream_incremental(client):
    """
    Real LLM call (no mocking, matching this repo's existing test convention —
    see tests/test_pipeline.py / tests/test_persistence.py). Verifies the SSE
    stream delivers multiple frames incrementally, ending in a 'final' event
    with a non-empty answer, without the request hanging or erroring.
    """
    events = []
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "thread_id": "test-phase2-pytest",
            "message": "What capabilities have been added to the system so far?",
        },
        timeout=90,
    ) as resp:
        assert resp.status_code == 200
        event_type = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event_type:
                events.append((event_type, json.loads(line.split(":", 1)[1].strip())))
                event_type = None

    assert len(events) >= 2, "expected at least an enhancer trace + a final event"
    assert events[0] == ("trace", {
        "node": "enhancer",
        "content": "What capabilities have been added to the system so far?",
        "is_ai": False,
    })
    final_events = [d for etype, d in events if etype == "final"]
    assert len(final_events) == 1
    assert final_events[0]["answer"].strip()
    assert not any(etype == "error" for etype, _ in events)
