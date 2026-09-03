#!/usr/bin/env python3
"""
Test: FastAPI backend — non-streaming endpoints (Phase 1) + streaming chat (Phase 2)
      + quick-start prompts / file upload (Phase 4)
─────────────────────────────────────────────────────────────────────────────────────
Verifies /api/health, /api/status, /api/threads (no LLM calls — pool
construction, restore_state(), response shapes only), /api/chat/stream
(real LLM call, matching this repo's no-mocking test convention),
/api/quickstart/prompts (byte-for-byte against the original app.py
PROMPT_T1A..T4B strings), and /api/tools/upload (round-trip).

Run:
    pytest tests/test_backend_api.py -v
"""

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dotenv
dotenv.load_dotenv()

import pytest
from starlette.testclient import TestClient

from backend import state
from backend.main import app
from backend.quickstart import PROMPTS
from backend.state import UPLOADS_DIR


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


def test_chat_stream_post_loop_failure_surfaces_as_error_event(client, monkeypatch):
    """
    Regression test: state.save_thread() (and the final answer bookkeeping
    around it) used to sit outside the try/except guarding the SSE
    generator, so a failure there silently killed the stream with no
    'error' event ever reaching the client. Simulates that failure and
    asserts it now surfaces as a proper 'error' event instead.
    """
    def _boom(*_args, **_kwargs):
        raise OSError("simulated disk failure writing threads.json")

    monkeypatch.setattr(state, "save_thread", _boom)

    events = []
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "thread_id": "test-phase2-error-path",
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

    error_events = [d for etype, d in events if etype == "error"]
    assert len(error_events) == 1
    assert "simulated disk failure" in error_events[0]["message"]
    assert not any(etype == "final" for etype, _ in events)


def test_quickstart_prompts_match_source(client):
    """
    The API's quickstart prompts must be byte-identical to backend.quickstart's
    PROMPTS dict (which is itself a verbatim port of app.py's PROMPT_T1A..T4B
    construction) — this is a regression guard against the two drifting apart.
    """
    resp = client.get("/api/quickstart/prompts")
    assert resp.status_code == 200
    body = resp.json()
    assert body == PROMPTS
    # sanity: T1a must actually embed the download_mace_model source, T2 must
    # reference the tutorials/ directory for its file paths.
    assert "def download_mace_model" in body["t1a"]
    assert "nacl_water_box.xyz" in body["t2"]


def test_tool_upload_roundtrip(client, tmp_path):
    src = tmp_path / "my_tool.py"
    src.write_text(
        "def my_test_tool(x: float) -> str:\n"
        '    """A test tool for the upload round-trip test."""\n'
        "    return str(x)\n"
    )

    with open(src, "rb") as f:
        resp = client.post(
            "/api/tools/upload",
            files={"file": ("my_tool.py", f, "text/x-python")},
        )
    assert resp.status_code == 200
    body = resp.json()
    expected_path = os.path.join(UPLOADS_DIR, "my_tool.py")
    assert body["path"] == expected_path
    assert expected_path in body["prompt"]
    assert os.path.exists(expected_path)
    with open(expected_path) as f:
        assert "def my_test_tool" in f.read()
    os.remove(expected_path)


def test_tool_upload_rejects_non_python(client):
    resp = client.post(
        "/api/tools/upload",
        files={"file": ("not_a_tool.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_register_two_tools_at_once_no_parallel_tool_call_race(tmp_path, monkeypatch):
    """
    Regression test: asking to register two functions "together" (T1b) can
    make the model emit two parallel tool calls to register_tool_from_code.
    LangGraph's ToolNode runs those via a ThreadPoolExecutor by default
    (see langgraph.prebuilt.ToolNode._func), and dynamate's AgentPool
    mutates plain, non-thread-safe dicts (_tool_registry, _agents) inside
    tool calls — two concurrent registrations can race on the same dict and
    raise "RuntimeError: dictionary changed size during iteration".
    chat.py now passes max_concurrency=1 in the LangGraph config to force
    sequential tool execution. Runs against an isolated scratch state dir
    (not the shared module-scoped `client` fixture's ui_state) so both
    tools are genuinely new registrations, not skipped re-registrations.
    """
    monkeypatch.setattr(state, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(state, "UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(state, "THREADS_DB", str(tmp_path / "threads.json"))

    with TestClient(app) as scratch_client:
        thread_id = scratch_client.post("/api/threads").json()["id"]

        events = []
        with scratch_client.stream(
            "POST",
            "/api/chat/stream",
            json={"thread_id": thread_id, "message": PROMPTS["t1b"]},
            timeout=120,
        ) as resp:
            assert resp.status_code == 200
            event_type = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event_type:
                    events.append((event_type, json.loads(line.split(":", 1)[1].strip())))
                    event_type = None

        error_events = [d for etype, d in events if etype == "error"]
        assert not error_events, f"unexpected error event(s): {error_events}"

        registry = scratch_client.get("/api/status").json()["registry"]
        assert set(registry) == {"smiles_to_xyz", "packmol_build_system"}


def test_enhancer_preserves_original_message_for_multistep_prompts(client):
    """
    Regression test for dynamate/prompt_enhancer.py's Rule 3 (multi-step
    requests). The system prompt used to instruct the LLM to REPLACE the
    whole message with just "First use <agent> with <tool_X>, then
    <tool_Y>." — discarding every concrete detail (file paths, box size,
    molecule counts) from the original request. The downstream specialist
    agent then had nothing to act on and would just bounce back to the
    supervisor in a loop instead of calling a tool (burning tokens on the
    growing conversation history each hop, which is what actually
    triggered the 429 rate-limit error this bug was discovered from).

    Fix: Rule 3 now appends the routing hint after the original message,
    matching how Rule 2 already handled single-tool routing. Asserts the
    enhanced output still contains the original request's file paths.
    """
    enhancer = client.app.state.enhancer
    enhanced = enhancer.enhance(PROMPTS["t2"])

    # Every concrete detail from the original T2 prompt must survive.
    for original_line in [
        "water.xyz",
        "na.xyz",
        "cl.xyz",
        "20.0 Angstrom",
        "267 water molecules",
        "nacl_water_box.xyz",
    ]:
        assert original_line in enhanced, (
            f"enhancer dropped {original_line!r} from the original message "
            f"— got: {enhanced!r}"
        )


def test_assign_tool_after_agent_creation_updates_supervisor(tmp_path):
    """
    Regression test for the root cause behind every "specialist never
    actually calls its tool" failure diagnosed today: AgentPoolWithSupervisor
    overrides add_agent/remove_agent to rebuild the supervisor afterward, but
    did NOT override assign_tool/remove_tool. Since restore_state() (and
    tool_manager's assign_tool_to_agent) always create/have the agent BEFORE
    assigning its tools, the supervisor's compiled graph kept routing to a
    stale, tool-less copy of the agent for the rest of the process's life —
    pool_state.json/the /api/status endpoint correctly showed the tool
    assigned, but the specialist could never actually call it, so it just
    bounced back to the supervisor and the supervisor fabricated a plausible
    success message from nothing.

    Directly mirrors the exact sequence that exposed this (add_agent with
    zero tools, first supervisor build via set_system_agents, THEN
    register+assign a tool) and verifies a message routed through the
    supervisor actually reaches and executes the real tool function.
    """
    from langchain_openai import ChatOpenAI

    from dynamate import AgentPoolWithSupervisor, build_tool_manager_v2

    # A marker file, not a captured Python closure: register_tool_from_code
    # exec()'s the source text in a fresh namespace, so a real closure over
    # a local list would be silently lost — file I/O survives that round-trip
    # the same way the real smiles_to_xyz/packmol_build_system tools do.
    marker = tmp_path / "called.txt"
    tool_source = f'''
def record_call(value: str) -> str:
    """Records that this tool was actually called; echoes the value back."""
    with open("{marker}", "a") as f:
        f.write(value + "\\n")
    return f"recorded: {{value}}"
'''

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    pool = AgentPoolWithSupervisor(
        supervisor_model=model,
        supervisor_prompt=(
            "You are the Supervisor. Any request mentioning 'echo_specialist' "
            "or asking to record/echo a value must be routed to echo_specialist."
        ),
    )
    # Agent created with NO tools yet, and the supervisor is first built
    # (via set_system_agents) around it in that state — mirrors
    # restore_state()'s exact sequence (add_agent, then assign_tool later).
    pool.add_agent(
        "echo_specialist",
        model,
        base_tools=[],
        system_prompt=(
            "You are echo_specialist. Your FIRST action must be to call "
            "record_call with the value from the request. Never respond "
            "with plain text first."
        ),
    )
    tool_manager = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tool_manager])  # first supervisor build; echo_specialist has 0 tools here

    # Tool registered and assigned AFTER the agent already exists in the
    # compiled supervisor graph — this is the exact gap that caused the bug.
    pool.register_tool_from_code(tool_source)
    result = pool.assign_tool("record_call", "echo_specialist")
    assert "Assigned" in result

    config = {"configurable": {"thread_id": "test-supervisor-sync"}}
    for _ in pool.supervisor.stream(
        {"messages": [{"role": "user", "content": "echo_specialist: record_call with value 'hello-regression-test'"}]},
        config=config,
        recursion_limit=15,
    ):
        pass

    assert marker.exists(), (
        "echo_specialist never actually called record_call (stale supervisor "
        "routing bug?) — no marker file was written"
    )
    assert marker.read_text().strip() == "hello-regression-test"


def test_status_endpoint_survives_concurrent_pool_mutation(client, monkeypatch):
    """
    Regression test for GET /api/status racing a concurrent pool mutation
    (e.g. a chat turn inside register_tool_from_code/assign_tool, running in
    a different worker thread).

    The natural race window is only a handful of dict-mutation bytecode ops
    wide — a plain "background thread churns the dict while the foreground
    hammers get_status() in a loop" stress test was tried first and did not
    reproduce it reliably even at 2000 iterations with sys.setswitchinterval
    lowered, because CPython's dict iteration usually completes faster than
    the OS actually preempts the thread. So this deterministically forces
    the exact failure instead of hoping for it: pool._tool_registry is
    swapped for a dict subclass whose keys() pauses after yielding the
    first key, hands off to a mutator thread that performs a real,
    synchronous mutation on that same live dict object, waits for it to
    finish, and only then resumes iterating — which is guaranteed to raise
    "RuntimeError: dictionary changed size during iteration" on continuation
    (verified directly against a plain dict/iterator pair before writing
    this test). get_status() must survive that via its retry loop.
    """
    import threading

    from backend.routes.status import get_status

    class _PausingDict(dict):
        """keys() pauses after the first key so a mutator thread can make a
        real, guaranteed-visible size change before iteration continues."""

        def __init__(self, *args, on_first_key=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._on_first_key = on_first_key

        def keys(self):
            it = dict.__iter__(self)

            def _paced():
                first = next(it)
                yield first
                if self._on_first_key:
                    self._on_first_key()
                yield from it  # continues the SAME live iterator post-mutation

            return _paced()

    pool = client.app.state.pool
    original_registry = pool._tool_registry
    ready_to_mutate = threading.Event()
    mutation_done = threading.Event()

    def on_first_key():
        ready_to_mutate.set()
        assert mutation_done.wait(timeout=5), "mutator thread never completed"

    racy_registry = _PausingDict(original_registry, on_first_key=on_first_key)

    def mutator():
        assert ready_to_mutate.wait(timeout=5), "iteration never reached the checkpoint"
        racy_registry["_race_injected_tool"] = next(iter(original_registry.values()))
        mutation_done.set()

    monkeypatch.setattr(pool, "_tool_registry", racy_registry)
    try:
        mutator_thread = threading.Thread(target=mutator, daemon=True)
        mutator_thread.start()
        result = get_status(pool)  # must not raise — the retry loop must catch it
        mutator_thread.join(timeout=5)
    finally:
        monkeypatch.setattr(pool, "_tool_registry", original_registry)

    assert "_race_injected_tool" in result.registry
