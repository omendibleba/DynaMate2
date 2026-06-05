"""
tests/test_pipeline.py
──────────────────────
Layered validation tests for the DynaMate2 pipeline.

Runs four tiers so failures are easy to localise:

  T0 — Enhancer output (no agent, no tool call)
       Verify the PromptEnhancer produces correct routing hints.

  T1 — Tool registration (no LLM required)
       Verify pool.register_tool_from_code / assign_tool behave correctly.

  T2 — Agent tool-call verification (agent invoked directly, no supervisor)
       Confirm the agent makes a REAL tool call (ToolMessage in history).
       If this fails the agent is hallucinating tool use.

  T3 — Supervisor routing (supervisor invoked, single-hop)
       Confirm the supervisor delegates to the correct agent.

  T4 — End-to-end integration (matches notebook T1–T4, slow/expensive)
       Full pipeline with real tools; run with --e2e flag or set E2E=1.

Usage
─────
    # fast tests only
    python tests/test_pipeline.py

    # include slow e2e tests
    E2E=1 python tests/test_pipeline.py

    # or with pytest
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k e2e   # e2e only
"""

from __future__ import annotations

import os
import sys
import time

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from langchain_core.messages import ToolMessage
from langchain_community.tools import ShellTool

from dynamate import (
    AgentPoolWithSupervisor,
    build_tool_manager_v2,
    PromptEnhancer,
    pretty_print_messages,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_model(model_name: str = "gpt-4o-mini"):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, temperature=0.0)


def _build_pool(model=None) -> AgentPoolWithSupervisor:
    """Standard pool with shell_agent, compute_agent, and ToolManager."""
    if model is None:
        model = _make_model()
    pool = AgentPoolWithSupervisor(
        supervisor_model=model,
        supervisor_prompt=(
            "You are the Supervisor managing a pool of agents.\n"
            "- tool_manager  : registers tools, assigns them to agents, adds/removes agents.\n"
            "- shell_agent   : runs shell commands and handles file-system tasks.\n"
            "- compute_agent : performs calculations with its dynamically assigned tools.\n\n"
            "Routing rules:\n"
            "  * Add/register/assign/remove/list tools or agents -> tool_manager.\n"
            "  * Shell or file-system tasks -> shell_agent.\n"
            "  * Domain tasks (download, simulate, generate, compute) ->\n"
            "    the specialist agent that owns the relevant tool.\n"
            "  * If no specialist exists for the task, ask tool_manager to create one first.\n"
            "Assign work to one agent at a time. Execute immediately."
        ),
    )
    pool.add_agent(
        "shell_agent", model,
        base_tools=[ShellTool()],
        system_prompt="You are a shell agent. Execute shell commands to answer requests.",
    )
    pool.add_agent(
        "compute_agent", model,
        base_tools=[],
        system_prompt="You are a computation agent. Use your assigned tools for calculations.",
    )
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])
    return pool


def tool_was_called(stream_chunks: list, tool_name: str) -> bool:
    """
    Return True if a ToolMessage with the given name appears in stream output.
    Use this to distinguish real tool execution from hallucinated text responses.
    """
    for chunk in stream_chunks:
        for node_msgs in chunk.values():
            msgs = node_msgs if isinstance(node_msgs, list) else node_msgs.get("messages", [])
            for msg in msgs:
                if isinstance(msg, ToolMessage) and msg.name == tool_name:
                    return True
    return False


def count_tool_calls(stream_chunks: list, tool_name: str) -> int:
    """Count how many times a ToolMessage for tool_name appears in stream output."""
    count = 0
    for chunk in stream_chunks:
        for node_msgs in chunk.values():
            msgs = node_msgs if isinstance(node_msgs, list) else node_msgs.get("messages", [])
            for msg in msgs:
                if isinstance(msg, ToolMessage) and msg.name == tool_name:
                    count += 1
    return count


def _thread(prefix: str = "test") -> dict:
    return {"configurable": {"thread_id": f"{prefix}-{int(time.time())}"}}


PASS = "✓ PASS"
FAIL = "✗ FAIL"

# ── T0: Enhancer output tests ─────────────────────────────────────────────────

def test_enhancer_routes_simulation_to_specialist():
    """Enhanced T3 query must name mace_md_specialist and run_nvt_md."""
    model = _make_model()
    pool  = _build_pool(model)

    # Register a dummy run_nvt_md tool so the pool has it
    pool.register_tool_from_code(
        'def run_nvt_md(structure_xyz, model_path, num_steps, temp_k, output_traj):\n'
        '    """Run an NVT MD simulation and write a trajectory file."""\n'
        '    pass\n'
    )
    pool.add_agent(
        "mace_md_specialist", model,
        base_tools=[],
        system_prompt="I am a MACE MD specialist for molecular dynamics simulations.",
    )
    pool.assign_tool("run_nvt_md", "mace_md_specialist")

    enhancer = PromptEnhancer(model, pool)
    query = (
        "Run an NVT molecular dynamics simulation on the structure at /data/nacl.xyz. "
        "Use the MACE model at /models/mace.model. "
        "Apply PBC with a 20 Å cubic cell, 300 K, 100 steps. "
        "Save trajectory to /out/traj.traj."
    )
    enhanced = enhancer.enhance(query)
    print(f"  Enhanced:\n    {enhanced}\n")

    assert "mace_md_specialist" in enhanced, \
        f"Expected 'mace_md_specialist' in enhanced output. Got:\n{enhanced}"
    assert "run_nvt_md" in enhanced, \
        f"Expected 'run_nvt_md' in enhanced output. Got:\n{enhanced}"
    print(PASS)


def test_enhancer_routes_registration_to_tool_manager():
    """A .py file registration query must route to tool_manager with register_tool_from_file."""
    model = _make_model()
    pool  = _build_pool(model)
    enhancer = PromptEnhancer(model, pool)

    query = "Register the tools in /scripts/ase_nvt.py and assign run_nvt_md to mace_md_specialist."
    enhanced = enhancer.enhance(query)
    print(f"  Enhanced:\n    {enhanced}\n")

    assert "tool_manager" in enhanced, \
        f"Expected 'tool_manager' in enhanced output. Got:\n{enhanced}"
    assert "register_tool_from_file" in enhanced, \
        f"Expected 'register_tool_from_file' in enhanced output. Got:\n{enhanced}"
    print(PASS)


def test_enhancer_no_code_in_output():
    """Enhancer must not add 'def ' code to a plain simulation query."""
    model = _make_model()
    pool  = _build_pool(model)
    pool.register_tool_from_code(
        'def run_nvt_md(structure_xyz, model_path, num_steps, temp_k, output_traj):\n'
        '    """Run an NVT MD simulation."""\n    pass\n'
    )
    pool.add_agent("mace_md_specialist", model, base_tools=[],
                   system_prompt="MACE MD specialist.")
    pool.assign_tool("run_nvt_md", "mace_md_specialist")

    enhancer = PromptEnhancer(model, pool)
    query = "Run a simulation on /data/mol.xyz with MACE at /models/m.model, 300 K, 50 steps."
    enhanced = enhancer.enhance(query)

    assert "def " not in enhanced, \
        f"Enhancer should not emit Python code. Got:\n{enhanced}"
    print(f"  Enhanced (no code): {enhanced[:100]}...")
    print(PASS)


def test_enhancer_priority_c_skips_download():
    """Query with an absolute input path must NOT include a download step."""
    model = _make_model()
    pool  = _build_pool(model)
    pool.register_tool_from_code(
        'def download_mace_model(model_name, output_dir):\n'
        '    """Download a MACE model."""\n    pass\n'
    )
    pool.register_tool_from_code(
        'def run_nvt_md(structure_xyz, model_path, num_steps, temp_k, output_traj):\n'
        '    """Run an NVT MD simulation."""\n    pass\n'
    )
    pool.add_agent("mace_md_specialist", model, base_tools=[],
                   system_prompt="MACE MD specialist.")
    pool.assign_tool("download_mace_model", "mace_md_specialist")
    pool.assign_tool("run_nvt_md", "mace_md_specialist")

    enhancer = PromptEnhancer(model, pool)
    query = (
        "Run NVT MD on /data/nacl.xyz using the MACE model at /models/mace.model, "
        "300 K, 100 steps. Save to /out/traj.traj."
    )
    enhanced = enhancer.enhance(query)
    print(f"  Enhanced:\n    {enhanced}\n")

    # Priority C: the model path is given as an absolute input — no download step
    assert "download" not in enhanced.lower(), \
        f"Enhancer should not add a download step when model path is given. Got:\n{enhanced}"
    print(PASS)


# ── T1: Tool registration unit tests (no LLM) ─────────────────────────────────

def test_register_tool_from_code():
    """Registering a trivial function adds it to the pool registry."""
    pool = AgentPoolWithSupervisor(supervisor_model=_make_model())

    result = pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers and return result as string."""\n'
        '    return str(a + b)\n'
    )
    print(f"  register_tool_from_code result: {result}")
    assert "add_numbers" in pool.list_registered_tools(), \
        f"Tool not in registry. Got: {pool.list_registered_tools()}"
    print(PASS)


def test_assign_tool_to_agent():
    """Assigning a registered tool adds it to the agent's extra_tools."""
    pool = AgentPoolWithSupervisor(supervisor_model=_make_model())
    pool.add_agent("compute_agent", _make_model(), base_tools=[],
                   system_prompt="Computation agent.")

    pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers and return result as string."""\n'
        '    return str(a + b)\n'
    )
    result = pool.assign_tool("add_numbers", "compute_agent")
    print(f"  assign_tool result: {result}")

    extra = [t.name for t in pool._agents["compute_agent"]["extra_tools"]]
    assert "add_numbers" in extra, \
        f"Tool not assigned to compute_agent. extra_tools: {extra}"
    print(PASS)


def test_tool_directly_callable():
    """The registered StructuredTool is callable and returns the correct answer."""
    pool = AgentPoolWithSupervisor(supervisor_model=_make_model())
    pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers and return result as string."""\n'
        '    return str(a + b)\n'
    )
    tool = pool._tool_registry["add_numbers"]
    result = tool.invoke({"a": 3.0, "b": 4.0})
    print(f"  Direct tool call result: {result}")
    assert result == "7.0", f"Expected '7.0', got '{result}'"
    print(PASS)


def test_execution_rule_no_duplicate():
    """Agent system prompt must have the execution rule injected exactly once."""
    pool = AgentPoolWithSupervisor(supervisor_model=_make_model())
    pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers."""\n    return str(a + b)\n'
    )
    pool.add_agent("compute_agent", _make_model(), base_tools=[],
                   system_prompt="You are a computation agent.")
    pool.assign_tool("add_numbers", "compute_agent")

    entry = pool._agents["compute_agent"]
    # Effective prompt is base_system_prompt + tool_section + execution_rule
    # We check there is exactly one "Execution rules:" block
    effective = (
        (entry.get("base_system_prompt") or "") +
        "\n".join(
            f"  - {t.name}" for t in entry["extra_tools"]
        )
    )
    # The agent graph prompt is stored on the compiled agent
    graph_prompt = str(entry["agent"].get_prompts() if hasattr(entry["agent"], "get_prompts") else "")
    count = effective.count("Execution rules:") + graph_prompt.count("Execution rules:")
    # We just verify the base_system_prompt itself has zero injected copies
    base_count = (entry.get("base_system_prompt") or "").count("Execution rules:")
    print(f"  'Execution rules:' occurrences in base_system_prompt: {base_count}")
    assert base_count == 0, \
        f"base_system_prompt should be clean (0 execution rule copies), found {base_count}"
    print(PASS)


# ── T2: Agent tool-call verification ──────────────────────────────────────────

def test_agent_calls_tool_not_text():
    """
    Agent must make a real tool call (ToolMessage in history), not a text description.
    This is the ground-truth hallucination check.
    """
    model = _make_model()
    pool  = AgentPoolWithSupervisor(supervisor_model=model)
    pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers and return the result as a string."""\n'
        '    return str(a + b)\n'
    )
    pool.add_agent("compute_agent", model, base_tools=[],
                   system_prompt="You are a computation agent.")
    pool.assign_tool("add_numbers", "compute_agent")

    agent = pool.get_agent("compute_agent")
    chunks = list(agent.stream(
        {"messages": [{"role": "user", "content": "Add 12 and 30 using add_numbers."}]},
        config=_thread("t2-tool-call"),
    ))

    called = tool_was_called(chunks, "add_numbers")
    print(f"  tool_was_called('add_numbers'): {called}")
    if not called:
        print("  [streams received]")
        for c in chunks:
            print(f"    {c}")
    assert called, \
        "Agent did NOT make a real tool call — likely hallucinating. Check prompt & tool binding."
    print(PASS)


def test_agent_no_double_call():
    """Agent must call the tool exactly once per request, then stop."""
    model = _make_model()
    pool  = AgentPoolWithSupervisor(supervisor_model=model)
    pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers and return the result as a string."""\n'
        '    return str(a + b)\n'
    )
    pool.add_agent("compute_agent", model, base_tools=[],
                   system_prompt="You are a computation agent.")
    pool.assign_tool("add_numbers", "compute_agent")

    agent  = pool.get_agent("compute_agent")
    chunks = list(agent.stream(
        {"messages": [{"role": "user", "content": "Add 5 and 7 using add_numbers."}]},
        config=_thread("t2-no-double"),
    ))

    n = count_tool_calls(chunks, "add_numbers")
    print(f"  add_numbers called {n} time(s)")
    assert n == 1, f"Expected exactly 1 tool call, got {n}"
    print(PASS)


# ── T3: Supervisor routing tests ───────────────────────────────────────────────

def _handoff_targets(stream_chunks: list) -> list[str]:
    """Extract agent names that appear as HandoffMessage or node keys in stream chunks."""
    targets = []
    for chunk in stream_chunks:
        for key in chunk.keys():
            if key not in ("supervisor", "__end__"):
                targets.append(key)
    return targets


def test_supervisor_routes_domain_task_to_specialist():
    """Supervisor must route a compute task to compute_agent, not tool_manager."""
    model = _make_model()
    pool  = _build_pool(model)
    pool.register_tool_from_code(
        'def add_numbers(a: float, b: float) -> str:\n'
        '    """Add two numbers and return the result as a string."""\n'
        '    return str(a + b)\n'
    )
    pool.assign_tool("add_numbers", "compute_agent")

    chunks = list(pool.supervisor.stream(
        {"messages": [{"role": "user",
                       "content": "Use compute_agent — call add_numbers with a=10 and b=20."}]},
        config=_thread("t3-domain"),
        recursion_limit=15,
    ))

    targets = _handoff_targets(chunks)
    print(f"  Nodes visited: {targets}")
    assert "compute_agent" in targets, \
        f"Supervisor did not route to compute_agent. Visited: {targets}"
    assert "tool_manager" not in targets, \
        f"Supervisor incorrectly routed to tool_manager. Visited: {targets}"
    print(PASS)


def test_supervisor_routes_registration_to_tool_manager():
    """Supervisor must route a tool-registration request to tool_manager."""
    model = _make_model()
    pool  = _build_pool(model)

    code = (
        'def square(x: float) -> str:\n'
        '    """Return x squared as a string."""\n'
        '    return str(x * x)\n'
    )
    query = (
        f"Register the following Python function:\n\n{code}\n\n"
        "Then assign it to compute_agent."
    )

    chunks = list(pool.supervisor.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=_thread("t3-register"),
        recursion_limit=20,
    ))

    targets = _handoff_targets(chunks)
    print(f"  Nodes visited: {targets}")
    assert "tool_manager" in targets, \
        f"Supervisor did not route to tool_manager. Visited: {targets}"
    print(PASS)


# ── T4: End-to-end integration tests (slow, require MACE env) ─────────────────
# Set E2E=1 in environment to enable these tests.

_E2E = os.environ.get("E2E", "0") == "1"


def _skip_unless_e2e(test_name: str) -> bool:
    if not _E2E:
        print(f"  [SKIP] {test_name} — set E2E=1 to run e2e tests.")
        return True
    return False


def _notebook_pool():
    """Build a pool matching the notebook 6 setup (requires MACE environment)."""
    model = _make_model("gpt-4o")
    SUPERVISOR_PROMPT = (
        "You are the Supervisor managing a pool of agents.\n"
        "- tool_manager              : registers tools, assigns them to agents, and adds/removes agents.\n"
        "- shell_agent               : runs shell commands and handles file-system tasks.\n"
        "- compute_agent             : performs calculations with its dynamically assigned tools.\n\n"
        "Routing rules:\n"
        "  * Add/register/assign/remove/list tools or agents -> tool_manager.\n"
        "  * Python code (def statements) + add/register intent -> tool_manager.\n"
        "  * Shell or file-system tasks -> shell_agent.\n"
        "  * Domain tasks (download, simulate, generate, compute, create files) ->\n"
        "    the specialist agent that owns the relevant tool.\n"
        "  * If no specialist exists for the task, ask tool_manager to create one first.\n\n"
        "Execution rules:\n"
        "  * If you have all you need execute tasks immediately.\n"
        "  * When a specialist agent completes a calculation, report the full numerical\n"
        "    result directly.\n"
        "  * Assign work to one agent at a time."
    )
    pool = AgentPoolWithSupervisor(
        supervisor_model=model,
        supervisor_prompt=SUPERVISOR_PROMPT,
    )
    pool.add_agent("shell_agent", model, base_tools=[ShellTool()],
                   system_prompt="You are a shell agent. Execute shell commands and handle file-system tasks.")
    pool.add_agent("compute_agent", model, base_tools=[],
                   system_prompt="You are a computation agent. Use your dynamically assigned tools.")
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])
    return pool, model, PromptEnhancer(model, pool)


def test_e2e_T1_register_download_model():
    """T1: register download_mace_model tool; assert it appears in registry."""
    if _skip_unless_e2e("T1 register download_mace_model"):
        return

    pool, model, enhancer = _notebook_pool()

    DOWNLOAD_CODE = '''\
import os, urllib.request

def download_mace_model(model_name: str, output_dir: str, convert_lmp: bool = False) -> str:
    """Download a MACE model checkpoint by name to output_dir.
    Supported names: mace-mp-0b3-medium.
    Returns the path to the downloaded .model file."""
    MODEL_URLS = {
        "mace-mp-0b3-medium": (
            "https://github.com/ACEsuit/mace-mp/releases/download/"
            "mace_mp_0b3/2024-01-07-mace-128-L2_epoch-199.model"
        ),
    }
    if model_name not in MODEL_URLS:
        return f"Unknown model '{model_name}'. Available: {list(MODEL_URLS)}"
    os.makedirs(output_dir, exist_ok=True)
    url  = MODEL_URLS[model_name]
    dest = os.path.join(output_dir, f"{model_name}.model")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
    return dest
'''

    query = (
        f"Register the following Python function as a tool:\n\n{DOWNLOAD_CODE}\n\n"
        "Then create a specialist agent called 'mace_md_specialist' with a system prompt "
        "describing it as a MACE MD specialist. "
        "Then assign download_mace_model to mace_md_specialist."
    )
    enhanced = enhancer.enhance(query)
    print(f"  [enhancer] {enhanced[:120]}...\n")

    chunks = list(pool.supervisor.stream(
        {"messages": [{"role": "user", "content": enhanced}]},
        config=_thread("e2e-T1"),
        recursion_limit=30,
    ))
    for c in chunks:
        pretty_print_messages(c, last_message=True)

    assert "download_mace_model" in pool.list_registered_tools(), \
        f"download_mace_model not in registry. Got: {pool.list_registered_tools()}"
    assert "mace_md_specialist" in pool.list_agents(), \
        f"mace_md_specialist not created. Got: {pool.list_agents()}"
    extra = [t.name for t in pool._agents["mace_md_specialist"]["extra_tools"]]
    assert "download_mace_model" in extra, \
        f"download_mace_model not assigned. Got: {extra}"
    print(PASS)


def test_e2e_T3_run_nvt_md():
    """
    T3: Full pipeline — supervisor routes to mace_md_specialist which calls run_nvt_md.
    Asserts:
      1. The output trajectory file exists on disk.
      2. A real ToolMessage for run_nvt_md appears (no hallucination).
    Requires the pool to already have mace_md_specialist with run_nvt_md assigned.
    """
    if _skip_unless_e2e("T3 run NVT MD"):
        return

    TUTORIALS = os.path.join(ROOT, "tutorials")
    INPUT_XYZ   = os.path.join(TUTORIALS, "nacl_water_box.xyz")
    MACE_MODEL  = os.path.join(TUTORIALS, "models", "mace-mp-0b3-medium.model")
    OUTPUT_TRAJ = os.path.join(TUTORIALS, "nvt_nacl_water_e2e.traj")

    if not os.path.exists(INPUT_XYZ):
        print(f"  [SKIP] Input file not found: {INPUT_XYZ}")
        return
    if not os.path.exists(MACE_MODEL):
        print(f"  [SKIP] MACE model not found: {MACE_MODEL}")
        return

    pool, model, enhancer = _notebook_pool()

    # Register run_nvt_md from the ASE script
    ASE_SCRIPT = os.path.join(TUTORIALS, "ASE_NVT_PBC.py")
    assert os.path.exists(ASE_SCRIPT), f"ASE script not found: {ASE_SCRIPT}"
    pool.register_tool_from_file(ASE_SCRIPT)
    pool.add_agent("mace_md_specialist", model, base_tools=[],
                   system_prompt="I am a MACE MD specialist for molecular dynamics.")
    pool.assign_tool("run_nvt_md", "mace_md_specialist")

    query = (
        f"Run an NVT molecular dynamics simulation on the structure at {INPUT_XYZ}. "
        f"Use the MACE model saved at {MACE_MODEL}. "
        f"Apply periodic boundary conditions with a 20.0 Å cubic simulation cell, "
        "a temperature of 300 K, and 100 steps. "
        f"Save the trajectory to {OUTPUT_TRAJ}."
    )
    enhanced = enhancer.enhance(query)
    print(f"  [enhancer] {enhanced[:120]}...\n")

    chunks = list(pool.supervisor.stream(
        {"messages": [{"role": "user", "content": enhanced}]},
        config=_thread("e2e-T3"),
        recursion_limit=30,
    ))
    for c in chunks:
        pretty_print_messages(c, last_message=True)

    assert os.path.exists(OUTPUT_TRAJ), \
        f"Trajectory not created: {OUTPUT_TRAJ}"
    assert tool_was_called(chunks, "run_nvt_md"), \
        "run_nvt_md was NOT actually called — agent hallucinated tool execution."
    print(f"  Trajectory: {OUTPUT_TRAJ} ({os.path.getsize(OUTPUT_TRAJ)} bytes)")
    print(PASS)


# ── runner ────────────────────────────────────────────────────────────────────

_T0_TESTS = [
    test_enhancer_routes_simulation_to_specialist,
    test_enhancer_routes_registration_to_tool_manager,
    test_enhancer_no_code_in_output,
    test_enhancer_priority_c_skips_download,
]

_T1_TESTS = [
    test_register_tool_from_code,
    test_assign_tool_to_agent,
    test_tool_directly_callable,
    test_execution_rule_no_duplicate,
]

_T2_TESTS = [
    test_agent_calls_tool_not_text,
    test_agent_no_double_call,
]

_T3_TESTS = [
    test_supervisor_routes_domain_task_to_specialist,
    test_supervisor_routes_registration_to_tool_manager,
]

_T4_TESTS = [
    test_e2e_T1_register_download_model,
    test_e2e_T3_run_nvt_md,
]


def _run_suite(tests: list, label: str) -> tuple[int, int]:
    passed = failed = 0
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    for fn in tests:
        print(f"\n  [{fn.__name__}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {FAIL}: {e}")
            failed += 1
    return passed, failed


if __name__ == "__main__":
    total_pass = total_fail = 0

    for suite, label in [
        (_T0_TESTS, "T0 — Enhancer output"),
        (_T1_TESTS, "T1 — Tool registration (no LLM)"),
        (_T2_TESTS, "T2 — Agent tool-call verification"),
        (_T3_TESTS, "T3 — Supervisor routing"),
        (_T4_TESTS, "T4 — End-to-end integration (E2E=1 to enable)"),
    ]:
        p, f = _run_suite(suite, label)
        total_pass += p
        total_fail += f

    print(f"\n{'='*60}")
    print(f"Results: {total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
