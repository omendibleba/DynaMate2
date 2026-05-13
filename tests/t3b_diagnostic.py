#!/usr/bin/env python3
"""
tests/t3b_diagnostic.py
────────────────────────
Three-level diagnostic for the T3b (NVT simulation) failure in the Gradio UI.

Runs against the REAL ui_state/ pool so it replicates the exact production
context.  Execute from the project root:

    python tests/t3b_diagnostic.py

Level 1 – direct tool call
    Calls run_nvt_md from the pool registry with the same arguments that
    PROMPT_T3B would pass.  Catches any exception and prints it verbatim.
    This is the fastest way to find CUDA errors, bad paths, or traj_interval
    surprises.

Level 2 – direct agent call (no supervisor)
    Sends a message straight to mace_md_specialist.  Confirms the agent calls
    the tool and reports the result correctly — without the supervisor or
    conversation history getting in the way.

Level 3 – fresh-thread pipeline
    Runs the full PromptEnhancer → Supervisor → Agent pipeline using a brand-
    new thread_id so prior conversation history does not bleed in.  This
    replicates the notebook's Cell 35 approach (which uses a fresh thread).

Level 4 – same-thread pipeline (replicates the UI bug)
    Re-runs the pipeline with the SAME thread_id that level 3 used, simulating
    the second time T3b is sent in the same session.  If the tool's early-exit
    guard fires ("Trajectory already exists — skipping"), or if the supervisor
    hallucinates, this level will reveal it.
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

from langchain_community.tools import ShellTool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
)

# ── constants — mirror PROMPT_T3B exactly ─────────────────────────────────────

TUTORIALS   = os.path.join(ROOT, "tutorials")
MODEL_PATH  = os.path.join(TUTORIALS, "models", "mace-mp-0b3-medium.model")
STRUCT_FILE = os.path.join(TUTORIALS, "nacl_water_box.xyz")
OUT_TRAJ    = os.path.join(TUTORIALS, "diag_nvt.traj")   # separate file so real traj is untouched
BOX_SIZE    = 20.0
TEMP_K      = 300.0
N_STEPS     = 10
TRAJ_INTV   = 10   # ← Level 1 uses 10, not the default 100 (see Issue 2 in plan)

TOOL_ARGS = dict(
    model_path=MODEL_PATH,
    structure_file=STRUCT_FILE,
    box_size=BOX_SIZE,
    temperature_K=TEMP_K,
    n_steps=N_STEPS,
    output_traj=OUT_TRAJ,
    traj_interval=TRAJ_INTV,
)

AGENT_QUERY = (
    f"Please run a short NVT molecular dynamics simulation using the run_nvt_md tool.\n"
    f"model_path      = {MODEL_PATH}\n"
    f"structure_file  = {STRUCT_FILE}\n"
    f"box_size        = {BOX_SIZE}\n"
    f"temperature_K   = {TEMP_K}\n"
    f"n_steps         = {N_STEPS}\n"
    f"output_traj     = {OUT_TRAJ}\n"
    f"traj_interval   = {TRAJ_INTV}\n"
    "Execute the tool immediately. Do not ask for confirmation."
)

PIPELINE_QUERY = (
    "Please run a short NVT molecular dynamics simulation using the run_nvt_md tool.\n"
    "The model file is already on disk — do NOT call download_mace_model.\n"
    f"model_path     = {MODEL_PATH}\n"
    f"structure_file = {STRUCT_FILE}\n"
    f"box_size        = {BOX_SIZE}\n"
    f"temperature_K   = {TEMP_K}\n"
    f"n_steps         = {N_STEPS}\n"
    f"traj_interval   = {TRAJ_INTV}\n"
    f"output_traj    = {OUT_TRAJ}\n"
    "Execute run_nvt_md immediately with these parameters."
)

SPECIALIST = "mace_md_specialist"
MODEL_NAME = os.getenv("DYNAMATE_MODEL", "gpt-4.1-mini")

_SEP = "─" * 58


def _build_pool() -> tuple:
    """Load the real ui_state/ pool (same as app.py does on startup)."""
    model      = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    saver      = PersistentSaver(os.path.join(ROOT, "ui_state", "conversations.db"))
    pool_store = PoolStore(os.path.join(ROOT, "ui_state", "pool_state.json"))

    SUPERVISOR_PROMPT = (
        "You are the Supervisor managing a pool of agents.\n"
        "- tool_manager  : registers tools, assigns them to agents, and adds/removes agents.\n"
        "- shell_agent   : runs shell commands and handles file-system tasks.\n"
        "- compute_agent : performs calculations with its dynamically assigned tools.\n\n"
        "Routing rules:\n"
        "  * Add/register/assign/remove/list tools or agents -> tool_manager.\n"
        "  * Python code (def statements) + add/register intent -> tool_manager.\n"
        "  * Shell or file-system tasks -> shell_agent.\n"
        "  * Domain tasks (download, simulate, generate, compute, create files) ->\n"
        "    the specialist agent that owns the relevant tool. Do NOT route these\n"
        "    to tool_manager — tool_manager only manages the pool, it cannot execute\n"
        "    domain work.\n"
        "  * If no specialist exists for the task, ask tool_manager to create one first.\n\n"
        "Execution rules:\n"
        "  * If you have all you need execute tasks immediately.\n"
        "  * When a specialist agent completes a calculation, report the full numerical\n"
        "    result directly. Do not say 'the agent is ready' or ask what to do next.\n"
        "  * Assign work to one agent at a time."
    )

    pool = PersistentAgentPoolWithSupervisor(
        supervisor_model=model,
        pool_store=pool_store,
        supervisor_prompt=SUPERVISOR_PROMPT,
        checkpointer=saver,
    )
    pool.add_agent(
        name="shell_agent",
        model=model,
        base_tools=[ShellTool()],
        system_prompt="You are a shell agent. Execute shell commands to answer requests.",
        _is_dynamic=False,
    )
    pool.add_agent(
        name="compute_agent",
        model=model,
        base_tools=[],
        system_prompt=(
            "You are a computation agent. "
            "Use your dynamically assigned tools to perform calculations."
        ),
        _is_dynamic=False,
    )
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])
    pool.restore_state(model_factory=lambda name: ChatOpenAI(model=name, temperature=0.0))

    # Mirrors app.py _build_system(): explicit rebuild so execution rules are fresh.
    _STATIC = {"shell_agent", "compute_agent"}
    for _name in list(pool._agents):
        if _name not in _STATIC:
            pool._rebuild_agent(_name)
    pool._rebuild_supervisor()

    enhancer = PromptEnhancer(model=model, pool=pool)
    return pool, enhancer


# ── Level 1: direct tool call ─────────────────────────────────────────────────

def level1(pool) -> bool:
    print(f"\n{_SEP}")
    print("  LEVEL 1 — direct tool call")
    print(_SEP)

    # Clean up any leftover file from a previous run
    if os.path.exists(OUT_TRAJ):
        os.remove(OUT_TRAJ)
        print(f"  (removed stale {OUT_TRAJ})")

    if SPECIALIST not in pool._agents:
        print(f"FAIL ✗  '{SPECIALIST}' not in pool.")
        return False
    if "run_nvt_md" not in pool._tool_registry:
        print(f"FAIL ✗  run_nvt_md not in registry. Registered: {pool.list_registered_tools()}")
        return False

    print(f"  Calling run_nvt_md with:")
    for k, v in TOOL_ARGS.items():
        print(f"    {k} = {v}")
    print()

    try:
        result = pool._tool_registry["run_nvt_md"].invoke(TOOL_ARGS)
        print(f"  Tool returned : {result}")
        exists = os.path.exists(OUT_TRAJ)
        size   = os.path.getsize(OUT_TRAJ) if exists else 0
        print(f"  File created  : {exists}  ({size} bytes)")
        if not exists:
            print("  WARN  Tool returned a result but file is absent.")
            print("        This means the tool function succeeded but the file was never written.")
            print("        Check traj_interval vs n_steps (Issue 2).")
        print(f"\n  PASS ✓" if exists else "\n  FAIL ✗  file absent after tool call")
        return exists
    except Exception as exc:
        print(f"  FAIL ✗  Exception: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ── Level 2: direct agent call ────────────────────────────────────────────────

def level2(pool) -> bool:
    print(f"\n{_SEP}")
    print("  LEVEL 2 — direct agent call (no supervisor)")
    print(_SEP)

    if os.path.exists(OUT_TRAJ):
        os.remove(OUT_TRAJ)
        print(f"  (removed stale {OUT_TRAJ})")

    if SPECIALIST not in pool._agents:
        print(f"FAIL ✗  '{SPECIALIST}' not in pool.")
        return False

    agent = pool._agents[SPECIALIST]["agent"]
    assigned = [t.name for t in pool._agents[SPECIALIST]["extra_tools"]]
    print(f"  Assigned tools: {assigned}")
    print(f"  Query         :\n    {AGENT_QUERY[:200]}...\n")

    try:
        response = agent.invoke({"messages": [HumanMessage(content=AGENT_QUERY)]})
    except Exception as exc:
        print(f"  FAIL ✗  Exception invoking agent: {exc}")
        import traceback
        traceback.print_exc()
        return False

    tool_calls_seen = 0
    for msg in response.get("messages", []):
        msg_type = getattr(msg, "type", type(msg).__name__)
        if msg_type == "tool":
            tool_calls_seen += 1
            print(f"  [tool call]  {getattr(msg, 'name', '?')}  →  {str(msg.content)[:300]}")
        elif msg_type == "ai" and getattr(msg, "content", ""):
            print(f"  [agent]      {msg.content[:300]}")

    exists = os.path.exists(OUT_TRAJ)
    size   = os.path.getsize(OUT_TRAJ) if exists else 0
    print(f"\n  Tool calls    : {tool_calls_seen}")
    print(f"  File created  : {exists}  ({size} bytes)")

    if tool_calls_seen == 0:
        print("\n  FAIL ✗  Agent never called a tool — pure hallucination.")
        sp = pool._agents[SPECIALIST].get("system_prompt", "")
        print(f"  Effective system prompt (first 400 chars):\n    {sp[:400]}")
        return False
    if not exists:
        print("\n  FAIL ✗  Tool was called but file not created (see Level 1 for root cause).")
        return False

    print("\n  PASS ✓")
    return True


# ── Level 3: fresh-thread pipeline ───────────────────────────────────────────

def level3(pool, enhancer) -> bool:
    print(f"\n{_SEP}")
    print("  LEVEL 3 — fresh-thread pipeline  (mirrors notebook Cell 35)")
    print(_SEP)

    if os.path.exists(OUT_TRAJ):
        os.remove(OUT_TRAJ)
        print(f"  (removed stale {OUT_TRAJ})")

    fresh_tid = f"diag-t3b-{int(time.time())}"
    config    = {"configurable": {"thread_id": fresh_tid}}

    print(f"  Thread ID     : {fresh_tid}")
    enhanced = enhancer.enhance(PIPELINE_QUERY)
    print(f"  Enhanced hint : {enhanced[len(PIPELINE_QUERY):].strip()[:200]}")
    print()

    tool_calls_seen = 0
    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": enhanced}]},
            config=config,
            recursion_limit=25,
        ):
            for node_name, data in chunk.items() if isinstance(chunk, dict) else []:
                for msg in (data.get("messages", []) if isinstance(data, dict) else []):
                    msg_type = getattr(msg, "type", type(msg).__name__)
                    if msg_type == "tool":
                        nm = getattr(msg, "name", "")
                        if not nm.startswith("transfer_"):
                            tool_calls_seen += 1
                            print(f"  [{node_name}/tool]  {nm}  →  {str(msg.content)[:200]}")
                    elif msg_type == "ai" and getattr(msg, "content", ""):
                        print(f"  [{node_name}]  {msg.content[:200]}")
    except Exception as exc:
        print(f"  FAIL ✗  Exception: {exc}")
        import traceback
        traceback.print_exc()
        return False

    exists = os.path.exists(OUT_TRAJ)
    size   = os.path.getsize(OUT_TRAJ) if exists else 0
    print(f"\n  Tool calls    : {tool_calls_seen}")
    print(f"  File created  : {exists}  ({size} bytes)")

    if tool_calls_seen == 0:
        print("\n  FAIL ✗  No domain tool calls in stream — supervisor hallucinated.")
        return False
    if not exists:
        print("\n  FAIL ✗  Tool called but file absent (see Level 1 for root cause).")
        return False

    print("\n  PASS ✓")
    return True


# ── Level 4: same-thread re-run (replicates UI early-exit bug) ───────────────

def level4(pool, enhancer, shared_tid: str) -> bool:
    print(f"\n{_SEP}")
    print("  LEVEL 4 — same-thread re-run  (replicates UI session bleed)")
    print(_SEP)
    print(f"  Thread ID     : {shared_tid}")
    print(f"  File exists   : {os.path.exists(OUT_TRAJ)}  (should exist from Level 3)")
    print()

    # Do NOT remove the file — we want to hit the early-exit guard
    config   = {"configurable": {"thread_id": shared_tid}}
    enhanced = enhancer.enhance(PIPELINE_QUERY)

    tool_calls_seen = 0
    skip_detected   = False
    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": enhanced}]},
            config=config,
            recursion_limit=25,
        ):
            for node_name, data in chunk.items() if isinstance(chunk, dict) else []:
                for msg in (data.get("messages", []) if isinstance(data, dict) else []):
                    msg_type = getattr(msg, "type", type(msg).__name__)
                    if msg_type == "tool":
                        nm = getattr(msg, "name", "")
                        content = str(msg.content)
                        if not nm.startswith("transfer_"):
                            tool_calls_seen += 1
                            print(f"  [{node_name}/tool]  {nm}  →  {content[:200]}")
                        if "already exists" in content.lower():
                            skip_detected = True
                            print("  WARN  ↑ Early-exit guard fired: trajectory file exists!")
                    elif msg_type == "ai" and getattr(msg, "content", ""):
                        print(f"  [{node_name}]  {msg.content[:200]}")
    except Exception as exc:
        print(f"  Exception: {exc}")

    print(f"\n  Tool calls    : {tool_calls_seen}")
    print(f"  Early-exit    : {skip_detected}")
    if skip_detected:
        print("  INFO  This confirms Issue 2: the tool skips if output_traj already exists.")
        print("        The UI must delete or rename the trajectory before re-running.")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  T3b Diagnostic  (real ui_state/ pool)")
    print("=" * 58)

    print("\nLoading pool from ui_state/…")
    pool, enhancer = _build_pool()

    print(f"\nPool agents   : {pool.list_agents()}")
    print(f"Registry      : {pool.list_registered_tools()}")
    assigned = [t.name for t in pool._agents.get(SPECIALIST, {}).get("extra_tools", [])]
    print(f"Specialist tools: {assigned}")

    p1 = level1(pool)
    p2 = level2(pool)

    # Level 3 uses a fresh thread; capture the tid so Level 4 can reuse it
    shared_tid = f"diag-t3b-{int(time.time())}"
    if os.path.exists(OUT_TRAJ):
        os.remove(OUT_TRAJ)

    p3 = False
    config3 = {"configurable": {"thread_id": shared_tid}}
    if os.path.exists(OUT_TRAJ):
        os.remove(OUT_TRAJ)
    enhanced3 = enhancer.enhance(PIPELINE_QUERY)

    print(f"\n{_SEP}")
    print("  LEVEL 3 — fresh-thread pipeline  (mirrors notebook Cell 35)")
    print(_SEP)
    print(f"  Thread ID     : {shared_tid}")
    print(f"  Enhanced hint : {enhanced3[len(PIPELINE_QUERY):].strip()[:200]}")
    print()

    tc3 = 0
    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": enhanced3}]},
            config=config3,
            recursion_limit=25,
        ):
            for node_name, data in chunk.items() if isinstance(chunk, dict) else []:
                for msg in (data.get("messages", []) if isinstance(data, dict) else []):
                    msg_type = getattr(msg, "type", type(msg).__name__)
                    if msg_type == "tool":
                        nm = getattr(msg, "name", "")
                        if not nm.startswith("transfer_"):
                            tc3 += 1
                            print(f"  [{node_name}/tool]  {nm}  →  {str(msg.content)[:200]}")
                    elif msg_type == "ai" and getattr(msg, "content", ""):
                        print(f"  [{node_name}]  {msg.content[:200]}")
    except Exception as exc:
        print(f"  FAIL ✗  Exception: {exc}")

    exists3 = os.path.exists(OUT_TRAJ)
    size3   = os.path.getsize(OUT_TRAJ) if exists3 else 0
    print(f"\n  Tool calls    : {tc3}")
    print(f"  File created  : {exists3}  ({size3} bytes)")
    p3 = tc3 > 0 and exists3
    print(f"\n  {'PASS ✓' if p3 else 'FAIL ✗'}")

    p4 = level4(pool, enhancer, shared_tid)

    # Clean up diagnostic file
    if os.path.exists(OUT_TRAJ):
        os.remove(OUT_TRAJ)
        print(f"\n  (cleaned up {OUT_TRAJ})")

    print(f"\n{'=' * 58}")
    print(f"  Level 1 (direct tool)      : {'PASS ✓' if p1 else 'FAIL ✗'}")
    print(f"  Level 2 (direct agent)     : {'PASS ✓' if p2 else 'FAIL ✗'}")
    print(f"  Level 3 (fresh thread)     : {'PASS ✓' if p3 else 'FAIL ✗'}")
    print(f"  Level 4 (early-exit check) : run — see output above")
    print("=" * 58)
    print()
    print("Read the FAIL/WARN lines above to identify the root cause.")
    print("Refer to the plan in app.py's comment block for the matching fix.")
