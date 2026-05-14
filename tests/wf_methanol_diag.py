#!/usr/bin/env python3
"""
wf_methanol_diag.py — four-level diagnostic for the methanol NVT workflow.

Level 0 : prerequisites — do all required files exist and are they readable?
Level 1 : direct run_nvt_md call  (1 step, fresh output path)
Level 2 : direct run_nvt_md call  (100 steps, same params as workflow)
Level 3 : full supervisor pipeline (fresh thread, mirrors workflow button)

Run with:
  /groups/ycolon/group-envs/agentic-tutorials/bin/python tests/wf_methanol_diag.py \
      2>&1 | tee /tmp/wf_methanol_diag.log
"""

import os
import sys
import traceback

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTORIALS = os.path.join(ROOT, "tutorials")
STATE_DIR = os.path.join(ROOT, "ui_state")

MODEL_PATH  = os.path.join(TUTORIALS, "models", "mace-mp-0b3-medium.model")
STRUCT_FILE = os.path.join(TUTORIALS, "methanol_box.xyz")
TRAJ_QUICK  = os.path.join(TUTORIALS, "diag_methanol_quick.traj")   # Level 1
TRAJ_FULL   = os.path.join(TUTORIALS, "diag_methanol_full.traj")    # Level 2
TRAJ_PIPE   = os.path.join(TUTORIALS, "diag_methanol_pipe.traj")    # Level 3
BOX_SIZE    = 15.0
TEMP_K      = 300.0
TRAJ_INTV   = 10
LOG_FILE    = os.path.join(TUTORIALS, "diag_methanol_nvt.log")

sys.path.insert(0, ROOT)

PASS = "PASS ✓"
FAIL = "FAIL ✗"

# ─────────────────────────────────────────────────────────────────────────────
# Level 0 — prerequisites
# ─────────────────────────────────────────────────────────────────────────────

def level0():
    print("\n" + "="*60)
    print("LEVEL 0 — Prerequisites")
    print("="*60)
    ok = True

    checks = [
        (MODEL_PATH,  "MACE model"),
        (STRUCT_FILE, "methanol_box.xyz"),
    ]
    for path, label in checks:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        status = PASS if exists and size > 0 else FAIL
        print(f"  {status}  {label}: {path} ({size:,} bytes)")
        if status == FAIL:
            ok = False

    # Quick ASE read
    try:
        from ase.io import read
        atoms = read(STRUCT_FILE)
        print(f"  {PASS}  ASE read: {len(atoms)} atoms, species={set(atoms.get_chemical_symbols())}")
    except Exception as e:
        print(f"  {FAIL}  ASE read failed: {e}")
        ok = False

    return ok

# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — direct tool call, 1 step
# ─────────────────────────────────────────────────────────────────────────────

def level1():
    print("\n" + "="*60)
    print("LEVEL 1 — Direct run_nvt_md (1 step, quick smoke-test)")
    print("="*60)

    # Remove stale file
    if os.path.exists(TRAJ_QUICK):
        os.remove(TRAJ_QUICK)
        print(f"  Removed existing {TRAJ_QUICK}")

    sys.path.insert(0, STATE_DIR)
    try:
        from tools.run_nvt_md import run_nvt_md
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_nvt_md",
            os.path.join(STATE_DIR, "tools", "run_nvt_md.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        run_nvt_md = mod.run_nvt_md

    try:
        result = run_nvt_md(
            model_path     = MODEL_PATH,
            structure_file = STRUCT_FILE,
            box_size       = BOX_SIZE,
            temperature_K  = TEMP_K,
            n_steps        = 1,
            output_traj    = TRAJ_QUICK,
            traj_interval  = 1,
            log_interval   = 1,
            log_file       = LOG_FILE,
            device         = "cuda",
        )
        size = os.path.getsize(TRAJ_QUICK) if os.path.exists(TRAJ_QUICK) else 0
        if size > 0:
            print(f"  {PASS}  Tool returned: {result}")
            print(f"  {PASS}  File created: {TRAJ_QUICK} ({size:,} bytes)")
            return True
        else:
            print(f"  {FAIL}  Tool returned: {result}")
            print(f"  {FAIL}  File empty or missing")
            return False
    except Exception as e:
        print(f"  {FAIL}  Exception: {e}")
        traceback.print_exc()
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — direct tool call, 100 steps (workflow parameters)
# ─────────────────────────────────────────────────────────────────────────────

def level2():
    print("\n" + "="*60)
    print("LEVEL 2 — Direct run_nvt_md (100 steps, workflow params)")
    print("="*60)

    if os.path.exists(TRAJ_FULL):
        os.remove(TRAJ_FULL)
        print(f"  Removed existing {TRAJ_FULL}")

    sys.path.insert(0, STATE_DIR)
    try:
        from tools.run_nvt_md import run_nvt_md
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_nvt_md",
            os.path.join(STATE_DIR, "tools", "run_nvt_md.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        run_nvt_md = mod.run_nvt_md

    try:
        result = run_nvt_md(
            model_path     = MODEL_PATH,
            structure_file = STRUCT_FILE,
            box_size       = BOX_SIZE,
            temperature_K  = TEMP_K,
            n_steps        = 100,
            output_traj    = TRAJ_FULL,
            traj_interval  = TRAJ_INTV,
            log_interval   = TRAJ_INTV,
            log_file       = LOG_FILE,
            device         = "cuda",
        )
        size = os.path.getsize(TRAJ_FULL) if os.path.exists(TRAJ_FULL) else 0
        if size > 0:
            print(f"  {PASS}  Tool returned: {result}")
            print(f"  {PASS}  File created: {TRAJ_FULL} ({size:,} bytes)")
            return True
        else:
            print(f"  {FAIL}  Tool returned: {result}")
            print(f"  {FAIL}  File empty or missing")
            return False
    except Exception as e:
        print(f"  {FAIL}  Exception: {e}")
        traceback.print_exc()
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — full supervisor pipeline (fresh thread)
# ─────────────────────────────────────────────────────────────────────────────

def level3():
    print("\n" + "="*60)
    print("LEVEL 3 — Full supervisor pipeline (fresh thread)")
    print("="*60)

    if os.path.exists(TRAJ_PIPE):
        os.remove(TRAJ_PIPE)
        print(f"  Removed existing {TRAJ_PIPE}")

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    from langchain_openai import ChatOpenAI
    from dynamate import (
        PersistentAgentPoolWithSupervisor,
        PersistentSaver,
        PoolStore,
        build_tool_manager_v2,
    )

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    saver = PersistentSaver(os.path.join(STATE_DIR, "conversations.db"))
    pool_store = PoolStore(os.path.join(STATE_DIR, "pool_state.json"))

    from langchain_community.tools import ShellTool
    pool = PersistentAgentPoolWithSupervisor(
        supervisor_model  = model,
        pool_store        = pool_store,
        supervisor_prompt = (
            "You are the Supervisor managing a pool of agents.\n"
            "Route domain tasks (download, simulate, generate) to the specialist agent "
            "that owns the relevant tool. Copy ALL parameters verbatim into the handoff message."
        ),
        checkpointer = saver,
    )
    pool.add_agent("shell_agent", model, base_tools=[ShellTool()],
                   system_prompt="You are a shell agent.", _is_dynamic=False)
    pool.add_agent("compute_agent", model, base_tools=[],
                   system_prompt="You are a computation agent.", _is_dynamic=False)
    tool_manager = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tool_manager])

    pool.restore_state(model_factory=lambda name: ChatOpenAI(model=name, temperature=0.0))
    _STATIC = {"shell_agent", "compute_agent"}
    for _name in list(pool._agents):
        if _name not in _STATIC:
            pool._rebuild_agent(_name)
    pool._rebuild_supervisor()

    print(f"  Agents: {pool.list_agents()}")
    print(f"  Registered tools: {pool.list_registered_tools()}")

    import uuid
    tid = f"diag-wf-{uuid.uuid4().hex[:8]}"

    query = (
        f"Please run a 100-step NVT MD simulation of methanol using run_nvt_md.\n"
        f"model_path     = {MODEL_PATH}\n"
        f"structure_file = {STRUCT_FILE}\n"
        f"box_size        = {BOX_SIZE}\n"
        f"temperature_K   = {TEMP_K}\n"
        f"n_steps         = 100\n"
        f"traj_interval   = {TRAJ_INTV}\n"
        f"output_traj    = {TRAJ_PIPE}\n"
        f"Execute run_nvt_md immediately with these parameters."
    )
    print(f"\n  Thread: {tid}")
    print(f"  Query: {query[:120]}...")

    tool_calls = 0
    final_answer = ""
    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": tid}},
            recursion_limit=80,
        ):
            for node, data in chunk.items():
                msgs = data.get("messages", []) if isinstance(data, dict) else []
                for m in msgs:
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            print(f"  [tool_call] {node} → {tc['name']}")
                            tool_calls += 1
                    if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip():
                        final_answer = m.content.strip()
    except Exception as e:
        print(f"  {FAIL}  Pipeline exception: {e}")
        traceback.print_exc()
        return False

    size = os.path.getsize(TRAJ_PIPE) if os.path.exists(TRAJ_PIPE) else 0
    print(f"\n  Tool calls   : {tool_calls}")
    print(f"  File created : {os.path.exists(TRAJ_PIPE)} ({size:,} bytes)")
    print(f"  Final answer : {final_answer[:200]}")

    if size > 0:
        print(f"  {PASS}  Pipeline produced a non-empty trajectory")
        return True
    else:
        print(f"  {FAIL}  Trajectory empty or missing after pipeline")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    results["L0"] = level0()
    if not results["L0"]:
        print("\nLevel 0 failed — fix prerequisites before running higher levels.")
        sys.exit(1)

    results["L1"] = level1()
    if not results["L1"]:
        print("\nLevel 1 failed — run_nvt_md crashes even directly. Fix the tool first.")
        sys.exit(1)

    results["L2"] = level2()
    if not results["L2"]:
        print("\nLevel 2 failed — tool works for 1 step but fails at 100 steps.")
        sys.exit(1)

    results["L3"] = level3()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for lvl, ok in results.items():
        print(f"  {lvl}: {PASS if ok else FAIL}")
