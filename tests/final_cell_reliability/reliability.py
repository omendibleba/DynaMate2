"""
Reliability of the tutorial notebook's final integration cell (Table 3 / Figure 6 workflow).

One run = fresh pool state, the five tools registered and assigned to mace_md_specialist (the state the notebook
reaches before its final cell), then the notebook's exact query through the PromptEnhancer and the supervisor.
Success is decided ONLY by the criteria below, which were fixed before any run.

usage: python reliability.py <model> <run_dir>
"""
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")

MODEL_NAME, RUN = sys.argv[1], sys.argv[2]
STATE_DIR, OUT = os.path.join(RUN, "state"), os.path.join(RUN, "out")
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

from dotenv import dotenv_values
for k, v in dotenv_values("/app/.env").items():
    if v and k.strip() not in os.environ:
        os.environ[k.strip()] = v.strip()

from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
from dynamate import (PersistentAgentPoolWithSupervisor, PersistentSaver, PoolStore,
                      PromptEnhancer, build_tool_manager_v2)

model = ChatOpenAI(model=MODEL_NAME, temperature=0)

SUPERVISOR_PROMPT = '''You are the Supervisor managing a pool of agents.
- tool_manager              : registers tools, assigns them to agents, and adds/removes agents.
- shell_agent               : runs shell commands and handles file-system tasks.
- compute_agent             : performs calculations with its dynamically assigned tools.

Routing rules:
  * Add/register/assign/remove/list tools or agents -> tool_manager.
  * Python code (def statements) + add/register intent -> tool_manager.
  * Shell or file-system tasks -> shell_agent.
  * Domain tasks (download, simulate, generate, compute, create files) ->
    the specialist agent that owns the relevant tool. Do NOT route these
    to tool_manager — tool_manager only manages the pool, it cannot execute
    domain work.
  * If no specialist exists for the task, ask tool_manager to create one first.

Execution rules:
  * If you have all you need execute tasks immediately.
  * When a specialist agent completes a calculation, report the full numerical
    result directly. Do not say "the agent is ready" or ask what to do next.
  * Assign work to one agent at a time.'''

saver = PersistentSaver(os.path.join(STATE_DIR, "conversations.db"))
pool_store = PoolStore(os.path.join(STATE_DIR, "pool_state.json"))
pool = PersistentAgentPoolWithSupervisor(supervisor_model=model, pool_store=pool_store,
                                         supervisor_prompt=SUPERVISOR_PROMPT, checkpointer=saver)
pool.add_agent(name="shell_agent", model=model, base_tools=[ShellTool()],
               system_prompt="You are a shell agent. Execute shell commands to answer requests.", _is_dynamic=False)
pool.add_agent(name="compute_agent", model=model, base_tools=[],
               system_prompt="You are a computation agent. Use your dynamically assigned tools to answer requests. Always call your tools and return the numerical result.",
               _is_dynamic=False)
tool_manager = build_tool_manager_v2(pool, model)
pool.set_system_agents([tool_manager])
enhancer = PromptEnhancer(model=model, pool=pool)

RULE = ("\nYour FIRST action must ALWAYS be to call one of your assigned tools."
        "\nDo NOT call transfer_back_to_supervisor until AFTER a tool has returned a result."
        "\nDo not ask for confirmation — call the tool immediately and return its output.")
FILES = ["download_mace_model.py", "smiles_to_xyz.py", "packmol_build_system.py", "ASE_NVT_PBC.py", "plot_nvt_trajectory.py"]
for f in FILES:
    pool.register_tool_from_file(f"/app/tutorials/{f}")
pool.add_agent(name="mace_md_specialist", model=model, base_tools=[],
               system_prompt="You are a dedicated specialist in MACE and molecular dynamics simulations." + RULE)
for t in ("download_mace_model", "smiles_to_xyz", "packmol_build_system", "run_nvt_md", "plot_nvt_trajectory"):
    pool.assign_tool(t, "mace_md_specialist")
print("Model:", MODEL_NAME, "| agents:", pool.list_agents(), "| tools:", pool.list_registered_tools(), flush=True)

W, NA, CL = (os.path.join(OUT, n) for n in ("water.xyz", "na.xyz", "cl.xyz"))
BOX, TRAJ, LOG, PNG = (os.path.join(OUT, n) for n in ("nacl_water_box.xyz", "nacl_water_nvt.traj", "nacl_water_nvt.log", "nacl_water_nvt_analysis.png"))
QUERY = (
    "Run a complete molecular dynamics study of a NaCl aqueous solution from scratch, "
    "using the MACE polar foundation model 'polar-1-m' -- run_nvt_md loads it by name "
    "itself, so do NOT call download_mace_model and do NOT look for a local model file. "
    f"Convert the water SMILES (O) to a 3D structure and save it to {W}. "
    f"Convert the Na ion SMILES ([Na+]) to a 3D structure and save it to {NA}. "
    f"Convert the Cl ion SMILES ([Cl-]) to a 3D structure and save it to {CL}. "
    f"Build a periodic cubic box of 12.5 Angstrom containing 64 water molecules, "
    f"1 Na ion, and 1 Cl ion, and save it to {BOX}. "
    "Then run an NPT equilibration so the box density can relax: a temperature of 300 K, "
    "a pressure of 1 bar (pressure_bar=1.0), 50 geometry-optimization steps first "
    "(minimize_steps=50), and 100 MD steps, saving a trajectory frame every step "
    f"(traj_interval=1). Save the trajectory to {TRAJ} and the log file to {LOG}. "
    "Finally, plot the potential and total energies and temperature over time "
    f"and save the figure as a PNG to {PNG}."
)
THREAD = {"configurable": {"thread_id": f"reliability-{int(time.time())}"}}
enhanced = enhancer.enhance(QUERY)
print("[enhancer] addition:", repr(enhanced[len(QUERY):] if enhanced.startswith(QUERY) else enhanced[-200:]), flush=True)

trace, seen = [], set()
t0 = time.time()
MAX_TOOL_RESULTS, MAX_WALL = 15, 600
n_results = 0


def evaluate(err):
    import numpy as np
    from ase.io import read
    calls = [dict(name=c["name"], args=c["args"]) for m in trace for c in m["tool_calls"]]
    by = lambda n: [c["args"] for c in calls if c["name"] == n]
    why = []
    if err:
        why.append(err[:120])
    # C1 files
    for n, p in (("water", W), ("na", NA), ("cl", CL), ("box", BOX), ("traj", TRAJ), ("log", LOG), ("png", PNG)):
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            why.append(f"missing or empty output: {n}")
    # C2 box composition
    box_ok = None
    try:
        a = read(BOX); box_ok = a.get_chemical_formula() == "H128ClNaO64"
        if not box_ok:
            why.append(f"box formula {a.get_chemical_formula()}")
    except Exception as e:
        why.append(f"box unreadable: {type(e).__name__}")
    # C3 trajectory
    n_frames = dens = None
    try:
        tr = read(TRAJ, index=":"); n_frames = len(tr)
        dens = float(sum(tr[-1].get_masses()) * 1.66053906660 / tr[-1].get_volume())
        if n_frames != 101:
            why.append(f"trajectory has {n_frames} frames, expected 101")
        if not (0.95 <= dens <= 1.10):
            why.append(f"final density {dens:.3f} outside 0.95 to 1.10")
    except Exception as e:
        why.append(f"trajectory unreadable: {type(e).__name__}")
    # C4 tool calls and arguments
    smiles = sorted(str(a.get("smiles")) for a in by("smiles_to_xyz"))
    if smiles != sorted(["O", "[Na+]", "[Cl-]"]):
        why.append(f"smiles_to_xyz calls: {smiles}")
    pk = by("packmol_build_system")
    ok_pk = False
    for a in pk:
        try:
            counts = {os.path.basename(f): int(n) for f, n in zip(a["xyz_files"], a["n_molecules"])}
            ok_pk = ok_pk or (counts == {"water.xyz": 64, "na.xyz": 1, "cl.xyz": 1} and float(a["box_size"]) == 12.5)
        except Exception:
            pass
    if not ok_pk:
        why.append("no correct packmol_build_system call (64 water, 1 Na, 1 Cl, 12.5 A)")
    md = by("run_nvt_md")
    ok_md = any(a.get("n_steps") == 100 and float(a.get("pressure_bar", 0)) == 1.0 and a.get("minimize_steps") == 50
                and float(a.get("temperature_K", 0)) == 300 and a.get("model_name", "polar-1-m") == "polar-1-m" for a in md)
    if not ok_md:
        why.append(f"no correct run_nvt_md call; got {[(a.get('n_steps'), a.get('pressure_bar'), a.get('minimize_steps')) for a in md]}")
    if len(md) != 1:
        why.append(f"run_nvt_md called {len(md)} times")
    if not by("plot_nvt_trajectory"):
        why.append("plot_nvt_trajectory never called")
    # C5 no download
    if by("download_mace_model"):
        why.append("download_mace_model was called")
    res = {"model": MODEL_NAME, "outcome": "SUCCESS" if not why else "FAIL", "reasons": why, "elapsed_s": round(time.time() - t0, 1),
           "n_tool_calls": len([c for c in calls if not c["name"].startswith("transfer")]), "n_frames": n_frames,
           "final_density": dens, "box_formula_ok": box_ok,
           "box_bytes_equal_committed": (open(BOX, "rb").read() == open("/app/tutorials/sample_outputs/end_to_end_test/nacl_water_box.xyz", "rb").read()) if os.path.exists(BOX) else None,
           "tool_sequence": [c["name"] for c in calls if not c["name"].startswith("transfer")], "error": err}
    json.dump(res, open(os.path.join(RUN, "result.json"), "w"), indent=2, default=float)
    print("\n=== RESULT ===", json.dumps({k: res[k] for k in ("outcome", "reasons", "elapsed_s", "n_tool_calls", "n_frames", "final_density")}, default=float), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


evt = open(os.path.join(RUN, "events.jsonl"), "w")
err = None
try:
    for item in pool.supervisor.stream({"messages": [{"role": "user", "content": enhanced}]}, config=THREAD,
                                       recursion_limit=60, subgraphs=True):
        ns, chunk = item if isinstance(item, tuple) and len(item) == 2 else ((), item)
        for node, upd in chunk.items():
            for m in (upd or {}).get("messages", []):
                key = getattr(m, "id", None) or (m.__class__.__name__, getattr(m, "tool_call_id", None), str(m.content)[:80])
                if key in seen:
                    continue
                seen.add(key)
                rec = {"t": round(time.time() - t0, 1), "type": m.__class__.__name__, "name": getattr(m, "name", None),
                       "content": (m.content if isinstance(m.content, str) else str(m.content))[:600],
                       "tool_calls": [{"name": c["name"], "args": c["args"]} for c in (getattr(m, "tool_calls", None) or [])]}
                trace.append(rec); evt.write(json.dumps(rec, default=str) + "\n"); evt.flush()
                for c in rec["tool_calls"]:
                    print(f"[EVT t={rec['t']}] CALL {c['name']}", flush=True)
                if rec["type"] == "ToolMessage" and not (rec["name"] or "").startswith("transfer"):
                    n_results += 1
                    print(f"[EVT t={rec['t']}] RESULT {rec['name']}", flush=True)
        if n_results > MAX_TOOL_RESULTS:
            evaluate(f"TOOL-CALL GUARD: more than {MAX_TOOL_RESULTS} tool results")
        if time.time() - t0 > MAX_WALL:
            evaluate(f"WALL GUARD: exceeded {MAX_WALL} s")
except BaseException as e:
    err = f"{type(e).__name__}: {e}"
    traceback.print_exc()
evaluate(err)
