"""
R3-2 single-instruction trial. Setup (tool registration, two agents) goes through the
framework API; the experiment is ONE natural-language instruction. Success is judged
only from the saved message trace, never from what the LLM says about itself.

usage: python trial.py <model> <run_dir>
"""
import ast
import json
import os
import shutil
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")

MODEL_NAME = sys.argv[1]
RUN = sys.argv[2]
STATE_DIR = os.path.join(RUN, "state")
OUT_DIR = os.path.join(RUN, "out")
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

from dotenv import dotenv_values
for k, v in dotenv_values("/app/.env").items():
    if v and k.strip() not in os.environ:
        os.environ[k.strip()] = v.strip()

from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
from dynamate import (PersistentAgentPoolWithSupervisor, PersistentSaver, PoolStore,
                      PromptEnhancer, build_tool_manager_v2)

BOX = os.path.join(OUT_DIR, "nacl_water_box.xyz")
TRAJ = os.path.join(OUT_DIR, "md.traj")
LOG = os.path.join(OUT_DIR, "md.log")
DFT_JSON = os.path.join(OUT_DIR, "dft_single_point.json")
shutil.copy("/app/tutorials/sample_outputs/end_to_end_test/nacl_water_box.xyz", BOX)

if MODEL_NAME == "gpt-6-luna":
    model = ChatOpenAI(model="gpt-6-luna", temperature=0, reasoning_effort="none")
else:
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)

SUPERVISOR_PROMPT = """You are the Supervisor managing a pool of agents.
- tool_manager              : registers tools, assigns them to agents, and adds/removes agents.
- shell_agent               : runs shell commands and handles file-system tasks.
- compute_agent             : performs calculations with its dynamically assigned tools.

Routing rules:
  * Add/register/assign/remove/list tools or agents -> tool_manager.
  * Python code (def statements) + add/register intent -> tool_manager.
  * Shell or file-system tasks -> shell_agent.
  * Domain tasks (download, simulate, generate, compute, create files) ->
    the specialist agent that owns the relevant tool. Do NOT route these
    to tool_manager -- tool_manager only manages the pool, it cannot execute
    domain work.
  * If no specialist exists for the task, ask tool_manager to create one first.

Execution rules:
  * If you have all you need execute tasks immediately.
  * When a specialist agent completes a calculation, report the full numerical
    result directly. Do not say "the agent is ready" or ask what to do next.
  * Assign work to one agent at a time."""

saver = PersistentSaver(os.path.join(STATE_DIR, "conversations.db"))
pool_store = PoolStore(os.path.join(STATE_DIR, "pool_state.json"))
pool = PersistentAgentPoolWithSupervisor(supervisor_model=model, pool_store=pool_store,
                                         supervisor_prompt=SUPERVISOR_PROMPT, checkpointer=saver)
pool.add_agent(name="shell_agent", model=model, base_tools=[ShellTool()],
               system_prompt="You are a shell agent. Execute shell commands to answer requests.",
               _is_dynamic=False)
pool.add_agent(name="compute_agent", model=model, base_tools=[],
               system_prompt="You are a computation agent. Use your dynamically assigned tools to answer requests. Always call your tools and return the numerical result.",
               _is_dynamic=False)
tool_manager = build_tool_manager_v2(pool, model)
pool.set_system_agents([tool_manager])
enhancer = PromptEnhancer(model=model, pool=pool)

RULE = ("\nYour FIRST action must ALWAYS be to call one of your assigned tools."
        "\nDo NOT call transfer_back_to_supervisor until AFTER a tool has returned a result."
        "\nDo not ask for confirmation — call the tool immediately and return its output.")

# ---- setup through the framework API (not part of the claim being tested) ----
print(pool.register_tool_from_file("/app/tutorials/ASE_NVT_PBC.py"), flush=True)
print(pool.register_tool_from_file("/app/tutorials/branching_md_to_dft_tools.py"), flush=True)
pool.add_agent(name="mace_md_specialist", model=model, base_tools=[],
               system_prompt="You are a dedicated specialist in MACE and molecular dynamics simulations." + RULE)
pool.add_agent(name="dft_specialist", model=model, base_tools=[],
               system_prompt="You are a dedicated specialist in electronic-structure single-point calculations." + RULE)
pool.assign_tool("run_nvt_md", "mace_md_specialist")
pool.assign_tool("check_density_convergence", "mace_md_specialist")
pool.assign_tool("run_dft_single_point", "dft_specialist")
assert "dft_specialist" in pool.list_agents()
print("Model:", MODEL_NAME, "| agents:", pool.list_agents(), flush=True)

# ---- the experiment: ONE instruction (identical in every attempt) ----
QUERY = (
    f"Using mace_md_specialist, run an NPT equilibration with run_nvt_md on the structure file {BOX}, "
    "using the MACE polar foundation model 'polar-1-m' (it loads by name itself, do NOT call "
    "download_mace_model). Use a starting box size of 12.5 Angstrom, temperature 300 K, pressure 1 bar "
    "(pressure_bar=1.0), 50 geometry-optimization steps first (minimize_steps=50), 14 MD steps, and "
    f"save a trajectory frame every step (traj_interval=1). Save the trajectory to {TRAJ} and the log to {LOG}. "
    "Then call check_density_convergence on that trajectory with window_frames=5 and tolerance_percent=0.05. "
    "If it reports that the density has not converged, run 16 more NPT MD steps with run_nvt_md, continuing "
    f"from the trajectory at {TRAJ} as the starting structure (same box size 12.5 Angstrom, temperature 300 K, "
    f"pressure_bar=1.0, minimize_steps=0, traj_interval=1), saving to the same trajectory {TRAJ} (overwrite) and "
    f"log {LOG}, and then check convergence again. Keep extending in 16-step increments and re-checking until "
    "the check reports convergence. Once the density has converged, use dft_specialist to run a single-point "
    "electronic-structure calculation with run_dft_single_point on the final frame (frame_index=-1) of the "
    f"trajectory at {TRAJ}, using the GFN2-xTB method, charge=0, spin_multiplicity=1, and save the result to {DFT_JSON}."
)
THREAD = {"configurable": {"thread_id": f"r3-2-trial-{int(time.time())}"}}
enhanced = enhancer.enhance(QUERY)
print("\n── [enhancer] addition:", repr(enhanced[len(QUERY):] if enhanced.startswith(QUERY) else enhanced[-200:]), flush=True)

trace, error, seen_ids = [], None, set()
t0 = time.time()


def finalize(err):
    sys.path.insert(0, "/mnt/work")
    from evaluate import evaluate
    result = evaluate(trace, MODEL_NAME, RUN, error=err, elapsed=round(time.time() - t0, 1))
    result["final_message"] = next((m["content"] for m in reversed(trace) if m["type"] == "AIMessage" and m["content"]), None)
    json.dump(result, open(os.path.join(RUN, "result.json"), "w"), indent=2, default=float)
    json.dump(trace, open(os.path.join(RUN, "trace.json"), "w"), indent=1, default=str)
    print("\n=== RESULT ===", json.dumps({k: result[k] for k in ("model", "outcome", "reasons", "md_n_steps_per_call",
          "n_checks", "branch_fired", "handoff_to_dft_specialist", "dft_energy_eV", "elapsed_s")}, default=float), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # background agent threads would otherwise keep the process alive

MAX_MD, MAX_WALL = 8, 1500
n_md_results = 0
evt_log = open(os.path.join(RUN, "events.jsonl"), "w")
try:
    for item in pool.supervisor.stream({"messages": [{"role": "user", "content": enhanced}]},
                                       config=THREAD, recursion_limit=120, subgraphs=True):
        ns, chunk = item if isinstance(item, tuple) and len(item) == 2 else ((), item)
        for node, upd in chunk.items():
            for m in (upd or {}).get("messages", []):
                key = getattr(m, "id", None) or (m.__class__.__name__, getattr(m, "tool_call_id", None), str(m.content)[:80])
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                rec = {
                    "t": round(time.time() - t0, 1), "ns": "/".join(ns) if ns else "", "node": node,
                    "type": m.__class__.__name__, "name": getattr(m, "name", None),
                    "content": (m.content if isinstance(m.content, str) else str(m.content)),
                    "tool_calls": [{"name": c["name"], "args": c["args"]} for c in (getattr(m, "tool_calls", None) or [])],
                }
                trace.append(rec)
                evt_log.write(json.dumps(rec, default=str) + "\n"); evt_log.flush()
                for c in rec["tool_calls"]:
                    print(f"[EVT t={rec['t']}] CALL {c['name']} {json.dumps(c['args'], default=str)[:230]}", flush=True)
                if rec["type"] == "ToolMessage":
                    print(f"[EVT t={rec['t']}] RESULT {rec['name']} {rec['content'][:230]!r}", flush=True)
                    if rec["name"] == "run_nvt_md":
                        n_md_results += 1
        if n_md_results > MAX_MD:
            finalize(f"LOOP GUARD: more than {MAX_MD} run_nvt_md results")
        if time.time() - t0 > MAX_WALL:
            finalize(f"WALL GUARD: exceeded {MAX_WALL} s")
except BaseException as e:  # includes GraphRecursionError
    error = f"{type(e).__name__}: {e}"
    traceback.print_exc()
finalize(error)
