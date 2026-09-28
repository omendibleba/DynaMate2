"""
R3-2 demonstration: a real, agent-executed, convergence-gated MD -> DFT
hand-off. mace_md_specialist runs NPT equilibration and checks trailing-
window density convergence via check_density_convergence; while not
converged, the run is genuinely extended (a real numerical branch, not a
fixed step count); once converged, the final configuration is handed off to
a newly created dft_specialist agent for a GFN2-xTB single-point energy via
run_dft_single_point.

The MD/convergence-check loop is orchestrated here by reading the real
check_density_convergence result after each agent-executed check and issuing
the next natural-language instruction accordingly -- the same class of
light external control flow already used elsewhere in this notebook/session
(e.g. T4's fallback assignment step) for reliability, while every actual
tool call (MD run, convergence check, DFT single point) is executed by the
real agent and appears in the real routing trace below.

As run for the manuscript, this executed inside the dynamate2_gpu Apptainer
container (--bind <repo>:/app), with `tblite[ase]` (GFN2-xTB) pip-installed
to a separate --target directory on PYTHONPATH (container filesystem is
read-only, PYTHONNOUSERSITE=1 is set deliberately -- same pattern as
tests/r1_8_fireworks_demo.py). Reuses the repo's own tutorial_state/ so it
extends the already-registered mace_md_specialist/run_nvt_md from Section
4.1, matching the manuscript's "extends the workflow of Section 4.1" framing.
Real output: see tutorials/sample_outputs/end_to_end_test/r3_2_summary.json.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/app")

from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
    pretty_print_messages,
)

STATE_DIR = "/app/tutorials/tutorial_state"
OUT_DIR = "/app/tutorials/sample_outputs/end_to_end_test"
BOX = os.path.join(OUT_DIR, "nacl_water_box.xyz")

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

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
saver = PersistentSaver(os.path.join(STATE_DIR, "conversations.db"))
pool_store = PoolStore(os.path.join(STATE_DIR, "pool_state.json"))

pool = PersistentAgentPoolWithSupervisor(
    supervisor_model=model,
    pool_store=pool_store,
    supervisor_prompt=SUPERVISOR_PROMPT,
    checkpointer=saver,
)
pool.add_agent(
    name="shell_agent", model=model, base_tools=[ShellTool()],
    system_prompt="You are a shell agent. Execute shell commands to answer requests.",
    _is_dynamic=False,
)
pool.add_agent(
    name="compute_agent", model=model, base_tools=[],
    system_prompt="You are a computation agent. Use your dynamically assigned tools to answer requests. Always call your tools and return the numerical result.",
    _is_dynamic=False,
)
tool_manager = build_tool_manager_v2(pool, model)
pool.set_system_agents([tool_manager])
model_factory = lambda name: ChatOpenAI(model=name, temperature=0.0)
pool.restore_state(model_factory)
enhancer = PromptEnhancer(model=model, pool=pool)

print("Agents:", pool.list_agents(), flush=True)
print("Tools :", pool.list_registered_tools(), flush=True)

# tutorial_state/tools/*.py was found stale -- reverted to the pre-this-session
# (June 8) versions of run_nvt_md/smiles_to_xyz/packmol_build_system/
# plot_nvt_trajectory by earlier `git checkout --` calls this session that
# assumed this directory was safe-to-discard "routine drift". It is NOT
# safe when nothing re-registers fresh afterward (unlike the notebook, which
# re-registers everything itself every full run). Force back to the real,
# fixed source files before doing anything else.
for _fname in ["ASE_NVT_PBC.py", "smiles_to_xyz.py", "packmol_build_system.py", "plot_nvt_trajectory.py"]:
    print(f"\n[fixup] re-registering from /app/tutorials/{_fname}", flush=True)
    print(pool.register_tool_from_file(f"/app/tutorials/{_fname}"), flush=True)

THREAD = {"configurable": {"thread_id": f"r3-2-md-dft-{int(time.time())}"}}


def ask(query, recursion_limit=15):
    print("\n" + "=" * 70, flush=True)
    print("[user] " + query[:300], flush=True)
    print("=" * 70, flush=True)
    enhanced = enhancer.enhance(query)
    print("── [enhancer] " + "─" * 44, flush=True)
    print(enhanced, flush=True)
    print("── [supervisor] " + "─" * 42, flush=True)
    for chunk in pool.supervisor.stream(
        {"messages": [{"role": "user", "content": enhanced}]},
        config=THREAD,
        recursion_limit=recursion_limit,
    ):
        pretty_print_messages(chunk, last_message=True)


# ── Step 1: register the two new tools, create dft_specialist ──────────────
ask(
    "Please re-register the tools defined in the file /app/tutorials/branching_md_to_dft_tools.py "
    "-- call register_tool_from_file with this exact path now, even if a tool by this name is "
    "already registered, since the file's contents may have changed since it was last registered. "
    "Do not skip this step just because the tool already exists. "
    "Then assign the check_density_convergence tool to mace_md_specialist. "
    "Then create a new agent named dft_specialist -- a dedicated specialist in electronic-structure "
    "single-point calculations -- and assign it the run_dft_single_point tool. "
    "Do not ask for confirmation -- execute all steps immediately."
)
assert "check_density_convergence" in pool.list_registered_tools()
assert "run_dft_single_point" in pool.list_registered_tools()
if "check_density_convergence" not in pool.list_agent_tools("mace_md_specialist"):
    ask("Assign the check_density_convergence tool to mace_md_specialist.")
if "dft_specialist" not in pool.list_agents():
    ask("Create a new agent named dft_specialist, a specialist in electronic-structure single-point calculations.")
if "run_dft_single_point" not in pool.list_agent_tools("dft_specialist"):
    ask("Assign the run_dft_single_point tool to dft_specialist.")

assert "check_density_convergence" in pool.list_agent_tools("mace_md_specialist")
assert "dft_specialist" in pool.list_agents()
assert "run_dft_single_point" in pool.list_agent_tools("dft_specialist")
print("\nSetup verified: tools registered, dft_specialist created.", flush=True)

# ── Step 2: initial short NPT segment ───────────────────────────────────────
TRAJ = os.path.join(OUT_DIR, "r3_2_md.traj")
LOG = os.path.join(OUT_DIR, "r3_2_md.log")
INITIAL_STEPS = 14
ask(
    f"Using mace_md_specialist, run an NPT equilibration with run_nvt_md on the structure file "
    f"{BOX}, using the MACE polar foundation model 'polar-1-m' (loads by name itself, do NOT call "
    f"download_mace_model). Use a starting box size of 12.5 Angstrom, temperature 300 K, pressure "
    f"1 bar (pressure_bar=1.0), 50 geometry-optimization steps first (minimize_steps=50), "
    f"{INITIAL_STEPS} MD steps, and save a trajectory frame every step (traj_interval=1). Save "
    f"the trajectory to {TRAJ} and the log to {LOG}."
)

# ── Step 3: convergence-gated loop -- real branching, driven by real results ──
CHECK_KW = dict(window_frames=5, tolerance_percent=0.05)
total_steps = INITIAL_STEPS
converged = False
max_extensions = 5
extension_log = []

sys.path.insert(0, "/app/tutorials")
from branching_md_to_dft_tools import check_density_convergence  # noqa: E402

for attempt in range(max_extensions + 1):
    ask(
        f"Using mace_md_specialist, call check_density_convergence on the trajectory file {TRAJ} "
        f"with window_frames={CHECK_KW['window_frames']} and "
        f"tolerance_percent={CHECK_KW['tolerance_percent']}, and report the full result."
    )
    result = check_density_convergence(TRAJ, **CHECK_KW)
    print(f"\n[real check_density_convergence result] {json.dumps(result, default=float)}", flush=True)
    extension_log.append({"total_steps": total_steps, **{k: (float(v) if v is not None else v) for k, v in result.items() if k != "recommendation"}, "recommendation": result["recommendation"]})

    if result["converged"]:
        converged = True
        break
    if attempt == max_extensions:
        print("\nReached max_extensions without convergence -- stopping loop.", flush=True)
        break

    extend_by = 16
    total_steps += extend_by
    ask(
        f"The density has not converged yet. Using mace_md_specialist, run {extend_by} more NPT "
        f"MD steps with run_nvt_md, continuing from the trajectory at {TRAJ} as the starting "
        f"structure (use the same box size 12.5 Angstrom, temperature 300 K, pressure_bar=1.0, "
        f"minimize_steps=0 this time since it's already equilibrated, traj_interval=1). Save the "
        f"new trajectory to {TRAJ} (overwrite) and the log to {LOG}."
    )

print(f"\n\n=== Convergence loop finished: converged={converged}, total_steps={total_steps} ===", flush=True)
print(json.dumps(extension_log, indent=2), flush=True)

# ── Step 4: hand off to dft_specialist ──────────────────────────────────────
DFT_JSON = os.path.join(OUT_DIR, "r3_2_dft_single_point.json")
ask(
    f"The equilibration has converged. Using dft_specialist, run a single-point electronic-"
    f"structure calculation with run_dft_single_point on the final frame (frame_index=-1) of the "
    f"trajectory at {TRAJ}, using the GFN2-xTB method, charge=0, spin_multiplicity=1. Save the "
    f"result to {DFT_JSON}."
)

dft_result = json.load(open(DFT_JSON)) if os.path.exists(DFT_JSON) else None
print("\n\n=== FINAL DFT RESULT ===", flush=True)
print(json.dumps(dft_result, indent=2), flush=True)

summary = {
    "initial_steps": INITIAL_STEPS,
    "total_md_steps": total_steps,
    "convergence_check_params": CHECK_KW,
    "extension_log": extension_log,
    "converged": converged,
    "dft_result": dft_result,
}
with open("/app/tutorials/sample_outputs/end_to_end_test/r3_2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary to r3_2_summary.json", flush=True)
