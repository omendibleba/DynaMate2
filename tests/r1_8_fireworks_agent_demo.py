"""
R1-8 demonstration (agent-mediated version): a real, agent-executed dispatch
of a molecular simulation workflow to a real FireWorks LaunchPad, matching
the same standard as the other Representative Example subsections (Section
5.1-5.4) -- natural-language instructions routed through the real
PersistentAgentPoolWithSupervisor, not a direct Python call to the tool
functions.

Supersedes tests/r1_8_fireworks_demo.py, which validated the same two tools
(submit_temperature_sweep_to_fireworks, check_fireworks_status) by calling
them directly in Python with no agent or LLM involved at all -- accurate as
a unit test of the tool functions, but not evidence that DynaMate2 itself
(registration + natural-language routing) can drive this integration. This
script closes that gap: mace_md_specialist registers and calls both tools
itself, in response to natural-language instructions, exactly as a user
would interact with the live framework.

What remains deliberately NOT agent-mediated: actually running the
submitted jobs through a FireWorks worker (`rapidfire`). That's the whole
point of handing a step to a production workflow-management backend --
FireWorks' own worker executes it, independently of DynaMate2 -- so it is
run here as plain infrastructure between two agent-mediated status checks,
not as a tool call.

As run for the manuscript, this executed inside the dynamate2_gpu Apptainer
container (--bind <repo>:/app), against a local, no-root MongoDB (the
portable mongodb-linux-x86_64-rhel90 tarball, `mongod --dbpath ... --port
27117 --bind_ip 127.0.0.1`), with `fireworks`/`pymongo` pip-installed to a
separate --target directory on PYTHONPATH (container filesystem is
read-only, PYTHONNOUSERSITE=1 is set deliberately -- same pattern as
tests/r1_8_fireworks_demo.py and tests/r3_2_md_dft_demo.py). Reuses the
repo's own tutorial_state/ so it extends the already-registered
mace_md_specialist/run_nvt_md from Section 5.1. A my_launchpad.yaml must
exist at /app/my_launchpad.yaml (host/port/name pointing at that mongod).
Real output: see tutorials/sample_outputs/end_to_end_test/r1_8_agent_summary.json.
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
FW_OUT_DIR = os.path.join(OUT_DIR, "r1_8_agent_fireworks_out")
LAUNCHPAD_FILE = "/app/my_launchpad.yaml"

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

# Same fixup as tests/r3_2_md_dft_demo.py: tutorial_state/tools/*.py can hold
# stale pre-fix sources if nothing re-registered it fresh this session.
for _fname in ["ASE_NVT_PBC.py", "smiles_to_xyz.py", "packmol_build_system.py", "plot_nvt_trajectory.py"]:
    print(f"\n[fixup] re-registering from /app/tutorials/{_fname}", flush=True)
    print(pool.register_tool_from_file(f"/app/tutorials/{_fname}"), flush=True)

THREAD = {"configurable": {"thread_id": f"r1-8-fireworks-agent-{int(time.time())}"}}


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


# ── Step 1: register the two FireWorks tools, assign to mace_md_specialist ──
ask(
    "Please register the tools defined in the file /app/tutorials/fireworks_dispatch_tools.py "
    "-- call register_tool_from_file with this exact path now, even if a tool by this name is "
    "already registered, since the file's contents may have changed since it was last registered. "
    "Then assign both submit_temperature_sweep_to_fireworks and check_fireworks_status to "
    "mace_md_specialist. Do not ask for confirmation -- execute all steps immediately."
)
assert "submit_temperature_sweep_to_fireworks" in pool.list_registered_tools()
assert "check_fireworks_status" in pool.list_registered_tools()
if "submit_temperature_sweep_to_fireworks" not in pool.list_agent_tools("mace_md_specialist"):
    ask("Assign the submit_temperature_sweep_to_fireworks tool to mace_md_specialist.")
if "check_fireworks_status" not in pool.list_agent_tools("mace_md_specialist"):
    ask("Assign the check_fireworks_status tool to mace_md_specialist.")
assert "submit_temperature_sweep_to_fireworks" in pool.list_agent_tools("mace_md_specialist")
assert "check_fireworks_status" in pool.list_agent_tools("mace_md_specialist")
print("\nSetup verified: FireWorks tools registered and assigned.", flush=True)

# ── Step 2: agent submits a real two-temperature sweep to a real LaunchPad ──
ask(
    f"Using mace_md_specialist, call submit_temperature_sweep_to_fireworks to submit a "
    f"temperature sweep on the structure file {BOX}, using the MACE polar foundation model "
    f"'polar-1-m', for temperatures_K=[290, 310], n_steps=20, box_size=12.5, "
    f"output_dir='{FW_OUT_DIR}', and launchpad_file='{LAUNCHPAD_FILE}'. Report the fw_ids and "
    f"workflow id returned."
)

# ── Step 3: agent checks status immediately (ground truth read directly too) ──
ask(
    "Using mace_md_specialist, call check_fireworks_status on the fw_ids from the submission you "
    "just made, and report the full status for each one."
)

sys.path.insert(0, "/scratchpad/pydeps")
from fireworks import LaunchPad  # noqa: E402

launchpad = LaunchPad.from_file(LAUNCHPAD_FILE)
fw_ids_ground_truth = sorted(launchpad.get_fw_ids())
status_before = {
    fw_id: launchpad.get_fw_by_id(fw_id).state for fw_id in fw_ids_ground_truth
}
print(f"\n[ground truth] fw_ids={fw_ids_ground_truth} status_before={status_before}", flush=True)

# ── Step 4: NOT agent-mediated -- a real FireWorks worker executes the jobs,
#    exactly as a production deployment would (this is the whole point of
#    handing the step to FireWorks rather than running it inline). ──
from fireworks.core.fworker import FWorker  # noqa: E402
from fireworks.core.rocket_launcher import rapidfire  # noqa: E402

print("\n=== Running the submitted jobs through a real FireWorks worker (rapidfire) ===", flush=True)
rapidfire(launchpad, fworker=FWorker(), nlaunches=0, sleep_time=1)

status_after_ground_truth = {
    fw_id: launchpad.get_fw_by_id(fw_id).state for fw_id in fw_ids_ground_truth
}
print(f"\n[ground truth] status_after={status_after_ground_truth}", flush=True)

# ── Step 5: agent checks status again, in the same thread, after the real
#    worker has finished -- this is the genuinely agent-mediated evidence
#    that DynaMate2 can report a production backend's job state, both
#    before and after execution. ──
ask(
    "Using mace_md_specialist, call check_fireworks_status again on the same fw_ids from "
    "earlier in this conversation, and report the full status for each one now."
)

summary = {
    "fw_ids": fw_ids_ground_truth,
    "status_before_worker": status_before,
    "status_after_worker": status_after_ground_truth,
}
with open(os.path.join(OUT_DIR, "r1_8_agent_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved summary to r1_8_agent_summary.json", flush=True)
print(json.dumps(summary, indent=2), flush=True)
