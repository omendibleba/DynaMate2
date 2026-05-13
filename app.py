#!/usr/bin/env python3
"""
DynaMate2 — Gradio Chat UI
───────────────────────────────────────────────────────────────────────────────
Production chat interface for the DynaMate2 multi-agent molecular simulation
assistant. Users interact with the supervisor architecture through plain
natural-language prompts; no notebooks or scripting required.

QUICK START
───────────
  pip install gradio
  cd /path/to/DynaMate2_prod
  python app.py          # http://localhost:8888

STATE
─────
  ui_state/conversations.db  — full conversation history (SQLite)
  ui_state/pool_state.json   — agent definitions and tool assignments
  ui_state/tools/            — one .py file per registered tool
  ui_state/threads.json      — thread metadata for the history dropdown
  ui_state/uploads/          — user-uploaded Python tool scripts
"""

import json
import os
import shutil
import sys
from datetime import datetime

import dotenv
import gradio as gr
from langchain_community.tools import ShellTool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
)

dotenv.load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────────

_ROOT       = os.path.dirname(os.path.abspath(__file__))
_STATE_DIR  = os.path.join(_ROOT, "ui_state")
_UPLOADS    = os.path.join(_STATE_DIR, "uploads")
_THREADS_DB = os.path.join(_STATE_DIR, "threads.json")
_TUTORIALS  = os.path.join(_ROOT, "tutorials")
_MODEL_NAME = os.getenv("DYNAMATE_MODEL", "gpt-4.1-mini")


def _tut(relpath: str) -> str:
    """Absolute path inside tutorials/."""
    return os.path.join(_TUTORIALS, relpath)


def _read_tool_code(filename: str) -> str:
    """Read a tool .py file from tutorials/ and return its source."""
    with open(_tut(filename)) as f:
        return f.read()

# ── Supervisor prompt ────────────────────────────────────────────────────────────

_SUPERVISOR_PROMPT = (
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
    "Handoff rules:\n"
    "  * When routing to a specialist, copy ALL specific parameters from the user\n"
    "    request verbatim into the handoff message: exact file paths, numerical values,\n"
    "    output locations. Do not paraphrase — specialists need exact values to call\n"
    "    their tools correctly.\n\n"
    "Execution rules:\n"
    "  * If you have all you need execute tasks immediately.\n"
    "  * When a specialist agent completes a calculation, report the full numerical\n"
    "    result directly. Do not say 'the agent is ready' or ask what to do next.\n"
    "  * Assign work to one agent at a time."
)

# ── Notebook-faithful prompts (paper_tests_7) ────────────────────────────────────
# Code is read from .py files in tutorials/ so the prompts stay in sync with
# the files rather than being hardcoded duplicates.

_DOWNLOAD_CODE = _read_tool_code("download_mace_model.py")
_SMILES_CODE   = _read_tool_code("smiles_to_xyz.py")
_PACKMOL_CODE  = _read_tool_code("packmol_build_system.py")

# ── T1a: Register download_mace_model from inline code ─────────────────────────
# Mirrors paper_tests_7 cell 10 — user_prompt_A
PROMPT_T1A = (
    "\n"
    "    I have a Python function that can download MACE machine learning potential \n"
    "    models by name — it knows the download URLs for several standard models \n"
    "    (MACE-MP-0b3, MACE-MPA-0, etc.) and skips re-downloading if the file \n"
    "    already exists. Please add it to the system so I can use it later.\n\n"
    "\n"
    + _DOWNLOAD_CODE
)

# ── T1b: Register smiles_to_xyz + packmol_build_system from inline code ─────────
# Mirrors paper_tests_7 cell 12 — user_prompt_B
PROMPT_T1B = (
    "\n"
    "    Here are two functions I would like to add to the system.\n"
    "    The first converts a SMILES string to a 3D XYZ file using RDKit.\n"
    "    The second builds a periodic molecular simulation box using Packmol —\n"
    "    it takes one or more XYZ files and places copies of the molecules inside\n"
    "    a cubic box of a given size. Please register both so I can use them later.\n\n"
    "\n"
    + _SMILES_CODE + "\n" + _PACKMOL_CODE
)

# ── T1c: Create mace_md_specialist ──────────────────────────────────────────────
# Mirrors paper_tests_7 cell 14 — user_prompt_C
PROMPT_T1C = (
    "\n"
    "I need a dedicated specialist in MACE and molecular dynamics simulations.\n"
    "Please create one and give it the three tools I just added.\n\n"
    "\n"
)

# ── T1_run: Download MACE-MP-0b3 model ─────────────────────────────────────────
# Mirrors paper_tests_7 cell 23 — p3
PROMPT_T1_RUN = (
    f"Download the MACE-MP-0b3 machine learning potential and save it "
    f"to {_tut('models')}. Please convert the model to LAMMPS format."
)

# ── T2: Build NaCl + water box ──────────────────────────────────────────────────
# Mirrors paper_tests_7 cell 27 — query_T2
# Notebook uses os.path.join('./', ...) from tutorials/; we use absolute paths.
PROMPT_T2 = (
    "I need a periodic simulation box containing 1 Na(+1), 1 Cl(-1) ions  and 267 water molecules. "
    f"First convert the water SMILES (O) to a 3D XYZ file at {_tut('water.xyz')}, "
    f"and the Na and CL ions with SMILES [Na+], and [Cl-] to {_tut('na.xyz')} and {_tut('cl.xyz')}. "
    "Then use packmol to build a cubic box of 20.0 Angstrom with 267 water molecules "
    f"and 1 NaCl pair, and save the result to {_tut('nacl_water_box.xyz')}."
)

# ── T3a: Register run_nvt_md from .py file ──────────────────────────────────────
# Mirrors paper_tests_7 cell 30 — query_T3_setup
PROMPT_T3A = (
    f"Please register the tools defined in the file {_tut('ASE_NVT_PBC.py')}. "
    "Update it if it already exists. "
    "Then assign the run_nvt_md tool to mace_md_specialist."
)

# ── T3b: Run NVT MD ─────────────────────────────────────────────────────────────
# Mirrors paper_tests_7 cell 35 — query_T3_run
PROMPT_T3B = (
    "Please run a short NVT molecular dynamics simulation using the run_nvt_md tool.\n"
    "The model file is already on disk — do NOT call download_mace_model.\n"
    f"model_path     = {_tut('models/mace-mp-0b3-medium.model')}\n"
    f"structure_file = {_tut('nacl_water_box.xyz')}\n"
    "box_size        = 20.0\n"
    "temperature_K   = 300.0\n"
    "n_steps         = 10\n"
    "traj_interval   = 10\n"
    f"output_traj    = {_tut('nvt_nacl_water.traj')}\n"
    "Execute run_nvt_md immediately with these parameters."
)

# ── T4a: Ask LLM to write, register, and assign plot_nvt_trajectory ─────────────
# Mirrors paper_tests_7 cell 38 — query_T4_register
PROMPT_T4A = (
    "Write, register, and assign a new Python tool called plot_nvt_trajectory.\n\n"
    "The function signature must be:\n"
    "    plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str\n\n"
    "Include a docstring that describes what the function does.\n"
    "All imports (ase, matplotlib, numpy) must be inside the function body.\n\n"
    "The function must:\n"
    "  1. Read an ASE .traj file using ase.io.read with index=':'.\n"
    "  2. Extract per-frame: potential energy (eV) via get_potential_energy(),\n"
    "     total energy (eV) as get_kinetic_energy() + get_potential_energy(),\n"
    "     and temperature (K) via get_temperature().\n"
    "  3. Build a time axis in picoseconds: frame_index * timestep_fs / 1000.\n"
    "  4. Normalize potential and total energy by dividing each by its mean.\n"
    "  5. Create a two-panel matplotlib figure (figsize=(10, 6), sharex=True):\n"
    "       - Top panel: normalized potential energy and normalized total energy vs time (ps).\n"
    "       - Bottom panel: temperature (K) vs time (ps).\n"
    "     Add axis labels, legends, and gridlines.\n"
    "  6. Save the figure to output_png with dpi=150, return output_png.\n\n"
    "After writing the function: register it as a tool, then assign it to mace_md_specialist.\n"
    "Do not ask for confirmation — execute all three steps immediately."
)

# ── T4b: Plot the NVT trajectory ────────────────────────────────────────────────
# Mirrors paper_tests_7 cell 40 — query_T4_run
PROMPT_T4B = (
    f"Plot the NVT trajectory at {_tut('nvt_nacl_water.traj')}. "
    "Use a timestep of 0.5 fs. "
    f"Save the figure to {_tut('nvt_nacl_water_analysis.png')}."
)

# ── Final integration workflow: full MD pipeline from a single prompt ────────────
# Mirrors paper_tests_7 cell 45 — query_final
_wf_mol_xyz  = _tut("methanol.xyz")
_wf_box_xyz  = _tut("methanol_box.xyz")
_wf_traj     = _tut("methanol_nvt.traj")
_wf_png      = _tut("methanol_nvt_analysis.png")
_wf_box_size = 15.0

PROMPT_WORKFLOW_FINAL = (
    "I need a complete NVT molecular dynamics study of liquid methanol from scratch. "
    f"Use the MACE-MP-0b3 force field and save the model to {_tut('models')}. "
    f"Convert the methanol SMILES (CO) to a 3D structure and save it to {_wf_mol_xyz}. "
    f"Build a periodic cubic box of {_wf_box_size} Angstrom containing 50 methanol "
    f"molecules and save it to {_wf_box_xyz}. "
    "Run a 100-step NVT simulation at 300 K with a 0.5 fs timestep using ASE "
    f"and save the trajectory to {_wf_traj}. "
    f"Finally, plot the energy and temperature evolution and save the figure to {_wf_png}."
)

# ── System initialisation ────────────────────────────────────────────────────────

def _build_system() -> tuple:
    os.makedirs(_STATE_DIR, exist_ok=True)
    os.makedirs(_UPLOADS, exist_ok=True)

    model      = ChatOpenAI(model=_MODEL_NAME, temperature=0.0)
    saver      = PersistentSaver(os.path.join(_STATE_DIR, "conversations.db"))
    pool_store = PoolStore(os.path.join(_STATE_DIR, "pool_state.json"))

    pool = PersistentAgentPoolWithSupervisor(
        supervisor_model=model,
        pool_store=pool_store,
        supervisor_prompt=_SUPERVISOR_PROMPT,
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
    tool_manager = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tool_manager])
    pool.restore_state(
        model_factory=lambda name: ChatOpenAI(model=name, temperature=0.0)
    )
    # Mirrors notebook Cell 22: rebuild every dynamic agent so the latest
    # execution rules (from pool._rebuild_agent) take effect after restore.
    _STATIC = {"shell_agent", "compute_agent"}
    for _name in list(pool._agents):
        if _name not in _STATIC:
            pool._rebuild_agent(_name)
    pool._rebuild_supervisor()

    enhancer = PromptEnhancer(model=model, pool=pool)
    return pool, enhancer


print("DynaMate2: initialising system…", flush=True)
pool, enhancer = _build_system()
print("DynaMate2: ready.", flush=True)

# ── Thread helpers ───────────────────────────────────────────────────────────────

def _new_thread_id() -> str:
    return "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_thread_choices() -> list[str]:
    if not os.path.exists(_THREADS_DB):
        return []
    try:
        with open(_THREADS_DB) as f:
            data = json.load(f)
        return [f"{t['id']}  —  {t['preview']}" for t in reversed(data)]
    except Exception:
        return []


def _save_thread(thread_id: str, preview: str) -> None:
    data: list = []
    if os.path.exists(_THREADS_DB):
        try:
            with open(_THREADS_DB) as f:
                data = json.load(f)
        except Exception:
            data = []
    if any(t["id"] == thread_id for t in data):
        return
    data.append({
        "id": thread_id,
        "preview": preview,
        "created_at": datetime.now().isoformat(),
    })
    with open(_THREADS_DB, "w") as f:
        json.dump(data, f, indent=2)

# ── Status helpers ───────────────────────────────────────────────────────────────

def _get_status_html() -> str:
    parts = [
        '<div class="status-panel">',
        '<div class="status-section-title">🤖 Agents</div>',
    ]
    for name in pool.list_agents():
        entry = pool._agents[name]
        base  = [t.name for t in entry.get("base_tools",  [])]
        extra = [t.name for t in entry.get("extra_tools", [])]
        parts.append('<div class="status-agent">')
        parts.append(f'<span class="agent-name">{name}</span>')
        for t in base:
            parts.append(f'<span class="chip chip-base">{t}</span>')
        for t in extra:
            parts.append(f'<span class="chip chip-tool">{t}</span>')
        parts.append('</div>')
    reg = pool.list_registered_tools()
    parts.append('<div class="status-section-title" style="margin-top:0.8rem;">🛠 Tool Registry</div>')
    if reg:
        parts.append('<div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-top:0.2rem;">')
        for t in reg:
            parts.append(f'<span class="chip chip-registry">{t}</span>')
        parts.append('</div>')
    else:
        parts.append('<span class="status-empty">empty — register tools above</span>')
    parts.append('</div>')
    return '\n'.join(parts)

# ── Stream chunk parsing ─────────────────────────────────────────────────────────

def _msg_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        ).strip()
    return str(content).strip()


def _parse_chunk(chunk) -> tuple[str | None, str, bool]:
    if isinstance(chunk, tuple):
        namespace, update = chunk
        prefix = " > ".join(namespace) if namespace else "subgraph"
        if isinstance(update, dict):
            for name, data in update.items():
                msgs = data.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    return f"{prefix}/{name}", _msg_text(last), isinstance(last, AIMessage)
        return prefix, "", False
    if isinstance(chunk, dict):
        for node_name, data in chunk.items():
            msgs = data.get("messages", [])
            if msgs:
                last = msgs[-1]
                return node_name, _msg_text(last), isinstance(last, AIMessage)
    return None, "", False

# ── Response generator ───────────────────────────────────────────────────────────

def respond(message: str, history: list, thread_id: str):
    if not message.strip():
        yield history, "", _get_status_html()
        return

    tid          = thread_id.split("  —")[0].strip()
    enhanced     = enhancer.enhance(message)
    config       = {"configurable": {"thread_id": tid}}
    trace_lines  = [f"[enhancer]  {enhanced}", ""]
    final_answer = ""

    def _build_history(answer: str) -> list:
        return history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": answer},
        ]

    yield _build_history("…"), "\n".join(trace_lines), _get_status_html()

    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": enhanced}]},
            config=config,
            recursion_limit=25,
        ):
            node, content, is_ai = _parse_chunk(chunk)
            if node and content:
                trace_lines.append(f"[{node}]  {content[:300]}")
                if is_ai and content:
                    final_answer = content
            yield _build_history(final_answer or "…"), "\n".join(trace_lines), _get_status_html()
    except Exception as exc:
        trace_lines.append(f"\n[error]  {exc}")
        yield _build_history(f"⚠ Error: {exc}"), "\n".join(trace_lines), _get_status_html()
        return

    final_answer = final_answer or "(No response)"
    _save_thread(tid, message[:60])
    yield _build_history(final_answer), "\n".join(trace_lines), _get_status_html()


def do_new_thread():
    tid     = _new_thread_id()
    choices = _load_thread_choices()
    return tid, gr.Dropdown(choices=choices, value=None)


def select_thread(selection: str | None) -> str:
    if not selection:
        return ""
    return selection.split("  —")[0].strip()


def handle_upload(filepath: str | None) -> tuple[str, str, str]:
    """Copy uploaded file to ui_state/uploads/ and return (prompt, path_display, prompt_preview)."""
    if not filepath:
        return "", "", ""
    filename = os.path.basename(filepath)
    dest = os.path.join(_UPLOADS, filename)
    shutil.copy(filepath, dest)
    prompt = (
        f"Please register the tools defined in the file {dest}. "
        "Update any existing tools with the same name."
    )
    return prompt, dest, prompt

# ── CSS ──────────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Base ── */
.monospace textarea, .monospace input {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    font-size: 0.80rem !important;
    line-height: 1.5 !important;
}
footer { display: none !important; }

/* ── Hero ── */
#hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0e4d6e 100%);
    border-radius: 12px;
    padding: 1.8rem 2.2rem 1.4rem 2.2rem;
    margin-bottom: 0.8rem;
    color: white;
}
#hero h1 {
    font-size: 2.0rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    margin: 0 0 0.2rem 0;
    color: #f0f9ff !important;
}
#hero .subtitle {
    font-size: 1.0rem;
    color: #94d2f7;
    margin: 0 0 0.4rem 0;
    font-weight: 500;
}
#hero .tagline {
    font-size: 0.85rem;
    color: #7ec8e3;
    margin: 0;
    opacity: 0.85;
}
#hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 0.12rem 0.65rem;
    font-size: 0.72rem;
    color: #bae6fd;
    margin-top: 0.55rem;
    margin-right: 0.35rem;
}

/* ── Quick-start accordion headers ── */
#acc-prompt > .label-wrap button,
#acc-script > .label-wrap button,
#acc-llm    > .label-wrap button,
#acc-run    > .label-wrap button {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
}

/* ── Group labels ── */
.group-label {
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
    margin: 0 0 0.5rem 0 !important;
    padding: 0 !important;
}

/* ── Step buttons by category ── */
.btn-prompt button {
    background: #eff6ff !important;
    border: 1.5px solid #bfdbfe !important;
    color: #1d4ed8 !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.btn-prompt button:hover { background: #dbeafe !important; border-color: #93c5fd !important; }

.btn-file button {
    background: #f0fdf4 !important;
    border: 1.5px solid #bbf7d0 !important;
    color: #15803d !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.btn-file button:hover { background: #dcfce7 !important; border-color: #86efac !important; }

.btn-llm button {
    background: #fdf4ff !important;
    border: 1.5px solid #e9d5ff !important;
    color: #7e22ce !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.btn-llm button:hover { background: #f3e8ff !important; }

.btn-run button {
    background: #fff7ed !important;
    border: 1.5px solid #fed7aa !important;
    color: #c2410c !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
.btn-run button:hover { background: #ffedd5 !important; }

/* ── Upload zone ── */
#upload-zone .wrap { border-radius: 10px !important; border: 2px dashed #86efac !important; }

/* ── Upload path display ── */
#upload-path textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #15803d !important;
    background: #f0fdf4 !important;
}

/* ── System Status panel ── */
.status-panel {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.77rem;
    line-height: 1.6;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    max-height: 340px;
    overflow-y: auto;
}
.status-section-title {
    font-weight: 700;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #475569;
    margin: 0 0 0.35rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #e2e8f0;
}
.status-agent {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.25rem;
    margin-bottom: 0.3rem;
}
.agent-name {
    font-weight: 600;
    color: #1e40af;
    min-width: 130px;
}
.chip {
    display: inline-block;
    border-radius: 10px;
    padding: 0.04rem 0.45rem;
    font-size: 0.67rem;
    font-weight: 600;
    white-space: nowrap;
}
.chip-base     { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.chip-tool     { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
.chip-registry { background: #fdf4ff; color: #7e22ce; border: 1px solid #e9d5ff; }
.status-empty  { color: #94a3b8; font-style: italic; font-size: 0.75rem; }

/* ── Chat input ── */
#send-row textarea {
    border-radius: 8px !important;
    border: 1.5px solid #cbd5e1 !important;
    font-size: 0.9rem !important;
}
#send-row textarea:focus { border-color: #3b82f6 !important; }
"""

# ── Gradio Blocks UI ─────────────────────────────────────────────────────────────

with gr.Blocks(title="DynaMate2") as demo:

    # ── Hero ──────────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="hero">
      <h1>⚗️ DynaMate2</h1>
      <p class="subtitle">Multi-Agent Molecular Simulation Assistant</p>
      <p class="tagline">
        Describe your simulation task in plain language — DynaMate2 routes it
        through a pool of specialist agents, registers tools on the fly, and
        returns full numerical results without any scripting.
      </p>
      <span class="badge">MACE-MP-0b3</span>
      <span class="badge">ASE MD</span>
      <span class="badge">Packmol</span>
      <span class="badge">RDKit</span>
      <span class="badge">LangGraph</span>
    </div>
    """)

    # ── Quick-start tabs ───────────────────────────────────────────────────────────
    # ── Section 1: From Prompt ────────────────────────────────────────────────────
    with gr.Accordion("📋  From Prompt", open=True, elem_id="acc-prompt"):
        gr.HTML(
            '<p class="group-label">Register tools by pasting Python code inline '
            '— mirrors paper_tests_7 cells 10 · 12 · 14</p>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                t1a_btn = gr.Button(
                    "① Register download_mace_model",
                    elem_classes=["btn-prompt"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'Sends full function source as part of the prompt</p>')
            with gr.Column(scale=1):
                t1b_btn = gr.Button(
                    "② Register smiles_to_xyz + packmol_build_system",
                    elem_classes=["btn-prompt"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'Sends both function sources inline</p>')
            with gr.Column(scale=1):
                t1c_btn = gr.Button(
                    "③ Create mace_md_specialist",
                    elem_classes=["btn-prompt"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'Natural-language request · no code pasted</p>')

    # ── Section 2: From Script ────────────────────────────────────────────────────
    with gr.Accordion("📄  From Script", open=False, elem_id="acc-script"):
        gr.HTML(
            '<p class="group-label">Upload a .py file — its path is injected into the '
            'prompt automatically · mirrors paper_tests_7 cell 30</p>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                upload_widget = gr.File(
                    label="Upload a Python tool script (.py)",
                    file_types=[".py"],
                    elem_id="upload-zone",
                )
                upload_path_box = gr.Textbox(
                    label="Saved path",
                    interactive=False,
                    lines=1,
                    elem_id="upload-path",
                    placeholder="Saved path appears here after upload.",
                )
                upload_preview_box = gr.Textbox(
                    label="Auto-generated prompt (editable before sending)",
                    interactive=True,
                    lines=3,
                    elem_classes=["monospace"],
                    placeholder="The prompt that will be sent to the agent appears here after upload.",
                )
            with gr.Column(scale=1):
                gr.HTML(
                    '<p class="group-label" style="margin-bottom:0.6rem;">'
                    'Or use the pre-loaded NVT script</p>'
                )
                t3a_btn = gr.Button(
                    "④ Register run_nvt_md from ASE_NVT_PBC.py",
                    elem_classes=["btn-file"], size="sm",
                )
                gr.HTML(
                    '<p style="font-size:0.72rem;color:#64748b;margin:0.3rem 0 0 0;">'
                    f'File: <code>tutorials/ASE_NVT_PBC.py</code><br>'
                    'Assigns run_nvt_md to mace_md_specialist</p>'
                )

    # ── Section 3: From LLM ───────────────────────────────────────────────────────
    with gr.Accordion("🤖  From LLM", open=False, elem_id="acc-llm"):
        gr.HTML(
            '<p class="group-label">Ask the agent to write, register, and assign a '
            'brand-new tool from a description — mirrors paper_tests_7 cell 38</p>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                t4a_btn = gr.Button(
                    "⑤ Write & register plot_nvt_trajectory",
                    elem_classes=["btn-llm"], size="sm",
                )
                gr.HTML(
                    '<p style="font-size:0.72rem;color:#64748b;margin:0.3rem 0 0 0;">'
                    'LLM writes the plotting function from a spec,<br>'
                    'registers it, and assigns it to mace_md_specialist</p>'
                )
            with gr.Column(scale=2):
                gr.Textbox(
                    value=PROMPT_T4A,
                    label="Prompt preview",
                    interactive=False,
                    lines=12,
                    elem_classes=["monospace"],
                )

    # ── Section 4: Run Simulations ────────────────────────────────────────────────
    with gr.Accordion("▶  Run Simulations", open=False, elem_id="acc-run"):
        gr.HTML(
            '<p class="group-label">Execute the full workflow '
            '— tools must be registered first via the sections above</p>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                t1_run_btn = gr.Button(
                    "T1 — Download MACE-Foundation Models",
                    elem_classes=["btn-run"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'Downloads MACE-MP-0b3 · skips if model already present</p>')
            with gr.Column(scale=1):
                t2_btn = gr.Button(
                    "T2 — Build NaCl + Water Box",
                    elem_classes=["btn-run"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        '1 NaCl pair + 267 H₂O · 20 Å periodic box</p>')
        with gr.Row():
            with gr.Column(scale=1):
                t3b_btn = gr.Button(
                    "T3 — Run NVT MD (10 steps, 300 K)",
                    elem_classes=["btn-run"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'ASE · nacl_water_box.xyz · 300 K · 10 steps</p>')
            with gr.Column(scale=1):
                t4b_btn = gr.Button(
                    "T4 — Plot NVT Trajectory",
                    elem_classes=["btn-run"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'Energy & temperature from nvt_nacl_water.traj</p>')

    # ── Section 5: Run Workflows ──────────────────────────────────────────────────
    with gr.Accordion("⚡  Run Workflows", open=False, elem_id="acc-wf"):
        gr.HTML(
            '<p class="group-label">End-to-end autonomous workflows '
            '— all tools must be registered before running</p>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                wf_final_btn = gr.Button(
                    "Full MD Workflow — Liquid Methanol",
                    elem_classes=["btn-run"], size="sm",
                )
                gr.HTML('<p style="font-size:0.72rem;color:#64748b;margin:0.2rem 0 0 0;">'
                        'Download · SMILES→XYZ · pack box · NVT MD · plot · all from one prompt</p>')

    # ── Main layout ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left — chat ──────────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=480,
                render_markdown=True,
            )
            with gr.Row(elem_id="send-row"):
                msg_box = gr.Textbox(
                    placeholder="Describe your task, or click a step above to auto-fill…",
                    show_label=False,
                    lines=2,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

            with gr.Accordion("🔍  Agent trace", open=False):
                trace_box = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=12,
                    max_lines=40,
                    elem_classes=["monospace"],
                    placeholder="Routing steps and tool calls will appear here…",
                )

        # Right — status + session ─────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=270):
            status_box = gr.HTML(
                value=_get_status_html,
                elem_id="status-panel-wrap",
            )
            refresh_btn = gr.Button("↻  Refresh", size="sm")

            gr.Markdown("### Session")
            thread_box = gr.Textbox(
                label="Thread ID",
                value=_new_thread_id,
                interactive=True,
            )
            new_thread_btn = gr.Button("＋  New Thread", size="sm")
            thread_hist = gr.Dropdown(
                label="Resume a previous thread",
                choices=_load_thread_choices(),
                interactive=True,
                allow_custom_value=False,
            )

    # ── Event wiring ──────────────────────────────────────────────────────────────

    # Send / submit
    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot, thread_box],
        outputs=[chatbot, trace_box, status_box],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot, thread_box],
        outputs=[chatbot, trace_box, status_box],
    ).then(fn=lambda: "", outputs=msg_box)

    # Refresh / new thread / history
    refresh_btn.click(fn=_get_status_html, outputs=status_box)

    new_thread_btn.click(
        fn=do_new_thread,
        outputs=[thread_box, thread_hist],
    ).then(fn=lambda: [], outputs=chatbot)

    thread_hist.change(
        fn=select_thread,
        inputs=thread_hist,
        outputs=thread_box,
    )

    # File upload → save to uploads/ + auto-fill msg_box, show saved path, show prompt preview
    upload_widget.upload(
        fn=handle_upload,
        inputs=upload_widget,
        outputs=[msg_box, upload_path_box, upload_preview_box],
    )
    upload_widget.clear(
        fn=lambda: ("", "", ""),
        outputs=[msg_box, upload_path_box, upload_preview_box],
    )

    def _fill_run(prompt: str) -> tuple[str, str]:
        """Fill msg_box + auto-generate a fresh thread_id for execution prompts.
        Mirrors the notebook's fresh-thread approach for T3b and other Run steps."""
        return prompt, _new_thread_id()

    # Setup buttons — use the current session thread (build context in-place)
    t1a_btn.click(fn=lambda: PROMPT_T1A, outputs=msg_box)
    t1b_btn.click(fn=lambda: PROMPT_T1B, outputs=msg_box)
    t1c_btn.click(fn=lambda: PROMPT_T1C, outputs=msg_box)
    t3a_btn.click(fn=lambda: PROMPT_T3A, outputs=msg_box)
    t4a_btn.click(fn=lambda: PROMPT_T4A, outputs=msg_box)

    # Execution buttons — always start a fresh thread (mirrors notebook Cell 35)
    t1_run_btn.click(fn=lambda: _fill_run(PROMPT_T1_RUN), outputs=[msg_box, thread_box])
    t2_btn.click(fn=lambda:     _fill_run(PROMPT_T2),     outputs=[msg_box, thread_box])
    t3b_btn.click(fn=lambda:    _fill_run(PROMPT_T3B),    outputs=[msg_box, thread_box])
    t4b_btn.click(fn=lambda:    _fill_run(PROMPT_T4B),    outputs=[msg_box, thread_box])
    wf_final_btn.click(fn=lambda: _fill_run(PROMPT_WORKFLOW_FINAL), outputs=[msg_box, thread_box])

# ── Entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "8888")),
        show_error=True,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
