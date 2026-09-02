# DynaMate2

**DynaMate2** is a dynamic multi-agent framework built on [LangGraph](https://github.com/langchain-ai/langgraph) that lets users register new Python functions as agent tools and create new specialist agents at runtime — through natural language prompts — without restarting the system. All tools, agents, and conversation history are persisted to disk and restored automatically on the next session.

A **Prompt Enhancer** layer sits between the user and the Supervisor. It reads the live pool state and rewrites each raw user query with explicit routing hints — agent names and relevant tool names — so users never need to know internal names to get correct routing.

Originally developed as a research framework for molecular simulation workflows (MACE force fields, ASE MD, PACKMOL box building), DynaMate2 is general-purpose: any Python function with a docstring can become a callable tool.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Tutorial](#tutorial)
- [Usage](#usage)
  - [Interactive CLI](#interactive-cli)
  - [Web UI (React)](#web-ui-react)
  - [Gradio Web UI (legacy)](#gradio-web-ui-legacy)
  - [Single Prompt](#single-prompt)
  - [Using as a Python Library](#using-as-a-python-library)
- [How Persistence Works](#how-persistence-works)
- [Core Concepts](#core-concepts)
- [Adding Tools and Agents](#adding-tools-and-agents)
- [Running Tests](#running-tests)
- [Limitations](#limitations)

---

## Architecture Overview

DynaMate2 is organised as a four-tier pipeline:

```
User prompt  (natural language — no tool/agent names required)
    │
    ▼
PromptEnhancer                ← rewrites query with routing hints
    │                            (reads live pool state on every call)
    ▼
Supervisor                    ← routes tasks to the right agent
    ├── ToolManager           ← manages the pool (register, assign, add agents)
    └── [domain agents...]    ← created at runtime, persisted across sessions
              │
              ▼ example
         mace_md_specialist   ← runs MACE/ASE simulations
           ├── download_mace_model
           ├── smiles_to_xyz
           ├── packmol_build_system
           ├── run_nvt_md
           └── plot_nvt_trajectory
              ▲
         AgentPool  (shared state)
         ├── _tool_registry   {tool_name → StructuredTool}
         ├── _source_registry {tool_name → source_code}   [persistent layer]
         ├── _agents          {name → {model, base_tools, extra_tools, ...}}
         └── _supervisor      current compiled supervisor graph
```

**Rebuild cost summary:**

| Operation | Rebuild cost |
|---|---|
| Assign a tool to an agent | Rebuilds **only that agent** |
| Add a new agent to the pool | Rebuilds **that agent + the supervisor** |
| Register a tool (no assignment) | No rebuild |
| Remove a tool | Rebuilds **all agents that had it assigned** |
| Remove an agent | Rebuilds **the supervisor** |

---

## Project Structure

```
DynaMate2/
├── main.py                        # CLI entry point
├── server.py                      # React UI production entry point
├── app.py                         # Gradio web UI (legacy — gradio-ui-legacy branch)
├── .env                           # OPENAI_API_KEY (not committed)
├── .env_sample                    # Template for .env
├── environment.yml                # Recommended conda environment
├── environment_pinned.yml         # Fully-pinned conda environment
├── requirements.txt               # pip requirements
│
├── backend/                       # FastAPI backend for the React UI
│   ├── main.py                    # App + lifespan (builds the pool once at startup)
│   ├── routes/                    # health, status, threads, chat, quickstart, tools
│   ├── state.py                   # ui_state/ paths, thread-index helpers
│   ├── streaming.py               # SSE bridge for pool.supervisor.stream()
│   ├── quickstart.py              # Tutorial-mirroring prompt strings
│   └── schemas.py                 # Pydantic request/response models
│
├── frontend/                      # React + TypeScript + Tailwind UI (Vite)
│   └── src/
│       ├── components/            # ChatPanel, QuickStartPanel, StatusSidebar, …
│       ├── hooks/useChatStream.ts # Drives one chat turn (SSE)
│       └── lib/api.ts             # Typed backend client
│
├── dynamate/                      # Core package
│   ├── __init__.py                # Public exports
│   ├── pool.py                    # AgentPool, AgentPoolWithSupervisor
│   ├── tool_manager.py            # build_tool_manager_v2 factory
│   ├── prompt_enhancer.py         # PromptEnhancer (query rewriting layer)
│   ├── dynamic_agent.py           # DynamicToolAgent (standalone, no supervisor)
│   ├── persistence.py             # PersistentSaver, PoolStore,
│   │                              # PersistentAgentPoolWithSupervisor
│   └── utils.py                   # pretty_print_messages
│
├── tests/
│   ├── _setup.py                  # Shared fixtures (model, pool builder)
│   ├── test_dynamic_agent.py      # Tests for DynamicToolAgent
│   ├── test_agent_pool.py         # Tests for AgentPool + ToolManager
│   ├── test_add_agent.py          # Tests for dynamic agent addition
│   ├── test_persistence.py        # Tests for cross-session persistence
│   └── agent_tool_diagnostic.py   # Three-level diagnostic helper
│
├── misc/
│   └── show_graph.py              # Pool inspector + Mermaid / PNG output
│
├── ui_state/                      # Auto-created: web UI persistent state (shared by both UIs)
│   ├── pool_state.json
│   ├── conversations.db
│   └── tools/
│
└── tutorials/
    ├── DynaMate2_tutorial.ipynb   # Main tutorial notebook
    ├── tutorial_state/            # Persistent state for the tutorial
    ├── ASE_NVT_PBC.py             # NVT MD tool source (Langevin + MACE)
    ├── smiles_to_xyz.py           # SMILES → XYZ tool source
    ├── packmol_build_system.py    # Packmol box-builder tool source
    ├── pretty_print.py            # Notebook display helper
    ├── models/                    # MACE model files (gitignored — download on first run)
    └── sample_outputs/            # Reference outputs from a completed tutorial run
        ├── dmf.xyz
        ├── dmf_30.xyz / dmf_30.traj / dmf_30.png
        ├── water.xyz / nacl_water_box.xyz
        └── end_to_end_test/       # Full NaCl-water workflow outputs
```

---

## Requirements

- Python 3.10+
- An OpenAI API key
- A CUDA-capable GPU (strongly recommended for MACE simulations; CPU fallback is very slow)
- [PACKMOL](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml) binary on your `PATH` (required by the `packmol_build_system` tool)

**Python packages** (see `environment.yml` for full list):

| Category | Packages |
|---|---|
| LLM stack | `langchain`, `langchain-openai`, `langchain-community`, `langgraph`, `langgraph-supervisor`, `langgraph-checkpoint-sqlite` |
| Molecular simulation | `ase`, `mace-torch`, `rdkit` |
| Numerics / plotting | `numpy`, `matplotlib` |
| Web UI | `gradio` |
| Notebook | `jupyter`, `ipykernel` |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/omendibleba/DynaMate2.git
cd DynaMate2
```

### 2. Create the conda environment

**Recommended (flexible, top-level packages):**
```bash
conda env create -f environment.yml
conda activate dynamate
```

**Alternative (fully pinned, maximum reproducibility):**
```bash
conda env create -f environment_pinned.yml
conda activate dynamate
```

Or with pip only:
```bash
pip install -r requirements.txt
```

### 3. Install PACKMOL

PACKMOL is an external binary (not pip-installable) required by the `packmol_build_system` tool:

```bash
conda install -c conda-forge packmol   # easiest
# or download from: http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml
```

### 4. Set your API key

```bash
cp .env_sample .env
# edit .env and add:  OPENAI_API_KEY=sk-...
```

### 5. Verify the installation

```bash
python -c "from dynamate import AgentPool; print('OK')"
```

---

## Tutorial

`tutorials/DynaMate2_tutorial.ipynb` demonstrates a complete molecular simulation workflow driven by natural language:

| Test | Task | Tools used |
|---|---|---|
| **T1** | Download a MACE foundation model | `download_mace_model` |
| **T2** | Build a periodic DMF liquid box (30 molecules) | `smiles_to_xyz`, `packmol_build_system` |
| **T2.1** | Build a NaCl aqueous solution box | `smiles_to_xyz`, `packmol_build_system` |
| **T3** | Run an NVT Langevin MD simulation | `run_nvt_md` |
| **T4** | Plot energies and temperature from the trajectory | `plot_nvt_trajectory` |
| **Integration** | Full end-to-end workflow from a single prompt | all five tools |

All tools are registered and assigned to `mace_md_specialist` through natural language prompts — no tool names need to appear in the user queries.

Reference outputs from a completed run are in `tutorials/sample_outputs/`. Running the notebook regenerates the same files.

**Before running:** download the MACE-POLAR-1-M model once:
```bash
mkdir -p tutorials/models
wget -P tutorials/models https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model
```

---

## Usage

### Interactive CLI

Start an interactive session. All tools and agents from previous sessions are restored automatically:

```bash
python main.py
```

Output:
```
Building system  [model=gpt-4.1-mini, state=.dynamate] ...
State restored — 5 tool(s), 1 dynamic agent(s), 5 assignment(s).
Ready.

DynaMate session  [thread: a3f9b1c2]
Agents: ['tool_manager', 'mace_md_specialist']
Tools in registry: ['download_mace_model', 'smiles_to_xyz', ...]
Type 'status' to inspect the pool, 'exit' to quit.

>>>
```

### Web UI (React)

The current web UI — a FastAPI backend (`backend/`) with a React + TypeScript +
Tailwind frontend (`frontend/`), replacing the earlier Gradio UI per reviewer
feedback. Same functionality as before (quick-start actions mirroring the
tutorial notebook, streaming chat, tool upload, thread history, live agent/tool
status) behind a more usable interface.

**Production (single command):**
```bash
cd frontend && npm install && npm run build && cd ..
python server.py
```
Starts at `http://localhost:8888`. State is saved to `ui_state/` and restored
between sessions, same as before.

**Development (hot-reload on both sides):**
```bash
uvicorn backend.main:app --reload --port 8000      # terminal 1
cd frontend && npm install && npm run dev            # terminal 2, proxies /api -> :8000
```
Open the Vite dev URL it prints (`http://localhost:5173`).

Node.js is required for the frontend; if it isn't already on your `PATH`,
`conda install -c conda-forge nodejs` into your environment.

### Gradio Web UI (legacy)

The original Gradio interface is preserved on the `gradio-ui-legacy` branch for
anyone who prefers it:
```bash
git checkout gradio-ui-legacy
python app.py
```
Starts at `http://localhost:8888`. Uses the same `ui_state/` as the React UI —
switching between the two preserves your tools, agents, and conversation
history.

### Single Prompt

```bash
python main.py --prompt "What tools are currently registered?"
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-4.1-mini` | OpenAI model name |
| `--state-dir` | `.dynamate/` | Directory for persisted state |
| `--thread-id` | random UUID | Conversation thread — reuse to continue a prior session |
| `--prompt` | — | Run one prompt and exit |
| `--verbose` | off | Print all messages, not just the last |
| `--status` | off | Print pool status and exit |

### Using as a Python Library

```python
from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
    pretty_print_messages,
)
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

# Set up persistence
saver      = PersistentSaver("my_state/conversations")
pool_store = PoolStore("my_state/pool_state.json")

# Build the pool
pool = PersistentAgentPoolWithSupervisor(
    supervisor_model=model,
    pool_store=pool_store,
    checkpointer=saver,
)

tool_manager = build_tool_manager_v2(pool, model)
pool.set_system_agents([tool_manager])

# Restore previous session
pool.restore_state(model_factory=lambda name: ChatOpenAI(model=name, temperature=0.0))

# Build the prompt enhancer
enhancer = PromptEnhancer(model=model, pool=pool)

# Stream a prompt
raw_query = "List all agents and their tools"
enhanced  = enhancer.enhance(raw_query)

config = {"configurable": {"thread_id": "my-thread"}}
for chunk in pool.supervisor.stream(
    {"messages": [{"role": "user", "content": enhanced}]},
    config=config,
    recursion_limit=25,
):
    pretty_print_messages(chunk, last_message=True)
```

---

## How Persistence Works

### What is Persisted

| What | Storage | Survives restart? |
|---|---|---|
| Registered tool **source code** | `pool_state.json` | ✓ Yes |
| Dynamic agent **name + system_prompt + model** | `pool_state.json` | ✓ Yes |
| Tool-to-agent **assignments** | `pool_state.json` | ✓ Yes |
| **Conversation history** (per thread_id) | `conversations.db` (SQLite) | ✓ Yes |

### What is Not Persisted

| What | Why |
|---|---|
| The supervisor graph itself | Compiled Python object — rebuilt at startup by replaying `pool_state.json` |
| In-flight / partial LLM responses | Only committed LangGraph checkpoints are saved |

### Implementation Details

**Layer 1 — Pool State (`pool_state.json`)**

Managed by `PoolStore` and `PersistentAgentPoolWithSupervisor` in `dynamate/persistence.py`:

```json
{
  "tools": {
    "run_nvt_md": "def run_nvt_md(model_path, structure_file, ...) ..."
  },
  "dynamic_agents": {
    "mace_md_specialist": {
      "system_prompt": "You are a specialist in MACE...",
      "model_name": "gpt-4.1-mini"
    }
  },
  "assignments": {
    "mace_md_specialist": ["download_mace_model", "smiles_to_xyz", "run_nvt_md", ...]
  }
}
```

`_autosave()` is called synchronously after every mutation (add, assign, remove), so the file is always current.

**Layer 2 — Conversation History (`conversations.db`)**

`PersistentSaver` is a thin subclass of `SqliteSaver` from `langgraph-checkpoint-sqlite`. All LangGraph checkpoints and intermediate writes are stored in two tables managed by `SqliteSaver.setup()`.

### Startup Restoration Sequence

```
1. Create PersistentSaver     → open SQLite DB
2. Create PoolStore           → read pool_state.json
3. Create PersistentAgentPoolWithSupervisor
4. Build ToolManager          → closures bind to live pool
5. pool.set_system_agents()   → first supervisor build
6. pool.restore_state()
   ├── Re-exec each saved tool source  → repopulate _tool_registry
   ├── Recreate each dynamic agent     → trigger supervisor rebuild
   └── Re-apply each assignment        → rebuild only target agent
7. PromptEnhancer(model, pool) → queries pool live on every enhance() call
```

---

## Core Concepts

### AgentPool

`dynamate/pool.py`

Shared state object. Holds named agents and a global tool registry.

```python
pool = AgentPool()
pool.add_agent("my_agent", model, base_tools=[])
pool.register_tool_from_code("def my_func(...): ...")
pool.assign_tool("my_func", "my_agent")
```

### AgentPoolWithSupervisor

`dynamate/pool.py`

Extends `AgentPool`. Owns the supervisor graph and rebuilds it automatically when agents are added or removed.

```python
pool = AgentPoolWithSupervisor(model)
pool.add_agent(...)
tm = build_tool_manager_v2(pool, model)
pool.set_system_agents([tm])   # first supervisor build
```

### ToolManager

`dynamate/tool_manager.py`

A dedicated ReAct agent that manages the pool. It exposes nine tools via closure over the pool:

| Tool | Description |
|---|---|
| `register_tool_from_code` | Register functions from a code string |
| `register_tool_from_file` | Register functions from a `.py` file |
| `assign_tool_to_agent` | Assign a registered tool to a named agent |
| `add_agent_to_pool` | Create a new agent |
| `remove_tool_from_registry` | Unregister a tool and unassign it from all agents |
| `remove_agent_from_pool` | Remove a dynamic agent |
| `list_registered_tools` | List tools in the global registry |
| `list_agent_tools` | List tools for a specific agent |
| `list_agents` | List all agents in the pool |

### PromptEnhancer

`dynamate/prompt_enhancer.py`

A lightweight LLM layer that rewrites user queries with routing hints before they reach the Supervisor.

```python
enhancer = PromptEnhancer(model=model, pool=pool)
raw      = "Run an NVT simulation with this structure file..."
enhanced = enhancer.enhance(raw)
# → "...Use mace_md_specialist — it should use run_nvt_md to complete the request.
#    All required input files are already present. Call run_nvt_md directly..."
```

**Routing rules (in priority order):**

| Rule | Trigger | Action |
|---|---|---|
| **A** | Message mentions a `.py` file path | Route to `tool_manager` → `register_tool_from_file`; chain `assign_tool_to_agent` if assignment requested |
| **B** | Message contains `def ` function definition | Route to `tool_manager` → `register_tool_from_code` |
| **2** | Single-tool request (run, compute, convert) | Append `"Use <agent> — call <tool> directly"` |
| **3** | Explicit multi-step request ("build AND run") | Chain `"First <tool_X>, then <tool_Y>"` |

**Key properties:**

| Property | Detail |
|---|---|
| Context | `_build_pool_context()` queries the live pool on every call |
| LLM call | Single `model.invoke()` — no ReAct loop |
| Fallback | Returns input unchanged if pool has no agents |
| State | Stateless — holds only a model reference and a pool reference |

### DynamicToolAgent

`dynamate/dynamic_agent.py`

A simpler, standalone ReAct agent that can register new tools at runtime without a supervisor. Useful for simple single-agent workflows.

```python
agent = DynamicToolAgent(model, base_tools=[])
agent.stream({"messages": [{"role": "user", "content": "Add def foo..."}]})
```

**Note:** Does not have persistence. Stateless across process restarts.

### PersistentAgentPoolWithSupervisor

`dynamate/persistence.py`

Extends `AgentPoolWithSupervisor` with automatic save/restore. Overrides `add_agent`, `register_tool_from_code`, `assign_tool`, `remove_tool`, and `remove_agent` to write `pool_state.json` after every mutation.

---

## Adding Tools and Agents

All of the following can be done through natural language prompts. The Supervisor routes them to the ToolManager automatically.

### Register a Tool from a Code String

```
>>> Please register this Python function as a tool:

def boltzmann_energy(temperature_K: float) -> str:
    """Compute thermal energy kT in eV for a given temperature in Kelvin."""
    kT = 8.617333e-5 * temperature_K
    return f"kT at {temperature_K} K = {kT:.6f} eV"
```

**Rules for registrable functions:**
- Must have a **docstring** — this becomes the tool description for the LLM
- Must be a **top-level** function
- All imports must be inside the function body (functions are executed via `exec()` in an isolated namespace)

### Register Tools from a File

```
>>> Load all tools from /path/to/my_tools.py
```

To also assign:

```
>>> Register the tools in /path/to/my_tools.py and assign run_nvt_md to mace_md_specialist
```

### Add a New Agent

```
>>> Create a new agent called mace_md_specialist whose job is to run
    MACE molecular dynamics simulations using ASE.
```

### Assign Tools to Agents

```
>>> Assign run_nvt_md to mace_md_specialist
```

The same tool can be assigned to multiple agents. Only the target agent is rebuilt.

### Remove a Tool

```
>>> Remove the boltzmann_energy tool
```

Every agent that had the tool assigned is rebuilt. The tool is removed from `pool_state.json`.

### Remove an Agent

```
>>> Remove mace_md_specialist from the pool
```

The supervisor is rebuilt. Any tools assigned to the removed agent remain in the registry.

---

## Running Tests

```bash
# DynamicToolAgent: add tools from code and file
python tests/test_dynamic_agent.py

# AgentPool + ToolManager: register and assign tools
python tests/test_agent_pool.py

# Dynamic agent addition: add a new agent via the supervisor
python tests/test_add_agent.py

# Persistence: cross-session save and restore (no LLM calls)
python tests/test_persistence.py
```

`test_dynamic_agent`, `test_agent_pool`, and `test_add_agent` make real LLM API calls and require a valid `OPENAI_API_KEY`. `test_persistence` does not make LLM calls.

---

## Limitations

### Functional Limitations

**Tool functions must be self-contained.**
Functions registered via `register_tool_from_code` run in an isolated `exec()` namespace. All imports and dependencies must be inside the function body:

```python
# ✓ Works
def compute_something(x: float) -> str:
    """Compute something."""
    import numpy as np
    return str(np.sqrt(x))

# ✗ Fails — 'np' is not in the exec namespace
import numpy as np
def compute_something(x: float) -> str:
    """Compute something."""
    return str(np.sqrt(x))
```

**Tools cannot be updated in place.**
Re-registering a function with the same name is silently skipped. To update a tool, remove it first (`remove_tool_from_registry`), then re-register.

**New dynamic agents start with no base tools.**
All capabilities must come from assigned tools.

### Simulation Limitations

**MACE requires a GPU.**
Running MACE simulations on CPU is technically possible but orders of magnitude slower. A CUDA-capable GPU is strongly recommended.

**PACKMOL must be installed separately.**
The `packmol_build_system` tool calls the `packmol` binary from the system PATH. If PACKMOL is not installed, that tool will fail at runtime.

### Persistence Limitations

**Conversation history grows indefinitely.**
The `conversations.db` SQLite database accumulates checkpoint rows across sessions. There is no built-in pruning. Inspect and manage with any SQLite tool:
```bash
sqlite3 .dynamate/conversations.db
```

**No concurrent access.**
`pool_state.json` is written without file locking. Running two sessions simultaneously against the same `--state-dir` may corrupt the state file.

### Model Limitations

**The Prompt Enhancer adds one extra LLM call per user turn.**
For bulk scripting or latency-sensitive use cases, bypass the enhancer by passing the raw query directly to `pool.supervisor.stream()`.

**Tool docstrings are critical.**
The LLM decides which tool to call based entirely on the docstring. Vague or missing docstrings cause incorrect or absent tool calls.

**Model context limits.**
Very long conversations can approach the model's context window. Use separate `--thread-id` values for distinct tasks.
