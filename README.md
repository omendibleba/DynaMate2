# DynaMate

**DynaMate** is a dynamic multi-agent framework built on [LangGraph](https://github.com/langchain-ai/langgraph) that allows users to register new Python functions as agent tools and create new agents at runtime — through natural language prompts — without restarting the system. All tools, agents, and conversation history are persisted to disk and restored automatically on the next session.

A **Prompt Enhancer** layer sits between the user and the Supervisor. It reads the current pool state and rewrites each raw user query with explicit routing hints — agent names and relevant tool names — so users never need to know internal names to get correct routing.

Originally developed as a research tool for molecular simulation workflows, DynaMate is general-purpose: any Python function with a docstring can become a callable tool.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Interactive CLI](#interactive-cli)
  - [Single Prompt](#single-prompt)
  - [Inspect Graph and Pool State](#inspect-graph-and-pool-state)
  - [Using as a Python Library](#using-as-a-python-library)
- [How Persistence Works](#how-persistence-works)
  - [What is Persisted](#what-is-persisted)
  - [What is Not Persisted](#what-is-not-persisted)
  - [Implementation Details](#implementation-details)
  - [Where State is Stored](#where-state-is-stored)
  - [Startup Restoration Sequence](#startup-restoration-sequence)
- [Core Concepts](#core-concepts)
  - [AgentPool](#agentpool)
  - [AgentPoolWithSupervisor](#agentpoolwithsupervisor)
  - [ToolManager](#toolmanager)
  - [PromptEnhancer](#promptenhancer)
  - [DynamicToolAgent](#dynamictoolagent)
  - [PersistentAgentPoolWithSupervisor](#persistentagentpoolwithsupervisor)
- [Adding Tools and Agents](#adding-tools-and-agents)
  - [Register a Tool from a Code String](#register-a-tool-from-a-code-string)
  - [Register Tools from a File](#register-tools-from-a-file)
  - [Add a New Agent](#add-a-new-agent)
  - [Assign Tools to Agents](#assign-tools-to-agents)
  - [Remove a Tool](#remove-a-tool)
  - [Remove an Agent](#remove-an-agent)
- [Running Tests](#running-tests)
- [Limitations](#limitations)

---

## Architecture Overview

DynaMate is organized as a four-tier pipeline:

```
User prompt  (natural language — no tool/agent names required)
    │
    ▼
PromptEnhancer                ← rewrites query with explicit routing hints
    │                            (reads live pool state on every call)
    ▼
Supervisor                    ← routes tasks to the right agent
    ├── ToolManager           ← manages the pool (register, assign, add agents)
    ├── shell_agent           ← runs shell commands (always present)
    ├── compute_agent         ← computation (starts empty, gets tools assigned)
    └── [dynamic agents...]   ← created at runtime, persisted across sessions
              ▲
         AgentPool  (shared state)
         ├── _tool_registry   {tool_name → StructuredTool}
         ├── _source_registry {tool_name → source_code}   [persistent layer]
         ├── _agents          {name → {model, base_tools, extra_tools, ...}}
         └── _supervisor      current compiled supervisor graph
```

**Key design principles:**

| Operation | Rebuild cost |
|---|---|
| Assign a tool to an agent | Rebuilds **only that agent** |
| Add a new agent to the pool | Rebuilds **that agent + the supervisor** |
| Register a tool (without assigning) | No rebuild |
| Remove a tool | Rebuilds **all agents that had it assigned** |
| Remove an agent | Rebuilds **the supervisor** |

The Supervisor is never modified directly. It is rebuilt by the `AgentPool` whenever the set of agents changes, so `pool.supervisor` always reflects the current state.

---

## Project Structure

```
DynaMate2_V4_Claude/
├── main.py                        # CLI entry point
├── .env                           # OpenAI API key (not committed)
├── .dynamate/                     # Auto-created on first run
│   ├── pool_state.json            # Persisted tools, agents, assignments
│   └── conversations                # Persisted conversation history (SQLite)
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
│   └── test_persistence.py        # Tests for cross-session persistence
│
├── misc/
│   └── show_graph.py              # Pool inspector + Mermaid / PNG output
│
└── tutorials/
    └── tutorial_1.ipynb           # Development notebook with all examples
```

---

## Requirements

- Python 3.10+
- An OpenAI API key
- The following Python packages (available in the project environment):

```
langchain
langchain-openai
langchain-community
langgraph
langgraph-supervisor
langgraph-checkpoint-sqlite
python-dotenv
```

## Setup

**1. Clone / navigate to the project directory:**
```bash
git clone https://github.com/omendibleba/DynaMate2.git
cd DynaMate2
```

**2. Create a `.env` file with your OpenAI API key:**
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

**3. Verify the setup:**
```bash
conda activate enviroment
python3 -c "from dynamate import AgentPool; print('OK')"
```

## Usage

### Interactive CLI

Start an interactive session. All tools and agents from previous sessions are restored automatically:

```bash
py main.py
```

You will see output like:
```
Building system  [model=gpt-4o-mini, state=.dynamate] ...
State restored — 3 tool(s), 1 dynamic agent(s), 4 assignment(s).
Ready.

DynaMate session  [thread: a3f9b1c2]
Agents: ['shell_agent', 'compute_agent', 'unit_conversion_agent']
Tools in registry: ['boltzmann_energy', 'angstrom_to_bohr', 'eV_to_hartree']
Type 'status' to inspect the pool, 'exit' to quit.

>>>
```

Type `status` at the prompt to inspect the current pool at any time. Type `exit` or press `Ctrl+C` to quit.

### Single Prompt

Run one prompt and exit (useful for scripting):

```bash
py main.py --prompt "What files are in the current directory?"
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-4o-mini` | OpenAI model name |
| `--state-dir` | `.dynamate/` | Directory for persisted state |
| `--thread-id` | random UUID | Conversation thread — reuse to continue a previous conversation |
| `--prompt` | — | Run one prompt and exit (non-interactive) |
| `--verbose` | off | Print all messages per chunk, not just the last |
| `--status` | off | Print pool status and exit without starting a session |

**Examples:**

```bash
# Use a more capable model
py main.py --model gpt-4o

# Continue a named conversation thread
py main.py --thread-id my-project

# Save state to a backed-up location (recommended on HPC)
py main.py --state-dir ~/dynamate_state

# Print pool status without starting a session
py main.py --status

# Single prompt with verbose output
py main.py --prompt "List all agents and their tools" --verbose
```

### Inspect Graph and Pool State

Print the supervisor Mermaid diagram and pool state to the terminal:

```bash
py misc/show_graph.py
```

Save the supervisor graph as a PNG image:

```bash
py misc/show_graph.py --png supervisor_graph.png
```

Preview what the graph looks like after adding a hypothetical agent:

```bash
py misc/show_graph.py --add-agent chemistry_agent \
    --agent-prompt "You compute molecular properties from SMILES strings."
```

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
from langchain_community.tools import ShellTool

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Set up persistence
saver      = PersistentSaver("my_state/conversations")
pool_store = PoolStore("my_state/pool_state.json")

# Build the pool
pool = PersistentAgentPoolWithSupervisor(
    supervisor_model=model,
    pool_store=pool_store,
    checkpointer=saver,
)
pool.add_agent("shell_agent",   model, base_tools=[ShellTool()], _is_dynamic=False)
pool.add_agent("compute_agent", model, base_tools=[],            _is_dynamic=False)

tool_manager = build_tool_manager_v2(pool, model)
pool.set_system_agents([tool_manager])

# Restore previous session
pool.restore_state(model_factory=lambda name: ChatOpenAI(model=name, temperature=0.0))

# Build the prompt enhancer (queries pool live on every call)
enhancer = PromptEnhancer(model=model, pool=pool)

# Stream a prompt — raw user input is enhanced before reaching the supervisor
raw_query = "What agents do you have?"
enhanced_query = enhancer.enhance(raw_query)

config = {"configurable": {"thread_id": "my-thread"}}
for chunk in pool.supervisor.stream(
    {"messages": [{"role": "user", "content": enhanced_query}]},
    config=config,
    recursion_limit=25,
):
    pretty_print_messages(chunk, last_message=True)
```

---

## Adding Tools and Agents

All of the following can be done through natural language prompts in the interactive CLI. The Supervisor routes them to the ToolManager automatically.

### Register a Tool from a Code String

Paste a function definition directly into your prompt:

```
>>> Please register this Python function as a tool:

def boltzmann_energy(temperature_K: float) -> str:
    """Compute thermal energy kT in eV for a given temperature in Kelvin."""
    kT = 8.617333e-5 * temperature_K
    return f"kT at {temperature_K} K = {kT:.6f} eV"
```

**Rules for registrable functions:**
- Must have a **docstring** — this becomes the tool's description for the LLM
- Must be a **top-level** function (no nested classes, etc.)
- Type annotations on arguments are recommended but not required
- The function is executed via `exec()` in an isolated namespace

### Register Tools from a File

```
>>> Load all tools from /path/to/my_tools.py
```

The file should contain plain Python function definitions with docstrings. All top-level callables will be registered.

### Add a New Agent

```
>>> Create a new agent called unit_conversion_agent whose job is to convert
    between physical units used in molecular simulations.
```

The new agent:
- Starts with no tools
- Uses the same model as the ToolManager (configurable via `--model`)
- Is immediately added to the supervisor's routing table
- Is persisted and will reappear in the next session

### Assign Tools to Agents

After registering tools, assign them to specific agents:

```
>>> Assign boltzmann_energy to compute_agent
>>> Assign angstrom_to_bohr to unit_conversion_agent
```

The same tool can be assigned to multiple agents. Only the target agent is rebuilt — others are unaffected.

### Remove a Tool

To unregister a tool and automatically unassign it from every agent that holds it:

```
>>> Remove the boltzmann_energy tool
```

The ToolManager calls `remove_tool_from_registry`. Every agent that had the tool assigned is rebuilt. The tool is also removed from `pool_state.json` so it will not reappear in the next session.

You can verify the result:

```
>>> List all registered tools
```

### Remove an Agent

To remove a dynamic agent from the pool:

```
>>> Remove compute_v2 from the pool
```

The ToolManager calls `remove_agent_from_pool`. The supervisor is rebuilt to exclude the removed agent. The agent is also removed from `pool_state.json`.

**Constraints:**
- Only dynamically added agents can be removed. The initial agents `shell_agent`, `compute_agent`, and `tool_manager` are protected.
- Any tools that were assigned to the removed agent remain in the registry and are still assignable to other agents.

---

## How Persistence Works

### What is Persisted

| What | Storage | Survives restart? |
|---|---|---|
| Registered tool **source code** | `pool_state.json` | ✓ Yes |
| Dynamic agent **name + system_prompt + model** | `pool_state.json` | ✓ Yes |
| Tool-to-agent **assignments** | `pool_state.json` | ✓ Yes |
| **Conversation history** (per thread_id) | `conversations` (SQLite) | ✓ Yes |

### What is Not Persisted

| What | Why |
|---|---|
| `shell_agent`, `compute_agent` | Initial agents are always rebuilt fresh from `main.py`. Their base tools (e.g. `ShellTool`) are Python objects that cannot be serialized. |
| The supervisor graph itself | It is a compiled Python object. It is rebuilt at startup by replaying persisted state. |
| In-flight / partial LLM responses | Only committed LangGraph checkpoints are saved. |

### Implementation Details

Persistence is split into two independent layers:

#### Layer 1 — Pool State (`pool_state.json`)

Managed by `PoolStore` and `PersistentAgentPoolWithSupervisor` in `dynamate/persistence.py`.

The JSON file has three keys:

```json
{
  "tools": {
    "boltzmann_energy": "def boltzmann_energy(temperature_K: float) -> str:\n    ..."
  },
  "dynamic_agents": {
    "unit_conversion_agent": {
      "system_prompt": "You convert physical units...",
      "model_name": "gpt-4o-mini"
    }
  },
  "assignments": {
    "compute_agent": ["boltzmann_energy"],
    "unit_conversion_agent": ["angstrom_to_bohr", "eV_to_hartree"]
  }
}
```

`PersistentAgentPoolWithSupervisor` overrides five methods of `AgentPool`:

- `add_agent()` — records new dynamic agents and calls `_autosave()`
- `register_tool_from_code()` — captures the source code string alongside the compiled tool and calls `_autosave()`
- `assign_tool()` — updates the assignment mapping and calls `_autosave()`
- `remove_tool()` — deletes from `_source_registry`, unassigns from all agents, and calls `_autosave()`
- `remove_agent()` — removes from `_dynamic_agent_names`, deletes the agent, and calls `_autosave()`

`_autosave()` is called **synchronously after every mutation**, so the file is always up to date. A `_loading` flag suppresses saves during the restoration phase to avoid overwriting saved state.

#### Layer 2 — Conversation History (`conversations`)

Managed by `PersistentSaver` in `dynamate/persistence.py`.

`PersistentSaver` is a thin subclass of `SqliteSaver` from `langgraph-checkpoint-sqlite`. It manages its own `sqlite3` connection from a file path, so callers only need to provide a path string — no connection management required:

```python
class PersistentSaver(SqliteSaver):
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        super().__init__(conn)
        self.setup()   # creates 'checkpoints' and 'writes' tables if absent
```

**Why SQLite over `shelve`?**

| Property | `shelve` (old) | `SqliteSaver` (new) |
|---|---|---|
| External dependency | None (stdlib) | `langgraph-checkpoint-sqlite` |
| File format | `dbm` (platform-specific) | Standard SQLite `.db` |
| Portability | OS/Python-version dependent | Any platform, any SQLite tool |
| Inspection | Binary only | Any SQLite browser or `sqlite3` CLI |
| Write strategy | Full re-serialize on every write | Incremental row-level writes |
| Concurrency | No locking | SQLite WAL mode |

Checkpoints and intermediate writes are stored in two tables (`checkpoints`, `writes`) managed by `SqliteSaver.setup()`. LangGraph serializes all state values with its built-in `JsonPlusSerializer` before writing, so no manual pickling is needed.

### Where State is Stored

By default, all state is written to:

```
DynaMate2_V4_Claude/.dynamate/
├── pool_state.json       ← human-readable JSON (open with any text editor)
└── conversations         ← SQLite database (inspect with any SQLite tool)
```

This path can be changed with `--state-dir`:

```bash
# Store state in your home directory (backed up, quota-limited)
py main.py --state-dir ~/dynamate_state


### Startup Restoration Sequence

Every time `main.py` runs, `build_system()` follows this exact sequence:

```
1. Create PersistentSaver     → open conversations SQLite DB (tables created if absent)
2. Create PoolStore           → read pool_state.json (parsed but not applied yet)
3. Create PersistentAgentPoolWithSupervisor
4. Add shell_agent            (_is_dynamic=False → not saved to JSON)
5. Add compute_agent          (_is_dynamic=False → not saved to JSON)
6. Build ToolManager          → closures bind to the live pool object
7. pool.set_system_agents()   → first supervisor build
8. pool.restore_state()
   ├── Re-exec each saved tool source block  → repopulates _tool_registry
   ├── Recreate each dynamic agent           → triggers supervisor rebuild
   └── Re-apply each assignment              → rebuilds only the target agent
9. PromptEnhancer(model, pool) → bound to live pool; no state of its own
```

After step 9 the system is identical to how it was at the end of the last session. The `PromptEnhancer` always reflects current pool state because it queries the pool on every `enhance()` call — no restart needed after adding agents or tools.

---

## Core Concepts

### AgentPool

`dynamate/pool.py`

The shared state object. Holds named agents and a global tool registry. Responsible for:
- Registering agents with a model and base tools
- Registering tools from code strings or files
- Assigning registered tools to specific agents (rebuilding only the target agent)

```python
pool = AgentPool()
pool.add_agent("my_agent", model, base_tools=[ShellTool()])
pool.register_tool_from_code("def my_func(...): ...")
pool.assign_tool("my_func", "my_agent")
```

### AgentPoolWithSupervisor

`dynamate/pool.py`

Extends `AgentPool`. Owns the supervisor graph and rebuilds it automatically whenever an agent is added. The supervisor is accessed via `pool.supervisor`.

Required initialization order:
```python
pool = AgentPoolWithSupervisor(model)
pool.add_agent(...)                    # add domain agents
tm = build_tool_manager_v2(pool, model)
pool.set_system_agents([tm])           # triggers first supervisor build
```

### ToolManager

`dynamate/tool_manager.py`

A dedicated ReAct agent whose only responsibility is managing the pool. It exposes nine tools via closure over the pool object:

| Tool | Description |
|---|---|
| `register_tool_from_code` | Register functions from a code string |
| `register_tool_from_file` | Register functions from a `.py` file |
| `assign_tool_to_agent` | Assign a registered tool to a named agent |
| `add_agent_to_pool` | Create a new agent and add it to the pool |
| `remove_tool_from_registry` | Unregister a tool and unassign it from all agents |
| `remove_agent_from_pool` | Remove a dynamic agent and rebuild the supervisor |
| `list_registered_tools` | List all tools in the global registry |
| `list_agent_tools` | List tools for a specific agent |
| `list_agents` | List all agents in the pool |

The ToolManager never performs domain work (computation, shell commands, etc.). The initial agents (`shell_agent`, `compute_agent`, `tool_manager`) are protected and cannot be removed.

### PromptEnhancer

`dynamate/prompt_enhancer.py`

A lightweight LLM layer that sits between the user and the Supervisor. On every call it reads the current pool state and rewrites the user's raw query to include explicit routing hints — the name of the most relevant agent and its tools — so the Supervisor can route directly without extra reasoning hops.

```python
enhancer = PromptEnhancer(model=model, pool=pool)

raw   = "What is the Arrhenius factor at 400 K for a 0.30 eV barrier?"
enhanced = enhancer.enhance(raw)
# → "...What is the Arrhenius factor at 400 K for a 0.30 eV barrier?
#    Use thermal_agent — it should use both boltzmann_energy and
#    arrhenius_factor tools to compute the answer step by step."
```

**Key properties:**

| Property | Detail |
|---|---|
| Context | `_build_pool_context()` calls `list_agents()` and `list_agent_tools()` on every invocation — always reflects the live pool |
| LLM call | Single `model.invoke([SystemMessage, HumanMessage])` — no ReAct loop, no graph |
| Output | Original question kept verbatim; routing instruction appended |
| Fallback | If the pool has no agents, returns the original input unchanged (no API call made) |
| State | Stateless — holds only a model reference and a pool reference |

The `PromptEnhancer` is constructed once in `build_system()` and passed to both `run_interactive()` and `run_single()`. Because it queries the pool at call time, adding a new agent or assigning a new tool is immediately visible to the enhancer without any restart.

### DynamicToolAgent

`dynamate/dynamic_agent.py`

A simpler, standalone alternative to the full pool architecture. A single ReAct agent that can register new tools at runtime without a supervisor or pool. Useful for simple workflows where multi-agent routing is not needed.

```python
agent = DynamicToolAgent(model, base_tools=[ShellTool()])
agent.stream({"messages": [{"role": "user", "content": "Add def foo..."}]})
```

**Note:** `DynamicToolAgent` does not have persistence. It is stateless across process restarts.

### PersistentAgentPoolWithSupervisor

`dynamate/persistence.py`

Extends `AgentPoolWithSupervisor` with automatic save/restore. Overrides `add_agent`, `register_tool_from_code`, `assign_tool`, `remove_tool`, and `remove_agent` to write to `pool_state.json` after every mutation. Provides `restore_state()` to replay saved state on startup.

---

## Running Tests

All tests are standalone scripts that can be run directly without a test runner:

```bash
PY=python3

# DynamicToolAgent: add tools from code and file
$PY tests/test_dynamic_agent.py

# AgentPool + ToolManager: register and assign tools selectively
$PY tests/test_agent_pool.py

# Dynamic agent addition: add a new agent at runtime via the supervisor
$PY tests/test_add_agent.py

# Persistence: cross-session save and restore (no LLM calls)
$PY tests/test_persistence.py
```

Tests `test_dynamic_agent`, `test_agent_pool`, and `test_add_agent` make real LLM API calls and require a valid `OPENAI_API_KEY` in `.env`. `test_persistence` uses only the pool internals and does not make LLM calls.

---

## Limitations

### Functional Limitations

**Tool functions must be self-contained.**
Functions registered via `register_tool_from_code` are executed with `exec()` in an isolated namespace. They cannot reference global variables, imported modules, or other functions defined outside the code block. All imports and dependencies must be declared inside the function body:

```python
# ✓ Works — imports inside the function
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
Re-registering a function with the same name is silently skipped (the registry already contains that name). To update a tool's implementation, remove it first with `remove_tool_from_registry`, then re-register the new version.

**Tool assignment is one-directional.**
A tool can be assigned to multiple agents, but assigning a tool to a new agent does not remove it from agents it was previously assigned to.

**New dynamic agents start with no base tools.**
When a new agent is created via `add_agent_to_pool`, it receives no base tools. All capabilities must come from assigned tools. If you need a new agent to have base tools like `ShellTool`, you must add it programmatically in `main.py`'s `build_system()` function.

### Persistence Limitations

**Conversation history resets when a new agent is added.**
`AgentPoolWithSupervisor._rebuild_supervisor()` is called every time a new agent is added, which compiles a new supervisor graph. The `PersistentSaver` (SQLite) checkpointer is reused across rebuilds, so history from previous sessions is preserved. However, in-memory state accumulated during the current session before the rebuild may not be visible to the new graph until it is next checkpointed.

**Conversation history grows indefinitely.**
The `conversations` SQLite database accumulates checkpoint rows across sessions. There is currently no mechanism to prune old threads. You can inspect and manage the file directly with any SQLite tool (e.g. `sqlite3 .dynamate/conversations`).

**No concurrent access.**
`pool_state.json` is written without file locking. Running two `main.py` sessions simultaneously against the same `--state-dir` will cause race conditions and may corrupt the JSON state. The SQLite database handles concurrent reads safely, but concurrent writes from two processes can still interleave.

### Model Limitations

**The Prompt Enhancer adds one extra LLM call per user turn.**
`PromptEnhancer.enhance()` makes a synchronous `model.invoke()` call before each supervisor invocation. For bulk scripting or latency-sensitive use cases, you can bypass the enhancer by passing the raw query directly to `pool.supervisor.stream()`. The enhancer is a convenience layer, not a hard dependency.

**The Prompt Enhancer may misroute when agents have overlapping tool sets.**
The enhancer selects the best agent based on its system prompt and assigned tools. If two agents have similar tools or descriptions, the routing hint may name the wrong agent. The Supervisor can still override the hint based on its own routing rules.

**The Supervisor may not always route correctly.**
The Supervisor is an LLM-based router. For tasks that span multiple agents (e.g. "register a tool, assign it, and then use it"), it may not always execute all steps in a single invocation. If a step is skipped, you can re-issue the specific sub-request.

**Tool docstrings are critical.**
The LLM decides which tool to call based entirely on the tool's docstring. Functions with vague or missing docstrings may not be called correctly or at all. Always write clear, specific docstrings that describe inputs, outputs, and units.

**Model context limits.**
Very long conversations (many turns across many sessions) can approach the model's context window limit. Use separate `--thread-id` values for distinct tasks to keep threads focused.
