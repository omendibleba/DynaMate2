"""
dynamate.persistence
─────────────────────
Two persistence layers that together make the system fully stateful
across process restarts:

  PersistentSaver
    Thin subclass of SqliteSaver (langgraph-checkpoint-sqlite) that opens
    and owns its own sqlite3 connection from a file path.  All checkpoint
    reads and writes go directly to a portable .db file with no extra
    serialisation overhead.

  PoolStore
    Reads/writes pool state (tool source, dynamic agents, assignments) to a
    JSON file.

  PersistentAgentPoolWithSupervisor
    Extends AgentPoolWithSupervisor to auto-save pool state after every
    mutation and restore it on startup.

Restoration sequence (handled by build_system in main.py)
──────────────────────────────────────────────────────────
  1. Create PersistentAgentPoolWithSupervisor with a PoolStore + PersistentSaver
  2. Add initial agents  (_is_dynamic=False  → not persisted)
  3. Build ToolManager and call set_system_agents()
  4. Call pool.restore_state()   → re-registers tools, recreates dynamic
     agents, re-applies assignments from the JSON store.
"""

import ast
import json
import os
import sqlite3
import textwrap

from langgraph.checkpoint.sqlite import SqliteSaver

from .pool import AgentPoolWithSupervisor


def _extract_func_source(code: str, func_name: str) -> str:
    """Return source lines for a single named function from a multi-function code block."""
    dedented = textwrap.dedent(code)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return dedented
    lines = dedented.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    return dedented


# ── Conversation-history persistence ──────────────────────────────────────────

class PersistentSaver(SqliteSaver):
    """
    SqliteSaver that manages its own sqlite3 connection from a file path.

    Conversation history is stored in a single portable .db file that
    survives process restarts, is platform-independent, and can be
    inspected or pruned with any SQLite tool.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g. '.dynamate/conversations.db').
        The file and any missing parent directories are created automatically.
    """

    def __init__(self, db_path: str):
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        super().__init__(conn)
        self.setup()   # create 'checkpoints' and 'writes' tables if not present


# ── Pool-state persistence ─────────────────────────────────────────────────────

class PoolStore:
    """
    Reads and writes dynamic pool state to a JSON file plus per-tool .py files.

    Persisted state
    ───────────────
    pool_state.json  : tool names (list), dynamic_agents, assignments
    tools/<name>.py  : full source code for each registered tool (human-editable)

    Initial agents (shell_agent, compute_agent) are always rebuilt fresh by
    build_system() and are NOT included in dynamic_agents.
    """

    _EMPTY: dict = {"tools": [], "dynamic_agents": {}, "assignments": {}}

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.tools_dir = os.path.join(parent, "tools")
        os.makedirs(self.tools_dir, exist_ok=True)

    # ── JSON state ────────────────────────────────────────────────────────────

    def load(self) -> dict:
        if not os.path.exists(self.path):
            return dict(self._EMPTY)
        with open(self.path) as f:
            return json.load(f)

    def save(self, state: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(state, f, indent=2)

    # ── per-tool .py files ────────────────────────────────────────────────────

    def tool_path(self, name: str) -> str:
        """Absolute path to the .py file for *name*."""
        return os.path.join(self.tools_dir, f"{name}.py")

    def save_tool(self, name: str, source: str) -> None:
        """Write tool source to tools/<name>.py."""
        with open(self.tool_path(name), "w") as f:
            f.write(source)

    def delete_tool(self, name: str) -> None:
        """Remove tools/<name>.py if it exists."""
        p = self.tool_path(name)
        if os.path.exists(p):
            os.remove(p)

    def load_tools(self) -> dict:
        """Return {name: source} for every .py file in the tools/ directory."""
        tools = {}
        if os.path.isdir(self.tools_dir):
            for fname in sorted(os.listdir(self.tools_dir)):
                if fname.endswith(".py"):
                    with open(os.path.join(self.tools_dir, fname)) as f:
                        tools[fname[:-3]] = f.read()
        return tools


# ── Persistent pool ────────────────────────────────────────────────────────────

class PersistentAgentPoolWithSupervisor(AgentPoolWithSupervisor):
    """
    AgentPoolWithSupervisor that automatically saves and restores:
      • Registered tool source code
      • Dynamically added agent definitions
      • Tool-to-agent assignments

    Any tool registered or agent added via the ToolManager is immediately
    written to the JSON store and will be present in the next session.

    Parameters
    ----------
    supervisor_model  : LangChain chat model for the supervisor
    pool_store        : PoolStore — where to read/write JSON state
    supervisor_prompt : optional system prompt for the supervisor
    checkpointer      : checkpoint saver (default: PersistentSaver);
                        pass None to use an in-memory MemorySaver instead
    """

    def __init__(
        self,
        supervisor_model,
        pool_store: PoolStore,
        supervisor_prompt: str = None,
        checkpointer=None,
    ):
        super().__init__(supervisor_model, supervisor_prompt, checkpointer)
        self._pool_store         = pool_store
        self._source_registry:  dict = {}   # {tool_name: source_code_string}
        self._dynamic_agent_names: set = set()
        self._loading:           bool = False  # suppress auto-save during restore

    # ── overrides ─────────────────────────────────────────────────────────────

    def add_agent(
        self,
        name: str,
        model,
        base_tools: list = None,
        system_prompt: str = None,
        _is_dynamic: bool = True,     # False for initial shell/compute agents
    ) -> "PersistentAgentPoolWithSupervisor":
        super().add_agent(name, model, base_tools=base_tools, system_prompt=system_prompt)
        # Record the model name string for later restoration
        self._agents[name]["model_name"] = getattr(
            model, "model_name", getattr(model, "model", str(model))
        )
        if _is_dynamic:
            self._dynamic_agent_names.add(name)
            if not self._loading:
                self._autosave()   # only save when pool state actually changed
        return self

    def register_tool_from_code(self, code: str) -> str:
        result = super().register_tool_from_code(code)
        # Track source code for each newly registered tool and write .py file
        if "Registered:" in result:
            namespace: dict = {}
            try:
                exec(textwrap.dedent(code), namespace)
                for name, obj in namespace.items():
                    if callable(obj) and not name.startswith("_") and name in self._tool_registry:
                        func_source = _extract_func_source(code, name)
                        self._source_registry[name] = func_source
                        self._pool_store.save_tool(name, func_source)
            except Exception:
                pass
        if not self._loading:
            self._autosave()
        return result

    def assign_tool(self, tool_name: str, agent_name: str) -> str:
        result = super().assign_tool(tool_name, agent_name)
        if not self._loading:
            self._autosave()
        return result

    def remove_tool(self, tool_name: str) -> str:
        result = super().remove_tool(tool_name)
        if "not in registry" not in result:
            self._source_registry.pop(tool_name, None)
            self._pool_store.delete_tool(tool_name)
            if not self._loading:
                self._autosave()
        return result

    def remove_agent(self, agent_name: str) -> str:
        result = super().remove_agent(agent_name)
        if "not found" not in result:
            self._dynamic_agent_names.discard(agent_name)
            if not self._loading:
                self._autosave()
        return result

    # ── restore ───────────────────────────────────────────────────────────────

    def restore_state(self, model_factory) -> None:
        """
        Reload previously saved pool state.

        Call this AFTER set_system_agents() so the supervisor exists before
        dynamic agents are added (which triggers a supervisor rebuild).

        Parameters
        ----------
        model_factory : callable(model_name: str) → LangChain chat model
        """
        state     = self._pool_store.load()
        tool_sources = self._pool_store.load_tools()
        n_tools   = len(tool_sources)
        n_agents  = len(state.get("dynamic_agents", {}))
        n_assigns = sum(len(v) for v in state.get("assignments", {}).values())

        if n_tools == 0 and n_agents == 0 and n_assigns == 0:
            print("No saved state found — starting fresh.")
            return

        self._loading = True
        try:
            # 1. Re-register tools from .py files (new format) or JSON dict (old format)
            raw_tools = state.get("tools", {})
            if isinstance(raw_tools, dict) and raw_tools:
                # Old format: migrate source from JSON into .py files on the fly
                for name, src in raw_tools.items():
                    self._pool_store.save_tool(name, src)
                tool_sources = self._pool_store.load_tools()
            seen_sources: set = set()
            for name, source in tool_sources.items():
                if source not in seen_sources:
                    self.register_tool_from_code(source)
                    seen_sources.add(source)

            # 2. Recreate dynamic agents not already in pool
            for name, info in state.get("dynamic_agents", {}).items():
                if name not in self._agents:
                    model = model_factory(info.get("model_name", "gpt-4o-mini"))
                    self.add_agent(
                        name=name,
                        model=model,
                        base_tools=[],
                        system_prompt=info.get("system_prompt", ""),
                        _is_dynamic=True,
                    )

            # 3. Re-apply tool assignments
            for agent_name, tool_names in state.get("assignments", {}).items():
                if agent_name not in self._agents:
                    continue
                current = [t.name for t in self._agents[agent_name]["extra_tools"]]
                for tool_name in tool_names:
                    if tool_name in self._tool_registry and tool_name not in current:
                        self.assign_tool(tool_name, agent_name)
        finally:
            self._loading = False

        print(
            f"State restored — "
            f"{n_tools} tool(s), {n_agents} dynamic agent(s), {n_assigns} assignment(s)."
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _autosave(self) -> None:
        state = {
            "tools": list(self._source_registry.keys()),
            "dynamic_agents": {
                name: {
                    "system_prompt": self._agents[name].get("system_prompt") or "",
                    "model_name":    self._agents[name].get("model_name", "gpt-4o-mini"),
                }
                for name in self._dynamic_agent_names
                if name in self._agents
            },
            "assignments": {
                name: [t.name for t in self._agents[name]["extra_tools"]]
                for name in self._agents
                if self._agents[name]["extra_tools"]
            },
        }
        try:
            self._pool_store.save(state)
        except Exception as e:
            print(f"[PersistentPool] Warning: could not save state: {e}")
