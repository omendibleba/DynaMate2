"""
dynamate.pool
─────────────
AgentPool          – shared registry of named agents and tools.
AgentPoolWithSupervisor – extends AgentPool; owns and rebuilds the supervisor
                          whenever an agent is added.

Rebuild cost summary
────────────────────
  pool.add_agent(...)    → rebuilds that agent  +  rebuilds supervisor
  pool.assign_tool(...)  → rebuilds only the target agent (supervisor untouched)
  pool.supervisor        → property; always returns the current compiled graph
"""

import inspect
import textwrap

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor


def _make_model_selector(base_model, domain_tools: list):
    """
    Build a state-aware model callable for create_react_agent.

    Forces tool_choice="required" only on an agent's FIRST inference after
    being activated by the supervisor — structurally preventing the model
    from responding with plain text describing an action instead of taking
    it (observed: "I will now call smiles_to_xyz... Executing now." with no
    actual tool call, which LangGraph treats as a completed turn, causing an
    unearned handoff back to the supervisor and a fabricated success
    report). Reverts to normal tool_choice="auto" after any real tool call,
    so the agent can still choose to stop on its own once genuine work is
    done — avoiding a forced-tool-call loop.
    """

    def _select_model(state, runtime):
        if not domain_tools:
            return base_model

        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        is_first_turn = (
            last is None
            or isinstance(last, HumanMessage)
            or (isinstance(last, ToolMessage) and (last.name or "").startswith("transfer_to_"))
        )
        if is_first_turn:
            return base_model.bind_tools(domain_tools, tool_choice="required")
        return base_model.bind_tools(domain_tools)

    return _select_model


class AgentPool:
    """
    Shared registry of named agents and a global tool store.

    Agents start with a fixed set of base tools.  Additional tools can be
    registered globally and then assigned to specific agents by name.
    Only the target agent is rebuilt on each assignment.
    """

    def __init__(self):
        # {name: {"agent", "model", "base_tools", "extra_tools", "system_prompt"}}
        self._agents: dict = {}
        # {tool_name: StructuredTool}
        self._tool_registry: dict = {}

    # ── agent management ──────────────────────────────────────────────────────

    def add_agent(
        self,
        name: str,
        model,
        base_tools: list = None,
        system_prompt: str = None,
    ) -> "AgentPool":
        """Register a new named agent.  Raises ValueError if name already exists."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists.")
        self._agents[name] = {
            "model": model,
            "base_tools": list(base_tools or []),
            "extra_tools": [],
            "system_prompt": system_prompt,        # persisted; user-visible
            "base_system_prompt": system_prompt,   # never overwritten; source of truth
        }
        self._rebuild_agent(name)
        return self

    def get_agent(self, name: str):
        """Return the compiled LangGraph graph for a named agent."""
        return self._agents[name]["agent"]

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    # ── tool registration ─────────────────────────────────────────────────────

    def register_tool_from_code(self, code: str) -> str:
        """Execute *code* and store every new top-level callable in the registry."""
        namespace: dict = {}
        try:
            exec(textwrap.dedent(code), namespace)
        except Exception as e:
            return f"Syntax/execution error: {e}"

        added, skipped = [], []
        for name, obj in namespace.items():
            if name.startswith("_"):
                continue
            # Only register plain functions defined in the exec'd code, not
            # imported callables (classes, imported functions, modules, etc.).
            # Functions defined by exec() have __module__ == None because the
            # namespace dict has no __name__ key.
            if not inspect.isfunction(obj) or obj.__module__ is not None:
                continue
            # Require a docstring — stubs the LLM invents have no docstring,
            # and an empty description makes the tool invisible to routing.
            if not (obj.__doc__ or "").strip():
                continue
            if name in self._tool_registry:
                skipped.append(name)
                continue
            try:
                self._tool_registry[name] = StructuredTool.from_function(obj)
                added.append(name)
            except Exception as e:
                return f"Could not convert '{name}' to a tool: {e}"

        if not added and not skipped:
            return "No callable functions found in the provided code."

        parts = []
        if added:
            parts.append(f"Registered: {', '.join(added)}")
        if skipped:
            parts.append(f"Already registered (skipped): {', '.join(skipped)}")
        return " | ".join(parts)

    def register_tool_from_file(self, file_path: str) -> str:
        """Load function definitions from a .py file into the global registry."""
        try:
            with open(file_path) as f:
                return self.register_tool_from_code(f.read())
        except FileNotFoundError:
            return f"File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {e}"

    def list_registered_tools(self) -> list[str]:
        return list(self._tool_registry.keys())

    # ── tool assignment ───────────────────────────────────────────────────────

    def assign_tool(self, tool_name: str, agent_name: str) -> str:
        """Assign a registered tool to an agent.  Only that agent is rebuilt."""
        if tool_name not in self._tool_registry:
            return f"Tool '{tool_name}' not in registry. Register it first."
        if agent_name not in self._agents:
            return f"Agent '{agent_name}' not found. Available: {self.list_agents()}"

        entry = self._agents[agent_name]
        if tool_name in [t.name for t in entry["extra_tools"]]:
            return f"Tool '{tool_name}' already assigned to '{agent_name}'."

        entry["extra_tools"].append(self._tool_registry[tool_name])
        self._rebuild_agent(agent_name)
        return f"Assigned '{tool_name}' to '{agent_name}' and rebuilt it."

    def list_agent_tools(self, agent_name: str) -> str:
        if agent_name not in self._agents:
            return f"Agent '{agent_name}' not found."
        entry = self._agents[agent_name]
        return (
            f"Agent '{agent_name}' tools:\n"
            f"  base    : {[t.name for t in entry['base_tools']]}\n"
            f"  assigned: {[t.name for t in entry['extra_tools']]}"
        )

    # ── removal ───────────────────────────────────────────────────────────────

    def remove_tool(self, tool_name: str) -> str:
        """Remove a tool from the registry and unassign it from all agents."""
        if tool_name not in self._tool_registry:
            return f"Tool '{tool_name}' not in registry."
        del self._tool_registry[tool_name]
        affected = []
        for name, entry in self._agents.items():
            before = len(entry["extra_tools"])
            entry["extra_tools"] = [t for t in entry["extra_tools"] if t.name != tool_name]
            if len(entry["extra_tools"]) < before:
                self._rebuild_agent(name)
                affected.append(name)
        msg = f"Removed tool '{tool_name}' from registry."
        if affected:
            msg += f" Unassigned from and rebuilt: {affected}."
        return msg

    def remove_agent(self, agent_name: str) -> str:
        """Remove an agent from the pool."""
        if agent_name not in self._agents:
            return f"Agent '{agent_name}' not found."
        del self._agents[agent_name]
        return f"Removed agent '{agent_name}'."

    # ── internal ──────────────────────────────────────────────────────────────

    def _rebuild_agent(self, name: str) -> None:
        entry   = self._agents[name]
        base_sp = entry.get("base_system_prompt") or entry.get("system_prompt") or ""

        # Auto-generate a tool-aware section from the currently assigned tools
        extra = entry["extra_tools"]
        base_tools = entry["base_tools"]
        all_domain_tools = base_tools + extra

        if extra:
            lines = ["\nYour assigned tools (call the matching one for each request):"]
            for t in extra:
                doc = (t.description or "").split("\n")[0].strip()
                lines.append(f"  - {t.name}: {doc}")
            tool_section = "\n".join(lines)
        else:
            tool_section = ""

        domain_tool_names = ", ".join(t.name for t in all_domain_tools) if all_domain_tools else "(none)"
        execution_rule = (
            "\nExecution rules:"
            "\n  * You have been activated by the supervisor to perform a specific task."
            f"\n  * Your domain tools are: {domain_tool_names}."
            "\n  * Your FIRST action MUST be to call one of these domain tools — NEVER transfer_back_to_supervisor first."
            "\n  * Do NOT write text saying you will call a tool or that you are executing it now."
            "\n    Only the tool-call mechanism itself counts as calling a tool — a sentence describing"
            "\n    the action is NOT the same as taking it, and produces no result at all."
            "\n  * If the message says 'it should use <tool_name>' or 'use <tool_name> to',"
            "\n    call that tool FIRST using the file paths from the"
            "\n    message directly as arguments. Do NOT call packmol_build_system,"
            "\n    download_mace_model, or smiles_to_xyz before it. The named tool handles"
            "\n    its own setup (e.g., box sizing) internally from the parameters provided."
            "\n  * If the message provides file paths for a simulation, call the simulation tool"
            "\n    directly with those paths and the box_size/temperature parameters — the simulation"
            "\n    tool sets up the periodic box internally; you do NOT need to call packmol first."
            "\n  * transfer_back_to_supervisor is ONLY allowed AFTER a domain tool has returned a result."
            "\n  * Calling transfer_back_to_supervisor without first calling a domain tool is a critical failure."
            "\n  * Do not ask for confirmation, do not explain — just call the domain tool immediately."
            "\n  * Never ask the user which tool to use or for implementation details about a tool."
            "\n    Match the task to the tool whose description best fits, then call it immediately."
            "\n  * Return the tool's output directly as your final answer."
        )

        effective_prompt = base_sp + tool_section + execution_rule

        entry["agent"] = create_react_agent(
            _make_model_selector(entry["model"], all_domain_tools),
            tools=entry["base_tools"] + entry["extra_tools"],
            name=name,
            prompt=effective_prompt,
        )


class AgentPoolWithSupervisor(AgentPool):
    """
    AgentPool that owns and automatically rebuilds the supervisor graph.

    Initialisation order (required)
    ────────────────────────────────
    1. pool = AgentPoolWithSupervisor(supervisor_model, ...)
    2. pool.add_agent(...)          # add domain agents
    3. tm   = build_tool_manager_v2(pool, model)
    4. pool.set_system_agents([tm]) # injects ToolManager → first supervisor build

    Checkpointing note
    ──────────────────
    Each supervisor rebuild creates a fresh MemorySaver by default, so
    conversation history resets after a new agent is added.  Pass a
    persistent checkpointer (e.g. SqliteSaver) to preserve history.
    """

    def __init__(
        self,
        supervisor_model,
        supervisor_prompt: str = None,
        checkpointer=None,
    ):
        super().__init__()
        self._supervisor_model  = supervisor_model
        self._supervisor_prompt = supervisor_prompt
        self._checkpointer      = checkpointer   # None → fresh MemorySaver each rebuild
        self._system_agents     = []             # always present in supervisor (e.g. ToolManager)
        self._supervisor        = None

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def supervisor(self):
        """Always returns the current compiled supervisor graph."""
        return self._supervisor

    def set_system_agents(self, agents: list) -> None:
        """
        Register agents that must always appear in the supervisor (e.g. ToolManager).
        Calling this triggers the first supervisor build.
        """
        self._system_agents = list(agents)
        self._rebuild_supervisor()

    def add_agent(
        self,
        name: str,
        model,
        base_tools: list = None,
        system_prompt: str = None,
    ) -> "AgentPoolWithSupervisor":
        """Add a domain agent to the pool and rebuild the supervisor."""
        super().add_agent(name, model, base_tools=base_tools, system_prompt=system_prompt)
        self._rebuild_supervisor()
        return self

    def remove_agent(self, agent_name: str) -> str:
        """Remove a domain agent and rebuild the supervisor."""
        result = super().remove_agent(agent_name)
        if "not found" not in result:
            self._rebuild_supervisor()
        return result

    def assign_tool(self, tool_name: str, agent_name: str) -> str:
        """
        Assign a tool to an agent and rebuild the supervisor.

        AgentPool.assign_tool() only rebuilds the target agent's own compiled
        graph (a fresh object each time) — it does NOT update the supervisor,
        whose compiled graph was wired to whatever agent object existed when
        _rebuild_supervisor() last ran. Without this override, an agent that
        already existed in the supervisor (e.g. one recreated by
        restore_state(), which always adds the agent before assigning its
        tools) permanently routes to a stale, tool-less copy of itself for
        the rest of the process's life — the agent can never actually call
        the tool, even though pool state/status correctly shows it assigned.
        """
        result = super().assign_tool(tool_name, agent_name)
        if "Assigned" in result:
            self._rebuild_supervisor()
        return result

    def remove_tool(self, tool_name: str) -> str:
        """Remove a tool from the registry and rebuild the supervisor if any
        agent currently in it was affected (see assign_tool for why this
        override is required — remove_tool has the identical staleness gap)."""
        result = super().remove_tool(tool_name)
        if "Unassigned from and rebuilt" in result:
            self._rebuild_supervisor()
        return result

    # ── internal ──────────────────────────────────────────────────────────────

    def _rebuild_supervisor(self) -> None:
        all_agents = self._system_agents + [
            self._agents[n]["agent"] for n in self._agents
        ]
        if not all_agents:
            return

        kwargs = {}
        if self._supervisor_prompt:
            # Build a live agent-description block so the supervisor always
            # knows every current agent, its tools, and its role.
            lines = ["Available agents (auto-updated):"]
            for sys_agent in self._system_agents:
                aname = getattr(sys_agent, "name", str(sys_agent))
                lines.append(f"  - {aname}")
            for aname, entry in self._agents.items():
                tools = (
                    [t.name for t in entry.get("base_tools", [])]
                    + [t.name for t in entry.get("extra_tools", [])]
                )
                tool_str = ", ".join(tools) if tools else "no tools yet"
                sp = (entry.get("system_prompt") or "").split("\n")[0][:80]
                lines.append(f"  - {aname}  [tools: {tool_str}]  — {sp}")
            agent_block = "\n".join(lines)
            kwargs["prompt"] = self._supervisor_prompt + "\n\n" + agent_block

        self._supervisor = create_supervisor(
            model=self._supervisor_model,
            agents=all_agents,
            add_handoff_back_messages=True,
            output_mode="full_history",
            **kwargs,
        ).compile(checkpointer=self._checkpointer or MemorySaver())
