"""
dynamate.tool_manager
─────────────────────
Factory function that builds the ToolManager agent.

The ToolManager is the single agent responsible for all pool management:
  • register_tool_from_code   – add functions from a code string
  • register_tool_from_file   – add functions from a .py file
  • assign_tool_to_agent      – assign a registered tool to a named agent
  • add_agent_to_pool         – create a new agent + rebuild supervisor
  • remove_tool_from_registry – unregister a tool and unassign it from all agents
  • remove_agent_from_pool    – remove a dynamic agent and rebuild supervisor
  • list_registered_tools     – inspect the global tool registry
  • list_agent_tools          – inspect one agent's tool set
  • list_agents               – list all agents in the pool
"""

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from .pool import AgentPoolWithSupervisor

_TOOL_MANAGER_PROMPT = (
    "You are the ToolManager. Your only job is to manage the agent pool:\n"
    "- Register new tools from code strings or files.\n"
    "- Assign registered tools to specific agents.\n"
    "- Add new agents to the pool when requested.\n"
    "- Remove tools from the registry (and unassign them from all agents).\n"
    "- Remove dynamic agents from the pool.\n"
    "- Report tool and agent status when asked.\n"
    "- If the user pastes Python function code and asks to add, register, or\n"
    "  use it as a tool, call register_tool_from_code with the exact code\n"
    "  string provided. Do not write files or use any shell commands.\n"
    "- When creating a new agent with add_agent_to_pool, always include in\n"
    "  its system_prompt: 'Always call your tools immediately and return the\n"
    "  complete numerical result. '\n"
    "Do not perform any domain work yourself."
)


def build_tool_manager_v2(pool: AgentPoolWithSupervisor, model):
    """
    Create the ToolManager agent backed by *pool*.

    All tools are thin wrappers around pool methods, bound via closure so
    they always operate on the live pool instance.

    Parameters
    ----------
    pool  : AgentPoolWithSupervisor
    model : any LangChain chat model

    Returns
    -------
    Compiled LangGraph ReAct agent (name="tool_manager")
    """

    @tool
    def register_tool_from_code(code: str) -> str:
        """Register one or more Python functions into the global tool registry.
        Pass the complete function definition(s) as a plain Python string.
        Each function's docstring becomes its tool description."""
        return pool.register_tool_from_code(code)

    @tool
    def register_tool_from_file(file_path: str) -> str:
        """Register Python functions from a .py file into the global tool registry.
        Pass the absolute or relative path to the file."""
        return pool.register_tool_from_file(file_path)

    @tool
    def assign_tool_to_agent(tool_name: str, agent_name: str) -> str:
        """Assign a registered tool to a specific agent by name.
        Only that agent is rebuilt; all others are unaffected."""
        return pool.assign_tool(tool_name, agent_name)

    _AGENT_EXECUTION_RULE = (
        "\nAlways call your available tools to complete every request and return "
        "the full result. Never transfer back to the supervisor before your tools "
        "have been executed and you have a concrete answer."
    )

    @tool
    def add_agent_to_pool(agent_name: str, system_prompt: str) -> str:
        """Create a new agent and add it to the pool.
        The supervisor is automatically rebuilt to include the new agent.
        The agent starts with no tools — use assign_tool_to_agent afterwards.

        agent_name    : unique snake_case name for the new agent.
        system_prompt : describes the agent's role and responsibilities."""
        try:
            pool.add_agent(
                name=agent_name,
                model=model,
                base_tools=[],
                system_prompt=system_prompt + _AGENT_EXECUTION_RULE,
            )
            return (
                f"Agent '{agent_name}' created. "
                f"Supervisor rebuilt. Current agents: {pool.list_agents()}"
            )
        except ValueError as e:
            return str(e)

    @tool
    def remove_tool_from_registry(tool_name: str) -> str:
        """Remove a tool from the global registry and unassign it from every agent.
        All agents that had this tool assigned are rebuilt automatically.
        Use list_registered_tools first to confirm the tool name."""
        return pool.remove_tool(tool_name)

    @tool
    def remove_agent_from_pool(agent_name: str) -> str:
        """Remove a dynamic agent from the pool and rebuild the supervisor.
        Only agents added at runtime (not the initial shell_agent or compute_agent)
        can be removed. Use list_agents first to confirm the agent name."""
        # Guard against removing system agents
        _PROTECTED = {"shell_agent", "compute_agent", "tool_manager"}
        if agent_name in _PROTECTED:
            return f"Cannot remove protected agent '{agent_name}'."
        return pool.remove_agent(agent_name)

    @tool
    def list_registered_tools() -> str:
        """List all tools currently in the global registry (available to assign)."""
        tools = pool.list_registered_tools()
        return (
            "Global registry:\n" + "\n".join(f"  - {t}" for t in tools)
            if tools else "Registry is empty."
        )

    @tool
    def list_agent_tools(agent_name: str) -> str:
        """List the base tools and assigned tools for a specific agent."""
        return pool.list_agent_tools(agent_name)

    @tool
    def list_agents() -> str:
        """List all agents currently in the pool."""
        return "Agents in pool:\n" + "\n".join(f"  - {a}" for a in pool.list_agents())

    return create_react_agent(
        model,
        tools=[
            register_tool_from_code,
            register_tool_from_file,
            assign_tool_to_agent,
            add_agent_to_pool,
            remove_tool_from_registry,
            remove_agent_from_pool,
            list_registered_tools,
            list_agent_tools,
            list_agents,
        ],
        name="tool_manager",
        prompt=_TOOL_MANAGER_PROMPT,
    )
