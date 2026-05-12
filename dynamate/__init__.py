"""
dynamate
────────
Dynamic multi-agent framework built on LangGraph.

Quick start
───────────
    from dynamate import AgentPoolWithSupervisor, build_tool_manager_v2

    pool = AgentPoolWithSupervisor(model, supervisor_prompt="...")
    pool.add_agent("shell_agent", model, base_tools=[ShellTool()])
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])          # first supervisor build

    pool.supervisor.stream({"messages": [...]}, config=...)
"""

from .pool import AgentPool, AgentPoolWithSupervisor
from .tool_manager import build_tool_manager_v2
from .dynamic_agent import DynamicToolAgent
from .persistence import PersistentSaver, PoolStore, PersistentAgentPoolWithSupervisor
from .utils import pretty_print_messages, pretty_print_message
from .prompt_enhancer import PromptEnhancer, register_tools_from_prompt

__all__ = [
    "AgentPool",
    "AgentPoolWithSupervisor",
    "build_tool_manager_v2",
    "DynamicToolAgent",
    "PersistentSaver",
    "PoolStore",
    "PersistentAgentPoolWithSupervisor",
    "pretty_print_messages",
    "pretty_print_message",
    "PromptEnhancer",
    "register_tools_from_prompt",
]
