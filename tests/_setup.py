"""
Shared setup helpers for all test scripts.
Adds the project root to sys.path so 'dynamate' can be imported
regardless of where the script is invoked from.
"""

import os
import sys

# Ensure project root is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool

from dynamate import (
    AgentPoolWithSupervisor,
    build_tool_manager_v2,
    DynamicToolAgent,
    pretty_print_messages,
)


def make_model(model_name: str = "gpt-4o-mini"):
    return ChatOpenAI(model=model_name, temperature=0.0)


def build_pool(model_name: str = "gpt-4o-mini") -> AgentPoolWithSupervisor:
    """
    Assemble the standard AgentPoolWithSupervisor used by most tests.
    Returns the pool; access supervisor via pool.supervisor.
    """
    model = make_model(model_name)

    pool = AgentPoolWithSupervisor(
        supervisor_model=model,
        supervisor_prompt=(
            "You are the Supervisor managing a pool of agents.\n"
            "- tool_manager : registers tools, assigns them to agents, adds new agents.\n"
            "- shell_agent  : runs shell commands and handles file-system tasks.\n"
            "- compute_agent: performs calculations with its dynamically assigned tools.\n\n"
            "Routing rules:\n"
            "  • Add/register/assign tools or add new agents → tool_manager.\n"
            "  • Shell or file-system tasks → shell_agent.\n"
            "  • Computation → compute_agent (only if it has the required tool).\n"
            "  • If an agent for a task doesn't exist, ask tool_manager to create it first.\n"
            "Assign work to one agent at a time."
        ),
    )
    pool.add_agent(
        "shell_agent", model,
        base_tools=[ShellTool()],
        system_prompt="You are a shell agent. Execute shell commands to answer requests.",
    )
    pool.add_agent(
        "compute_agent", model,
        base_tools=[],
        system_prompt="You are a computation agent. Use your assigned tools for calculations.",
    )
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])
    return pool


def print_pool_state(pool: AgentPoolWithSupervisor, label: str = "Pool state") -> None:
    print(f"\n── {label} ──")
    print(f"  Agents   : {pool.list_agents()}")
    print(f"  Registry : {pool.list_registered_tools()}")
    for name in pool.list_agents():
        print(f"  {pool.list_agent_tools(name)}")
    print()
