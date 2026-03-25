#!/usr/bin/env python3
"""
DynaMate — persistent dynamic multi-agent CLI
──────────────────────────────────────────────
Assembles an AgentPoolWithSupervisor and restores all tools, agents,
and assignments from the previous session automatically.

State is stored in --state-dir (default: .dynamate/ in the project root):
  pool_state.json    — tool source, dynamic agents, assignments
  conversations.*    — shelve files with full conversation history

Usage
─────
  Interactive REPL (state auto-loaded and auto-saved):
      python main.py

  Single prompt:
      python main.py --prompt "Register a boltzmann_energy tool and assign to compute_agent"

  Custom state directory:
      python main.py --state-dir /scratch/my_session

  Different model:
      python main.py --model gpt-4o

  Named conversation thread:
      python main.py --thread-id project-abc

  Verbose (print all messages per chunk, not just last):
      python main.py --verbose
"""

import argparse
import os
import sys
import uuid

import dotenv
from langchain_community.tools import ShellTool
from langchain_openai import ChatOpenAI

from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
    pretty_print_messages,
)

# ── constants ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_STATE_DIR = os.path.join(_PROJECT_ROOT, ".dynamate")

_SUPERVISOR_PROMPT = (
    "You are the Supervisor managing a pool of agents.\n"
    "- tool_manager : registers tools, assigns them to agents, and adds new agents.\n"
    "- shell_agent  : runs shell commands and handles file-system tasks.\n"
    "- compute_agent: performs calculations with its dynamically assigned tools.\n\n"
    "Routing rules:\n"
    "  • Add/register/assign tools or add new agents → tool_manager.\n"
    "  • Shell or file-system tasks → shell_agent.\n"
    "  • Computation → compute_agent (only if it has the required tool).\n"
    "  • If an agent for a task does not exist yet, ask tool_manager to create it first.\n"
    "Assign work to one agent at a time."
)

# ── system factory ─────────────────────────────────────────────────────────────

def make_model(model_name: str):
    return ChatOpenAI(model=model_name, temperature=0.0)


def build_system(
    model_name: str,
    state_dir: str,
) -> tuple[PersistentAgentPoolWithSupervisor, PromptEnhancer]:
    """
    Assemble the full multi-agent system and restore previous session state.

    Steps
    ─────
    1. Create PersistentSaver (conversation history) and PoolStore (pool state).
    2. Build PersistentAgentPoolWithSupervisor with initial domain agents.
    3. Build ToolManager and set it as a system agent (triggers first supervisor build).
    4. Restore previous state: tools, dynamic agents, assignments.
    """
    os.makedirs(state_dir, exist_ok=True)

    saver = PersistentSaver(
        db_path=os.path.join(state_dir, "conversations")
    )
    pool_store = PoolStore(
        path=os.path.join(state_dir, "pool_state.json")
    )
    model = make_model(model_name)

    # Step 1 — create pool (no supervisor yet)
    pool = PersistentAgentPoolWithSupervisor(
        supervisor_model=model,
        pool_store=pool_store,
        supervisor_prompt=_SUPERVISOR_PROMPT,
        checkpointer=saver,
    )

    # Step 2 — add initial domain agents (_is_dynamic=False: never persisted)
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

    # Step 3 — build ToolManager and trigger first supervisor build
    tool_manager = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tool_manager])

    # Step 4 — restore previous session state
    pool.restore_state(model_factory=make_model)

    enhancer = PromptEnhancer(model=model, pool=pool)
    return pool, enhancer

# ── REPL ──────────────────────────────────────────────────────────────────────

def run_interactive(
    pool: PersistentAgentPoolWithSupervisor,
    enhancer: PromptEnhancer,
    thread_id: str,
    verbose: bool,
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\nDynaMate session  [thread: {thread_id}]")
    print("Agents:", pool.list_agents())
    print("Tools in registry:", pool.list_registered_tools() or "(none)")
    print("Type 'status' to inspect the pool, 'exit' to quit.\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            break

        if user_input.lower() == "status":
            _print_status(pool)
            continue

        try:
            enhanced_input = enhancer.enhance(user_input)
            if enhanced_input != user_input:
                print(f"[enhancer] {enhanced_input}\n")
            for chunk in pool.supervisor.stream(
                {"messages": [{"role": "user", "content": enhanced_input}]},
                config=config,
                recursion_limit=25,
            ):
                pretty_print_messages(chunk, last_message=not verbose)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)


def run_single(
    pool: PersistentAgentPoolWithSupervisor,
    enhancer: PromptEnhancer,
    prompt: str,
    thread_id: str,
    verbose: bool,
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    enhanced_prompt = enhancer.enhance(prompt)
    if enhanced_prompt != prompt:
        print(f"[enhancer] {enhanced_prompt}\n")
    for chunk in pool.supervisor.stream(
        {"messages": [{"role": "user", "content": enhanced_prompt}]},
        config=config,
        recursion_limit=25,
    ):
        pretty_print_messages(chunk, last_message=not verbose)


def _print_status(pool: PersistentAgentPoolWithSupervisor) -> None:
    print("\n── Pool status ──────────────────────────────────")
    print(f"  Agents   : {pool.list_agents()}")
    print(f"  Registry : {pool.list_registered_tools() or '(empty)'}")
    for name in pool.list_agents():
        print(f"  {pool.list_agent_tools(name)}")
    print("─────────────────────────────────────────────────\n")

# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DynaMate — persistent dynamic multi-agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--state-dir", default=_DEFAULT_STATE_DIR,
        help=f"Directory for persisted state (default: {_DEFAULT_STATE_DIR})",
    )
    parser.add_argument(
        "--thread-id", default=None,
        help="Conversation thread ID — reuse to continue a previous conversation",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Run a single prompt and exit (non-interactive)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print all messages per chunk, not just the last",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print pool status and exit without starting a session",
    )
    args = parser.parse_args()

    dotenv.load_dotenv()
    thread_id = args.thread_id or str(uuid.uuid4())[:8]

    print(f"Building system  [model={args.model}, state={args.state_dir}] ...")
    pool, enhancer = build_system(args.model, args.state_dir)
    print(f"Ready.\n")

    if args.status:
        _print_status(pool)
        return

    if args.prompt:
        run_single(pool, enhancer, args.prompt, thread_id, args.verbose)
    else:
        run_interactive(pool, enhancer, thread_id, args.verbose)


if __name__ == "__main__":
    main()
