#!/usr/bin/env python3
"""
DynaMate — graph & pool state inspector
─────────────────────────────────────────
Builds the system and prints a snapshot of:
  • All agents in the pool and their tools
  • The global tool registry
  • The supervisor graph as a Mermaid diagram (text)
  • Optionally saves the graph as a PNG image

Usage
─────
  # Print Mermaid diagram to terminal
  python misc/show_graph.py

  # Save PNG to file
  python misc/show_graph.py --png graph.png

  # Inspect after adding an agent programmatically
  python misc/show_graph.py --add-agent chemistry_agent \
      --agent-prompt "You convert SMILES to molecular properties."
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from langchain_community.tools import ShellTool
from langchain_openai import ChatOpenAI

from dynamate import AgentPoolWithSupervisor, build_tool_manager_v2


# ── system builder (same as main.py) ──────────────────────────────────────────

_SUPERVISOR_PROMPT = (
    "You are the Supervisor managing a pool of agents.\n"
    "- tool_manager : registers tools, assigns them to agents, adds new agents.\n"
    "- shell_agent  : runs shell commands and handles file-system tasks.\n"
    "- compute_agent: performs calculations with its dynamically assigned tools.\n\n"
    "Assign work to one agent at a time."
)


def build_system(model_name: str = "gpt-4o-mini") -> AgentPoolWithSupervisor:
    model = ChatOpenAI(model=model_name, temperature=0.0)
    pool = AgentPoolWithSupervisor(
        supervisor_model=model,
        supervisor_prompt=_SUPERVISOR_PROMPT,
    )
    pool.add_agent(
        "shell_agent", model,
        base_tools=[ShellTool()],
        system_prompt="You are a shell agent. Execute shell commands to answer requests.",
    )
    pool.add_agent(
        "compute_agent", model,
        base_tools=[],
        system_prompt="You are a computation agent. Use assigned tools for calculations.",
    )
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])
    return pool


# ── display helpers ────────────────────────────────────────────────────────────

def print_pool_state(pool: AgentPoolWithSupervisor) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  POOL STATE")
    print(sep)
    print(f"  Agents in pool : {pool.list_agents()}")
    print(f"  Global registry: {pool.list_registered_tools() or '(empty)'}")
    print()
    for name in pool.list_agents():
        entry  = pool._agents[name]
        base   = [t.name for t in entry["base_tools"]]
        extra  = [t.name for t in entry["extra_tools"]]
        prompt = (entry["system_prompt"] or "")[:80]
        print(f"  [{name}]")
        print(f"    base tools    : {base or '(none)'}")
        print(f"    assigned tools: {extra or '(none)'}")
        print(f"    system_prompt : {prompt!r}{'...' if len(entry['system_prompt'] or '') > 80 else ''}")
        print()
    print(sep)


def print_mermaid(pool: AgentPoolWithSupervisor) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  SUPERVISOR GRAPH  (Mermaid)")
    print(sep)
    mermaid_str = pool.supervisor.get_graph().draw_mermaid()
    print(mermaid_str)
    print(sep)


def save_png(pool: AgentPoolWithSupervisor, path: str) -> None:
    print(f"\nGenerating PNG → {path} ...")
    try:
        png_bytes = pool.supervisor.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"Saved: {path}  ({len(png_bytes):,} bytes)")
    except Exception as e:
        print(f"[warn] PNG generation failed: {e}")
        print("       Falling back to Mermaid text output.")
        print_mermaid(pool)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the DynaMate supervisor graph and pool state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--png", metavar="FILE",
        help="Save the supervisor graph as a PNG image to FILE",
    )
    parser.add_argument(
        "--add-agent", metavar="NAME",
        help="Add an extra agent to the pool before displaying",
    )
    parser.add_argument(
        "--agent-prompt", metavar="PROMPT", default="",
        help="System prompt for the agent added via --add-agent",
    )
    args = parser.parse_args()

    print(f"Building system  [model={args.model}] ...")
    pool = build_system(args.model)

    if args.add_agent:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=args.model, temperature=0.0)
        pool.add_agent(
            name=args.add_agent,
            model=model,
            base_tools=[],
            system_prompt=args.agent_prompt or f"You are {args.add_agent}.",
        )
        print(f"Added agent '{args.add_agent}' to pool.")

    print_pool_state(pool)

    if args.png:
        save_png(pool, args.png)
    else:
        print_mermaid(pool)


if __name__ == "__main__":
    main()
