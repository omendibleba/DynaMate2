#!/usr/bin/env python3
"""
Test: Dynamic agent addition  (notebook Test C)
────────────────────────────────────────────────
Verifies that an entirely new agent can be created at runtime via the
ToolManager and immediately routed to by the Supervisor.

Sequence tested:
  1. Create 'unit_conversion_agent' via add_agent_to_pool
     → supervisor is rebuilt to include it
  2. Register two unit-conversion functions and assign both to the new agent
  3. Confirm the new agent appears in the pool and has its tools
  4. Route a computation request to the new agent

NOTE ON PERSISTENCE
───────────────────
Agents added at runtime exist only for the duration of the process.
When this script exits, the pool resets to its initial state on the next
run (shell_agent + compute_agent only).  For persistence across runs you
would need to serialize the tool source to disk and replay it on startup,
and use a persistent checkpointer such as SqliteSaver.

Run:
    python tests/test_add_agent.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dotenv; dotenv.load_dotenv()

from _setup import build_pool, print_pool_state
from dynamate import pretty_print_messages


def test_C_add_agent_and_use():
    print("\n" + "="*60)
    print("Test C — dynamically add an agent and route to it")
    print("="*60)

    pool = build_pool()
    initial_agents = pool.list_agents()
    config = {"configurable": {"thread_id": "test_c"}}

    code = '''
def angstrom_to_bohr(angstrom: float) -> str:
    """Convert a length in Angstrom to Bohr radii."""
    return f"{angstrom} Å = {angstrom * 1.8897259886:.6f} a0"

def eV_to_hartree(eV: float) -> str:
    """Convert an energy in eV to Hartree atomic units."""
    return f"{eV} eV = {eV / 27.211396:.8f} Ha"
'''

    prompt = f"""\
Please do the following steps in order:

1. Create a new agent called 'unit_conversion_agent' whose job is to convert \
between physical units used in molecular simulation.

2. Register these two Python functions and assign both to unit_conversion_agent:
{code}

3. List all agents and confirm unit_conversion_agent has both tools.

4. Use unit_conversion_agent to convert 3.5 Angstrom to Bohr radii.
"""

    for chunk in pool.supervisor.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        recursion_limit=25,
    ):
        pretty_print_messages(chunk, last_message=True)

    print_pool_state(pool, "Pool state after Test C")

    # Assertions
    assert "unit_conversion_agent" in pool.list_agents(), \
        f"unit_conversion_agent not found in pool. Got: {pool.list_agents()}"

    new_agent_extra = [t.name for t in pool._agents["unit_conversion_agent"]["extra_tools"]]
    assert "angstrom_to_bohr" in new_agent_extra, \
        f"angstrom_to_bohr not assigned to unit_conversion_agent. Got: {new_agent_extra}"
    assert "eV_to_hartree" in new_agent_extra, \
        f"eV_to_hartree not assigned to unit_conversion_agent. Got: {new_agent_extra}"

    # Original agents should be unmodified
    for original in initial_agents:
        extra = [t.name for t in pool._agents[original]["extra_tools"]]
        assert "angstrom_to_bohr" not in extra, \
            f"angstrom_to_bohr incorrectly leaked into {original}"

    print("✓ Test C passed.")


if __name__ == "__main__":
    test_C_add_agent_and_use()
    print("\nAll dynamic agent addition tests passed.")
