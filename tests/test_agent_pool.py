#!/usr/bin/env python3
"""
Test: AgentPool + ToolManager  (notebook Tests A & B)
──────────────────────────────────────────────────────
Verifies that tools can be registered globally and assigned selectively
to specific agents, with only the target agent being rebuilt each time.

NOTE ON PERSISTENCE
───────────────────
Registered tools and agent assignments are in-memory only.  They do not
survive process restarts.  Each test run starts with an empty registry
and agents that have only their base tools.

Run:
    python tests/test_agent_pool.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dotenv; dotenv.load_dotenv()

from _setup import build_pool, print_pool_state
from dynamate import pretty_print_messages


# ── Test A — register a tool from code and assign to compute_agent ────────────

def test_A_register_and_assign_from_code():
    print("\n" + "="*60)
    print("Test A — register tool from code string, assign to compute_agent")
    print("="*60)

    pool = build_pool()
    config = {"configurable": {"thread_id": "test_a"}}

    code = '''
def boltzmann_energy(temperature_K: float) -> str:
    """Compute thermal energy kT in eV for a given temperature in Kelvin."""
    kT = 8.617333e-5 * temperature_K
    return f"kT at {temperature_K} K = {kT:.6f} eV"
'''

    prompt = (
        f"Please register the following Python function and assign it to "
        f"compute_agent:\n\n{code}\n\n"
        "Then use compute_agent to calculate kT at 300 K."
    )

    for chunk in pool.supervisor.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        recursion_limit=15,
    ):
        pretty_print_messages(chunk, last_message=True)

    print_pool_state(pool, "Pool state after Test A")

    # Ground-truth assertions on the pool directly
    assert "boltzmann_energy" in pool.list_registered_tools(), \
        "Tool not in global registry"
    compute_entry = pool._agents["compute_agent"]
    extra_names = [t.name for t in compute_entry["extra_tools"]]
    assert "boltzmann_energy" in extra_names, \
        f"Tool not assigned to compute_agent. Got extra_tools: {extra_names}"
    shell_entry = pool._agents["shell_agent"]
    shell_extra = [t.name for t in shell_entry["extra_tools"]]
    assert "boltzmann_energy" not in shell_extra, \
        "Tool incorrectly assigned to shell_agent"
    print("✓ Test A passed.")


# ── Test B — load tools from file, assign selectively to different agents ──────

def test_B_assign_selectively_from_file():
    print("\n" + "="*60)
    print("Test B — load tools from file, assign to different agents")
    print("="*60)

    pool = build_pool()
    config = {"configurable": {"thread_id": "test_b"}}

    sample_file = "/tmp/dynamate_unit_tools.py"
    with open(sample_file, "w") as f:
        f.write('''\
def angstrom_to_bohr(angstrom: float) -> str:
    """Convert a length in Angstrom to Bohr radii."""
    return f"{angstrom} Å = {angstrom * 1.8897259886:.6f} a0"

def eV_to_hartree(eV: float) -> str:
    """Convert an energy in eV to Hartree atomic units."""
    return f"{eV} eV = {eV / 27.211396:.8f} Ha"
''')

    prompt = (
        f"Load all tools from {sample_file}. "
        "Assign angstrom_to_bohr to compute_agent and eV_to_hartree to shell_agent. "
        "Then list the tools of each agent to confirm."
    )

    for chunk in pool.supervisor.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        recursion_limit=20,
    ):
        pretty_print_messages(chunk, last_message=True)

    print_pool_state(pool, "Pool state after Test B")

    compute_extra = [t.name for t in pool._agents["compute_agent"]["extra_tools"]]
    shell_extra   = [t.name for t in pool._agents["shell_agent"]["extra_tools"]]

    assert "angstrom_to_bohr" in compute_extra, \
        f"angstrom_to_bohr not in compute_agent. Got: {compute_extra}"
    assert "eV_to_hartree" in shell_extra, \
        f"eV_to_hartree not in shell_agent. Got: {shell_extra}"
    assert "angstrom_to_bohr" not in shell_extra, \
        "angstrom_to_bohr incorrectly assigned to shell_agent"
    assert "eV_to_hartree" not in compute_extra, \
        "eV_to_hartree incorrectly assigned to compute_agent"
    print("✓ Test B passed.")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_A_register_and_assign_from_code()
    test_B_assign_selectively_from_file()
    print("\nAll AgentPool + ToolManager tests passed.")
