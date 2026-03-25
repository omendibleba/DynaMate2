#!/usr/bin/env python3
"""
Test: DynamicToolAgent  (notebook Tests 1 & 2)
───────────────────────────────────────────────
Standalone single-agent that hot-swaps its tool list at runtime.

NOTE ON PERSISTENCE
───────────────────
Tools added during a run are held in memory only.  When this process
exits, all registered tools are lost.  Each new run starts with only the
base tools supplied at construction time.

Run:
    python tests/test_dynamic_agent.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dotenv; dotenv.load_dotenv()

from _setup import make_model, print_pool_state
from dynamate import DynamicToolAgent, pretty_print_messages


# ── shared fixture ─────────────────────────────────────────────────────────────

def make_agent():
    return DynamicToolAgent(
        model=make_model(),
        system_prompt=(
            "You are a helpful assistant. When asked to add a tool, use "
            "add_tool_from_code or add_tool_from_file. Then use the tool if asked."
        ),
    )


# ── Test 1 — add a tool from an inline code string ────────────────────────────

def test_1_add_tool_from_code():
    print("\n" + "="*60)
    print("Test 1 — add tool from inline code string")
    print("="*60)

    agent = make_agent()

    code = '''
def boltzmann_energy(temperature_K: float) -> str:
    """Compute thermal energy kT in eV for a given temperature in Kelvin."""
    kT = 8.617333e-5 * temperature_K
    return f"kT at {temperature_K} K = {kT:.6f} eV"
'''

    prompt = (
        f"Please add the following Python function as a tool and then "
        f"use it at 300 K:\n\n{code}"
    )

    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
        pretty_print_messages(chunk, last_message=True)

    registered = [t.name for t in agent.user_tools]
    assert "boltzmann_energy" in registered, f"Tool not registered. Got: {registered}"
    print(f"\n✓ boltzmann_energy registered. All user tools: {registered}")


# ── Test 2 — add tools from a .py file ────────────────────────────────────────

def test_2_add_tool_from_file():
    print("\n" + "="*60)
    print("Test 2 — add tools from a .py file")
    print("="*60)

    sample_file = "/tmp/dynamate_test_tools.py"
    with open(sample_file, "w") as f:
        f.write('''\
def angstrom_to_bohr(angstrom: float) -> str:
    """Convert a length in Angstrom to Bohr radii."""
    return f"{angstrom} Å = {angstrom * 1.8897259886:.6f} a0"

def eV_to_hartree(eV: float) -> str:
    """Convert an energy in eV to Hartree atomic units."""
    return f"{eV} eV = {eV / 27.211396:.8f} Ha"
''')

    agent = make_agent()

    prompt = (
        f"Add all the tools defined in {sample_file}, "
        "then list all available tools, "
        "and finally convert 2.5 Angstrom to Bohr radii."
    )

    for chunk in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
        pretty_print_messages(chunk, last_message=True)

    registered = [t.name for t in agent.user_tools]
    for expected in ("angstrom_to_bohr", "eV_to_hartree"):
        assert expected in registered, f"Tool '{expected}' not registered. Got: {registered}"
    print(f"\n✓ File tools registered. All user tools: {registered}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_1_add_tool_from_code()
    test_2_add_tool_from_file()
    print("\nAll DynamicToolAgent tests passed.")
