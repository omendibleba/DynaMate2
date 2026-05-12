#!/usr/bin/env python3
"""
Test: Persistence across sessions
───────────────────────────────────
Verifies that tools, dynamic agents, and assignments survive process
restarts by simulating three independent sessions against the same store.

Session 1 — register a tool, add a dynamic agent, assign the tool.
Session 2 — restore from store, confirm everything is present,
             then add a second tool and assign it to a different agent.
Session 3 — restore from store again, confirm both sessions' state is intact.

Run:
    python tests/test_persistence.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import dotenv; dotenv.load_dotenv()

from langchain_community.tools import ShellTool
from langchain_openai import ChatOpenAI

from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    build_tool_manager_v2,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def fresh_pool(model, store: PoolStore) -> PersistentAgentPoolWithSupervisor:
    """Build a pool from scratch — mirrors what main.py does on startup."""
    pool = PersistentAgentPoolWithSupervisor(model, store)
    pool.add_agent(
        "shell_agent", model,
        base_tools=[ShellTool()],
        system_prompt="Shell agent.",
        _is_dynamic=False,
    )
    pool.add_agent(
        "compute_agent", model,
        base_tools=[],
        system_prompt="Compute agent.",
        _is_dynamic=False,
    )
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])
    return pool


# ── PoolStore round-trip ───────────────────────────────────────────────────────

def test_pool_store_roundtrip(tmp: str) -> None:
    print("\n── PoolStore JSON round-trip ──")
    store = PoolStore(os.path.join(tmp, "pool_rt.json"))
    state = {
        "tools": {"my_tool": "def my_tool(): pass"},
        "dynamic_agents": {"agent_x": {"system_prompt": "hi", "model_name": "gpt-4o-mini"}},
        "assignments": {"compute_agent": ["my_tool"]},
    }
    store.save(state)
    loaded = store.load()
    assert loaded == state, f"Mismatch: {loaded}"
    print("  ✓ save → load round-trip correct")


# ── PersistentSaver init ───────────────────────────────────────────────────────

def test_persistent_saver_init(tmp: str) -> None:
    print("\n── PersistentSaver init ──")
    saver = PersistentSaver(os.path.join(tmp, "conv"))
    assert saver is not None
    print("  ✓ PersistentSaver created without error")


# ── Three-session persistence ──────────────────────────────────────────────────

def test_three_sessions(tmp: str, model) -> None:
    store = PoolStore(os.path.join(tmp, "pool_state.json"))

    # ── Session 1 ─────────────────────────────────────────────────────────────
    print("\n── Session 1: register tool, add agent, assign ──")
    pool1 = fresh_pool(model, store)
    pool1.register_tool_from_code('''\
def boltzmann_energy(temperature_K: float) -> str:
    """Compute thermal energy kT in eV for a given temperature in Kelvin."""
    kT = 8.617333e-5 * temperature_K
    return f"kT at {temperature_K} K = {kT:.6f} eV"
''')
    pool1.add_agent(
        "compute_v2", model,
        base_tools=[], system_prompt="Advanced computation agent.",
        _is_dynamic=True,
    )
    pool1.assign_tool("boltzmann_energy", "compute_v2")

    saved = store.load()
    assert "boltzmann_energy" in saved["tools"], "Tool not in JSON store"
    assert "compute_v2" in saved["dynamic_agents"], "Dynamic agent not in JSON store"
    assert "boltzmann_energy" in saved["assignments"]["compute_v2"], "Assignment not in JSON store"
    print("  ✓ Session 1 state saved correctly")

    # ── Session 2: restore + extend ───────────────────────────────────────────
    print("\n── Session 2: restore, add second tool, assign to different agent ──")
    pool2 = fresh_pool(model, store)
    pool2.restore_state(model_factory=lambda n: ChatOpenAI(model=n, temperature=0.0))

    assert "boltzmann_energy" in pool2.list_registered_tools(), "Tool not restored"
    assert "compute_v2" in pool2.list_agents(), "Dynamic agent not restored"
    restored_extra = [t.name for t in pool2._agents["compute_v2"]["extra_tools"]]
    assert "boltzmann_energy" in restored_extra, "Assignment not restored"
    print("  ✓ Session 1 state fully restored")

    # Extend: add a new tool and assign to a different agent
    pool2.register_tool_from_code('''\
def angstrom_to_bohr(angstrom: float) -> str:
    """Convert a length in Angstrom to Bohr radii."""
    return f"{angstrom} Å = {angstrom * 1.8897259886:.6f} a0"
''')
    pool2.assign_tool("angstrom_to_bohr", "compute_agent")
    print("  ✓ Session 2 extended with angstrom_to_bohr → compute_agent")

    # ── Session 3: restore both sessions' state ────────────────────────────────
    print("\n── Session 3: restore and verify cumulative state ──")
    pool3 = fresh_pool(model, store)
    pool3.restore_state(model_factory=lambda n: ChatOpenAI(model=n, temperature=0.0))

    # From session 1
    assert "boltzmann_energy" in pool3.list_registered_tools(), "boltzmann_energy lost"
    assert "compute_v2" in pool3.list_agents(), "compute_v2 lost"
    assert "boltzmann_energy" in [
        t.name for t in pool3._agents["compute_v2"]["extra_tools"]
    ], "compute_v2 assignment lost"

    # From session 2
    assert "angstrom_to_bohr" in pool3.list_registered_tools(), "angstrom_to_bohr lost"
    assert "angstrom_to_bohr" in [
        t.name for t in pool3._agents["compute_agent"]["extra_tools"]
    ], "compute_agent assignment lost"

    # Shell agent untouched across all sessions
    assert pool3._agents["shell_agent"]["extra_tools"] == [], \
        "shell_agent should have no extra tools"

    print("  ✓ Cumulative state from sessions 1 & 2 fully restored in session 3")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    with tempfile.TemporaryDirectory() as tmp:
        test_pool_store_roundtrip(tmp)
        test_persistent_saver_init(tmp)
        test_three_sessions(tmp, model)

    print("\nAll persistence tests passed.")
