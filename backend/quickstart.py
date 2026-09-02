"""
backend.quickstart
────────────────────
Quick-start prompt strings mirroring the DynaMate2 tutorial notebook
(tutorials/DynaMate2_tutorial.ipynb), ported verbatim from app.py so the
paths/prompts stay in sync with the tutorial .py files rather than being
duplicated as frontend static strings.
"""

import os

from backend.state import TUTORIALS_DIR


def _tut(relpath: str) -> str:
    """Absolute path inside tutorials/."""
    return os.path.join(TUTORIALS_DIR, relpath)


def _read_tool_code(filename: str) -> str:
    """Read a tool .py file from tutorials/ and return its source."""
    with open(_tut(filename)) as f:
        return f.read()


_DOWNLOAD_CODE = _read_tool_code("download_mace_model.py")
_SMILES_CODE   = _read_tool_code("smiles_to_xyz.py")
_PACKMOL_CODE  = _read_tool_code("packmol_build_system.py")

# ── T1a: Register download_mace_model from inline code ─────────────────────────
PROMPT_T1A = (
    "\n"
    "    I have a Python function that can download MACE machine learning potential \n"
    "    models by name — it knows the download URLs for several standard models \n"
    "    (MACE-MP-0b3, MACE-MPA-0, etc.) and skips re-downloading if the file \n"
    "    already exists. Please add it to the system so I can use it later.\n\n"
    "\n"
    + _DOWNLOAD_CODE
)

# ── T1b: Register smiles_to_xyz + packmol_build_system from inline code ─────────
PROMPT_T1B = (
    "\n"
    "    Here are two functions I would like to add to the system.\n"
    "    The first converts a SMILES string to a 3D XYZ file using RDKit.\n"
    "    The second builds a periodic molecular simulation box using Packmol —\n"
    "    it takes one or more XYZ files and places copies of the molecules inside\n"
    "    a cubic box of a given size. Please register both so I can use them later.\n\n"
    "\n"
    + _SMILES_CODE + "\n" + _PACKMOL_CODE
)

# ── T1c: Create mace_md_specialist ──────────────────────────────────────────────
PROMPT_T1C = (
    "\n"
    "I need a dedicated specialist in MACE and molecular dynamics simulations.\n"
    "Please create one and give it the three tools I just added.\n\n"
    "\n"
)

# ── T2: Build NaCl + water box ──────────────────────────────────────────────────
PROMPT_T2 = (
    "I need a periodic simulation box containing 1 Na(+1), 1 Cl(-1) ions  and 267 water molecules. "
    f"First convert the water SMILES (O) to a 3D XYZ file at {_tut('water.xyz')}, "
    f"and the Na and CL ions with SMILES [Na+], and [Cl-] to {_tut('na.xyz')} and {_tut('cl.xyz')}. "
    "Then use packmol to build a cubic box of 20.0 Angstrom with 267 water molecules "
    f"and 1 NaCl pair, and save the result to {_tut('nacl_water_box.xyz')}."
)

# ── T3a: Register run_nvt_md from .py file ──────────────────────────────────────
PROMPT_T3A = (
    f"Please register the tools defined in the file {_tut('ASE_NVT_PBC.py')}. "
    "Update it if it already exists. "
    "Then assign the run_nvt_md tool to mace_md_specialist."
)

# ── T3b: Run NVT MD ─────────────────────────────────────────────────────────────
PROMPT_T3B = (
    "Please run a short NVT molecular dynamics simulation using ASE. "
    "(use the run_nvt_md tool from the mace_md_specialist ) "
    f"Use the MACE model at {_tut('models/mace-mp-0b3-medium.model')}, "
    f"the structure file {_tut('nacl_water_box.xyz')}, "
    "a box size of 20.0 Angstrom, "
    "a temperature of 300 K, and 10 steps. "
    f"Save the trajectory to {_tut('nvt_nacl_water.traj')}."
)

# ── T4a: Ask LLM to write, register, and assign plot_nvt_trajectory ─────────────
PROMPT_T4A = (
    "Write, register, and assign a new Python tool called plot_nvt_trajectory.\n\n"
    "The function signature must be:\n"
    "    plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str\n\n"
    "Include a docstring that describes what the function does.\n"
    "All imports (ase, matplotlib, numpy) must be inside the function body.\n\n"
    "The function must:\n"
    "  1. Read an ASE .traj file using ase.io.read with index=':'.\n"
    "  2. Extract per-frame: potential energy (eV) via get_potential_energy(),\n"
    "     total energy (eV) as get_kinetic_energy() + get_potential_energy(),\n"
    "     and temperature (K) via get_temperature().\n"
    "  3. Build a time axis in picoseconds: frame_index * timestep_fs / 1000.\n"
    "  4. Normalize potential and total energy by dividing each by its mean.\n"
    "  5. Create a two-panel matplotlib figure (figsize=(10, 6), sharex=True):\n"
    "       - Top panel: normalized potential energy and normalized total energy vs time (ps).\n"
    "       - Bottom panel: temperature (K) vs time (ps).\n"
    "     Add axis labels, legends, and gridlines.\n"
    "  6. Save the figure to output_png with dpi=150, return output_png.\n\n"
    "After writing the function: register it as a tool, then assign it to mace_md_specialist.\n"
    "Do not ask for confirmation — execute all three steps immediately."
)

# ── T4b: Plot the NVT trajectory ────────────────────────────────────────────────
PROMPT_T4B = (
    f"Plot the NVT trajectory at {_tut('nvt_nacl_water.traj')}. "
    "Use a timestep of 0.5 fs. "
    f"Save the figure to {_tut('nvt_nacl_water_analysis.png')}."
)

PROMPTS = {
    "t1a": PROMPT_T1A,
    "t1b": PROMPT_T1B,
    "t1c": PROMPT_T1C,
    "t2":  PROMPT_T2,
    "t3a": PROMPT_T3A,
    "t3b": PROMPT_T3B,
    "t4a": PROMPT_T4A,
    "t4b": PROMPT_T4B,
}
