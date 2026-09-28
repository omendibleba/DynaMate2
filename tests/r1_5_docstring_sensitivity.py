"""
R1-5 sensitivity test: does docstring quality affect tool-routing / argument-
passing accuracy, and does that effect differ between gpt-4o-mini (the model
the live Quick Start UI's prompts -- backend/quickstart.py -- were tuned
against) and gpt-6-luna (a newer, economy-tier model)?

Methodology: isolate the docstring -> tool-selection mechanism from the rest
of the framework (supervisor handoffs, PromptEnhancer, multi-agent routing)
by binding the 4 T1-T4 tools directly to a bare ChatOpenAI model and
inspecting the first tool call for a single natural-language message -- no
LangGraph, no supervisor. This directly tests the thing R1-5 asks about
(does the LLM pick the right tool given its schema+docstring) without
conflating it with orchestration-layer behavior already covered elsewhere.

Each of the 4 tools is a real function with its real signature/type hints
(so the JSON schema sent to the model is exactly what the real framework
would derive) but a stub body that just records its own call -- no real
packmol/MACE/RDKit execution, since this measures routing+argument
correctness, not execution correctness (already covered by Item 1's
end-to-end run).
"""
import json
import time

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

# ── Stub tool implementations (real signatures, no-op bodies) ──────────────

def smiles_to_xyz(smiles: str, output_path: str = "molecule.xyz", random_seed: int = 42) -> str:
    return output_path


def packmol_build_system(xyz_files, box_size: float, n_molecules=1,
                          output_file: str = "system.xyz", tolerance: float = 2.5) -> str:
    return output_file


def run_nvt_md(
    structure_file: str, box_size: float, temperature_K: float, n_steps: int,
    model_name: str = "polar-1-m", output_traj: str = "nvt.traj",
    timestep_fs: float = 0.5, friction: float = 0.01, traj_interval: int = None,
    log_interval: int = 10, log_file: str = "nvt.log", device: str = "cuda",
    default_dtype: str = "float32", enable_cueq: bool = True, charge: int = 0,
    spin: int = 1, external_field: list = [0.0, 0.0, 0.0], pressure_bar: float = None,
    minimize_steps: int = 0, fmax: float = 0.5,
) -> str:
    return output_traj


def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    return output_png


FUNCS = {
    "smiles_to_xyz": smiles_to_xyz,
    "packmol_build_system": packmol_build_system,
    "run_nvt_md": run_nvt_md,
    "plot_nvt_trajectory": plot_nvt_trajectory,
}

# ── Docstring levels ────────────────────────────────────────────────────────

DOCSTRINGS = {
    "smiles_to_xyz": {
        "full": (
            "Convert a SMILES string to a 3D XYZ file using RDKit.\n\n"
            "Generates 3D coordinates with ETKDG and optimises with UFF force field.\n\n"
            "Parameters\n----------\n"
            "smiles      : str -- SMILES string of the molecule\n"
            "output_path : str -- path for the output .xyz file\n"
            "random_seed : int -- seed for ETKDG conformer generation (default 42)\n\n"
            "Returns\n-------\nstr -- path to the written XYZ file"
        ),
        "no_types": (
            "Convert a SMILES string to a 3D XYZ file using RDKit.\n\n"
            "Generates 3D coordinates with ETKDG and optimises with UFF force field.\n\n"
            "Parameters\n----------\n"
            "smiles      -- SMILES string of the molecule\n"
            "output_path -- path for the output .xyz file\n"
            "random_seed -- seed for ETKDG conformer generation\n\n"
            "Returns\n-------\npath to the written XYZ file"
        ),
        "one_line": "Convert a SMILES string to a 3D XYZ file using RDKit.",
        "none": "Tool.",
    },
    "packmol_build_system": {
        "full": (
            "Build a molecular system using Packmol by placing one or more molecules "
            "in a cubic box.\n\n"
            "Parameters\n----------\n"
            "xyz_files   : str or list of str -- path(s) to input XYZ file(s)\n"
            "box_size    : float              -- size of the cubic box (Angstrom)\n"
            "n_molecules : int or list of int -- number of copies for each molecule\n"
            "output_file : str                -- output file path\n"
            "tolerance   : float              -- minimum atom-atom distance (Angstrom)\n\n"
            "Returns\n-------\nstr -- path to the generated system XYZ file"
        ),
        "no_types": (
            "Build a molecular system using Packmol by placing one or more molecules "
            "in a cubic box.\n\n"
            "Parameters\n----------\n"
            "xyz_files   -- path(s) to input XYZ file(s)\n"
            "box_size    -- size of the cubic box (Angstrom)\n"
            "n_molecules -- number of copies for each molecule\n"
            "output_file -- output file path\n"
            "tolerance   -- minimum atom-atom distance\n\n"
            "Returns\n-------\npath to the generated system XYZ file"
        ),
        "one_line": "Build a molecular system using Packmol by placing one or more molecules in a cubic box.",
        "none": "Tool.",
    },
    "run_nvt_md": {
        "full": (
            "Run a molecular dynamics simulation with a MACE polar calculator: NVT "
            "(Langevin thermostat) by default, or NPT (Berendsen thermostat + "
            "barostat) if pressure_bar is given. Optionally relaxes the structure "
            "first (minimize_steps > 0) to remove bad contacts left by packing.\n\n"
            "Parameters\n----------\n"
            "structure_file : str   -- path to the input structure (XYZ or extxyz)\n"
            "box_size       : float -- size of the cubic simulation box in Angstroms\n"
            "temperature_K  : float -- target temperature in Kelvin\n"
            "n_steps        : int   -- number of MD steps to run\n"
            "model_name     : str   -- name of the MACE polar foundation model to use\n"
            "output_traj    : str   -- path for the output ASE trajectory file\n"
            "pressure_bar   : float -- if given, run NPT at this pressure in bar instead of NVT\n"
            "minimize_steps : int   -- max geometry-optimization steps before MD; 0 = skip\n\n"
            "Returns\n-------\nstr -- path to the written trajectory file"
        ),
        "no_types": (
            "Run a molecular dynamics simulation with a MACE polar calculator: NVT "
            "by default, or NPT if pressure_bar is given.\n\n"
            "Parameters\n----------\n"
            "structure_file -- path to the input structure\n"
            "box_size       -- size of the cubic simulation box\n"
            "temperature_K  -- target temperature\n"
            "n_steps        -- number of MD steps to run\n"
            "model_name     -- name of the MACE polar foundation model to use\n"
            "output_traj    -- path for the output trajectory file\n"
            "pressure_bar   -- if given, run NPT at this pressure instead of NVT\n"
            "minimize_steps -- geometry-optimization steps before MD\n\n"
            "Returns\n-------\npath to the written trajectory file"
        ),
        "one_line": "Run a molecular dynamics simulation (NVT or NPT) using a MACE machine-learning potential.",
        "none": "Tool.",
    },
    "plot_nvt_trajectory": {
        "full": (
            "Plot potential energy, total energy, and temperature vs time from an "
            "ASE NVT/NPT trajectory.\n\n"
            "Parameters\n----------\n"
            "traj_file   : str   -- path to the input ASE .traj trajectory file\n"
            "output_png  : str   -- path to save the output plot PNG file\n"
            "timestep_fs : float -- MD timestep in femtoseconds used during the simulation\n\n"
            "Returns\n-------\nstr -- path to the saved PNG file"
        ),
        "no_types": (
            "Plot potential energy, total energy, and temperature vs time from an "
            "ASE NVT/NPT trajectory.\n\n"
            "Parameters\n----------\n"
            "traj_file   -- path to the input trajectory file\n"
            "output_png  -- path to save the output plot PNG file\n"
            "timestep_fs -- MD timestep in femtoseconds used during the simulation\n\n"
            "Returns\n-------\npath to the saved PNG file"
        ),
        "one_line": "Plot potential energy, total energy, and temperature vs time from an ASE trajectory.",
        "none": "Tool.",
    },
}

LEVELS = ["full", "no_types", "one_line", "none"]

# ── Fixed prompts, each targeting exactly one tool, with checkable expected args ──

def _contains(val, substr):
    return isinstance(val, str) and substr in val


def _num_eq(val, expected):
    try:
        return float(val) == float(expected)
    except (TypeError, ValueError):
        return False


PROMPTS = [
    dict(tool="smiles_to_xyz",
         text="Convert the SMILES string 'CCO' into a 3D XYZ structure and save it to ethanol.xyz.",
         check=lambda a: a.get("smiles") == "CCO" and _contains(a.get("output_path", ""), "ethanol.xyz")),
    dict(tool="smiles_to_xyz",
         text="I have a SMILES 'O' for water -- generate 3D coordinates and write them to water.xyz.",
         check=lambda a: a.get("smiles") == "O" and _contains(a.get("output_path", ""), "water.xyz")),
    dict(tool="smiles_to_xyz",
         text="Turn this SMILES into a molecule file: c1ccccc1, save as benzene.xyz.",
         check=lambda a: a.get("smiles") == "c1ccccc1" and _contains(a.get("output_path", ""), "benzene.xyz")),

    dict(tool="packmol_build_system",
         text="Build a periodic simulation box of 15 Angstrom containing 30 copies of water.xyz using packmol.",
         check=lambda a: _num_eq(a.get("box_size"), 15) and str(a.get("n_molecules")) .find("30") >= 0
                          and "water.xyz" in json.dumps(a.get("xyz_files", ""))),
    dict(tool="packmol_build_system",
         text="Pack 10 molecules from methane.xyz into a cubic box of size 10 Angstrom and save the result to box.xyz.",
         check=lambda a: _num_eq(a.get("box_size"), 10) and str(a.get("n_molecules")).find("10") >= 0
                          and "methane.xyz" in json.dumps(a.get("xyz_files", ""))
                          and _contains(a.get("output_file", ""), "box.xyz")),
    dict(tool="packmol_build_system",
         text="Use packmol to place 5 copies of ion.xyz and 50 copies of water.xyz into a 20 Angstrom box, saved to mix.xyz.",
         check=lambda a: _num_eq(a.get("box_size"), 20)
                          and "ion.xyz" in json.dumps(a.get("xyz_files", ""))
                          and "water.xyz" in json.dumps(a.get("xyz_files", ""))
                          and _contains(a.get("output_file", ""), "mix.xyz")),

    dict(tool="run_nvt_md",
         text="Run a 200-step NVT molecular dynamics simulation at 300 K on structure.xyz using MACE, saving the trajectory to traj.traj.",
         check=lambda a: _num_eq(a.get("n_steps"), 200) and _num_eq(a.get("temperature_K"), 300)
                          and _contains(a.get("structure_file", ""), "structure.xyz")
                          and _contains(a.get("output_traj", ""), "traj.traj")),
    dict(tool="run_nvt_md",
         text="Simulate NPT dynamics on box.xyz at 310 K and 1 bar for 500 steps with a MACE potential, save to result.traj.",
         check=lambda a: _num_eq(a.get("n_steps"), 500) and _num_eq(a.get("temperature_K"), 310)
                          and _num_eq(a.get("pressure_bar"), 1) and _contains(a.get("structure_file", ""), "box.xyz")),
    dict(tool="run_nvt_md",
         text="Perform molecular dynamics on water_box.xyz for 1000 steps at 350 K using the MACE polar model 'polar-1-m', output to sim.traj.",
         check=lambda a: _num_eq(a.get("n_steps"), 1000) and _num_eq(a.get("temperature_K"), 350)
                          and _contains(a.get("model_name", ""), "polar-1-m")
                          and _contains(a.get("structure_file", ""), "water_box.xyz")),

    dict(tool="plot_nvt_trajectory",
         text="Plot the potential energy and temperature over time from the trajectory nvt.traj and save it as analysis.png.",
         check=lambda a: _contains(a.get("traj_file", ""), "nvt.traj") and _contains(a.get("output_png", ""), "analysis.png")),
    dict(tool="plot_nvt_trajectory",
         text="Generate a figure showing energy and temperature vs time from run.traj, output to plot.png.",
         check=lambda a: _contains(a.get("traj_file", ""), "run.traj") and _contains(a.get("output_png", ""), "plot.png")),
    dict(tool="plot_nvt_trajectory",
         text="Create a chart of the MD trajectory data in sim.traj using a timestep of 1.0 fs, save the image as result.png.",
         check=lambda a: _contains(a.get("traj_file", ""), "sim.traj") and _contains(a.get("output_png", ""), "result.png")
                          and _num_eq(a.get("timestep_fs"), 1.0)),
]

MODEL_TIERS = {
    "gpt-4o-mini": dict(model="gpt-4o-mini", temperature=0),
    "gpt-6-luna": dict(model="gpt-6-luna", temperature=0, reasoning_effort="none"),
}


def build_tools(level):
    tools = []
    for name, func in FUNCS.items():
        tools.append(StructuredTool.from_function(func=func, name=name, description=DOCSTRINGS[name][level]))
    return tools


def run_trial(level, tier_name):
    tools = build_tools(level)
    model = ChatOpenAI(**MODEL_TIERS[tier_name])
    bound = model.bind_tools(tools)

    results = []
    for p in PROMPTS:
        t0 = time.time()
        try:
            resp = bound.invoke(p["text"])
            calls = resp.tool_calls or []
        except Exception as e:
            results.append(dict(tool=p["tool"], prompt=p["text"], error=f"{type(e).__name__}: {e}",
                                 routing_correct=False, arg_correct=False, elapsed=time.time() - t0))
            continue

        routing_correct = len(calls) == 1 and calls[0]["name"] == p["tool"]
        arg_correct = False
        called_name = calls[0]["name"] if len(calls) == 1 else ([c["name"] for c in calls] if calls else None)
        called_args = calls[0]["args"] if len(calls) == 1 else None
        if routing_correct:
            try:
                arg_correct = bool(p["check"](called_args))
            except Exception:
                arg_correct = False

        results.append(dict(
            tool=p["tool"], prompt=p["text"], called=called_name, args=called_args,
            routing_correct=routing_correct, arg_correct=arg_correct, elapsed=time.time() - t0,
        ))
    return results


def main():
    all_results = {}
    for tier_name in MODEL_TIERS:
        for level in LEVELS:
            key = f"{tier_name}__{level}"
            print(f"\n=== {key} ===", flush=True)
            trial = run_trial(level, tier_name)
            n_route = sum(r["routing_correct"] for r in trial)
            n_arg = sum(r["arg_correct"] for r in trial)
            print(f"routing: {n_route}/{len(trial)}  args: {n_arg}/{len(trial)}", flush=True)
            for r in trial:
                status = "OK" if r["routing_correct"] and r["arg_correct"] else (
                    "ROUTE-OK/ARG-FAIL" if r["routing_correct"] else "ROUTE-FAIL")
                called = r.get("called")
                print(f"  [{status}] expected={r['tool']:<22} called={called}", flush=True)
            all_results[key] = trial

    out_path = "/scratchpad/r1_5_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved raw results to {out_path}", flush=True)

    # Summary table
    print("\n\n=== SUMMARY (routing_correct / arg_correct, out of 12) ===")
    header = f"{'docstring level':<12}" + "".join(f"{t:>28}" for t in MODEL_TIERS)
    print(header)
    for level in LEVELS:
        row = f"{level:<12}"
        for tier_name in MODEL_TIERS:
            trial = all_results[f"{tier_name}__{level}"]
            n_route = sum(r["routing_correct"] for r in trial)
            n_arg = sum(r["arg_correct"] for r in trial)
            row += f"{f'{n_route}/12 route, {n_arg}/12 args':>28}"
        print(row)


if __name__ == "__main__":
    main()
