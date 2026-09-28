def check_density_convergence(
    traj_file: str,
    window_frames: int = 20,
    tolerance_percent: float = 1.0,
) -> dict:
    """
    Check whether an NPT/NVT trajectory's density has converged, by comparing
    the mean density of the last window_frames frames against the window
    immediately before it. Returns a converged flag the supervisor can use as
    a genuine branching condition: continue the same equilibration for more
    steps if not converged, or hand off to the next stage of the workflow if
    it is. This is the conditional step distinguishing this example from a
    fixed-length, strictly sequential pipeline.

    Parameters
    ----------
    traj_file         : str   -- path to an ASE .traj file (e.g. from run_nvt_md
                                  with pressure_bar set, so cell volume changes)
    window_frames      : int   -- number of trailing frames averaged for each
                                  of the two comparison windows
    tolerance_percent  : float -- convergence threshold; converged if the two
                                  windows' mean densities differ by less than
                                  this percentage of the earlier window's value

    Returns
    -------
    dict -- {
        "converged": bool,
        "density_g_per_cm3_recent": float,
        "density_g_per_cm3_previous": float,
        "percent_change": float,
        "n_frames_available": int,
        "recommendation": str -- "extend_equilibration" or "proceed",
    }
    """
    from ase.io import read

    traj = read(traj_file, index=':')
    n_frames = len(traj)

    if n_frames < 2 * window_frames:
        return {
            'converged': False,
            'density_g_per_cm3_recent': None,
            'density_g_per_cm3_previous': None,
            'percent_change': None,
            'n_frames_available': n_frames,
            'recommendation': 'extend_equilibration',
        }

    def mean_density(frames):
        densities = []
        for atoms in frames:
            mass_amu = sum(atoms.get_masses())
            volume_A3 = atoms.get_volume()
            # amu / Angstrom^3 -> g / cm^3
            density = mass_amu * 1.66053906660 / volume_A3
            densities.append(density)
        return sum(densities) / len(densities)

    recent = traj[-window_frames:]
    previous = traj[-2 * window_frames:-window_frames]

    density_recent = mean_density(recent)
    density_previous = mean_density(previous)
    percent_change = 100.0 * abs(density_recent - density_previous) / density_previous

    converged = percent_change < tolerance_percent

    return {
        'converged': converged,
        'density_g_per_cm3_recent': density_recent,
        'density_g_per_cm3_previous': density_previous,
        'percent_change': percent_change,
        'n_frames_available': n_frames,
        'recommendation': 'proceed' if converged else 'extend_equilibration',
    }


def run_dft_single_point(
    structure_file: str,
    frame_index: int = -1,
    charge: int = 0,
    spin_multiplicity: int = 1,
    method: str = 'GFN2-xTB',
    output_json: str = 'dft_single_point.json',
) -> dict:
    """
    Run a single-point electronic-structure calculation on one frame of a
    structure or trajectory file, using a fast tight-binding DFT method by
    default (GFN2-xTB, via the tblite ASE calculator) so the example completes
    in seconds rather than requiring an HPC allocation. TEMPLATE NOTE: replace
    the calculator construction below with whichever DFT code your group runs
    routinely (e.g. an ASE Espresso/VASP/ORCA calculator) if a specific level
    of theory is required for the paper; the surrounding docstring, argument
    names, and return structure are designed to stay the same either way.

    Parameters
    ----------
    structure_file     : str   -- path to a structure or trajectory file readable
                                   by ase.io.read (e.g. the final frame of an
                                   equilibrated MD run)
    frame_index         : int   -- which frame to use if structure_file contains
                                   multiple frames (default: last frame, -1)
    charge              : int   -- total system charge
    spin_multiplicity   : int   -- spin multiplicity (2S + 1)
    method              : str   -- level of theory identifier passed to the
                                   underlying calculator
    output_json         : str   -- path to write the result as JSON

    Returns
    -------
    dict -- {
        "energy_eV": float,
        "method": str,
        "charge": int,
        "spin_multiplicity": int,
        "structure_file": str,
        "frame_index": int,
        "output_json": str,
    }
    """
    import json

    from ase.io import read
    from tblite.ase import TBLite

    atoms = read(structure_file, index=frame_index)
    atoms.calc = TBLite(method=method, charge=charge, multiplicity=spin_multiplicity)

    energy_eV = atoms.get_potential_energy()

    result = {
        'energy_eV': energy_eV,
        'method': method,
        'charge': charge,
        'spin_multiplicity': spin_multiplicity,
        'structure_file': structure_file,
        'frame_index': frame_index,
        'output_json': output_json,
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    return result
