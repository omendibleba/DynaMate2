def run_nvt_md(
    structure_file: str,
    box_size: float,
    temperature_K: float,
    n_steps: int,
    model_name: str = "polar-1-m",
    output_traj: str = "nvt.traj",
    timestep_fs: float = 0.5,
    friction: float = 0.01,
    traj_interval: int = 100,
    log_interval: int = 10,
    log_file: str = "nvt.log",
    device: str = "cuda",
    default_dtype: str = "float64",
    charge: int = 0,
    spin: int = 1,
    external_field=(0.0, 0.0, 0.0),
) -> str:
    """
    Run an NVT molecular dynamics simulation using the Langevin thermostat using a MACE polar calculator.

    Parameters
    ----------
    structure_file : str   -- path to the input structure (XYZ or extxyz)
    box_size       : float -- size of the cubic simulation box in Angstroms
    temperature_K  : float -- target temperature in Kelvin
    n_steps        : int   -- number of MD steps to run
    model_name     : str   -- name of the MACE polar foundation model to use (e.g. 'polar-1-m');
                               downloaded and cached automatically, no local model file needed
    output_traj    : str   -- path for the output ASE trajectory file
    timestep_fs    : float -- MD timestep in femtoseconds (default 0.5)
    friction       : float -- Langevin friction coefficient in 1/fs (default 0.01)
    traj_interval  : int   -- write trajectory every N steps (default 100)
    log_interval   : int   -- write log every N steps (default 10)
    log_file       : str   -- path for the MDLogger output file
    device         : str   -- compute device: 'cuda' or 'cpu'
    default_dtype  : str   -- 'float64' (default, more precise) or 'float32' (faster MD)
    charge         : int   -- total system charge, set on atoms.info before the calculator runs
    spin           : int   -- spin multiplicity, set on atoms.info before the calculator runs
    external_field : sequence of 3 floats -- external field vector, set on atoms.info

    Returns
    -------
    str -- path to the written trajectory file
    """
    import os
    import numpy as np
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import mace_polar

    os.makedirs(os.path.dirname(os.path.abspath(output_traj)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    # Load structure and enforce PBC, and box size
    atoms = read(structure_file)
    atoms.set_pbc([True, True, True])
    atoms.set_cell([box_size, box_size, box_size])

    # Attach MACE polar calculator -- a named foundation model, downloaded
    # and cached automatically, no local .model file needed.
    calculator = mace_polar(model=model_name, device=device, default_dtype=default_dtype)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.info["external_field"] = list(external_field)
    atoms.calc = calculator

    # Initialise velocities from Maxwell-Boltzmann distribution
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    # Set up Langevin thermostat
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction / units.fs,
    )

    # Print energy and density at each log interval
    def print_properties():
        pot_energy = atoms.get_potential_energy()
        mass_g     = np.sum(atoms.get_masses()) * 1.660539e-24  # g
        volume_cm3 = atoms.get_volume() * 1e-24                 # cm³
        density    = mass_g / volume_cm3
        print(
            f"Step: {dyn.get_number_of_steps():>7d} | "
            f"Pot. Energy: {pot_energy:>12.4f} eV | "
            f"Density: {density:.4f} g/cm³"
        )

    dyn.attach(print_properties, interval=log_interval)

    # Trajectory output
    traj = Trajectory(output_traj, "w", atoms)
    dyn.attach(traj.write, interval=traj_interval)

    # MDLogger output
    logger = MDLogger(
        dyn, atoms, log_file,
        header=True, stress=False, peratom=False, mode="w",
    )
    dyn.attach(logger, interval=log_interval)

    # Run simulation
    dyn.run(n_steps)
    traj.close()

    return output_traj


if __name__ == "__main__":
    import os

    # ── Edit these paths/values before running ─────────────────────────────────
    MODEL_NAME     = "polar-1-m"
    STRUCTURE_FILE = os.path.join(os.path.dirname(__file__), "nacl_water_box.xyz")
    BOX_SIZE       = 20.0   # Å  (matches the box built by packmol in T2.1)
    TEMPERATURE_K  = 300.0  # K
    N_STEPS        = 100    # short run for testing
    OUTPUT_TRAJ    = "test_nvt.traj"
    LOG_FILE       = "test_nvt.log"
    DEVICE         = "cuda"
    # ──────────────────────────────────────────────────────────────────────────

    print(f"Model          : {MODEL_NAME}")
    print(f"Structure      : {STRUCTURE_FILE}")
    print(f"Box size       : {BOX_SIZE} Å")
    print(f"Temperature    : {TEMPERATURE_K} K")
    print(f"Steps          : {N_STEPS}")
    print(f"Output traj    : {OUTPUT_TRAJ}")
    print(f"Device         : {DEVICE}")
    print()

    result = run_nvt_md(
        model_name=MODEL_NAME,
        structure_file=STRUCTURE_FILE,
        box_size=BOX_SIZE,
        temperature_K=TEMPERATURE_K,
        n_steps=N_STEPS,
        output_traj=OUTPUT_TRAJ,
        log_file=LOG_FILE,
        device=DEVICE,
    )

    print(f"\nDone. Trajectory written to: {result}")
    print(f"Log written to             : {LOG_FILE}")
