def run_nvt_md(
    model_path: str,
    structure_file: str,
    box_size: float,
    temperature_K: float,
    n_steps: int,
    output_traj: str = "nvt.traj",
    timestep_fs: float = 0.5,
    friction: float = 0.01,
    traj_interval: int = 100,
    log_interval: int = 10,
    log_file: str = "nvt.log",
    device: str = "cuda",
) -> str:
    """
    Run an NVT Langevin MD simulation with PBC using a MACE calculator.

    Parameters
    ----------
    model_path     : str   -- path to the MACE .model or -lammps.pt file
    structure_file : str   -- path to the input structure (XYZ or extxyz)
    box_size       : float -- size of the cubic simulation box in Angstroms
    temperature_K  : float -- target temperature in Kelvin
    n_steps        : int   -- number of MD steps to run
    output_traj    : str   -- path for the output ASE trajectory file
    timestep_fs    : float -- MD timestep in femtoseconds (default 0.5)
    friction       : float -- Langevin friction coefficient in 1/fs (default 0.01)
    traj_interval  : int   -- write trajectory every N steps (default 100)
    log_interval   : int   -- write log every N steps (default 10)
    log_file       : str   -- path for the MDLogger output file
    device         : str   -- compute device: 'cuda' or 'cpu'

    Returns
    -------
    str -- path to the written trajectory file
    """
    import numpy as np
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import MACECalculator

    # Load structure and enforce PBC, and box size
    atoms = read(structure_file)
    atoms.set_pbc([True, True, True])
    atoms.set_cell([box_size, box_size, box_size])

    # Attach MACE calculator
    calculator = MACECalculator(model_path=model_path, device=device)
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

    # ── Edit these paths before running ───────────────────────────────────────
    MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "mace-mp-0b3-medium.model")
    STRUCTURE_FILE = os.path.join(os.path.dirname(__file__), "nacl_water_box.xyz")
    BOX_SIZE       = 20.0   # Å  (matches the box built by packmol in T2.1)
    TEMPERATURE_K  = 300.0  # K
    N_STEPS        = 100    # short run for testing
    OUTPUT_TRAJ    = "test_nvt.traj"
    LOG_FILE       = "test_nvt.log"
    DEVICE         = "cuda"
    # ──────────────────────────────────────────────────────────────────────────

    print(f"Model          : {MODEL_PATH}")
    print(f"Structure      : {STRUCTURE_FILE}")
    print(f"Box size       : {BOX_SIZE} Å")
    print(f"Temperature    : {TEMPERATURE_K} K")
    print(f"Steps          : {N_STEPS}")
    print(f"Output traj    : {OUTPUT_TRAJ}")
    print(f"Device         : {DEVICE}")
    print()

    result = run_nvt_md(
        model_paths=MODEL_PATH,
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
