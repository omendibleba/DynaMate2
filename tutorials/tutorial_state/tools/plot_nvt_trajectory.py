def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """Plot potential energy, total energy, and temperature vs time from an ASE NVT trajectory.

    Args:
        traj_file (str): Path to the ASE binary trajectory (.traj) file.
        output_png (str): Path to save the output PNG image.
        timestep_fs (float): Timestep in femtoseconds used during the simulation. Defaults to 0.5.

    Returns:
        str: Path to the saved PNG image.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io.trajectory import Trajectory

    traj = Trajectory(traj_file, 'r')
    frames = list(traj)

    times = np.arange(len(frames)) * timestep_fs
    pot_energies = [atoms.get_potential_energy() for atoms in frames]
    kin_energies  = [atoms.get_kinetic_energy()   for atoms in frames]
    tot_energies  = [pe + ke for pe, ke in zip(pot_energies, kin_energies)]
    temperatures  = [atoms.get_temperature()       for atoms in frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(times, pot_energies, label='Potential Energy')
    ax1.plot(times, tot_energies, label='Total Energy')
    ax1.set_ylabel('Energy (eV)')
    ax1.legend()

    ax2.plot(times, temperatures)
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Temperature (K)')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    return output_png
