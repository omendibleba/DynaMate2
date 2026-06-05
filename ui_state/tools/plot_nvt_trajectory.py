def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """Read an ASE .traj file and plot normalized potential and total energies and temperature vs time.

    Parameters:
    traj_file (str): Path to the ASE trajectory file (.traj).
    output_png (str): Path to save the output plot PNG file.
    timestep_fs (float): Timestep between frames in femtoseconds (default 0.5 fs).

    Returns:
    str: The output PNG file path.
    """
    import ase.io
    import matplotlib
    matplotlib.use('Agg')  # thread-safe non-GUI backend; must precede pyplot import
    import matplotlib.pyplot as plt
    import numpy as np

    # Read all frames from the trajectory
    frames = ase.io.read(traj_file, index=':')

    # Extract energies and temperature
    pot_energies = np.array([frame.get_potential_energy() for frame in frames])
    kin_energies = np.array([frame.get_kinetic_energy() for frame in frames])
    total_energies = pot_energies + kin_energies
    temperatures = np.array([frame.get_temperature() for frame in frames])

    # Time axis in picoseconds
    times_ps = np.arange(len(frames)) * timestep_fs / 1000.0

    # Normalize energies by their mean
    pot_energies_norm = pot_energies / np.mean(pot_energies)
    total_energies_norm = total_energies / np.mean(total_energies)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(times_ps, pot_energies_norm, label='Normalized Potential Energy')
    ax1.plot(times_ps, total_energies_norm, label='Normalized Total Energy')
    ax1.set_ylabel('Normalized Energy (eV)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times_ps, temperatures, label='Temperature')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Temperature (K)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    return output_png
