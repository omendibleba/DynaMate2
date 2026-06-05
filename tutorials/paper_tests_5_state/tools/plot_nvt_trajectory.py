def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """Reads an ASE .traj file and plots normalized potential and total energy, and temperature vs time.

    Parameters:
    traj_file (str): Path to the ASE .traj trajectory file.
    output_png (str): Path to save the output plot PNG file.
    timestep_fs (float): Timestep in femtoseconds between frames (default 0.5 fs).

    Returns:
    str: The output PNG file path.
    """
    import ase.io
    import matplotlib.pyplot as plt
    import numpy as np

    # Read all frames from the trajectory
    frames = ase.io.read(traj_file, index=":")

    # Extract energies and temperature
    potential_energies = np.array([frame.get_potential_energy() for frame in frames])
    kinetic_energies = np.array([frame.get_kinetic_energy() for frame in frames])
    total_energies = kinetic_energies + potential_energies
    temperatures = np.array([frame.get_temperature() for frame in frames])

    # Time axis in picoseconds
    time_ps = np.arange(len(frames)) * timestep_fs / 1000.0

    # Normalize energies by their mean
    norm_potential = potential_energies / np.mean(potential_energies)
    norm_total = total_energies / np.mean(total_energies)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(time_ps, norm_potential, label='Normalized Potential Energy')
    ax1.plot(time_ps, norm_total, label='Normalized Total Energy')
    ax1.set_ylabel('Normalized Energy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_ps, temperatures, label='Temperature (K)', color='tab:orange')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Temperature (K)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    return output_png
