def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """Read an ASE .traj file and plot normalized potential and total energy, and temperature vs time.

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

    traj = ase.io.read(traj_file, index=":")
    n_frames = len(traj)

    potential_energies = np.array([frame.get_potential_energy() for frame in traj])
    kinetic_energies = np.array([frame.get_kinetic_energy() for frame in traj])
    total_energies = kinetic_energies + potential_energies
    temperatures = np.array([frame.get_temperature() for frame in traj])

    time_ps = np.arange(n_frames) * timestep_fs / 1000.0

    potential_norm = potential_energies / np.mean(potential_energies)
    total_norm = total_energies / np.mean(total_energies)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(time_ps, potential_norm, label='Normalized Potential Energy')
    ax1.plot(time_ps, total_norm, label='Normalized Total Energy')
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