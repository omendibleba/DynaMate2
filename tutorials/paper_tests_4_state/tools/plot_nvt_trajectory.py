def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """
    Reads an ASE .traj file and plots normalized potential and total energies along with temperature over time.
    The time axis is constructed in picoseconds based on the provided timestep in femtoseconds.
    The plot is saved as a PNG file and the output file path is returned.
    """
    import ase.io
    import matplotlib.pyplot as plt
    import numpy as np

    traj = ase.io.read(traj_file, index=":")
    potential_energies = np.array([frame.get_potential_energy() for frame in traj])
    kinetic_energies = np.array([frame.get_kinetic_energy() for frame in traj])
    total_energies = kinetic_energies + potential_energies
    temperatures = np.array([frame.get_temperature() for frame in traj])

    time_ps = np.arange(len(traj)) * timestep_fs / 1000.0

    norm_potential = potential_energies / np.mean(potential_energies)
    norm_total = total_energies / np.mean(total_energies)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(time_ps, norm_potential, label="Normalized Potential Energy")
    ax1.plot(time_ps, norm_total, label="Normalized Total Energy")
    ax1.set_ylabel("Normalized Energy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_ps, temperatures, label="Temperature (K)", color="tab:red")
    ax2.set_xlabel("Time (ps)")
    ax2.set_ylabel("Temperature (K)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    return output_png
