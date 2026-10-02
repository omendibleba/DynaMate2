def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """
    Plot potential energy, total energy, and temperature vs time from an ASE
    NVT/NPT trajectory.

    Reads the trajectory with ase.io.read, extracts per-frame potential energy
    (get_potential_energy), total energy (get_kinetic_energy + get_potential_energy),
    and temperature (get_temperature), builds a time axis in picoseconds from
    timestep_fs, and saves a two-panel figure: normalized potential/total energy
    on top, temperature on the bottom.

    Parameters
    ----------
    traj_file   : str   -- path to the input ASE .traj trajectory file
    output_png  : str   -- path to save the output plot PNG file
    timestep_fs : float -- MD timestep in femtoseconds used during the simulation
                            (default 0.5), used to build the time axis

    Returns
    -------
    str -- path to the saved PNG file
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read

    frames = read(traj_file, index=":")

    potential_energies = np.array([f.get_potential_energy() for f in frames])
    total_energies = np.array(
        [f.get_kinetic_energy() + f.get_potential_energy() for f in frames]
    )
    temperatures = np.array([f.get_temperature() for f in frames])

    times_ps = np.arange(len(frames)) * timestep_fs / 1000.0

    norm_potential = potential_energies / potential_energies.mean()
    norm_total = total_energies / total_energies.mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(times_ps, norm_potential, label="Potential Energy (normalized)")
    ax1.plot(times_ps, norm_total, label="Total Energy (normalized)")
    ax1.set_ylabel("Normalized Energy", fontsize=15)
    ax1.legend(fontsize=15)
    ax1.tick_params(axis="both", labelsize=15)
    ax1.grid(True)

    ax2.plot(times_ps, temperatures, color="tab:red", label="Temperature (K)")
    ax2.set_xlabel("Time (ps)", fontsize=15)
    ax2.set_ylabel("Temperature (K)", fontsize=15)
    ax2.legend(fontsize=15)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    return output_png
