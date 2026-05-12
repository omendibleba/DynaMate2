def plot_nvt_trajectory(traj_file: str, output_png: str, timestep_fs: float = 0.5) -> str:
    """
    Plots the normalized potential energy, total energy, and temperature from an ASE trajectory file.

    Parameters:
    traj_file (str): Path to the ASE .traj file.
    output_png (str): Path to save the output PNG file.
    timestep_fs (float): Time step in femtoseconds (default is 0.5 fs).

    Returns:
    str: Path to the saved PNG file.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.io import read

    # Read the trajectory file
    atoms = read(traj_file, index=':')
    potential_energy = np.array([a.get_potential_energy() for a in atoms])
    total_energy = np.array([a.get_kinetic_energy() + a.get_potential_energy() for a in atoms])
    temperature = np.array([a.get_temperature() for a in atoms])

    # Build time axis in picoseconds
    time_axis = np.arange(len(atoms)) * timestep_fs / 1000

    # Normalize energies
    norm_potential_energy = potential_energy / np.mean(potential_energy)
    norm_total_energy = total_energy / np.mean(total_energy)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(time_axis, norm_potential_energy, label='Normalized Potential Energy', color='blue')
    ax1.plot(time_axis, norm_total_energy, label='Normalized Total Energy', color='orange')
    ax1.set_ylabel('Normalized Energy')
    ax1.legend()
    ax1.grid()

    ax2.plot(time_axis, temperature, label='Temperature', color='green')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Temperature (K)')
    ax2.legend()
    ax2.grid()

    # Save the figure
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png