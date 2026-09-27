def packmol_build_system(
    xyz_files,
    box_size: float,
    n_molecules=1,
    output_file: str = 'system.xyz',
    tolerance: float = 2.5
):
    """
    Build a molecular system using Packmol by placing one or more molecules
    in a cubic box.

    If the output file already exists it is returned immediately without
    re-running Packmol.

    Parameters
    ----------
    xyz_files   : str or list of str -- path(s) to input XYZ file(s)
    box_size    : float              -- size of the cubic box (Angstrom)
    n_molecules : int or list of int -- number of copies for each molecule
    output_file : str                -- output file path
    tolerance   : float              -- minimum atom-atom distance (Angstrom)

    Returns
    -------
    str -- path to the generated system XYZ file
    """
    import subprocess
    import os
    import tempfile
    import time

    if isinstance(xyz_files, str):
        xyz_files = [xyz_files]
    if isinstance(n_molecules, int):
        n_molecules = [n_molecules] * len(xyz_files)
    if len(xyz_files) != len(n_molecules):
        raise ValueError('Length of xyz_files and n_molecules must match')

    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists. Skipping Packmol build.")
        return output_file

    # The agent calling this tool often also just called smiles_to_xyz (or
    # similar) to produce these same xyz_files, in the same turn. LangGraph's
    # ToolNode can dispatch multiple tool calls from one turn concurrently via
    # a thread pool, so this tool can start running before that write has
    # landed on disk even though the prompt said "first...then". Wait briefly
    # for each input file to appear rather than failing immediately.
    for xyz in xyz_files:
        waited = 0.0
        while not os.path.exists(xyz) and waited < 15.0:
            time.sleep(0.25)
            waited += 0.25
        if not os.path.exists(xyz):
            raise FileNotFoundError(
                f"Required input file '{xyz}' never appeared (waited {waited:.1f}s). "
                "It may not have been generated yet, or generation failed."
            )

    if subprocess.call(['which', 'packmol'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise EnvironmentError(
            'Packmol is not found in PATH. Please install or load it first.'
        )

    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    # A real system temp file, not a bare relative filename -- the caller's
    # current working directory isn't guaranteed to be writable (e.g. a
    # read-only container root filesystem), while the OS temp directory is.
    fd, input_filename = tempfile.mkstemp(suffix='.inp', prefix='packmol_input_')
    os.close(fd)
    with open(input_filename, 'w') as f:
        f.write(f'tolerance {tolerance}\n')
        f.write('filetype xyz\n')
        f.write(f'output {output_file}\n')
        # periodic boundary conditions: without this, molecules near opposite
        # faces overlap once the box is treated as periodic in the MD run
        f.write(f'pbc 0. 0. 0. {box_size} {box_size} {box_size}\n')
        f.write('seed 12345\n\n')
        for xyz, n in zip(xyz_files, n_molecules):
            f.write(f'structure {xyz}\n')
            f.write(f'  number {n}\n')
            f.write(f'  inside box 0. 0. 0. {box_size} {box_size} {box_size}\n')
            f.write('end structure\n\n')

    print(f'Running Packmol to build system in a {box_size} A box...')
    os.system(f'packmol < {input_filename}')

    if not os.path.exists(output_file):
        raise FileNotFoundError('Packmol did not produce the expected output file.')

    print(f'System built successfully: {output_file}')
    return output_file
