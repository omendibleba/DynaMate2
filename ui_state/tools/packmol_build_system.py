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

    if isinstance(xyz_files, str):
        xyz_files = [xyz_files]
    if isinstance(n_molecules, int):
        n_molecules = [n_molecules] * len(xyz_files)
    if len(xyz_files) != len(n_molecules):
        raise ValueError('Length of xyz_files and n_molecules must match')

    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists. Skipping Packmol build.")
        return output_file

    if subprocess.call(['which', 'packmol'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise EnvironmentError(
            'Packmol is not found in PATH. Please install or load it first.'
        )

    input_filename = 'packmol_input.inp'
    with open(input_filename, 'w') as f:
        f.write(f'tolerance {tolerance}\n')
        f.write('filetype xyz\n')
        f.write(f'output {output_file}\n')
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
