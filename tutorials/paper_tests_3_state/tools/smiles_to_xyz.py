def smiles_to_xyz(smiles, output_path='molecule.xyz'):
    """
    Convert a SMILES string to a 3D XYZ file using RDKit.

    Generates 3D coordinates with ETKDG and optimises with UFF force field.

    Parameters
    ----------
    smiles      : str -- SMILES string of the molecule
    output_path : str -- path for the output .xyz file

    Returns
    -------
    str -- path to the written XYZ file
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import os

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        raise RuntimeError('3D embedding failed for SMILES: ' + smiles)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    symbols   = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [
        (conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f'{len(symbols)}\n')
        f.write(f'SMILES: {smiles}\n')
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f'{sym:<3} {x:>12.6f} {y:>12.6f} {z:>12.6f}\n')
    return output_path

import subprocess
import os

def packmol_build_system(xyz_files, box_size, n_molecules=1, output_file='system.xyz', tolerance=2.5):
    """
    Build a molecular system using Packmol by placing one or more molecules
    in a cubic box.

    Parameters
    ----------
    xyz_files   : str or list of str -- path(s) to input XYZ file(s)
    box_size    : float              -- size of the cubic box (angstroms)
    n_molecules : int or list of int -- number of copies for each molecule
    output_file : str                -- output file path (default: 'system.xyz')
    tolerance   : float              -- minimum atom-atom distance (angstroms)

    Returns
    -------
    str -- path to the generated system XYZ file
    """
    if isinstance(xyz_files, str):
        xyz_files = [xyz_files]
    if isinstance(n_molecules, int):
        n_molecules = [n_molecules] * len(xyz_files)
    if len(xyz_files) != len(n_molecules):
        raise ValueError('Length of xyz_files and n_molecules must match')

    if subprocess.call(['which', 'packmol'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise EnvironmentError('Packmol is not found in PATH. Please install or load it first.')

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
