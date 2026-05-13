def smiles_to_xyz(smiles: str, output_path: str = 'molecule.xyz') -> str:
    """
    Convert a SMILES string to a 3D XYZ file using RDKit.

    Generates 3D coordinates with ETKDG and optimises with UFF force field.
    If the output file already exists it is returned immediately without
    regenerating coordinates.

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

    if os.path.exists(output_path):
        print(f"Output file '{output_path}' already exists. Skipping SMILES conversion.")
        return output_path

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
