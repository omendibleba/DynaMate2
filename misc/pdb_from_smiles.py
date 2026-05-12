from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_pdb(smiles: str, output_file: str):
    """
    Convert a SMILES string to a 3D PDB file.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule
    output_file : str
        Name of the output PDB file
    """

    # Create RDKit molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # Optimize geometry (MMFF if available, otherwise UFF)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        AllChem.UFFOptimizeMolecule(mol)

    # Write PDB file
    output_file = output_file + ".pdb"
    Chem.MolToPDBFile(mol, output_file)

    print(f"\n\nPDB file written to {output_file}")

#smiles_to_pdb("CCO", "ethanol.pdb")