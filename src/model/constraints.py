
# TODO Trial version that can be updated

# TODO FIX THIS
from rdkit.Chem import Descriptors

def check_constraints(mol):
    """Return penalty score for violations (lower is better)."""
    penalties = 0

    # Size constraint example
    if mol.rdkit_mol.GetNumAtoms() > 30:
        penalties += 50

    # LogP constraint
    logp = Descriptors.MolLogP(mol.rdkit_mol)
    if logp > 5:
        penalties += 20

    return penalties