
# TODO Trial version that can be updated

# TODO FIX THIS
# from rdkit.Chem import Descriptors
#
# def check_constraints(mol):
#     """Return penalty score for violations (lower is better)."""
#     penalties = 0
#
#     # Size constraint example
#     if mol.rdkit_mol.GetNumAtoms() > 30:
#         penalties += 50
#
#     # LogP constraint
#     logp = Descriptors.MolLogP(mol.rdkit_mol)
#     if logp > 5:
#         penalties += 20
#
#     return penalties

from rdkit import Chem

def size_constraint(molecule, max_size: int) -> bool:
    """Returns True if molecule exceeds max size (violation)."""
    return molecule.heavy_atom_count > max_size

def sanitization_constraint(molecule) -> bool:
    """
    Returns True if sanitization fails (violation).
    """
    if molecule.smiles is None:
        return True  # invalid

    try:
        mol = Chem.MolFromSmiles(molecule.smiles, sanitize=True)
        return mol is None  # True → violation
    except:
        return True  # sanitizer crashed → definitely invalid

def check_constraints(molecule, constraints: dict={"sanitize":True}) -> bool:
    CONSTRAINT_FUNCTIONS = {
    "size": size_constraint,
    "sanitize": sanitization_constraint,
    # add new constraints here later without touching main code
}
    """
    Evaluates all constraints in the dictionary.
    Returns:
        True  -> at least one constraint violated
        False -> all constraints satisfied
    """
    for name, target in constraints.items():
        
        if name not in CONSTRAINT_FUNCTIONS:
            raise ValueError(f"Unknown constraint: {name}")

        func = CONSTRAINT_FUNCTIONS[name]

        # If constraint has a parameter (e.g., max_size)
        if isinstance(target, bool):
            violated = func(molecule)   # no parameters
        else:
            violated = func(molecule, target)  # pass parameter (e.g., size limit)

        if violated:
            return True  # stop early: violation found

    return False 

