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

def carbon_pct_constraint(molecule, min_pct: float) -> bool:
    """Returns True if percentage of carbon atoms exceeds max_carbons (violation)."""
    if molecule.heavy_atom_count == 0:
        return True  # avoid division by zero; treat as violation

    if molecule.num_carbons is None:
        molecule.count_carbons()
    carbon_pct = molecule.num_carbon / molecule.heavy_atom_count
    return carbon_pct < min_pct

def check_constraints(molecule, constraints: dict={"sanitize":True, "min_carbon_pct":0.3, "size":25}) -> bool:
    CONSTRAINT_FUNCTIONS = {
    "size": size_constraint,
    "sanitize": sanitization_constraint,
    "min_carbon_pct": carbon_pct_constraint
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

