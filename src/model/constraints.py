from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

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
    carbon_pct = molecule.num_carbons / molecule.heavy_atom_count
    return carbon_pct < min_pct


def ring_size_constraint(molecule, min_size=5, max_size=6) -> bool:
    """Returns True if any ring is outside the allowed size window."""
    rm = molecule.rdkit_mol
    if rm is None:
        return True
    sizes = [len(r) for r in rm.GetRingInfo().AtomRings()]
    return any(s < min_size or s > max_size for s in sizes)


def charge_constraint(molecule, max_abs=1) -> bool:
    """Returns True if formal charge magnitude exceeds max_abs."""
    rm = molecule.rdkit_mol
    if rm is None:
        return True
    return abs(Chem.GetFormalCharge(rm)) > max_abs


def check_constraints(molecule, constraints=None) -> bool:
    """
        Evaluates all constraints in the dictionary.
        Returns:
            False -> at least one constraint violated;
            True -> all constraints satisfied.
    """
    if constraints is None:
        constraints = {
            "size": 100,
            "sanitize": True,
            "min_carbon_pct": 0.4,
            "ring_size": (5, 6),
            "max_abs_charge": 1,
        }
    constraint_function = {
        "size": size_constraint,
        "sanitize": sanitization_constraint,
        "min_carbon_pct": carbon_pct_constraint,
        "ring_size": ring_size_constraint,
        "max_abs_charge": charge_constraint
        # add new constraints here later without touching main code
    }

    for name, target in constraints.items():
        
        if name not in constraint_function:
            raise ValueError(f"Unknown constraint: {name}")

        func = constraint_function[name]

        # If constraint has a parameter (e.g., max_size)
        if isinstance(target, bool):
            violated = func(molecule)   # no parameters
        elif isinstance(target, tuple):
            violated = func(molecule, *target)
        else:
            violated = func(molecule, target)  # pass parameter (e.g., size limit)

        if violated:
            return False  # stop early: violation found

    return True
