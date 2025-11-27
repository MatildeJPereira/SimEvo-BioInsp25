from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# ---------- raw molecule metrics ----------
def _mmff(mol):
    rd = Chem.MolFromSmiles(mol.smiles)
    if rd is None:
        return None
    rd = Chem.AddHs(rd)
    try:
        AllChem.EmbedMolecule(rd, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(rd)
        props = AllChem.MMFFGetMoleculeProperties(rd)
        ff = AllChem.MMFFGetMoleculeForceField(rd, props)
        return ff.CalcEnergy()
    except:
        return None


def _tpsa(mol):
    rd = Chem.MolFromSmiles(mol.smiles)
    return None if rd is None else Descriptors.TPSA(rd)


def _logp(mol):
    rd = Chem.MolFromSmiles(mol.smiles)
    return None if rd is None else Descriptors.MolLogP(rd)


def _normalize(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return [0] * len(values)
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) if v is not None else 0 for v in values]

def build_population_fitness(population):
    """
    Precompute normalized descriptors for the whole population,
    then return a function f(molecule) -> fitness_value.
    """

    mols = list(population.molecules)

    # 1) Compute raw descriptors
    raw_E = []
    raw_T = []
    raw_L = []
    raw_C = []

    for mol in mols:
        mol.count_carbons()

        e = _mmff(mol)
        t = _tpsa(mol)
        l = _logp(mol)
        c = mol.num_carbons / max(1, mol.heavy_atom_count)

        raw_E.append(None if e is None else e / max(1, mol.heavy_atom_count))
        raw_T.append(t)
        raw_L.append(l)
        raw_C.append(c)

        # store raw results inside molecule (optional but useful)
        mol.mmff_energy = e
        mol.tpsa = t
        mol.log_p = l

    # 2) Normalize population-wide
    E_norm = _normalize(raw_E)
    T_norm = _normalize(raw_T)
    L_norm = _normalize(raw_L)
    C_norm = _normalize(raw_C)

    # 3) Map molecule → its normalized descriptor set
    descriptor_map = {
        mol: (e, t, l, c)
        for mol, e, t, l, c in zip(mols, E_norm, T_norm, L_norm, C_norm)
    }

    # 4) Return the per-molecule fitness function
    def fitness_fn(mol):
        # When GA calls fitness_fn on newly created offspring,
        # they won't be in descriptor_map → recompute raw score only.
        if mol not in descriptor_map:
            # fallback: compute *unnormalized* score
            e = raw_E[0] if raw_E else 0
            return e  # or something safe

        e, t, l, c = descriptor_map[mol]

        return (
            e
            - 0.35 * t
            + 0.15 * l
            - 0.10 * c
        )

    return fitness_fn
