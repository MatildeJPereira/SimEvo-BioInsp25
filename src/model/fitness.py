# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def compute_mmff94_energy(molecule):
    mol = Chem.MolFromSmiles(molecule.smiles)
    if mol is None:
        return None  # invalid
    molH = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(molH, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(molH)
        props = AllChem.MMFFGetMoleculeProperties(molH)
        ff = AllChem.MMFFGetMoleculeForceField(molH, props)
        return ff.CalcEnergy()
    except:
        return None  # optimization failed
    
def compute_descriptors(molecule):
    mol = Chem.MolFromSmiles(molecule.smiles)
    if mol is None:
        return None, None
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    return tpsa, logp

def normalize(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return [0 for _ in values]

    min_v = min(vals)
    max_v = max(vals)

    if max_v == min_v:  # avoid division by zero
        return [0.5 for _ in values]  # all identical → neutral

    return [( (v - min_v) / (max_v - min_v) ) if v is not None else 0 
            for v in values]

def compute_population_fitness(population):
    """
    population: list of molecule objects
    Each molecule must have .smiles attribute.
    Computes energy, TPSA, LogP, normalizes, 
    then computes and assigns molecule.fitness
    """

    energies = []
    tpsas = []
    logps = []

    # --- compute raw properties for each molecule ---
    for mol in population:
        e = compute_mmff94_energy(mol)
        tpsa, logp = compute_descriptors(mol)

        mol.mmff_energy = e
        mol.tpsa = tpsa
        mol.log_p = logp

        energies.append(e/mol.heavy_atom_count if e is not None else None)
        tpsas.append(tpsa)
        logps.append(logp)

    # --- normalize ---
    E_norm = normalize(energies)
    TPSA_norm = normalize(tpsas)
    LogP_norm = normalize(logps)

    # --- compute fitness for all ---
    for mol, e_n, t_n, l_n in zip(population, E_norm, TPSA_norm, LogP_norm):
        # Your proposed formula:
        # Fitness = stability + 0.35*TPSA_norm - 0.15*LogP_norm
        # Here, energy should be *inverted* (lower energy = better):
        #stability_score = 1.0 - e_n

        mol.fitness = e_n - 0.35*t_n + 0.15*l_n

    return population


# Nromilization and scaling functions can be added as needed, or change the weights so it works better in practice.