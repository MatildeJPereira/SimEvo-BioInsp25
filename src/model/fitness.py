# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function


from .constraints import check_constraints
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from .novelty import compute_population_novelty

# Deprecated
# def stability_fitness(mol):
#     energy = mol.compute_mmff_energy()
#     penalty = check_constraints(mol)
#     return energy + penalty

# Working Fitness function
def compute_fitness(molecule, w_energy=1.0, w_tpsa=0.35, w_logP=0.15):
    E = molecule.compute_mmff_energy() / max(1, molecule.heavy_atom_count)
    TPSA = molecule.tpsa
    logP = molecule.log_p

    # MINIMIZATION fitness function
    fitness = (
            w_energy * E  # lower is better
            - w_tpsa * TPSA  # higher TPSA lowers fitness (good)
            + w_logP * logP  # higher logP raises fitness (bad)
    )
    return fitness
    
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
    Computes all descriptors, normalizes them, computes fitness,
    and stores fitness in molecule.fitness.
    """

    # --- 1) FIRST compute novelty ---
    compute_population_novelty(population)

    # --- 2) Compute raw descriptors ---
    energies = []
    tpsas = []
    logps = []
    novelties = []
    carbon_counts = []

    for mol in population:
        mol.count_carbons()
        e = mol.compute_mmff_energy()
        tpsa, logp = compute_descriptors(mol)

        mol.mmff_energy = e
        mol.tpsa = tpsa
        mol.log_p = logp
        
        energies.append(e/max(1,mol.heavy_atom_count))  # normalize by size
        tpsas.append(tpsa)
        logps.append(logp)
        novelties.append(mol.novelty)
        carbon_counts.append(mol.num_carbons/mol.heavy_atom_count)

    # --- 3) Normalize all properties ---
    E_norm     = normalize(energies)
    TPSA_norm  = normalize(tpsas)
    LogP_norm  = normalize(logps)
    Novelty_norm = normalize(novelties)
    Carbon_norm = normalize(carbon_counts)

    # --- 4) Compute final fitness for each molecule ---
    for mol, e_n, t_n, l_n, n_n, c_n in zip(population, E_norm, TPSA_norm, LogP_norm, Novelty_norm, Carbon_norm):
        
        # Lower energy = better → stability = (1 - normalized energy)
        #stability_score = 1.0 - e_n
        
        mol.fitness = (
            e_n
            - 0.35 * t_n
            + 0.15 * l_n
            - 0.05 * n_n
            - 0.10 * c_n
        )

    return population

# Normalization and scaling functions can be added as needed, or change the weights so it works better in practice.
