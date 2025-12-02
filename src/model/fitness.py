# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function


from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from .novelty import NoveltyArchive
from .constraints import check_constraints

# Global novelty archive
archive = NoveltyArchive(k=5)

# Deprecated Fitness Functions
# def stability_fitness(mol):
#     energy = mol.compute_mmff_energy()
#     penalty = check_constraints(mol)
#     return energy + penalty

# TODO verify
def novelty_augmented_fitness(mol, novelty_weight=1):
    penalized_fitness = compute_fitness_penalized(mol)
    novelty = archive.novelty_score(mol)
    return penalized_fitness + novelty_weight * (1 - novelty)

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

# Updated fitness function with symmetric penalties
def compute_fitness_penalized(
        molecule,
        w_energy=0.01, #1
        w_tpsa=0.1,    #1
        w_logp=0.2):    #1

    # Normalize MMFF energy per heavy atom
    E = molecule.compute_mmff_energy() / max(1, molecule.heavy_atom_count)
    TPSA = molecule.tpsa
    logP = molecule.log_p

    # Target ranges (tunable)
    TPSA_low, TPSA_high = 40, 180
    logP_low, logP_high = 0, 5
    E_low, E_high = 3, 40   # kcal/mol per heavy atom

    # Compute penalties
    p_tpsa = range_penalty(TPSA, TPSA_low, TPSA_high, w_tpsa)
    p_logp = range_penalty(logP, logP_low, logP_high, w_logp)
    p_energy = range_penalty(E, E_low, E_high, w_energy)

    # Fitness = sum of penalties (lower = better)
    fitness = p_energy + p_tpsa + p_logp
    molecule.fitness = fitness
    return fitness


# TODO this needs to be changed to receive a molecule and not a population
# This is weird, but maybe we could calculate it without normalization first, and then normalize afterwards
def compute_population_fitness(molecule, population):
    """
    Computes all descriptors, normalizes them, computes fitness,
    and stores fitness in molecule.fitness.
    """

    # --- 1) FIRST compute novelty ---
    archive = NoveltyArchive()
    score = archive.novelty_score(molecule)
    # compute_population_novelty(population) # TODO this shouldn't be here, maybe put it up a level

    # --- 2) Compute raw descriptors --- # TODO these don't need to be lists
    energies = []
    tpsas = []
    logps = []
    novelties = []
    carbon_counts = []

    for mol in population:  # TODO remove the for and make it just a single molecule
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

    # --- 3) Normalize all properties --- # TODO This might all be able to be done in a single line
    E_norm     = normalize(energies)
    TPSA_norm  = normalize(tpsas)
    LogP_norm  = normalize(logps)
    Novelty_norm = normalize(novelties)
    Carbon_norm = normalize(carbon_counts)

    # --- 4) Compute final fitness for each molecule --- # TODO remove the for, do only for the one molecule
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

    return population # TODO return fitness

# Normalization and scaling functions can be added as needed, or change the weights so it works better in practice.

## Utility Functions
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


# Fitness penalized for abs(MMFF, TPSA, logP)
# Two-sided penalty helper
def range_penalty(x, low, high, weight):
    """
    penalizes x when it falls outside [low, high].
    Returns 0 when inside the range.
    """
    if x < low:
        return weight * (low - x)**2
    elif x > high:
        return weight * (x - high)**2
    return 0.0