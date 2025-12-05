# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function

from .novelty import NoveltyArchive

# Global novelty archive
archive = NoveltyArchive(k=5)

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
    return fitness

# Normalization and scaling functions can be added as needed, or change the weights so it works better in practice.

## Utility/Helper Functions
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