# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function

from .constraints import check_constraints

# Deprecated
def stability_fitness(mol):
    energy = mol.compute_mmff_energy()
    penalty = check_constraints(mol)
    return energy + penalty

# Working Fitness function
def compute_fitness(molecule, w_energy=1.0, w_tpsa=0.35, w_logP=0.15):
    E = molecule.compute_mmff_energy()
    TPSA = molecule.tpsa
    logP = molecule.log_p

    # MINIMIZATION fitness function
    fitness = (
            w_energy * E  # lower is better
            - w_tpsa * TPSA  # higher TPSA lowers fitness (good)
            + w_logP * logP  # higher logP raises fitness (bad)
    )
    return fitness