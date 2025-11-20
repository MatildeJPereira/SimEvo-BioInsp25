# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function

# TODO Trial version that can be updated
from .constraints import check_constraints

def stability_fitness(mol):
    energy = mol.compute_mmff_energy()
    penalty = check_constraints(mol)
    return energy + penalty