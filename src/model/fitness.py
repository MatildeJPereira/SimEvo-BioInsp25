# MMFF94 energy computation
# Penalties for chemical cnstraints (charge, valence, size)
# Combined fitness function

def compute_mmff94_energy(molecule):
    # Placeholder for MMFF94 energy computation
    energy = 0.0
    # ... compute energy based on molecule structure ...
    return energy

def compute_fitness(molecule):
    energy = compute_mmff94_energy(molecule)
    MolTPSA = molecule.log_p  # Topological Polar Surface Area
    MolLogP = molecule.tpsa # Octanol-water partition coefficient
    fitness = energy + 0.35 * MolTPSA - 0.15 * MolLogP
    return fitness

# Nromilization and scaling functions can be added as needed, or change the weights so it works better in practice.