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

    for mol in population:
        e = compute_mmff94_energy(mol)
        tpsa, logp = compute_descriptors(mol)

        mol.mmff_energy = e
        mol.tpsa = tpsa
        mol.log_p = logp
        
        energies.append(e/max(1,mol.heavy_atom_count))  # normalize by size
        tpsas.append(tpsa)
        logps.append(logp)
        novelties.append(mol.novelty)

    # --- 3) Normalize all properties ---
    E_norm     = normalize(energies)
    TPSA_norm  = normalize(tpsas)
    LogP_norm  = normalize(logps)
    Novelty_norm = normalize(novelties)

    # --- 4) Compute final fitness for each molecule ---
    for mol, e_n, t_n, l_n, n_n in zip(population, E_norm, TPSA_norm, LogP_norm, Novelty_norm):
        
        # Lower energy = better → stability = (1 - normalized energy)
        #stability_score = 1.0 - e_n
        
        mol.fitness = (
            e_n
            - 0.35 * t_n
            + 0.15 * l_n
            - 0.05 * n_n
        )

    return population


# Nromilization and scaling functions can be added as needed, or change the weights so it works better in practice.