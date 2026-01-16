# Here we have different fitness functions (penalized and with novelty components)
# We have several weighted components in the combined penalty 
# (including MMFF94 energy computation, topological surface area, 
# logp as a hydrophobicity measure, number of heteroatoms in carbon chains)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from .novelty import NoveltyArchive
from .constraints import check_constraints

# global novelty archive
archive = NoveltyArchive(k=5)

# TODO verify

# the first naive fitness function which minimizes MMFF energy and logP (hydrophobicity),
# and maximizes tpsa (solubility). Used just as a test
def compute_fitness(molecule, w_energy=1.0, w_tpsa=0.35, w_logP=0.2):
    E = molecule.compute_mmff_energy() / max(1, molecule.heavy_atom_count)
    TPSA = molecule.tpsa
    logP = molecule.log_p

    # MINIMIZATION fitness function
    fitness = (
            w_energy * E  # lower is better
            - w_tpsa * TPSA  # higher TPSA is better
            + w_logP * logP  # lower logP is better
    )
    return fitness

# another fitness function with a range_penalty 
# (treats MMFF, tpsa, logp and heteroatom parameters thats fall inside 
# a given range as 0 penalty)
def compute_dummy_fitness(
                molecule,
        w_energy=0.01,
        w_tpsa=0.1,
        w_logp=0.2,
        w_hetero=0.1):

    E = molecule.compute_mmff_energy() / max(1, molecule.heavy_atom_count)
    TPSA = molecule.tpsa
    logP = molecule.log_p

    TPSA_low, TPSA_high = 40, 180
    logP_low, logP_high = 0, 5
    E_low, E_high = 3, 40

    p_tpsa = range_penalty(TPSA, TPSA_low, TPSA_high, w_tpsa)
    p_logp = range_penalty(logP, logP_low, logP_high, w_logp)
    p_energy = range_penalty(E, E_low, E_high, w_energy)
    p_hetero = w_hetero * hetero_distribution_penalty(molecule)

    fitness = p_energy + p_tpsa + p_logp + p_hetero
    molecule.fitness = fitness
    return fitness



# updated penalty fitness function with band penalties
# (penalizes deviation from the given "sweet spot"
# even inside the plausible range of each descriptor)
def compute_fitness_penalized(
        molecule,
        w_energy=0.01,
        w_tpsa=0.1,
        w_logp=0.2,
        w_hetero=0.1):

    E = molecule.compute_mmff_energy() / max(1, molecule.heavy_atom_count)
    TPSA = molecule.tpsa
    logP = molecule.log_p

    # setting low and high values for each descriptor (based on common chemical sense and literature)
    TPSA_low, TPSA_high = 0, 200
    logP_low, logP_high = -3, 9
    E_low, E_high = -8, 6

    p_tpsa = band_penalty(TPSA, TPSA_low, TPSA_high, w_tpsa)
    p_logp = band_penalty(logP, logP_low, logP_high, w_logp)
    p_energy = band_penalty(E, E_low, E_high, w_energy)
    p_hetero = w_hetero * hetero_distribution_penalty(molecule)

    fitness = p_energy + p_tpsa + p_logp + p_hetero
    molecule.fitness = fitness
    return fitness

# fitness function with novelty component (select the weight manually)
# based on band penalized fitness function
def novelty_augmented_fitness(mol, novelty_weight=0.1):
    penalized = compute_fitness_penalized(mol)
    novelty = archive.novelty_score(mol)
    return penalized + novelty_weight * (1 - novelty)


# Helper functions:
# function for penalizing many heteroatoms inside carbon chains
# this is not in the constraints due to the fact that we want some heteroatoms inside,
# and don't want to restrict them fully
def hetero_distribution_penalty(
    mol,
    max_hetero_frac=0.4,
    interior_w=0.4,
    hh_adj_w=0.2,
    frac_w=0.3,
):
    """
    Penalizes (a) interior hetero atoms (non-C) in aliphatic chains,
    (b) hetero–hetero adjacencies, and (c) overall hetero fraction above a band.
    Lower = better. Tune weights manually.
    """
    rm = mol.rdkit_mol
    if rm is None:
        return 5.0  # hard penalty on invalid mol

    atoms = list(rm.GetAtoms())
    heavy = rm.GetNumHeavyAtoms() or 1
    hetero_idxs = [a.GetIdx() for a in atoms if a.GetAtomicNum() != 6]

    # overall heteroatom fraction penalty (soft cap)
    hetero_frac = len(hetero_idxs) / heavy
    p_frac = 0.0
    if hetero_frac > max_hetero_frac:
        p_frac = frac_w * (hetero_frac - max_hetero_frac) ** 2

    # interior heteroatoms in chains: not in ring, degree >=2, with >=2 carbon neighbors (sp3-ish chain)
    p_interior = 0.0
    for idx in hetero_idxs:
        a = atoms[idx]
        if a.IsInRing():
            continue  # allow hetero in rings
        carbon_neighbors = [n for n in a.GetNeighbors() if n.GetAtomicNum() == 6]
        if len(carbon_neighbors) >= 2:
            p_interior += interior_w  # flat penalty per interior hetero

    # heteroatom-heteroatom adjacencies (penalty for having heteroatoms together)
    p_hh = 0.0
    for bond in rm.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() != 6 and a2.GetAtomicNum() != 6:
            p_hh += hh_adj_w

    return p_frac + p_interior + p_hh

# Fitness penalized for abs(MMFF, TPSA, logP)
# Two-sided range penalty helper (inside the range penalty = 0, soft)
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

# band penalty (quadratic penalty depending on how far from the "sweet spot"
# of each deascriptor we are). The sweet spot is computed based on the low and high values 
# passed as parameters for each descriptor
def band_penalty(x, low, high, weight):
    if x is None:
        return weight * 10
    mid = 0.5 * (low + high)
    halfw = 0.5 * (high - low)
    return weight * ((x - mid) / halfw) ** 2
