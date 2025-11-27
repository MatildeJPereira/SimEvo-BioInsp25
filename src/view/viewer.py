# Uses RDKit to:
# - draw molecules
# - maybe render small grids of top molecules per generation

from rdkit.Chem.Draw import MolsToImage

def population_grid(population, n=16):
    mols = [m.rdkit_mol for m in population.molecules[:n]]
    return MolsToImage(mols, molsPerRow=4)