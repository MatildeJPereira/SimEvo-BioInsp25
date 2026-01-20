# Utility functions for visualizing molecules outside the pygame environment.
# This module provides:
# - A grid-based visualization of population molecules using RDKit's
#   built-in drawing utilities.

from rdkit.Chem.Draw import MolsToImage

def population_grid(population, n=16, subimg_size=(300, 300)):
    """Returns an RDKit-generated grid image of the first n molecules in the population."""
    mols = [m.rdkit_mol for m in population.molecules[:n]]
    return MolsToImage(mols, molsPerRow=4, subImgSize=subimg_size)