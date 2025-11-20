import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit.Chem.Draw import MolToImage
import random

class MolecularSoup:
    def __init__(self, population, box_size=10):
        self.population = population
        self.box_size = box_size

        # Random positions
        self.positions = {
            mol: np.array([random.uniform(0, box_size), random.uniform(0, box_size)])
            for mol in population.molecules
        }

        # Pre-render molecule images
        self.images = {
            mol: MolToImage(mol.rdkit_mol, size=(80, 80))
            for mol in population.molecules
        }

    def update_population(self, population):
        """Update soup with new generation molecules."""
        self.population = population

        # Reassign positions but keep same mapping count
        old_positions = list(self.positions.values())
        self.positions = {
            mol: old_positions[i % len(old_positions)]
            for i, mol in enumerate(population.molecules)
        }

        # Re-render images
        self.images = {
            mol: MolToImage(mol.rdkit_mol, size=(80, 80))
            for mol in population.molecules
        }

    def step_motion(self):
        """Brownian random motion."""
        for mol, pos in self.positions.items():
            self.positions[mol] += np.random.normal(scale=0.2, size=2)

            # Keep within bounds
            self.positions[mol] = np.clip(self.positions[mol], 0, self.box_size)

    def draw(self, ax):
        ax.clear()
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)

        for mol, pos in self.positions.items():
            img = OffsetImage(self.images[mol], zoom=0.3)
            ab = AnnotationBbox(img, pos, frameon=False)
            ax.add_artist(ab)
