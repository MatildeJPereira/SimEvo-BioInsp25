# Population class for storing molecules and their fitness values.
# Provides evaluation and parent-selection utilities.

import random

class Population:
    """Holds a list of molecules and a fitness dictionary."""

    def __init__(self, molecules):
        self.molecules = molecules
        self.fitness = {} # maps molecule -> fitness value

    def evaluate(self, fitness_fn):
        """Compute and store fitness values for all molecules."""
        for mol in self.molecules:
            self.fitness[mol] = fitness_fn(mol)

    def select_tournament(self, k=3):
        """
        k-way tournament selection.

        Returns the fittest among k random molecules.
        """
        candidates = random.sample(self.molecules, k)
        return min(candidates, key=lambda m: self.fitness[m])