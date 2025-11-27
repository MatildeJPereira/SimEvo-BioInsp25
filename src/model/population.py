# Population class storing molecules, their fitness, and metadata
# Utility methods for selection (tournament, roulette, novelty-based…)

import random

class Population:
    def __init__(self, molecules):
        self.molecules = molecules
        self.fitness = {}

    def evaluate(self, fitness_fn):
        for mol in self.molecules:
            self.fitness[mol] = fitness_fn(mol)

    def select_tournament(self, k=3):
        candidates = random.sample(self.molecules, k)
        return min(candidates, key=lambda m: self.fitness[m])