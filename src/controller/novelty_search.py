# A variant using:
# - novelty score instead of fitness
# - archive updating strategy
# - selection based on novelty ranking or multi-objective (fitness + novelty)

# TODO trial version that can be changed
from ..model.novelty import NoveltyArchive
from ..model.operators import mutate_selfies
from ..model.molecule import Molecule
import random

class NoveltySearch:
    def __init__(self, archive_k=5, mutation_rate=0.5):
        self.archive = NoveltyArchive(k=archive_k)
        self.mutation_rate = mutation_rate

    def evolve(self, population, generations):
        history = []

        for gen in range(generations):
            scores = {m: self.archive.novelty_score(m) for m in population.molecules}

            population.molecules.sort(key=lambda m: scores[m], reverse=True)
            self.archive.add(population.molecules[0])

            offspring = []
            for parent in population.molecules:
                child_selfies = mutate_selfies(parent.selfies)
                offspring.append(Molecule(child_selfies))

            population.molecules = offspring
            history.append(population)

        return history