# Main GA pipeline
# selection -> crossover -> mutation -> evaluation -> replacement (μ + λ)

#Try saving the top 10 of each generation (like in hall of fame) but then delete them from the pop to increase exploration

import random
from dataclasses import dataclass

from ..model.constraints import check_constraints
from ..model.population import Population
from ..model.operators import mutate_selfies, crossover_selfies
from ..model.molecule import Molecule
from ..model.novelty import NoveltyArchive



@dataclass
class GAConfig:
    mu: int = 50
    lam: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.9
    tournament_k: int = 3
    elitism: bool = True
    random_seed: int = 42

def tournament_selection(pop, fitness,k):
    candidates = random.sample(pop,k)
    return min(candidates, key=lambda m: fitness[m])

def mu_plus_lambda(parents, offspring, fitness_fn, mu):
    combined = parents + offspring
    combined.sort(key=lambda m: fitness_fn(m))
    return combined[:mu]

class GeneticAlgorithm:
    def __init__(self, config: GAConfig, fitness_fn):
        self.cfg = config
        self.fitness_fn = fitness_fn
        random.seed(config.random_seed)

    def initialize(self, population):
        population.evaluate(self.fitness_fn)

    def select_parent(self, population):
        return tournament_selection(
            population.molecules,
            population.fitness,
            self.cfg.tournament_k,
        )

    def produce_offspring(self, parent1, parent2):
        violated = True
        n = 0

        while violated and n < 20:
            if random.random() < self.cfg.crossover_rate:
                child_selfies = crossover_selfies(parent1.selfies, parent2.selfies)
            else:
                child_selfies = parent1.selfies

            if random.random() < self.cfg.mutation_rate:
                child_selfies = mutate_selfies(child_selfies)

            new_mol = Molecule(child_selfies)

            violated = check_constraints(new_mol)
            n += 1

        # If we failed 20 times, return *something* valid-ish
        if violated:
            return None  # fallback

        return new_mol

    def evolve_one_generation(self, population):
        parents = population.molecules

        offspring = []

        for _ in range(self.cfg.lam):
            p1 = self.select_parent(population)
            p2 = self.select_parent(population)
            new_offspring = self.produce_offspring(p1, p2)
            if new_offspring is not None:
                offspring.append(new_offspring)

        new_pop = mu_plus_lambda(parents, offspring, self.fitness_fn, self.cfg.mu)

        new_population = Population(new_pop)
        new_population.evaluate(self.fitness_fn)
        return new_population

    def evolve(self, population, generations):
        self.initialize(population)
        history = []

        for gen in range(generations):
            print("Generation ", gen)
            population = self.evolve_one_generation(population)
            avg_carbons = population.compute_carbon_avg()
            avg_other_atoms = population.compute_other_atoms_avg()
            validation_distance = population.compute_validation_knn_distance([

    # Amino acids
    "NCC(=O)O", "NC(C)C(=O)O", "NCC(O)C(=O)O", "NC(CC(=O)O)C(=O)O",
    "NC(CCC(=O)O)C(=O)O", "NC(C(C)C)C(=O)O", "NCC(CO)C(=O)O", "N1CCCC1C(=O)O",
    "NC(C(CC)C)C(=O)O", "NC(Cc1ccccc1)C(=O)O",

    # Nucleobases
    "O=c1[nH]c[nH]c1=O", "Nc1ncnc2[nH]cnc12", "O=c1cc[nH]c(=O)[nH]1",
    "NC(=O)c1ncc[nH]1", "O=c1[nH]cc(=O)n(C)1", "O=c1[nH]c(=O)[nH]c[nH]1",

    # Metabolites
    "CC(=O)C(=O)O", "CC(O)C(=O)O", "OCC(CC(=O)O)C(=O)O", "O=C(O)C=CC(=O)O",

    # Cofactor fragments
    "OC[C@H](O)[C@H](O)[C@H](O)CO", "NC(=O)c1ccccn1", "c1ncc[nH]1", "O=C(O)c1ccccn1"
])

            history.append((population,avg_carbons,avg_other_atoms,validation_distance))

        return history
