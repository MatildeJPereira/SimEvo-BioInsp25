# Main GA pipeline
# selection -> crossover -> mutation -> evaluation -> replacement (μ + λ)

import random
from dataclasses import dataclass

from ..model.constraints import check_constraints
from ..model.population import Population
from ..model.operators import mutate_selfies, crossover_selfies
from ..model.molecule import Molecule

# ----------------------------
# GA Configuration
# ----------------------------
@dataclass
class GAConfig:
    mu: int = 50
    lam: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.9
    tournament_k: int = 3
    elitism: bool = True
    random_seed: int = 42

# ----------------------------
# Selection
# ----------------------------
def tournament_selection(pop, fitness,k):
    candidates = random.sample(pop,k)
    return min(candidates, key=lambda m: fitness[m])

# ----------------------------
# Replacement (μ + λ)
# ----------------------------
def mu_plus_lambda(parents, offspring, fitness_fn, mu):
    combined = parents + offspring
    combined.sort(key=lambda m: fitness_fn(m))
    return combined[:mu]

# ----------------------------
# Modular Genetic Algorithm
# ----------------------------
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
        if random.random() < self.cfg.crossover_rate:
            child_selfies = crossover_selfies(parent1.selfies, parent2.selfies)
        else:
            child_selfies = parent1.selfies

        if random.random() < self.cfg.mutation_rate:
            child_selfies = mutate_selfies(child_selfies)

        new_mol = Molecule(child_selfies)
        if check_constraints(new_mol):
            return new_mol

        return None

    def evolve_one_generation(self, population):
        from ..model.fitness import archive
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

        # Update novelty archive with best molecule
        best = min(new_population.molecules, key=lambda m: new_population.fitness[m])
        archive.add(best)

        return new_population

    def evolve(self, population, generations):
        self.initialize(population)
        history = []

        for gen in range(generations):
            print("Generation ", gen)
            population = self.evolve_one_generation(population)
            history.append(population)

        return history