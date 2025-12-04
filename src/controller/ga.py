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
    tournament_k: int = 2
    rank_bias: float = 1.7
    elitism: bool = True
    random_seed: int = 42

# ----------------------------
# Selection
# ----------------------------
def tournament_selection(pop, fitness,k):
    candidates = random.sample(pop,k)
    return min(candidates, key=lambda m: fitness[m])


def rank_selection(pop, fitness, bias=1.7):
    """
    Rank-based parent selection for minimization problems.
    - Sorts molecules by fitness (lower = better).
    - Assigns exponentially decaying weights controlled by `bias` (>1 → stronger pressure).
    """
    if not pop:
        return None

    ranked = sorted(pop, key=lambda m: fitness[m])  # best first
    n = len(ranked)
    # Exponential rank weights: best gets the highest weight
    weights = [bias ** (n - 1 - i) for i in range(n)]
    total = sum(weights)

    r = random.random() * total
    acc = 0.0
    for mol, w in zip(ranked, weights):
        acc += w
        if acc >= r:
            return mol
    return ranked[-1]

# ----------------------------
# Replacement (μ + λ)
# ----------------------------
def mu_plus_lambda(parents, offspring, fitness_fn, mu):
    combined = parents + offspring
    combined.sort(key=lambda m: fitness_fn(m))
    return combined[:mu]

# another replacement strategy
def mu_comma_lambda(offspring, fitness_fn, mu):
    """
    (μ, λ) selection:
    - parents are used only to generate offspring
    - next generation is the best μ offspring
    """
    offspring.sort(key=lambda m: fitness_fn(m))
    return offspring[:mu]

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
