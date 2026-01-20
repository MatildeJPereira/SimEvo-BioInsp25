# Genetic Algorithm implementation (μ + λ) for evolving SELFIES-encoded molecules.
# Provides parent selection, mutation, crossover, constraint filtering, and evaluation.

import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..model.constraints import check_constraints
from ..model.population import Population
from ..model.operators import mutate_selfies, crossover_selfies
from ..model.molecule import Molecule

# ----------------------------
# GA Configuration
# ----------------------------
@dataclass
class GAConfig:
    """Configuration object for GA hyperparameters."""
    mu: int = 50                # parent population size
    lam: int = 50               # offspring population size
    mutation_rate: float = 0.3
    crossover_rate: float = 0.9
    tournament_k: int = 2
    rank_bias: float = 1.7      # controls pressure in rank selection
    elitism: bool = True
    random_seed: int = 42

# ----------------------------
# Selection
# ----------------------------
def tournament_selection(pop, fitness, k):
    """Return best of k randomly sampled molecules (minimization)."""
    candidates = random.sample(pop,k)
    return min(candidates, key=lambda m: fitness[m])


def rank_selection(pop, fitness, bias=1.7):
    """
    Rank-based parent selection for minimization problems.

    - Sorts molecules by fitness (lower = better).
    - Assigns exponentially decaying weights controlled by `bias` (>1 -> stronger pressure).
    """
    if not pop:
        return None

    ranked = sorted(pop, key=lambda m: fitness[m])  # best first
    n = len(ranked)

    # Exponential rank weights
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
# Replacement
# ----------------------------
def mu_plus_lambda(parents, offspring, fitness_fn, mu):
    """(μ + λ) replacement: select best μ from combined pool."""
    combined = parents + offspring
    combined.sort(key=lambda m: fitness_fn(m))
    return combined[:mu]

# mu comma lambda replacement strategy
def mu_comma_lambda(parents, offspring, fitness_fn, mu):
    """
    Strict (μ,λ) selection:

    - next generation is the best μ offspring only
    - parents never survive
    """
    if len(offspring) < mu:
        raise RuntimeError(
            f"(μ,λ) requires at least mu={mu} offspring, got {len(offspring)}. "
            "Increase max_attempts / lam, or relax constraints."
        )
    offspring.sort(key=lambda m: fitness_fn(m))
    return offspring[:mu]

# ----------------------------
# GA Class
# ----------------------------
ParentSelector = Callable[[Population], Molecule]
Replacer = Callable[[list, list, Callable, int], list]

class GeneticAlgorithm:
    """Main GA executor performing selection, variation, and replacement."""

    def __init__(
        self,
        config: GAConfig,
        fitness_fn: Callable,
        parent_selector: Optional[ParentSelector] = None,
        replacer: Optional[Replacer] = None,
    ):
        self.cfg = config
        self.fitness_fn = fitness_fn
        self.parent_selector = parent_selector
        self.replacer = replacer
        random.seed(config.random_seed)

    # Initialization
    def initialize(self, population):
        """Compute initial fitness for all molecules."""
        population.evaluate(self.fitness_fn)

    # Selection helpers
    def select_parent(self, population):
        """Select a parent using configured method (default: tournament)."""
        if self.parent_selector is None:
            return tournament_selection(
                population.molecules,
                population.fitness,
                self.cfg.tournament_k,
            )
        return self.parent_selector(population)

    # Variation operators
    def produce_offspring(self, parent1, parent2):
        """
        Apply crossover and mutation.
        Return new Molecule or None if invalid.
        """
        # Crossover
        if random.random() < self.cfg.crossover_rate:
            child_selfies = crossover_selfies(parent1.selfies, parent2.selfies)
        else:
            child_selfies = parent1.selfies

        # Mutation
        if random.random() < self.cfg.mutation_rate:
            child_selfies = mutate_selfies(child_selfies)

        new_mol = Molecule(child_selfies)

        # Reject if constraint violation occurs
        return new_mol if check_constraints(new_mol) else None

    #Single generation
    def evolve_one_generation(self, population):
        """Produce one new generation using μ + λ replacement (default)."""
        from ..model.fitness import archive # global novelty archive

        parents = population.molecules
        offspring = []
        attempts = 0

        target = max(self.cfg.lam, self.cfg.mu) # ensure enough offspring
        max_attempts = 50 * target              # safety cap

        # Variation loop
        while len(offspring) < target and attempts < max_attempts:
            attempts += 1
            p1 = self.select_parent(population)
            p2 = self.select_parent(population)

            new_offspring = self.produce_offspring(p1, p2)
            if new_offspring is not None:
                offspring.append(new_offspring)

        if len(offspring) < target:
            raise RuntimeError(
                f"Could not generate enough valid offspring: {len(offspring)}/{target} "
                f"after {attempts} attempts. Constraints too strict or max_attempts too low."
            )

        acceptance_rate = len(offspring) / attempts if attempts else 0.0
        print(f"Accepted offspring: {len(offspring)}/{attempts} (rate={acceptance_rate:.2f})")

        # Replacement (default μ + λ)
        replacer = self.replacer or mu_plus_lambda
        new_pop = replacer(parents, offspring, self.fitness_fn, self.cfg.mu)

        # Evaluate new population
        new_population = Population(new_pop)
        new_population.evaluate(self.fitness_fn)

        # Update novelty archive
        best = min(new_population.molecules, key=lambda m: new_population.fitness[m])
        archive.add(best)

        return new_population

    # Multi-generation driver
    def evolve(self, population, generations):
        """Run evolution for a number of generations and return history list."""
        self.initialize(population)
        history = [population]  # generation 0

        for gen in range(generations):
            print("Generation ", gen)
            population = self.evolve_one_generation(population)
            history.append(population)

        return history
