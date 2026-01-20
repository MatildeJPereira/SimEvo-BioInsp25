# Command-line runner for molecular evolution using the Genetic Algorithm.
# This script initializes a population, configures the GA, and runs evolution for a user-specified
# number of generations.

import argparse
from ..model.molecule import Molecule
from ..model.population import Population
from ..model.fitness import novelty_augmented_fitness
from .ga import GeneticAlgorithm, GAConfig

# Command-line arguments
parser = argparse.ArgumentParser(description="Run molecular evolution experiments.")

parser.add_argument("--gens", default=50, type=int, help="Number of generations.")

# GA Hyperparameters
parser.add_argument("--mu", type=int, default=50, help="Parent population size μ.")
parser.add_argument("--lam", type=int, default=50, help="Offspring population size λ.")
parser.add_argument("--mutation", type=float, default=0.3, help="Mutation rate.")
parser.add_argument("--crossover", type=float, default=0.9, help="Crossover rate.")
parser.add_argument("--tournament_k", type=int, default=3, help="Tournament selection size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

args = parser.parse_args()

# Initial seed molecules
initial_selfies = [
        '[C][#N]', '[C][=O]', '[C][O]', '[C][C][O]', '[C][C][=O]', '[O][=C][C][O]', '[O][=C][O]',
        '[N][C][=Branch1][C][=O][N]', '[N]', '[O]', '[N][C][C][=Branch1][C][=O][O]',
        '[C][C][Branch1][=Branch1][C][=Branch1][C][=O][O][N]', '[C][C][=Branch1][C][=O][O]', '[C][C][N]', '[C][S]',
        '[C][C][=Branch1][C][=O][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][C]', '[O][=C][=O]', '[O][=C][=S]',
        '[O][P][=Branch1][C][=O][Branch1][C][O][O]', '[C][=C][C][=C][C][=C][Ring1][=Branch1]',
        '[C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=C][NH1][C][=Ring1][Branch1]', '[C][C][C][C][C][Ring1][Branch1]',
        '[C][C][C][C][C][C][Ring1][=Branch1]', '[N][C][=N][C][=C][N][Ring1][Branch1]',
        '[C][C][=C][O][C][=Ring1][Branch1]', '[O][C][C][=Branch1][C][=O][O]',
        '[C][=N][C][=N][C][NH1][C][=N][C][Ring1][=Branch2][=Ring1][Branch1]',
        '[O][P][=Branch1][C][=O][Branch1][C][O][O][P][=Branch1][C][=O][Branch1][C][O][O]',
        '[C][C][Branch1][C][O][C][=Branch1][C][=O][O]', '[O][=C][C][Branch1][C][O][C][O]',
        '[O][=C][Branch1][Ring1][C][O][C][O]', '[N][C][=O]', '[C][=C]', '[C][C][C][=Branch1][C][=O][O]',
        '[O][=C][Branch1][C][O][C][C][C][=Branch1][C][=O][O]', '[N][C][C][S]', '[N][C][=Branch1][C][=S][N]',
        '[O][C][C@H1][O][C][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Ring1][#Branch1][O]']

# Build initial population
pop = Population([Molecule(s) for s in initial_selfies])

# Configure GA
config = GAConfig(
    mu=args.mu,
    lam=args.lam,
    mutation_rate=args.mutation,
    crossover_rate=args.crossover,
    tournament_k=args.tournament_k,
    random_seed=args.seed,
)

# Use novelty-augmented penalized fitness
fitness_fn = lambda m: novelty_augmented_fitness(m, novelty_weight=2.0)
ga = GeneticAlgorithm(config, fitness_fn)

# Run evolution
history = ga.evolve(pop, args.gens)

# Print results
for gen_idx, pop in enumerate(history):
    print(f"\n=== Generation {gen_idx} ===")
    for mol in pop.molecules:
        print(f"SMILES: {mol.smiles}    SELFIES: {mol.selfies}")

print("Finished evolution.")