# Defines experiments you can run from CLI:
# Defines experiments you can run from CLI:

import argparse
from ..model.molecule import Molecule
from ..model.population import Population
from ..model.fitness import compute_fitness
from .ga import GeneticAlgorithm, GAConfig
from .novelty_search import NoveltySearch

# Default initial molecules
initial_selfies = ["[C][C][O]", "[C][O][O]", "[C][C][C]"]

parser = argparse.ArgumentParser(description="Run molecular evolution experiments.")

parser.add_argument("--algo", choices=["ga","novelty"], default="ga", help="Algorithm to use.")
parser.add_argument("--gens", default=50, type=int, help="Number of generations.")

parser.add_argument("--mu", type=int, default=50, help="Parent population size μ.")
parser.add_argument("--lam", type=int, default=50, help="Offspring population size λ.")
parser.add_argument("--mutation", type=float, default=0.3, help="Mutation rate.")
parser.add_argument("--crossover", type=float, default=0.9, help="Crossover rate.")
parser.add_argument("--tournament_k", type=int, default=3, help="Tournament selection size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

args = parser.parse_args()

mol_objs = [Molecule(s) for s in initial_selfies]
pop = Population(mol_objs)

algo = None
if args.algo == "ga":
    config = GAConfig(
        mu=args.mu,
        lam=args.lam,
        mutation_rate=args.mutation,
        crossover_rate=args.crossover,
        tournament_k=args.tournament_k,
        random_seed=args.seed,
    )
    algo = GeneticAlgorithm(config, stability_fitness)

elif args.algo == "novelty":
    algo = NoveltySearch()

# Run evolution
history = algo.evolve(pop, args.gens)

# print history
count = 1
for h in history:
    print("Generation: ", count)
    for mol in h.molecules:
        print("Smiles: ", mol.smiles, "Selfies: ", mol.selfies)
    count += 1

print("Finished evolution.")