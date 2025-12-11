from src.controller.ga import GeneticAlgorithm, GAConfig
from src.model.molecule import Molecule
from src.model.population import Population
from src.model.fitness import novelty_augmented_fitness
from src.view.soup import MolecularSoupPygame

soup = ['[C][#N]', '[C][=O]', '[C][O]', '[C][C][O]', '[C][C][=O]', '[O][=C][C][O]', '[O][=C][O]',
        '[N][C][=Branch1][C][=O][N]', '[N]', '[O]', '[N][C][C][=Branch1][C][=O][O]',
        '[C][C][Branch1][=Branch1][C][=Branch1][C][=O][O][N]', '[C][C][=Branch1][C][=O][O]', '[C][C][N]', '[C][S]',
        '[C][C][=Branch1][C][=O][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][C]', '[O][=C][=O]', '[O][=C][=S]',
        '[O][P][=Branch1][C][=O][Branch1][C][O][O]', '[C][=C][C][=C][C][=C][Ring1][=Branch1]',
        '[C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=C][NH1][C][=Ring1][Branch1]', '[C][C][C][C][C][Ring1][Branch1]',
        '[C][C][C][C][C][C][Ring1][=Branch1]']

initial = []
for s in soup:
    initial.append(Molecule(s))

pop = Population(initial)

cfg = GAConfig(mu=25,
    lam=25,
    mutation_rate=0.5,
    crossover_rate=0.5,
    tournament_k=2,
    random_seed=0)

ga = GeneticAlgorithm(cfg, lambda m: novelty_augmented_fitness(m))

def evolve_callback(population):
    new_pop = ga.evolve_one_generation(population)
    print("Generation updated!")
    return new_pop

ga.initialize(pop)
soup = MolecularSoupPygame(pop)
soup.run(update_callback=evolve_callback)