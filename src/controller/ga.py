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
    def __init__(self, config: GAConfig, fitness_fn, novelty_weight=0.05,w_energy=0.01,w_tpsa=0.02,w_logp=0.1, w_hetero=0.5):
        self.cfg = config
        self.fitness_fn = fitness_fn
        random.seed(config.random_seed)
        self.novelty_weight = novelty_weight
        self.w_energy = w_energy
        self.w_tpsa = w_tpsa
        self.w_logp = w_logp
        self.w_hetero = w_hetero
    def initialize(self, population):
        population.evaluate(
            self.fitness_fn,
            novelty_weight=self.novelty_weight,
            w_energy=self.w_energy,
            w_tpsa=self.w_tpsa,
            w_logp=self.w_logp,
            w_hetero=self.w_hetero
        )

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

    def has_converged(self, history, threshold=1e-3, patience=5):
        if len(history) < patience:
            return False

        vals = [p.fitness_stats["mean"] for p in history[-patience:]]
        return max(vals) - min(vals) < threshold

    def evolve_one_generation(self, population):


            # 🔥 FIX: ensure fitness exists before parent selection
        if len(population.fitness) == 0:
            population.evaluate(
                self.fitness_fn,
                novelty_weight=self.novelty_weight,
                w_energy=self.w_energy,
                w_tpsa=self.w_tpsa,
                w_logp=self.w_logp,
                w_hetero=self.w_hetero
            )
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
        new_population.evaluate(self.fitness_fn,
    novelty_weight=self.novelty_weight,
    w_energy=self.w_energy,
    w_tpsa=self.w_tpsa,
    w_logp=self.w_logp,
    w_hetero=self.w_hetero)
        return new_population

    def evolve(self, population, generations):
        self.initialize(population)
        history = []

        for gen in range(generations):
            print("Generation ", gen)
            population = self.evolve_one_generation(population)
            avg_carbons = population.compute_carbon_avg()
            avg_other_atoms = population.compute_other_atoms_avg()
            validation_distance = population.compute_validation_knn_distance(validation_smiles=[
    # === 1. Alpha amino acids ===
    "NCC(=O)O",
    "NC(C)C(=O)O",
    "NCC(O)C(=O)O",
    "NC(CC(=O)O)C(=O)O",
    "NC(CCC(=O)O)C(=O)O",
    "NC(C(C)C)C(=O)O",
    "NCC(CO)C(=O)O",
    "N1CCCC1C(=O)O",
    "NC(Cc1ccccc1)C(=O)O",
    "NC(CS)C(=O)O",
    "NC(C=O)C(=O)O",
    "NC(CN)C(=O)O",
    "NCC(C)C(=O)O",
    "NC(CO)C(=O)O",
    "NC(Cc1[nH]cnc1)C(=O)O",
    "NC(Cc1ccc(O)cc1)C(=O)O",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "NC(Cc1ccc(CO)cc1)C(=O)O",
    "NC(COO)C(=O)O",
    "NC(CCO)C(=O)O",

    # === 2. Beta/gamma amino acids ===
    "NCCC(=O)O",
    "NCCCC(=O)O",
    "NCC(O)C(=O)O",
    "NCCCO",
    "NCCC(O)C(=O)O",

    # === 3. Hydroxy acids ===
    "CC(O)C(=O)O",        # lactic acid
    "OCC(=O)O",           # glycolic acid
    "OCC(O)C(=O)O",       # malic acid
    "OC(=O)CC(=O)O",      # succinic acid
    "O=C(O)CCC(=O)O",     # 4-hydroxybutyrate precursor

    # === 4. Keto acids ===
    "CC(=O)C(=O)O",       # pyruvate
    "O=C(O)C(=O)O",       # oxalate
    "O=C(O)CC(=O)O",      # malonate
    "O=C(O)CCC(=O)O",     # succinate
    "O=C(O)C(=O)C(=O)O",  # oxalosuccinate-like

    # === 5. Short dipeptides (very proto-bio) ===
    "NCC(=O)NCC(=O)O",        # Gly-Gly
    "NC(C)C(=O)NCC(=O)O",     # Ala-Gly
    "NCC(=O)NC(C)C(=O)O",     # Gly-Ala
    "NCC(=O)NC(CO)C(=O)O",    # Gly-Ser

    # === 6. Polyamines (prebiotic catalysts) ===
    "NCCN",
    "NCCCCN",
    "NCCCN",
    "NCCNCCN",
    "NCCCCCCN",
    "NCCCNCCN",

    # === 7. Small prebiotic N-containing molecules ===
    "NC=O",          # formamide
    "NC#N",          # cyanamide
    "N=C=O",         # isocyanic acid
    "N=C(N)N",       # diaminocarbene precursor
    "CNC=O",         # methylformamide

    # === 8. Nucleobase-like fragments ===
    "O=C1NC=NC=N1",            # uracil-like core
    "NC1=NC=NC=N1",            # adenine fragment
    "N1C=NC=N1",               # diaminopyrimidine
    "O=CNC=N",                 # formamidine-urea
    "O=C1NCCN1",               # imidazolidone

    # === 9. TCA/glycolysis intermediates ===
    "O=CC(O)C(=O)O",           # glycerate
    "O=CC(=O)CO",              # glyoxylate
    "O=CC(O)CO",               # glyceraldehyde
    "OC(CO)C(=O)O",            # hydroxypropionate

    # === 10. Cofactor/fragments (small size only) ===
    "NC(=O)c1ccccn1",          # nicotinamide fragment
    "O=C(O)c1ccccn1",          # pyridinecarboxylate
    "c1ncc[nH]1",              # pyrimidine fragment
    "OC[C@H](O)[C@H](O)CO",    # sugar alcohol fragment (glycerol aldehyde-like)
    "O=C(O)c1ccncc1",          # pyridyl-carboxylate

    # === 11. Extra proto-bio organic acids ===
    "CC(=O)OC(=O)C",           # acetoacetate-like
    "CCOC(=O)C",               # ethyl acetate precursor
    "CC(=O)CO",                # acetoacetic alcohol
    "CCC(O)C(=O)O",            # 2-hydroxybutyrate
])

            history.append((population,avg_carbons,avg_other_atoms,validation_distance))

        return history
