from src.controller.ga import GeneticAlgorithm, GAConfig
from src.view.plots import plot_param_pca
from ..model.molecule import Molecule
from ..model.population import Population
from ..model.fitness import compute_fitness_penalized, novelty_augmented_fitness
import random

soup = ['[C][#N]', '[C][=O]', '[C][O]', '[C][C][O]', '[C][C][=O]', '[O][=C][C][O]', '[O][=C][O]', '[N][C][=Branch1][C][=O][N]', '[N]', '[O]', '[N][C][C][=Branch1][C][=O][O]', '[C][C][Branch1][=Branch1][C][=Branch1][C][=O][O][N]', '[C][C][=Branch1][C][=O][O]', '[C][C][N]', '[C][S]', '[C][C][=Branch1][C][=O][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][C]', '[O][=C][=O]', '[O][=C][=S]', '[O][P][=Branch1][C][=O][Branch1][C][O][O]', '[C][=C][C][=C][C][=C][Ring1][=Branch1]', '[C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=C][NH1][C][=Ring1][Branch1]', '[C][C][C][C][C][Ring1][Branch1]', '[C][C][C][C][C][C][Ring1][=Branch1]']

initial = []
for s in soup:
    initial.append(Molecule(s))

pop = Population(initial)

cfg = GAConfig(
    mu=20,
    lam=40,
    mutation_rate=0.5,
    crossover_rate=0.8,
    tournament_k=2,
    random_seed=0
)



def tune_hyperparameters(
    base_cfg,
    fitness_fn,
    validation_smiles,
    n_trials=10
):
    best_params = None
    best_score = float("inf")

    top_results = [] 

    for trial in range(n_trials):
        # sample random weights
        params = {
            "novelty_weight": random.uniform(-25, 25),
            "w_energy": random.uniform(-5, 5),
            "w_tpsa": random.uniform(-10, 10),
            "w_logp": random.uniform(-50, 50),
            "w_carbonpct": random.uniform(-250, 250)
        }

        print("\n=== Trial", trial, "params:", params)

        # initialize soup
        initial = [Molecule(s) for s in soup]
        pop = Population(initial)

        # create GA with sampled params
        ga = GeneticAlgorithm(
            base_cfg, fitness_fn,
            **params
        )

        # evolve with early stopping
        history = []
        gens_to_converge = 50


        for gen in range(50):      # max generations
            pop = ga.evolve_one_generation(pop)
            history.append(pop)

            if ga.has_converged(history):
                print(f"Converged at generation {gen}")
                break
            if gens_to_converge == gen:
                print(f"Converged at generation {gen}")
                break

        # compute validation score
        score = pop.compute_validation_knn_distance(validation_smiles)
        print("Validation score:", score)

        top_results.append((score, params, gen))

        # update best
        if score < best_score:
            best_score = score
            best_params = params

    top_results = sorted(top_results, key=lambda x: x[0])[:10]
    print("\n======= TOP 10 RESULTS =======")
    for rank, (sc, pr, gen) in enumerate(top_results, 1):
        print(f"#{rank} | score={sc:.4f} | gens={gen} | params={pr}")

    return best_params, best_score

validation_smiles=[
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
]

best_params, score = tune_hyperparameters(cfg, novelty_augmented_fitness, validation_smiles)
print("Best:", best_params, "Score:", score)
