import optuna
from functools import partial
from src.view.plots import plot_param_pca,plot_param_pca_3d 
from src.controller.ga import GeneticAlgorithm, GAConfig
from src.model.molecule import Molecule
from src.model.population import Population
from src.model.fitness import novelty_augmented_fitness


soup = ['[C][#N]', '[C][=O]', '[C][O]', '[C][C][O]', '[C][C][=O]', '[O][=C][C][O]', '[O][=C][O]', '[N][C][=Branch1][C][=O][N]', '[N]', '[O]', '[N][C][C][=Branch1][C][=O][O]', '[C][C][Branch1][=Branch1][C][=Branch1][C][=O][O][N]', '[C][C][=Branch1][C][=O][O]', '[C][C][N]', '[C][S]', '[C][C][=Branch1][C][=O][C][=Branch1][C][=O][O]', '[C][C][=Branch1][C][=O][C]', '[O][=C][=O]', '[O][=C][=S]', '[O][P][=Branch1][C][=O][Branch1][C][O][O]', '[C][=C][C][=C][C][=C][Ring1][=Branch1]', '[C][=C][N][=C][NH1][Ring1][Branch1]', '[C][C][=C][NH1][C][=Ring1][Branch1]', '[C][C][C][C][C][Ring1][Branch1]', '[C][C][C][C][C][C][Ring1][=Branch1]']
 # your prebiotic SELFIES

cfg = GAConfig(
    mu=20,
    lam=40,
    mutation_rate=0.5,
    crossover_rate=0.8,
    tournament_k=2,
    random_seed=0
)
# ---------------------------
# OBJECTIVE FUNCTION
# ---------------------------
def objective(trial, base_cfg, validation_smiles, top_results):

    # Suggest weights
    params = {
        "novelty_weight": trial.suggest_float("novelty_weight", -5, 5),
        "w_energy": trial.suggest_float("w_energy", -1, 1),
        "w_tpsa": trial.suggest_float("w_tpsa", -2, 2),
        "w_logp": trial.suggest_float("w_logp", -10, 10),
        "w_carbonpct": trial.suggest_float("w_carbonpct", -50, 50),
    }

    print("\n=== Trial params:", params)

    # Initialize population
    pop = Population([Molecule(s) for s in soup])

    # Create GA instance with dynamic weights
    ga = GeneticAlgorithm(base_cfg, novelty_augmented_fitness, **params)

    # Run with early stopping
    history = []
    max_gens = 30

    for gen in range(max_gens):
        pop = ga.evolve_one_generation(pop)
        history.append(pop)

        if ga.has_converged(history):
            print(f"Converged at generation {gen}")
            break

    # Compute validation score
    score = pop.compute_validation_knn_distance(validation_smiles)
    print("Validation score =", score)

    # Save into top_results
    top_results.append((score, params, gen))

    return score


# ---------------------------
# OPTUNA TUNING WRAPPER
# ---------------------------
def tune_with_optuna(base_cfg, validation_smiles, n_trials=10):

    top_results = []   # <-- store all trial results

    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(trial, base_cfg, validation_smiles, top_results),
        n_trials=n_trials
    )

    # ----------------------------------------------------------
    # Produce TOP 10 summary as before
    # ----------------------------------------------------------
    plot_param_pca_3d(top_results)
    top_results_sorted = sorted(top_results, key=lambda x: x[0])[:10]

    print("\n======= TOP 10 RESULTS =======")
    for rank, (sc, pr, gen) in enumerate(top_results_sorted, 1):
        print(f"#{rank} | score={sc:.4f} | gens={gen} | params={pr}")

    print("\n===== BEST OVERALL =====")
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)

    return study.best_params, study.best_value

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

best_params, best_score = tune_with_optuna(cfg, validation_smiles, n_trials=30)