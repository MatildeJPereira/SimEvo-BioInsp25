import numpy as np
import random
from src.controller.ga import GeneticAlgorithm, GAConfig
from src.model.molecule import Molecule
from src.model.population import Population
from src.model.fitness import novelty_augmented_fitness
from src.view.plots import pca_landscape

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

# ===============================================================
#  CMA-ES IMPLEMENTATION (lightweight, numpy-only)
# ===============================================================

def build_fitness_wrapper(base_fitness_fn, params):
    def wrapped_fitness(m, novelty_weight=None, w_energy=None, w_tpsa=None, w_logp=None, w_carbonpct=None):
        return base_fitness_fn(
            m,
            novelty_weight=params["novelty_weight"],
            w_energy=params["w_energy"],
            w_tpsa=params["w_tpsa"],
            w_logp=params["w_logp"],
            w_carbonpct=params["w_carbonpct"]
        )
    return wrapped_fitness

def cma_es_optimize(objective_fn, x0, sigma=1.0, population_size=20, generations=30):
    """
    Very lightweight CMA-ES implementation (no sklearn, no external libs).
    objective_fn: function taking a vector -> fitness
    x0: initial parameter guess (vector)
    sigma: initial search step size
    population_size: λ offspring
    """

    n = len(x0)
    mu = population_size // 2       # number of parents
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1 / np.sum(weights**2)

    # Strategy parameter settings
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))

    # Initialize
    mean = np.array(x0)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    C = np.eye(n)

    history = []

    for gen in range(generations):
        # === Sample offspring ===
        R = np.linalg.cholesky(C)
        population = [mean + sigma * (R @ np.random.randn(n)) for _ in range(population_size)]

        # === Evaluate ===
        fitnesses = [objective_fn(ind) for ind in population]

        # === Sort ===
        idx = np.argsort(fitnesses)
        population = [population[i] for i in idx]
        fitnesses = [fitnesses[i] for i in idx]

        # === Save history ===
        history.append((fitnesses[0], population[0]))

        # === Recompute mean ===
        old_mean = mean.copy()
        mean = np.sum(weights[:, None] * np.array(population[:mu]), axis=0)

        # === Evolution paths ===
        y = mean - old_mean
        C_inv_sqrt = np.linalg.inv(R.T)
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y / sigma))

        h_sigma = int((np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) < (1.4 + 2 / (n + 1)))

        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y / sigma)

        # === Update covariance matrix ===
        artmp = [(pop - old_mean) / sigma for pop in population[:mu]]
        C = (1 - c1 - c_mu) * C + \
            c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + \
            c_mu * sum(w * np.outer(a, a) for w, a in zip(weights, artmp))

        # === Step-size control ===
        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))

    return history


def tune_params_cmaes(
    base_cfg,
    fitness_fn,
    validation_smiles,
    initial_guess=[1, 1, 1, 1, 1],
    sigma=2.0,
    generations=20,
    pop_size=30
):

    param_names = ["novelty_weight", "w_energy", "w_tpsa", "w_logp", "w_carbonpct"]

    def objective(vec):
        params = dict(zip(param_names, vec))
        fitness_wrapped = build_fitness_wrapper(fitness_fn, params)

        initial = [Molecule(s) for s in soup]
        pop = Population(initial)

        ga = GeneticAlgorithm(base_cfg, fitness_wrapped)

        history = []
        for gen in range(40):
            pop = ga.evolve_one_generation(pop)
            history.append(pop)
            if ga.has_converged(history):
                break

        return pop.compute_validation_knn_distance(validation_smiles)

    # === Run CMA-ES ===
    history = cma_es_optimize(
        objective_fn=objective,
        x0=np.array(initial_guess),
        sigma=sigma,
        population_size=pop_size,
        generations=generations
    )

    # history contains (score, params_vector)
    scored = [(score, dict(zip(param_names, vec))) for score, vec in history]


    # === Top 10 ===
    top10 = sorted(scored, key=lambda x: x[0])[:10]

    print("\n===== TOP 10 CMA-ES RESULTS =====")
    for i, (score, params) in enumerate(top10, 1):
        print(f"#{i} | score={score:.4f} | params={params}")

    # === PCA landscape ===
    param_matrix = np.array([list(d.values()) for _, d in scored])
    param_matrix_centered = param_matrix - param_matrix.mean(axis=0)
    cov = np.cov(param_matrix_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    PC = param_matrix_centered @ eigvecs[:, :2]

    # === PCA LOADINGS ===
    loadings = eigvecs[:, :2]  # params × PC

    print("\n===== PCA LOADINGS =====")
    for name, (a, b) in zip(param_names, loadings):
        print(f"{name:>15} | PC1={a:.4f} | PC2={b:.4f}")

    return top10, PC, loadings


top10, PC, loadings = tune_params_cmaes(
    cfg,
    novelty_augmented_fitness,
    validation_smiles,
    initial_guess=[0, 1, 1, 1, 10],
    sigma=4.0,
    generations=3,
    pop_size=12
)

