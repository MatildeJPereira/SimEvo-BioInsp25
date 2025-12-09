# Plots with matplotlib or seaborn:
# - mean/median fitness per generation
# - diversity scores
# - archive growth

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_fitness_over_time(history):
    means = []
    for pop in history:
        vals = list(pop.fitness.values())
        means.append(sum(vals) / len(vals))

    plt.plot(means)
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.show()

def plot_all_atom_stats(history):
    comp = [h[3] for h in history]
    plt.plot(comp, label="distance")

    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Atom Composition Evolution")
    plt.legend()
    plt.show()


def plot_param_pca(top_results):
    """
    top_results = list of (score, params, gens)
    """

    # convert to DataFrame
    df = pd.DataFrame([
        {
            **params,
            "score": score,
            "gens": gens
        }
        for score, params, gens in top_results
    ])

    # extract only the parameter columns
    param_cols = [c for c in df.columns if c not in ("score", "gens")]
    X = df[param_cols].values

    # PCA compress to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df["score"], cmap="viridis", s=70
    )
    plt.colorbar(scatter, label="Validation Score (lower = better)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Hyperparameter Landscape (PCA)")
    plt.show()

    print("Explained variance:", pca.explained_variance_ratio_)

###############################################################################
# 1. Convert your parameter search results into a matrix
###############################################################################
def params_to_matrix(top_results):
    param_names = list(top_results[0][1].keys())
    X = []
    fitness = []

    for score, params, gen in top_results:
        row = [params[p] for p in param_names]
        X.append(row)
        fitness.append(score)

    return np.array(X, dtype=float), np.array(fitness), param_names


###############################################################################
# 2. Manual PCA implementation (no sklearn needed)
###############################################################################
def manual_pca(X, n_components=2):
    # standardize
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]

    # project onto principal components
    PC = X_centered @ eigvecs[:, :n_components]

    return PC, eigvecs[:, :n_components], eigvals[:n_components], X_mean


###############################################################################
# 3. Plotting utilities
###############################################################################
from scipy.interpolate import griddata

def plot_surface_landscape(PC, fitness, resolution=80):
    """
    Interpolates irregular PCA points into a smooth surface.
    """
    x = PC[:, 0]
    y = PC[:, 1]
    z = fitness

    # Create a grid in PCA space
    xi = np.linspace(min(x), max(x), resolution)
    yi = np.linspace(min(y), max(y), resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate fitness onto the grid
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

    # Plot surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        Xi, Yi, Zi,
        cmap="viridis",
        alpha=0.85,
        linewidth=0,
        antialiased=True
    )

    # Scatter original points on top (optional)
    ax.scatter(x, y, z, color="black", s=20, label="Samples")

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("Fitness")
    ax.set_title("Hyperparameter Landscape (Smoothed Surface)")

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def plot_contour_landscape(PC, fitness, resolution=120):
    x = PC[:, 0]
    y = PC[:, 1]
    z = fitness

    xi = np.linspace(min(x), max(x), resolution)
    yi = np.linspace(min(y), max(y), resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(Xi, Yi, Zi, levels=40, cmap="viridis")
    plt.scatter(x, y, c=z, cmap="viridis", edgecolor="white", s=50)

    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("Hyperparameter Landscape (2D Contour Map)")
    plt.colorbar(cp, label="Fitness")
    plt.show()



###############################################################################
# 4. Print parameters + PCA coordinates
###############################################################################
def print_interpretable_table(top_results, PC, param_names):
    print("\n========= PCA Coordinates for Each Trial =========")
    print("(Lower fitness = better)")
    header = ["Fitness", "PCA1", "PCA2"] + param_names
    print("\t".join(f"{h:>12}" for h in header))
    print("-"*120)

    for (score, params, gen), (pc1, pc2) in zip(top_results, PC):
        row = [f"{score:.4f}", f"{pc1:.4f}", f"{pc2:.4f}"] + [f"{params[p]:.3f}" for p in param_names]
        print("\t".join(f"{v:>12}" for v in row))


###############################################################################
# MAIN FUNCTION TO CALL
###############################################################################
def pca_landscape(top_results):
    X, fitness, param_names = params_to_matrix(top_results)

    # compute PCA
    PC, eigvecs, eigvals, mean = manual_pca(X)

    # print table
    print_interpretable_table(top_results, PC, param_names)

    # 3D landscape plot
    plot_surface_landscape(PC, fitness)

    plot_contour_landscape(PC, fitness)


