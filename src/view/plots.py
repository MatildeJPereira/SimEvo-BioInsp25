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



def plot_param_pca_3d(top_results):
    df = pd.DataFrame([
        {**params, "score": score}
        for score, params, gens in top_results
    ])

    param_cols = [c for c in df.columns if c != "score"]
    X = df[param_cols].values

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_pca[:,0], X_pca[:,1], X_pca[:,2],
        c=df["score"], cmap="viridis", s=60
    )
    fig.colorbar(scatter, label="Validation Score")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title("3D PCA Hyperparameter Landscape")
    plt.show()


