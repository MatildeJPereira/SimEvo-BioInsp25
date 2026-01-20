# Utility plotting functions for analyzing GA history objects.
# This module provides:
# - Mean fitness curve per generation
# - Multi-run comparison plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# Global seaborn theme
sns.set_theme(style="whitegrid", context="talk")

# Internal helper
def _mean_scores_from_history(history):
    """
    Compute mean fitness per generation from GA history.

    Notes:

    - Fitness is minimized in the GA, so the plot uses the negative mean to display 'better' values as upward trends.
    - None or NaN entries are skipped.
    """
    means = []
    for pop in history:
        # Filter out invalid or missing values
        vals = [v for v in pop.fitness.values() if v is not None and pd.notna(v)]
        if len(vals) == 0:
            means.append(None)
        else:
            means.append(-(sum(vals) / len(vals)))

    # Forward-fill initial None values so curve starts at generation 0
    first_valid = next((m for m in means if m is not None), None)
    if first_valid is not None:
        means = [first_valid if m is None else m for m in means]

    return means


# Single-run Plotting
def plot_fitness_over_time(history, label=None, ax=None, **line_kws):
    """
    Plot a single run. Can also be used inside a multi-plot by passing an Axes.
    """
    means = _mean_scores_from_history(history)
    gens = list(range(len(means)))

    # Create a new figure if no axes were provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    default_kws = dict(linewidth=2.5, marker="o")
    default_kws.update(line_kws)

    sns.lineplot(x=gens, y=means, ax=ax, label=label, **default_kws)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    if label is None:
        ax.set_title("GA Mean Fitness Over Time")

    if ax.figure:
        ax.figure.tight_layout()

    if ax is None:
        plt.show()


# Multi-run Comparison
def plot_multiple_fitness_histories(histories, labels=None):
    """
    Plot mean fitness curves from multiple GA runs on a single figure.

    - histories: list of GA histories
    - labels:   list of strings (same length as histories)
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(histories))]

    rows = []
    for run_idx, history in enumerate(histories):
        run_label = labels[run_idx]
        means = _mean_scores_from_history(history)
        for gen, score in enumerate(means):
            rows.append(
                {"Generation": gen, "Mean Fitness": score, "Run": run_label}
            )

    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6),  facecolor="#84cda1")
    ax = sns.lineplot(
        data=df,
        x="Generation",
        y="Mean Fitness",
        hue="Run",
        style="Run",
        palette="magma",
        markers=True,
        dashes=False,
        linewidth=2.5,
    )
    ax.grid(False)
    ax.set_title("Fitness Comparison Across Selection and Replacement Strategies")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness (-Penalty Score)")
    plt.tight_layout()
    plt.show()