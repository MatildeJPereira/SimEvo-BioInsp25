# Plots with matplotlib or seaborn:
# - mean/median fitness per generation
# - diversity scores
# - archive growth

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd   # <- new

sns.set_theme(style="whitegrid", context="talk")  # nicer default style


def _mean_scores_from_history(history):
    """Convert a GA history into a list of mean fitness scores per generation."""
    means = []
    for pop in history:
        vals = list(pop.fitness.values())
        if not vals:
            means.append(float("nan"))
            continue
        mean_penalty = sum(vals) / len(vals)
        means.append(-mean_penalty)   # penalties -> scores (higher = better)
    return means


def plot_fitness_over_time(history, label=None, ax=None, **line_kws):
    """
    Plot a single run. Can also be used inside a multi-plot by passing an Axes.
    """
    means = _mean_scores_from_history(history)
    gens = list(range(len(means)))

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


def plot_multiple_fitness_histories(histories, labels=None):
    """
    histories: list of GA histories
    labels:   list of strings (same length as histories)
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

    plt.figure(figsize=(10, 6),  facecolor="#C2A5CF")
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
    ax.set_title("Fitness Across Simulations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    plt.tight_layout()
    plt.show()