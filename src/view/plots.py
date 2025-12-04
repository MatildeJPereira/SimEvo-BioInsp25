# Plots with matplotlib or seaborn:
# - mean/median fitness per generation
# - diversity scores
# - archive growth

import matplotlib.pyplot as plt
import seaborn as sns

def plot_fitness_over_time(history):
    means = []
    for pop in history:
        vals = list(pop.fitness.values())
        if not vals:
            continue
        # Stored fitness values are penalties (lower is better). Plot as a score: higher is better.
        mean_penalty = sum(vals) / len(vals)
        mean_score = -mean_penalty
        means.append(mean_score)

    sns.set_style("darkgrid")
    sns.lineplot(x=range(len(means)), y=means, color="#990F4B")

    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.show()
