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
        # invert so higher = better
        mean_val = -1 * (sum(vals) / len(vals))
        means.append(mean_val)

    sns.set_style("darkgrid")
    sns.lineplot(x=range(len(means)), y=means, color="#990F4B")

    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.show()