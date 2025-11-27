# Plots with matplotlib or seaborn:
# - mean/median fitness per generation
# - diversity scores
# - archive growth

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
