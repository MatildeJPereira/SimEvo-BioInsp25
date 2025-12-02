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
        means.append(sum(vals) / len(vals))   # same computation as before

    sns.set_style("darkgrid") # optional: just changes background style


    # Same line plot as plt.plot(means), but with seaborn and explicit color
    sns.lineplot(x=range(len(means)), y=means, color="#990F4B")

    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.show()