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

def plot_all_atom_stats(history):
    carb = [h[1] for h in history]
    oth = [h[2] for h in history]
    comp = [h[3] for h in history]

    plt.plot(carb, label="Avg Carbons")
    plt.plot(oth, label="Avg Other Atoms")
    plt.plot(comp, label="Avg complexity")

    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Atom Composition Evolution")
    plt.legend()
    plt.show()

