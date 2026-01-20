# Statistical tests: AUC per run + Wilcoxon Rank-Sum 
import numpy as np
from scipy.stats import mannwhitneyu

def mean_fitness_curve(history):
    """
    Returns array of mean fitness scores (higher = better)
    for each generation in one GA run.
    """
    curve = []
    for pop in history:
        penalties = list(pop.fitness.values())
        mean_penalty = sum(penalties) / len(penalties)
        curve.append(-mean_penalty)  # convert penalty → score
    return np.array(curve)


def auc_from_curve(curve):
    """
    Computes AUC using trapezoidal rule.
    x = generation index (0, 1, 2, ...)
    """
    generations = np.arange(len(curve))
    return np.trapezoid(curve, generations)


def wilcoxon_rank_sum(aucs_a, aucs_b, alternative="two-sided"):
    """
    Wilcoxon rank-sum test (Mann–Whitney U).
    
    Parameters
    ----------
    aucs_a, aucs_b : list or array-like
        AUC values from independent GA runs.
    alternative : {"two-sided", "less", "greater"}
        Hypothesis type.

    Returns
    -------
    p_value : float
        P-value of the test.
    """
    _, p_value = mannwhitneyu(
        aucs_a,
        aucs_b,
        alternative=alternative
    )
    return p_value

def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size.
    
    Returns
    -------
    delta : float
        Range [-1, 1]
        0   → no effect
        ±1  → complete separation
    """
    nx = len(x)
    ny = len(y)

    greater = 0
    less = 0

    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1

    delta = (greater - less) / (nx * ny)
    return delta