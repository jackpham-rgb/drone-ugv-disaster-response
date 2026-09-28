"""Fairness metrics and reward term, reused from the REU work this project
extends. Gini and Jain measure how evenly rescues (or any effort) are spread
across agents; fairness_penalty is the term that can be subtracted from an
agent's Q-value so agents with more completed work are penalized when
bidding for new tasks (see docs/research_elevation.md, section C.5, for the
alpha-fairness reading of this).
"""
import numpy as np


def gini(counts):
    x = np.sort(np.asarray(counts, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def jain(counts):
    x = np.asarray(counts, float)
    return 1.0 if x.sum() == 0 else (x.sum() ** 2) / (len(x) * np.sum(x ** 2))


def fairness_penalty(effort_i, mean_effort, eps=1e-3):
    return (eps + abs(effort_i / max(mean_effort, eps) - 1.0)) / max(mean_effort, eps)
