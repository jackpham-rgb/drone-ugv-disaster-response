"""Plot IPPO learning curves across seeds with a 95% confidence band, the
figure a reviewer looks for first: proof of real learning, not a single
cherry-picked run.

    python -m scripts.make_plots
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(pattern):
    runs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    steps = np.array([s for s, _ in runs[0]])
    Y = np.array([[v for _, v in r] for r in runs])  # (seeds, T)
    return steps, Y


def plot(pattern, label):
    steps, Y = load(pattern)
    mean = Y.mean(0)
    ci = 1.96 * Y.std(0) / np.sqrt(len(Y))
    plt.plot(steps, mean, label=label)
    plt.fill_between(steps, mean - ci, mean + ci, alpha=0.2)


if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    plot("results/ippo_seed*.json", "IPPO")
    plt.xlabel("env steps")
    plt.ylabel("mean episode return")
    plt.title("DisasterEnv: IPPO learning curve (mean ± 95% CI, 3 seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150)
    print("saved results/learning_curve.png")
