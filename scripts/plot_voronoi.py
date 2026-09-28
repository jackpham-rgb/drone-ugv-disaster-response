"""Plot the Voronoi coverage regions for a set of drone positions, showing
"each agent owns a region" directly, for the coverage-novelty figure.

    python -m scripts.plot_voronoi
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.marl.coverage.voronoi import voronoi_owner


def plot_voronoi(drones, grid=16, out="results/voronoi.png"):
    owner = voronoi_owner(drones, grid)
    plt.figure(figsize=(5, 5))
    plt.imshow(owner, origin="lower", cmap="tab10", alpha=.55)
    plt.scatter(drones[:, 1], drones[:, 0], c="k", marker="^", s=90)
    plt.title("Voronoi coverage regions (one per drone)")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    plot_voronoi(rng.integers(0, 16, size=(4, 2)).astype(float))
