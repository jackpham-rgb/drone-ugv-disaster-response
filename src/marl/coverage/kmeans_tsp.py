"""Cluster discovered survivors into one group per rescue agent, then route
each agent through its cluster with 2-opt local search on the travelling-
salesman tour. Ties back to the LP task-allocation and routing work from the
REU project this repo extends.

K-means is implemented here in plain NumPy (Lloyd's algorithm again, the same
structure as coverage/voronoi.py) rather than imported from scikit-learn,
because scikit-learn's compiled clustering extension is blocked by this
machine's Application Control policy. If that is not an issue in your
environment, `sklearn.cluster.KMeans` is a drop-in replacement.
"""
import numpy as np


def kmeans(points, k, n_init=5, iters=100, seed=0):
    """Minimal Lloyd's-algorithm K-means. Returns integer labels, one per
    point. Runs n_init random restarts and keeps the lowest-distortion one."""
    rng = np.random.default_rng(seed)
    best_labels, best_cost = None, np.inf
    for _ in range(n_init):
        centers = points[rng.choice(len(points), size=k, replace=False)].copy()
        labels = np.zeros(len(points), int)
        for _ in range(iters):
            d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            new_labels = d.argmin(1)
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = points[m].mean(0)
        cost = np.linalg.norm(points - centers[labels], axis=1).sum()
        if cost < best_cost:
            best_cost, best_labels = cost, labels
    return best_labels


def two_opt(route, D):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                a, b, c, d = route[i - 1], route[i], route[j - 1], route[j % len(route)]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d]:
                    route[i:j] = route[i:j][::-1]
                    improved = True
    return route


def cluster_and_route(points, k):
    labels = kmeans(points, k)
    routes = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        P = points[idx]
        D = np.linalg.norm(P[:, None] - P[None, :], axis=2)
        routes.append(idx[two_opt(list(range(len(P))), D)])
    return labels, routes
