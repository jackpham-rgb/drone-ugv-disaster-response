"""Centroidal Voronoi coverage (Lloyd's algorithm): each drone owns the grid
cells nearest to it and moves toward that cell's (importance-weighted)
centroid. This is the geometric baseline for "each agent owns a region",
minimizing the locational cost of Cortes, Martinez, Karatas and Bullo (2004);
see docs/research_elevation.md section C.6 for why moving to the centroid is
gradient descent on that cost, hence guaranteed to converge.
"""
import numpy as np


def lloyd_step(drones, grid, weight=None):
    """drones: (K, 2) positions. grid: side length G. weight: (G, G) importance,
    for example the discovered/undiscovered density. Returns the new (K, 2)
    positions, one Lloyd iteration closer to a centroidal Voronoi tessellation.
    """
    G = grid
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    pts = np.stack([ii.ravel(), jj.ravel()], 1).astype(float)  # (G*G, 2)
    d = np.linalg.norm(pts[:, None, :] - drones[None, :, :], axis=2)  # (G*G, K)
    owner = d.argmin(1)
    w = np.ones(G * G) if weight is None else weight.ravel()
    new = drones.astype(float).copy()
    for k in range(len(drones)):
        m = owner == k
        if m.sum() == 0:
            continue
        wk = w[m]
        new[k] = (pts[m] * wk[:, None]).sum(0) / (wk.sum() + 1e-9)
    return new


def voronoi_owner(drones, grid):
    """(G, G) array of which drone index owns each cell, for plotting or
    for handing regions to a coverage-path planner."""
    G = grid
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    pts = np.stack([ii.ravel(), jj.ravel()], 1).astype(float)
    d = np.linalg.norm(pts[:, None, :] - drones[None, :, :], axis=2)
    return d.argmin(1).reshape(G, G)

# RESEARCH QUESTION this backs: does LEARNED scouting (the IPPO/MAPPO policy
# in src/marl/algos/) beat, match, or complement this geometric coverage? A
# hybrid where Voronoi assigns regions and a learned policy acts within them
# is the third condition worth comparing.
