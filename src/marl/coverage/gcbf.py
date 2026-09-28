"""A classical control-barrier-function (CBF) quadratic program for pairwise
collision avoidance: the smallest edit to an agent's desired velocity that
keeps it at least r_safe from every neighbor. This is the correct, publishable
baby step; GCBF+ (Zhang et al., 2024/25) is the learned, graph-neural upgrade
that scales this idea to many agents. See docs/research_elevation.md section
C.9 for why the constraint below makes the safe set forward invariant.
"""
import numpy as np
import cvxpy as cp


def safe_velocity(u_nom, pos, others, r_safe=1.0, gamma=1.0):
    u = cp.Variable(2)
    cons = []
    for o in np.atleast_2d(others):
        p = pos - o
        h = float(p @ p - r_safe ** 2)  # barrier: h >= 0 is safe
        cons.append(2 * p @ u >= -gamma * h)  # CBF condition: h_dot >= -gamma h
    cp.Problem(cp.Minimize(cp.sum_squares(u - u_nom)), cons).solve()
    return u.value if u.value is not None else u_nom
