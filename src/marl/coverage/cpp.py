"""Coverage path planning within one region: boustrophedon (back-and-forth)
sweep, the simplest complete-coverage pattern (Choset; surveyed in Galceran
and Carreras, 2013). Given a rectangular region this produces waypoints that
visit every cell exactly once.
"""


def boustrophedon(r0, c0, r1, c1, step=1):
    wps = []
    l2r = True
    for r in range(r0, r1 + 1, step):
        cols = range(c0, c1 + 1, step) if l2r else range(c1, c0 - 1, -step)
        for c in cols:
            wps.append((r, c))
        l2r = not l2r
    return wps
