"""Render a static top-down preview of a disaster-response scenario, for the
README's general-audience section: terrain, a fire, drones, ground robots and
survivors, with the Voronoi coverage regions from src/marl/coverage/voronoi.py
tinted in the background.

This is NOT a screenshot of the interactive 3D sim in web/sim3d_v3.html; it is
a separate, simpler 2D rendering meant to give a reader who will not click
through to the live sim a sense of what the scenario looks like from above.
The real thing has 3D terrain, fire spread over time, and is interactive; see
the link in the README.

    python -m scripts.render_map_preview
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.marl.coverage.voronoi import voronoi_owner

G = 28
rng = np.random.default_rng(3)


def fbm(size, octaves=5, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((size, size))
    amp, freq = 1.0, 2
    for _ in range(octaves):
        n = r.standard_normal((freq + 1, freq + 1))
        xs = np.linspace(0, freq, size)
        ys = np.linspace(0, freq, size)
        xi = xs.astype(int).clip(0, freq - 1)
        yi = ys.astype(int).clip(0, freq - 1)
        tx = (xs - xi)[None, :]
        ty = (ys - yi)[:, None]
        top = n[yi][:, xi] * (1 - tx) + n[yi][:, xi + 1] * tx
        bot = n[yi + 1][:, xi] * (1 - tx) + n[yi + 1][:, xi + 1] * tx
        out += amp * (top * (1 - ty) + bot * ty)
        amp *= 0.5
        freq *= 2
    return (out - out.min()) / (out.max() - out.min())


def main():
    terrain = fbm(G, seed=7)
    terrain_cmap = LinearSegmentedColormap.from_list(
        "terrain2", ["#2f5d3a", "#5c8a4a", "#9cb26a", "#c9b479", "#8a7256"]
    )

    drones = rng.integers(4, G - 4, size=(3, 2)).astype(float)
    ugvs = rng.integers(4, G - 4, size=(2, 2)).astype(float)
    trucks = rng.integers(4, G - 4, size=(1, 2)).astype(float)
    survivors = rng.integers(1, G - 1, size=(7, 2))
    rescued = np.array([True, True, False, False, False, False, False])
    fire_center = np.array([G * 0.7, G * 0.3])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(terrain, cmap=terrain_cmap, origin="lower", extent=(0, G, 0, G), zorder=0)

    owner = voronoi_owner(drones, G)
    ax.imshow(owner, cmap="tab10", alpha=0.16, origin="lower", extent=(0, G, 0, G), zorder=1)

    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    fire_d = np.sqrt((ii - fire_center[0]) ** 2 + (jj - fire_center[1]) ** 2)
    fire = np.clip(1.3 - fire_d / 4.0, 0, 1) ** 2
    ax.imshow(np.ma.masked_less(fire, 0.05), cmap="autumn_r", alpha=0.75,
              origin="lower", extent=(0, G, 0, G), zorder=2, vmin=0, vmax=1)

    for i, (sr, sc) in enumerate(survivors):
        if rescued[i]:
            ax.scatter(sc, sr, marker="*", s=260, c="#2ecc71", edgecolors="black", zorder=4)
        else:
            ax.scatter(sc, sr, marker="*", s=260, c="#f4d03f", edgecolors="black", zorder=4)
    ax.scatter(drones[:, 1], drones[:, 0], marker="^", s=220, c="#2e86ff", edgecolors="black",
               linewidths=1.2, zorder=5, label="Drone (UAV)")
    ax.scatter(ugvs[:, 1], ugvs[:, 0], marker="s", s=200, c="#e74c3c", edgecolors="black",
               linewidths=1.2, zorder=5, label="Ground robot (UGV)")
    ax.scatter(trucks[:, 1], trucks[:, 0], marker="D", s=200, c="#f39c12", edgecolors="black",
               linewidths=1.2, zorder=5, label="Fire truck")

    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sample scenario: drones scout, ground robots rescue, trucks fight fire",
                 fontsize=12)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.text(0.5, 0.01,
              "Static 2D preview. The real simulation is 3D, animated and interactive.",
              ha="center", fontsize=9, color="#444")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig("results/map_preview.png", dpi=150)
    print("saved results/map_preview.png")


if __name__ == "__main__":
    main()
