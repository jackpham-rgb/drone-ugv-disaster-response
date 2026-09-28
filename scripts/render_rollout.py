"""Render one rollout of a trained policy to a GIF, for the README teaser.
Legend: blue triangle = drone, red square = UGV; gold star = undiscovered
survivor, orange = discovered, green = rescued.

    python -m scripts.render_rollout
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.distributions import Categorical

from src.marl.envs.disaster_env import DisasterEnv


def make_gif(net, out="results/demo.gif", seed=7):
    env = DisasterEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    frames = []
    while env.agents:
        ag = list(env.agents)
        o = torch.tensor(np.stack([obs[a] for a in ag]), dtype=torch.float32)
        with torch.no_grad():
            logits, _ = net(o)
            act = Categorical(logits=logits).sample()
        frames.append((dict(env.pos), env.surv.copy(), env.rescued.copy(), env.discovered.copy()))
        obs, _, term, trunc, _ = env.step({a: int(act[i]) for i, a in enumerate(ag)})
    g = env.grid
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(k):
        ax.clear()
        ax.set_xlim(-.5, g - .5)
        ax.set_ylim(-.5, g - .5)
        ax.set_xticks([])
        ax.set_yticks([])
        pos, surv, resc, disc = frames[k]
        for i, (sr, sc) in enumerate(surv):
            if resc[i]:
                ax.scatter(sc, sr, c="green", marker="*", s=130)
            elif disc[i]:
                ax.scatter(sc, sr, c="orange", marker="*", s=130)
            else:
                ax.scatter(sc, sr, c="gold", marker="*", s=60, alpha=.5)
        for a, (r, c) in pos.items():
            ax.scatter(c, r, c=("tab:blue" if a.startswith("drone") else "tab:red"),
                       marker=("^" if a.startswith("drone") else "s"), s=100, edgecolors="k")
        ax.set_title(f"t={k}   rescued={int(resc.sum())}/{len(surv)}")

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=120)
    anim.save(out, writer="pillow", fps=8)
    print("saved", out)


if __name__ == "__main__":
    from src.marl.algos.ippo import ActorCritic
    env = DisasterEnv()
    net = ActorCritic(env.observation_space("drone_0").shape[0], 5)
    net.load_state_dict(torch.load("results/ippo.pt"))
    make_gif(net)
