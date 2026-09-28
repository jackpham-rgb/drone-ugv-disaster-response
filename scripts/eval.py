"""Evaluate a trained policy over held-out episodes: rescue rate, fairness
(Gini and Jain over per-agent rescue counts), and episode length, each with a
95% confidence interval over episodes.

    python -m scripts.eval
"""
import numpy as np
import torch
from torch.distributions import Categorical

from src.marl.algos.ippo import ActorCritic
from src.marl.envs.disaster_env import DisasterEnv
from src.marl.fairness import gini, jain


def evaluate(net, n_ep=50, seed=1000, verbose=True):
    env = DisasterEnv(seed=seed)
    R, G, J, T = [], [], [], []
    for e in range(n_ep):
        obs, _ = env.reset(seed=seed + e)
        steps = 0
        info = {}
        while env.agents:
            ag = list(env.agents)
            o = torch.tensor(np.stack([obs[a] for a in ag]), dtype=torch.float32)
            with torch.no_grad():
                logits, _ = net(o)
                act = Categorical(logits=logits).sample()
            obs, _, term, trunc, info = env.step({a: int(act[i]) for i, a in enumerate(ag)})
            steps += 1
        counts = list(env.rescue_count.values())
        rescued = info[list(info)[0]]["rescued"] if info else 0
        R.append(rescued / env.n_survivors)
        G.append(gini(counts))
        J.append(jain(counts))
        T.append(steps)

    def ci(x):
        x = np.array(x, float)
        return x.mean(), 1.96 * x.std() / np.sqrt(len(x))

    if verbose:
        for name, arr in (("rescue_rate", R), ("gini", G), ("jain", J), ("ep_len", T)):
            m, c = ci(arr)
            print(f"{name}: {m:.3f} +/- {c:.3f}")
    return dict(rescue=R, gini=G, jain=J, ep_len=T)


if __name__ == "__main__":
    env = DisasterEnv()
    net = ActorCritic(env.observation_space("drone_0").shape[0], 5)
    net.load_state_dict(torch.load("results/ippo.pt"))
    evaluate(net)
