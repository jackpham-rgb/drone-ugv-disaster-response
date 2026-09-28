"""Independent PPO with parameter sharing: one actor-critic network used by
every agent (a role bit in the observation lets it tell drones from UGVs
apart). This is the first REAL training in this repo: it learns a neural
network from environment reward, not a hand-scored heuristic, and produces a
learning curve. QMIX and MAPPO (the two-tier structure described in the
README) are the next step, run through EPyMARL for trusted baselines; this
file is the from-scratch, minimal-but-real starting point.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, h=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(),
                                   nn.Linear(h, h), nn.Tanh())
        self.pi = nn.Linear(h, n_act)
        self.v = nn.Linear(h, 1)

    def forward(self, x):
        hh = self.body(x)
        return self.pi(hh), self.v(hh).squeeze(-1)


def gae(rew, val, done, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation: trades bias (low lambda) against
    variance (high lambda) in the advantage estimate used by the PPO update."""
    adv = np.zeros_like(rew, float)
    last = 0.0
    for t in reversed(range(len(rew))):
        nt = 1.0 - done[t]
        nextv = val[t + 1] if t + 1 < len(val) else 0.0
        delta = rew[t] + gamma * nextv * nt - val[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
    return adv


def train(env_fn, total_steps=300_000, rollout=2048, epochs=4, mb=256,
          clip=0.2, lr=3e-4, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = env_fn(seed=seed)
    obs, _ = env.reset(seed=seed)
    a0 = env.possible_agents[0]
    odim = env.observation_space(a0).shape[0]
    nact = env.action_space(a0).n
    net = ActorCritic(odim, nact)
    opt = torch.optim.Adam(net.parameters(), lr)
    curve = []
    steps = 0
    while steps < total_steps:
        B = {k: [] for k in ("obs", "act", "logp", "rew", "val", "done")}
        ep_returns = []
        ep = 0.0
        while len(B["rew"]) < rollout:
            if not env.agents:
                ep_returns.append(ep)
                ep = 0.0
                obs, _ = env.reset()
            ag = list(env.agents)
            o = torch.tensor(np.stack([obs[a] for a in ag]), dtype=torch.float32)
            with torch.no_grad():
                logits, val = net(o)
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            nobs, rew, term, trunc, _ = env.step({a: int(act[i]) for i, a in enumerate(ag)})
            for i, a in enumerate(ag):
                B["obs"].append(obs[a])
                B["act"].append(int(act[i]))
                B["logp"].append(float(logp[i]))
                B["rew"].append(rew[a])
                B["val"].append(float(val[i]))
                B["done"].append(float(term[a] or trunc[a]))
                ep += rew[a]
            obs = nobs
            steps += len(ag)
        rewA = np.array(B["rew"])
        valA = np.array(B["val"])
        doneA = np.array(B["done"])
        adv = gae(rewA, valA, doneA)
        ret = adv + valA
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O = torch.tensor(np.array(B["obs"]), dtype=torch.float32)
        A = torch.tensor(B["act"])
        OLD = torch.tensor(B["logp"])
        ADV = torch.tensor(adv, dtype=torch.float32)
        RET = torch.tensor(ret, dtype=torch.float32)
        idx = np.arange(len(A))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                m = idx[s:s + mb]
                logits, v = net(O[m])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(A[m])
                ratio = torch.exp(lp - OLD[m])
                s1 = ratio * ADV[m]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[m]
                loss = -torch.min(s1, s2).mean() + 0.5 * ((v - RET[m]) ** 2).mean() \
                    - 0.01 * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        if ep_returns:
            m = float(np.mean(ep_returns))
            curve.append((steps, m))
            if verbose:
                print(f"steps={steps} mean_ep_return={m:.2f}")
    return net, curve
