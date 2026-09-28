"""Prove the training env runs, with random actions. No learning happens here.

    python -m scripts.sanity_random
"""
from src.marl.envs.disaster_env import DisasterEnv

env = DisasterEnv(seed=0)
obs, info = env.reset(seed=0)
total = {a: 0.0 for a in env.possible_agents}
last = None
while env.agents:
    acts = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(acts)
    for a, r in rew.items():
        total[a] += r
    last = info
print("returns:", {k: round(v, 2) for k, v in total.items()})
print(last)
