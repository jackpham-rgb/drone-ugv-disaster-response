"""Train IPPO on DisasterEnv across a few seeds and save a learning curve per
seed. This is the "it actually learns" milestone: a return that rises with
training steps, not a fixed heuristic score.

    python -m scripts.train_ippo
"""
import json
import os

from src.marl.algos.ippo import train
from src.marl.envs.disaster_env import DisasterEnv

os.makedirs("results", exist_ok=True)
for seed in (0, 1, 2):
    _, curve = train(lambda seed=seed: DisasterEnv(seed=seed), total_steps=200_000, seed=seed)
    json.dump(curve, open(f"results/ippo_seed{seed}.json", "w"))
print("done, plot with scripts.make_plots")
