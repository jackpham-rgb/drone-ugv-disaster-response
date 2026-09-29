"""Plot EPyMARL QMIX/MAPPO training curves (test_return_mean, the greedy
evaluation return) from their Sacred metrics.json files, alongside this
project's own IPPO curve, on one axis for comparison.

    python -m scripts.plot_epymarl_baselines

Reads epymarl/results/sacred/<alg>/<env>/<run_id>/metrics.json for each
(alg, run_id) in RUNS below, and results/ippo_seed*.json for IPPO. Writes
results/baseline_comparison.png and results/baseline_summary.json.

Scale note: EPyMARL's gymma wrapper runs this env with common_reward=True
(the default), which SUMS the reward across all N_AGENTS agents into one
team return each step. This project's own IPPO script reports the MEAN
episode return per agent instead. The two are not on the same scale, so this
script divides the QMIX/MAPPO curves by N_AGENTS before plotting, so all
three lines read as "mean return per agent" and can be compared directly.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (algorithm label, sacred run directory) pairs actually run so far.
RUNS = {
    "QMIX": ["epymarl/results/sacred/qmix/disaster-v0/1"],
    "MAPPO": ["epymarl/results/sacred/mappo/disaster-v0/1"],
}
METRIC = "test_return_mean"
N_AGENTS = 4  # DisasterEnv() defaults: 2 drones + 2 UGVs, used on both sides


def load_epymarl(run_dirs, metric=METRIC):
    curves = []
    for d in run_dirs:
        m = json.load(open(f"{d}/metrics.json"))
        if metric not in m:
            continue
        curves.append((np.array(m[metric]["steps"]), np.array(m[metric]["values"])))
    return curves


def load_ippo():
    runs = [json.load(open(f)) for f in sorted(glob.glob("results/ippo_seed*.json"))]
    steps = np.array([s for s, _ in runs[0]])
    Y = np.array([[v for _, v in r] for r in runs])
    return steps, Y


def plot_curve(steps, values, label):
    plt.plot(steps, values, label=label, marker="o", markersize=3)


if __name__ == "__main__":
    plt.figure(figsize=(7, 4.5))
    summary = {}
    for alg, dirs in RUNS.items():
        curves = load_epymarl(dirs)
        if not curves:
            continue
        steps, values = curves[0]
        per_agent = values / N_AGENTS
        plot_curve(steps, per_agent, f"{alg} (1 seed)")
        summary[alg] = {
            "final_team_return": float(values[-1]),
            "final_per_agent": float(per_agent[-1]),
            "n_points": int(len(values)),
        }

    ippo_steps, ippo_Y = load_ippo()
    mean = ippo_Y.mean(0)
    ci = 1.96 * ippo_Y.std(0) / np.sqrt(len(ippo_Y))
    plt.plot(ippo_steps, mean, label="IPPO, from scratch (mean of 3 seeds)")
    plt.fill_between(ippo_steps, mean - ci, mean + ci, alpha=0.15)
    summary["IPPO"] = {"final_per_agent_mean": float(mean[-1]), "n_seeds": int(len(ippo_Y))}

    plt.xlabel("env steps")
    plt.ylabel("mean return per agent")
    plt.title("DisasterEnv: QMIX / MAPPO (EPyMARL) vs IPPO (from scratch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/baseline_comparison.png", dpi=150)
    json.dump(summary, open("results/baseline_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("saved results/baseline_comparison.png")
