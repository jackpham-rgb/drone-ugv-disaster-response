# Fair Regional Coverage in Heterogeneous Multi-Tier MARL

## Cooperative Drone-UGV Disaster Response

> **This is a follow-up to my 2024 NSF REU research.** The original work used
> Linear Programming and FE-MADDPG on a 2D grid. This project takes a
> different approach: a two-tier system where aerial drones scout terrain and
> hand off to ground robots that perform the rescue, with fire trucks added
> for suppression. Built as a personal extension after the REU.

**[Interactive 3D sim](https://jackpham-rgb.github.io/drone-ugv-disaster-response/web/sim3d_v3.html)**
· **[Research portal](https://jackpham-rgb.github.io/drone-ugv-disaster-response/web/index.html)**
· **[Research notebook](notebooks/drone_ugv_notebook.ipynb)**
· **[Research elevation roadmap](docs/research_elevation.md)**

## Where the research stands

The browser sim's "MAPPO" and "QMIX" labels describe the two-tier
architecture and the fairness idea below; they are not yet backed by a
trained deep RL policy in this repo. I am closing that gap now: moving from
the visualization into an actual PyTorch training pipeline, benchmarked with
confidence intervals against standard baselines, plus a novel regional-
coverage contribution. The plan, current status and how to reproduce each
result are in [docs/research_elevation.md](docs/research_elevation.md).

First concrete milestone, done: a real PettingZoo training environment
(`src/marl/envs/disaster_env.py`) and Independent PPO
(`src/marl/algos/ippo.py`) that trains a neural network from environment
reward and produces a rising learning curve, not a fixed heuristic score.

```bash
pip install -r requirements.txt
python -m scripts.sanity_random   # the env runs
python -m scripts.train_ippo      # trains 3 seeds
python -m scripts.make_plots      # results/learning_curve.png
```

## The core idea

The original REU paper had all robots doing everything. This version splits
responsibilities:

| Tier | Agent | Algorithm | What it does |
|---|---|---|---|
| Air | Drones (UAV) | MAPPO | Scouts terrain, discovers survivors, maps fire, hands off to ground |
| Ground | Rescue robots (UGV) | QMIX | Reads the drones' belief map, navigates to survivors, performs rescue |
| Ground | Fire trucks | Greedy | Suppresses fire spread in a radius around target cells |

The drones and ground robots never communicate directly. Instead, drones
write to a shared belief map, and ground robots are blind to undiscovered
areas and can only act on what the drones have found. This information
asymmetry is the interesting part: it turns coordination into an information-
flow problem, not just a control problem.

## What's different from the original approach

| | Original REU (2024) | This project |
|---|---|---|
| Architecture | Single-tier robots | Two-tier UAV + UGV + FireTruck |
| Drone policy | Not present | MAPPO (CTDE framework) |
| Ground policy | FE-MADDPG | QMIX with an embedded fairness term |
| Fairness reward | Policy-gradient level | Built into the QMIX Q-value directly |
| Environment | Static 2D hex grid | 3D isometric, dynamic fire spread |
| Fire model | None | Cellular automaton with wind |
| Terrain | Uniform | Procedural mountain or urban biome |

The fairness reward from the original paper is still here, just wired
differently now:

$$r_t^i = \frac{\varepsilon + \left|e_t^i / \bar{e}_t - 1\right|}{\bar{e}_t}$$

In the original, this drove the FE-MADDPG policy gradient. Here it is
subtracted directly from the QMIX per-agent Q-value, so rescue robots that
already have more completed rescues get penalized when bidding for new
tasks. `src/marl/fairness.py` reimplements this reward and the Gini and Jain
metrics used to score it.

## Try it

**[Launch the 3D simulation](https://jackpham-rgb.github.io/drone-ugv-disaster-response/web/sim3d_v3.html)**

Or clone and open `web/sim3d_v3.html` directly in any browser: no install
needed.

In the simulation you can:
- Switch between mountain wildfire and urban disaster terrain
- Swap drone policy between MAPPO, Boids+RL, and Random mid-run
- Swap UGV policy between QMIX, Greedy, and Random
- Configure the number of drones, ground robots, and fire trucks independently
- Set fire spread rate and seed
- Choose random spawn or station spawn
- Drag to rotate the view, scroll to zoom
- Step through frame by frame or run at 1 to 10x speed
- Get a full performance report with a Gini fairness score at the end

Victim colors: gold/yellow flashing = undiscovered, green = rescued, maroon =
lost to fire.

## Jupyter notebook

`notebooks/drone_ugv_notebook.ipynb` walks through everything:

1. MDP formulation: two coupled MDPs, state space size
2. Fairness reward derivation and visualization
3. Terrain generation from scratch (fBm noise, no dependencies)
4. MAPPO drone scoring function
5. QMIX Q-value with an embedded fairness term
6. Fire spread cellular automaton implementation
7. Full episode run and trajectory plots
8. Multi-episode algorithm comparison (6 configurations, 10 seeds)
9. A custom experiment panel

To run it, either open in Colab or run locally alongside `drone_ugv_sim.py`.
The first cell auto-downloads the sim module from this repo if it is not
present.

## Repository layout

```
src/marl/envs/         training environment (PettingZoo ParallelEnv)
src/marl/algos/         RL algorithms (IPPO now; QMIX/MAPPO via EPyMARL next)
src/marl/coverage/      Voronoi partitioning, coverage path planning, K-means+2-opt routing, CBF safety
src/marl/fairness.py    fairness reward and Gini/Jain metrics
scripts/                train, evaluate, plot
notebooks/              the original research notebook (terrain, fire model, 6-config comparison)
web/                    the interactive 3D browser demo, the research portal, and the visualizations page
docs/                   research elevation roadmap and status
drone_ugv_sim.py        the simulation module the notebook imports
```

## What I learned so far

Splitting agents by role turned coordination from a control problem into an
information-flow problem: ground robots act on a belief map built from what
drones have discovered, not on the true state, so the interesting behavior is
in how information gets shared and used, not just in how each agent moves.
Writing the fairness term directly into the QMIX Q-value, instead of as a
separate rule, also showed me that fairness can be built into what an agent
learns to value, not bolted on afterward.

## Honesty line

This is a personal research extension, not a published paper, and the
results are on my own environment, not a standardized competition benchmark.
The browser sim visualizes the two-tier architecture and fairness idea;
actual trained-policy results are tracked in
[docs/research_elevation.md](docs/research_elevation.md) as they land.

## Algorithms used

MAPPO: Yu et al. (2021). *The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games.* NeurIPS.

QMIX: Rashid et al. (2018). *QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent RL.* ICML.

Fairness reward: Liu et al. (2022). *A fairness-aware cooperation strategy
for multi-agent systems driven by DRL.* CCC. (Also from the original REU
paper below.)

Boids: Reynolds (1987). *Flocks, herds, and schools: A distributed behavioral
model.* SIGGRAPH.

## Related

The original REU paper this builds on:
[area-disaster-response](https://github.com/jackpham-rgb/area-disaster-response)

## Acknowledgments

Thanks to Dr. Adam Thorpe and Dr. Ufuk Topcu at UT Austin for the original
REU mentorship, and to TACC and the NSF for supporting the summer research
that started this.

NSF REU Site: CI Research for Social Change, Award #2150390
