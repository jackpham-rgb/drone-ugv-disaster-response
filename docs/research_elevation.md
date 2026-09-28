# Research elevation roadmap

This is the durable plan for moving this project from a browser simulation to
trained, benchmarked, published multi-agent RL research: real PyTorch
training, a standard PettingZoo environment, baselines with confidence
intervals, a coverage-and-safety novelty, and a short paper.

The roadmap, including the baby-step implementation guide with code and the
mathematical foundations for each component, lives in the author's private
planning notes. This file tracks what has actually landed in the repo.

## Status

| Phase | What it is | State |
|---|---|---|
| 0: repo hygiene | `.gitattributes`, HTML moved to `web/`, notebook to `notebooks/`, package skeleton | Done |
| 1: environment | `src/marl/envs/disaster_env.py`, a PettingZoo `ParallelEnv` | Done |
| 2: real training | `src/marl/algos/ippo.py` (Independent PPO, parameter sharing), `scripts/train_ippo.py` | Done: real gradient-based training, see below |
| 3: baselines + rigor | QMIX, MAPPO, IQL, VDN, MADDPG via EPyMARL; fairness ablation | Wrapper written (`src/marl/envs/epymarl_wrapper.py`), not yet run |
| 4: coverage novelty | Voronoi partitioning, boustrophedon CPP, K-means+2-opt routing, CBF-QP safety | Modules written (`src/marl/coverage/`), not yet integrated into training |
| 5: extra extension | Learned communication or GNN coordination, with an ablation | Not started |
| 6: sim-to-real validation | Gazebo or Isaac Sim demo of the trained policy | Not started |
| 7: generalization | Train on one map set, test on unseen; a second application domain | Not started |
| 8: paper | Short paper targeting a workshop, arXiv | Not started |

## First training result, honestly

`scripts/train_ippo.py` trains 3 seeds for 200,000 environment steps each on
the 4-agent, 8-survivor grid. The result (`results/learning_curve.png`) is a
real but noisy improvement: the mean return across seeds rises from around
-3 early in training to around -2 by the end, with a wide confidence band and
individual seeds that do not all improve monotonically (seed 0 goes from
-3.5 to -0.13, seed 1 from -4.25 to -1.77, seed 2 from -2.5 to -3.97). This is
the expected shape for a single shared network with no reward shaping beyond
the raw scout/rescue/time-penalty signal, not a tuned result. It is enough to
say training is real, not enough to claim the policy is good. Next steps for
a stronger curve: separate networks per role, reward shaping or curriculum,
and more seeds with variance reduction, before moving to QMIX/MAPPO via
EPyMARL.

## What is real here

`DisasterEnv` is deliberately small and fast, a 16x16 grid rather than the
full fBm terrain and fire cellular automaton in
`notebooks/drone_ugv_notebook.ipynb`. Deep RL training needs millions of
environment steps, so the fast grid is what gets trained on; porting the
richer terrain and fire model into this env, or building it as
`coverage_env.py`, is future work. The browser 3D simulation in `web/` is a
visualization layer, not the training environment, and is being kept as an
outreach demo, clearly separated from the research pipeline.

## Reproduce the current milestone

```bash
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.sanity_random   # confirms the env runs
python -m scripts.train_ippo      # trains 3 seeds, ~10-20 min on CPU
python -m scripts.make_plots      # results/learning_curve.png
python -m scripts.render_rollout  # results/demo.gif, needs results/ippo.pt saved separately
```
