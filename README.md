# Drone–UGV Cooperative Disaster Response
## A Two-Tier Multi-Agent Rescue System using MAPPO + QMIX

> **This is a follow-up to my 2024 NSF REU research.** The original work used Linear Programming + FE-MADDPG on a 2D grid. This repo takes a different approach — a two-tier system where aerial drones scout terrain and communicate with ground robots that perform the actual rescue, with fire trucks added for suppression. Built as a personal extension project after the REU.

---

## The core idea

The original paper had all robots doing everything. This version splits responsibilities:

| Tier | Agent | Algorithm | What it does |
|------|-------|-----------|--------------|
| Air | Drones (UAV) | **MAPPO** | Scouts terrain, discovers survivors, maps fire, hands off to ground |
| Ground | Rescue robots (UGV) | **QMIX** | Reads drone's belief map, navigates to survivors, performs rescue |
| Ground | Fire trucks | Greedy | Suppresses fire spread in radius around target cells |

The drones and ground robots never communicate directly. Instead, drones write to a **shared belief map** — ground robots are blind to undiscovered areas and can only act on what the drones have found. This information asymmetry is the interesting part.

---

## What's different from the original approach

| | Original REU (2024) | This project |
|---|---|---|
| Architecture | Single-tier robots | Two-tier UAV + UGV + FireTruck |
| Drone policy | Not present | MAPPO (CTDE framework) |
| Ground policy | FE-MADDPG | QMIX with embedded fairness term |
| Fairness reward | Policy gradient level | Built into QMIX Q-value directly |
| Environment | Static 2D hex grid | 3D isometric, dynamic fire spread |
| Fire model | None | Cellular automaton with wind |
| Terrain | Uniform | Procedural mountain or urban biome |

The fairness reward from the original paper is still here — it's just wired differently now:

$$r_t^i = \frac{\varepsilon + \left|e_t^i / \bar{e}_t - 1\right|}{\bar{e}_t}$$

In the original, this drove the FE-MADDPG policy gradient. Here it's subtracted directly from the QMIX per-agent Q-value, so rescue robots that already have more completed rescues get penalized when bidding for new tasks.

---

## Try it

**[Launch the 3D simulation](https://jackpham-rgb.github.io/drone-ugv-disaster-response/sim3d_v3.html)**

**[Research portal](https://jackpham-rgb.github.io/drone-ugv-disaster-response/index.html)**

Or clone and open `sim3d_v3.html` directly in any browser — no install needed.

In the simulation you can:
- Switch between **mountain wildfire** and **urban disaster** terrain
- Swap drone policy between MAPPO, Boids+RL, and Random mid-run
- Swap UGV policy between QMIX, Greedy, and Random
- Configure number of drones, ground robots, and fire trucks independently
- Set fire spread rate and seed
- Choose random spawn or station spawn
- Drag to rotate the view, scroll to zoom
- Step through frame by frame or run at 1–10× speed
- Get a full performance report with Gini fairness score at the end

**Victim colors:** gold/yellow flashing = undiscovered, green = rescued, maroon = lost to fire.

---

## Jupyter Notebook

The notebook `drone_ugv_notebook.ipynb` walks through everything:

1. MDP formulation — two coupled MDPs, state space size
2. Fairness reward derivation and visualization
3. Terrain generation from scratch (fBm noise, no dependencies)
4. MAPPO drone scoring function
5. QMIX Q-value with embedded fairness term
6. Fire spread cellular automaton implementation
7. Full episode run and trajectory plots
8. Multi-episode algorithm comparison (6 configurations, 10 seeds)
9. Custom experiment panel

To run it — either open in Colab or run locally alongside `drone_ugv_sim.py`. The first cell auto-downloads the sim module from this repo if it's not present.

---

## Files

| File | Description |
|------|-------------|
| `sim3d_v3.html` | 3D isometric simulation — open in any browser |
| `drone_ugv_notebook.ipynb` | Jupyter research notebook |
| `index.html` | Research portal landing page |
| `notebook.html` | HTML notebook viewer |
| `visualizations.html` | Interactive data visualizations |
| `paper/` | Original REU paper (Pham et al., 2024) |

---

## Algorithms used

**MAPPO** — Yu et al. (2021). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.* NeurIPS.

**QMIX** — Rashid et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL.* ICML.

**Fairness reward** — Liu et al. (2022). *A fairness-aware cooperation strategy for multi-agent systems driven by DRL.* CCC. *(Also from the original REU paper below.)*

**Boids** — Reynolds (1987). *Flocks, herds, and schools: A distributed behavioral model.* SIGGRAPH.

---

## Related

The original REU paper this builds on: **[area-disaster-response](https://github.com/jackpham-rgb/area-disaster-response)**

---

## Acknowledgments

Thanks to Dr. Adam Thorpe and Dr. Ufuk Topcu at UT Austin for the original REU mentorship, and to TACC and the NSF for supporting the summer research that started this.

NSF REU Site: CI Research for Social Change, Award #2150390
