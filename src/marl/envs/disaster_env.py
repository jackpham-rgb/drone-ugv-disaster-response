"""A minimal, real multi-agent training environment for the disaster-response
problem: drones scout a grid and reveal survivors into a shared belief map,
ground robots (UGVs) rescue the survivors that have been revealed.

This is a PettingZoo ParallelEnv (the standard multi-agent Gym API), so it
plugs into PyTorch training loops directly and, later, into EPyMARL for
QMIX/MAPPO baselines. It is deliberately small and fast: training needs
millions of environment steps, so this stays a lightweight grid rather than
the full 3D terrain and fire model in notebooks/drone_ugv_notebook.ipynb.
Porting that terrain and fire cellular automaton into this env is future
work (see docs/research_elevation.md).
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

_MOVES = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # stay, up, down, left, right


class DisasterEnv(ParallelEnv):
    """Drones (scout) and UGVs (rescue) share a grid and a belief map.

    Reward: a drone gets +0.5 for discovering a survivor it hasn't seen before;
    a UGV gets +1.0 for rescuing a discovered survivor. Every agent pays a
    small per-step time penalty, so idling is never optimal.
    """

    metadata = {"name": "disaster_v0"}

    def __init__(self, grid=16, n_drones=2, n_ugv=2, n_survivors=8,
                 max_steps=200, view=2, seed=None):
        self.grid, self.view, self.max_steps = grid, view, max_steps
        self.n_drones, self.n_ugv, self.n_survivors = n_drones, n_ugv, n_survivors
        self._rng = np.random.default_rng(seed)
        self.possible_agents = ([f"drone_{i}" for i in range(n_drones)]
                                 + [f"ugv_{i}" for i in range(n_ugv)])

    def _is_drone(self, a):
        return a.startswith("drone")

    def observation_space(self, agent):
        w = 2 * self.view + 1
        return spaces.Box(0.0, 1.0, shape=(3 + w * w,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        g = self.grid
        self.agents = list(self.possible_agents)
        self.t = 0
        self.pos = {a: self._rng.integers(0, g, size=2) for a in self.agents}
        self.surv = self._rng.integers(0, g, size=(self.n_survivors, 2))
        self.discovered = np.zeros(self.n_survivors, bool)
        self.rescued = np.zeros(self.n_survivors, bool)
        self.belief = np.zeros((g, g), np.float32)
        self.rescue_count = {a: 0 for a in self.agents if not self._is_drone(a)}
        return {a: self._obs(a) for a in self.agents}, {a: {} for a in self.agents}

    def _obs(self, a):
        g, v = self.grid, self.view
        r, c = self.pos[a]
        win = np.zeros((2 * v + 1, 2 * v + 1), np.float32)
        for dr in range(-v, v + 1):
            for dc in range(-v, v + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < g and 0 <= cc < g:
                    win[dr + v, dc + v] = self.belief[rr, cc]
        role = 1.0 if self._is_drone(a) else 0.0
        return np.concatenate([[r / g, c / g, role], win.ravel()]).astype(np.float32)

    def step(self, actions):
        g = self.grid
        rew = {a: 0.0 for a in self.agents}
        for a, act in actions.items():
            dr, dc = _MOVES[int(act)]
            r, c = self.pos[a]
            self.pos[a] = np.array([np.clip(r + dr, 0, g - 1), np.clip(c + dc, 0, g - 1)])
        for a in self.agents:  # drones scout
            if not self._is_drone(a):
                continue
            r, c = self.pos[a]
            for i in range(self.n_survivors):
                if self.rescued[i]:
                    continue
                sr, sc = self.surv[i]
                if abs(sr - r) <= self.view and abs(sc - c) <= self.view and not self.discovered[i]:
                    self.discovered[i] = True
                    self.belief[sr, sc] = 1.0
                    rew[a] += 0.5
        for a in self.agents:  # ugvs rescue
            if self._is_drone(a):
                continue
            r, c = self.pos[a]
            for i in range(self.n_survivors):
                if self.rescued[i] or not self.discovered[i]:
                    continue
                if self.surv[i][0] == r and self.surv[i][1] == c:
                    self.rescued[i] = True
                    self.belief[r, c] = 0.0
                    self.rescue_count[a] += 1
                    rew[a] += 1.0
        self.t += 1
        for a in self.agents:
            rew[a] -= 0.01  # time penalty
        done = bool(self.rescued.all()) or self.t >= self.max_steps
        term = {a: done for a in self.agents}
        trunc = {a: self.t >= self.max_steps for a in self.agents}
        info = {a: {"rescued": int(self.rescued.sum())} for a in self.agents}
        obs = {a: self._obs(a) for a in self.agents}
        if done:
            self.agents = []
        return obs, rew, term, trunc, info
