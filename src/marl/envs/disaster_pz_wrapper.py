"""Registers DisasterEnv (this project's own PettingZoo ParallelEnv, in
../../../src/marl/envs/disaster_env.py, three levels up from this epymarl
checkout) as a gym environment EPyMARL's `gymma` wrapper can drive.

This copy is the source of truth and is not imported from here. Copy it into
a separate `epymarl` checkout's `src/envs/` folder before use; see
docs/research_elevation.md for the exact steps and the small edit that
`epymarl/src/envs/__init__.py` also needs.

This mirrors envs/pz_wrapper.py's PettingZooWrapper, which only auto-
registers environments bundled inside the `pettingzoo` package itself.
DisasterEnv lives outside that package, so it needs its own thin gym.Env
adapter and its own gym.register call instead of relying on that
auto-discovery.

Registers as "disaster-v0". Run with, for example:

    python src/main.py --config=qmix --env-config=gymma \
        with env_args.key="disaster-v0" env_args.time_limit=200 seed=0
"""
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Tuple

# Add the repo root (three levels up: envs/ -> src/ -> epymarl/ -> repo root)
# so `from src.marl...` below resolves to THIS project's src/, not epymarl's
# own src/ (which is already on sys.path as the script directory and holds
# unrelated top-level modules like envs, learners, components).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.marl.envs.disaster_env import DisasterEnv  # noqa: E402


class DisasterPZWrapper(gym.Env):
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(self, **kwargs):
        self._env = DisasterEnv(**kwargs)
        self._env.reset()
        self.n_agents = len(self._env.possible_agents)
        self.last_obs = None
        self.action_space = Tuple(
            tuple(self._env.action_space(a) for a in self._env.possible_agents)
        )
        self.observation_space = Tuple(
            tuple(self._env.observation_space(a) for a in self._env.possible_agents)
        )

    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        obs = tuple(obs[a] for a in self._env.possible_agents)
        self.last_obs = obs
        return obs, info

    def step(self, actions):
        dict_actions = {a: int(act) for a, act in zip(self._env.possible_agents, actions)}
        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)
        agents = self._env.possible_agents
        if observations:
            obs = tuple(observations[a] for a in agents)
            rewards = [rewards[a] for a in agents]
            self.last_obs = obs
        else:
            # DisasterEnv empties `agents` on the terminal step, matching the
            # PettingZoo convention pz_wrapper.py also handles this way.
            obs = self.last_obs
            rewards = [0.0] * self.n_agents
        done = all(dones.values()) if dones else True
        trunc = all(truncated.values()) if truncated else True
        info = {f"{a}_{k}": v for a, d in infos.items() for k, v in d.items()}
        return obs, rewards, done, trunc, info

    def close(self):
        pass


gym.register("disaster-v0", entry_point="envs.disaster_pz_wrapper:DisasterPZWrapper")
