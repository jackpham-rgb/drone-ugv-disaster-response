"""Adapter that exposes DisasterEnv through EPyMARL's SMAC-like MultiAgentEnv
API, so QMIX and MAPPO (and the IPPO/IQL/VDN/MADDPG baselines) can be run
through EPyMARL for a trusted, third-party-implemented benchmark instead of
from-scratch reimplementations. Not wired into a training run yet; see
docs/research_elevation.md STEP 8 for how to register it.
"""
import numpy as np

from .disaster_env import DisasterEnv


class DisasterMAEnv:
    def __init__(self, **kwargs):
        self.env = DisasterEnv(**kwargs)
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        self.n_actions = self.env.action_space(self.agents[0]).n
        self.episode_limit = self.env.max_steps
        self._obs = None

    def reset(self):
        self._obs, _ = self.env.reset()
        return self.get_obs(), self.get_state()

    def step(self, actions):
        acts = {a: int(actions[i]) for i, a in enumerate(self.agents)}
        self._obs, rew, term, trunc, info = self.env.step(acts)
        reward = float(np.mean(list(rew.values())))  # shared team reward
        done = (all(term.values()) or all(trunc.values())) if term else True
        return reward, done, info

    def get_obs(self):
        z = np.zeros(self.get_obs_size(), np.float32)
        return [self._obs.get(a, z) for a in self.agents]

    def get_obs_agent(self, i):
        return self.get_obs()[i]

    def get_obs_size(self):
        return self.env.observation_space(self.agents[0]).shape[0]

    def get_state(self):
        return np.concatenate(self.get_obs()).astype(np.float32)

    def get_state_size(self):
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [[1] * self.n_actions for _ in self.agents]

    def get_avail_agent_actions(self, i):
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def close(self):
        pass

# To register in EPyMARL, in epymarl/src/envs/__init__.py add:
#   from marl.envs.epymarl_wrapper import DisasterMAEnv
#   REGISTRY["disaster"] = partial(env_fn, env=DisasterMAEnv)
# then run:  python src/main.py --config=qmix --env-config=disaster with seed=0
