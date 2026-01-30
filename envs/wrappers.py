# envs/wrappers.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ObsSelectWrapper(gym.ObservationWrapper):
    """
    从原始 obs 中选择部分维度给 PPO
    """
    def __init__(self, env: gym.Env, idxs):
        super().__init__(env)
        self.idxs = list(idxs)

        low = -np.inf * np.ones(len(self.idxs), dtype=np.float32)
        high = np.inf * np.ones(len(self.idxs), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[self.idxs].astype(np.float32)
