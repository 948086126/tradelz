import gym
import numpy as np
from gym.spaces import Discrete, Box


class DailyPositionEnv(gym.Env):
    def __init__(self, df, feature_cols):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols

        self.action_space = Discrete(5)
        self.positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(feature_cols),),
            dtype=np.float32,
        )

        self.reset()

    def reset(self):
        self.t = 0
        self.position = 0.0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return self.df.loc[self.t, self.feature_cols].values.astype(np.float32)

    def step(self, action):
        target_position = self.positions[action]

        ret_next = self.df.loc[self.t + 1, "ret_1"]

        reward = (
            target_position * ret_next
            - 0.001 * abs(target_position - self.position)
        )

        self.position = target_position
        self.t += 1

        if self.t >= len(self.df) - 2:
            self.done = True

        return self._get_obs(), reward, self.done, {
            "position": self.position,
            "ret_next": ret_next,
        }
