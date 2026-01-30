# policies/ppo_policy.py
from typing import Any, Dict, Optional
import numpy as np
from stable_baselines3 import PPO


class PPOPolicy:
    def __init__(self, model_path: str, deterministic: bool = True):
        self.model = PPO.load(model_path)
        self.deterministic = bool(deterministic)

    def __call__(self, obs: np.ndarray, *, state: Optional[dict] = None, **kwargs) -> np.ndarray:
        # obs 必须与训练时 wrapper 输出的维度一致
        obs = np.asarray(obs, dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        # action 通常是 shape (1,) 的 np.ndarray
        return np.array(action, dtype=np.float32)
