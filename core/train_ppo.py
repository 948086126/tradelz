from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


ACTION_TO_POS = {
    0: 0.0,
    1: 0.3,
    2: 0.6,
    3: 1.0,
}


@dataclass
class TrainConfig:
    symbol: str = "601138"
    fee: float = 0.0015
    seed: int = 42
    total_timesteps: int = 300_000


class DailyTradingEnv(gym.Env):
    """
    日频：t日观察特征 -> 选择仓位 p_t -> reward 用 next_ret_1 + 换手惩罚
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], fee: float = 0.0015):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.fee = fee

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32
        )

        self._i = 0
        self._prev_pos = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self._prev_pos = 0.0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.loc[self._i, self.feature_cols]
        return row.to_numpy(dtype=np.float32)

    def step(self, action: int):
        pos = float(ACTION_TO_POS[int(action)])
        row = self.df.loc[self._i]

        next_ret_1 = float(row["next_ret_1"])
        turnover_cost = self.fee * abs(pos - self._prev_pos)

        reward = pos * next_ret_1 - turnover_cost

        self._prev_pos = pos
        self._i += 1

        terminated = self._i >= (len(self.df) - 1)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros((len(self.feature_cols),), dtype=np.float32)

        info = {
            "pos": pos,
            "next_ret_1": next_ret_1,
            "turnover_cost": turnover_cost,
        }
        return obs, float(reward), terminated, truncated, info


def load_processed(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "601138_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed features: {path}. Run script/02_build_features.py first.")
    return pd.read_csv(path, parse_dates=["date"])


def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "open", "high", "low", "close", "volume",
               "next_ret_1", "next_ret_5", "ret_oc_1", "split", "day_idx", "tr"}
    cols = [c for c in df.columns if c not in exclude]
    if len(cols) < 8:
        raise ValueError(f"Too few features ({len(cols)}). Feature cols: {cols}")
    return cols


def main():
    cfg = TrainConfig()
    root = Path(__file__).resolve().parents[1]  # tradeLz/

    df = load_processed(root)
    df_train = df[df["split"] == "train"].copy()
    df_valid = df[df["split"] == "valid"].copy()

    feature_cols = pick_feature_cols(df_train)

    env = DummyVecEnv([lambda: DailyTradingEnv(df_train, feature_cols, fee=cfg.fee)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=cfg.seed,
        verbose=1,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
    )

    model.learn(total_timesteps=cfg.total_timesteps)

    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_601138.zip"
    model.save(str(model_path))

    meta = {
        "symbol": cfg.symbol,
        "fee": cfg.fee,
        "feature_cols": feature_cols,
        "action_to_pos": ACTION_TO_POS,
        "train_rows": int(len(df_train)),
        "valid_rows": int(len(df_valid)),
        "total_timesteps": int(cfg.total_timesteps),
    }
    meta_path = out_dir / "ppo_601138_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved model -> {model_path}")
    print(f"[OK] Saved meta  -> {meta_path}")


if __name__ == "__main__":
    main()
