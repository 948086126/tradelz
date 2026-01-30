import numpy as np
from stable_baselines3 import PPO
from envs.trading_env import TradingEnv
import pandas as pd

df = pd.read_csv("data/gongye_fulian_features.csv")
env = TradingEnv(df=df, use_action_shield=True, train_random_start_pos=False)
obs, _ = env.reset()

model = PPO.load("core/models/ppo_601138_obsS_shieldON.zip")  # 按你实际路径改
acts = []
for _ in range(100):
    a, _ = model.predict(obs, deterministic=False)
    acts.append(float(np.array(a).reshape(-1)[0]))
print("action mean/std/min/max:", np.mean(acts), np.std(acts), np.min(acts), np.max(acts))
