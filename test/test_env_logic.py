import numpy as np
import pandas as pd
from envs.trading_env import TradingEnv

# =========================================================
# 1️⃣ 读取混合行情数据
# =========================================================
df = pd.read_csv("../data/mixed_market.csv")

# =========================================================
# 2️⃣ 创建交易环境
# =========================================================
env = TradingEnv(
    df=df,
    window_size=3,
    transaction_cost=0.001,  # 稍微大一点，方便观察换手惩罚
)

# =========================================================
# 3️⃣ reset 环境
# =========================================================
obs, info = env.reset()
print("Initial obs:", obs)

# =========================================================
# 4️⃣ 随机动作测试（只验证环境行为是否合理）
# =========================================================
print("\n=== Random actions ===")

for i in range(50):
    # 随机给一个 [0,1] 的目标仓位
    action = np.random.uniform(0.0, 1.0)

    obs, reward, done, _, info = env.step([action])

    print(
        f"step={env.current_step:4d} | "
        f"ret={info['price_return']:+.4f} | "
        f"pos={info['position']:.2f} | "
        f"target={info['target_position']:.2f} | "
        f"eq={info['equity']:.3f} | "
        f"reward={reward:+.5f}"
    )

    if done:
        print("Episode finished.")
        break
