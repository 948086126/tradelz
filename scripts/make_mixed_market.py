import numpy as np
import pandas as pd

np.random.seed(42)

price = 10.0
prices = []

def apply_return(p, r):
    return p * (1 + r)

# -------- A. 上涨趋势（低波动）--------
for _ in range(200):
    ret = np.random.normal(loc=0.0015, scale=0.002)  # 均值 +0.15%
    ret = np.clip(ret, -0.03, 0.03)
    price = apply_return(price, ret)
    prices.append(price)

# -------- B. 高位震荡（波动放大）--------
for _ in range(150):
    ret = np.random.normal(loc=0.0, scale=0.01)
    ret = np.clip(ret, -0.05, 0.05)
    price = apply_return(price, ret)
    prices.append(price)

# -------- C. 突然暴跌（模拟黑天鹅）--------
for r in [-0.06, -0.08, -0.10]:
    price = apply_return(price, r)
    prices.append(price)

# -------- D. 连续恐慌下跌（接近跌停）--------
for _ in range(5):
    ret = np.random.uniform(-0.095, -0.085)
    price = apply_return(price, ret)
    prices.append(price)

# -------- E. 低位震荡（情绪修复）--------
for _ in range(200):
    ret = np.random.normal(loc=0.0005, scale=0.015)
    ret = np.clip(ret, -0.08, 0.08)
    price = apply_return(price, ret)
    prices.append(price)

# -------- F. V 型反转 --------
for _ in range(150):
    ret = np.random.normal(loc=0.002, scale=0.01)
    ret = np.clip(ret, -0.05, 0.05)
    price = apply_return(price, ret)
    prices.append(price)

df = pd.DataFrame({"close": prices})
df.to_csv("../data/mixed_market.csv", index=False)

print("saved mixed_market.csv, len =", len(df))
