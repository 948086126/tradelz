import pandas as pd
import numpy as np

IN_PATH = "../data/gongye_fulian_features.csv"
OUT_PATH = "../data/gongye_fulian_features_fixed.csv"

df = pd.read_csv(IN_PATH)
df = df.sort_values("date").reset_index(drop=True)

# 你原来的 drawdown 是负数（<=0），我们改成正的回撤幅度
# dd = max(0, -drawdown)
if "drawdown" in df.columns:
    df["dd"] = (-df["drawdown"]).clip(lower=0.0)
else:
    # 如果没有 drawdown，就用 equity 自己算
    eq = df["equity"].astype(float)
    rolling_max = eq.cummax()
    df["dd"] = ((rolling_max - eq) / (rolling_max + 1e-12)).clip(lower=0.0)

# dd_increase：回撤增加的那部分（>=0）
df["dd_increase"] = (df["dd"] - df["dd"].shift(1).fillna(df["dd"].iloc[0])).clip(lower=0.0)

# 清理一下 NaN / inf
for c in ["ret_1", "ret_5", "ret_20", "vol_20", "price_vs_ma20", "equity", "dd", "dd_increase"]:
    if c in df.columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print("Saved:", OUT_PATH)
print("Columns:", list(df.columns))
