import pandas as pd

df = pd.read_csv("../logs/behavior_baseline_ddcontrol.csv")
print("pos min/max:", df["position"].min(), df["position"].max())
print("trade nonzero ratio:", (df["executed_trade"].abs() > 1e-9).mean())
if "price_vs_ma20" in df.columns:
    print("price_vs_ma20 p10/p50/p90:",
          df["price_vs_ma20"].quantile(0.1),
          df["price_vs_ma20"].quantile(0.5),
          df["price_vs_ma20"].quantile(0.9))
