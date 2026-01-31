from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ACTION_TO_POS = {0: 0.0, 1: 0.3, 2: 0.6, 3: 1.0}


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def main():
    root = Path(__file__).resolve().parents[1]  # tradeLz/
    data_path = root / "data" / "processed" / "601138_features.csv"
    meta_path = root / "models" / "ppo_601138_meta.json"
    model_path = root / "models" / "ppo_601138.zip"

    if not (data_path.exists() and meta_path.exists() and model_path.exists()):
        raise FileNotFoundError("Missing data/model. Ensure features + model are created first.")

    df = pd.read_csv(data_path, parse_dates=["date"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols: List[str] = meta["feature_cols"]

    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    if len(df_test) < 50:
        raise ValueError(f"Test rows too few: {len(df_test)}")

    model = PPO.load(str(model_path))

    fee = float(meta.get("fee", 0.0015))

    equity = 1.0
    pos_prev = 0.0

    rows = []
    trades = 0
    turnover_sum = 0.0

    # 重要：t日收盘做决策，t+1日开盘成交并持有到收盘
    # 因为我们的 df_test 每行都有当日 ret_oc_1（open->close），所以用它更新资金曲线
    for i in range(len(df_test)):
        obs = df_test.loc[i, feature_cols].to_numpy(dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        pos_target = float(ACTION_TO_POS[int(action)])

        delta = pos_target - pos_prev
        cost = fee * abs(delta)
        if abs(delta) > 1e-9:
            trades += 1
            turnover_sum += abs(delta)

        # 当日开盘成交 -> 持有到收盘
        day_ret = float(df_test.loc[i, "ret_oc_1"])
        equity = equity * (1.0 + pos_target * day_ret - cost)

        rows.append({
            "date": df_test.loc[i, "date"],
            "equity": equity,
            "position": pos_target,
            "day_ret_oc": day_ret,
            "cost": cost,
            "delta_pos": delta,
            "action": int(action),
        })

        pos_prev = pos_target

    out_dir = root / "backtest" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame(rows)
    equity_path = out_dir / "equity_test.csv"
    equity_df.to_csv(equity_path, index=False, encoding="utf-8-sig")

    total_return = float(equity_df["equity"].iloc[-1] - 1.0)
    mdd = max_drawdown(equity_df["equity"])
    metrics = {
        "total_return": total_return,
        "max_drawdown": mdd,
        "trades": int(trades),
        "turnover_sum": float(turnover_sum),
        "rows": int(len(equity_df)),
    }
    metrics_path = out_dir / "metrics_test.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved equity -> {equity_path}")
    print(f"[OK] Saved metrics -> {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
