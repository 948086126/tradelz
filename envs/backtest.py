"""
日频回测模块（里程碑 A）
规则：
- 使用 PPO 输出的 target_position ∈ [0, 1]
- T 日收盘得到 target_position
- T+1 日开盘一次性成交到目标仓位
- 单股票、全资金、无滑点（可选手续费）

输入：
- features.csv（包含 date, open, close, ret_1 等）
- positions.csv（date, target_position）  # 由 PPO 推理得到

输出：
- equity.csv（每日资金曲线）
- metrics.json（简单回测指标）
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# =====================
# 参数区
# =====================
INITIAL_CASH = 1_000_000  # 初始资金
FEE_RATE = 0.001          # 单边手续费（0.1%），可先设为 0

# =====================
# 核心回测逻辑
# =====================

def run_backtest(features_csv: str, positions_csv: str, output_dir: str):
    features = pd.read_csv(features_csv, parse_dates=["date"])
    positions = pd.read_csv(positions_csv, parse_dates=["date"])

    features = features.sort_values("date").reset_index(drop=True)
    positions = positions.sort_values("date").reset_index(drop=True)

    # 对齐：positions 的 date 代表 T 日收盘后的决策
    df = pd.merge(features, positions, on="date", how="inner")

    # target_position 用于下一交易日
    df["target_position"] = df["target_position"].clip(0, 1)
    df["next_target_position"] = df["target_position"].shift(1)

    df = df.dropna().reset_index(drop=True)

    cash = INITIAL_CASH
    position = 0.0  # 持有股票市值
    equity_curve = []

    prev_target = 0.0

    for i, row in df.iterrows():
        open_price = row["open"]
        close_price = row["close"]
        target = row["next_target_position"]

        total_equity = cash + position

        # === 开盘调仓 ===
        desired_position_value = total_equity * target
        delta = desired_position_value - position

        if abs(delta) > 1e-6:
            fee = abs(delta) * FEE_RATE
            cash -= fee
            position += delta
            cash -= delta

        # === 收盘市值更新 ===
        if open_price > 0:
            position = position * (close_price / open_price)

        total_equity = cash + position

        equity_curve.append({
            "date": row["date"],
            "cash": cash,
            "position": position,
            "equity": total_equity,
            "target_position": target,
        })

        prev_target = target

    equity_df = pd.DataFrame(equity_curve)
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0)

    # =====================
    # 指标计算
    # =====================
    total_return = equity_df["equity"].iloc[-1] / INITIAL_CASH - 1

    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    sharpe = (
        equity_df["daily_return"].mean()
        / equity_df["daily_return"].std()
        * np.sqrt(252)
        if equity_df["daily_return"].std() > 0
        else 0.0
    )

    metrics = {
        "initial_cash": INITIAL_CASH,
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
    }

    # =====================
    # 输出
    # =====================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_df.to_csv(output_dir / "equity.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Backtest finished.")
    print(metrics)


# =====================
# CLI
# =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", default="./backtest_output")

    args = parser.parse_args()

    run_backtest(
        features_csv=args.features,
        positions_csv=args.positions,
        output_dir=args.out,
    )
