import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_col(df: pd.DataFrame, candidates: list[str]):
    """Helper function to find the first column from candidates that exists in the DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def analyze(csv_path: str, out_dir: str = "logs"):
    # 读取行为数据
    df = pd.read_csv(csv_path)

    # 确定行为日志列
    col_pos = _safe_col(df, ["position", "pos"])
    col_trade = _safe_col(df, ["executed_trade", "trade", "trade_amount"])
    col_eq = _safe_col(df, ["equity", "nav"])
    col_dd = _safe_col(df, ["drawdown", "dd"])
    col_ddi = _safe_col(df, ["dd_increase", "ddi"])

    if col_pos is None or col_eq is None or col_dd is None:
        raise ValueError(f"缺少关键列。需要至少包含 position/equity/drawdown。当前列：{list(df.columns)}")

    # 处理列数据
    pos = df[col_pos].astype(float).to_numpy()
    eq = df[col_eq].astype(float).to_numpy()
    dd = df[col_dd].astype(float).to_numpy()

    trade = None
    if col_trade is not None:
        trade = df[col_trade].astype(float).to_numpy()

    ddi = None
    if col_ddi is not None:
        ddi = df[col_ddi].astype(float).to_numpy()

    # ---- 指标 ----
    avg_pos = float(np.mean(pos))
    p_pos = np.percentile(pos, [0, 10, 25, 50, 75, 90, 100]).tolist()

    # 仓位分桶
    buckets = {
        "pos==0": float(np.mean(pos == 0.0)),
        "0<pos<=0.2": float(np.mean((pos > 0.0) & (pos <= 0.2))),
        "0.2<pos<=0.8": float(np.mean((pos > 0.2) & (pos <= 0.8))),
        "pos>0.8": float(np.mean(pos > 0.8)),
    }

    max_dd = float(np.max(dd))
    # 回撤恢复：最后一个 equity >= 之前峰值 的位置（简单版：看是否创过新高）
    recovered = bool(eq[-1] >= np.max(eq) - 1e-12)

    turnover_mean = None
    turnover_p = None
    if trade is not None:
        turnover = np.abs(trade)
        turnover_mean = float(np.mean(turnover))
        turnover_p = np.percentile(turnover, [50, 90, 99]).tolist()

    # 风险响应度：新增回撤时（ddi>0）的平均仓位变化
    risk_response = None
    if ddi is not None and len(ddi) == len(pos):
        idx = np.where(ddi > 1e-12)[0]
        if len(idx) > 5:
            # 看这些时刻之后一格的仓位变化（粗略）
            dpos = np.diff(pos, prepend=pos[0])
            risk_response = {
                "events": int(len(idx)),
                "avg_dpos_at_ddi": float(np.mean(dpos[idx])),
                "median_dpos_at_ddi": float(np.median(dpos[idx])),
                "frac_reduce_at_ddi": float(np.mean(dpos[idx] < 0)),
            }

    # 粗略收益指标（基于 equity）
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    avg_ret = float(np.mean(rets))
    vol_ret = float(np.std(rets))
    sharpe = float(avg_ret / (vol_ret + 1e-12) * np.sqrt(252))  # 日频粗算

    # ---- 输出 ----
    print("=" * 80)
    print("File:", csv_path)
    print(f"len={len(df)}")
    print(f"Avg Position: {avg_pos:.4f}")
    print(f"Position Percentiles (0/10/25/50/75/90/100): {[round(x,4) for x in p_pos]}")
    print("Position Buckets:")
    for k, v in buckets.items():
        print(f"  {k:<12}: {v*100:.1f}%")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"Recovered to new high at end?: {recovered}")
    print(f"Avg daily return (eq): {avg_ret:.6f}, Vol: {vol_ret:.6f}, Sharpe~: {sharpe:.3f}")

    if turnover_mean is not None:
        print(f"Turnover mean(|trade|): {turnover_mean:.6f}, p50/p90/p99={ [round(x,6) for x in turnover_p] }")

    if risk_response is not None:
        print("Risk response at dd_increase events:")
        for k, v in risk_response.items():
            print(f"  {k}: {v}")

    # ---- 画图 ----
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # 1) Equity curve
    plt.figure()
    plt.plot(eq)
    plt.title(f"Equity - {base}")
    plt.xlabel("step")
    plt.ylabel("equity")
    plt.tight_layout()
    f1 = os.path.join(out_dir, f"{base}_equity.png")
    plt.savefig(f1, dpi=150)
    plt.close()

    # 2) Position curve
    plt.figure()
    plt.plot(pos)
    plt.title(f"Position - {base}")
    plt.xlabel("step")
    plt.ylabel("position")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    f2 = os.path.join(out_dir, f"{base}_position.png")
    plt.savefig(f2, dpi=150)
    plt.close()

    print("Saved plots:")
    print(" ", f1)
    print(" ", f2)
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="path to behavior_*.csv")
    ap.add_argument("--out", type=str, default="logs", help="output dir for plots")
    args = ap.parse_args()
    analyze(args.csv, args.out)


if __name__ == "__main__":
    main()
