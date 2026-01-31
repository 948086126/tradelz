# tradelz/script/build_daily_features.py

import akshare as ak
import pandas as pd
import numpy as np


# =========================
# 1. 拉取日线数据（前复权）
# =========================
def load_daily_data(symbol: str, start_date="20180101"):
    """
    拉取 A 股前复权日线数据
    symbol: e.g. "601138"
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        adjust="qfq"
    )

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 2. 构建因子
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ===== 收益类 =====
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    # ===== 波动率 =====
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # ===== 均线 & 趋势 =====
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_60"] = df["close"].rolling(60).mean()

    df["price_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["ma5_vs_ma20"] = (df["ma_5"] - df["ma_20"]) / df["ma_20"]
    df["ma20_vs_ma60"] = (df["ma_20"] - df["ma_60"]) / df["ma_60"]

    # ===== 动量 =====
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # ===== 成交量 =====
    df["vol_mean_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_mean_20"]

    # ===== 风险状态（回撤）=====
    df["equity_full"] = (1 + df["ret_1"].fillna(0)).cumprod()
    df["rolling_max"] = df["equity_full"].cummax()
    df["drawdown"] = (df["equity_full"] - df["rolling_max"]) / df["rolling_max"]

    # ===== 监督标签（给 PPO reward 用）=====
    df["next_ret_1"] = df["ret_1"].shift(-1)

    # ===== 清洗 =====
    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# 3. 主入口
# =========================
if __name__ == "__main__":
    SYMBOL = "601138"  # 工业富联
    OUTPUT_PATH = "../data/gongye_fulian_daily_features.csv"

    df_raw = load_daily_data(SYMBOL)
    df_feat = build_features(df_raw)

    print("Feature columns:")
    print(df_feat.columns.tolist())

    print("\nPreview:")
    print(df_feat.head())

    df_feat.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved features to {OUTPUT_PATH}")
