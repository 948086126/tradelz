# webapp/services/feature_service.py
import pandas as pd

import numpy as np

def build_features(df: pd.DataFrame, dd_window: int = 252) -> pd.DataFrame:
    """
    生成：ret_1/5/20, vol_20, ma_20, price_vs_ma20
    rolling drawdown（dd_window）与 dd_increase
    """
    df = df.copy()
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["ma_20"] = df["close"].rolling(20).mean()
    df["price_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

    # 满仓假设 equity（仅用于计算rolling dd特征，不参与交易）
    df["equity_full"] = (1 + df["ret_1"].fillna(0)).cumprod()

    # rolling drawdown（关键：避免“2018高点阴影”导致长期压仓位）
    roll_max = df["equity_full"].rolling(dd_window, min_periods=1).max()
    dd = (roll_max - df["equity_full"]) / roll_max
    df["dd"] = dd

    df["dd_increase"] = (df["dd"] - df["dd"].shift(1)).clip(lower=0.0).fillna(0.0)

    # 清理 NaN
    for c in ["ret_1","ret_5","ret_20","vol_20","ma_20","price_vs_ma20","dd","dd_increase"]:
        df[c] = df[c].replace([np.inf,-np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)
    return df
