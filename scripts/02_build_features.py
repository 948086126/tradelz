import numpy as np
import pandas as pd
from pathlib import Path


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 基础收益
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    # 动量
    df["mom_10"] = df["close"] / df["close"].shift(10) - 1.0

    # 波动
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()

    # 均线与偏离
    for w in [5, 10, 20, 60]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"price_vs_ma{w}"] = (df["close"] - df[f"ma_{w}"]) / df[f"ma_{w}"]

    # ATR
    df["tr"] = _true_range(df)
    df["atr_14"] = df["tr"].rolling(14).mean()

    # 区间位置
    df["hhv_20"] = df["high"].rolling(20).max()
    df["llv_20"] = df["low"].rolling(20).min()
    denom = (df["hhv_20"] - df["llv_20"]).replace(0, np.nan)
    df["pos_20"] = (df["close"] - df["llv_20"]) / denom

    # 成交量
    df["vol_chg_5"] = df["volume"].pct_change(5)
    vmean = df["volume"].rolling(20).mean()
    vstd = df["volume"].rolling(20).std().replace(0, np.nan)
    df["volume_z_20"] = (df["volume"] - vmean) / vstd

    # 标签：未来收益（训练reward用）
    df["next_ret_1"] = df["close"].shift(-1) / df["close"] - 1.0
    df["next_ret_5"] = df["close"].shift(-5) / df["close"] - 1.0

    # 回测用：当日开盘到收盘收益（执行在开盘）
    df["ret_oc_1"] = df["close"] / df["open"] - 1.0

    # 清理
    df = df.dropna().reset_index(drop=True)

    # day index
    df["day_idx"] = np.arange(len(df), dtype=int)

    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 你可以按需要改日期；先用稳定的默认
    train_end = pd.Timestamp("2022-12-31")
    valid_end = pd.Timestamp("2024-12-31")

    df["split"] = "test"
    df.loc[df["date"] <= train_end, "split"] = "train"
    df.loc[(df["date"] > train_end) & (df["date"] <= valid_end), "split"] = "valid"
    return df


def main():
    root = Path(__file__).resolve().parents[1]  # tradeLz/
    raw_path = root / "data" / "raw" / "601138_daily_qfq.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data: {raw_path}. Run 01_fetch_daily.py first.")

    df_raw = pd.read_csv(raw_path, parse_dates=["date"])
    df_feat = build_features(df_raw)
    df_feat = add_split(df_feat)

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "601138_features.csv"
    df_feat.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 打印检查
    feature_cols = [
        c for c in df_feat.columns
        if c not in {"date", "open", "high", "low", "close", "volume",
                     "next_ret_1", "next_ret_5", "ret_oc_1", "split", "day_idx", "tr"}
    ]
    print(f"[OK] Saved processed features -> {out_path}")
    print(f"Rows: {len(df_feat)}")
    print(f"Features({len(feature_cols)}): {feature_cols}")
    print(df_feat[["date", "split", "ret_oc_1", "next_ret_1"]].head())
    print(df_feat[["date", "split", "ret_oc_1", "next_ret_1"]].tail())


if __name__ == "__main__":
    main()
