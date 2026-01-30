import akshare as ak
import pandas as pd
import numpy as np

def load_gongye_fulian_daily():
    """
    拉取工业富联（601138.SH）前复权日线数据
    """
    df = ak.stock_zh_a_hist(
        symbol="601138",
        period="daily",
        start_date="20180101",
        adjust="qfq"
    )

    df = df.rename(columns={
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume"
    })



    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df

def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === 收益 ===
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    # === 波动率（风险） ===
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # === 趋势 ===
    df["ma_20"] = df["close"].rolling(20).mean()
    df["price_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

    # === 用“满仓假设”计算 equity（只用于分析风险） ===
    df["equity"] = (1 + df["ret_1"].fillna(0)).cumprod()
    df["rolling_max_equity"] = df["equity"].cummax()
    df["drawdown"] = (
        df["equity"] - df["rolling_max_equity"]
    ) / df["rolling_max_equity"]

    # 清理前期 NaN
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    df_raw = load_gongye_fulian_daily()
    df_feat = build_risk_features(df_raw)

    print(df_feat.head())
    print(df_feat.describe())

    df_feat.to_csv(
        "../data/gongye_fulian_features.csv",
        index=False,
        encoding="utf-8"
    )

    print("Saved features to data/gongye_fulian_features.csv")
