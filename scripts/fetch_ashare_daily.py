import akshare as ak
import pandas as pd
import os

# ========= 配置 =========
SYMBOL = "601138"          # 工业富联
START_DATE = "20180101"    # 上市后较完整
END_DATE = "20241231"      # 可写今天，AkShare会自动处理
OUT_PATH = "../data/601138_daily.csv"


def fetch_daily_ohlcv():
    print("Fetching daily data for", SYMBOL)

    df = ak.stock_zh_a_hist(
        symbol=SYMBOL,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="qfq",   # 前复权（推荐）
    )

    if df.empty:
        raise RuntimeError("AkShare 返回空数据")

    # 统一列名（你 env / analyze 用这个标准）
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    })

    df = df[[
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    fetch_daily_ohlcv()


