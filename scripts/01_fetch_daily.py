import pandas as pd
import akshare as ak
from pathlib import Path


def fetch_daily_qfq(symbol: str = "601138", start_date: str = "20180101") -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        adjust="qfq",
    )

    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    })

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # 保留我们需要的列（别让杂列污染后续）
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[keep]

    # 基本检查
    if df.isna().any().any():
        raise ValueError("Raw daily data contains NaN, check akshare output.")

    return df


def main():
    root = Path(__file__).resolve().parents[1]  # tradeLz/
    out_dir = root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol = "601138"
    df = fetch_daily_qfq(symbol=symbol, start_date="20180101")
    out_path = out_dir / f"{symbol}_daily_qfq.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved raw daily data -> {out_path}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
