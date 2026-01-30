from pathlib import Path
import pandas as pd
import akshare as ak

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_daily_akshare(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """
    AkShare 日线：symbol 如 "601138"
    start_date/end_date: "YYYYMMDD"
    adjust: "", "qfq", "hfq"
    """
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust
    )

    # 兼容字段
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # 保留常用列（存在就保留，不存在就忽略）
    keep = [c for c in ["date","open","high","low","close","volume","成交额","振幅","涨跌幅","换手率","股票代码"] if c in df.columns]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df

def cache_path(symbol: str, start_date: str, end_date: str, adjust: str) -> Path:
    name = f"{symbol}_{start_date}_{end_date}_{adjust}.csv"
    return CACHE_DIR / name

def save_cache(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False, encoding="utf-8")
    return str(path)
