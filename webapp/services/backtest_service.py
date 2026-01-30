# webapp/services/backtest_service.py
import json
import os
from typing import Dict, Any

import pandas as pd

from webapp.db import get_conn


def save_backtest_to_db(
    *,
    symbol: str,
    start: str,
    end: str,
    policy_name: str,
    params: Dict[str, Any],
    result_csv: str,
    equity_png: str,
    pos_png: str,
    metrics: Dict[str, Any],
) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO backtests(symbol, start, end, policy_name, params_json, metrics_json, csv_path, equity_png, pos_png)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            start,
            end,
            policy_name,
            json.dumps(params, ensure_ascii=False),
            json.dumps(metrics, ensure_ascii=False),
            result_csv,
            equity_png,
            pos_png,
        ),
    )
    conn.commit()
    backtest_id = int(cur.lastrowid)
    conn.close()
    return backtest_id


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
