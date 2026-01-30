# webapp/db.py
import os
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional, List

DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,

            symbol TEXT NOT NULL,
            start_date TEXT,
            end_date TEXT, 

            policy_name TEXT NOT NULL,
            policy_kwargs_json TEXT NOT NULL,
            env_kwargs_json TEXT NOT NULL,

            metrics_json TEXT NOT NULL,

            csv_path TEXT NOT NULL,
            equity_png TEXT NOT NULL,
            pos_png TEXT NOT NULL
        );
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_created_at ON backtest_runs(created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_symbol ON backtest_runs(symbol);")
    # ===== jobs: 训练/微调/回测等任务队列 =====
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,

            job_type TEXT NOT NULL,         -- train / finetune / backtest / fetch_data / live
            status TEXT NOT NULL,           -- queued / running / done / failed

            params_json TEXT NOT NULL,
            log_path TEXT,
            result_json TEXT NOT NULL,
            error_msg TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(job_type);")

    # ===== models: 模型产物登记 =====
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,

            symbol TEXT NOT NULL,
            obs_mode TEXT NOT NULL,
            phase INTEGER NOT NULL,
            shield INTEGER NOT NULL,
            use_vecnorm INTEGER NOT NULL,

            model_path TEXT NOT NULL,
            vecnorm_path TEXT,
            meta_path TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_models_symbol ON models(symbol);")


    conn.commit()
    conn.close()


def insert_run(
    *,
    run_id: str,
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
    policy_name: str,
    policy_kwargs: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    metrics: Dict[str, Any],
    csv_path: str,
    equity_png: str,
    pos_png: str,
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO backtest_runs (
            run_id, created_at,
            symbol, start_date, end_date,
            policy_name, policy_kwargs_json, env_kwargs_json,
            metrics_json,
            csv_path, equity_png, pos_png
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            run_id,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            symbol,
            start_date,
            end_date,
            policy_name,
            json.dumps(policy_kwargs, ensure_ascii=False),
            json.dumps(env_kwargs, ensure_ascii=False),
            json.dumps(metrics, ensure_ascii=False),
            csv_path,
            equity_png,
            pos_png,
        ),
    )
    conn.commit()
    conn.close()


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id, created_at, symbol, start_date, end_date,
               policy_name, metrics_json, csv_path, equity_png, pos_png
        FROM backtest_runs
        ORDER BY created_at DESC
        LIMIT ?;
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["metrics"] = json.loads(d.pop("metrics_json"))
        except Exception:
            d["metrics"] = {}
        out.append(d)
    return out


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM backtest_runs
        WHERE run_id = ?;
        """,
        (run_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None

    d = dict(row)
    for k in ["policy_kwargs_json", "env_kwargs_json", "metrics_json"]:
        try:
            d[k.replace("_json", "")] = json.loads(d.pop(k))
        except Exception:
            d[k.replace("_json", "")] = {}
    return d
# ===== jobs API =====
def insert_job(*, job_id: str, job_type: str, status: str, params: dict, log_path: str = "") -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs (job_id, created_at, job_type, status, params_json, log_path, result_json, error_msg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            job_id,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            job_type,
            status,
            json.dumps(params, ensure_ascii=False),
            log_path or "",
            json.dumps({}, ensure_ascii=False),
            "",
        ),
    )
    conn.commit()
    conn.close()


def update_job_status(*, job_id: str, status: str, log_path: str = "", result: dict = None, error_msg: str = "") -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE jobs
        SET status = ?,
            log_path = COALESCE(NULLIF(?, ''), log_path),
            result_json = ?,
            error_msg = ?
        WHERE job_id = ?;
        """,
        (
            status,
            log_path or "",
            json.dumps(result or {}, ensure_ascii=False),
            error_msg or "",
            job_id,
        ),
    )
    conn.commit()
    conn.close()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    try:
        d["params"] = json.loads(d.pop("params_json"))
    except Exception:
        d["params"] = {}
    try:
        d["result"] = json.loads(d.pop("result_json"))
    except Exception:
        d["result"] = {}
    return d


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT job_id, created_at, job_type, status, params_json, log_path, result_json, error_msg
        FROM jobs
        ORDER BY created_at DESC
        LIMIT ?;
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        d = dict(r)
        try:
            d["params"] = json.loads(d.pop("params_json"))
        except Exception:
            d["params"] = {}
        try:
            d["result"] = json.loads(d.pop("result_json"))
        except Exception:
            d["result"] = {}
        out.append(d)
    return out


# ===== models API =====
def insert_model(
    *,
    model_id: str,
    symbol: str,
    obs_mode: str,
    phase: int,
    shield: bool,
    use_vecnorm: bool,
    model_path: str,
    vecnorm_path: str = "",
    meta_path: str = "",
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO models (
            model_id, created_at,
            symbol, obs_mode, phase, shield, use_vecnorm,
            model_path, vecnorm_path, meta_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            model_id,
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            symbol,
            obs_mode,
            int(phase),
            1 if shield else 0,
            1 if use_vecnorm else 0,
            model_path,
            vecnorm_path or "",
            meta_path or "",
        ),
    )
    conn.commit()
    conn.close()


def list_models(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM models
        ORDER BY created_at DESC
        LIMIT ?;
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
