# webapp/app.py
import os
import uuid
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from webapp.db import init_db, insert_run, list_runs, get_run
from webapp.schemas import BacktestRunRequest, BacktestRunResponse, BacktestRunListResponse, BacktestRunItem
from core.backtest import run_backtest  # 你已有的函数
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from webapp.jobs.store import insert_job, get_job, list_jobs, list_models
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi.responses import PlainTextResponse

app = FastAPI(title="tradeLz")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True}
@app.get("/train/models/list")
def train_models_list(limit: int = 50):
    return {"items": list_models(limit=limit)}

# webapp/app.py (替换 backtest_run 函数)
from webapp.schemas import (
    BacktestRunRequest,
    BacktestRunResponse,
    BacktestCompareResponse,
    BacktestSingleResult,
)

class TrainStartRequest(BaseModel):
    data: str
    symbol: str = "601138"
    obs_mode: str = "S"
    phase: int = 2
    total_steps: int = 300000
    seed: int = 42
    shield: bool = True
    use_vecnorm: bool = True
    init_model: str = ""  # 可选：微调/续训
    extra_env_kwargs: Optional[Dict[str, Any]] = None  # 预留：后续 UI 微调参数用

class TrainStartResponse(BaseModel):
    job_id: str

@app.post("/train/start", response_model=TrainStartResponse)
def train_start(req: TrainStartRequest):
    job_id = uuid.uuid4().hex[:12]

    log_dir = "logs_jobs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{req.symbol}_{job_id}.log")

    params = req.model_dump()
    params["log_path"] = log_path

    insert_job(
        job_id=job_id,
        job_type="train",
        status="queued",
        params=params,
        log_path=log_path,
    )

    from webapp.jobs.queue import get_queue
    q = get_queue()
    q.enqueue("webapp.jobs.tasks.run_train_job", job_id, job_timeout=60*60*6)

    return TrainStartResponse(job_id=job_id)



@app.get("/train/log/{job_id}", response_class=PlainTextResponse)
def train_log(job_id: str, tail: int = 400):
    """
    读取训练日志。默认返回最后 400 行，可用 ?tail=1000 调大
    """
    d = get_job(job_id)
    if d is None:
        raise HTTPException(status_code=404, detail="job not found")

    log_path = d.get("log_path") or ""
    if not log_path or not os.path.exists(log_path):
        return f"[log not found] {log_path}"

    # 读最后 tail 行，避免大文件卡死
    try:
        with open(log_path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()[-int(tail):]
        return "\n".join(lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{job_id}")
def train_get(job_id: str):
    d = get_job(job_id)
    if d is None:
        raise HTTPException(status_code=404, detail="job not found")
    return d

@app.get("/train/jobs/list")
def train_jobs_list(limit: int = 50):
    return {"items": list_jobs(limit=limit)}

@app.get("/train", response_class=HTMLResponse)
def train_page(request: Request):
    jobs = list_jobs(limit=50)
    models = list_models(limit=50)
    return templates.TemplateResponse(
        "train.html",
        {"request": request, "jobs": jobs, "models": models},
    )

@app.post("/backtest/run")
def backtest_run(req: BacktestRunRequest):
    if not os.path.exists(req.data_csv_path):
        raise HTTPException(status_code=400, detail="data_csv_path not found: %s" % req.data_csv_path)

    df = pd.read_csv(req.data_csv_path)
    if "close" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain column 'close'")

    log_dir = "logs"

    env_kwargs = dict(req.env_kwargs or {})
    env_kwargs.setdefault("cost_rate", 0.001)
    env_kwargs.setdefault("max_step_change", 0.10)
    env_kwargs.setdefault("ddi_penalty", 2.0)
    env_kwargs.setdefault("use_action_shield", True)

    policy_kwargs = dict(req.policy_kwargs or {})
    if "max_step_change" not in policy_kwargs:
        policy_kwargs["max_step_change"] = env_kwargs.get("max_step_change", 0.10)

    # ===== 单跑 =====
    if not req.compare_shield:
        run_id = uuid.uuid4().hex[:12]
        log_stem = "run_%s_%s" % (req.symbol, run_id)

        result = run_backtest(
            df,
            policy_name=req.policy_name,
            policy_kwargs=policy_kwargs,
            env_kwargs=env_kwargs,
            log_dir=log_dir,
            log_stem=log_stem,
            force_shield=None,  # 按 env_kwargs
            obs_mode=req.obs_mode,  # ✅ 新增
        )

        insert_run(
            run_id=run_id,
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            policy_name=req.policy_name,
            policy_kwargs=policy_kwargs,
            env_kwargs=env_kwargs,
            metrics=result.metrics,
            csv_path=result.csv_path,
            equity_png=result.equity_png,
            pos_png=result.pos_png,
        )

        return BacktestRunResponse(
            run_id=run_id,
            metrics=result.metrics,
            csv_path=result.csv_path,
            equity_png=result.equity_png,
            pos_png=result.pos_png,
        )

    # ===== 对照：Shield ON vs OFF =====
    run_id_on = uuid.uuid4().hex[:12]
    run_id_off = uuid.uuid4().hex[:12]

    stem_on = "run_%s_%s_shieldON" % (req.symbol, run_id_on)
    stem_off = "run_%s_%s_shieldOFF" % (req.symbol, run_id_off)

    result_on = run_backtest(
        df,
        policy_name=req.policy_name,
        policy_kwargs=policy_kwargs,
        env_kwargs=env_kwargs,
        log_dir=log_dir,
        log_stem=stem_on,
        force_shield=True,
    )
    result_off = run_backtest(
        df,
        policy_name=req.policy_name,
        policy_kwargs=policy_kwargs,
        env_kwargs=env_kwargs,
        log_dir=log_dir,
        log_stem=stem_off,
        force_shield=False,
    )

    # 入库（两条）
    env_on = dict(env_kwargs); env_on["use_action_shield"] = True
    env_off = dict(env_kwargs); env_off["use_action_shield"] = False

    insert_run(
        run_id=run_id_on,
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        policy_name=req.policy_name + "_shieldON",
        policy_kwargs=policy_kwargs,
        env_kwargs=env_on,
        metrics=result_on.metrics,
        csv_path=result_on.csv_path,
        equity_png=result_on.equity_png,
        pos_png=result_on.pos_png,
    )
    insert_run(
        run_id=run_id_off,
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        policy_name=req.policy_name + "_shieldOFF",
        policy_kwargs=policy_kwargs,
        env_kwargs=env_off,
        metrics=result_off.metrics,
        csv_path=result_off.csv_path,
        equity_png=result_off.equity_png,
        pos_png=result_off.pos_png,
    )

    # diff（只做关键几项，够用了）
    def _get(m, k, default=0.0):
        try:
            return float(m.get(k, default))
        except Exception:
            return float(default)

    diff = {
        "equity_end_diff": _get(result_on.metrics, "equity_end") - _get(result_off.metrics, "equity_end"),
        "max_drawdown_diff": _get(result_on.metrics, "max_drawdown") - _get(result_off.metrics, "max_drawdown"),
        "avg_position_diff": _get(result_on.metrics, "avg_position") - _get(result_off.metrics, "avg_position"),
        "turnover_mean_diff": _get(result_on.metrics, "turnover_mean") - _get(result_off.metrics, "turnover_mean"),
        "cap_hit_ratio_on": _get(result_on.metrics, "cap_hit_ratio", 0.0),
        "avg_cap_on": _get(result_on.metrics, "avg_cap", 1.0),
    }

    return BacktestCompareResponse(
        symbol=req.symbol,
        policy_name=req.policy_name,
        shield_on=BacktestSingleResult(
            run_id=run_id_on,
            metrics=result_on.metrics,
            csv_path=result_on.csv_path,
            equity_png=result_on.equity_png,
            pos_png=result_on.pos_png,
        ),
        shield_off=BacktestSingleResult(
            run_id=run_id_off,
            metrics=result_off.metrics,
            csv_path=result_off.csv_path,
            equity_png=result_off.equity_png,
            pos_png=result_off.pos_png,
        ),
        diff=diff,
    )

@app.get("/runs/list", response_model=BacktestRunListResponse)
def runs_list(limit: int = 50):
    items_raw = list_runs(limit=limit)
    items = []
    for d in items_raw:
        items.append(
            BacktestRunItem(
                run_id=d["run_id"],
                created_at=d["created_at"],
                symbol=d["symbol"],
                start_date=d.get("start_date"),
                end_date=d.get("end_date"),
                policy_name=d["policy_name"],
                metrics=d.get("metrics", {}),
                csv_path=d["csv_path"],
                equity_png=d["equity_png"],
                pos_png=d["pos_png"],
            )
        )
    return BacktestRunListResponse(items=items)


@app.get("/runs/{run_id}")
def runs_get(run_id: str):
    d = get_run(run_id)
    if d is None:
        raise HTTPException(status_code=404, detail="run not found")
    return d
@app.get("/train/models/list")
def train_models_list(limit: int = 50):
    return {"items": list_models(limit=limit)}


# ===== 新增：里程碑 A - 资金曲线展示 =====

# ===== 里程碑 A：资金曲线展示 =====

@app.get("/api/backtest/equity/{run_id}")
def get_equity_data(run_id: str):
    """
    返回指定 run_id 的资金曲线数据（用于前端绘图）
    """
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    csv_path = run.get("csv_path")
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")

    df = pd.read_csv(csv_path)

    # 确保必要的列存在
    required_cols = ["date", "equity", "position"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"CSV missing columns: {missing}")

    # 转换为前端需要的格式
    data = {
        "dates": df["date"].tolist(),
        "equity": df["equity"].tolist(),
        "position": df["position"].tolist(),
    }

    return data


@app.get("/milestone-a", response_class=HTMLResponse)
def milestone_a_page(request: Request):
    """
    里程碑 A 专用页面：展示最新的回测结果
    """
    runs = list_runs(limit=1)  # 获取最新的一条回测记录

    if not runs:
        return templates.TemplateResponse(
            "milestone_a.html",
            {"request": request, "has_data": False, "run": None},
        )

    latest_run = runs[0]

    return templates.TemplateResponse(
        "milestone_a.html",
        {
            "request": request,
            "has_data": True,
            "run": latest_run,
        },
    )