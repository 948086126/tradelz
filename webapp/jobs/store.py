# webapp/jobs/store.py
import os
import json
import time
from typing import Any, Dict, List, Optional

from redis import Redis

# key 前缀
JOB_KEY = "tradelz:job:"
MODEL_KEY = "tradelz:model:"
JOB_INDEX = "tradelz:jobs"      # zset by updated_at
MODEL_INDEX = "tradelz:models"  # zset by created_at


def _redis() -> Redis:
    redis_url = os.getenv("REDIS_URL", "redis://82.157.246.81:6379/0")
    # redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    return Redis.from_url(redis_url, decode_responses=True)


def _now_iso() -> str:
    # 简单 ISO，不强依赖 datetime
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def insert_job(*, job_id: str, job_type: str, status: str, params: Dict[str, Any], log_path: str) -> None:
    r = _redis()
    now = _now_iso()
    doc = {
        "job_id": job_id,
        "created_at": now,
        "updated_at": now,
        "job_type": job_type,
        "status": status,
        "log_path": log_path,
        "error_msg": "",
        "params": params or {},
        "result": {},
    }
    r.set(JOB_KEY + job_id, json.dumps(doc, ensure_ascii=False))
    r.zadd(JOB_INDEX, {job_id: time.time()})


def update_job_status(*, job_id: str, status: str, log_path: Optional[str] = None,
                      result: Optional[Dict[str, Any]] = None, error_msg: str = "") -> None:
    r = _redis()
    key = JOB_KEY + job_id
    raw = r.get(key)
    if not raw:
        # 不抛错，让 worker 自己决定怎么处理
        return
    doc = json.loads(raw)
    doc["status"] = status
    doc["updated_at"] = _now_iso()
    if log_path is not None:
        doc["log_path"] = log_path
    if result is not None:
        doc["result"] = result
    doc["error_msg"] = error_msg or ""
    r.set(key, json.dumps(doc, ensure_ascii=False))
    r.zadd(JOB_INDEX, {job_id: time.time()})


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    r = _redis()
    raw = r.get(JOB_KEY + job_id)
    if not raw:
        return None
    return json.loads(raw)


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    r = _redis()
    ids = r.zrevrange(JOB_INDEX, 0, int(limit) - 1)
    out = []
    for jid in ids:
        d = get_job(jid)
        if d:
            out.append(d)
    return out


def insert_model(*, model_id: str, symbol: str, obs_mode: str, phase: int, shield: bool, use_vecnorm: bool,
                 model_path: str, vecnorm_path: str, meta_path: str) -> None:
    r = _redis()
    now = _now_iso()
    doc = {
        "model_id": model_id,
        "created_at": now,
        "symbol": symbol,
        "obs_mode": obs_mode,
        "phase": int(phase),
        "shield": bool(shield),
        "use_vecnorm": bool(use_vecnorm),
        "model_path": model_path,
        "vecnorm_path": vecnorm_path,
        "meta_path": meta_path,
    }
    r.set(MODEL_KEY + model_id, json.dumps(doc, ensure_ascii=False))
    r.zadd(MODEL_INDEX, {model_id: time.time()})


def list_models(limit: int = 50) -> List[Dict[str, Any]]:
    r = _redis()
    ids = r.zrevrange(MODEL_INDEX, 0, int(limit) - 1)
    out = []
    for mid in ids:
        raw = r.get(MODEL_KEY + mid)
        if raw:
            out.append(json.loads(raw))
    return out

