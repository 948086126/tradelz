# webapp/jobs/tasks.py
import sys
import os
import subprocess
from typing import Dict, Any

from webapp.jobs.store import get_job, update_job_status, insert_model



def _run_tag(symbol: str, obs_mode: str, phase: int, shield: bool) -> str:
    shield_tag = "shieldON" if shield else "shieldOFF"
    return f"{symbol}_obs{obs_mode}_{shield_tag}_p{phase}"


def _artifact_paths(run_tag: str):
    model_path = f"models/ppo_{run_tag}.zip"
    vecnorm_path = f"models/vecnorm_{run_tag}.pkl"
    meta_path = f"logs_rl/runmeta_{run_tag}.json"
    return model_path, vecnorm_path, meta_path


def run_train_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"job not found in redis: {job_id}")

    params: Dict[str, Any] = job.get("params", {}) or {}

    symbol = str(params.get("symbol", "601138"))
    obs_mode = str(params.get("obs_mode", "S")).upper()
    phase = int(params.get("phase", 1))
    total_steps = int(params.get("total_steps", 300_000))
    seed = int(params.get("seed", 42))
    shield = bool(params.get("shield", True))
    use_vecnorm = bool(params.get("use_vecnorm", True))
    data_path = str(params.get("data", "data/gongye_fulian_features.csv"))
    init_model = str(params.get("init_model", "")).strip()

    log_path = str(params.get("log_path", f"logs_jobs/train_{job_id}.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    update_job_status(job_id=job_id, status="running", log_path=log_path, result={}, error_msg="")

    project_root = os.getenv("TRADELZ_ROOT") or os.getcwd()

    abs_data = data_path if os.path.isabs(data_path) else os.path.join(project_root, data_path)
    if not os.path.exists(abs_data):
        msg = f"data not found: {data_path} (abs={abs_data})"
        update_job_status(job_id=job_id, status="failed", log_path=log_path, result={}, error_msg=msg)
        raise RuntimeError(msg)

    cmd = [
        sys.executable, "-m", "core.train_ppo",
        "--data", data_path,
        "--symbol", symbol,
        "--obs_mode", obs_mode,
        "--phase", str(phase),
        "--total_steps", str(total_steps),
        "--seed", str(seed),
    ]

    cmd.append("--shield" if shield else "--no_shield")
    if use_vecnorm:
        cmd.append("--use_vecnorm")
    if init_model:
        cmd += ["--init_model", init_model]

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[CWD] " + project_root + "\n")
            f.write("[CMD] " + " ".join(cmd) + "\n\n")
            f.flush()

            p = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root,
                env=os.environ.copy(),
            )
            code = p.wait()

        if code != 0:
            msg = f"train exit_code={code} (see log)"
            update_job_status(job_id=job_id, status="failed", log_path=log_path, result={}, error_msg=msg)
            raise RuntimeError(msg)

        run_tag = _run_tag(symbol, obs_mode, phase, shield)
        model_path, vecnorm_path, meta_path = _artifact_paths(run_tag)

        result = {
            "run_tag": run_tag,
            "model_path": model_path if os.path.exists(os.path.join(project_root, model_path)) else "",
            "vecnorm_path": vecnorm_path if os.path.exists(os.path.join(project_root, vecnorm_path)) else "",
            "meta_path": meta_path if os.path.exists(os.path.join(project_root, meta_path)) else "",
        }

        if not result["model_path"]:
            msg = "model zip not found after training"
            update_job_status(job_id=job_id, status="failed", log_path=log_path, result=result, error_msg=msg)
            raise RuntimeError(msg)

        update_job_status(job_id=job_id, status="done", log_path=log_path, result=result, error_msg="")

        insert_model(
            model_id=job_id,
            symbol=symbol,
            obs_mode=obs_mode,
            phase=phase,
            shield=shield,
            use_vecnorm=use_vecnorm,
            model_path=result["model_path"],
            vecnorm_path=result["vecnorm_path"],
            meta_path=result["meta_path"],  # ✅
        )


    except Exception as e:
        update_job_status(job_id=job_id, status="failed", log_path=log_path, result={}, error_msg=str(e))
        raise
