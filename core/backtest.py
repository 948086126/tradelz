# core/backtest.py
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from envs.trading_env import TradingEnv
from envs.wrappers import ObsSelectWrapper

from policies.policies import always_zero_policy, buy_and_hold_policy
from policies.ddcontrol import ddcontrol_with_features

from stable_baselines3 import PPO

from pathlib import Path
from uuid import uuid4
from datetime import datetime


# -------------------------
# PPO policy wrapper
# -------------------------
class PPOPolicy:
    """
    只负责 predict，不负责 load（load 在 run_backtest 里做，这样可以：
    1) 记录 model_load.txt 证据
    2) 避免重复 load
    """
    def __init__(self, model: PPO, deterministic: bool = True):
        self.model = model
        self.deterministic = bool(deterministic)

    def __call__(self, obs: np.ndarray, *, state: Optional[dict] = None, **kwargs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return action


POLICY_REGISTRY = {
    "always0": always_zero_policy,
    "buyhold": buy_and_hold_policy,
    "ddcontrol": ddcontrol_with_features,
    "ppo": None,  # 特判
}


@dataclass
class BacktestResult:
    csv_path: str
    equity_png: str
    pos_png: str
    metrics: Dict[str, Any]


def _make_run_dir(base_dir: Optional[str] = None) -> Path:
    """
    在 webapp/cache/runs 下创建唯一目录，或者在 base_dir 下创建子目录（更通用）。
    """
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    if base_dir:
        run_dir = Path(base_dir) / run_id
    else:
        run_dir = Path("webapp") / "cache" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _log_model_load(run_dir: Path, model_path: str, model: PPO) -> None:
    p = Path(model_path).expanduser()
    p_abs = p.resolve()
    mtime = p_abs.stat().st_mtime if p_abs.exists() else None
    txt = (
        f"model_path_input = {model_path}\n"
        f"model_path_abs   = {p_abs}\n"
        f"exists           = {p_abs.exists()}\n"
        f"mtime_epoch      = {mtime}\n"
        f"observation_space= {getattr(model, 'observation_space', None)}\n"
        f"action_space     = {getattr(model, 'action_space', None)}\n"
    )
    (run_dir / "model_load.txt").write_text(txt, encoding="utf-8")


def _ensure_zip_suffix(model_path: str) -> str:
    p = Path(model_path)
    # 防止 .zip.zip
    if p.suffix.lower() != ".zip":
        p = p.with_suffix(".zip")
    return str(p)


def _obs_idxs_from_mode(obs_mode: str):
    """
    TradingEnv obs 9维顺序（固定）：
    [0 position,
     1 ret_1,
     2 ret_5,
     3 ret_20,
     4 vol_20,
     5 price_vs_ma20,
     6 dd,
     7 dd_increase,
     8 equity_log]
    """
    m = (obs_mode or "S").upper()

    if m == "S":
        # ✅ 必须与最新 train_ppo.py 对齐（你 meta 里也是这套）
        # position, ret_1, ret_5, ret_20, price_vs_ma20
        return [0, 1, 2, 3, 5]

    if m == "M":
        return [0, 1, 2, 3, 4, 5]

    if m == "L":
        return [0, 1, 2, 3, 4, 5, 7]

    raise ValueError("Unknown obs_mode: %s" % obs_mode)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _calc_metrics_from_csv(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if len(df) < 5:
        return {"len": int(len(df))}

    eq = df["equity"].astype(float).values
    pos = df["position"].astype(float).values
    trade = df["executed_trade"].astype(float).values

    rets = np.zeros_like(eq)
    rets[1:] = (eq[1:] / np.maximum(eq[:-1], 1e-12)) - 1.0

    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    max_dd = float(np.max(dd))

    abs_trade = np.abs(trade)
    turnover_mean = float(np.mean(abs_trade))
    nz_ratio = float(np.mean(abs_trade > 1e-12))

    mu = float(np.mean(rets))
    sigma = float(np.std(rets) + 1e-12)
    sharpe = float(mu / sigma)

    metrics = {
        "len": int(len(df)),
        "equity_end": float(eq[-1]),
        "max_drawdown": max_dd,
        "avg_position": float(np.mean(pos)),
        "pos_min": float(np.min(pos)),
        "pos_max": float(np.max(pos)),
        "turnover_mean": turnover_mean,
        "trade_nonzero_ratio": nz_ratio,
        "avg_step_return": mu,
        "vol_step": float(np.std(rets)),
        "sharpe_step": sharpe,
    }

    cols = set(df.columns)
    if ("raw_target" in cols) and ("risk_cap" in cols) and ("shielded_target" in cols):
        raw_t = df["raw_target"].astype(float).values
        cap = df["risk_cap"].astype(float).values
        sh_t = df["shielded_target"].astype(float).values

        cap_hit_ratio = float(np.mean(sh_t < (raw_t - 1e-12)))
        metrics["cap_hit_ratio"] = cap_hit_ratio
        metrics["avg_cap"] = float(np.mean(cap))
        metrics["cap_min"] = float(np.min(cap))
        metrics["cap_max"] = float(np.max(cap))

    return metrics


def _plot_equity_pos(csv_path: str, out_equity_png: str, out_pos_png: str) -> None:
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["equity"].astype(float).values)
    plt.title("Equity")
    plt.xlabel("step")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(out_equity_png)
    plt.close()

    plt.figure()
    plt.plot(df["position"].astype(float).values)
    plt.title("Position")
    plt.xlabel("step")
    plt.ylabel("pos")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_pos_png)
    plt.close()


def _dump_first_n_steps_from_csv(csv_path: str, out_txt: str, n: int = 30) -> None:
    """
    不依赖 env 的 print，直接从 csv 抽前 N 步，写到 first_30_steps.txt
    """
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        Path(out_txt).write_text("EMPTY CSV\n", encoding="utf-8")
        return

    take = df.head(n).copy()

    # 尽量兼容不同列名：你现在行为日志里常见这些列
    col_map = {
        "step": "step",
        "price_return": "price_return",
        "action_delta": "action_delta",
        "risk_cap": "risk_cap",
        "raw_target": "raw_target",
        "shielded_target": "shielded_target",
        "position": "position",
        "equity": "equity",
        "dd": "dd",
        "dd_increase": "dd_increase",
        "reward": "reward",
    }
    cols = set(take.columns)

    lines = []
    for _, r in take.iterrows():
        # 兼容：某些日志可能没有 step 列（用 index 代替）
        step = int(r["step"]) if "step" in cols else int(_)

        ret = float(r["price_return"]) if "price_return" in cols else float("nan")
        act = float(r["action_delta"]) if "action_delta" in cols else float("nan")
        cap = float(r["risk_cap"]) if "risk_cap" in cols else float("nan")
        raw = float(r["raw_target"]) if "raw_target" in cols else float("nan")
        sh = float(r["shielded_target"]) if "shielded_target" in cols else float("nan")
        pos = float(r["position"]) if "position" in cols else float("nan")
        eq = float(r["equity"]) if "equity" in cols else float("nan")
        dd = float(r["dd"]) if "dd" in cols else float("nan")
        ddi = float(r["dd_increase"]) if "dd_increase" in cols else float("nan")
        rew = float(r["reward"]) if "reward" in cols else float("nan")

        line = (
            f"step={step:4d} | ret={ret:+.4f} | act={act:+.2f} | cap={cap:.2f} | "
            f"raw={raw:.2f} | sh={sh:.2f} | pos={pos:.2f} | eq={eq:.3f} | "
            f"dd={dd:.3f} | ddi={ddi:.4f} | reward={rew:+.6f}"
        )
        lines.append(line)

    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backtest(
    df: pd.DataFrame,
    *,
    policy_name: str,
    policy_kwargs: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    log_dir: str,
    log_stem: str,
    force_shield: Optional[bool] = None,
    obs_mode: str = "S",
    debug_first_n: int = 30,   # ✅ 新增：默认写前30步到文件
    auto_run_subdir: bool = True,  # ✅ 新增：在 log_dir 下自动建 run_id 子目录
) -> BacktestResult:
    if policy_name not in POLICY_REGISTRY:
        raise ValueError("Unknown policy_name: %s" % policy_name)

    # 1) 创建本次 run 的输出目录（避免相互覆盖）
    _ensure_dir(log_dir)
    out_dir = Path(log_dir)
    if auto_run_subdir:
        out_dir = _make_run_dir(base_dir=log_dir)

    csv_path = str(out_dir / f"{log_stem}.csv")
    equity_png = str(out_dir / f"{log_stem}_equity.png")
    pos_png = str(out_dir / f"{log_stem}_pos.png")
    first_n_txt = str(out_dir / "first_30_steps.txt")

    # 2) env_kwargs 复制 + 覆盖 shield（关键修复）
    _env_kwargs = dict(env_kwargs or {})
    if force_shield is not None:
        _env_kwargs["use_action_shield"] = bool(force_shield)

    # ✅ 强制日志写到 out_dir
    _env_kwargs["log_dir"] = str(out_dir)
    _env_kwargs["log_name"] = f"{log_stem}.csv"

    # 3) 创建 env
    env = TradingEnv(df=df, **_env_kwargs)

    # 4) obs wrapper：必须与训练一致
    obs_idxs = _obs_idxs_from_mode(obs_mode)
    env = ObsSelectWrapper(env, obs_idxs)

    state: Dict[str, Any] = {}
    obs, _ = env.reset()
    done = False

    # 5) build policy
    if policy_name == "ppo":
        model_path = (policy_kwargs or {}).get("model_path")
        if not model_path:
            raise ValueError("ppo requires policy_kwargs['model_path']")

        # ✅ 统一补 .zip（避免 zip.zip）
        model_path = _ensure_zip_suffix(model_path)

        # ✅ 先 load 一次，写 model_load.txt 证据
        model = PPO.load(model_path)
        _log_model_load(out_dir, model_path, model)

        policy_fn = PPOPolicy(model=model, deterministic=True)
    else:
        policy_fn = POLICY_REGISTRY[policy_name]

    # 6) rollout
    while not done:
        action = policy_fn(obs, state=state, **(policy_kwargs or {}))
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

    env.close()

    # 7) 绘图 + 指标
    _plot_equity_pos(csv_path, equity_png, pos_png)
    metrics = _calc_metrics_from_csv(csv_path)

    # 8) ✅ 2.3：把前30步写入文件（从 csv 抽）
    if debug_first_n and debug_first_n > 0:
        _dump_first_n_steps_from_csv(csv_path, first_n_txt, n=int(debug_first_n))

    return BacktestResult(
        csv_path=csv_path,
        equity_png=equity_png,
        pos_png=pos_png,
        metrics=metrics,
    )
