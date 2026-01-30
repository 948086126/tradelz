import numpy as np
from typing import Optional

def _delta_from_target(position: float, target_position: float, max_step_change: float) -> float:
    target_position = float(np.clip(target_position, 0.0, 1.0))
    position = float(np.clip(position, 0.0, 1.0))
    need = target_position - position
    action_delta = need / max_step_change
    return float(np.clip(action_delta, -1.0, 1.0))


def always_zero_policy(obs: np.ndarray, *, max_step_change: float, **kwargs) -> np.ndarray:
    """
    永远目标仓位 0
    obs: [position, price_return, drawdown, equity_log]
    """
    position = float(obs[0])
    action_delta = _delta_from_target(position, 0.0, max_step_change)
    return np.array([action_delta], dtype=np.float32)


def buy_and_hold_policy(obs: np.ndarray, *, max_step_change: float, **kwargs) -> np.ndarray:
    """
    尽快加到 1，然后不动（在 delta env 下就是每步往 1 推，直到到达）
    """
    position = float(obs[0])
    action_delta = _delta_from_target(position, 1.0, max_step_change)
    return np.array([action_delta], dtype=np.float32)


def drawdown_control_policy(
    obs: np.ndarray,
    *,
    max_step_change: float,
    # 仓位参数
    base_pos: float = 0.6,
    min_pos: float = 0.1,
    risk_cap: float = 0.2,          # 风险状态最大仓位上限
    # 回撤阈值
    dd_low: float = 0.02,
    dd_high: float = 0.06,
    # 恢复机制
    cooldown_steps: int = 10,       # 需要连续多少天“低风险”才允许退出风险状态
    recover_step: float = 0.05,     # 每天最多恢复多少仓位（斜率限制）
    # 跨步状态
    state: Optional[dict] = None,
    **kwargs
) -> np.ndarray:
    """
    解释性强的风控策略（带冷静期 + 慢恢复）：
    - 触发风险：dd >= dd_high -> 进入 RISK
    - 退出风险：dd <= dd_low 连续 cooldown_steps 天 -> 回 NORMAL
    - RISK 状态：目标仓位被 cap 到 risk_cap，并按线性规则靠近 min_pos
    - NORMAL 状态：目标仓位逐步恢复到 base_pos（每天最多 +recover_step）
    """
    if state is None:
        state = {}

    position = float(obs[0])
    dd = float(obs[6]) if len(obs) >= 7 else float(obs[2])

    base_pos = float(np.clip(base_pos, 0.0, 1.0))
    min_pos = float(np.clip(min_pos, 0.0, 1.0))
    risk_cap = float(np.clip(risk_cap, 0.0, 1.0))
    recover_step = float(max(recover_step, 0.0))

    dd_low = float(max(dd_low, 0.0))
    dd_high = float(max(dd_high, dd_low + 1e-9))

    mode = state.get("mode", "NORMAL")  # "NORMAL" or "RISK"
    cool = int(state.get("cool", 0))
    target_pos_smooth = float(state.get("target_pos_smooth", base_pos))

    # ---------- 状态转移 ----------
    if dd >= dd_high:
        mode = "RISK"
        cool = 0  # 重置冷却计数

    if mode == "RISK":
        # 满足低风险条件则累积冷却天数
        if dd <= dd_low:
            cool += 1
        else:
            cool = 0

        # 冷却满足 -> 退出风险
        if cool >= cooldown_steps:
            mode = "NORMAL"
            cool = 0

    # ---------- 计算“基础目标仓位”（不含 smooth） ----------
    # 线性降仓：dd_low -> base_pos, dd_high -> min_pos
    if dd <= dd_low:
        target_base = base_pos
    elif dd >= dd_high:
        target_base = min_pos
    else:
        t = (dd - dd_low) / (dd_high - dd_low)
        target_base = base_pos + t * (min_pos - base_pos)

    if mode == "RISK":
        # 风险模式：压住上限（防止反弹期过快加仓）
        target = min(target_base, risk_cap)
        # 风险模式下可以允许更快减仓：直接用 target（不做慢恢复）
        target_pos_smooth = target
    else:
        # 正常模式：慢恢复（只限制加仓斜率，不限制减仓）
        if target_base > target_pos_smooth:
            target_pos_smooth = min(target_pos_smooth + recover_step, target_base)
        else:
            target_pos_smooth = target_base

        target = target_pos_smooth

    # 保存状态
    state["mode"] = mode
    state["cool"] = cool
    state["target_pos_smooth"] = float(target_pos_smooth)

    # delta action
    action_delta = _delta_from_target(position, target, max_step_change)
    return np.array([action_delta], dtype=np.float32)
