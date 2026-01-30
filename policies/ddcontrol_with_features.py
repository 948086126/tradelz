import numpy as np


def _piecewise_max_pos(dd, dd_points, pos_points):
    dd = float(max(dd, 0.0))

    if dd <= dd_points[0]:
        return float(pos_points[0])
    if dd >= dd_points[-1]:
        return float(pos_points[-1])

    for i in range(len(dd_points) - 1):
        d0, d1 = float(dd_points[i]), float(dd_points[i + 1])
        if d0 <= dd <= d1:
            p0, p1 = float(pos_points[i]), float(pos_points[i + 1])
            t = (dd - d0) / (d1 - d0 + 1e-12)
            return float(p0 + t * (p1 - p0))

    return float(pos_points[-1])


def _delta_from_target(position, target_position, max_step_change):
    target_position = float(np.clip(target_position, 0.0, 1.0))
    position = float(np.clip(position, 0.0, 1.0))
    need = target_position - position
    action_delta = need / float(max_step_change)
    return float(np.clip(action_delta, -1.0, 1.0))


def ddcontrol_with_features(
    obs,
    *,
    max_step_change,

    base_pos=0.6,
    min_pos=0.1,

    vol_low=0.02,
    vol_high=0.05,     # 放宽一点
    vol_cut=0.75,      # 折扣别太狠（原来 0.6）

    ma_band=0.01,
    block_add_when_below_ma=True,

    recover_step=0.02,        # 加快恢复
    min_rebalance_gap=0.005,  # 更容易触发调仓

    dd_slope_k=3.0,           # 先别太敏感（原来 10）
    crash_ret=-0.03,
    cooldown_steps=20,

    state=None,
    **kwargs
):
    """
    obs 约定（9维）：
    [position, ret_1, ret_5, ret_20, vol_20, price_vs_ma20, drawdown, dd_increase, equity_log]
    """

    if state is None:
        state = {}

    position = float(obs[0])
    ret_1 = float(obs[1])
    vol_20 = float(obs[4])
    price_vs_ma20 = float(obs[5])
    dd = float(obs[6])
    dd_increase = float(obs[7])

    base_pos = float(np.clip(base_pos, 0.0, 1.0))
    min_pos = float(np.clip(min_pos, 0.0, 1.0))

    # ========= 0) 冷却机制（暴跌/回撤突变/波动爆炸） =========
    cooldown_left = int(state.get("cooldown_left", 0))
    if cooldown_left > 0:
        state["cooldown_left"] = cooldown_left - 1
        target = min_pos
        return np.array([_delta_from_target(position, target, max_step_change)], dtype=np.float32)

    if (ret_1 <= crash_ret) or (dd_increase > 0.015) or (vol_20 >= 0.08):
        state["cooldown_left"] = int(cooldown_steps)
        target = min_pos
        return np.array([_delta_from_target(position, target, max_step_change)], dtype=np.float32)

    # ========= 1) 回撤 -> 允许最大仓位（路线A） =========
    dd_points = kwargs.get("dd_points")
    pos_points = kwargs.get("pos_points")

    if dd_points is None or pos_points is None:
        dd_points = [0.02, 0.06, 0.10, 0.18]
        pos_points = [0.8,  0.6,  0.35, 0.1]

    dd_max_pos = _piecewise_max_pos(dd, dd_points, pos_points)
    dd_max_pos = float(np.clip(dd_max_pos, min_pos, 1.0))

    # ========= 2) 波动闸门：只在“明显高波动”时打折 =========
    vol_low = float(max(vol_low, 0.0))
    vol_high = float(max(vol_high, vol_low + 1e-9))
    vol_cut = float(np.clip(vol_cut, 0.0, 1.0))

    if vol_20 <= vol_low:
        vol_factor = 1.0
    elif vol_20 >= vol_high:
        vol_factor = vol_cut
    else:
        # 中间缓慢折扣（更温和）
        u = (vol_20 - vol_low) / (vol_high - vol_low)
        vol_factor = 1.0 + u * (vol_cut - 1.0)

    target = dd_max_pos * vol_factor
    target = float(np.clip(target, min_pos, dd_max_pos))

    # ========= 3) 回撤斜率刹车（轻一点） =========
    if dd_increase > 0.0:
        slope_factor = float(np.exp(-dd_slope_k * dd_increase))
        target *= slope_factor
        target = float(np.clip(target, min_pos, dd_max_pos))

    # ========= 4) 趋势闸门（避免在 MA 下方加仓） =========
    if block_add_when_below_ma:
        allow_add = bool(state.get("allow_add", True))
        if price_vs_ma20 < -ma_band:
            allow_add = False
        elif price_vs_ma20 > ma_band:
            allow_add = True
        state["allow_add"] = allow_add
        if not allow_add:
            target = min(target, position)

    # ========= 5) 慢恢复（只限制“加仓速度”） =========
    target_smooth = float(state.get("target_smooth", position))
    if target > target_smooth:
        target_smooth = min(target_smooth + recover_step, target)
    else:
        target_smooth = target

    # ========= 6) 最小调仓阈值 =========
    if abs(target_smooth - position) < min_rebalance_gap:
        action_delta = 0.0
    else:
        action_delta = _delta_from_target(position, target_smooth, max_step_change)

    state["target_smooth"] = float(target_smooth)
    return np.array([action_delta], dtype=np.float32)
