# core/risk_cap.py

def risk_cap_from_obs(
    obs,
    *,
    dd_points=None,
    pos_points=None,
    min_pos=0.1,
):
    """
    只做一件事：根据风险状态，返回允许的最大仓位
    """
    dd = float(obs[6])

    if dd_points is None:
        dd_points = [0.02, 0.06, 0.10, 0.18]
    if pos_points is None:
        pos_points = [0.8, 0.6, 0.35, 0.1]

    # 分段线性
    if dd <= dd_points[0]:
        cap = pos_points[0]
    elif dd >= dd_points[-1]:
        cap = pos_points[-1]
    else:
        cap = pos_points[-1]
        for i in range(len(dd_points) - 1):
            d0, d1 = dd_points[i], dd_points[i + 1]
            if d0 <= dd <= d1:
                p0, p1 = pos_points[i], pos_points[i + 1]
                t = (dd - d0) / (d1 - d0 + 1e-12)
                cap = p0 + t * (p1 - p0)
                break

    return max(cap, min_pos)
