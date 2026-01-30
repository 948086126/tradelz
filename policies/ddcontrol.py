import numpy as np
from .ddcontrol_with_features import ddcontrol_with_features

def ddcontrol(obs: np.ndarray, *, max_step_change: float, state=None, **kwargs) -> np.ndarray:
    """
    对外统一入口：run_backtest / future PPO-shield 都只调用这个函数
    kwargs 会透传给实现层，便于你在 run_backtest 里调参数
    """
    return ddcontrol_with_features(
        obs,
        max_step_change=max_step_change,
        state=state,
        **kwargs
    )
