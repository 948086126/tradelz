# tradelz/core/trade/signal_interpreter.py

from dataclasses import dataclass
from typing import Optional
import pandas as pd


# ======================
# 数据结构
# ======================

@dataclass
class SignalInput:
    date: pd.Timestamp
    target_position: float      # PPO 输出 [0, 1]
    current_position: float     # 当前仓位 [0, 1]
    price: float                # 成交价（里程碑A：open）
    cash: float                 # 当前现金


@dataclass
class TradeSignal:
    date: pd.Timestamp
    action: str                 # BUY / SELL / HOLD
    trade_ratio: float          # 本次交易涉及的仓位比例（绝对值）
    target_position: float
    reason: str                 # 用于日志和调试


# ======================
# 核心解释器
# ======================

class SignalInterpreter:
    """
    将 PPO 输出的 target_position
    转换为可执行的交易信号
    """

    MIN_TRADE_DELTA = 0.05   # 最小调仓阈值（5%）

    def interpret(self, inp: SignalInput) -> TradeSignal:
        target = float(max(0.0, min(1.0, inp.target_position)))
        current = float(max(0.0, min(1.0, inp.current_position)))

        delta = target - current

        # === 不交易 ===
        if abs(delta) < self.MIN_TRADE_DELTA:
            return TradeSignal(
                date=inp.date,
                action="HOLD",
                trade_ratio=0.0,
                target_position=current,
                reason=f"delta {delta:.3f} < MIN_TRADE_DELTA"
            )

        # === 买入 ===
        if delta > 0:
            return TradeSignal(
                date=inp.date,
                action="BUY",
                trade_ratio=abs(delta),
                target_position=target,
                reason=f"increase position {current:.2f} → {target:.2f}"
            )

        # === 卖出 ===
        return TradeSignal(
            date=inp.date,
            action="SELL",
            trade_ratio=abs(delta),
            target_position=target,
            reason=f"decrease position {current:.2f} → {target:.2f}"
        )
