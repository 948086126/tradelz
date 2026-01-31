# core/trade/twap_executor.py

from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np


@dataclass
class TWAPSignal:
    """
    单次交易信号（用于邮件通知）
    """
    time_window: str  # "开盘20分钟" / "盘中60分钟" / "尾盘20分钟"
    action: str  # BUY / SELL / HOLD
    trade_ratio: float  # 本次交易占总调仓量的比例（0-1）
    absolute_ratio: float  # 本次交易占总资产的比例（0-1）
    reason: str


@dataclass
class DailyTWAPPlan:
    """
    一天的完整交易计划
    """
    date: pd.Timestamp
    symbol: str
    current_position: float  # 当前仓位
    target_position: float  # 目标仓位
    total_delta: float  # 需要调仓的量

    signals: List[TWAPSignal]  # 3 个信号（开盘、盘中、尾盘）


class SimpleTWAPExecutor:
    """
    简化版 20/60/20 执行器

    核心逻辑：
    - 开盘 20 分钟：执行 20%
    - 盘中 60 分钟：执行 60%
    - 尾盘 20 分钟：执行 20%
    """

    OPEN_RATIO = 0.20  # 开盘执行 20%
    MID_RATIO = 0.60  # 盘中执行 60%
    CLOSE_RATIO = 0.20  # 尾盘执行 20%

    MIN_TRADE_DELTA = 0.05  # 最小调仓阈值（5%）

    def generate_plan(
            self,
            date: pd.Timestamp,
            symbol: str,
            current_position: float,
            target_position: float,
    ) -> DailyTWAPPlan:
        """
        生成一天的交易计划
        """
        current = float(np.clip(current_position, 0.0, 1.0))
        target = float(np.clip(target_position, 0.0, 1.0))
        delta = target - current

        signals = []

        # === 不交易 ===
        if abs(delta) < self.MIN_TRADE_DELTA:
            signals.append(TWAPSignal(
                time_window="全天",
                action="HOLD",
                trade_ratio=0.0,
                absolute_ratio=0.0,
                reason=f"调仓量 {delta:.2%} < 最小阈值 {self.MIN_TRADE_DELTA:.2%}"
            ))
            return DailyTWAPPlan(
                date=date,
                symbol=symbol,
                current_position=current,
                target_position=current,  # 不调仓
                total_delta=0.0,
                signals=signals,
            )

        # === 需要交易 ===
        action = "BUY" if delta > 0 else "SELL"
        abs_delta = abs(delta)

        # 开盘 20%
        open_amount = abs_delta * self.OPEN_RATIO
        signals.append(TWAPSignal(
            time_window="开盘20分钟（9:30-9:50）",
            action=action,
            trade_ratio=self.OPEN_RATIO,
            absolute_ratio=open_amount,
            reason=f"执行总调仓量的 {self.OPEN_RATIO:.0%}"
        ))

        # 盘中 60%
        mid_amount = abs_delta * self.MID_RATIO
        signals.append(TWAPSignal(
            time_window="盘中60分钟（10:00-14:00）",
            action=action,
            trade_ratio=self.MID_RATIO,
            absolute_ratio=mid_amount,
            reason=f"执行总调仓量的 {self.MID_RATIO:.0%}（分批执行）"
        ))

        # 尾盘 20%
        close_amount = abs_delta * self.CLOSE_RATIO
        signals.append(TWAPSignal(
            time_window="尾盘20分钟（14:40-15:00）",
            action=action,
            trade_ratio=self.CLOSE_RATIO,
            absolute_ratio=close_amount,
            reason=f"执行总调仓量的 {self.CLOSE_RATIO:.0%}（兜底）"
        ))

        return DailyTWAPPlan(
            date=date,
            symbol=symbol,
            current_position=current,
            target_position=target,
            total_delta=delta,
            signals=signals,
        )
