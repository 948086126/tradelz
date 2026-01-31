import pandas as pd
import numpy as np


class TradeExecutor:
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        commission_rate: float = 0.001,  # 千分之一
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate

        # 账户状态
        self.cash = initial_cash
        self.shares = 0.0

        # 记录
        self.records = []

    def reset(self):
        self.cash = self.initial_cash
        self.shares = 0.0
        self.records = []

    def step(
        self,
        date,
        open_price: float,
        close_price: float,
        target_position: float,
    ):
        """
        在开盘价成交，将仓位调整到 target_position
        """

        # 当前总资产（按开盘价估值）
        equity = self.cash + self.shares * open_price

        # 目标持仓市值
        target_value = equity * np.clip(target_position, 0.0, 1.0)

        # 当前持仓市值
        current_value = self.shares * open_price

        # 需要调整的市值
        delta_value = target_value - current_value

        # === 买入 ===
        if delta_value > 0:
            buy_value = min(delta_value, self.cash)
            commission = buy_value * self.commission_rate
            actual_buy = buy_value - commission

            buy_shares = actual_buy / open_price
            self.shares += buy_shares
            self.cash -= buy_value

        # === 卖出 ===
        elif delta_value < 0:
            sell_value = min(-delta_value, current_value)
            sell_shares = sell_value / open_price
            commission = sell_value * self.commission_rate
            actual_sell = sell_value - commission

            self.shares -= sell_shares
            self.cash += actual_sell

        # 收盘后资产
        equity_close = self.cash + self.shares * close_price
        position = (self.shares * close_price) / equity_close if equity_close > 0 else 0

        self.records.append(
            {
                "date": date,
                "cash": self.cash,
                "shares": self.shares,
                "equity": equity_close,
                "position": position,
                "target_position": target_position,
                "open_price": open_price,
                "close_price": close_price,
            }
        )

    def run_backtest(
        self,
        df: pd.DataFrame,
        target_position_col: str = "target_position",
    ) -> pd.DataFrame:
        """
        df 必须包含：
        date, open, close, target_position
        """

        self.reset()

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]

            self.step(
                date=next_row["date"],
                open_price=next_row["open"],
                close_price=next_row["close"],
                target_position=row[target_position_col],
            )

        result = pd.DataFrame(self.records)
        result["return"] = result["equity"].pct_change().fillna(0)
        result["cum_return"] = (1 + result["return"]).cumprod()

        return result
