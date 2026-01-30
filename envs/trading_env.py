# envs/trading_env.py
import os
import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces


ENV_VERSION = "2026-01-29.v5"  # ✅ 用它确认运行的就是这份 env


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        cost_rate: float = 0.001,

        # ===== 动作控制 =====
        max_step_change: float = 0.10,

        # ===== 风控惩罚 =====
        ddi_penalty: float = 2.0,
        turnover_penalty: float = 0.0005,
        hold_a: float = 0.0006,
        hold_b: float = 0.0006,

        # ===== Action Shield =====
        use_action_shield: bool = True,
        dd_points=None,
        pos_points=None,

        # ===== 训练：随机初始仓位 =====
        train_random_start_pos: bool = False,
        random_pos_prob: float = 0.5,
        random_pos_low: float = 0.1,
        random_pos_high: float = 0.6,

        # ===== shaping：回撤/交易稳定性 =====
        ddi_free: float = 0.01,
        smooth_penalty: float = 0.003,
        cap_penalty: float = 0.5,

        # ===== “老投资者”风格：不预测未来，只是趋势状态下的参与激励 =====
        opp_cost: float = 0.02,
        trend_bonus: float = 0.01,

        # ✅ 空仓惩罚（弱信号很小，强信号更大）
        idle_base: float = 0.00005,
        idle_signal: float = 0.00060,

        # ✅ 参与缺口惩罚（关键：打破 0 仓吸引态）
        part_penalty: float = 0.06,
        part_power: float = 1.5,
        min_participation: float = 0.08,

        # ✅ desired_pos 平滑（避免目标跳变导致不可达惩罚）
        desired_ema_alpha: float = 0.35,  # 0~1，越大越跟随当下，越小越平滑

        # 信号函数形状
        signal_k_ret5: float = 6.0,
        signal_k_ma: float = 3.0,

        # ===== 日志 =====
        log_dir: str = "logs",
        log_name: str = "behavior.csv",
        print_first_n: int = 0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        if "close" not in self.df.columns:
            raise ValueError("df 必须包含 'close' 列")

        # params
        self.cost_rate = float(cost_rate)
        self.max_step_change = float(max_step_change)

        self.ddi_penalty = float(ddi_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.hold_a = float(hold_a)
        self.hold_b = float(hold_b)

        self.use_action_shield = bool(use_action_shield)
        self.dd_points = dd_points if dd_points is not None else [0.02, 0.06, 0.10, 0.18]
        self.pos_points = pos_points if pos_points is not None else [0.8, 0.6, 0.35, 0.1]

        self.train_random_start_pos = bool(train_random_start_pos)
        self.random_pos_prob = float(random_pos_prob)
        self.random_pos_low = float(random_pos_low)
        self.random_pos_high = float(random_pos_high)

        self.ddi_free = float(ddi_free)
        self.smooth_penalty = float(smooth_penalty)
        self.cap_penalty = float(cap_penalty)

        self.opp_cost = float(opp_cost)
        self.trend_bonus = float(trend_bonus)

        self.idle_base = float(idle_base)
        self.idle_signal = float(idle_signal)

        self.part_penalty = float(part_penalty)
        self.part_power = float(part_power)
        self.min_participation = float(min_participation)

        self.desired_ema_alpha = float(np.clip(desired_ema_alpha, 0.0, 1.0))

        self.signal_k_ret5 = float(signal_k_ret5)
        self.signal_k_ma = float(signal_k_ma)

        self.log_dir = log_dir
        self.log_name = log_name
        self.print_first_n = int(print_first_n)

        # Action: [-1,1] -> position delta ratio
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Obs 9维固定顺序：
        # [position, ret_1, ret_5, ret_20, vol_20, price_vs_ma20, drawdown, dd_increase, equity_log]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # runtime log
        self._csv_file = None
        self._csv_writer = None

        # runtime state
        self.current_step = 0
        self.position = 0.0
        self.locked_position = 0.0
        self.today_bought = 0.0
        self.equity = 1.0
        self.max_equity = 1.0
        self.prev_drawdown = 0.0

        # ✅ desired_pos 平滑状态（不泄露未来）
        self._desired_smooth = 0.0

        print(f"[TradingEnv] loaded env version = {ENV_VERSION}")

    # -----------------------
    # logging
    # -----------------------
    def _open_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, self.log_name)

        # ✅ 追加写入，避免 VecEnv auto-reset 覆盖
        file_exists = os.path.exists(path)
        file_empty = (not file_exists) or (os.path.getsize(path) == 0)

        self._csv_file = open(path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        # 只有空文件才写表头
        if file_empty:
            self._csv_writer.writerow([
                "step", "close", "price_return", "action_delta", "raw_target", "risk_cap", "shielded_target",
                "target_position", "prev_position", "position", "executed_trade", "locked_position",
                "sellable_position",
                "limit_up", "limit_down", "pnl", "cost", "equity", "max_equity", "drawdown", "dd_increase",
                "signal", "desired_pos_raw", "desired_pos", "part_gap", "part_pen", "reward"
            ])

    def _close_logger(self):
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file = None
        self._csv_writer = None

    # -----------------------
    # Gym API
    # -----------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        self.position = 0.0
        if self.train_random_start_pos and (np.random.rand() < self.random_pos_prob):
            self.position = float(np.random.uniform(self.random_pos_low, self.random_pos_high))

        self.locked_position = 0.0
        self.today_bought = 0.0

        self.equity = 1.0
        self.max_equity = 1.0
        self.prev_drawdown = 0.0

        self._desired_smooth = float(self.position)

        self._close_logger()
        self._open_logger()

        return self._get_obs(), {}

    def _get_limits(self, price_return: float):
        if "limit_up" in self.df.columns and "limit_down" in self.df.columns:
            limit_up = bool(self.df.iloc[self.current_step]["limit_up"])
            limit_down = bool(self.df.iloc[self.current_step]["limit_down"])
            return limit_up, limit_down
        return (price_return >= 0.10), (price_return <= -0.10)

    def _risk_cap(self, obs):
        dd = float(obs[6])
        if not np.isfinite(dd):
            dd = 0.0
        dd = max(dd, 0.0)

        dd_points = self.dd_points
        pos_points = self.pos_points
        if len(dd_points) != len(pos_points):
            raise ValueError("dd_points 与 pos_points 必须等长")

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

        return float(np.clip(cap, 0.0, 1.0))

    def _get_obs(self):
        # ✅ 强制有限，彻底避免 dd=nan
        if not np.isfinite(self.equity) or self.equity <= 0:
            self.equity = 1e-6
        if not np.isfinite(self.max_equity) or self.max_equity <= 0:
            self.max_equity = self.equity

        self.max_equity = max(float(self.max_equity), float(self.equity))
        drawdown = (self.max_equity - self.equity) / max(self.max_equity, 1e-12)
        if not np.isfinite(drawdown):
            drawdown = 0.0
        drawdown = float(np.clip(drawdown, 0.0, 1.0))

        equity_log = float(np.log(max(self.equity, 1e-12)))

        idx = self.current_step

        def _get(col, default=0.0):
            if col in self.df.columns and idx < len(self.df):
                v = self.df.iloc[idx][col]
                try:
                    v = float(v)
                except Exception:
                    return float(default)
                if not np.isfinite(v):
                    return float(default)
                return float(v)
            return float(default)

        ret_1 = _get("ret_1", 0.0)
        ret_5 = _get("ret_5", 0.0)
        ret_20 = _get("ret_20", 0.0)
        vol_20 = _get("vol_20", 0.0)
        p_vs_ma = _get("price_vs_ma20", 0.0)

        prev_dd = float(self.prev_drawdown) if np.isfinite(self.prev_drawdown) else 0.0
        dd_inc_obs = max(drawdown - prev_dd, 0.0)

        return np.array(
            [
                float(self.position),
                float(ret_1),
                float(ret_5),
                float(ret_20),
                float(vol_20),
                float(p_vs_ma),
                float(drawdown),
                float(dd_inc_obs),
                float(equity_log),
            ],
            dtype=np.float32
        )

    def _signal_pos(self, obs_before) -> float:
        ret_5 = float(obs_before[2])
        p_vs_ma20 = float(obs_before[5])
        s = 0.5 * (np.tanh(self.signal_k_ret5 * ret_5) + np.tanh(self.signal_k_ma * p_vs_ma20))  # [-1,1]
        return float(max(s, 0.0))  # [0,1]

    def step(self, action):
        # 1) action -> raw target
        a = float(action[0])
        if not np.isfinite(a):
            a = 0.0
        a = float(np.clip(a, -1.0, 1.0))

        raw_target = float(np.clip(self.position + a * self.max_step_change, 0.0, 1.0))
        obs_before = self._get_obs()

        # 2) shield
        risk_cap = 1.0
        shielded_target = raw_target
        if self.use_action_shield:
            risk_cap = self._risk_cap(obs_before)
            shielded_target = min(raw_target, risk_cap)

        target_position = float(shielded_target)
        prev_position = float(self.position)

        prev_price = float(self.df.iloc[self.current_step]["close"])
        if (not np.isfinite(prev_price)) or (prev_price <= 0):
            prev_price = 1.0

        # T+1 lock
        self.locked_position = float(self.today_bought)
        self.today_bought = 0.0

        # advance time
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        price = float(self.df.iloc[self.current_step]["close"])
        if (not np.isfinite(price)) or (price <= 0):
            price = prev_price

        price_return = (price - prev_price) / max(prev_price, 1e-12)
        if not np.isfinite(price_return):
            price_return = 0.0

        limit_up, limit_down = self._get_limits(price_return)
        sellable_position = max(self.position - self.locked_position, 0.0)

        # execute (T+1 sell lock)
        executed_trade = 0.0
        if target_position < self.position:
            if not limit_down:
                desired_sell = self.position - target_position
                actual_sell = min(desired_sell, sellable_position)
                self.position -= actual_sell
                executed_trade = -actual_sell
        elif target_position > self.position:
            if not limit_up:
                buy_amount = target_position - self.position
                self.position += buy_amount
                self.today_bought += buy_amount
                executed_trade = buy_amount

        trade_amount = float(abs(executed_trade))
        cost = trade_amount * self.cost_rate

        # 用 prev_position 吃当步收益，避免同bar调仓吃收益
        pnl = prev_position * price_return

        self.equity *= (1.0 + pnl - cost)
        if (not np.isfinite(self.equity)) or self.equity <= 0:
            self.equity = 1e-6

        self.max_equity = max(float(self.max_equity), float(self.equity))
        if (not np.isfinite(self.max_equity)) or self.max_equity <= 0:
            self.max_equity = self.equity

        drawdown = (self.max_equity - self.equity) / max(self.max_equity, 1e-12)
        if not np.isfinite(drawdown):
            drawdown = 0.0
        drawdown = float(np.clip(drawdown, 0.0, 1.0))

        prev_dd = float(self.prev_drawdown) if np.isfinite(self.prev_drawdown) else 0.0
        dd_increase = max(drawdown - prev_dd, 0.0)
        self.prev_drawdown = float(drawdown)

        # ===== reward components =====
        ddi_excess = max(dd_increase - self.ddi_free, 0.0)
        dd_pen = self.ddi_penalty * ddi_excess

        turnover_pen = self.turnover_penalty * trade_amount
        smooth_pen = self.smooth_penalty * abs(self.position - prev_position)

        cap_hit = max(raw_target - shielded_target, 0.0)
        cap_hit_pen = self.cap_penalty * cap_hit

        signal_pos = self._signal_pos(obs_before)

        # 机会成本 / 趋势奖励
        opp_pen = self.opp_cost * (1.0 - self.position) * signal_pos
        trend_b = self.trend_bonus * self.position * signal_pos

        # ✅ desired_pos（原始）
        desired_raw = float(np.clip(risk_cap * signal_pos, 0.0, risk_cap))
        if signal_pos > 0.7:
            desired_raw = max(desired_raw, min(self.min_participation, risk_cap))

        # ✅ 可达化：desired 不能比上一时刻提升超过 max_step_change
        desired_step = min(desired_raw, self._desired_smooth + self.max_step_change)

        # ✅ EMA 平滑（防止目标跳变）
        alpha = self.desired_ema_alpha
        desired_pos = alpha * desired_step + (1.0 - alpha) * self._desired_smooth
        desired_pos = float(np.clip(desired_pos, 0.0, risk_cap))
        self._desired_smooth = desired_pos

        # ✅ 关键修复：用“当步决策目标 target_position”算 gap，避免不可达惩罚
        part_gap = max(desired_pos - target_position, 0.0)
        part_pen = self.part_penalty * (part_gap ** self.part_power)

        # 空仓惩罚：趋势越强越痛
        idle_pen = 0.0
        if self.position < 1e-6:
            idle_pen = self.idle_base + self.idle_signal * signal_pos

        hold_term = (+ self.hold_a * self.position) - (self.hold_b * (self.position ** 2))

        reward = (
            pnl
            - cost
            - dd_pen
            - turnover_pen
            - smooth_pen
            - cap_hit_pen
            - opp_pen
            - idle_pen
            - part_pen
            + trend_b
            + hold_term
        )
        reward = float(reward)

        obs = self._get_obs()

        info = {
            "position": float(self.position),
            "raw_target": float(raw_target),
            "risk_cap": float(risk_cap),
            "shielded_target": float(shielded_target),
            "target_position": float(target_position),
            "executed_trade": float(executed_trade),
            "locked_position": float(self.locked_position),
            "sellable_position": float(sellable_position),
            "price_return": float(price_return),
            "limit_up": bool(limit_up),
            "limit_down": bool(limit_down),
            "pnl": float(pnl),
            "cost": float(cost),
            "equity": float(self.equity),
            "drawdown": float(drawdown),
            "dd_increase": float(dd_increase),
            "signal": float(signal_pos),
            "desired_pos_raw": float(desired_raw),
            "desired_pos": float(desired_pos),
            "part_gap": float(part_gap),
            "part_pen": float(part_pen),
        }

        if self._csv_writer is not None:
            self._csv_writer.writerow([
                self.current_step,
                price,
                price_return,
                a,
                raw_target,
                risk_cap,
                shielded_target,
                target_position,
                prev_position,
                self.position,
                executed_trade,
                self.locked_position,
                sellable_position,
                int(limit_up),
                int(limit_down),
                pnl,
                cost,
                self.equity,
                self.max_equity,
                drawdown,
                dd_increase,
                signal_pos,
                desired_raw,
                desired_pos,
                part_gap,
                part_pen,
                reward,
            ])

        if self.print_first_n > 0 and self.current_step <= self.print_first_n:
            print(
                f"step={self.current_step:4d} | "
                f"ret={price_return:+.4f} | "
                f"act={a:+.2f} | "
                f"cap={risk_cap:.2f} | "
                f"raw={raw_target:.2f} | "
                f"sh={shielded_target:.2f} | "
                f"pos={self.position:.2f} | "
                f"sig={signal_pos:.2f} | "
                f"des={desired_pos:.2f} | "
                f"gap={part_gap:.2f} | "
                f"eq={self.equity:.3f} | "
                f"dd={drawdown:.3f} | "
                f"ddi={dd_increase:.4f} | "
                f"reward={reward:+.6f}"
            )

        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self):
        self._close_logger()
        super().close()
