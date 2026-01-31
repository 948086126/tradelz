# core/predict_next_day.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict
import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from core.trade.twap_executor import SimpleTWAPExecutor
from core.trade.email_notifier import EmailNotifier
from core.trade.position_manager import PositionManager

ACTION_TO_POS = {0: 0.0, 1: 0.3, 2: 0.6, 3: 1.0}


def predict_next_day(
        symbol: str = "601138",
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        send_email: bool = True
) -> Optional[Dict]:
    """
    预测明日持仓（供 Celery 调用）

    Args:
        symbol: 股票代码
        model_path: 模型路径（可选，默认使用标准路径）
        data_path: 数据路径（可选，默认使用标准路径）
        send_email: 是否发送邮件

    Returns:
        dict: {
            'symbol': 股票代码,
            'date': 预测日期,
            'current_position': 当前持仓,
            'recommended_position': 推荐持仓,
            'action': 动作,
            'confidence': 置信度,
            'plan': TWAP计划
        }
    """
    try:
        root = Path(__file__).resolve().parents[1]

        # 使用默认路径
        if data_path is None:
            data_path = root / "data" / "processed" / f"{symbol}_features.csv"
        else:
            data_path = Path(data_path)

        if model_path is None:
            model_path = root / "models" / f"ppo_{symbol}.zip"
            meta_path = root / "models" / f"ppo_{symbol}_meta.json"
        else:
            model_path = Path(model_path)
            meta_path = model_path.parent / f"{model_path.stem}_meta.json"

        # 加载数据
        df = pd.read_csv(data_path, parse_dates=["date"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols: List[str] = meta["feature_cols"]

        # 加载模型
        model = PPO.load(str(model_path))

        # 获取最新数据
        last = df.iloc[-1]
        obs = last[feature_cols].to_numpy(dtype=np.float32)

        # 预测
        action, _ = model.predict(obs, deterministic=True)
        target_pos = float(ACTION_TO_POS[int(action)])

        # 获取当前持仓
        pos_manager = PositionManager()
        current_position = pos_manager.get_position(symbol)

        # 生成 TWAP 计划
        executor = SimpleTWAPExecutor()
        plan = executor.generate_plan(
            date=last['date'] + pd.Timedelta(days=1),
            symbol=symbol,
            current_position=current_position,
            target_position=target_pos,
        )

        # 构建返回结果
        result = {
            'symbol': symbol,
            'date': str(last['date'].date()),
            'current_position': current_position,
            'recommended_position': target_pos,
            'action': int(action),
            'confidence': abs(target_pos - current_position),  # 简单的置信度计算
            'plan': plan,
            'total_delta': plan.total_delta
        }

        # 打印结果
        print("=" * 60)
        print(f"Latest date: {last['date'].date()}")
        print(f"Predicted action: {int(action)}")
        print(f"Current position: {current_position:.2%}")
        print(f"Target position: {target_pos:.2%}")
        print(f"Position change: {plan.total_delta:+.2%}")
        print("=" * 60)

        # 发送邮件
        if send_email:
            receiver_email = os.getenv("RECEIVER_EMAIL")
            if receiver_email:
                try:
                    notifier = EmailNotifier()
                    notifier.send_daily_plan(plan, receiver_email)
                    print(f"✅ 邮件已发送到: {receiver_email}")
                except Exception as e:
                    print(f"❌ 邮件发送失败: {e}")

        # 保存信号文件
        out_dir = root / "data" / "signals"
        out_dir.mkdir(parents=True, exist_ok=True)

        signal = {
            "symbol": symbol,
            "signal_date": str(last["date"].date()),
            "current_position": current_position,
            "target_position": target_pos,
            "action": int(action),
            "twap_plan": {
                "open": plan.signals[0].absolute_ratio if len(plan.signals) > 0 else 0.0,
                "mid": plan.signals[1].absolute_ratio if len(plan.signals) > 1 else 0.0,
                "close": plan.signals[2].absolute_ratio if len(plan.signals) > 2 else 0.0,
            }
        }

        out_path = out_dir / "next_day_signal.json"
        out_path.write_text(
            json.dumps(signal, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ 信号已保存: {out_path}")

        return result

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """命令行入口（保留原版逻辑）"""
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "601138_features.csv"
    meta_path = root / "models" / "ppo_601138_meta.json"
    model_path = root / "models" / "ppo_601138.zip"

    df = pd.read_csv(data_path, parse_dates=["date"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols: List[str] = meta["feature_cols"]

    model = PPO.load(str(model_path))

    last = df.iloc[-1]
    obs = last[feature_cols].to_numpy(dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)
    target_pos = float(ACTION_TO_POS[int(action)])

    print(f"Latest date: {last['date'].date()}")
    print(f"Predicted action: {int(action)}")
    print(f"Target position for next day: {target_pos * 100:.0f}%")

    # ===== 获取当前持仓 =====
    pos_manager = PositionManager()
    symbol = "601138"
    current_position = pos_manager.get_position(symbol)

    print(f"当前持仓: {current_position:.2%}")

    # ===== 生成 TWAP 计划 =====
    executor = SimpleTWAPExecutor()

    plan = executor.generate_plan(
        date=last['date'] + pd.Timedelta(days=1),
        symbol=symbol,
        current_position=current_position,
        target_position=target_pos,
    )

    # ===== 打印计划 =====
    print("=" * 60)
    print("📋 交易计划")
    print("=" * 60)
    print(f"当前仓位: {plan.current_position:.2%}")
    print(f"目标仓位: {plan.target_position:.2%}")
    print(f"需要调仓: {plan.total_delta:+.2%}")
    print()

    for sig in plan.signals:
        print(f"[{sig.time_window}]")
        print(f"  操作: {sig.action}")
        print(f"  交易量: {sig.absolute_ratio:.2%}")
        print(f"  说明: {sig.reason}")
        print()

    # ===== 发送邮件 =====
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if receiver_email:
        try:
            notifier = EmailNotifier()
            notifier.send_daily_plan(plan, receiver_email)
        except Exception as e:
            print(f"❌ 邮件发送异常: {e}")
            import traceback
            traceback.print_exc()

    # ===== 保存信号文件 =====
    out_dir = root / "data" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    signal = {
        "symbol": symbol,
        "signal_date": str(last["date"].date()),
        "current_position": current_position,
        "target_position": target_pos,
        "action": int(action),
        "twap_plan": {
            "open": plan.signals[0].absolute_ratio if len(plan.signals) > 0 else 0.0,
            "mid": plan.signals[1].absolute_ratio if len(plan.signals) > 1 else 0.0,
            "close": plan.signals[2].absolute_ratio if len(plan.signals) > 2 else 0.0,
        }
    }

    out_path = out_dir / "next_day_signal.json"
    out_path.write_text(json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved signal -> {out_path}")


if __name__ == "__main__":
    main()
