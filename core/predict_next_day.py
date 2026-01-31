from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ACTION_TO_POS = {0: 0.0, 1: 0.3, 2: 0.6, 3: 1.0}


def main():
    root = Path(__file__).resolve().parents[1]  # tradeLz/
    data_path = root / "data" / "processed" / "601138_features.csv"
    meta_path = root / "models" / "ppo_601138_meta.json"
    model_path = root / "models" / "ppo_601138.zip"

    df = pd.read_csv(data_path, parse_dates=["date"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols: List[str] = meta["feature_cols"]

    model = PPO.load(str(model_path))

    # 用最新一行特征预测“下一交易日仓位建议”
    last = df.iloc[-1]
    obs = last[feature_cols].to_numpy(dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)
    target_pos = float(ACTION_TO_POS[int(action)])

    print(f"Latest date: {last['date'].date()}")
    print(f"Predicted action: {int(action)}")
    print(f"Target position for next day: {target_pos * 100:.0f}%")

    # 可选：写出一个信号文件（后面做邮件/页面会用到）
    out_dir = root / "data" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    signal = {
        "symbol": "601138",
        "signal_date": str(last["date"].date()),
        "target_position": target_pos,
        "action": int(action),
    }
    out_path = out_dir / "next_day_signal.json"
    out_path.write_text(json.dumps(signal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved signal -> {out_path}")


if __name__ == "__main__":
    main()
