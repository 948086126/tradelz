# core/trade/position_manager.py

import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime


class PositionManager:
    """
    持仓管理器：记录和更新当前持仓

    功能：
    1. 获取当前持仓
    2. 更新持仓
    3. 记录历史变更
    4. 查询历史记录
    """

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            root = Path(__file__).resolve().parents[2]
            data_dir = root / "data" / "positions"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_position(self, symbol: str) -> float:
        """
        获取当前持仓

        Returns:
            float: 持仓比例（0-1）
        """
        pos_file = self.data_dir / f"{symbol}_position.json"

        if not pos_file.exists():
            print(f"[INFO] {symbol} 持仓文件不存在，返回默认值 0.0")
            return 0.0

        try:
            data = json.loads(pos_file.read_text(encoding="utf-8"))
            position = float(data.get("position", 0.0))
            print(f"[INFO] {symbol} 当前持仓: {position:.2%}")
            return position
        except Exception as e:
            print(f"[ERROR] 读取持仓文件失败: {e}")
            return 0.0

    def get_position_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取完整的持仓信息

        Returns:
            dict: {
                "symbol": str,
                "position": float,
                "update_date": str,
                "reason": str,
            }
        """
        pos_file = self.data_dir / f"{symbol}_position.json"

        if not pos_file.exists():
            return {
                "symbol": symbol,
                "position": 0.0,
                "update_date": None,
                "reason": "初始化",
            }

        try:
            data = json.loads(pos_file.read_text(encoding="utf-8"))
            return data
        except Exception as e:
            print(f"[ERROR] 读取持仓信息失败: {e}")
            return {
                "symbol": symbol,
                "position": 0.0,
                "update_date": None,
                "reason": "读取失败",
            }

    def update_position(
            self,
            symbol: str,
            position: float,
            date: Optional[pd.Timestamp] = None,
            reason: str = "",
    ) -> None:
        """
        更新持仓

        Args:
            symbol: 股票代码
            position: 新持仓比例（0-1）
            date: 更新日期（默认当前日期）
            reason: 更新原因
        """
        if date is None:
            date = pd.Timestamp.now()

        # 限制范围
        position = float(max(0.0, min(1.0, position)))

        pos_file = self.data_dir / f"{symbol}_position.json"

        # 获取旧持仓（用于历史记录）
        old_position = self.get_position(symbol)

        # 保存新持仓
        data = {
            "symbol": symbol,
            "position": position,
            "update_date": str(date.date()),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
        }

        pos_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"[OK] 持仓已更新: {symbol} {old_position:.2%} → {position:.2%}")

        # 自动添加历史记录
        if abs(position - old_position) > 1e-6:
            self.add_history(
                symbol=symbol,
                date=date,
                old_position=old_position,
                new_position=position,
                reason=reason,
            )

    def get_history(self, symbol: str, limit: Optional[int] = None) -> list:
        """
        获取历史持仓记录

        Args:
            symbol: 股票代码
            limit: 返回最近 N 条记录（None 表示全部）

        Returns:
            list: 历史记录列表（按时间倒序）
        """
        history_file = self.data_dir / f"{symbol}_history.json"

        if not history_file.exists():
            return []

        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))

            # 按日期倒序排序
            history = sorted(
                history,
                key=lambda x: x.get("date", ""),
                reverse=True
            )

            if limit is not None:
                history = history[:limit]

            return history
        except Exception as e:
            print(f"[ERROR] 读取历史记录失败: {e}")
            return []

    def add_history(
            self,
            symbol: str,
            date: pd.Timestamp,
            old_position: float,
            new_position: float,
            reason: str = "",
    ) -> None:
        """
        添加历史记录
        """
        history_file = self.data_dir / f"{symbol}_history.json"

        history = self.get_history(symbol)

        record = {
            "date": str(date.date()),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "old_position": float(old_position),
            "new_position": float(new_position),
            "delta": float(new_position - old_position),
            "reason": reason,
        }

        # 插入到开头（最新的在前面）
        history.insert(0, record)

        history_file.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def print_history(self, symbol: str, limit: int = 10) -> None:
        """
        打印历史记录
        """
        history = self.get_history(symbol, limit=limit)

        if not history:
            print(f"[INFO] {symbol} 暂无历史记录")
            return

        print(f"\n{'=' * 80}")
        print(f"📊 {symbol} 持仓变更历史（最近 {len(history)} 条）")
        print(f"{'=' * 80}")

        for i, record in enumerate(history, 1):
            date = record.get("date", "N/A")
            old = record.get("old_position", 0.0)
            new = record.get("new_position", 0.0)
            delta = record.get("delta", 0.0)
            reason = record.get("reason", "")

            arrow = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"

            print(f"{i:2d}. {date} | {arrow} {old:.2%} → {new:.2%} ({delta:+.2%})")
            if reason:
                print(f"    原因: {reason}")

        print(f"{'=' * 80}\n")

    def reset_position(self, symbol: str, reason: str = "重置") -> None:
        """
        重置持仓为 0
        """
        self.update_position(
            symbol=symbol,
            position=0.0,
            date=pd.Timestamp.now(),
            reason=reason,
        )
