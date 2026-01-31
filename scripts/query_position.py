# scripts/query_position.py

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core.trade.position_manager import PositionManager
# ✅ 修改：从 core/trade/ 到项目根目录需要往上两级
root = Path(__file__).resolve().parents[2]  # ← 改成 parents[2]
sys.path.insert(0, str(root))


def main():
    """
    查询持仓信息
    """
    pos_manager = PositionManager()

    print(" " + " = "*60)
    print("📊 持仓管理器 - 查询")
    print("=" * 60 + " ")

    symbol = input("股票代码（默认 601138）: ").strip() or "601138"

    # 当前持仓
    info = pos_manager.get_position_info(symbol)

    print(f"\n当前持仓信息:")
    print(f"  股票代码: {info['symbol']}")
    print(f"  持仓比例: {info['position']:.2%}")
    print(f"  更新日期: {info.get('update_date', 'N/A')}")
    print(f"  更新时间: {info.get('update_time', 'N/A')}")
    print(f"  更新原因: {info.get('reason', 'N/A')}")

    # 历史记录
    limit = int(input("显示最近N条历史记录（默认10）: ").strip() or "10")
    pos_manager.print_history(symbol, limit=limit)

    if __name__ == "__main__":
        main()
