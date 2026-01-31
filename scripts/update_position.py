


import pandas as pd
from core.trade.position_manager import PositionManager


def main():
    pos_manager = PositionManager()

    print(" " + " = "*60)
    print("📝 持仓管理器 - 手动更新")
    print("=" * 60 + "     ")

    # 1. 输入股票代码
    symbol = input("股票代码（默认 601138）: ").strip()
    if not symbol:
        symbol = "601138"

    # 2. 显示当前持仓
    current_info = pos_manager.get_position_info(symbol)
    current = current_info["position"]
    last_update = current_info.get("update_date", "N/A")
    last_reason = current_info.get("reason", "N/A")

    print(f"当前持仓: {current: .2 %}")
    print(f"上次更新: {last_update}")
    print(f"更新原因: {last_reason}")

    # 3. 显示历史记录
    print("最近5次变更: ")
    pos_manager.print_history(symbol, limit=5)

    # 4. 输入新持仓
    print("请输入新持仓:")
    print("  - 输入 0-1 之间的小数（如 0.6 表示 60%）")
    print("  - 输入 0-100 之间的整数（如 60 表示 60%）")
    print("  - 输入 'q' 退出")

    while True:
        new_pos_input = input("新持仓: ").strip()

        if new_pos_input.lower() == 'q':
            print("\n已取消\n")
            return

        try:
            new_pos = float(new_pos_input)

            # 如果输入的是 0-100 的整数，转换为 0-1
            if new_pos > 1.0:
                new_pos = new_pos / 100.0

            # 限制范围
            new_pos = max(0.0, min(1.0, new_pos))
            break

        except ValueError:
            print("❌ 输入格式错误，请重新输入")
            continue

    # 5. 输入原因
    reason = input("更新原因（可选）: ").strip()
    if not reason:
        reason = "手动更新"

    # 6. 确认
    print(f"{'=' * 60}")
    print("确认更新:")
    print(f"  股票: {symbol}")
    print(f"  {current:.2%} → {new_pos:.2%} ({new_pos - current:+.2%})")
    print(f"  原因: {reason}")
    print(f"{'=' * 60}")

    while True:
        confirm = input("确认？(y/n): ").strip().lower()
        if confirm in ['y', 'n']:
            break
        print("请输入 y 或 n")

    if confirm != 'y':
        print("\n已取消")
        return

    # 7. 更新
    try:
        pos_manager.update_position(
            symbol=symbol,
            position=new_pos,
            date=pd.Timestamp.now(),
            reason=reason,
        )
        print("✅ 持仓更新成功")
    except Exception as e:
        print(f"❌ 更新失败: {e}\n")
        import traceback
        traceback.print_exc()

        if __name__ == "__main__":
            try:
                main()
            except KeyboardInterrupt:
                print("\n已取消")
            except Exception as e:
                print(f"❌ 发生错误: {e}\n")
                import traceback
                traceback.print_exc()
