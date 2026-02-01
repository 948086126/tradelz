import requests
import json
from typing import Optional, Dict, List
from datetime import datetime
import os


class WeChatNotifier:
    """企业微信机器人通知"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv('WECHAT_WEBHOOK')
        if not self.webhook_url:
            raise ValueError("未配置企业微信 Webhook URL")

    def send_text(self, content: str, mentioned_list: Optional[List[str]] = None):
        """
        发送文本消息

        Args:
            content: 消息内容
            mentioned_list: @的用户列表，["@all"] 表示@所有人
        """
        data = {
            "msgtype": "text",
            "text": {
                "content": content,
                "mentioned_list": mentioned_list or []
            }
        }
        return self._send(data)

    def send_markdown(self, content: str):
        """
        发送 Markdown 消息

        Args:
            content: Markdown 格式的内容
        """
        data = {
            "msgtype": "markdown",
            "markdown": {
                "content": content
            }
        }
        return self._send(data)

    def send_trading_signal(self,
                            signal_type: str,
                            stock_code: str,
                            stock_name: str,
                            action: str,
                            price: float = None,
                            quantity: int = None,
                            reason: str = ""):
        """
        发送交易信号

        Args:
            signal_type: 信号类型（如：开盘提醒、收盘预测）
            stock_code: 股票代码
            stock_name: 股票名称
            action: 操作（买入/卖出/持有）
            price: 价格
            quantity: 数量
            reason: 原因说明
        """
        # 根据操作类型选择颜色和图标
        action_config = {
            "买入": {"color": "info", "icon": "📈"},
            "卖出": {"color": "warning", "icon": "📉"},
            "持有": {"color": "comment", "icon": "💎"}
        }

        config = action_config.get(action, {"color": "comment", "icon": "📊"})

        content = f"""## {config['icon']} {signal_type}
> **股票**: <font color="info">{stock_code} {stock_name}</font>
> **操作**: <font color="{config['color']}">{action}</font>"""

        if price is not None:
            content += f"\n> **价格**: ¥{price:.2f}"

        if quantity is not None:
            content += f"\n> **数量**: {quantity}股"

        content += f"\n> **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        if reason:
            content += f"\n> **原因**: {reason}"

        return self.send_markdown(content)

    def send_prediction_result(self, predictions: Dict):
        """
        发送预测结果

        Args:
            predictions: 预测结果字典
        """
        pred_date = predictions.get('date', datetime.now().strftime('%Y-%m-%d'))

        content = f"""## 📊 每日预测结果
> **预测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **预测日期**: {pred_date}

### 持仓建议
"""

        positions = predictions.get('positions', [])
        if positions:
            for stock in positions[:5]:  # 只显示前5个
                code = stock.get('code', 'N/A')
                name = stock.get('name', 'N/A')
                weight = stock.get('weight', 0) * 100
                expected_return = stock.get('expected_return', 0) * 100

                content += f"""
> **{code} {name}**
> 建议仓位: {weight:.1f}% | 预期收益: {expected_return:+.2f}%
"""
        else:
            content += "\n> 暂无持仓建议"

        # 添加总结
        total_stocks = len(positions)
        if total_stocks > 5:
            content += f"\n> ... 共 {total_stocks} 只股票"

        content += "\n\n---\n💡 详细报告已发送至邮箱"

        return self.send_markdown(content)

    def send_trading_plan(self, plan: Dict):
        """
        发送交易计划

        Args:
            plan: 交易计划字典
        """
        content = f"""## 🌅 开盘提醒
> **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **交易日**: {plan.get('date', 'N/A')}

### 今日操作计划
"""

        actions = plan.get('actions', [])
        if actions:
            for action in actions:
                stock_code = action.get('stock_code', 'N/A')
                stock_name = action.get('stock_name', 'N/A')
                operation = action.get('action', 'N/A')
                quantity = action.get('quantity', 0)

                icon = "📈" if operation == "买入" else "📉" if operation == "卖出" else "💎"
                content += f"\n> {icon} **{stock_code} {stock_name}**: {operation} {quantity}股"
        else:
            content += "\n> 今日无操作计划"

        content += "\n\n---\n⏰ 请在 9:30 开盘后执行"

        return self.send_markdown(content)

    def send_error(self, error_msg: str, at_all: bool = True):
        """
        发送错误通知

        Args:
            error_msg: 错误信息
            at_all: 是否@所有人
        """
        content = f"""## ❌ 系统错误
> **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **错误**: {error_msg}

请及时检查系统日志！"""

        # 先发送 Markdown
        self.send_markdown(content)

        # 再发送文本 @所有人
        if at_all:
            self.send_text(f"❌ 系统错误: {error_msg}", mentioned_list=["@all"])

        return True

    def send_system_status(self, status: str, details: str = ""):
        """
        发送系统状态

        Args:
            status: 状态（启动/停止/重启）
            details: 详细信息
        """
        icon_map = {
            "启动": "🚀",
            "停止": "🛑",
            "重启": "🔄"
        }

        icon = icon_map.get(status, "ℹ️")

        content = f"""## {icon} 系统{status}
> **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        if details:
            content += f"\n> **详情**: {details}"

        return self.send_markdown(content)

    def _send(self, data: Dict) -> bool:
        """
        发送消息到企业微信

        Args:
            data: 消息数据

        Returns:
            是否发送成功
        """
        try:
            response = requests.post(
                self.webhook_url,
                json=data,
                timeout=5
            )
            result = response.json()

            if result.get('errcode') == 0:
                return True
            else:
                error_msg = result.get('errmsg', '未知错误')
                print(f"❌ 企业微信通知失败: {error_msg}")
                return False

        except requests.exceptions.Timeout:
            print("❌ 企业微信通知超时")
            return False
        except Exception as e:
            print(f"❌ 企业微信通知异常: {e}")
            return False


# ===== 测试代码 =====
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    notifier = WeChatNotifier()

    # 测试1：发送文本
    print("测试1：发送文本消息...")
    notifier.send_text("🎉 交易系统测试消息")

    # 测试2：发送交易信号
    print("测试2：发送交易信号...")
    notifier.send_trading_signal(
        signal_type="测试信号",
        stock_code="600519",
        stock_name="贵州茅台",
        action="买入",
        price=1680.50,
        quantity=100,
        reason="测试原因"
    )

    # 测试3：发送预测结果
    print("测试3：发送预测结果...")
    test_predictions = {
        'date': '2026-02-03',
        'positions': [
            {'code': '600519', 'name': '贵州茅台', 'weight': 0.3, 'expected_return': 0.05},
            {'code': '000858', 'name': '五粮液', 'weight': 0.2, 'expected_return': 0.03}
        ]
    }
    notifier.send_prediction_result(test_predictions)

    print("✅ 测试完成！请检查企业微信群是否收到消息")


python << 'EOF'
from tasks.trading_tasks import manual_prediction

print("🚀 测试手动预测任务...")
result = manual_prediction.delay()
print(f"任务 ID: {result.id}")

print("⏳ 等待结果（最多30秒）...")
try:
    output = result.get(timeout=30)
    print(f"\n✅ 任务成功:")
    print(output)
except Exception as e:
    print(f"❌ 任务失败:")
    print(e)
    import traceback
    traceback.print_exc()
EOF
