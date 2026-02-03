# utils/wecom_notifier.py

import requests
import json
from datetime import datetime
import os

# 企业微信机器人 Webhook URL（从环境变量或配置文件读取）
WECOM_WEBHOOK_URL = os.getenv('WECOM_WEBHOOK_URL', '')


def send_wecom_message(message, webhook_url=None):
    """
    发送企业微信消息

    Args:
        message: 消息内容
        webhook_url: 可选的 webhook URL，如果不提供则使用默认配置

    Returns:
        bool: 发送是否成功
    """
    url = webhook_url or WECOM_WEBHOOK_URL

    # 如果没有配置 webhook，只打印日志
    if not url:
        print(f"[企业微信通知] {message}")
        print("⚠️ 未配置 WECOM_WEBHOOK_URL，消息仅打印到日志")
        return False

    try:
        # 构造消息体
        data = {
            "msgtype": "text",
            "text": {
                "content": message
            }
        }

        # 发送请求
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )

        # 检查响应
        if response.status_code == 200:
            result = response.json()
            if result.get('errcode') == 0:
                print(f"[企业微信通知] ✅ 发送成功: {message[:50]}...")
                return True
            else:
                print(f"[企业微信通知] ❌ 发送失败: {result}")
                return False
        else:
            print(f"[企业微信通知] ❌ HTTP错误: {response.status_code}")
            return False

    except Exception as e:
        print(f"[企业微信通知] ❌ 异常: {str(e)}")
        return False


def send_wecom_markdown(title, content, webhook_url=None):
    """
    发送 Markdown 格式的企业微信消息

    Args:
        title: 标题
        content: Markdown 内容
        webhook_url: 可选的 webhook URL

    Returns:
        bool: 发送是否成功
    """
    url = webhook_url or WECOM_WEBHOOK_URL

    if not url:
        print(f"[企业微信通知] {title}\n{content}")
        print("⚠️ 未配置 WECOM_WEBHOOK_URL，消息仅打印到日志")
        return False

    try:
        data = {
            "msgtype": "markdown",
            "markdown": {
                "content": f"**{title}**{content}"
            }
        }

        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('errcode') == 0:
                print(f"[企业微信通知] ✅ Markdown发送成功")
                return True
            else:
                print(f"[企业微信通知] ❌ 发送失败: {result}")
                return False
        else:
            print(f"[企业微信通知] ❌ HTTP错误: {response.status_code}")
            return False

    except Exception as e:
        print(f"[企业微信通知] ❌ 异常: {str(e)}")
        return False


def send_trading_signal(signal_type, stock_code, stock_name, reason, webhook_url=None):
    """
    发送交易信号通知

    Args:
        signal_type: 信号类型（买入/卖出）
        stock_code: 股票代码
        stock_name: 股票名称
        reason: 交易原因
        webhook_url: 可选的 webhook URL
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    message = f"""
📊 交易信号通知

类型: {signal_type}
股票: {stock_name} ({stock_code})
原因: {reason}
时间: {timestamp}
"""

    return send_wecom_message(message.strip(), webhook_url)


def send_task_notification(task_name, status, details="", webhook_url=None):
    """
    发送任务执行通知

    Args:
        task_name: 任务名称
        status: 状态（成功/失败）
        details: 详细信息
        webhook_url: 可选的 webhook URL
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    emoji = "✅" if status == "成功" else "❌"

    message = f"""
{emoji} 任务通知

任务: {task_name}
状态: {status}
时间: {timestamp}
"""

    if details:
        message += f"\n详情: {details}"

    return send_wecom_message(message.strip(), webhook_url)


# 测试函数
if __name__ == '__main__':
    print("测试企业微信通知...")

    # 测试文本消息
    send_wecom_message("这是一条测试消息")

    # 测试交易信号
    send_trading_signal("买入", "000001", "平安银行", "技术指标突破")

    # 测试任务通知
    send_task_notification("每日预测", "成功", "预测完成，生成10个信号")


