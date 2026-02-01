# tasks/trading_tasks.py

from celery_app import app
from pathlib import Path
import sys
import pandas as pd
import logging
import json
import os
from datetime import datetime

# 添加项目路径
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core.predict_next_day import predict_next_day
from core.trade.position_manager import PositionManager
from core.trade.twap_executor import SimpleTWAPExecutor
from utils.wechat_notifier import WeChatNotifier

# 初始化通知器
wechat_notifier = WeChatNotifier()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== 辅助函数 =====

def load_latest_predictions():
    """加载最新的预测结果"""
    try:
        signal_path = root / "data" / "signals" / "next_day_signal.json"

        if not signal_path.exists():
            logger.warning("未找到预测信号文件")
            return None

        with open(signal_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载预测结果失败: {e}")
        return None


def generate_trading_plan(predictions):
    """生成交易计划"""
    if not predictions:
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'actions': [],
            'message': '无预测数据'
        }

    current_pos = predictions.get('current_position', 0)
    target_pos = predictions.get('target_position', 0)
    symbol = predictions.get('symbol', 'N/A')

    actions = []

    if abs(target_pos - current_pos) >= 0.05:
        action_type = "买入" if target_pos > current_pos else "卖出"
        delta = abs(target_pos - current_pos)

        actions.append({
            'stock_code': symbol,
            'stock_name': predictions.get('name', symbol),
            'action': action_type,
            'quantity': int(delta * 100000),  # 假设总资金100万
            'current_position': f"{current_pos:.1%}",
            'target_position': f"{target_pos:.1%}"
        })

    return {
        'date': predictions.get('date', datetime.now().strftime('%Y-%m-%d')),
        'actions': actions
    }


def generate_detailed_report(result):
    """生成详细的邮件报告"""
    report = f"""每日预测报告{'=' * 60}预测日期: {result.get('date', 'N/A')}股票代码: {result.get('symbol', 'N/A')}

持仓建议:
- 当前持仓: {result.get('current_position', 0):.2%}
- 目标持仓: {result.get('recommended_position', 0):.2%}
- 持仓变化: {result.get('total_delta', 0):+.2%}

操作建议: {result.get('action', 'N/A')}

TWAP执行计划:
- 开盘时段 (9:30-9:50): {result.get('twap_plan', {}).get('open', 0):.2%}
- 盘中时段 (10:00-14:00): {result.get('twap_plan', {}).get('mid', 0):.2%}
- 尾盘时段 (14:40-14:57): {result.get('twap_plan', {}).get('close', 0):.2%}

{'=' * 60}
此报告由自动交易系统生成
"""
    return report


def generate_plan_report(plan):
    """生成交易计划报告"""
    report = f"""
交易计划
{'=' * 60}

日期: {plan.get('date', 'N/A')}

今日操作:
"""

    actions = plan.get('actions', [])
    if actions:
        for action in actions:
            report += f"""
- {action['stock_code']} {action['stock_name']}
  操作: {action['action']}
  数量: {action['quantity']}股
  当前持仓: {action['current_position']}
  目标持仓: {action['target_position']}
"""
    else:
        report += "\n无操作计划\n"

    report += f"\n{'=' * 60}"
    return report


def send_email_safe(subject, body, to_email=None):
    """安全发送邮件（如果配置了邮件）"""
    try:
        # 检查是否配置了邮件
        if not os.getenv('EMAIL_HOST'):
            logger.info("未配置邮件，跳过邮件发送")
            return False

        # 尝试导入邮件发送模块
        try:
            from utils.email_sender import send_email
            send_email(subject=subject, body=body, to_email=to_email or os.getenv('EMAIL_TO'))
            return True
        except ImportError:
            logger.warning("邮件发送模块不存在，跳过")
            return False

    except Exception as e:
        logger.warning(f"邮件发送失败: {e}")
        return False


# ===== Celery 任务 =====

@app.task(bind=True, name='tasks.trading_tasks.daily_prediction_task')
def daily_prediction_task(self):
    """
    每日预测任务（15:05执行）
    """
    try:
        logger.info("=" * 60)
        logger.info("开始每日预测任务")
        logger.info("=" * 60)

        # 🚀 发送开始通知
        wechat_notifier.send_text("📊 开始执行每日预测...")

        # 调用预测函数
        result = predict_next_day(
            symbol='601138',
            send_email=False  # 我们自己控制邮件发送
        )

        if result is None:
            error_msg = "预测失败：返回结果为空"
            logger.error(error_msg)
            wechat_notifier.send_error(error_msg, at_all=True)
            return {'status': 'error', 'message': error_msg}

        logger.info("每日预测任务完成")
        logger.info(f"当前持仓: {result['current_position']:.2%}")
        logger.info(f"目标持仓: {result['recommended_position']:.2%}")
        logger.info(f"持仓变化: {result['total_delta']:+.2%}")

        # 🚀 企业微信通知（秒级到达）
        prediction_data = {
            'date': result.get('date', datetime.now().strftime('%Y-%m-%d')),
            'positions': [{
                'code': result.get('symbol', 'N/A'),
                'name': result.get('symbol', 'N/A'),
                'weight': result.get('recommended_position', 0),
                'expected_return': result.get('total_delta', 0)
            }]
        }
        wechat_notifier.send_prediction_result(prediction_data)

        # 📧 邮件通知（详细报告，可选）
        send_email_safe(
            subject=f"每日预测报告 - {result.get('date', 'N/A')}",
            body=generate_detailed_report(result)
        )

        return {
            'status': 'success',
            'symbol': result['symbol'],
            'date': result['date'],
            'current_position': result['current_position'],
            'target_position': result['recommended_position'],
            'action': result['action'],
            'total_delta': result['total_delta']
        }

    except Exception as e:
        error_msg = f"每日预测任务失败: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # 🚀 发送错误通知
        wechat_notifier.send_error(error_msg, at_all=True)

        return {'status': 'error', 'message': str(e)}


@app.task(bind=True, name='tasks.trading_tasks.prepare_trading_task')
def prepare_trading_task(self):
    """
    交易准备任务（9:25执行）
    检查今天是否需要交易
    """
    try:
        logger.info("=" * 60)
        logger.info("开始交易准备任务")
        logger.info("=" * 60)

        # 读取最新预测结果
        predictions = load_latest_predictions()

        if not predictions:
            msg = "未找到预测数据"
            logger.warning(msg)
            wechat_notifier.send_text(f"⚠️ {msg}")
            return {'status': 'no_signal', 'message': msg}

        # 生成交易计划
        plan = generate_trading_plan(predictions)

        current_pos = predictions.get('current_position', 0)
        target_pos = predictions.get('target_position', 0)

        if abs(target_pos - current_pos) < 0.05:
            logger.info("无需交易")
            wechat_notifier.send_text("💎 今日无需交易，持仓已达目标")
            return {'status': 'no_trade', 'message': '持仓已达目标'}

        logger.info(f"需要交易: {current_pos:.2%} → {target_pos:.2%}")

        # 🚀 企业微信通知
        wechat_notifier.send_trading_plan(plan)

        # 📧 邮件通知（可选）
        send_email_safe(
            subject=f"交易计划 - {plan.get('date', 'N/A')}",
            body=generate_plan_report(plan)
        )

        return {
            'status': 'ready',
            'symbol': predictions['symbol'],
            'current_position': current_pos,
            'target_position': target_pos,
            'twap_plan': predictions.get('twap_plan', {})
        }

    except Exception as e:
        error_msg = f"交易准备任务失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        wechat_notifier.send_error(error_msg, at_all=True)
        return {'status': 'error', 'message': str(e)}


@app.task(bind=True, name='tasks.trading_tasks.check_twap_task')
def check_twap_task(self):
    """
    检查TWAP执行（每5分钟）
    在交易时段内执行
    """
    try:
        now = pd.Timestamp.now()

        # 检查是否在交易时段
        morning_start = now.replace(hour=9, minute=30, second=0)
        morning_end = now.replace(hour=11, minute=30, second=0)
        afternoon_start = now.replace(hour=13, minute=0, second=0)
        afternoon_end = now.replace(hour=14, minute=57, second=0)

        in_trading_hours = (
                (morning_start <= now <= morning_end) or
                (afternoon_start <= now <= afternoon_end)
        )

        if not in_trading_hours:
            return {'status': 'skip', 'message': '非交易时段'}

        logger.info(f"检查TWAP执行 - {now}")

        # 读取交易信号
        predictions = load_latest_predictions()

        if not predictions:
            return {'status': 'no_signal', 'message': '未找到交易信号'}

        # 判断当前应该执行哪个时间窗口
        if now.hour == 9 and 30 <= now.minute <= 50:
            window = "open"
            ratio = predictions.get('twap_plan', {}).get('open', 0)
            logger.info(f"开盘时段 - 执行 {ratio:.2%}")
        elif 10 <= now.hour < 14 or (now.hour == 14 and now.minute == 0):
            window = "mid"
            ratio = predictions.get('twap_plan', {}).get('mid', 0)
            logger.info(f"盘中时段 - 执行 {ratio:.2%}")
        elif now.hour == 14 and 40 <= now.minute <= 57:
            window = "close"
            ratio = predictions.get('twap_plan', {}).get('close', 0)
            logger.info(f"尾盘时段 - 执行 {ratio:.2%}")
        else:
            return {'status': 'skip', 'message': '不在TWAP执行窗口'}

        # TODO: 这里接入实际的交易接口
        # 例如：broker.place_order(symbol, ratio, ...)

        logger.info(f"[模拟] 执行交易: {predictions['symbol']}, 比例: {ratio:.2%}")

        # 🚀 发送执行通知
        wechat_notifier.send_trading_signal(
            signal_type=f"TWAP执行 - {window}",
            stock_code=predictions['symbol'],
            stock_name=predictions.get('name', predictions['symbol']),
            action="执行中",
            quantity=int(ratio * 100000),
            reason=f"按计划执行 {ratio:.1%}"
        )

        return {
            'status': 'executed',
            'window': window,
            'ratio': ratio,
            'time': str(now)
        }

    except Exception as e:
        logger.error(f"TWAP检查任务失败: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}


@app.task(bind=True, name='tasks.trading_tasks.manual_prediction')
def manual_prediction(self, symbol='601138'):
    """
    手动触发预测（Web界面调用）
    """
    try:
        logger.info(f"手动触发预测: {symbol}")
        wechat_notifier.send_text(f"🔄 手动预测任务开始 - {symbol}")

        # 调用每日预测任务
        result = daily_prediction_task()

        return result

    except Exception as e:
        error_msg = f"手动预测失败: {str(e)}"
        logger.error(error_msg)
        wechat_notifier.send_error(error_msg)
        raise
