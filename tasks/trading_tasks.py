# tasks/trading_tasks.py

from celery_app import app
from pathlib import Path
import sys
import pandas as pd
import logging

# 添加项目路径
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from core.predict_next_day import predict_next_day
from core.trade.position_manager import PositionManager
from core.trade.twap_executor import SimpleTWAPExecutor  # ✅ 修改这里
from core.trade.email_notifier import EmailNotifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.task(bind=True, name='tasks.trading_tasks.daily_prediction_task')
def daily_prediction_task(self):
    """
    每日预测任务（15:05执行）
    """
    try:
        logger.info("=" * 60)
        logger.info("开始每日预测任务")
        logger.info("=" * 60)

        # 调用预测函数
        result = predict_next_day(
            symbol='601138',
            send_email=True
        )

        if result is None:
            logger.error("预测失败")
            return {'status': 'error', 'message': '预测失败'}

        logger.info("每日预测任务完成")
        logger.info(f"当前持仓: {result['current_position']:.2%}")
        logger.info(f"目标持仓: {result['recommended_position']:.2%}")
        logger.info(f"持仓变化: {result['total_delta']:+.2%}")

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
        logger.error(f"每日预测任务失败: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}


@app.task(bind=True, name='tasks.trading_tasks.prepare_trading_task')
def prepare_trading_task(self):
    """
    交易准备任务（9:25执行）
    检查今天是否需要交易
    """
    try:
        logger.info("检查今日交易计划")

        # 读取昨天生成的信号文件
        root = Path(__file__).resolve().parents[1]
        signal_path = root / "data" / "signals" / "next_day_signal.json"

        if not signal_path.exists():
            logger.warning("未找到交易信号文件")
            return {'status': 'no_signal', 'message': '未找到交易信号'}

        import json
        signal = json.loads(signal_path.read_text(encoding='utf-8'))

        current_pos = signal['current_position']
        target_pos = signal['target_position']

        if abs(target_pos - current_pos) < 0.05:
            logger.info("无需交易")
            return {'status': 'no_trade', 'message': '持仓已达目标'}

        logger.info(f"需要交易: {current_pos:.2%} → {target_pos:.2%}")

        return {
            'status': 'ready',
            'symbol': signal['symbol'],
            'current_position': current_pos,
            'target_position': target_pos,
            'twap_plan': signal['twap_plan']
        }

    except Exception as e:
        logger.error(f"交易准备任务失败: {e}", exc_info=True)
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
        root = Path(__file__).resolve().parents[1]
        signal_path = root / "data" / "signals" / "next_day_signal.json"

        if not signal_path.exists():
            return {'status': 'no_signal', 'message': '未找到交易信号'}

        import json
        signal = json.loads(signal_path.read_text(encoding='utf-8'))

        # 判断当前应该执行哪个时间窗口
        if now.hour == 9 and 30 <= now.minute <= 50:
            window = "open"
            ratio = signal['twap_plan']['open']
            logger.info(f"开盘时段 - 执行 {ratio:.2%}")
        elif 10 <= now.hour < 14 or (now.hour == 14 and now.minute == 0):
            window = "mid"
            ratio = signal['twap_plan']['mid']
            logger.info(f"盘中时段 - 执行 {ratio:.2%}")
        elif now.hour == 14 and 40 <= now.minute <= 57:
            window = "close"
            ratio = signal['twap_plan']['close']
            logger.info(f"尾盘时段 - 执行 {ratio:.2%}")
        else:
            return {'status': 'skip', 'message': '不在TWAP执行窗口'}

        # TODO: 这里接入实际的交易接口
        # 例如：broker.place_order(symbol, ratio, ...)

        logger.info(f"[模拟] 执行交易: {signal['symbol']}, 比例: {ratio:.2%}")

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
    logger.info(f"手动触发预测: {symbol}")
    return daily_prediction_task()
