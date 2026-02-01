# tasks/trading_task.py
import os
import sys
from celery import shared_task
from datetime import datetime
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, '/root/tradelz')

from utils.wecom_notifier import send_wecom_message, send_task_notification

@shared_task(name='tasks.trading_tasks.manual_prediction')  # ← 添加这行
def manual_prediction():
    """手动预测任务"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] 🚀 开始执行手动预测...")

        # 模拟预测逻辑
        result = f"预测完成 - {timestamp}"

        send_task_notification("手动预测", "成功", result)
        return result

    except Exception as e:
        error_msg = f"手动预测失败: {str(e)}"
        print(f"❌ {error_msg}")
        send_task_notification("手动预测", "失败", error_msg)
        raise

@shared_task(name='tasks.trading_tasks.daily_prediction_task')
def daily_prediction_task():
    """每日预测任务"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] 📊 开始每日预测...")

        # 调用 main.py 的预测功能
        cmd = "cd /root/tradelz && /root/miniconda3/envs/tradelz/bin/python main.py --mode predict"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            msg = f"每日预测成功 - {timestamp}"
            print(f"✅ {msg}")
            send_task_notification("每日预测", "成功", msg)
            return msg
        else:
            error = result.stderr or result.stdout
            print(f"❌ 预测失败: {error}")
            send_task_notification("每日预测", "失败", error[:200])
            raise Exception(error)

    except Exception as e:
        error_msg = f"每日预测异常: {str(e)}"
        print(f"❌ {error_msg}")
        send_task_notification("每日预测", "失败", error_msg)
        raise

@shared_task(name='tasks.trading_tasks.prepare_trading_task')
def prepare_trading_task():
    """准备交易任务"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] 🔧 准备交易...")

        # 调用 main.py 的交易准备功能
        cmd = "cd /root/tradelz && /root/miniconda3/envs/tradelz/bin/python main.py --mode prepare"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            msg = f"交易准备完成 - {timestamp}"
            print(f"✅ {msg}")
            send_task_notification("交易准备", "成功", msg)
            return msg
        else:
            error = result.stderr or result.stdout
            print(f"❌ 准备失败: {error}")
            send_task_notification("交易准备", "失败", error[:200])
            raise Exception(error)

    except Exception as e:
        error_msg = f"交易准备异常: {str(e)}"
        print(f"❌ {error_msg}")
        send_task_notification("交易准备", "失败", error_msg)
        raise

@shared_task(name='tasks.trading_tasks.check_twap_task')
def check_twap_task():
    """检查 TWAP 执行任务"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] 🔍 检查 TWAP 执行...")

        # 调用 main.py 的 TWAP 检查功能
        cmd = "cd /root/tradelz && /root/miniconda3/envs/tradelz/bin/python main.py --mode check_twap"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            msg = f"TWAP 检查完成 - {timestamp}"
            print(f"✅ {msg}")
            return msg
        else:
            error = result.stderr or result.stdout
            print(f"❌ TWAP 检查失败: {error}")
            raise Exception(error)

    except Exception as e:
        error_msg = f"TWAP 检查异常: {str(e)}"
        print(f"❌ {error_msg}")
        raise