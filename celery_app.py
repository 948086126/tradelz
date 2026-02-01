# celery_app.py

from celery import Celery
from celery.schedules import crontab
import os

# Redis 配置
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_DB = os.getenv('REDIS_DB', 0)

BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
BACKEND_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# 创建 Celery 应用
app = Celery(
    'trading_system',
    broker=BROKER_URL,
    backend=BACKEND_URL,
    include=['tasks.trading_tasks']
)

# Celery 配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=False,

    # 任务结果过期时间（1天）
    result_expires=86400,

    # 任务超时时间
    task_time_limit=3600,  # 1小时
    task_soft_time_limit=3000,  # 50分钟

    # 并发配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)

# 定时任务配置
app.conf.beat_schedule = {
    # 每天15:05运行预测
    'daily-prediction': {
        'tasks': 'tasks.trading_tasks.daily_prediction_task',
        'schedule': crontab(hour=15, minute=5),
        'args': ()
    },

    # 交易日早上9:25准备
    'morning-prepare': {
        'tasks': 'tasks.trading_tasks.prepare_trading_task',
        'schedule': crontab(hour=9, minute=25, day_of_week='1-5'),
        'args': ()
    },

    # 每5分钟检查是否需要执行TWAP
    'check-twap-execution': {
        'tasks': 'tasks.trading_tasks.check_twap_task',
        'schedule': crontab(minute='*/5', day_of_week='1-5'),
        'args': ()
    },
}

if __name__ == '__main__':
    app.start()
