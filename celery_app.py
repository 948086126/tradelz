import os
import sys
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

# ==========================================
# 核心修复: 强制将当前目录加入 Python 搜索路径
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 加载 .env 环境变量
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Redis 配置
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_DB = os.getenv('REDIS_DB', 0)

# 如果有密码，格式为: redis://:password@host:port/db
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
if REDIS_PASSWORD:
    BROKER_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    BACKEND_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
else:
    BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    BACKEND_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# 创建 Celery 应用
app = Celery(
    'tradelz',  # 应用名称建议简短
    broker=BROKER_URL,
    backend=BACKEND_URL,
    # 既然路径已经修复，直接 include 即可
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
# 注意：确保 tasks/trading_tasks.py 里有对应的函数名
app.conf.beat_schedule = {
    # 每天15:05运行预测
    'daily-prediction': {
        'task': 'tasks.trading_tasks.daily_prediction_task', # 注意这里 key 是 'task' 不是 'tasks'
        'schedule': crontab(hour=15, minute=5),
        'args': ()
    },

    # 交易日早上9:25准备
    'morning-prepare': {
        'task': 'tasks.trading_tasks.prepare_trading_task',
        'schedule': crontab(hour=9, minute=25, day_of_week='1-5'),
        'args': ()
    },

    # 每5分钟检查是否需要执行TWAP
    'check-twap-execution': {
        'task': 'tasks.trading_tasks.check_twap_task',
        'schedule': crontab(minute='*/5', day_of_week='1-5'),
        'args': ()
    },
}

if __name__ == '__main__':
    app.start()
