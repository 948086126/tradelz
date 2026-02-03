# celery_config.py

from celery.schedules import crontab

# Celery 配置
broker_url = 'redis://127.0.0.1:6379/0'
result_backend = 'redis://127.0.0.1:6379/0'

# 时区设置
timezone = 'Asia/Shanghai'
enable_utc = False

# 任务序列化
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

# 任务结果过期时间（秒）
result_expires = 3600

# Worker 配置
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000

# 定时任务配置
beat_schedule = {
    # 每日预测任务 - 每天 15:05 执行
    'daily-prediction': {
        'task': 'tasks.trading_tasks.daily_prediction_task',
        'schedule': crontab(hour=15, minute=5),
        'options': {
            'expires': 300,
        }
    },

    # 交易准备任务 - 每天 9:25 执行
    'prepare-trading': {
        'task': 'tasks.trading_tasks.prepare_trading_task',
        'schedule': crontab(hour=9, minute=25),
        'options': {
            'expires': 300,
        }
    },

    # TWAP 检查任务 - 每5分钟执行一次
    'check-twap': {
        'task': 'tasks.trading_tasks.check_twap_task',
        'schedule': crontab(minute='*/5'),
        'options': {
            'expires': 240,
        }
    },
}

# 任务路由
task_routes = {
    'tasks.trading_tasks.*': {'queue': 'trading'},
}
