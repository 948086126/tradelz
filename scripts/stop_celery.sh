#!/bin/bash

echo "正在停止 Celery 服务..."

# 方法1：使用 PID 文件
if [ -f logs/celery_worker.pid ]; then
    kill $(cat logs/celery_worker.pid) 2>/dev/null
    echo "Worker 已停止"
fi

if [ -f logs/celery_beat.pid ]; then
    kill $(cat logs/celery_beat.pid) 2>/dev/null
    echo "Beat 已停止"
fi

# 方法2：强制杀死所有 celery 进程
pkill -f "celery worker"
pkill -f "celery beat"

sleep 2

# 检查是否还有残留进程
if ps aux | grep celery | grep -v grep > /dev/null; then
    echo "警告：仍有 Celery 进程在运行"
    ps aux | grep celery | grep -v grep
else
    echo "✅ 所有 Celery 进程已停止"
fi
