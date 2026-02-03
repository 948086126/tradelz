#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

echo "停止 Celery 服务..."

# 方法1：通过 PID 文件停止
if [ -f "$PROJECT_ROOT/logs/celery_worker.pid" ]; then
    WORKER_PID=$(cat "$PROJECT_ROOT/logs/celery_worker.pid")
    echo "停止 Worker (PID: $WORKER_PID)..."
    kill -TERM $WORKER_PID 2>/dev/null || kill -9 $WORKER_PID 2>/dev/null
    rm -f "$PROJECT_ROOT/logs/celery_worker.pid"
fi

if [ -f "$PROJECT_ROOT/logs/celery_beat.pid" ]; then
    BEAT_PID=$(cat "$PROJECT_ROOT/logs/celery_beat.pid")
    echo "停止 Beat (PID: $BEAT_PID)..."
    kill -TERM $BEAT_PID 2>/dev/null || kill -9 $BEAT_PID 2>/dev/null
    rm -f "$PROJECT_ROOT/logs/celery_beat.pid"
fi

# 方法2：强制杀掉所有 celery 进程
echo "清理残留进程..."
pkill -f "celery -A celery_app worker"
pkill -f "celery -A celery_app beat"

sleep 2

# 检查是否还有进程
REMAINING=$(ps aux | grep celery | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ Celery 服务已完全停止"
else
    echo "⚠️  还有 $REMAINING 个进程残留，强制终止..."
    pkill -9 -f celery
    sleep 1
    echo "✅ 已强制终止所有进程"
fi
