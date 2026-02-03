#!/bin/bash

cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

echo "===== Celery 服务状态 ====="
echo ""

WORKER_COUNT=$(ps aux | grep "celery -A celery_app worker" | grep -v grep | wc -l)
BEAT_COUNT=$(ps aux | grep "celery -A celery_app beat" | grep -v grep | wc -l)

echo "Worker 进程数: $WORKER_COUNT"
echo "Beat 进程数: $BEAT_COUNT"
echo ""

if [ $WORKER_COUNT -gt 0 ]; then
    echo "✅ Worker 运行中:"
    ps aux | grep "celery -A celery_app worker" | grep -v grep | head -3
else
    echo "❌ Worker 未运行"
fi

echo ""

if [ $BEAT_COUNT -gt 0 ]; then
    echo "✅ Beat 运行中:"
    ps aux | grep "celery -A celery_app beat" | grep -v grep
else
    echo "❌ Beat 未运行"
fi

echo ""
echo "===== 最新日志 ====="
echo ""

if [ -f "$PROJECT_ROOT/logs/celery_worker.log" ]; then
    echo "Worker 日志 (最后 5 行):"
    tail -5 "$PROJECT_ROOT/logs/celery_worker.log"
else
    echo "❌ Worker 日志不存在"
fi

echo ""

if [ -f "$PROJECT_ROOT/logs/celery_beat.log" ]; then
    echo "Beat 日志 (最后 5 行):"
    tail -5 "$PROJECT_ROOT/logs/celery_beat.log"
else
    echo "❌ Beat 日志不存在"
fi