#!/bin/bash

# 项目根目录
PROJECT_DIR="/root/tradelz"

# 切换到项目目录
cd $PROJECT_DIR

# 设置 Python 路径
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# 激活虚拟环境（如果使用）
source /root/miniconda3/envs/tradelz/bin/activate

# 创建日志目录
mkdir -p logs

# 停止旧进程
pkill -f "celery worker"
pkill -f "celery beat"

# 等待进程结束
sleep 2

# 启动 Worker
nohup celery -A celery_app worker --loglevel=info \
    --logfile=logs/celery_worker.log \
    --pidfile=logs/celery_worker.pid \
    > logs/celery_worker_stdout.log 2>&1 &

echo "Celery Worker 已启动 (PID: $!)"

# 启动 Beat
nohup celery -A celery_app beat --loglevel=info \
    --logfile=logs/celery_beat.log \
    --pidfile=logs/celery_beat.pid \
    > logs/celery_beat_stdout.log 2>&1 &

echo "Celery Beat 已启动 (PID: $!)"

# 显示状态
sleep 2
ps aux | grep celery | grep -v grep

echo "Celery 服务启动完成！"
echo "查看日志："
echo "  Worker: tail -f logs/celery_worker.log"
echo "  Beat:   tail -f logs/celery_beat.log"
