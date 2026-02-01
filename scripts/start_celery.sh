#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)

echo "项目根目录: $PROJECT_ROOT"

# 创建日志目录
mkdir -p "$PROJECT_ROOT/logs"

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tradelz

# 检查环境
if [ $? -ne 0 ]; then
    echo "❌ 激活 conda 环境失败"
    exit 1
fi

echo "✅ Conda 环境已激活: $(which python)"

# 停止旧进程（如果存在）
echo "停止旧进程..."
pkill -f "celery -A celery_app worker"
pkill -f "celery -A celery_app beat"
sleep 2

# 清理旧的 PID 文件
rm -f "$PROJECT_ROOT/logs/celery_worker.pid"
rm -f "$PROJECT_ROOT/logs/celery_beat.pid"

# 启动 Celery Worker
echo "启动 Celery Worker..."
cd "$PROJECT_ROOT"
nohup celery -A celery_app worker \
    --loglevel=info \
    --logfile="$PROJECT_ROOT/logs/celery_worker.log" \
    --pidfile="$PROJECT_ROOT/logs/celery_worker.pid" \
    > "$PROJECT_ROOT/logs/celery_worker_stdout.log" 2>&1 &

WORKER_PID=$!
echo "Celery Worker 已启动 (PID: $WORKER_PID)"

# 等待 Worker 启动
sleep 3

# 启动 Celery Beat
echo "启动 Celery Beat..."
nohup celery -A celery_app beat \
    --loglevel=info \
    --logfile="$PROJECT_ROOT/logs/celery_beat.log" \
    --pidfile="$PROJECT_ROOT/logs/celery_beat.pid" \
    > "$PROJECT_ROOT/logs/celery_beat_stdout.log" 2>&1 &

BEAT_PID=$!
echo "Celery Beat 已启动 (PID: $BEAT_PID)"

# 等待服务启动
sleep 2

# 检查进程状态
echo ""
echo "检查进程状态..."
ps aux | grep celery | grep -v grep

echo ""
echo "✅ Celery 服务启动完成！"
echo ""
echo "查看日志："
echo "  Worker: tail -f $PROJECT_ROOT/logs/celery_worker.log"
echo "  Beat:   tail -f $PROJECT_ROOT/logs/celery_beat.log"
echo ""
echo "查看实时输出："
echo "  Worker: tail -f $PROJECT_ROOT/logs/celery_worker_stdout.log"
echo "  Beat:   tail -f $PROJECT_ROOT/logs/celery_beat_stdout.log"
