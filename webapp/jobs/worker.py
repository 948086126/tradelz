# webapp/jobs/worker.py
import os
from redis import Redis
from rq import Queue
from rq.worker import SimpleWorker

def main():
    redis_url = os.getenv("REDIS_URL", "redis://82.157.246.81:6379/0")
    conn = Redis.from_url(redis_url)

    q = Queue("train", connection=conn)

    # ✅ Windows-safe：不 fork
    worker = SimpleWorker([q], connection=conn)
    worker.work(with_scheduler=False)

if __name__ == "__main__":
    main()
