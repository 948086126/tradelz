

import os
from redis import Redis
from rq import Queue

def get_queue() -> Queue:
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    # redis_url = os.getenv("REDIS_URL", "redis://82.157.246.81:6379/0")
    conn = Redis.from_url(redis_url)
    return Queue("train", connection=conn)





