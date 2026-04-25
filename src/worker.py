import os

from rq import Worker

from src.queue import get_redis
from src.utils.logging_setup import configure_logging


def main():
    configure_logging()
    queues = os.getenv("RQ_QUEUES", "default").split(",")
    queues = [q.strip() for q in queues if q.strip()]
    worker = Worker(queues, connection=get_redis())
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
