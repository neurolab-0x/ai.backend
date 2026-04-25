import os

from rq import Worker

from src.queue import get_redis


def main():
    queues = os.getenv("RQ_QUEUES", "default").split(",")
    queues = [q.strip() for q in queues if q.strip()]
    worker = Worker(queues, connection=get_redis())
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()

