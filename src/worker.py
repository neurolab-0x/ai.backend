import os
import logging
import subprocess
import sys
from dataclasses import dataclass

try:
    from rq import Worker
    RQ_AVAILABLE = True
except ImportError:
    Worker = None
    RQ_AVAILABLE = False

from src.queue import get_redis
from src.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _job_exception_handler(job, exc_type, exc_value, _traceback):
    logger.exception(
        "Worker job failed queue=%s job_id=%s func=%s error=%s",
        getattr(job, "origin", "unknown"),
        getattr(job, "id", "unknown"),
        getattr(job, "func_name", "unknown"),
        exc_value,
    )
    return True


def configured_queues() -> list[str]:
    requested = os.getenv("RQ_QUEUES", "default,persistence,chat,reports").split(",")
    queues = [q.strip() for q in requested if q.strip()]
    for required in ("default", "persistence", "chat", "reports"):
        if required not in queues:
            queues.append(required)
    return queues


def create_worker() -> "Worker":
    if not RQ_AVAILABLE:
        raise RuntimeError("RQ is not installed; worker cannot be started")

    redis = get_redis()
    worker = Worker(configured_queues(), connection=redis)
    worker.push_exc_handler(_job_exception_handler)
    return worker


@dataclass
class ManagedWorkerProcessHandle:
    process: subprocess.Popen
    queues: list[str]
    redis_url: str

    def stop(self, timeout_seconds: float = 10.0) -> None:
        logger.info(
            "Stopping managed RQ worker process pid=%s queues=%s",
            self.process.pid,
            ",".join(self.queues),
        )
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Managed RQ worker pid=%s did not stop within %.1fs; killing",
                self.process.pid,
                timeout_seconds,
            )
            self.process.kill()
            self.process.wait(timeout=5)
        logger.info("Managed RQ worker process stopped pid=%s", self.process.pid)


def start_managed_worker_process() -> ManagedWorkerProcessHandle:
    configure_logging()
    queues = configured_queues()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    env = os.environ.copy()
    env["RQ_QUEUES"] = ",".join(queues)

    logger.info(
        "Starting managed RQ worker process queues=%s redis_url=%s",
        ",".join(queues),
        redis_url,
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "src.worker"],
        env=env,
        cwd=os.getcwd(),
    )
    return ManagedWorkerProcessHandle(
        process=process,
        queues=queues,
        redis_url=redis_url,
    )


def main():
    configure_logging()
    queues = configured_queues()
    logger.info(
        "Starting RQ worker queues=%s redis_url=%s",
        ",".join(queues),
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )
    worker = create_worker()
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
