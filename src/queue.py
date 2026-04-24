import os
import logging
from typing import Optional

from redis import Redis
from rq import Queue

logger = logging.getLogger(__name__)

_JOB_SET_PREFIX = "neurolab:rq:jobs:"


def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> Redis:
    return Redis.from_url(redis_url())


def get_queue(name: str = "default") -> Queue:
    return Queue(name, connection=get_redis())


def track_job(category: str, job_id: str, ttl_seconds: int = 60 * 60 * 24) -> None:
    try:
        r = get_redis()
        key = f"{_JOB_SET_PREFIX}{category}"
        r.sadd(key, job_id)
        r.expire(key, ttl_seconds)
    except Exception as e:
        logger.warning(f"Failed to track job_id={job_id}: {e}")


def untrack_job(category: str, job_id: str) -> None:
    try:
        r = get_redis()
        key = f"{_JOB_SET_PREFIX}{category}"
        r.srem(key, job_id)
    except Exception as e:
        logger.warning(f"Failed to untrack job_id={job_id}: {e}")


def list_tracked_jobs(category: str):
    r = get_redis()
    key = f"{_JOB_SET_PREFIX}{category}"
    return [jid.decode("utf-8") if isinstance(jid, (bytes, bytearray)) else str(jid) for jid in r.smembers(key)]


def safe_enqueue(queue_name: str, func_path: str, *args, **kwargs):
    """
    Best-effort enqueue. Returns rq.job.Job or None if Redis is unavailable.

    func_path is an importable path like "src.jobs.persistence.store_session_summary".
    """
    try:
        q = get_queue(queue_name)
        return q.enqueue(func_path, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to enqueue job to queue='{queue_name}': {e}")
        return None
