import os
import logging
import json
from datetime import datetime, timezone
from typing import Optional

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
try:
    from rq import Queue
    RQ_AVAILABLE = True
except ImportError:
    Queue = None
    RQ_AVAILABLE = False

logger = logging.getLogger(__name__)

_JOB_SET_PREFIX = "neurolab:rq:jobs:"
_JOB_STATE_PREFIX = "neurolab:job:state:"
_JOB_CHANNEL_PREFIX = "neurolab:job:events:"


def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> Redis:
    return Redis.from_url(redis_url())


def get_async_redis() -> AsyncRedis:
    return AsyncRedis.from_url(redis_url())


def get_queue(name: str = "default") -> Queue:
    if not RQ_AVAILABLE:
        raise RuntimeError("RQ is not installed")
    return Queue(name, connection=get_redis())


def _summarize_payload(payload: Optional[dict]) -> str:
    if not payload:
        return "payload={}"

    summary_parts = []
    for key in ("stage", "progress", "status", "message", "epoch", "total_epochs", "model_type", "subject_id", "session_id", "conversation_id"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            summary_parts.append(f"{key}={value}")

    if not summary_parts:
        summary_parts.append(f"keys={sorted(payload.keys())}")

    return ", ".join(summary_parts)


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


def _job_state_key(category: str, job_id: str) -> str:
    return f"{_JOB_STATE_PREFIX}{category}:{job_id}"


def job_event_channel(category: str, job_id: str) -> str:
    return f"{_JOB_CHANNEL_PREFIX}{category}:{job_id}"


def publish_job_event(
    category: str,
    job_id: str,
    event: str,
    payload: Optional[dict] = None,
    ttl_seconds: int = 60 * 60 * 24,
    persist_state: bool = True,
) -> dict:
    message = {
        "job_id": job_id,
        "category": category,
        "event": event,
        "payload": payload or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    raw_message = json.dumps(message, default=str)
    logger.info(
        "Worker event category=%s job_id=%s event=%s %s",
        category,
        job_id,
        event,
        _summarize_payload(payload),
    )
    try:
        r = get_redis()
        if persist_state:
            r.setex(_job_state_key(category, job_id), ttl_seconds, raw_message)
        r.publish(job_event_channel(category, job_id), raw_message)
    except Exception as e:
        logger.warning(f"Failed to publish event for {category} job_id={job_id}: {e}")
    return message


def read_job_state(category: str, job_id: str) -> Optional[dict]:
    try:
        raw = get_redis().get(_job_state_key(category, job_id))
        if not raw:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Failed to read state for {category} job_id={job_id}: {e}")
        return None


def safe_enqueue(queue_name: str, func_path: str, *args, **kwargs):
    """
    Best-effort enqueue. Returns rq.job.Job or None if Redis is unavailable.

    func_path is an importable path like "src.jobs.persistence.store_session_summary".
    """
    try:
        q = get_queue(queue_name)
        job = q.enqueue(func_path, *args, **kwargs)
        logger.info(
            "Enqueued background job queue=%s job_id=%s func=%s",
            queue_name,
            job.id,
            func_path,
        )
        return job
    except Exception as e:
        logger.warning(f"Failed to enqueue job to queue='{queue_name}': {e}")
        return None
