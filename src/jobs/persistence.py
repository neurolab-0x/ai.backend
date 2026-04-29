import asyncio
import logging
from typing import Any, Dict, Optional

from src.services.database import db_service

try:
    from rq import get_current_job
except ImportError:
    def get_current_job():
        return None

logger = logging.getLogger(__name__)


def _run(coro):
    try:
        asyncio.run(coro)
        return True
    except RuntimeError:
        # Already in an event loop: create a new loop explicitly.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
            return True
        finally:
            loop.close()


def _job_id() -> str:
    job = get_current_job()
    return job.id if job else "persistence-unknown"


def store_eeg_data(features: Dict[str, Any], subject_id: str, session_id: str) -> bool:
    job_id = _job_id()
    logger.info(
        "Persistence job started job_id=%s task=store_eeg_data subject_id=%s session_id=%s fields=%s",
        job_id,
        subject_id,
        session_id,
        sorted(features.keys()),
    )
    result = _run(db_service.store_eeg_data(features, subject_id, session_id))
    logger.info(
        "Persistence job completed job_id=%s task=store_eeg_data success=%s",
        job_id,
        result,
    )
    return result


def store_session_summary(session_data: Dict[str, Any]) -> Optional[str]:
    job_id = _job_id()
    logger.info(
        "Persistence job started job_id=%s task=store_session_summary subject_id=%s session_id=%s",
        job_id,
        session_data.get("subject_id"),
        session_data.get("session_id"),
    )
    result_container = {"id": None}

    async def _inner():
        result_container["id"] = await db_service.store_session_summary(session_data)

    _run(_inner())
    logger.info(
        "Persistence job completed job_id=%s task=store_session_summary record_id=%s",
        job_id,
        result_container["id"],
    )
    return result_container["id"]


def store_voice_data(voice_results: Dict[str, Any], subject_id: str, session_id: str) -> Optional[str]:
    job_id = _job_id()
    logger.info(
        "Persistence job started job_id=%s task=store_voice_data subject_id=%s session_id=%s",
        job_id,
        subject_id,
        session_id,
    )
    result_container = {"id": None}

    async def _inner():
        result_container["id"] = await db_service.store_voice_data(voice_results, subject_id, session_id)

    _run(_inner())
    logger.info(
        "Persistence job completed job_id=%s task=store_voice_data record_id=%s",
        job_id,
        result_container["id"],
    )
    return result_container["id"]
