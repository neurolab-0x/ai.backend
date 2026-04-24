import asyncio
import logging
from typing import Any, Dict, Optional

from src.services.database import db_service

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


def store_eeg_data(features: Dict[str, Any], subject_id: str, session_id: str) -> bool:
    return _run(db_service.store_eeg_data(features, subject_id, session_id))


def store_session_summary(session_data: Dict[str, Any]) -> Optional[str]:
    result_container = {"id": None}

    async def _inner():
        result_container["id"] = await db_service.store_session_summary(session_data)

    _run(_inner())
    return result_container["id"]


def store_voice_data(voice_results: Dict[str, Any], subject_id: str, session_id: str) -> Optional[str]:
    result_container = {"id": None}

    async def _inner():
        result_container["id"] = await db_service.store_voice_data(voice_results, subject_id, session_id)

    _run(_inner())
    return result_container["id"]

