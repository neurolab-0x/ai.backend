import asyncio
from typing import Any, Dict, List, Optional

from src.services.database import db_service


def _run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def create_training_run(run_data: Dict[str, Any]) -> Optional[str]:
    return _run(db_service.create_training_run(run_data))


def update_training_run(job_id: str, updates: Dict[str, Any]) -> bool:
    return _run(db_service.update_training_run(job_id, updates))


def get_training_run(job_id: str) -> Optional[Dict[str, Any]]:
    return _run(db_service.get_training_run(job_id))


def list_training_runs(limit: int = 20) -> List[Dict[str, Any]]:
    return _run(db_service.list_training_runs(limit=limit))


def delete_training_run(job_id: str) -> bool:
    return _run(db_service.delete_training_run(job_id))
