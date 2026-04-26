import asyncio
import logging
from typing import Any, Dict, Optional

from rq import get_current_job

from src.queue import publish_job_event
from src.services.chat import generate_chat_exchange, retrieve_chat_context

logger = logging.getLogger(__name__)


def _run(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def _process_chat_request(request: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    publish_job_event("chat", job_id, "started", {"stage": "retrieving_context"})
    context = await retrieve_chat_context(
        subject_id=request.get("subject_id"),
        history=request.get("history"),
        include_health_data=bool(request.get("include_health_data", True)),
        limit=int(request.get("context_limit", 8)),
    )
    publish_job_event(
        "chat",
        job_id,
        "context_retrieved",
        {
            "history_items": len(context.get("history", [])),
            "health_history_items": len(context.get("health_history", [])),
        },
    )

    publish_job_event("chat", job_id, "generating_response", {"stage": "llm_inference"})
    result = await generate_chat_exchange(
        message=str(request.get("message", "")).strip(),
        subject_id=request.get("subject_id"),
        history=context.get("history"),
        current_title=request.get("current_title"),
        include_health_data=bool(request.get("include_health_data", True)),
        retrieval_context=context,
    )
    result["conversation_id"] = request.get("conversation_id")
    publish_job_event("chat", job_id, "completed", result)
    return result


def process_chat_request(request: Dict[str, Any]) -> Dict[str, Any]:
    job = get_current_job()
    job_id = job.id if job else request.get("job_id") or "chat-unknown"
    try:
        return _run(_process_chat_request(request, job_id))
    except Exception as exc:
        logger.exception("Chat job failed for job_id=%s", job_id)
        publish_job_event("chat", job_id, "failed", {"error": str(exc)})
        raise
