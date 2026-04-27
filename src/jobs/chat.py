import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from rq import get_current_job

from src.queue import publish_job_event
from src.services.chat import generate_chat_exchange, generate_conversation_title, retrieve_chat_context
from src.services.database import db_service

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
        auth_token=request.get("auth_token"),
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
        auth_token=request.get("auth_token"),
        history=context.get("history"),
        current_title=request.get("current_title"),
        include_health_data=bool(request.get("include_health_data", True)),
        retrieval_context=context,
        generate_title=False,
    )
    result["conversation_id"] = request.get("conversation_id")
    result["job_id"] = job_id
    result["status"] = "completed"
    result["message"] = str(request.get("message", "")).strip()
    result["subject_id"] = request.get("subject_id")
    result["timestamp"] = datetime.now()

    try:
        stored_id = await db_service.store_chat_exchange(result.copy())
        if stored_id:
            result["storage_id"] = stored_id
    except Exception:
        logger.exception("Failed to persist chat exchange for job_id=%s", job_id)

    publish_job_event("chat", job_id, "completed", result)

    if bool(request.get("generate_title", False)):
        try:
            title = await generate_conversation_title(
                [
                    *context.get("history", []),
                    {"role": "user", "content": str(request.get("message", "")).strip()},
                    {"role": "assistant", "content": result["response"]},
                ],
                subject_id=request.get("subject_id"),
                current_title=request.get("current_title"),
            )
            title_payload = {
                "job_id": job_id,
                "conversation_id": request.get("conversation_id"),
                "suggested_title": title,
                "should_update_title": bool(title and title != (request.get("current_title") or "")),
            }
            publish_job_event("chat", job_id, "title_generated", title_payload, persist_state=False)
        except Exception:
            logger.exception("Failed to generate title for chat job_id=%s", job_id)
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
