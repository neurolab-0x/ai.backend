import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Request, status
from fastapi.responses import StreamingResponse as StarletteStreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.queue import get_async_redis, publish_job_event, read_job_state
from src.services.chat import (
    generate_chat_exchange,
    generate_conversation_title,
    retrieve_chat_context,
)
from src.utils.validation import require_safe_id_or_400, validate_optional_safe_id

try:
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    from src.queue import get_queue, track_job
    RQ_AVAILABLE = True
except ImportError:
    Job = None

    class NoSuchJobError(Exception):
        pass

    RQ_AVAILABLE = False
    get_queue = None
    track_job = None

logger = logging.getLogger(__name__)
router = APIRouter()

TERMINAL_CHAT_EVENTS = {"completed", "failed"}
MAX_CHAT_MESSAGE_CHARS = 8000
MAX_CHAT_HISTORY_ITEMS = 20
MAX_CHAT_HISTORY_ITEM_CHARS = 4000


def require_rq() -> None:
    if not RQ_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat queue is unavailable because RQ is not installed.",
        )


class ChatJobRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS, description="User message to send to the assistant")
    subject_id: Optional[str] = Field(None, description="User ID for personalized retrieval")
    conversation_id: Optional[str] = Field(None, description="Backend conversation identifier")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Recent conversation history")
    current_title: Optional[str] = Field(None, description="Current conversation title")
    include_health_data: bool = Field(True, description="Whether health history should be retrieved")
    context_limit: int = Field(8, ge=1, le=25, description="How many history items to retrieve from storage")
    generate_title: bool = Field(False, description="Generate a suggested title after the main answer is ready")

    @field_validator("history")
    @classmethod
    def validate_history(cls, value: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if value is None:
            return value
        if len(value) > MAX_CHAT_HISTORY_ITEMS:
            raise ValueError(f"history cannot exceed {MAX_CHAT_HISTORY_ITEMS} items")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("history items must be objects")
            content = str(item.get("content", ""))
            if len(content) > MAX_CHAT_HISTORY_ITEM_CHARS:
                raise ValueError(
                    f"history item content cannot exceed {MAX_CHAT_HISTORY_ITEM_CHARS} characters"
                )
        return value

    @field_validator("subject_id", "conversation_id")
    @classmethod
    def validate_optional_ids(cls, value: Optional[str], info):
        return validate_optional_safe_id(value, info.field_name)

    @field_validator("current_title")
    @classmethod
    def validate_title(cls, value: Optional[str]):
        if value is None:
            return value
        return value[:200]


async def build_sync_chat_response(data: ChatJobRequest) -> Dict[str, Any]:
    context = await retrieve_chat_context(
        subject_id=data.subject_id,
        history=data.history,
        include_health_data=data.include_health_data,
        limit=data.context_limit,
    )
    result = await generate_chat_exchange(
        message=data.message,
        subject_id=data.subject_id,
        history=context.get("history"),
        current_title=data.current_title,
        include_health_data=data.include_health_data,
        retrieval_context=context,
        generate_title=data.generate_title,
    )
    result["conversation_id"] = data.conversation_id
    return result


@router.post('/', summary="Generate AI chat response synchronously")
async def generate_chat_response(data: ChatJobRequest):
    """Generate an immediate chat response using the shared chat service."""
    try:
        return await build_sync_chat_response(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating synchronous chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/submit', status_code=status.HTTP_202_ACCEPTED, summary="Submit AI chat request for background processing")
async def submit_chat_job(data: ChatJobRequest):
    """Queue a chat request and return immediately with a job id."""
    try:
        require_rq()
        q = get_queue("chat")
        job = q.enqueue(
            "src.jobs.chat.process_chat_request",
            data.model_dump(),
            job_timeout=60 * 10,
            result_ttl=60 * 60 * 24,
        )
        track_job("chat", job.id)
        initial_event = publish_job_event(
            "chat",
            job.id,
            "queued",
            {
                "conversation_id": data.conversation_id,
                "subject_id": data.subject_id,
            },
        )
        return {
            "job_id": job.id,
            "status": "queued",
            "conversation_id": data.conversation_id,
            "events_url": f"/api/v1/chat/sse?job_id={job.id}",
            "status_url": f"/api/v1/chat/status/{job.id}",
            "submitted_at": initial_event["timestamp"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing chat job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/status/{job_id}', summary="Get background chat job status")
async def get_chat_job_status(job_id: str):
    """Return the latest known state for a background chat job."""
    job_id = require_safe_id_or_400(job_id, "job_id")
    state = read_job_state("chat", job_id)
    rq_status = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("chat").connection)
            rq_status = job.get_status(refresh=True)
        except NoSuchJobError:
            rq_status = None
        except Exception as exc:
            logger.warning(f"Failed to read RQ status for chat job {job_id}: {exc}")

    if state is None and rq_status is None:
        raise HTTPException(status_code=404, detail=f"Chat job {job_id} not found")

    return {
        "job_id": job_id,
        "status": (state or {}).get("event") or rq_status,
        "rq_status": rq_status,
        "state": state,
    }


@router.get('/sse', summary="Listen for background chat job events")
async def stream_chat_job_events(
    request: Request,
    job_id: str,
) -> StarletteStreamingResponse:
    """Server-Sent Events stream for queued chat retrieval and LLM generation updates."""
    job_id = require_safe_id_or_400(job_id, "job_id")

    async def event_gen():
        redis = get_async_redis()
        pubsub = redis.pubsub()
        channel = f"neurolab:job:events:chat:{job_id}"
        try:
            await pubsub.subscribe(channel)
            state = read_job_state("chat", job_id)
            yield f"event: ready\ndata: {json.dumps({'job_id': job_id, 'state': state}, default=str)}\n\n".encode("utf-8")
            if state and state.get("event") in TERMINAL_CHAT_EVENTS:
                yield f"event: {state['event']}\ndata: {json.dumps(state, default=str)}\n\n".encode("utf-8")
                return

            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if message and message.get("data"):
                    data = message["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8")
                    payload = json.loads(data)
                    event_name = payload.get("event", "message")
                    yield f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
                    if event_name in TERMINAL_CHAT_EVENTS:
                        break
                else:
                    yield b"event: ping\ndata: {}\n\n"
                await asyncio.sleep(0.05)
        finally:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                logger.debug("Chat SSE unsubscribe failed for job %s", job_id)
            await pubsub.close()
            await redis.close()

    return StarletteStreamingResponse(event_gen(), media_type="text/event-stream")


@router.post('/generate-name', summary="Generate a short chat conversation name")
async def generate_chat_name(
    history: List[Dict[str, Any]] = Body(..., embed=True),
    subject_id: Optional[str] = Body(None, description="User ID for personalization")
):
    """Generate a short title for a chat conversation."""
    try:
        subject_id = validate_optional_safe_id(subject_id, "subject_id")
        title = await generate_conversation_title(history, subject_id=subject_id)
        return {"name": title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chat name: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
