import asyncio
import json
import logging
import time
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
TERMINAL_RQ_STATUSES = {"finished", "failed", "stopped", "canceled"}
MAX_CHAT_MESSAGE_CHARS = 8000
MAX_CHAT_HISTORY_ITEMS = 20
MAX_CHAT_HISTORY_ITEM_CHARS = 4000
CHAT_STALLED_SECONDS = 30
CHAT_RQ_STATUS_POLL_SECONDS = 5


def require_rq() -> None:
    if not RQ_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat queue is unavailable because RQ is not installed.",
        )


def _get_rq_job_details(job_id: str) -> Dict[str, Any]:
    if not RQ_AVAILABLE:
        return {
            "status": None,
            "error": None,
            "queue": None,
            "enqueued_at": None,
            "started_at": None,
            "ended_at": None,
        }

    try:
        queue = get_queue("chat")
        job = Job.fetch(job_id, connection=queue.connection)
        return {
            "status": job.get_status(refresh=True),
            "error": job.exc_info,
            "queue": job.origin,
            "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        }
    except NoSuchJobError:
        return {
            "status": None,
            "error": None,
            "queue": None,
            "enqueued_at": None,
            "started_at": None,
            "ended_at": None,
        }
    except Exception as exc:
        logger.warning("Failed to inspect RQ job %s: %s", job_id, exc)
        return {
            "status": None,
            "error": str(exc),
            "queue": None,
            "enqueued_at": None,
            "started_at": None,
            "ended_at": None,
        }


class ChatJobRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS, description="User message to send to the assistant")
    subject_id: Optional[str] = Field(None, description="User ID for personalized retrieval")
    conversation_id: Optional[str] = Field(None, description="Backend conversation identifier")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Recent conversation history")
    current_title: Optional[str] = Field(None, description="Current conversation title")
    auth_token: Optional[str] = Field(None, description="Internal bearer token forwarded by the backend")
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
        auth_token=data.auth_token,
        history=data.history,
        include_health_data=data.include_health_data,
        limit=data.context_limit,
    )
    result = await generate_chat_exchange(
        message=data.message,
        subject_id=data.subject_id,
        auth_token=data.auth_token,
        history=context.get("history"),
        current_title=data.current_title,
        include_health_data=data.include_health_data,
        retrieval_context=context,
        generate_title=data.generate_title,
    )
    result["conversation_id"] = data.conversation_id
    return result


@router.post('/', summary="Generate AI chat response synchronously")
async def generate_chat_response(data: ChatJobRequest, request: Request):
    """Generate an immediate chat response using the shared chat service."""
    try:
        data.auth_token = data.auth_token or request.headers.get("authorization")
        return await build_sync_chat_response(data)
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Chat generation unavailable: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error generating synchronous chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/submit', status_code=status.HTTP_202_ACCEPTED, summary="Submit AI chat request for background processing")
async def submit_chat_job(data: ChatJobRequest, request: Request):
    """Queue a chat request and return immediately with a job id."""
    try:
        data.auth_token = data.auth_token or request.headers.get("authorization")
        require_rq()
        q = get_queue("chat")
        job = q.enqueue(
            "src.jobs.chat.process_chat_request",
            data.model_dump(),
            job_timeout=60 * 10,
            result_ttl=60 * 60 * 24,
        )
        logger.info(
            "Queued chat job job_id=%s queue=%s conversation_id=%s subject_id=%s queue_size=%s",
            job.id,
            q.name,
            data.conversation_id,
            data.subject_id,
            len(q.job_ids),
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
    rq_details = _get_rq_job_details(job_id)
    rq_status = rq_details["status"]

    if state is None and rq_status is None:
        raise HTTPException(status_code=404, detail=f"Chat job {job_id} not found")

    if state is None:
        logger.warning(
            "Chat job has no published state yet job_id=%s rq_status=%s queue=%s",
            job_id,
            rq_status,
            rq_details["queue"],
        )
    if rq_status == "failed":
        logger.error(
            "Chat job failed in RQ job_id=%s error=%s",
            job_id,
            rq_details["error"],
        )

    return {
        "job_id": job_id,
        "status": (state or {}).get("event") or rq_status,
        "rq_status": rq_status,
        "state": state,
        "rq_details": rq_details,
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
        last_status_poll = 0.0
        last_non_ping_event_at = time.monotonic()
        last_rq_status = None
        stalled_emitted = False
        try:
            logger.info("Opening chat SSE stream job_id=%s channel=%s", job_id, channel)
            await pubsub.subscribe(channel)
            state = read_job_state("chat", job_id)
            rq_details = _get_rq_job_details(job_id)
            last_rq_status = rq_details["status"]
            initial_payload = {
                "job_id": job_id,
                "state": state,
                "rq_status": rq_details["status"],
                "rq_details": rq_details,
            }
            logger.info(
                "Chat SSE initial state job_id=%s rq_status=%s has_state=%s",
                job_id,
                rq_details["status"],
                bool(state),
            )
            yield f"event: ready\ndata: {json.dumps(initial_payload, default=str)}\n\n".encode("utf-8")
            if state and state.get("event") in TERMINAL_CHAT_EVENTS:
                yield f"event: {state['event']}\ndata: {json.dumps(state, default=str)}\n\n".encode("utf-8")
                return
            if rq_details["status"] == "failed":
                failed_payload = {
                    "job_id": job_id,
                    "error": rq_details["error"] or "RQ job failed before publishing state",
                    "rq_status": rq_details["status"],
                }
                logger.error("Chat SSE immediate failure job_id=%s error=%s", job_id, failed_payload["error"])
                yield f"event: failed\ndata: {json.dumps(failed_payload, default=str)}\n\n".encode("utf-8")
                return

            while True:
                if await request.is_disconnected():
                    logger.info("Chat SSE client disconnected job_id=%s", job_id)
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if message and message.get("data"):
                    data = message["data"]
                    if isinstance(data, (bytes, bytearray)):
                        data = data.decode("utf-8")
                    payload = json.loads(data)
                    event_name = payload.get("event", "message")
                    last_non_ping_event_at = time.monotonic()
                    stalled_emitted = False
                    logger.info("Chat SSE event job_id=%s event=%s", job_id, event_name)
                    yield f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
                    if event_name in TERMINAL_CHAT_EVENTS:
                        break
                else:
                    now = time.monotonic()
                    if now - last_status_poll >= CHAT_RQ_STATUS_POLL_SECONDS:
                        last_status_poll = now
                        rq_details = _get_rq_job_details(job_id)
                        rq_status = rq_details["status"]
                        if rq_status != last_rq_status:
                            last_rq_status = rq_status
                            logger.info(
                                "Chat SSE RQ status update job_id=%s rq_status=%s",
                                job_id,
                                rq_status,
                            )
                            rq_payload = {
                                "job_id": job_id,
                                "rq_status": rq_status,
                                "rq_details": rq_details,
                            }
                            yield f"event: rq_status\ndata: {json.dumps(rq_payload, default=str)}\n\n".encode("utf-8")
                        if rq_status == "failed":
                            failed_payload = {
                                "job_id": job_id,
                                "error": rq_details["error"] or "RQ job failed before publishing terminal state",
                                "rq_status": rq_status,
                            }
                            logger.error("Chat SSE detected RQ failure job_id=%s error=%s", job_id, failed_payload["error"])
                            yield f"event: failed\ndata: {json.dumps(failed_payload, default=str)}\n\n".encode("utf-8")
                            break
                        if (
                            not stalled_emitted
                            and now - last_non_ping_event_at >= CHAT_STALLED_SECONDS
                            and rq_status not in TERMINAL_RQ_STATUSES
                        ):
                            stalled_emitted = True
                            stalled_payload = {
                                "job_id": job_id,
                                "message": "Chat job has not emitted progress updates recently",
                                "rq_status": rq_status,
                                "rq_details": rq_details,
                                "stalled_seconds": CHAT_STALLED_SECONDS,
                            }
                            logger.warning(
                                "Chat SSE stalled job_id=%s rq_status=%s stalled_seconds=%s",
                                job_id,
                                rq_status,
                                CHAT_STALLED_SECONDS,
                            )
                            yield f"event: stalled\ndata: {json.dumps(stalled_payload, default=str)}\n\n".encode("utf-8")
                    yield b"event: ping\ndata: {}\n\n"
                await asyncio.sleep(0.05)
        finally:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                logger.debug("Chat SSE unsubscribe failed for job %s", job_id)
            await pubsub.close()
            await redis.close()
            logger.info("Closed chat SSE stream job_id=%s", job_id)

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
    except RuntimeError as e:
        logger.error(f"Chat title generation unavailable: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error generating chat name: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
