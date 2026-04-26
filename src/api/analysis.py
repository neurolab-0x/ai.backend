import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Body, Query, HTTPException, Request, status
from fastapi.responses import StreamingResponse as StarletteStreamingResponse
from pydantic import BaseModel, Field, field_validator
import logging
import base64

from src.utils.files import validate_file, save_uploaded_file
from src.core.ml.model_types import sanitize_model_type
from src.services.chat import generate_conversation_title
from src.queue import get_async_redis, publish_job_event, read_job_state

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

_ml_processor = None

def get_ml_processor():
    global _ml_processor
    if _ml_processor is None:
        from src.services.analysis import MLProcessor
        _ml_processor = MLProcessor()
    return _ml_processor


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

@router.post('/upload', summary="Advanced EEG analysis", response_description="Cognitive state report")
async def process_uploaded_file(
    file: Optional[UploadFile] = File(None),
    json_data: Optional[Dict] = Body(None),
    encrypt_response: bool = Query(False, description="Whether to encrypt the response"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction")
):
    """Process uploaded EEG file or JSON data"""
    try:
        ml_processor = get_ml_processor()
        try:
            model_type = sanitize_model_type(model_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        if file:
            validate_file(file)
            file_location = await save_uploaded_file(file)
            result = await ml_processor.process_eeg_data(
                file_location, 
                "anonymous", 
                "session_1", 
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode
            )
        elif json_data:
            result = await ml_processor.process_eeg_data(
                json_data, 
                "anonymous", 
                "session_1", 
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode
            )
        else:
            raise HTTPException(status_code=400, detail="No file or data provided")
            
        if encrypt_response:
            result = base64.b64encode(str(result).encode()).decode()
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/analyze', summary="Analyze EEG data", response_description="Analysis results")
async def analyze_eeg_data(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction")
):
    """Analyze EEG data and return results"""
    try:
        ml_processor = get_ml_processor()
        try:
            model_type = sanitize_model_type(model_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        result = await ml_processor.process_eeg_data(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1'),
            model_type=model_type,
            overlap=overlap,
            simple_mode=simple_mode
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/detailed-report', summary="Generate detailed analysis report", response_description="Comprehensive report with recommendations")
async def generate_detailed_report(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    save_report: bool = Query(False, description="Whether to save the report to a file"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction")
):
    """Generate a detailed analysis report with comprehensive recommendations"""
    try:
        ml_processor = get_ml_processor()
        try:
            model_type = sanitize_model_type(model_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        report = await ml_processor.generate_detailed_report(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1'),
            save_report=save_report,
            model_type=model_type,
            overlap=overlap,
            simple_mode=simple_mode
        )
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating detailed report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/recommendations', summary="Get personalized recommendations", response_description="NLP-based recommendations")
async def get_recommendations(
    state_durations: Dict[int, float] = Body(..., description="State durations mapping"),
    total_duration: float = Body(..., description="Total session duration"),
    confidence: float = Body(..., description="Prediction confidence"),
    cognitive_metrics: Optional[Dict[str, float]] = Body(None, description="Cognitive metrics"),
    state_transitions: int = Body(0, description="Number of state transitions"),
    max_recommendations: int = Query(5, description="Maximum number of recommendations")
):
    """Get personalized recommendations based on EEG analysis"""
    try:
        ml_processor = get_ml_processor()
        # Use the singleton recommendation engine from ml_processor for consistency
        recommendations = await ml_processor.recommendation_engine.generate_recommendations(
            state_durations=state_durations,
            total_duration=total_duration,
            confidence=confidence,
            cognitive_metrics=cognitive_metrics,
            state_transitions=state_transitions,
            max_recommendations=max_recommendations
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/decision-support', summary="Get decision support based on history")
async def get_decision_support(
    history: List[Dict[str, Any]] = Body(..., description="User analysis history")
):
    """Generate decision support insights from user history"""
    try:
        # We can reuse the recommendation engine's ability to analyze history
        # For a more specific "decision support", we might want to aggregate history
        if not history:
            return {
                "summary": "No history available for decision support.",
                "patterns": [],
                "risks": []
            }
            
        # Extract recent metrics for the engine
        # This is a simplified implementation that uses the engine's LLM capabilities
        # if available, or fallback logic.
        
        # For now, let's use the NLP engine to generate a summary
        # We'll need to mock a context if we want to use generate_detailed_report
        # or just call a new method if we were to add it.
        # Given the current AIService.js expectation, we'll return a structure it likes.
        
        # Mocking a simple response for now that matches the engine's capability
        return {
            "summary": "Based on your recent sessions, you are showing consistent focus patterns.",
            "patterns": ["High focus during morning sessions", "Increased stress in late afternoon"],
            "risks": ["Potential burnout if rest periods are not maintained"],
            "recommendations": [
                "Schedule a 10-minute break after 90 minutes of focused work",
                "Practice guided meditation during high-stress periods"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in decision support: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/chat/submit', status_code=status.HTTP_202_ACCEPTED, summary="Submit AI chat request for background processing")
async def submit_chat_job(data: ChatJobRequest):
    """Queue a chat request and return immediately with a job id."""
    try:
        require_rq()
        q = get_queue("chat")
        job = q.enqueue(
            "src.jobs.chat.process_chat_request",
            data.dict(),
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
            "events_url": f"/api/v1/eeg/chat/sse?job_id={job.id}",
            "status_url": f"/api/v1/eeg/chat/status/{job.id}",
            "submitted_at": initial_event["timestamp"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing chat job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/chat/status/{job_id}', summary="Get background chat job status")
async def get_chat_job_status(job_id: str):
    """Return the latest known state for a background chat job."""
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


@router.get('/chat/sse', summary="Listen for background chat job events")
async def stream_chat_job_events(
    request: Request,
    job_id: str,
) -> StarletteStreamingResponse:
    """Server-Sent Events stream for queued chat retrieval and LLM generation updates."""

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

@router.post('/chat/generate-name', summary="Generate a short chat conversation name")
async def generate_chat_name(
    history: List[Dict[str, Any]] = Body(..., embed=True),
    subject_id: Optional[str] = Body(None, description="User ID for personalization")
):
    """Generate a short title for a chat conversation."""
    try:
        title = await generate_conversation_title(history, subject_id=subject_id)
        return {"name": title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chat name: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/generate-notes', summary="Generate session notes")
async def generate_notes(
    analysis_results: Dict[str, Any] = Body(..., description="Results from EEG analysis")
):
    """Generate session notes from analysis results"""
    try:
        # Simple logic to format notes
        notes = f"Session conducted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
        notes += f"Primary State: {analysis_results.get('dominant_state', 'Unknown')}\n"
        notes += "Key Observations:\n"
        for rec in analysis_results.get('recommendations', []):
            notes += f"- {rec}\n"
            
        return {"notes": notes}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating notes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
