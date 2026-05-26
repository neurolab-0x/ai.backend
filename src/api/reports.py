import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

try:
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    from src.queue import get_queue, track_job, list_tracked_jobs, untrack_job, publish_job_event, read_job_state, get_async_redis
    RQ_AVAILABLE = True
except ImportError:
    Job = None

    class NoSuchJobError(Exception):
        pass

    RQ_AVAILABLE = False
    get_queue = None
    track_job = None
    list_tracked_jobs = None
    untrack_job = None
    publish_job_event = None
    read_job_state = None
    get_async_redis = None

from src.services.database import db_service
from src.services.storage import MinioStorageService
from src.utils.validation import require_safe_id_or_400, validate_optional_safe_id

logger = logging.getLogger(__name__)
router = APIRouter()
_storage_service: Optional[MinioStorageService] = None


def require_rq() -> None:
    if not RQ_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report queue is unavailable because RQ is not installed.",
        )


class ReportJobRequest(BaseModel):
    subject_id: str = Field(..., description="Subject identifier for the report")
    session_id: Optional[str] = Field(None, description="Optional session identifier to narrow the report scope")
    report_type: str = Field(default="summary", description="Report type label")
    prompt: Optional[str] = Field(None, description="Optional custom instructions for the report")
    start_time: Optional[datetime] = Field(None, description="Optional inclusive UTC start time")
    end_time: Optional[datetime] = Field(None, description="Optional inclusive UTC end time")
    lookback_days: int = Field(default=30, ge=1, le=365, description="Fallback lookback window if explicit times are not provided")
    include_sessions: bool = Field(default=True)
    include_chat: bool = Field(default=True)
    context_limit: int = Field(default=20, ge=1, le=100)
    external_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional backend-supplied user/analysis context to enrich the report",
    )

    @validator("subject_id")
    def validate_subject_id(cls, value: str) -> str:
        return require_safe_id_or_400(value, "subject_id")

    @validator("session_id")
    def validate_session_id(cls, value: Optional[str]) -> Optional[str]:
        return validate_optional_safe_id(value, "session_id")


class ReportJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    submitted_at: str
    events_url: str
    status_url: str


class ReportJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    context_counts: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class ReportRunDetail(ReportJobStatus):
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    report_type: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def _serialize_dt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _get_storage_service() -> MinioStorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = MinioStorageService()
    return _storage_service


def _hydrate_artifacts(artifacts: Any) -> Any:
    if not isinstance(artifacts, (dict, list)):
        return artifacts
    return _get_storage_service().hydrate_artifact_urls(artifacts)


def _build_status(job: Optional[Job], persisted_run: Optional[Dict[str, Any]]) -> ReportJobStatus:
    state = persisted_run or {}
    rq_status = job.get_status() if job is not None else None
    return ReportJobStatus(
        job_id=state.get("job_id") or (job.id if job else ""),
        status=state.get("status") or rq_status or "unknown",
        progress=float(state.get("progress", job.meta.get("progress", 0.0) if job else 0.0)),
        message=str(state.get("message") or (job.meta.get("message") if job else None) or "unknown"),
        started_at=_serialize_dt(state.get("created_at") or state.get("started_at") or (job.enqueued_at if job else datetime.now())),
        completed_at=_serialize_dt(state.get("completed_at") or (job.ended_at if job else None)),
        error=state.get("error") or (str(job.exc_info) if job is not None and job.is_failed else None),
        artifacts=_hydrate_artifacts(state.get("artifacts")),
        summary=state.get("summary"),
        context_counts=state.get("context_counts"),
        config=state.get("config"),
    )


def _build_detail(job: Optional[Job], persisted_run: Dict[str, Any]) -> ReportRunDetail:
    base = _build_status(job, persisted_run)
    return ReportRunDetail(
        **base.model_dump(),
        subject_id=persisted_run.get("subject_id"),
        session_id=persisted_run.get("session_id"),
        report_type=persisted_run.get("report_type"),
        created_at=_serialize_dt(persisted_run.get("created_at")),
        updated_at=_serialize_dt(persisted_run.get("updated_at")),
        result=_json_safe(persisted_run.get("result")) if persisted_run.get("result") else None,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return str(value)


@router.post("/submit", response_model=ReportJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_report_job(data: ReportJobRequest):
    require_rq()
    job_id = f"report_{uuid4().hex}"
    run_record = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Report job queued",
        "subject_id": data.subject_id,
        "session_id": data.session_id,
        "report_type": data.report_type,
        "config": data.model_dump(),
        "artifacts": {},
        "summary": None,
        "context_counts": None,
        "result": None,
        "error": None,
        "started_at": datetime.now(),
        "completed_at": None,
    }
    await db_service.create_report_run(run_record)

    request_payload = data.model_dump()
    if request_payload.get("start_time") is not None:
        request_payload["start_time"] = request_payload["start_time"]
    if request_payload.get("end_time") is not None:
        request_payload["end_time"] = request_payload["end_time"]

    q = get_queue("reports")
    job = q.enqueue(
        "src.jobs.reports.process_report_request",
        request_payload,
        job_id=job_id,
        job_timeout=60 * 60,
        result_ttl=60 * 60 * 24,
    )
    track_job("reports", job.id)
    event = publish_job_event(
        "reports",
        job.id,
        "queued",
        {"subject_id": data.subject_id, "session_id": data.session_id, "report_type": data.report_type},
    )
    return ReportJobResponse(
        job_id=job.id,
        status="queued",
        message="Report job queued",
        submitted_at=event["timestamp"],
        events_url=f"/api/v1/reports/sse?job_id={job.id}",
        status_url=f"/api/v1/reports/status/{job.id}",
    )


@router.get("/status/{job_id}", response_model=ReportJobStatus)
async def get_report_status(job_id: str):
    job_id = require_safe_id_or_400(job_id, "job_id")
    persisted_run = await db_service.get_report_run(job_id)
    job = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("reports").connection)
        except NoSuchJobError:
            job = None
    if persisted_run is None and job is None:
        raise HTTPException(status_code=404, detail=f"Report job {job_id} not found")
    return _build_status(job, persisted_run)


@router.get("/jobs", response_model=List[ReportJobStatus])
async def list_report_jobs(limit: int = 10):
    persisted_runs = await db_service.list_report_runs(limit=max(limit, 20))
    jobs: List[ReportJobStatus] = []
    tracked_job_ids = list_tracked_jobs("reports") if RQ_AVAILABLE else []
    seen = {run.get("job_id") for run in persisted_runs if run.get("job_id")}
    ordered_ids = [run["job_id"] for run in persisted_runs if run.get("job_id")] + [jid for jid in tracked_job_ids if jid not in seen]
    for jid in ordered_ids:
        persisted_run = next((run for run in persisted_runs if run.get("job_id") == jid), None)
        job = None
        if RQ_AVAILABLE:
            try:
                job = Job.fetch(jid, connection=get_queue("reports").connection)
            except Exception:
                job = None
        if job is None and persisted_run is None:
            continue
        jobs.append(_build_status(job, persisted_run))
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return jobs[:limit]


@router.delete("/job/{job_id}")
async def archive_report_job(job_id: str):
    job_id = require_safe_id_or_400(job_id, "job_id")
    job = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("reports").connection)
        except NoSuchJobError:
            job = None
    persisted_run = await db_service.get_report_run(job_id)
    if job is None and persisted_run is None:
        raise HTTPException(status_code=404, detail=f"Report job {job_id} not found")
    if job is not None:
        job.delete()
    if RQ_AVAILABLE:
        untrack_job("reports", job_id)
    await db_service.archive_report_run(job_id, reason="archived_by_user")
    return {"status": "success", "message": f"Report job {job_id} archived"}


@router.get("/runs/{job_id}", response_model=ReportRunDetail)
async def get_report_run_detail(job_id: str):
    job_id = require_safe_id_or_400(job_id, "job_id")
    persisted_run = await db_service.get_report_run(job_id)
    if persisted_run is None:
        raise HTTPException(status_code=404, detail=f"Report run {job_id} not found")
    job = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("reports").connection)
        except Exception:
            job = None
    return _build_detail(job, persisted_run)


@router.get("/history", response_model=List[ReportRunDetail])
async def get_report_history(limit: int = 20, include_archived: bool = False):
    persisted_runs = await db_service.list_report_runs(limit=limit, include_archived=include_archived)
    details: List[ReportRunDetail] = []
    for run in persisted_runs:
        job = None
        if RQ_AVAILABLE:
            try:
                job = Job.fetch(run["job_id"], connection=get_queue("reports").connection)
            except Exception:
                job = None
        details.append(_build_detail(job, run))
    return details


@router.get("/runs/{job_id}/artifacts")
async def get_report_artifacts(job_id: str):
    job_id = require_safe_id_or_400(job_id, "job_id")
    run = await db_service.get_report_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Report run {job_id} not found")
    return {"job_id": job_id, "artifacts": _hydrate_artifacts(run.get("artifacts") or {})}


@router.get("/sse")
async def stream_report_events(request: Request, job_id: str):
    job_id = require_safe_id_or_400(job_id, "job_id")
    if get_async_redis is None:
        raise HTTPException(status_code=503, detail="Report event stream is unavailable")

    async def event_gen():
        redis = get_async_redis()
        pubsub = redis.pubsub()
        channel = f"neurolab:job:events:reports:{job_id}"
        try:
            await pubsub.subscribe(channel)
            state = read_job_state("reports", job_id) if read_job_state else None
            yield f"event: ready\ndata: {json.dumps({'job_id': job_id, 'state': state}, default=str)}\n\n".encode("utf-8")
            if state and state.get("event") in {"completed", "failed"}:
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
                    if event_name in {"completed", "failed"}:
                        break
                else:
                    yield b"event: ping\ndata: {}\n\n"
                await asyncio.sleep(0.05)
        finally:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                logger.debug("Report SSE unsubscribe failed for job %s", job_id)
            await pubsub.close()
            await redis.close()

    return StreamingResponse(event_gen(), media_type="text/event-stream")
