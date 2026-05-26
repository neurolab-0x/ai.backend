import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

try:
    from rq import get_current_job
except ImportError:
    def get_current_job():
        return None

from src.queue import publish_job_event
from src.services.database import db_service
from src.services.reporting import (
    build_report_document,
    collect_report_context,
    generate_report_content,
    persist_report_artifact,
)

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


async def _update_report_run(job_id: str, updates: Dict[str, Any]) -> None:
    await db_service.update_report_run(job_id, updates)


async def _process_report_request(request: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    publish_job_event("reports", job_id, "started", {"stage": "retrieving_context"})
    await _update_report_run(job_id, {"status": "retrieving_context", "progress": 0.1, "message": "Retrieving report context"})

    context = await collect_report_context(
        subject_id=request["subject_id"],
        session_id=request.get("session_id"),
        start_time=request.get("start_time"),
        end_time=request.get("end_time"),
        lookback_days=int(request.get("lookback_days", 30)),
        include_sessions=bool(request.get("include_sessions", True)),
        include_chat=bool(request.get("include_chat", True)),
        limit=int(request.get("context_limit", 20)),
        external_context=request.get("external_context"),
    )
    context_counts = {
        "sessions": len(context.get("sessions", [])),
        "chat_exchanges": len(context.get("chat_exchanges", [])),
        "external_analyses": len(context.get("external_analysis_history", [])),
    }
    publish_job_event("reports", job_id, "context_retrieved", context_counts)
    await _update_report_run(
        job_id,
        {
            "status": "context_retrieved",
            "progress": 0.35,
            "message": "Context retrieved",
            "context_counts": context_counts,
        },
    )

    publish_job_event("reports", job_id, "generating_report", {"stage": "llm_inference"})
    await _update_report_run(job_id, {"status": "generating_report", "progress": 0.6, "message": "Generating report"})
    content = await generate_report_content(
        subject_id=request["subject_id"],
        report_type=request.get("report_type", "summary"),
        prompt=request.get("prompt"),
        context=context,
    )
    report_document = build_report_document(
        job_id=job_id,
        request=request,
        context=context,
        content=content,
    )

    publish_job_event("reports", job_id, "persisting_report", {"stage": "object_storage"})
    await _update_report_run(job_id, {"status": "persisting_report", "progress": 0.85, "message": "Persisting report artifact"})
    artifact = persist_report_artifact(report_document, job_id=job_id)
    report_document["artifact"] = artifact

    completed_payload = {
        "job_id": job_id,
        "status": "completed",
        "subject_id": request["subject_id"],
        "session_id": request.get("session_id"),
        "report_type": request.get("report_type", "summary"),
        "artifact": artifact,
        "summary": report_document["summary"],
        "context_counts": report_document["context_counts"],
        "generated_at": report_document["generated_at"],
    }
    await _update_report_run(
        job_id,
        {
            "status": "completed",
            "progress": 1.0,
            "message": "Report generation completed",
            "result": report_document,
            "artifacts": {"report": artifact},
            "summary": report_document["summary"],
            "context_counts": report_document["context_counts"],
            "error": None,
            "completed_at": datetime.now(),
        },
    )
    publish_job_event("reports", job_id, "completed", completed_payload)
    return report_document


def process_report_request(request: Dict[str, Any]) -> Dict[str, Any]:
    job = get_current_job()
    job_id = job.id if job else request.get("job_id") or "report-unknown"
    try:
        return _run(_process_report_request(request, job_id))
    except Exception as exc:
        logger.exception("Report job failed for job_id=%s", job_id)
        _run(
            _update_report_run(
                job_id,
                {
                    "status": "failed",
                    "progress": 1.0,
                    "message": f"Report generation failed: {exc}",
                    "error": str(exc),
                    "completed_at": datetime.now(),
                },
            )
        )
        publish_job_event("reports", job_id, "failed", {"error": str(exc)})
        raise
