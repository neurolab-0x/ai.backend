import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.services.database import db_service
from src.services.llm import get_async_llm_client
from src.services.storage import MinioStorageService

logger = logging.getLogger(__name__)

REPORT_DISCLAIMER = (
    "This report is for wellness tracking and operational review only. "
    "It is non-diagnostic and should not replace professional medical evaluation."
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


async def collect_report_context(
    *,
    subject_id: str,
    session_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    lookback_days: int = 30,
    include_sessions: bool = True,
    include_training: bool = True,
    include_chat: bool = True,
    limit: int = 20,
    external_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    end = end_time or datetime.now()
    start = start_time or (end - timedelta(days=max(1, lookback_days)))
    context: Dict[str, Any] = {
        "subject_id": subject_id,
        "session_id": session_id,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "lookback_days": lookback_days,
        "sessions": [],
        "training_runs": [],
        "chat_exchanges": [],
        "external_subject_profile": {},
        "external_analysis_history": [],
    }

    if isinstance(external_context, dict):
        subject_profile = external_context.get("subject_profile")
        analysis_history = external_context.get("analysis_history")
        if isinstance(subject_profile, dict):
            context["external_subject_profile"] = subject_profile
        if isinstance(analysis_history, list):
            context["external_analysis_history"] = analysis_history[:limit]

    if include_sessions:
        context["sessions"] = await db_service.get_session_summaries(
            subject_id=subject_id,
            session_id=session_id,
            start_time=start,
            end_time=end,
            limit=limit,
        )

    if include_training:
        context["training_runs"] = await db_service.get_training_runs_for_subject(
            subject_id=subject_id,
            start_time=start,
            end_time=end,
            limit=limit,
        )

    if include_chat:
        context["chat_exchanges"] = await db_service.get_chat_exchanges(
            subject_id=subject_id,
            start_time=start,
            end_time=end,
            limit=limit,
        )

    return context


def build_report_summary(context: Dict[str, Any]) -> Dict[str, Any]:
    sessions = context.get("sessions", [])
    training_runs = context.get("training_runs", [])
    chat_exchanges = context.get("chat_exchanges", [])

    confidence_values = [
        float(item.get("confidence"))
        for item in sessions
        if isinstance(item.get("confidence"), (int, float))
    ]
    state_counts: Dict[str, int] = {}
    for item in sessions:
        state = str(item.get("dominant_state", "")).strip()
        if state:
            state_counts[state] = state_counts.get(state, 0) + 1

    summary = {
        "session_count": len(sessions),
        "training_run_count": len(training_runs),
        "chat_exchange_count": len(chat_exchanges),
        "external_analysis_count": len(context.get("external_analysis_history", [])),
        "average_session_confidence": (
            round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else None
        ),
        "dominant_state_counts": state_counts,
    }
    return summary


def _format_sessions(sessions: List[Dict[str, Any]]) -> str:
    if not sessions:
        return "No session summaries found in the requested time range."
    lines = []
    for item in sessions[:20]:
        timestamp = item.get("timestamp")
        ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        lines.append(
            f"- {ts}: session={item.get('session_id')} state={item.get('dominant_state')} "
            f"confidence={item.get('confidence')} state_percentages={item.get('state_percentages')}"
        )
    return "\n".join(lines)


def _format_training_runs(training_runs: List[Dict[str, Any]]) -> str:
    if not training_runs:
        return "No completed training runs found in the requested time range."
    lines = []
    for item in training_runs[:20]:
        metrics = item.get("metrics") or {}
        lines.append(
            f"- {item.get('created_at')}: job_id={item.get('job_id')} model_type={item.get('model_type')} "
            f"train_accuracy={metrics.get('final_train_accuracy')} val_accuracy={metrics.get('final_val_accuracy')}"
        )
    return "\n".join(lines)


def _format_chat_exchanges(chat_exchanges: List[Dict[str, Any]]) -> str:
    if not chat_exchanges:
        return "No chat exchanges found in the requested time range."
    lines = []
    for item in chat_exchanges[:20]:
        timestamp = item.get("timestamp")
        ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        message = str(item.get("message", "")).strip().replace("\n", " ")
        response = str(item.get("response", "")).strip().replace("\n", " ")
        lines.append(f"- {ts}: user='{message[:120]}' assistant='{response[:160]}'")
    return "\n".join(lines)


def _format_external_subject_profile(subject_profile: Dict[str, Any]) -> str:
    if not subject_profile:
        return "No backend subject profile supplied."

    lines = []
    for key, label in (
        ("full_name", "Full name"),
        ("username", "Username"),
        ("email", "Email"),
        ("phone", "Phone"),
        ("role", "Role"),
        ("address", "Address"),
        ("member_since", "Member since"),
    ):
        value = subject_profile.get(key)
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "No backend subject profile supplied."


def _format_external_analysis_history(analysis_history: List[Dict[str, Any]]) -> str:
    if not analysis_history:
        return "No backend analysis history supplied."

    lines = []
    for item in analysis_history[:20]:
        timestamp = item.get("timestamp")
        ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        lines.append(
            f"- {ts}: analysis={item.get('id')} status={item.get('status')} "
            f"notes={item.get('ai_notes') or 'none'} results={item.get('results') or {}}"
        )
    return "\n".join(lines)


async def generate_report_content(
    *,
    subject_id: str,
    report_type: str,
    prompt: Optional[str],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    summary = build_report_summary(context)
    client = get_async_llm_client()

    if not client.enabled:
        return {
            "title": f"{report_type.replace('_', ' ').title()} Report",
            "executive_summary": (
                f"Offline report for subject {subject_id}. "
                f"Collected {summary['session_count']} sessions, {summary['training_run_count']} training runs, "
                f"and {summary['chat_exchange_count']} chat exchanges."
            ),
            "key_findings": [
                f"Dominant state distribution: {summary['dominant_state_counts'] or 'no session data available'}",
                f"Average session confidence: {summary['average_session_confidence']}",
            ],
            "recommendations": [
                "Review the requested time window with additional domain context before drawing conclusions.",
                "Use this report as operational and wellness support, not as diagnosis.",
            ],
        }

    instructions = prompt.strip() if prompt else (
        "Generate a concise, high-signal non-diagnostic report focused on trends, changes, and operationally useful recommendations."
    )
    report_prompt = f"""
You are the NeuroLab reporting assistant.
Generate a JSON report for subject '{subject_id}'.

Constraints:
- The report must be non-diagnostic.
- Focus on trend analysis, operational observations, and cautious recommendations.
- Do not claim clinical diagnosis.
- Return valid JSON only.

Requested report type: {report_type}
Custom instructions: {instructions}

Summary statistics:
{json.dumps(summary, default=str)}

Session summaries:
{_format_sessions(context.get("sessions", []))}

Training runs:
{_format_training_runs(context.get("training_runs", []))}

Chat exchanges:
{_format_chat_exchanges(context.get("chat_exchanges", []))}

Backend subject profile:
{_format_external_subject_profile(context.get("external_subject_profile", {}))}

Backend analysis history:
{_format_external_analysis_history(context.get("external_analysis_history", []))}

Return JSON with this shape:
{{
  "title": "string",
  "executive_summary": "string",
  "key_findings": ["string"],
  "recommendations": ["string"]
}}
"""
    raw = await client.create_chat_completion(
        messages=[{"role": "user", "content": report_prompt}],
        temperature=0.3,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    payload = json.loads(raw)
    payload.setdefault("title", f"{report_type.replace('_', ' ').title()} Report")
    payload.setdefault("executive_summary", "")
    payload.setdefault("key_findings", [])
    payload.setdefault("recommendations", [])
    return payload


def build_report_document(
    *,
    job_id: str,
    request: Dict[str, Any],
    context: Dict[str, Any],
    content: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "report_type": request.get("report_type", "summary"),
        "subject_id": request.get("subject_id"),
        "session_id": request.get("session_id"),
        "time_window": {
            "start_time": context.get("start_time"),
            "end_time": context.get("end_time"),
            "lookback_days": context.get("lookback_days"),
        },
        "summary": build_report_summary(context),
        "content": content,
        "context_counts": {
            "sessions": len(context.get("sessions", [])),
            "training_runs": len(context.get("training_runs", [])),
            "chat_exchanges": len(context.get("chat_exchanges", [])),
            "external_analyses": len(context.get("external_analysis_history", [])),
        },
        "medical_disclaimer": REPORT_DISCLAIMER,
        "generated_at": datetime.now().isoformat(),
    }


def persist_report_artifact(report_document: Dict[str, Any], *, job_id: str) -> Dict[str, Any]:
    storage = MinioStorageService()
    if not storage.enabled:
        raise RuntimeError("MinIO storage is unavailable for report artifacts")

    with tempfile.TemporaryDirectory(prefix=f"report_{job_id}_") as tmp_dir:
        local_path = os.path.join(tmp_dir, "report.json")
        with open(local_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(report_document), handle, indent=2, sort_keys=True)
        object_name = storage.upload_file(local_path, "reports", f"reports/{job_id}/report.json")
        if not object_name:
            raise RuntimeError("Failed to upload report artifact to object storage")
        return storage.build_artifact_descriptor(
            "reports",
            object_name,
            label="Generated report artifact",
            kind="report_json",
            content_type="application/json",
            metadata={"job_id": job_id, "report_type": report_document.get("report_type")},
        )
