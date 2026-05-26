from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api import reports as reports_api


class FakeQueue:
    def __init__(self):
        self.calls = []

    def enqueue(self, func_name, *args, **kwargs):
        self.calls.append((func_name, args, kwargs))
        return SimpleNamespace(id=kwargs["job_id"])


@pytest.fixture
def report_setup(monkeypatch):
    fake_queue = FakeQueue()
    monkeypatch.setattr(reports_api, "require_rq", lambda: None)
    monkeypatch.setattr(reports_api, "get_queue", lambda name: fake_queue)
    monkeypatch.setattr(reports_api, "track_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(reports_api, "publish_job_event", lambda *args, **kwargs: {"timestamp": "2026-04-27T00:00:00Z"})
    monkeypatch.setattr(reports_api.db_service, "create_report_run", AsyncMock(return_value="db-id"))
    return fake_queue


@pytest.mark.asyncio
async def test_submit_report_job_enqueues_background_worker(report_setup):
    fake_queue = report_setup

    response = await reports_api.submit_report_job(
        reports_api.ReportJobRequest(
            subject_id="subject_1",
            session_id="session_1",
            report_type="summary",
            prompt="Focus on the last 7 days",
            lookback_days=7,
            include_sessions=True,
            include_chat=False,
            context_limit=12,
        )
    )

    assert response.status == "queued"
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.reports.process_report_request"
    assert args[0]["subject_id"] == "subject_1"
    assert args[0]["session_id"] == "session_1"
    assert args[0]["lookback_days"] == 7
    assert args[0]["include_chat"] is False
    assert kwargs["job_id"].startswith("report_")


@pytest.mark.asyncio
async def test_submit_report_job_returns_report_urls(report_setup):
    response = await reports_api.submit_report_job(
        reports_api.ReportJobRequest(
            subject_id="subject_2",
            report_type="executive_summary",
        )
    )

    assert response.events_url.endswith(f"/api/v1/reports/sse?job_id={response.job_id}")
    assert response.status_url.endswith(f"/api/v1/reports/status/{response.job_id}")
