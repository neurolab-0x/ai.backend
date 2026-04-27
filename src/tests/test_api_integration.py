"""
Integration tests for the public FastAPI surface using lightweight API-level mocks.
"""
from types import SimpleNamespace
import io
import sys
import types

import httpx
import pytest
from fastapi import FastAPI

pd = pytest.importorskip("pandas")

fake_model_manager_module = types.ModuleType("src.services.model_manager")
fake_model_manager_module.get_model_manager = lambda: SimpleNamespace(
    tensorflow_available=False,
    get_health_status=lambda: {
        "status": "ok",
        "models_loaded": [],
        "models_count": 0,
    },
    list_model_files=lambda: [],
)
sys.modules.setdefault("src.services.model_manager", fake_model_manager_module)

import src.api.analysis as analysis_api
import src.api.system as system_api


app = FastAPI()
app.include_router(system_api.router, prefix="/api/v1")
app.include_router(analysis_api.router, prefix="/api/v1/eeg")
app.include_router(system_api.router)
app.include_router(analysis_api.router)


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    async def process_eeg_data(*args, **kwargs):
        return {
            "state_label": "engaged",
            "confidence": 87.5,
            "dominant_state": 1,
            "recommendations": ["Take a short break"],
        }

    async def generate_recommendations(**kwargs):
        return ["Take a short break", "Hydrate"]

    mock_ml_processor = SimpleNamespace(
        process_eeg_data=process_eeg_data,
        recommendation_engine=SimpleNamespace(
            generate_recommendations=generate_recommendations,
        ),
    )

    monkeypatch.setattr(analysis_api, "get_ml_processor", lambda: mock_ml_processor)


class TestAPIIntegration:
    @pytest.fixture
    async def async_client(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client

    @pytest.fixture
    def sample_eeg_csv(self):
        data = {
            "alpha": [0.5, 0.6],
            "beta": [0.3, 0.4],
            "theta": [0.2, 0.3],
            "delta": [0.1, 0.2],
            "gamma": [0.4, 0.5],
        }
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @pytest.mark.anyio
    async def test_root_endpoint(self, async_client):
        response = await async_client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert "NeuroLab" in data["name"]
        assert "features" in data

    @pytest.mark.anyio
    async def test_health_check(self, async_client):
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "diagnostics" in data

    @pytest.mark.anyio
    async def test_analyze_endpoint_valid_json(self, async_client):
        payload = {
            "alpha": 0.5,
            "beta": 0.3,
            "theta": 0.2,
            "delta": 0.1,
            "gamma": 0.4,
            "subject_id": "api_test_sub",
            "session_id": "api_test_sess",
        }

        response = await async_client.post(
            "/api/v1/eeg/analyze",
            params={"model_type": "enhanced_cnn_lstm"},
            json=payload,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["state_label"] == "engaged"
        assert result["confidence"] == 87.5

    @pytest.mark.anyio
    async def test_analyze_endpoint_invalid_model_type(self, async_client):
        payload = {
            "alpha": 0.5,
            "beta": 0.3,
            "theta": 0.2,
            "delta": 0.1,
            "gamma": 0.4,
        }

        response = await async_client.post(
            "/api/v1/eeg/analyze",
            params={"model_type": "invalid_model"},
            json=payload,
        )

        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_upload_endpoint_csv(self, async_client, sample_eeg_csv):
        files = {
            "file": ("test_eeg.csv", sample_eeg_csv, "text/csv"),
        }

        response = await async_client.post(
            "/api/v1/eeg/upload",
            params={"model_type": "enhanced_cnn_lstm"},
            files=files,
        )

        assert response.status_code == 200
        assert response.json()["state_label"] == "engaged"

    @pytest.mark.anyio
    async def test_upload_endpoint_invalid_file_type(self, async_client):
        files = {
            "file": ("test.txt", io.BytesIO(b"bad data"), "text/plain"),
        }

        response = await async_client.post(
            "/api/v1/eeg/upload",
            params={"model_type": "enhanced_cnn_lstm"},
            files=files,
        )

        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_recommendations_endpoint(self, async_client):
        payload = {
            "state_durations": {"0": 10, "1": 20, "2": 5},
            "total_duration": 35,
            "confidence": 85.5,
            "state_transitions": 2,
        }

        response = await async_client.post("/api/v1/eeg/recommendations", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["recommendations"] == ["Take a short break", "Hydrate"]
