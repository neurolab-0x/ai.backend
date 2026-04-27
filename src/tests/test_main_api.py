import io
import sys
import types
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

np = pytest.importorskip("numpy")
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


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    async def process_eeg_data(*args, **kwargs):
        return {
            "state_label": "engaged",
            "confidence": 84.0,
            "dominant_state": 1,
            "recommendations": ["Take a short break"],
        }

    async def generate_recommendations(**kwargs):
        return ["Take a short break"]

    mock_ml_processor = SimpleNamespace(
        process_eeg_data=process_eeg_data,
        recommendation_engine=SimpleNamespace(
            generate_recommendations=generate_recommendations,
        ),
    )

    monkeypatch.setattr(analysis_api, "get_ml_processor", lambda: mock_ml_processor)


@pytest.fixture
async def async_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class TestMainAPI:
    @pytest.fixture
    def sample_eeg_csv(self):
        df = pd.DataFrame(
            {
                "alpha": [0.5, 0.6],
                "beta": [0.3, 0.4],
                "theta": [0.2, 0.3],
                "delta": [0.1, 0.2],
                "gamma": [0.4, 0.5],
            }
        )
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @pytest.mark.anyio
    async def test_health_check(self, async_client):
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "diagnostics" in data

    @pytest.mark.anyio
    async def test_upload_endpoint(self, async_client, sample_eeg_csv):
        response = await async_client.post(
            "/api/v1/eeg/upload",
            params={"model_type": "original"},
            files={"file": ("test_eeg.csv", sample_eeg_csv, "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["state_label"] == "engaged"

    @pytest.mark.anyio
    async def test_analyze_endpoint(self, async_client):
        test_data = {
            "alpha": 0.5,
            "beta": 0.3,
            "theta": 0.2,
            "delta": 0.1,
            "gamma": 0.4,
            "subject_id": "test_subject",
            "session_id": "test_session_001",
        }

        response = await async_client.post(
            "/api/v1/eeg/analyze",
            params={"model_type": "original"},
            json=test_data,
        )

        assert response.status_code == 200
        assert response.json()["confidence"] == 84.0

    @pytest.mark.anyio
    async def test_root_endpoint(self, async_client):
        response = await async_client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "features" in data

    @pytest.mark.anyio
    async def test_invalid_file_upload(self, async_client):
        response = await async_client.post(
            "/api/v1/eeg/upload",
            params={"model_type": "original"},
            files={"file": ("test.txt", io.BytesIO(b"Invalid data"), "text/plain")},
        )
        assert response.status_code == 400

    @pytest.mark.anyio
    async def test_analyze_invalid_data(self, async_client):
        response = await async_client.post(
            "/api/v1/eeg/analyze",
            params={"model_type": "original"},
            json={"session_id": "test_session_001"},
        )
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_recommendations_endpoint(self, async_client):
        response = await async_client.post(
            "/api/v1/eeg/recommendations",
            json={
                "state_durations": {"0": 10, "1": 5, "2": 3},
                "total_duration": 18,
                "confidence": 75,
                "cognitive_metrics": {"focus_index": 0.6},
                "state_transitions": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendations"] == ["Take a short break"]
        assert data["count"] == 1
