"""
In-process tests for the voice API router.
"""
import base64
import io
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

import src.api.voice as voice_api


app = FastAPI()
app.include_router(voice_api.router, prefix="/api/v1/voice")


@pytest.fixture(autouse=True)
def mock_voice_processor(monkeypatch):
    processor = SimpleNamespace(
        model=object(),
        processor=object(),
        device="cpu",
        sample_rate=16000,
        emotion_to_state={
            "calm": 0,
            "happy": 1,
            "angry": 2,
        },
        process_audio=lambda audio_bytes, sample_rate=None: {
            "emotion": "calm",
            "confidence": 0.93,
            "mental_state": 0,
            "emotion_probabilities": {"calm": 0.93, "happy": 0.04, "angry": 0.03},
            "features": {"rms": 0.5},
        },
    )
    monkeypatch.setattr(voice_api, "get_voice_processor", lambda: processor)


@pytest.fixture
async def async_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.anyio
async def test_voice_health(async_client):
    response = await async_client.get("/api/v1/voice/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@pytest.mark.anyio
async def test_get_emotions(async_client):
    response = await async_client.get("/api/v1/voice/emotions")
    assert response.status_code == 200
    data = response.json()
    assert "emotions" in data
    assert "emotion_to_state_mapping" in data


@pytest.mark.anyio
async def test_analyze_audio(async_client):
    files = {
        "file": ("test.wav", io.BytesIO(b"RIFFfakeWAVEdata"), "audio/wav"),
    }
    response = await async_client.post("/api/v1/voice/analyze", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["emotion"] == "calm"


@pytest.mark.anyio
async def test_analyze_raw_audio(async_client):
    sample_rate = 16000
    duration = 0.05
    frequency = 440
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

    response = await async_client.post(
        "/api/v1/voice/analyze-raw",
        json={
            "audio_data": {
                "data": audio_base64,
                "format": "base64",
            },
            "sample_rate": sample_rate,
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["mental_state"] == 0


@pytest.mark.anyio
async def test_batch_analysis(async_client):
    files = [
        ("files", ("segment1.wav", io.BytesIO(b"RIFFsegment1"), "audio/wav")),
        ("files", ("segment2.wav", io.BytesIO(b"RIFFsegment2"), "audio/wav")),
    ]

    response = await async_client.post("/api/v1/voice/analyze-batch", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["processed_files"] == 2
    assert body["pattern_analysis"]["dominant_emotion"] == "calm"
