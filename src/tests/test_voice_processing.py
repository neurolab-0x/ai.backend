"""
Additional in-process tests for voice endpoints.
"""
import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

import src.api.voice as voice_api


app = FastAPI()
app.include_router(voice_api.router, prefix="/voice")


@pytest.fixture(autouse=True)
def mock_voice_processor(monkeypatch):
    processor = SimpleNamespace(
        model=object(),
        processor=object(),
        device="cpu",
        sample_rate=16000,
        emotion_to_state={"calm": 0, "happy": 1, "angry": 2},
        process_audio=lambda audio_bytes, sample_rate=None: {
            "emotion": "happy",
            "confidence": 0.88,
            "mental_state": 1,
            "emotion_probabilities": {"happy": 0.88, "calm": 0.1, "angry": 0.02},
            "features": {"rms": 0.4},
        },
    )
    monkeypatch.setattr(voice_api, "get_voice_processor", lambda: processor)


@pytest.fixture
async def async_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class TestVoiceAPI:
    @pytest.fixture
    def mock_audio_file(self):
        wav_header = (
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00"
            b"\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        return io.BytesIO(wav_header)

    @pytest.mark.anyio
    async def test_voice_health(self, async_client):
        response = await async_client.get("/voice/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.anyio
    async def test_voice_emotions_list(self, async_client):
        response = await async_client.get("/voice/emotions")
        assert response.status_code == 200
        data = response.json()
        assert "emotions" in data
        assert isinstance(data["emotions"], list)

    @pytest.mark.anyio
    async def test_analyze_audio_file(self, async_client, mock_audio_file):
        files = {
            "file": ("test_audio.wav", mock_audio_file, "audio/wav"),
        }

        response = await async_client.post("/voice/analyze", files=files)

        assert response.status_code == 200
        result = response.json()
        assert "data" in result
        assert result["data"]["emotion"] == "happy"
        assert result["data"]["mental_state"] == 1
