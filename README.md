# NeuroLab AI Service

FastAPI service for EEG analysis, voice analysis, recommendations, chat, model calibration, and optional streaming.

## What this service does
- EEG analysis from JSON or uploaded files
- Voice emotion analysis from uploaded/raw audio
- Recommendation and non-diagnostic history summary endpoints
- Chat endpoint with async OpenRouter background mode
- Model calibration endpoint
- Optional Redis queue + MongoDB/InfluxDB persistence

## Tech stack
- Python + FastAPI + Uvicorn
- TensorFlow + scikit-learn
- Redis/RQ (background jobs)
- MongoDB + InfluxDB (optional persistence)

## Quick start (local)
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the API.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Optional install groups:
- `pip install -r requirements-runtime.txt`
- `pip install -r requirements-ml.txt`
- `pip install -r requirements-voice.txt`
- `pip install -r requirements-dev.txt`

Docs:
- `http://localhost:8000/api/v1/docs`
- `http://localhost:8000/docs` (redirect)

## Environment variables
Important variables:
- `API_PREFIX` (default: `/api/v1`)
- `OPENROUTER_API_KEY` (optional, enables async LLM chat features)
- `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`)
- `OPENROUTER_SITE_URL` (optional, sent as OpenRouter referer)
- `OPENROUTER_APP_NAME` (optional, sent as OpenRouter title)
- `ENABLE_CHAT_GRPC` (default: `false`; keeps the old gRPC chat server disabled)
- `REQUIRE_AUTH` (default: `false`)
- `API_BEARER_TOKEN` (required when `REQUIRE_AUTH=true`)
- `ALLOWED_ORIGINS` (required in production)
- `ENABLE_DATABASES` (default: `true`)
- `MONGODB_URI`, `MONGODB_DB`
- `INFLUXDB_URL`, `INFLUXDB_TOKEN`, `INFLUXDB_ORG`, `INFLUXDB_BUCKET`
- `REDIS_URL` (for RQ jobs)

## API structure
Versioned routes:
- `/api/v1/health`
- `/api/v1/eeg/*`
- `/api/v1/voice/*`
- `/api/v1/models/*`
- `/api/v1/training/*`
- `/api/v1/streaming/*` (if available)

Legacy compatibility routes also exist (hidden from OpenAPI), including `/health`, `/upload`, `/analyze`, etc.

## Key endpoints
EEG:
- `POST /api/v1/eeg/upload`
- `POST /api/v1/eeg/analyze`
- `POST /api/v1/eeg/recommendations`
- `POST /api/v1/eeg/decision-support`

Chat:
- `POST /api/v1/chat`
- `POST /api/v1/chat/submit`
- `GET /api/v1/chat/status/{job_id}`
- `GET /api/v1/chat/sse?job_id=...`
- `POST /api/v1/chat/generate-name`

Reports:
- `POST /api/v1/reports/submit`
- `GET /api/v1/reports/status/{job_id}`
- `GET /api/v1/reports/jobs`
- `GET /api/v1/reports/runs/{job_id}`
- `GET /api/v1/reports/history`
- `GET /api/v1/reports/runs/{job_id}/artifacts`
- `GET /api/v1/reports/sse?job_id=...`

Voice:
- `POST /api/v1/voice/analyze`
- `POST /api/v1/voice/analyze-batch`
- `POST /api/v1/voice/analyze-raw`
- `GET /api/v1/voice/health`

Models:
- `POST /api/v1/models/calibrate`

## Important request contract
For EEG endpoints (`/upload`, `/analyze`), `model_type` is required as a query param.

Allowed values:
- `original`
- `enhanced_cnn_lstm`
- `resnet_lstm`
- `transformer`
- `trained_model`

Example:
```bash
curl -X POST "http://localhost:8000/api/v1/eeg/analyze?model_type=original" \
  -H "Content-Type: application/json" \
  -d '{"alpha":10.5,"beta":15.2,"theta":6.3,"delta":2.1,"gamma":30.5}'
```

## Docker
Build:
```bash
docker build -t neurolab-ai .
```

Run:
```bash
docker run --rm -p 8000:8000 --env-file .env neurolab-ai
```

## Tests
Run all tests:
```bash
pytest
```

Run a single file:
```bash
pytest src/tests/test_main_api.py
```

## GitHub workflows
Defined in `.github/workflows/`:
- `python-app.yml`
- `docker-publish.yml`

## Notes
- If `OPENROUTER_API_KEY` is missing, chat falls back to non-LLM behavior.
- Persistence and queue failures are handled best-effort in most paths.
- In production, the service now refuses to start with unsafe CORS, DB secret, MinIO secret, or auth-token defaults.
- Keep credentials out of git; use secret management in CI/CD.

## Background chat flow
1. Submit a request with `POST /api/v1/chat/submit`.
2. Open `GET /api/v1/chat/sse?job_id=...` to receive `queued`, `started`, `context_retrieved`, `generating_response`, `completed`, or `failed` events.
3. Poll `GET /api/v1/chat/status/{job_id}` if SSE is not available.
4. Set `generate_title=true` only if you want a follow-up `title_generated` event; it is no longer part of the main answer critical path.

Run an RQ worker for the new queue:

```bash
python -m src.worker
```
