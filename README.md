# NeuroLab AI Service

FastAPI service for EEG analysis, voice analysis, recommendations, chat, model calibration, and optional streaming.

## What this service does
- EEG analysis from JSON or uploaded files
- Voice emotion analysis from uploaded/raw audio
- Recommendation and decision-support endpoints
- Chat endpoint (uses Groq when configured)
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

Docs:
- `http://localhost:8000/api/v1/docs`
- `http://localhost:8000/docs` (redirect)

## Environment variables
Important variables:
- `API_PREFIX` (default: `/api/v1`)
- `GROQ_API_KEY` (optional, enables LLM features)
- `REQUIRE_AUTH` (default: `false`)
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
- `POST /api/v1/eeg/detailed-report`
- `POST /api/v1/eeg/recommendations`
- `POST /api/v1/eeg/decision-support`
- `POST /api/v1/eeg/chat`

Voice:
- `POST /api/v1/voice/analyze`
- `POST /api/v1/voice/analyze-batch`
- `POST /api/v1/voice/analyze-raw`
- `GET /api/v1/voice/health`

Models:
- `POST /api/v1/models/calibrate`

## Important request contract
For EEG endpoints (`/upload`, `/analyze`, `/detailed-report`), `model_type` is required as a query param.

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
- If `GROQ_API_KEY` is missing, recommendation/chat fall back to non-LLM behavior.
- Persistence and queue failures are handled best-effort in most paths.
- Keep credentials out of git; use secret management in CI/CD.
