"""
Training API endpoints for model training and retraining.
"""
import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import numpy as np
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

from src.utils.files import validate_file, save_uploaded_file
from src.core.ml.model_types import sanitize_model_type
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
            detail="Training queue is unavailable because RQ is not installed.",
        )


class TrainingConfig(BaseModel):
    """Configuration for model training"""
    model_type: str = Field(..., description="Type of model to train (required)")
    epochs: int = Field(default=30, ge=1, le=200, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for training")
    learning_rate: float = Field(default=0.001, gt=0, lt=1, description="Learning rate")
    dropout_rate: float = Field(default=0.3, ge=0, le=0.9, description="Dropout rate")
    use_separable: bool = Field(default=True, description="Use separable convolutions")
    use_relative_pos: bool = Field(default=True, description="Use relative positional encoding")
    l1_reg: float = Field(default=1e-5, ge=0, description="L1 regularization factor")
    l2_reg: float = Field(default=1e-4, ge=0, description="L2 regularization factor")
    subject_id: Optional[str] = Field(None, description="Subject ID for personalized training")
    session_id: Optional[str] = Field(None, description="Session ID")
    validation_mode: str = Field(default='split', description="Validation mode: 'split', 'kfold', or 'loso'")
    overlap: float = Field(default=0.5, ge=0.0, le=0.9, description="Overlap between epochs (0.0 to 0.9)")
    simple_mode: bool = Field(default=True, description="Whether to use simplified feature extraction")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        return sanitize_model_type(v)

    @validator('validation_mode')
    def validate_validation_mode(cls, v):
        normalized = str(v).strip().lower()
        if normalized != "split":
            raise ValueError("Only validation_mode='split' is currently supported")
        return normalized


class TrainingData(BaseModel):
    """Training data input"""
    X_train: List[List[float]] = Field(..., description="Training features")
    y_train: List[int] = Field(..., description="Training labels")
    X_test: Optional[List[List[float]]] = Field(None, description="Test features (optional)")
    y_test: Optional[List[int]] = Field(None, description="Test labels (optional)")
    config: Optional[TrainingConfig] = Field(None, description="Training configuration")
    
    @validator('X_train')
    def validate_X_train(cls, v):
        if len(v) == 0:
            raise ValueError("Training data cannot be empty")
        if len(v) > 100000:
            raise ValueError("Training data too large (max 100,000 samples)")
        return v
    
    @validator('y_train')
    def validate_y_train(cls, v, values):
        if 'X_train' in values and len(v) != len(values['X_train']):
            raise ValueError("X_train and y_train must have the same length")
        return v


class TrainingResponse(BaseModel):
    """Response for training request"""
    job_id: str
    status: str
    message: str
    started_at: str


class TrainingStatus(BaseModel):
    """Training job status"""
    job_id: str
    status: str
    progress: float
    message: str
    started_at: str
    completed_at: Optional[str]
    metrics: Optional[Dict[str, Any]]
    error: Optional[str]
    artifacts: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class TrainingRunDetail(TrainingStatus):
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    model_type: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _sanitize_model_type_or_400(model_type: str) -> str:
    try:
        return sanitize_model_type(model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _serialize_dt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _sanitize_training_config_ids(config: Optional[TrainingConfig]) -> Optional[TrainingConfig]:
    if config is None:
        return None
    payload = config.dict()
    payload["subject_id"] = validate_optional_safe_id(payload.get("subject_id"), "subject_id")
    payload["session_id"] = validate_optional_safe_id(payload.get("session_id"), "session_id")
    return TrainingConfig(**payload)


def _get_storage_service() -> MinioStorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = MinioStorageService()
    return _storage_service


def _require_training_storage() -> MinioStorageService:
    storage = _get_storage_service()
    if not storage.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training storage is unavailable. MinIO must be enabled for training jobs.",
        )
    return storage


def _hydrate_artifacts(artifacts: Any) -> Any:
    if not isinstance(artifacts, (dict, list)):
        return artifacts
    return _get_storage_service().hydrate_artifact_urls(artifacts)


def _build_training_status(job: Optional[Job], persisted_run: Optional[Dict[str, Any]]) -> TrainingStatus:
    state = persisted_run or {}
    rq_status = job.get_status() if job is not None else None
    status_str = state.get("status") or rq_status or "unknown"
    progress = float(state.get("progress", job.meta.get("progress", 0.0) if job else 0.0))
    message = str(state.get("message") or (job.meta.get("message") if job else None) or status_str)
    metrics = state.get("metrics")
    error = state.get("error") or (str(job.exc_info) if job is not None and job.is_failed else None)
    artifacts = _hydrate_artifacts(state.get("artifacts"))
    config = state.get("config")
    return TrainingStatus(
        job_id=(state.get("job_id") or (job.id if job else "")),
        status=status_str,
        progress=progress,
        message=message,
        started_at=_serialize_dt(state.get("started_at") or (job.enqueued_at if job else None) or datetime.now()),
        completed_at=_serialize_dt(state.get("completed_at") or (job.ended_at if job else None)),
        metrics=metrics if isinstance(metrics, dict) else None,
        error=error,
        artifacts=artifacts if isinstance(artifacts, dict) else None,
        config=config if isinstance(config, dict) else None,
    )


def _build_training_run_detail(job: Optional[Job], persisted_run: Dict[str, Any]) -> TrainingRunDetail:
    base = _build_training_status(job, persisted_run)
    return TrainingRunDetail(
        **base.model_dump(),
        subject_id=persisted_run.get("subject_id"),
        session_id=persisted_run.get("session_id"),
        model_type=persisted_run.get("model_type"),
        created_at=_serialize_dt(persisted_run.get("created_at")),
        updated_at=_serialize_dt(persisted_run.get("updated_at")),
    )


@router.post("/train", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    data: TrainingData,
    model_type: str = Query(..., description="Architecture to use for training (required)"),
    # current_user: Dict = Depends(require_admin_role)
):
    """
    Train a new model with provided data (Admin only).
    
    This endpoint starts a background training job and returns immediately.
    Use the job_id to check training status via /api/train/status/{job_id}
    """
    try:
        require_rq()
        model_type = _sanitize_model_type_or_400(model_type)
        # Convert data to numpy arrays
        X_train = np.array(data.X_train)
        y_train = np.array(data.y_train)
        X_test = np.array(data.X_test) if data.X_test else None
        y_test = np.array(data.y_test) if data.y_test else None
        
        # Ensure config is populated and has the correct model_type
        if data.config is None:
            data.config = TrainingConfig(model_type=model_type)
        else:
            config_payload = data.config.dict()
            config_payload["model_type"] = model_type
            data.config = TrainingConfig(**config_payload)
        data.config = _sanitize_training_config_ids(data.config)

        job_id = f"train_{uuid4().hex}"
        storage = _require_training_storage()
        # Stage bundle locally only long enough to upload it to MinIO.
        os.makedirs("temp", exist_ok=True)
        temp_npz = os.path.join("temp", f"{job_id}.npz")
        if X_test is None or y_test is None:
            np.savez_compressed(temp_npz, X_train=X_train, y_train=y_train)
        else:
            np.savez_compressed(temp_npz, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        uploaded_bundle = storage.upload_file(temp_npz, "training", f"runs/{job_id}/input/training_bundle.npz")
        if not uploaded_bundle:
            try:
                os.remove(temp_npz)
            except OSError:
                pass
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to upload training bundle to object storage.",
            )
        bundle_descriptor = storage.build_artifact_descriptor(
            "training",
            uploaded_bundle,
            label="Training bundle",
            kind="dataset_bundle",
            content_type="application/octet-stream",
            metadata={"model_type": model_type, "source": "api_train_payload"},
        )
        try:
            os.remove(temp_npz)
        except OSError:
            pass

        run_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Training job queued",
            "run_type": "training_data",
            "run_family": "training",
            "model_type": model_type,
            "subject_id": data.config.subject_id,
            "session_id": data.config.session_id,
            "config": {**data.config.dict(), "run_type": "training_data"},
            "artifacts": {
                "objects": {"training_bundle": bundle_descriptor},
            },
            "metrics": None,
            "error": None,
            "started_at": datetime.now(),
            "completed_at": None,
        }
        await db_service.create_training_run(run_record)

        try:
            q = get_queue("training")
            job = q.enqueue(
                "src.jobs.training.train_from_bundle_object",
                bundle_descriptor,
                {**data.config.dict(), "run_type": "training_data", "source": "api_train_payload"},
                model_type,
                job_id=job_id,
                job_timeout=60 * 60 * 6,  # 6h
                result_ttl=60 * 60 * 24,
            )
        except Exception:
            await db_service.archive_training_run(job_id, reason="enqueue_failed")
            raise
        track_job("training", job.id)
        publish_job_event("training", job.id, "queued", {"run_type": "training_data", "model_type": model_type})
        logger.info(f"Training job {job.id} enqueued")
        
        return TrainingResponse(
            job_id=job.id,
            status='queued',
            message='Training job started in background',
            started_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post("/file", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model_from_file(
    file: UploadFile = File(...),
    model_type: str = Query(..., description="Architecture to use for training (required)"),
    overlap: float = Query(0.5, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction"),
    config: Optional[str] = None,
    #current_user: Dict = Depends(require_admin_role)
):
    """
    Train a model from uploaded data file (Admin only).
    
    Accepts CSV files with EEG data and labels.
    """
    try:
        require_rq()
        model_type = _sanitize_model_type_or_400(model_type)
        storage = _require_training_storage()
        # Validate file
        validate_file(file)
        
        # Save uploaded file only long enough to push it into object storage.
        file_location = await save_uploaded_file(file)
        
        # Parse config if provided
        if config:
            import json
            config_dict = json.loads(config)
            config_dict['model_type'] = model_type
            if 'overlap' not in config_dict:
                config_dict['overlap'] = overlap
            if 'simple_mode' not in config_dict:
                config_dict['simple_mode'] = simple_mode
            training_config = TrainingConfig(**config_dict)
        else:
            training_config = TrainingConfig(
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode
            )
        training_config = _sanitize_training_config_ids(training_config)

        job_id = f"train_file_{uuid4().hex}"
        uploaded_dataset = storage.upload_file(
            file_location,
            "training",
            f"runs/{job_id}/input/{os.path.basename(file_location)}",
        )
        try:
            os.remove(file_location)
        except OSError:
            pass
        if not uploaded_dataset:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to upload dataset file to object storage.",
            )
        dataset_descriptor = storage.build_artifact_descriptor(
            "training",
            uploaded_dataset,
            label="Uploaded training dataset",
            kind="uploaded_dataset",
            content_type=file.content_type or "application/octet-stream",
            metadata={"model_type": model_type, "original_filename": file.filename},
        )
        run_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Training-from-file job queued",
            "run_type": "training_file",
            "run_family": "training",
            "model_type": model_type,
            "subject_id": training_config.subject_id,
            "session_id": training_config.session_id,
            "config": {**training_config.dict(), "run_type": "training_file"},
            "artifacts": {
                "uploaded_file_name": file.filename,
                "objects": {"uploaded_dataset": dataset_descriptor},
            },
            "metrics": None,
            "error": None,
            "started_at": datetime.now(),
            "completed_at": None,
        }
        await db_service.create_training_run(run_record)

        try:
            q = get_queue("training")
            job = q.enqueue(
                "src.jobs.training.train_from_file_object",
                dataset_descriptor,
                {**training_config.dict(), "run_type": "training_file", "source": "uploaded_dataset_file"},
                model_type,
                job_id=job_id,
                job_timeout=60 * 60 * 6,  # 6h
                result_ttl=60 * 60 * 24,
            )
        except Exception:
            await db_service.archive_training_run(job_id, reason="enqueue_failed")
            raise
        track_job("training", job.id)
        publish_job_event("training", job.id, "queued", {"run_type": "training_file", "model_type": model_type})
        logger.info(f"Training-from-file job {job.id} enqueued for {file.filename}")
        
        return TrainingResponse(
            job_id=job.id,
            status='queued',
            message=f'Training job started from file {file.filename}',
            started_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error starting training from file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(
    job_id: str,
    # current_user: Dict = Depends(get_current_user)
):
    """
    Get the status of a training job.
    """
    job_id = require_safe_id_or_400(job_id, "job_id")
    persisted_run = await db_service.get_training_run(job_id)
    job = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("training").connection)
        except NoSuchJobError:
            job = None

    if persisted_run is None and job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found")
    return _build_training_status(job, persisted_run)


@router.get("/jobs", response_model=List[TrainingStatus])
async def list_training_jobs(
    # current_user: Dict = Depends(get_current_user),
    limit: int = 10
):
    """
    List training jobs.
    Returns all jobs (authentication disabled).
    """
    persisted_runs = await db_service.list_training_runs(limit=max(limit, 20))
    jobs: List[TrainingStatus] = []
    job_ids = list_tracked_jobs("training") if RQ_AVAILABLE else []
    seen = {run.get("job_id") for run in persisted_runs if run.get("job_id")}
    ordered_ids = [run["job_id"] for run in persisted_runs if run.get("job_id")] + [jid for jid in job_ids if jid not in seen]
    for jid in ordered_ids:
        persisted_run = next((run for run in persisted_runs if run.get("job_id") == jid), None)
        job = None
        if RQ_AVAILABLE:
            try:
                job = Job.fetch(jid, connection=get_queue("training").connection)
            except Exception:
                job = None
        if job is None and persisted_run is None:
            continue
        jobs.append(_build_training_status(job, persisted_run))
    jobs.sort(key=lambda j: j.started_at, reverse=True)
    return jobs[:limit]


@router.delete("/job/{job_id}")
async def delete_training_job(
    job_id: str,
    # current_user: Dict = Depends(require_admin_role)
):
    """
    Delete a training job record (Admin only).
    """
    job_id = require_safe_id_or_400(job_id, "job_id")
    job = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("training").connection)
        except NoSuchJobError:
            job = None
    persisted_run = await db_service.get_training_run(job_id)
    if job is None and persisted_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found")
    if job is not None:
        job.delete()
    if RQ_AVAILABLE:
        untrack_job("training", job_id)
    await db_service.archive_training_run(job_id, reason="archived_by_user")
    return {"status": "success", "message": f"Training job {job_id} archived"}


@router.post("/compare", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def compare_models(
    data: TrainingData,
    n_repeats: int = 3,
    # current_user: Dict = Depends(require_admin_role)
):
    """
    Compare multiple model architectures.
    Trains and evaluates all available model types and returns comparison metrics.
    (Authentication disabled)
    """
    try:
        require_rq()
        storage = _require_training_storage()
        # Convert data to numpy arrays
        X_train = np.array(data.X_train)
        y_train = np.array(data.y_train)
        X_test = np.array(data.X_test) if data.X_test else None
        y_test = np.array(data.y_test) if data.y_test else None
        data.config = _sanitize_training_config_ids(data.config)
        
        if X_test is None or y_test is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Test data is required for model comparison"
            )
        
        os.makedirs("temp", exist_ok=True)
        job_id = f"compare_{uuid4().hex}"
        temp_npz = os.path.join("temp", f"{job_id}.npz")
        np.savez_compressed(temp_npz, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        uploaded_bundle = storage.upload_file(temp_npz, "training", f"runs/{job_id}/input/comparison_bundle.npz")
        try:
            os.remove(temp_npz)
        except OSError:
            pass
        if not uploaded_bundle:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to upload comparison bundle to object storage.",
            )
        bundle_descriptor = storage.build_artifact_descriptor(
            "training",
            uploaded_bundle,
            label="Comparison bundle",
            kind="dataset_bundle",
            content_type="application/octet-stream",
            metadata={"source": "api_compare_payload"},
        )

        run_record = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Model comparison queued",
            "run_type": "comparison",
            "run_family": "comparison",
            "model_type": None,
            "subject_id": data.config.subject_id if data.config else None,
            "session_id": data.config.session_id if data.config else None,
            "config": {"n_repeats": n_repeats, "run_type": "comparison", **(data.config.dict() if data.config else {})},
            "artifacts": {"objects": {"comparison_bundle": bundle_descriptor}},
            "metrics": None,
            "error": None,
            "started_at": datetime.now(),
            "completed_at": None,
        }
        await db_service.create_training_run(run_record)

        try:
            q = get_queue("training")
            job = q.enqueue(
                "src.jobs.training.compare_models_from_object",
                bundle_descriptor,
                n_repeats=n_repeats,
                config={"run_type": "comparison", **(data.config.dict() if data.config else {})},
                job_id=job_id,
                job_timeout=60 * 60 * 6,
                result_ttl=60 * 60 * 24,
            )
        except Exception:
            await db_service.archive_training_run(job_id, reason="enqueue_failed")
            raise
        track_job("training", job.id)
        publish_job_event("training", job.id, "queued", {"run_type": "comparison"})
        
        return TrainingResponse(
            job_id=job.id,
            status='queued',
            message='Model comparison started in background',
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model comparison: {str(e)}"
        )


@router.get("/runs/{job_id}", response_model=TrainingRunDetail)
async def get_training_run_detail(job_id: str):
    """Return the canonical persisted training run view by job id."""
    job_id = require_safe_id_or_400(job_id, "job_id")
    persisted_run = await db_service.get_training_run(job_id)
    if persisted_run is None:
        raise HTTPException(status_code=404, detail=f"Training run {job_id} not found")
    job = None
    if RQ_AVAILABLE:
        try:
            job = Job.fetch(job_id, connection=get_queue("training").connection)
        except Exception:
            job = None
    return _build_training_run_detail(job, persisted_run)


@router.get("/history", response_model=List[TrainingRunDetail])
async def get_training_history(limit: int = 20, include_archived: bool = False):
    """Return persisted training run history newest first."""
    persisted_runs = await db_service.list_training_runs(limit=limit, include_archived=include_archived)
    details: List[TrainingRunDetail] = []
    for run in persisted_runs:
        job = None
        if RQ_AVAILABLE:
            try:
                job = Job.fetch(run["job_id"], connection=get_queue("training").connection)
            except Exception:
                job = None
        details.append(_build_training_run_detail(job, run))
    return details


@router.get("/runs/{job_id}/artifacts")
async def get_training_run_artifacts(job_id: str):
    """Return persisted artifact metadata for a training run."""
    job_id = require_safe_id_or_400(job_id, "job_id")
    run = await db_service.get_training_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Training run {job_id} not found")
    return {
        "job_id": job_id,
        "artifacts": _hydrate_artifacts(run.get("artifacts") or {}),
    }


@router.get("/sse")
async def stream_training_events(job_id: str):
    """Server-Sent Events stream for training job events."""
    job_id = require_safe_id_or_400(job_id, "job_id")
    if get_async_redis is None:
        raise HTTPException(status_code=503, detail="Training event stream is unavailable")

    async def event_gen():
        redis = get_async_redis()
        pubsub = redis.pubsub()
        channel = f"neurolab:job:events:training:{job_id}"
        try:
            await pubsub.subscribe(channel)
            state = read_job_state("training", job_id) if read_job_state else None
            yield f"event: ready\ndata: {json.dumps({'job_id': job_id, 'state': state}, default=str)}\n\n".encode("utf-8")
            if state and state.get("event") in {"completed", "failed"}:
                yield f"event: {state['event']}\ndata: {json.dumps(state, default=str)}\n\n".encode("utf-8")
                return

            while True:
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
                logger.debug("Training SSE unsubscribe failed for job %s", job_id)
            await pubsub.close()
            await redis.close()

    return StreamingResponse(event_gen(), media_type="text/event-stream")
