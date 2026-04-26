"""
Training API endpoints for model training and retraining.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Query
from pydantic import BaseModel, Field, validator
import numpy as np
try:
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    from src.queue import get_queue, track_job, list_tracked_jobs, untrack_job
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

from src.utils.files import validate_file, save_uploaded_file
from src.core.ml.model_types import sanitize_model_type


logger = logging.getLogger(__name__)

router = APIRouter()


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
        # Convert data to numpy arrays
        X_train = np.array(data.X_train)
        y_train = np.array(data.y_train)
        X_test = np.array(data.X_test) if data.X_test else None
        y_test = np.array(data.y_test) if data.y_test else None
        
        # Ensure config is populated and has the correct model_type
        if data.config is None:
            data.config = TrainingConfig(model_type=model_type)
        else:
            # Sync model_type from query param to config
            data.config.model_type = model_type

        # Persist arrays to disk to avoid pickling huge payloads into Redis.
        os.makedirs("temp", exist_ok=True)
        temp_npz = os.path.join("temp", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz")
        if X_test is None or y_test is None:
            np.savez_compressed(temp_npz, X_train=X_train, y_train=y_train)
        else:
            np.savez_compressed(temp_npz, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        try:
            q = get_queue("training")
            job = q.enqueue(
                "src.jobs.training.train_from_npz",
                temp_npz,
                data.config.dict(),
                model_type,
                job_timeout=60 * 60 * 6,  # 6h
                result_ttl=60 * 60 * 24,
            )
        except Exception:
            try:
                os.remove(temp_npz)
            except OSError:
                pass
            raise
        track_job("training", job.id)
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
        # Validate file
        validate_file(file)
        
        # Save uploaded file
        file_location = await save_uploaded_file(file)
        
        # Parse config if provided
        if config:
            import json
            config_dict = json.loads(config)
            if 'model_type' not in config_dict:
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

        try:
            q = get_queue("training")
            job = q.enqueue(
                "src.jobs.training.train_from_file",
                file_location,
                training_config.dict(),
                model_type,
                job_timeout=60 * 60 * 6,  # 6h
                result_ttl=60 * 60 * 24,
            )
        except Exception:
            try:
                os.remove(file_location)
            except OSError:
                pass
            raise
        track_job("training", job.id)
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
    require_rq()
    try:
        job = Job.fetch(job_id, connection=get_queue("training").connection)
    except NoSuchJobError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found")

    status_str = job.get_status()
    progress = float(job.meta.get("progress", 0.0))
    message = job.meta.get("message", status_str)

    result = job.result if job.is_finished else None
    error = str(job.exc_info) if job.is_failed else None

    return TrainingStatus(
        job_id=job.id,
        status=status_str,
        progress=progress,
        message=message,
        started_at=job.enqueued_at.isoformat() if job.enqueued_at else datetime.now().isoformat(),
        completed_at=job.ended_at.isoformat() if job.ended_at else None,
        metrics=result if isinstance(result, dict) else None,
        error=error,
    )


@router.get("/jobs", response_model=List[TrainingStatus])
async def list_training_jobs(
    # current_user: Dict = Depends(get_current_user),
    limit: int = 10
):
    """
    List training jobs.
    Returns all jobs (authentication disabled).
    """
    job_ids = list_tracked_jobs("training")
    jobs: List[TrainingStatus] = []
    for jid in job_ids:
        try:
            job = Job.fetch(jid, connection=get_queue("training").connection)
        except Exception:
            continue
        status_str = job.get_status()
        jobs.append(
            TrainingStatus(
                job_id=job.id,
                status=status_str,
                progress=float(job.meta.get("progress", 0.0)),
                message=str(job.meta.get("message", status_str)),
                started_at=job.enqueued_at.isoformat() if job.enqueued_at else datetime.now().isoformat(),
                completed_at=job.ended_at.isoformat() if job.ended_at else None,
                metrics=job.result if job.is_finished and isinstance(job.result, dict) else None,
                error=str(job.exc_info) if job.is_failed else None,
            )
        )
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
    try:
        job = Job.fetch(job_id, connection=get_queue("training").connection)
    except NoSuchJobError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found")
    job.delete()
    untrack_job("training", job_id)
    return {"status": "success", "message": f"Training job {job_id} deleted"}


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
        # Convert data to numpy arrays
        X_train = np.array(data.X_train)
        y_train = np.array(data.y_train)
        X_test = np.array(data.X_test) if data.X_test else None
        y_test = np.array(data.y_test) if data.y_test else None
        
        if X_test is None or y_test is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Test data is required for model comparison"
            )
        
        os.makedirs("temp", exist_ok=True)
        temp_npz = os.path.join("temp", f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npz")
        np.savez_compressed(temp_npz, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        try:
            q = get_queue("training")
            job = q.enqueue(
                "src.jobs.training.compare_models_from_npz",
                temp_npz,
                n_repeats=n_repeats,
                job_timeout=60 * 60 * 6,
                result_ttl=60 * 60 * 24,
            )
        except Exception:
            try:
                os.remove(temp_npz)
            except OSError:
                pass
            raise
        track_job("training", job.id)
        
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
