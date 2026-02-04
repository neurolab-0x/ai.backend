"""
Training API endpoints for model training and retraining.
"""
import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status, UploadFile, File, Query
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd

from src.core.ml.model import train_hybrid_model, evaluate_model, model_comparison, build_model
from src.utils.files import validate_file, save_uploaded_file
from src.preprocessing.load_data import load_data
from src.preprocessing.labeling import label_eeg_states
from src.preprocessing.features import extract_features
from src.preprocessing.preprocess import preprocess_data

logger = logging.getLogger(__name__)

router = APIRouter()

# Store training jobs status
training_jobs = {}


class TrainingConfig(BaseModel):
    """Configuration for model training"""
    model_type: str = Field(default='enhanced_cnn_lstm', description="Type of model to train")
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
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['original', 'enhanced_cnn_lstm', 'resnet_lstm', 'transformer']
        if v not in valid_types:
            raise ValueError(f"Invalid model type. Must be one of: {valid_types}")
        return v


class TrainingData(BaseModel):
    """Training data input"""
    X_train: List[List[float]] = Field(..., description="Training features")
    y_train: List[int] = Field(..., description="Training labels")
    X_test: Optional[List[List[float]]] = Field(None, description="Test features (optional)")
    y_test: Optional[List[int]] = Field(None, description="Test labels (optional)")
    config: Optional[TrainingConfig] = Field(default_factory=TrainingConfig, description="Training configuration")
    
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


async def train_model_background(
    job_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    config: TrainingConfig,
    model_type : str
):
    """Background task for model training"""
    try:
        logger.info(f"Starting training job {job_id}")
        training_jobs[job_id]['status'] = 'training'
        training_jobs[job_id]['progress'] = 0.1
        
        # Train model
        model, history = train_hybrid_model(
            X_train, y_train,
            model_type=model_type,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            dropout_rate=config.dropout_rate,
            use_separable=config.use_separable,
            use_relative_pos=config.use_relative_pos,
            l1_reg=config.l1_reg,
            l2_reg=config.l2_reg,
            subject_id=config.subject_id,
            session_id=config.session_id
        )
        
        # Performance check
        if config.validation_mode == 'loso':
            logger.info(f"LOSO validation requested for job {job_id}")
            # Note: For LOSO to work, we'd need a DataFrame with subject_ids. 
            # In a real API, we'd reconstruct this or pass it in.
            # For now, we'll signal it's handled in the training logic if data permits.
            pass
        
        training_jobs[job_id]['progress'] = 0.8
        training_jobs[job_id]['message'] = 'Training complete, evaluating model...'
        
        # Evaluate model if test data provided
        metrics = {}
        if X_test is not None and y_test is not None:
            metrics = evaluate_model(model, X_test, y_test, calibrate=True)
            logger.info(f"Model evaluation complete for job {job_id}")
        
        # Update job status
        training_jobs[job_id]['status'] = 'completed'
        training_jobs[job_id]['progress'] = 1.0
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        training_jobs[job_id]['message'] = 'Training completed successfully'
        training_jobs[job_id]['metrics'] = {
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'test_metrics': metrics if metrics else None
        }
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}")
        training_jobs[job_id]['status'] = 'failed'
        training_jobs[job_id]['error'] = str(e)
        training_jobs[job_id]['completed_at'] = datetime.now().isoformat()


@router.post("/train", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    data: TrainingData,
    model_type: str,
    background_tasks: BackgroundTasks,
    # current_user: Dict = Depends(require_admin_role)
):
    """
    Train a new model with provided data (Admin only).
    
    This endpoint starts a background training job and returns immediately.
    Use the job_id to check training status via /api/train/status/{job_id}
    """
    try:
        # Generate job ID
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert data to numpy arrays
        X_train = np.array(data.X_train)
        y_train = np.array(data.y_train)
        X_test = np.array(data.X_test) if data.X_test else None
        y_test = np.array(data.y_test) if data.y_test else None
        
        # Initialize job status
        training_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0.0,
            'message': 'Training job queued',
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'metrics': None,
            'error': None,
            'user': 'test_user',
            'config': data.config.dict()
        }
        
        # Start background training
        background_tasks.add_task(
            train_model_background,
            job_id, X_train, y_train, X_test, y_test, data.config, model_type
        )
        
        logger.info(f"Training job {job_id} queued by testing user")
        
        return TrainingResponse(
            job_id=job_id,
            status='queued',
            message='Training job started in background',
            started_at=training_jobs[job_id]['started_at']
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
    model_type: str = Query("enhanced_cnn_lstm", description="Architecture to use for training"),
    config: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    #current_user: Dict = Depends(require_admin_role)
):
    """
    Train a model from uploaded data file (Admin only).
    
    Accepts CSV files with EEG data and labels.
    """
    try:
        # Validate file
        validate_file(file)
        
        # Save uploaded file
        file_location = await save_uploaded_file(file)
        
        # Load and process data
        df = load_data(file_location)
        df = label_eeg_states(df)
        features_df = extract_features(df)
        X_train, X_test, y_train, y_test, metadata = preprocess_data(features_df)
        
        # Parse config if provided
        training_config = TrainingConfig()
        if config:
            import json
            config_dict = json.loads(config)
            training_config = TrainingConfig(**config_dict)
        
        # Generate job ID
        job_id = f"train_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize job status
        training_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0.0,
            'message': 'Training job queued',
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'metrics': None,
            'error': None,
            'user': 'test_user',
            'config': training_config.dict(),
            'file': file.filename,
            'model_type': model_type
        }
        
        # Start background training
        background_tasks.add_task(
            train_model_background,
            job_id, X_train, y_train, X_test, y_test, training_config, model_type
        )
        
        # Clean up uploaded file
        os.remove(file_location)
        
        logger.info(f"Training job {job_id} queued from file {file.filename}")
        
        return TrainingResponse(
            job_id=job_id,
            status='queued',
            message=f'Training job started from file {file.filename}',
            started_at=training_jobs[job_id]['started_at']
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
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job = training_jobs[job_id]
    
    return TrainingStatus(**job)


@router.get("/jobs", response_model=List[TrainingStatus])
async def list_training_jobs(
    # current_user: Dict = Depends(get_current_user),
    limit: int = 10
):
    """
    List training jobs for the current user.
    Admins can see all jobs.
    """
    is_admin = 'admin' in current_user.get('roles', [])
    
    # Filter jobs based on user permissions
    user_jobs = []
    for job_id, job in training_jobs.items():
        if is_admin or job['user'] == current_user['sub']:
            user_jobs.append(TrainingStatus(**job))
    
    # Sort by started_at (most recent first) and limit
    user_jobs.sort(key=lambda x: x.started_at, reverse=True)
    return user_jobs[:limit]


@router.delete("/job/{job_id}")
async def delete_training_job(
    job_id: str,
    # current_user: Dict = Depends(require_admin_role)
):
    """
    Delete a training job record (Admin only).
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    del training_jobs[job_id]
    logger.info(f"Training job {job_id} deleted by {current_user['sub']}")
    
    return {"status": "success", "message": f"Training job {job_id} deleted"}


@router.post("/compare", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def compare_models(
    data: TrainingData,
    background_tasks: BackgroundTasks,
    n_repeats: int = 3,
    # current_user: Dict = Depends(require_admin_role)
):
    """
    Compare multiple model architectures (Admin only).
    
    Trains and evaluates all available model types and returns comparison metrics.
    """
    try:
        # Generate job ID
        job_id = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user['sub']}"
        
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
        
        # Initialize job status
        training_jobs[job_id] = {
            'job_id': job_id,
            'status': 'queued',
            'progress': 0.0,
            'message': 'Model comparison job queued',
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'metrics': None,
            'error': None,
            'user': current_user['sub'],
            'type': 'comparison'
        }
        
        # Start background comparison
        async def compare_models_background():
            try:
                training_jobs[job_id]['status'] = 'running'
                training_jobs[job_id]['message'] = 'Comparing models...'
                
                results = model_comparison(
                    X_train, y_train, X_test, y_test,
                    n_repeats=n_repeats
                )
                
                training_jobs[job_id]['status'] = 'completed'
                training_jobs[job_id]['progress'] = 1.0
                training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
                training_jobs[job_id]['message'] = 'Model comparison completed'
                training_jobs[job_id]['metrics'] = results
                
                logger.info(f"Model comparison job {job_id} completed")
                
            except Exception as e:
                logger.error(f"Model comparison job {job_id} failed: {str(e)}")
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = str(e)
                training_jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        background_tasks.add_task(compare_models_background)
        
        logger.info(f"Model comparison job {job_id} queued by user {current_user['sub']}")
        
        return TrainingResponse(
            job_id=job_id,
            status='queued',
            message='Model comparison started in background',
            started_at=training_jobs[job_id]['started_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model comparison: {str(e)}"
        )
