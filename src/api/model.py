from fastapi import APIRouter, Body, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging
import os

from src.services.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize component
model_manager = ModelManager()

@router.post('/calibrate', summary="Calibrate model", response_description="Calibration results")
async def calibrate_model(
    request: Dict[str, Any] = Body(..., description="Calibration request with model_name and data"),
    background_tasks: BackgroundTasks = None
):
    """
    Calibrate a model with new data.
    
    Request body should include:
    - model_name: Name of the model to calibrate (e.g., 'enhanced_cnn_lstm') or 'all'
    - calibration_data: The calibration dataset
    """
    try:
        model_name = request.get('model_name')
        if not model_name:
            raise HTTPException(
                status_code=400, 
                detail="model_name is required in request body (e.g., 'enhanced_cnn_lstm' or 'all')"
            )
        
        calibration_data = request.get('calibration_data', {})
        
        # Check if model exists
        model_path = f"model/{model_name}.h5"
        if model_name != 'all' and not os.path.exists(model_path):
            available_models = [f.replace('.h5', '') for f in os.listdir('model') if f.endswith('.h5')]
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        # Placeholder for actual calibration logic
        # In a real implementation, this would:
        # 1. Load the model
        # 2. Apply calibration techniques (e.g., temperature scaling, Platt scaling)
        # 3. Save the calibrated model
        
        return {
            "status": "calibration_started", 
            "message": f"Calibration process initiated for model: {model_name}",
            "model_name": model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
