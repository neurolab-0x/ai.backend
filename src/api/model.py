from fastapi import APIRouter, Body, HTTPException
from typing import Dict, Any
import logging
import os
from src.core.ml.model_types import sanitize_model_type
from src.core.ml.model import get_model_artifact_paths

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/calibrate', summary="Calibrate model", response_description="Calibration results")
async def calibrate_model_endpoint(
    request: Dict[str, Any] = Body(..., description="Calibration request with model_name and data"),
):
    """
    Calibrate a model with new data using temperature scaling.
    
    Request body should include:
    - model_name: Name of the model to calibrate (e.g., 'enhanced_cnn_lstm') or 'all'
    - calibration_data: Dict with 'X' (features) and 'y' (labels) for calibration
    """
    try:
        from src.core.ml.model import calibrate_model, load_calibrated_model
        import numpy as np
        
        model_name = request.get('model_name')
        if not model_name:
            raise HTTPException(
                status_code=400, 
                detail="model_name is required in request body (e.g., 'enhanced_cnn_lstm' or 'all')"
            )
        if model_name != 'all':
            try:
                model_name = sanitize_model_type(model_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        
        calibration_data = request.get('calibration_data')
        if not calibration_data:
            raise HTTPException(
                status_code=400,
                detail="calibration_data is required with 'X' (features) and 'y' (labels)"
            )
        
        # Extract calibration features and labels
        X_cal = np.array(calibration_data.get('X', []))
        y_cal = np.array(calibration_data.get('y', []))
        
        if len(X_cal) == 0 or len(y_cal) == 0:
            raise HTTPException(
                status_code=400,
                detail="calibration_data must contain non-empty 'X' and 'y' arrays"
            )
        
        # Reshape if needed
        if len(X_cal.shape) == 2:
            X_cal = X_cal.reshape(-1, X_cal.shape[1], 1)
        
        # Determine which models to calibrate
        if model_name == 'all':
            model_names = []
            if os.path.exists("model"):
                for entry in os.listdir("model"):
                    entry_path = os.path.join("model", entry)
                    if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "model.keras")):
                        model_names.append(entry)
                    elif entry.endswith(".h5"):
                        model_names.append(entry.replace(".h5", ""))
        else:
            model_names = [model_name]
        
        results = {}
        for name in model_names:
            paths = get_model_artifact_paths(name)
            model_path = paths["model_path"] if os.path.exists(paths["model_path"]) else paths["legacy_model_path"]
            if not os.path.exists(model_path):
                results[name] = {"status": "error", "message": f"Model file not found for model '{name}'"}
                continue
            
            try:
                # Load the model
                logger.info(f"Loading model: {name}")
                model = load_calibrated_model(name)
                
                if model is None:
                    results[name] = {"status": "error", "message": "Failed to load model"}
                    continue
                
                # Calibrate the model
                logger.info(f"Calibrating model: {name}")
                calibrated_model = calibrate_model(model, X_cal, y_cal)
                
                # Save calibrated model
                calibrated_path = os.path.join(paths["artifact_dir"], "model_calibrated.keras")
                os.makedirs(paths["artifact_dir"], exist_ok=True)
                calibrated_model.save(calibrated_path)
                
                results[name] = {
                    "status": "success",
                    "message": f"Model calibrated and saved to {calibrated_path}",
                    "calibrated_path": calibrated_path
                }
                logger.info(f"Successfully calibrated {name}")
                
            except Exception as e:
                logger.error(f"Error calibrating {name}: {str(e)}")
                results[name] = {"status": "error", "message": str(e)}
        
        return {
            "status": "completed",
            "results": results,
            "models_processed": len(model_names)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in calibration endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
