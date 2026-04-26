import os
import logging
from typing import Dict, Any
from src.core.ml.model import get_model_artifact_paths, load_calibrated_model as _load_model

logger = logging.getLogger(__name__)

def get_available_models() -> Dict[str, Any]:
    """
    Get information about available trained models.
    
    Returns:
        Dict[str, Any]: Dictionary containing model information
    """
    models_dir = "./model"
    available_models = {}
    
    try:
        if not os.path.exists(models_dir):
            return available_models
            
        for filename in os.listdir(models_dir):
            model_path = os.path.join(models_dir, filename)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model.keras")):
                model_info = {
                    "path": os.path.join(model_path, "model.keras"),
                    "size": os.path.getsize(os.path.join(model_path, "model.keras")),
                    "last_modified": os.path.getmtime(os.path.join(model_path, "model.keras"))
                }
                available_models[filename] = model_info
            elif filename.endswith('.h5'):
                model_info = {
                    "path": model_path,
                    "size": os.path.getsize(model_path),
                    "last_modified": os.path.getmtime(model_path)
                }
                available_models[filename] = model_info
                
        return available_models
        
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        return {}

def load_calibrated_model(model_path: str):
    """Alias for src.core.ml.model.load_calibrated_model for backward compatibility"""
    return _load_model(model_path)
