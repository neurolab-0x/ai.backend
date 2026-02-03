import os
import logging
import numpy as np
from tensorflow import keras
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

# load_calibrated_model has been moved to core/ml/model.py
# Import it from there: from core.ml.model import load_calibrated_model

def get_available_models() -> Dict[str, Any]:
    """
    Get information about available trained models.
    
    Returns:
        Dict[str, Any]: Dictionary containing model information
    """
    models_dir = "./processed"
    available_models = {}
    
    try:
        if not os.path.exists(models_dir):
            return available_models
            
        for filename in os.listdir(models_dir):
            if filename.endswith('.h5'):
                model_path = os.path.join(models_dir, filename)
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

def save_model(model: keras.Model, model_path: str) -> None:
    """
    Save a trained model to the specified path.
    
    Args:
        model (keras.Model): Model to save
        model_path (str): Path where to save the model
        
    Raises:
        ValueError: If model saving fails
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        model.save(model_path)
        logger.info(f"Model saved successfully to {model_path}")
        
    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}")
        raise ValueError(f"Failed to save model: {str(e)}")

def load_model_weights(model: keras.Model, weights_path: str) -> keras.Model:
    """
    Load weights into a model from the specified path.
    
    Args:
        model (keras.Model): Model to load weights into
        weights_path (str): Path to the weights file
        
    Returns:
        keras.Model: Model with loaded weights
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        ValueError: If weight loading fails
    """
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        model.load_weights(weights_path)
        logger.info(f"Model weights loaded successfully from {weights_path}")
        return model
        
    except Exception as e:
        logger.error(f"Model weights loading failed: {str(e)}")
        raise 