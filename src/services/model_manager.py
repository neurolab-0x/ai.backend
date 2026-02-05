from typing import List, Dict
import os
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from src.core.ml.model import load_calibrated_model

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_types: List[str] = ["enhanced_cnn_lstm", "resnet_lstm", "transformer", "original"]):
        self.model_types = model_types
        self.models = {}
        self.tensorflow_available = False
        self._initialize_tensorflow()
        
    def _initialize_tensorflow(self):
        """Initialize TensorFlow and load models if available"""
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            logger.info("TensorFlow loaded successfully")
            self._load_models()
        except ImportError as e:
            logger.warning(f"TensorFlow import failed: {str(e)}")
            self.tensorflow_available = False
    
    def _load_models(self):
        """Load and warm up all configured models"""
        if not self.tensorflow_available:
            return
            
        for model_type in self.model_types:
            model_path = f"./model/{model_type}.h5"
            try:
                logger.info(f"Initializing {model_type} from {model_path}")
                if os.path.exists(model_path):
                    model = load_calibrated_model(model_path)
                    if model is not None:
                        # Compile to avoid warnings/errors during inference
                        model.compile(
                            optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy']
                        )
                        # Warm up the model
                        try:
                            dummy_input = np.zeros((1, *model.input_shape[1:]))
                            _ = model.predict(dummy_input, verbose=0)
                            self.models[model_type] = model
                            logger.info(f"Model {model_type} loaded and warmed up")
                        except Exception as e:
                            logger.warning(f"Model {model_type} loaded but warmup failed: {e}")
                            self.models[model_type] = model
                    else:
                        logger.error(f"Failed to load or create model {model_type}")
                else:
                    logger.info(f"Model file not found at {model_path}. Skipping pre-load.")
            except Exception as e:
                logger.error(f"Model initialization failed for {model_type}: {str(e)}")
    
    def get_health_status(self) -> dict:
        """Get model health status"""
        status = {
            "status": "ok",
            "models_loaded": list(self.models.keys()),
            "models_count": len(self.models),
            "tensorflow_available": self.tensorflow_available,
            "model_version": "3.1.0",
            "system_time": datetime.now().isoformat()
        }
        
        # Check latency for first available model
        if self.models:
            try:
                first_model = next(iter(self.models.values()))
                dummy_input = np.zeros((1, *first_model.input_shape[1:]))
                start_time = datetime.now()
                _ = first_model.predict(dummy_input, verbose=0)
                latency = (datetime.now() - start_time).total_seconds() * 1000
                status["inference_latency_ms"] = round(latency, 2)
            except Exception as e:
                logger.error(f"Model health check failed: {str(e)}")
                status["health_check_error"] = str(e)
        
        return status