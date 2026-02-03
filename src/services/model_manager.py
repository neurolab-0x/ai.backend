import os
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from src.core.ml.model import load_calibrated_model

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path: str = "./model/trained_model.h5"):
        self.model_path = model_path
        self.model = None
        self.tensorflow_available = False
        self._initialize_tensorflow()
    
    def _initialize_tensorflow(self):
        """Initialize TensorFlow and load model if available"""
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            logger.info("TensorFlow loaded successfully")
            self._load_model()
        except ImportError as e:
            logger.warning(f"TensorFlow import failed: {str(e)}")
            self.tensorflow_available = False
    
    def _load_model(self):
        """Load and warm up the model"""
        if not self.tensorflow_available:
            return
            
        try:
            logger.info(f"Initializing model from {self.model_path}")
            if os.path.exists(self.model_path):
                self.model = load_calibrated_model(self.model_path)
                if self.model is not None:
                    self.model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    # Warm up the model
                    dummy_input = np.zeros((1, *self.model.input_shape[1:]))
                    _ = self.model.predict(dummy_input)
                    logger.info("Model loaded and warmed up")
                else:
                    logger.error("Failed to load or create model")
            else:
                logger.warning(f"Model file not found at {self.model_path}. Running in minimal mode.")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            self.model = None
    
    def get_health_status(self) -> dict:
        """Get model health status"""
        status = {
            "status": "ok",
            "model_loaded": self.model is not None,
            "tensorflow_available": self.tensorflow_available,
            "model_path": self.model_path,
            "model_version": "1.0.0",
            "system_time": datetime.now().isoformat()
        }
        
        if self.model is not None:
            try:
                dummy_input = np.zeros((1, *self.model.input_shape[1:]))
                start_time = datetime.now()
                _ = self.model.predict(dummy_input)
                latency = (datetime.now() - start_time).total_seconds() * 1000
                status["inference_latency_ms"] = round(latency, 2)
            except Exception as e:
                logger.error(f"Model health check failed: {str(e)}")
                status["model_loaded"] = False
        
        return status