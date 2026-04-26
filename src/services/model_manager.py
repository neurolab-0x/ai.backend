from __future__ import annotations

from datetime import datetime
import logging
import os
from threading import Lock
from typing import Dict, List, Optional

import numpy as np

from src.core.ml.model import (
    get_model_artifact_paths,
    load_calibrated_model,
    load_model_metadata,
    load_scaler_artifact,
)
from src.core.ml.model_types import sanitize_model_type, VALID_MODEL_TYPES

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized model registry/cache.

    This replaces multiple ad-hoc caches spread across the codebase.
    """

    def __init__(self, model_types: Optional[List[str]] = None, model_dir: str = "model"):
        self.model_types = model_types or sorted(VALID_MODEL_TYPES)
        self.model_dir = model_dir
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.metadata: Dict[str, Dict[str, object]] = {}
        self.tensorflow_available = False
        self._lock = Lock()
        self._initialize_tensorflow()
        
    def _initialize_tensorflow(self):
        """Initialize TensorFlow and load models if available"""
        try:
            import tensorflow as _tf  # noqa: F401
            self.tensorflow_available = True
            logger.info("TensorFlow loaded successfully")
        except ImportError as e:
            logger.warning(f"TensorFlow import failed: {str(e)}")
            self.tensorflow_available = False
    
    def _model_path(self, model_type: str) -> str:
        model_type = sanitize_model_type(model_type)
        return get_model_artifact_paths(model_type, base_dir=self.model_dir)["model_path"]

    def list_model_files(self) -> List[str]:
        """List model filenames present on disk."""
        if not os.path.exists(self.model_dir):
            return []
        entries: List[str] = []
        for name in sorted(os.listdir(self.model_dir)):
            path = os.path.join(self.model_dir, name)
            if os.path.isdir(path):
                artifact_model = os.path.join(path, "model.keras")
                if os.path.exists(artifact_model):
                    entries.append(f"{name}/model.keras")
            elif name.endswith(".h5") or name.endswith(".keras"):
                entries.append(name)
        return entries

    def get_model(self, model_type: str, warmup: bool = True):
        """
        Load a model once and cache it.
        model_type is a validated architecture identifier (not an arbitrary path).
        """
        if not self.tensorflow_available:
            return None

        model_type = sanitize_model_type(model_type)
        with self._lock:
            if model_type in self.models:
                return self.models[model_type]

            model_path = self._model_path(model_type)
            logger.info(f"Loading model '{model_type}' from {model_path}")
            model = load_calibrated_model(model_type)
            if model is None:
                logger.error(f"Failed to load model '{model_type}'")
                return None

            if warmup:
                try:
                    dummy_input = np.zeros((1, *model.input_shape[1:]))
                    _ = model.predict(dummy_input, verbose=0)
                except Exception as e:
                    logger.warning(f"Model '{model_type}' warmup failed: {e}")

            self.models[model_type] = model
            return model

    def get_scaler(self, model_type: str):
        model_type = sanitize_model_type(model_type)
        with self._lock:
            if model_type in self.scalers:
                return self.scalers[model_type]

            scaler = load_scaler_artifact(model_type, base_dir=self.model_dir)
            if scaler is None:
                logger.error(f"Scaler artifact missing for model '{model_type}'")
                return None
            self.scalers[model_type] = scaler
            return scaler

    def get_metadata(self, model_type: str) -> Optional[Dict[str, object]]:
        model_type = sanitize_model_type(model_type)
        with self._lock:
            if model_type in self.metadata:
                return self.metadata[model_type]

            metadata = load_model_metadata(model_type, base_dir=self.model_dir)
            if metadata is None:
                logger.error(f"Metadata artifact missing for model '{model_type}'")
                return None
            self.metadata[model_type] = metadata
            return metadata

    def warmup_all(self) -> None:
        """Best-effort warmup of all known model types."""
        for mt in self.model_types:
            try:
                self.get_model(mt, warmup=True)
            except Exception as e:
                logger.warning(f"Warmup failed for model '{mt}': {e}")
    
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


_global_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager()
    return _global_manager
