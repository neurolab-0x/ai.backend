from __future__ import annotations

from datetime import datetime
import logging
import os
import tempfile
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
from src.services.storage import MinioStorageService

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized model registry/cache.

    This replaces multiple ad-hoc caches spread across the codebase.
    """

    def __init__(self, model_types: Optional[List[str]] = None, model_dir: str = "model"):
        self.model_types = model_types or sorted(VALID_MODEL_TYPES)
        self.model_dir = model_dir
        self.model_cache_dir = os.getenv(
            "MODEL_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "neurolab_active_models"),
        )
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.metadata: Dict[str, Dict[str, object]] = {}
        self.model_mtimes: Dict[str, float] = {}
        self.scaler_mtimes: Dict[str, float] = {}
        self.metadata_mtimes: Dict[str, float] = {}
        self.remote_model_etags: Dict[str, Optional[str]] = {}
        self.remote_scaler_etags: Dict[str, Optional[str]] = {}
        self.remote_metadata_etags: Dict[str, Optional[str]] = {}
        self.tensorflow_available = False
        self.storage = MinioStorageService()
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
        return get_model_artifact_paths(
            model_type,
            base_dir=self._resolve_artifact_base_dir(model_type),
        )["model_path"]

    def _local_artifacts_complete(self, model_type: str, base_dir: str) -> bool:
        paths = get_model_artifact_paths(model_type, base_dir=base_dir)
        return all(
            os.path.exists(paths[key])
            for key in ("model_path", "scaler_path", "metadata_path")
        )

    def _remote_active_object_names(self, model_type: str) -> Dict[str, str]:
        return {
            "model_path": f"active/{model_type}/model.keras",
            "scaler_path": f"active/{model_type}/scaler.joblib",
            "metadata_path": f"active/{model_type}/metadata.json",
        }

    def _ensure_cached_remote_bundle(self, model_type: str) -> bool:
        if not self.storage.enabled:
            return False

        remote_objects = self._remote_active_object_names(model_type)
        remote_stats = {
            key: self.storage.stat_file("models", object_name)
            for key, object_name in remote_objects.items()
        }
        if not all(remote_stats.values()):
            return False

        cache_paths = get_model_artifact_paths(model_type, base_dir=self.model_cache_dir)
        os.makedirs(cache_paths["artifact_dir"], exist_ok=True)

        expected_etags = {
            "model_path": remote_stats["model_path"]["etag"],
            "scaler_path": remote_stats["scaler_path"]["etag"],
            "metadata_path": remote_stats["metadata_path"]["etag"],
        }
        cached_etags = {
            "model_path": self.remote_model_etags.get(model_type),
            "scaler_path": self.remote_scaler_etags.get(model_type),
            "metadata_path": self.remote_metadata_etags.get(model_type),
        }

        for key, object_name in remote_objects.items():
            local_path = cache_paths[key]
            if (
                not os.path.exists(local_path)
                or cached_etags[key] != expected_etags[key]
            ):
                downloaded = self.storage.download_file("models", object_name, local_path)
                if not downloaded:
                    return False

        self.remote_model_etags[model_type] = expected_etags["model_path"]
        self.remote_scaler_etags[model_type] = expected_etags["scaler_path"]
        self.remote_metadata_etags[model_type] = expected_etags["metadata_path"]
        return self._local_artifacts_complete(model_type, self.model_cache_dir)

    def _resolve_artifact_base_dir(self, model_type: str) -> str:
        model_type = sanitize_model_type(model_type)
        if self._local_artifacts_complete(model_type, self.model_dir):
            return self.model_dir
        if self._ensure_cached_remote_bundle(model_type):
            return self.model_cache_dir
        return self.model_dir

    def _artifact_mtime(self, path: str) -> Optional[float]:
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def list_model_files(self) -> List[str]:
        """List model filenames present on disk."""
        entries: List[str] = []
        for base_dir in (self.model_dir, self.model_cache_dir):
            if not os.path.exists(base_dir):
                continue
            for name in sorted(os.listdir(base_dir)):
                path = os.path.join(base_dir, name)
                if os.path.isdir(path):
                    artifact_model = os.path.join(path, "model.keras")
                    if os.path.exists(artifact_model):
                        rel = f"{name}/model.keras"
                        if rel not in entries:
                            entries.append(rel)
                elif base_dir == self.model_dir and (name.endswith(".h5") or name.endswith(".keras")):
                    if name not in entries:
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
            base_dir = self._resolve_artifact_base_dir(model_type)
            model_path = get_model_artifact_paths(model_type, base_dir=base_dir)["model_path"]
            current_mtime = self._artifact_mtime(model_path)
            cached_mtime = self.model_mtimes.get(model_type)
            if model_type in self.models and current_mtime == cached_mtime:
                return self.models[model_type]

            logger.info(f"Loading model '{model_type}' from {model_path}")
            model = load_calibrated_model(model_type, base_dir=base_dir)
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
            self.model_mtimes[model_type] = current_mtime or -1.0
            return model

    def get_scaler(self, model_type: str):
        model_type = sanitize_model_type(model_type)
        with self._lock:
            base_dir = self._resolve_artifact_base_dir(model_type)
            scaler_path = get_model_artifact_paths(model_type, base_dir=base_dir)["scaler_path"]
            current_mtime = self._artifact_mtime(scaler_path)
            cached_mtime = self.scaler_mtimes.get(model_type)
            if model_type in self.scalers and current_mtime == cached_mtime:
                return self.scalers[model_type]

            scaler = load_scaler_artifact(model_type, base_dir=base_dir)
            if scaler is None:
                logger.error(f"Scaler artifact missing for model '{model_type}'")
                return None
            self.scalers[model_type] = scaler
            self.scaler_mtimes[model_type] = current_mtime or -1.0
            return scaler

    def get_metadata(self, model_type: str) -> Optional[Dict[str, object]]:
        model_type = sanitize_model_type(model_type)
        with self._lock:
            base_dir = self._resolve_artifact_base_dir(model_type)
            metadata_path = get_model_artifact_paths(model_type, base_dir=base_dir)["metadata_path"]
            current_mtime = self._artifact_mtime(metadata_path)
            cached_mtime = self.metadata_mtimes.get(model_type)
            if model_type in self.metadata and current_mtime == cached_mtime:
                return self.metadata[model_type]

            metadata = load_model_metadata(model_type, base_dir=base_dir)
            if metadata is None:
                logger.error(f"Metadata artifact missing for model '{model_type}'")
                return None
            self.metadata[model_type] = metadata
            self.metadata_mtimes[model_type] = current_mtime or -1.0
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
