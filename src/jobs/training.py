import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from rq import get_current_job

from src.core.ml.model import train_hybrid_model, evaluate_model, model_comparison
from src.preprocessing.load_data import load_data
from src.preprocessing.labeling import label_eeg_states
from src.preprocessing.preprocess import preprocess_data
from src.services.storage import MinioStorageService
from src.services.training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)


def _set_meta(progress: float, message: str, **extra):
    job = get_current_job()
    if not job:
        return
    job.meta["progress"] = float(progress)
    job.meta["message"] = message
    for k, v in extra.items():
        job.meta[k] = v
    job.save_meta()


def train_from_npz(npz_path: str, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    RQ job: loads arrays from npz, trains a model, evaluates, uploads artifact.
    Returns a serializable metrics dict.
    """
    _set_meta(0.05, "Loading training data...")
    data = np.load(npz_path, allow_pickle=False)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"] if "X_test" in data else None
    y_test = data["y_test"] if "y_test" in data else None

    monitor = TrainingMonitor()
    storage_service = MinioStorageService()

    job = get_current_job()
    job_id = job.id if job else f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    monitor.log_training_event(job_id, "STARTED", {"model_type": model_type, "epochs": config.get("epochs")})

    _set_meta(0.1, "Training model...")
    model, history = train_hybrid_model(
        X_train,
        y_train,
        model_type=model_type,
        epochs=config.get("epochs", 30),
        batch_size=config.get("batch_size", 32),
        learning_rate=config.get("learning_rate", 0.001),
        dropout_rate=config.get("dropout_rate", 0.3),
        use_separable=config.get("use_separable", True),
        use_relative_pos=config.get("use_relative_pos", True),
        l1_reg=config.get("l1_reg", 1e-5),
        l2_reg=config.get("l2_reg", 1e-4),
        subject_id=config.get("subject_id"),
        session_id=config.get("session_id"),
        overlap=config.get("overlap", 0.5),
        simple_mode=config.get("simple_mode", True),
    )

    _set_meta(0.8, "Evaluating model...")
    metrics = {}
    if X_test is not None and y_test is not None and len(X_test) and len(y_test):
        metrics = evaluate_model(model, X_test, y_test, calibrate=True)

    _set_meta(0.9, "Saving model artifact...")
    os.makedirs("processed", exist_ok=True)
    temp_model_path = f"processed/model_{job_id}.h5"
    model.save(temp_model_path)

    model_url = None
    if storage_service.enabled:
        try:
            _set_meta(0.95, "Uploading model to object storage...")
            object_name = storage_service.upload_file(temp_model_path, "models", f"{job_id}/model.h5")
            if object_name:
                model_url = storage_service.get_file_url("models", f"{job_id}/model.h5")
        except Exception as e:
            logger.warning(f"MinIO upload failed: {e}")

    _set_meta(1.0, "Training completed.", model_url=model_url)
    monitor.log_training_event(job_id, "COMPLETED", {"final_accuracy": float(history.history["accuracy"][-1])})

    result = {
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history.get("val_accuracy", [None])[-1]) if history.history.get("val_accuracy") else None,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [None])[-1]) if history.history.get("val_loss") else None,
        "test_metrics": metrics or None,
        "model_url": model_url,
    }

    try:
        os.remove(npz_path)
    except OSError:
        pass

    return result


def train_from_file(file_path: str, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    _set_meta(0.05, "Loading and preprocessing file...")
    df = load_data(file_path)
    df = label_eeg_states(df)

    X_train, X_test, y_train, y_test, _metadata = preprocess_data(
        df,
        overlap=config.get("overlap", 0.5),
        simple_mode=config.get("simple_mode", True),
    )

    os.makedirs("temp", exist_ok=True)
    npz_path = os.path.join("temp", f"train_{get_current_job().id}.npz")
    np.savez_compressed(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    try:
        return train_from_npz(npz_path, config, model_type)
    finally:
        try:
            os.remove(npz_path)
        except OSError:
            pass
        try:
            os.remove(file_path)
        except OSError:
            pass


def compare_models_from_npz(npz_path: str, n_repeats: int = 3) -> Dict[str, Any]:
    _set_meta(0.05, "Loading data for comparison...")
    data = np.load(npz_path, allow_pickle=False)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    _set_meta(0.1, "Comparing model architectures...")
    results = model_comparison(X_train, y_train, X_test, y_test, n_repeats=n_repeats)
    _set_meta(1.0, "Model comparison completed.")
    try:
        os.remove(npz_path)
    except OSError:
        pass
    return results
