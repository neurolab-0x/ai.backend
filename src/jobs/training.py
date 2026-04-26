import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from rq import get_current_job

from src.core.ml.model import evaluate_model, model_comparison, train_hybrid_model
from src.jobs.training_persistence import update_training_run
from src.preprocessing.labeling import label_eeg_states
from src.preprocessing.load_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.queue import publish_job_event
from src.services.storage import MinioStorageService
from src.services.training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)


def _current_job_id() -> str:
    job = get_current_job()
    return job.id if job else f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _set_meta(progress: float, message: str, **extra):
    job = get_current_job()
    if not job:
        return
    job.meta["progress"] = float(progress)
    job.meta["message"] = message
    for k, v in extra.items():
        job.meta[k] = v
    job.save_meta()


def _update_run(job_id: str, status: str, progress: float, message: str, **extra) -> None:
    payload = {
        "status": status,
        "progress": float(progress),
        "message": message,
    }
    payload.update(extra)
    update_training_run(job_id, payload)


def _publish(job_id: str, event: str, payload: Optional[Dict[str, Any]] = None, persist_state: bool = True) -> None:
    publish_job_event("training", job_id, event, payload or {}, persist_state=persist_state)


def _cleanup_file(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _build_completed_event(
    *,
    job_id: str,
    model_type: Optional[str],
    config: Dict[str, Any],
    result: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "completed",
        "model_type": model_type,
        "subject_id": config.get("subject_id"),
        "session_id": config.get("session_id"),
        "metrics": result,
        "artifacts": artifacts,
        "completed_at": datetime.now().isoformat(),
    }


def train_from_npz(
    npz_path: str,
    config: Dict[str, Any],
    model_type: str,
    existing_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    RQ job: loads arrays from npz, trains a model, evaluates, uploads artifact.
    Returns a serializable metrics dict.
    """
    job_id = _current_job_id()
    monitor = TrainingMonitor()
    storage_service = MinioStorageService()
    temp_model_path = f"processed/model_{job_id}.h5"
    artifacts: Dict[str, Any] = dict(existing_artifacts or {})
    artifacts["training_bundle_local_path"] = npz_path

    try:
        _set_meta(0.05, "Loading training data...")
        _update_run(job_id, "loading_data", 0.05, "Loading training data...", artifacts=artifacts)
        _publish(job_id, "loading_data", {"progress": 0.05})
        data = np.load(npz_path, allow_pickle=False)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"] if "X_test" in data else None
        y_test = data["y_test"] if "y_test" in data else None

        monitor.log_training_event(
            job_id,
            "STARTED",
            {
                "job_id": job_id,
                "subject_id": config.get("subject_id"),
                "session_id": config.get("session_id"),
                "model_type": model_type,
                "epochs": config.get("epochs"),
            },
        )

        _set_meta(0.1, "Training model...")
        _update_run(job_id, "training", 0.1, "Training model...", artifacts=artifacts)
        _publish(job_id, "training", {"progress": 0.1, "model_type": model_type})
        validation_data = None
        if X_test is not None and y_test is not None and len(X_test) and len(y_test):
            validation_data = (X_test, y_test)
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
            validation_data=validation_data,
        )

        _set_meta(0.8, "Evaluating model...")
        _update_run(job_id, "evaluating", 0.8, "Evaluating model...", artifacts=artifacts)
        _publish(job_id, "evaluating", {"progress": 0.8})
        metrics: Dict[str, Any] = {}
        if X_test is not None and y_test is not None and len(X_test) and len(y_test):
            metrics = evaluate_model(model, X_test, y_test, calibrate=True)

        _set_meta(0.9, "Saving model artifact...")
        _update_run(job_id, "saving_artifacts", 0.9, "Saving model artifact...", artifacts=artifacts)
        _publish(job_id, "saving_artifacts", {"progress": 0.9})
        os.makedirs("processed", exist_ok=True)
        model.save(temp_model_path)
        artifacts["model_local_path"] = temp_model_path

        if storage_service.enabled:
            try:
                _set_meta(0.95, "Uploading model to object storage...")
                _update_run(job_id, "uploading_artifacts", 0.95, "Uploading model artifact...", artifacts=artifacts)
                _publish(job_id, "uploading_artifacts", {"progress": 0.95})
                object_name = storage_service.upload_file(temp_model_path, "models", f"{job_id}/model.h5")
                if object_name:
                    artifacts["model_object_name"] = object_name
                    artifacts["model_url"] = storage_service.get_file_url("models", f"{job_id}/model.h5")
            except Exception as exc:
                logger.warning(f"MinIO upload failed: {exc}")

        result = {
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history.get("val_accuracy", [None])[-1]) if history.history.get("val_accuracy") else None,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history.get("val_loss", [None])[-1]) if history.history.get("val_loss") else None,
            "test_metrics": metrics or None,
        }

        _set_meta(1.0, "Training completed.", artifacts=artifacts)
        _update_run(
            job_id,
            "completed",
            1.0,
            "Training completed.",
            metrics=result,
            artifacts=artifacts,
            error=None,
            completed_at=datetime.now(),
        )
        completed_payload = _build_completed_event(
            job_id=job_id,
            model_type=model_type,
            config=config,
            result=result,
            artifacts=artifacts,
        )
        _publish(job_id, "completed", completed_payload)
        monitor.log_training_event(
            job_id,
            "COMPLETED",
            {
                "job_id": job_id,
                "subject_id": config.get("subject_id"),
                "session_id": config.get("session_id"),
                "model_type": model_type,
                "final_accuracy": result["final_train_accuracy"],
                "final_loss": result["final_train_loss"],
            },
        )
        return {
            **result,
            "artifacts": artifacts,
        }
    except Exception as exc:
        _set_meta(1.0, f"Training failed: {exc}")
        _update_run(
            job_id,
            "failed",
            1.0,
            f"Training failed: {exc}",
            error=str(exc),
            artifacts=artifacts,
            completed_at=datetime.now(),
        )
        _publish(job_id, "failed", {"job_id": job_id, "error": str(exc), "artifacts": artifacts})
        monitor.log_training_event(
            job_id,
            "FAILED",
            {
                "job_id": job_id,
                "subject_id": config.get("subject_id"),
                "session_id": config.get("session_id"),
                "model_type": model_type,
                "error": str(exc),
            },
        )
        raise
    finally:
        _cleanup_file(npz_path)
        _cleanup_file(temp_model_path)
        monitor.close()


def train_from_file(file_path: str, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    job_id = _current_job_id()
    storage_service = MinioStorageService()
    artifacts: Dict[str, Any] = {
        "uploaded_file_local_path": file_path,
        "uploaded_file_name": os.path.basename(file_path),
    }
    npz_path = os.path.join("temp", f"{job_id}.npz")

    try:
        if storage_service.enabled:
            try:
                object_name = storage_service.upload_file(file_path, "training", f"{job_id}/input/{os.path.basename(file_path)}")
                if object_name:
                    artifacts["uploaded_file_object_name"] = object_name
                    artifacts["uploaded_file_url"] = storage_service.get_file_url("training", f"{job_id}/input/{os.path.basename(file_path)}")
            except Exception as exc:
                logger.warning(f"Training input upload failed: {exc}")

        _set_meta(0.05, "Loading and preprocessing file...")
        _update_run(job_id, "preprocessing", 0.05, "Loading and preprocessing file...", artifacts=artifacts)
        _publish(job_id, "preprocessing", {"progress": 0.05})
        df = load_data(file_path)
        df = label_eeg_states(df)

        X_train, X_test, y_train, y_test, _metadata = preprocess_data(
            df,
            overlap=config.get("overlap", 0.5),
            simple_mode=config.get("simple_mode", True),
        )
        np.savez_compressed(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        artifacts["training_bundle_local_path"] = npz_path
        update_training_run(job_id, {"artifacts": artifacts})
        return train_from_npz(npz_path, config, model_type, existing_artifacts=artifacts)
    finally:
        _cleanup_file(npz_path)
        _cleanup_file(file_path)


def compare_models_from_npz(npz_path: str, n_repeats: int = 3) -> Dict[str, Any]:
    job_id = _current_job_id()
    artifacts: Dict[str, Any] = {"comparison_bundle_local_path": npz_path}
    try:
        _set_meta(0.05, "Loading data for comparison...")
        _update_run(job_id, "loading_data", 0.05, "Loading data for comparison...", artifacts=artifacts)
        _publish(job_id, "loading_data", {"progress": 0.05})
        data = np.load(npz_path, allow_pickle=False)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        _set_meta(0.1, "Comparing model architectures...")
        _update_run(job_id, "comparing", 0.1, "Comparing model architectures...", artifacts=artifacts)
        _publish(job_id, "comparing", {"progress": 0.1, "n_repeats": n_repeats})
        results = model_comparison(X_train, y_train, X_test, y_test, n_repeats=n_repeats)
        _set_meta(1.0, "Model comparison completed.")
        _update_run(
            job_id,
            "completed",
            1.0,
            "Model comparison completed.",
            metrics=results,
            artifacts=artifacts,
            completed_at=datetime.now(),
        )
        _publish(job_id, "completed", {"job_id": job_id, "metrics": results, "artifacts": artifacts})
        return results
    except Exception as exc:
        _set_meta(1.0, f"Model comparison failed: {exc}")
        _update_run(
            job_id,
            "failed",
            1.0,
            f"Model comparison failed: {exc}",
            error=str(exc),
            artifacts=artifacts,
            completed_at=datetime.now(),
        )
        _publish(job_id, "failed", {"job_id": job_id, "error": str(exc), "artifacts": artifacts})
        raise
    finally:
        _cleanup_file(npz_path)
