import logging
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from rq import get_current_job
except ImportError:
    def get_current_job():
        return None

from src.core.ml.model import (
    DEFAULT_FEATURE_NAMES,
    evaluate_model,
    get_model_artifact_paths,
    model_comparison,
    promote_model_artifacts,
    save_model_metadata,
    save_scaler_artifact,
    train_hybrid_model,
    tf,
)
from src.jobs.training_persistence import update_training_run
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
    if isinstance(payload.get("artifacts"), dict):
        payload["artifacts"] = _public_artifacts(payload["artifacts"])
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "scaler"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _job_run_paths(job_id: str) -> Dict[str, str]:
    run_root = os.path.join("temp", "training_runs", job_id)
    return {
        "run_root": run_root,
        "artifact_base_dir": os.path.join(run_root, "model_bundle"),
        "metrics_dir": os.path.join(run_root, "metrics"),
        "logs_dir": os.path.join(run_root, "logs"),
    }


def _artifact_descriptor(label: str, kind: str, **extra) -> Dict[str, Any]:
    descriptor = {
        "label": label,
        "kind": kind,
    }
    descriptor.update(extra)
    return descriptor


def _download_object_artifact(
    storage_service: MinioStorageService,
    descriptor: Dict[str, Any],
    destination_path: str,
) -> str:
    downloaded = storage_service.download_artifact(descriptor, destination_path)
    if not downloaded:
        bucket_key = descriptor.get("bucket_key", "unknown")
        object_name = descriptor.get("object_name", "unknown")
        raise FileNotFoundError(f"Failed to download artifact {bucket_key}/{object_name}")
    return downloaded


def _register_object_artifact(
    storage_service: MinioStorageService,
    artifacts: Dict[str, Any],
    *,
    key: str,
    local_path: str,
    bucket_key: str,
    object_name: str,
    label: str,
    kind: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    artifacts.setdefault("objects", {})
    if storage_service.enabled:
        uploaded_name = storage_service.upload_file(local_path, bucket_key, object_name)
        if uploaded_name:
            artifacts["objects"][key] = storage_service.build_artifact_descriptor(
                bucket_key,
                uploaded_name,
                label=label,
                kind=kind,
                content_type=content_type,
                metadata=metadata,
            )
            return
    artifacts["objects"][key] = _artifact_descriptor(
        label,
        kind,
        local_reference=os.path.basename(local_path),
        content_type=content_type,
        metadata=metadata,
    )


def _public_artifacts(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if key.endswith("_local_path") or key.endswith("_local_dir") or key == "model_upload_temp_path":
            continue
        if isinstance(value, dict):
            cleaned[key] = _public_artifacts(value)
        elif isinstance(value, list):
            cleaned[key] = [_public_artifacts(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def _build_training_history_payload(history) -> Dict[str, Any]:
    return {
        "epochs": len(history.history.get("loss", [])),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }


def _save_training_visualizations(run_paths: Dict[str, str], history, test_metrics: Optional[Dict[str, Any]]) -> List[str]:
    created_paths: List[str] = []
    metrics_dir = run_paths["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    history_json_path = os.path.join(metrics_dir, "history.json")
    _write_json(history_json_path, _build_training_history_payload(history))
    created_paths.append(history_json_path)

    if test_metrics:
        metrics_json_path = os.path.join(metrics_dir, "evaluation.json")
        _write_json(metrics_json_path, test_metrics)
        created_paths.append(metrics_json_path)

    try:
        import matplotlib.pyplot as plt

        accuracy_values = history.history.get("accuracy", [])
        val_accuracy_values = history.history.get("val_accuracy", [])
        loss_values = history.history.get("loss", [])
        val_loss_values = history.history.get("val_loss", [])
        if accuracy_values or loss_values:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            if accuracy_values:
                axes[0].plot(accuracy_values, label="train_accuracy")
            if val_accuracy_values:
                axes[0].plot(val_accuracy_values, label="val_accuracy")
            axes[0].set_title("Accuracy")
            axes[0].legend()
            if loss_values:
                axes[1].plot(loss_values, label="train_loss")
            if val_loss_values:
                axes[1].plot(val_loss_values, label="val_loss")
            axes[1].set_title("Loss")
            axes[1].legend()
            fig.tight_layout()
            curves_path = os.path.join(metrics_dir, "training_curves.png")
            fig.savefig(curves_path)
            plt.close(fig)
            created_paths.append(curves_path)

        confusion_matrix = (test_metrics or {}).get("confusion_matrix")
        if confusion_matrix:
            fig, ax = plt.subplots(figsize=(6, 5))
            matrix = np.array(confusion_matrix)
            heatmap = ax.imshow(matrix, cmap="Blues")
            fig.colorbar(heatmap)
            ax.set_title("Confusion Matrix")
            matrix_path = os.path.join(metrics_dir, "confusion_matrix.png")
            fig.savefig(matrix_path)
            plt.close(fig)
            created_paths.append(matrix_path)
    except Exception as exc:
        logger.warning(f"Failed to create training visualizations for {run_paths['run_root']}: {exc}")

    return created_paths


def _epoch_progress_callback(
    *,
    job_id: str,
    total_epochs: int,
    monitor: TrainingMonitor,
    artifacts: Dict[str, Any],
):
    if tf is None:
        return None

    def _on_epoch_end(epoch: int, logs: Optional[Dict[str, Any]] = None):
        metrics = {key: float(val) for key, val in (logs or {}).items() if isinstance(val, (int, float))}
        progress = 0.1 + (((epoch + 1) / max(total_epochs, 1)) * 0.65)
        message = f"Training epoch {epoch + 1}/{total_epochs}"
        _set_meta(progress, message, epoch=epoch + 1, epoch_metrics=metrics)
        _update_run(
            job_id,
            "training",
            progress,
            message,
            artifacts=artifacts,
            latest_epoch=epoch + 1,
            latest_epoch_metrics=metrics,
        )
        _publish(
            job_id,
            "epoch_completed",
            {
                "progress": progress,
                "epoch": epoch + 1,
                "total_epochs": total_epochs,
                "metrics": metrics,
            },
            persist_state=False,
        )
        monitor.log_metrics(job_id, epoch + 1, metrics)

    return tf.keras.callbacks.LambdaCallback(on_epoch_end=_on_epoch_end)


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
    existing_scaler: Any = None,
    feature_names: Optional[list[str]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    RQ job: loads arrays from npz, trains a model, evaluates, uploads artifact.
    Returns a serializable metrics dict.
    """
    job_id = _current_job_id()
    monitor = TrainingMonitor()
    storage_service = MinioStorageService()
    run_paths = _job_run_paths(job_id)
    os.makedirs(run_paths["artifact_base_dir"], exist_ok=True)
    os.makedirs(run_paths["metrics_dir"], exist_ok=True)
    os.makedirs(run_paths["logs_dir"], exist_ok=True)
    temp_model_path = os.path.join(run_paths["run_root"], "model_export.keras")
    artifacts: Dict[str, Any] = dict(existing_artifacts or {})
    artifacts["training_bundle_local_path"] = npz_path
    artifacts["run_root_local_dir"] = run_paths["run_root"]
    artifacts["storage_backend"] = "minio" if storage_service.enabled else "local"

    try:
        _set_meta(0.05, "Loading training data...")
        _update_run(job_id, "loading_data", 0.05, "Loading training data...", artifacts=artifacts)
        _publish(job_id, "loading_data", {"progress": 0.05})
        data = np.load(npz_path, allow_pickle=False)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"] if "X_test" in data else None
        y_test = data["y_test"] if "y_test" in data else None
        if os.path.exists(npz_path) and "training_bundle" not in (artifacts.get("objects") or {}):
            _register_object_artifact(
                storage_service,
                artifacts,
                key="training_bundle",
                local_path=npz_path,
                bucket_key="training",
                object_name=f"runs/{job_id}/input/training_bundle.npz",
                label="Training bundle",
                kind="dataset_bundle",
                content_type="application/octet-stream",
            )
        input_feature_names = feature_names or (
            DEFAULT_FEATURE_NAMES if X_train.shape[1] == len(DEFAULT_FEATURE_NAMES) else [f"feature_{i}" for i in range(X_train.shape[1])]
        )
        scaler = existing_scaler or StandardScaler()
        if existing_scaler is None:
            X_train = scaler.fit_transform(X_train)
            if X_test is not None and y_test is not None and len(X_test) and len(y_test):
                X_test = scaler.transform(X_test)
        save_scaler_artifact(model_type, scaler, base_dir=run_paths["artifact_base_dir"])
        metadata_payload = {
            "input_features": input_feature_names,
            "scaler": scaler.__class__.__name__,
            "model_type": model_type,
            "job_id": job_id,
            "dataset_version": config.get("dataset_version", "unknown"),
            "training_mode": config.get("validation_mode", "split"),
            "source": config.get("source", "training_job"),
        }
        if dataset_metadata:
            metadata_payload["dataset_metadata"] = _json_safe(dataset_metadata)
        metadata_path = save_model_metadata(model_type, metadata_payload, base_dir=run_paths["artifact_base_dir"])
        run_model_paths = get_model_artifact_paths(model_type, base_dir=run_paths["artifact_base_dir"])
        artifacts["scaler_local_path"] = run_model_paths["scaler_path"]
        artifacts["metadata_local_path"] = metadata_path
        artifacts["model_local_path"] = run_model_paths["model_path"]

        monitor.log_training_event(
            job_id,
            "STARTED",
            {
                "job_id": job_id,
                "run_type": config.get("run_type", "training"),
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
        epoch_callback = _epoch_progress_callback(
            job_id=job_id,
            total_epochs=int(config.get("epochs", 30)),
            monitor=monitor,
            artifacts=artifacts,
        )
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
            artifact_base_dir=run_paths["artifact_base_dir"],
            initial_model_base_dir="model",
            tensorboard_log_dir=run_paths["logs_dir"],
            extra_callbacks=[epoch_callback],
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
        artifacts["model_upload_temp_path"] = temp_model_path

        result = {
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history.get("val_accuracy", [None])[-1]) if history.history.get("val_accuracy") else None,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history.get("val_loss", [None])[-1]) if history.history.get("val_loss") else None,
            "test_metrics": metrics or None,
        }
        visualization_paths = _save_training_visualizations(run_paths, history, metrics or None)

        _set_meta(0.92, "Uploading run artifacts...")
        _update_run(job_id, "uploading_artifacts", 0.92, "Uploading run artifacts...", artifacts=artifacts)
        _publish(job_id, "uploading_artifacts", {"progress": 0.92})
        _register_object_artifact(
            storage_service,
            artifacts,
            key="run_model",
            local_path=run_model_paths["model_path"],
            bucket_key="training",
            object_name=f"runs/{job_id}/artifacts/model.keras",
            label="Run model artifact",
            kind="model",
            content_type="application/octet-stream",
            metadata={"model_type": model_type},
        )
        _register_object_artifact(
            storage_service,
            artifacts,
            key="run_scaler",
            local_path=run_model_paths["scaler_path"],
            bucket_key="training",
            object_name=f"runs/{job_id}/artifacts/scaler.joblib",
            label="Run scaler artifact",
            kind="scaler",
            content_type="application/octet-stream",
            metadata={"model_type": model_type},
        )
        _register_object_artifact(
            storage_service,
            artifacts,
            key="run_metadata",
            local_path=run_model_paths["metadata_path"],
            bucket_key="training",
            object_name=f"runs/{job_id}/artifacts/metadata.json",
            label="Run metadata artifact",
            kind="metadata",
            content_type="application/json",
            metadata={"model_type": model_type},
        )
        for path in visualization_paths:
            filename = os.path.basename(path)
            content_type = "application/json" if filename.endswith(".json") else "image/png"
            _register_object_artifact(
                storage_service,
                artifacts,
                key=filename.replace(".", "_"),
                local_path=path,
                bucket_key="training",
                object_name=f"runs/{job_id}/metrics/{filename}",
                label=filename,
                kind="training_metric_artifact",
                content_type=content_type,
                metadata={"model_type": model_type},
            )

        _set_meta(0.97, "Promoting active model artifacts...")
        _update_run(job_id, "promoting_model", 0.97, "Promoting active model artifacts...", artifacts=artifacts)
        _publish(job_id, "promoting_model", {"progress": 0.97, "model_type": model_type})
        promoted_paths = promote_model_artifacts(model_type, run_paths["artifact_base_dir"], target_base_dir="model")
        artifacts["promotion"] = {
            "promoted_at": datetime.now().isoformat(),
            "model_type": model_type,
            "active_artifact_dir": promoted_paths["artifact_dir"],
        }
        _register_object_artifact(
            storage_service,
            artifacts,
            key="active_model",
            local_path=promoted_paths["model_path"],
            bucket_key="models",
            object_name=f"active/{model_type}/model.keras",
            label="Active model artifact",
            kind="active_model",
            content_type="application/octet-stream",
            metadata={"model_type": model_type},
        )
        _register_object_artifact(
            storage_service,
            artifacts,
            key="active_scaler",
            local_path=promoted_paths["scaler_path"],
            bucket_key="models",
            object_name=f"active/{model_type}/scaler.joblib",
            label="Active scaler artifact",
            kind="active_scaler",
            content_type="application/octet-stream",
            metadata={"model_type": model_type},
        )
        _register_object_artifact(
            storage_service,
            artifacts,
            key="active_metadata",
            local_path=promoted_paths["metadata_path"],
            bucket_key="models",
            object_name=f"active/{model_type}/metadata.json",
            label="Active metadata artifact",
            kind="active_metadata",
            content_type="application/json",
            metadata={"model_type": model_type},
        )
        public_artifacts = _public_artifacts(artifacts)

        _set_meta(1.0, "Training completed.", artifacts=artifacts)
        _update_run(
            job_id,
            "completed",
            1.0,
            "Training completed.",
            metrics=result,
            artifacts=public_artifacts,
            error=None,
            completed_at=datetime.now(),
        )
        completed_payload = _build_completed_event(
            job_id=job_id,
            model_type=model_type,
            config=config,
            result=result,
            artifacts=public_artifacts,
        )
        _publish(job_id, "completed", completed_payload)
        monitor.log_training_event(
            job_id,
            "COMPLETED",
            {
                "job_id": job_id,
                "run_type": config.get("run_type", "training"),
                "subject_id": config.get("subject_id"),
                "session_id": config.get("session_id"),
                "model_type": model_type,
                "final_accuracy": result["final_train_accuracy"],
                "final_loss": result["final_train_loss"],
            },
        )
        return {
            **result,
            "artifacts": public_artifacts,
        }
    except Exception as exc:
        _set_meta(1.0, f"Training failed: {exc}")
        public_artifacts = _public_artifacts(artifacts)
        _update_run(
            job_id,
            "failed",
            1.0,
            f"Training failed: {exc}",
            error=str(exc),
            artifacts=public_artifacts,
            completed_at=datetime.now(),
        )
        _publish(job_id, "failed", {"job_id": job_id, "error": str(exc), "artifacts": public_artifacts})
        monitor.log_training_event(
            job_id,
            "FAILED",
            {
                "job_id": job_id,
                "run_type": config.get("run_type", "training"),
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
        try:
            shutil.rmtree(run_paths["run_root"], ignore_errors=True)
        except Exception:
            logger.debug("Failed to remove run directory %s", run_paths["run_root"])
        monitor.close()


def train_from_file(
    file_path: str,
    config: Dict[str, Any],
    model_type: str,
    existing_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    job_id = _current_job_id()
    storage_service = MinioStorageService()
    artifacts: Dict[str, Any] = dict(existing_artifacts or {})
    artifacts.update({
        "uploaded_file_local_path": file_path,
        "uploaded_file_name": os.path.basename(file_path),
    })
    npz_path = os.path.join("temp", f"{job_id}.npz")

    try:
        from src.preprocessing.labeling import label_eeg_states
        from src.preprocessing.load_data import load_data
        from src.preprocessing.preprocess import preprocess_data

        if "uploaded_dataset" not in (artifacts.get("objects") or {}) and storage_service.enabled:
            try:
                object_name = storage_service.upload_file(file_path, "training", f"runs/{job_id}/input/{os.path.basename(file_path)}")
                if object_name:
                    artifacts.setdefault("objects", {})
                    artifacts["objects"]["uploaded_dataset"] = storage_service.build_artifact_descriptor(
                        "training",
                        object_name,
                        label="Uploaded training dataset",
                        kind="uploaded_dataset",
                        content_type="application/octet-stream",
                    )
            except Exception as exc:
                logger.warning(f"Training input upload failed: {exc}")
        elif "uploaded_dataset" not in (artifacts.get("objects") or {}):
            artifacts.setdefault("objects", {})
            artifacts["objects"]["uploaded_dataset"] = _artifact_descriptor(
                "Uploaded training dataset",
                "uploaded_dataset",
                local_reference=os.path.basename(file_path),
                content_type="application/octet-stream",
            )

        _set_meta(0.05, "Loading and preprocessing file...")
        _update_run(job_id, "preprocessing", 0.05, "Loading and preprocessing file...", artifacts=artifacts)
        _publish(job_id, "preprocessing", {"progress": 0.05})
        df = load_data(file_path)
        df = label_eeg_states(df)

        X_train, X_test, y_train, y_test, preprocessing_metadata = preprocess_data(
            df,
            overlap=config.get("overlap", 0.5),
            simple_mode=config.get("simple_mode", True),
        )
        np.savez_compressed(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        artifacts["training_bundle_local_path"] = npz_path
        update_training_run(job_id, {"artifacts": artifacts})
        return train_from_npz(
            npz_path,
            {**config, "run_type": config.get("run_type", "training_file"), "source": "dataset_file"},
            model_type,
            existing_artifacts=artifacts,
            existing_scaler=preprocessing_metadata.get("scaler"),
            feature_names=preprocessing_metadata.get("feature_names"),
            dataset_metadata=preprocessing_metadata,
        )
    finally:
        _cleanup_file(npz_path)
        _cleanup_file(file_path)


def train_from_bundle_object(bundle_descriptor: Dict[str, Any], config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    job_id = _current_job_id()
    storage_service = MinioStorageService()
    run_paths = _job_run_paths(job_id)
    input_dir = os.path.join(run_paths["run_root"], "input")
    local_npz_path = os.path.join(input_dir, "training_bundle.npz")
    artifacts: Dict[str, Any] = {
        "storage_backend": "minio" if storage_service.enabled else "local",
        "objects": {"training_bundle": dict(bundle_descriptor)},
    }
    _update_run(job_id, "downloading_input", 0.02, "Downloading training bundle...", artifacts=artifacts)
    _publish(job_id, "downloading_input", {"progress": 0.02})
    downloaded_path = _download_object_artifact(storage_service, bundle_descriptor, local_npz_path)
    return train_from_npz(
        downloaded_path,
        config,
        model_type,
        existing_artifacts=artifacts,
    )


def train_from_file_object(file_descriptor: Dict[str, Any], config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    job_id = _current_job_id()
    storage_service = MinioStorageService()
    run_paths = _job_run_paths(job_id)
    input_dir = os.path.join(run_paths["run_root"], "input")
    filename = os.path.basename(file_descriptor.get("object_name", "dataset.bin"))
    local_file_path = os.path.join(input_dir, filename)
    artifacts: Dict[str, Any] = {
        "storage_backend": "minio" if storage_service.enabled else "local",
        "uploaded_file_name": file_descriptor.get("metadata", {}).get("original_filename") or filename,
        "objects": {"uploaded_dataset": dict(file_descriptor)},
    }
    _update_run(job_id, "downloading_input", 0.02, "Downloading uploaded dataset...", artifacts=artifacts)
    _publish(job_id, "downloading_input", {"progress": 0.02})
    downloaded_path = _download_object_artifact(storage_service, file_descriptor, local_file_path)
    return train_from_file(downloaded_path, config, model_type, existing_artifacts=artifacts)


def compare_models_from_npz(
    npz_path: str,
    n_repeats: int = 3,
    existing_artifacts: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    job_id = _current_job_id()
    storage_service = MinioStorageService()
    run_paths = _job_run_paths(job_id)
    os.makedirs(run_paths["metrics_dir"], exist_ok=True)
    comparison_config = dict(config or {})
    artifacts: Dict[str, Any] = dict(existing_artifacts or {})
    artifacts.update({
        "comparison_bundle_local_path": npz_path,
        "storage_backend": "minio" if storage_service.enabled else "local",
        "run_root_local_dir": run_paths["run_root"],
    })
    monitor = TrainingMonitor()
    try:
        _set_meta(0.05, "Loading data for comparison...")
        _update_run(job_id, "loading_data", 0.05, "Loading data for comparison...", artifacts=artifacts)
        _publish(job_id, "loading_data", {"progress": 0.05})
        data = np.load(npz_path, allow_pickle=False)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        if os.path.exists(npz_path) and "comparison_bundle" not in (artifacts.get("objects") or {}):
            _register_object_artifact(
                storage_service,
                artifacts,
                key="comparison_bundle",
                local_path=npz_path,
                bucket_key="training",
                object_name=f"runs/{job_id}/input/comparison_bundle.npz",
                label="Comparison bundle",
                kind="dataset_bundle",
                content_type="application/octet-stream",
            )
        monitor.log_training_event(
            job_id,
            "STARTED",
            {
                "job_id": job_id,
                "run_type": "comparison",
                "n_repeats": n_repeats,
                "subject_id": comparison_config.get("subject_id"),
                "session_id": comparison_config.get("session_id"),
            },
        )

        _set_meta(0.1, "Comparing model architectures...")
        _update_run(job_id, "comparing", 0.1, "Comparing model architectures...", artifacts=artifacts)
        _publish(job_id, "comparing", {"progress": 0.1, "n_repeats": n_repeats})
        results = model_comparison(X_train, y_train, X_test, y_test, n_repeats=n_repeats)
        results_path = os.path.join(run_paths["metrics_dir"], "comparison_results.json")
        _write_json(results_path, results)
        _register_object_artifact(
            storage_service,
            artifacts,
            key="comparison_results",
            local_path=results_path,
            bucket_key="training",
            object_name=f"runs/{job_id}/metrics/comparison_results.json",
            label="Comparison results",
            kind="comparison_metrics",
            content_type="application/json",
        )
        public_artifacts = _public_artifacts(artifacts)
        _set_meta(1.0, "Model comparison completed.")
        _update_run(
            job_id,
            "completed",
            1.0,
            "Model comparison completed.",
            metrics=results,
            artifacts=public_artifacts,
            completed_at=datetime.now(),
        )
        _publish(
            job_id,
            "completed",
            {
                "job_id": job_id,
                "run_type": "comparison",
                "subject_id": comparison_config.get("subject_id"),
                "session_id": comparison_config.get("session_id"),
                "metrics": results,
                "artifacts": public_artifacts,
            },
        )
        monitor.log_training_event(
            job_id,
            "COMPLETED",
            {
                "job_id": job_id,
                "run_type": "comparison",
                "subject_id": comparison_config.get("subject_id"),
                "session_id": comparison_config.get("session_id"),
            },
        )
        return results
    except Exception as exc:
        _set_meta(1.0, f"Model comparison failed: {exc}")
        public_artifacts = _public_artifacts(artifacts)
        _update_run(
            job_id,
            "failed",
            1.0,
            f"Model comparison failed: {exc}",
            error=str(exc),
            artifacts=public_artifacts,
            completed_at=datetime.now(),
        )
        _publish(
            job_id,
            "failed",
            {
                "job_id": job_id,
                "run_type": "comparison",
                "subject_id": comparison_config.get("subject_id"),
                "session_id": comparison_config.get("session_id"),
                "error": str(exc),
                "artifacts": public_artifacts,
            },
        )
        monitor.log_training_event(
            job_id,
            "FAILED",
            {
                "job_id": job_id,
                "run_type": "comparison",
                "subject_id": comparison_config.get("subject_id"),
                "session_id": comparison_config.get("session_id"),
                "error": str(exc),
            },
        )
        raise
    finally:
        _cleanup_file(npz_path)
        try:
            shutil.rmtree(run_paths["run_root"], ignore_errors=True)
        except Exception:
            logger.debug("Failed to remove comparison run directory %s", run_paths["run_root"])
        monitor.close()


def compare_models_from_object(bundle_descriptor: Dict[str, Any], n_repeats: int = 3, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    job_id = _current_job_id()
    storage_service = MinioStorageService()
    run_paths = _job_run_paths(job_id)
    input_dir = os.path.join(run_paths["run_root"], "input")
    local_npz_path = os.path.join(input_dir, "comparison_bundle.npz")
    artifacts: Dict[str, Any] = {
        "storage_backend": "minio" if storage_service.enabled else "local",
        "objects": {"comparison_bundle": dict(bundle_descriptor)},
    }
    _update_run(job_id, "downloading_input", 0.02, "Downloading comparison bundle...", artifacts=artifacts)
    _publish(job_id, "downloading_input", {"progress": 0.02})
    downloaded_path = _download_object_artifact(storage_service, bundle_descriptor, local_npz_path)
    return compare_models_from_npz(
        downloaded_path,
        n_repeats=n_repeats,
        existing_artifacts=artifacts,
        config=config,
    )
