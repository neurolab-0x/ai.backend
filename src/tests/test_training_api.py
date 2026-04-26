import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import training as training_api

fake_matplotlib = types.ModuleType("matplotlib")
fake_pyplot = types.ModuleType("matplotlib.pyplot")
fake_pyplot.figure = lambda *args, **kwargs: None
fake_pyplot.subplots = lambda *args, **kwargs: (SimpleNamespace(), [SimpleNamespace(), SimpleNamespace()])
fake_pyplot.close = lambda *args, **kwargs: None
sys.modules.setdefault("matplotlib", fake_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", fake_pyplot)

from src.jobs import training as training_jobs


class FakeStorage:
    def __init__(self):
        self.enabled = True
        self.uploads = []
        self.downloads = []

    def upload_file(self, file_path, bucket_key, object_name=None):
        self.uploads.append((file_path, bucket_key, object_name))
        return object_name or Path(file_path).name

    def build_artifact_descriptor(self, bucket_key, object_name, **kwargs):
        descriptor = {"bucket_key": bucket_key, "object_name": object_name}
        descriptor.update(kwargs)
        return descriptor

    def download_artifact(self, descriptor, destination_path):
        self.downloads.append((descriptor, destination_path))
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        Path(destination_path).write_bytes(b"bundle")
        return destination_path


class FakeQueue:
    def __init__(self):
        self.calls = []

    def enqueue(self, func_name, *args, **kwargs):
        self.calls.append((func_name, args, kwargs))
        return SimpleNamespace(id=kwargs["job_id"])


@pytest.fixture
def training_client(monkeypatch):
    app = FastAPI()
    app.include_router(training_api.router, prefix="/api/v1/training")
    client = TestClient(app)

    fake_queue = FakeQueue()
    fake_storage = FakeStorage()

    monkeypatch.setattr(training_api, "_storage_service", fake_storage)
    monkeypatch.setattr(training_api, "get_queue", lambda name: fake_queue)
    monkeypatch.setattr(training_api, "track_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_api, "publish_job_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_api, "validate_file", lambda file: None)
    monkeypatch.setattr(training_api.db_service, "create_training_run", AsyncMock(return_value="db-id"))
    monkeypatch.setattr(training_api.db_service, "archive_training_run", AsyncMock(return_value=True))

    return client, fake_queue, fake_storage


def test_train_submit_uses_minio_object_reference(training_client):
    client, fake_queue, fake_storage = training_client

    response = client.post(
        "/api/v1/training/train?model_type=original",
        json={
            "X_train": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
            "y_train": [0, 1],
        },
    )

    assert response.status_code == 202
    assert fake_storage.uploads
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.training.train_from_bundle_object"
    assert isinstance(args[0], dict)
    assert args[0]["bucket_key"] == "training"
    assert args[0]["object_name"].endswith("/input/training_bundle.npz")
    assert kwargs["job_id"].startswith("train_")


def test_train_file_submit_uses_minio_object_reference(training_client, tmp_path, monkeypatch):
    client, fake_queue, _ = training_client

    uploaded_path = tmp_path / "dataset.csv"
    uploaded_path.write_text("alpha,beta,theta,delta,gamma,state\n0.1,0.2,0.3,0.4,0.5,0\n", encoding="utf-8")
    monkeypatch.setattr(training_api, "save_uploaded_file", AsyncMock(return_value=str(uploaded_path)))

    response = client.post(
        "/api/v1/training/file?model_type=original",
        files={"file": ("dataset.csv", io.BytesIO(uploaded_path.read_bytes()), "text/csv")},
    )

    assert response.status_code == 202
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.training.train_from_file_object"
    assert args[0]["bucket_key"] == "training"
    assert args[0]["kind"] == "uploaded_dataset"
    assert args[0]["metadata"]["original_filename"] == "dataset.csv"
    assert kwargs["job_id"].startswith("train_file_")


def test_compare_submit_uses_minio_object_reference_and_context(training_client):
    client, fake_queue, _ = training_client

    response = client.post(
        "/api/v1/training/compare?n_repeats=2",
        json={
            "X_train": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
            "y_train": [0, 1],
            "X_test": [[0.2, 0.2, 0.3, 0.4, 0.4]],
            "y_test": [1],
            "config": {
                "model_type": "original",
                "subject_id": "subject_1",
                "session_id": "session_1",
            },
        },
    )

    assert response.status_code == 202
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.training.compare_models_from_object"
    assert args[0]["object_name"].endswith("/input/comparison_bundle.npz")
    assert kwargs["n_repeats"] == 2
    assert kwargs["config"]["subject_id"] == "subject_1"
    assert kwargs["config"]["session_id"] == "session_1"


def test_train_from_bundle_object_downloads_then_delegates(monkeypatch):
    fake_storage = FakeStorage()
    bundle_descriptor = {"bucket_key": "training", "object_name": "runs/job/input/training_bundle.npz"}
    delegated = MagicMock(return_value={"status": "ok"})

    monkeypatch.setattr(training_jobs, "MinioStorageService", lambda: fake_storage)
    monkeypatch.setattr(training_jobs, "_current_job_id", lambda: "train_job_1")
    monkeypatch.setattr(training_jobs, "_update_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_jobs, "_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_jobs, "train_from_npz", delegated)

    result = training_jobs.train_from_bundle_object(bundle_descriptor, {"epochs": 1}, "original")

    assert result == {"status": "ok"}
    assert fake_storage.downloads
    delegated.assert_called_once()
    assert delegated.call_args.args[0].endswith("temp/training_runs/train_job_1/input/training_bundle.npz")
    assert delegated.call_args.kwargs["existing_artifacts"]["objects"]["training_bundle"] == bundle_descriptor


def test_compare_from_object_downloads_then_delegates(monkeypatch):
    fake_storage = FakeStorage()
    bundle_descriptor = {"bucket_key": "training", "object_name": "runs/job/input/comparison_bundle.npz"}
    delegated = MagicMock(return_value={"original": {"mean_accuracy": 0.8}})

    monkeypatch.setattr(training_jobs, "MinioStorageService", lambda: fake_storage)
    monkeypatch.setattr(training_jobs, "_current_job_id", lambda: "compare_job_1")
    monkeypatch.setattr(training_jobs, "_update_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_jobs, "_publish", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_jobs, "compare_models_from_npz", delegated)

    result = training_jobs.compare_models_from_object(bundle_descriptor, n_repeats=4, config={"subject_id": "s1"})

    assert result["original"]["mean_accuracy"] == 0.8
    delegated.assert_called_once()
    assert delegated.call_args.args[0].endswith("temp/training_runs/compare_job_1/input/comparison_bundle.npz")
    assert delegated.call_args.kwargs["n_repeats"] == 4
    assert delegated.call_args.kwargs["config"]["subject_id"] == "s1"
