import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.datastructures import UploadFile

from src.api import training as training_api


class FakeStorage:
    def __init__(self):
        self.enabled = True
        self.uploads = []

    def upload_file(self, file_path, bucket_key, object_name=None):
        self.uploads.append((file_path, bucket_key, object_name))
        return object_name or "uploaded-object"

    def build_artifact_descriptor(self, bucket_key, object_name, **kwargs):
        descriptor = {"bucket_key": bucket_key, "object_name": object_name}
        descriptor.update(kwargs)
        return descriptor


class FakeQueue:
    def __init__(self):
        self.calls = []

    def enqueue(self, func_name, *args, **kwargs):
        self.calls.append((func_name, args, kwargs))
        return SimpleNamespace(id=kwargs["job_id"])


@pytest.fixture
def training_setup(monkeypatch):
    fake_queue = FakeQueue()
    fake_storage = FakeStorage()

    monkeypatch.setattr(training_api, "_storage_service", fake_storage)
    monkeypatch.setattr(training_api, "require_rq", lambda: None)
    monkeypatch.setattr(training_api, "get_queue", lambda name: fake_queue)
    monkeypatch.setattr(training_api, "track_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_api, "publish_job_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(training_api, "validate_file", lambda file: None)
    monkeypatch.setattr(training_api.db_service, "create_training_run", AsyncMock(return_value="db-id"))
    monkeypatch.setattr(training_api.db_service, "archive_training_run", AsyncMock(return_value=True))

    return fake_queue, fake_storage


@pytest.mark.asyncio
async def test_train_submit_uses_minio_object_reference(training_setup):
    fake_queue, fake_storage = training_setup

    response = await training_api.train_model(
        data=training_api.TrainingData(
            X_train=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
            y_train=[0, 1],
        ),
        model_type="original",
    )

    assert response.status == "queued"
    assert fake_storage.uploads
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.training.train_from_bundle_object"
    assert args[0]["bucket_key"] == "training"
    assert args[0]["object_name"].endswith("/input/training_bundle.npz")
    assert kwargs["job_id"].startswith("train_")


@pytest.mark.asyncio
async def test_train_file_submit_uses_minio_object_reference(training_setup, tmp_path, monkeypatch):
    fake_queue, _ = training_setup

    uploaded_path = tmp_path / "dataset.csv"
    uploaded_path.write_text("alpha,beta,theta,delta,gamma,state\n0.1,0.2,0.3,0.4,0.5,0\n", encoding="utf-8")
    monkeypatch.setattr(training_api, "save_uploaded_file", AsyncMock(return_value=str(uploaded_path)))

    upload = UploadFile(filename="dataset.csv", file=io.BytesIO(uploaded_path.read_bytes()))
    upload.headers = {"content-type": "text/csv"}

    response = await training_api.train_model_from_file(
        file=upload,
        model_type="original",
        overlap=0.5,
        simple_mode=True,
        config=None,
    )

    assert response.status == "queued"
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.training.train_from_file_object"
    assert args[0]["bucket_key"] == "training"
    assert args[0]["kind"] == "uploaded_dataset"
    assert args[0]["metadata"]["original_filename"] == "dataset.csv"
    assert kwargs["job_id"].startswith("train_file_")


@pytest.mark.asyncio
async def test_compare_submit_uses_minio_object_reference_and_context(training_setup):
    fake_queue, _ = training_setup

    response = await training_api.compare_models(
        data=training_api.TrainingData(
            X_train=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
            y_train=[0, 1],
            X_test=[[0.2, 0.2, 0.3, 0.4, 0.4]],
            y_test=[1],
            config=training_api.TrainingConfig(
                model_type="original",
                subject_id="subject_1",
                session_id="session_1",
            ),
        ),
        n_repeats=2,
    )

    assert response.status == "queued"
    func_name, args, kwargs = fake_queue.calls[0]
    assert func_name == "src.jobs.training.compare_models_from_object"
    assert args[0]["object_name"].endswith("/input/comparison_bundle.npz")
    assert kwargs["n_repeats"] == 2
    assert kwargs["config"]["subject_id"] == "subject_1"
    assert kwargs["config"]["session_id"] == "session_1"
