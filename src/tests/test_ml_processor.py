"""
Tests for the async MLProcessor using lightweight model and recommendation mocks.
"""
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import src.services.analysis as analysis_module


class IdentityScaler:
    def transform(self, values):
        return values


class FakeModel:
    input_shape = (None, 5, 1)

    def predict(self, values, verbose=0):
        _ = verbose
        rows = values.shape[0]
        base = np.array([[0.1, 0.8, 0.1]])
        return np.repeat(base, rows, axis=0)


@pytest.fixture
def processor(monkeypatch):
    async def generate_recommendations(*args, **kwargs):
        return ["Take a short break", "Hydrate"]

    async def generate_detailed_report(*args, **kwargs):
        return {
            "dominant_state": "engaged",
            "wellness_rating": "low",
            "clinical_observation": "stable engagement",
            "technical_analysis": "ok",
            "recommendations": ["Take a short break", "Hydrate"],
            "metrics": {"confidence": 80.0},
        }

    mock_manager = SimpleNamespace(
        models={},
        model_dir="model",
        tensorflow_available=True,
        list_model_files=lambda: ["enhanced_cnn_lstm/model.keras"],
        get_model=lambda model_type, warmup=False: FakeModel(),
        get_scaler=lambda model_type: IdentityScaler(),
        get_metadata=lambda model_type: {"input_features": ["alpha", "beta", "theta", "delta", "gamma"]},
    )

    monkeypatch.setattr(analysis_module, "get_model_manager", lambda: mock_manager)
    monkeypatch.setattr(
        analysis_module,
        "NLPRecommendationEngine",
        lambda: SimpleNamespace(
            generate_recommendations=generate_recommendations,
            generate_detailed_report=generate_detailed_report,
            save_report=lambda report: "reports/test_report.json",
        ),
    )
    monkeypatch.setattr(analysis_module, "safe_enqueue", lambda *args, **kwargs: None)
    return analysis_module.MLProcessor(default_model="enhanced_cnn_lstm")


class TestMLProcessor:
    @pytest.fixture
    def sample_dict_data(self):
        return {
            "alpha": 0.5,
            "beta": 0.3,
            "theta": 0.2,
            "delta": 0.1,
            "gamma": 0.4,
        }

    @pytest.fixture
    def sample_array_data(self):
        return np.array(
            [
                [0.5, 0.3, 0.2, 0.1, 0.4],
                [0.6, 0.4, 0.3, 0.2, 0.5],
                [0.4, 0.2, 0.1, 0.05, 0.3],
                [0.7, 0.5, 0.4, 0.3, 0.6],
                [0.3, 0.1, 0.05, 0.02, 0.2],
            ]
        )

    def test_initialization(self, processor):
        status = processor.get_status()
        assert status["model_path"] == "model"
        assert status["recommendation_engine_loaded"] is True
        assert status["tensorflow_available"] is True

    @pytest.mark.asyncio
    async def test_process_single_sample_dict(self, processor, sample_dict_data):
        result = await processor.process_eeg_data(
            sample_dict_data,
            subject_id="test_sub_001",
            session_id="test_sess_001",
            model_type="enhanced_cnn_lstm",
        )

        assert result["state_label"] == "engaged"
        assert result["dominant_state"] == 1
        assert result["metadata"]["subject_id"] == "test_sub_001"
        assert result["recommendations"] == ["Take a short break", "Hydrate"]

    @pytest.mark.asyncio
    async def test_process_batch_array(self, processor, sample_array_data):
        result = await processor.process_eeg_data(
            sample_array_data,
            subject_id="test_sub_002",
            session_id="test_sess_002",
            model_type="enhanced_cnn_lstm",
        )

        assert result["temporal_analysis"]["total_samples"] == 5
        assert len(result["predicted_state"]) == 5
        assert len(result["smoothed_states"]) == 5

    @pytest.mark.asyncio
    async def test_process_dataframe(self, processor, sample_array_data):
        columns = ["alpha", "beta", "theta", "delta", "gamma"]
        df = pd.DataFrame(sample_array_data, columns=columns)

        result = await processor.process_eeg_data(
            df,
            subject_id="test_sub_003",
            model_type="enhanced_cnn_lstm",
        )

        assert result["temporal_analysis"]["total_samples"] == 5
        assert result["metadata"]["subject_id"] == "test_sub_003"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_input(self, processor):
        with pytest.raises(Exception):
            await processor.process_eeg_data(
                {"invalid_key": 123},
                model_type="enhanced_cnn_lstm",
            )

    @pytest.mark.asyncio
    async def test_cognitive_metrics_calculation(self, processor, sample_dict_data):
        result = await processor.process_eeg_data(
            sample_dict_data,
            model_type="enhanced_cnn_lstm",
        )
        metrics = result["cognitive_metrics"]

        expected_keys = [
            "attention_index",
            "relaxation_index",
            "stress_index",
            "cognitive_load",
            "mental_fatigue",
            "alertness",
        ]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

    @pytest.mark.asyncio
    async def test_detailed_report_generation(self, processor, sample_dict_data):
        report = await processor.generate_detailed_report(
            sample_dict_data,
            subject_id="report_test_sub",
            save_report=False,
            model_type="enhanced_cnn_lstm",
        )

        assert "analysis_results" in report
        assert "recommendations" in report
        assert report["dominant_state"] == "engaged"
        assert report["metrics"]["confidence"] == 80.0
