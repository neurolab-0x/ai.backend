"""
Tests for recommendation/report generation with the shared async LLM client mocked out.
"""
from types import SimpleNamespace

import pytest

import src.services.analysis as analysis_module
import src.services.recommendation as recommendation_module
from src.services.analysis import MLProcessor
from src.services.recommendation import NLPRecommendationEngine


@pytest.fixture
def recommendation_engine(monkeypatch):
    async def fake_history(subject_id, limit=3):
        return [
            {
                "time": "2026-04-01T10:00:00",
                "type": "session",
                "session_id": "sess_1",
                "dominant_state": "engaged",
            }
        ]

    mock_client = SimpleNamespace(
        enabled=True,
        create_chat_completion=lambda **kwargs: None,
    )

    async def fake_completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if 'Format as JSON' in prompt:
            return (
                '{"clinical_observation":"stable engagement",'
                '"technical_analysis":"ok",'
                '"interpretation":"ok",'
                '"safety_assessment":{"alert_level":"low","immediate_actions":[]}}'
            )
        return "- Take a short break\n- Hydrate\n- Maintain posture"

    mock_client.create_chat_completion = fake_completion

    monkeypatch.setattr(recommendation_module, "get_async_llm_client", lambda: mock_client)
    monkeypatch.setattr(recommendation_module.db_service, "get_user_history", fake_history)
    return NLPRecommendationEngine()


@pytest.mark.asyncio
async def test_basic_recommendations(recommendation_engine):
    recommendations = await recommendation_engine.generate_recommendations(
        state_durations={0: 10, 1: 20, 2: 70},
        total_duration=100,
        confidence=85.0,
        cognitive_metrics={"cognitive_load": 2.5, "mental_fatigue": 0.6},
        state_transitions=12,
    )

    assert len(recommendations) == 3
    assert recommendations[0].startswith("-")


@pytest.mark.asyncio
async def test_detailed_report(recommendation_engine):
    async def fake_generate_recommendations(*args, **kwargs):
        return ["Take a short break", "Hydrate", "Maintain posture"]

    async def fake_generate_medical_explanation(context, occupation="default"):
        _ = occupation
        return {
            "clinical_observation": f"state is {context.state_label}",
            "technical_analysis": "ok",
            "interpretation": "ok",
            "safety_assessment": {"alert_level": "low", "immediate_actions": []},
        }

    recommendation_engine.generate_recommendations = fake_generate_recommendations
    recommendation_engine.generate_medical_explanation = fake_generate_medical_explanation

    report = await recommendation_engine.generate_detailed_report(
        state_durations={0: 25, 1: 35, 2: 40},
        total_duration=100,
        confidence=82.5,
        cognitive_metrics={
            "cognitive_load": 2.1,
            "mental_fatigue": 0.55,
            "attention_index": 1.8,
            "stress_index": 1.9,
        },
        state_transitions=15,
        subject_id="subject_1",
        session_id="session_1",
    )

    assert report["dominant_state"] == "elevated_stress"
    assert report["wellness_rating"] == "low"
    assert len(report["recommendations"]) == 3
    assert report["metrics"]["confidence"] == 82.5


@pytest.mark.asyncio
async def test_ml_processor_integration(monkeypatch):
    class IdentityScaler:
        def transform(self, values):
            return values

    class FakeModel:
        input_shape = (None, 5, 1)

        def predict(self, values, verbose=0):
            import numpy as np

            return np.repeat(np.array([[0.1, 0.8, 0.1]]), values.shape[0], axis=0)

    async def fake_generate_recommendations(*args, **kwargs):
        return ["Take a short break"]

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
        lambda: SimpleNamespace(generate_recommendations=fake_generate_recommendations),
    )
    monkeypatch.setattr(analysis_module, "safe_enqueue", lambda *args, **kwargs: None)

    processor = MLProcessor(default_model="enhanced_cnn_lstm")
    result = await processor.process_eeg_data(
        {
            "alpha": 8.5,
            "beta": 15.2,
            "theta": 6.3,
            "delta": 3.1,
            "gamma": 2.8,
        },
        model_type="enhanced_cnn_lstm",
    )

    assert result["state_label"] == "engaged"
    assert result["confidence"] > 0
    assert result["recommendations"] == ["Take a short break"]
