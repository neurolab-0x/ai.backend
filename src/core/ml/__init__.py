"""
ML package for the NeuroLab AI Model Server.
Contains machine learning models and processing components.
"""

from src.preprocessing import (
    load_data,
    label_eeg_states,
    extract_features,
    preprocess_data
)
from src.core.processing.temporal import (
    temporal_smoothing,
    calculate_state_durations
)
from src.services.recommendation import get_recommendations as generate_recommendations
from .model import (
    create_model,
    load_calibrated_model,
    calibrate_model,
    save_model,
    evaluate_model
)

__all__ = [
    'load_data',
    'label_eeg_states',
    'extract_features',
    'preprocess_data',
    'temporal_smoothing',
    'calculate_state_durations',
    'generate_recommendations',
    'create_model',
    'load_calibrated_model',
    'calibrate_model',
    'save_model',
    'evaluate_model'
] 