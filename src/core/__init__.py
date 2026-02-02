"""
Core package for the NeuroLab AI Model Server.
Contains models, data handling, and ML processing components.
"""

from .models.eeg import EEGDataPoint, EEGSession, EEGFeatures
from .data.handler import DataHandler
from src.preprocessing import (
    load_data,
    label_eeg_states,
    extract_features,
    preprocess_data
)
from src.utils.temporal_processing import (
    temporal_smoothing,
    calculate_state_durations
)
from src.utils.nlp_recommendations import get_recommendations as generate_recommendations
from .ml.model import (
    create_model,
    load_calibrated_model,
    calibrate_model,
    save_model,
    evaluate_model
)

__all__ = [
    'EEGDataPoint',
    'EEGSession',
    'EEGFeatures',
    'DataHandler',
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