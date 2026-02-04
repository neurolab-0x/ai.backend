"""
Core package for the NeuroLab AI Model Server.
Contains models, data handling, and ML processing components.
"""

from src.schemas.eeg import EEGDataPoint, EEGSession, EEGFeatures
from .data.handler import DataHandler
from src.core.processing.temporal import (
    temporal_smoothing,
    calculate_state_durations
)
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
    'temporal_smoothing',
    'calculate_state_durations',
    'create_model',
    'load_calibrated_model',
    'calibrate_model',
    'save_model',
    'evaluate_model'
]
 