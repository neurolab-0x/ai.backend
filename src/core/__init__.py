"""
Core package for the NeuroLab AI Model Server.
Contains models, data handling, and ML processing components.
"""

from importlib import import_module

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


def __getattr__(name: str):
    """
    Lazy attribute loading to avoid importing TensorFlow-heavy modules at package import time.
    """
    mapping = {
        "EEGDataPoint": ("src.schemas.eeg", "EEGDataPoint"),
        "EEGSession": ("src.schemas.eeg", "EEGSession"),
        "EEGFeatures": ("src.schemas.eeg", "EEGFeatures"),
        "DataHandler": ("src.core.data.handler", "DataHandler"),
        "temporal_smoothing": ("src.core.processing.temporal", "temporal_smoothing"),
        "calculate_state_durations": ("src.core.processing.temporal", "calculate_state_durations"),
        "create_model": ("src.core.ml.model", "create_model"),
        "load_calibrated_model": ("src.core.ml.model", "load_calibrated_model"),
        "calibrate_model": ("src.core.ml.model", "calibrate_model"),
        "save_model": ("src.core.ml.model", "save_model"),
        "evaluate_model": ("src.core.ml.model", "evaluate_model"),
    }
    if name not in mapping:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = mapping[name]
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
