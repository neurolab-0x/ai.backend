"""
ML package for the NeuroLab AI Model Server.
Contains machine learning models and processing components.
"""

from importlib import import_module

__all__ = [
    'create_model',
    'load_calibrated_model',
    'calibrate_model',
    'save_model',
    'evaluate_model'
]


def __getattr__(name: str):
    mapping = {
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
