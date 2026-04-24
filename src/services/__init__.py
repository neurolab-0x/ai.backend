"""
Services package for the NeuroLab AI Model Server.
Contains business logic and high-level application services.
"""

from importlib import import_module

__all__ = [
    'MLProcessor',
    'NLPRecommendationEngine',
    'DataHandler',
    'EEGDataPoint',
    'VoiceProcessor',
    'DatabaseService'
]


def __getattr__(name: str):
    """
    Lazy attribute loading to avoid importing heavy dependencies (e.g. TensorFlow)
    at package import time.
    """
    mapping = {
        "MLProcessor": ("src.services.analysis", "MLProcessor"),
        "NLPRecommendationEngine": ("src.services.recommendation", "NLPRecommendationEngine"),
        "DataHandler": ("src.services.data_service", "DataHandler"),
        "EEGDataPoint": ("src.services.data_service", "EEGDataPoint"),
        "VoiceProcessor": ("src.services.voice", "VoiceProcessor"),
        "DatabaseService": ("src.services.database", "DatabaseService"),
    }
    if name not in mapping:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = mapping[name]
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
