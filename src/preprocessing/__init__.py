"""
Preprocessing module for EEG data.
Consolidated data processing functionality.
"""

from .load_data import load_data, load_csv_data, load_edf_data, load_biosignal_data, load_matlab_data
from .features import extract_features
from .labeling import label_eeg_states


def _missing_preprocess_dependency(*args, **kwargs):
    raise RuntimeError(
        "Training preprocessing utilities require optional ML dependencies "
        "such as imbalanced-learn."
    )


try:
    from .preprocess import preprocess_data, split_data, augment_data, remove_outliers
except ImportError:  # pragma: no cover - optional training dependency
    preprocess_data = _missing_preprocess_dependency
    split_data = _missing_preprocess_dependency
    augment_data = _missing_preprocess_dependency
    remove_outliers = _missing_preprocess_dependency

__all__ = [
    # Data loading
    'load_data',
    'load_csv_data',
    'load_edf_data',
    'load_biosignal_data',
    'load_matlab_data',
    
    # Feature extraction
    'extract_features',
    
    # Labeling
    'label_eeg_states',
    
    # Preprocessing
    'preprocess_data',
    'split_data',
    'augment_data',
    'remove_outliers',
]
