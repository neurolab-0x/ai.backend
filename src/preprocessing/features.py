from scipy.signal import welch
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
CANONICAL_BAND_COLUMNS = {'delta', 'theta', 'alpha', 'beta', 'gamma'}

class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors"""
    pass

def compute_psd(signal: np.ndarray, fs: float = 250) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced PSD computation with validation"""
    try:
        if len(signal) < 2:
            raise FeatureExtractionError("Signal too short for PSD computation")
        
        # Adjust nperseg based on signal length
        # Welch's method requires nperseg to be less than signal length
        # and ideally a power of 2
        if len(signal) < 256:
            # For short signals, use a smaller window
            nperseg = max(4, min(len(signal) // 2, 128))
        else:
            nperseg = 256
            
        # Ensure noverlap is less than nperseg
        noverlap = nperseg // 2                                                                                                                                                                                                                                                                                                                           
        
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        return freqs, psd
    except Exception as e:
        raise FeatureExtractionError(f"Error in PSD computation: {str(e)}")

def compute_band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    """Enhanced band power computation with validation"""
    try:
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        if not any(idx):
            return 0
        return simpson(psd[idx], freqs[idx])
    except Exception as e:
        raise FeatureExtractionError(f"Error in band power computation: {str(e)}")

def extract_features(df: pd.DataFrame, simple_mode: bool = True, overlap: float = 0.0) -> pd.DataFrame:
    """
    Inference-oriented feature extraction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing EEG data.
        Can be either:
        - Raw time-series data (rows=timepoints, columns=channels)
        - Pre-computed features (rows=samples, columns=features)
    simple_mode : bool
        Retained for API compatibility. The backend now always uses the
        simple 5-band feature path.
    overlap : float
        Overlap fraction between epochs (0.0 to 0.9)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted band features
    """
    try:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude timestamp and state columns
        eeg_channels = [col for col in numerical_cols if col.lower() not in ['timestamp', 'time', 'eeg_state', 'state', 'label']]
        
        lower_cols = {col.lower() for col in df.columns}
        has_canonical_feature_columns = CANONICAL_BAND_COLUMNS.issubset(lower_cols)
        all_scalar_features = True
        if eeg_channels and not df.empty:
            sample_row = df[eeg_channels].head(5)
            for _, row in sample_row.iterrows():
                for value in row:
                    if isinstance(value, (np.ndarray, list, tuple)):
                        all_scalar_features = False
                        break
                if not all_scalar_features:
                    break

        # Check if this is raw time-series data or pre-computed features
        # If the dataset already looks like canonical spectral features, keep it as tabular features.
        is_raw_timeseries = (
            len(df) > 50
            and len(eeg_channels) < 100
            and not (has_canonical_feature_columns and all_scalar_features)
        )
        
        if not simple_mode:
            logger.info("Legacy comprehensive backend feature extraction has been removed; using simple band features")

        if is_raw_timeseries:
            logger.info(f"Processing raw time-series data: {len(df)} timepoints, {len(eeg_channels)} channels with overlap {overlap}")
            # Process as time-series: extract features from each channel's full signal
            return extract_features_from_timeseries(df, eeg_channels, simple_mode=True, overlap=overlap)
        else:
            logger.info(f"Processing pre-computed features: {len(df)} samples")
            # Already features, just return (maybe with some processing)
            return df
            
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise FeatureExtractionError(f"Error in feature extraction: {str(e)}")


def segment_into_epochs(df: pd.DataFrame, epoch_length_samples: int = 257, overlap: float = 0.0) -> List[pd.DataFrame]:
    """
    Segment continuous EEG data into fixed-length epochs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Continuous EEG data (rows=timepoints, columns=channels)
    epoch_length_samples : int
        Number of samples per epoch (default 257 = 1.028s at 250Hz)
    overlap : float
        Overlap between epochs as fraction (0.0 = no overlap, 0.5 = 50% overlap)
        
    Returns:
    --------
    List[pd.DataFrame]
        List of epoch dataframes
    """
    epochs = []
    step_size = int(epoch_length_samples * (1 - overlap))
    
    for start_idx in range(0, len(df) - epoch_length_samples + 1, step_size):
        end_idx = start_idx + epoch_length_samples
        epoch = df.iloc[start_idx:end_idx].copy()
        epochs.append(epoch)
    
    logger.info(f"Segmented {len(df)} samples into {len(epochs)} epochs of {epoch_length_samples} samples each")
    return epochs

def extract_features_from_timeseries(df: pd.DataFrame, eeg_channels: List[str], simple_mode: bool = True, overlap: float = 0.0) -> pd.DataFrame:
    """
    Extract features from raw time-series EEG data.
    Automatically segments long recordings into epochs and processes each separately.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rows=timepoints, columns=channels
    eeg_channels : List[str]
        List of EEG channel names
    simple_mode : bool
        Retained for API compatibility. The backend now always uses the
        simple 5-band feature path.
    overlap : float
        Overlap fraction between epochs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per epoch containing extracted features
    """
    try:
        # Segment into epochs if data is long enough
        epoch_length = 257  # 1.028s at 250Hz
        
        if len(df) >= epoch_length:
            epochs = segment_into_epochs(df, epoch_length_samples=epoch_length, overlap=overlap)
            logger.info(f"Processing {len(epochs)} epochs...")
            
            # Process each epoch
            all_epoch_features = []
            for epoch_idx, epoch_df in enumerate(epochs):
                epoch_features = extract_features_from_single_epoch(epoch_df, eeg_channels, simple_mode=True)
                all_epoch_features.append(epoch_features)
            
            # Combine all epochs into a single dataframe
            features_df = pd.DataFrame(all_epoch_features)
            logger.info(f"Extracted features from {len(features_df)} epochs, shape: {features_df.shape}")
            return features_df
        else:
            # Data too short for epochs, process as single sample
            logger.warning(f"Data too short ({len(df)} samples) for epoch segmentation, processing as single sample")
            epoch_features = extract_features_from_single_epoch(df, eeg_channels, simple_mode=True)
            return pd.DataFrame([epoch_features])
            
    except Exception as e:
        logger.error(f"Error in feature extraction from timeseries: {str(e)}")
        raise FeatureExtractionError(f"Error in feature extraction from timeseries: {str(e)}")

def extract_features_from_single_epoch(df: pd.DataFrame, eeg_channels: List[str], simple_mode: bool = True) -> Dict[str, float]:
    """
    Extract features from a single epoch of EEG data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Single epoch DataFrame (rows=timepoints, columns=channels)
    eeg_channels : List[str]
        List of EEG channel names
    simple_mode : bool
        Retained for API compatibility. The backend now always uses the
        simple 5-band feature path.
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of extracted features
    """
    try:
        if not eeg_channels:
            raise FeatureExtractionError("No EEG channels available for feature extraction")
        if not simple_mode:
            logger.info("Legacy comprehensive backend feature extraction has been removed; using simple band features")

        all_band_powers = {'alpha': [], 'beta': [], 'theta': [], 'delta': [], 'gamma': []}

        for col in eeg_channels:
            channel_signal = df[col].values
            try:
                freqs, psd = compute_psd(channel_signal, fs=250)
                bands = {
                    'delta': (0.5, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30),
                    'gamma': (30, 45),
                }
                for band_name, band_range in bands.items():
                    power = compute_band_power(freqs, psd, band_range)
                    all_band_powers[band_name].append(power)
            except Exception as e:
                logger.warning(f"Error processing channel %s: %s", col, e)
                for band_name in all_band_powers.keys():
                    all_band_powers[band_name].append(0)

        feature_data = {
            band_name: (float(np.mean(powers)) if powers else 0.0)
            for band_name, powers in all_band_powers.items()
        }

        if 'eeg_state' in df.columns:
            feature_data['eeg_state'] = df['eeg_state'].iloc[0]
        elif 'state' in df.columns:
            feature_data['eeg_state'] = df['state'].iloc[0]
        elif 'label' in df.columns:
            feature_data['eeg_state'] = df['label'].iloc[0]

        return feature_data
    except Exception as e:
        logger.error(f"Error extracting features from single epoch: {str(e)}")
        raise FeatureExtractionError(f"Error extracting features from single epoch: {str(e)}")
