from scipy.signal import welch, butter, filtfilt, hilbert, coherence
import numpy as np
from scipy.stats import skew, kurtosis, entropy, pearsonr
import pandas as pd
from scipy.integrate import simpson
import antropy as ant
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import pywt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

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

def compute_hjorth_parameters(data: np.ndarray) -> Tuple[float, float, float]:
    """Enhanced Hjorth parameters computation with validation"""
    try:
        if len(data) < 2:
            return 0, 0, 0
        
        # Activity - variance of the signal
        activity = np.var(data)
        
        # First derivative
        diff1 = np.diff(data)
        # Mobility
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        
        # Second derivative
        diff2 = np.diff(diff1)
        # Complexity
        complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 and np.var(diff1) > 0 else 0
        
        return activity, mobility, complexity
    except Exception as e:
        raise FeatureExtractionError(f"Error in Hjorth parameters computation: {str(e)}")

def compute_spectral_entropy(psd: np.ndarray) -> float:
    """Enhanced spectral entropy computation with validation"""
    try:
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        return entropy(psd_norm)
    except Exception as e:
        raise FeatureExtractionError(f"Error in spectral entropy computation: {str(e)}")

def compute_nonlinear_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute nonlinear features with error handling"""
    features = {}
    try:
        if len(signal) > 10:
            features['sample_entropy'] = ant.sample_entropy(signal)
            features['app_entropy'] = ant.app_entropy(signal)
            features['perm_entropy'] = ant.perm_entropy(signal)
            features['spectral_entropy'] = ant.spectral_entropy(signal, sf=250)
            features['svd_entropy'] = ant.svd_entropy(signal)
        else:
            features.update({k: 0 for k in ['sample_entropy', 'app_entropy', 'perm_entropy', 
                                          'spectral_entropy', 'svd_entropy']})
    except Exception as e:
        logger.warning(f"Error computing nonlinear features: {str(e)}")
        features.update({k: 0 for k in ['sample_entropy', 'app_entropy', 'perm_entropy', 
                                      'spectral_entropy', 'svd_entropy']})
    return features

def compute_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute time domain features with validation"""
    features = {}
    try:
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['skew'] = skew(signal) if len(signal) > 2 else 0
        features['kurtosis'] = kurtosis(signal) if len(signal) > 2 else 0
        features['zero_crossings'] = np.sum(np.diff(np.signbit(signal).astype(int)) != 0)
        features['peak_to_peak'] = np.max(signal) - np.min(signal)
        features['rms'] = np.sqrt(np.mean(signal ** 2))
    except Exception as e:
        raise FeatureExtractionError(f"Error in time domain feature computation: {str(e)}")
    return features

def compute_frequency_domain_features(signal: np.ndarray, fs: float = 250) -> Dict[str, float]:
    """Compute frequency domain features with validation"""
    features = {}
    try:
        freqs, psd = compute_psd(signal, fs)
        
        # Define standard EEG frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Compute band powers
        for band_name, band_range in bands.items():
            features[f'{band_name}_power'] = compute_band_power(freqs, psd, band_range)
        
        # Compute band power ratios
        alpha_power = features['alpha_power']
        beta_power = features['beta_power']
        theta_power = features['theta_power']
        
        if beta_power > 0:
            features['alpha_beta_ratio'] = alpha_power / beta_power
        else:
            features['alpha_beta_ratio'] = 0
            
        if theta_power > 0:
            features['beta_theta_ratio'] = beta_power / theta_power
        else:
            features['beta_theta_ratio'] = 0
        
        # Spectral entropy
        features['spectral_entropy'] = compute_spectral_entropy(psd)
        
        # Peak frequency
        peak_freq_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_freq_idx]
        
    except Exception as e:
        raise FeatureExtractionError(f"Error in frequency domain feature computation: {str(e)}")
    return features

def compute_wavelet_features(signal: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict[str, float]:
    """Compute wavelet decomposition features"""
    features = {}
    try:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Compute features for each decomposition level
        for i, coeff in enumerate(coeffs):
            # Energy
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff ** 2)
            
            # Mean absolute value
            features[f'wavelet_mav_level_{i}'] = np.mean(np.abs(coeff))
            
            # Standard deviation
            features[f'wavelet_std_level_{i}'] = np.std(coeff)
            
            # Entropy
            hist, _ = np.histogram(coeff, bins=50, density=True)
            features[f'wavelet_entropy_level_{i}'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return features
    except Exception as e:
        logger.warning(f"Error in wavelet feature computation: {str(e)}")
        return {f'wavelet_{k}_level_{i}': 0 for i in range(level+1) for k in ['energy', 'mav', 'std', 'entropy']}

def compute_harmonic_features(signal: np.ndarray, fs: float = 250) -> Dict[str, float]:
    """Compute harmonic analysis features"""
    features = {}
    try:
        # Compute FFT
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, 1/fs)
        
        # Get positive frequencies only
        pos_freq_mask = freqs > 0
        freqs = freqs[pos_freq_mask]
        fft_vals = np.abs(fft_vals[pos_freq_mask])
        
        # Find peaks in frequency domain
        peaks, properties = find_peaks(fft_vals, height=0)
        
        if len(peaks) > 0:
            # Dominant frequency
            features['dominant_freq'] = freqs[peaks[np.argmax(properties['peak_heights'])]]
            
            # Harmonic frequencies
            peak_freqs = freqs[peaks]
            peak_heights = properties['peak_heights']
            
            # Sort peaks by height
            sorted_idx = np.argsort(peak_heights)[::-1]
            peak_freqs = peak_freqs[sorted_idx]
            peak_heights = peak_heights[sorted_idx]
            
            # Store top 3 harmonic frequencies and their amplitudes
            for i in range(min(3, len(peak_freqs))):
                features[f'harmonic_freq_{i+1}'] = peak_freqs[i]
                features[f'harmonic_amp_{i+1}'] = peak_heights[i]
        else:
            features.update({
                'dominant_freq': 0,
                'harmonic_freq_1': 0, 'harmonic_amp_1': 0,
                'harmonic_freq_2': 0, 'harmonic_amp_2': 0,
                'harmonic_freq_3': 0, 'harmonic_amp_3': 0
            })
        
        return features
    except Exception as e:
        logger.warning(f"Error in harmonic feature computation: {str(e)}")
        return {
            'dominant_freq': 0,
            'harmonic_freq_1': 0, 'harmonic_amp_1': 0,
            'harmonic_freq_2': 0, 'harmonic_amp_2': 0,
            'harmonic_freq_3': 0, 'harmonic_amp_3': 0
        }

def compute_phase_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute phase-based features using Hilbert transform"""
    features = {}
    try:
        # Compute analytic signal
        analytic_signal = hilbert(signal)
        
        # Instantaneous phase
        phase = np.unwrap(np.angle(analytic_signal))
        
        # Phase features
        features['phase_mean'] = np.mean(phase)
        features['phase_std'] = np.std(phase)
        features['phase_range'] = np.max(phase) - np.min(phase)
        
        # Phase velocity
        phase_velocity = np.diff(phase)
        features['phase_velocity_mean'] = np.mean(phase_velocity)
        features['phase_velocity_std'] = np.std(phase_velocity)
        
        return features
    except Exception as e:
        logger.warning(f"Error in phase feature computation: {str(e)}")
        return {
            'phase_mean': 0, 'phase_std': 0, 'phase_range': 0,
            'phase_velocity_mean': 0, 'phase_velocity_std': 0
        }

def compute_cross_channel_features(signal1: np.ndarray, signal2: np.ndarray, fs: float = 250) -> Dict[str, float]:
    """Compute cross-channel features between two signals"""
    features = {}
    try:
        # Cross-correlation
        corr = np.correlate(signal1, signal2, mode='full')
        features['cross_corr_max'] = np.max(np.abs(corr))
        features['cross_corr_lag'] = np.argmax(np.abs(corr)) - len(signal1) + 1
        
        # Coherence
        freqs, coh = coherence(signal1, signal2, fs=fs)
        features['mean_coherence'] = np.mean(coh)
        features['max_coherence'] = np.max(coh)
        features['coherence_bandwidth'] = np.sum(coh > 0.5) * (freqs[1] - freqs[0])
        
        # Phase synchronization
        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)
        phase_diff = np.angle(analytic1) - np.angle(analytic2)
        features['phase_sync'] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return features
    except Exception as e:
        logger.warning(f"Error in cross-channel feature computation: {str(e)}")
        return {
            'cross_corr_max': 0, 'cross_corr_lag': 0,
            'mean_coherence': 0, 'max_coherence': 0, 'coherence_bandwidth': 0,
            'phase_sync': 0
        }

def compute_pca_features(signals: np.ndarray, n_components: int = 3) -> Dict[str, float]:
    """Compute PCA-based features from multiple channels"""
    features = {}
    try:
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(signals)
        
        # Store explained variance ratios
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            features[f'pca_var_ratio_{i+1}'] = var_ratio
        
        # Store total explained variance
        features['pca_total_var_ratio'] = np.sum(pca.explained_variance_ratio_)
        
        return features
    except Exception as e:
        logger.warning(f"Error in PCA feature computation: {str(e)}")
        return {f'pca_var_ratio_{i+1}': 0 for i in range(n_components)}

def extract_features(df: pd.DataFrame, simple_mode: bool = True) -> pd.DataFrame:
    """
    Enhanced feature extraction with advanced EEG-specific features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing EEG data.
        Can be either:
        - Raw time-series data (rows=timepoints, columns=channels)
        - Pre-computed features (rows=samples, columns=features)
    simple_mode : bool
        If True, extract only 5 core frequency band features (alpha, beta, theta, delta, gamma)
        If False, extract comprehensive 930-feature set
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted features
    """
    try:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude timestamp and state columns
        eeg_channels = [col for col in numerical_cols if col.lower() not in ['timestamp', 'time', 'eeg_state', 'state', 'label']]
        
        # Check if this is raw time-series data or pre-computed features
        # Raw data: many rows (>50), few columns (channels)
        # Features: few rows (samples), many columns (features)
        is_raw_timeseries = len(df) > 50 and len(eeg_channels) < 100
        
        if is_raw_timeseries:
            logger.info(f"Processing raw time-series data: {len(df)} timepoints, {len(eeg_channels)} channels")
            # Process as time-series: extract features from each channel's full signal
            return extract_features_from_timeseries(df, eeg_channels, simple_mode=simple_mode)
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

def extract_features_from_timeseries(df: pd.DataFrame, eeg_channels: List[str], simple_mode: bool = True) -> pd.DataFrame:
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
        If True, extract only 5 core frequency band features averaged across channels
        If False, extract comprehensive per-channel feature set
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per epoch containing extracted features
    """
    try:
        # Segment into epochs if data is long enough
        epoch_length = 257  # 1.028s at 250Hz
        
        if len(df) >= epoch_length:
            epochs = segment_into_epochs(df, epoch_length_samples=epoch_length, overlap=0.0)
            logger.info(f"Processing {len(epochs)} epochs...")
            
            # Process each epoch
            all_epoch_features = []
            for epoch_idx, epoch_df in enumerate(epochs):
                epoch_features = extract_features_from_single_epoch(epoch_df, eeg_channels, simple_mode=simple_mode)
                all_epoch_features.append(epoch_features)
            
            # Combine all epochs into a single dataframe
            features_df = pd.DataFrame(all_epoch_features)
            logger.info(f"Extracted features from {len(features_df)} epochs, shape: {features_df.shape}")
            return features_df
        else:
            # Data too short for epochs, process as single sample
            logger.warning(f"Data too short ({len(df)} samples) for epoch segmentation, processing as single sample")
            epoch_features = extract_features_from_single_epoch(df, eeg_channels, simple_mode=simple_mode)
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
        If True, extract only 5 averaged frequency band features
        If False, extract comprehensive per-channel feature set
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of extracted features
    """
    try:
        if simple_mode:
            # Simple mode: Extract only 5 core frequency band features averaged across all channels
            all_band_powers = {'alpha': [], 'beta': [], 'theta': [], 'delta': [], 'gamma': []}
            
            for col in eeg_channels:
                channel_signal = df[col].values
                try:
                    freqs, psd = compute_psd(channel_signal, fs=250)
                    
                    # Define standard EEG frequency bands
                    bands = {
                        'delta': (0.5, 4),
                        'theta': (4, 8),
                        'alpha': (8, 13),
                        'beta': (13, 30),
                        'gamma': (30, 45)
                    }
                    
                    # Compute band powers for this channel
                    for band_name, band_range in bands.items():
                        power = compute_band_power(freqs, psd, band_range)
                        all_band_powers[band_name].append(power)
                except Exception as e:
                    logger.warning(f"Error processing channel {col}: {str(e)}")
                    # Add zeros if processing fails
                    for band_name in all_band_powers.keys():
                        all_band_powers[band_name].append(0)
            
            # Average across all channels
            feature_data = {}
            for band_name, powers in all_band_powers.items():
                feature_data[band_name] = np.mean(powers) if powers else 0
            
            # Preserve target column if present
            if 'eeg_state' in df.columns:
                feature_data['eeg_state'] = df['eeg_state'].iloc[0]
            elif 'state' in df.columns:
                feature_data['eeg_state'] = df['state'].iloc[0]
            elif 'label' in df.columns:
                feature_data['eeg_state'] = df['label'].iloc[0]
            
            return feature_data
        
        # Complex mode: Extract comprehensive features per channel
        def process_channel(channel_data: np.ndarray, col_name: str) -> Dict[str, float]:
            """Process a single channel's entire time series"""
            features = {}
            
            # Time domain features
            features.update({f'{col_name}_{k}': v for k, v in 
                           compute_time_domain_features(channel_data).items()})
            
            # Hjorth parameters
            activity, mobility, complexity = compute_hjorth_parameters(channel_data)
            features[f'{col_name}_activity'] = activity
            features[f'{col_name}_mobility'] = mobility
            features[f'{col_name}_complexity'] = complexity
            
            # Frequency domain features
            freq_features = compute_frequency_domain_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in freq_features.items()})
            
            # Nonlinear features
            nl_features = compute_nonlinear_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in nl_features.items()})
            
            # Wavelet features
            wavelet_features = compute_wavelet_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in wavelet_features.items()})
            
            # Harmonic features
            harmonic_features = compute_harmonic_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in harmonic_features.items()})
            
            # Phase features
            phase_features = compute_phase_features(channel_data)
            features.update({f'{col_name}_{k}': v for k, v in phase_features.items()})
            
            return features
        
        feature_data = {}
        
        # Process each channel's time series for this epoch
        for col in eeg_channels:
            channel_signal = df[col].values  # Get entire column as numpy array
            feature_data.update(process_channel(channel_signal, col))
        
        # Multi-channel features
        if len(eeg_channels) > 1:
            # Compute cross-channel features for first few channel pairs (to avoid explosion)
            max_pairs = min(5, len(eeg_channels) - 1)
            for i in range(max_pairs):
                ch1 = eeg_channels[i]
                ch2 = eeg_channels[i + 1]
                sig1 = df[ch1].values
                sig2 = df[ch2].values
                
                try:
                    cross_features = compute_cross_channel_features(sig1, sig2)
                    feature_data.update({
                        f'{ch1}_{ch2}_{k}': v for k, v in cross_features.items()
                    })
                except Exception as e:
                    logger.warning(f"Could not compute cross-channel features for {ch1}-{ch2}: {str(e)}")
            
            # Compute PCA features for all channels
            try:
                signals = np.array([df[col].values for col in eeg_channels])
                pca_features = compute_pca_features(signals)
                feature_data.update(pca_features)
            except Exception as e:
                logger.warning(f"Could not compute PCA features: {str(e)}")
        
        # Preserve target column if present
        if 'eeg_state' in df.columns:
            feature_data['eeg_state'] = df['eeg_state'].iloc[0]
        elif 'state' in df.columns:
            feature_data['eeg_state'] = df['state'].iloc[0]
        elif 'label' in df.columns:
            feature_data['eeg_state'] = df['label'].iloc[0]
            
        # Return as dictionary
        return feature_data
        
    except Exception as e:
        logger.error(f"Error extracting features from single epoch: {str(e)}")
        raise FeatureExtractionError(f"Error extracting features from single epoch: {str(e)}")