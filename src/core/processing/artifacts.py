import numpy as np
from scipy import signal
from scipy.stats import iqr, zscore


def detect_outliers(eeg_signal, threshold=3.0, method='zscore'):
    """
    Detect outliers in EEG signal using various methods.
    
    Parameters:
    -----------
    eeg_signal : array-like
        The EEG signal to process
    threshold : float
        Threshold for outlier detection
    method : str
        Method to use: 'zscore', 'iqr', or 'abs'
        
    Returns:
    --------
    outlier_mask : array
        Boolean mask of outliers (True for outliers)
    """
    if method == 'zscore':
        z_scores = zscore(eeg_signal, nan_policy='omit')
        return np.abs(z_scores) > threshold
    
    elif method == 'iqr':
        q1 = np.percentile(eeg_signal, 25)
        q3 = np.percentile(eeg_signal, 75)
        iqr_value = q3 - q1
        lower_bound = q1 - threshold * iqr_value
        upper_bound = q3 + threshold * iqr_value
        return (eeg_signal < lower_bound) | (eeg_signal > upper_bound)
    
    elif method == 'abs':
        return np.abs(eeg_signal) > threshold
    
    else:
        raise ValueError(f"Method '{method}' not recognized")


def detect_muscle_artifacts(eeg_signal, fs=250, threshold=0.1):
    """
    Detect muscle artifacts based on high frequency content.
    
    Parameters:
    -----------
    eeg_signal : array-like
        The EEG signal to process
    fs : float
        Sampling frequency in Hz
    threshold : float
        Threshold for high frequency power ratio
        
    Returns:
    --------
    artifact_mask : array
        Boolean mask of detected artifacts (True for artifacts)
    """
    # Design bandpass filters
    high_freq_band = [20, 45]  # Hz, typical muscle activity band
    
    # Calculate normalized power in high frequency band
    freqs, psd = signal.welch(eeg_signal, fs=fs)
    
    # Find indices for the high frequency band
    idx_high = (freqs >= high_freq_band[0]) & (freqs <= high_freq_band[1])
    idx_total = freqs <= 45  # Total power up to 45 Hz
    
    # Calculate power
    power_high = np.sum(psd[idx_high])
    power_total = np.sum(psd[idx_total])
    
    # Calculate ratio
    if power_total > 0:
        high_freq_ratio = power_high / power_total
    else:
        high_freq_ratio = 0
    
    # Create mask for segments with high muscle activity
    return high_freq_ratio > threshold


def detect_eye_blinks(eeg_signal, fs=250, threshold=2.5):
    """
    Detect eye blinks in frontal channels.
    
    Parameters:
    -----------
    eeg_signal : array-like
        The EEG signal to process
    fs : float
        Sampling frequency in Hz
    threshold : float
        Threshold in standard deviations for peak detection
        
    Returns:
    --------
    blink_mask : array
        Boolean mask of detected blinks (True for blinks)
    """
    # Filter signal to isolate slow waves typical of eye movements
    b, a = signal.butter(3, [1, 10], btype='bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, eeg_signal)
    
    # Detect peaks that exceed threshold
    std_dev = np.std(filtered)
    mean_val = np.mean(filtered)
    threshold_value = mean_val + threshold * std_dev
    
    # Create mask for blink artifacts
    return filtered > threshold_value


def remove_artifacts_interpolation(eeg_signal, artifact_mask, window_size=10):
    """
    Remove artifacts by interpolating over artifact regions.
    
    Parameters:
    -----------
    eeg_signal : array-like
        The EEG signal to process
    artifact_mask : array
        Boolean mask identifying artifacts
    window_size : int
        Window size for interpolation context
        
    Returns:
    --------
    cleaned_signal : array
        EEG signal with artifacts removed
    """
    # Create a copy of the signal
    cleaned = np.copy(eeg_signal)
    
    # Find contiguous artifact segments
    artifact_indices = np.where(artifact_mask)[0]
    
    if len(artifact_indices) == 0:
        return cleaned
    
    # Process each artifact segment
    i = 0
    while i < len(artifact_indices):
        # Find contiguous segment
        start = artifact_indices[i]
        end = start
        while i + 1 < len(artifact_indices) and artifact_indices[i + 1] == end + 1:
            i += 1
            end = artifact_indices[i]
        
        # Get values before and after artifact
        pre_window = max(0, start - window_size)
        post_window = min(len(eeg_signal), end + window_size + 1)
        
        # Get valid values for interpolation
        pre_values = eeg_signal[pre_window:start]
        post_values = eeg_signal[end+1:post_window]
        
        # Interpolate if we have enough context
        if len(pre_values) > 0 and len(post_values) > 0:
            # Use linear interpolation
            x_interp = np.arange(start, end+1)
            x_known = np.concatenate([np.arange(pre_window, start), np.arange(end+1, post_window)])
            y_known = np.concatenate([pre_values, post_values])
            
            # Linear interpolation
            cleaned[start:end+1] = np.interp(x_interp, x_known, y_known)
        
        i += 1
    
    return cleaned


def wavelet_denoise(eeg_signal, wavelet='db4', level=4, threshold_multiplier=1.0):
    """
    Denoise EEG signal using wavelet thresholding.
    
    Parameters:
    -----------
    eeg_signal : array-like
        The EEG signal to process
    wavelet : str
        Wavelet to use (e.g. 'db4', 'sym4')
    level : int
        Decomposition level
    threshold_multiplier : float
        Multiplier for threshold calculation
        
    Returns:
    --------
    denoised_signal : array
        Denoised EEG signal
    """
    try:
        import pywt
        
        # Pad signal to power of 2 if needed
        orig_len = len(eeg_signal)
        power = 2**np.ceil(np.log2(orig_len))
        padded = np.zeros(int(power))
        padded[:orig_len] = eeg_signal
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(padded, wavelet, level=level)
        
        # Threshold detail coefficients
        for i in range(1, len(coeffs)):
            # Calculate threshold
            detail = coeffs[i]
            sigma = np.median(np.abs(detail)) / 0.6745
            threshold = sigma * threshold_multiplier * np.sqrt(2 * np.log(len(detail)))
            
            # Apply soft thresholding
            coeffs[i] = pywt.threshold(detail, threshold, mode='soft')
        
        # Reconstruct signal
        denoised = pywt.waverec(coeffs, wavelet)
        
        # Return signal of original length
        return denoised[:orig_len]
    
    except ImportError:
        print("PyWavelets not installed. Returning original signal.")
        return eeg_signal


def clean_eeg(eeg_data, fs=250, channels=None):
    """
    Main function to clean EEG data with multiple strategies.
    
    Parameters:
    -----------
    eeg_data : 2D array
        EEG data with shape (channels, samples) or (samples, channels)
    fs : float
        Sampling frequency in Hz
    channels : list
        List of channel types/names (frontal, temporal, etc.)
        
    Returns:
    --------
    cleaned_data : 2D array
        Cleaned EEG data with same shape as input
    artifact_report : dict
        Report of detected artifacts by channel
    """
    # Determine data shape
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
        single_channel = True
    else:
        single_channel = False
        
    n_channels, n_samples = eeg_data.shape
    
    if channels is None:
        channels = [f"channel_{i}" for i in range(n_channels)]
    
    # Initialize output and artifact report
    cleaned_data = np.zeros_like(eeg_data)
    artifact_report = {ch: {"outliers": 0, "muscle": 0, "blinks": 0} for ch in channels}
    
    # Process each channel
    for ch_idx, ch_name in enumerate(channels):
        signal = eeg_data[ch_idx]
        
        # 1. Detect and remove outliers
        outlier_mask = detect_outliers(signal, threshold=3.0)
        artifact_report[ch_name]["outliers"] = np.sum(outlier_mask) / n_samples * 100  # percentage
        
        # 2. Detect muscle artifacts
        muscle_mask = detect_muscle_artifacts(signal, fs=fs)
        artifact_report[ch_name]["muscle"] = np.sum(muscle_mask) / n_samples * 100  # percentage
        
        # 3. Detect eye blinks (mainly for frontal channels)
        if "frontal" in ch_name.lower() or "fp" in ch_name.lower() or ch_idx == 0:
            blink_mask = detect_eye_blinks(signal, fs=fs)
            artifact_report[ch_name]["blinks"] = np.sum(blink_mask) / n_samples * 100  # percentage
        else:
            blink_mask = np.zeros(n_samples, dtype=bool)
        
        # Combine all artifact masks
        combined_mask = outlier_mask | muscle_mask | blink_mask
        
        # Apply artifact removal
        cleaned = remove_artifacts_interpolation(signal, combined_mask)
        
        # Apply wavelet denoising
        cleaned = wavelet_denoise(cleaned)
        
        # Store cleaned data
        cleaned_data[ch_idx] = cleaned
    
    # Return in original format
    if single_channel:
        return cleaned_data[0], artifact_report
    else:
        return cleaned_data, artifact_report
