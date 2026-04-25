import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def apply_bandpass_filter(data, low_freq, high_freq, fs=250, order=4):
    """
    Apply a bandpass filter to EEG data.
    
    Parameters:
    -----------
    data : array-like
        The signal to filter, can be 1D or 2D
    low_freq : float
        Lower cutoff frequency in Hz
    high_freq : float
        Higher cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order
    
    Returns:
    --------
    filtered_data : array
        Filtered signal with same shape as input
    """
    # Check if input is 1D or 2D
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(1, -1)
    
    # Design filter
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    # Return in original format
    if is_1d:
        return filtered_data[0]
    else:
        return filtered_data


def apply_notch_filter(data, notch_freq, fs=250, q=30):
    """
    Apply a notch filter to remove power line interference.
    
    Parameters:
    -----------
    data : array-like
        The signal to filter
    notch_freq : float
        Frequency to remove (typically 50 or 60 Hz)
    fs : float
        Sampling frequency in Hz
    q : float
        Quality factor. Higher values mean narrower notch
    
    Returns:
    --------
    filtered_data : array
        Filtered signal with same shape as input
    """
    # Check if input is 1D or 2D
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(1, -1)
    
    # Design filter
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = signal.iirnotch(freq, q)
    
    # Apply filter
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    # Return in original format
    if is_1d:
        return filtered_data[0]
    else:
        return filtered_data


def apply_highpass_filter(data, cutoff_freq, fs=250, order=4):
    """
    Apply a high-pass filter to remove slow drift.
    
    Parameters:
    -----------
    data : array-like
        The signal to filter
    cutoff_freq : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order
    
    Returns:
    --------
    filtered_data : array
        Filtered signal with same shape as input
    """
    # Check if input is 1D or 2D
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(1, -1)
    
    # Design filter
    nyquist = 0.5 * fs
    cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, cutoff, btype='high')
    
    # Apply filter
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    # Return in original format
    if is_1d:
        return filtered_data[0]
    else:
        return filtered_data


def apply_lowpass_filter(data, cutoff_freq, fs=250, order=4):
    """
    Apply a low-pass filter.
    
    Parameters:
    -----------
    data : array-like
        The signal to filter
    cutoff_freq : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order
    
    Returns:
    --------
    filtered_data : array
        Filtered signal with same shape as input
    """
    # Check if input is 1D or 2D
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(1, -1)
    
    # Design filter
    nyquist = 0.5 * fs
    cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, cutoff, btype='low')
    
    # Apply filter
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    # Return in original format
    if is_1d:
        return filtered_data[0]
    else:
        return filtered_data


def apply_savgol_filter(data, window_length=11, poly_order=3):
    """
    Apply a Savitzky-Golay filter for smooth filtering with better preservation of peaks.
    
    Parameters:
    -----------
    data : array-like
        The signal to filter
    window_length : int
        The length of the filter window (must be odd)
    poly_order : int
        The order of the polynomial used to fit the samples (must be less than window_length)
    
    Returns:
    --------
    filtered_data : array
        Filtered signal with same shape as input
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Check if input is 1D or 2D
    is_1d = data.ndim == 1
    if is_1d:
        data = data.reshape(1, -1)
    
    # Apply filter
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.savgol_filter(data[i], window_length, poly_order)
    
    # Return in original format
    if is_1d:
        return filtered_data[0]
    else:
        return filtered_data


def apply_eeg_preprocessing(data, fs=250, notch_freq=50):
    """
    Standard EEG preprocessing pipeline
    
    Parameters:
    -----------
    data : array-like
        The EEG data to process
    fs : float
        Sampling frequency in Hz
    notch_freq : float
        Frequency to remove (typically 50 or 60 Hz)
    
    Returns:
    --------
    filtered_data : array
        Preprocessed EEG data
    """
    # Apply high-pass filter to remove slow drift (0.5 Hz)
    data = apply_highpass_filter(data, 0.5, fs)
    
    # Apply notch filter to remove power line interference
    data = apply_notch_filter(data, notch_freq, fs)
    
    # Apply low-pass filter to remove high-frequency noise (45 Hz)
    data = apply_lowpass_filter(data, 45, fs)
    
    return data


def filter_eeg_bands(data, fs=250, band=None):
    """
    Filter EEG into standard frequency bands.
    
    Parameters:
    -----------
    data : array-like
        The EEG data to filter
    fs : float
        Sampling frequency in Hz
    band : str or None
        Specific band to extract ('delta', 'theta', 'alpha', 'beta', 'gamma')
        If None, returns a dictionary with all bands
    
    Returns:
    --------
    filtered_bands : dict or array
        Dictionary with filtered data for each band or specific band data
    """
    # Define standard EEG frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Check if specific band is requested
    if band is not None:
        if band not in bands:
            raise ValueError(f"Unknown band: {band}. Available bands: {list(bands.keys())}")
        
        low_freq, high_freq = bands[band]
        return apply_bandpass_filter(data, low_freq, high_freq, fs)
    
    # Process all bands
    filtered_bands = {}
    for band_name, (low_freq, high_freq) in bands.items():
        filtered_bands[band_name] = apply_bandpass_filter(data, low_freq, high_freq, fs)
    
    return filtered_bands


def plot_filter_response(fs=250):
    """
    Plot frequency response of different EEG filters.
    
    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    """
    # Set up figure
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Frequency vector for plotting
    freq = np.linspace(0, fs/2, 1000)
    
    # EEG bands for plotting
    bands = {
        'delta': (0.5, 4, 'royalblue'),
        'theta': (4, 8, 'green'),
        'alpha': (8, 13, 'red'),
        'beta': (13, 30, 'purple'),
        'gamma': (30, 45, 'orange')
    }
    
    # Plot 1: Individual band filters
    for band_name, (low, high, color) in bands.items():
        # Design bandpass filter
        nyquist = 0.5 * fs
        b, a = signal.butter(4, [low/nyquist, high/nyquist], btype='band')
        
        # Calculate frequency response
        w, h = signal.freqz(b, a, worN=freq)
        
        # Convert to Hz and dB
        f = w * fs / (2 * np.pi)
        db = 20 * np.log10(abs(h))
        
        # Plot
        axs[0].plot(f, db, label=f'{band_name} ({low}-{high} Hz)', color=color)
    
    axs[0].set_title('EEG Band Filters')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Gain (dB)')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_ylim(-60, 5)
    
    # Plot 2: Notch filter at 50 Hz and 60 Hz
    for notch_freq, color, label in [(50, 'red', '50 Hz'), (60, 'blue', '60 Hz')]:
        # Design notch filter
        nyquist = 0.5 * fs
        b, a = signal.iirnotch(notch_freq/nyquist, 30)
        
        # Calculate frequency response
        w, h = signal.freqz(b, a, worN=freq)
        
        # Convert to Hz and dB
        f = w * fs / (2 * np.pi)
        db = 20 * np.log10(abs(h))
        
        # Plot
        axs[1].plot(f, db, label=f'Notch filter at {label}', color=color)
    
    axs[1].set_title('Notch Filters')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Gain (dB)')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_ylim(-60, 5)
    
    # Plot 3: High-pass filters for drift removal
    for cutoff, color, label in [(0.1, 'green', '0.1 Hz'), (0.5, 'blue', '0.5 Hz'), (1.0, 'red', '1.0 Hz')]:
        # Design high-pass filter
        nyquist = 0.5 * fs
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        
        # Calculate frequency response
        w, h = signal.freqz(b, a, worN=freq)
        
        # Convert to Hz and dB
        f = w * fs / (2 * np.pi)
        db = 20 * np.log10(abs(h))
        
        # Plot
        axs[2].plot(f, db, label=f'High-pass at {label}', color=color)
    
    axs[2].set_title('High-pass Filters for Drift Removal')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Gain (dB)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_ylim(-60, 5)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
