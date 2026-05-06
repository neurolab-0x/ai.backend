import numpy as np
import pandas as pd
from scipy import stats
from scipy import signal
from src.core.processing.filters import filter_eeg_bands
import logging

logger = logging.getLogger(__name__)

def label_eeg_states(df, method='auto', threshold_params=None):
    """
    Label EEG data with mental states based on signal characteristics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing EEG data
    method : str
        Labeling method ('auto', 'frequency', 'threshold', 'cluster')
    threshold_params : dict or None
        Optional parameters for thresholding
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with added 'eeg_state' column containing state labels
        0: calm/relaxed, 1: neutral, 2: stressed/focused
    """
    logger.info(f"Labeling EEG states using method: {method}")
    
    # Make a copy of the dataframe to avoid modifying the original
    df_out = df.copy()
    
    # Preserve ground-truth labels when they already exist.
    if 'eeg_state' in df.columns:
        logger.info("Using existing 'eeg_state' labels from dataset")
        return df_out
    if 'state' in df.columns:
        logger.info("Using existing 'state' column as ground-truth EEG labels")
        df_out['eeg_state'] = df_out['state']
        return df_out
    if 'label' in df.columns:
        logger.info("Using existing 'label' column as ground-truth EEG labels")
        df_out['eeg_state'] = df_out['label']
        return df_out

    # Extract only EEG data columns (exclude any existing labels)
    eeg_cols = [col for col in df.columns if col not in {'eeg_state', 'state', 'label'}]
    
    # Default threshold parameters
    default_threshold = {
        'alpha_high': 0.25,  # High alpha power threshold (relaxed)
        'beta_high': 0.3,    # High beta power threshold (stressed/focused)
        'alpha_beta_low': 0.5,  # Low alpha/beta ratio (stressed/focused)
        'alpha_beta_high': 1.5   # High alpha/beta ratio (relaxed)
    }
    
    # Use provided parameters or defaults
    if threshold_params is None:
        threshold_params = default_threshold
    else:
        # Merge with defaults for any missing parameters
        for key, value in default_threshold.items():
            if key not in threshold_params:
                threshold_params[key] = value
    
    # Auto-detect method if data has attributes
    if method == 'auto' and hasattr(df, 'attrs') and 'ch_types' in df.attrs:
        # If we have channel types, prefer frequency-based analysis
        method = 'frequency'
    elif method == 'auto':
        # Default to threshold-based
        method = 'threshold'
    
    # Apply the selected method
    if method == 'frequency':
        df_out = label_by_frequency_bands(df_out, eeg_cols, threshold_params)
    elif method == 'threshold':
        df_out = label_by_threshold(df_out, eeg_cols, threshold_params)
    elif method == 'cluster':
        df_out = label_by_clustering(df_out, eeg_cols)
    else:
        raise ValueError(f"Unknown labeling method: {method}")
    
    logger.info(f"Labeled data: {df_out['eeg_state'].value_counts().to_dict()}")
    return df_out


def label_by_frequency_bands(df, eeg_cols, thresholds):
    """Label based on frequency band power and ratios."""
    # Extract sampling frequency from attributes if available
    fs = 250  # Default
    if hasattr(df, 'attrs') and 'sfreq' in df.attrs:
        fs = df.attrs['sfreq']
    
    # Collect all channel data
    all_channels = []
    for col in eeg_cols:
        all_channels.append(df[col].values)
    
    # Stack all channels
    eeg_data = np.vstack(all_channels)
    
    # Extract frequency bands
    try:
        bands = filter_eeg_bands(eeg_data, fs=fs)
        
        # Calculate average band powers across channels
        alpha_power = np.mean([np.mean(bands['alpha'][i]**2) for i in range(len(bands['alpha']))])
        beta_power = np.mean([np.mean(bands['beta'][i]**2) for i in range(len(bands['beta']))])
        theta_power = np.mean([np.mean(bands['theta'][i]**2) for i in range(len(bands['theta']))])
        
        # Normalize powers
        total_power = alpha_power + beta_power + theta_power + 1e-10  # Avoid division by zero
        alpha_norm = alpha_power / total_power
        beta_norm = beta_power / total_power
        
        # Calculate alpha/beta ratio
        alpha_beta_ratio = alpha_power / (beta_power + 1e-10)
        
        # Determine states based on band powers and ratios
        states = np.ones(len(df), dtype=int)  # Default: neutral (1)
        
        # High alpha, low beta: relaxed (0)
        relaxed_mask = ((alpha_norm > thresholds['alpha_high']) | 
                        (alpha_beta_ratio > thresholds['alpha_beta_high']))
        states[relaxed_mask] = 0
        
        # High beta, low alpha: stressed/focused (2)
        stressed_mask = ((beta_norm > thresholds['beta_high']) | 
                         (alpha_beta_ratio < thresholds['alpha_beta_low']))
        states[stressed_mask] = 2
        
        # Add states to dataframe
        df['eeg_state'] = states
        
    except Exception as e:
        logger.error(f"Error in frequency band labeling: {str(e)}")
        # Fall back to simpler method
        df = label_by_threshold(df, eeg_cols, thresholds)
    
    return df


def label_by_threshold(df, eeg_cols, thresholds):
    """Label based on simple signal statistics thresholds."""
    # Calculate standard deviation and mean for each channel
    std_values = []
    for col in eeg_cols:
        std_values.append(df[col].std())
    
    # Use average standard deviation as a measure of activity
    avg_std = np.mean(std_values)
    
    # Calculate z-score for activity level
    z_scores = stats.zscore(df[eeg_cols].mean(axis=1))
    
    # Label based on activity level
    states = np.ones(len(df), dtype=int)  # Default: neutral (1)
    
    # Low activity: relaxed (0)
    relaxed_mask = z_scores < -0.5
    states[relaxed_mask] = 0
    
    # High activity: stressed/focused (2)
    stressed_mask = z_scores > 0.5
    states[stressed_mask] = 2
    
    # Add states to dataframe
    df['eeg_state'] = states
    
    return df


def label_by_clustering(df, eeg_cols):
    """Label using unsupervised clustering."""
    try:
        from sklearn.cluster import KMeans
        
        # Extract features for clustering
        features = df[eeg_cols].values
        
        # Apply K-means clustering with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Sort clusters by feature means to align with our state definitions
        cluster_means = [np.mean(features[clusters == i]) for i in range(3)]
        cluster_order = np.argsort(cluster_means)
        
        # Map clusters to states (0: relaxed, 1: neutral, 2: stressed)
        state_map = {
            cluster_order[0]: 0,  # Lowest mean -> relaxed
            cluster_order[1]: 1,  # Middle mean -> neutral
            cluster_order[2]: 2   # Highest mean -> stressed
        }
        
        # Create state column
        df['eeg_state'] = [state_map[c] for c in clusters]
        
    except Exception as e:
        logger.error(f"Error in clustering labeling: {str(e)}")
        # Fall back to simpler method
        df = label_by_threshold(df, eeg_cols, None)
    
    return df
