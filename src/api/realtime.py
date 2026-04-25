import numpy as np
import logging
import time
from datetime import datetime
import pandas as pd
from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.features import extract_features, extract_features_from_timeseries
from src.core.processing.temporal import temporal_smoothing
from src.core.processing.artifacts import clean_eeg
from src.core.processing.filters import apply_eeg_preprocessing
from src.config.settings import PROCESSING_CONFIG
from src.services.data_service import DataHandler, EEGDataPoint
from src.services.recommendation import NLPRecommendationEngine
from src.services.database import db_service
from src.queue import safe_enqueue
import asyncio
from typing import Dict, Any
from src.core.ml.model_types import sanitize_model_type
from src.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)

# Streaming buffer for continuous data processing
class StreamBuffer:
    def __init__(self, max_size=5000, channels=None):
        """Initialize a streaming buffer for EEG data
        
        Parameters:
        -----------
        max_size : int
            Maximum number of samples to store in buffer
        channels : int or None
            Number of EEG channels
        """
        self.max_size = max_size
        self.channels = channels
        self.buffer = None
        self.last_processed_idx = 0
        
    def add_data(self, data):
        """Add new data to the buffer
        
        Parameters:
        -----------
        data : array-like
            New EEG data samples
            
        Returns:
        --------
        new_data_range : tuple
            (start_idx, end_idx) of newly added data
        """
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Handle dimensionality
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        # Initialize buffer if needed
        if self.buffer is None:
            self.channels = data.shape[0]
            self.buffer = data
            return (0, data.shape[1])
        
        # Append new data
        self.buffer = np.hstack([self.buffer, data])
        
        # Trim buffer if it exceeds max size
        if self.buffer.shape[1] > self.max_size:
            trim_size = self.buffer.shape[1] - self.max_size
            self.buffer = self.buffer[:, trim_size:]
            self.last_processed_idx = max(0, self.last_processed_idx - trim_size)
            
        # Return range of new data
        return (self.buffer.shape[1] - data.shape[1], self.buffer.shape[1])
    
    def get_unprocessed_data(self):
        """Get data that hasn't been processed yet
        
        Returns:
        --------
        data : array-like
            Unprocessed data
        """
        if self.buffer is None or self.last_processed_idx >= self.buffer.shape[1]:
            return None
            
        data = self.buffer[:, self.last_processed_idx:]
        self.last_processed_idx = self.buffer.shape[1]
        return data
    
    def get_window(self, window_size=None):
        """Get the most recent window of data
        
        Parameters:
        -----------
        window_size : int or None
            Size of window to return, if None uses all available data
            
        Returns:
        --------
        data : array-like
            Windowed data
        """
        if self.buffer is None:
            return None
            
        if window_size is None or window_size >= self.buffer.shape[1]:
            return self.buffer
            
        return self.buffer[:, -window_size:]


# Calculate adaptive window size based on data characteristics
def calculate_adaptive_window(data, min_window=50, max_window=500):
    """Calculate adaptive window size based on signal characteristics
    
    Parameters:
    -----------
    data : array-like
        EEG data
    min_window : int
        Minimum window size
    max_window : int
        Maximum window size
        
    Returns:
    --------
    window_size : int
        Calculated window size
    """
    if data is None or data.size == 0:
        return min_window
        
    # Use the amplitude variance to determine stability
    variance = np.var(data)
    
    # Higher variance (less stable) = smaller window
    # Lower variance (more stable) = larger window
    if variance > 100:  # High variance
        return min_window
    elif variance < 10:  # Low variance
        return max_window
    else:
        # Scale linearly between min and max window
        norm_var = (variance - 10) / 90  # Normalize between 0 and 1
        window_size = max_window - norm_var * (max_window - min_window)
        return int(window_size)


# Initialize components
data_handler = DataHandler(buffer_size=1000)
model_manager = get_model_manager()


def _ensure_channels_samples(arr: np.ndarray) -> np.ndarray:
    """
    Normalize EEG chunk to (channels, samples).
    Accepts (channels, samples) or (samples, channels).
    """
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    # Heuristic: channels are usually <= 64
    if arr.shape[0] <= 64 and arr.shape[0] < arr.shape[1]:
        return arr
    if arr.shape[1] <= 64 and arr.shape[1] < arr.shape[0]:
        return arr.T
    # Default: assume (channels, samples)
    return arr

def process_streaming_chunk(
    data: np.ndarray, 
    model_type: str = "trained_model",
    clean_artifacts: bool = True, 
    stream_buffer=None,
    subject_id: str = "anonymous",
    session_id: str = "default_streaming"
) -> Dict[str, Any]:
    """
    Process a chunk of streaming EEG data (array-based).
    
    Args:
        data (np.ndarray): EEG data chunk (channels x samples)
        model_type (str): Architecture identifier to use
        clean_artifacts (bool): Whether to apply artifact cleaning
        stream_buffer (StreamBuffer): Buffer for statefulness
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        # 1. Update Buffer / Get Window
        current_data = data
        if stream_buffer:
            stream_buffer.add_data(data)
            # Use whole buffer or substantial window for context
            current_data = stream_buffer.get_window(window_size=None) 
            
        # 2. Preprocessing (Cleaning)
        current_data = _ensure_channels_samples(np.asarray(current_data))
        if clean_artifacts:
            # Standard preprocessing first (filters/notch/high/low-pass).
            processed_data = apply_eeg_preprocessing(current_data, fs=250, notch_freq=60)
            processed_data, _artifact_report = clean_eeg(processed_data, fs=250)
        else:
            processed_data = current_data
            
        # 3. Model Inference
        model_type = sanitize_model_type(model_type)
        model = model_manager.get_model(model_type, warmup=False)
        
        if model:
            # Prepare input shape
            # Model expects (batch, time, channels) or (batch, features)
            
            # Check model input shape
            input_shape = model.input_shape
            
            if len(input_shape) == 3 and input_shape[1] == 5 and input_shape[2] == 1:
                # Feature-based model (e.g., 5 frequency bands)
                try:
                    # processed_data is (channels, samples). extract_features expects (samples, channels) df
                    df = pd.DataFrame(processed_data.T)
                    df.columns = df.columns.astype(str)
                    
                    # Use simple_mode=True to get 5 bands
                    # Must use extract_features_from_timeseries directly for short chunks
                    features_df = extract_features_from_timeseries(
                        df, 
                        eeg_channels=list(df.columns), 
                        simple_mode=True
                    )
                    
                    # features_df is (n_epochs, 5). Reshape to (n_epochs, 5, 1)
                    X_input = features_df.values.reshape(-1, 5, 1)
                except Exception as e:
                    logger.error(f"Feature extraction failed in streaming: {e}")
                    # Fallback or re-raise
                    raise e
                    
            elif len(input_shape) == 3: # (None, time, channels) - RAW DATA MODEL
                # Reshape data to (1, time, channels)
                # Ensure correct channel count
                if processed_data.shape[1] != input_shape[2]:
                    # Mismatch channels
                    # Attempt resize or fail. For now, use dummy logic if mismatch
                    logger.warning(f"Shape mismatch: data {processed_data.shape} vs model {input_shape}")
                    # Try to use only required channels or pad
                    if processed_data.shape[1] > input_shape[2]:
                        processed_data = processed_data[:, :input_shape[2]]
                    else:
                        # Pad
                        pad = np.zeros((processed_data.shape[0], input_shape[2] - processed_data.shape[1]))
                        processed_data = np.hstack([processed_data, pad])
                
                # Ensure time dimension matches or is flexible? 
                # LSTM usually flexible on Time, but some models fixed.
                # enhanced_cnn_lstm usually flexible or fixed window.
                # Let's assume we pass the whole window as one sample
                # Model expects (batch, time, channels). We treat samples as time.
                X_input = processed_data.T.reshape(1, processed_data.shape[1], processed_data.shape[0])
                
            else: # Feature based
                # Fallback: band features from timeseries
                df = pd.DataFrame(processed_data.T)
                df.columns = df.columns.astype(str)
                features_df = extract_features_from_timeseries(df, eeg_channels=list(df.columns), simple_mode=True)
                X_input = features_df.values.reshape(-1, features_df.shape[1])
                
            # Predict
            try:
                probs = model.predict(X_input, verbose=0)
                if probs.ndim == 1:
                    probs = probs.reshape(1, -1)
                predicted_states = np.argmax(probs, axis=1).astype(int).tolist()
                dominant_state = int(np.bincount(predicted_states).argmax()) if predicted_states else 0
                confidence = float(np.max(np.mean(probs, axis=0))) if len(probs) else 0.0
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                dominant_state = 0
                confidence = 0.0
                predicted_states = []
                
        else:
            # No model, return dummy
            dominant_state = 0
            confidence = 0.0
            predicted_states = []
            
        result = {
            "predicted_states": predicted_states,
            "dominant_state": dominant_state,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": 0.0 # Will be calc by caller
        }
        
        # Step 4: Persistence (Async)
        try:
            safe_enqueue(
                "persistence",
                "src.jobs.persistence.store_session_summary",
                {
                    "type": "streaming_chunk",
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "dominant_state": dominant_state,
                    "confidence": confidence,
                    "timestamp": datetime.now(),
                },
            )
        except Exception as pe:
            logger.warning(f"Streaming persistence skipped: {pe}")
            
        return result
            
    except Exception as e:
        logger.error(f"Streaming processing error: {str(e)}")
        raise ValueError(f"Streaming processing failed: {str(e)}")
