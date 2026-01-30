import numpy as np
import logging
import time
from collections import deque
from threading import Lock
from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.features import extract_features
from src.utils.model_loading import load_calibrated_model
from src.utils.temporal_processing import temporal_smoothing
from src.utils.artifacts import clean_eeg
from src.utils.filters import apply_eeg_preprocessing
from src.config.settings import PROCESSING_CONFIG
from src.utils.data_handler import DataHandler, EEGDataPoint
from src.utils.explanation_generator import ExplanationGenerator
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Singleton model cache to prevent redundant model loading
class ModelCache:
    _instance = None
    _lock = Lock()
    _loaded_models = {}
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_path):
        """Get model from cache or load it if not available"""
        if model_path not in self._loaded_models:
            logger.info(f"Loading model from {model_path} (not in cache)")
            self._loaded_models[model_path] = load_calibrated_model(model_path)
        return self._loaded_models[model_path]
    
    def clear_cache(self):
        """Clear model cache"""
        self._loaded_models = {}


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
explanation_generator = ExplanationGenerator()

async def process_realtime_data(data: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Process real-time EEG data with immediate state classification and explanation.
    
    Args:
        data (Dict[str, Any]): Input data containing EEG features
        model: Loaded and calibrated model for inference
        
    Returns:
        Dict[str, Any]: Processing results including state classification and explanation
        
    Raises:
        ValueError: If data processing fails
    """
    try:
        # Extract features
        features = data.get('features', {})
        if not features:
            raise ValueError("No features provided in input data")
            
        # Create data point
        data_point = EEGDataPoint(
            timestamp=data.get('timestamp'),
            features=features,
            subject_id=data.get('subject_id'),
            session_id=data.get('session_id'),
            state=None,  # Will be determined by model
            confidence=None  # Will be determined by model
        )
        
        # Process data point
        explanation = await data_handler.process_data_point(
            data_point,
            explanation_generator
        )
        
        # Add to buffer for temporal analysis
        data_handler.buffer.append(data_point)
        
        return {
            "status": "success",
            "explanation": explanation,
            "timestamp": data_point.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Real-time processing error: {str(e)}")
        raise ValueError(f"Real-time processing failed: {str(e)}")

def validate_realtime_data(data: Dict[str, Any]) -> bool:
    """
    Validate real-time data format and content.
    
    Args:
        data (Dict[str, Any]): Input data to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        # Check required fields
        required_fields = ['features', 'subject_id', 'session_id']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
                
        # Validate features
        features = data['features']
        if not isinstance(features, dict):
            logger.error("Features must be a dictionary")
            return False
            
        # Check for minimum required channels
        required_channels = ['channel_1', 'channel_2', 'channel_3']
        for channel in required_channels:
            if channel not in features:
                logger.error(f"Missing required channel: {channel}")
                return False
                
        # Validate feature values
        for channel, value in features.items():
            if not isinstance(value, (int, float)):
                logger.error(f"Invalid feature value type for {channel}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        return False

def get_buffer_statistics() -> Dict[str, Any]:
    """
    Get statistics about the current data buffer.
    
    Returns:
        Dict[str, Any]: Buffer statistics including size and temporal information
    """
    try:
        buffer_data = data_handler.get_buffer_data()
        
        if not buffer_data:
            return {
                "buffer_size": 0,
                "time_span": 0,
                "data_points": 0
            }
            
        # Calculate time span
        timestamps = [dp.timestamp for dp in buffer_data]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        
        return {
            "buffer_size": len(buffer_data),
            "time_span": time_span,
            "data_points": len(buffer_data),
            "start_time": min(timestamps).isoformat(),
            "end_time": max(timestamps).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Buffer statistics error: {str(e)}")
        return {
            "error": str(e),
            "buffer_size": 0,
            "time_span": 0,
            "data_points": 0
        }

# Create a default stream buffer for application-wide use
default_stream_buffer = StreamBuffer(max_size=PROCESSING_CONFIG.get('max_buffer_size', 5000))
