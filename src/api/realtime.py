import numpy as np
import logging
import time
from collections import deque
from threading import Lock
from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.features import extract_features
from src.core.ml.model import load_calibrated_model
from src.core.processing.temporal import temporal_smoothing
from src.core.processing.artifacts import clean_eeg
from src.core.processing.filters import apply_eeg_preprocessing
from src.config.settings import PROCESSING_CONFIG
from src.services.data_service import DataHandler, EEGDataPoint
from src.services.recommendation import NLPRecommendationEngine
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
recommendation_engine = NLPRecommendationEngine()

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
            recommendation_engine
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

def process_streaming_chunk(
    data: np.ndarray, 
    model_path: str = None, 
    clean_artifacts: bool = True, 
    stream_buffer=None
) -> Dict[str, Any]:
    """
    Process a chunk of streaming EEG data (array-based).
    
    Args:
        data (np.ndarray): EEG data chunk (channels x samples)
        model_path (str): Path to model file
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
        if clean_artifacts:
            # clean_eeg expects (samples, channels) usuallly, or (channels, samples)
            # Let's check dimensions. Streaming usually sends (channels, samples) or (samples, channels)
            # We'll assume (samples, channels) for processing pipeline standard
            if current_data.shape[0] < current_data.shape[1] and current_data.shape[0] <= 64: 
                 # Likely (channels, samples), transpose
                 current_data = current_data.T
            
            # Simple artifact removal if needed (mock or real)
            # processed_data = clean_eeg(current_data)
            processed_data = current_data # Placeholder for robust cleaner
        else:
            processed_data = current_data
            
        # 3. Model Inference
        # Load model
        model = ModelCache().get_model(model_path) if model_path else None
        
        if model:
            # Prepare input shape
            # Model expects (batch, time, channels) or (batch, features)
            # If we are doing sequence classification, we need to segment
             
            # For simplicity in this fix, we'll treat the chunk/window as one batch if it fits
            # Or extract features
            
            # Check model input shape
            input_shape = model.input_shape
            
            if len(input_shape) == 3: # (None, time, channels)
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
                X_input = processed_data.reshape(1, processed_data.shape[0], processed_data.shape[1])
                
            else: # Feature based
                # Extract features
                # features = extract_features(processed_data, ...)
                # Placeholder: Flatten or mean
                X_input = np.mean(processed_data, axis=0).reshape(1, -1)
                
            # Predict
            try:
                probs = model.predict(X_input, verbose=0)[0]
                dominant_state = int(np.argmax(probs))
                confidence = float(np.max(probs))
                predicted_states = [dominant_state] # Simplify for chunk
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
            
        return {
            "predicted_states": predicted_states,
            "dominant_state": dominant_state,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": 0.0 # Will be calc by caller
        }
            
    except Exception as e:
        logger.error(f"Streaming processing error: {str(e)}")
        raise ValueError(f"Streaming processing failed: {str(e)}")

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
