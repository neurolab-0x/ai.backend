import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
from src.schemas.eeg import EEGDataPoint, EEGSession, EEGFeatures

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles EEG data processing and management"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffer: List[EEGDataPoint] = []
        self.current_session: Optional[EEGSession] = None
        logger.info(f"Initialized DataHandler with buffer size: {buffer_size}")
        
    def load_manual_data(self, file_path: str, subject_id: str, session_id: str) -> List[EEGDataPoint]:
        """Load data from a manually created CSV file"""
        logger.info(f"Loading manual data from {file_path} for subject {subject_id}, session {session_id}")
        data_points = []
        
        try:
            logger.debug("Reading CSV file")
            df = pd.read_csv(file_path)
            logger.debug(f"CSV columns: {list(df.columns)}")
            
            # Validate required columns
            required_columns = ['timestamp', 'alpha', 'beta', 'theta', 'delta', 'gamma']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert numeric columns to float, handling errors
            numeric_columns = ['alpha', 'beta', 'theta', 'delta', 'gamma', 'confidence']
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Fill NaN values with 0 for numeric columns
                        df[col] = df[col].fillna(0.0)
                    except Exception as e:
                        logger.warning(f"Error converting column {col} to numeric: {str(e)}")
                        df[col] = 0.0
            
            for index, row in df.iterrows():
                try:
                    logger.debug(f"Processing row {row.get('timestamp', index)}")
                    
                    # Parse timestamp
                    timestamp = row.get('timestamp')
                    if isinstance(timestamp, str):
                        try:
                            timestamp = pd.to_datetime(timestamp)
                        except:
                            timestamp = datetime.now()
                    elif pd.isna(timestamp):
                        timestamp = datetime.now()
                    
                    logger.debug(f"Parsed timestamp: {timestamp}")
                    
                    # Parse metadata
                    metadata = {}
                    metadata_str = row.get('metadata', '{}')
                    if isinstance(metadata_str, str):
                        try:
                            # Clean up the metadata string
                            metadata_str = metadata_str.strip()
                            if not metadata_str.startswith('{'):
                                metadata_str = '{' + metadata_str
                            if not metadata_str.endswith('}'):
                                metadata_str = metadata_str + '}'
                            metadata = json.loads(metadata_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing metadata: {str(e)}")
                            metadata = {}
                    
                    # Calculate signal quality
                    logger.debug("Calculating signal quality from frequency bands")
                    try:
                        signal_quality = self._calculate_signal_quality(
                            alpha=float(row.get('alpha', 0.0)),
                            beta=float(row.get('beta', 0.0)),
                            theta=float(row.get('theta', 0.0)),
                            delta=float(row.get('delta', 0.0)),
                            gamma=float(row.get('gamma', 0.0))
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error calculating signal quality: {str(e)}")
                        signal_quality = 0.0
                    
                    logger.debug(f"Calculated signal quality: {signal_quality}")
                    
                    # Create EEGFeatures
                    logger.debug("Creating EEGFeatures")
                    features = EEGFeatures(
                        alpha=float(row.get('alpha', 0.0)),
                        beta=float(row.get('beta', 0.0)),
                        theta=float(row.get('theta', 0.0)),
                        delta=float(row.get('delta', 0.0)),
                        gamma=float(row.get('gamma', 0.0))
                    )
                    
                    # Create EEGDataPoint
                    logger.debug("Creating EEGDataPoint")
                    data_point = EEGDataPoint(
                        timestamp=timestamp,
                        features=features,
                        state=int(row.get('state', 0)),
                        confidence=float(row.get('confidence', 0.0)),
                        metadata={
                            'signal_quality': signal_quality,
                            'device': metadata.get('device', 'unknown'),
                            'session_id': session_id,
                            'subject_id': subject_id
                        }
                    )
                    data_points.append(data_point)
                    
                except Exception as e:
                    logger.error(f"Error processing row {row.get('timestamp', index)}: {str(e)}")
                    continue
            
            if not data_points:
                raise ValueError("No valid data points could be processed from the file")
                
            return data_points
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def add_data_point(self, data_point: EEGDataPoint) -> None:
        """Add a new data point to the buffer"""
        self.data_buffer.append(data_point)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
    
    def get_latest_data(self, n_points: int = 100) -> List[EEGDataPoint]:
        """Get the most recent n data points"""
        return self.data_buffer[-n_points:]
    
    def clear_buffer(self) -> None:
        """Clear the data buffer"""
        self.data_buffer.clear()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session"""
        if not self.current_session:
            return {}
            
        data_points = self.current_session.data_points
        if not data_points:
            return {}
            
        states = [dp.state for dp in data_points]
        confidences = [dp.confidence for dp in data_points]
        signal_qualities = [dp.features.signal_quality for dp in data_points]
        
        return {
            "session_id": self.current_session.session_id,
            "subject_id": self.current_session.subject_id,
            "duration_seconds": (self.current_session.end_time - self.current_session.start_time).total_seconds(),
            "total_points": len(data_points),
            "state_distribution": {
                str(state): states.count(state) / len(states)
                for state in set(states)
            },
            "mean_confidence": np.mean(confidences) if confidences else 0.0,
            "mean_signal_quality": np.mean(signal_qualities) if signal_qualities else 0.0,
            "start_time": self.current_session.start_time.isoformat(),
            "end_time": self.current_session.end_time.isoformat()
        } 