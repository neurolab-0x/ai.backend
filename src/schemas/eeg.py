from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np

@dataclass
class EEGChannelData:
    """Class representing raw EEG channel data"""
    channel_name: str
    sampling_rate: float
    data: np.ndarray
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EEGRecording:
    """Class representing a complete EEG recording with multiple channels"""
    channels: List[EEGChannelData]
    start_time: datetime
    end_time: datetime
    subject_id: str
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        """Get recording duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def channel_names(self) -> List[str]:
        """Get list of channel names"""
        return [ch.channel_name for ch in self.channels]

    def get_channel_data(self, channel_name: str) -> Optional[EEGChannelData]:
        """Get data for a specific channel"""
        for channel in self.channels:
            if channel.channel_name == channel_name:
                return channel
        return None

    def to_array(self) -> np.ndarray:
        """Convert all channel data to a numpy array (channels x timepoints)"""
        return np.array([ch.data for ch in self.channels])

@dataclass
class EEGDataPoint:
    """Base class for EEG data points"""
    timestamp: datetime
    features: Union[Dict[str, float], 'EEGFeatures']
    subject_id: str
    session_id: str
    state: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EEGSession:
    """Class representing an EEG recording session"""
    session_id: str
    subject_id: str
    start_time: datetime
    end_time: datetime
    data_points: List[EEGDataPoint]
    metadata: Dict[str, Any]
    duration: float = 0.0  # Default value moved to end

    @property
    def duration(self) -> float:
        """Get session duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()

@dataclass
class EEGFeatures:
    """Class representing extracted EEG features"""
    alpha: float
    beta: float
    theta: float
    gamma: float
    delta: float
    signal_quality: float = 1.0  # Default to perfect quality
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EEGFeatures':
        """Create EEGFeatures from a dictionary"""
        return cls(
            alpha=float(data.get('alpha', 0.0)),
            beta=float(data.get('beta', 0.0)),
            theta=float(data.get('theta', 0.0)),
            gamma=float(data.get('gamma', 0.0)),
            delta=float(data.get('delta', 0.0)),
            signal_quality=float(data.get('signal_quality', 1.0)),
            metadata=data.get('metadata')
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'theta': self.theta,
            'gamma': self.gamma,
            'delta': self.delta,
            'signal_quality': self.signal_quality,
            'metadata': self.metadata
        } 