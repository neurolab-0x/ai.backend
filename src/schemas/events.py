from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class EventSeverity(Enum):
    """Enum for event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    """Enum for event types"""
    STRESS = "stress"
    RELAXATION = "relaxation"
    ATTENTION = "attention"
    ANOMALY = "anomaly"
    SIGNAL_QUALITY = "signal_quality"

@dataclass
class DetectedEvent:
    """Class representing a detected neurological event"""
    event_type: EventType
    confidence: float
    timestamp: datetime
    session_id: str
    subject_id: str
    features: Dict[str, float]
    severity: EventSeverity
    description: str
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectedEvent':
        """Create DetectedEvent from a dictionary"""
        return cls(
            event_type=EventType(data['event_type']),
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            session_id=data['session_id'],
            subject_id=data['subject_id'],
            features=data['features'],
            severity=EventSeverity(data['severity']),
            description=data['description'],
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'subject_id': self.subject_id,
            'features': self.features,
            'severity': self.severity.value,
            'description': self.description,
            'metadata': self.metadata
        }

@dataclass
class EventPattern:
    """Class representing an event detection pattern"""
    name: EventType
    threshold: float
    min_duration: float  # in seconds
    required_channels: List[str]
    weights: Dict[str, float]
    description_template: str

    @classmethod
    def default_patterns(cls) -> Dict[EventType, 'EventPattern']:
        """Get default event patterns"""
        return {
            EventType.STRESS: cls(
                name=EventType.STRESS,
                threshold=0.7,
                min_duration=5.0,
                required_channels=['beta', 'alpha'],
                weights={'beta': 0.4, 'alpha': 0.3},
                description_template="Detected stress pattern with beta activity at {beta:.2f}"
            ),
            EventType.RELAXATION: cls(
                name=EventType.RELAXATION,
                threshold=0.6,
                min_duration=10.0,
                required_channels=['theta', 'alpha'],
                weights={'theta': 0.3, 'alpha': 0.4},
                description_template="Detected relaxation pattern with alpha activity at {alpha:.2f}"
            ),
            EventType.ATTENTION: cls(
                name=EventType.ATTENTION,
                threshold=0.8,
                min_duration=3.0,
                required_channels=['beta', 'gamma'],
                weights={'beta': 0.4, 'gamma': 0.3},
                description_template="Detected attention pattern with beta/gamma activity at {beta:.2f}"
            )
        } 