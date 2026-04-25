from typing import Dict, List, Union, Optional, Generator, AsyncGenerator
import numpy as np
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass
import json
import os
import pandas as pd
from queue import Queue
import threading
import time
from src.services.database import db_service
from src.queue import safe_enqueue
from src.preprocessing.features import extract_features_from_timeseries

@dataclass
class EEGDataPoint:
    """Class to represent a single EEG data point."""
    timestamp: datetime
    features: Dict[str, float]
    subject_id: str
    session_id: str
    state: Optional[str] = None
    confidence: Optional[float] = None

class DataHandler:
    """Handles both manual and streaming EEG data inputs."""
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize the data handler.
        
        Parameters:
        -----------
        buffer_size : int
            Size of the buffer for streaming data
        """
        self.logger = logging.getLogger(__name__)
        self.buffer_size = buffer_size
        self.data_buffer = Queue(maxsize=buffer_size)
        self.is_streaming = False
        self.stream_thread = None
        
    def load_manual_data(self, 
                        file_path: str,
                        subject_id: str,
                        session_id: str) -> List[EEGDataPoint]:
        """
        Load EEG data from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file (CSV, JSON, or EDF)
        subject_id : str
            ID of the subject
        session_id : str
            ID of the session
            
        Returns:
        --------
        List[EEGDataPoint]
            List of EEG data points
        """
        try:
            data_points = []
            
            # Determine file type and load accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    data_point = EEGDataPoint(
                        timestamp=pd.to_datetime(row['timestamp']),
                        features=row.to_dict(),
                        subject_id=subject_id,
                        session_id=session_id
                    )
                    data_points.append(data_point)
                    
            elif file_ext == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        data_point = EEGDataPoint(
                            timestamp=datetime.fromisoformat(entry['timestamp']),
                            features=entry['features'],
                            subject_id=subject_id,
                            session_id=session_id
                        )
                        data_points.append(data_point)
                        
            elif file_ext == '.edf':
                # EDF loading via MNE; return one EEGDataPoint per extracted epoch.
                try:
                    import mne
                except ImportError as e:
                    raise RuntimeError("EDF support requires 'mne' to be installed") from e

                raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
                sfreq = float(raw.info.get("sfreq", 250.0))
                data = raw.get_data()  # (channels, samples)
                ch_names = [str(c) for c in raw.ch_names]

                # Convert to DataFrame for feature extraction (rows=timepoints).
                df = pd.DataFrame(data.T, columns=ch_names)
                features_df = extract_features_from_timeseries(df, eeg_channels=ch_names, simple_mode=True, overlap=0.0)

                # Assign approximate timestamps per epoch.
                epoch_len = 257
                step = epoch_len
                start_dt = raw.info.get("meas_date")
                if isinstance(start_dt, (tuple, list)):
                    start_dt = start_dt[0]
                if start_dt is None:
                    start_dt = datetime.now()
                start_dt = pd.to_datetime(start_dt).to_pydatetime()

                for epoch_idx, row in features_df.iterrows():
                    epoch_start_seconds = (epoch_idx * step) / sfreq
                    ts = start_dt + pd.to_timedelta(epoch_start_seconds, unit="s").to_pytimedelta()
                    data_points.append(
                        EEGDataPoint(
                            timestamp=ts,
                            features={k: float(v) for k, v in row.to_dict().items() if isinstance(v, (int, float, np.floating, np.integer))},
                            subject_id=subject_id,
                            session_id=session_id,
                        )
                    )
                
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            return data_points
            
        except Exception as e:
            self.logger.error(f"Error loading manual data: {str(e)}")
            raise
            
    async def start_streaming(self, 
                            stream_source: Union[str, Generator],
                            subject_id: str,
                            session_id: str) -> None:
        """
        Start streaming EEG data from a source.
        
        Parameters:
        -----------
        stream_source : Union[str, Generator]
            Source of the streaming data (URL or generator)
        subject_id : str
            ID of the subject
        session_id : str
            ID of the session
        """
        try:
            self.is_streaming = True
            self.stream_thread = threading.Thread(
                target=self._process_stream,
                args=(stream_source, subject_id, session_id)
            )
            self.stream_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error starting stream: {str(e)}")
            raise
            
    def _process_stream(self,
                       stream_source: Union[str, Generator],
                       subject_id: str,
                       session_id: str) -> None:
        """Process the streaming data in a separate thread."""
        try:
            if isinstance(stream_source, str):
                # URL-based streaming: expect newline-delimited JSON objects.
                from urllib.request import Request, urlopen

                req = Request(stream_source, headers={"Accept": "application/x-ndjson"})
                with urlopen(req, timeout=10) as resp:
                    for line in resp:
                        if not self.is_streaming:
                            break
                        if not line:
                            continue
                        try:
                            payload = json.loads(line.decode("utf-8").strip())
                        except Exception:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        data_point = EEGDataPoint(
                            timestamp=datetime.now(),
                            features=payload,
                            subject_id=subject_id,
                            session_id=session_id,
                        )
                        if self.data_buffer.full():
                            self.data_buffer.get()
                        self.data_buffer.put(data_point)
            else:
                # Handle generator-based streaming
                for data in stream_source:
                    if not self.is_streaming:
                        break
                        
                    data_point = EEGDataPoint(
                        timestamp=datetime.now(),
                        features=data,
                        subject_id=subject_id,
                        session_id=session_id
                    )
                    
                    # Add to buffer, remove oldest if full
                    if self.data_buffer.full():
                        self.data_buffer.get()
                    self.data_buffer.put(data_point)
                    
        except Exception as e:
            self.logger.error(f"Error processing stream: {str(e)}")
            self.is_streaming = False
            raise
            
    def stop_streaming(self) -> None:
        """Stop the streaming process."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
            
    def get_buffer_data(self) -> List[EEGDataPoint]:
        """
        Get all data points from the buffer.
        
        Returns:
        --------
        List[EEGDataPoint]
            List of buffered EEG data points
        """
        data_points = []
        while not self.data_buffer.empty():
            data_points.append(self.data_buffer.get())
        return data_points
        
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        while not self.data_buffer.empty():
            self.data_buffer.get()
            
    async def process_data_point(self, 
                               data_point: EEGDataPoint,
                               recommendation_engine) -> Dict:
        """
        Process a single data point and generate medical explanation.
        
        Parameters:
        -----------
        data_point : EEGDataPoint
            The EEG data point to process
        recommendation_engine : NLPRecommendationEngine
            The recommendation and explanation engine instance
            
        Returns:
        --------
        Dict
            Generated medical explanation for the data point
        """
        try:
            # Build context for the recommendation engine
            # For a single data point, we treat it as an instantaneous snapshot
            context = recommendation_engine._build_context(
                state_durations={0: 0, 1: 0, 2: 0},  # Initialized empty
                total_duration=1.0,
                confidence=data_point.confidence or 0.0,
                cognitive_metrics={},
                state_transitions=0,
                timestamp=data_point.timestamp,
                subject_id=data_point.subject_id,
                session_id=data_point.session_id,
                features=data_point.features
            )
            
            # Map state label back to dominant index for context consistency
            state_map = {"relaxed": 0, "focused": 1, "stressed": 2}
            context.state_label = data_point.state.lower() if data_point.state else "relaxed"
            
            # Generate medical explanation
            explanation = await recommendation_engine.generate_medical_explanation(context)
            
            # Step 3: Persistence (Async)
            try:
                safe_enqueue(
                    "persistence",
                    "src.jobs.persistence.store_eeg_data",
                    data_point.features,
                    data_point.subject_id,
                    data_point.session_id,
                )
                safe_enqueue(
                    "persistence",
                    "src.jobs.persistence.store_session_summary",
                    {
                        "type": "realtime_datapoint",
                        "subject_id": data_point.subject_id,
                        "session_id": data_point.session_id,
                        "timestamp": data_point.timestamp,
                        "state": context.state_label,
                        "confidence": data_point.confidence,
                        "explanation": explanation,
                    },
                )
            except Exception as pe:
                self.logger.warning(f"Real-time persistence skipped: {pe}")
                
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error processing data point: {str(e)}")
            raise
            
    def save_data(self,
                 data_points: List[EEGDataPoint],
                 file_path: str) -> None:
        """
        Save EEG data points to a file.
        
        Parameters:
        -----------
        data_points : List[EEGDataPoint]
            List of EEG data points to save
        file_path : str
            Path to save the data
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.DataFrame([
                    {
                        'timestamp': dp.timestamp,
                        'subject_id': dp.subject_id,
                        'session_id': dp.session_id,
                        'state': dp.state,
                        'confidence': dp.confidence,
                        **dp.features
                    }
                    for dp in data_points
                ])
                df.to_csv(file_path, index=False)
                
            elif file_ext == '.json':
                data = [
                    {
                        'timestamp': dp.timestamp.isoformat(),
                        'subject_id': dp.subject_id,
                        'session_id': dp.session_id,
                        'state': dp.state,
                        'confidence': dp.confidence,
                        'features': dp.features
                    }
                    for dp in data_points
                ]
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise 
