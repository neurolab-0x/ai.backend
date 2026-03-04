from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from src.config.database import ENABLE_DATABASES, INFLUXDB_CONFIG

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service layer for database operations.
    Handles retrieval of historical EEG data and metrics from InfluxDB.
    """
    
    def __init__(self):
        """Initialize database service using InfluxDB"""
        self.enabled = ENABLE_DATABASES
        self.client = None
        self.query_api = None
        self.bucket = INFLUXDB_CONFIG['bucket']
        self.org = INFLUXDB_CONFIG['org']
        
        if self.enabled:
            try:
                self.client = InfluxDBClient(
                    url=INFLUXDB_CONFIG['url'],
                    token=INFLUXDB_CONFIG['token'],
                    org=self.org
                )
                self.query_api = self.client.query_api()
                logger.info("DatabaseService (InfluxDB) initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB client in DatabaseService: {e}")
                self.enabled = False
        
    async def store_eeg_data(self, data: Dict[str, Any], session_id: str) -> None:
        """Storage currently handled via TrainingMonitor or specific ingestion logic"""
        pass
            
    async def store_session_summary(self, session_data: Dict[str, Any]) -> str:
        """Placeholder for session summary storage"""
        return "not_implemented"
            
    async def store_detected_event(self, event_data: Dict[str, Any]) -> str:
        """No-op storage of detected event"""
        return "disabled"
            
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metrics for a specific session from InfluxDB"""
        if not self.enabled or not self.query_api:
            return None
            
        try:
            query = f'from(bucket: "{self.bucket}") \
                |> range(start: -30d) \
                |> filter(fn: (r) => r["run_id"] == "{session_id}") \
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
            
            result = self.query_api.query(org=self.org, query=query)
            
            data = []
            for table in result:
                for record in table.records:
                    data.append(record.values)
            
            if not data:
                return None
                
            # Aggregate or return most recent for current context
            return data[-1] if data else None
        except Exception as e:
            logger.error(f"Error retrieving session data for {session_id}: {e}")
            return None
            
    async def get_user_history(self, subject_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve historical session summaries for a user"""
        if not self.enabled or not self.query_api:
            return []
            
        try:
            # Query for completed training runs or analysis events for this subject
            query = f'from(bucket: "{self.bucket}") \
                |> range(start: -90d) \
                |> filter(fn: (r) => r["_measurement"] == "training_events") \
                |> filter(fn: (r) => r["event_type"] == "COMPLETED") \
                |> filter(fn: (r) => r["subject_id"] == "{subject_id}") \
                |> limit(n: {limit})'
            
            result = self.query_api.query(org=self.org, query=query)
            
            history = []
            for table in result:
                for record in table.records:
                    history.append({
                        "time": record.get_time(),
                        "run_id": record.values.get("run_id"),
                        "accuracy": record.values.get("final_accuracy"),
                        "loss": record.values.get("final_loss")
                    })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving user history for {subject_id}: {e}")
            return []
 
    async def get_eeg_data_range(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve EEG metrics for a specific time range"""
        if not self.enabled or not self.query_api:
            return []
            
        try:
            query = f'from(bucket: "{self.bucket}") \
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z) \
                |> filter(fn: (r) => r["run_id"] == "{session_id}")'
            
            result = self.query_api.query(org=self.org, query=query)
            return [record.values for table in result for record in table.records]
        except Exception as e:
            logger.error(f"Error retrieving data range: {e}")
            return []
            
    async def store_model_version(self, model_data: Dict[str, Any]) -> str:
        """No-op storage of model version"""
        return "disabled"
            
    async def close(self):
        """Close InfluxDB client"""
        if self.client:
            self.client.close()
 
db_service = DatabaseService()