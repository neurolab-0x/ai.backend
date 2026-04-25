from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from motor.motor_asyncio import AsyncIOMotorClient

from src.config.database import ENABLE_DATABASES, INFLUXDB_CONFIG, MONGODB_CONFIG

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service layer for database operations.
    Handles dual-persistence: InfluxDB for time-series and MongoDB for metadata/artifacts.
    """
    
    def __init__(self):
        """Initialize database services"""
        self.enabled = ENABLE_DATABASES
        
        # InfluxDB attributes
        self.influx_client = None
        self.query_api = None
        self.write_api = None
        self.bucket = INFLUXDB_CONFIG['bucket']
        self.org = INFLUXDB_CONFIG['org']
        
        # MongoDB attributes
        self.mongo_client = None
        self.db = None
        
        if self.enabled:
            self._init_influx()
            self._init_mongo()
            
    def _init_influx(self):
        """Initialize InfluxDB connection"""
        try:
            self.influx_client = InfluxDBClient(
                url=INFLUXDB_CONFIG['url'],
                token=INFLUXDB_CONFIG['token'],
                org=self.org
            )
            self.query_api = self.influx_client.query_api()
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            logger.info("DatabaseService: InfluxDB initialized")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB: {e}")
            
    def _init_mongo(self):
        """Initialize MongoDB connection via Motor"""
        try:
            self.mongo_client = AsyncIOMotorClient(MONGODB_CONFIG['uri'])
            self.db = self.mongo_client[MONGODB_CONFIG['database']]
            logger.info(f"DatabaseService: MongoDB initialized on db '{MONGODB_CONFIG['database']}'")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")

    async def store_eeg_data(self, features: Dict[str, float], subject_id: str, session_id: str) -> bool:
        """Store processed EEG features to InfluxDB as time-series data"""
        if not self.enabled or not self.write_api:
            return False
            
        try:
            point = Point("eeg_metrics") \
                .tag("subject_id", subject_id) \
                .tag("session_id", session_id) \
                .time(datetime.now())
            
            for k, v in features.items():
                if isinstance(v, (int, float)):
                    point = point.field(k, float(v))
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            logger.error(f"Error storing EEG data to InfluxDB: {e}")
            return False
            
    async def store_session_summary(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Store a comprehensive session summary (multimodal metadata) to MongoDB"""
        if not self.enabled or self.db is None:
            return None
            
        try:
            # Ensure timestamp exists
            if 'timestamp' not in session_data:
                session_data['timestamp'] = datetime.now()
                
            result = await self.db.sessions.insert_one(session_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing session summary to MongoDB: {e}")
            return None
            
    async def store_voice_data(self, voice_results: Dict[str, Any], subject_id: str, session_id: str) -> Optional[str]:
        """Store voice analysis results to MongoDB"""
        if not self.enabled or self.db is None:
            return None
            
        try:
            record = {
                "subject_id": subject_id,
                "session_id": session_id,
                "timestamp": datetime.now(),
                "results": voice_results
            }
            result = await self.db.voice_analysis.insert_one(record)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing voice data to MongoDB: {e}")
            return None

    async def store_detected_event(self, event_data: Dict[str, Any]) -> str:
        """Store a specific detected neural event to MongoDB"""
        if not self.enabled or self.db is None:
            return "disabled"
            
        try:
            result = await self.db.events.insert_one(event_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing event: {e}")
            return "error"
            
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve aggregated metrics for a specific session"""
        if not self.enabled:
            return None
            
        # First check MongoDB for summary
        summary = await self.db.sessions.find_one({"session_id": session_id})
        if summary:
            return summary
            
        # Fallback to InfluxDB for raw metrics
        if self.query_api:
            try:
                query = f'from(bucket: "{self.bucket}") \
                    |> range(start: -30d) \
                    |> filter(fn: (r) => r["session_id"] == "{session_id}") \
                    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
                
                result = self.query_api.query(org=self.org, query=query)
                data = [record.values for table in result for record in table.records]
                return data[-1] if data else None
            except Exception as e:
                logger.error(f"Error retrieving session from InfluxDB: {e}")
        
        return None
            
    async def get_user_history(self, subject_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve historical session summaries for a user from both DBs"""
        if not self.enabled:
            return []
            
        history = []
        
        # 1. Fetch from MongoDB (Sessions)
        try:
            cursor = self.db.sessions.find({"subject_id": subject_id}).sort("timestamp", -1).limit(limit)
            async for doc in cursor:
                doc['_id'] = str(doc['_id']) # Make JSON serializable
                history.append({
                    "type": "session",
                    "time": doc.get("timestamp"),
                    "session_id": doc.get("session_id"),
                    "dominant_state": doc.get("dominant_state"),
                    "details": doc
                })
        except Exception as e:
            logger.error(f"Error fetching Mongo history: {e}")

        # 2. Fetch from InfluxDB (Training Events)
        if self.query_api:
            try:
                query = f'from(bucket: "{self.bucket}") \
                    |> range(start: -90d) \
                    |> filter(fn: (r) => r["_measurement"] == "training_events") \
                    |> filter(fn: (r) => r["event_type"] == "COMPLETED") \
                    |> filter(fn: (r) => r["subject_id"] == "{subject_id}") \
                    |> limit(n: {limit})'
                
                result = self.query_api.query(org=self.org, query=query)
                for table in result:
                    for record in table.records:
                        history.append({
                            "type": "training",
                            "time": record.get_time(),
                            "run_id": record.values.get("run_id"),
                            "accuracy": record.values.get("final_accuracy"),
                            "loss": record.values.get("final_loss")
                        })
            except Exception as e:
                logger.error(f"Error fetching InfluxDB history: {e}")
            
        # Sort combined history by time
        history.sort(key=lambda x: x['time'], reverse=True)
        return history[:limit]

    async def get_eeg_data_range(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve high-resolution EEG metrics from InfluxDB"""
        if not self.enabled or not self.query_api:
            return []
            
        try:
            query = f'from(bucket: "{self.bucket}") \
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z) \
                |> filter(fn: (r) => r["session_id"] == "{session_id}")'
            
            result = self.query_api.query(org=self.org, query=query)
            return [record.values for table in result for record in table.records]
        except Exception as e:
            logger.error(f"Error retrieving data range from InfluxDB: {e}")
            return []
            
    async def store_model_version(self, model_data: Dict[str, Any]) -> str:
        """Store model metadata in MongoDB"""
        if not self.enabled or self.db is None:
            return "disabled"
            
        try:
            result = await self.db.models.insert_one(model_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing model version: {e}")
            return "error"
            
    async def close(self):
        """Close both database clients"""
        if self.influx_client:
            self.influx_client.close()
        if self.mongo_client:
            self.mongo_client.close()
 
db_service = DatabaseService()
