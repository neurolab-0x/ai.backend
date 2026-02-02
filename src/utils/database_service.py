from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from motor.motor_asyncio import AsyncIOMotorClient
from src.config.database import INFLUXDB_CONFIG, MONGODB_CONFIG, SCHEMA_VERSIONS, ENABLE_DATABASES

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service layer for database operations"""
    
    def __init__(self):
        """Initialize database connections"""
        self.enabled = ENABLE_DATABASES
        self.influx_client = None
        self.write_api = None
        self.query_api = None
        self.mongo_client = None
        self.db = None
        
        if self.enabled:
            try:
                self.influx_client = InfluxDBClient(
                    url=INFLUXDB_CONFIG['url'],
                    token=INFLUXDB_CONFIG['token'],
                    org=INFLUXDB_CONFIG['org']
                )
                self.write_api = self.influx_client.write_api(write_options=ASYNCHRONOUS)
                self.query_api = self.influx_client.query_api()
                
                # Initialize MongoDB client
                self.mongo_client = AsyncIOMotorClient(MONGODB_CONFIG['url'])
                self.db = self.mongo_client[MONGODB_CONFIG['database']]
                logger.info("Database connections initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database connections: {str(e)}")
                self.enabled = False
        else:
            logger.info("Database interactions are disabled")
        
    async def store_eeg_data(self, data: Dict[str, Any], session_id: str) -> None:
        """Store EEG data in InfluxDB (No-op if disabled)"""
        if not self.enabled:
            return
            
        try:
            point = Point("eeg_measurements")\
                .tag("session_id", session_id)\
                .tag("schema_version", SCHEMA_VERSIONS['eeg_data'])\
                .time(datetime.utcnow())
                
            # Add all EEG channels as fields
            for channel, value in data['features'].items():
                point = point.field(channel, value)
                
            await self.write_api.write(
                bucket=INFLUXDB_CONFIG['bucket'],
                record=point
            )
        except Exception as e:
            logger.error(f"Error storing EEG data: {str(e)}")
            raise
            
    async def store_session_summary(self, session_data: Dict[str, Any]) -> str:
        """Store session summary in MongoDB (No-op if disabled)"""
        if not self.enabled:
            return "disabled"
            
        try:
            session_data['schema_version'] = SCHEMA_VERSIONS['session_data']
            session_data['created_at'] = datetime.utcnow()
            
            result = await self.db[MONGODB_CONFIG['collections']['sessions']].insert_one(session_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing session summary: {str(e)}")
            raise
            
    async def store_detected_event(self, event_data: Dict[str, Any]) -> str:
        """Store detected event in MongoDB (No-op if disabled)"""
        if not self.enabled:
            return "disabled"
            
        try:
            event_data['schema_version'] = SCHEMA_VERSIONS['event_data']
            event_data['detected_at'] = datetime.utcnow()
            
            result = await self.db[MONGODB_CONFIG['collections']['events']].insert_one(event_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing detected event: {str(e)}")
            raise
            
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data from MongoDB (Empty if disabled)"""
        if not self.enabled:
            return None
            
        try:
            return await self.db[MONGODB_CONFIG['collections']['sessions']].find_one(
                {"session_id": session_id}
            )
        except Exception as e:
            logger.error(f"Error retrieving session data: {str(e)}")
            raise
            
    async def get_eeg_data_range(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve EEG data from InfluxDB for a specific time range (Empty if disabled)"""
        if not self.enabled:
            return []
            
        try:
            query = f'''
                from(bucket: "{INFLUXDB_CONFIG['bucket']}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r["session_id"] == "{session_id}")
            '''
            
            result = self.query_api.query(query)
            return [record for table in result for record in table.records]
        except Exception as e:
            logger.error(f"Error retrieving EEG data: {str(e)}")
            raise
            
    async def store_model_version(self, model_data: Dict[str, Any]) -> str:
        """Store model version information in MongoDB (No-op if disabled)"""
        if not self.enabled:
            return "disabled"
            
        try:
            model_data['schema_version'] = SCHEMA_VERSIONS['model_data']
            model_data['created_at'] = datetime.utcnow()
            
            result = await self.db[MONGODB_CONFIG['collections']['models']].insert_one(model_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing model version: {str(e)}")
            raise
            
    async def close(self):
        """Close database connections"""
        if not self.enabled:
            return
            
        try:
            if self.influx_client:
                self.influx_client.close()
            if self.mongo_client:
                self.mongo_client.close()
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
            raise

# Create singleton instance
db_service = DatabaseService() 