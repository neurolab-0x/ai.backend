import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from src.config.database import INFLUXDB_CONFIG, ENABLE_DATABASES

logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self):
        self.enabled = ENABLE_DATABASES
        self.client = None
        self.write_api = None
        self.bucket = INFLUXDB_CONFIG['bucket']
        self.org = INFLUXDB_CONFIG['org']
        
        if self.enabled:
            try:
                self.client = InfluxDBClient(
                    url=INFLUXDB_CONFIG['url'],
                    token=INFLUXDB_CONFIG['token'],
                    org=self.org
                )
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB client: {e}")
                self.enabled = False

    def log_training_event(self, run_id: str, event_type: str, metadata: Dict[str, Any] = None):
        """Log a training lifecycle event (START, STOP, EPOCH_END)"""
        if not self.enabled or not self.write_api:
            return

        try:
            point = Point("training_events") \
                .tag("run_id", run_id) \
                .tag("event_type", event_type) \
                .field("value", 1)  # Marker field
                
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                         point = point.field(k, v)
                    else:
                         point = point.field(k, str(v))

            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as e:
            logger.error(f"Error logging training event: {e}")

    def log_metrics(self, run_id: str, step: int, metrics: Dict[str, float]):
        """Log training metrics (loss, accuracy, etc.)"""
        if not self.enabled or not self.write_api:
            return

        try:
            point = Point("training_metrics") \
                .tag("run_id", run_id) \
                .field("step", step)
            
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    point = point.field(k, v)
            
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def close(self):
        if self.client:
            self.client.close()
