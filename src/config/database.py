from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Global database toggle
ENABLE_DATABASES = os.getenv('ENABLE_DATABASES', 'false').lower() == 'true'

# InfluxDB Configuration
INFLUXDB_CONFIG = {
    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
    'token': os.getenv('INFLUXDB_TOKEN', ''),
    'org': os.getenv('INFLUXDB_ORG', 'neurolab'),
    'bucket': os.getenv('INFLUXDB_BUCKET', 'eeg_data'),
    'retention_policy': os.getenv('INFLUXDB_RETENTION', '30d'),
    'batch_size': 1000,
    'flush_interval': 10,  # seconds
}

# MongoDB Configuration
MONGODB_CONFIG = {
    'url': os.getenv('MONGODB_URL', 'mongodb://localhost:27017'),
    'database': os.getenv('MONGODB_DATABASE', 'neurolab'),
    'collections': {
        'sessions': 'eeg_sessions',
        'models': 'model_versions',
        'events': 'detected_events',
        'analytics': 'session_analytics'
    },
    'max_pool_size': 100,
    'min_pool_size': 10,
    'max_idle_time_ms': 30000,
    'wait_queue_timeout_ms': 10000,
}

# Database Schema Versions
SCHEMA_VERSIONS = {
    'eeg_data': '1.0',
    'session_data': '1.0',
    'event_data': '1.0',
    'model_data': '1.2'
}

# Data Retention Policies
RETENTION_POLICIES = {
    'raw_eeg_data': '30d',  # Raw EEG data retention
    'processed_data': '90d',  # Processed data retention
    'session_summaries': '365d',  # Session summaries retention
    'model_versions': '180d',  # Model version history retention
    'event_logs': '90d'  # Event logs retention
} 