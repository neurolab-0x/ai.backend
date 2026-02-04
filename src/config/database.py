"""
Database configuration for the NeuroLab AI Model Server.
"""

import os

# Global switch for database interactions
# Set to False to disable all database operations (No-op mode)
ENABLE_DATABASES = os.getenv('ENABLE_DATABASES', 'false').lower() == 'true'

# Database Connection Settings (Mocks for development)
MONGODB_CONFIG = {
    'uri': os.getenv('MONGODB_URI', 'mongodb://localhost:27017'),
    'database': os.getenv('MONGODB_DB', 'neurolab_ai')
}

INFLUXDB_CONFIG = {
    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
    'token': os.getenv('INFLUXDB_TOKEN', 'test-token'),
    'org': os.getenv('INFLUXDB_ORG', 'neurolab'),
    'bucket': os.getenv('INFLUXDB_BUCKET', 'eeg_signals')
}
