import sys
import os
import logging
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock env vars to ensure we don't fail on missing real credentials
os.environ['MINIO_ENDPOINT'] = 'localhost:9000'
os.environ['INFLUXDB_URL'] = 'http://localhost:8086'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationVerify")

def verify_storage_service():
    print("\n--- Verifying MinioStorageService ---")
    try:
        from src.services.storage import MinioStorageService
        # Mock Minio client to avoid actual connection error during CI/Verification if service not up
        # Mocking context or pass
        pass 
             
        service = MinioStorageService()
        print(f"Service Initialized: {service.enabled}")
        
        if service.enabled:
            print("MinIO Client: Active")
            # Mock client interactions for safety
            service.client = MagicMock()
            service.client.bucket_exists.return_value = True
            
            res = service.upload_file("README.md", "models", "test/readme.md")
            print(f"Mock Upload Result: {res}")
        else:
            print("MinIO Service: Disabled (Expected if server not running)")
            
    except ImportError as e:
        print(f"FAILED to import storage service: {e}")
    except Exception as e:
        print(f"FAILED with error: {e}")

def verify_training_monitor():
    print("\n--- Verifying TrainingMonitor ---")
    try:
        from src.services.training_monitor import TrainingMonitor
        
        monitor = TrainingMonitor()
        print(f"Monitor Initialized: {monitor.enabled}")
        
        if monitor.enabled:
             print("InfluxDB Client: Active")
             # Mock write api
             monitor.write_api = MagicMock()
             monitor.log_training_event("test_run", "START", {"epochs": 10})
             print("Mock Log Event: Success")
        else:
            print("InfluxDB Service: Disabled (Expected if server not running)")

    except ImportError as e:
        print(f"FAILED to import monitor service: {e}")
    except Exception as e:
        print(f"FAILED with error: {e}")

if __name__ == "__main__":
    verify_storage_service()
    verify_training_monitor()
