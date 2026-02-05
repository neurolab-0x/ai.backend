import logging
import os
from minio import Minio
from minio.error import S3Error
from src.config.database import MINIO_CONFIG, ENABLE_DATABASES

logger = logging.getLogger(__name__)

class MinioStorageService:
    def __init__(self):
        self.enabled = ENABLE_DATABASES
        self.client = None
        if self.enabled:
            try:
                self.client = Minio(
                    MINIO_CONFIG['endpoint'],
                    access_key=MINIO_CONFIG['access_key'],
                    secret_key=MINIO_CONFIG['secret_key'],
                    secure=MINIO_CONFIG['secure']
                )
                self._ensure_buckets()
            except Exception as e:
                logger.error(f"Failed to initialize MinIO client: {e}")
                self.enabled = False

    def _ensure_buckets(self):
        """Ensure configured buckets exist"""
        if not self.client:
            return

        for bucket_name in MINIO_CONFIG['buckets'].values():
            try:
                if not self.client.bucket_exists(bucket_name):
                    self.client.make_bucket(bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
            except Exception as e:
                logger.error(f"Error checking/creating bucket {bucket_name}: {e}")

    def upload_file(self, file_path: str, bucket_key: str, object_name: str = None) -> str:
        """
        Upload a file to MinIO
        
        Args:
            file_path: Local path to file
            bucket_key: Key in MINIO_CONFIG['buckets'] (e.g., 'training', 'models')
            object_name: Optional name for object in bucket (defaults to filename)
            
        Returns:
            str: Object name if successful, None otherwise
        """
        if not self.enabled or not self.client:
            logger.warning("MinIO storage disabled or not initialized")
            return None

        bucket_name = MINIO_CONFIG['buckets'].get(bucket_key)
        if not bucket_name:
            logger.error(f"Invalid bucket key: {bucket_key}")
            return None

        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                file_path,
            )
            logger.info(f"Uploaded {file_path} to {bucket_name}/{object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"MinIO upload error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to MinIO: {e}")
            return None

    def get_file_url(self, bucket_key: str, object_name: str, expiry_hours: int = 24) -> str:
        """Get presigned URL for a file"""
        if not self.enabled or not self.client:
            return None
            
        bucket_name = MINIO_CONFIG['buckets'].get(bucket_key)
        if not bucket_name:
            return None

        try:
            return self.client.get_presigned_url(
                "GET",
                bucket_name,
                object_name,
                expires=timedelta(hours=expiry_hours)
            )
        except Exception as e:
            logger.error(f"Error generating URL: {e}")
            return None
