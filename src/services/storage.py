import logging
import os
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, Optional
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

    def download_file(self, bucket_key: str, object_name: str, destination_path: str) -> Optional[str]:
        """Download an object from MinIO to a local path."""
        if not self.enabled or not self.client:
            logger.warning("MinIO storage disabled or not initialized")
            return None

        bucket_name = MINIO_CONFIG['buckets'].get(bucket_key)
        if not bucket_name:
            logger.error(f"Invalid bucket key: {bucket_key}")
            return None

        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            self.client.fget_object(bucket_name, object_name, destination_path)
            logger.info(f"Downloaded {bucket_name}/{object_name} to {destination_path}")
            return destination_path
        except S3Error as e:
            logger.error(f"MinIO download error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading from MinIO: {e}")
            return None

    def download_artifact(self, descriptor: Dict[str, Any], destination_path: str) -> Optional[str]:
        """Download a persisted artifact descriptor to a local path."""
        bucket_key = descriptor.get("bucket_key") if isinstance(descriptor, dict) else None
        object_name = descriptor.get("object_name") if isinstance(descriptor, dict) else None
        if not bucket_key or not object_name:
            logger.error("Artifact descriptor missing bucket_key/object_name")
            return None
        return self.download_file(bucket_key, object_name, destination_path)

    def stat_file(self, bucket_key: str, object_name: str) -> Optional[Dict[str, Any]]:
        """Return object metadata for a stored file."""
        if not self.enabled or not self.client:
            return None

        bucket_name = MINIO_CONFIG['buckets'].get(bucket_key)
        if not bucket_name:
            logger.error(f"Invalid bucket key: {bucket_key}")
            return None

        try:
            stat = self.client.stat_object(bucket_name, object_name)
            return {
                "bucket_key": bucket_key,
                "bucket_name": bucket_name,
                "object_name": object_name,
                "etag": getattr(stat, "etag", None),
                "size": getattr(stat, "size", None),
                "last_modified": (
                    stat.last_modified.isoformat()
                    if getattr(stat, "last_modified", None) is not None
                    else None
                ),
            }
        except S3Error as e:
            logger.warning(f"MinIO stat error for {bucket_name}/{object_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error statting {bucket_name}/{object_name}: {e}")
            return None

    def object_exists(self, bucket_key: str, object_name: str) -> bool:
        """Check whether an object exists in MinIO."""
        return self.stat_file(bucket_key, object_name) is not None

    def build_artifact_descriptor(
        self,
        bucket_key: str,
        object_name: str,
        *,
        label: Optional[str] = None,
        kind: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        descriptor: Dict[str, Any] = {
            "bucket_key": bucket_key,
            "bucket_name": MINIO_CONFIG["buckets"].get(bucket_key),
            "object_name": object_name,
        }
        if label:
            descriptor["label"] = label
        if kind:
            descriptor["kind"] = kind
        if content_type:
            descriptor["content_type"] = content_type
        if metadata:
            descriptor["metadata"] = metadata
        return descriptor

    def hydrate_artifact_urls(self, value: Any, expiry_hours: int = 24) -> Any:
        """Attach fresh signed URLs to nested artifact descriptors."""
        if isinstance(value, dict):
            hydrated = {key: self.hydrate_artifact_urls(val, expiry_hours=expiry_hours) for key, val in value.items()}
            if "bucket_key" in hydrated and "object_name" in hydrated:
                hydrated["signed_url"] = self.get_file_url(
                    hydrated["bucket_key"],
                    hydrated["object_name"],
                    expiry_hours=expiry_hours,
                )
            return hydrated
        if isinstance(value, list):
            return [self.hydrate_artifact_urls(item, expiry_hours=expiry_hours) for item in value]
        return deepcopy(value) if isinstance(value, (tuple, set)) else value
