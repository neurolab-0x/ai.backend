import os
from datetime import datetime
from fastapi import UploadFile, HTTPException
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

from src.utils.validation import validate_safe_id

# Constants
ALLOWED_EXTENSIONS = {'.edf', '.bdf', '.gdf', '.csv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_CONTENT_TYPES: Dict[str, Set[str]] = {
    ".edf": {"application/octet-stream", "application/edf", "application/x-edf"},
    ".bdf": {"application/octet-stream", "application/bdf", "application/x-bdf"},
    ".gdf": {"application/octet-stream", "application/gdf", "application/x-gdf"},
    ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel"},
}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/flac",
    "application/octet-stream",
}
MAX_AUDIO_FILE_SIZE = 25 * 1024 * 1024  # 25MB


def _safe_extension(filename: str) -> str:
    return os.path.splitext((filename or "").strip().lower())[1]


def _validate_content_type(file: UploadFile, extension: str, allowed: Dict[str, Set[str]]) -> None:
    content_type = (file.content_type or "").strip().lower()
    if not content_type:
        return
    allowed_types = allowed.get(extension, set())
    if allowed_types and content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}' for {extension} file",
        )

def validate_file(file: UploadFile):
    """Validate uploaded file parameters"""
    extension = _safe_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    _validate_content_type(file, extension, ALLOWED_CONTENT_TYPES)
    # UploadFile doesn't reliably expose size across servers; enforce limit while saving.

async def save_uploaded_file(file: UploadFile, user_id: str = "anonymous") -> str:
    """Save uploaded file with timestamp prefixing"""
    try:
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_user_id = validate_safe_id(user_id, "user_id")
        safe_filename = os.path.basename(file.filename or "upload.bin")
        safe_filename = ''.join(c for c in safe_filename if c.isalnum() or c in '._-')[:128]
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        file_location = f"temp/{timestamp}_{safe_user_id}_{safe_filename}"
        
        with open(file_location, "wb") as f:
            total_bytes = 0
            while content := await file.read(1024 * 1024):
                total_bytes += len(content)
                if total_bytes > MAX_FILE_SIZE:
                    f.close()
                    try:
                        os.remove(file_location)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds size limit of {MAX_FILE_SIZE//1024//1024}MB"
                    )
                f.write(content)
                
        if os.path.getsize(file_location) == 0:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        return file_location
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File handling failure: {str(e)}")
        raise HTTPException(500, "File processing error") from e 


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file parameters."""
    extension = _safe_extension(file.filename)
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio file type. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
    _validate_content_type(
        file,
        extension,
        {ext: ALLOWED_AUDIO_CONTENT_TYPES for ext in ALLOWED_AUDIO_EXTENSIONS},
    )


async def read_validated_audio_bytes(
    file: UploadFile,
    max_size: int = MAX_AUDIO_FILE_SIZE,
) -> bytes:
    """Read an uploaded audio file with size enforcement."""
    validate_audio_file(file)
    audio_data = await file.read()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds size limit of {max_size // 1024 // 1024}MB",
        )
    return audio_data
