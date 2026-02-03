import os
from datetime import datetime
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'.edf', '.bdf', '.gdf', '.csv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

def validate_file(file: UploadFile):
    """Validate uploaded file parameters"""
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds size limit of {MAX_FILE_SIZE//1024//1024}MB"
        )

async def save_uploaded_file(file: UploadFile, user_id: str = "anonymous") -> str:
    """Save uploaded file with timestamp prefixing"""
    try:
        os.makedirs("temp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        safe_filename = ''.join(c for c in file.filename if c.isalnum() or c in '._-')
        file_location = f"temp/{timestamp}_{user_id}_{safe_filename}"
        
        with open(file_location, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)
                
        if os.path.getsize(file_location) == 0:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        return file_location
    except Exception as e:
        logger.error(f"File handling failure: {str(e)}")
        raise HTTPException(500, "File processing error") from e 