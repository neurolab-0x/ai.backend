from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging
from src.config.database import ENABLE_DATABASES

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service layer for database operations.
    Refactored to be a no-op service to remove direct DB dependencies.
    """
    
    def __init__(self):
        """Initialize database service (No-op)"""
        self.enabled = ENABLE_DATABASES
        if not self.enabled:
            logger.info("Database interactions are globally disabled")
        else:
            # Implementation pending or moved to specific services (Storage/Monitor)
            pass
        
    async def store_eeg_data(self, data: Dict[str, Any], session_id: str) -> None:
        """No-op storage of EEG data"""
        return
            
    async def store_session_summary(self, session_data: Dict[str, Any]) -> str:
        """No-op storage of session summary"""
        return "disabled"
            
    async def store_detected_event(self, event_data: Dict[str, Any]) -> str:
        """No-op storage of detected event"""
        return "disabled"
            
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """No-op retrieval of session data"""
        return None
            
    async def get_eeg_data_range(
        self,
        session_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """No-op retrieval of EEG data range"""
        return []
            
    async def store_model_version(self, model_data: Dict[str, Any]) -> str:
        """No-op storage of model version"""
        return "disabled"
            
    async def close(self):
        """No-op close"""
        pass

# Create singleton instance
db_service = DatabaseService()