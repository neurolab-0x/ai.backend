from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging
from src.config.settings import SECURITY_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()

from src.services.model_manager import get_model_manager

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_manager = get_model_manager()
        health = model_manager.get_health_status()
        return {
            "status": health,
            "authentication": {
                "enabled": SECURITY_CONFIG['require_authentication'],
                "status": "enabled" if SECURITY_CONFIG['require_authentication'] else "disabled"
            },
            "diagnostics": {
                "tensorflow_available": model_manager.tensorflow_available,
                "models_loaded": health.get("models_loaded", []),
                "models_count": health.get("models_count", 0),
                "model_files": model_manager.list_model_files(),
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@router.get("/")
async def root():
    """API root with basic information"""
    return {
        "name": "NeuroLab Axon Prime - Cloud Server",
        "version": "2.0.1",
        "description": "API for EEG signal processing and mental state classification with NLP-based recommendations",
        "features": [
            "Real-time EEG analysis",
            "Mental state classification (calm, engaged, elevated stress)",
            "NLP-based personalized recommendations",
            "Cognitive metrics calculation",
            "Wellness scoring",
            "Detailed reporting with insights",
            "Voice emotion detection and analysis"
        ]
    }
