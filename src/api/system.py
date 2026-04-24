from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging
from src.config.settings import SECURITY_CONFIG

logger = logging.getLogger(__name__)
router = APIRouter()

_model_manager = None

def get_model_manager():
    global _model_manager
    if _model_manager is None:
        from src.services.model_manager import ModelManager
        _model_manager = ModelManager()
    return _model_manager

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_manager = get_model_manager()
        return {
            "status": model_manager.get_health_status(),
            "authentication": {
                "enabled": SECURITY_CONFIG['require_authentication'],
                "status": "enabled" if SECURITY_CONFIG['require_authentication'] else "disabled"
            },
            "diagnostics": {
                "model_loaded": model_manager.model is not None,
                "tensorflow_available": model_manager.tensorflow_available
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
            "Mental state classification (relaxed, focused, stressed)",
            "NLP-based personalized recommendations",
            "Cognitive metrics calculation",
            "Wellness scoring",
            "Detailed reporting with insights",
            "Voice emotion detection and analysis"
        ]
    }
