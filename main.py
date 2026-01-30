from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Optional, Dict, Any, List
import base64
import pandas as pd

# Import utility modules
from src.utils.file_handler import validate_file, save_uploaded_file
from src.utils.model_manager import ModelManager
from src.utils.ml_processor import MLProcessor
from src.config.settings import SECURITY_CONFIG

# Import API routers
from src.api.training import router as training_router
from src.api.voice import router as voice_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurolab_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NeuroLab Axon Prime API")

try:
    from src.api.streaming import router as streaming_router
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logger.warning("Streaming endpoint not available")

logger.setLevel(logging.DEBUG)

# Initialize components
model_manager = ModelManager()
ml_processor = MLProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for the application"""
    yield
    logger.info("Application shutdown initiated")

app = FastAPI(
    title="NeuroLab Axon Prime - Cloud Server",
    description="API for EEG signal processing and mental state classification",
    version="2.0.1",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(training_router, tags=["Training"])
app.include_router(voice_router, tags=["Voice Analysis"])
if STREAMING_AVAILABLE:
    app.include_router(streaming_router, tags=["Streaming"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
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

@app.get("/")
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

@app.post('/upload', summary="Advanced EEG analysis", response_description="Cognitive state report", tags=["Analysis"])
async def process_uploaded_file(
    file: Optional[UploadFile] = File(None),
    json_data: Optional[Dict] = Body(None),
    encrypt_response: bool = Query(False, description="Whether to encrypt the response")
):
    """Process uploaded EEG file or JSON data"""
    try:
        if file:
            validate_file(file)
            file_location = await save_uploaded_file(file)
            result = ml_processor.process_eeg_data(file_location, "anonymous", "session_1")
        elif json_data:
            result = ml_processor.process_eeg_data(json_data, "anonymous", "session_1")
        else:
            raise HTTPException(status_code=400, detail="No file or data provided")
            
        if encrypt_response:
            result = base64.b64encode(str(result).encode()).decode()
            
        return result
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analyze', summary="Analyze EEG data", response_description="Analysis results", tags=["Analysis"])
async def analyze_eeg_data(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    background_tasks: BackgroundTasks = None
):
    """Analyze EEG data and return results"""
    try:
        result = ml_processor.process_eeg_data(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1')
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/calibrate', summary="Calibrate model", response_description="Calibration results", tags=["Model"])
async def calibrate_model(
    calibration_data: Dict[str, Any] = Body(..., description="Calibration data"),
    background_tasks: BackgroundTasks = None
):
    """Calibrate the model with new data"""
    try:
        if not model_manager.model:
            raise HTTPException(status_code=503, detail="Model not available")
            
        # Add calibration logic here
        return {"status": "calibration_started", "message": "Calibration process initiated"}
    except Exception as e:
        logger.error(f"Error calibrating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/detailed-report', summary="Generate detailed analysis report", response_description="Comprehensive report with recommendations", tags=["Analysis"])
async def generate_detailed_report(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    save_report: bool = Query(False, description="Whether to save the report to a file")
):
    """Generate a detailed analysis report with comprehensive recommendations"""
    try:
        report = ml_processor.generate_detailed_report(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1'),
            save_report=save_report
        )
        return report
    except Exception as e:
        logger.error(f"Error generating detailed report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/recommendations', summary="Get personalized recommendations", response_description="NLP-based recommendations", tags=["Analysis"])
async def get_recommendations(
    state_durations: Dict[int, float] = Body(..., description="State durations mapping"),
    total_duration: float = Body(..., description="Total session duration"),
    confidence: float = Body(..., description="Prediction confidence"),
    cognitive_metrics: Optional[Dict[str, float]] = Body(None, description="Cognitive metrics"),
    state_transitions: int = Body(0, description="Number of state transitions"),
    max_recommendations: int = Query(5, description="Maximum number of recommendations")
):
    """Get personalized recommendations based on EEG analysis"""
    try:
        from src.utils.nlp_recommendations import get_recommendations
        
        recommendations = get_recommendations(
            state_durations=state_durations,
            total_duration=total_duration,
            confidence=confidence,
            cognitive_metrics=cognitive_metrics,
            state_transitions=state_transitions,
            max_recommendations=max_recommendations
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
