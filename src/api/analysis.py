from fastapi import APIRouter, UploadFile, File, Body, Query, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any, List
import logging
import base64
from datetime import datetime

from src.utils.files import validate_file, save_uploaded_file
from src.services.analysis import MLProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize component
ml_processor = MLProcessor()

@router.post('/upload', summary="Advanced EEG analysis", response_description="Cognitive state report")
async def process_uploaded_file(
    file: Optional[UploadFile] = File(None),
    json_data: Optional[Dict] = Body(None),
    encrypt_response: bool = Query(False, description="Whether to encrypt the response"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction")
):
    """Process uploaded EEG file or JSON data"""
    try:
        if file:
            validate_file(file)
            file_location = await save_uploaded_file(file)
            result = await ml_processor.process_eeg_data(
                file_location, 
                "anonymous", 
                "session_1", 
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode
            )
        elif json_data:
            result = ml_processor.process_eeg_data(
                json_data, 
                "anonymous", 
                "session_1", 
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode
            )
        else:
            raise HTTPException(status_code=400, detail="No file or data provided")
            
        if encrypt_response:
            result = base64.b64encode(str(result).encode()).decode()
            
        return result
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/analyze', summary="Analyze EEG data", response_description="Analysis results")
async def analyze_eeg_data(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction"),
    background_tasks: BackgroundTasks = None
):
    """Analyze EEG data and return results"""
    try:
        result = await ml_processor.process_eeg_data(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1'),
            model_type=model_type,
            overlap=overlap,
            simple_mode=simple_mode
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/detailed-report', summary="Generate detailed analysis report", response_description="Comprehensive report with recommendations")
async def generate_detailed_report(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    save_report: bool = Query(False, description="Whether to save the report to a file"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction")
):
    """Generate a detailed analysis report with comprehensive recommendations"""
    try:
        report = await ml_processor.generate_detailed_report(
            data,
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1'),
            save_report=save_report,
            model_type=model_type,
            overlap=overlap,
            simple_mode=simple_mode
        )
        return report
    except Exception as e:
        logger.error(f"Error generating detailed report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/recommendations', summary="Get personalized recommendations", response_description="NLP-based recommendations")
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
        # Use the singleton recommendation engine from ml_processor for consistency
        recommendations = await ml_processor.recommendation_engine.generate_recommendations(
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
