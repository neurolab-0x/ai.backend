import asyncio
from collections import Counter
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Body, Query, HTTPException
import logging
import base64

from src.utils.files import validate_file, save_uploaded_file
from src.core.ml.model_types import sanitize_model_type
from src.utils.validation import validate_safe_id

logger = logging.getLogger(__name__)
router = APIRouter()

_ml_processor = None

def get_ml_processor():
    global _ml_processor
    if _ml_processor is None:
        from src.services.analysis import MLProcessor
        _ml_processor = MLProcessor()
    return _ml_processor
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
        ml_processor = get_ml_processor()
        try:
            model_type = sanitize_model_type(model_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
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
            subject_id = validate_safe_id(str(json_data.get('subject_id', 'anonymous')), "subject_id")
            session_id = validate_safe_id(str(json_data.get('session_id', 'session_1')), "session_id")
            result = await ml_processor.process_eeg_data(
                json_data, 
                subject_id,
                session_id,
                model_type=model_type,
                overlap=overlap,
                simple_mode=simple_mode
            )
        else:
            raise HTTPException(status_code=400, detail="No file or data provided")
            
        if encrypt_response:
            result = base64.b64encode(str(result).encode()).decode()
            
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/analyze', summary="Analyze EEG data", response_description="Analysis results")
async def analyze_eeg_data(
    data: Dict[str, Any] = Body(..., description="EEG data to analyze"),
    model_type: str = Query(..., description="Architecture to use for analysis (required)"),
    overlap: float = Query(0.0, ge=0.0, le=0.9, description="Overlap between epochs"),
    simple_mode: bool = Query(True, description="Whether to use simplified feature extraction")
):
    """Analyze EEG data and return results"""
    try:
        ml_processor = get_ml_processor()
        try:
            model_type = sanitize_model_type(model_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        result = await ml_processor.process_eeg_data(
            data,
            subject_id=validate_safe_id(str(data.get('subject_id', 'anonymous')), "subject_id"),
            session_id=validate_safe_id(str(data.get('session_id', 'session_1')), "session_id"),
            model_type=model_type,
            overlap=overlap,
            simple_mode=simple_mode
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
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
        ml_processor = get_ml_processor()
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/decision-support', summary="Get decision support based on history")
async def get_decision_support(
    history: List[Dict[str, Any]] = Body(..., description="User analysis history")
):
    """Generate non-diagnostic history summaries from prior sessions."""
    try:
        if not history:
            return {
                "summary": {
                    "sessions_analyzed": 0,
                    "average_confidence": None,
                    "dominant_state_counts": {},
                },
                "observed_patterns": [],
                "supportive_recommendations": [],
                "medical_disclaimer": (
                    "This output is non-diagnostic and should not replace professional medical evaluation."
                ),
            }

        states = [str(item.get("dominant_state")) for item in history if item.get("dominant_state") is not None]
        confidences = [
            float(item.get("confidence"))
            for item in history
            if isinstance(item.get("confidence"), (int, float))
        ]
        state_counts = Counter(states)
        observed_patterns: List[str] = []
        recommendations: List[str] = []

        if state_counts:
            dominant_state, dominant_count = state_counts.most_common(1)[0]
            observed_patterns.append(
                f"Most recent sessions most often reported '{dominant_state}' ({dominant_count} of {len(history)} sessions)."
            )
        average_confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
        if average_confidence is not None:
            observed_patterns.append(f"Average model confidence across sessions was {average_confidence}%.")
            if average_confidence < 60:
                recommendations.append("Collect additional sessions before treating this pattern as stable.")
        if len(state_counts) > 1:
            recommendations.append("Review session context alongside these shifts before drawing conclusions.")
        if not recommendations:
            recommendations.append("Use these summaries as supportive wellness context, not as a diagnosis.")

        return {
            "summary": {
                "sessions_analyzed": len(history),
                "average_confidence": average_confidence,
                "dominant_state_counts": dict(state_counts),
            },
            "observed_patterns": observed_patterns,
            "supportive_recommendations": recommendations,
            "medical_disclaimer": (
                "This output is non-diagnostic and should not replace professional medical evaluation."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in decision support: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
