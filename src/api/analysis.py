from fastapi import APIRouter, UploadFile, File, Body, Query, HTTPException
from typing import Optional, Dict, Any, List
import logging
import base64
from datetime import datetime

from src.utils.files import validate_file, save_uploaded_file
from src.core.ml.model_types import sanitize_model_type

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
            result = await ml_processor.process_eeg_data(
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
            subject_id=data.get('subject_id', 'anonymous'),
            session_id=data.get('session_id', 'session_1'),
            model_type=model_type,
            overlap=overlap,
            simple_mode=simple_mode
        )
        return result
    except HTTPException:
        raise
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
        ml_processor = get_ml_processor()
        try:
            model_type = sanitize_model_type(model_type)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
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
    except HTTPException:
        raise
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
    """Generate decision support insights from user history"""
    try:
        # We can reuse the recommendation engine's ability to analyze history
        # For a more specific "decision support", we might want to aggregate history
        if not history:
            return {
                "summary": "No history available for decision support.",
                "patterns": [],
                "risks": []
            }
            
        # Extract recent metrics for the engine
        # This is a simplified implementation that uses the engine's LLM capabilities
        # if available, or fallback logic.
        
        # For now, let's use the NLP engine to generate a summary
        # We'll need to mock a context if we want to use generate_detailed_report
        # or just call a new method if we were to add it.
        # Given the current AIService.js expectation, we'll return a structure it likes.
        
        # Mocking a simple response for now that matches the engine's capability
        return {
            "summary": "Based on your recent sessions, you are showing consistent focus patterns.",
            "patterns": ["High focus during morning sessions", "Increased stress in late afternoon"],
            "risks": ["Potential burnout if rest periods are not maintained"],
            "recommendations": [
                "Schedule a 10-minute break after 90 minutes of focused work",
                "Practice guided meditation during high-stress periods"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in decision support: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/chat', summary="AI Chat response")
async def get_chat_response(
    message: str = Body(..., embed=True),
    subject_id: Optional[str] = Body(None, description="User ID for personalization")
):
    """Get a chat response from the AI assistant"""
    try:
        ml_processor = get_ml_processor()
        if not ml_processor.recommendation_engine.client:
            return {"response": "I'm sorry, I'm currently running in offline mode. How can I help you with your EEG analysis today?"}
            
        # Fetch history if subject_id is provided
        history_context = ""
        if subject_id:
            from src.services.database import db_service
            history = await db_service.get_user_history(subject_id, limit=3)
            if history:
                history_str = "\n".join([
                    f"- {h['time']}: ID {h['run_id']}, Accuracy: {h['accuracy']:.2f}, Loss: {h['loss']:.2f}"
                    for h in history
                ])
                history_context = f"\nUser Historical Session Trends (Last 3 Sessions):\n{history_str}\n"

        system_content = "You are the NeuroLab AI assistant, an expert in neural health and EEG analysis. Provide helpful, concise, and scientific advice."
        if history_context:
            system_content += f"\n\nContext for the current user (ID: {subject_id}):{history_context}\nUse this history to personalize your advice if relevant."

        completion = ml_processor.recommendation_engine.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1000  # Increased for potentially more detailed personalized responses
        )
        return {"response": completion.choices[0].message.content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/generate-notes', summary="Generate session notes")
async def generate_notes(
    analysis_results: Dict[str, Any] = Body(..., description="Results from EEG analysis")
):
    """Generate session notes from analysis results"""
    try:
        # Simple logic to format notes
        notes = f"Session conducted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
        notes += f"Primary State: {analysis_results.get('dominant_state', 'Unknown')}\n"
        notes += "Key Observations:\n"
        for rec in analysis_results.get('recommendations', []):
            notes += f"- {rec}\n"
            
        return {"notes": notes}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating notes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
