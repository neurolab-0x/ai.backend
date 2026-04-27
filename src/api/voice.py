"""
Voice Processing API Endpoints
Handles audio upload and emotion/mental state analysis
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from src.utils.files import read_validated_audio_bytes

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize voice processor (singleton)
voice_processor = None

def get_voice_processor():
    """Get or initialize voice processor"""
    global voice_processor
    if voice_processor is None:
        try:
            from src.services.voice import VoiceProcessor
            voice_processor = VoiceProcessor()
            logger.info("Voice processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize voice processor: {str(e)}")
            raise HTTPException(status_code=503, detail="Voice processor unavailable")
    return voice_processor


@router.post("/analyze", summary="Analyze audio for emotion and mental state")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    sample_rate: Optional[int] = Query(None, description="Audio sample rate if known")
):
    """
    Analyze uploaded audio file for emotion detection and mental state classification
    
    Returns:
    - emotion: Detected emotion (angry, calm, fear, happy, neutral, sad, surprise)
    - confidence: Prediction confidence (0-1)
    - mental_state: Mapped signal state (0=calm, 1=engaged, 2=elevated_stress)
    - emotion_probabilities: Probability distribution across all emotions
    - features: Extracted audio features
    """
    try:
        processor = get_voice_processor()
        
        # Read audio file
        audio_data = await read_validated_audio_bytes(file)
        
        # Process audio
        result = processor.process_audio(audio_data, sample_rate)
        
        return {
            "status": "success",
            "data": result,
            "filename": file.filename,
            "file_size": len(audio_data),
            "medical_disclaimer": "This output is non-diagnostic and should not replace professional medical evaluation.",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")


@router.post("/analyze-batch", summary="Analyze multiple audio segments")
async def analyze_audio_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files"),
    sample_rate: Optional[int] = Query(None, description="Audio sample rate if known")
):
    """
    Analyze multiple audio files and provide pattern analysis
    
    Returns individual results plus aggregate analysis including:
    - dominant_emotion: Most common emotion across segments
    - emotion_distribution: Count of each emotion
    - average_confidence: Mean confidence across predictions
    - average_mental_state: Mean mental state
    - state_variability: Standard deviation of mental states
    """
    try:
        processor = get_voice_processor()
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 files allowed")
        
        # Process each file
        results = []
        
        for file in files:
            try:
                audio_data = await read_validated_audio_bytes(file)
            except HTTPException as exc:
                logger.warning("Skipping invalid audio file %s: %s", file.filename, exc.detail)
                continue
            
            result = processor.process_audio(audio_data, sample_rate)
            results.append({
                "filename": file.filename,
                "result": result
            })
        
        # Perform pattern analysis
        if len(results) > 1:
            # Extract emotions and states for analysis
            emotions = [r['result']['emotion'] for r in results]
            states = [r['result']['mental_state'] for r in results]
            confidences = [r['result']['confidence'] for r in results]
            
            from collections import Counter
            import numpy as np
            
            emotion_counts = Counter(emotions)
            
            pattern_analysis = {
                'total_segments': len(results),
                'dominant_emotion': emotion_counts.most_common(1)[0][0] if emotion_counts else 'neutral',
                'emotion_distribution': dict(emotion_counts),
                'average_confidence': float(np.mean(confidences)),
                'average_mental_state': float(np.mean(states)),
                'state_variability': float(np.std(states)),
                'emotions': emotions,
                'mental_states': states
            }
        else:
            pattern_analysis = None
        
        return {
            "status": "success",
            "total_files": len(files),
            "processed_files": len(results),
            "results": results,
            "pattern_analysis": pattern_analysis,
            "medical_disclaimer": "This output is non-diagnostic and should not replace professional medical evaluation.",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.post("/analyze-raw", summary="Analyze raw audio data")
async def analyze_raw_audio(
    audio_data: Dict[str, Any] = Body(..., description="Raw audio data"),
    sample_rate: int = Body(16000, description="Audio sample rate")
):
    """
    Analyze raw audio data provided as base64 or bytes array
    
    Expected format:
    {
        "data": "base64_encoded_audio" or [byte_array],
        "format": "base64" or "bytes"
    }
    """
    try:
        processor = get_voice_processor()
        
        # Parse audio data
        if audio_data.get('format') == 'base64':
            import base64
            audio_bytes = base64.b64decode(audio_data['data'])
        elif audio_data.get('format') == 'bytes':
            audio_bytes = bytes(audio_data['data'])
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'base64' or 'bytes'")
        
        # Process audio
        result = processor.process_audio(audio_bytes, sample_rate)
        
        return {
            "status": "success",
            "data": result,
            "medical_disclaimer": "This output is non-diagnostic and should not replace professional medical evaluation.",
        }
        
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error analyzing raw audio")
        raise HTTPException(status_code=500, detail="Raw audio analysis failed")


@router.get("/health", summary="Check voice processor health")
async def voice_health_check():
    """Check if voice processor is initialized and ready"""
    try:
        processor = get_voice_processor()
        
        return {
            "status": "healthy",
            "model_loaded": processor.model is not None,
            "processor_loaded": processor.processor is not None,
            "device": processor.device,
            "sample_rate": processor.sample_rate,
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        logger.exception("Voice health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Voice processor unavailable",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/emotions", summary="Get supported emotions")
async def get_supported_emotions():
    """Get list of supported emotions and their mental state mappings"""
    processor = get_voice_processor()
    
    return {
        "emotions": list(processor.emotion_to_state.keys()),
        "emotion_to_state_mapping": processor.emotion_to_state,
        "mental_states": {
            0: "calm",
            1: "engaged",
            2: "elevated_stress"
        }
    }
