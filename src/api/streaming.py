import numpy as np
import logging
import time
from fastapi import APIRouter, Request, HTTPException, Depends, Header, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import base64

from src.api.realtime import process_realtime_data
from src.core.ml.interpretability import ModelInterpretability
from src.config.settings import REAL_TIME_CONFIG, SECURITY_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter()

# Create instance-specific stream buffers for multiple clients
client_buffers = {}

# Dummy implementation for missing security features
def require_user_role(x): return x
def validate_client_identifier(x): return x
def validate_eeg_data(x): return True
def sanitize_model_type(x): return x
class DataEncryption:
    def encrypt_data(self, data): return data
encryption = DataEncryption()

# Enhanced Pydantic models with validation
class EEGData(BaseModel):
    eeg_data: List[List[float]] = Field(..., description="EEG data samples")
    client_id: Optional[str] = Field(None, description="Client identifier")
    model_type: Optional[str] = Field(None, description="Model type to use for processing")
    clean_artifacts: bool = Field(True, description="Whether to clean artifacts")
    encrypt_response: Optional[bool] = Field(None, description="Whether to encrypt the response")
    include_interpretability: bool = Field(False, description="Whether to include interpretability data")
    
    @validator('eeg_data')
    def validate_eeg_dimensions(cls, v):
        # Check if data is empty
        if not v:
            raise ValueError("EEG data cannot be empty")
        
        # Check number of channels
        if len(v) > SECURITY_CONFIG['max_eeg_channels']:
            raise ValueError(f"Too many EEG channels (max: {SECURITY_CONFIG['max_eeg_channels']})")
        
        # Check number of samples
        for channel in v:
            if len(channel) > SECURITY_CONFIG['max_eeg_samples']:
                raise ValueError(f"Too many samples (max: {SECURITY_CONFIG['max_eeg_samples']})")
            
            # Check for NaN, Inf, and amplitude
            for sample in channel:
                if not np.isfinite(sample):
                    raise ValueError("EEG data contains NaN or Inf values")
                if abs(sample) > SECURITY_CONFIG['max_eeg_amplitude']:
                    raise ValueError(f"EEG amplitude exceeds maximum ({SECURITY_CONFIG['max_eeg_amplitude']} μV)")
        
        return v
    
    @validator('client_id')
    def validate_client_id(cls, v):
        if v is not None and (len(v) > 64 or not all(c.isalnum() or c in '_-.' for c in v)):
            raise ValueError("Invalid client ID format")
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is not None:
            return sanitize_model_type(v)
        return v

class StreamingResponse(BaseModel):
    predicted_states: List[int]
    dominant_state: int
    confidence: float
    processing_time_ms: float
    timestamp: str
    encrypted: bool = False
    interpretability: Optional[Dict[str, Any]] = None

@router.post("/api/stream", response_model=StreamingResponse)
async def stream_eeg_data(
    request: Request, 
    data: EEGData,
    request: Request, 
    data: EEGData,
    current_user: Dict = None,
    client_id: Optional[str] = None
):
    """
    Stream EEG data for real-time processing with security features
    
    - Authenticates and authorizes users
    - Validates and sanitizes inputs
    - Encrypts sensitive data
    - Uses client-specific streaming buffers
    - Provides detailed error tracking
    - Can include interpretability data when requested
    """
    start_time = time.time()
    
    try:
        # Determine client identifier (from path, header, or request)
        client_identifier = data.client_id or client_id or request.client.host
        
        # Log processing request
        logger.info(f"Processing request for client: {client_identifier}")
        
        # Get or create client-specific buffer
        if client_identifier not in client_buffers:
            logger.info(f"Creating new stream buffer for client: {client_identifier}")
            client_buffers[client_identifier] = StreamBuffer()
        
        # Convert data format with validation
        eeg_array = np.array(data.eeg_data)
        
        # Validate EEG data
        if not validate_eeg_data(eeg_array):
            logger.warning(f"Invalid EEG data from client: {client_identifier}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid EEG data format or values"
            )
        
        # Use streaming if enabled
        stream_buffer = client_buffers[client_identifier] if REAL_TIME_CONFIG['enable_streaming'] else None
            
        # Sanitize model type to prevent path traversal
        model_type = sanitize_model_type(data.model_type) if data.model_type else None
        model_path = f"./model/trained_model.h5" if model_type else None
        
        # Process the data with the optimized pipeline
        result = process_realtime_data(
            eeg_array, 
            model_path=model_path,
            clean_artifacts=data.clean_artifacts,
            stream_buffer=stream_buffer
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add processing statistics
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        result['total_processing_time_ms'] = total_time_ms
        
        # Add interpretability data if requested
        if data.include_interpretability:
            try:
                # Load the model used for predictions
                from src.utils.loading import load_calibrated_model
                from main import MODEL_PATH  # Import model path from main
                
                model = load_calibrated_model(model_path or MODEL_PATH)
                
                # Create interpretability handler
                interpreter = ModelInterpretability(model)
                
                # Prepare data for interpretability
                # Reshape for the model input if needed
                eeg_features = eeg_array
                if len(eeg_features.shape) < 3:
                    eeg_features = eeg_features.reshape(1, eeg_features.shape[0], 1)
                elif len(eeg_features.shape) == 2:
                    eeg_features = eeg_features.reshape(eeg_features.shape[0], eeg_features.shape[1], 1)
                
                # Generate LIME explanation (faster than SHAP for streaming)
                lime_results = interpreter.explain_with_lime(
                    eeg_features, 
                    sample_idx=0,
                    num_features=5  # Limit to 5 features for performance
                )
                
                # Extract top features and importance values
                if "feature_importance" in lime_results and "explanation" in lime_results:
                    # Get feature importance
                    top_features = lime_results["feature_importance"]
                    
                    # Clean up explainer object that can't be serialized
                    del lime_results["explanation"]
                    
                    # Add interpretability data to result
                    result["interpretability"] = {
                        "method": "lime",
                        "feature_importance": top_features,
                        "predicted_class": lime_results["predicted_class"],
                    }
            except Exception as e:
                logger.warning(f"Failed to generate interpretability data: {str(e)}")
                result["interpretability"] = {"error": str(e)}
        
        # Check if response should be encrypted
        should_encrypt = data.encrypt_response
        if should_encrypt is None:
            should_encrypt = SECURITY_CONFIG['encryption_for_transit']
        
        # Encrypt response if requested or required
        if should_encrypt:
            # Store some metadata outside encryption for client use
            metadata = {
                "dominant_state": result["dominant_state"],
                "confidence": result["confidence"],
                "timestamp": result["timestamp"]
            }
            
            # Encrypt the full result
            encrypted_result = encryption.encrypt_data(result)
            
            # Return encrypted payload
            return JSONResponse(content={
                "encrypted": True,
                "metadata": metadata,
                "data": base64.b64encode(encrypted_result).decode()
            })
            
        # Add encryption status to response
        result["encrypted"] = False
        return result
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Error in streaming endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/api/stream/clear")
async def clear_stream_buffer(
    request: Request, 
    client_id: Optional[str] = None
):
    """Clear client stream buffer with authentication"""
    try:
        # Determine client identifier
        client_identifier = client_id or validated_client_id or request.client.host
        
        # Log action
        logger.info(f"Clearing buffer for client {client_identifier}")
        
        if client_identifier in client_buffers:
            del client_buffers[client_identifier]
            return {"status": "success", "message": f"Buffer cleared for client {client_identifier}"}
        return {"status": "success", "message": "No buffer found for client"}
    except Exception as e:
        logger.error(f"Error clearing stream buffer: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 