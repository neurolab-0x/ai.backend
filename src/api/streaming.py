import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional

import numpy as np
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse as StarletteStreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List

from fastapi.concurrency import run_in_threadpool

from src.api.realtime import process_streaming_chunk, StreamBuffer
from src.config.settings import REAL_TIME_CONFIG, SECURITY_CONFIG
from src.core.ml.model_types import sanitize_model_type

logger = logging.getLogger(__name__)

router = APIRouter()

# Per-client streaming buffers and SSE queues
client_buffers: Dict[str, StreamBuffer] = {}
client_result_queues: Dict[str, "asyncio.Queue[Dict[str, Any]]"] = {}
_clients_lock = asyncio.Lock()


def _validate_client_identifier(client_id: str) -> str:
    if not client_id:
        raise ValueError("client_id is required")
    if len(client_id) > 64 or not all(c.isalnum() or c in "_-." for c in client_id):
        raise ValueError("Invalid client_id format")
    return client_id

# Enhanced Pydantic models with validation
class EEGData(BaseModel):
    eeg_data: List[List[float]] = Field(..., description="EEG data samples")
    client_id: Optional[str] = Field(None, description="Client identifier")
    model_type: Optional[str] = Field(None, description="Model type to use for processing")
    clean_artifacts: bool = Field(True, description="Whether to clean artifacts")
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
        if v is None:
            return v
        return _validate_client_identifier(v)
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is not None:
            return sanitize_model_type(v)
        return v

class StreamingInferenceResponse(BaseModel):
    predicted_states: List[int]
    dominant_state: int
    confidence: float
    processing_time_ms: float
    timestamp: str
    interpretability: Optional[Dict[str, Any]] = None

@router.post("/", response_model=StreamingInferenceResponse)
async def stream_eeg_data(
    request: Request, 
    data: EEGData,
    client_id: Optional[str] = None
):
    """
    Stream EEG data for real-time processing.
    
    - Uses client-specific streaming buffers
    - Provides detailed error tracking
    - Can include interpretability data when requested
    """
    start_time = time.time()
    
    try:
        # Determine client identifier (from path, header, or request)
        client_identifier = data.client_id or client_id or (request.client.host if request.client else "unknown")
        client_identifier = _validate_client_identifier(client_identifier)
        
        # Log processing request
        logger.info(f"Processing request for client: {client_identifier}")
        
        # Get or create client-specific buffer
        async with _clients_lock:
            if client_identifier not in client_buffers:
                logger.info(f"Creating new stream buffer for client: {client_identifier}")
                client_buffers[client_identifier] = StreamBuffer()
            if client_identifier not in client_result_queues:
                client_result_queues[client_identifier] = asyncio.Queue(maxsize=100)
        
        # Convert data format with validation
        eeg_array = np.array(data.eeg_data)
        
        # Use streaming if enabled
        stream_buffer = client_buffers[client_identifier] if REAL_TIME_CONFIG['enable_streaming'] else None
            
        model_type = sanitize_model_type(data.model_type) if data.model_type else "trained_model"
        model_path = f"./model/{model_type}.h5"
        
        # Process the data with the optimized pipeline
        result = await run_in_threadpool(
            process_streaming_chunk,
            eeg_array,
            model_type,
            data.clean_artifacts,
            stream_buffer,
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add processing statistics
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        result['processing_time_ms'] = total_time_ms
        
        # Add interpretability data if requested
        if data.include_interpretability:
            try:
                # Load the model used for predictions
                from src.core.ml.model import load_calibrated_model
                from src.core.ml.interpretability import ModelInterpretability
                model = load_calibrated_model(model_path or "./model/trained_model.h5")
                
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
        
        # Publish to SSE listeners (best-effort)
        try:
            q = client_result_queues.get(client_identifier)
            if q:
                if q.full():
                    _ = q.get_nowait()
                q.put_nowait(result)
        except Exception as e:
            logger.warning(f"Failed to publish SSE update for client {client_identifier}: {str(e)}")

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

@router.post("/chunk", status_code=status.HTTP_202_ACCEPTED)
async def submit_streaming_chunk(
    request: Request,
    data: EEGData,
    client_id: Optional[str] = None,
):
    """
    Submit an EEG chunk for processing and publish results to the client's SSE stream.
    """
    client_identifier = data.client_id or client_id or (request.client.host if request.client else "unknown")
    try:
        client_identifier = _validate_client_identifier(client_identifier)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    async with _clients_lock:
        if client_identifier not in client_buffers:
            client_buffers[client_identifier] = StreamBuffer()
        if client_identifier not in client_result_queues:
            client_result_queues[client_identifier] = asyncio.Queue(maxsize=100)

    eeg_array = np.array(data.eeg_data)
    model_type = sanitize_model_type(data.model_type) if data.model_type else "trained_model"
    stream_buffer = client_buffers[client_identifier] if REAL_TIME_CONFIG["enable_streaming"] else None
    start_time = time.time()
    result = await run_in_threadpool(
        process_streaming_chunk,
        eeg_array,
        model_type,
        data.clean_artifacts,
        stream_buffer,
    )
    result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)

    try:
        q = client_result_queues.get(client_identifier)
        if q:
            if q.full():
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    logger.debug("Result queue was empty while attempting to drop oldest item for client '%s'.", client_identifier)
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                logger.warning("Result queue is full; dropping streaming result for client '%s'.", client_identifier)
    except Exception:
        logger.exception("Unexpected error while queuing streaming result for client '%s'.", client_identifier)

    return {"status": "queued", "client_id": client_identifier, "timestamp": result.get("timestamp")}


@router.get("/sse")
async def stream_results_sse(
    request: Request,
    client_id: str,
) -> StarletteStreamingResponse:
    """
    Server-Sent Events stream for real-time EEG results.

    Clients:
    1) `POST /api/v1/streaming/chunk` to submit data
    2) `GET  /api/v1/streaming/sse?client_id=...` to receive results
    """
    try:
        client_identifier = _validate_client_identifier(client_id)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    async with _clients_lock:
        if client_identifier not in client_result_queues:
            client_result_queues[client_identifier] = asyncio.Queue(maxsize=100)
        q = client_result_queues[client_identifier]

    async def event_gen() -> AsyncGenerator[bytes, None]:
        # Initial hello event
        yield f"event: ready\ndata: {json.dumps({'client_id': client_identifier})}\n\n".encode("utf-8")

        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=15.0)
                payload = json.dumps(item, default=str)
                yield f"event: eeg\ndata: {payload}\n\n".encode("utf-8")
            except asyncio.TimeoutError:
                # Keep-alive ping so proxies don't close idle connections.
                yield b"event: ping\ndata: {}\n\n"

    return StarletteStreamingResponse(event_gen(), media_type="text/event-stream")

@router.post("/clear")
async def clear_stream_buffer(
    request: Request, 
    client_id: Optional[str] = None
):
    """Clear client stream buffer"""
    try:
        # Determine client identifier
        client_identifier = client_id or (request.client.host if request.client else "unknown")
        client_identifier = _validate_client_identifier(client_identifier)
        
        # Log action
        logger.info(f"Clearing buffer for client {client_identifier}")
        
        async with _clients_lock:
            if client_identifier in client_buffers:
                del client_buffers[client_identifier]
            if client_identifier in client_result_queues:
                del client_result_queues[client_identifier]
            return {"status": "success", "message": f"Buffer cleared for client {client_identifier}"}
        return {"status": "success", "message": "No buffer found for client"}
    except Exception as e:
        logger.error(f"Error clearing stream buffer: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 
