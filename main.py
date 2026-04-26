import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import logging
import uvicorn

from src.utils.logging_setup import configure_logging

# Import API routers
from src.api.training import router as training_router
from src.api.voice import router as voice_router
from src.api.analysis import router as analysis_router
from src.api.system import router as system_router
from src.api.model import router as model_mgmt_router

configure_logging()

logger = logging.getLogger("NeuroLab Axon Prime API")

try:
    from src.api.streaming import router as streaming_router
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logger.warning("Streaming endpoint not available")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for the application"""
    logger.info("Application starting up")
    grpc_server = None
    try:
        from src.grpc.chat_server import create_chat_grpc_server

        grpc_server = await create_chat_grpc_server()
        await grpc_server.start()
        logger.info("AI chat gRPC server started")
    except ImportError as exc:
        logger.warning(f"Chat gRPC server unavailable: {exc}")
    except Exception as exc:
        logger.warning(f"Failed to start chat gRPC server: {exc}")

    yield
    if grpc_server is not None:
        await grpc_server.stop(grace=5)
    logger.info("Application shutdown initiated")

API_PREFIX = os.getenv("API_PREFIX", "/api/v1").rstrip("/")

app = FastAPI(
    title="NeuroLab Axon Prime - Cloud Server",
    description="API for EEG signal processing and mental state classification",
    version="2.0.1",
    lifespan=lifespan,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect convenience endpoints to the versioned docs.
@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url=app.docs_url)

@app.get("/openapi.json", include_in_schema=False)
async def openapi_redirect():
    return RedirectResponse(url=app.openapi_url)

# Versioned API routers (standardized)
app.include_router(system_router, prefix=API_PREFIX, tags=["System"])
app.include_router(analysis_router, prefix=f"{API_PREFIX}/eeg", tags=["EEG"])
app.include_router(training_router, prefix=f"{API_PREFIX}/training", tags=["Training"])
app.include_router(voice_router, prefix=f"{API_PREFIX}/voice", tags=["Voice"])
app.include_router(model_mgmt_router, prefix=f"{API_PREFIX}/models", tags=["Models"])

if STREAMING_AVAILABLE:
    app.include_router(streaming_router, prefix=f"{API_PREFIX}/streaming", tags=["Streaming"])

# Legacy routes (backward-compatible, hidden from Swagger)
app.include_router(system_router, include_in_schema=False)
app.include_router(analysis_router, include_in_schema=False)
app.include_router(training_router, prefix="/api/training", include_in_schema=False)
app.include_router(voice_router, prefix="/api/voice", include_in_schema=False)
app.include_router(model_mgmt_router, prefix="/api/model", include_in_schema=False)
if STREAMING_AVAILABLE:
    app.include_router(streaming_router, prefix="/api/streaming", include_in_schema=False)

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
