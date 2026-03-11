from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

# Import API routers
from src.api.training import router as training_router
from src.api.voice import router as voice_router
from src.api.analysis import router as analysis_router
from src.api.system import router as system_router
from src.api.model import router as model_mgmt_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for the application"""
    logger.info("Application starting up")
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
app.include_router(system_router)
app.include_router(analysis_router, tags=["Analysis"])
app.include_router(training_router, prefix="/api/training", tags=["Training"])
app.include_router(voice_router, prefix="/api/voice", tags=["Voice Analysis"])
app.include_router(model_mgmt_router, prefix="/api/model", tags=["Model Management"])

if STREAMING_AVAILABLE:
    app.include_router(streaming_router, prefix="/api/streaming", tags=["Streaming"])

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
