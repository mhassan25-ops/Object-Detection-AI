"""
FastAPI application entry point for the RTSP Multi-Camera Detection Dashboard.
Initializes the app with CORS middleware, multiprocessing Manager for shared state,
and startup/shutdown event hooks.
"""

import multiprocessing as mp
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from database import init_db
from config import CORS_ORIGINS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared state (initialized in startup event)
manager: mp.Manager = None
registry: dict = None        # {camera_id: {process, stop_event, frame_queue, status}}
status_dict: dict = None     # {camera_id: {running, counts}}
video_status_dict: dict = None  # {job_id: {status, progress, frame_index, total_frames, counts, error}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup and shutdown events.
    Initializes multiprocessing Manager and database on startup.
    Cleans up worker processes on shutdown.
    """
    global manager, registry, status_dict, video_status_dict
    
    # Startup
    logger.info("Starting RTSP Multi-Camera Detection Dashboard...")
    try:
        # Initialize multiprocessing Manager
        manager = mp.Manager()
        registry = manager.dict()
        status_dict = manager.dict()
        video_status_dict = manager.dict()
        
        # Initialize database
        init_db()
        
        logger.info("Startup complete. Manager and database initialized.")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RTSP Multi-Camera Detection Dashboard...")
    try:
        # Gracefully terminate all running camera worker processes
        for camera_id, entry in list(registry.items()):
            process = entry.get("process")
            stop_event = entry.get("stop_event")
            
            if process and process.is_alive():
                logger.info(f"Terminating detection worker for camera {camera_id}...")
                if stop_event:
                    stop_event.set()
                process.join(timeout=5)
                if process.is_alive():
                    logger.warning(f"Force-terminating worker for camera {camera_id}...")
                    process.terminate()
                    process.join(timeout=2)

        # Gracefully terminate all running video job processes
        try:
            from api.video import _LOCAL_JOBS
            for job_id, entry in list(_LOCAL_JOBS.items()):
                process = entry.get("process")
                stop_event = entry.get("stop_event")
                if process and process.is_alive():
                    logger.info(f"Terminating video worker for job {job_id}...")
                    if stop_event:
                        stop_event.set()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
        except Exception:
            pass
        
        # Close manager
        if manager:
            manager.shutdown()
        
        logger.info("Shutdown complete.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI app with lifespan
app = FastAPI(
    title="RTSP Multi-Camera Detection Dashboard",
    description="Backend API for real-time object detection on multiple RTSP streams",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers (lazy import to avoid circular dependencies)
def include_routers():
    """Include all API routers after app initialization."""
    from api import cameras, detections, health, stats, websocket
    from api.video import router as video_router, ws_router as video_ws_router

    app.include_router(cameras.router, prefix="/api", tags=["cameras"])
    app.include_router(detections.router, prefix="/api", tags=["detections"])
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(stats.router, prefix="/api", tags=["stats"])
    app.include_router(websocket.router, tags=["websocket"])
    # Video: REST routes under /api, WebSocket routes at root (already has /ws/ prefix)
    app.include_router(video_router, prefix="/api", tags=["video"])
    app.include_router(video_ws_router, tags=["video-ws"])


# Include routers
include_routers()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "RTSP Multi-Camera Detection Dashboard",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
