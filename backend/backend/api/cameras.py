"""
Camera CRUD endpoints: GET /api/cameras, POST /api/cameras, DELETE /api/cameras/{id}

These endpoints manage camera registration and provide status information by querying
the in-memory registry to determine if detections are currently running.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from schemas import CameraCreate, CameraResponse
from services.camera_service import create_camera, list_cameras, get_camera, delete_camera
import main  # For access to registry and status_dict

logger = logging.getLogger(__name__)

router = APIRouter()


def get_camera_status(camera_id: int, registry: dict, status_dict: dict) -> str:
    """
    Determine camera status based on in-memory registry.
    
    Args:
        camera_id: Camera ID
        registry: In-memory registry of active detections
        status_dict: In-memory status dictionary
    
    Returns:
        Status string: "idle", "detecting", or "error"
    """
    if camera_id not in registry:
        return "idle"
    
    status_entry = registry[camera_id].get("status", {})
    if status_entry.get("running", False):
        return "detecting"
    
    if status_entry.get("error"):
        return "error"
    
    return "idle"


@router.get("/cameras", response_model=list[CameraResponse])
async def list_all_cameras(db: Session = Depends(get_db)):
    """
    List all registered cameras with current status.
    
    For each camera, lookup in the in-memory registry to determine status:
    - "idle": Camera is registered but not currently detecting
    - "detecting": Detection is currently running on this camera
    - "error": Last detection on this camera failed
    
    Returns:
        List of cameras with id, name, rtsp_url, status, created_at
    """
    try:
        # Query all cameras from database
        cameras = list_cameras(db)
        
        # Get registry and status_dict from main module
        registry = main.registry or {}
        
        # Build response list with status from registry
        response = []
        for camera in cameras:
            status_str = get_camera_status(camera.id, registry, main.status_dict or {})
            response.append(CameraResponse(
                id=camera.id,
                name=camera.name,
                rtsp_url=camera.rtsp_url,
                status=status_str,
                created_at=camera.created_at,
            ))
        
        return response
    
    except Exception as e:
        logger.error(f"Error listing cameras: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving cameras",
        )


@router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_new_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """
    Register a new camera.
    
    Validates:
    - name: non-empty, at most 255 characters
    - rtsp_url: valid RTSP URL format (starts with rtsp:// or rtsps://)
    
    Request body:
    {
        "name": "Factory Floor Camera 1",
        "rtsp_url": "rtsp://192.168.1.100:554/stream"
    }
    
    Returns:
        201 Created with new camera details
    """
    try:
        # Schema validation happens automatically via Pydantic
        # (validates name length and RTSP URL format)
        
        # Create camera in database
        created_camera = create_camera(db, camera.name, camera.rtsp_url)
        
        # Return with "idle" status (new camera is not yet detecting)
        return CameraResponse(
            id=created_camera.id,
            name=created_camera.name,
            rtsp_url=created_camera.rtsp_url,
            status="idle",
            created_at=created_camera.created_at,
        )
    
    except ValueError as e:
        logger.warning(f"Validation error creating camera: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    except Exception as e:
        logger.error(f"Error creating camera: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating camera",
        )


@router.delete("/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_camera(camera_id: int, db: Session = Depends(get_db)):
    """
    Delete a registered camera.
    
    Preconditions:
    - Camera must not have an active detection running
    
    Returns:
        204 No Content on success
        409 Conflict if detection is currently running
        404 Not Found if camera does not exist
    """
    try:
        # Check if camera exists
        camera = get_camera(db, camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera not found",
            )
        
        # Get registry from main module
        registry = main.registry or {}
        
        # Check if detection is currently running on this camera
        if camera_id in registry:
            status_entry = registry[camera_id].get("status", {})
            if status_entry.get("running", False):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Cannot delete camera: detection is currently running",
                )
        
        # Delete camera from database
        delete_camera(db, camera_id)
        
        # Return 204 No Content (no body)
        return None
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error deleting camera {camera_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting camera",
        )

