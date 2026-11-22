"""
Camera service: Business logic for camera operations.
Handles camera creation, deletion, and status lookup.
"""

import logging
from sqlalchemy.orm import Session
from models import Camera

logger = logging.getLogger(__name__)


def create_camera(db: Session, name: str, rtsp_url: str) -> Camera:
    """
    Create and persist a new camera.
    
    Args:
        db: Database session
        name: Camera name
        rtsp_url: RTSP stream URL
    
    Returns:
        Created Camera object
    """
    camera = Camera(name=name, rtsp_url=rtsp_url)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    logger.info(f"Created camera: id={camera.id}, name={camera.name}")
    return camera


def get_camera(db: Session, camera_id: int) -> Camera:
    """
    Retrieve a camera by ID.
    
    Args:
        db: Database session
        camera_id: Camera ID
    
    Returns:
        Camera object or None if not found
    """
    return db.query(Camera).filter(Camera.id == camera_id).first()


def list_cameras(db: Session) -> list:
    """
    List all cameras.
    
    Args:
        db: Database session
    
    Returns:
        List of Camera objects
    """
    return db.query(Camera).all()


def delete_camera(db: Session, camera_id: int) -> bool:
    """
    Delete a camera by ID.
    
    Args:
        db: Database session
        camera_id: Camera ID
    
    Returns:
        True if deleted, False if not found
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        return False
    
    db.delete(camera)
    db.commit()
    logger.info(f"Deleted camera: id={camera_id}")
    return True
