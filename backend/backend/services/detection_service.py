"""
Detection service: Business logic for detection operations.
Handles registry management, concurrency control, and process lifecycle.
"""

import logging
from multiprocessing import Process, Event, Queue, Manager

logger = logging.getLogger(__name__)


def get_active_detection_count(registry: dict) -> int:
    """
    Count the number of active (running) detections in the registry.
    
    Args:
        registry: In-memory detection registry
    
    Returns:
        Number of active detections
    """
    count = 0
    for entry in registry.values():
        if entry.get("status", {}).get("running", False):
            count += 1
    return count


def can_start_detection(registry: dict) -> bool:
    """
    Check if a new detection can be started (active_count < max_concurrent).
    
    Args:
        registry: In-memory detection registry
    
    Returns:
        True if new detection can be started, False if max reached
    """
    active_count = get_active_detection_count(registry)
    return active_count < 2


def register_detection(
    registry: dict,
    camera_id: int,
    process: Process,
    stop_event: Event,
    frame_queue: Queue,
    status_entry: dict,
    start_time
) -> None:
    """
    Register a new detection in the in-memory registry.
    
    Args:
        registry: In-memory detection registry
        camera_id: Camera ID
        process: Multiprocessing Process object
        stop_event: Multiprocessing Event for graceful stop signaling
        frame_queue: Multiprocessing Queue for frame delivery
        status_entry: Shared status dict entry for this camera
        start_time: Detection start time
    """
    registry[camera_id] = {
        "process": process,
        "stop_event": stop_event,
        "frame_queue": frame_queue,
        "status": status_entry,
        "start_time": start_time,
    }


def unregister_detection(registry: dict, camera_id: int) -> None:
    """
    Unregister a detection from the registry (called after detection stops).
    
    Args:
        registry: In-memory detection registry
        camera_id: Camera ID
    """
    if camera_id in registry:
        del registry[camera_id]
