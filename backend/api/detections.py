"""
Detection control endpoints:
  POST /api/cameras/{id}/start   – start detection worker for a camera
  POST /api/cameras/{id}/stop    – stop detection worker and save run record
  GET  /api/runs                 – list all completed runs (newest first)
  GET  /api/runs/{run_id}        – single run detail
"""

import json
import logging
from datetime import datetime
from multiprocessing import Process, Event, Queue

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import main
from config import (
    WEIGHTS_PATH,
    TRACKER_CONFIG,
    DETECTION_CONFIDENCE,
    MIN_HITS_MAP,
    MAX_CONCURRENT_DETECTIONS,
    DETECTION_STOP_TIMEOUT,
    FRAME_QUEUE_MAXSIZE,
)
from database import get_db
from models import Camera, Run
from schemas import StartDetectionResponse, StopDetectionResponse, RunResponse
from services.camera_service import get_camera
from services.detection_service import (
    can_start_detection,
    get_active_detection_count,
    register_detection,
    unregister_detection,
)
from worker import detection_worker

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_run_with_camera(db: Session, run_id: int):
    """Return (Run, Camera) or raise 404."""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    camera = db.query(Camera).filter(Camera.id == run.camera_id).first()
    return run, camera


def _run_to_response(run: Run, camera: Camera) -> RunResponse:
    counts = {}
    if run.counts_per_class:
        try:
            counts = json.loads(run.counts_per_class)
        except Exception:
            pass
    resource = None
    if run.resource_stats:
        try:
            resource = json.loads(run.resource_stats)
        except Exception:
            pass
    return RunResponse(
        id=run.id,
        camera_id=run.camera_id,
        camera_name=camera.name if camera else "Unknown",
        start_time=run.start_time,
        end_time=run.end_time,
        duration_seconds=run.duration_seconds,
        total_confirmed_count=run.total_confirmed_count,
        counts_per_class=counts,
        resource_stats=resource,
        status=run.status,
    )


# ---------------------------------------------------------------------------
# POST /api/cameras/{camera_id}/start
# ---------------------------------------------------------------------------

@router.post(
    "/cameras/{camera_id}/start",
    response_model=StartDetectionResponse,
    status_code=status.HTTP_200_OK,
)
async def start_detection(camera_id: int, db: Session = Depends(get_db)):
    """
    Start a detection worker process for the specified camera.

    Enforces:
    - Camera must exist
    - Camera must not already be detecting
    - Global concurrent detection limit (MAX_CONCURRENT_DETECTIONS=2)
    """
    # Verify camera exists
    camera = get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    registry = main.registry
    status_d = main.status_dict

    # Check if already running
    if camera_id in registry:
        entry = registry[camera_id]
        if entry.get("status", {}).get("running", False):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Detection is already running for this camera",
            )

    # Enforce concurrency limit
    if not can_start_detection(registry):
        active = get_active_detection_count(registry)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Max {MAX_CONCURRENT_DETECTIONS} concurrent detections reached "
                   f"(currently {active} active)",
        )

    # Create IPC primitives via Manager so they are shareable across processes
    manager = main.manager
    stop_event = manager.Event()
    frame_queue = manager.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
    status_entry = manager.dict()

    # Spawn worker process
    proc = Process(
        target=detection_worker,
        args=(
            camera_id,
            camera.rtsp_url,
            WEIGHTS_PATH,
            TRACKER_CONFIG,
            DETECTION_CONFIDENCE,
            MIN_HITS_MAP,
            frame_queue,
            status_d,
            stop_event,
        ),
        daemon=True,
    )
    proc.start()

    register_detection(
        registry=registry,
        camera_id=camera_id,
        process=proc,
        stop_event=stop_event,
        frame_queue=frame_queue,
        status_entry=status_entry,
        start_time=datetime.utcnow(),
    )

    logger.info(f"Started detection worker for camera {camera_id} (pid={proc.pid})")
    return StartDetectionResponse(success=True, message="Detection started")


# ---------------------------------------------------------------------------
# POST /api/cameras/{camera_id}/stop
# ---------------------------------------------------------------------------

@router.post(
    "/cameras/{camera_id}/stop",
    response_model=StopDetectionResponse,
    status_code=status.HTTP_200_OK,
)
async def stop_detection(camera_id: int, db: Session = Depends(get_db)):
    """
    Stop the detection worker for a camera and persist a Run record.

    Returns run_id so the frontend can navigate to the results page.
    """
    camera = get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    registry = main.registry
    status_d = main.status_dict

    if camera_id not in registry:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active detection for this camera",
        )

    entry = registry[camera_id]
    start_time: datetime = entry.get("start_time", datetime.utcnow())
    stop_event = entry.get("stop_event")
    process: Process = entry.get("process")
    frame_queue = entry.get("frame_queue")

    # Signal worker to stop
    if stop_event:
        stop_event.set()

    # Wait for process to terminate
    if process and process.is_alive():
        process.join(timeout=DETECTION_STOP_TIMEOUT)
        if process.is_alive():
            logger.warning(f"Worker for camera {camera_id} did not stop gracefully; terminating")
            process.terminate()
            process.join(timeout=3)

    # Drain queue to release shared memory
    if frame_queue:
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
        except Exception:
            pass

    # Read final counts from status_dict
    final_status = status_d.get(camera_id, {})
    final_counts: dict = final_status.get("counts", {})
    had_error: bool = bool(final_status.get("error"))

    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    total_count = sum(final_counts.values())
    run_status = "errored" if had_error else "completed"

    # Persist Run record
    run = Run(
        camera_id=camera_id,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        total_confirmed_count=total_count,
        counts_per_class=json.dumps(final_counts),
        resource_stats=None,
        status=run_status,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Clean up registry and status_dict
    unregister_detection(registry, camera_id)
    try:
        del status_d[camera_id]
    except KeyError:
        pass

    logger.info(
        f"Stopped detection for camera {camera_id}. "
        f"Run id={run.id}, counts={final_counts}, duration={duration:.1f}s"
    )
    return StopDetectionResponse(run_id=run.id, message="Detection stopped")


# ---------------------------------------------------------------------------
# GET /api/runs
# ---------------------------------------------------------------------------

@router.get("/runs", response_model=list[RunResponse])
async def list_runs(db: Session = Depends(get_db)):
    """Return all completed runs ordered newest-first."""
    runs = db.query(Run).order_by(Run.start_time.desc()).all()
    result = []
    for run in runs:
        camera = db.query(Camera).filter(Camera.id == run.camera_id).first()
        result.append(_run_to_response(run, camera))
    return result


# ---------------------------------------------------------------------------
# GET /api/runs/{run_id}
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: int, db: Session = Depends(get_db)):
    """Return a single run by ID."""
    run, camera = _get_run_with_camera(db, run_id)
    return _run_to_response(run, camera)
