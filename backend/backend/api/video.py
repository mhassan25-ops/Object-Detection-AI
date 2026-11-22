"""
Video upload and processing endpoints.

  POST /api/video/upload          – upload a video file, start detection job, return job_id
  GET  /api/video/{job_id}        – get current job status / progress / counts
  POST /api/video/{job_id}/cancel – cancel a running job
  WS   /ws/video/{job_id}/live    – stream annotated frames + progress in real time

Design note — two separate stores:
  _LOCAL_JOBS  (plain dict, lives in this process only)
      { job_id: { "process": Process, "stop_event": Event,
                  "frame_queue": Queue, "filename": str,
                  "file_path": str, "start_time": datetime } }

  main.video_status_dict  (Manager.dict, shared with worker process)
      { job_id: { "status", "progress", "frame_index",
                  "total_frames", "counts", "error" } }

We keep Process/Event/Queue objects out of the Manager.dict because they
contain un-picklable AuthenticationString data and cannot be serialised
across the manager proxy boundary.
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from multiprocessing import Process

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status

import main
from config import (
    WEIGHTS_PATH,
    TRACKER_CONFIG,
    DETECTION_CONFIDENCE,
    MIN_HITS_MAP,
)
from video_worker import video_detection_worker

logger = logging.getLogger(__name__)

# ── routers ───────────────────────────────────────────────────────────────────
# REST routes → mounted under /api  (paths become /api/video/...)
router = APIRouter()
# WebSocket routes → mounted at root (paths stay /ws/video/...)
ws_router = APIRouter()

# ── in-process job registry (NOT shared across processes) ────────────────────
# Stores un-picklable objects: Process, Event, Queue.
_LOCAL_JOBS: dict = {}

# ── upload directory ──────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
VIDEO_FRAME_QUEUE_MAXSIZE = 5
VIDEO_FRAME_QUEUE_TIMEOUT = 0.5


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_status_dict() -> dict:
    return getattr(main, "video_status_dict", {}) or {}


def _job_status_response(job_id: str) -> dict:
    status_d = _get_status_dict()
    entry = status_d.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Job not found")
    meta = _LOCAL_JOBS.get(job_id, {})
    start = meta.get("start_time", datetime.utcnow())
    return {
        "job_id": job_id,
        "filename": meta.get("filename", ""),
        "status": entry.get("status", "unknown"),
        "progress": entry.get("progress", 0.0),
        "frame_index": entry.get("frame_index", 0),
        "total_frames": entry.get("total_frames", 0),
        "counts": entry.get("counts", {}),
        "error": entry.get("error"),
        "start_time": start.isoformat() if hasattr(start, "isoformat") else str(start),
    }


def _spawn_worker(job_id, file_path, stop_event, frame_queue, status_d) -> Process:
    proc = Process(
        target=video_detection_worker,
        args=(
            job_id,
            file_path,
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
    return proc


# ── POST /api/video/upload ────────────────────────────────────────────────────

@router.post("/video/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and start a detection job. Returns { job_id, filename, message }."""
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job_id = str(uuid.uuid4())
    safe_name = f"{job_id}{ext.lower()}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        with open(file_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as exc:
        logger.error(f"Failed to save uploaded file: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded video")

    if os.path.getsize(file_path) == 0:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    mgr = main.manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="Server not ready (manager not initialized)")

    # IPC primitives — created via Manager so the worker process can share them
    stop_event = mgr.Event()
    frame_queue = mgr.Queue(maxsize=VIDEO_FRAME_QUEUE_MAXSIZE)
    status_d = main.video_status_dict   # Manager.dict — safe to share with worker

    # Spawn worker using the Python that has ultralytics
    proc = _spawn_worker(job_id, file_path, stop_event, frame_queue, status_d)
    logger.info(f"Started video detection job {job_id} for file '{file.filename}' (pid={proc.pid})")

    # Store un-picklable handles in plain in-process dict only
    _LOCAL_JOBS[job_id] = {
        "process":     proc,
        "stop_event":  stop_event,
        "frame_queue": frame_queue,
        "filename":    file.filename,
        "file_path":   file_path,
        "start_time":  datetime.utcnow(),
    }

    return {"job_id": job_id, "filename": file.filename, "message": "Detection job started"}


# ── GET /api/video/{job_id} ───────────────────────────────────────────────────

@router.get("/video/{job_id}")
async def get_video_job_status(job_id: str):
    """Return current status, progress, and counts for a video detection job."""
    return _job_status_response(job_id)


# ── POST /api/video/{job_id}/cancel ──────────────────────────────────────────

@router.post("/video/{job_id}/cancel")
async def cancel_video_job(job_id: str):
    """Cancel a running video detection job."""
    if job_id not in _LOCAL_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    entry = _LOCAL_JOBS[job_id]
    stop_event = entry.get("stop_event")
    proc = entry.get("process")

    if stop_event:
        stop_event.set()
    if proc and proc.is_alive():
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)

    status_d = _get_status_dict()
    existing = dict(status_d.get(job_id, {}))
    existing["status"] = "cancelled"
    status_d[job_id] = existing

    logger.info(f"Cancelled video detection job {job_id}")
    return {"job_id": job_id, "message": "Job cancelled"}


# ── WS /ws/video/{job_id}/live ────────────────────────────────────────────────

@ws_router.websocket("/ws/video/{job_id}/live")
async def websocket_video_live(websocket: WebSocket, job_id: str):
    """
    Stream annotated frames, per-class counts, and progress for a video job.

    Messages sent to client:
      { "type": "frame",  "frame_base64": "...", "counts": {...},
        "progress": 0.42, "frame_index": N, "total_frames": N }
      { "type": "status", "status": "completed"|"error"|"cancelled",
        "counts": {...}, "progress": 1.0, "error": null }
      { "type": "error",  "message": "..." }
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for video job {job_id}")

    try:
        # Wait up to 20 s for the job to be registered (model load takes time)
        for _ in range(66):
            if job_id in _LOCAL_JOBS:
                break
            await asyncio.sleep(0.3)
        else:
            await websocket.send_text(json.dumps({"type": "error", "message": "Job not found"}))
            await websocket.close(code=4000)
            return

        while True:
            if job_id not in _LOCAL_JOBS:
                await websocket.send_text(json.dumps({"type": "error", "message": "Job not found"}))
                await websocket.close(code=4000)
                return

            entry = _LOCAL_JOBS[job_id]
            frame_queue = entry.get("frame_queue")
            status_d = _get_status_dict()
            current = status_d.get(job_id, {})
            job_state = current.get("status", "starting")

            payload = None
            if frame_queue is not None:
                try:
                    payload = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _safe_queue_get(frame_queue, timeout=VIDEO_FRAME_QUEUE_TIMEOUT),
                    )
                except Exception:
                    pass

            if payload is not None:
                if payload.get("done"):
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "status": "completed",
                        "counts": payload.get("counts", {}),
                        "progress": 1.0,
                    }))
                    return
                else:
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "frame_base64": payload.get("frame_base64", ""),
                        "counts": payload.get("counts", {}),
                        "progress": payload.get("progress", 0.0),
                        "frame_index": payload.get("frame_index", 0),
                        "total_frames": payload.get("total_frames", 0),
                    }))
            else:
                if job_state in ("completed", "error", "cancelled"):
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "status": job_state,
                        "counts": current.get("counts", {}),
                        "progress": current.get("progress", 0.0),
                        "error": current.get("error"),
                    }))
                    return
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client for job {job_id}")
    except Exception as exc:
        logger.error(f"WebSocket error for job {job_id}: {exc}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def _safe_queue_get(queue, timeout: float):
    try:
        return queue.get(block=True, timeout=timeout)
    except Exception:
        return None
