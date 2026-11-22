"""
WebSocket endpoint for live detection frame streaming.
  WS /ws/cameras/{camera_id}/live

Message format sent to client (JSON):
  { "frame_base64": "<base64 JPEG>", "counts": { "box": 3, "sack": 1 } }

Connection is closed with code 4000 if no detection is running for the camera.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import main
from config import FRAME_QUEUE_TIMEOUT

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/cameras/{camera_id}/live")
async def websocket_live(websocket: WebSocket, camera_id: int):
    """
    Stream annotated frames and confirmed counts to the browser.

    Reads frames from the shared frame_queue that the detection worker pushes to.
    Closes the connection (code 4000) when:
      - No detection is registered for this camera, OR
      - The worker process has stopped
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for camera {camera_id}")

    try:
        while True:
            registry = main.registry or {}

            # Check if detection is active for this camera
            if camera_id not in registry:
                logger.info(f"No detection running for camera {camera_id}; closing WS")
                await websocket.close(code=4000)
                return

            entry = registry[camera_id]
            status_d = main.status_dict or {}
            cam_status = status_d.get(camera_id, {})

            # If worker has stopped, close connection
            if not cam_status.get("running", True):
                # Give it a brief grace period in case it's still starting up
                await asyncio.sleep(0.5)
                cam_status = (main.status_dict or {}).get(camera_id, {})
                if not cam_status.get("running", False):
                    logger.info(f"Worker stopped for camera {camera_id}; closing WS")
                    await websocket.close(code=4000)
                    return

            frame_queue = entry.get("frame_queue")
            counts = cam_status.get("counts", {})

            # Try to read a frame (non-blocking via run_in_executor)
            frame_b64 = None
            if frame_queue is not None:
                try:
                    frame_b64 = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _safe_queue_get(frame_queue, timeout=FRAME_QUEUE_TIMEOUT),
                    )
                except Exception:
                    pass

            if frame_b64:
                message = json.dumps({"frame_base64": frame_b64, "counts": counts})
                await websocket.send_text(message)
            else:
                # No frame available yet — yield to event loop briefly
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client for camera {camera_id}")
    except Exception as exc:
        logger.error(f"WebSocket error for camera {camera_id}: {exc}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def _safe_queue_get(queue, timeout: float):
    """Blocking queue.get with timeout; returns None on Empty."""
    try:
        return queue.get(block=True, timeout=timeout)
    except Exception:
        return None
