"""
Detection worker process function.
Runs YOLOv8n + BoT-SORT inference on an RTSP stream in a separate
multiprocessing process. Falls back to CPU when no CUDA device is available.

Flow:
  1. Load YOLOv8n model from weights_path (CPU or GPU based on availability)
  2. Open RTSP stream via OpenCV; retry with exponential back-off on failure
  3. Main loop: run model.track() frame-by-frame
     - Increment hit_counts[class][track_id] for every tracked box
     - When hit_count >= min_hits[class], mark track as confirmed (once only)
     - Annotate frame with r.plot()
     - Push annotated JPEG bytes to frame_queue (drop oldest if full)
     - Update shared status_dict
     - Check stop_event each iteration
  4. Finalise: set running=False, write final counts to status_dict
"""

import base64
import logging
import time
from collections import defaultdict
from multiprocessing import Event, Queue

import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _encode_frame(frame) -> str:
    """Encode a BGR numpy frame as a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _push_frame(frame_queue: Queue, frame_b64: str):
    """Push a frame to the queue; drop the oldest entry if full."""
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except Exception:
            pass
    try:
        frame_queue.put_nowait(frame_b64)
    except Exception:
        pass


def _open_stream(rtsp_url: str, max_retries: int, backoff_base: float):
    """
    Open an OpenCV VideoCapture for the given RTSP URL.
    Retries up to max_retries times with exponential back-off.
    Returns the capture object or None on persistent failure.
    """
    for attempt in range(max_retries + 1):
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            logger.info(f"RTSP stream opened: {rtsp_url}")
            return cap
        cap.release()
        if attempt < max_retries:
            wait = backoff_base ** attempt
            logger.warning(
                f"RTSP open failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait:.0f}s …"
            )
            time.sleep(wait)
    logger.error(f"Could not open RTSP stream after {max_retries} retries: {rtsp_url}")
    return None


# ---------------------------------------------------------------------------
# main worker entry point
# ---------------------------------------------------------------------------

def detection_worker(
    camera_id: int,
    rtsp_url: str,
    weights_path: str,
    tracker_cfg: str,
    conf: float,
    min_hits_map: dict,
    frame_queue: Queue,
    status_dict: dict,
    stop_event: Event,
    logger_queue: Queue = None,
):
    """
    Detection worker process.

    Designed to be launched via multiprocessing.Process(target=detection_worker, …).
    All arguments must be picklable.

    Args:
        camera_id      – used as the key in status_dict
        rtsp_url       – RTSP stream URL; also accepts local file paths / webcam int
        weights_path   – path to YOLOv8 weights (.pt)
        tracker_cfg    – BoT-SORT tracker YAML name (e.g. "botsort.yaml")
        conf           – detection confidence threshold
        min_hits_map   – {class_name: min_frames} to confirm a track
        frame_queue    – Queue(maxsize=1) shared with WebSocket handler
        status_dict    – Manager.dict(); keyed by camera_id
        stop_event     – set() to request graceful shutdown
        logger_queue   – unused, reserved for future log forwarding
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(f"worker.{camera_id}")

    # Initialise status entry
    status_dict[camera_id] = {"running": False, "counts": {}, "error": None}

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading model from {weights_path} on device={device}")
        model = YOLO(weights_path)
    except Exception as exc:
        log.error(f"Failed to load model: {exc}", exc_info=True)
        status_dict[camera_id] = {"running": False, "counts": {}, "error": str(exc)}
        return

    # ------------------------------------------------------------------
    # 2. Open RTSP stream
    # ------------------------------------------------------------------
    from config import RTSP_RETRY_CONFIG

    cap = _open_stream(
        rtsp_url,
        max_retries=int(RTSP_RETRY_CONFIG["max_retries"]),
        backoff_base=float(RTSP_RETRY_CONFIG["backoff_base"]),
    )
    if cap is None:
        err = f"Could not open RTSP stream: {rtsp_url}"
        status_dict[camera_id] = {"running": False, "counts": {}, "error": err}
        return

    # ------------------------------------------------------------------
    # 3. Per-track state
    # ------------------------------------------------------------------
    hit_counts: dict = defaultdict(lambda: defaultdict(int))   # {class_name: {track_id: count}}
    confirmed_ids: dict = defaultdict(set)                      # {class_name: {track_id, …}}
    counts: dict = defaultdict(int)                             # {class_name: confirmed_count}

    status_dict[camera_id] = {"running": True, "counts": dict(counts), "error": None}
    log.info(f"Detection started for camera {camera_id}")

    consecutive_read_failures = 0
    max_read_failures = 30  # ~1 second at 30 fps before re-open attempt

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_read_failures:
                    log.warning("Too many consecutive read failures, attempting stream re-open…")
                    cap.release()
                    cap = _open_stream(
                        rtsp_url,
                        max_retries=int(RTSP_RETRY_CONFIG["max_retries"]),
                        backoff_base=float(RTSP_RETRY_CONFIG["backoff_base"]),
                    )
                    if cap is None:
                        err = "RTSP stream lost and could not be re-opened"
                        log.error(err)
                        status_dict[camera_id] = {
                            "running": False,
                            "counts": dict(counts),
                            "error": err,
                        }
                        return
                    consecutive_read_failures = 0
                time.sleep(0.03)
                continue

            consecutive_read_failures = 0

            # Run tracker
            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=conf,
                    tracker=tracker_cfg,
                    device=device,
                    verbose=False,
                )
            except Exception as exc:
                log.warning(f"model.track() error (skipping frame): {exc}")
                continue

            annotated = results[0].plot() if results else frame

            # Process tracked boxes
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    cls_idx = int(box.cls.item())
                    cls_name = model.names.get(cls_idx, str(cls_idx))
                    min_h = min_hits_map.get(cls_name, 15)

                    hit_counts[cls_name][track_id] += 1

                    if (
                        hit_counts[cls_name][track_id] >= min_h
                        and track_id not in confirmed_ids[cls_name]
                    ):
                        confirmed_ids[cls_name].add(track_id)
                        counts[cls_name] += 1

            # Push annotated frame
            frame_b64 = _encode_frame(annotated)
            if frame_b64:
                _push_frame(frame_queue, frame_b64)

            # Update shared status
            status_dict[camera_id] = {
                "running": True,
                "counts": dict(counts),
                "error": None,
            }

    except Exception as exc:
        log.error(f"Unhandled error in detection loop: {exc}", exc_info=True)
        status_dict[camera_id] = {
            "running": False,
            "counts": dict(counts),
            "error": str(exc),
        }
    finally:
        cap.release()
        log.info(f"Detection stopped for camera {camera_id}. Final counts: {dict(counts)}")
        # Mark as stopped (do not overwrite an error already set above)
        current = status_dict.get(camera_id, {})
        if current.get("running", False):
            status_dict[camera_id] = {
                "running": False,
                "counts": dict(counts),
                "error": current.get("error"),
            }
