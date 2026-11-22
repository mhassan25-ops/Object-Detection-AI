"""
Video file detection worker.

Processes an uploaded video file frame-by-frame using YOLOv8 + BoT-SORT,
mirrors the same hit-counting confirmation logic from worker.py but reads
from a local file path instead of an RTSP stream.

Runs in a separate multiprocessing.Process so it doesn't block the FastAPI
event loop.

Status dict entry shape:
  {
    "status": "processing" | "completed" | "error",
    "progress": 0.0-1.0,       # fraction of frames processed
    "frame_index": int,
    "total_frames": int,
    "counts": {"box": N, "sack": M},
    "error": str | None,
  }
"""

import base64
import logging
import time
from collections import defaultdict
from multiprocessing import Event, Queue

import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers (mirror of worker.py helpers)
# ---------------------------------------------------------------------------

def _encode_frame(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _push_frame(frame_queue: Queue, payload: dict):
    """Push a frame payload dict; drop oldest if full."""
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except Exception:
            pass
    try:
        frame_queue.put_nowait(payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# main worker entry point
# ---------------------------------------------------------------------------

def video_detection_worker(
    job_id: str,
    video_path: str,
    weights_path: str,
    tracker_cfg: str,
    conf: float,
    min_hits_map: dict,
    frame_queue: Queue,
    status_dict: dict,
    stop_event: Event,
):
    """
    Video file detection worker process.

    Designed to be launched via multiprocessing.Process(target=video_detection_worker, …).
    All arguments must be picklable.

    Args:
        job_id        – unique job identifier (key in status_dict)
        video_path    – local path to the uploaded video file
        weights_path  – path to YOLOv8 weights (.pt)
        tracker_cfg   – BoT-SORT tracker YAML name
        conf          – detection confidence threshold
        min_hits_map  – {class_name: min_frames} to confirm a track
        frame_queue   – Queue(maxsize=5) shared with WebSocket handler
        status_dict   – Manager.dict(); keyed by job_id
        stop_event    – set() to request graceful shutdown
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(f"video_worker.{job_id}")

    # Initialise status
    status_dict[job_id] = {
        "status": "starting",
        "progress": 0.0,
        "frame_index": 0,
        "total_frames": 0,
        "counts": {},
        "error": None,
    }

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
        status_dict[job_id] = {
            "status": "error",
            "progress": 0.0,
            "frame_index": 0,
            "total_frames": 0,
            "counts": {},
            "error": str(exc),
        }
        return

    # ------------------------------------------------------------------
    # 2. Open video file
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err = f"Could not open video file: {video_path}"
        log.error(err)
        status_dict[job_id] = {
            "status": "error",
            "progress": 0.0,
            "frame_index": 0,
            "total_frames": 0,
            "counts": {},
            "error": err,
        }
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info(f"Video opened: {video_path}, total_frames={total_frames}, fps={fps:.1f}")

    # ------------------------------------------------------------------
    # 3. Per-track state
    # ------------------------------------------------------------------
    hit_counts: dict = defaultdict(lambda: defaultdict(int))
    confirmed_ids: dict = defaultdict(set)
    counts: dict = defaultdict(int)

    status_dict[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "frame_index": 0,
        "total_frames": total_frames,
        "counts": dict(counts),
        "error": None,
    }

    frame_index = 0

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                # End of video
                break

            frame_index += 1
            progress = frame_index / total_frames if total_frames > 0 else 1.0

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
                log.warning(f"model.track() error (skipping frame {frame_index}): {exc}")
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

            # Encode and push frame with metadata
            frame_b64 = _encode_frame(annotated)
            if frame_b64:
                payload = {
                    "frame_base64": frame_b64,
                    "counts": dict(counts),
                    "progress": progress,
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                }
                _push_frame(frame_queue, payload)

            # Update shared status
            status_dict[job_id] = {
                "status": "processing",
                "progress": progress,
                "frame_index": frame_index,
                "total_frames": total_frames,
                "counts": dict(counts),
                "error": None,
            }

    except Exception as exc:
        log.error(f"Unhandled error in video detection loop: {exc}", exc_info=True)
        status_dict[job_id] = {
            "status": "error",
            "progress": frame_index / total_frames if total_frames > 0 else 0.0,
            "frame_index": frame_index,
            "total_frames": total_frames,
            "counts": dict(counts),
            "error": str(exc),
        }
    else:
        # Clean completion
        log.info(f"Video processing complete for job {job_id}. Counts: {dict(counts)}")
        status_dict[job_id] = {
            "status": "completed",
            "progress": 1.0,
            "frame_index": frame_index,
            "total_frames": total_frames,
            "counts": dict(counts),
            "error": None,
        }
        # Push a sentinel so the WebSocket knows we're done
        _push_frame(frame_queue, {"done": True, "counts": dict(counts)})
    finally:
        cap.release()
