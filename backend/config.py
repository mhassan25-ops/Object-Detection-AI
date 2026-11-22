"""
Configuration settings for the RTSP Multi-Camera Detection Dashboard backend.

These settings include:
- Model and tracker paths
- Detection parameters (confidence, min-hits)
- RTSP retry configuration
- API and CORS settings
"""

import os
from typing import Dict

# ============================================================================
# Model and Weights Configuration
# ============================================================================

# Path to YOLOv8n weights file
# Can be overridden via environment variable WEIGHTS_PATH
WEIGHTS_PATH = os.getenv(
    "WEIGHTS_PATH",
    r"C:\Users\pc\Downloads\sack-detection-interface-main (1)\sack-detection-yolov8-main\runs\detect\sack_box_detect\v1\weights\best.pt"
)

# Path to BoT-SORT tracker configuration file
# accuracy-tuned tracker with ReID enabled
TRACKER_CONFIG = os.getenv(
    "TRACKER_CONFIG",
    "botsort.yaml"  # built-in ultralytics tracker config
)

# Detection confidence threshold (0.0 to 1.0)
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.4"))

# Image size for inference (YOLOv8 input resolution)
IMGSZ = int(os.getenv("IMGSZ", "640"))

# ============================================================================
# Hit Counting and Confirmation Configuration
# ============================================================================

# Minimum number of consecutive frames a track must appear in to be "confirmed"
# (i.e., counted as a real object, not noise)
# Can be overridden per-class via MIN_HITS_MAP
MIN_HITS_DEFAULT = int(os.getenv("MIN_HITS_DEFAULT", "15"))

# Per-class min-hits configuration
# Map class name (string) to minimum hit threshold (int)
# Classes not listed use MIN_HITS_DEFAULT
MIN_HITS_MAP: Dict[str, int] = {
    "box": int(os.getenv("MIN_HITS_BOX", "15")),
    "sack": int(os.getenv("MIN_HITS_SACK", "15")),
}

# ============================================================================
# RTSP Stream and Retry Configuration
# ============================================================================

# RTSP connection retry configuration
# max_retries: Maximum number of retry attempts for transient connection failures
# backoff_base: Base for exponential backoff (wait_time = base ** attempt)
#   E.g., attempt 1 → 2s, attempt 2 → 4s, attempt 3 → 8s, etc.
RTSP_RETRY_CONFIG: Dict[str, float] = {
    "max_retries": int(os.getenv("RTSP_MAX_RETRIES", "5")),
    "backoff_base": float(os.getenv("RTSP_BACKOFF_BASE", "2.0")),
}

# Connection timeout for RTSP streams (seconds)
RTSP_TIMEOUT = float(os.getenv("RTSP_TIMEOUT", "10.0"))

# ============================================================================
# Detection Process and GPU Configuration
# ============================================================================

# GPU device ID to use for inference (0-indexed)
# Use "0" for single GPU, or auto-select with "cpu"
GPU_DEVICE = os.getenv("GPU_DEVICE", "0")

# Maximum number of concurrent detections allowed
# Set to 2 to fit within ~227 MB VRAM per detection on NVIDIA RTX 3050
MAX_CONCURRENT_DETECTIONS = int(os.getenv("MAX_CONCURRENT_DETECTIONS", "2"))

# Timeout (seconds) when joining worker process on stop
# If exceeded, process is force-terminated
DETECTION_STOP_TIMEOUT = int(os.getenv("DETECTION_STOP_TIMEOUT", "10"))

# ============================================================================
# WebSocket and Live Feed Configuration
# ============================================================================

# Frame queue max size (queue behavior: drop-oldest-on-full)
# Set to 1 to always have the latest frame available
FRAME_QUEUE_MAXSIZE = int(os.getenv("FRAME_QUEUE_MAXSIZE", "1"))

# Timeout (seconds) when reading from frame_queue in WebSocket handler
FRAME_QUEUE_TIMEOUT = float(os.getenv("FRAME_QUEUE_TIMEOUT", "0.5"))

# JPEG quality for base64 encoding (0-100)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))

# ============================================================================
# Database Configuration
# ============================================================================

# Database file path (absolute path)
# By default, stored in backend/ directory as data.db
DB_PATH = os.getenv(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "data.db")
)

# ============================================================================
# API and CORS Configuration
# ============================================================================

# Frontend URL for CORS (allows requests from frontend)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# CORS allowed origins
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:80",
    "http://127.0.0.1:80",
    FRONTEND_URL,
]

# ============================================================================
# Health Check Configuration
# ============================================================================

# Timeout for database health check query (seconds)
HEALTH_CHECK_DB_TIMEOUT = float(os.getenv("HEALTH_CHECK_DB_TIMEOUT", "1.0"))

# ============================================================================
# Resource Monitoring Configuration
# ============================================================================

# Interval (seconds) for resource monitoring samples during detection
RESOURCE_MONITOR_INTERVAL = float(os.getenv("RESOURCE_MONITOR_INTERVAL", "1.0"))

# ============================================================================
# Logging Configuration
# ============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
