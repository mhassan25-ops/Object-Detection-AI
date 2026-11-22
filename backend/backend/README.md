# RTSP Multi-Camera Detection Dashboard - Backend

This is the FastAPI backend for the RTSP Multi-Camera Detection Dashboard, a real-time object detection system for YOLOv8n + BoT-SORT tracking on multiple RTSP camera streams.

## Project Structure

```
backend/
├── main.py                      # FastAPI application entry point
├── database.py                  # SQLAlchemy database setup and session factory
├── models.py                    # ORM models: Camera, Run
├── schemas.py                   # Pydantic request/response models
├── config.py                    # Configuration settings
├── api/                         # Route handlers
│   ├── cameras.py              # Camera CRUD endpoints
│   ├── detections.py           # Detection control endpoints
│   ├── health.py               # Health check endpoint
│   ├── stats.py                # Dashboard stats endpoint
│   └── websocket.py            # WebSocket live feed endpoint
├── services/                    # Business logic layer
│   ├── camera_service.py       # Camera operations
│   ├── detection_service.py    # Detection management
│   └── health_service.py       # Health checks
├── worker.py                    # Detection worker process (YOLOv8n + BoT-SORT)
├── requirements.txt            # Python dependencies
└── test_models_schemas.py      # Unit tests for models and schemas
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- NVIDIA GPU (RTX 3050 recommended, ~227 MB VRAM per detection)
- CUDA and cuDNN (pre-installed in `ultralytics` Docker image)

### Setup

1. **Clone the repository** (or navigate to the backend directory):
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights** (if not already present):
   ```bash
   mkdir -p weights
   # Copy your trained best.pt to weights/best.pt
   # Or the weights will be auto-downloaded on first run
   ```

## Running the Backend

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Note: Keep `--workers 1` to ensure multiprocessing Manager works correctly.

## Configuration

Configuration is managed through `config.py` and can be overridden via environment variables.

### Key Settings

- `WEIGHTS_PATH`: Path to YOLOv8n weights file (default: `weights/best.pt`)
- `TRACKER_CONFIG`: BoT-SORT tracker configuration (default: `botsort_accurate.yaml`)
- `DETECTION_CONFIDENCE`: Confidence threshold (default: `0.4`)
- `MIN_HITS_MAP`: Per-class minimum hit thresholds (default: `{"box": 15, "sack": 15}`)
- `MAX_CONCURRENT_DETECTIONS`: Max parallel detections (default: `2`)
- `GPU_DEVICE`: GPU device ID (default: `"0"`)
- `DB_PATH`: SQLite database path (default: `backend/data.db`)
- `CORS_ORIGINS`: Allowed CORS origins (default: `localhost:3000`)

### Environment Variables

Create a `.env` file to override defaults:

```env
WEIGHTS_PATH=weights/best.pt
MIN_HITS_BOX=15
MIN_HITS_SACK=15
MAX_CONCURRENT_DETECTIONS=2
GPU_DEVICE=0
DETECTION_CONFIDENCE=0.4
```

Load with: `python-dotenv` (automatically loaded by the app).

## Database

The backend uses **SQLite** for persistent storage of cameras and detection results.

### Schema

**cameras** table:
- `id`: Primary key
- `name`: Camera name (required)
- `rtsp_url`: RTSP stream URL (required)
- `created_at`: Creation timestamp

**runs** table:
- `id`: Primary key
- `camera_id`: Foreign key to cameras
- `start_time`: Detection start timestamp
- `end_time`: Detection end timestamp
- `duration_seconds`: Total runtime
- `total_confirmed_count`: Total objects counted
- `counts_per_class`: JSON string of per-class counts
- `resource_stats`: JSON string of resource usage (CPU, RAM, GPU)
- `status`: Run status (completed, stopped_forcefully, errored)

### Initialization

Database is automatically initialized on startup via `init_db()` in `main.py`. Tables are created if they don't exist (CREATE TABLE IF NOT EXISTS).

## API Endpoints

### Health Check
- `GET /api/health` - Check backend, database, and GPU status

### Camera Management
- `GET /api/cameras` - List all cameras
- `POST /api/cameras` - Create a new camera
- `DELETE /api/cameras/{id}` - Delete a camera

### Detection Control
- `POST /api/cameras/{id}/start` - Start detection on a camera
- `POST /api/cameras/{id}/stop` - Stop detection and save results
- `GET /api/runs` - List all completed runs
- `GET /api/runs/{run_id}` - Get details of a specific run

### Dashboard
- `GET /api/stats/summary` - Get dashboard statistics

### WebSocket
- `WS /ws/cameras/{id}/live` - Live detection feed (frames and counts)

## Testing

Run unit tests for models and schemas:

```bash
python3 test_models_schemas.py
```

This validates:
- Pydantic schema validation (RTSP URL format, name length, etc.)
- Configuration loading
- Model structure

## Architecture

### Multi-Processing Design

- **Main FastAPI Process**: Handles HTTP requests, WebSocket connections, registry management
- **Detection Worker Processes**: Each detection runs in its own `multiprocessing.Process`
  - Loads YOLOv8n model and BoT-SORT tracker
  - Reads frames from RTSP stream
  - Tracks objects across frames
  - Maintains hit counts for noise rejection
  - Pushes annotated frames to shared Queue
  - Updates shared status dictionary

### Registry

In-memory registry (multiprocessing.Manager.dict) tracks active detections:

```python
registry[camera_id] = {
    "process": multiprocessing.Process,
    "stop_event": multiprocessing.Event,
    "frame_queue": multiprocessing.Queue(maxsize=1),
    "status": status_dict[camera_id],
    "start_time": datetime
}

status_dict[camera_id] = {
    "running": bool,
    "counts": {class_name: confirmed_count}
}
```

### Concurrency Control

- **Max 2 concurrent detections** to stay within GPU VRAM budget (~227 MB per detection on RTX 3050)
- Enforced server-side in `/start` endpoint (not just UI)
- Checked before spawning new detection worker process

### Hit Counting and Confirmation

Noise rejection via hit counting:

1. Track appears in frame → hit_count incremented
2. When hit_count ≥ min_hits (default 15) → track is confirmed
3. Confirmed count = number of unique confirmed tracks per class
4. Prevents single-frame false positives from inflating counts

## Troubleshooting

### GPU Not Detected

```bash
# Verify CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

If False, ensure:
- NVIDIA drivers installed
- CUDA toolkit compatible with PyTorch version
- GPU not in exclusive mode

### RTSP Connection Fails

- Verify RTSP URL format: `rtsp://host:port/stream`
- Check network connectivity: `ping 192.168.x.x`
- Verify camera stream is active: `ffplay rtsp://...`
- Retry logic will attempt up to 5 times with exponential backoff

### Database Issues

```bash
# Remove stale database
rm backend/data.db
# Will be recreated on next startup
```

### Port Already in Use

```bash
# Use a different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Performance Notes

- Frame rate: 2–5 fps live display (intentional for bandwidth efficiency)
- Latency: 5–15 seconds end-to-end (acceptable for this use case)
- Resource usage: ~227 MB VRAM per detection on RTX 3050
- Processing: Single frame inference time depends on image size (default 640x640)

## Dependencies

See `requirements.txt` for full list. Key packages:

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `sqlalchemy` - ORM
- `pydantic` - Data validation
- `ultralytics` - YOLOv8 models
- `torch` - PyTorch (included in ultralytics)
- `opencv-python` - Image processing
- `psutil` - System resource monitoring

## Contributing

Follow these guidelines:

1. Update `schemas.py` when adding new API endpoints
2. Add tests in `test_models_schemas.py`
3. Document configuration changes in `config.py`
4. Use type hints in function signatures
5. Add docstrings to classes and functions

## License

[Your License Here]

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review API documentation at `/docs` (Swagger UI)
3. Check logs in stdout/stderr

---

**Last Updated**: [Current Date]
**Version**: 1.0.0
