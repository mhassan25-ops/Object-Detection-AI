# RTSP Multi-Camera Detection Dashboard

A local web application for registering CCTV cameras, running YOLOv8n + BoT-SORT object detection on live RTSP streams, and reviewing results in real time and historically.

The system enforces a hard maximum of **2 concurrent GPU-accelerated detections** to stay within the VRAM budget of an NVIDIA RTX 3050 (~227 MB VRAM per detection instance). The entire stack runs via Docker Compose (Docker Desktop with WSL2 backend).

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  Browser  →  http://localhost  (nginx on port 80)        │
│                                                          │
│  nginx proxies:                                          │
│    /api/*  →  backend:8000  (REST API)                   │
│    /ws/*   →  backend:8000  (WebSocket live feed)        │
└──────────────────────────────────────────────────────────┘
         │
┌────────▼─────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                            │
│    - Camera CRUD                                         │
│    - Detection worker process management                 │
│    - WebSocket live frame streaming                      │
│    - SQLite persistence (data.db)                        │
│    - GPU-accelerated YOLOv8n + BoT-SORT inference        │
└──────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11 with WSL2 enabled |
| **Docker Desktop** | Version 4.x or newer (with WSL2 backend) |
| **NVIDIA GPU** | RTX 3050 or better (1 GB+ VRAM) |
| **NVIDIA Driver** | Windows driver ≥ 527.41 (enables GPU passthrough to WSL2) |
| **Model weights** | `weights/best.pt` (YOLOv8n trained on box/sack dataset) |

### Verify GPU Passthrough (once, before first run)

```bash
docker run --rm --gpus=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` prints your GPU name and driver version, GPU passthrough is working. If it fails, see the [GPU Troubleshooting](#gpu-troubleshooting) section below.

---

## Quick Start

### 1. Clone and set up model weights

```bash
git clone <repo-url>
cd sack-detection-interface-main

# Create the weights directory and copy your trained model
mkdir -p weights
cp /path/to/your/best.pt weights/best.pt
```

### 2. Start with the startup script (recommended)

The `startup.sh` script verifies GPU passthrough, creates required files, builds images, and starts the stack:

```bash
chmod +x startup.sh
./startup.sh
```

To skip the GPU verification step:

```bash
./startup.sh --skip-gpu-check
```

### 3. Or start manually

```bash
# Create data.db file (needed for the volume mount)
touch data.db

# Build images
docker compose build

# Start services in the background
docker compose up -d
```

---

## Access the Application

| Service | URL |
|---------|-----|
| **Frontend** (React app) | http://localhost |
| **Backend API** | http://localhost:8000 |
| **Swagger / OpenAPI docs** | http://localhost:8000/docs |

---

## Stopping and Managing the Stack

```bash
# Stop all services (containers removed, volumes/images kept)
docker compose down

# View logs for all services
docker compose logs -f

# View logs for a specific service
docker compose logs -f backend
docker compose logs -f frontend

# Rebuild after code changes
docker compose build
docker compose up -d

# Full teardown (removes volumes too — deletes data.db!)
docker compose down -v
```

---

## Project Structure

```
.
├── backend/                 # FastAPI backend
│   ├── api/                 # Route handlers (cameras, detections, health, stats, ws)
│   ├── services/            # Business logic layer
│   ├── main.py              # App entry point, registry setup
│   ├── worker.py            # Detection worker process (YOLOv8n + BoT-SORT)
│   ├── models.py            # SQLAlchemy ORM (Camera, Run)
│   ├── schemas.py           # Pydantic request/response models
│   ├── config.py            # Configuration (weights path, min_hits, etc.)
│   ├── database.py          # SQLite session factory
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile
├── frontend/                # React frontend (Vite + React Router)
│   ├── src/
│   │   ├── components/      # Page components (Dashboard, Cameras, Live, Results, History, Navbar)
│   │   ├── services/        # API and WebSocket clients
│   │   └── hooks/           # Custom React hooks
│   ├── nginx.conf           # nginx config (serves SPA, proxies /api and /ws to backend)
│   └── Dockerfile           # Multi-stage: Node build → nginx serve
├── weights/                 # Model weights (not committed to git)
│   └── best.pt              # YOLOv8n trained model
├── data.db                  # SQLite database (persisted via volume mount)
├── docker-compose.yml       # Service orchestration with GPU reservation
├── startup.sh               # GPU verification + startup helper script
└── README.md                # This file
```

---

## Configuration

Backend configuration is managed via environment variables (with defaults in `backend/config.py`).

Add an optional `.env` file in the project root to override defaults:

```env
# Model inference
WEIGHTS_PATH=weights/best.pt
INFERENCE_CONF=0.4
MIN_HITS_BOX=15
MIN_HITS_SACK=15

# RTSP retry behavior
RTSP_MAX_RETRIES=5
RTSP_BACKOFF_BASE=2

# GPU device index
GPU_DEVICE=0
```

Pass `.env` to docker compose by adding `env_file: .env` under the `backend` service in `docker-compose.yml`, or set them inline under `environment:`.

---

## Data Persistence

The SQLite database (`data.db`) is mounted as a volume into the backend container:

```yaml
volumes:
  - ./data.db:/app/data.db
```

This means **camera registrations and detection run history persist across container restarts and rebuilds**. To reset all data:

```bash
docker compose down
rm data.db
touch data.db
docker compose up -d
```

Similarly, model weights are mounted at `./weights:/app/weights` so you can swap the model without rebuilding the image.

---

## Troubleshooting

### GPU passthrough issues

**Symptom**: `docker run --rm --gpus=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` fails or backend health reports `"gpu": "unreachable"`.

**Checklist**:

1. **Update NVIDIA driver** (Windows host): https://www.nvidia.com/Download/index.aspx — you need driver version ≥ 527.41 for WSL2 CUDA passthrough.
2. **Enable WSL2 backend**: Docker Desktop → Settings → General → "Use the WSL 2 based engine" must be checked.
3. **Enable WSL integration**: Docker Desktop → Settings → Resources → WSL Integration → enable your default distro.
4. **Restart Docker Desktop** after any driver or settings change (full quit + reopen, not just restart daemon).
5. **Verify NVIDIA Container Toolkit** is included — it ships with Docker Desktop on Windows, but can be missing in some configurations:
   ```bash
   docker info | findstr "Runtimes"
   # Should show: Runtimes: nvidia runc
   ```

**Check GPU availability inside the container:**

```bash
docker compose exec backend python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**If GPU remains unavailable**, detection falls back to CPU automatically, which is significantly slower (~2–5 fps → ~0.5–1 fps). The navbar will show "Service unavailable" because the health check GPU sub-check fails. You can still use the application — just expect slower inference.

---

### RTSP connection failures and retry logic

**Symptom**: Detection starts but immediately errors, or camera shows as "errored" in the list.

**Diagnose connectivity**:

```bash
# Ping the camera from inside the backend container
docker compose exec backend ping -c 3 192.168.1.100

# Test RTSP stream directly
docker compose exec backend ffprobe rtsp://192.168.1.100:554/stream
```

**Common causes and fixes**:

| Problem | Symptom | Fix |
|---------|---------|-----|
| Wrong IP/port | Immediate connection refused | Verify URL with `ffplay rtsp://...` on host |
| Camera requires auth | 401 Unauthorized in logs | Include credentials: `rtsp://user:pass@host:port/path` |
| Camera on different VLAN | Ping fails from container | Ensure Docker host networking or correct bridge |
| Camera stream inactive | Connection timeout | Check camera power and network cable |
| Firewall blocking RTSP | Timeout after TCP connect | Open port 554 (RTSP) in Windows Firewall |

**Retry logic**: The detection worker automatically retries RTSP connections up to 5 times with exponential backoff (2s, 4s, 8s, 16s, 32s) before marking the detection as errored. Transient network drops during an active stream are also retried. You can tune this via environment variables:

```env
RTSP_MAX_RETRIES=5    # number of retry attempts
RTSP_BACKOFF_BASE=2   # base for exponential backoff (seconds)
```

---

### Port conflicts

**Symptom**: `docker compose up` fails with "port is already allocated" or "address already in use".

```bash
# Find what's using port 80 or 8000
netstat -ano | findstr ":80 "
netstat -ano | findstr ":8000"

# Kill the process by PID (replace 1234 with actual PID)
taskkill /F /PID 1234
```

**Or remap ports** in `docker-compose.yml`:

```yaml
services:
  frontend:
    ports:
      - "8080:80"    # serve frontend on http://localhost:8080

  backend:
    ports:
      - "8001:8000"  # serve backend API on http://localhost:8001
```

If you remap the backend port, also update `frontend/nginx.conf` proxy_pass targets to match, and rebuild both images.

---

### Common startup errors

**`data.db` volume mount fails** — backend exits with "no such file or directory":

The `docker-compose.yml` mounts `./data.db` as a file volume. Docker requires the file to exist before mounting.

```bash
# Create it first (only needed once)
New-Item -ItemType File -Name "data.db" -Force   # PowerShell
# or on Windows CMD:
type nul > data.db
```

**Backend container exits immediately with `ImportError`**:

The ultralytics base image may not have all requirements. Check logs:

```bash
docker compose logs backend
```

If a package is missing, add it to `backend/requirements.txt` and rebuild:

```bash
docker compose build backend
docker compose up -d backend
```

**Frontend shows blank page after `docker compose up`**:

The nginx config requires the backend API to be reachable. If the backend is still starting, nginx's proxy may return 502 for the first few requests. Wait 10–20 seconds and reload.

Check that both containers are running:

```bash
docker compose ps
# Both 'backend' and 'frontend' should show status 'Up'
```

**WebSocket connection refused or immediately closes**:

nginx proxies `/ws/*` to the backend with upgraded headers. If you see WebSocket errors in the browser console, verify:

1. `nginx.conf` has `proxy_http_version 1.1` and `Upgrade`/`Connection` headers for `/ws` location.
2. The backend is healthy: `curl http://localhost:8000/api/health`
3. A detection is actually running on that camera (WebSocket closes with code 4000 if not).

**`Max 2 concurrent detections reached` (409 error)**:

This is expected behavior, not a bug. Stop one of the running detections from its live page before starting a new one.

**`Cannot delete camera: detection is currently running` (409 error)**:

Stop the active detection on that camera first, then delete it.

---

### Database issues

**Reset the database** (deletes all cameras and run history):

```bash
docker compose down
Remove-Item data.db        # PowerShell
New-Item -ItemType File -Name "data.db" -Force
docker compose up -d
```

**SQLite locked errors** (rare, usually from multiple processes):

The backend must run with `--workers 1` (single uvicorn worker process). Check that `docker-compose.yml` does not set `--workers > 1` in the CMD.

**Inspect the database directly** (from host):

```bash
# Install sqlite3 CLI (Windows: winget install SQLite.SQLite)
sqlite3 data.db ".tables"
sqlite3 data.db "SELECT * FROM cameras;"
sqlite3 data.db "SELECT id, camera_id, status, total_confirmed_count FROM runs ORDER BY id DESC LIMIT 10;"
```

---

## Running Tests

### Backend unit and property-based tests

```bash
cd backend
pip install -r requirements.txt
python -m pytest -v
```

### Frontend unit tests

```bash
cd frontend
npm install
npm run test:run
```

For the full testing guide including the 20 correctness properties coverage table, see [TESTING.md](./TESTING.md).

---

## License

[Your License Here]
