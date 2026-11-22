# Sack and Box Detection and Counting System

An AI-powered warehouse object detection system that counts distinct sacks and boxes from video footage using YOLOv8 object detection, BoT-SORT multi-object tracking, and a Bayesian probabilistic layer. Includes a full-stack web interface for uploading videos and viewing live annotated results.

---

## Group Members

| Name | Roll Number | Built |
|---|---|---|
| Muhammad Hassan | 24K-3051 | AI core — detection workers, Bayesian HMM counter, OOP layer, experiments |
| Muhammad Ali | 24K-3103 | Backend API — FastAPI server, REST endpoints, WebSocket, database |
| Asad Chotani | 24K-3072 | Frontend — React web interface, video upload page, live viewer, all components |

---

## Features

- Upload a video and watch the model detect and count objects frame by frame in real time
- Live RTSP camera stream support for warehouse monitoring
- Two independent counting methods: hit-counter (BoT-SORT tracking) and Bayesian HMM layer
- Per-class confirmed counts: `box` and `sack`
- Detection history with run records stored in SQLite
- Dashboard with aggregate statistics across all runs

**Known limitations:**
- CPU-only inference runs at 2–6 fps (GPU recommended for real-time use)
- Bayesian layer may miss one object when two entries happen in rapid temporal succession
- Model detects only `box` and `sack` classes; retraining required for other objects

---

## Setup Instructions

### Requirements

- Python 3.14 (must have `ultralytics`, `opencv-python`, `fastapi`, `uvicorn`, `sqlalchemy`)
- Node.js 18+ and npm

### 1. Clone the repository

```bash
git clone https://github.com/mhassan25-ops/Object-Detection-AI.git
cd Object-Detection-AI
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
```

### 4. Set the model weights path

The trained model weights (`best.pt`) are not committed to the repository due to file size (~21 MB). Set the path in `backend/config.py`:

```python
WEIGHTS_PATH = r"path\to\your\best.pt"
```

Or set the environment variable before starting:

```bash
$env:WEIGHTS_PATH = "C:\path\to\best.pt"   # PowerShell
```

---

## Running the Project

### Start the backend

```bash
cd backend
C:\Python314\python.exe -m uvicorn main:app --reload --port 8000
```

### Start the frontend (separate terminal)

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Reproducing the Experiments

Runs the three detection configurations on `video.mp4` and generates the results CSV and all three charts:

```bash
python final_experiments.py
```

**Output:**
- `outputs/results/experiment_results.csv` — per-configuration counts, errors, fps
- `outputs/figures/fig1_counts_vs_gt.png` — predicted counts vs ground truth
- `outputs/figures/fig2_abs_error.png` — absolute error per method
- `outputs/figures/fig3_noise_vs_fps.png` — noise rejection vs processing speed

Seed is fixed at `random.seed(42)` — results are fully reproducible.

---

## Project Structure

```
├── README.md
├── requirements.txt          ← backend Python dependencies
├── .gitignore
├── final_experiments.py      ← reproduces all experiment results
├── count_objects.py          ← standalone CLI detection tool
│
├── src/                      ← AI core (Member 1)
│   ├── environment.py        ← DetectionEnvironment class (PEAS wrapper)
│   ├── bayesian_counter.py   ← BayesianObjectCounter class (two-state HMM)
│   ├── agent.py              ← SackDetectorAgent class (model-based reflex agent)
│   └── main.py               ← CLI entry point
│
├── backend/                  ← FastAPI server (Member 2)
│   ├── main.py               ← app entry point, multiprocessing Manager
│   ├── worker.py             ← RTSP stream detection worker process
│   ├── video_worker.py       ← video file detection worker process
│   ├── config.py             ← all configuration constants
│   ├── models.py             ← SQLAlchemy ORM (Camera, Run tables)
│   ├── schemas.py            ← Pydantic request/response schemas
│   ├── database.py           ← SQLite session factory
│   ├── requirements.txt
│   └── api/
│       ├── cameras.py        ← GET/POST/DELETE /api/cameras
│       ├── detections.py     ← start/stop detection, run history
│       ├── video.py          ← video upload, job status, WebSocket stream
│       ├── websocket.py      ← live RTSP frame streaming
│       ├── health.py         ← /api/health
│       └── stats.py          ← /api/stats/summary
│
├── frontend/                 ← React web interface (Member 3)
│   ├── src/
│   │   ├── components/       ← VideoUpload, Live, Cameras, Dashboard, History, Results, Navbar
│   │   ├── hooks/            ← useWebSocket, useHealth
│   │   ├── services/         ← api.js (Axios), websocket.js
│   │   └── styles/           ← App.css (all global styles)
│   └── vite.config.js        ← dev proxy for /api and /ws
│
├── docs/
│   └── report.md             ← project report
│
└── outputs/
    ├── figures/              ← fig1, fig2, fig3 (generated by final_experiments.py)
    └── results/              ← experiment_results.csv
```

---

## Data Source

- **Training data:** Custom-annotated warehouse video footage (sacks and boxes)
- **Test video:** `video.mp4` — 70 seconds, 848×480, 25 fps, manually counted ground truth: 3 sacks + 6 boxes = 9 total
- **Model:** YOLOv8n fine-tuned on the custom dataset — weights at `runs/detect/sack_box_detect/v1/weights/best.pt`
