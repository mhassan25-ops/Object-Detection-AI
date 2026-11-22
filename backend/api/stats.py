"""
Dashboard statistics endpoint.
  GET /api/stats/summary
"""

import json
import logging
from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import main
from database import get_db
from models import Camera, Run
from services.detection_service import get_active_detection_count

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats/summary")
async def stats_summary(db: Session = Depends(get_db)):
    """
    Return dashboard summary statistics:
      - total_cameras          total registered cameras
      - active_detections      currently-running detection workers
      - total_runs             all persisted run records
      - aggregate_counts_by_class  sum of counts across all completed runs
    """
    try:
        total_cameras = db.query(Camera).count()
    except Exception:
        total_cameras = 0

    try:
        registry = main.registry or {}
        active_detections = get_active_detection_count(registry)
    except Exception:
        active_detections = 0

    try:
        total_runs = db.query(Run).count()
    except Exception:
        total_runs = 0

    # Aggregate per-class counts from all completed run records
    aggregate: dict = defaultdict(int)
    try:
        runs = db.query(Run).all()
        for run in runs:
            if run.counts_per_class:
                try:
                    counts = json.loads(run.counts_per_class)
                    for cls, cnt in counts.items():
                        aggregate[cls] += cnt
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "total_cameras": total_cameras,
        "active_detections": active_detections,
        "total_runs": total_runs,
        "aggregate_counts_by_class": dict(aggregate),
    }
