"""
Health check endpoint for the RTSP Multi-Camera Detection Dashboard.
Verifies backend availability, SQLite connectivity, and GPU visibility.

GPU failure is intentionally NON-FATAL: the system operates on CPU when no
GPU is available, so overall status is "ok" as long as backend + DB are up.
"""

import logging
import time
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from database import get_db

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.

    Checks:
      1. backend  — always "ok" if this handler is reached
      2. database — SQLite SELECT 1
      3. gpu      — torch.cuda.is_available() (informational only, NOT fatal)

    Overall status is "ok" when backend + database are healthy.
    GPU unavailability downgrades gpu sub-check to "unavailable" but does NOT
    change the top-level status, because inference falls back to CPU automatically.
    """
    start_time = time.time()
    checks = {}

    # 1. Backend process
    checks["backend"] = "ok"

    # 2. Database connectivity
    try:
        result = db.execute(text("SELECT 1"))
        result.close()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = "error"
        logger.warning(f"Database health check failed: {e}")

    # 3. GPU availability (informational)
    try:
        if torch is not None:
            checks["gpu"] = "ok" if torch.cuda.is_available() else "unavailable"
        else:
            checks["gpu"] = "unavailable"
    except Exception as e:
        checks["gpu"] = "error"
        logger.warning(f"GPU health check failed: {e}")

    elapsed = time.time() - start_time
    if elapsed > 1.0:
        logger.warning(f"Health check took {elapsed:.3f}s")

    # Only backend + database are required for "ok" overall status
    critical_ok = checks["backend"] == "ok" and checks["database"] == "ok"
    overall = "ok" if critical_ok else "unavailable"

    return {"status": overall, "checks": checks}
