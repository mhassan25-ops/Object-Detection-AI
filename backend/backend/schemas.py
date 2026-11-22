"""
Pydantic request and response schemas for API endpoints.
Provides data validation and serialization for API payloads.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ===== Camera Schemas =====

class CameraCreate(BaseModel):
    """Schema for creating a new camera (POST request body)."""
    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    rtsp_url: str = Field(..., min_length=1, description="RTSP stream URL")
    
    @field_validator("rtsp_url")
    @classmethod
    def validate_rtsp_url(cls, v):
        """Validate that URL starts with rtsp:// or rtsps://."""
        if not (v.startswith("rtsp://") or v.startswith("rtsps://")):
            raise ValueError("RTSP URL must start with rtsp:// or rtsps://")
        return v


class CameraResponse(BaseModel):
    """Schema for camera response (GET /api/cameras)."""
    id: int
    name: str
    rtsp_url: str
    status: str = Field(..., description="Status: 'idle', 'detecting', or 'error'")
    created_at: datetime
    
    class Config:
        from_attributes = True


class CameraDetail(BaseModel):
    """Detailed camera schema."""
    id: int
    name: str
    rtsp_url: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ===== Run/Detection Result Schemas =====

class RunResponse(BaseModel):
    """Schema for run response (GET /api/runs, GET /api/runs/{run_id})."""
    id: int
    camera_id: int
    camera_name: Optional[str] = None
    start_time: datetime
    end_time: datetime
    duration_seconds: Optional[float] = None
    total_confirmed_count: int
    counts_per_class: Optional[Dict[str, int]] = None
    resource_stats: Optional[Dict[str, Any]] = None
    status: str = Field(..., description="Status: 'completed', 'stopped_forcefully', 'errored'")
    
    class Config:
        from_attributes = True


class RunCreate(BaseModel):
    """Schema for creating a run record (internal use)."""
    camera_id: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_confirmed_count: int
    counts_per_class: Dict[str, int]
    resource_stats: Optional[Dict[str, Any]] = None
    status: str = "completed"


class RunDetail(BaseModel):
    """Detailed run schema with all fields."""
    id: int
    camera_id: int
    camera_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_confirmed_count: int
    counts_per_class: Dict[str, int]
    resource_stats: Optional[Dict[str, Any]] = None
    status: str
    
    class Config:
        from_attributes = True


# ===== Detection Control Schemas =====

class StartDetectionResponse(BaseModel):
    """Response from POST /api/cameras/{id}/start."""
    success: bool
    message: Optional[str] = None


class StopDetectionResponse(BaseModel):
    """Response from POST /api/cameras/{id}/stop."""
    run_id: int
    message: Optional[str] = None


# ===== Health Check Schemas =====

class HealthCheckResponse(BaseModel):
    """Response from GET /api/health."""
    status: str = Field(..., description="'ok' or 'unavailable'")
    checks: Dict[str, str] = Field(..., description="Sub-check results: {database, gpu}")


# ===== Stats Schemas =====

class StatsResponse(BaseModel):
    """Response from GET /api/stats/summary."""
    total_cameras: int
    active_detections: int
    total_runs: int
    aggregate_counts_by_class: Dict[str, int]


# ===== WebSocket Schemas =====

class WebSocketLiveMessage(BaseModel):
    """Message sent over WebSocket /ws/cameras/{id}/live."""
    frame_base64: str = Field(..., description="Base64-encoded JPEG frame")
    counts: Dict[str, int] = Field(..., description="Per-class confirmed counts")


# ===== Error Schemas =====

class ErrorResponse(BaseModel):
    """Generic error response."""
    detail: str
    status_code: int


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    detail: str
    field: Optional[str] = None
