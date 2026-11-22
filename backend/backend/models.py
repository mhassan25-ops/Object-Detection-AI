"""
SQLAlchemy ORM models for Camera and Run entities.
These models define the database schema for persistent storage.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON
from database import Base


class Camera(Base):
    """
    Camera model representing a registered RTSP stream source.
    
    Attributes:
        id: Unique identifier (primary key)
        name: Camera name (e.g., "Factory Floor Camera 1")
        rtsp_url: RTSP stream URL (e.g., "rtsp://192.168.1.100:554/stream")
        created_at: Timestamp when camera was registered
    """
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String(512), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name='{self.name}', rtsp_url='{self.rtsp_url}')>"


class Run(Base):
    """
    Run model representing a completed detection session.
    
    Attributes:
        id: Unique identifier (primary key)
        camera_id: Foreign key to Camera
        start_time: When the detection started
        end_time: When the detection stopped
        duration_seconds: Total duration of the detection run
        total_confirmed_count: Total number of confirmed objects across all classes
        counts_per_class: JSON string mapping class names to confirmed counts (e.g., '{"box": 12, "sack": 8}')
        resource_stats: JSON string with resource usage stats (CPU %, GPU %, RAM, etc.)
        status: Status of the run ('completed', 'stopped_forcefully', 'errored')
    """
    __tablename__ = "runs"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    total_confirmed_count = Column(Integer, nullable=False, default=0)
    counts_per_class = Column(String, nullable=True)  # JSON string
    resource_stats = Column(String, nullable=True)  # JSON string
    status = Column(String(50), nullable=False, default="completed")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Run(id={self.id}, camera_id={self.camera_id}, total_count={self.total_confirmed_count})>"
