"""
Unit tests for models.py and schemas.py validation.
Tests ORM model structure and Pydantic schema validation.

Note: This test file can be run without a full database setup or all dependencies.
It focuses on model definitions and schema validation logic.
"""

import json
from datetime import datetime
from typing import Dict, Any

# Test for Pydantic schemas (no SQLAlchemy needed)
def test_camera_create_schema_valid():
    """Test CameraCreate validates valid input."""
    from schemas import CameraCreate
    
    camera = CameraCreate(
        name="Factory Floor Camera",
        rtsp_url="rtsp://192.168.1.100:554/stream"
    )
    assert camera.name == "Factory Floor Camera"
    assert camera.rtsp_url == "rtsp://192.168.1.100:554/stream"
    print("✓ CameraCreate accepts valid RTSP URL (rtsp://)")


def test_camera_create_schema_rtsps():
    """Test CameraCreate accepts rtsps:// URLs."""
    from schemas import CameraCreate
    
    camera = CameraCreate(
        name="Secure Camera",
        rtsp_url="rtsps://192.168.1.100:322/stream"
    )
    assert camera.rtsp_url == "rtsps://192.168.1.100:322/stream"
    print("✓ CameraCreate accepts valid RTSP URL (rtsps://)")


def test_camera_create_schema_invalid_url():
    """Test CameraCreate rejects invalid RTSP URLs."""
    from schemas import CameraCreate
    from pydantic import ValidationError
    
    try:
        CameraCreate(
            name="Bad Camera",
            rtsp_url="http://192.168.1.100/stream"
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "rtsp://" in str(e).lower() or "rtsps://" in str(e).lower()
        print("✓ CameraCreate rejects http:// URL")


def test_camera_create_schema_empty_name():
    """Test CameraCreate rejects empty name."""
    from schemas import CameraCreate
    from pydantic import ValidationError
    
    try:
        CameraCreate(
            name="",
            rtsp_url="rtsp://192.168.1.100:554/stream"
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("✓ CameraCreate rejects empty name")


def test_camera_create_schema_name_too_long():
    """Test CameraCreate rejects names > 255 chars."""
    from schemas import CameraCreate
    from pydantic import ValidationError
    
    try:
        CameraCreate(
            name="x" * 256,  # 256 chars, exceeds limit
            rtsp_url="rtsp://192.168.1.100:554/stream"
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("✓ CameraCreate rejects name > 255 chars")


def test_camera_response_schema():
    """Test CameraResponse can be constructed."""
    from schemas import CameraResponse
    
    now = datetime.utcnow()
    response = CameraResponse(
        id=1,
        name="Test Camera",
        rtsp_url="rtsp://192.168.1.100:554/stream",
        status="idle",
        created_at=now
    )
    assert response.id == 1
    assert response.status == "idle"
    print("✓ CameraResponse schema valid")


def test_run_response_schema():
    """Test RunResponse with parsed JSON fields."""
    from schemas import RunResponse
    
    now = datetime.utcnow()
    response = RunResponse(
        id=42,
        camera_id=1,
        camera_name="Test Camera",
        start_time=now,
        end_time=now,
        duration_seconds=300.5,
        total_confirmed_count=25,
        counts_per_class={"box": 12, "sack": 13},
        resource_stats={
            "cpu_util_avg_pct": 45.2,
            "gpu_util_avg_pct": 87.5,
            "gpu_mem_used_avg_mb": 227
        },
        status="completed"
    )
    assert response.total_confirmed_count == 25
    assert response.counts_per_class["box"] == 12
    assert response.resource_stats["cpu_util_avg_pct"] == 45.2
    print("✓ RunResponse schema valid with parsed JSON")


def test_health_check_response():
    """Test HealthCheckResponse schema."""
    from schemas import HealthCheckResponse
    
    response = HealthCheckResponse(
        status="ok",
        checks={"database": "ok", "gpu": "ok"}
    )
    assert response.status == "ok"
    assert response.checks["database"] == "ok"
    print("✓ HealthCheckResponse schema valid")


def test_detection_start_response():
    """Test StartDetectionResponse schema."""
    from schemas import StartDetectionResponse
    
    response = StartDetectionResponse(
        success=True,
        message="Detection started on camera 1"
    )
    assert response.success is True
    print("✓ StartDetectionResponse schema valid")


def test_detection_stop_response():
    """Test StopDetectionResponse schema."""
    from schemas import StopDetectionResponse
    
    response = StopDetectionResponse(
        run_id=42,
        message="Detection stopped, run saved"
    )
    assert response.run_id == 42
    print("✓ StopDetectionResponse schema valid")


def test_stats_response():
    """Test StatsResponse schema."""
    from schemas import StatsResponse
    
    response = StatsResponse(
        total_cameras=5,
        active_detections=1,
        total_runs=42,
        aggregate_counts_by_class={"box": 312, "sack": 287}
    )
    assert response.total_cameras == 5
    assert response.aggregate_counts_by_class["box"] == 312
    print("✓ StatsResponse schema valid")


def test_websocket_live_message():
    """Test WebSocketLiveMessage schema."""
    from schemas import WebSocketLiveMessage
    
    message = WebSocketLiveMessage(
        frame_base64="iVBORw0KGgoAAAA...",  # fake base64
        counts={"box": 12, "sack": 8}
    )
    assert len(message.frame_base64) > 0
    assert message.counts["box"] == 12
    print("✓ WebSocketLiveMessage schema valid")


def test_config_imports():
    """Test that config.py can be imported and basic settings exist."""
    try:
        import config
        
        # Core settings should exist
        assert hasattr(config, 'WEIGHTS_PATH')
        assert hasattr(config, 'MIN_HITS_MAP')
        
        # Verify MIN_HITS_MAP has expected classes
        assert "box" in config.MIN_HITS_MAP
        assert "sack" in config.MIN_HITS_MAP
        assert isinstance(config.MIN_HITS_MAP["box"], int)
        assert isinstance(config.MIN_HITS_MAP["sack"], int)
        
        # Verify concurrency limit
        assert hasattr(config, 'MAX_CONCURRENT_DETECTIONS')
        assert config.MAX_CONCURRENT_DETECTIONS == 2, "Must enforce max 2 concurrent detections"
        
        # Verify CORS origins
        assert hasattr(config, 'CORS_ORIGINS')
        assert isinstance(config.CORS_ORIGINS, list)
        assert len(config.CORS_ORIGINS) > 0
        
        print("✓ Config module loads with expected settings")
    except Exception as e:
        print(f"⊘ Config module test failed: {e}")
        raise


def test_database_module_structure():
    """Test database.py module exports expected symbols."""
    try:
        import database
        
        assert hasattr(database, 'Base')
        assert hasattr(database, 'engine')
        assert hasattr(database, 'SessionLocal')
        assert hasattr(database, 'DATABASE_PATH')
        assert hasattr(database, 'init_db')
        assert hasattr(database, 'get_db')
        
        print("✓ Database module has expected structure")
    except ImportError as e:
        if 'sqlalchemy' in str(e):
            print("⊘ Database module test skipped (sqlalchemy not installed)")
        else:
            raise


def test_models_module_structure():
    """Test models.py defines Camera and Run classes."""
    try:
        # This tests structure only; ORM validation requires SQLAlchemy to be installed
        import models
        
        assert hasattr(models, 'Camera')
        assert hasattr(models, 'Run')
        assert hasattr(models, 'Base')
        
        # Check that Camera has expected attributes (as class attributes)
        Camera = models.Camera
        assert hasattr(Camera, '__tablename__')
        assert Camera.__tablename__ == "cameras"
        
        # Check that Run has expected attributes
        Run = models.Run
        assert hasattr(Run, '__tablename__')
        assert Run.__tablename__ == "runs"
        
        print("✓ Models module defines Camera and Run with correct table names")
    except ImportError as e:
        if 'sqlalchemy' in str(e):
            print("⊘ Models module test skipped (sqlalchemy not installed)")
        else:
            raise


def test_run_create_schema():
    """Test RunCreate schema for internal use."""
    from schemas import RunCreate
    
    now = datetime.utcnow()
    run = RunCreate(
        camera_id=1,
        start_time=now,
        end_time=now,
        duration_seconds=300.0,
        total_confirmed_count=25,
        counts_per_class={"box": 12, "sack": 13},
        resource_stats={"cpu": 45.2, "gpu": 87.5},
        status="completed"
    )
    assert run.camera_id == 1
    assert run.total_confirmed_count == 25
    print("✓ RunCreate schema valid")


def test_run_detail_response():
    """Test RunDetail schema (full details)."""
    from schemas import RunDetail
    
    now = datetime.utcnow()
    response = RunDetail(
        id=42,
        camera_id=1,
        camera_name="Test Camera",
        start_time=now,
        end_time=now,
        duration_seconds=300.0,
        total_confirmed_count=25,
        counts_per_class={"box": 12, "sack": 13},
        resource_stats={"cpu": 45.2, "gpu": 87.5},
        status="completed"
    )
    assert response.id == 42
    assert response.total_confirmed_count == 25
    print("✓ RunDetail schema valid")


if __name__ == "__main__":
    # Run all tests
    print("\n" + "=" * 60)
    print("Testing Models and Schemas")
    print("=" * 60 + "\n")
    
    try:
        test_camera_create_schema_valid()
        test_camera_create_schema_rtsps()
        test_camera_create_schema_invalid_url()
        test_camera_create_schema_empty_name()
        test_camera_create_schema_name_too_long()
        test_camera_response_schema()
        test_run_response_schema()
        test_health_check_response()
        test_detection_start_response()
        test_detection_stop_response()
        test_stats_response()
        test_websocket_live_message()
        test_config_imports()
        test_database_module_structure()
        test_models_module_structure()
        test_run_create_schema()
        test_run_detail_response()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
