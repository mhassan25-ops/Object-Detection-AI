"""
Unit tests for Camera CRUD endpoints: GET /api/cameras, POST /api/cameras, DELETE /api/cameras/{id}

Tests cover:
- Successful camera creation with valid inputs
- Successful camera listing with status determination
- Successful camera deletion when idle
- Validation errors for invalid inputs
- Conflict errors when attempting to delete while detecting
- 404 Not Found errors for nonexistent cameras
"""

import pytest
import multiprocessing as mp
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from main import app
from database import Base, get_db
from models import Camera, Run
import main as main_module


# Test database setup - unique file per module to prevent cross-module conflicts
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_cameras.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override get_db dependency with test database."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# Initialize multiprocessing manager for tests if not already done
@pytest.fixture(scope="session", autouse=True)
def init_test_registry():
    """Initialize the multiprocessing registry for tests."""
    if main_module.manager is None:
        main_module.manager = mp.Manager()
        main_module.registry = main_module.manager.dict()
        main_module.status_dict = main_module.manager.dict()
    yield
    # Cleanup is handled by OS


@pytest.fixture(autouse=True)
def setup_teardown():
    """Clear database before and after each test."""
    # Re-apply our override to ensure this module's engine is used
    app.dependency_overrides[get_db] = override_get_db
    # Clear before
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    if main_module.registry is not None:
        main_module.registry.clear()
    yield
    # Clear after
    Base.metadata.drop_all(bind=engine)
    if main_module.registry is not None:
        main_module.registry.clear()


# ===== Tests for POST /api/cameras (Create Camera) =====

def test_create_camera_with_valid_inputs():
    """Test successful camera creation with valid name and RTSP URL."""
    response = client.post(
        "/api/cameras",
        json={
            "name": "Factory Floor Camera 1",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Factory Floor Camera 1"
    assert data["rtsp_url"] == "rtsp://192.168.1.100:554/stream"
    assert data["status"] == "idle"
    assert data["id"] is not None
    assert data["created_at"] is not None


def test_create_camera_with_rtsps_url():
    """Test camera creation with secure RTSP URL (rtsps://)."""
    response = client.post(
        "/api/cameras",
        json={
            "name": "Secure Camera",
            "rtsp_url": "rtsps://secure.example.com:322/stream",
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["rtsp_url"] == "rtsps://secure.example.com:322/stream"


def test_create_camera_with_empty_name():
    """Test camera creation fails with empty name."""
    response = client.post(
        "/api/cameras",
        json={
            "name": "",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_create_camera_with_name_exceeding_255_chars():
    """Test camera creation fails when name exceeds 255 characters."""
    long_name = "a" * 256
    response = client.post(
        "/api/cameras",
        json={
            "name": long_name,
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_create_camera_with_invalid_rtsp_format_http():
    """Test camera creation fails when RTSP URL format is invalid (http://)."""
    response = client.post(
        "/api/cameras",
        json={
            "name": "Bad Camera",
            "rtsp_url": "http://192.168.1.100:554/stream",
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_create_camera_with_invalid_rtsp_format_no_protocol():
    """Test camera creation fails when RTSP URL has no protocol."""
    response = client.post(
        "/api/cameras",
        json={
            "name": "Bad Camera",
            "rtsp_url": "192.168.1.100:554/stream",
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_create_camera_with_missing_name():
    """Test camera creation fails when name is missing."""
    response = client.post(
        "/api/cameras",
        json={
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_create_camera_with_missing_rtsp_url():
    """Test camera creation fails when RTSP URL is missing."""
    response = client.post(
        "/api/cameras",
        json={
            "name": "Test Camera",
        }
    )
    
    assert response.status_code == 422  # Validation error


# ===== Tests for GET /api/cameras (List Cameras) =====

def test_list_cameras_empty():
    """Test listing cameras when database is empty."""
    response = client.get("/api/cameras")
    
    assert response.status_code == 200
    data = response.json()
    assert data == []


def test_list_cameras_with_single_idle_camera():
    """Test listing cameras with one idle camera."""
    # Create a camera
    client.post(
        "/api/cameras",
        json={
            "name": "Camera 1",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    
    # List cameras
    response = client.get("/api/cameras")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Camera 1"
    assert data[0]["status"] == "idle"


def test_list_cameras_with_multiple_cameras():
    """Test listing multiple cameras."""
    # Create 3 cameras
    for i in range(3):
        client.post(
            "/api/cameras",
            json={
                "name": f"Camera {i+1}",
                "rtsp_url": f"rtsp://192.168.1.{100+i}:554/stream",
            }
        )
    
    # List cameras
    response = client.get("/api/cameras")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    for i, camera in enumerate(data):
        assert camera["name"] == f"Camera {i+1}"
        assert camera["status"] == "idle"


def test_list_cameras_response_structure():
    """Test that list response includes all required fields."""
    # Create a camera
    create_response = client.post(
        "/api/cameras",
        json={
            "name": "Test Camera",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    camera_id = create_response.json()["id"]
    
    # List cameras
    response = client.get("/api/cameras")
    data = response.json()
    
    assert len(data) == 1
    camera = data[0]
    assert "id" in camera
    assert "name" in camera
    assert "rtsp_url" in camera
    assert "status" in camera
    assert "created_at" in camera
    assert camera["id"] == camera_id


# ===== Tests for DELETE /api/cameras/{id} (Delete Camera) =====

def test_delete_camera_success():
    """Test successful camera deletion."""
    # Create a camera
    create_response = client.post(
        "/api/cameras",
        json={
            "name": "Camera to Delete",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    camera_id = create_response.json()["id"]
    
    # Delete the camera
    response = client.delete(f"/api/cameras/{camera_id}")
    
    assert response.status_code == 204
    
    # Verify camera is gone
    list_response = client.get("/api/cameras")
    assert len(list_response.json()) == 0


def test_delete_nonexistent_camera():
    """Test deletion of nonexistent camera returns 404."""
    response = client.delete("/api/cameras/9999")
    
    assert response.status_code == 404
    assert "Camera not found" in response.json()["detail"]


def test_delete_camera_conflict_when_detecting():
    """Test deletion fails with 409 when camera is detecting.
    
    This test mocks an active detection by inserting a status entry
    into the registry with running=True.
    """
    # Create a camera
    create_response = client.post(
        "/api/cameras",
        json={
            "name": "Detecting Camera",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    camera_id = create_response.json()["id"]
    
    # Simulate active detection by adding to registry
    assert main_module.registry is not None, "Registry should be initialized"
    
    main_module.registry[camera_id] = {
        "process": None,
        "stop_event": None,
        "frame_queue": None,
        "status": {
            "running": True,
            "counts": {},
        },
    }
    
    # Attempt to delete camera
    response = client.delete(f"/api/cameras/{camera_id}")
    
    assert response.status_code == 409
    assert "Cannot delete camera" in response.json()["detail"]
    
    # Clean up registry
    del main_module.registry[camera_id]


def test_delete_camera_success_after_detection_stops():
    """Test that camera can be deleted once detection stops.
    
    This verifies the conflict check is based on running status,
    not just presence in registry.
    """
    # Create a camera
    create_response = client.post(
        "/api/cameras",
        json={
            "name": "Camera After Detection",
            "rtsp_url": "rtsp://192.168.1.100:554/stream",
        }
    )
    camera_id = create_response.json()["id"]
    
    # Simulate completed detection (running=False)
    assert main_module.registry is not None, "Registry should be initialized"
    
    main_module.registry[camera_id] = {
        "process": None,
        "stop_event": None,
        "frame_queue": None,
        "status": {
            "running": False,
            "counts": {"box": 5},
        },
    }
    
    # Attempt to delete camera (should succeed)
    response = client.delete(f"/api/cameras/{camera_id}")
    
    assert response.status_code == 204
    
    # Clean up registry
    del main_module.registry[camera_id]


# ===== Property-Based Tests (using pytest parametrize) =====

class TestCameraValidation:
    """Property-based validation tests for camera inputs."""
    
    @pytest.mark.parametrize("name_length", [1, 10, 100, 255])
    def test_valid_name_lengths(self, name_length):
        """Property 1.1: All valid name lengths (1-255) are accepted."""
        name = "a" * name_length
        response = client.post(
            "/api/cameras",
            json={
                "name": name,
                "rtsp_url": "rtsp://192.168.1.100:554/stream",
            }
        )
        
        assert response.status_code == 201
        assert response.json()["name"] == name
    
    @pytest.mark.parametrize("invalid_length", [0, 256, 500])
    def test_invalid_name_lengths(self, invalid_length):
        """Property 1.2: Invalid name lengths (0 or >255) are rejected."""
        name = "a" * invalid_length
        response = client.post(
            "/api/cameras",
            json={
                "name": name,
                "rtsp_url": "rtsp://192.168.1.100:554/stream",
            }
        )
        
        assert response.status_code == 422
    
    @pytest.mark.parametrize("valid_url", [
        "rtsp://192.168.1.100:554/stream",
        "rtsp://camera.local:554/stream",
        "rtsps://secure.camera.local:322/stream",
        "rtsp://192.168.1.1/",
        "rtsps://192.168.1.1:554/path/to/stream",
    ])
    def test_valid_rtsp_urls(self, valid_url):
        """Property 1.3: All valid RTSP URLs are accepted."""
        response = client.post(
            "/api/cameras",
            json={
                "name": "Test Camera",
                "rtsp_url": valid_url,
            }
        )
        
        assert response.status_code == 201
        assert response.json()["rtsp_url"] == valid_url
    
    @pytest.mark.parametrize("invalid_url", [
        "http://192.168.1.100:554/stream",
        "https://192.168.1.100:554/stream",
        "192.168.1.100:554/stream",
        "ftp://192.168.1.100:554/stream",
        "rtps://192.168.1.100:554/stream",  # typo: rtps instead of rtsp
    ])
    def test_invalid_rtsp_urls(self, invalid_url):
        """Property 1.4: All invalid RTSP URLs are rejected."""
        response = client.post(
            "/api/cameras",
            json={
                "name": "Test Camera",
                "rtsp_url": invalid_url,
            }
        )
        
        assert response.status_code == 422


class TestCameraUniqueness:
    """Tests for camera uniqueness and database constraints.
    
    Note: The design doesn't specify unique constraint on RTSP URL,
    but this test verifies that duplicate URLs can be created
    (if needed, constraint can be added later).
    """
    
    def test_duplicate_rtsp_urls_allowed(self):
        """Verify that duplicate RTSP URLs can be created (no unique constraint).
        
        This is a deliberate design choice - the spec doesn't require
        RTSP URL uniqueness, so multiple cameras can point to the same stream.
        """
        rtsp_url = "rtsp://192.168.1.100:554/stream"
        
        # Create first camera
        response1 = client.post(
            "/api/cameras",
            json={
                "name": "Camera 1",
                "rtsp_url": rtsp_url,
            }
        )
        assert response1.status_code == 201
        
        # Create second camera with same RTSP URL
        response2 = client.post(
            "/api/cameras",
            json={
                "name": "Camera 2",
                "rtsp_url": rtsp_url,
            }
        )
        assert response2.status_code == 201
        
        # Both should have different IDs
        camera1_id = response1.json()["id"]
        camera2_id = response2.json()["id"]
        assert camera1_id != camera2_id
        
        # Both should be in the list
        list_response = client.get("/api/cameras")
        assert len(list_response.json()) == 2
    
    def test_camera_ids_are_unique(self):
        """Verify that created cameras have unique IDs."""
        ids = set()
        
        for i in range(5):
            response = client.post(
                "/api/cameras",
                json={
                    "name": f"Camera {i}",
                    "rtsp_url": f"rtsp://192.168.1.{100+i}:554/stream",
                }
            )
            assert response.status_code == 201
            camera_id = response.json()["id"]
            
            # ID should be unique
            assert camera_id not in ids
            ids.add(camera_id)
        
        assert len(ids) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
