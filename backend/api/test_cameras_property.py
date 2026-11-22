"""
Property-Based Tests for Camera CRUD endpoints using Hypothesis.

**Property 1: Camera Uniqueness by RTSP URL**
**Validates: Requirement 1.1**

This property test verifies that:
- Creating a camera with RTSP URL X succeeds
- Attempting to create a duplicate camera with the same RTSP URL also succeeds
- However, each camera maintains a unique ID in the database
"""

import pytest
import multiprocessing as mp
from hypothesis import given, strategies as st, settings, HealthCheck
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from database import Base, get_db
import main as main_module


# Test database setup - unique file per module to prevent cross-module conflicts
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_cameras_property.db"
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


# Hypothesis settings for all tests
hypothesis_settings = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    max_examples=5,
)


@given(
    rtsp_url=st.sampled_from([
        "rtsp://192.168.1.100:554/stream",
        "rtsp://camera.local:554/stream",
        "rtsps://secure.example.com:322/stream",
    ])
)
@hypothesis_settings
def test_property_duplicate_rtsp_urls_create_unique_cameras(rtsp_url):
    """
    **Property 1: Camera Uniqueness by RTSP URL**
    **Validates: Requirement 1.1**
    
    Test: Create camera with URL X, attempt to create duplicate → verify entries have unique IDs.
    """
    # Create first camera
    response1 = client.post(
        "/api/cameras",
        json={
            "name": "Camera A",
            "rtsp_url": rtsp_url,
        }
    )
    assert response1.status_code == 201
    camera1_id = response1.json()["id"]
    
    # Create second camera with same RTSP URL
    response2 = client.post(
        "/api/cameras",
        json={
            "name": "Camera B",
            "rtsp_url": rtsp_url,
        }
    )
    assert response2.status_code == 201
    camera2_id = response2.json()["id"]
    
    # Verify unique IDs
    assert camera1_id != camera2_id
    
    # Verify both in list
    list_response = client.get("/api/cameras")
    cameras = {c["id"]: c for c in list_response.json()}
    assert camera1_id in cameras
    assert camera2_id in cameras


@given(
    rtsp_urls=st.lists(
        st.sampled_from([
            "rtsp://192.168.1.100:554/stream",
            "rtsp://192.168.1.101:554/stream",
            "rtsps://secure.example.com:322/stream",
        ]),
        min_size=1,
        max_size=3,
        unique=True
    )
)
@hypothesis_settings
def test_property_multiple_cameras_unique_ids(rtsp_urls):
    """
    **Property 2: Multiple Cameras Have Unique IDs**
    **Validates: Requirement 1.1**
    """
    created_ids = set()
    
    for i, rtsp_url in enumerate(rtsp_urls):
        response = client.post(
            "/api/cameras",
            json={
                "name": f"Camera_{i}",
                "rtsp_url": rtsp_url,
            }
        )
        assert response.status_code == 201
        camera_id = response.json()["id"]
        assert camera_id not in created_ids
        created_ids.add(camera_id)
    
    list_response = client.get("/api/cameras")
    cameras = list_response.json()
    list_ids = {c["id"] for c in cameras}
    assert created_ids.issubset(list_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
