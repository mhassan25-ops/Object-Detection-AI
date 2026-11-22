"""
Test suite for detection_service.py

Tests include:
- Unit tests for registry management functions
- Property-based tests for concurrency limit enforcement
"""

import pytest
from multiprocessing import Manager, Process, Event, Queue
from datetime import datetime
from services.detection_service import (
    get_active_detection_count,
    can_start_detection,
    register_detection,
    unregister_detection,
)


class TestRegistryStateConsistency:
    """Unit tests for registry state consistency."""

    def test_get_active_detection_count_empty_registry(self):
        """Test that empty registry returns 0 active detections."""
        registry = {}
        assert get_active_detection_count(registry) == 0

    def test_get_active_detection_count_single_running(self):
        """Test counting a single running detection."""
        manager = Manager()
        registry = {}
        
        # Create a mock detection entry with running=True
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        registry[1] = {
            "process": None,
            "stop_event": None,
            "frame_queue": None,
            "status": status,
            "start_time": datetime.now(),
        }
        
        assert get_active_detection_count(registry) == 1
        manager.shutdown()

    def test_get_active_detection_count_multiple_running(self):
        """Test counting multiple running detections."""
        manager = Manager()
        registry = {}
        
        # Create two running detection entries
        for camera_id in [1, 2]:
            status = manager.dict()
            status["running"] = True
            status["counts"] = {}
            
            registry[camera_id] = {
                "process": None,
                "stop_event": None,
                "frame_queue": None,
                "status": status,
                "start_time": datetime.now(),
            }
        
        assert get_active_detection_count(registry) == 2
        manager.shutdown()

    def test_get_active_detection_count_mixed_running_state(self):
        """Test counting with mixed running/stopped states."""
        manager = Manager()
        registry = {}
        
        # Create three detection entries, only first two running
        for camera_id in [1, 2, 3]:
            status = manager.dict()
            status["running"] = camera_id < 3  # Only 1 and 2 are running
            status["counts"] = {}
            
            registry[camera_id] = {
                "process": None,
                "stop_event": None,
                "frame_queue": None,
                "status": status,
                "start_time": datetime.now(),
            }
        
        assert get_active_detection_count(registry) == 2
        manager.shutdown()

    def test_can_start_detection_empty_registry(self):
        """Test that new detection can start when registry is empty."""
        registry = {}
        assert can_start_detection(registry) is True

    def test_can_start_detection_one_running(self):
        """Test that new detection can start when 1 is running."""
        manager = Manager()
        registry = {}
        
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        registry[1] = {
            "process": None,
            "stop_event": None,
            "frame_queue": None,
            "status": status,
            "start_time": datetime.now(),
        }
        
        assert can_start_detection(registry) is True
        manager.shutdown()

    def test_can_start_detection_two_running(self):
        """Test that new detection cannot start when 2 are running."""
        manager = Manager()
        registry = {}
        
        for camera_id in [1, 2]:
            status = manager.dict()
            status["running"] = True
            status["counts"] = {}
            
            registry[camera_id] = {
                "process": None,
                "stop_event": None,
                "frame_queue": None,
                "status": status,
                "start_time": datetime.now(),
            }
        
        assert can_start_detection(registry) is False
        manager.shutdown()

    def test_register_detection_adds_to_registry(self):
        """Test that registering a detection adds it to the registry."""
        manager = Manager()
        registry = {}
        
        camera_id = 1
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        start_time = datetime.now()
        
        register_detection(
            registry=registry,
            camera_id=camera_id,
            process=None,
            stop_event=None,
            frame_queue=None,
            status_entry=status,
            start_time=start_time,
        )
        
        assert camera_id in registry
        assert registry[camera_id]["status"]["running"] is True
        assert registry[camera_id]["start_time"] == start_time
        manager.shutdown()

    def test_register_detection_has_all_required_fields(self):
        """Test that registered detection has all required fields."""
        manager = Manager()
        registry = {}
        
        camera_id = 1
        stop_event = manager.Event()
        frame_queue = manager.Queue()
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        start_time = datetime.now()
        
        register_detection(
            registry=registry,
            camera_id=camera_id,
            process=None,
            stop_event=stop_event,
            frame_queue=frame_queue,
            status_entry=status,
            start_time=start_time,
        )
        
        entry = registry[camera_id]
        assert "process" in entry
        assert "stop_event" in entry
        assert "frame_queue" in entry
        assert "status" in entry
        assert "start_time" in entry
        
        # Verify they are correct objects
        assert entry["stop_event"] is stop_event
        assert entry["frame_queue"] is frame_queue
        assert entry["status"] is status
        
        manager.shutdown()

    def test_unregister_detection_removes_from_registry(self):
        """Test that unregistering removes a detection from registry."""
        manager = Manager()
        registry = {}
        
        camera_id = 1
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        register_detection(
            registry=registry,
            camera_id=camera_id,
            process=None,
            stop_event=None,
            frame_queue=None,
            status_entry=status,
            start_time=datetime.now(),
        )
        
        assert camera_id in registry
        
        unregister_detection(registry, camera_id)
        
        assert camera_id not in registry
        manager.shutdown()

    def test_unregister_detection_idempotent(self):
        """Test that unregistering a non-existent camera doesn't error."""
        registry = {}
        
        # Should not raise
        unregister_detection(registry, 999)
        assert 999 not in registry

    def test_registry_state_consistency_start_stop_cycle(self):
        """Test registry state consistency through start/stop cycles."""
        manager = Manager()
        registry = {}
        
        # Simulate 3 start/stop cycles
        for cycle in range(3):
            camera_id = cycle + 1
            
            # Start detection
            status = manager.dict()
            status["running"] = True
            status["counts"] = {"box": 0, "sack": 0}
            
            register_detection(
                registry=registry,
                camera_id=camera_id,
                process=None,
                stop_event=None,
                frame_queue=None,
                status_entry=status,
                start_time=datetime.now(),
            )
            
            # Verify state
            assert get_active_detection_count(registry) == 1
            assert can_start_detection(registry) is True
            
            # Stop detection
            unregister_detection(registry, camera_id)
            
            # Verify state
            assert get_active_detection_count(registry) == 0
            assert can_start_detection(registry) is True
        
        manager.shutdown()


class TestConcurrencyLimitEnforcement:
    """Property-based tests for concurrency limit enforcement.
    
    **Validates: Requirement 15**
    
    Test: Start 2 detections successfully; 3rd attempt returns 409; 
          stop one; can start a new one.
    """

    def test_concurrency_start_two_successfully(self):
        """Test starting first 2 detections succeeds."""
        manager = Manager()
        registry = {}
        
        # Start 2 detections
        for camera_id in [1, 2]:
            status = manager.dict()
            status["running"] = True
            status["counts"] = {}
            
            # Can start before registration
            assert can_start_detection(registry) is True
            
            register_detection(
                registry=registry,
                camera_id=camera_id,
                process=None,
                stop_event=None,
                frame_queue=None,
                status_entry=status,
                start_time=datetime.now(),
            )
        
        # Verify both are active
        assert get_active_detection_count(registry) == 2
        
        manager.shutdown()

    def test_concurrency_third_attempt_fails(self):
        """Test that 3rd detection start is rejected (returns 409 state)."""
        manager = Manager()
        registry = {}
        
        # Start 2 detections
        for camera_id in [1, 2]:
            status = manager.dict()
            status["running"] = True
            status["counts"] = {}
            
            register_detection(
                registry=registry,
                camera_id=camera_id,
                process=None,
                stop_event=None,
                frame_queue=None,
                status_entry=status,
                start_time=datetime.now(),
            )
        
        # Verify can_start_detection returns False for 3rd
        assert can_start_detection(registry) is False
        
        manager.shutdown()

    def test_concurrency_stop_one_allows_new_start(self):
        """Test that stopping one detection allows starting a new one."""
        manager = Manager()
        registry = {}
        
        # Start 2 detections
        for camera_id in [1, 2]:
            status = manager.dict()
            status["running"] = True
            status["counts"] = {}
            
            register_detection(
                registry=registry,
                camera_id=camera_id,
                process=None,
                stop_event=None,
                frame_queue=None,
                status_entry=status,
                start_time=datetime.now(),
            )
        
        # Verify can't start new one
        assert can_start_detection(registry) is False
        
        # Stop one
        unregister_detection(registry, 1)
        
        # Now can start new one
        assert can_start_detection(registry) is True
        assert get_active_detection_count(registry) == 1
        
        # Start new one
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        register_detection(
            registry=registry,
            camera_id=3,
            process=None,
            stop_event=None,
            frame_queue=None,
            status_entry=status,
            start_time=datetime.now(),
        )
        
        # Verify now have 2 again
        assert get_active_detection_count(registry) == 2
        
        manager.shutdown()

    def test_concurrency_never_exceeds_two(self):
        """Test property: active detection count never exceeds 2."""
        manager = Manager()
        registry = {}
        
        # Simulate 5 attempt sequences
        active_cameras = set()
        
        for attempt in range(5):
            if can_start_detection(registry):
                camera_id = attempt + 1
                status = manager.dict()
                status["running"] = True
                status["counts"] = {}
                
                register_detection(
                    registry=registry,
                    camera_id=camera_id,
                    process=None,
                    stop_event=None,
                    frame_queue=None,
                    status_entry=status,
                    start_time=datetime.now(),
                )
                
                active_cameras.add(camera_id)
            
            # PROPERTY: count never exceeds 2
            count = get_active_detection_count(registry)
            assert count <= 2, f"Active count {count} exceeds max of 2"
            
            # Stop oldest if we have 2
            if len(active_cameras) == 2:
                oldest = min(active_cameras)
                unregister_detection(registry, oldest)
                active_cameras.remove(oldest)
        
        manager.shutdown()

    def test_concurrency_max_capacity_boundary(self):
        """Test concurrency enforcement at max capacity boundary."""
        manager = Manager()
        registry = {}
        
        # Start exactly 2
        for camera_id in [1, 2]:
            status = manager.dict()
            status["running"] = True
            status["counts"] = {}
            
            assert can_start_detection(registry) is True
            
            register_detection(
                registry=registry,
                camera_id=camera_id,
                process=None,
                stop_event=None,
                frame_queue=None,
                status_entry=status,
                start_time=datetime.now(),
            )
        
        # Now at capacity
        assert get_active_detection_count(registry) == 2
        assert can_start_detection(registry) is False
        
        # Unregister one
        unregister_detection(registry, 1)
        
        # Back below capacity
        assert get_active_detection_count(registry) == 1
        assert can_start_detection(registry) is True
        
        # Start new one
        status = manager.dict()
        status["running"] = True
        status["counts"] = {}
        
        register_detection(
            registry=registry,
            camera_id=3,
            process=None,
            stop_event=None,
            frame_queue=None,
            status_entry=status,
            start_time=datetime.now(),
        )
        
        assert get_active_detection_count(registry) == 2
        assert can_start_detection(registry) is False
        
        manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
