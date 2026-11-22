"""
Root conftest.py for the backend test suite.

Cleans up stale test database files before the test session starts,
ensuring full test isolation when running all test modules together.
"""

import os
import pytest


def pytest_sessionstart(session):
    """Remove stale test database files at session start."""
    stale_dbs = [
        "test_cameras.db",
        "test_cameras_property.db",
    ]
    for db_file in stale_dbs:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass  # If removal fails, the fixture's drop_all will handle it
