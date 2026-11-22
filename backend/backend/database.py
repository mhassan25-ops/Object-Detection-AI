"""
Database initialization and session management for SQLite using SQLAlchemy.
Provides Base class, session factory, and init_db() function.
"""

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# Database URL (SQLite)
DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(DATABASE_DIR, "data.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# SQLAlchemy engine configuration
# Use StaticPool for SQLite to ensure connections are reused in-memory
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,  # Set to True for SQL debug logging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for all ORM models
Base = declarative_base()


def init_db():
    """
    Initialize the database by creating all tables defined in the ORM models.
    Called on application startup.
    """
    try:
        # Import models to ensure they are registered with Base
        from models import Camera, Run  # noqa: F401
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized at {DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise


def get_db():
    """
    Dependency injection function for FastAPI to get a database session.
    Usage: def my_endpoint(db: Session = Depends(get_db)):
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
