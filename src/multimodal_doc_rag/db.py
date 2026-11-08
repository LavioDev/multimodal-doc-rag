"""Synchronous PostgreSQL connection and readiness checks."""

from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session

from multimodal_doc_rag.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create one SQLAlchemy engine for the configured database."""
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def check_database(engine: Engine | None = None) -> None:
    """Raise when PostgreSQL or its pgvector extension is unavailable."""
    database = engine or get_engine()
    with database.connect() as connection:
        connection.execute(text("SELECT 1"))
        extension = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        if extension is None:
            raise RuntimeError("PostgreSQL extension 'vector' is not installed")


@contextmanager
def session_scope():
    """Commit one unit of work or roll it back, always releasing the connection."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
