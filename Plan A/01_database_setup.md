# Database Setup

## File: `backend/app/component/database.py` (NEW)

```python
"""Database connection and session management."""
from sqlmodel import Session, create_engine
from app.component.environment import env, env_or_fail
from loguru import logger

# Get database URL from environment, default to SQLite for development
database_url = env("database_url", "sqlite:///./task_completion.db")

# Create database engine
engine = create_engine(
    database_url,
    echo=env("debug") == "on",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
)

logger.info(f"Database engine created with URL: {database_url.split('@')[-1] if '@' in database_url else database_url}")


def session_make() -> Session:
    """Create a new database session."""
    return Session(engine)


def session():
    """Context manager for database session."""
    with Session(engine) as session:
        yield session
```

## Environment Variable

Add to `.env`:
```bash
database_url=postgresql://user:password@localhost:5432/hachiai
# Or for SQLite:
# database_url=sqlite:///./task_completion.db
```

