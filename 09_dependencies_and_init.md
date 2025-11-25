# Dependencies and Initialization

## Dependencies Update

### File: `backend/pyproject.toml` (MODIFY)

Add to the `dependencies` list:

```toml
dependencies = [
    # ... existing dependencies ...
    "sqlmodel>=0.0.24",
    "psycopg2-binary>=2.9.10",
]
```

**Complete dependencies section:**
```toml
dependencies = [
    "camel-ai[eigent]>=0.2.76a6",
    "fastapi>=0.115.12",
    "fastapi-babel>=1.0.0",
    "uvicorn[standard]>=0.34.2",
    "pydantic-i18n>=0.4.5",
    "python-dotenv>=1.1.0",
    "httpx[socks]>=0.28.1",
    "loguru>=0.7.3",
    "pydash>=8.0.5",
    "inflection>=0.5.1",
    "aiofiles>=24.1.0",
    "openai>=1.99.3,<2",
    "traceroot>=0.0.4a9",
    "nodejs-wheel>=22.18.0",
    "sqlmodel>=0.0.24",
    "psycopg2-binary>=2.9.10",
]
```

## Database Initialization

### File: `backend/app/__init__.py` (MODIFY)

Add database table creation on startup:

**BEFORE:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI with title
api = FastAPI(title="HachiAI Multi-Agent System API")

# Add CORS middleware
api.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
```

**AFTER:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel
from loguru import logger

# Import models to register them
from app.model.task_completion import TaskCompletion
from app.component.database import engine

# Initialize FastAPI with title
api = FastAPI(title="HachiAI Multi-Agent System API")

# Add CORS middleware
api.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


@api.on_event("startup")
async def startup_event():
    """Create database tables on startup."""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
```

## Environment Configuration

### File: `.env` (ADD/UPDATE)

Add database configuration:

```bash
# Database URL
# For PostgreSQL:
database_url=postgresql://user:password@localhost:5432/hachiai

# For SQLite (development):
# database_url=sqlite:///./task_completion.db

# Debug mode (enables SQL echo)
debug=off
```

## Installation Steps

1. **Install dependencies:**
```bash
cd backend
uv sync
# or
pip install sqlmodel psycopg2-binary
```

2. **Set up database:**
   - For PostgreSQL: Create database and update connection string
   - For SQLite: Database file will be created automatically

3. **Run application:**
```bash
uv run uvicorn main:api --port 5001
```

4. **Verify tables:**
   - Check logs for "Database tables created/verified successfully"
   - Or query database directly to verify `task_completion` table exists

## Database Migration (Optional)

For production, consider using Alembic for migrations:

1. **Initialize Alembic:**
```bash
alembic init alembic
```

2. **Create migration:**
```bash
alembic revision --autogenerate -m "Add task_completion table"
```

3. **Apply migration:**
```bash
alembic upgrade head
```

## Testing Database Connection

Create a test script `test_db.py`:

```python
from app.component.database import engine, session_make
from app.model.task_completion import TaskCompletion
from sqlmodel import SQLModel

# Create tables
SQLModel.metadata.create_all(engine)

# Test connection
with session_make() as session:
    count = session.query(TaskCompletion).count()
    print(f"Task completion records: {count}")
```

Run:
```bash
python test_db.py
```

