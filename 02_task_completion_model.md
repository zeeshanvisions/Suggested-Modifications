# Task Completion Model

## File: `backend/app/model/task_completion.py` (NEW)

```python
"""Task completion database model."""
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from sqlmodel import Field, SQLModel, Column, JSON, String, Text, Integer, Float, DateTime
from sqlalchemy import Enum as SQLEnum


class TaskStatus(str, Enum):
    """Task completion status."""
    completed = "completed"
    failed = "failed"


class TaskCompletion(SQLModel, table=True):
    """Task completion record with full task state and LLM summaries."""
    
    __tablename__ = "task_completion"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(index=True, unique=True, max_length=255, description="Unique task identifier")
    workflow_id: str = Field(index=True, max_length=255, description="Workflow identifier for grouping tasks")
    user_email: str = Field(index=True, max_length=255, description="Email of user who created the task")
    
    # Full task state stored as JSON
    task_data: dict[str, Any] = Field(
        sa_column=Column(JSON),
        description="Complete task state including messages, taskInfo, taskRunning, taskAssigning, etc."
    )
    
    original_question: str = Field(sa_column=Column(Text), description="Original user query/question")
    
    # LLM-generated summaries (nullable initially, populated after processing)
    summary_steps: Optional[str] = Field(default=None, sa_column=Column(Text), description="LLM-generated summary of all execution steps")
    summary_dos_donts: Optional[str] = Field(default=None, sa_column=Column(Text), description="LLM-generated do's and don'ts")
    summary_learnings: Optional[str] = Field(default=None, sa_column=Column(Text), description="LLM-generated learnings and insights")
    summary_expected_outcome: Optional[str] = Field(default=None, sa_column=Column(Text), description="LLM-generated expected outcome description")
    
    status: TaskStatus = Field(
        default=TaskStatus.completed,
        sa_column=Column(SQLEnum(TaskStatus)),
        description="Task completion status"
    )
    
    tokens_used: int = Field(default=0, description="Total tokens consumed during task execution")
    execution_time: float = Field(default=0.0, description="Task execution time in seconds")
    
    created_at: datetime = Field(default_factory=datetime.now, sa_column=Column(DateTime), description="When task was created")
    completed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime), description="When task was completed")
    updated_at: datetime = Field(
        default_factory=datetime.now,
        sa_column=Column(DateTime),
        description="Last update timestamp"
    )


class TaskCompletionCreate(SQLModel):
    """Schema for creating a task completion record."""
    task_id: str
    workflow_id: str
    user_email: str
    task_data: dict[str, Any]
    original_question: str
    status: TaskStatus = TaskStatus.completed
    tokens_used: int = 0
    execution_time: float = 0.0


class TaskCompletionUpdate(SQLModel):
    """Schema for updating task completion summaries."""
    summary_steps: Optional[str] = None
    summary_dos_donts: Optional[str] = None
    summary_learnings: Optional[str] = None
    summary_expected_outcome: Optional[str] = None
    status: Optional[TaskStatus] = None


class TaskCompletionOut(SQLModel):
    """Schema for returning task completion data."""
    id: int
    task_id: str
    workflow_id: str
    user_email: str
    original_question: str
    summary_steps: Optional[str]
    summary_dos_donts: Optional[str]
    summary_learnings: Optional[str]
    summary_expected_outcome: Optional[str]
    status: TaskStatus
    tokens_used: int
    execution_time: float
    created_at: datetime
    completed_at: Optional[datetime]
    updated_at: datetime
```

