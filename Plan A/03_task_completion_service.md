# Task Completion Service

## File: `backend/app/service/task_completion_service.py` (NEW)

```python
"""Service for detecting and saving task completion."""
import asyncio
from datetime import datetime
from typing import Any, Optional
from loguru import logger
from sqlmodel import Session, select
from app.component.database import session_make
from app.model.task_completion import TaskCompletion, TaskCompletionCreate, TaskStatus
from app.service.task import TaskLock


async def save_task_completion(
    task_id: str,
    workflow_id: str,
    user_email: str,
    task_data: dict[str, Any],
    original_question: str,
    tokens_used: int = 0,
    execution_time: float = 0.0,
    status: TaskStatus = TaskStatus.completed,
) -> Optional[TaskCompletion]:
    """
    Save task completion to database.
    
    Args:
        task_id: Unique task identifier
        workflow_id: Workflow identifier
        user_email: User email
        task_data: Full task state as dictionary
        original_question: Original user query
        tokens_used: Total tokens consumed
        execution_time: Execution time in seconds
        status: Task completion status
        
    Returns:
        TaskCompletion record or None if save failed
    """
    try:
        with session_make() as session:
            # Check if task already exists
            existing = session.exec(
                select(TaskCompletion).where(TaskCompletion.task_id == task_id)
            ).first()
            
            if existing:
                logger.warning(f"Task {task_id} already exists in database, skipping save")
                return existing
            
            # Create new record
            task_completion = TaskCompletion(
                task_id=task_id,
                workflow_id=workflow_id,
                user_email=user_email,
                task_data=task_data,
                original_question=original_question,
                status=status,
                tokens_used=tokens_used,
                execution_time=execution_time,
                created_at=datetime.now(),
                completed_at=datetime.now(),
                updated_at=datetime.now(),
            )
            
            session.add(task_completion)
            session.commit()
            session.refresh(task_completion)
            
            logger.info(f"Task completion saved: {task_id} for workflow {workflow_id}")
            return task_completion
            
    except Exception as e:
        logger.error(f"Error saving task completion {task_id}: {e}", exc_info=True)
        return None


def extract_task_data_from_lock(task_lock: TaskLock, options: Any) -> dict[str, Any]:
    """
    Extract full task state from TaskLock and options.
    
    Note: This is a placeholder. The actual implementation needs to
    collect task data from the frontend store or from accumulated
    state during task execution.
    
    Args:
        task_lock: TaskLock instance
        options: Chat options containing task information
        
    Returns:
        Dictionary containing full task state
    """
    # This is a simplified version. In practice, you'll need to:
    # 1. Collect messages from task execution
    # 2. Collect taskInfo, taskRunning, taskAssigning from workforce
    # 3. Collect fileList, webViewUrls, snapshots, etc.
    
    return {
        "task_id": task_lock.id,
        "status": task_lock.status.value if hasattr(task_lock.status, 'value') else str(task_lock.status),
        "active_agent": task_lock.active_agent,
        "created_at": task_lock.created_at.isoformat() if hasattr(task_lock.created_at, 'isoformat') else str(task_lock.created_at),
        "original_question": options.question if hasattr(options, 'question') else "",
        # Add more fields as needed
    }


async def handle_task_completion(
    task_lock: TaskLock,
    options: Any,
    workflow_id: str,
    tokens_used: int = 0,
    execution_time: float = 0.0,
) -> Optional[TaskCompletion]:
    """
    Handle task completion: extract data and save to database.
    
    Args:
        task_lock: TaskLock instance
        options: Chat options
        workflow_id: Workflow identifier
        tokens_used: Total tokens consumed
        execution_time: Execution time in seconds
        
    Returns:
        TaskCompletion record or None
    """
    try:
        task_data = extract_task_data_from_lock(task_lock, options)
        user_email = options.email if hasattr(options, 'email') else ""
        original_question = options.question if hasattr(options, 'question') else ""
        
        return await save_task_completion(
            task_id=task_lock.id,
            workflow_id=workflow_id,
            user_email=user_email,
            task_data=task_data,
            original_question=original_question,
            tokens_used=tokens_used,
            execution_time=execution_time,
            status=TaskStatus.completed,
        )
    except Exception as e:
        logger.error(f"Error handling task completion: {e}", exc_info=True)
        return None
```

## Note on Task Data Extraction

The `extract_task_data_from_lock` function is a placeholder. In practice, you'll need to:

1. **Collect from Frontend**: The frontend store (`chatStore`) has the complete task state. You may need to:
   - Add an API endpoint to retrieve task state from frontend
   - Or accumulate state during backend execution

2. **Collect from Backend**: Accumulate task data during execution:
   - Messages from agent responses
   - TaskInfo from workforce
   - TaskRunning from task state updates
   - TaskAssigning from agent assignments
   - File operations, webview URLs, snapshots, etc.

3. **Hybrid Approach**: Combine both - use backend accumulated data and request final state from frontend if needed.

