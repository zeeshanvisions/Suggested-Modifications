# Task Completion API Controller

## File: `backend/app/controller/task_completion_controller.py` (NEW)

```python
"""API endpoints for task completion and knowledge retrieval."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from loguru import logger
from app.service.knowledge_retrieval_service import (
    get_task_by_id,
    get_workflow_tasks,
    get_previous_knowledge,
    format_knowledge_for_prompt,
)
from app.model.task_completion import TaskCompletionOut

router = APIRouter()


@router.get("/task-completion/{task_id}", response_model=TaskCompletionOut)
async def get_task_completion(task_id: str):
    """
    Retrieve a specific task completion record by task_id.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        TaskCompletion record
    """
    task = get_task_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@router.get("/task-completion/workflow/{workflow_id}", response_model=List[TaskCompletionOut])
async def get_workflow_completions(
    workflow_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get all task completions for a workflow.
    
    Args:
        workflow_id: Workflow identifier
        limit: Maximum number of records to return
        offset: Offset for pagination
        
    Returns:
        List of TaskCompletion records
    """
    tasks = get_workflow_tasks(workflow_id, limit=limit, offset=offset)
    return tasks


@router.get("/task-completion/knowledge/{workflow_id}")
async def get_workflow_knowledge(
    workflow_id: str,
    limit: int = Query(default=5, ge=1, le=20),
    formatted: bool = Query(default=False, description="Return formatted for prompt injection"),
):
    """
    Get knowledge summaries for a workflow.
    
    Args:
        workflow_id: Workflow identifier
        limit: Maximum number of previous tasks to retrieve
        formatted: If True, return formatted string for prompt injection
        
    Returns:
        Either list of TaskCompletion records or formatted string
    """
    knowledge = get_previous_knowledge(workflow_id, limit=limit)
    
    if formatted:
        return {
            "knowledge": format_knowledge_for_prompt(knowledge),
            "count": len(knowledge),
        }
    else:
        return {
            "tasks": knowledge,
            "count": len(knowledge),
        }


@router.get("/task-completion/knowledge/{workflow_id}/context")
async def get_knowledge_context(
    workflow_id: str,
    current_question: Optional[str] = Query(default=None),
    limit: int = Query(default=3, ge=1, le=10),
):
    """
    Get formatted knowledge context for prompt injection.
    
    Args:
        workflow_id: Workflow identifier
        current_question: Current task question (for future similarity matching)
        limit: Maximum number of previous tasks to include
        
    Returns:
        Formatted knowledge context string
    """
    from app.service.knowledge_retrieval_service import get_knowledge_context
    
    context = get_knowledge_context(workflow_id, current_question=current_question, limit=limit)
    
    return {
        "context": context,
        "workflow_id": workflow_id,
        "has_knowledge": len(context) > 0,
    }
```

## Router Registration

The router will be automatically registered by the `auto_include_routers` function in `app/__init__.py` since it follows the naming convention `*_controller.py`.

## API Endpoints Summary

1. **GET `/api/task-completion/{task_id}`**
   - Retrieve specific task completion record
   - Returns: `TaskCompletionOut`

2. **GET `/api/task-completion/workflow/{workflow_id}`**
   - Get all tasks for a workflow
   - Query params: `limit`, `offset`
   - Returns: `List[TaskCompletionOut]`

3. **GET `/api/task-completion/knowledge/{workflow_id}`**
   - Get knowledge summaries for a workflow
   - Query params: `limit`, `formatted`
   - Returns: Knowledge data (formatted or raw)

4. **GET `/api/task-completion/knowledge/{workflow_id}/context`**
   - Get formatted knowledge context for prompt injection
   - Query params: `current_question`, `limit`
   - Returns: Formatted context string

## Example Usage

```bash
# Get specific task
curl http://localhost:5001/api/task-completion/task-123

# Get all tasks for workflow
curl http://localhost:5001/api/task-completion/workflow/workflow-456?limit=10

# Get knowledge summaries
curl http://localhost:5001/api/task-completion/knowledge/workflow-456?limit=5&formatted=true

# Get knowledge context
curl "http://localhost:5001/api/task-completion/knowledge/workflow-456/context?limit=3"
```

