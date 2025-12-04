# Knowledge Retrieval Service

## File: `backend/app/service/knowledge_retrieval_service.py` (NEW)

```python
"""Service for retrieving previous task knowledge."""
from typing import Optional, List
from loguru import logger
from sqlmodel import Session, select, func
from app.component.database import session_make
from app.model.task_completion import TaskCompletion


def get_previous_knowledge(
    workflow_id: str,
    limit: int = 5,
) -> List[TaskCompletion]:
    """
    Retrieve previous task knowledge for a workflow.
    
    Args:
        workflow_id: Workflow identifier
        limit: Maximum number of previous tasks to retrieve
        
    Returns:
        List of TaskCompletion records with summaries
    """
    try:
        with session_make() as session:
            # Get most recent completed tasks for this workflow
            stmt = (
                select(TaskCompletion)
                .where(TaskCompletion.workflow_id == workflow_id)
                .where(TaskCompletion.status == "completed")
                .where(TaskCompletion.summary_steps.isnot(None))
                .order_by(TaskCompletion.completed_at.desc())
                .limit(limit)
            )
            
            results = session.exec(stmt).all()
            logger.info(f"Retrieved {len(results)} previous knowledge records for workflow {workflow_id}")
            return list(results)
            
    except Exception as e:
        logger.error(f"Error retrieving previous knowledge: {e}", exc_info=True)
        return []


def format_knowledge_for_prompt(
    knowledge_records: List[TaskCompletion],
) -> str:
    """
    Format knowledge records into a prompt-friendly string.
    
    Args:
        knowledge_records: List of TaskCompletion records
        
    Returns:
        Formatted knowledge string for injection into prompts
    """
    if not knowledge_records:
        return ""
    
    sections = []
    sections.append("## Previous Task Knowledge\n")
    sections.append("The following knowledge from previous similar tasks may be helpful:\n")
    
    for i, record in enumerate(knowledge_records, 1):
        sections.append(f"\n### Previous Task {i}: {record.original_question[:100]}\n")
        
        if record.summary_steps:
            sections.append(f"**Steps Taken:**\n{record.summary_steps[:500]}\n")
        
        if record.summary_dos_donts:
            sections.append(f"**Do's and Don'ts:**\n{record.summary_dos_donts[:500]}\n")
        
        if record.summary_learnings:
            sections.append(f"**Key Learnings:**\n{record.summary_learnings[:500]}\n")
        
        if record.summary_expected_outcome:
            sections.append(f"**Expected Outcome:**\n{record.summary_expected_outcome[:500]}\n")
    
    return "\n".join(sections)


def get_knowledge_context(
    workflow_id: str,
    current_question: Optional[str] = None,
    limit: int = 3,
) -> str:
    """
    Get formatted knowledge context for injection into prompts.
    
    Args:
        workflow_id: Workflow identifier
        current_question: Current task question (for similarity matching, future enhancement)
        limit: Maximum number of previous tasks to include
        
    Returns:
        Formatted knowledge context string
    """
    knowledge_records = get_previous_knowledge(workflow_id, limit=limit)
    return format_knowledge_for_prompt(knowledge_records)


def get_task_by_id(task_id: str) -> Optional[TaskCompletion]:
    """
    Retrieve a specific task completion record by task_id.
    
    Args:
        task_id: Task identifier
        
    Returns:
        TaskCompletion record or None
    """
    try:
        with session_make() as session:
            stmt = select(TaskCompletion).where(TaskCompletion.task_id == task_id)
            return session.exec(stmt).first()
    except Exception as e:
        logger.error(f"Error retrieving task {task_id}: {e}", exc_info=True)
        return None


def get_workflow_tasks(
    workflow_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[TaskCompletion]:
    """
    Get all tasks for a workflow.
    
    Args:
        workflow_id: Workflow identifier
        limit: Maximum number of records
        offset: Offset for pagination
        
    Returns:
        List of TaskCompletion records
    """
    try:
        with session_make() as session:
            stmt = (
                select(TaskCompletion)
                .where(TaskCompletion.workflow_id == workflow_id)
                .order_by(TaskCompletion.completed_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(session.exec(stmt).all())
    except Exception as e:
        logger.error(f"Error retrieving workflow tasks: {e}", exc_info=True)
        return []
```

