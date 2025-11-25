# Chat Service Modifications

## File: `backend/app/service/chat_service.py` (MODIFY)

### Changes Required

#### 1. Add Imports

Add at the top of the file:

```python
from app.service.task_completion_service import handle_task_completion
from app.service.task_summarization_service import summarize_task_completion
from app.service.knowledge_retrieval_service import get_knowledge_context
from app.model.chat import Chat
```

#### 2. Modify `step_solve` Function

Find the section where `Action.end` is handled (around line 212-218) and modify it:

**BEFORE:**
```python
elif item.action == Action.stop:
    if workforce is not None:
        if workforce._running:
            workforce.stop()
        workforce.stop_gracefully()
    await delete_task_lock(task_lock.id)
    break
```

**AFTER:**
```python
elif item.action == Action.end:
    # Task completion detected - save to database
    try:
        # Extract workflow_id from options or task_lock
        workflow_id = getattr(options, 'workflow_id', task_lock.id)
        
        # Calculate execution time
        execution_time = (datetime.now() - task_lock.created_at).total_seconds()
        
        # Get tokens used (accumulate from task execution if available)
        tokens_used = getattr(options, 'tokens', 0)
        
        # Save task completion (async, don't block)
        task_completion = await handle_task_completion(
            task_lock=task_lock,
            options=options,
            workflow_id=workflow_id,
            tokens_used=tokens_used,
            execution_time=execution_time,
        )
        
        # Trigger summarization in background
        if task_completion:
            asyncio.create_task(
                summarize_task_completion(task_completion, options)
            )
    except Exception as e:
        logger.error(f"Error handling task completion: {e}", exc_info=True)
    
    if workforce is not None:
        if workforce._running:
            workforce.stop()
        workforce.stop_gracefully()
    await delete_task_lock(task_lock.id)
    break
elif item.action == Action.stop:
    if workforce is not None:
        if workforce._running:
            workforce.stop()
        workforce.stop_gracefully()
    await delete_task_lock(task_lock.id)
    break
```

#### 3. Inject Knowledge at Task Start

Find where the task is initialized (around line 111-120) and add knowledge injection:

**BEFORE:**
```python
summary_task_agent = task_summary_agent(options)
task_lock.status = Status.confirmed
question = question + options.summary_prompt
camel_task = Task(content=question, id=options.task_id)
```

**AFTER:**
```python
summary_task_agent = task_summary_agent(options)
task_lock.status = Status.confirmed

# Retrieve previous knowledge for this workflow
workflow_id = getattr(options, 'workflow_id', options.task_id)
knowledge_context = get_knowledge_context(workflow_id, current_question=question)

# Inject knowledge into prompt if available
if knowledge_context:
    question = f"{knowledge_context}\n\n{question}"

question = question + options.summary_prompt
camel_task = Task(content=question, id=options.task_id)
```

### Complete Modified Section

Here's the complete modified section around the `Action.end` handler:

```python
# ... existing code ...

elif item.action == Action.end:
    # Task completion detected - save to database
    try:
        # Extract workflow_id from options or task_lock
        workflow_id = getattr(options, 'workflow_id', task_lock.id)
        
        # Calculate execution time
        execution_time = (datetime.now() - task_lock.created_at).total_seconds()
        
        # Get tokens used (accumulate from task execution if available)
        tokens_used = getattr(options, 'tokens', 0)
        
        # Save task completion (async, don't block)
        task_completion = await handle_task_completion(
            task_lock=task_lock,
            options=options,
            workflow_id=workflow_id,
            tokens_used=tokens_used,
            execution_time=execution_time,
        )
        
        # Trigger summarization in background
        if task_completion:
            asyncio.create_task(
                summarize_task_completion(task_completion, options)
            )
    except Exception as e:
        logger.error(f"Error handling task completion: {e}", exc_info=True)
    
    if workforce is not None:
        if workforce._running:
            workforce.stop()
        workforce.stop_gracefully()
    await delete_task_lock(task_lock.id)
    break

# ... existing code ...
```

### Notes

1. **Workflow ID**: You'll need to ensure `workflow_id` is passed in the `Chat` model or extracted from task context
2. **Token Tracking**: You may need to accumulate tokens during task execution
3. **Task Data**: The `extract_task_data_from_lock` function needs to be enhanced to collect full task state
4. **Error Handling**: All database operations are wrapped in try-except to prevent task completion failures

