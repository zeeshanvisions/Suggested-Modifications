# Quick Reference Guide

## File Locations

### New Files
```
backend/app/component/database.py
backend/app/model/task_completion.py
backend/app/service/task_completion_service.py
backend/app/service/task_summarization_service.py
backend/app/service/knowledge_retrieval_service.py
backend/app/controller/task_completion_controller.py
```

### Modified Files
```
backend/pyproject.toml
backend/app/__init__.py
backend/app/model/chat.py
backend/app/service/task.py
backend/app/service/chat_service.py
```

## Key Functions

### Task Completion
- `save_task_completion()` - Save task to database
- `handle_task_completion()` - Main handler for task completion
- `extract_task_data_from_lock()` - Extract task state

### Summarization
- `summarize_task_steps()` - Generate steps summary
- `summarize_dos_donts()` - Extract do's and don'ts
- `summarize_learnings()` - Extract learnings
- `summarize_expected_outcome()` - Describe outcomes
- `summarize_task_completion()` - Generate all summaries

### Knowledge Retrieval
- `get_previous_knowledge()` - Get previous tasks for workflow
- `format_knowledge_for_prompt()` - Format for prompt injection
- `get_knowledge_context()` - Get formatted context
- `get_task_by_id()` - Get specific task
- `get_workflow_tasks()` - Get all tasks for workflow

## API Endpoints

```
GET  /api/task-completion/{task_id}
GET  /api/task-completion/workflow/{workflow_id}
GET  /api/task-completion/knowledge/{workflow_id}
GET  /api/task-completion/knowledge/{workflow_id}/context
```

## Database Schema

**Table**: `task_completion`

**Key Fields**:
- `task_id` (unique, indexed)
- `workflow_id` (indexed)
- `task_data` (JSON)
- `summary_steps`, `summary_dos_donts`, `summary_learnings`, `summary_expected_outcome`

## Environment Variables

```bash
database_url=postgresql://user:password@localhost:5432/hachiai
debug=off
```

## Common Issues

### Database Connection Failed
- Check `database_url` in `.env`
- Verify database is running
- Check credentials

### Tables Not Created
- Check startup logs
- Verify `app/__init__.py` has startup event
- Manually create tables if needed

### Summarization Not Working
- Check LLM API credentials
- Verify agent creation
- Check logs for errors

### Knowledge Not Retrieved
- Verify `workflow_id` is set
- Check database has completed tasks
- Verify summaries are generated

## Code Snippets

### Save Task Completion
```python
from app.service.task_completion_service import handle_task_completion

task_completion = await handle_task_completion(
    task_lock=task_lock,
    options=options,
    workflow_id=workflow_id,
    tokens_used=tokens_used,
    execution_time=execution_time,
)
```

### Retrieve Knowledge
```python
from app.service.knowledge_retrieval_service import get_knowledge_context

context = get_knowledge_context(workflow_id, limit=3)
```

### Generate Summaries
```python
from app.service.task_summarization_service import summarize_task_completion

await summarize_task_completion(task_completion, options)
```

## Testing Commands

```bash
# Test database connection
python -c "from app.component.database import engine; print('OK')"

# Test model
python -c "from app.model.task_completion import TaskCompletion; print('OK')"

# Run tests
pytest tests/
```

