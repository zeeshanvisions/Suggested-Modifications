# Task Service Modifications

## File: `backend/app/service/task.py` (MODIFY)

### Changes Required

#### 1. Add workflow_id to TaskLock

Modify the `TaskLock` class to include `workflow_id`:

**BEFORE:**
```python
class TaskLock:
    id: str
    status: Status = Status.confirming
    active_agent: str = ""
    mcp: list[str]
    queue: asyncio.Queue[ActionData]
    human_input: dict[str, asyncio.Queue[str]]
    created_at: datetime
    last_accessed: datetime
    background_tasks: set[asyncio.Task]

    def __init__(self, id: str, queue: asyncio.Queue, human_input: dict) -> None:
        self.id = id
        self.queue = queue
        self.human_input = human_input
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.background_tasks = set()
```

**AFTER:**
```python
class TaskLock:
    id: str
    workflow_id: str | None = None  # NEW: Workflow identifier
    status: Status = Status.confirming
    active_agent: str = ""
    mcp: list[str]
    queue: asyncio.Queue[ActionData]
    human_input: dict[str, asyncio.Queue[str]]
    created_at: datetime
    last_accessed: datetime
    background_tasks: set[asyncio.Task]

    def __init__(self, id: str, queue: asyncio.Queue, human_input: dict, workflow_id: str | None = None) -> None:
        self.id = id
        self.workflow_id = workflow_id  # NEW: Set workflow_id
        self.queue = queue
        self.human_input = human_input
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.background_tasks = set()
```

#### 2. Update create_task_lock Function

Find the `create_task_lock` function and update it to accept and set `workflow_id`:

**BEFORE:**
```python
def create_task_lock(id: str) -> TaskLock:
    # ... existing code ...
```

**AFTER:**
```python
def create_task_lock(id: str, workflow_id: str | None = None) -> TaskLock:
    # ... existing code ...
    # When creating TaskLock, pass workflow_id:
    task_lock = TaskLock(id=id, queue=queue, human_input=human_input, workflow_id=workflow_id)
    # ... rest of existing code ...
```

#### 3. Update Chat Model

**File:** `backend/app/model/chat.py` (MODIFY)

Add `workflow_id` field to `Chat` model:

```python
class Chat(BaseModel):
    task_id: str
    workflow_id: str | None = None  # NEW: Workflow identifier
    question: str
    email: str
    # ... rest of existing fields ...
```

#### 4. Update Controller

**File:** `backend/app/controller/chat_controller.py` (MODIFY)

When creating task_lock, pass workflow_id:

```python
# Find where create_task_lock is called
task_lock = create_task_lock(data.task_id, workflow_id=data.workflow_id)
```

### Complete Modified TaskLock Class

```python
class TaskLock:
    id: str
    workflow_id: str | None = None  # Workflow identifier for grouping tasks
    status: Status = Status.confirming
    active_agent: str = ""
    mcp: list[str]
    queue: asyncio.Queue[ActionData]
    """Queue monitoring for SSE response"""
    human_input: dict[str, asyncio.Queue[str]]
    """After receiving user's reply, put the reply into the corresponding agent's queue"""
    created_at: datetime
    last_accessed: datetime
    background_tasks: set[asyncio.Task]
    """Track all background tasks for cleanup"""

    def __init__(self, id: str, queue: asyncio.Queue, human_input: dict, workflow_id: str | None = None) -> None:
        self.id = id
        self.workflow_id = workflow_id
        self.queue = queue
        self.human_input = human_input
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.background_tasks = set()

    # ... rest of methods remain the same ...
```

### Notes

1. **Backward Compatibility**: Make `workflow_id` optional to maintain compatibility
2. **Default Value**: If `workflow_id` is not provided, use `task_id` as fallback
3. **Frontend Integration**: Ensure frontend sends `workflow_id` in the Chat request

