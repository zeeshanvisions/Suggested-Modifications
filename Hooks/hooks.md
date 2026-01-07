## Hooks Feature - Design Document

```markdown
# Hooks Feature - Design Document

## Executive Summary

The Hooks feature enables **sequential task chaining** where multiple tasks can be executed in sequence, sharing browser sessions and state. This allows complex workflows to be broken into multiple tasks that continue where the previous task left off.

---

## Current Architecture Analysis

### Current State

```

┌─────────────────────────────────────────────────────────────┐
│                    Current Task Isolation                    │
└─────────────────────────────────────────────────────────────┘

Task A (task_id: "abc-123")
├── TaskLock (isolated)
├── Browser Session (session_id: "uuid-1")
├── File System (~/hachiai/user/task_abc-123/)
└── Workforce (manages subtasks within Task A)

Task B (task_id: "def-456")
├── TaskLock (isolated)
├── Browser Session (session_id: "uuid-2")
├── File System (~/hachiai/user/task_def-456/)
└── Workforce (manages subtasks within Task B)

❌ No communication between Task A and Task B
❌ No shared browser session
❌ No state sharing
❌ No automatic triggering

```

### Key Components

1. **TaskLock** (`backend/app/service/task.py`)
   - Manages task state and communication queue
   - One per task_id
   - Contains: queue, human_input, background_tasks

2. **Browser Sessions** (`backend/app/utils/toolkit/hybrid_browser_python_toolkit.py`)
   - Each toolkit has `session_id` (defaults to UUID)
   - Connects to CDP endpoint (`browser_port`)
   - Sessions are isolated per task

3. **Workforce** (`backend/app/utils/workforce.py`)
   - Manages subtask execution within a single task
   - Handles dependencies between subtasks
   - No cross-task coordination

4. **File System**
   - Each task has isolated directory: `~/hachiai/{user}/task_{task_id}/`
   - No shared state between tasks

---

## Requirements

### Functional Requirements

1. **Sequential Task Execution**
   - Task B should start automatically when Task A completes
   - Support for multiple tasks in a chain (A → B → C → ...)

2. **Shared Browser Session**
   - All tasks in a hook chain share the same browser session
   - Browser state persists between tasks
   - Continue on the same page/URL where previous task left off

3. **State Sharing**
   - Tasks can access results from previous tasks
   - Shared context/data between tasks
   - File system access to previous task's outputs

4. **Triggering Mechanism**
   - Automatic triggering when previous task completes
   - Support for conditional triggering (on success/failure)
   - Manual trigger option

### Non-Functional Requirements

1. **Backward Compatibility**
   - Existing tasks should continue to work in isolation
   - Hooks should be opt-in feature

2. **Error Handling**
   - Handle failures in hook chain
   - Option to continue or halt on error

3. **Observability**
   - Track hook chain execution
   - Log hook transitions
   - UI visibility of hook relationships

---

## Architectural Options

### Option 1: Hook Chain Manager (RECOMMENDED ⭐)

**Overview:** Central manager that orchestrates task chains with shared session management.

#### Architecture Diagram

```

┌─────────────────────────────────────────────────────────────────┐
│                      Hook Chain Manager                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  HookChain                                                │ │
│  │  - chain_id: "hook-xyz"                                   │ │
│  │  - tasks: [TaskA, TaskB, TaskC]                           │ │
│  │  - shared_session_id: "session-abc"                      │ │
│  │  - shared_context: {data, files, state}                  │ │
│  │  - current_task_index: 0                                  │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ manages
                            ▼
        ┌───────────────────┴───────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────┐                        ┌───────────────┐
│   Task A      │                        │   Task B      │
│ task_id: "a"  │ ──completes──►        │ task_id: "b"  │
│               │                        │               │
│ Uses:         │                        │ Uses:         │
│ - session_id:│                        │ - session_id:│
│   "session-abc"│                      │   "session-abc"│
│ - context from│                        │ - context from│
│   previous    │                        │   Task A      │
└───────────────┘                        └───────────────┘

```

#### Data Flow

```

1. User creates Hook Chain
   └─> HookChain created with:
       - chain_id
       - task_configs: [{task_id, question, ...}, ...]
       - shared_session_id
       - trigger_conditions

2. Task A starts
   └─> HookChainManager creates TaskLock for Task A
   └─> Browser toolkit initialized with shared_session_id
   └─> Task A executes normally

3. Task A completes
   └─> HookChainManager detects completion
   └─> Extracts Task A results/context
   └─> Stores in HookChain.shared_context
   └─> Triggers Task B

4. Task B starts
   └─> HookChainManager creates TaskLock for Task B
   └─> Browser toolkit initialized with same shared_session_id
   └─> Task B receives context from Task A
   └─> Task B continues on same browser page

```

#### Database Schema

```sql
CREATE TABLE hook_chain (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    name VARCHAR(255),
    description TEXT,
    shared_session_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    INDEX idx_user_id (user_id)
);

CREATE TABLE hook_chain_task (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    task_order INT NOT NULL,
    task_id VARCHAR(64) NOT NULL,
    task_config JSON,  -- Chat model config
    trigger_condition ENUM('always', 'on_success', 'on_failure'),
    context_mapping JSON,  -- How to pass context to next task
    created_at TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id),
    INDEX idx_chain_order (chain_id, task_order)
);

CREATE TABLE hook_execution (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    execution_id VARCHAR(64) NOT NULL,
    current_task_index INT DEFAULT 0,
    status ENUM('running', 'completed', 'failed', 'paused'),
    shared_context JSON,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id),
    INDEX idx_execution (execution_id),
    INDEX idx_chain_status (chain_id, status)
);
```

#### Pros

- ✅ Clean separation of concerns
- ✅ Reusable shared sessions
- ✅ Centralized state management
- ✅ Easy to extend (parallel chains, conditional branching)
- ✅ Good observability

#### Cons

- ❌ More complex implementation
- ❌ Requires new database tables
- ❌ Additional service to maintain

---

### Option 2: Task-to-Task Direct Triggering

**Overview:** Tasks directly trigger next tasks via completion callbacks.

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Task A                                │
│  task_id: "task-a"                                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  On Completion:                                  │   │
│  │  └─> Trigger Task B                              │   │
│  │      - Pass session_id                           │   │
│  │      - Pass context data                         │   │
│  │      - Create new TaskLock for Task B            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                    │
                    │ triggers
                    ▼
┌─────────────────────────────────────────────────────────┐
│                    Task B                                │
│  task_id: "task-b"                                       │
│  hook_parent_id: "task-a"                                │
│  hook_session_id: "session-abc"                          │
│  hook_context: {task_a_result: "..."}                   │
└─────────────────────────────────────────────────────────┘
```

#### Pros

- ✅ Simpler implementation
- ✅ No central manager needed
- ✅ Direct task relationships

#### Cons

- ❌ Harder to manage complex chains
- ❌ Session management scattered
- ❌ Limited observability

---

### Option 3: Workflow Template Extension

**Overview:** Extend existing workflow templates to support multi-task execution.

#### Pros

- ✅ Leverages existing template system
- ✅ Users familiar with templates

#### Cons

- ❌ Templates are single-task focused
- ❌ Mixing concerns
- ❌ Less flexible

---

## Recommended Solution: Option 1 (Hook Chain Manager) ⭐

### Why Option 1?

1. **Scalability**: Can handle complex chains, parallel execution, conditional branching
2. **Maintainability**: Clear separation of concerns
3. **Observability**: Centralized tracking and logging
4. **Flexibility**: Easy to extend with new features
5. **Session Management**: Proper shared session lifecycle

---

## Detailed Sequence Diagrams

### Hook Chain Execution Flow

```
User                    Frontend              HookManager          Task A              Task B
 │                        │                        │                  │                    │
 │  Create Hook Chain     │                        │                  │                    │
 ├───────────────────────►│                        │                  │                    │
 │                        │  POST /hooks/chains    │                  │                    │
 │                        ├───────────────────────►│                  │                    │
 │                        │                        │  Create Chain     │                    │
 │                        │                        │  (chain_id)      │                    │
 │                        │◄───────────────────────┤                  │                    │
 │                        │                        │                  │                    │
 │  Execute Chain         │                        │                  │                    │
 ├───────────────────────►│                        │                  │                    │
 │                        │  POST /hooks/chains/   │                  │                    │
 │                        │    {id}/execute        │                  │                    │
 │                        ├───────────────────────►│                  │                    │
 │                        │                        │  Start Task A    │                    │
 │                        │                        ├─────────────────►│                    │
 │                        │                        │                  │  Initialize        │
 │                        │                        │                  │  (shared_session) │
 │                        │                        │                  │                    │
 │                        │                        │                  │  Execute...        │
 │                        │                        │                  │                    │
 │                        │                        │                  │  Complete          │
 │                        │                        │◄─────────────────┤                    │
 │                        │                        │  Extract Context │                    │
 │                        │                        │  Trigger Task B  │                    │
 │                        │                        ├───────────────────────────────────────►│
 │                        │                        │                  │  Initialize        │
 │                        │                        │                  │  (same session)    │
 │                        │                        │                  │                    │
 │                        │                        │                  │  Execute...        │
 │                        │                        │                  │                    │
 │                        │                        │                  │  Complete          │
 │                        │                        │◄───────────────────────────────────────┤
 │                        │                        │  Chain Complete  │                    │
 │                        │◄───────────────────────┤                  │                    │
 │  Chain Complete        │                        │                  │                    │
 │◄───────────────────────┤                        │                  │                    │
```

### Browser Session Sharing

```
Task A Toolkit          BrowserSession          HookManager          Task B Toolkit
 │                        │                        │                    │
 │  Initialize            │                        │                    │
 │  (shared_session_id)   │                        │                    │
 ├───────────────────────►│                        │                    │
 │                        │  Check if exists       │                    │
 │                        ├───────────────────────►│                    │
 │                        │                        │  Create/Get        │
 │                        │                        │  Session            │
 │                        │◄───────────────────────┤                    │
 │                        │  Return Session       │                    │
 │◄───────────────────────┤                        │                    │
 │  Use Session           │                        │                    │
 │  (navigate, click...)   │                        │                    │
 │                        │                        │                    │
 │  Task A Complete       │                        │                    │
 │                        │                        │                    │
 │                        │                        │                    │  Initialize
 │                        │                        │                    │  (same session_id)
 │                        │                        │                    ├───────────────────►
 │                        │                        │                    │  Get Session
 │                        │                        │                    ├───────────────────►
 │                        │                        │                    │  Return Existing
 │                        │                        │                    │  Session
 │                        │                        │                    │◄───────────────────┤
 │                        │                        │                    │  Continue on
 │                        │                        │                    │  Same Page
```

---

## Implementation Details

### Backend Components

#### 1. HookChainManager Service

```python
# backend/app/service/hook_service.py

class HookChainManager:
    """Manages hook chain execution and shared sessions"""
    
    def __init__(self):
        self.active_chains: Dict[str, HookChain] = {}
        self.shared_sessions: Dict[str, BrowserSession] = {}
    
    async def start_chain(self, chain_id: str, execution_id: str):
        """Start executing a hook chain"""
        chain = await self.get_chain(chain_id)
        execution = HookExecution(
            chain_id=chain_id,
            execution_id=execution_id,
            shared_context={}
        )
        
        # Start first task
        await self.start_task_in_chain(chain, execution, 0)
    
    async def start_task_in_chain(
        self, 
        chain: HookChain, 
        execution: HookExecution, 
        task_index: int
    ):
        """Start a specific task in the chain"""
        task_config = chain.tasks[task_index]
        
        # Create Chat config with shared session
        chat_config = Chat(
            task_id=task_config.task_id,
            question=task_config.question,
            hook_chain_id=chain.id,
            hook_execution_id=execution.execution_id,
            shared_session_id=chain.shared_session_id,
            shared_context=execution.shared_context
        )
        
        # Start task via existing chat endpoint
    
    async def on_task_complete(
        self, 
        execution_id: str, 
        task_id: str, 
        result: str
    ):
        """Called when a task in chain completes"""
        execution = await self.get_execution(execution_id)
        chain = await self.get_chain(execution.chain_id)
        
        # Update shared context
        execution.shared_context[task_id] = {
            'result': result,
            'completed_at': datetime.now()
        }
        
        # Check if more tasks to run
        current_index = execution.current_task_index
        if current_index + 1 < len(chain.tasks):
            # Trigger next task
            await self.start_task_in_chain(
                chain, 
                execution, 
                current_index + 1
            )
        else:
            # Chain complete
            execution.status = 'completed'
```

#### 2. Modified Browser Toolkit

```python
# backend/app/utils/toolkit/hybrid_browser_python_toolkit.py

class HybridBrowserPythonToolkit:
    def __init__(
        self,
        api_task_id: str,
        *,
        session_id: str | None = None,
        shared_session_id: str | None = None,  # NEW: For hook chains
        # ... other params
    ):
        # If in hook chain, use shared session
        if shared_session_id:
            self._session_id = shared_session_id
            self._shared_session = True
        else:
            self._session_id = session_id or "default"
            self._shared_session = False
```

#### 3. Modified Chat Service

```python
# backend/app/service/chat_service.py

@sync_step
async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # Check if this task is part of a hook chain
    if options.hook_chain_id and options.hook_execution_id:
        hook_manager = get_hook_chain_manager()
        
        # Register task completion callback
        async def on_task_complete(result: str):
            await hook_manager.on_task_complete(
                options.hook_execution_id,
                options.task_id,
                result
            )
        
        # Inject shared context into task
        if options.shared_context:
            # Add context to initial prompt or task data
            pass
    
    # ... rest of existing execution ...
```

---

## API Design

### Create Hook Chain

```http
POST /api/chat/hooks/chains
Content-Type: application/json

{
  "name": "E-commerce Workflow",
  "description": "Research → Purchase → Review",
  "shared_session_id": "session-abc-123",
  "tasks": [
    {
      "task_order": 0,
      "task_config": {
        "question": "Research best laptops under $1000",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success"
    },
    {
      "task_order": 1,
      "task_config": {
        "question": "Purchase the top recommendation",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success",
      "context_mapping": {
        "product_name": "{{task_results.task-0.result.product}}"
      }
    }
  ]
}
```

### Execute Hook Chain

```http
POST /api/chat/hooks/chains/{chain_id}/execute
Content-Type: application/json

{
  "execution_id": "exec-xyz-789",
  "initial_context": {
    "budget": 1000,
    "category": "laptops"
  }
}
```

---

## UI/UX Design

### Hook Chain Builder

```
┌─────────────────────────────────────────────────────────────┐
│  Create Hook Chain                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Chain Name: [E-commerce Workflow        ]                  │
│  Description: [Research → Purchase → Review]                │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Task 1: Research Products                           │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Question: Research best laptops under $1000   │  │  │
│  │  │ Model: [GPT-4 ▼]                              │  │  │
│  │  │ Trigger Next: [On Success ▼]                 │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                    │                                   │  │
│  │                    ▼                                   │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Task 2: Purchase Product                      │  │  │
│  │  │ ┌────────────────────────────────────────────┐ │  │  │
│  │  │ │ Question: Purchase {{task1.product}}     │ │  │  │
│  │  │ │ Context Mapping:                          │ │  │  │
│  │  │ │   product_name ← task1.result.product   │ │  │  │
│  │  │ └────────────────────────────────────────────┘ │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Shared Session: [session-abc-123]                         │
│  [✓] Share browser session across tasks                    │
│  [✓] Share file system                                     │
│                                                              │
│  [Cancel]  [Save Chain]  [Execute Now]                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

- Database migrations
- HookChainManager service
- API endpoints
- Browser session sharing

### Phase 2: Integration (Week 3)

- Integrate with chat_service
- Modify browser toolkit
- Context passing
- Testing

### Phase 3: Frontend (Week 4)

- Hook chain builder UI
- Execution monitor
- Chain management UI

### Phase 4: Polish (Week 5)

- Error handling improvements
- Performance optimization
- Documentation
- User testing

---

## Conclusion

The **Hook Chain Manager (Option 1)** provides the best foundation for the Hooks feature. It offers:

- ✅ Clean architecture
- ✅ Scalability
- ✅ Maintainability
- ✅ Extensibility
- ✅ Good user experience

The implementation can be done incrementally, starting with core functionality and adding features over time.

```

## Summary

Recommended approach: Option 1 — Hook Chain Manager.

### Why this option
1. Centralized orchestration for complex chains
2. Shared session management
3. Clear separation of concerns
4. Extensible (parallel execution, branching)
5. Better observability and debugging

### Key components
1. HookChainManager — orchestrates task chains
2. Shared browser sessions — persistent across tasks
3. Context passing — results flow between tasks
4. Database schema — stores chain definitions and executions

### Implementation path
- Phase 1: Backend infrastructure (database, service, APIs)
- Phase 2: Integration with existing systems
- Phase 3: Frontend UI
- Phase 4: Polish and optimization

This design supports sequential task execution with shared browser sessions and state, while remaining backward compatible with existing isolated tasks.

Should I create the implementation files, or do you want to review the design first?

```markdown
# Hooks Feature - Design Document

## Executive Summary

The Hooks feature enables **sequential task chaining** where multiple tasks can be executed in sequence, sharing browser sessions and state. This allows complex workflows to be broken into multiple tasks that continue where the previous task left off.

---

## Current Architecture Analysis

### Current State

```

```plaintext

### Key Components

1. **TaskLock** (`backend/app/service/task.py`)
   - Manages task state and communication queue
   - One per task_id
   - Contains: queue, human_input, background_tasks

2. **Browser Sessions** (`backend/app/utils/toolkit/hybrid_browser_python_toolkit.py`)
   - Each toolkit has `session_id` (defaults to UUID)
   - Connects to CDP endpoint (`browser_port`)
   - Sessions are isolated per task

3. **Workforce** (`backend/app/utils/workforce.py`)
   - Manages subtask execution within a single task
   - Handles dependencies between subtasks
   - No cross-task coordination

4. **File System**
   - Each task has isolated directory: `~/hachiai/{user}/task_{task_id}/`
   - No shared state between tasks

---

## Requirements

### Functional Requirements

1. **Sequential Task Execution**
   - Task B should start automatically when Task A completes
   - Support for multiple tasks in a chain (A → B → C → ...)

2. **Shared Browser Session**
   - All tasks in a hook chain share the same browser session
   - Browser state persists between tasks
   - Continue on the same page/URL where previous task left off

3. **State Sharing**
   - Tasks can access results from previous tasks
   - Shared context/data between tasks
   - File system access to previous task's outputs

4. **Triggering Mechanism**
   - Automatic triggering when previous task completes
   - Support for conditional triggering (on success/failure)
   - Manual trigger option

### Non-Functional Requirements

1. **Backward Compatibility**
   - Existing tasks should continue to work in isolation
   - Hooks should be opt-in feature

2. **Error Handling**
   - Handle failures in hook chain
   - Option to continue or halt on error

3. **Observability**
   - Track hook chain execution
   - Log hook transitions
   - UI visibility of hook relationships

---

## Architectural Options

### Option 1: Hook Chain Manager (RECOMMENDED ⭐)

**Overview:** Central manager that orchestrates task chains with shared session management.

#### Architecture Diagram

```

```plaintext

#### Data Flow

```

```plaintext

#### Database Schema

CREATE TABLE hook_chain (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    name VARCHAR(255),
    description TEXT,
    shared_session_id VARCHAR(64) NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    INDEX idx_user_id (user_id)
);

CREATE TABLE hook_chain_task (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    task_order INT NOT NULL,
    task_id VARCHAR(64) NOT NULL,
    task_config JSON,  -- Chat model config
    trigger_condition ENUM('always', 'on_success', 'on_failure'),
    context_mapping JSON,  -- How to pass context to next task
    created_at TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id),
    INDEX idx_chain_order (chain_id, task_order)
);

CREATE TABLE hook_execution (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    execution_id VARCHAR(64) NOT NULL,
    current_task_index INT DEFAULT 0,
    status ENUM('running', 'completed', 'failed', 'paused'),
    shared_context JSON,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id),
    INDEX idx_execution (execution_id),
    INDEX idx_chain_status (chain_id, status)
);#### Pros
- ✅ Clean separation of concerns
- ✅ Reusable shared sessions
- ✅ Centralized state management
- ✅ Easy to extend (parallel chains, conditional branching)
- ✅ Good observability

#### Cons
- ❌ More complex implementation
- ❌ Requires new database tables
- ❌ Additional service to maintain

---

### Option 2: Task-to-Task Direct Triggering

**Overview:** Tasks directly trigger next tasks via completion callbacks.

#### Architecture Diagram

```

```plaintext

#### Pros
- ✅ Simpler implementation
- ✅ No central manager needed
- ✅ Direct task relationships

#### Cons
- ❌ Harder to manage complex chains
- ❌ Session management scattered
- ❌ Limited observability

---

### Option 3: Workflow Template Extension

**Overview:** Extend existing workflow templates to support multi-task execution.

#### Pros
- ✅ Leverages existing template system
- ✅ Users familiar with templates

#### Cons
- ❌ Templates are single-task focused
- ❌ Mixing concerns
- ❌ Less flexible

---

## Recommended Solution: Option 1 (Hook Chain Manager) ⭐

### Why Option 1?

1. **Scalability**: Can handle complex chains, parallel execution, conditional branching
2. **Maintainability**: Clear separation of concerns
3. **Observability**: Centralized tracking and logging
4. **Flexibility**: Easy to extend with new features
5. **Session Management**: Proper shared session lifecycle

---

## Detailed Sequence Diagrams

### Hook Chain Execution Flow

```

```plaintext

#### Pros
- ✅ Simpler implementation
- ✅ No central manager needed
- ✅ Direct task relationships

#### Cons
- ❌ Harder to manage complex chains
- ❌ Session management scattered
- ❌ Limited observability

---

### Option 3: Workflow Template Extension

**Overview:** Extend existing workflow templates to support multi-task execution.

#### Pros
- ✅ Leverages existing template system
- ✅ Users familiar with templates

#### Cons
- ❌ Templates are single-task focused
- ❌ Mixing concerns
- ❌ Less flexible

---

## Recommended Solution: Option 1 (Hook Chain Manager) ⭐

### Why Option 1?

1. **Scalability**: Can handle complex chains, parallel execution, conditional branching
2. **Maintainability**: Clear separation of concerns
3. **Observability**: Centralized tracking and logging
4. **Flexibility**: Easy to extend with new features
5. **Session Management**: Proper shared session lifecycle

---

## Detailed Sequence Diagrams

### Hook Chain Execution Flow

```

```plaintext

### Browser Session Sharing

```

```plaintext

### Browser Session Sharing

```

```plaintext

---

## Implementation Details

### Backend Components

#### 1. HookChainManager Service

# backend/app/service/hook_service.py

class HookChainManager:
    """Manages hook chain execution and shared sessions"""
    
    def __init__(self):
        self.active_chains: Dict[str, HookChain] = {}
        self.shared_sessions: Dict[str, BrowserSession] = {}
    
    async def start_chain(self, chain_id: str, execution_id: str):
        """Start executing a hook chain"""
        chain = await self.get_chain(chain_id)
        execution = HookExecution(
            chain_id=chain_id,
            execution_id=execution_id,
            shared_context={}
        )
        
        # Start first task
        await self.start_task_in_chain(chain, execution, 0)
    
    async def start_task_in_chain(
        self, 
        chain: HookChain, 
        execution: HookExecution, 
        task_index: int
    ):
        """Start a specific task in the chain"""
        task_config = chain.tasks[task_index]
        
        # Create Chat config with shared session
        chat_config = Chat(
            task_id=task_config.task_id,
            question=task_config.question,
            hook_chain_id=chain.id,
            hook_execution_id=execution.execution_id,
            shared_session_id=chain.shared_session_id,
            shared_context=execution.shared_context
        )
        
        # Start task via existing chat endpoint
    
    async def on_task_complete(
        self, 
        execution_id: str, 
        task_id: str, 
        result: str
    ):
        """Called when a task in chain completes"""
        execution = await self.get_execution(execution_id)
        chain = await self.get_chain(execution.chain_id)
        
        # Update shared context
        execution.shared_context[task_id] = {
            'result': result,
            'completed_at': datetime.now()
        }
        
        # Check if more tasks to run
        current_index = execution.current_task_index
        if current_index + 1 < len(chain.tasks):
            # Trigger next task
            await self.start_task_in_chain(
                chain, 
                execution, 
                current_index + 1
            )
        else:
            # Chain complete
            execution.status = 'completed'#### 2. Modified Browser Toolkit

# backend/app/utils/toolkit/hybrid_browser_python_toolkit.py

class HybridBrowserPythonToolkit:
    def __init__(
        self,
        api_task_id: str,
        *,
        session_id: str | None = None,
        shared_session_id: str | None = None,  # NEW: For hook chains
        # ... other params
    ):
        # If in hook chain, use shared session
        if shared_session_id:
            self._session_id = shared_session_id
            self._shared_session = True
        else:
            self._session_id = session_id or "default"
            self._shared_session = False#### 3. Modified Chat Service

# backend/app/service/chat_service.py

@sync_step
async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # Check if this task is part of a hook chain
    if options.hook_chain_id and options.hook_execution_id:
        hook_manager = get_hook_chain_manager()
        
        # Register task completion callback
        async def on_task_complete(result: str):
            await hook_manager.on_task_complete(
                options.hook_execution_id,
                options.task_id,
                result
            )
        
        # Inject shared context into task
        if options.shared_context:
            # Add context to initial prompt or task data
            pass
    
    # ... rest of existing execution ...---

## API Design

### Create Hook Chain

POST /api/chat/hooks/chains
Content-Type: application/json

{
  "name": "E-commerce Workflow",
  "description": "Research → Purchase → Review",
  "shared_session_id": "session-abc-123",
  "tasks": [
    {
      "task_order": 0,
      "task_config": {
        "question": "Research best laptops under $1000",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success"
    },
    {
      "task_order": 1,
      "task_config": {
        "question": "Purchase the top recommendation",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success",
      "context_mapping": {
        "product_name": "{{task_results.task-0.result.product}}"
      }
    }
  ]
}### Execute Hook Chain
p
POST /api/chat/hooks/chains/{chain_id}/execute
Content-Type: application/json

{
  "execution_id": "exec-xyz-789",
  "initial_context": {
    "budget": 1000,
    "category": "laptops"
  }
}---

## UI/UX Design

### Hook Chain Builder

```

```plaintext

---

## Implementation Details

### Backend Components

#### 1. HookChainManager Service

# backend/app/service/hook_service.py

class HookChainManager:
    """Manages hook chain execution and shared sessions"""
    
    def __init__(self):
        self.active_chains: Dict[str, HookChain] = {}
        self.shared_sessions: Dict[str, BrowserSession] = {}
    
    async def start_chain(self, chain_id: str, execution_id: str):
        """Start executing a hook chain"""
        chain = await self.get_chain(chain_id)
        execution = HookExecution(
            chain_id=chain_id,
            execution_id=execution_id,
            shared_context={}
        )
        
        # Start first task
        await self.start_task_in_chain(chain, execution, 0)
    
    async def start_task_in_chain(
        self, 
        chain: HookChain, 
        execution: HookExecution, 
        task_index: int
    ):
        """Start a specific task in the chain"""
        task_config = chain.tasks[task_index]
        
        # Create Chat config with shared session
        chat_config = Chat(
            task_id=task_config.task_id,
            question=task_config.question,
            hook_chain_id=chain.id,
            hook_execution_id=execution.execution_id,
            shared_session_id=chain.shared_session_id,
            shared_context=execution.shared_context
        )
        
        # Start task via existing chat endpoint
    
    async def on_task_complete(
        self, 
        execution_id: str, 
        task_id: str, 
        result: str
    ):
        """Called when a task in chain completes"""
        execution = await self.get_execution(execution_id)
        chain = await self.get_chain(execution.chain_id)
        
        # Update shared context
        execution.shared_context[task_id] = {
            'result': result,
            'completed_at': datetime.now()
        }
        
        # Check if more tasks to run
        current_index = execution.current_task_index
        if current_index + 1 < len(chain.tasks):
            # Trigger next task
            await self.start_task_in_chain(
                chain, 
                execution, 
                current_index + 1
            )
        else:
            # Chain complete
            execution.status = 'completed'#### 2. Modified Browser Toolkit

# backend/app/utils/toolkit/hybrid_browser_python_toolkit.py

class HybridBrowserPythonToolkit:
    def __init__(
        self,
        api_task_id: str,
        *,
        session_id: str | None = None,
        shared_session_id: str | None = None,  # NEW: For hook chains
        # ... other params
    ):
        # If in hook chain, use shared session
        if shared_session_id:
            self._session_id = shared_session_id
            self._shared_session = True
        else:
            self._session_id = session_id or "default"
            self._shared_session = False#### 3. Modified Chat Service

# backend/app/service/chat_service.py

@sync_step
async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # Check if this task is part of a hook chain
    if options.hook_chain_id and options.hook_execution_id:
        hook_manager = get_hook_chain_manager()
        
        # Register task completion callback
        async def on_task_complete(result: str):
            await hook_manager.on_task_complete(
                options.hook_execution_id,
                options.task_id,
                result
            )
        
        # Inject shared context into task
        if options.shared_context:
            # Add context to initial prompt or task data
            pass
    
    # ... rest of existing execution ...---

## API Design

### Create Hook Chain

POST /api/chat/hooks/chains
Content-Type: application/json

{
  "name": "E-commerce Workflow",
  "description": "Research → Purchase → Review",
  "shared_session_id": "session-abc-123",
  "tasks": [
    {
      "task_order": 0,
      "task_config": {
        "question": "Research best laptops under $1000",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success"
    },
    {
      "task_order": 1,
      "task_config": {
        "question": "Purchase the top recommendation",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success",
      "context_mapping": {
        "product_name": "{{task_results.task-0.result.product}}"
      }
    }
  ]
}### Execute Hook Chain
p
POST /api/chat/hooks/chains/{chain_id}/execute
Content-Type: application/json

{
  "execution_id": "exec-xyz-789",
  "initial_context": {
    "budget": 1000,
    "category": "laptops"
  }
}---

## UI/UX Design

### Hook Chain Builder

```

```python

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Database migrations
- HookChainManager service
- API endpoints
- Browser session sharing

### Phase 2: Integration (Week 3)
- Integrate with chat_service
- Modify browser toolkit
- Context passing
- Testing

### Phase 3: Frontend (Week 4)
- Hook chain builder UI
- Execution monitor
- Chain management UI

### Phase 4: Polish (Week 5)
- Error handling improvements
- Performance optimization
- Documentation
- User testing

---

## Conclusion

The **Hook Chain Manager (Option 1)** provides the best foundation for the Hooks feature. It offers:

- ✅ Clean architecture
- ✅ Scalability
- ✅ Maintainability
- ✅ Extensibility
- ✅ Good user experience

The implementation can be done incrementally, starting with core functionality and adding features over time.
```

```plaintext

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Database migrations
- HookChainManager service
- API endpoints
- Browser session sharing

### Phase 2: Integration (Week 3)
- Integrate with chat_service
- Modify browser toolkit
- Context passing
- Testing

### Phase 3: Frontend (Week 4)
- Hook chain builder UI
- Execution monitor
- Chain management UI

### Phase 4: Polish (Week 5)
- Error handling improvements
- Performance optimization
- Documentation
- User testing

---

## Conclusion

The **Hook Chain Manager (Option 1)** provides the best foundation for the Hooks feature. It offers:

- ✅ Clean architecture
- ✅ Scalability
- ✅ Maintainability
- ✅ Extensibility
- ✅ Good user experience

The implementation can be done incrementally, starting with core functionality and adding features over time.
```

```python
# backend/app/utils/toolkit/hybrid_browser_python_toolkit.py

class HybridBrowserPythonToolkit:
    def __init__(
        self,
        api_task_id: str,
        *,
        session_id: str | None = None,
        shared_session_id: str | None = None,  # NEW: For hook chains
        # ... other params
    ):
        # If in hook chain, use shared session
        if shared_session_id:
            self._session_id = shared_session_id
            self._shared_session = True
        else:
            self._session_id = session_id or "default"
            self._shared_session = False
```

```python
# backend/app/service/chat_service.py

@sync_step
async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # Check if this task is part of a hook chain
    if options.hook_chain_id and options.hook_execution_id:
        hook_manager = get_hook_chain_manager()
        
        # Register task completion callback
        async def on_task_complete(result: str):
            await hook_manager.on_task_complete(
                options.hook_execution_id,
                options.task_id,
                result
            )
        
        # Inject shared context into task
        if options.shared_context:
            # Add context to initial prompt or task data
            pass
    
    # ... rest of existing execution ...
```

```plaintext
POST /api/chat/hooks/chains
Content-Type: application/json

{
  "name": "E-commerce Workflow",
  "description": "Research → Purchase → Review",
  "shared_session_id": "session-abc-123",
  "tasks": [
    {
      "task_order": 0,
      "task_config": {
        "question": "Research best laptops under $1000",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success"
    },
    {
      "task_order": 1,
      "task_config": {
        "question": "Purchase the top recommendation",
        "model_platform": "openai",
        "model_type": "gpt-4"
      },
      "trigger_condition": "on_success",
      "context_mapping": {
        "product_name": "{{task_results.task-0.result.product}}"
      }
    }
  ]
}
```

```plaintext
POST /api/chat/hooks/chains/{chain_id}/execute
Content-Type: application/json

{
  "execution_id": "exec-xyz-789",
  "initial_context": {
    "budget": 1000,
    "category": "laptops"
  }
}
```

```plaintext
┌─────────────────────────────────────────────────────────────┐
│  Create Hook Chain                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Chain Name: [E-commerce Workflow        ]                  │
│  Description: [Research → Purchase → Review]                │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Task 1: Research Products                           │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Question: Research best laptops under $1000   │  │  │
│  │  │ Model: [GPT-4 ▼]                              │  │  │
│  │  │ Trigger Next: [On Success ▼]                 │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                    │                                   │  │
│  │                    ▼                                   │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Task 2: Purchase Product                      │  │  │
│  │  │ ┌────────────────────────────────────────────┐ │  │  │
│  │  │ │ Question: Purchase {{task1.product}}     │ │  │  │
│  │  │ │ Context Mapping:                          │ │  │  │
│  │  │ │   product_name ← task1.result.product   │ │  │  │
│  │  │ └────────────────────────────────────────────┘ │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Shared Session: [session-abc-123]                         │
│  [✓] Share browser session across tasks                    │
│  [✓] Share file system                                     │
│                                                              │
│  [Cancel]  [Save Chain]  [Execute Now]                     │
└─────────────────────────────────────────────────────────────┘
```

```plaintext

## Summary

Recommended approach: Option 1 — Hook Chain Manager.

### Why this option
1. Centralized orchestration for complex chains
2. Shared session management
3. Clear separation of concerns
4. Extensible (parallel execution, branching)
5. Better observability and debugging

### Key components
1. HookChainManager — orchestrates task chains
2. Shared browser sessions — persistent across tasks
3. Context passing — results flow between tasks
4. Database schema — stores chain definitions and executions

### Implementation path
- Phase 1: Backend infrastructure (database, service, APIs)
- Phase 2: Integration with existing systems
- Phase 3: Frontend UI
- Phase 4: Polish and optimization

This design supports sequential task execution with shared browser sessions and state, while remaining backward compatible with existing isolated tasks.

Should I create the implementation files, or do you want to review the design first?
```

---

**User**

I am more interested in Option 1. We should have a seperate modal or window to create chain. Where you can easily create sequece of exisiting tasks. Review the plan with new requirements. And return a details diagrams and plan along with advantages of implementing it

---

**Cursor**

Reviewing the codebase to understand task storage, UI patterns, and modal/dialog structure to design a separate chain creation modal that sequences existing tasks.

Design document for the Hook Chain feature with a separate modal for creating chains from existing tasks.

## Hook Chain Feature - Detailed Design Document

### Updated Requirements

1. Separate modal/window for chain creation
2. Select and sequence existing tasks
3. Visual chain builder interface
4. Shared browser session across tasks
5. Context passing between tasks

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐         ┌──────────────────────┐            │
│  │  ChatBox         │         │  HookChainModal      │            │
│  │  (Main UI)       │────────►│  (Chain Builder)     │            │
│  │                  │  opens  │                      │            │
│  │  - Task List     │         │  - Task Selector     │            │
│  │  - Task History  │         │  - Chain Sequence    │            │
│  │  - [Create Chain]│         │  - Configuration     │            │
│  └──────────────────┘         └──────────────────────┘            │
│           │                              │                          │
│           │                              │                          │
│           └──────────────┬───────────────┘                          │
│                          │                                          │
│                          ▼                                          │
│                  ┌──────────────┐                                  │
│                  │  chatStore    │                                  │
│                  │  (Zustand)    │                                  │
│                  └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          │ API Calls
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend Layer                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │  HookChainController  │      │  HookChainManager    │            │
│  │  (API Endpoints)      │◄────►│  (Service Layer)     │            │
│  │                       │      │                      │            │
│  │  - POST /chains      │      │  - Chain Creation    │            │
│  │  - GET /chains       │      │  - Execution Control  │            │
│  │  - POST /execute     │      │  - Session Management │            │
│  └──────────────────────┘      │  - Context Passing   │            │
│           │                     └──────────────────────┘            │
│           │                              │                          │
│           │                              │                          │
│           ▼                              ▼                          │
│  ┌──────────────────┐         ┌──────────────────────┐            │
│  │  Database        │         │  ChatService        │            │
│  │  - hook_chain     │         │  (Task Execution)   │            │
│  │  - hook_execution │         │                      │            │
│  └──────────────────┘         └──────────────────────┘            │
│                                         │                          │
│                                         ▼                          │
│                              ┌──────────────────────┐            │
│                              │  BrowserToolkit       │            │
│                              │  (Shared Sessions)    │            │
│                              └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Hook Chain Modal Design

### Modal Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  Create Hook Chain                                    [×]           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Chain Configuration                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Chain Name: [E-commerce Automation Workflow        ]        │  │
│  │ Description: [Research → Purchase → Review Process]         │  │
│  │                                                              │  │
│  │ ☑ Share browser session across all tasks                   │  │
│  │ ☑ Share file system between tasks                          │  │
│  │                                                              │  │
│  │ Shared Session ID: [session-ecom-123] (auto-generated)     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Task Sequence                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ Step 1: Research Products                         │     │  │
│  │  │ ┌──────────────────────────────────────────────┐ │     │  │
│  │  │ │ Selected Task:                                │ │     │  │
│  │  │ │ [Search: "Research best laptops under $1000"] │ │     │  │
│  │  │ │ Task ID: task-abc-123                          │ │     │  │
│  │  │ │ Status: ✓ Completed                           │ │     │  │
│  │  │ │                                                │ │     │  │
│  │  │ │ Trigger Next Task:                             │ │     │  │
│  │  │ │ ( ) Always    (•) On Success    ( ) On Failure │ │     │  │
│  │  │ │                                                │ │     │  │
│  │  │ │ Context to Pass:                               │ │     │  │
│  │  │ │ ☑ Task result                                 │ │     │  │
│  │  │ │ ☑ Generated files                             │ │     │  │
│  │  │ │ ☑ Browser state (URL, cookies)                │ │     │  │
│  │  │ └──────────────────────────────────────────────┘ │     │  │
│  │  │                    │                             │     │  │
│  │  │                    ▼                             │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ Step 2: Purchase Product                            │     │  │
│  │  │ ┌──────────────────────────────────────────────┐ │     │  │
│  │  │ │ Selected Task:                                │ │     │  │
│  │  │ │ [+ Select Task from History ▼]                │ │     │  │
│  │  │ │                                                │ │     │  │
│  │  │ │ Or Create New Task:                            │ │     │  │
│  │  │ │ [Create New Task]                              │ │     │  │
│  │  │ └──────────────────────────────────────────────┘ │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  [+ Add Next Task]                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Available Tasks (Click to add to chain)                       │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ 📋 Research Competitor Pricing                      │     │  │
│  │  │ Task ID: task-xyz-789 | Status: ✓ Completed        │     │  │
│  │  │ Question: "Research competitor pricing for..."     │     │  │
│  │  │ [Add to Chain]                                      │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ 📋 Generate Marketing Copy                           │     │  │
│  │  │ Task ID: task-def-456 | Status: ✓ Completed         │     │  │
│  │  │ Question: "Generate marketing copy for..."          │     │  │
│  │  │ [Add to Chain]                                      │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  [Load More Tasks]                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [Cancel]                    [Save Chain]  [Execute Now]       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Task Selection Flow

```
User Opens Hook Chain Modal
         │
         ▼
┌────────────────────────┐
│  Fetch Task History    │
│  (from chatStore or    │
│   API endpoint)        │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  Display Task List     │
│  - Filter by status    │
│  - Search by question │
│  - Sort by date       │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  User Selects Task     │
│  (Click "Add to Chain")│
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  Add to Sequence       │
│  - Auto-increment step │
│  - Configure trigger   │
│  - Set context mapping │
└────────────────────────┘
```

---

## Data Flow Diagrams

### Chain Creation Flow

```
┌──────────┐
│   User   │
└────┬─────┘
     │
     │ 1. Click "Create Hook Chain"
     ▼
┌─────────────────────┐
│  HookChainModal      │
│  Opens               │
└────┬─────────────────┘
     │
     │ 2. Fetch Task History
     ▼
┌─────────────────────┐
│  GET /api/chat/      │
│  history?status=done │
└────┬─────────────────┘
     │
     │ 3. Return Task List
     ▼
┌─────────────────────┐
│  Display Tasks       │
│  in Modal            │
└────┬─────────────────┘
     │
     │ 4. User Selects Tasks
     │    and Configures Chain
     ▼
┌─────────────────────┐
│  User Clicks         │
│  "Save Chain"        │
└────┬─────────────────┘
     │
     │ 5. POST /api/chat/hooks/chains
     ▼
┌─────────────────────┐
│  HookChainController │
│  - Validate data     │
│  - Create chain      │
│  - Store in DB       │
└────┬─────────────────┘
     │
     │ 6. Return chain_id
     ▼
┌─────────────────────┐
│  Show Success        │
│  Close Modal          │
└─────────────────────┘
```

### Chain Execution Flow

```
┌──────────┐
│   User   │
└────┬─────┘
     │
     │ 1. Click "Execute Chain"
     ▼
┌─────────────────────┐
│  POST /api/chat/    │
│  hooks/chains/{id}/  │
│  execute             │
└────┬─────────────────┘
     │
     │ 2. Create Execution
     ▼
┌─────────────────────┐
│  HookChainManager    │
│  - Load chain config │
│  - Create execution  │
│  - Start Task 1      │
└────┬─────────────────┘
     │
     │ 3. Initialize Task 1
     │    with shared_session_id
     ▼
┌─────────────────────┐
│  ChatService         │
│  - Create TaskLock   │
│  - Start Workforce   │
│  - Execute Task 1    │
└────┬─────────────────┘
     │
     │ 4. Task 1 Completes
     ▼
┌─────────────────────┐
│  HookChainManager    │
│  - Capture result    │
│  - Update context    │
│  - Trigger Task 2    │
└────┬─────────────────┘
     │
     │ 5. Initialize Task 2
     │    with same session_id
     ▼
┌─────────────────────┐
│  ChatService         │
│  - Create TaskLock   │
│  - Reuse browser     │
│  - Execute Task 2    │
└────┬─────────────────┘
     │
     │ 6. Continue until
     │    all tasks complete
     ▼
┌─────────────────────┐
│  Chain Complete      │
│  - Cleanup           │
│  - Notify user       │
└─────────────────────┘
```

### Browser Session Sharing Flow

```
Task 1 Starts
     │
     │ Initialize Browser Toolkit
     ▼
┌─────────────────────────────┐
│  HybridBrowserToolkit        │
│  - shared_session_id provided│
│  - Check if session exists  │
└────┬─────────────────────────┘
     │
     │ Session exists?
     ├─── NO ──► Create new session
     │           Store in HookManager
     │
     └─── YES ──► Reuse existing session
                 
Task 1 Executes
     │
     │ Navigate, click, etc.
     ▼
┌─────────────────────────────┐
│  Browser State Preserved     │
│  - Current URL               │
│  - Cookies                   │
│  - Local Storage             │
└─────────────────────────────┘
     │
     │ Task 1 Completes
     ▼
Task 2 Starts
     │
     │ Initialize Browser Toolkit
     │ (same shared_session_id)
     ▼
┌─────────────────────────────┐
│  HybridBrowserToolkit        │
│  - Reuse session from        │
│    HookManager              │
└────┬─────────────────────────┘
     │
     │ Continue on same page
     ▼
┌─────────────────────────────┐
│  Task 2 continues where     │
│  Task 1 left off             │
└─────────────────────────────┘
```

---

## Database Schema

```sql
-- Hook Chain Definition
CREATE TABLE hook_chain (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    shared_session_id VARCHAR(64) NOT NULL,
    share_browser_session BOOLEAN DEFAULT TRUE,
    share_file_system BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_shared_session (shared_session_id)
);

-- Tasks in a Chain (References existing tasks)
CREATE TABLE hook_chain_task (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    task_order INT NOT NULL,
    source_task_id VARCHAR(64) NOT NULL,  -- Reference to existing task
    task_question TEXT,  -- Cached question for display
    task_status VARCHAR(20),  -- Cached status: 'completed', 'failed', etc.
    trigger_condition ENUM('always', 'on_success', 'on_failure') DEFAULT 'on_success',
    context_mapping JSON,  -- What context to pass to next task
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id) ON DELETE CASCADE,
    INDEX idx_chain_order (chain_id, task_order),
    INDEX idx_source_task (source_task_id)
);

-- Chain Execution Tracking
CREATE TABLE hook_execution (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    execution_id VARCHAR(64) NOT NULL UNIQUE,
    current_task_index INT DEFAULT 0,
    status ENUM('running', 'completed', 'failed', 'paused', 'cancelled') DEFAULT 'running',
    shared_context JSON,  -- Accumulated context from all tasks
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id),
    INDEX idx_execution (execution_id),
    INDEX idx_chain_status (chain_id, status),
    INDEX idx_started_at (started_at)
);

-- Task Execution Results (for context passing)
CREATE TABLE hook_task_result (
    id INT PRIMARY KEY AUTO_INCREMENT,
    execution_id VARCHAR(64) NOT NULL,
    task_id VARCHAR(64) NOT NULL,
    task_order INT NOT NULL,
    result TEXT,
    status VARCHAR(20),
    files JSON,  -- List of files created
    browser_state JSON,  -- URL, cookies, etc.
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_execution_task (execution_id, task_id),
    INDEX idx_execution_order (execution_id, task_order)
);
```

---

## API Endpoints

### Chain Management

```typescript
// Get task history for selection
GET /api/chat/history
Query: ?status=done&limit=50&offset=0
Response: {
  items: [
    {
      task_id: "task-abc-123",
      question: "Research best laptops...",
      status: "completed",
      completed_at: "2025-01-15T10:30:00Z",
      summary_task: "Research completed"
    },
    ...
  ],
  total: 100
}

// Create hook chain
POST /api/chat/hooks/chains
Body: {
  name: "E-commerce Workflow",
  description: "Research → Purchase → Review",
  shared_session_id: "session-abc-123",  // Auto-generated if not provided
  share_browser_session: true,
  share_file_system: true,
  tasks: [
    {
      task_order: 0,
      source_task_id: "task-abc-123",
      trigger_condition: "on_success",
      context_mapping: {
        include_result: true,
        include_files: true,
        include_browser_state: true
      }
    },
    {
      task_order: 1,
      source_task_id: "task-def-456",
      trigger_condition: "on_success",
      context_mapping: {
        include_result: true,
        include_files: false,
        include_browser_state: true
      }
    }
  ]
}
Response: {
  id: 1,
  name: "E-commerce Workflow",
  shared_session_id: "session-abc-123",
  created_at: "2025-01-15T10:30:00Z"
}

// List user's chains
GET /api/chat/hooks/chains
Response: {
  items: [
    {
      id: 1,
      name: "E-commerce Workflow",
      description: "...",
      task_count: 3,
      created_at: "...",
      last_executed: "..."
    },
    ...
  ],
  total: 10
}

// Get chain details
GET /api/chat/hooks/chains/{chain_id}
Response: {
  id: 1,
  name: "E-commerce Workflow",
  description: "...",
  shared_session_id: "session-abc-123",
  tasks: [
    {
      id: 1,
      task_order: 0,
      source_task_id: "task-abc-123",
      task_question: "Research best laptops...",
      task_status: "completed",
      trigger_condition: "on_success"
    },
    ...
  ]
}

// Execute chain
POST /api/chat/hooks/chains/{chain_id}/execute
Body: {
  execution_id: "exec-xyz-789",  // Optional, auto-generated if not provided
  initial_context: {  // Optional
    budget: 1000,
    category: "laptops"
  }
}
Response: {
  execution_id: "exec-xyz-789",
  chain_id: 1,
  status: "running",
  first_task_id: "task-new-001"
}

// Get execution status
GET /api/chat/hooks/executions/{execution_id}
Response: {
  execution_id: "exec-xyz-789",
  chain_id: 1,
  chain_name: "E-commerce Workflow",
  status: "running",
  current_task_index: 1,
  current_task_id: "task-new-002",
  tasks: [
    {
      task_id: "task-new-001",
      task_order: 0,
      status: "completed",
      result: "Found 5 laptops...",
      completed_at: "2025-01-15T10:35:00Z"
    },
    {
      task_id: "task-new-002",
      task_order: 1,
      status: "running",
      started_at: "2025-01-15T10:35:05Z"
    }
  ],
  shared_context: {
    "task-new-001": {
      result: "Found 5 laptops...",
      files: ["/path/to/research.txt"],
      browser_state: {
        url: "https://example.com/results"
      }
    }
  }
}
```

---

## Component Structure

### Frontend Components

```typescript
// HookChainModal.tsx
interface HookChainModalProps {
  open: boolean;
  onClose: () => void;
  onChainCreated?: (chainId: number) => void;
}

// TaskSelector.tsx (used in modal)
interface TaskSelectorProps {
  onTaskSelect: (task: TaskHistoryItem) => void;
  selectedTaskIds: string[];
}

// ChainSequenceBuilder.tsx (used in modal)
interface ChainSequenceBuilderProps {
  tasks: ChainTask[];
  onTaskAdd: (task: ChainTask) => void;
  onTaskRemove: (index: number) => void;
  onTaskReorder: (from: number, to: number) => void;
  onTaskConfigure: (index: number, config: TaskConfig) => void;
}

// HookExecutionMonitor.tsx (separate component)
interface HookExecutionMonitorProps {
  executionId: string;
  onClose?: () => void;
}
```

---

## Implementation Plan

### Phase 1: Backend Foundation (Week 1-2)

1. Database migrations
   - Create `hook_chain`, `hook_chain_task`, `hook_execution`, `hook_task_result` tables
   - Add indexes

2. Models
   - `HookChain`, `HookChainTask`, `HookExecution`, `HookTaskResult`
   - Pydantic models for API

3. HookChainManager service
   - Chain creation/retrieval
   - Execution orchestration
   - Context management
   - Session management

4. API endpoints
   - CRUD for chains
   - Execution endpoints
   - Status endpoints

### Phase 2: Browser Session Sharing (Week 2-3)

1. Session manager
   - Create/retrieve shared sessions
   - Session lifecycle
   - Cleanup

2. Browser toolkit modifications
   - Support `shared_session_id`
   - Reuse existing browser context
   - State persistence

3. Integration with ChatService
   - Detect hook chain execution
   - Initialize with shared session
   - Pass context

### Phase 3: Frontend Modal (Week 3-4)

1. HookChainModal component
   - Modal structure
   - Task history fetching
   - Task selection UI
   - Chain sequence builder
   - Configuration options

2. TaskSelector component
   - Task list display
   - Search/filter
   - Selection handling

3. ChainSequenceBuilder component
   - Drag-and-drop reordering
   - Task configuration
   - Visual flow

4. Integration with ChatBox
   - "Create Hook Chain" button
   - Modal trigger
   - Chain list display

### Phase 4: Execution & Monitoring (Week 4-5)

1. Execution flow
   - Chain execution trigger
   - Task sequencing
   - Context passing
   - Error handling

2. Execution monitor
   - Real-time status
   - Task progress
   - Context visualization
   - Pause/resume/cancel

3. UI integration
   - Execution status in ChatBox
   - Notifications
   - Error display

### Phase 5: Polish & Testing (Week 5-6)

1. Error handling
   - Failure scenarios
   - Recovery mechanisms
   - User notifications

2. Performance
   - Optimization
   - Caching
   - Database queries

3. Testing
   - Unit tests
   - Integration tests
   - E2E tests

4. Documentation
   - User guide
   - API docs
   - Code comments

---

## Advantages

### 1. User experience

- Visual chain builder
- Reuse existing tasks
- Clear sequence visualization
- Easy configuration

### 2. Technical benefits

- Centralized orchestration
- Shared session management
- Context passing
- Observability

### 3. Business value

- Automation of multi-step workflows
- Time savings
- Consistency
- Reusability

### 4. Scalability

- Supports long chains
- Parallel execution (future)
- Conditional branching (future)
- Template integration

### 5. Maintainability

- Clear separation of concerns
- Modular design
- Extensible architecture
- Testable components

---

## UI Mockups

### Task Selection View

```
┌─────────────────────────────────────────────────────────────┐
│  Select Tasks for Chain                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Search: [Search tasks...                    ] [Filter ▼]   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ✓ Research Competitor Pricing                        │  │
│  │   Task ID: task-xyz-789 | Completed 2 days ago      │  │
│  │   "Research competitor pricing for laptops..."       │  │
│  │   [Add to Chain]                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Generate Marketing Copy                            │  │
│  │   Task ID: task-def-456 | Completed 1 week ago       │  │
│  │   "Generate marketing copy for product launch..."    │  │
│  │   [Add to Chain]                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  [Load More]                                                 │
└─────────────────────────────────────────────────────────────┘
```

### Chain Sequence View

```
┌─────────────────────────────────────────────────────────────┐
│  Chain Sequence (Drag to reorder)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 1: Research Products                    [⋮⋮]     │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ Task: Research Competitor Pricing               │  │  │
│  │ │ Trigger: On Success                             │  │  │
│  │ │ [Configure] [Remove]                            │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 2: Generate Content                    [⋮⋮]     │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ Task: Generate Marketing Copy                  │  │  │
│  │ │ Trigger: On Success                             │  │  │
│  │ │ [Configure] [Remove]                            │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  [+ Add Next Task]                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

This design provides:

1. Separate modal for chain creation
2. Easy selection and sequencing of existing tasks
3. Visual chain builder
4. Shared browser session management
5. Context passing between tasks
6. Execution monitoring

The architecture is scalable, maintainable, and user-friendly, enabling complex multi-task workflows while keeping the UI simple.

Should I proceed with implementation, or do you want to review specific parts first?

```plaintext
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐         ┌──────────────────────┐            │
│  │  ChatBox         │         │  HookChainModal      │            │
│  │  (Main UI)       │────────►│  (Chain Builder)     │            │
│  │                  │  opens  │                      │            │
│  │  - Task List     │         │  - Task Selector     │            │
│  │  - Task History  │         │  - Chain Sequence    │            │
│  │  - [Create Chain]│         │  - Configuration     │            │
│  └──────────────────┘         └──────────────────────┘            │
│           │                              │                          │
│           │                              │                          │
│           └──────────────┬───────────────┘                          │
│                          │                                          │
│                          ▼                                          │
│                  ┌──────────────┐                                  │
│                  │  chatStore    │                                  │
│                  │  (Zustand)    │                                  │
│                  └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          │ API Calls
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend Layer                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │  HookChainController  │      │  HookChainManager    │            │
│  │  (API Endpoints)      │◄────►│  (Service Layer)     │            │
│  │                       │      │                      │            │
│  │  - POST /chains      │      │  - Chain Creation    │            │
│  │  - GET /chains       │      │  - Execution Control  │            │
│  │  - POST /execute     │      │  - Session Management │            │
│  └──────────────────────┘      │  - Context Passing   │            │
│           │                     └──────────────────────┘            │
│           │                              │                          │
│           │                              │                          │
│           ▼                              ▼                          │
│  ┌──────────────────┐         ┌──────────────────────┐            │
│  │  Database        │         │  ChatService        │            │
│  │  - hook_chain     │         │  (Task Execution)   │            │
│  │  - hook_execution │         │                      │            │
│  └──────────────────┘         └──────────────────────┘            │
│                                         │                          │
│                                         ▼                          │
│                              ┌──────────────────────┐            │
│                              │  BrowserToolkit       │            │
│                              │  (Shared Sessions)    │            │
│                              └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────┐
│  Create Hook Chain                                    [×]           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Chain Configuration                                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Chain Name: [E-commerce Automation Workflow        ]        │  │
│  │ Description: [Research → Purchase → Review Process]         │  │
│  │                                                              │  │
│  │ ☑ Share browser session across all tasks                   │  │
│  │ ☑ Share file system between tasks                          │  │
│  │                                                              │  │
│  │ Shared Session ID: [session-ecom-123] (auto-generated)     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Task Sequence                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ Step 1: Research Products                         │     │  │
│  │  │ ┌──────────────────────────────────────────────┐ │     │  │
│  │  │ │ Selected Task:                                │ │     │  │
│  │  │ │ [Search: "Research best laptops under $1000"] │ │     │  │
│  │  │ │ Task ID: task-abc-123                          │ │     │  │
│  │  │ │ Status: ✓ Completed                           │ │     │  │
│  │  │ │                                                │ │     │  │
│  │  │ │ Trigger Next Task:                             │ │     │  │
│  │  │ │ ( ) Always    (•) On Success    ( ) On Failure │ │     │  │
│  │  │ │                                                │ │     │  │
│  │  │ │ Context to Pass:                               │ │     │  │
│  │  │ │ ☑ Task result                                 │ │     │  │
│  │  │ │ ☑ Generated files                             │ │     │  │
│  │  │ │ ☑ Browser state (URL, cookies)                │ │     │  │
│  │  │ └──────────────────────────────────────────────┘ │     │  │
│  │  │                    │                             │     │  │
│  │  │                    ▼                             │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ Step 2: Purchase Product                            │     │  │
│  │  │ ┌──────────────────────────────────────────────┐ │     │  │
│  │  │ │ Selected Task:                                │ │     │  │
│  │  │ │ [+ Select Task from History ▼]                │ │     │  │
│  │  │ │                                                │ │     │  │
│  │  │ │ Or Create New Task:                            │ │     │  │
│  │  │ │ [Create New Task]                              │ │     │  │
│  │  │ └──────────────────────────────────────────────┘ │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  [+ Add Next Task]                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Available Tasks (Click to add to chain)                       │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ 📋 Research Competitor Pricing                      │     │  │
│  │  │ Task ID: task-xyz-789 | Status: ✓ Completed        │     │  │
│  │  │ Question: "Research competitor pricing for..."     │     │  │
│  │  │ [Add to Chain]                                      │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  ┌────────────────────────────────────────────────────┐     │  │
│  │  │ 📋 Generate Marketing Copy                           │     │  │
│  │  │ Task ID: task-def-456 | Status: ✓ Completed         │     │  │
│  │  │ Question: "Generate marketing copy for..."          │     │  │
│  │  │ [Add to Chain]                                      │     │  │
│  │  └────────────────────────────────────────────────────┘     │  │
│  │                                                              │  │
│  │  [Load More Tasks]                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [Cancel]                    [Save Chain]  [Execute Now]       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

```plaintext
User Opens Hook Chain Modal
         │
         ▼
┌────────────────────────┐
│  Fetch Task History    │
│  (from chatStore or    │
│   API endpoint)        │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  Display Task List     │
│  - Filter by status    │
│  - Search by question │
│  - Sort by date       │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  User Selects Task     │
│  (Click "Add to Chain")│
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  Add to Sequence       │
│  - Auto-increment step │
│  - Configure trigger   │
│  - Set context mapping │
└────────────────────────┘
```

```plaintext
┌──────────┐
│   User   │
└────┬─────┘
     │
     │ 1. Click "Create Hook Chain"
     ▼
┌─────────────────────┐
│  HookChainModal      │
│  Opens               │
└────┬─────────────────┘
     │
     │ 2. Fetch Task History
     ▼
┌─────────────────────┐
│  GET /api/chat/      │
│  history?status=done │
└────┬─────────────────┘
     │
     │ 3. Return Task List
     ▼
┌─────────────────────┐
│  Display Tasks       │
│  in Modal            │
└────┬─────────────────┘
     │
     │ 4. User Selects Tasks
     │    and Configures Chain
     ▼
┌─────────────────────┐
│  User Clicks         │
│  "Save Chain"        │
└────┬─────────────────┘
     │
     │ 5. POST /api/chat/hooks/chains
     ▼
┌─────────────────────┐
│  HookChainController │
│  - Validate data     │
│  - Create chain      │
│  - Store in DB       │
└────┬─────────────────┘
     │
     │ 6. Return chain_id
     ▼
┌─────────────────────┐
│  Show Success        │
│  Close Modal          │
└─────────────────────┘
```

```plaintext
┌──────────┐
│   User   │
└────┬─────┘
     │
     │ 1. Click "Execute Chain"
     ▼
┌─────────────────────┐
│  POST /api/chat/    │
│  hooks/chains/{id}/  │
│  execute             │
└────┬─────────────────┘
     │
     │ 2. Create Execution
     ▼
┌─────────────────────┐
│  HookChainManager    │
│  - Load chain config │
│  - Create execution  │
│  - Start Task 1      │
└────┬─────────────────┘
     │
     │ 3. Initialize Task 1
     │    with shared_session_id
     ▼
┌─────────────────────┐
│  ChatService         │
│  - Create TaskLock   │
│  - Start Workforce   │
│  - Execute Task 1    │
└────┬─────────────────┘
     │
     │ 4. Task 1 Completes
     ▼
┌─────────────────────┐
│  HookChainManager    │
│  - Capture result    │
│  - Update context    │
│  - Trigger Task 2    │
└────┬─────────────────┘
     │
     │ 5. Initialize Task 2
     │    with same session_id
     ▼
┌─────────────────────┐
│  ChatService         │
│  - Create TaskLock   │
│  - Reuse browser     │
│  - Execute Task 2    │
└────┬─────────────────┘
     │
     │ 6. Continue until
     │    all tasks complete
     ▼
┌─────────────────────┐
│  Chain Complete      │
│  - Cleanup           │
│  - Notify user       │
└─────────────────────┘
```

```plaintext
Task 1 Starts
     │
     │ Initialize Browser Toolkit
     ▼
┌─────────────────────────────┐
│  HybridBrowserToolkit        │
│  - shared_session_id provided│
│  - Check if session exists  │
└────┬─────────────────────────┘
     │
     │ Session exists?
     ├─── NO ──► Create new session
     │           Store in HookManager
     │
     └─── YES ──► Reuse existing session
                 
Task 1 Executes
     │
     │ Navigate, click, etc.
     ▼
┌─────────────────────────────┐
│  Browser State Preserved     │
│  - Current URL               │
│  - Cookies                   │
│  - Local Storage             │
└─────────────────────────────┘
     │
     │ Task 1 Completes
     ▼
Task 2 Starts
     │
     │ Initialize Browser Toolkit
     │ (same shared_session_id)
     ▼
┌─────────────────────────────┐
│  HybridBrowserToolkit        │
│  - Reuse session from        │
│    HookManager              │
└────┬─────────────────────────┘
     │
     │ Continue on same page
     ▼
┌─────────────────────────────┐
│  Task 2 continues where     │
│  Task 1 left off             │
└─────────────────────────────┘
```

```sql
-- Hook Chain Definition
CREATE TABLE hook_chain (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    shared_session_id VARCHAR(64) NOT NULL,
    share_browser_session BOOLEAN DEFAULT TRUE,
    share_file_system BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_shared_session (shared_session_id)
);

-- Tasks in a Chain (References existing tasks)
CREATE TABLE hook_chain_task (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    task_order INT NOT NULL,
    source_task_id VARCHAR(64) NOT NULL,  -- Reference to existing task
    task_question TEXT,  -- Cached question for display
    task_status VARCHAR(20),  -- Cached status: 'completed', 'failed', etc.
    trigger_condition ENUM('always', 'on_success', 'on_failure') DEFAULT 'on_success',
    context_mapping JSON,  -- What context to pass to next task
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id) ON DELETE CASCADE,
    INDEX idx_chain_order (chain_id, task_order),
    INDEX idx_source_task (source_task_id)
);

-- Chain Execution Tracking
CREATE TABLE hook_execution (
    id INT PRIMARY KEY AUTO_INCREMENT,
    chain_id INT NOT NULL,
    execution_id VARCHAR(64) NOT NULL UNIQUE,
    current_task_index INT DEFAULT 0,
    status ENUM('running', 'completed', 'failed', 'paused', 'cancelled') DEFAULT 'running',
    shared_context JSON,  -- Accumulated context from all tasks
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (chain_id) REFERENCES hook_chain(id),
    INDEX idx_execution (execution_id),
    INDEX idx_chain_status (chain_id, status),
    INDEX idx_started_at (started_at)
);

-- Task Execution Results (for context passing)
CREATE TABLE hook_task_result (
    id INT PRIMARY KEY AUTO_INCREMENT,
    execution_id VARCHAR(64) NOT NULL,
    task_id VARCHAR(64) NOT NULL,
    task_order INT NOT NULL,
    result TEXT,
    status VARCHAR(20),
    files JSON,  -- List of files created
    browser_state JSON,  -- URL, cookies, etc.
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_execution_task (execution_id, task_id),
    INDEX idx_execution_order (execution_id, task_order)
);
```

```typescript
// Get task history for selection
GET /api/chat/history
Query: ?status=done&limit=50&offset=0
Response: {
  items: [
    {
      task_id: "task-abc-123",
      question: "Research best laptops...",
      status: "completed",
      completed_at: "2025-01-15T10:30:00Z",
      summary_task: "Research completed"
    },
    ...
  ],
  total: 100
}

// Create hook chain
POST /api/chat/hooks/chains
Body: {
  name: "E-commerce Workflow",
  description: "Research → Purchase → Review",
  shared_session_id: "session-abc-123",  // Auto-generated if not provided
  share_browser_session: true,
  share_file_system: true,
  tasks: [
    {
      task_order: 0,
      source_task_id: "task-abc-123",
      trigger_condition: "on_success",
      context_mapping: {
        include_result: true,
        include_files: true,
        include_browser_state: true
      }
    },
    {
      task_order: 1,
      source_task_id: "task-def-456",
      trigger_condition: "on_success",
      context_mapping: {
        include_result: true,
        include_files: false,
        include_browser_state: true
      }
    }
  ]
}
Response: {
  id: 1,
  name: "E-commerce Workflow",
  shared_session_id: "session-abc-123",
  created_at: "2025-01-15T10:30:00Z"
}

// List user's chains
GET /api/chat/hooks/chains
Response: {
  items: [
    {
      id: 1,
      name: "E-commerce Workflow",
      description: "...",
      task_count: 3,
      created_at: "...",
      last_executed: "..."
    },
    ...
  ],
  total: 10
}

// Get chain details
GET /api/chat/hooks/chains/{chain_id}
Response: {
  id: 1,
  name: "E-commerce Workflow",
  description: "...",
  shared_session_id: "session-abc-123",
  tasks: [
    {
      id: 1,
      task_order: 0,
      source_task_id: "task-abc-123",
      task_question: "Research best laptops...",
      task_status: "completed",
      trigger_condition: "on_success"
    },
    ...
  ]
}

// Execute chain
POST /api/chat/hooks/chains/{chain_id}/execute
Body: {
  execution_id: "exec-xyz-789",  // Optional, auto-generated if not provided
  initial_context: {  // Optional
    budget: 1000,
    category: "laptops"
  }
}
Response: {
  execution_id: "exec-xyz-789",
  chain_id: 1,
  status: "running",
  first_task_id: "task-new-001"
}

// Get execution status
GET /api/chat/hooks/executions/{execution_id}
Response: {
  execution_id: "exec-xyz-789",
  chain_id: 1,
  chain_name: "E-commerce Workflow",
  status: "running",
  current_task_index: 1,
  current_task_id: "task-new-002",
  tasks: [
    {
      task_id: "task-new-001",
      task_order: 0,
      status: "completed",
      result: "Found 5 laptops...",
      completed_at: "2025-01-15T10:35:00Z"
    },
    {
      task_id: "task-new-002",
      task_order: 1,
      status: "running",
      started_at: "2025-01-15T10:35:05Z"
    }
  ],
  shared_context: {
    "task-new-001": {
      result: "Found 5 laptops...",
      files: ["/path/to/research.txt"],
      browser_state: {
        url: "https://example.com/results"
      }
    }
  }
}
```

```typescript
// HookChainModal.tsx
interface HookChainModalProps {
  open: boolean;
  onClose: () => void;
  onChainCreated?: (chainId: number) => void;
}

// TaskSelector.tsx (used in modal)
interface TaskSelectorProps {
  onTaskSelect: (task: TaskHistoryItem) => void;
  selectedTaskIds: string[];
}

// ChainSequenceBuilder.tsx (used in modal)
interface ChainSequenceBuilderProps {
  tasks: ChainTask[];
  onTaskAdd: (task: ChainTask) => void;
  onTaskRemove: (index: number) => void;
  onTaskReorder: (from: number, to: number) => void;
  onTaskConfigure: (index: number, config: TaskConfig) => void;
}

// HookExecutionMonitor.tsx (separate component)
interface HookExecutionMonitorProps {
  executionId: string;
  onClose?: () => void;
}
```

```plaintext
┌─────────────────────────────────────────────────────────────┐
│  Select Tasks for Chain                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Search: [Search tasks...                    ] [Filter ▼]   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ✓ Research Competitor Pricing                        │  │
│  │   Task ID: task-xyz-789 | Completed 2 days ago      │  │
│  │   "Research competitor pricing for laptops..."       │  │
│  │   [Add to Chain]                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Generate Marketing Copy                            │  │
│  │   Task ID: task-def-456 | Completed 1 week ago       │  │
│  │   "Generate marketing copy for product launch..."    │  │
│  │   [Add to Chain]                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  [Load More]                                                 │
└─────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────┐
│  Chain Sequence (Drag to reorder)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 1: Research Products                    [⋮⋮]     │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ Task: Research Competitor Pricing               │  │  │
│  │ │ Trigger: On Success                             │  │  │
│  │ │ [Configure] [Remove]                            │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                    │                                         │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Step 2: Generate Content                    [⋮⋮]     │  │
│  │ ┌────────────────────────────────────────────────┐  │  │
│  │ │ Task: Generate Marketing Copy                  │  │  │
│  │ │ Trigger: On Success                             │  │  │
│  │ │ [Configure] [Remove]                            │  │  │
│  │ └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  [+ Add Next Task]                                           │
└─────────────────────────────────────────────────────────────┘
```
