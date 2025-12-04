# Understanding replay functionality and limitations
_Exported on 04/12/2025 at 17:01:30 GMT+5 from Cursor (2.1.39)_

---

**User**

Tell me about the replay functionality and its limitations

---

**Cursor**

Based on my analysis of the codebase, here's a comprehensive overview of the **replay functionality** and its **limitations**:

## How Replay Works

The replay feature allows users to re-watch completed task workflows. It works by:

1. **Fetching stored chat steps** from the database (`ChatStep` model) via the `/api/chat/steps/playback/{task_id}` endpoint
2. **Streaming the steps** back to the frontend using Server-Sent Events (SSE)
3. **Reconstructing the UI state** in the Zustand store (`chatStore`)

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Frontend store | `src/store/chatStore.ts` | `replay()` function at line 2358 |
| Playback API | `server/app/controller/chat/step_controller.py` | SSE streaming endpoint |
| Share playback | `server/app/controller/chat/share_controller.py` | Shared workflow replay |
| Snapshots | `server/app/controller/chat/snapshot_controller.py` | Browser session images |

### Replay Flow

```
User clicks replay → chatStore.replay() → Creates task in store → 
Calls startTask() with type="replay" → SSE connection to /playback/{task_id} →
Steps streamed to frontend → UI state reconstructed
```

## Limitations

### 1. **Only Finished Tasks Can Be Replayed**
The replay button is disabled unless `activeTask.status === 'finished'`:

```823:823:src/components/ChatBox/index.tsx
                        disabled={loading || !activeTask || activeTask.status !== 'finished'}
```

### 2. **Maximum Playback Delay of 5 Seconds**
The `delay_time` parameter (for controlling playback speed) is capped at 5 seconds:

```35:36:server/app/controller/chat/step_controller.py
    if delay_time > 5:
        delay_time = 5
```

### 3. **Backwards Compatibility with Old Tasks**
Older workflows may not have `user_message` steps stored. The code handles this by manually adding the initial question:

```2388:2408:src/store/chatStore.ts
    // Check if there are any user_message steps in the database
    // For backwards compatibility with old tasks that don't have user_message steps
    let hasUserMessages = false;
    // ... 
    // If no user_message steps exist (old workflow), add the initial question manually
    if (!hasUserMessages && question) {
      addMessages(taskId, {
        id: generateUniqueId(),
        role: "user",
        content: question.split("|")[0],
      });
    }
```

### 4. **Task History Must Exist**
If the task history cannot be found in the database, replay fails with error:
> "Could not find task history. Please wait a moment and try again."

### 5. **No Live Browser Sessions**
During replay, browser sessions are shown via **pre-stored snapshots** only - no actual live browser interaction is possible. From the documentation:
> "Snapshots are pre-loaded for browser sessions"

### 6. **Confirmation Dialogs Are Auto-Skipped**
During replay, `isConfirm` is auto-set to skip confirmation dialogs that would have appeared during the original execution.

### 7. **Shared Link Expiration**
For shared workflows, the replay token expires (default 1 day). Attempting to replay with an expired token results in:
> "Share link has expired."

### 8. **No Steps = No Replay**
If no steps are stored for a task, the endpoint returns an error:

```42:44:server/app/controller/chat/step_controller.py
        if not steps:
            yield f"data: {json.dumps({'error': 'No steps found for this task.'})}\n\n"
            return
```

### 9. **Read-Only Playback**
Replay is purely for viewing - you cannot interact with or modify the workflow during playback. It's essentially a recording, not a resumable session.

```plaintext
User clicks replay → chatStore.replay() → Creates task in store → 
Calls startTask() with type="replay" → SSE connection to /playback/{task_id} →
Steps streamed to frontend → UI state reconstructed
```

```typescriptreact
                        disabled={loading || !activeTask || activeTask.status !== 'finished'}
```

```python
    if delay_time > 5:
        delay_time = 5
```

```typescript
    // Check if there are any user_message steps in the database
    // For backwards compatibility with old tasks that don't have user_message steps
    let hasUserMessages = false;
    // ... 
    // If no user_message steps exist (old workflow), add the initial question manually
    if (!hasUserMessages && question) {
      addMessages(taskId, {
        id: generateUniqueId(),
        role: "user",
        content: question.split("|")[0],
      });
    }
```

```python
        if not steps:
            yield f"data: {json.dumps({'error': 'No steps found for this task.'})}\n\n"
            return
```

---

**User**

Than whats the benefit of this functionality

---

**Cursor**

Great question! Despite the limitations, the **replay functionality** provides several valuable benefits:

## Key Benefits

### 1. **Debugging & Troubleshooting**
- Review what went wrong if a task failed or produced unexpected results
- See the exact sequence of steps, agent assignments, and tool calls that occurred
- Identify bottlenecks or errors in the workflow execution

### 2. **Learning & Training**
- New users can watch how successful tasks were completed
- Understand how agents break down complex tasks into subtasks
- Learn from past workflows to craft better prompts in the future

### 3. **Audit Trail & Documentation**
- Provides a complete record of what happened during task execution
- Useful for compliance or accountability purposes
- Document workflows for team knowledge sharing

### 4. **Sharing with Others**
The share functionality (`share_controller.py`) lets you:
- Generate a shareable link (valid for 1 day) to show others a workflow
- Demonstrate results to stakeholders without re-running the task
- Collaborate by sharing successful task patterns

### 5. **Cost Savings**
- Replay doesn't consume tokens or API calls - it's free to re-watch
- No need to re-run expensive tasks just to show someone the process
- Review results without incurring additional LLM/browser costs

### 6. **Time Savings**
- Instantly review completed work without waiting for re-execution
- Can control playback speed with `delayTime` (0-5 seconds) to watch faster or slower
- Jump back to any previous workflow from history

### 7. **Visual Context via Snapshots**
The snapshot system (`snapshot_controller.py`) preserves:
- Browser screenshots from browser automation tasks
- Visual proof of what the agent "saw" during execution
- Context that would otherwise be lost after task completion

---

**In essence**: Replay turns ephemeral task execution into a **persistent, reviewable artifact** - similar to how screen recordings let you review meetings. You can't interact with it, but you can learn from it, share it, and use it as documentation without any additional compute cost.

---

**User**

I want to get over this limitation. When user start a chat it creates a coordinator/orchestrator agent which creates in to subtasks and save it in database. It also creates other agents as well and sabe them in database. Than user approves the tasks and agent start execution.



Than it creates multiple steps to get to a final result. 



After that we can not execute again to get the same output. My idea is I want to create a new button If the task was previously completed it will use its same subtasks and agents to give the same realtime output again. Instead for starting from scratch.



What are the solutions you suggest. How we can use databse to save it and reuse it. Give me best approaches

---

**Cursor**

Let me first understand the current data models and task execution flow to give you the best approach.

Now I have a complete picture of your architecture. Here are the **best approaches** for implementing a "Re-execute" feature that reuses subtasks and agents:

---

## Current Data Already Stored

You're in a good position! The database already stores:

| Table | What's Stored | Useful For Re-execution |
|-------|--------------|-------------------------|
| `chat_history` | Task metadata, model config, question | ✅ Recreate task configuration |
| `chat_step` (step=`to_sub_tasks`) | Subtask breakdown with IDs and content | ✅ Reuse subtask plan |
| `chat_step` (step=`create_agent`) | Agent names, IDs, and tools | ✅ Recreate agents |
| `chat_step` (step=`assign_task`) | Which agent handles which subtask | ✅ Reuse assignments |

---

## Recommended Approaches

### **Approach 1: Template-Based Re-execution (Recommended)**

Create a new database model `workflow_template` that extracts and stores reusable workflow structures.

#### New Database Model

```python
# server/app/model/chat/workflow_template.py

from sqlmodel import SQLModel, Field, JSON, Column
from app.model.abstract.model import AbstractModel, DefaultTimes
from typing import Optional

class WorkflowTemplate(AbstractModel, DefaultTimes, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    source_task_id: str = Field(index=True)  # Original task this was created from
    name: str  # User-friendly name
    
    # Stored workflow structure
    subtasks: str = Field(sa_type=JSON)  # [{id, content, agent_type}, ...]
    agents: str = Field(sa_type=JSON)    # [{name, tools}, ...]
    assignments: str = Field(sa_type=JSON)  # {subtask_id: agent_name, ...}
    
    # Optional: Store model preferences
    model_platform: Optional[str] = None
    model_type: Optional[str] = None
```

#### Workflow

1. **After task completes** → Extract structure from `chat_step` records
2. **Save as template** → Store in `workflow_template`
3. **Re-execute** → Load template, skip orchestration, jump to execution

#### Backend Endpoint

```python
# server/app/controller/chat/template_controller.py

@router.post("/template/from-task/{task_id}", name="Create template from completed task")
async def create_template_from_task(
    task_id: str, 
    name: str,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    # 1. Get the to_sub_tasks step
    subtasks_step = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id, ChatStep.step == "to_sub_tasks")
    ).first()
    
    # 2. Get all create_agent steps
    agent_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id, ChatStep.step == "create_agent")
    ).all()
    
    # 3. Get all assign_task steps
    assignment_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id, ChatStep.step == "assign_task")
    ).all()
    
    # 4. Create template
    template = WorkflowTemplate(
        user_id=auth.user.id,
        source_task_id=task_id,
        name=name,
        subtasks=subtasks_step.data,
        agents=[step.data for step in agent_steps],
        assignments=[step.data for step in assignment_steps],
    )
    session.add(template)
    session.commit()
    return template


@router.post("/execute-template/{template_id}", name="Execute from template")
async def execute_from_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    template = session.get(WorkflowTemplate, template_id)
    # Returns template data for backend worker to use
    return {
        "mode": "template_execution",
        "subtasks": template.subtasks,
        "agents": template.agents,
        "assignments": template.assignments
    }
```

---

### **Approach 2: Direct Re-execution from Chat Steps (Simpler)**

Skip template creation and directly reuse existing `chat_step` data.

#### Backend Changes

```python
# backend/app/service/chat_service.py

async def step_solve_from_template(options: Chat, request: Request, task_lock: TaskLock, source_task_id: str):
    """Re-execute using saved workflow structure from a previous task."""
    
    # 1. Fetch previous workflow structure via API
    previous_steps = await fetch_previous_workflow(source_task_id)
    
    # 2. Extract subtasks from to_sub_tasks step
    to_sub_tasks_step = next(s for s in previous_steps if s['step'] == 'to_sub_tasks')
    predefined_subtasks = to_sub_tasks_step['data']['sub_tasks']
    
    # 3. Extract agent configurations
    agent_configs = [s['data'] for s in previous_steps if s['step'] == 'create_agent']
    
    # 4. Skip orchestration, directly create workforce with predefined structure
    (workforce, mcp) = await construct_workforce_from_template(
        options, 
        predefined_subtasks,
        agent_configs
    )
    
    # 5. Start execution (same as normal flow)
    yield sse_json("to_sub_tasks", {
        "summary_task": to_sub_tasks_step['data']['summary_task'],
        "sub_tasks": predefined_subtasks
    })
    
    # ... continue with execution


async def construct_workforce_from_template(
    options: Chat, 
    subtasks: list, 
    agent_configs: list
) -> tuple[Workforce, ListenChatAgent]:
    """Build workforce from saved template instead of AI orchestration."""
    
    # Create agents based on saved configs
    agents = []
    for config in agent_configs:
        agent = agent_model(
            config['agent_name'],
            get_system_prompt_for_agent(config['agent_name']),
            options,
            tools=get_tools_for_agent(config['agent_name'], config.get('tools', []))
        )
        agents.append(agent)
    
    # Build workforce
    workforce = Workforce(...)
    for agent in agents:
        workforce.add_single_agent_worker(agent.description, agent)
    
    return workforce, mcp
```

---

### **Approach 3: Clone & Execute (Fastest Implementation)**

Add a simple "Clone Task" button that creates a new task with pre-filled subtasks.

#### Frontend Changes

```typescript
// src/store/chatStore.ts

reExecute: async (sourceTaskId: string) => {
  const { create, setActiveTaskId, startTask, setTaskInfo, setSummaryTask } = get();
  
  // 1. Fetch original task's steps
  const steps = await proxyFetchGet(`/api/chat/steps`, { task_id: sourceTaskId });
  
  // 2. Find the to_sub_tasks step
  const subTasksStep = steps.find((s: any) => s.step === 'to_sub_tasks');
  
  // 3. Create new task
  const newTaskId = generateUniqueId();
  create(newTaskId, "re_execute");
  
  // 4. Pre-populate with saved structure (skip AI orchestration)
  if (subTasksStep?.data) {
    setSummaryTask(newTaskId, subTasksStep.data.summary_task);
    setTaskInfo(newTaskId, subTasksStep.data.sub_tasks);
  }
  
  // 5. Start with special mode flag
  await startTask(newTaskId, "re_execute", undefined, 0, { 
    sourceTaskId,
    skipOrchestration: true 
  });
  
  setActiveTaskId(newTaskId);
}
```

#### Backend: New Action Type

```python
# backend/app/service/task.py

class Action(str, Enum):
    # ... existing actions ...
    re_execute = "re_execute"  # New: Re-execute from template
```

---

## Comparison of Approaches

| Aspect | Approach 1 (Templates) | Approach 2 (Direct) | Approach 3 (Clone) |
|--------|----------------------|--------------------|--------------------|
| **Complexity** | High | Medium | Low |
| **New DB Tables** | Yes (1 table) | No | No |
| **User Control** | Can name & manage templates | Less flexible | Simple |
| **Best For** | Repeated workflows | One-off re-runs | Quick implementation |
| **Scalability** | Best | Good | Basic |

---

## My Recommendation

**Start with Approach 3 (Clone & Execute)** for a quick win, then evolve to **Approach 1 (Templates)** for production.

### Key Implementation Steps:

1. **Add a "Re-Execute" button** next to the existing Replay button (only for finished tasks)
2. **Create new endpoint** `/api/chat/re-execute/{task_id}` that:
   - Fetches the `to_sub_tasks` step from original task
   - Creates a new task with a new `task_id`
   - Passes the saved subtask structure to the backend
3. **Modify `step_solve`** to accept an optional `predefined_subtasks` parameter that skips AI orchestration
4. **Send same SSE events** as normal execution (so UI works unchanged)

Would you like me to provide more detailed implementation code for any of these approaches?

```python
# server/app/model/chat/workflow_template.py

from sqlmodel import SQLModel, Field, JSON, Column
from app.model.abstract.model import AbstractModel, DefaultTimes
from typing import Optional

class WorkflowTemplate(AbstractModel, DefaultTimes, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    source_task_id: str = Field(index=True)  # Original task this was created from
    name: str  # User-friendly name
    
    # Stored workflow structure
    subtasks: str = Field(sa_type=JSON)  # [{id, content, agent_type}, ...]
    agents: str = Field(sa_type=JSON)    # [{name, tools}, ...]
    assignments: str = Field(sa_type=JSON)  # {subtask_id: agent_name, ...}
    
    # Optional: Store model preferences
    model_platform: Optional[str] = None
    model_type: Optional[str] = None
```

```python
# server/app/controller/chat/template_controller.py

@router.post("/template/from-task/{task_id}", name="Create template from completed task")
async def create_template_from_task(
    task_id: str, 
    name: str,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    # 1. Get the to_sub_tasks step
    subtasks_step = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id, ChatStep.step == "to_sub_tasks")
    ).first()
    
    # 2. Get all create_agent steps
    agent_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id, ChatStep.step == "create_agent")
    ).all()
    
    # 3. Get all assign_task steps
    assignment_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id, ChatStep.step == "assign_task")
    ).all()
    
    # 4. Create template
    template = WorkflowTemplate(
        user_id=auth.user.id,
        source_task_id=task_id,
        name=name,
        subtasks=subtasks_step.data,
        agents=[step.data for step in agent_steps],
        assignments=[step.data for step in assignment_steps],
    )
    session.add(template)
    session.commit()
    return template


@router.post("/execute-template/{template_id}", name="Execute from template")
async def execute_from_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    template = session.get(WorkflowTemplate, template_id)
    # Returns template data for backend worker to use
    return {
        "mode": "template_execution",
        "subtasks": template.subtasks,
        "agents": template.agents,
        "assignments": template.assignments
    }
```

```python
# backend/app/service/chat_service.py

async def step_solve_from_template(options: Chat, request: Request, task_lock: TaskLock, source_task_id: str):
    """Re-execute using saved workflow structure from a previous task."""
    
    # 1. Fetch previous workflow structure via API
    previous_steps = await fetch_previous_workflow(source_task_id)
    
    # 2. Extract subtasks from to_sub_tasks step
    to_sub_tasks_step = next(s for s in previous_steps if s['step'] == 'to_sub_tasks')
    predefined_subtasks = to_sub_tasks_step['data']['sub_tasks']
    
    # 3. Extract agent configurations
    agent_configs = [s['data'] for s in previous_steps if s['step'] == 'create_agent']
    
    # 4. Skip orchestration, directly create workforce with predefined structure
    (workforce, mcp) = await construct_workforce_from_template(
        options, 
        predefined_subtasks,
        agent_configs
    )
    
    # 5. Start execution (same as normal flow)
    yield sse_json("to_sub_tasks", {
        "summary_task": to_sub_tasks_step['data']['summary_task'],
        "sub_tasks": predefined_subtasks
    })
    
    # ... continue with execution


async def construct_workforce_from_template(
    options: Chat, 
    subtasks: list, 
    agent_configs: list
) -> tuple[Workforce, ListenChatAgent]:
    """Build workforce from saved template instead of AI orchestration."""
    
    # Create agents based on saved configs
    agents = []
    for config in agent_configs:
        agent = agent_model(
            config['agent_name'],
            get_system_prompt_for_agent(config['agent_name']),
            options,
            tools=get_tools_for_agent(config['agent_name'], config.get('tools', []))
        )
        agents.append(agent)
    
    # Build workforce
    workforce = Workforce(...)
    for agent in agents:
        workforce.add_single_agent_worker(agent.description, agent)
    
    return workforce, mcp
```

```typescript
// src/store/chatStore.ts

reExecute: async (sourceTaskId: string) => {
  const { create, setActiveTaskId, startTask, setTaskInfo, setSummaryTask } = get();
  
  // 1. Fetch original task's steps
  const steps = await proxyFetchGet(`/api/chat/steps`, { task_id: sourceTaskId });
  
  // 2. Find the to_sub_tasks step
  const subTasksStep = steps.find((s: any) => s.step === 'to_sub_tasks');
  
  // 3. Create new task
  const newTaskId = generateUniqueId();
  create(newTaskId, "re_execute");
  
  // 4. Pre-populate with saved structure (skip AI orchestration)
  if (subTasksStep?.data) {
    setSummaryTask(newTaskId, subTasksStep.data.summary_task);
    setTaskInfo(newTaskId, subTasksStep.data.sub_tasks);
  }
  
  // 5. Start with special mode flag
  await startTask(newTaskId, "re_execute", undefined, 0, { 
    sourceTaskId,
    skipOrchestration: true 
  });
  
  setActiveTaskId(newTaskId);
}
```

```python
# backend/app/service/task.py

class Action(str, Enum):
    # ... existing actions ...
    re_execute = "re_execute"  # New: Re-execute from template
```

---

**User**

I like the template based approach.Go though the entire codebase and don't skip anything. Give me a detailed plan how to achieve it. Along with the necessary diagrams

---

**Cursor**

I'll do a comprehensive analysis of the codebase to create a detailed implementation plan. Let me explore all the relevant components.

Now I have a complete understanding of the codebase. Here's a **comprehensive implementation plan** for the Template-Based Re-execution feature:

---

# 🚀 Template-Based Workflow Re-Execution - Detailed Implementation Plan

## 📊 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            CURRENT EXECUTION FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │  User    │───▶│   Backend   │───▶│  AI Orchestrator │───▶│  Task Decomposer   │     │
│  │  Query   │    │   /chat     │    │  (Coordinator)   │    │  (_decompose_task) │     │
│  └──────────┘    └─────────────┘    └──────────────────┘    └────────────────────┘     │
│                                                                         │               │
│                                                                         ▼               │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │   End    │◀───│   Results   │◀───│  Agent Execution │◀───│   Subtasks List    │     │
│  │  Result  │    │   Stream    │    │  (Workforce)     │    │   (from AI)        │     │
│  └──────────┘    └─────────────┘    └──────────────────┘    └────────────────────┘     │
│                                                                                          │
│                        💰 Cost: AI tokens for decomposition + execution                  │
│                        ⏱️ Time: Includes AI thinking time for planning                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         NEW TEMPLATE EXECUTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐                               │
│  │  Select  │───▶│   Backend   │───▶│ Load Template    │                               │
│  │ Template │    │ /re-execute │    │ from Database    │                               │
│  └──────────┘    └─────────────┘    └──────────────────┘                               │
│                                              │                                          │
│                                              ▼ SKIP AI DECOMPOSITION                    │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │   End    │◀───│   Results   │◀───│  Agent Execution │◀───│  Pre-saved         │     │
│  │  Result  │    │   Stream    │    │  (Workforce)     │    │  Subtasks          │     │
│  └──────────┘    └─────────────┘    └──────────────────┘    └────────────────────┘     │
│                                                                                          │
│                        💰 Cost: Only AI tokens for execution (save ~30-50%)             │
│                        ⏱️ Time: Immediate start (no planning delay)                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Database Design

### New Table: `workflow_template`

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              workflow_template                                         │
├──────────────────┬──────────────┬─────────────────────────────────────────────────────┤
│ Column           │ Type         │ Description                                         │
├──────────────────┼──────────────┼─────────────────────────────────────────────────────┤
│ id               │ INTEGER PK   │ Auto-increment primary key                          │
│ user_id          │ INTEGER FK   │ Owner of the template                               │
│ source_task_id   │ VARCHAR(64)  │ Original task_id this template was created from    │
│ name             │ VARCHAR(128) │ User-friendly template name                         │
│ description      │ TEXT         │ Optional description of what this template does    │
│ summary_task     │ VARCHAR(512) │ The summary generated by AI                         │
│ subtasks         │ JSON         │ Array of subtask objects from to_sub_tasks          │
│ agents           │ JSON         │ Array of agent configs from create_agent events     │
│ agent_assignments│ JSON         │ Map of subtask_id -> agent_id/agent_name            │
│ model_config     │ JSON         │ Optional: store model preferences                   │
│ is_public        │ BOOLEAN      │ Whether template can be shared                      │
│ use_count        │ INTEGER      │ How many times this template has been used          │
│ created_at       │ TIMESTAMP    │ Creation timestamp                                  │
│ updated_at       │ TIMESTAMP    │ Last update timestamp                               │
│ deleted_at       │ TIMESTAMP    │ Soft delete timestamp                               │
└──────────────────┴──────────────┴─────────────────────────────────────────────────────┘
```

### Data Extraction from Existing Steps

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          DATA MAPPING: chat_step → workflow_template                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  chat_step (step="to_sub_tasks")                  workflow_template                    │
│  ┌──────────────────────────────┐                 ┌──────────────────────────────┐    │
│  │ {                            │                 │                              │    │
│  │   "summary_task": "...",     │ ──────────────▶ │ summary_task: "..."          │    │
│  │   "sub_tasks": [             │                 │                              │    │
│  │     {                        │                 │ subtasks: [                  │    │
│  │       "id": "task.1",        │ ──────────────▶ │   {id, content, state}       │    │
│  │       "content": "...",      │                 │ ]                            │    │
│  │       "state": "..."         │                 │                              │    │
│  │     }                        │                 │                              │    │
│  │   ]                          │                 │                              │    │
│  │ }                            │                 │                              │    │
│  └──────────────────────────────┘                 │                              │    │
│                                                   │                              │    │
│  chat_step (step="create_agent")                  │                              │    │
│  ┌──────────────────────────────┐                 │                              │    │
│  │ {                            │                 │ agents: [                    │    │
│  │   "agent_name": "...",       │ ──────────────▶ │   {agent_name, agent_id,     │    │
│  │   "agent_id": "...",         │                 │    tools}                    │    │
│  │   "tools": [...]             │                 │ ]                            │    │
│  │ }                            │                 │                              │    │
│  └──────────────────────────────┘                 │                              │    │
│                                                   │                              │    │
│  chat_step (step="assign_task")                   │                              │    │
│  ┌──────────────────────────────┐                 │                              │    │
│  │ {                            │                 │ agent_assignments: {         │    │
│  │   "assignee_id": "...",      │ ──────────────▶ │   "task.1": "agent_id",      │    │
│  │   "task_id": "..."           │                 │   "task.2": "agent_id"       │    │
│  │ }                            │                 │ }                            │    │
│  └──────────────────────────────┘                 └──────────────────────────────┘    │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    COMPLETE DATA FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────── PHASE 1: TEMPLATE CREATION ──────────────────────────┐│
│  │                                                                                          ││
│  │     User completes task                                                                  ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐                                                                     ││
│  │  │ "Save Template" │                                                                     ││
│  │  │    Button       │                                                                     ││
│  │  └────────┬────────┘                                                                     ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ POST            │───▶│ Server API      │───▶│ Extract from chat_step:         │      ││
│  │  │ /api/template/  │    │ template_       │    │ • to_sub_tasks → subtasks       │      ││
│  │  │ from-task/{id}  │    │ controller.py   │    │ • create_agent → agents         │      ││
│  │  └─────────────────┘    └─────────────────┘    │ • assign_task → assignments     │      ││
│  │                                                 └──────────────┬──────────────────┘      ││
│  │                                                                │                         ││
│  │                                                                ▼                         ││
│  │                                                 ┌─────────────────────────────────┐      ││
│  │                                                 │ INSERT INTO workflow_template   │      ││
│  │                                                 │ (subtasks, agents, assignments) │      ││
│  │                                                 └─────────────────────────────────┘      ││
│  │                                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                              │
│  ┌─────────────────────────────────── PHASE 2: TEMPLATE EXECUTION ─────────────────────────┐│
│  │                                                                                          ││
│  │     User selects template                                                                ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ "Re-Execute"    │───▶│ GET             │───▶│ SELECT * FROM workflow_template │      ││
│  │  │    Button       │    │ /api/template/  │    │ WHERE id = :template_id         │      ││
│  │  └─────────────────┘    │ {id}            │    └──────────────┬──────────────────┘      ││
│  │                         └─────────────────┘                   │                         ││
│  │                                                               ▼                         ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ Frontend Store  │◀───│ Template Data   │◀───│ Return template JSON            │      ││
│  │  │ chatStore       │    │ Response        │    │ {subtasks, agents, assignments} │      ││
│  │  └────────┬────────┘    └─────────────────┘    └─────────────────────────────────┘      ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ POST /chat      │───▶│ step_solve_     │───▶│ SKIP: _decompose_task()         │      ││
│  │  │ (with template  │    │ from_template() │    │ USE:  template.subtasks         │      ││
│  │  │  mode flag)     │    │                 │    │ SKIP: AI planning               │      ││
│  │  └─────────────────┘    └─────────────────┘    └──────────────┬──────────────────┘      ││
│  │                                                               │                         ││
│  │                                                               ▼                         ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ SSE Stream      │◀───│ Results via SSE │◀───│ workforce.eigent_start()        │      ││
│  │  │ to Frontend     │    │ (same as normal)│    │ with pre-defined subtasks       │      ││
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────────────┘      ││
│  │                                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementation Phases

### Phase 1: Database Model (Server)

**File: `server/app/model/chat/workflow_template.py`**

```python
from typing import Optional
from sqlalchemy import Column, Integer, text, Text
from sqlmodel import Field, JSON
from app.model.abstract.model import AbstractModel, DefaultTimes
from pydantic import BaseModel


class WorkflowTemplate(AbstractModel, DefaultTimes, table=True):
    """Stores reusable workflow templates extracted from completed tasks."""
    
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    source_task_id: str = Field(index=True)  # Original task this was created from
    name: str = Field(sa_column=Column(Text))
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Workflow structure (extracted from chat_step records)
    summary_task: str = Field(default="", sa_column=Column(Text))
    subtasks: str = Field(sa_type=JSON)  # From to_sub_tasks step
    agents: str = Field(sa_type=JSON)    # From create_agent steps  
    agent_assignments: str = Field(sa_type=JSON, default={})  # From assign_task steps
    
    # Optional model configuration
    model_config_data: Optional[str] = Field(default=None, sa_type=JSON)
    
    # Sharing and usage tracking
    is_public: bool = Field(default=False)
    use_count: int = Field(default=0, sa_column=Column(Integer, server_default=text("0")))


class WorkflowTemplateIn(BaseModel):
    """Input model for creating a template from a completed task."""
    source_task_id: str
    name: str
    description: Optional[str] = None
    is_public: bool = False


class WorkflowTemplateOut(BaseModel):
    """Output model for template data."""
    id: int
    user_id: int
    source_task_id: str
    name: str
    description: Optional[str]
    summary_task: str
    subtasks: list
    agents: list
    agent_assignments: dict
    is_public: bool
    use_count: int
    
    class Config:
        from_attributes = True


class WorkflowTemplateListOut(BaseModel):
    """Output model for template list."""
    id: int
    name: str
    description: Optional[str]
    summary_task: str
    source_task_id: str
    use_count: int
    is_public: bool
    
    class Config:
        from_attributes = True
```

---

### Phase 2: Database Migration

**File: `server/alembic/versions/XXXX_add_workflow_template.py`**

```python
"""Add workflow_template table

Revision ID: add_workflow_template
Revises: b3e436e9188b
Create Date: 2025-XX-XX
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'add_workflow_template'
down_revision = 'b3e436e9188b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'workflow_template',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('source_task_id', sa.String(64), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('summary_task', sa.Text(), nullable=True, server_default=''),
        sa.Column('subtasks', sa.JSON(), nullable=False),
        sa.Column('agents', sa.JSON(), nullable=False),
        sa.Column('agent_assignments', sa.JSON(), nullable=True, server_default='{}'),
        sa.Column('model_config_data', sa.JSON(), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('use_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workflow_template_user_id', 'workflow_template', ['user_id'])
    op.create_index('ix_workflow_template_source_task_id', 'workflow_template', ['source_task_id'])


def downgrade() -> None:
    op.drop_index('ix_workflow_template_source_task_id')
    op.drop_index('ix_workflow_template_user_id')
    op.drop_table('workflow_template')
```

---

### Phase 3: Server API Controller

**File: `server/app/controller/chat/template_controller.py`**

```python
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select, desc, asc
from app.component.database import session
from app.component.auth import Auth, auth_must
from app.model.chat.workflow_template import (
    WorkflowTemplate, 
    WorkflowTemplateIn, 
    WorkflowTemplateOut,
    WorkflowTemplateListOut
)
from app.model.chat.chat_step import ChatStep
from app.model.chat.chat_history import ChatHistory

router = APIRouter(prefix="/chat", tags=["Workflow Templates"])


@router.post("/template/from-task/{task_id}", name="Create template from completed task")
async def create_template_from_task(
    task_id: str,
    data: WorkflowTemplateIn,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> WorkflowTemplateOut:
    """
    Extract workflow structure from a completed task and save as reusable template.
    """
    # 1. Verify the task belongs to this user and is completed
    history = session.exec(
        select(ChatHistory)
        .where(ChatHistory.task_id == task_id)
        .where(ChatHistory.user_id == auth.user.id)
    ).first()
    
    if not history:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if history.status != 2:  # 2 = done/completed
        raise HTTPException(status_code=400, detail="Task must be completed to create template")
    
    # 2. Extract to_sub_tasks step
    subtasks_step = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id)
        .where(ChatStep.step == "to_sub_tasks")
        .order_by(desc(ChatStep.id))  # Get latest if multiple
    ).first()
    
    if not subtasks_step:
        raise HTTPException(status_code=400, detail="No subtasks found for this task")
    
    # Parse the subtasks data
    subtasks_data = subtasks_step.data if isinstance(subtasks_step.data, dict) else json.loads(subtasks_step.data)
    
    # 3. Extract all create_agent steps
    agent_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id)
        .where(ChatStep.step == "create_agent")
        .order_by(asc(ChatStep.id))
    ).all()
    
    agents = []
    for step in agent_steps:
        agent_data = step.data if isinstance(step.data, dict) else json.loads(step.data)
        agents.append(agent_data)
    
    # 4. Extract all assign_task steps to build assignment map
    assignment_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id)
        .where(ChatStep.step == "assign_task")
        .order_by(asc(ChatStep.id))
    ).all()
    
    agent_assignments = {}
    for step in assignment_steps:
        assign_data = step.data if isinstance(step.data, dict) else json.loads(step.data)
        if 'task_id' in assign_data and 'assignee_id' in assign_data:
            agent_assignments[assign_data['task_id']] = {
                'assignee_id': assign_data['assignee_id'],
                'agent_name': assign_data.get('agent_name', ''),
            }
    
    # 5. Create the template
    template = WorkflowTemplate(
        user_id=auth.user.id,
        source_task_id=task_id,
        name=data.name,
        description=data.description,
        summary_task=subtasks_data.get('summary_task', ''),
        subtasks=json.dumps(subtasks_data.get('sub_tasks', [])),
        agents=json.dumps(agents),
        agent_assignments=json.dumps(agent_assignments),
        is_public=data.is_public,
    )
    
    session.add(template)
    session.commit()
    session.refresh(template)
    
    return WorkflowTemplateOut(
        id=template.id,
        user_id=template.user_id,
        source_task_id=template.source_task_id,
        name=template.name,
        description=template.description,
        summary_task=template.summary_task,
        subtasks=json.loads(template.subtasks) if isinstance(template.subtasks, str) else template.subtasks,
        agents=json.loads(template.agents) if isinstance(template.agents, str) else template.agents,
        agent_assignments=json.loads(template.agent_assignments) if isinstance(template.agent_assignments, str) else template.agent_assignments,
        is_public=template.is_public,
        use_count=template.use_count,
    )


@router.get("/templates", name="List user's templates")
async def list_templates(
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> Page[WorkflowTemplateListOut]:
    """List all templates belonging to the current user."""
    stmt = (
        select(WorkflowTemplate)
        .where(WorkflowTemplate.user_id == auth.user.id)
        .where(WorkflowTemplate.deleted_at.is_(None))
        .order_by(desc(WorkflowTemplate.created_at))
    )
    return paginate(session, stmt)


@router.get("/template/{template_id}", name="Get template details")
async def get_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> WorkflowTemplateOut:
    """Get full template details for execution."""
    template = session.get(WorkflowTemplate, template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Check ownership or public access
    if template.user_id != auth.user.id and not template.is_public:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return WorkflowTemplateOut(
        id=template.id,
        user_id=template.user_id,
        source_task_id=template.source_task_id,
        name=template.name,
        description=template.description,
        summary_task=template.summary_task,
        subtasks=json.loads(template.subtasks) if isinstance(template.subtasks, str) else template.subtasks,
        agents=json.loads(template.agents) if isinstance(template.agents, str) else template.agents,
        agent_assignments=json.loads(template.agent_assignments) if isinstance(template.agent_assignments, str) else template.agent_assignments,
        is_public=template.is_public,
        use_count=template.use_count,
    )


@router.post("/template/{template_id}/use", name="Increment template use count")
async def use_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    """Increment the use count when a template is executed."""
    template = session.get(WorkflowTemplate, template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template.use_count += 1
    session.add(template)
    session.commit()
    
    return {"use_count": template.use_count}


@router.delete("/template/{template_id}", name="Delete template")
async def delete_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    """Soft delete a template."""
    template = session.get(WorkflowTemplate, template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    if template.user_id != auth.user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    template.delete(session)
    return Response(status_code=204)
```

---

### Phase 4: Backend Chat Service Modifications

**File: `backend/app/model/chat.py` - Add new fields**

```python
# ... existing code ...

class Chat(BaseModel):
    task_id: str
    question: str
    email: str
    attaches: list[str] = []
    model_platform: str
    model_type: str
    api_key: str
    api_url: str | None = None
    language: str = "en"
    browser_port: int = 9222
    max_retries: int = 3
    allow_local_system: bool = False
    installed_mcp: McpServers = {"mcpServers": {}}
    bun_mirror: str = ""
    uvx_mirror: str = ""
    env_path: str | None = None
    summary_prompt: str = "..."
    new_agents: list["NewAgent"] = []
    extra_params: dict | None = None
    agent_providers: dict[str, AgentProviderConfig] = Field(default_factory=dict)
    
    # NEW: Template execution fields
    template_mode: bool = False  # Flag to indicate template execution
    template_subtasks: list[dict] | None = None  # Pre-defined subtasks
    template_summary: str | None = None  # Pre-defined summary
    template_id: int | None = None  # Reference to the template being used
```

**File: `backend/app/service/chat_service.py` - Add template execution logic**

```python
# ... existing imports ...

@sync_step
async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    """
    Main task execution entry point.
    Now supports both normal execution and template-based execution.
    """
    start_event_loop = True
    question_agent = question_confirm_agent(options)
    camel_task = None
    workforce = None
    
    while True:
        if await request.is_disconnected():
            if workforce is not None:
                workforce.stop()
            break
        
        try:
            item = await task_lock.get_queue()
        except Exception as e:
            logger.error(f"Error getting item from queue: {e}")
            break

        try:
            if item.action == Action.improve or start_event_loop:
                if start_event_loop is True:
                    question = options.question
                    start_event_loop = False
                else:
                    assert isinstance(item, ActionImproveData)
                    question = item.data
                
                # Short question confirmation (skip for template mode)
                if not options.template_mode and len(question) < 12 and len(options.attaches) == 0:
                    confirm = await question_confirm(question_agent, question)
                else:
                    confirm = True

                if confirm is not True:
                    yield confirm
                else:
                    yield sse_json("confirmed", "")
                    (workforce, mcp) = await construct_workforce(options)
                    
                    # Add custom agents
                    for new_agent in options.new_agents:
                        agent_desc = format_agent_description(new_agent)
                        agent_model_instance = await new_agent_model(new_agent, options)
                        workforce.add_single_agent_worker(agent_desc, agent_model_instance)
                    
                    task_lock.status = Status.confirmed
                    question = question + options.summary_prompt
                    camel_task = Task(content=question, id=options.task_id)
                    
                    if len(options.attaches) > 0:
                        camel_task.additional_info = {
                            Path(file_path).name: file_path 
                            for file_path in options.attaches
                        }

                    # ===== TEMPLATE MODE EXECUTION =====
                    if options.template_mode and options.template_subtasks:
                        logger.info(f"[TEMPLATE] Executing from template for task {options.task_id}")
                        
                        # Skip AI decomposition - use pre-defined subtasks
                        sub_tasks = create_subtasks_from_template(
                            camel_task, 
                            options.template_subtasks
                        )
                        summary_task_content = options.template_summary or "Template Execution"
                        
                        # Initialize workforce state without AI decomposition
                        workforce.reset()
                        workforce._task = camel_task
                        workforce.set_channel(TaskChannel())
                        workforce._state = WorkforceState.RUNNING
                        camel_task.state = TaskState.OPEN
                        workforce._pending_tasks.append(camel_task)
                        
                    # ===== NORMAL EXECUTION =====
                    else:
                        logger.info(f"[NORMAL] Executing with AI decomposition for task {options.task_id}")
                        
                        summary_task_agent = task_summary_agent(options)
                        sub_tasks = await asyncio.to_thread(
                            workforce.eigent_make_sub_tasks, 
                            camel_task
                        )
                        summary_task_content = await summary_task(summary_task_agent, camel_task)
                    
                    # Send subtasks to frontend (same for both modes)
                    yield to_sub_tasks(camel_task, summary_task_content)
                    
                    # Auto-start in debug mode
                    if env("debug") == "on":
                        task_lock.status = Status.processing
                        task = asyncio.create_task(workforce.eigent_start(sub_tasks))
                        task_lock.add_background_task(task)

            # ... rest of the action handlers remain the same ...
            
        except Exception as e:
            logger.error(f"Error processing action {item.action}: {e}")
            yield sse_json("error", {"message": str(e)})


def create_subtasks_from_template(parent_task: Task, template_subtasks: list[dict]) -> list[Task]:
    """
    Create Task objects from template subtask definitions.
    This bypasses AI decomposition by using pre-saved structure.
    """
    subtasks = []
    
    for subtask_data in template_subtasks:
        # Create new task ID based on parent
        task_id = f"{parent_task.id}.{len(subtasks) + 1}"
        
        subtask = Task(
            content=subtask_data.get('content', ''),
            id=task_id,
        )
        subtask.state = TaskState.OPEN
        
        # Handle nested subtasks if present
        if 'subtasks' in subtask_data and subtask_data['subtasks']:
            for nested in subtask_data['subtasks']:
                nested_task = Task(
                    content=nested.get('content', ''),
                    id=f"{task_id}.{len(subtask.subtasks) + 1}",
                )
                nested_task.state = TaskState.OPEN
                subtask.add_subtask(nested_task)
        
        parent_task.add_subtask(subtask)
        subtasks.append(subtask)
    
    return subtasks
```

---

### Phase 5: Frontend Store Modifications

**File: `src/store/chatStore.ts` - Add template functions**

```typescript
// Add to ChatStore interface
interface ChatStore {
  // ... existing properties ...
  
  // Template management
  executeFromTemplate: (
    templateId: number,
    templateData: {
      subtasks: any[];
      summary_task: string;
      agents: any[];
    }
  ) => Promise<void>;
  
  saveAsTemplate: (taskId: string, name: string, description?: string) => Promise<any>;
}

// Add to the store implementation
const chatStore = create<ChatStore>()((set, get) => ({
  // ... existing code ...

  /**
   * Execute a task using a pre-saved template (skip AI orchestration)
   */
  executeFromTemplate: async (templateId: number, templateData: {
    subtasks: any[];
    summary_task: string;
    agents: any[];
  }) => {
    const {
      create,
      setActiveTaskId,
      startTaskFromTemplate,
      setSummaryTask,
      setTaskInfo,
      setTaskRunning,
      addMessages,
      setHasMessages,
    } = get();

    // 1. Generate new task ID
    const newTaskId = generateUniqueId();
    
    // 2. Create task in store
    create(newTaskId, "template");
    setHasMessages(newTaskId, true);
    
    // 3. Pre-populate UI with template data (immediate feedback)
    setSummaryTask(newTaskId, templateData.summary_task);
    
    // Map subtasks for UI display
    const mappedSubtasks = templateData.subtasks.map((st: any, idx: number) => ({
      id: `${newTaskId}.${idx + 1}`,
      content: st.content,
      status: "",
      subtasks: st.subtasks || [],
    }));
    setTaskInfo(newTaskId, mappedSubtasks);
    setTaskRunning(newTaskId, mappedSubtasks);
    
    // 4. Add initial message
    addMessages(newTaskId, {
      id: generateUniqueId(),
      role: "user",
      content: `Re-executing template: ${templateData.summary_task}`,
    });
    
    // 5. Notify server about template usage
    await proxyFetchPost(`/api/chat/template/${templateId}/use`);
    
    // 6. Start task execution with template data
    await startTaskFromTemplate(newTaskId, templateData);
    
    setActiveTaskId(newTaskId);
  },

  /**
   * Start task execution using template data
   */
  startTaskFromTemplate: async (
    taskId: string,
    templateData: {
      subtasks: any[];
      summary_task: string;
      agents: any[];
    }
  ) => {
    const { token, language, modelType, cloud_model_type, email } = getAuthStore();
    const workerList = getWorkerList();
    const baseURL = await getBaseURL();
    
    // Get model configuration (same as startTask)
    let apiModel = { /* ... same as existing startTask ... */ };
    // ... model fetching logic ...
    
    // Get MCP tools
    let mcpLocal = {};
    if (window.ipcRenderer) {
      mcpLocal = await window.ipcRenderer.invoke("mcp-list");
    }
    
    // Get env path
    let envPath = "";
    try {
      envPath = await window.ipcRenderer.invoke("get-env-path", email);
    } catch (error) {
      console.log("get-env-path error", error);
    }
    
    const browser_port = await window.ipcRenderer.invoke("get-browser-port");

    // Create history record
    const authStore = getAuthStore();
    await proxyFetchPost(`/api/chat/history`, {
      task_id: taskId,
      user_id: authStore.user_id,
      question: `Template: ${templateData.summary_task}`,
      language: language,
      model_platform: apiModel.model_platform,
      model_type: apiModel.model_type,
      api_url: modelType === "cloud" ? "cloud" : apiModel.api_url,
      max_retries: 3,
      status: 1,
      tokens: 0,
    });

    // Start SSE connection with template mode
    fetchEventSource(`${baseURL}/chat`, {
      method: "POST",
      openWhenHidden: true,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        task_id: taskId,
        question: `Execute template: ${templateData.summary_task}`,
        model_platform: apiModel.model_platform,
        email,
        model_type: apiModel.model_type,
        api_key: apiModel.api_key,
        api_url: apiModel.api_url,
        extra_params: apiModel.extra_params,
        installed_mcp: mcpLocal,
        language: language,
        allow_local_system: true,
        attaches: [],
        browser_port: browser_port,
        env_path: envPath,
        // TEMPLATE MODE FLAGS
        template_mode: true,
        template_subtasks: templateData.subtasks,
        template_summary: templateData.summary_task,
      }),
      
      async onmessage(event: any) {
        // Same SSE handling as existing startTask
        // ... existing onmessage logic ...
      },
      
      onerror(err) {
        console.error("Template execution SSE error:", err);
      },
      
      async onclose() {
        console.log("Template execution complete");
      },
    });
  },

  /**
   * Save current task as a reusable template
   */
  saveAsTemplate: async (taskId: string, name: string, description?: string) => {
    try {
      const response = await proxyFetchPost(`/api/chat/template/from-task/${taskId}`, {
        source_task_id: taskId,
        name,
        description,
        is_public: false,
      });
      
      return response;
    } catch (error) {
      console.error("Failed to save template:", error);
      throw error;
    }
  },
}));
```

---

### Phase 6: Frontend UI Components

**New Component: `src/components/TemplateDialog.tsx`**

```tsx
import React, { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { proxyFetchGet, proxyFetchPost } from "@/api/http";
import { useChatStore } from "@/store/chatStore";
import { toast } from "sonner";
import { PlayCircle, Save, Trash2, Clock, Zap } from "lucide-react";

interface Template {
  id: number;
  name: string;
  description?: string;
  summary_task: string;
  use_count: number;
  source_task_id: string;
}

interface TemplateDialogProps {
  open: boolean;
  onClose: () => void;
  mode: "save" | "execute";
  taskId?: string;
}

export function TemplateDialog({ open, onClose, mode, taskId }: TemplateDialogProps) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const chatStore = useChatStore();

  useEffect(() => {
    if (open && mode === "execute") {
      fetchTemplates();
    }
  }, [open, mode]);

  const fetchTemplates = async () => {
    setLoading(true);
    try {
      const res = await proxyFetchGet("/api/chat/templates");
      setTemplates(res.items || []);
    } catch (error) {
      console.error("Failed to fetch templates:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveTemplate = async () => {
    if (!taskId || !name.trim()) {
      toast.error("Please enter a template name");
      return;
    }

    setLoading(true);
    try {
      await chatStore.saveAsTemplate(taskId, name.trim(), description.trim());
      toast.success("Template saved successfully!");
      onClose();
    } catch (error) {
      toast.error("Failed to save template");
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteTemplate = async (template: Template) => {
    setLoading(true);
    try {
      // Fetch full template data
      const fullTemplate = await proxyFetchGet(`/api/chat/template/${template.id}`);
      
      // Execute from template
      await chatStore.executeFromTemplate(template.id, {
        subtasks: fullTemplate.subtasks,
        summary_task: fullTemplate.summary_task,
        agents: fullTemplate.agents,
      });
      
      toast.success("Executing template...");
      onClose();
    } catch (error) {
      toast.error("Failed to execute template");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {mode === "save" ? (
              <div className="flex items-center gap-2">
                <Save className="w-5 h-5" />
                Save as Template
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Execute from Template
              </div>
            )}
          </DialogTitle>
        </DialogHeader>

        {mode === "save" ? (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Template Name</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Web Research Workflow"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Description (optional)</label>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe what this template does..."
                rows={3}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button onClick={handleSaveTemplate} disabled={loading || !name.trim()}>
                {loading ? "Saving..." : "Save Template"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {loading ? (
              <div className="text-center py-8">Loading templates...</div>
            ) : templates.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No templates saved yet. Complete a task and save it as a template!
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {templates.map((template) => (
                  <div
                    key={template.id}
                    className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                    onClick={() => handleExecuteTemplate(template)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium">{template.name}</h3>
                        <p className="text-sm text-gray-500 truncate">
                          {template.summary_task}
                        </p>
                        {template.description && (
                          <p className="text-xs text-gray-400 mt-1">
                            {template.description}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-xs text-gray-400 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Used {template.use_count}x
                        </div>
                        <Button size="sm" variant="ghost">
                          <PlayCircle className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
```

---

## 🎯 Implementation Summary

### Files to Create (New)

| File | Purpose |
|------|---------|
| `server/app/model/chat/workflow_template.py` | Database model |
| `server/alembic/versions/XXX_add_workflow_template.py` | DB migration |
| `server/app/controller/chat/template_controller.py` | REST API |
| `src/components/TemplateDialog.tsx` | UI component |

### Files to Modify

| File | Changes |
|------|---------|
| `backend/app/model/chat.py` | Add template_mode fields to Chat |
| `backend/app/service/chat_service.py` | Add template execution path |
| `src/store/chatStore.ts` | Add template functions |
| `src/components/ChatBox/index.tsx` | Add Save/Execute template buttons |
| `server/alembic/env.py` | Import new model |

---

## 📈 Benefits Summary

```
┌────────────────────────────────────────────────────────────────────────┐
│                         EXPECTED BENEFITS                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  💰 COST SAVINGS                                                       │
│  ├── Skip AI orchestration tokens (~30-50% reduction)                  │
│  ├── No planning/decomposition LLM calls                               │
│  └── Only execution tokens consumed                                    │
│                                                                        │
│  ⏱️ TIME SAVINGS                                                        │
│  ├── Immediate start (no 5-30 second planning delay)                   │
│  ├── Instant UI feedback with pre-loaded subtasks                      │
│  └── Predictable execution time                                        │
│                                                                        │
│  🔄 CONSISTENCY                                                         │
│  ├── Same subtask structure every time                                 │
│  ├── Same agent assignments                                            │
│  └── Reproducible workflows                                            │
│                                                                        │
│  📚 KNOWLEDGE CAPTURE                                                   │
│  ├── Save successful workflow patterns                                 │
│  ├── Share templates across team (is_public flag)                      │
│  └── Build library of proven workflows                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

Would you like me to elaborate on any specific phase, or shall I help you get started with the implementation? Since you're in **ask mode**, you'll need to switch to **agent mode** to have me create these files for you.

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            CURRENT EXECUTION FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │  User    │───▶│   Backend   │───▶│  AI Orchestrator │───▶│  Task Decomposer   │     │
│  │  Query   │    │   /chat     │    │  (Coordinator)   │    │  (_decompose_task) │     │
│  └──────────┘    └─────────────┘    └──────────────────┘    └────────────────────┘     │
│                                                                         │               │
│                                                                         ▼               │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │   End    │◀───│   Results   │◀───│  Agent Execution │◀───│   Subtasks List    │     │
│  │  Result  │    │   Stream    │    │  (Workforce)     │    │   (from AI)        │     │
│  └──────────┘    └─────────────┘    └──────────────────┘    └────────────────────┘     │
│                                                                                          │
│                        💰 Cost: AI tokens for decomposition + execution                  │
│                        ⏱️ Time: Includes AI thinking time for planning                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         NEW TEMPLATE EXECUTION FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐                               │
│  │  Select  │───▶│   Backend   │───▶│ Load Template    │                               │
│  │ Template │    │ /re-execute │    │ from Database    │                               │
│  └──────────┘    └─────────────┘    └──────────────────┘                               │
│                                              │                                          │
│                                              ▼ SKIP AI DECOMPOSITION                    │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │   End    │◀───│   Results   │◀───│  Agent Execution │◀───│  Pre-saved         │     │
│  │  Result  │    │   Stream    │    │  (Workforce)     │    │  Subtasks          │     │
│  └──────────┘    └─────────────┘    └──────────────────┘    └────────────────────┘     │
│                                                                                          │
│                        💰 Cost: Only AI tokens for execution (save ~30-50%)             │
│                        ⏱️ Time: Immediate start (no planning delay)                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              workflow_template                                         │
├──────────────────┬──────────────┬─────────────────────────────────────────────────────┤
│ Column           │ Type         │ Description                                         │
├──────────────────┼──────────────┼─────────────────────────────────────────────────────┤
│ id               │ INTEGER PK   │ Auto-increment primary key                          │
│ user_id          │ INTEGER FK   │ Owner of the template                               │
│ source_task_id   │ VARCHAR(64)  │ Original task_id this template was created from    │
│ name             │ VARCHAR(128) │ User-friendly template name                         │
│ description      │ TEXT         │ Optional description of what this template does    │
│ summary_task     │ VARCHAR(512) │ The summary generated by AI                         │
│ subtasks         │ JSON         │ Array of subtask objects from to_sub_tasks          │
│ agents           │ JSON         │ Array of agent configs from create_agent events     │
│ agent_assignments│ JSON         │ Map of subtask_id -> agent_id/agent_name            │
│ model_config     │ JSON         │ Optional: store model preferences                   │
│ is_public        │ BOOLEAN      │ Whether template can be shared                      │
│ use_count        │ INTEGER      │ How many times this template has been used          │
│ created_at       │ TIMESTAMP    │ Creation timestamp                                  │
│ updated_at       │ TIMESTAMP    │ Last update timestamp                               │
│ deleted_at       │ TIMESTAMP    │ Soft delete timestamp                               │
└──────────────────┴──────────────┴─────────────────────────────────────────────────────┘
```

```plaintext
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          DATA MAPPING: chat_step → workflow_template                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  chat_step (step="to_sub_tasks")                  workflow_template                    │
│  ┌──────────────────────────────┐                 ┌──────────────────────────────┐    │
│  │ {                            │                 │                              │    │
│  │   "summary_task": "...",     │ ──────────────▶ │ summary_task: "..."          │    │
│  │   "sub_tasks": [             │                 │                              │    │
│  │     {                        │                 │ subtasks: [                  │    │
│  │       "id": "task.1",        │ ──────────────▶ │   {id, content, state}       │    │
│  │       "content": "...",      │                 │ ]                            │    │
│  │       "state": "..."         │                 │                              │    │
│  │     }                        │                 │                              │    │
│  │   ]                          │                 │                              │    │
│  │ }                            │                 │                              │    │
│  └──────────────────────────────┘                 │                              │    │
│                                                   │                              │    │
│  chat_step (step="create_agent")                  │                              │    │
│  ┌──────────────────────────────┐                 │                              │    │
│  │ {                            │                 │ agents: [                    │    │
│  │   "agent_name": "...",       │ ──────────────▶ │   {agent_name, agent_id,     │    │
│  │   "agent_id": "...",         │                 │    tools}                    │    │
│  │   "tools": [...]             │                 │ ]                            │    │
│  │ }                            │                 │                              │    │
│  └──────────────────────────────┘                 │                              │    │
│                                                   │                              │    │
│  chat_step (step="assign_task")                   │                              │    │
│  ┌──────────────────────────────┐                 │                              │    │
│  │ {                            │                 │ agent_assignments: {         │    │
│  │   "assignee_id": "...",      │ ──────────────▶ │   "task.1": "agent_id",      │    │
│  │   "task_id": "..."           │                 │   "task.2": "agent_id"       │    │
│  │ }                            │                 │ }                            │    │
│  └──────────────────────────────┘                 └──────────────────────────────┘    │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    COMPLETE DATA FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  ┌─────────────────────────────────── PHASE 1: TEMPLATE CREATION ──────────────────────────┐│
│  │                                                                                          ││
│  │     User completes task                                                                  ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐                                                                     ││
│  │  │ "Save Template" │                                                                     ││
│  │  │    Button       │                                                                     ││
│  │  └────────┬────────┘                                                                     ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ POST            │───▶│ Server API      │───▶│ Extract from chat_step:         │      ││
│  │  │ /api/template/  │    │ template_       │    │ • to_sub_tasks → subtasks       │      ││
│  │  │ from-task/{id}  │    │ controller.py   │    │ • create_agent → agents         │      ││
│  │  └─────────────────┘    └─────────────────┘    │ • assign_task → assignments     │      ││
│  │                                                 └──────────────┬──────────────────┘      ││
│  │                                                                │                         ││
│  │                                                                ▼                         ││
│  │                                                 ┌─────────────────────────────────┐      ││
│  │                                                 │ INSERT INTO workflow_template   │      ││
│  │                                                 │ (subtasks, agents, assignments) │      ││
│  │                                                 └─────────────────────────────────┘      ││
│  │                                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                              │
│  ┌─────────────────────────────────── PHASE 2: TEMPLATE EXECUTION ─────────────────────────┐│
│  │                                                                                          ││
│  │     User selects template                                                                ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ "Re-Execute"    │───▶│ GET             │───▶│ SELECT * FROM workflow_template │      ││
│  │  │    Button       │    │ /api/template/  │    │ WHERE id = :template_id         │      ││
│  │  └─────────────────┘    │ {id}            │    └──────────────┬──────────────────┘      ││
│  │                         └─────────────────┘                   │                         ││
│  │                                                               ▼                         ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ Frontend Store  │◀───│ Template Data   │◀───│ Return template JSON            │      ││
│  │  │ chatStore       │    │ Response        │    │ {subtasks, agents, assignments} │      ││
│  │  └────────┬────────┘    └─────────────────┘    └─────────────────────────────────┘      ││
│  │           │                                                                              ││
│  │           ▼                                                                              ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ POST /chat      │───▶│ step_solve_     │───▶│ SKIP: _decompose_task()         │      ││
│  │  │ (with template  │    │ from_template() │    │ USE:  template.subtasks         │      ││
│  │  │  mode flag)     │    │                 │    │ SKIP: AI planning               │      ││
│  │  └─────────────────┘    └─────────────────┘    └──────────────┬──────────────────┘      ││
│  │                                                               │                         ││
│  │                                                               ▼                         ││
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐      ││
│  │  │ SSE Stream      │◀───│ Results via SSE │◀───│ workforce.eigent_start()        │      ││
│  │  │ to Frontend     │    │ (same as normal)│    │ with pre-defined subtasks       │      ││
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────────────┘      ││
│  │                                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

```python
from typing import Optional
from sqlalchemy import Column, Integer, text, Text
from sqlmodel import Field, JSON
from app.model.abstract.model import AbstractModel, DefaultTimes
from pydantic import BaseModel


class WorkflowTemplate(AbstractModel, DefaultTimes, table=True):
    """Stores reusable workflow templates extracted from completed tasks."""
    
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    source_task_id: str = Field(index=True)  # Original task this was created from
    name: str = Field(sa_column=Column(Text))
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    
    # Workflow structure (extracted from chat_step records)
    summary_task: str = Field(default="", sa_column=Column(Text))
    subtasks: str = Field(sa_type=JSON)  # From to_sub_tasks step
    agents: str = Field(sa_type=JSON)    # From create_agent steps  
    agent_assignments: str = Field(sa_type=JSON, default={})  # From assign_task steps
    
    # Optional model configuration
    model_config_data: Optional[str] = Field(default=None, sa_type=JSON)
    
    # Sharing and usage tracking
    is_public: bool = Field(default=False)
    use_count: int = Field(default=0, sa_column=Column(Integer, server_default=text("0")))


class WorkflowTemplateIn(BaseModel):
    """Input model for creating a template from a completed task."""
    source_task_id: str
    name: str
    description: Optional[str] = None
    is_public: bool = False


class WorkflowTemplateOut(BaseModel):
    """Output model for template data."""
    id: int
    user_id: int
    source_task_id: str
    name: str
    description: Optional[str]
    summary_task: str
    subtasks: list
    agents: list
    agent_assignments: dict
    is_public: bool
    use_count: int
    
    class Config:
        from_attributes = True


class WorkflowTemplateListOut(BaseModel):
    """Output model for template list."""
    id: int
    name: str
    description: Optional[str]
    summary_task: str
    source_task_id: str
    use_count: int
    is_public: bool
    
    class Config:
        from_attributes = True
```

```python
"""Add workflow_template table

Revision ID: add_workflow_template
Revises: b3e436e9188b
Create Date: 2025-XX-XX
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'add_workflow_template'
down_revision = 'b3e436e9188b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'workflow_template',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('source_task_id', sa.String(64), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('summary_task', sa.Text(), nullable=True, server_default=''),
        sa.Column('subtasks', sa.JSON(), nullable=False),
        sa.Column('agents', sa.JSON(), nullable=False),
        sa.Column('agent_assignments', sa.JSON(), nullable=True, server_default='{}'),
        sa.Column('model_config_data', sa.JSON(), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('use_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workflow_template_user_id', 'workflow_template', ['user_id'])
    op.create_index('ix_workflow_template_source_task_id', 'workflow_template', ['source_task_id'])


def downgrade() -> None:
    op.drop_index('ix_workflow_template_source_task_id')
    op.drop_index('ix_workflow_template_user_id')
    op.drop_table('workflow_template')
```

```python
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select, desc, asc
from app.component.database import session
from app.component.auth import Auth, auth_must
from app.model.chat.workflow_template import (
    WorkflowTemplate, 
    WorkflowTemplateIn, 
    WorkflowTemplateOut,
    WorkflowTemplateListOut
)
from app.model.chat.chat_step import ChatStep
from app.model.chat.chat_history import ChatHistory

router = APIRouter(prefix="/chat", tags=["Workflow Templates"])


@router.post("/template/from-task/{task_id}", name="Create template from completed task")
async def create_template_from_task(
    task_id: str,
    data: WorkflowTemplateIn,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> WorkflowTemplateOut:
    """
    Extract workflow structure from a completed task and save as reusable template.
    """
    # 1. Verify the task belongs to this user and is completed
    history = session.exec(
        select(ChatHistory)
        .where(ChatHistory.task_id == task_id)
        .where(ChatHistory.user_id == auth.user.id)
    ).first()
    
    if not history:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if history.status != 2:  # 2 = done/completed
        raise HTTPException(status_code=400, detail="Task must be completed to create template")
    
    # 2. Extract to_sub_tasks step
    subtasks_step = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id)
        .where(ChatStep.step == "to_sub_tasks")
        .order_by(desc(ChatStep.id))  # Get latest if multiple
    ).first()
    
    if not subtasks_step:
        raise HTTPException(status_code=400, detail="No subtasks found for this task")
    
    # Parse the subtasks data
    subtasks_data = subtasks_step.data if isinstance(subtasks_step.data, dict) else json.loads(subtasks_step.data)
    
    # 3. Extract all create_agent steps
    agent_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id)
        .where(ChatStep.step == "create_agent")
        .order_by(asc(ChatStep.id))
    ).all()
    
    agents = []
    for step in agent_steps:
        agent_data = step.data if isinstance(step.data, dict) else json.loads(step.data)
        agents.append(agent_data)
    
    # 4. Extract all assign_task steps to build assignment map
    assignment_steps = session.exec(
        select(ChatStep)
        .where(ChatStep.task_id == task_id)
        .where(ChatStep.step == "assign_task")
        .order_by(asc(ChatStep.id))
    ).all()
    
    agent_assignments = {}
    for step in assignment_steps:
        assign_data = step.data if isinstance(step.data, dict) else json.loads(step.data)
        if 'task_id' in assign_data and 'assignee_id' in assign_data:
            agent_assignments[assign_data['task_id']] = {
                'assignee_id': assign_data['assignee_id'],
                'agent_name': assign_data.get('agent_name', ''),
            }
    
    # 5. Create the template
    template = WorkflowTemplate(
        user_id=auth.user.id,
        source_task_id=task_id,
        name=data.name,
        description=data.description,
        summary_task=subtasks_data.get('summary_task', ''),
        subtasks=json.dumps(subtasks_data.get('sub_tasks', [])),
        agents=json.dumps(agents),
        agent_assignments=json.dumps(agent_assignments),
        is_public=data.is_public,
    )
    
    session.add(template)
    session.commit()
    session.refresh(template)
    
    return WorkflowTemplateOut(
        id=template.id,
        user_id=template.user_id,
        source_task_id=template.source_task_id,
        name=template.name,
        description=template.description,
        summary_task=template.summary_task,
        subtasks=json.loads(template.subtasks) if isinstance(template.subtasks, str) else template.subtasks,
        agents=json.loads(template.agents) if isinstance(template.agents, str) else template.agents,
        agent_assignments=json.loads(template.agent_assignments) if isinstance(template.agent_assignments, str) else template.agent_assignments,
        is_public=template.is_public,
        use_count=template.use_count,
    )


@router.get("/templates", name="List user's templates")
async def list_templates(
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> Page[WorkflowTemplateListOut]:
    """List all templates belonging to the current user."""
    stmt = (
        select(WorkflowTemplate)
        .where(WorkflowTemplate.user_id == auth.user.id)
        .where(WorkflowTemplate.deleted_at.is_(None))
        .order_by(desc(WorkflowTemplate.created_at))
    )
    return paginate(session, stmt)


@router.get("/template/{template_id}", name="Get template details")
async def get_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> WorkflowTemplateOut:
    """Get full template details for execution."""
    template = session.get(WorkflowTemplate, template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Check ownership or public access
    if template.user_id != auth.user.id and not template.is_public:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return WorkflowTemplateOut(
        id=template.id,
        user_id=template.user_id,
        source_task_id=template.source_task_id,
        name=template.name,
        description=template.description,
        summary_task=template.summary_task,
        subtasks=json.loads(template.subtasks) if isinstance(template.subtasks, str) else template.subtasks,
        agents=json.loads(template.agents) if isinstance(template.agents, str) else template.agents,
        agent_assignments=json.loads(template.agent_assignments) if isinstance(template.agent_assignments, str) else template.agent_assignments,
        is_public=template.is_public,
        use_count=template.use_count,
    )


@router.post("/template/{template_id}/use", name="Increment template use count")
async def use_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    """Increment the use count when a template is executed."""
    template = session.get(WorkflowTemplate, template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template.use_count += 1
    session.add(template)
    session.commit()
    
    return {"use_count": template.use_count}


@router.delete("/template/{template_id}", name="Delete template")
async def delete_template(
    template_id: int,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    """Soft delete a template."""
    template = session.get(WorkflowTemplate, template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    if template.user_id != auth.user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    template.delete(session)
    return Response(status_code=204)
```

```python
# ... existing code ...

class Chat(BaseModel):
    task_id: str
    question: str
    email: str
    attaches: list[str] = []
    model_platform: str
    model_type: str
    api_key: str
    api_url: str | None = None
    language: str = "en"
    browser_port: int = 9222
    max_retries: int = 3
    allow_local_system: bool = False
    installed_mcp: McpServers = {"mcpServers": {}}
    bun_mirror: str = ""
    uvx_mirror: str = ""
    env_path: str | None = None
    summary_prompt: str = "..."
    new_agents: list["NewAgent"] = []
    extra_params: dict | None = None
    agent_providers: dict[str, AgentProviderConfig] = Field(default_factory=dict)
    
    # NEW: Template execution fields
    template_mode: bool = False  # Flag to indicate template execution
    template_subtasks: list[dict] | None = None  # Pre-defined subtasks
    template_summary: str | None = None  # Pre-defined summary
    template_id: int | None = None  # Reference to the template being used
```

```python
# ... existing imports ...

@sync_step
async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    """
    Main task execution entry point.
    Now supports both normal execution and template-based execution.
    """
    start_event_loop = True
    question_agent = question_confirm_agent(options)
    camel_task = None
    workforce = None
    
    while True:
        if await request.is_disconnected():
            if workforce is not None:
                workforce.stop()
            break
        
        try:
            item = await task_lock.get_queue()
        except Exception as e:
            logger.error(f"Error getting item from queue: {e}")
            break

        try:
            if item.action == Action.improve or start_event_loop:
                if start_event_loop is True:
                    question = options.question
                    start_event_loop = False
                else:
                    assert isinstance(item, ActionImproveData)
                    question = item.data
                
                # Short question confirmation (skip for template mode)
                if not options.template_mode and len(question) < 12 and len(options.attaches) == 0:
                    confirm = await question_confirm(question_agent, question)
                else:
                    confirm = True

                if confirm is not True:
                    yield confirm
                else:
                    yield sse_json("confirmed", "")
                    (workforce, mcp) = await construct_workforce(options)
                    
                    # Add custom agents
                    for new_agent in options.new_agents:
                        agent_desc = format_agent_description(new_agent)
                        agent_model_instance = await new_agent_model(new_agent, options)
                        workforce.add_single_agent_worker(agent_desc, agent_model_instance)
                    
                    task_lock.status = Status.confirmed
                    question = question + options.summary_prompt
                    camel_task = Task(content=question, id=options.task_id)
                    
                    if len(options.attaches) > 0:
                        camel_task.additional_info = {
                            Path(file_path).name: file_path 
                            for file_path in options.attaches
                        }

                    # ===== TEMPLATE MODE EXECUTION =====
                    if options.template_mode and options.template_subtasks:
                        logger.info(f"[TEMPLATE] Executing from template for task {options.task_id}")
                        
                        # Skip AI decomposition - use pre-defined subtasks
                        sub_tasks = create_subtasks_from_template(
                            camel_task, 
                            options.template_subtasks
                        )
                        summary_task_content = options.template_summary or "Template Execution"
                        
                        # Initialize workforce state without AI decomposition
                        workforce.reset()
                        workforce._task = camel_task
                        workforce.set_channel(TaskChannel())
                        workforce._state = WorkforceState.RUNNING
                        camel_task.state = TaskState.OPEN
                        workforce._pending_tasks.append(camel_task)
                        
                    # ===== NORMAL EXECUTION =====
                    else:
                        logger.info(f"[NORMAL] Executing with AI decomposition for task {options.task_id}")
                        
                        summary_task_agent = task_summary_agent(options)
                        sub_tasks = await asyncio.to_thread(
                            workforce.eigent_make_sub_tasks, 
                            camel_task
                        )
                        summary_task_content = await summary_task(summary_task_agent, camel_task)
                    
                    # Send subtasks to frontend (same for both modes)
                    yield to_sub_tasks(camel_task, summary_task_content)
                    
                    # Auto-start in debug mode
                    if env("debug") == "on":
                        task_lock.status = Status.processing
                        task = asyncio.create_task(workforce.eigent_start(sub_tasks))
                        task_lock.add_background_task(task)

            # ... rest of the action handlers remain the same ...
            
        except Exception as e:
            logger.error(f"Error processing action {item.action}: {e}")
            yield sse_json("error", {"message": str(e)})


def create_subtasks_from_template(parent_task: Task, template_subtasks: list[dict]) -> list[Task]:
    """
    Create Task objects from template subtask definitions.
    This bypasses AI decomposition by using pre-saved structure.
    """
    subtasks = []
    
    for subtask_data in template_subtasks:
        # Create new task ID based on parent
        task_id = f"{parent_task.id}.{len(subtasks) + 1}"
        
        subtask = Task(
            content=subtask_data.get('content', ''),
            id=task_id,
        )
        subtask.state = TaskState.OPEN
        
        # Handle nested subtasks if present
        if 'subtasks' in subtask_data and subtask_data['subtasks']:
            for nested in subtask_data['subtasks']:
                nested_task = Task(
                    content=nested.get('content', ''),
                    id=f"{task_id}.{len(subtask.subtasks) + 1}",
                )
                nested_task.state = TaskState.OPEN
                subtask.add_subtask(nested_task)
        
        parent_task.add_subtask(subtask)
        subtasks.append(subtask)
    
    return subtasks
```

```typescript
// Add to ChatStore interface
interface ChatStore {
  // ... existing properties ...
  
  // Template management
  executeFromTemplate: (
    templateId: number,
    templateData: {
      subtasks: any[];
      summary_task: string;
      agents: any[];
    }
  ) => Promise<void>;
  
  saveAsTemplate: (taskId: string, name: string, description?: string) => Promise<any>;
}

// Add to the store implementation
const chatStore = create<ChatStore>()((set, get) => ({
  // ... existing code ...

  /**
   * Execute a task using a pre-saved template (skip AI orchestration)
   */
  executeFromTemplate: async (templateId: number, templateData: {
    subtasks: any[];
    summary_task: string;
    agents: any[];
  }) => {
    const {
      create,
      setActiveTaskId,
      startTaskFromTemplate,
      setSummaryTask,
      setTaskInfo,
      setTaskRunning,
      addMessages,
      setHasMessages,
    } = get();

    // 1. Generate new task ID
    const newTaskId = generateUniqueId();
    
    // 2. Create task in store
    create(newTaskId, "template");
    setHasMessages(newTaskId, true);
    
    // 3. Pre-populate UI with template data (immediate feedback)
    setSummaryTask(newTaskId, templateData.summary_task);
    
    // Map subtasks for UI display
    const mappedSubtasks = templateData.subtasks.map((st: any, idx: number) => ({
      id: `${newTaskId}.${idx + 1}`,
      content: st.content,
      status: "",
      subtasks: st.subtasks || [],
    }));
    setTaskInfo(newTaskId, mappedSubtasks);
    setTaskRunning(newTaskId, mappedSubtasks);
    
    // 4. Add initial message
    addMessages(newTaskId, {
      id: generateUniqueId(),
      role: "user",
      content: `Re-executing template: ${templateData.summary_task}`,
    });
    
    // 5. Notify server about template usage
    await proxyFetchPost(`/api/chat/template/${templateId}/use`);
    
    // 6. Start task execution with template data
    await startTaskFromTemplate(newTaskId, templateData);
    
    setActiveTaskId(newTaskId);
  },

  /**
   * Start task execution using template data
   */
  startTaskFromTemplate: async (
    taskId: string,
    templateData: {
      subtasks: any[];
      summary_task: string;
      agents: any[];
    }
  ) => {
    const { token, language, modelType, cloud_model_type, email } = getAuthStore();
    const workerList = getWorkerList();
    const baseURL = await getBaseURL();
    
    // Get model configuration (same as startTask)
    let apiModel = { /* ... same as existing startTask ... */ };
    // ... model fetching logic ...
    
    // Get MCP tools
    let mcpLocal = {};
    if (window.ipcRenderer) {
      mcpLocal = await window.ipcRenderer.invoke("mcp-list");
    }
    
    // Get env path
    let envPath = "";
    try {
      envPath = await window.ipcRenderer.invoke("get-env-path", email);
    } catch (error) {
      console.log("get-env-path error", error);
    }
    
    const browser_port = await window.ipcRenderer.invoke("get-browser-port");

    // Create history record
    const authStore = getAuthStore();
    await proxyFetchPost(`/api/chat/history`, {
      task_id: taskId,
      user_id: authStore.user_id,
      question: `Template: ${templateData.summary_task}`,
      language: language,
      model_platform: apiModel.model_platform,
      model_type: apiModel.model_type,
      api_url: modelType === "cloud" ? "cloud" : apiModel.api_url,
      max_retries: 3,
      status: 1,
      tokens: 0,
    });

    // Start SSE connection with template mode
    fetchEventSource(`${baseURL}/chat`, {
      method: "POST",
      openWhenHidden: true,
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        task_id: taskId,
        question: `Execute template: ${templateData.summary_task}`,
        model_platform: apiModel.model_platform,
        email,
        model_type: apiModel.model_type,
        api_key: apiModel.api_key,
        api_url: apiModel.api_url,
        extra_params: apiModel.extra_params,
        installed_mcp: mcpLocal,
        language: language,
        allow_local_system: true,
        attaches: [],
        browser_port: browser_port,
        env_path: envPath,
        // TEMPLATE MODE FLAGS
        template_mode: true,
        template_subtasks: templateData.subtasks,
        template_summary: templateData.summary_task,
      }),
      
      async onmessage(event: any) {
        // Same SSE handling as existing startTask
        // ... existing onmessage logic ...
      },
      
      onerror(err) {
        console.error("Template execution SSE error:", err);
      },
      
      async onclose() {
        console.log("Template execution complete");
      },
    });
  },

  /**
   * Save current task as a reusable template
   */
  saveAsTemplate: async (taskId: string, name: string, description?: string) => {
    try {
      const response = await proxyFetchPost(`/api/chat/template/from-task/${taskId}`, {
        source_task_id: taskId,
        name,
        description,
        is_public: false,
      });
      
      return response;
    } catch (error) {
      console.error("Failed to save template:", error);
      throw error;
    }
  },
}));
```

```typescriptreact
import React, { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { proxyFetchGet, proxyFetchPost } from "@/api/http";
import { useChatStore } from "@/store/chatStore";
import { toast } from "sonner";
import { PlayCircle, Save, Trash2, Clock, Zap } from "lucide-react";

interface Template {
  id: number;
  name: string;
  description?: string;
  summary_task: string;
  use_count: number;
  source_task_id: string;
}

interface TemplateDialogProps {
  open: boolean;
  onClose: () => void;
  mode: "save" | "execute";
  taskId?: string;
}

export function TemplateDialog({ open, onClose, mode, taskId }: TemplateDialogProps) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const chatStore = useChatStore();

  useEffect(() => {
    if (open && mode === "execute") {
      fetchTemplates();
    }
  }, [open, mode]);

  const fetchTemplates = async () => {
    setLoading(true);
    try {
      const res = await proxyFetchGet("/api/chat/templates");
      setTemplates(res.items || []);
    } catch (error) {
      console.error("Failed to fetch templates:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveTemplate = async () => {
    if (!taskId || !name.trim()) {
      toast.error("Please enter a template name");
      return;
    }

    setLoading(true);
    try {
      await chatStore.saveAsTemplate(taskId, name.trim(), description.trim());
      toast.success("Template saved successfully!");
      onClose();
    } catch (error) {
      toast.error("Failed to save template");
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteTemplate = async (template: Template) => {
    setLoading(true);
    try {
      // Fetch full template data
      const fullTemplate = await proxyFetchGet(`/api/chat/template/${template.id}`);
      
      // Execute from template
      await chatStore.executeFromTemplate(template.id, {
        subtasks: fullTemplate.subtasks,
        summary_task: fullTemplate.summary_task,
        agents: fullTemplate.agents,
      });
      
      toast.success("Executing template...");
      onClose();
    } catch (error) {
      toast.error("Failed to execute template");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {mode === "save" ? (
              <div className="flex items-center gap-2">
                <Save className="w-5 h-5" />
                Save as Template
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Execute from Template
              </div>
            )}
          </DialogTitle>
        </DialogHeader>

        {mode === "save" ? (
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">Template Name</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Web Research Workflow"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Description (optional)</label>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe what this template does..."
                rows={3}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button onClick={handleSaveTemplate} disabled={loading || !name.trim()}>
                {loading ? "Saving..." : "Save Template"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {loading ? (
              <div className="text-center py-8">Loading templates...</div>
            ) : templates.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                No templates saved yet. Complete a task and save it as a template!
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {templates.map((template) => (
                  <div
                    key={template.id}
                    className="p-4 border rounded-lg hover:bg-gray-50 cursor-pointer"
                    onClick={() => handleExecuteTemplate(template)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium">{template.name}</h3>
                        <p className="text-sm text-gray-500 truncate">
                          {template.summary_task}
                        </p>
                        {template.description && (
                          <p className="text-xs text-gray-400 mt-1">
                            {template.description}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-xs text-gray-400 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Used {template.use_count}x
                        </div>
                        <Button size="sm" variant="ghost">
                          <PlayCircle className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
```

```plaintext
┌────────────────────────────────────────────────────────────────────────┐
│                         EXPECTED BENEFITS                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  💰 COST SAVINGS                                                       │
│  ├── Skip AI orchestration tokens (~30-50% reduction)                  │
│  ├── No planning/decomposition LLM calls                               │
│  └── Only execution tokens consumed                                    │
│                                                                        │
│  ⏱️ TIME SAVINGS                                                        │
│  ├── Immediate start (no 5-30 second planning delay)                   │
│  ├── Instant UI feedback with pre-loaded subtasks                      │
│  └── Predictable execution time                                        │
│                                                                        │
│  🔄 CONSISTENCY                                                         │
│  ├── Same subtask structure every time                                 │
│  ├── Same agent assignments                                            │
│  └── Reproducible workflows                                            │
│                                                                        │
│  📚 KNOWLEDGE CAPTURE                                                   │
│  ├── Save successful workflow patterns                                 │
│  ├── Share templates across team (is_public flag)                      │
│  └── Build library of proven workflows                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

