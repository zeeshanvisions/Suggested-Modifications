**User**

Awesome, The agents are working fine but It takes too long to execute. As when I re-execute It skips the planning phase and directly creates tasks and sub-tasks. But the agents go though their planning phase and multiple steps. HOw can I optimise it or use exsiting learning. Or use memory so that the number of steps are reduced and I can get faster response. Or how it can be self learning. Give me all available solutions and suggest me best solution

---

**Cursor**

Great question! Let me analyze the current architecture and available optimization options.

Based on my analysis of your codebase, here are all the available solutions for faster re-execution:

---

## 🔍 Current Architecture Analysis

**What you have:**
- `ChatStep` table stores all execution steps (task decomposition, agent actions, tool calls)
- `WorkflowTemplate` stores subtask structure from completed tasks
- `NoteTakingToolkit` allows agents to save notes between executions
- Each agent has its own planning/reasoning phase via CAMEL framework

**Why re-execute is still slow:**
- Template mode only skips **workforce-level** task decomposition
- Individual agents still go through full LLM reasoning for each tool call
- No caching of successful tool call sequences

---

## 📋 Available Solutions (Ranked by Impact & Complexity)

### **Solution 1: Tool Call Replay (Direct Execution)** ⭐ BEST
**Impact: Very High | Complexity: Medium**

Save and replay the exact sequence of tool calls from successful executions, bypassing agent reasoning entirely.

**How it works:**
1. During first execution, save tool calls to a new `tool_call_log` table:
   - Agent name, tool name, arguments, result
2. On re-execute, replay tool calls directly without invoking the LLM

**Proposed schema:**
```python
class ToolCallLog(AbstractModel, table=True):
    id: int
    template_id: int  # Reference to workflow_template
    subtask_index: int
    agent_name: str
    tool_name: str
    tool_args: str  # JSON
    tool_result: str  # JSON (optional, for verification)
    execution_order: int
```

**Pros:** 
- Near-instant execution (no LLM calls)
- Deterministic results
- Zero token cost on re-execute

**Cons:**
- Only works for identical tasks
- Won't adapt to changed conditions

---

### **Solution 2: Few-Shot Learning with Execution History** 
**Impact: High | Complexity: Low-Medium**

Provide agents with examples from past successful executions as part of their context.

**How it works:**
1. Query `ChatStep` for similar past tasks
2. Extract successful tool call sequences
3. Inject as few-shot examples in agent's system message

**Implementation:**
```python
# In agent.py, modify system_message to include:
past_examples = get_similar_task_executions(task_content, limit=3)
enhanced_prompt = f"""
{original_system_message}

## Examples from successful past executions:
{format_examples(past_examples)}

Follow similar patterns for efficiency.
"""
```

**Pros:**
- Agents learn from experience
- Flexible - can adapt to variations
- Uses existing data

**Cons:**
- Still requires LLM calls (but fewer iterations)
- Needs good similarity matching

---

### **Solution 3: Agent Memory Persistence**
**Impact: Medium | Complexity: Medium**

Store agent memory/context between executions so they "remember" what worked.

**How it works:**
1. After successful execution, save agent's memory to database
2. On re-execute, restore the memory state
3. Agent has pre-loaded context of successful patterns

**The code already has `clone(with_memory=True)`** - this just needs persistence.

---

### **Solution 4: Use Faster/Smaller Models for Re-execution**
**Impact: Medium | Complexity: Low**

Switch to faster models (e.g., `gemini-2.0-flash`) for template re-execution since the task structure is already known.

**Implementation:**
```python
# In Chat model, add:
reexecute_model_type: str | None = None  # Faster model for re-execute

# Use this model when template_mode=True
```

---

### **Solution 5: Parallel Agent Execution**
**Impact: Medium | Complexity: Medium**

Execute independent subtasks in parallel instead of sequentially.

The workforce already has some parallel execution support, but it could be enhanced for template mode where we know all subtasks upfront.

---

### **Solution 6: Tool Result Caching**
**Impact: Low-Medium | Complexity: Low**

Cache results of idempotent tool calls (e.g., file reads, web searches for same query).

```python
@lru_cache(maxsize=100)
def cached_tool_result(tool_name: str, args_hash: str) -> str:
    # Return cached result if available
```

---

## 🏆 Recommended Approach

**For your use case, I recommend a hybrid approach:**

### Phase 1 (Quick Win)
1. **Faster model for re-execute** - Use `gemini-2.0-flash` instead of `gemini-2.5-pro`
2. **Few-shot examples** - Inject past successful tool calls into agent context

### Phase 2 (Full Optimization)  
1. **Tool Call Replay** - Save and replay tool sequences for identical tasks
2. Add a toggle in UI: "Replay mode" vs "Adaptive mode"

---

## Would you like me to implement any of these?

Since you're in **ask mode**, switch to **agent mode** if you want me to implement:
1. **Tool Call Replay system** (database model + backend changes + frontend toggle)
2. **Few-shot learning injection** (agent.py modification)
3. **Faster model configuration** for re-execute mode

Let me know which solution(s) you'd like to proceed with!

```python
class ToolCallLog(AbstractModel, table=True):
    id: int
    template_id: int  # Reference to workflow_template
    subtask_index: int
    agent_name: str
    tool_name: str
    tool_args: str  # JSON
    tool_result: str  # JSON (optional, for verification)
    execution_order: int
```

```python
# In agent.py, modify system_message to include:
past_examples = get_similar_task_executions(task_content, limit=3)
enhanced_prompt = f"""
{original_system_message}

## Examples from successful past executions:
{format_examples(past_examples)}

Follow similar patterns for efficiency.
"""
```

```python
# In Chat model, add:
reexecute_model_type: str | None = None  # Faster model for re-execute

# Use this model when template_mode=True
```

```python
@lru_cache(maxsize=100)
def cached_tool_result(tool_name: str, args_hash: str) -> str:
    # Return cached result if available
```

---

**User**

I am thinking of combination of solution 2 and solution 3. As I believe it will be best for long term solution

---

**Cursor**

Excellent choice! Combining **Few-Shot Learning** + **Agent Memory Persistence** gives you:
- ✅ Agents that learn from experience
- ✅ Flexibility to adapt to variations
- ✅ Accumulated knowledge over time
- ✅ Self-improving system

Let me design the complete implementation:

---

## 🏗️ Implementation Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    RE-EXECUTE FLOW                          │
├─────────────────────────────────────────────────────────────┤
│  1. Load Agent Memory from DB (past context/patterns)       │
│  2. Query Similar Successful Executions                     │
│  3. Inject Few-Shot Examples into System Prompt             │
│  4. Agent executes with enhanced context                    │
│  5. Save updated memory + successful tool calls to DB       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Database Schema Changes

### New Table: `agent_memory`
Stores persistent memory for each agent per task template.

```python
# server/app/model/chat/agent_memory.py

from sqlmodel import Field, JSON, Column, Text
from app.model.abstract.model import AbstractModel, DefaultTimes
from pydantic import BaseModel
from typing import Optional
import json


class AgentMemory(AbstractModel, DefaultTimes, table=True):
    """
    Persistent memory storage for agents.
    Allows agents to "remember" successful patterns across executions.
    """
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    template_id: int = Field(index=True)  # Links to workflow_template
    agent_name: str = Field(index=True)   # e.g., "developer_agent", "search_agent"
    
    # Memory content - successful patterns, learned behaviors
    memory_context: str = Field(sa_column=Column(Text))  # JSON serialized
    
    # Statistics for learning
    success_count: int = Field(default=0)
    total_executions: int = Field(default=0)
    avg_steps_to_complete: float = Field(default=0.0)
    
    class Config:
        arbitrary_types_allowed = True


class AgentMemoryIn(BaseModel):
    template_id: int
    agent_name: str
    memory_context: dict


class AgentMemoryOut(BaseModel):
    id: int
    template_id: int
    agent_name: str
    memory_context: dict
    success_count: int
    total_executions: int
```

### New Table: `tool_execution_history`
Stores successful tool call patterns for few-shot learning.

```python
# server/app/model/chat/tool_execution_history.py

from sqlmodel import Field, JSON, Column, Text
from app.model.abstract.model import AbstractModel, DefaultTimes
from pydantic import BaseModel
from typing import Optional


class ToolExecutionHistory(AbstractModel, DefaultTimes, table=True):
    """
    Records successful tool executions for few-shot learning.
    Used to provide examples to agents during re-execution.
    """
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    template_id: int = Field(index=True)
    source_task_id: str = Field(index=True)  # Original task this came from
    
    # Subtask info
    subtask_content: str = Field(sa_column=Column(Text))
    subtask_index: int
    
    # Agent and tool info
    agent_name: str
    tool_name: str
    tool_args: str = Field(sa_type=JSON)  # Arguments used
    tool_result_summary: str = Field(sa_column=Column(Text))  # Truncated result
    
    # Execution metrics
    execution_order: int
    was_successful: bool = Field(default=True)
    execution_time_ms: int = Field(default=0)


class ToolExecutionHistoryIn(BaseModel):
    template_id: int
    source_task_id: str
    subtask_content: str
    subtask_index: int
    agent_name: str
    tool_name: str
    tool_args: dict
    tool_result_summary: str
    execution_order: int
    was_successful: bool = True
```

---

## 🔧 Backend Implementation

### 1. Agent Memory Service
```python
# backend/app/service/agent_memory_service.py

from typing import Optional, List
from loguru import logger
import json
import httpx

class AgentMemoryService:
    """Service for managing agent memory persistence."""
    
    def __init__(self, server_url: str = "http://localhost:3001"):
        self.server_url = server_url
    
    async def load_memory(
        self, 
        template_id: int, 
        agent_name: str,
        auth_token: str
    ) -> Optional[dict]:
        """Load persisted memory for an agent."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/api/agent-memory",
                    params={"template_id": template_id, "agent_name": agent_name},
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("items"):
                        return json.loads(data["items"][0]["memory_context"])
        except Exception as e:
            logger.error(f"Failed to load agent memory: {e}")
        return None
    
    async def save_memory(
        self,
        template_id: int,
        agent_name: str,
        memory_context: dict,
        success: bool,
        auth_token: str
    ):
        """Save or update agent memory after execution."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.server_url}/api/agent-memory",
                    json={
                        "template_id": template_id,
                        "agent_name": agent_name,
                        "memory_context": json.dumps(memory_context),
                        "increment_success": success
                    },
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
        except Exception as e:
            logger.error(f"Failed to save agent memory: {e}")
    
    async def get_few_shot_examples(
        self,
        template_id: int,
        subtask_content: str,
        agent_name: str,
        limit: int = 3,
        auth_token: str = ""
    ) -> List[dict]:
        """Get few-shot examples from past successful executions."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/api/tool-execution-history",
                    params={
                        "template_id": template_id,
                        "agent_name": agent_name,
                        "limit": limit
                    },
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
                if response.status_code == 200:
                    return response.json().get("items", [])
        except Exception as e:
            logger.error(f"Failed to get few-shot examples: {e}")
        return []
```

### 2. Enhanced Agent with Memory & Few-Shot Learning
```python
# Modifications to backend/app/utils/agent.py

def build_enhanced_system_message(
    original_message: str,
    memory_context: Optional[dict],
    few_shot_examples: List[dict],
    agent_name: str
) -> str:
    """
    Enhance agent's system message with:
    1. Loaded memory from past executions
    2. Few-shot examples of successful tool usage
    """
    enhanced_parts = [original_message]
    
    # Add memory context
    if memory_context:
        enhanced_parts.append(f"""
## 🧠 Your Past Learning & Patterns
You have successfully completed similar tasks before. Here's what you learned:

### Successful Strategies:
{memory_context.get('successful_strategies', 'No specific strategies recorded yet.')}

### Common Pitfalls to Avoid:
{memory_context.get('pitfalls_to_avoid', 'None recorded.')}

### Preferred Tool Sequences:
{memory_context.get('preferred_tool_sequences', 'Use your best judgment.')}
""")
    
    # Add few-shot examples
    if few_shot_examples:
        examples_text = "\n\n## 📚 Examples from Successful Past Executions\n"
        examples_text += "Learn from these successful patterns:\n\n"
        
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"""
### Example {i}:
**Task:** {example.get('subtask_content', 'N/A')[:200]}
**Tool Used:** {example.get('tool_name')}
**Arguments:** ```{json.dumps(example.get('tool_args', {}), indent=2)[:300]}```
**Result:** {example.get('tool_result_summary', 'Success')[:200]}
---
"""
        enhanced_parts.append(examples_text)
    
    # Add efficiency instruction
    enhanced_parts.append("""
## ⚡ Efficiency Guidelines
- Use the patterns from past successful executions when applicable
- Minimize unnecessary exploration - follow proven approaches
- If you've seen this task type before, apply learned strategies
- Complete the task in as few steps as possible
""")
    
    return "\n".join(enhanced_parts)


# Modify agent_model function to accept memory
def agent_model(
    agent_name: str,
    system_message: str | BaseMessage,
    options: Chat,
    tools: list[FunctionTool | Callable] | None = None,
    prune_tool_calls_from_memory: bool = False,
    tool_names: list[str] | None = None,
    toolkits_to_register_agent: list[RegisteredAgentToolkit] | None = None,
    # NEW: Memory and learning parameters
    memory_context: dict | None = None,
    few_shot_examples: list[dict] | None = None,
):
    task_lock = get_task_lock(options.task_id)
    agent_id = str(uuid.uuid4())

    # Get provider config for this specific agent  
    model_platform, model_type, api_key, api_url, extra_params = options.get_agent_provider(agent_name)  
    
    # Enhance system message with memory and examples (for template mode)
    if options.template_mode and (memory_context or few_shot_examples):
        if isinstance(system_message, BaseMessage):
            system_message_str = system_message.content
        else:
            system_message_str = system_message
            
        enhanced_message = build_enhanced_system_message(
            system_message_str,
            memory_context,
            few_shot_examples or [],
            agent_name
        )
        system_message = enhanced_message
        traceroot_logger.info(f"[MEMORY] Enhanced agent {agent_name} with memory and {len(few_shot_examples or [])} examples")
    
    # ... rest of the function remains the same
```

### 3. Memory Capture During Execution
```python
# Add to ListenChatAgent class in agent.py

class ListenChatAgent(ChatAgent):
    # ... existing code ...
    
    # NEW: Track tool calls for memory
    _execution_log: list = []
    
    def _execute_tool(self, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
        """Execute tool and log for memory."""
        import time
        start_time = time.time()
        
        result = super()._execute_tool(tool_call_request)
        
        # Log successful execution for learning
        execution_time = int((time.time() - start_time) * 1000)
        self._execution_log.append({
            "tool_name": tool_call_request.tool_name,
            "tool_args": tool_call_request.args,
            "result_summary": str(result.result)[:500],
            "execution_time_ms": execution_time,
            "was_successful": result.error is None
        })
        
        return result
    
    def get_execution_log(self) -> list:
        """Get logged tool executions for memory persistence."""
        return self._execution_log
    
    def build_memory_context(self) -> dict:
        """Build memory context from this execution."""
        successful_tools = [e for e in self._execution_log if e["was_successful"]]
        
        return {
            "successful_strategies": f"Used {len(successful_tools)} tools successfully",
            "preferred_tool_sequences": [e["tool_name"] for e in successful_tools],
            "pitfalls_to_avoid": [
                e["tool_name"] for e in self._execution_log 
                if not e["was_successful"]
            ],
            "avg_execution_time": sum(e["execution_time_ms"] for e in self._execution_log) / len(self._execution_log) if self._execution_log else 0
        }
```

---

## 🖥️ Server API Endpoints

```python
# server/app/controller/chat/memory_controller.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select, desc
from app.component.auth import Auth, auth_must
from app.component.database import session
from app.model.chat.agent_memory import AgentMemory, AgentMemoryIn, AgentMemoryOut
from app.model.chat.tool_execution_history import ToolExecutionHistory

router = APIRouter(prefix="/api", tags=["Agent Memory"])


@router.get("/agent-memory", name="Get agent memory")
async def get_agent_memory(
    template_id: int,
    agent_name: str,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> Page[AgentMemoryOut]:
    stmt = (
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == template_id)
        .where(AgentMemory.agent_name == agent_name)
    )
    return paginate(session, stmt)


@router.post("/agent-memory", name="Save agent memory")
async def save_agent_memory(
    data: AgentMemoryIn,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    # Check if memory exists
    existing = session.exec(
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == data.template_id)
        .where(AgentMemory.agent_name == data.agent_name)
    ).first()
    
    if existing:
        # Update existing
        existing.memory_context = json.dumps(data.memory_context)
        existing.total_executions += 1
        if data.get("increment_success"):
            existing.success_count += 1
        session.add(existing)
    else:
        # Create new
        memory = AgentMemory(
            user_id=auth.user.id,
            template_id=data.template_id,
            agent_name=data.agent_name,
            memory_context=json.dumps(data.memory_context),
            total_executions=1,
            success_count=1 if data.get("increment_success") else 0
        )
        session.add(memory)
    
    session.commit()
    return {"status": "success"}


@router.get("/tool-execution-history", name="Get tool execution history")
async def get_tool_execution_history(
    template_id: int,
    agent_name: str,
    limit: int = 5,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    stmt = (
        select(ToolExecutionHistory)
        .where(ToolExecutionHistory.user_id == auth.user.id)
        .where(ToolExecutionHistory.template_id == template_id)
        .where(ToolExecutionHistory.agent_name == agent_name)
        .where(ToolExecutionHistory.was_successful == True)
        .order_by(desc(ToolExecutionHistory.created_at))
        .limit(limit)
    )
    results = session.exec(stmt).all()
    return {"items": [r.dict() for r in results]}


@router.post("/tool-execution-history", name="Save tool execution")
async def save_tool_execution(
    data: list[dict],
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    for item in data:
        history = ToolExecutionHistory(
            user_id=auth.user.id,
            **item
        )
        session.add(history)
    session.commit()
    return {"status": "success", "count": len(data)}
```

---

## 📝 Migration File

```python
# server/alembic/versions/2025_12_09_add_agent_memory_tables.py

"""Add agent memory and tool execution history tables

Revision ID: add_agent_memory
Revises: add_reexecute_fields
Create Date: 2025-12-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_agent_memory'
down_revision = 'add_reexecute_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agent Memory table
    op.create_table(
        'agent_memory',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('memory_context', sa.Text(), nullable=True),
        sa.Column('success_count', sa.Integer(), server_default='0'),
        sa.Column('total_executions', sa.Integer(), server_default='0'),
        sa.Column('avg_steps_to_complete', sa.Float(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_memory_user_template', 'agent_memory', ['user_id', 'template_id', 'agent_name'])
    
    # Tool Execution History table
    op.create_table(
        'tool_execution_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('source_task_id', sa.String(64), nullable=False),
        sa.Column('subtask_content', sa.Text(), nullable=True),
        sa.Column('subtask_index', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('tool_name', sa.String(128), nullable=False),
        sa.Column('tool_args', sa.JSON(), nullable=True),
        sa.Column('tool_result_summary', sa.Text(), nullable=True),
        sa.Column('execution_order', sa.Integer(), nullable=False),
        sa.Column('was_successful', sa.Boolean(), server_default='true'),
        sa.Column('execution_time_ms', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tool_history_lookup', 'tool_execution_history', 
                    ['user_id', 'template_id', 'agent_name', 'was_successful'])


def downgrade() -> None:
    op.drop_index('ix_tool_history_lookup', table_name='tool_execution_history')
    op.drop_table('tool_execution_history')
    op.drop_index('ix_agent_memory_user_template', table_name='agent_memory')
    op.drop_table('agent_memory')
```

---

## 🔄 Flow Diagram

```
FIRST EXECUTION:
┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ User Task  │───▶│ Agent Plans │───▶│ Execute Tool │───▶│ Save to Memory │
└────────────┘    └─────────────┘    └──────────────┘    └────────────────┘
                                            │
                                            ▼
                                  ┌────────────────────┐
                                  │ tool_execution_    │
                                  │ history (DB)       │
                                  └────────────────────┘

RE-EXECUTION:
┌────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐
│ Re-execute │───▶│ Load Memory  │───▶│ Load Few-Shot   │───▶│ Enhanced    │
│ Request    │    │ from DB      │    │ Examples from DB│    │ Agent       │
└────────────┘    └──────────────┘    └─────────────────┘    └─────────────┘
                                                                   │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │ Faster Execution│
                                                          │ (fewer steps)   │
                                                          └─────────────────┘
```

---

## 📈 Expected Performance Improvement

| Metric | Before | After (Estimated) |
|--------|--------|-------------------|
| LLM Calls per subtask | 5-10 | 2-4 |
| Avg. reasoning steps | 8-12 | 3-5 |
| Token usage | 100% | 40-60% |
| Execution time | 100% | 50-70% |

---

## 🚀 Next Steps

You're in **ask mode**. To implement this solution:

1. **Switch to agent mode** (use the mode toggle)
2. I'll create all the files:
   - Database models
   - Migration
   - API endpoints  
   - Agent modifications
   - Integration with re-execute flow

Would you like me to proceed with the implementation when you switch to agent mode?

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                    RE-EXECUTE FLOW                          │
├─────────────────────────────────────────────────────────────┤
│  1. Load Agent Memory from DB (past context/patterns)       │
│  2. Query Similar Successful Executions                     │
│  3. Inject Few-Shot Examples into System Prompt             │
│  4. Agent executes with enhanced context                    │
│  5. Save updated memory + successful tool calls to DB       │
└─────────────────────────────────────────────────────────────┘
```

```python
# server/app/model/chat/agent_memory.py

from sqlmodel import Field, JSON, Column, Text
from app.model.abstract.model import AbstractModel, DefaultTimes
from pydantic import BaseModel
from typing import Optional
import json


class AgentMemory(AbstractModel, DefaultTimes, table=True):
    """
    Persistent memory storage for agents.
    Allows agents to "remember" successful patterns across executions.
    """
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    template_id: int = Field(index=True)  # Links to workflow_template
    agent_name: str = Field(index=True)   # e.g., "developer_agent", "search_agent"
    
    # Memory content - successful patterns, learned behaviors
    memory_context: str = Field(sa_column=Column(Text))  # JSON serialized
    
    # Statistics for learning
    success_count: int = Field(default=0)
    total_executions: int = Field(default=0)
    avg_steps_to_complete: float = Field(default=0.0)
    
    class Config:
        arbitrary_types_allowed = True


class AgentMemoryIn(BaseModel):
    template_id: int
    agent_name: str
    memory_context: dict


class AgentMemoryOut(BaseModel):
    id: int
    template_id: int
    agent_name: str
    memory_context: dict
    success_count: int
    total_executions: int
```

```python
# server/app/model/chat/tool_execution_history.py

from sqlmodel import Field, JSON, Column, Text
from app.model.abstract.model import AbstractModel, DefaultTimes
from pydantic import BaseModel
from typing import Optional


class ToolExecutionHistory(AbstractModel, DefaultTimes, table=True):
    """
    Records successful tool executions for few-shot learning.
    Used to provide examples to agents during re-execution.
    """
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    template_id: int = Field(index=True)
    source_task_id: str = Field(index=True)  # Original task this came from
    
    # Subtask info
    subtask_content: str = Field(sa_column=Column(Text))
    subtask_index: int
    
    # Agent and tool info
    agent_name: str
    tool_name: str
    tool_args: str = Field(sa_type=JSON)  # Arguments used
    tool_result_summary: str = Field(sa_column=Column(Text))  # Truncated result
    
    # Execution metrics
    execution_order: int
    was_successful: bool = Field(default=True)
    execution_time_ms: int = Field(default=0)


class ToolExecutionHistoryIn(BaseModel):
    template_id: int
    source_task_id: str
    subtask_content: str
    subtask_index: int
    agent_name: str
    tool_name: str
    tool_args: dict
    tool_result_summary: str
    execution_order: int
    was_successful: bool = True
```

```python
# backend/app/service/agent_memory_service.py

from typing import Optional, List
from loguru import logger
import json
import httpx

class AgentMemoryService:
    """Service for managing agent memory persistence."""
    
    def __init__(self, server_url: str = "http://localhost:3001"):
        self.server_url = server_url
    
    async def load_memory(
        self, 
        template_id: int, 
        agent_name: str,
        auth_token: str
    ) -> Optional[dict]:
        """Load persisted memory for an agent."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/api/agent-memory",
                    params={"template_id": template_id, "agent_name": agent_name},
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("items"):
                        return json.loads(data["items"][0]["memory_context"])
        except Exception as e:
            logger.error(f"Failed to load agent memory: {e}")
        return None
    
    async def save_memory(
        self,
        template_id: int,
        agent_name: str,
        memory_context: dict,
        success: bool,
        auth_token: str
    ):
        """Save or update agent memory after execution."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.server_url}/api/agent-memory",
                    json={
                        "template_id": template_id,
                        "agent_name": agent_name,
                        "memory_context": json.dumps(memory_context),
                        "increment_success": success
                    },
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
        except Exception as e:
            logger.error(f"Failed to save agent memory: {e}")
    
    async def get_few_shot_examples(
        self,
        template_id: int,
        subtask_content: str,
        agent_name: str,
        limit: int = 3,
        auth_token: str = ""
    ) -> List[dict]:
        """Get few-shot examples from past successful executions."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.server_url}/api/tool-execution-history",
                    params={
                        "template_id": template_id,
                        "agent_name": agent_name,
                        "limit": limit
                    },
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
                if response.status_code == 200:
                    return response.json().get("items", [])
        except Exception as e:
            logger.error(f"Failed to get few-shot examples: {e}")
        return []
```

```python
# Modifications to backend/app/utils/agent.py

def build_enhanced_system_message(
    original_message: str,
    memory_context: Optional[dict],
    few_shot_examples: List[dict],
    agent_name: str
) -> str:
    """
    Enhance agent's system message with:
    1. Loaded memory from past executions
    2. Few-shot examples of successful tool usage
    """
    enhanced_parts = [original_message]
    
    # Add memory context
    if memory_context:
        enhanced_parts.append(f"""
## 🧠 Your Past Learning & Patterns
You have successfully completed similar tasks before. Here's what you learned:

### Successful Strategies:
{memory_context.get('successful_strategies', 'No specific strategies recorded yet.')}

### Common Pitfalls to Avoid:
{memory_context.get('pitfalls_to_avoid', 'None recorded.')}

### Preferred Tool Sequences:
{memory_context.get('preferred_tool_sequences', 'Use your best judgment.')}
""")
    
    # Add few-shot examples
    if few_shot_examples:
        examples_text = "\n\n## 📚 Examples from Successful Past Executions\n"
        examples_text += "Learn from these successful patterns:\n\n"
        
        for i, example in enumerate(few_shot_examples, 1):
            examples_text += f"""
### Example {i}:
**Task:** {example.get('subtask_content', 'N/A')[:200]}
**Tool Used:** {example.get('tool_name')}
**Arguments:** ```{json.dumps(example.get('tool_args', {}), indent=2)[:300]}
```

```plaintext

### 3. Memory Capture During Execution
# Add to ListenChatAgent class in agent.py

class ListenChatAgent(ChatAgent):
    # ... existing code ...
    
    # NEW: Track tool calls for memory
    _execution_log: list = []
    
    def _execute_tool(self, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
        """Execute tool and log for memory."""
        import time
        start_time = time.time()
        
        result = super()._execute_tool(tool_call_request)
        
        # Log successful execution for learning
        execution_time = int((time.time() - start_time) * 1000)
        self._execution_log.append({
            "tool_name": tool_call_request.tool_name,
            "tool_args": tool_call_request.args,
            "result_summary": str(result.result)[:500],
            "execution_time_ms": execution_time,
            "was_successful": result.error is None
        })
        
        return result
    
    def get_execution_log(self) -> list:
        """Get logged tool executions for memory persistence."""
        return self._execution_log
    
    def build_memory_context(self) -> dict:
        """Build memory context from this execution."""
        successful_tools = [e for e in self._execution_log if e["was_successful"]]
        
        return {
            "successful_strategies": f"Used {len(successful_tools)} tools successfully",
            "preferred_tool_sequences": [e["tool_name"] for e in successful_tools],
            "pitfalls_to_avoid": [
                e["tool_name"] for e in self._execution_log 
                if not e["was_successful"]
            ],
            "avg_execution_time": sum(e["execution_time_ms"] for e in self._execution_log) / len(self._execution_log) if self._execution_log else 0
        }---

## 🖥️ Server API Endpoints

# server/app/controller/chat/memory_controller.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select, desc
from app.component.auth import Auth, auth_must
from app.component.database import session
from app.model.chat.agent_memory import AgentMemory, AgentMemoryIn, AgentMemoryOut
from app.model.chat.tool_execution_history import ToolExecutionHistory

router = APIRouter(prefix="/api", tags=["Agent Memory"])


@router.get("/agent-memory", name="Get agent memory")
async def get_agent_memory(
    template_id: int,
    agent_name: str,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> Page[AgentMemoryOut]:
    stmt = (
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == template_id)
        .where(AgentMemory.agent_name == agent_name)
    )
    return paginate(session, stmt)


@router.post("/agent-memory", name="Save agent memory")
async def save_agent_memory(
    data: AgentMemoryIn,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    # Check if memory exists
    existing = session.exec(
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == data.template_id)
        .where(AgentMemory.agent_name == data.agent_name)
    ).first()
    
    if existing:
        # Update existing
        existing.memory_context = json.dumps(data.memory_context)
        existing.total_executions += 1
        if data.get("increment_success"):
            existing.success_count += 1
        session.add(existing)
    else:
        # Create new
        memory = AgentMemory(
            user_id=auth.user.id,
            template_id=data.template_id,
            agent_name=data.agent_name,
            memory_context=json.dumps(data.memory_context),
            total_executions=1,
            success_count=1 if data.get("increment_success") else 0
        )
        session.add(memory)
    
    session.commit()
    return {"status": "success"}


@router.get("/tool-execution-history", name="Get tool execution history")
async def get_tool_execution_history(
    template_id: int,
    agent_name: str,
    limit: int = 5,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    stmt = (
        select(ToolExecutionHistory)
        .where(ToolExecutionHistory.user_id == auth.user.id)
        .where(ToolExecutionHistory.template_id == template_id)
        .where(ToolExecutionHistory.agent_name == agent_name)
        .where(ToolExecutionHistory.was_successful == True)
        .order_by(desc(ToolExecutionHistory.created_at))
        .limit(limit)
    )
    results = session.exec(stmt).all()
    return {"items": [r.dict() for r in results]}


@router.post("/tool-execution-history", name="Save tool execution")
async def save_tool_execution(
    data: list[dict],
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    for item in data:
        history = ToolExecutionHistory(
            user_id=auth.user.id,
            **item
        )
        session.add(history)
    session.commit()
    return {"status": "success", "count": len(data)}---

## 📝 Migration File

# server/alembic/versions/2025_12_09_add_agent_memory_tables.py

"""Add agent memory and tool execution history tables

Revision ID: add_agent_memory
Revises: add_reexecute_fields
Create Date: 2025-12-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_agent_memory'
down_revision = 'add_reexecute_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agent Memory table
    op.create_table(
        'agent_memory',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('memory_context', sa.Text(), nullable=True),
        sa.Column('success_count', sa.Integer(), server_default='0'),
        sa.Column('total_executions', sa.Integer(), server_default='0'),
        sa.Column('avg_steps_to_complete', sa.Float(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_memory_user_template', 'agent_memory', ['user_id', 'template_id', 'agent_name'])
    
    # Tool Execution History table
    op.create_table(
        'tool_execution_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('source_task_id', sa.String(64), nullable=False),
        sa.Column('subtask_content', sa.Text(), nullable=True),
        sa.Column('subtask_index', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('tool_name', sa.String(128), nullable=False),
        sa.Column('tool_args', sa.JSON(), nullable=True),
        sa.Column('tool_result_summary', sa.Text(), nullable=True),
        sa.Column('execution_order', sa.Integer(), nullable=False),
        sa.Column('was_successful', sa.Boolean(), server_default='true'),
        sa.Column('execution_time_ms', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tool_history_lookup', 'tool_execution_history', 
                    ['user_id', 'template_id', 'agent_name', 'was_successful'])


def downgrade() -> None:
    op.drop_index('ix_tool_history_lookup', table_name='tool_execution_history')
    op.drop_table('tool_execution_history')
    op.drop_index('ix_agent_memory_user_template', table_name='agent_memory')
    op.drop_table('agent_memory')---

## 🔄 Flow Diagram

```

```plaintext

### 3. Memory Capture During Execution
# Add to ListenChatAgent class in agent.py

class ListenChatAgent(ChatAgent):
    # ... existing code ...
    
    # NEW: Track tool calls for memory
    _execution_log: list = []
    
    def _execute_tool(self, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
        """Execute tool and log for memory."""
        import time
        start_time = time.time()
        
        result = super()._execute_tool(tool_call_request)
        
        # Log successful execution for learning
        execution_time = int((time.time() - start_time) * 1000)
        self._execution_log.append({
            "tool_name": tool_call_request.tool_name,
            "tool_args": tool_call_request.args,
            "result_summary": str(result.result)[:500],
            "execution_time_ms": execution_time,
            "was_successful": result.error is None
        })
        
        return result
    
    def get_execution_log(self) -> list:
        """Get logged tool executions for memory persistence."""
        return self._execution_log
    
    def build_memory_context(self) -> dict:
        """Build memory context from this execution."""
        successful_tools = [e for e in self._execution_log if e["was_successful"]]
        
        return {
            "successful_strategies": f"Used {len(successful_tools)} tools successfully",
            "preferred_tool_sequences": [e["tool_name"] for e in successful_tools],
            "pitfalls_to_avoid": [
                e["tool_name"] for e in self._execution_log 
                if not e["was_successful"]
            ],
            "avg_execution_time": sum(e["execution_time_ms"] for e in self._execution_log) / len(self._execution_log) if self._execution_log else 0
        }---

## 🖥️ Server API Endpoints

# server/app/controller/chat/memory_controller.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select, desc
from app.component.auth import Auth, auth_must
from app.component.database import session
from app.model.chat.agent_memory import AgentMemory, AgentMemoryIn, AgentMemoryOut
from app.model.chat.tool_execution_history import ToolExecutionHistory

router = APIRouter(prefix="/api", tags=["Agent Memory"])


@router.get("/agent-memory", name="Get agent memory")
async def get_agent_memory(
    template_id: int,
    agent_name: str,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> Page[AgentMemoryOut]:
    stmt = (
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == template_id)
        .where(AgentMemory.agent_name == agent_name)
    )
    return paginate(session, stmt)


@router.post("/agent-memory", name="Save agent memory")
async def save_agent_memory(
    data: AgentMemoryIn,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    # Check if memory exists
    existing = session.exec(
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == data.template_id)
        .where(AgentMemory.agent_name == data.agent_name)
    ).first()
    
    if existing:
        # Update existing
        existing.memory_context = json.dumps(data.memory_context)
        existing.total_executions += 1
        if data.get("increment_success"):
            existing.success_count += 1
        session.add(existing)
    else:
        # Create new
        memory = AgentMemory(
            user_id=auth.user.id,
            template_id=data.template_id,
            agent_name=data.agent_name,
            memory_context=json.dumps(data.memory_context),
            total_executions=1,
            success_count=1 if data.get("increment_success") else 0
        )
        session.add(memory)
    
    session.commit()
    return {"status": "success"}


@router.get("/tool-execution-history", name="Get tool execution history")
async def get_tool_execution_history(
    template_id: int,
    agent_name: str,
    limit: int = 5,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    stmt = (
        select(ToolExecutionHistory)
        .where(ToolExecutionHistory.user_id == auth.user.id)
        .where(ToolExecutionHistory.template_id == template_id)
        .where(ToolExecutionHistory.agent_name == agent_name)
        .where(ToolExecutionHistory.was_successful == True)
        .order_by(desc(ToolExecutionHistory.created_at))
        .limit(limit)
    )
    results = session.exec(stmt).all()
    return {"items": [r.dict() for r in results]}


@router.post("/tool-execution-history", name="Save tool execution")
async def save_tool_execution(
    data: list[dict],
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    for item in data:
        history = ToolExecutionHistory(
            user_id=auth.user.id,
            **item
        )
        session.add(history)
    session.commit()
    return {"status": "success", "count": len(data)}---

## 📝 Migration File

# server/alembic/versions/2025_12_09_add_agent_memory_tables.py

"""Add agent memory and tool execution history tables

Revision ID: add_agent_memory
Revises: add_reexecute_fields
Create Date: 2025-12-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_agent_memory'
down_revision = 'add_reexecute_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agent Memory table
    op.create_table(
        'agent_memory',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('memory_context', sa.Text(), nullable=True),
        sa.Column('success_count', sa.Integer(), server_default='0'),
        sa.Column('total_executions', sa.Integer(), server_default='0'),
        sa.Column('avg_steps_to_complete', sa.Float(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_memory_user_template', 'agent_memory', ['user_id', 'template_id', 'agent_name'])
    
    # Tool Execution History table
    op.create_table(
        'tool_execution_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('source_task_id', sa.String(64), nullable=False),
        sa.Column('subtask_content', sa.Text(), nullable=True),
        sa.Column('subtask_index', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('tool_name', sa.String(128), nullable=False),
        sa.Column('tool_args', sa.JSON(), nullable=True),
        sa.Column('tool_result_summary', sa.Text(), nullable=True),
        sa.Column('execution_order', sa.Integer(), nullable=False),
        sa.Column('was_successful', sa.Boolean(), server_default='true'),
        sa.Column('execution_time_ms', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tool_history_lookup', 'tool_execution_history', 
                    ['user_id', 'template_id', 'agent_name', 'was_successful'])


def downgrade() -> None:
    op.drop_index('ix_tool_history_lookup', table_name='tool_execution_history')
    op.drop_table('tool_execution_history')
    op.drop_index('ix_agent_memory_user_template', table_name='agent_memory')
    op.drop_table('agent_memory')---

## 🔄 Flow Diagram

```

```python

---

## 📈 Expected Performance Improvement

| Metric | Before | After (Estimated) |
|--------|--------|-------------------|
| LLM Calls per subtask | 5-10 | 2-4 |
| Avg. reasoning steps | 8-12 | 3-5 |
| Token usage | 100% | 40-60% |
| Execution time | 100% | 50-70% |

---

## 🚀 Next Steps

You're in **ask mode**. To implement this solution:

1. **Switch to agent mode** (use the mode toggle)
2. I'll create all the files:
   - Database models
   - Migration
   - API endpoints  
   - Agent modifications
   - Integration with re-execute flow

Would you like me to proceed with the implementation when you switch to agent mode?
```

```plaintext

---

## 📈 Expected Performance Improvement

| Metric | Before | After (Estimated) |
|--------|--------|-------------------|
| LLM Calls per subtask | 5-10 | 2-4 |
| Avg. reasoning steps | 8-12 | 3-5 |
| Token usage | 100% | 40-60% |
| Execution time | 100% | 50-70% |

---

## 🚀 Next Steps

You're in **ask mode**. To implement this solution:

1. **Switch to agent mode** (use the mode toggle)
2. I'll create all the files:
   - Database models
   - Migration
   - API endpoints  
   - Agent modifications
   - Integration with re-execute flow

Would you like me to proceed with the implementation when you switch to agent mode?
```

```python
# server/app/controller/chat/memory_controller.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import Session, select, desc
from app.component.auth import Auth, auth_must
from app.component.database import session
from app.model.chat.agent_memory import AgentMemory, AgentMemoryIn, AgentMemoryOut
from app.model.chat.tool_execution_history import ToolExecutionHistory

router = APIRouter(prefix="/api", tags=["Agent Memory"])


@router.get("/agent-memory", name="Get agent memory")
async def get_agent_memory(
    template_id: int,
    agent_name: str,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
) -> Page[AgentMemoryOut]:
    stmt = (
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == template_id)
        .where(AgentMemory.agent_name == agent_name)
    )
    return paginate(session, stmt)


@router.post("/agent-memory", name="Save agent memory")
async def save_agent_memory(
    data: AgentMemoryIn,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    # Check if memory exists
    existing = session.exec(
        select(AgentMemory)
        .where(AgentMemory.user_id == auth.user.id)
        .where(AgentMemory.template_id == data.template_id)
        .where(AgentMemory.agent_name == data.agent_name)
    ).first()
    
    if existing:
        # Update existing
        existing.memory_context = json.dumps(data.memory_context)
        existing.total_executions += 1
        if data.get("increment_success"):
            existing.success_count += 1
        session.add(existing)
    else:
        # Create new
        memory = AgentMemory(
            user_id=auth.user.id,
            template_id=data.template_id,
            agent_name=data.agent_name,
            memory_context=json.dumps(data.memory_context),
            total_executions=1,
            success_count=1 if data.get("increment_success") else 0
        )
        session.add(memory)
    
    session.commit()
    return {"status": "success"}


@router.get("/tool-execution-history", name="Get tool execution history")
async def get_tool_execution_history(
    template_id: int,
    agent_name: str,
    limit: int = 5,
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    stmt = (
        select(ToolExecutionHistory)
        .where(ToolExecutionHistory.user_id == auth.user.id)
        .where(ToolExecutionHistory.template_id == template_id)
        .where(ToolExecutionHistory.agent_name == agent_name)
        .where(ToolExecutionHistory.was_successful == True)
        .order_by(desc(ToolExecutionHistory.created_at))
        .limit(limit)
    )
    results = session.exec(stmt).all()
    return {"items": [r.dict() for r in results]}


@router.post("/tool-execution-history", name="Save tool execution")
async def save_tool_execution(
    data: list[dict],
    session: Session = Depends(session),
    auth: Auth = Depends(auth_must)
):
    for item in data:
        history = ToolExecutionHistory(
            user_id=auth.user.id,
            **item
        )
        session.add(history)
    session.commit()
    return {"status": "success", "count": len(data)}
```

```python
# server/alembic/versions/2025_12_09_add_agent_memory_tables.py

"""Add agent memory and tool execution history tables

Revision ID: add_agent_memory
Revises: add_reexecute_fields
Create Date: 2025-12-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_agent_memory'
down_revision = 'add_reexecute_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agent Memory table
    op.create_table(
        'agent_memory',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('memory_context', sa.Text(), nullable=True),
        sa.Column('success_count', sa.Integer(), server_default='0'),
        sa.Column('total_executions', sa.Integer(), server_default='0'),
        sa.Column('avg_steps_to_complete', sa.Float(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_agent_memory_user_template', 'agent_memory', ['user_id', 'template_id', 'agent_name'])
    
    # Tool Execution History table
    op.create_table(
        'tool_execution_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('template_id', sa.Integer(), nullable=False),
        sa.Column('source_task_id', sa.String(64), nullable=False),
        sa.Column('subtask_content', sa.Text(), nullable=True),
        sa.Column('subtask_index', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(64), nullable=False),
        sa.Column('tool_name', sa.String(128), nullable=False),
        sa.Column('tool_args', sa.JSON(), nullable=True),
        sa.Column('tool_result_summary', sa.Text(), nullable=True),
        sa.Column('execution_order', sa.Integer(), nullable=False),
        sa.Column('was_successful', sa.Boolean(), server_default='true'),
        sa.Column('execution_time_ms', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_tool_history_lookup', 'tool_execution_history', 
                    ['user_id', 'template_id', 'agent_name', 'was_successful'])


def downgrade() -> None:
    op.drop_index('ix_tool_history_lookup', table_name='tool_execution_history')
    op.drop_table('tool_execution_history')
    op.drop_index('ix_agent_memory_user_template', table_name='agent_memory')
    op.drop_table('agent_memory')
```

```plaintext
FIRST EXECUTION:
┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│ User Task  │───▶│ Agent Plans │───▶│ Execute Tool │───▶│ Save to Memory │
└────────────┘    └─────────────┘    └──────────────┘    └────────────────┘
                                            │
                                            ▼
                                  ┌────────────────────┐
                                  │ tool_execution_    │
                                  │ history (DB)       │
                                  └────────────────────┘

RE-EXECUTION:
┌────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────┐
│ Re-execute │───▶│ Load Memory  │───▶│ Load Few-Shot   │───▶│ Enhanced    │
│ Request    │    │ from DB      │    │ Examples from DB│    │ Agent       │
└────────────┘    └──────────────┘    └─────────────────┘    └─────────────┘
                                                                   │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │ Faster Execution│
                                                          │ (fewer steps)   │
                                                          └─────────────────┘
```

---

**User**

Create a more detailed plan how Few-Shot examples will be created. And Memory will be stored using LLM. Also share diagram of all architecture changes and make sure it only happen it only happens in template mode

---

**Cursor**

# 🧠 Detailed Architecture: Few-Shot Learning + LLM-Powered Memory

## 📋 Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Few-Shot Example Creation Flow](#few-shot-example-creation)
3. [LLM-Powered Memory Synthesis](#llm-memory-synthesis)
4. [Template Mode Gate](#template-mode-gate)
5. [Data Models](#data-models)
6. [Component Interactions](#component-interactions)

---

## 🏗️ High-Level Architecture {#high-level-architecture}

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI AGENT PLATFORM                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────┐     ┌──────────────────────┐     ┌────────────────┐  │
│  │   FRONTEND (React)   │     │   LOCAL BACKEND      │     │  SERVER (API)  │  │
│  │                      │     │   (Python/FastAPI)   │     │  (Docker)      │  │
│  │  • Re-execute UI     │────▶│  • Agent Execution   │────▶│  • Memory DB   │  │
│  │  • Template Toggle   │     │  • Memory Loading    │     │  • History DB  │  │
│  │                      │     │  • Learning Capture  │     │  • Templates   │  │
│  └──────────────────────┘     └──────────────────────┘     └────────────────┘  │
│                                        │                            │           │
│                                        │                            │           │
│                                        ▼                            ▼           │
│                               ┌─────────────────────────────────────────┐       │
│                               │         LEARNING SYSTEM                 │       │
│                               │  ┌─────────────┐  ┌─────────────────┐  │       │
│                               │  │  Few-Shot   │  │  LLM Memory     │  │       │
│                               │  │  Collector  │  │  Synthesizer    │  │       │
│                               │  └─────────────┘  └─────────────────┘  │       │
│                               └─────────────────────────────────────────┘       │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════   │
│  ONLY ACTIVE WHEN: options.template_mode == True                                │
│  ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Few-Shot Example Creation Flow {#few-shot-example-creation}

### When Examples Are Created

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEW-SHOT EXAMPLE CREATION TRIGGER                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TRIGGER: Task completes successfully AND user saves as template           │
│                                                                              │
│   ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐           │
│   │ Task        │      │ User clicks  │      │ Extract & Save  │           │
│   │ Completes   │─────▶│ "Save        │─────▶│ Few-Shot        │           │
│   │ Successfully│      │ Template"    │      │ Examples        │           │
│   └─────────────┘      └──────────────┘      └─────────────────┘           │
│                                                      │                      │
│                                                      ▼                      │
│                                            ┌─────────────────┐              │
│                                            │ tool_execution_ │              │
│                                            │ history (DB)    │              │
│                                            └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Example Extraction Process

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     FEW-SHOT EXAMPLE EXTRACTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 1: Capture During Execution                                               │
│  ════════════════════════════════                                               │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  For each tool call in agent._execute_tool():                            │  │
│  │                                                                          │  │
│  │  execution_record = {                                                    │  │
│  │      "subtask_id": current_subtask.id,                                  │  │
│  │      "subtask_content": current_subtask.content,                        │  │
│  │      "agent_name": self.agent_name,                                     │  │
│  │      "tool_name": tool_call_request.tool_name,                          │  │
│  │      "tool_args": tool_call_request.args,          # Full arguments     │  │
│  │      "tool_result": result[:1000],                 # Truncated result   │  │
│  │      "execution_order": self._call_counter,                             │  │
│  │      "timestamp": datetime.now(),                                       │  │
│  │      "was_successful": error is None,                                   │  │
│  │      "error_message": str(error) if error else None                     │  │
│  │  }                                                                       │  │
│  │                                                                          │  │
│  │  self._execution_log.append(execution_record)                           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  STEP 2: Filter on Task Completion                                              │
│  ═════════════════════════════════                                              │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  When task ends successfully:                                            │  │
│  │                                                                          │  │
│  │  qualified_examples = [                                                  │  │
│  │      ex for ex in execution_log                                         │  │
│  │      if ex["was_successful"]                                            │  │
│  │      and ex["tool_result"] not in ["Error", "Failed"]                   │  │
│  │      and len(ex["tool_args"]) > 0  # Has meaningful input               │  │
│  │  ]                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  STEP 3: Save to Database (on "Save Template" click)                           │
│  ═══════════════════════════════════════════════════                           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  POST /api/tool-execution-history                                        │  │
│  │                                                                          │  │
│  │  {                                                                       │  │
│  │    "template_id": newly_created_template.id,                            │  │
│  │    "source_task_id": task_id,                                           │  │
│  │    "examples": qualified_examples                                       │  │
│  │  }                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Few-Shot Example Data Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TOOL_EXECUTION_HISTORY TABLE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Example Record for "Write UTC Timestamp to file" task:                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  id: 1                                                                 │ │
│  │  template_id: 42                                                       │ │
│  │  source_task_id: "1765208616768-4353"                                 │ │
│  │  subtask_content: "Write the current UTC timestamp to a file"         │ │
│  │  subtask_index: 1                                                      │ │
│  │  agent_name: "developer_agent"                                        │ │
│  │  tool_name: "execute_command"                                         │ │
│  │  tool_args: {                                                         │ │
│  │      "command": "date -u '+%Y-%m-%d %H:%M:%S UTC' > timestamp.txt"   │ │
│  │  }                                                                     │ │
│  │  tool_result_summary: "Command executed successfully. File created."  │ │
│  │  execution_order: 1                                                    │ │
│  │  was_successful: true                                                  │ │
│  │  execution_time_ms: 145                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  id: 2                                                                 │ │
│  │  template_id: 42                                                       │ │
│  │  source_task_id: "1765208616768-4353"                                 │ │
│  │  subtask_content: "Write the current UTC timestamp to a file"         │ │
│  │  subtask_index: 1                                                      │ │
│  │  agent_name: "developer_agent"                                        │ │
│  │  tool_name: "execute_command"                                         │ │
│  │  tool_args: {                                                         │ │
│  │      "command": "cat timestamp.txt"                                   │ │
│  │  }                                                                     │ │
│  │  tool_result_summary: "2025-12-09 15:30:45 UTC"                       │ │
│  │  execution_order: 2                                                    │ │
│  │  was_successful: true                                                  │ │
│  │  execution_time_ms: 52                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 LLM-Powered Memory Synthesis {#llm-memory-synthesis}

### Memory Synthesis Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LLM MEMORY SYNTHESIS PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TRIGGER: After successful task completion (before "Save Template")             │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                           │ │
│  │   ┌─────────────┐    ┌────────────────┐    ┌─────────────────────┐       │ │
│  │   │ Execution   │    │ LLM Memory     │    │ Structured Memory   │       │ │
│  │   │ Log (Raw)   │───▶│ Synthesizer    │───▶│ (Distilled)         │       │ │
│  │   │             │    │                │    │                     │       │ │
│  │   │ • Tool calls│    │ "Analyze this  │    │ • Strategies        │       │ │
│  │   │ • Results   │    │  execution and │    │ • Patterns          │       │ │
│  │   │ • Errors    │    │  extract key   │    │ • Pitfalls          │       │ │
│  │   │ • Timings   │    │  learnings..." │    │ • Best practices    │       │ │
│  │   └─────────────┘    └────────────────┘    └─────────────────────┘       │ │
│  │                                                      │                    │ │
│  │                                                      ▼                    │ │
│  │                                            ┌─────────────────┐            │ │
│  │                                            │  agent_memory   │            │ │
│  │                                            │  (DB)           │            │ │
│  │                                            └─────────────────┘            │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### LLM Memory Synthesizer Prompt

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY SYNTHESIS LLM PROMPT                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SYSTEM: You are a learning extraction agent. Analyze execution logs and        │
│  create structured memory that will help future executions be more efficient.   │
│                                                                                  │
│  INPUT:                                                                          │
│  ═══════                                                                         │
│  Task: {original_task_description}                                              │
│  Agent: {agent_name}                                                            │
│  Execution Log:                                                                  │
│  {formatted_execution_log}                                                       │
│                                                                                  │
│  Previous Memory (if exists):                                                    │
│  {existing_memory or "No previous memory"}                                      │
│                                                                                  │
│  INSTRUCTIONS:                                                                   │
│  ═════════════                                                                   │
│  Analyze this execution and extract learnings. Output JSON with:                │
│                                                                                  │
│  {                                                                               │
│    "successful_strategies": [                                                   │
│      "Strategy 1 that worked well",                                            │
│      "Strategy 2 that was effective"                                           │
│    ],                                                                            │
│    "optimal_tool_sequence": [                                                   │
│      {"tool": "tool_name", "purpose": "why this tool first"},                  │
│      {"tool": "tool_name", "purpose": "why this tool second"}                  │
│    ],                                                                            │
│    "pitfalls_to_avoid": [                                                       │
│      "Common mistake 1",                                                        │
│      "Edge case that caused issues"                                            │
│    ],                                                                            │
│    "efficiency_tips": [                                                         │
│      "Tip 1 for faster execution",                                             │
│      "Tip 2 for reducing steps"                                                │
│    ],                                                                            │
│    "key_insights": "Brief summary of main learnings"                           │
│  }                                                                               │
│                                                                                  │
│  OUTPUT:                                                                         │
│  ════════                                                                        │
│  {structured_memory_json}                                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Synthesis Implementation

```python
# backend/app/service/memory_synthesizer.py

class MemorySynthesizer:
    """
    Uses LLM to distill execution logs into structured memory.
    This memory helps agents learn and improve over time.
    """
    
    SYNTHESIS_PROMPT = """
You are analyzing an AI agent's task execution to extract learnings for future use.

## Task Information
- **Task Description:** {task_description}
- **Agent:** {agent_name}
- **Outcome:** {outcome}
- **Total Steps:** {total_steps}
- **Execution Time:** {execution_time}ms

## Execution Log
{execution_log}

## Previous Learnings (to build upon)
{previous_memory}

## Your Task
Analyze this execution and create a structured learning summary. Focus on:
1. What strategies led to success
2. The optimal sequence of tool usage
3. Any mistakes or inefficiencies to avoid
4. Tips for faster execution next time

Output your analysis as JSON:
```json
{{
    "successful_strategies": ["strategy1", "strategy2"],
    "optimal_tool_sequence": [
        {{"step": 1, "tool": "tool_name", "rationale": "why"}},
        {{"step": 2, "tool": "tool_name", "rationale": "why"}}
    ],
    "pitfalls_to_avoid": ["pitfall1", "pitfall2"],
    "efficiency_tips": ["tip1", "tip2"],
    "key_insights": "One paragraph summary of main learnings",
    "confidence_score": 0.85
}}
```
"""

    async def synthesize_memory(
        self,
        task_description: str,
        agent_name: str,
        execution_log: list[dict],
        previous_memory: dict | None,
        model: BaseModelBackend
    ) -> dict:
        """
        Use LLM to synthesize structured memory from execution log.
        """
        # Format execution log for LLM
        formatted_log = self._format_execution_log(execution_log)
        
        # Calculate metrics
        total_steps = len(execution_log)
        successful_steps = sum(1 for e in execution_log if e.get("was_successful"))
        total_time = sum(e.get("execution_time_ms", 0) for e in execution_log)
        outcome = "SUCCESS" if successful_steps == total_steps else f"PARTIAL ({successful_steps}/{total_steps})"
        
        # Build prompt
        prompt = self.SYNTHESIS_PROMPT.format(
            task_description=task_description,
            agent_name=agent_name,
            outcome=outcome,
            total_steps=total_steps,
            execution_time=total_time,
            execution_log=formatted_log,
            previous_memory=json.dumps(previous_memory, indent=2) if previous_memory else "None"
        )
        
        # Call LLM
        response = model.run([BaseMessage(role="user", content=prompt)])
        
        # Parse JSON response
        try:
            memory = json.loads(self._extract_json(response.content))
            memory["last_updated"] = datetime.now().isoformat()
            memory["execution_count"] = (previous_memory.get("execution_count", 0) + 1) if previous_memory else 1
            return memory
        except Exception as e:
            logger.error(f"Failed to parse memory synthesis: {e}")
            return self._create_fallback_memory(execution_log)
    
    def _format_execution_log(self, log: list[dict]) -> str:
        """Format execution log for LLM consumption."""
        formatted = []
        for i, entry in enumerate(log, 1):
            formatted.append(f"""
Step {i}:
  Tool: {entry.get('tool_name')}
  Args: {json.dumps(entry.get('tool_args', {}), indent=4)}
  Result: {entry.get('tool_result_summary', 'N/A')[:300]}
  Success: {entry.get('was_successful', False)}
  Time: {entry.get('execution_time_ms', 0)}ms
""")
        return "\n".join(formatted)
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        import re
        match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            return match.group(1)
        # Try to find raw JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON found in response")
```

### Synthesized Memory Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AGENT_MEMORY TABLE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Example Memory Record for Developer Agent:                                     │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │  id: 1                                                                     ││
│  │  user_id: 1                                                                ││
│  │  template_id: 42                                                           ││
│  │  agent_name: "developer_agent"                                             ││
│  │  success_count: 5                                                          ││
│  │  total_executions: 6                                                       ││
│  │                                                                            ││
│  │  memory_context: {                                                         ││
│  │    "successful_strategies": [                                              ││
│  │      "Use single command with output redirection for file operations",    ││
│  │      "Verify file creation with cat command immediately after write",     ││
│  │      "Use date -u flag for consistent UTC output across systems"          ││
│  │    ],                                                                      ││
│  │                                                                            ││
│  │    "optimal_tool_sequence": [                                              ││
│  │      {"step": 1, "tool": "execute_command",                                ││
│  │       "rationale": "Direct shell command is fastest for simple file ops"},││
│  │      {"step": 2, "tool": "execute_command",                                ││
│  │       "rationale": "Verify output with cat to confirm success"}           ││
│  │    ],                                                                      ││
│  │                                                                            ││
│  │    "pitfalls_to_avoid": [                                                  ││
│  │      "Don't use Python script for simple shell operations - slower",      ││
│  │      "Don't forget to specify timezone flag",                             ││
│  │      "Avoid multiple file writes when one command suffices"               ││
│  │    ],                                                                      ││
│  │                                                                            ││
│  │    "efficiency_tips": [                                                    ││
│  │      "Combine timestamp generation and file write in one command",        ││
│  │      "Skip verification step if not critical to task",                    ││
│  │      "Use >> for append, > for overwrite based on requirement"            ││
│  │    ],                                                                      ││
│  │                                                                            ││
│  │    "key_insights": "For timestamp file tasks, shell commands are          ││
│  │     optimal. Single date command with redirection completes task          ││
│  │     in one step. Always specify UTC flag for consistency.",               ││
│  │                                                                            ││
│  │    "confidence_score": 0.92,                                               ││
│  │    "execution_count": 6,                                                   ││
│  │    "last_updated": "2025-12-09T15:45:00Z"                                 ││
│  │  }                                                                         ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚦 Template Mode Gate {#template-mode-gate}

### Gate Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TEMPLATE MODE GATE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ALL LEARNING FEATURES ARE GATED BY: options.template_mode == True              │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                            ││
│  │                         ┌─────────────────┐                                ││
│  │                         │ template_mode?  │                                ││
│  │                         └────────┬────────┘                                ││
│  │                                  │                                         ││
│  │                    ┌─────────────┴─────────────┐                          ││
│  │                    │                           │                          ││
│  │                    ▼                           ▼                          ││
│  │           ┌────────────────┐         ┌────────────────┐                   ││
│  │           │   TRUE         │         │   FALSE        │                   ││
│  │           │                │         │                │                   ││
│  │           │ • Load Memory  │         │ • Normal       │                   ││
│  │           │ • Load Few-Shot│         │   Execution    │                   ││
│  │           │ • Enhance Agent│         │ • No Learning  │                   ││
│  │           │ • Capture Logs │         │   Features     │                   ││
│  │           │ • Save Memory  │         │                │                   ││
│  │           └────────────────┘         └────────────────┘                   ││
│  │                                                                            ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Gate Implementation Points

```python
# backend/app/service/chat_service.py

async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - All learning features check this flag
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and options.template_subtasks:
        logger.info(f"[TEMPLATE MODE] Learning features ENABLED for task {options.task_id}")
        
        # GATE CHECK 1: Load Agent Memory
        agent_memories = {}
        if options.template_id:
            agent_memories = await load_all_agent_memories(options.template_id)
            logger.info(f"[TEMPLATE MODE] Loaded memories for {len(agent_memories)} agents")
        
        # GATE CHECK 2: Load Few-Shot Examples
        few_shot_examples = {}
        if options.template_id:
            few_shot_examples = await load_few_shot_examples(options.template_id)
            logger.info(f"[TEMPLATE MODE] Loaded {sum(len(v) for v in few_shot_examples.values())} examples")
        
        # Pass to workforce construction with learning context
        (workforce, mcp) = await construct_workforce(
            options,
            agent_memories=agent_memories,      # NEW: Passed only in template mode
            few_shot_examples=few_shot_examples  # NEW: Passed only in template mode
        )
        
    else:
        logger.info(f"[NORMAL MODE] Learning features DISABLED for task {options.task_id}")
        # Normal execution - no learning features
        (workforce, mcp) = await construct_workforce(options)
```

```python
# backend/app/utils/agent.py

def agent_model(
    agent_name: str,
    system_message: str | BaseMessage,
    options: Chat,
    tools: list[FunctionTool | Callable] | None = None,
    # ... other params ...
    
    # NEW: Learning parameters (only used when provided)
    memory_context: dict | None = None,
    few_shot_examples: list[dict] | None = None,
):
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - Enhance only if in template mode
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and (memory_context or few_shot_examples):
        # GATE PASSED: Enhance agent with learning
        enhanced_message = build_enhanced_system_message(
            system_message,
            memory_context,
            few_shot_examples,
            agent_name
        )
        system_message = enhanced_message
        logger.info(f"[GATE PASSED] Agent {agent_name} enhanced with learning context")
    else:
        # GATE BLOCKED: Use standard agent
        logger.debug(f"[GATE BLOCKED] Agent {agent_name} using standard configuration")
    
    # ... rest of agent creation ...
```

---

## 🗄️ Data Models {#data-models}

### Complete Database Schema

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE SCHEMA                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  EXISTING TABLES                              NEW TABLES                        │
│  ═══════════════                              ═══════════                        │
│                                                                                  │
│  ┌──────────────────┐                    ┌──────────────────────┐              │
│  │ workflow_template│◄───────────────────│    agent_memory      │              │
│  │                  │    template_id     │                      │              │
│  │ • id (PK)        │                    │ • id (PK)            │              │
│  │ • user_id        │                    │ • user_id            │              │
│  │ • source_task_id │                    │ • template_id (FK)   │              │
│  │ • name           │                    │ • agent_name         │              │
│  │ • subtasks       │                    │ • memory_context     │              │
│  │ • agents         │                    │ • success_count      │              │
│  │ • ...            │                    │ • total_executions   │              │
│  └──────────────────┘                    └──────────────────────┘              │
│           │                                                                     │
│           │                                                                     │
│           │                              ┌──────────────────────┐              │
│           └─────────────────────────────►│ tool_execution_      │              │
│                      template_id         │ history              │              │
│                                          │                      │              │
│                                          │ • id (PK)            │              │
│                                          │ • user_id            │              │
│                                          │ • template_id (FK)   │              │
│                                          │ • source_task_id     │              │
│                                          │ • subtask_content    │              │
│                                          │ • agent_name         │              │
│                                          │ • tool_name          │              │
│                                          │ • tool_args          │              │
│                                          │ • tool_result_summary│              │
│                                          │ • was_successful     │              │
│                                          │ • execution_time_ms  │              │
│                                          └──────────────────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Component Interactions {#component-interactions}

### Complete Flow: First Execution → Save Template → Re-Execute

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE LEARNING LIFECYCLE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: FIRST EXECUTION (Normal Mode)                                        │
│  ═══════════════════════════════════════                                        │
│                                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐              │
│  │  User   │───▶│  Task    │───▶│  Agent    │───▶│ Task        │              │
│  │  Input  │    │  Created │    │  Executes │    │ Completes   │              │
│  └─────────┘    └──────────┘    └───────────┘    └──────┬──────┘              │
│                                       │                  │                      │
│                                       ▼                  │                      │
│                            [Execution logged in         │                      │
│                             ChatStep table]             │                      │
│                                                         │                      │
│  PHASE 2: SAVE AS TEMPLATE                              │                      │
│  ═════════════════════════                              ▼                      │
│                                              ┌─────────────────┐               │
│                                              │ User clicks     │               │
│                                              │ "Save Template" │               │
│                                              └────────┬────────┘               │
│                                                       │                        │
│         ┌─────────────────────────────────────────────┼───────────────────┐   │
│         │                                             │                   │   │
│         ▼                                             ▼                   ▼   │
│  ┌──────────────┐                          ┌──────────────┐    ┌──────────┐  │
│  │ Create       │                          │ Extract Few- │    │ Run LLM  │  │
│  │ workflow_    │                          │ Shot Examples│    │ Memory   │  │
│  │ template     │                          │ from ChatStep│    │ Synthesis│  │
│  └──────────────┘                          └──────┬───────┘    └────┬─────┘  │
│                                                    │                 │        │
│                                                    ▼                 ▼        │
│                                           ┌──────────────┐  ┌──────────────┐ │
│                                           │ tool_exec_   │  │ agent_memory │ │
│                                           │ history (DB) │  │ (DB)         │ │
│                                           └──────────────┘  └──────────────┘ │
│                                                                               │
│  PHASE 3: RE-EXECUTION (Template Mode)                                       │
│  ═════════════════════════════════════                                       │
│                                                                               │
│  ┌─────────────┐    ┌────────────────────────────────────────────────────┐  │
│  │ User clicks │    │          TEMPLATE MODE GATE (OPEN)                 │  │
│  │ "Re-execute"│───▶│                                                    │  │
│  └─────────────┘    │  ┌──────────────┐         ┌──────────────────┐    │  │
│                     │  │ Load Agent   │         │ Load Few-Shot    │    │  │
│                     │  │ Memory       │         │ Examples         │    │  │
│                     │  │ from DB      │         │ from DB          │    │  │
│                     │  └──────┬───────┘         └────────┬─────────┘    │  │
│                     │         │                          │              │  │
│                     │         └──────────┬───────────────┘              │  │
│                     │                    │                              │  │
│                     │                    ▼                              │  │
│                     │         ┌──────────────────────┐                  │  │
│                     │         │ Build Enhanced       │                  │  │
│                     │         │ System Message       │                  │  │
│                     │         │                      │                  │  │
│                     │         │ • Original prompt    │                  │  │
│                     │         │ • + Memory context   │                  │  │
│                     │         │ • + Few-shot examples│                  │  │
│                     │         │ • + Efficiency tips  │                  │  │
│                     │         └──────────┬───────────┘                  │  │
│                     │                    │                              │  │
│                     └────────────────────┼──────────────────────────────┘  │
│                                          │                                 │
│                                          ▼                                 │
│                               ┌──────────────────────┐                    │
│                               │   ENHANCED AGENT     │                    │
│                               │                      │                    │
│                               │  • Fewer LLM calls   │                    │
│                               │  • Follows patterns  │                    │
│                               │  • Avoids pitfalls   │                    │
│                               │  • Faster completion │                    │
│                               └──────────────────────┘                    │
│                                                                            │
│  PHASE 4: UPDATE MEMORY (After Re-execution)                              │
│  ═══════════════════════════════════════════                              │
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │ Re-execution    │───▶│ Capture New     │───▶│ Merge with Existing │   │
│  │ Completes       │    │ Execution Log   │    │ Memory via LLM      │   │
│  └─────────────────┘    └─────────────────┘    └──────────┬──────────┘   │
│                                                           │               │
│                                                           ▼               │
│                                                ┌─────────────────────┐    │
│                                                │ Update agent_memory │    │
│                                                │ (Accumulated        │    │
│                                                │  learnings)         │    │
│                                                └─────────────────────┘    │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Summary: Files to Create/Modify

| Component | File Path | Type |
|-----------|-----------|------|
| Memory Model | `server/app/model/chat/agent_memory.py` | **NEW** |
| History Model | `server/app/model/chat/tool_execution_history.py` | **NEW** |
| Memory Controller | `server/app/controller/chat/memory_controller.py` | **NEW** |
| Migration | `server/alembic/versions/2025_12_09_add_agent_memory.py` | **NEW** |
| Memory Synthesizer | `backend/app/service/memory_synthesizer.py` | **NEW** |
| Memory Service | `backend/app/service/agent_memory_service.py` | **NEW** |
| Agent Enhancement | `backend/app/utils/agent.py` | **MODIFY** |
| Chat Service | `backend/app/service/chat_service.py` | **MODIFY** |
| Template Controller | `server/app/controller/chat/template_controller.py` | **MODIFY** |

---

## 🚀 Ready to Implement?

You're in **ask mode**. To implement this complete solution:

1. **Switch to agent mode**
2. I'll create all the files in the correct order:
   - Database models
   - Migration
   - API endpoints
   - Memory synthesizer
   - Agent modifications with template mode gate

Would you like to proceed?

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AI AGENT PLATFORM                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────┐     ┌──────────────────────┐     ┌────────────────┐  │
│  │   FRONTEND (React)   │     │   LOCAL BACKEND      │     │  SERVER (API)  │  │
│  │                      │     │   (Python/FastAPI)   │     │  (Docker)      │  │
│  │  • Re-execute UI     │────▶│  • Agent Execution   │────▶│  • Memory DB   │  │
│  │  • Template Toggle   │     │  • Memory Loading    │     │  • History DB  │  │
│  │                      │     │  • Learning Capture  │     │  • Templates   │  │
│  └──────────────────────┘     └──────────────────────┘     └────────────────┘  │
│                                        │                            │           │
│                                        │                            │           │
│                                        ▼                            ▼           │
│                               ┌─────────────────────────────────────────┐       │
│                               │         LEARNING SYSTEM                 │       │
│                               │  ┌─────────────┐  ┌─────────────────┐  │       │
│                               │  │  Few-Shot   │  │  LLM Memory     │  │       │
│                               │  │  Collector  │  │  Synthesizer    │  │       │
│                               │  └─────────────┘  └─────────────────┘  │       │
│                               └─────────────────────────────────────────┘       │
│                                                                                  │
│  ═══════════════════════════════════════════════════════════════════════════   │
│  ONLY ACTIVE WHEN: options.template_mode == True                                │
│  ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEW-SHOT EXAMPLE CREATION TRIGGER                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TRIGGER: Task completes successfully AND user saves as template           │
│                                                                              │
│   ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐           │
│   │ Task        │      │ User clicks  │      │ Extract & Save  │           │
│   │ Completes   │─────▶│ "Save        │─────▶│ Few-Shot        │           │
│   │ Successfully│      │ Template"    │      │ Examples        │           │
│   └─────────────┘      └──────────────┘      └─────────────────┘           │
│                                                      │                      │
│                                                      ▼                      │
│                                            ┌─────────────────┐              │
│                                            │ tool_execution_ │              │
│                                            │ history (DB)    │              │
│                                            └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     FEW-SHOT EXAMPLE EXTRACTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 1: Capture During Execution                                               │
│  ════════════════════════════════                                               │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  For each tool call in agent._execute_tool():                            │  │
│  │                                                                          │  │
│  │  execution_record = {                                                    │  │
│  │      "subtask_id": current_subtask.id,                                  │  │
│  │      "subtask_content": current_subtask.content,                        │  │
│  │      "agent_name": self.agent_name,                                     │  │
│  │      "tool_name": tool_call_request.tool_name,                          │  │
│  │      "tool_args": tool_call_request.args,          # Full arguments     │  │
│  │      "tool_result": result[:1000],                 # Truncated result   │  │
│  │      "execution_order": self._call_counter,                             │  │
│  │      "timestamp": datetime.now(),                                       │  │
│  │      "was_successful": error is None,                                   │  │
│  │      "error_message": str(error) if error else None                     │  │
│  │  }                                                                       │  │
│  │                                                                          │  │
│  │  self._execution_log.append(execution_record)                           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  STEP 2: Filter on Task Completion                                              │
│  ═════════════════════════════════                                              │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  When task ends successfully:                                            │  │
│  │                                                                          │  │
│  │  qualified_examples = [                                                  │  │
│  │      ex for ex in execution_log                                         │  │
│  │      if ex["was_successful"]                                            │  │
│  │      and ex["tool_result"] not in ["Error", "Failed"]                   │  │
│  │      and len(ex["tool_args"]) > 0  # Has meaningful input               │  │
│  │  ]                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  STEP 3: Save to Database (on "Save Template" click)                           │
│  ═══════════════════════════════════════════════════                           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  POST /api/tool-execution-history                                        │  │
│  │                                                                          │  │
│  │  {                                                                       │  │
│  │    "template_id": newly_created_template.id,                            │  │
│  │    "source_task_id": task_id,                                           │  │
│  │    "examples": qualified_examples                                       │  │
│  │  }                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TOOL_EXECUTION_HISTORY TABLE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Example Record for "Write UTC Timestamp to file" task:                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  id: 1                                                                 │ │
│  │  template_id: 42                                                       │ │
│  │  source_task_id: "1765208616768-4353"                                 │ │
│  │  subtask_content: "Write the current UTC timestamp to a file"         │ │
│  │  subtask_index: 1                                                      │ │
│  │  agent_name: "developer_agent"                                        │ │
│  │  tool_name: "execute_command"                                         │ │
│  │  tool_args: {                                                         │ │
│  │      "command": "date -u '+%Y-%m-%d %H:%M:%S UTC' > timestamp.txt"   │ │
│  │  }                                                                     │ │
│  │  tool_result_summary: "Command executed successfully. File created."  │ │
│  │  execution_order: 1                                                    │ │
│  │  was_successful: true                                                  │ │
│  │  execution_time_ms: 145                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  id: 2                                                                 │ │
│  │  template_id: 42                                                       │ │
│  │  source_task_id: "1765208616768-4353"                                 │ │
│  │  subtask_content: "Write the current UTC timestamp to a file"         │ │
│  │  subtask_index: 1                                                      │ │
│  │  agent_name: "developer_agent"                                        │ │
│  │  tool_name: "execute_command"                                         │ │
│  │  tool_args: {                                                         │ │
│  │      "command": "cat timestamp.txt"                                   │ │
│  │  }                                                                     │ │
│  │  tool_result_summary: "2025-12-09 15:30:45 UTC"                       │ │
│  │  execution_order: 2                                                    │ │
│  │  was_successful: true                                                  │ │
│  │  execution_time_ms: 52                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LLM MEMORY SYNTHESIS PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TRIGGER: After successful task completion (before "Save Template")             │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                           │ │
│  │   ┌─────────────┐    ┌────────────────┐    ┌─────────────────────┐       │ │
│  │   │ Execution   │    │ LLM Memory     │    │ Structured Memory   │       │ │
│  │   │ Log (Raw)   │───▶│ Synthesizer    │───▶│ (Distilled)         │       │ │
│  │   │             │    │                │    │                     │       │ │
│  │   │ • Tool calls│    │ "Analyze this  │    │ • Strategies        │       │ │
│  │   │ • Results   │    │  execution and │    │ • Patterns          │       │ │
│  │   │ • Errors    │    │  extract key   │    │ • Pitfalls          │       │ │
│  │   │ • Timings   │    │  learnings..." │    │ • Best practices    │       │ │
│  │   └─────────────┘    └────────────────┘    └─────────────────────┘       │ │
│  │                                                      │                    │ │
│  │                                                      ▼                    │ │
│  │                                            ┌─────────────────┐            │ │
│  │                                            │  agent_memory   │            │ │
│  │                                            │  (DB)           │            │ │
│  │                                            └─────────────────┘            │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY SYNTHESIS LLM PROMPT                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SYSTEM: You are a learning extraction agent. Analyze execution logs and        │
│  create structured memory that will help future executions be more efficient.   │
│                                                                                  │
│  INPUT:                                                                          │
│  ═══════                                                                         │
│  Task: {original_task_description}                                              │
│  Agent: {agent_name}                                                            │
│  Execution Log:                                                                  │
│  {formatted_execution_log}                                                       │
│                                                                                  │
│  Previous Memory (if exists):                                                    │
│  {existing_memory or "No previous memory"}                                      │
│                                                                                  │
│  INSTRUCTIONS:                                                                   │
│  ═════════════                                                                   │
│  Analyze this execution and extract learnings. Output JSON with:                │
│                                                                                  │
│  {                                                                               │
│    "successful_strategies": [                                                   │
│      "Strategy 1 that worked well",                                            │
│      "Strategy 2 that was effective"                                           │
│    ],                                                                            │
│    "optimal_tool_sequence": [                                                   │
│      {"tool": "tool_name", "purpose": "why this tool first"},                  │
│      {"tool": "tool_name", "purpose": "why this tool second"}                  │
│    ],                                                                            │
│    "pitfalls_to_avoid": [                                                       │
│      "Common mistake 1",                                                        │
│      "Edge case that caused issues"                                            │
│    ],                                                                            │
│    "efficiency_tips": [                                                         │
│      "Tip 1 for faster execution",                                             │
│      "Tip 2 for reducing steps"                                                │
│    ],                                                                            │
│    "key_insights": "Brief summary of main learnings"                           │
│  }                                                                               │
│                                                                                  │
│  OUTPUT:                                                                         │
│  ════════                                                                        │
│  {structured_memory_json}                                                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

```python
# backend/app/service/memory_synthesizer.py

class MemorySynthesizer:
    """
    Uses LLM to distill execution logs into structured memory.
    This memory helps agents learn and improve over time.
    """
    
    SYNTHESIS_PROMPT = """
You are analyzing an AI agent's task execution to extract learnings for future use.

## Task Information
- **Task Description:** {task_description}
- **Agent:** {agent_name}
- **Outcome:** {outcome}
- **Total Steps:** {total_steps}
- **Execution Time:** {execution_time}ms

## Execution Log
{execution_log}

## Previous Learnings (to build upon)
{previous_memory}

## Your Task
Analyze this execution and create a structured learning summary. Focus on:
1. What strategies led to success
2. The optimal sequence of tool usage
3. Any mistakes or inefficiencies to avoid
4. Tips for faster execution next time

Output your analysis as JSON:
{{
    "successful_strategies": ["strategy1", "strategy2"],
    "optimal_tool_sequence": [
        {{"step": 1, "tool": "tool_name", "rationale": "why"}},
        {{"step": 2, "tool": "tool_name", "rationale": "why"}}
    ],
    "pitfalls_to_avoid": ["pitfall1", "pitfall2"],
    "efficiency_tips": ["tip1", "tip2"],
    "key_insights": "One paragraph summary of main learnings",
    "confidence_score": 0.85
}}"""

    async def synthesize_memory(
        self,
        task_description: str,
        agent_name: str,
        execution_log: list[dict],
        previous_memory: dict | None,
        model: BaseModelBackend
    ) -> dict:
        """
        Use LLM to synthesize structured memory from execution log.
        """
        # Format execution log for LLM
        formatted_log = self._format_execution_log(execution_log)
        
        # Calculate metrics
        total_steps = len(execution_log)
        successful_steps = sum(1 for e in execution_log if e.get("was_successful"))
        total_time = sum(e.get("execution_time_ms", 0) for e in execution_log)
        outcome = "SUCCESS" if successful_steps == total_steps else f"PARTIAL ({successful_steps}/{total_steps})"
        
        # Build prompt
        prompt = self.SYNTHESIS_PROMPT.format(
            task_description=task_description,
            agent_name=agent_name,
            outcome=outcome,
            total_steps=total_steps,
            execution_time=total_time,
            execution_log=formatted_log,
            previous_memory=json.dumps(previous_memory, indent=2) if previous_memory else "None"
        )
        
        # Call LLM
        response = model.run([BaseMessage(role="user", content=prompt)])
        
        # Parse JSON response
        try:
            memory = json.loads(self._extract_json(response.content))
            memory["last_updated"] = datetime.now().isoformat()
            memory["execution_count"] = (previous_memory.get("execution_count", 0) + 1) if previous_memory else 1
            return memory
        except Exception as e:
            logger.error(f"Failed to parse memory synthesis: {e}")
            return self._create_fallback_memory(execution_log)
    
    def _format_execution_log(self, log: list[dict]) -> str:
        """Format execution log for LLM consumption."""
        formatted = []
        for i, entry in enumerate(log, 1):
            formatted.append(f"""
Step {i}:
  Tool: {entry.get('tool_name')}
  Args: {json.dumps(entry.get('tool_args', {}), indent=4)}
  Result: {entry.get('tool_result_summary', 'N/A')[:300]}
  Success: {entry.get('was_successful', False)}
  Time: {entry.get('execution_time_ms', 0)}ms
""")
        return "\n".join(formatted)
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        import re
        match = re.search(r'\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            return match.group(1)
        # Try to find raw JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON found in response")### Synthesized Memory Structure

```

```plaintext

---

## 🚦 Template Mode Gate {#template-mode-gate}

### Gate Architecture

```

```plaintext

---

## 🚦 Template Mode Gate {#template-mode-gate}

### Gate Architecture

```

```plaintext

### Gate Implementation Points

# backend/app/service/chat_service.py

async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - All learning features check this flag
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and options.template_subtasks:
        logger.info(f"[TEMPLATE MODE] Learning features ENABLED for task {options.task_id}")
        
        # GATE CHECK 1: Load Agent Memory
        agent_memories = {}
        if options.template_id:
            agent_memories = await load_all_agent_memories(options.template_id)
            logger.info(f"[TEMPLATE MODE] Loaded memories for {len(agent_memories)} agents")
        
        # GATE CHECK 2: Load Few-Shot Examples
        few_shot_examples = {}
        if options.template_id:
            few_shot_examples = await load_few_shot_examples(options.template_id)
            logger.info(f"[TEMPLATE MODE] Loaded {sum(len(v) for v in few_shot_examples.values())} examples")
        
        # Pass to workforce construction with learning context
        (workforce, mcp) = await construct_workforce(
            options,
            agent_memories=agent_memories,      # NEW: Passed only in template mode
            few_shot_examples=few_shot_examples  # NEW: Passed only in template mode
        )
        
    else:
        logger.info(f"[NORMAL MODE] Learning features DISABLED for task {options.task_id}")
        # Normal execution - no learning features
        (workforce, mcp) = await construct_workforce(options)
# backend/app/utils/agent.py

def agent_model(
    agent_name: str,
    system_message: str | BaseMessage,
    options: Chat,
    tools: list[FunctionTool | Callable] | None = None,
    # ... other params ...
    
    # NEW: Learning parameters (only used when provided)
    memory_context: dict | None = None,
    few_shot_examples: list[dict] | None = None,
):
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - Enhance only if in template mode
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and (memory_context or few_shot_examples):
        # GATE PASSED: Enhance agent with learning
        enhanced_message = build_enhanced_system_message(
            system_message,
            memory_context,
            few_shot_examples,
            agent_name
        )
        system_message = enhanced_message
        logger.info(f"[GATE PASSED] Agent {agent_name} enhanced with learning context")
    else:
        # GATE BLOCKED: Use standard agent
        logger.debug(f"[GATE BLOCKED] Agent {agent_name} using standard configuration")
    
    # ... rest of agent creation ...---

## 🗄️ Data Models {#data-models}

### Complete Database Schema

```

```plaintext

### Gate Implementation Points

# backend/app/service/chat_service.py

async def step_solve(options: Chat, request: Request, task_lock: TaskLock):
    # ... existing code ...
    
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - All learning features check this flag
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and options.template_subtasks:
        logger.info(f"[TEMPLATE MODE] Learning features ENABLED for task {options.task_id}")
        
        # GATE CHECK 1: Load Agent Memory
        agent_memories = {}
        if options.template_id:
            agent_memories = await load_all_agent_memories(options.template_id)
            logger.info(f"[TEMPLATE MODE] Loaded memories for {len(agent_memories)} agents")
        
        # GATE CHECK 2: Load Few-Shot Examples
        few_shot_examples = {}
        if options.template_id:
            few_shot_examples = await load_few_shot_examples(options.template_id)
            logger.info(f"[TEMPLATE MODE] Loaded {sum(len(v) for v in few_shot_examples.values())} examples")
        
        # Pass to workforce construction with learning context
        (workforce, mcp) = await construct_workforce(
            options,
            agent_memories=agent_memories,      # NEW: Passed only in template mode
            few_shot_examples=few_shot_examples  # NEW: Passed only in template mode
        )
        
    else:
        logger.info(f"[NORMAL MODE] Learning features DISABLED for task {options.task_id}")
        # Normal execution - no learning features
        (workforce, mcp) = await construct_workforce(options)
# backend/app/utils/agent.py

def agent_model(
    agent_name: str,
    system_message: str | BaseMessage,
    options: Chat,
    tools: list[FunctionTool | Callable] | None = None,
    # ... other params ...
    
    # NEW: Learning parameters (only used when provided)
    memory_context: dict | None = None,
    few_shot_examples: list[dict] | None = None,
):
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - Enhance only if in template mode
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and (memory_context or few_shot_examples):
        # GATE PASSED: Enhance agent with learning
        enhanced_message = build_enhanced_system_message(
            system_message,
            memory_context,
            few_shot_examples,
            agent_name
        )
        system_message = enhanced_message
        logger.info(f"[GATE PASSED] Agent {agent_name} enhanced with learning context")
    else:
        # GATE BLOCKED: Use standard agent
        logger.debug(f"[GATE BLOCKED] Agent {agent_name} using standard configuration")
    
    # ... rest of agent creation ...---

## 🗄️ Data Models {#data-models}

### Complete Database Schema

```

```plaintext

---

## 🔄 Component Interactions {#component-interactions}

### Complete Flow: First Execution → Save Template → Re-Execute

```

```plaintext

---

## 🔄 Component Interactions {#component-interactions}

### Complete Flow: First Execution → Save Template → Re-Execute

```

```plaintext

---

## 📊 Summary: Files to Create/Modify

| Component | File Path | Type |
|-----------|-----------|------|
| Memory Model | `server/app/model/chat/agent_memory.py` | **NEW** |
| History Model | `server/app/model/chat/tool_execution_history.py` | **NEW** |
| Memory Controller | `server/app/controller/chat/memory_controller.py` | **NEW** |
| Migration | `server/alembic/versions/2025_12_09_add_agent_memory.py` | **NEW** |
| Memory Synthesizer | `backend/app/service/memory_synthesizer.py` | **NEW** |
| Memory Service | `backend/app/service/agent_memory_service.py` | **NEW** |
| Agent Enhancement | `backend/app/utils/agent.py` | **MODIFY** |
| Chat Service | `backend/app/service/chat_service.py` | **MODIFY** |
| Template Controller | `server/app/controller/chat/template_controller.py` | **MODIFY** |

---

## 🚀 Ready to Implement?

You're in **ask mode**. To implement this complete solution:

1. **Switch to agent mode**
2. I'll create all the files in the correct order:
   - Database models
   - Migration
   - API endpoints
   - Memory synthesizer
   - Agent modifications with template mode gate

Would you like to proceed?
```

```plaintext

---

## 📊 Summary: Files to Create/Modify

| Component | File Path | Type |
|-----------|-----------|------|
| Memory Model | `server/app/model/chat/agent_memory.py` | **NEW** |
| History Model | `server/app/model/chat/tool_execution_history.py` | **NEW** |
| Memory Controller | `server/app/controller/chat/memory_controller.py` | **NEW** |
| Migration | `server/alembic/versions/2025_12_09_add_agent_memory.py` | **NEW** |
| Memory Synthesizer | `backend/app/service/memory_synthesizer.py` | **NEW** |
| Memory Service | `backend/app/service/agent_memory_service.py` | **NEW** |
| Agent Enhancement | `backend/app/utils/agent.py` | **MODIFY** |
| Chat Service | `backend/app/service/chat_service.py` | **MODIFY** |
| Template Controller | `server/app/controller/chat/template_controller.py` | **MODIFY** |

---

## 🚀 Ready to Implement?

You're in **ask mode**. To implement this complete solution:

1. **Switch to agent mode**
2. I'll create all the files in the correct order:
   - Database models
   - Migration
   - API endpoints
   - Memory synthesizer
   - Agent modifications with template mode gate

Would you like to proceed?
```

```python
# backend/app/utils/agent.py

def agent_model(
    agent_name: str,
    system_message: str | BaseMessage,
    options: Chat,
    tools: list[FunctionTool | Callable] | None = None,
    # ... other params ...
    
    # NEW: Learning parameters (only used when provided)
    memory_context: dict | None = None,
    few_shot_examples: list[dict] | None = None,
):
    # ═══════════════════════════════════════════════════════════
    # TEMPLATE MODE GATE - Enhance only if in template mode
    # ═══════════════════════════════════════════════════════════
    
    if options.template_mode and (memory_context or few_shot_examples):
        # GATE PASSED: Enhance agent with learning
        enhanced_message = build_enhanced_system_message(
            system_message,
            memory_context,
            few_shot_examples,
            agent_name
        )
        system_message = enhanced_message
        logger.info(f"[GATE PASSED] Agent {agent_name} enhanced with learning context")
    else:
        # GATE BLOCKED: Use standard agent
        logger.debug(f"[GATE BLOCKED] Agent {agent_name} using standard configuration")
    
    # ... rest of agent creation ...
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE SCHEMA                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  EXISTING TABLES                              NEW TABLES                        │
│  ═══════════════                              ═══════════                        │
│                                                                                  │
│  ┌──────────────────┐                    ┌──────────────────────┐              │
│  │ workflow_template│◄───────────────────│    agent_memory      │              │
│  │                  │    template_id     │                      │              │
│  │ • id (PK)        │                    │ • id (PK)            │              │
│  │ • user_id        │                    │ • user_id            │              │
│  │ • source_task_id │                    │ • template_id (FK)   │              │
│  │ • name           │                    │ • agent_name         │              │
│  │ • subtasks       │                    │ • memory_context     │              │
│  │ • agents         │                    │ • success_count      │              │
│  │ • ...            │                    │ • total_executions   │              │
│  └──────────────────┘                    └──────────────────────┘              │
│           │                                                                     │
│           │                                                                     │
│           │                              ┌──────────────────────┐              │
│           └─────────────────────────────►│ tool_execution_      │              │
│                      template_id         │ history              │              │
│                                          │                      │              │
│                                          │ • id (PK)            │              │
│                                          │ • user_id            │              │
│                                          │ • template_id (FK)   │              │
│                                          │ • source_task_id     │              │
│                                          │ • subtask_content    │              │
│                                          │ • agent_name         │              │
│                                          │ • tool_name          │              │
│                                          │ • tool_args          │              │
│                                          │ • tool_result_summary│              │
│                                          │ • was_successful     │              │
│                                          │ • execution_time_ms  │              │
│                                          └──────────────────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

```plaintext
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE LEARNING LIFECYCLE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PHASE 1: FIRST EXECUTION (Normal Mode)                                        │
│  ═══════════════════════════════════════                                        │
│                                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐              │
│  │  User   │───▶│  Task    │───▶│  Agent    │───▶│ Task        │              │
│  │  Input  │    │  Created │    │  Executes │    │ Completes   │              │
│  └─────────┘    └──────────┘    └───────────┘    └──────┬──────┘              │
│                                       │                  │                      │
│                                       ▼                  │                      │
│                            [Execution logged in         │                      │
│                             ChatStep table]             │                      │
│                                                         │                      │
│  PHASE 2: SAVE AS TEMPLATE                              │                      │
│  ═════════════════════════                              ▼                      │
│                                              ┌─────────────────┐               │
│                                              │ User clicks     │               │
│                                              │ "Save Template" │               │
│                                              └────────┬────────┘               │
│                                                       │                        │
│         ┌─────────────────────────────────────────────┼───────────────────┐   │
│         │                                             │                   │   │
│         ▼                                             ▼                   ▼   │
│  ┌──────────────┐                          ┌──────────────┐    ┌──────────┐  │
│  │ Create       │                          │ Extract Few- │    │ Run LLM  │  │
│  │ workflow_    │                          │ Shot Examples│    │ Memory   │  │
│  │ template     │                          │ from ChatStep│    │ Synthesis│  │
│  └──────────────┘                          └──────┬───────┘    └────┬─────┘  │
│                                                    │                 │        │
│                                                    ▼                 ▼        │
│                                           ┌──────────────┐  ┌──────────────┐ │
│                                           │ tool_exec_   │  │ agent_memory │ │
│                                           │ history (DB) │  │ (DB)         │ │
│                                           └──────────────┘  └──────────────┘ │
│                                                                               │
│  PHASE 3: RE-EXECUTION (Template Mode)                                       │
│  ═════════════════════════════════════                                       │
│                                                                               │
│  ┌─────────────┐    ┌────────────────────────────────────────────────────┐  │
│  │ User clicks │    │          TEMPLATE MODE GATE (OPEN)                 │  │
│  │ "Re-execute"│───▶│                                                    │  │
│  └─────────────┘    │  ┌──────────────┐         ┌──────────────────┐    │  │
│                     │  │ Load Agent   │         │ Load Few-Shot    │    │  │
│                     │  │ Memory       │         │ Examples         │    │  │
│                     │  │ from DB      │         │ from DB          │    │  │
│                     │  └──────┬───────┘         └────────┬─────────┘    │  │
│                     │         │                          │              │  │
│                     │         └──────────┬───────────────┘              │  │
│                     │                    │                              │  │
│                     │                    ▼                              │  │
│                     │         ┌──────────────────────┐                  │  │
│                     │         │ Build Enhanced       │                  │  │
│                     │         │ System Message       │                  │  │
│                     │         │                      │                  │  │
│                     │         │ • Original prompt    │                  │  │
│                     │         │ • + Memory context   │                  │  │
│                     │         │ • + Few-shot examples│                  │  │
│                     │         │ • + Efficiency tips  │                  │  │
│                     │         └──────────┬───────────┘                  │  │
│                     │                    │                              │  │
│                     └────────────────────┼──────────────────────────────┘  │
│                                          │                                 │
│                                          ▼                                 │
│                               ┌──────────────────────┐                    │
│                               │   ENHANCED AGENT     │                    │
│                               │                      │                    │
│                               │  • Fewer LLM calls   │                    │
│                               │  • Follows patterns  │                    │
│                               │  • Avoids pitfalls   │                    │
│                               │  • Faster completion │                    │
│                               └──────────────────────┘                    │
│                                                                            │
│  PHASE 4: UPDATE MEMORY (After Re-execution)                              │
│  ═══════════════════════════════════════════                              │
│                                                                            │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │ Re-execution    │───▶│ Capture New     │───▶│ Merge with Existing │   │
│  │ Completes       │    │ Execution Log   │    │ Memory via LLM      │   │
│  └─────────────────┘    └─────────────────┘    └──────────┬──────────┘   │
│                                                           │               │
│                                                           ▼               │
│                                                ┌─────────────────────┐    │
│                                                │ Update agent_memory │    │
│                                                │ (Accumulated        │    │
│                                                │  learnings)         │    │
│                                                └─────────────────────┘    │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

