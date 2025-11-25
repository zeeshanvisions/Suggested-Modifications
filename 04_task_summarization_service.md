# Task Summarization Service

## File: `backend/app/service/task_summarization_service.py` (NEW)

```python
"""Service for LLM-powered task summarization."""
import asyncio
from datetime import datetime
from typing import Optional
from loguru import logger
from sqlmodel import Session, select
from app.component.database import session_make
from app.model.task_completion import TaskCompletion
from app.utils.agent import ListenChatAgent
from app.model.chat import Chat


async def summarize_task_steps(
    agent: ListenChatAgent,
    task_data: dict,
    original_question: str,
) -> str:
    """
    Generate summary of all execution steps.
    
    Args:
        agent: LLM agent for summarization
        task_data: Full task state data
        original_question: Original user query
        
    Returns:
        Summary of execution steps
    """
    messages = task_data.get("messages", [])
    task_info = task_data.get("taskInfo", [])
    task_running = task_data.get("taskRunning", [])
    
    # Extract step information
    steps_text = "\n".join([
        f"Step {i+1}: {msg.get('content', '')[:200]}"
        for i, msg in enumerate(messages)
        if msg.get("role") == "agent" or msg.get("step")
    ])
    
    prompt = f"""Analyze the following task execution and provide a comprehensive summary of all steps taken.

Original Task: {original_question}

Task Breakdown:
{chr(10).join([f"- {task.get('content', '')}" for task in task_info])}

Execution Steps:
{steps_text}

Please provide a clear, structured summary of:
1. All steps that were executed
2. The sequence of operations
3. Key actions taken at each step
4. Any branching or decision points

Format your response as a numbered list of steps with brief descriptions."""
    
    try:
        res = await agent.astep(prompt)
        summary = res.msg.content if res.msg else ""
        logger.info(f"Generated steps summary: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Error generating steps summary: {e}", exc_info=True)
        return ""


async def summarize_dos_donts(
    agent: ListenChatAgent,
    task_data: dict,
    original_question: str,
) -> str:
    """
    Extract do's and don'ts from task execution.
    
    Args:
        agent: LLM agent for summarization
        task_data: Full task state data
        original_question: Original user query
        
    Returns:
        Do's and don'ts summary
    """
    messages = task_data.get("messages", [])
    task_running = task_data.get("taskRunning", [])
    
    # Extract error messages and successful operations
    errors = []
    successes = []
    
    for task in task_running:
        if task.get("status") == "failed":
            errors.append(task.get("content", ""))
        elif task.get("status") == "completed":
            successes.append(task.get("content", ""))
    
    prompt = f"""Based on the following task execution, identify key do's and don'ts that would help in similar future tasks.

Original Task: {original_question}

Successful Operations:
{chr(10).join([f"- {s}" for s in successes[:10]])}

Failed Operations:
{chr(10).join([f"- {e}" for e in errors[:10]])}

Please provide:
1. DO's: Best practices and successful approaches
2. DON'Ts: Common mistakes and pitfalls to avoid

Format as two clear sections with bullet points."""
    
    try:
        res = await agent.astep(prompt)
        summary = res.msg.content if res.msg else ""
        logger.info(f"Generated do's/don'ts summary: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Error generating do's/don'ts summary: {e}", exc_info=True)
        return ""


async def summarize_learnings(
    agent: ListenChatAgent,
    task_data: dict,
    original_question: str,
) -> str:
    """
    Extract key learnings and insights from task execution.
    
    Args:
        agent: LLM agent for summarization
        task_data: Full task state data
        original_question: Original user query
        
    Returns:
        Learnings summary
    """
    messages = task_data.get("messages", [])
    task_info = task_data.get("taskInfo", [])
    
    # Extract insights from agent messages
    insights_text = "\n".join([
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "agent" and len(msg.get("content", "")) > 100
    ][:20])
    
    prompt = f"""Analyze the following task execution and extract key learnings and insights.

Original Task: {original_question}

Task Information:
{chr(10).join([f"- {task.get('content', '')}" for task in task_info])}

Agent Insights:
{insights_text[:2000]}

Please identify:
1. Key learnings from this task execution
2. Insights about the problem domain
3. Patterns or approaches that worked well
4. Technical knowledge gained
5. Any unexpected discoveries

Format as a structured list of learnings with brief explanations."""
    
    try:
        res = await agent.astep(prompt)
        summary = res.msg.content if res.msg else ""
        logger.info(f"Generated learnings summary: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Error generating learnings summary: {e}", exc_info=True)
        return ""


async def summarize_expected_outcome(
    agent: ListenChatAgent,
    task_data: dict,
    original_question: str,
) -> str:
    """
    Describe expected vs actual outcome.
    
    Args:
        agent: LLM agent for summarization
        task_data: Full task state data
        original_question: Original user query
        
    Returns:
        Expected outcome summary
    """
    messages = task_data.get("messages", [])
    summary_task = task_data.get("summaryTask", "")
    
    # Extract final result
    final_messages = [msg for msg in messages if msg.get("step") == "end"][-5:]
    final_result = "\n".join([msg.get("content", "") for msg in final_messages])
    
    prompt = f"""Based on the task execution, describe the expected outcome and how it relates to the actual result.

Original Task: {original_question}

Task Summary: {summary_task}

Final Result:
{final_result[:1000]}

Please provide:
1. What was the expected outcome based on the original task?
2. What was the actual outcome achieved?
3. How well did the result match expectations?
4. Any deviations or surprises?
5. Overall assessment of task completion

Format as a structured analysis."""
    
    try:
        res = await agent.astep(prompt)
        summary = res.msg.content if res.msg else ""
        logger.info(f"Generated expected outcome summary: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Error generating expected outcome summary: {e}", exc_info=True)
        return ""


async def summarize_task_completion(
    task_completion: TaskCompletion,
    options: Chat,
) -> TaskCompletion:
    """
    Generate all summaries for a task completion record.
    
    Args:
        task_completion: TaskCompletion record
        options: Chat options for agent creation
        
    Returns:
        Updated TaskCompletion record
    """
    try:
        # Create summarization agent
        from app.utils.agent import task_summary_agent
        agent = task_summary_agent(options)
        
        task_data = task_completion.task_data
        original_question = task_completion.original_question
        
        # Generate all summaries
        logger.info(f"Starting summarization for task {task_completion.task_id}")
        
        summaries = await asyncio.gather(
            summarize_task_steps(agent, task_data, original_question),
            summarize_dos_donts(agent, task_data, original_question),
            summarize_learnings(agent, task_data, original_question),
            summarize_expected_outcome(agent, task_data, original_question),
            return_exceptions=True,
        )
        
        summary_steps, summary_dos_donts, summary_learnings, summary_expected_outcome = summaries
        
        # Update record
        with session_make() as session:
            task_completion.summary_steps = summary_steps if not isinstance(summary_steps, Exception) else ""
            task_completion.summary_dos_donts = summary_dos_donts if not isinstance(summary_dos_donts, Exception) else ""
            task_completion.summary_learnings = summary_learnings if not isinstance(summary_learnings, Exception) else ""
            task_completion.summary_expected_outcome = summary_expected_outcome if not isinstance(summary_expected_outcome, Exception) else ""
            task_completion.updated_at = datetime.now()
            
            session.add(task_completion)
            session.commit()
            session.refresh(task_completion)
            
            logger.info(f"Summarization completed for task {task_completion.task_id}")
            return task_completion
            
    except Exception as e:
        logger.error(f"Error summarizing task completion: {e}", exc_info=True)
        return task_completion
```

