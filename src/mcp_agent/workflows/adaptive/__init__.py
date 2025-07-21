"""
Adaptive Workflow - A multi-agent system based on Claude Deep Research architecture

This workflow implements an adaptive multi-agent pattern that can handle both
research tasks and action-oriented tasks with the following key features:

- Dynamic agent creation based on task requirements
- Non-cascading iteration limits (prevents explosion)
- Comprehensive time and cost tracking with budgets
- Memory persistence for long-running tasks
- Parallel execution of independent subtasks
- Citation tracking for research tasks
- Learning from past executions
"""

from .adaptive_workflow import AdaptiveWorkflow
from .models import (
    TaskType,
    SubagentTask,
    WorkflowStrategy,
    WorkflowMemory,
    WorkflowResult,
    CitationSource,
)

__all__ = [
    "AdaptiveWorkflow",
    "TaskType",
    "SubagentTask",
    "WorkflowMemory",
    "WorkflowResult",
    "CitationSource",
]
