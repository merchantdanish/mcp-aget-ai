"""
Data models for the Deep Orchestrator workflow.

This module contains all the Pydantic models and dataclasses used by the
Deep Orchestrator for task planning, execution, and result tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class TaskStatus(str, Enum):
    """Status of a task execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # For dependency failures


class PolicyAction(str, Enum):
    """Actions the policy engine can recommend."""

    CONTINUE = "continue"
    REPLAN = "replan"
    FORCE_COMPLETE = "force_complete"
    EMERGENCY_STOP = "emergency_stop"


# ============================================================================
# Knowledge and Memory Models
# ============================================================================


@dataclass
class KnowledgeItem:
    """A piece of extracted knowledge from task execution."""

    key: str
    value: Any
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "category": self.category,
        }


@dataclass
class TaskResult:
    """Result from executing a task."""

    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    knowledge_extracted: List[KnowledgeItem] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    retry_count: int = 0

    @property
    def success(self) -> bool:
        """Check if the task was successful."""
        return self.status == TaskStatus.COMPLETED


# ============================================================================
# Planning Models
# ============================================================================


class Task(BaseModel):
    """Enhanced task model with dependency handling."""

    description: str = Field(
        description="Clear, specific description of what needs to be done"
    )
    agent: Optional[str] = Field(
        default="AUTO", description="Agent name or AUTO for dynamic creation"
    )
    servers: List[str] = Field(default_factory=list, description="Required MCP servers")
    dependencies: List[str] = Field(
        default_factory=list, description="Task IDs this depends on"
    )

    # Runtime fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    def get_hash_key(self) -> Tuple[str, ...]:
        """Get a hash key for deduplication."""
        return (self.description.strip().lower(), tuple(sorted(self.servers)))


class Step(BaseModel):
    """A step containing tasks that can run in parallel."""

    description: str = Field(description="What this step accomplishes")
    tasks: List[Task] = Field(description="Tasks that can run in parallel")

    # Runtime fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    completed: bool = Field(default=False)


class Plan(BaseModel):
    """A complete execution plan."""

    steps: List[Step] = Field(description="Sequential steps to execute")
    is_complete: bool = Field(
        default=False, description="Whether objective is already satisfied"
    )
    reasoning: str = Field(default="", description="Explanation of the plan")


# ============================================================================
# Knowledge Extraction Models
# ============================================================================


class ExtractedKnowledge(BaseModel):
    """Model for knowledge extraction results."""

    items: List[Dict[str, Any]] = Field(
        description="Knowledge items with key, value, category, and confidence"
    )


# ============================================================================
# Agent Design Models
# ============================================================================


class AgentDesign(BaseModel):
    """Model for dynamically designed agents."""

    name: str = Field(
        description="Short, descriptive name (e.g., 'DataAnalyzer', 'ReportWriter')"
    )
    role: str = Field(description="The agent's specialty and expertise")
    instruction: str = Field(
        description="Detailed instruction for optimal task completion"
    )
    key_behaviors: List[str] = Field(
        description="Important behaviors the agent should exhibit"
    )
    tool_usage_tips: List[str] = Field(
        description="Specific tips for using the required tools"
    )


# ============================================================================
# Verification Models
# ============================================================================


class VerificationResult(BaseModel):
    """Result of objective verification."""

    is_complete: bool = Field(description="Whether objective is satisfied")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level (0-1)")
    reasoning: str = Field(description="Detailed explanation of the assessment")
    missing_elements: List[str] = Field(
        default_factory=list, description="Critical missing elements"
    )
    achievements: List[str] = Field(
        default_factory=list, description="What was successfully completed"
    )
