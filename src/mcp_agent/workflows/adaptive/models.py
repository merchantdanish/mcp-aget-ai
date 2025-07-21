"""
Data models for the Adaptive Workflow
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Types of tasks the workflow can handle"""
    RESEARCH = "research"    # Information gathering
    ACTION = "action"        # Making changes/executing operations
    HYBRID = "hybrid"        # Research followed by action


class TaskComplexity(str, Enum):
    """
    Task complexity estimation - used for initial resource allocation.
    
    TODO: This should be refactored to dynamic resource allocation based on:
    - Discovered scope during execution
    - Historical data for similar objectives
    - Available budget constraints
    
    The current implementation uses these as buckets for initial estimates,
    but a continuous scale would be more appropriate.
    """
    SIMPLE = "simple"       # Quick focused query
    MODERATE = "moderate"   # Multiple aspects to explore
    COMPLEX = "complex"     # Broad investigation needed
    EXTENSIVE = "extensive" # Comprehensive analysis required


class SubagentSpec(BaseModel):
    """Specification for creating a subagent"""
    name: str
    instruction: str
    server_names: List[str] = Field(default_factory=list)
    expected_iterations: int = Field(default=5, le=10)  # Non-cascading limit
    parallel_tools: bool = True
    timeout_seconds: int = Field(default=300)  # 5 min default


class SubagentTask(BaseModel):
    """A task assigned to a subagent"""
    task_id: str
    description: str
    objective: str
    agent_spec: SubagentSpec
    dependencies: List[str] = Field(default_factory=list)  # IDs of tasks this depends on
    context: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    citations: List["CitationSource"] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class CitationSource(BaseModel):
    """A source that can be cited in results"""
    url: Optional[str] = None
    title: str
    author: Optional[str] = None
    date: Optional[str] = None
    content_snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_type: str = "web"  # web, document, database, etc.


class WorkflowStrategy(BaseModel):
    """Strategy for executing the workflow"""
    approach: str  # breadth-first, depth-first, hybrid
    parallelism_level: int = Field(default=5, ge=1, le=20)
    subagent_budget: int = Field(default=10, ge=1, le=50)
    time_allocation: Dict[str, float] = Field(default_factory=dict)  # Phase -> time %
    tool_preferences: Dict[str, List[str]] = Field(default_factory=dict)  # Task type -> tools


class WorkflowMemory(BaseModel):
    """Persistent memory for workflow state"""
    workflow_id: str
    objective: str
    task_type: TaskType
    complexity: TaskComplexity
    strategy: WorkflowStrategy
    
    # Execution state
    phase: str = "planning"  # planning, executing, synthesizing, complete
    completed_tasks: List[SubagentTask] = Field(default_factory=list)
    pending_tasks: List[SubagentTask] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    all_citations: List[CitationSource] = Field(default_factory=list)
    
    # Metrics
    start_time: datetime = Field(default_factory=datetime.now)
    last_checkpoint: datetime = Field(default_factory=datetime.now)
    iterations: int = 0
    total_subagents_created: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    
    # Learning data
    successful_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    failed_patterns: List[Dict[str, Any]] = Field(default_factory=list)


class TaskBatch(BaseModel):
    """A batch of tasks to execute"""
    tasks: List[SubagentTask]
    rationale: str
    estimated_time: float  # seconds
    can_parallelize: bool = True


class ProgressEvaluation(BaseModel):
    """Evaluation of workflow progress"""
    is_complete: bool
    confidence: float = Field(ge=0.0, le=1.0)
    missing_aspects: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    should_pivot: bool = False
    pivot_reason: Optional[str] = None


class WorkflowResult(BaseModel):
    """Final result of the workflow"""
    workflow_id: str
    objective: str
    task_type: TaskType
    
    # Results
    result: str
    citations: List[CitationSource] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Metrics
    tasks_completed: int
    tasks_failed: int
    subagents_used: int
    total_time_seconds: float
    iterations: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    
    # Quality indicators
    success: bool = True
    limitations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)