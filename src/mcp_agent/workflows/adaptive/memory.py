"""
Memory management for Adaptive Workflow
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from mcp_agent.logging.logger import get_logger
from .models import WorkflowMemory, WorkflowStrategy, TaskComplexity, TaskType

logger = get_logger(__name__)


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends"""

    @abstractmethod
    def save(self, workflow_id: str, memory: WorkflowMemory) -> None:
        """Save workflow memory"""
        pass

    @abstractmethod
    def load(self, workflow_id: str) -> Optional[WorkflowMemory]:
        """Load workflow memory"""
        pass

    @abstractmethod
    def delete(self, workflow_id: str) -> None:
        """Delete workflow memory"""
        pass

    @abstractmethod
    def list_workflows(self) -> Dict[str, Dict[str, Any]]:
        """List all workflows with basic info"""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory storage backend (default)"""

    def __init__(self):
        self._storage: Dict[str, WorkflowMemory] = {}

    def save(self, workflow_id: str, memory: WorkflowMemory) -> None:
        """Save workflow memory in memory"""
        memory.last_checkpoint = datetime.now()
        self._storage[workflow_id] = memory.model_copy(deep=True)
        logger.debug(f"Saved workflow {workflow_id} to in-memory storage")

    def load(self, workflow_id: str) -> Optional[WorkflowMemory]:
        """Load workflow memory from memory"""
        if workflow_id in self._storage:
            return self._storage[workflow_id].model_copy(deep=True)
        return None

    def delete(self, workflow_id: str) -> None:
        """Delete workflow memory from memory"""
        if workflow_id in self._storage:
            del self._storage[workflow_id]
            logger.debug(f"Deleted workflow {workflow_id} from in-memory storage")

    def list_workflows(self) -> Dict[str, Dict[str, Any]]:
        """List all workflows in memory"""
        workflows = {}
        for workflow_id, memory in self._storage.items():
            workflows[workflow_id] = {
                "objective": memory.objective,
                "task_type": memory.task_type,
                "phase": memory.phase,
                "start_time": memory.start_time.isoformat(),
                "last_checkpoint": memory.last_checkpoint.isoformat(),
                "iterations": memory.iterations,
                "total_cost": memory.total_cost,
            }
        return workflows


class MemoryManager:
    """Manages workflow memory with configurable backend"""

    def __init__(self, backend: Optional[MemoryBackend] = None):
        """
        Initialize memory manager

        Args:
            backend: Storage backend to use (defaults to InMemoryBackend)
        """
        self.backend = backend or InMemoryBackend()

    def save_memory(self, memory: WorkflowMemory) -> None:
        """Save workflow memory"""
        self.backend.save(memory.workflow_id, memory)

    def load_memory(self, workflow_id: str) -> Optional[WorkflowMemory]:
        """Load workflow memory"""
        return self.backend.load(workflow_id)

    def delete_memory(self, workflow_id: str) -> None:
        """Delete workflow memory"""
        self.backend.delete(workflow_id)

    def list_workflows(self) -> Dict[str, Dict[str, Any]]:
        """List all stored workflows with basic info"""
        return self.backend.list_workflows()

    def compress_memory(self, memory: WorkflowMemory, max_findings: int = 50) -> None:
        """
        Compress memory by keeping only the most important information

        Args:
            memory: The workflow memory to compress
            max_findings: Maximum number of key findings to keep
        """
        # Keep only the most recent key findings
        if len(memory.key_findings) > max_findings:
            memory.key_findings = memory.key_findings[-max_findings:]

        # Clear completed task results if they're too old
        for task in memory.completed_tasks:
            if task.result and len(task.result) > 1000:
                # Keep only a summary
                task.result = task.result[:1000] + "... [truncated]"

        # Limit citation history
        if len(memory.all_citations) > 100:
            # Keep only the most relevant citations
            memory.all_citations.sort(key=lambda c: c.relevance_score, reverse=True)
            memory.all_citations = memory.all_citations[:100]


class LearningManager:
    """Manages learning from past workflow executions"""

    def __init__(self):
        """Initialize learning manager with in-memory storage"""
        self.patterns = {
            "task_strategies": defaultdict(
                lambda: {
                    "successes": 0,
                    "failures": 0,
                    "avg_time": 0.0,
                    "avg_cost": 0.0,
                    "avg_iterations": 0.0,
                }
            ),
            "complexity_estimates": defaultdict(list),
            "tool_effectiveness": defaultdict(lambda: defaultdict(float)),
            "timing_data": defaultdict(list),
        }

    def record_execution(self, memory: WorkflowMemory, success: bool = True) -> None:
        """Record a workflow execution for learning"""
        task_key = f"{memory.task_type}_{memory.complexity}"
        strategy_key = f"{memory.task_type}_{memory.strategy.approach}"

        # Calculate execution time
        execution_time = (datetime.now() - memory.start_time).total_seconds()

        # Update timing data
        self.patterns["timing_data"][task_key].append(
            {
                "time": execution_time,
                "iterations": memory.iterations,
                "subagents": memory.total_subagents_created,
                "cost": memory.total_cost,
                "success": success,
            }
        )

        # Keep only recent data (last 100 executions)
        self.patterns["timing_data"][task_key] = self.patterns["timing_data"][task_key][
            -100:
        ]

        # Update strategy effectiveness
        stats = self.patterns["task_strategies"][strategy_key]
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        # Update rolling averages
        total = stats["successes"] + stats["failures"]
        weight = 1.0 / total
        stats["avg_time"] = stats["avg_time"] * (1 - weight) + execution_time * weight
        stats["avg_cost"] = (
            stats["avg_cost"] * (1 - weight) + memory.total_cost * weight
        )
        stats["avg_iterations"] = (
            stats["avg_iterations"] * (1 - weight) + memory.iterations * weight
        )

        # Track tool effectiveness
        for task in memory.completed_tasks:
            if task.agent_spec.server_names:
                for server in task.agent_spec.server_names:
                    effectiveness = 1.0 if task.status == "completed" else 0.0
                    current = self.patterns["tool_effectiveness"][memory.task_type][
                        server
                    ]
                    self.patterns["tool_effectiveness"][memory.task_type][server] = (
                        current * 0.9
                        + effectiveness * 0.1  # Exponential moving average
                    )

    def suggest_strategy(
        self, task_type: TaskType, complexity: TaskComplexity
    ) -> Optional[WorkflowStrategy]:
        """Suggest a strategy based on past executions"""
        best_strategy = None
        best_score = -1.0

        # Evaluate all strategies for this task type
        for strategy_key, stats in self.patterns["task_strategies"].items():
            if strategy_key.startswith(str(task_type)):
                total = stats["successes"] + stats["failures"]
                if total >= 3:  # Need at least 3 executions
                    success_rate = stats["successes"] / total
                    # Score based on success rate and efficiency
                    time_factor = 1.0 / (
                        1.0 + stats["avg_time"] / 1800
                    )  # 30 min baseline
                    cost_factor = 1.0 / (1.0 + stats["avg_cost"] / 10.0)  # $10 baseline

                    score = success_rate * time_factor * cost_factor

                    if score > best_score:
                        best_score = score
                        approach = strategy_key.split("_", 1)[1]

                        # Determine parallelism based on complexity
                        parallelism = {
                            TaskComplexity.SIMPLE: 3,
                            TaskComplexity.MODERATE: 5,
                            TaskComplexity.COMPLEX: 10,
                            TaskComplexity.EXTENSIVE: 15,
                        }.get(complexity, 5)

                        best_strategy = WorkflowStrategy(
                            approach=approach,
                            parallelism_level=parallelism,
                            subagent_budget=parallelism * 2,
                        )

        return best_strategy

    def estimate_complexity(
        self, objective: str, task_type: TaskType
    ) -> TaskComplexity:
        """Estimate task complexity based on objective and type"""
        objective_lower = objective.lower()

        # Task type specific heuristics
        if task_type == TaskType.RESEARCH:
            if any(
                kw in objective_lower
                for kw in ["comprehensive", "all", "complete", "detailed analysis"]
            ):
                return TaskComplexity.EXTENSIVE
            elif any(
                kw in objective_lower
                for kw in ["compare", "analyze", "investigate", "multiple"]
            ):
                return TaskComplexity.COMPLEX
            elif any(
                kw in objective_lower
                for kw in ["find", "what is", "definition", "explain"]
            ):
                return TaskComplexity.SIMPLE
            else:
                return TaskComplexity.MODERATE

        elif task_type == TaskType.ACTION:
            if any(
                kw in objective_lower
                for kw in ["implement", "create", "build", "develop"]
            ):
                return TaskComplexity.COMPLEX
            elif any(
                kw in objective_lower for kw in ["update", "modify", "fix", "change"]
            ):
                return TaskComplexity.MODERATE
            else:
                return TaskComplexity.SIMPLE

        else:  # HYBRID or ANALYSIS
            # Count the number of distinct tasks mentioned
            action_words = ["create", "update", "delete", "implement", "build", "fix"]
            research_words = ["find", "analyze", "compare", "investigate", "research"]

            action_count = sum(1 for word in action_words if word in objective_lower)
            research_count = sum(
                1 for word in research_words if word in objective_lower
            )
            total_tasks = action_count + research_count

            if total_tasks >= 4:
                return TaskComplexity.EXTENSIVE
            elif total_tasks >= 2:
                return TaskComplexity.COMPLEX
            else:
                return TaskComplexity.MODERATE

    def get_effective_tools(
        self, task_type: TaskType, threshold: float = 0.7
    ) -> List[str]:
        """Get list of effective tools for a task type"""
        effective_tools = []

        if task_type in self.patterns["tool_effectiveness"]:
            for tool, effectiveness in self.patterns["tool_effectiveness"][
                task_type
            ].items():
                if effectiveness >= threshold:
                    effective_tools.append(tool)

        return effective_tools

    def estimate_resources(
        self, task_type: TaskType, complexity: TaskComplexity
    ) -> Dict[str, float]:
        """Estimate resource requirements based on historical data"""
        task_key = f"{task_type}_{complexity}"
        timing_data = self.patterns["timing_data"].get(task_key, [])

        if not timing_data:
            # Default estimates
            base_times = {
                TaskComplexity.SIMPLE: 300,  # 5 minutes
                TaskComplexity.MODERATE: 900,  # 15 minutes
                TaskComplexity.COMPLEX: 1800,  # 30 minutes
                TaskComplexity.EXTENSIVE: 3600,  # 60 minutes
            }
            return {
                "estimated_time": base_times.get(complexity, 900),
                "estimated_cost": base_times.get(complexity, 900)
                / 300,  # Rough cost estimate
                "estimated_iterations": 5
                + (5 * list(TaskComplexity).index(complexity)),
                "confidence": 0.0,
            }

        # Calculate estimates from historical data
        successful_runs = [d for d in timing_data if d["success"]]
        if successful_runs:
            avg_time = sum(d["time"] for d in successful_runs) / len(successful_runs)
            avg_cost = sum(d["cost"] for d in successful_runs) / len(successful_runs)
            avg_iterations = sum(d["iterations"] for d in successful_runs) / len(
                successful_runs
            )

            return {
                "estimated_time": avg_time * 1.2,  # Add 20% buffer
                "estimated_cost": avg_cost * 1.2,
                "estimated_iterations": int(avg_iterations * 1.2),
                "confidence": min(
                    len(successful_runs) / 10.0, 1.0
                ),  # Confidence based on sample size
            }
        else:
            # Use all data if no successful runs
            avg_time = sum(d["time"] for d in timing_data) / len(timing_data)
            avg_cost = sum(d["cost"] for d in timing_data) / len(timing_data)
            avg_iterations = sum(d["iterations"] for d in timing_data) / len(
                timing_data
            )

            return {
                "estimated_time": avg_time
                * 1.5,  # Add 50% buffer for unsuccessful patterns
                "estimated_cost": avg_cost * 1.5,
                "estimated_iterations": int(avg_iterations * 1.5),
                "confidence": min(len(timing_data) / 20.0, 0.5),  # Lower confidence
            }
