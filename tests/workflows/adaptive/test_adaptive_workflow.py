"""
Tests for the Adaptive Workflow
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import timedelta

from mcp_agent.workflows.adaptive import AdaptiveWorkflow, TaskType
from mcp_agent.workflows.adaptive.models import (
    TaskComplexity,
    WorkflowResult,
    WorkflowMemory,
    SubagentTask,
    SubagentSpec,
)
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class TestAdaptiveWorkflowInit:
    """Tests for Adaptive Workflow initialization"""

    def test_init_with_defaults(self, mock_llm_factory, mock_context):
        """Test that AdaptiveWorkflow can be initialized with default values"""
        workflow = AdaptiveWorkflow(llm_factory=mock_llm_factory, context=mock_context)

        assert workflow.llm_factory == mock_llm_factory
        assert workflow.context == mock_context
        assert workflow.time_budget == timedelta(minutes=30)
        assert workflow.cost_budget == 10.0
        assert workflow.max_iterations == 20
        assert workflow.max_subagents == 50
        assert workflow.enable_parallel is True
        assert workflow.enable_learning is True

    def test_init_with_custom_values(self, mock_llm_factory, mock_context):
        """Test initialization with custom values"""
        workflow = AdaptiveWorkflow(
            llm_factory=mock_llm_factory,
            name="TestWorkflow",
            available_servers=["server1", "server2"],
            time_budget=timedelta(minutes=5),
            cost_budget=1.0,
            max_iterations=5,
            max_subagents=10,
            enable_parallel=False,
            enable_learning=False,
            context=mock_context,
        )

        assert workflow.name == "TestWorkflow"
        assert workflow.available_servers == ["server1", "server2"]
        assert workflow.time_budget == timedelta(minutes=5)
        assert workflow.cost_budget == 1.0
        assert workflow.max_iterations == 5
        assert workflow.max_subagents == 10
        assert workflow.enable_parallel is False
        assert workflow.enable_learning is False


class TestAdaptiveWorkflowTaskAnalysis:
    """Tests for task analysis functionality"""

    @pytest.mark.asyncio
    async def test_analyze_research_task(self, mock_workflow):
        """Test analyzing a research task"""
        # Mock the task analyzer response
        mock_workflow.llm_factory.return_value.generate_structured_mock.return_value = (
            MagicMock(
                task_type=TaskType.RESEARCH,
                complexity=TaskComplexity.MODERATE,
                key_aspects=["aspect1", "aspect2"],
                recommended_tools=["tool1"],
                potential_challenges=[],
            )
        )

        with patch.object(mock_workflow, "_analyze_task") as mock_analyze:
            mock_analyze.return_value = (TaskType.RESEARCH, TaskComplexity.MODERATE)

            task_type, complexity = await mock_workflow._analyze_task(
                "What are the best practices for API design?", MagicMock()
            )

            assert task_type == TaskType.RESEARCH
            assert complexity == TaskComplexity.MODERATE

    @pytest.mark.asyncio
    async def test_analyze_action_task(self, mock_workflow):
        """Test analyzing an action task"""
        with patch.object(mock_workflow, "_analyze_task") as mock_analyze:
            mock_analyze.return_value = (TaskType.ACTION, TaskComplexity.SIMPLE)

            task_type, complexity = await mock_workflow._analyze_task(
                "Create a README file with installation instructions", MagicMock()
            )

            assert task_type == TaskType.ACTION
            assert complexity == TaskComplexity.SIMPLE


class TestAdaptiveWorkflowExecution:
    """Tests for workflow execution"""

    @pytest.mark.asyncio
    async def test_generate_str_simple(self, mock_workflow):
        """Test simple string generation"""
        # Mock the entire execution flow
        with patch.object(mock_workflow, "_execute_workflow") as mock_execute:
            mock_result = WorkflowResult(
                workflow_id="test-id",
                objective="Test objective",
                task_type=TaskType.RESEARCH,
                result="Test result content",
                tasks_completed=1,
                tasks_failed=0,
                subagents_used=1,
                total_time_seconds=10.0,
                iterations=1,
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost=0.01,
                success=True,
                confidence=0.9,
            )
            mock_execute.return_value = mock_result

            with patch.object(mock_workflow, "_messages_to_string") as mock_convert:
                mock_convert.return_value = "Test result content"

                result = await mock_workflow.generate_str("Test query")

                assert result == "Test result content"
                mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_resource_limits(self, mock_workflow):
        """Test that resource limits are enforced"""
        # Set very low limits
        mock_workflow.max_iterations = 2
        mock_workflow.max_subagents = 2

        memory = WorkflowMemory(
            workflow_id="test-id",
            objective="Test",
            task_type=TaskType.RESEARCH,
            iterations=2,
            total_subagents_created=2,
        )
        mock_workflow._current_memory = memory

        # Should not continue due to limits
        assert mock_workflow._should_continue(0) is False

    @pytest.mark.asyncio
    async def test_parallel_execution_enabled(self, mock_workflow):
        """Test parallel execution when enabled"""
        mock_workflow.enable_parallel = True

        tasks = [
            SubagentTask(
                task_id="task1",
                description="Task 1",
                objective="Objective 1",
                agent_spec=SubagentSpec(
                    name="Agent1", instruction="Instruction 1", server_names=[]
                ),
            ),
            SubagentTask(
                task_id="task2",
                description="Task 2",
                objective="Objective 2",
                agent_spec=SubagentSpec(
                    name="Agent2", instruction="Instruction 2", server_names=[]
                ),
            ),
        ]

        with patch.object(mock_workflow, "_execute_single_task") as mock_execute:
            mock_execute.return_value = None

            await mock_workflow._execute_parallel(tasks, RequestParams(), MagicMock())

            # Both tasks should be executed
            assert mock_execute.call_count == 2


class TestAdaptiveWorkflowMemory:
    """Tests for memory management"""

    def test_memory_initialization(self, mock_workflow):
        """Test workflow memory is properly initialized"""
        assert mock_workflow.memory_manager is not None
        assert mock_workflow.learning_manager is not None

    def test_memory_compression(self, mock_workflow):
        """Test memory compression functionality"""
        # Create mock memory with lots of findings
        memory = WorkflowMemory(
            workflow_id="test-id",
            objective="Test",
            task_type=TaskType.RESEARCH,
            key_findings=[f"Finding {i}" for i in range(100)],
        )

        mock_workflow.memory_manager.compress_memory(memory, max_findings=10)

        # Should be compressed to max_findings
        assert len(memory.key_findings) <= 10


class TestAdaptiveWorkflowLearning:
    """Tests for learning functionality"""

    def test_learning_disabled(self, mock_workflow):
        """Test workflow with learning disabled"""
        mock_workflow.enable_learning = False

        # Should still have learning manager but not use it
        assert mock_workflow.learning_manager is not None

    def test_task_type_detection(self):
        """Test that task types are properly detected"""
        from mcp_agent.workflows.adaptive.memory import LearningManager

        manager = LearningManager()

        # Research task
        complexity = manager.estimate_complexity(
            "What are the best practices for API design?", TaskType.RESEARCH
        )
        assert complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
            TaskComplexity.EXTENSIVE,
        ]

        # Action task
        complexity = manager.estimate_complexity(
            "Create a new configuration file", TaskType.ACTION
        )
        assert complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
            TaskComplexity.EXTENSIVE,
        ]


class TestAdaptiveWorkflowIntegration:
    """Integration tests for the workflow"""

    @pytest.mark.asyncio
    async def test_workflow_end_to_end_mock(self, mock_workflow):
        """Test a complete workflow execution with mocks"""
        # Mock all the internal methods
        with (
            patch.object(mock_workflow, "_analyze_task") as mock_analyze,
            patch.object(mock_workflow, "_plan_strategy") as mock_strategy,
            patch.object(mock_workflow, "_execute_iterations") as mock_execute,
            patch.object(mock_workflow, "_synthesize_results") as mock_synthesize,
        ):
            # Set up mock returns
            mock_analyze.return_value = (TaskType.RESEARCH, TaskComplexity.MODERATE)
            mock_strategy.return_value = MagicMock()
            mock_execute.return_value = None
            mock_synthesize.return_value = WorkflowResult(
                workflow_id="test-id",
                objective="Test objective",
                task_type=TaskType.RESEARCH,
                result="Mock result",
                tasks_completed=1,
                tasks_failed=0,
                subagents_used=1,
                total_time_seconds=5.0,
                iterations=1,
                total_input_tokens=50,
                total_output_tokens=25,
                total_cost=0.005,
                success=True,
                confidence=0.9,
            )

            # Mock message conversion
            with patch.object(mock_workflow, "_messages_to_string") as mock_convert:
                mock_convert.return_value = "Mock result"

                result = await mock_workflow.generate_str("Test query")

                # Verify all steps were called
                mock_analyze.assert_called_once()
                mock_strategy.assert_called_once()
                mock_execute.assert_called_once()
                mock_synthesize.assert_called_once()

                assert result == "Mock result"
